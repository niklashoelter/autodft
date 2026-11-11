"""State machine -- all task/job state transitions driven by SQLModel queries.

Every public function accepts a ``Session`` as its first argument so the
caller controls the transaction boundary.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from time import monotonic
from pathlib import Path
from typing import Collection, NamedTuple, Optional

from sqlalchemy import ColumnElement, and_, case, exists, or_, update
from sqlmodel import Session, col, select

from autodft import categories
from autodft.config import Settings
from autodft.engine import circuit_breaker
from autodft.engine.scheduler import NOT_IN_SLURM, Scheduler, SubmitResult
from autodft.extraction import results
from autodft.models.enums import (
    TERMINAL_SLURM_STATES,
    TRANSIENT_SLURM_STATES,
    ProjectJobStatus,
    SlurmStatus,
    TaskStatus,
    TaskType,
)
from autodft.models.geometry import MoleculeGeometry
from autodft.models.header import ComputationHeader
from autodft.models.job import ComputationJob
from autodft.models.molecule import Molecule
from autodft.models.project_job import ProjectJob
from autodft.models.state import MoleculeState
from autodft.models.task import ComputationTask
from autodft.qm.base import QMEngine, QMResult
from autodft.qm.orca.parser import DENSITY_CUBES, MP2_RESPONSE_MEMORY, OrcaParser

logger = logging.getLogger(__name__)


# The API dashboard runs in a daemon thread of this same process and shares
# one SQLite file with the worker, and SQLite allows exactly one writer. The
# heavy steps below interleave row updates with filesystem work on a network
# mount -- parsing ORCA outputs, writing input decks -- so a step that held a
# single transaction across the whole batch kept the write lock for minutes,
# and every other writer timed out behind it. They now commit per item, and
# step 2 parses each output before its first write, with the lock free.

# Failed checks that recur identically on every attempt, whatever else failed.
DETERMINISTIC_CHECKS = ("LibXC Needed", "S1 Stays Bound")

# Consecutive raising job builds after which the task fails.
MAX_BUILD_FAILURES = 10

# Why an MP2 response retry at its %maxcore cap fails its task instead of rerunning unchanged.
MP2_CAP_REACHED = "MP2 response memory cap reached"

# fail_reason prefix of a job SLURM can never run; its task fails without a retry.
UNSATISFIABLE = "unsatisfiable: "

# One unsatisfiable Reason on more pending jobs than this is a cluster change: nothing is cancelled.
MAX_UNSATISFIABLE_PER_TICK = 10

# Consecutive sbatch rejections of a job after which it fails as an attempt.
MAX_SBATCH_REJECTIONS = 3
# fail_reason prefix of an unjudged job whose sbatch failed; SLURM may have accepted it.
SBATCH_FAILED = "sbatch failed"
_SBATCH_RETRYING = f"{SBATCH_FAILED} (retrying)"
# fail_reason prefix of such a job SLURM was found not to have.
_NOT_ACCEPTED = "not in SLURM after"
_SBATCH_REJECTED = re.compile(r"^sbatch failed \(rejected (\d+)/")


# ======================================================================
# Step 1 -- Update running jobs from the scheduler
# ======================================================================

def update_running_jobs(session: Session, scheduler: Scheduler) -> None:
    """Refresh every submitted, unfinished job from the scheduler."""

    # Not a transient whitelist: a state in neither set (a new SLURM state,
    # or a width-truncated one from an older parser) would never be polled
    # again nor judged. Unrecognised states are polled until judged below.
    statement = select(ComputationJob).where(
        col(ComputationJob.slurm_jobid).is_not(None),
        or_(
            col(ComputationJob.slurm_status).is_(None),
            col(ComputationJob.slurm_status).in_(sorted(TRANSIENT_SLURM_STATES)),
            and_(
                col(ComputationJob.slurm_status).not_in(sorted(TERMINAL_SLURM_STATES)),
                col(ComputationJob.success).is_(None),
            ),
        ),
    )
    jobs = session.exec(statement).all()

    # One batched query instead of one subprocess per job. See
    # SlurmScheduler.get_statuses -- the per-job loop cost minutes per tick
    # once a few thousand jobs were in flight.
    statuses = scheduler.get_statuses([str(j.slurm_jobid) for j in jobs])

    for job in jobs:
        new_status = statuses.get(str(job.slurm_jobid), SlurmStatus.UNKNOWN)
        logger.debug(
            "Job %d (slurm %s): %s -> %s",
            job.id, job.slurm_jobid, job.slurm_status, new_status,
        )
        job.slurm_status = new_status
        if new_status in TERMINAL_SLURM_STATES:
            job.time_end = datetime.now(timezone.utc)
        elif new_status not in TRANSIENT_SLURM_STATES and job.success is None:
            reason = (
                "SLURM has no record of the job" if new_status == NOT_IN_SLURM
                else f"Unrecognised SLURM state {new_status}"
            )
            logger.warning("Job %d (slurm %s): %s; failing the job", job.id, job.slurm_jobid, reason)
            job.success = False
            job.fail_reason = reason[:500]
            job.time_end = datetime.now(timezone.utc)
        session.add(job)

    pending = [j for j in jobs if j.slurm_status == SlurmStatus.PENDING and j.success is None]
    unsatisfiable = scheduler.get_unsatisfiable([str(j.slurm_jobid) for j in pending]) if pending else {}
    for reason, count in Counter(unsatisfiable.values()).items():
        if count > MAX_UNSATISFIABLE_PER_TICK:
            logger.critical("%d of %d pending jobs turned unsatisfiable (%s) in one tick; "
                            "taken as a cluster-side change, nothing cancelled", count, len(pending), reason)
            unsatisfiable = {}
            break
    for job in pending:
        reason = unsatisfiable.get(str(job.slurm_jobid))
        if reason is None or not scheduler.cancel(str(job.slurm_jobid)):
            continue  # a failed scancel is tried again next tick
        logger.warning("Job %d (slurm %s) can never run (%s); cancelled", job.id, job.slurm_jobid, reason)
        job.slurm_status = SlurmStatus.CANCELLED
        job.success = False
        job.fail_reason = f"{UNSATISFIABLE}{reason}"[:500]
        job.time_end = datetime.now(timezone.utc)
        session.add(job)

    session.flush()


# ======================================================================
# Step 2 -- Process finished jobs (check QM output)
# ======================================================================

def process_finished_jobs(session: Session, qm_engine: QMEngine, max_attempts: int = 3) -> None:
    """For jobs with a terminal slurm_status but ``success IS NULL``:
    check the QM output, set ``success`` / ``fail_reason``, and extract
    geometries for successful confsearch/optimization jobs. Each job's files
    are read before its first write, and each job commits on its own.
    """

    # An explicit terminal whitelist. The original list (COMPLETED / FAILED /
    # TIMEOUT) silently swallowed CANCELLED, OUT_OF_MEMORY, NODE_FAIL and
    # PREEMPTED -- those jobs kept `success = NULL` forever and their tasks
    # sat `pending` with no error anywhere, which also meant the resource
    # escalation could never fire for a real OOM kill.
    #
    # Inverting it to "anything not RUNNING/PENDING/UNKNOWN" over-corrected:
    # COMPLETING, SUSPENDED, CONFIGURING and REQUEUED are *not* finished, and
    # parsing a job mid-epilog marks a healthy run failed with
    # ['Termination'] -- which is exactly the trigger that escalates its
    # retry to 32 ranks for 4 days.
    statement = select(ComputationJob.id).where(
        col(ComputationJob.slurm_status).in_(sorted(TERMINAL_SLURM_STATES)),
        col(ComputationJob.success).is_(None),
    )
    for job_id in session.exec(statement).all():
        _process_finished_job(session, job_id, qm_engine, max_attempts)
        session.commit()


class _Verdict(NamedTuple):
    success: bool
    fail_reason: Optional[str] = None
    result: Optional[QMResult] = None
    # The optimised XYZ, or why it could not be read.
    xyz: Optional[str] = None
    xyz_error: Optional[Exception] = None
    record: Optional[dict] = None


def _process_finished_job(session: Session, job_id: int, qm_engine: QMEngine, max_attempts: int) -> None:
    """Judge one finished job: read its files with no write pending, then write it under one claim."""
    job = session.get(ComputationJob, job_id)
    if job is None or job.success is not None:
        return
    task = session.get(ComputationTask, job.task_id)
    if task is None:
        logger.error("Job %d references missing task %d", job.id, job.task_id)
        return
    job_path = Path(job.job_path) if job.job_path else None
    verdict = _judge(session, job, task, job_path, qm_engine, max_attempts)

    # Judged or removed since it was read: someone else's write stands.
    if not _claim_finished(session, job_id):
        session.rollback()
        return
    job.success = verdict.success
    if verdict.fail_reason is not None:
        job.fail_reason = verdict.fail_reason
    job.time_end = job.time_end or datetime.now(timezone.utc)
    session.add(job)

    if not job.success:
        return

    # -- Extract geometries for successful jobs -----------------------

    result = verdict.result
    try:
        if task.task_type == TaskType.confsearch and result.conformers:
            _create_conformer_geometries(
                session, task, result.conformers, result.energy,
                result.conformer_energies, result.conformer_degeneracies,
            )

        elif task.task_type == TaskType.optimization:
            if verdict.xyz_error is not None:
                raise verdict.xyz_error
            _create_optimization_geometry(session, task, verdict.xyz, result.energy)
    except Exception as exc:  # noqa: BLE001 - deliberately broad
        logger.exception("Failed to extract geometries for job %d", job.id)
        job.success = False
        job.fail_reason = f"Geometry extraction failed: {type(exc).__name__}: {exc}"[:500]
        session.add(job)
        # Drop anything the failed extraction already wrote. Leaving
        # partial rows behind means the eventual successful retry mixes
        # them into "keep the N lowest conformers", since they carry the
        # same origin_task_id.
        for partial in session.exec(
            select(MoleculeGeometry).where(MoleculeGeometry.origin_task_id == task.id)
        ).all():
            session.delete(partial)

    if job.success and verdict.record is not None:
        # Parsed once here; analyses and exports read it from the database.
        try:
            results.save(session, task, job, verdict.record)
        except Exception:  # noqa: BLE001 - readers re-parse a job without a record
            logger.exception("Could not store the parsed results of job %d", job_id)


def _judge(
    session: Session, job: ComputationJob, task: ComputationTask, job_path: Optional[Path],
    qm_engine: QMEngine, max_attempts: int,
) -> _Verdict:
    """*job*'s outcome from its files; reads only, so no write lock is held across them."""
    if job_path is None or not job_path.exists():
        return _Verdict(False, "Job path missing or does not exist")

    # Output parsing touches files a killed job may have left truncated,
    # and the parser raises on several of those shapes. Isolate it per
    # job: an unhandled exception here used to unwind out of the whole
    # pipeline tick, discarding every other molecule's progress and
    # re-raising on the same job forever.
    try:
        result = qm_engine.check_output(
            job_path, _check_type(session, task), _imag_freq_tolerance(session, task),
        )
    except Exception as exc:  # noqa: BLE001 - deliberately broad
        logger.exception("Failed to parse output for job %d", job.id)
        return _Verdict(False, f"Output parsing failed: {type(exc).__name__}: {exc}"[:500])

    if not result.success:
        failed_checks = [k for k, v in result.checks.items() if not v]
        # A SLURM OOM kill usually truncates output.out before ORCA can
        # print anything, so the only reliable memory signal is the SLURM
        # state. Fold it into fail_reason so the retry escalates per-rank
        # memory (the one failure mode where that is the right response).
        if job.slurm_status == "OUT_OF_MEMORY":
            failed_checks.append("OUT_OF_MEMORY")
        if not (failed_checks == [DENSITY_CUBES] and _accept_cube_miss(session, job, max_attempts)):
            return _Verdict(False, str(failed_checks), result)
        # Retried once already, or no retry left; the energy is fine.
        logger.warning(
            "Job %d: density cubes still missing or empty after a retry in %s (%s); "
            "accepting the job", job.id, job_path, ", ".join(OrcaParser.missing_cubes(job_path)),
        )

    xyz = xyz_error = None
    if task.task_type == TaskType.optimization:
        try:
            xyz = _read_optimised_xyz(job_path)
        except Exception as exc:  # noqa: BLE001 - fails the job as a geometry extraction
            xyz_error = exc
    record = None
    if xyz_error is None:
        try:
            record = results.parse(session, task, job_path)
        except Exception:  # noqa: BLE001 - readers re-parse a job without a record
            logger.exception("Could not store the parsed results of job %d", job.id)
    return _Verdict(True, None, result, xyz, xyz_error, record)


def _claim_finished(session: Session, job_id: int) -> bool:
    """Take the write lock for judging *job_id*; False when it is gone, judged or no longer finished."""
    return bool(session.exec(
        update(ComputationJob)
        .where(col(ComputationJob.id) == job_id, col(ComputationJob.success).is_(None),
               col(ComputationJob.slurm_status).in_(sorted(TERMINAL_SLURM_STATES)))
        .values(success=None)
        .execution_options(synchronize_session=False)
    ).rowcount)


def _read_optimised_xyz(job_path: Path) -> Optional[str]:
    """The optimised XYZ ORCA left in *job_path*, or None when there is none."""
    xyz_file = job_path / "input.xyz"
    if not xyz_file.exists():
        logger.error("Optimised XYZ not found at %s", xyz_file)
        return None
    return xyz_file.read_text(encoding="utf-8")


def _create_conformer_geometries(
    session: Session,
    task: ComputationTask,
    conformers: list[str],
    energy: Optional[float],
    conformer_energies: Optional[list[float]] = None,
    conformer_degeneracies: Optional[list[Optional[int]]] = None,
) -> None:
    """Store each conformer XYZ as a :class:`MoleculeGeometry`.

    Each conformer keeps its own energy from the GOAT ensemble table. Storing
    the run's single scalar energy on all of them (the previous behaviour)
    made the "keep the N lowest" ORDER BY in _followup_confsearch an
    all-ties sort, so which conformers survived depended on SQLite's row
    order rather than on their energies.

    Note the scale: GOAT's ensemble table reports energies **relative to the
    global minimum in kcal/mol**, while _create_optimization_geometry stores
    absolute Hartree. Both only ever feed per-task ORDER BYs, which stay
    internally homogeneous, but never mix the two in one comparison -- which
    is why the fallback below is a positional rank rather than the run's
    absolute energy. (The parser already returns the ensemble sorted by
    energy, so the index preserves that order.)
    """
    for i, xyz_data in enumerate(conformers):
        if conformer_energies is not None and i < len(conformer_energies):
            geom_energy = conformer_energies[i]
        else:
            geom_energy = float(i)
        geom = MoleculeGeometry(
            state_id=task.state_id,
            xyz_data=xyz_data,
            energy=geom_energy,
            origin_task_id=task.id,
            label=f"conformer_{i}",
            degeneracy=conformer_degeneracies[i]
            if conformer_degeneracies is not None and i < len(conformer_degeneracies) else None,
        )
        session.add(geom)
        logger.debug(
            "Created conformer geometry %d for task %d", i, task.id,
        )


def _create_optimization_geometry(
    session: Session,
    task: ComputationTask,
    xyz_data: Optional[str],
    energy: Optional[float],
) -> None:
    """Store the optimised XYZ as a geometry; nothing without one."""
    if xyz_data is None:
        return
    geom = MoleculeGeometry(
        state_id=task.state_id,
        xyz_data=xyz_data,
        energy=energy,
        origin_task_id=task.id,
        label="optimised",
    )
    session.add(geom)
    session.flush()

    # Link output geometry to the task
    task.output_geometry_id = geom.id
    session.add(task)
    logger.debug("Created optimised geometry (id=%s) for task %d", geom.id, task.id)


# ======================================================================
# Step 3 -- Update task statuses based on job results
# ======================================================================

def _accept_cube_miss(session: Session, job: ComputationJob, max_attempts: int) -> bool:
    """Whether a cubes-only failure of *job* is accepted: its run's previous attempt
    failed on the cubes alone, or it was the run's last attempt."""
    task = session.get(ComputationTask, job.task_id)
    base = (task.retry_base or 0) if task is not None else 0
    if job.attempt - base >= max_attempts:
        return True
    previous = session.exec(
        select(ComputationJob)
        .where(ComputationJob.task_id == job.task_id,
               ComputationJob.attempt < job.attempt, ComputationJob.attempt > base)
        .order_by(col(ComputationJob.attempt).desc())
    ).first()
    return previous is not None and previous.fail_reason == str([DENSITY_CUBES])


def update_task_statuses(session: Session, max_attempts: int = 3) -> None:
    """Mark tasks successful if a job of the current run succeeded, failed if
    all its attempts are exhausted with failures."""

    pending_tasks = session.exec(
        select(ComputationTask).where(
            ComputationTask.status == TaskStatus.pending,
        )
    ).all()

    for task in pending_tasks:
        jobs = session.exec(
            select(ComputationJob).where(ComputationJob.task_id == task.id)
        ).all()

        if not jobs:
            continue

        # A success from before the last requeue/reset does not count.
        current = _current_run(task, jobs)
        if any(j.success is True for j in current):
            task.status = TaskStatus.successful
            # For optimization tasks, find the output geometry
            if task.task_type == TaskType.optimization and task.output_geometry_id is None:
                _set_output_geometry(session, task)
            # Singlepoint-type tasks never have followups
            if task.task_type.value.startswith("singlepoint"):
                task.has_followups = False
            task.updated_at = datetime.now(timezone.utc)
            session.add(task)
            logger.info("Task %d marked successful", task.id)
            continue

        # All judged (success is not None) and all failed, in the current run
        judged = [j for j in current if j.success is not None]
        if len(judged) >= max_attempts and all(j.success is False for j in judged):
            task.status = TaskStatus.failed
            task.updated_at = datetime.now(timezone.utc)
            session.add(task)
            logger.info("Task %d marked failed after %d attempts", task.id, len(judged))

    session.flush()


def _set_output_geometry(session: Session, task: ComputationTask) -> None:
    """Find the geometry produced by this task (lowest energy) and link it."""
    geom = session.exec(
        select(MoleculeGeometry)
        .where(MoleculeGeometry.origin_task_id == task.id)
        .order_by(col(MoleculeGeometry.energy).asc())
    ).first()
    if geom:
        task.output_geometry_id = geom.id


# ======================================================================
# Step 4 -- Start follow-up tasks
# ======================================================================

def start_followup_tasks(session: Session, settings: Settings) -> None:
    """For successful tasks with ``has_followups=True``: create the
    appropriate downstream tasks."""

    tasks = session.exec(
        select(ComputationTask).where(
            ComputationTask.status == TaskStatus.successful,
            ComputationTask.has_followups == True,  # noqa: E712
        )
    ).all()

    # Leave a task's has_followups set while its project is paused, so the
    # followups are created after the archive/export job finishes rather than
    # lost. See paused_project_names.
    paused = paused_project_names(session)
    if paused:
        tasks = [t for t in tasks if _task_project(session, t) not in paused]

    for task in tasks:
        if _molecule_is_archived(session, task):
            # Marked successful during its archive's wait; nothing can start a followup now.
            task.has_followups = False
            session.add(task)
            continue
        state = session.get(MoleculeState, task.state_id)
        if state is None:
            logger.error("Task %d references missing state %d", task.id, task.state_id)
            continue

        metadata = json.loads(state.metadata_json) if state.metadata_json else {}

        before = _count_tasks_depending_on(session, task.id)

        if task.task_type == TaskType.confsearch:
            _followup_confsearch(session, task, state, metadata)
        elif task.task_type == TaskType.optimization:
            _followup_optimization(session, task, state, metadata)
            _followup_categories(session, task, state, metadata)

        session.flush()
        created = _count_tasks_depending_on(session, task.id) - before

        # Mark follow-ups as consumed
        task.has_followups = False
        task.updated_at = datetime.now(timezone.utc)

        # A task that was supposed to spawn work and spawned none is a dead
        # end: has_followups is cleared and nothing ever revisits it, so the
        # state just stops with everything looking green. Fail it instead so
        # it is visible in the dashboard. Zero is only legitimate when the
        # downstream stage was not requested.
        if created == 0 and _followups_were_expected(task, metadata, state.description):
            logger.error(
                "Task %d (%s) completed but produced no follow-up tasks; "
                "marking it failed so the dead end is visible",
                task.id, task.task_type.value,
            )
            task.status = TaskStatus.failed

        session.add(task)

    session.flush()


def _count_tasks_depending_on(session: Session, task_id: int) -> int:
    """How many tasks currently hang off *task_id*."""
    return len(session.exec(
        select(ComputationTask.id).where(ComputationTask.depends_on_task_id == task_id)
    ).all())


def _followups_were_expected(
    task: ComputationTask, metadata: dict, description: str = "S0",
) -> bool:
    """Whether this task should have produced downstream work.

    False when the user explicitly turned the next stage off, in which case
    creating nothing is the correct outcome rather than a dead end.
    """
    if task.task_type == TaskType.confsearch:
        if not metadata.get("request_optimization", True):
            return False
        # Asking for zero conformers legitimately produces zero followups.
        for key, value in metadata.items():
            if key.startswith("max_conformers_") and not value:
                return False
        return True
    if task.task_type == TaskType.optimization:
        return bool(metadata.get("request_singlepoint", True) or _category_followups(description, metadata))
    return False


def _followup_confsearch(
    session: Session,
    task: ComputationTask,
    state: MoleculeState,
    metadata: dict,
) -> None:
    """After a successful confsearch: create optimization tasks for each
    conformer geometry (up to ``max_conformers``)."""

    if not metadata.get("request_optimization", True):
        logger.info("Skipping optimization follow-up for task %d (not requested)", task.id)
        return

    conformer_geoms = session.exec(
        select(MoleculeGeometry)
        .where(MoleculeGeometry.origin_task_id == task.id)
        .order_by(col(MoleculeGeometry.energy).asc())
    ).all()

    max_key = f"max_conformers_{state.description}"
    max_conformers = metadata.get(max_key, 10)
    conformer_geoms = conformer_geoms[:max_conformers]

    opt_header_id = state.optimization_header_id
    if opt_header_id is None:
        logger.error("No optimization header for state %d", state.id)
        return

    for geom in conformer_geoms:
        opt_task = ComputationTask(
            task_type=TaskType.optimization,
            state_id=state.id,
            header_id=opt_header_id,
            input_geometry_id=geom.id,
            depends_on_task_id=task.id,
            has_followups=True,
            status=TaskStatus.created,
        )
        session.add(opt_task)
        logger.info(
            "Created optimization task for geometry %d (state %d)",
            geom.id, state.id,
        )


def _followup_optimization(
    session: Session,
    task: ComputationTask,
    state: MoleculeState,
    metadata: dict,
) -> None:
    """After a successful optimization: create singlepoint tasks and
    vertical excitation variants."""

    output_geom_id = task.output_geometry_id
    if output_geom_id is None:
        logger.error("Optimization task %d has no output geometry", task.id)
        return

    sp_header_id = state.singlepoint_header_id
    if sp_header_id is None:
        logger.error("No singlepoint header for state %d", state.id)
        return

    if not metadata.get("request_singlepoint", True):
        logger.info("Skipping singlepoint follow-up for task %d", task.id)
        return

    # Base singlepoint
    _create_singlepoint_task(
        session, state.id, sp_header_id, output_geom_id,
        task.id, TaskType.singlepoint,
    )

    # Vertical excitation variants
    if not metadata.get("request_singlepoint_vertical_excitations", True):
        return

    desc = state.description

    # The spin-change partner only exists for the singlet/triplet pair. For
    # an open-shell reference (a radical cation/anion submitted as S0, i.e.
    # a doublet) there is no T1 state and no meaningful vertical spin flip,
    # so the task is not created at all rather than run as a duplicate of
    # the base singlepoint.
    spin_change_defined = state.multiplicity in (1, 3)

    if desc == "S0":
        _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_ox)
        _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_red)
        if spin_change_defined:
            _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_spin_change)
        else:
            logger.info(
                "State %d (S0, multiplicity %d) is open-shell; skipping the "
                "vertical spin-change singlepoint", state.id, state.multiplicity,
            )
    elif desc == "S1":
        _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_ox)
        _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_red)
    elif desc == "T1":
        if spin_change_defined:
            _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_spin_change)
    elif desc == "ox":
        _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_red)
    elif desc == "red":
        _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_ox)


def _category_followups(description: str, metadata: dict) -> list[TaskType]:
    """The category tasks a successful optimisation of this state gets, in creation order."""
    followups = []
    if categories.on_s0(description, metadata, categories.UVVIS):
        followups.append(TaskType.singlepoint_uvvis)
    if categories.on_s0(description, metadata, categories.NMR):
        followups.append(TaskType.singlepoint_nmr)
    # ESD states run a TDDFT/SOC singlepoint at their optimised geometry.
    if metadata.get("esd_role"):
        followups.append(TaskType.singlepoint_soc)
    return followups


def _followup_categories(
    session: Session,
    task: ComputationTask,
    state: MoleculeState,
    metadata: dict,
) -> None:
    """Opt-in category tasks of a successful optimisation.

    Created after the energy singlepoints, so they queue behind them.
    """
    if task.output_geometry_id is None or state.singlepoint_header_id is None:
        return
    for task_type in _category_followups(state.description, metadata):
        _create_singlepoint_task(
            session, state.id, state.singlepoint_header_id, task.output_geometry_id,
            task.id, task_type,
        )


def _create_singlepoint_task(
    session: Session,
    state_id: int,
    header_id: int,
    input_geometry_id: int,
    depends_on_task_id: int,
    task_type: TaskType,
) -> ComputationTask:
    """Create a singlepoint-family task (has_followups=False)."""
    task = ComputationTask(
        task_type=task_type,
        state_id=state_id,
        header_id=header_id,
        input_geometry_id=input_geometry_id,
        depends_on_task_id=depends_on_task_id,
        has_followups=False,
        status=TaskStatus.created,
    )
    session.add(task)
    logger.info("Created %s task for state %d", task_type.value, state_id)
    return task


# ======================================================================
# Step 6a -- Create retry jobs for pending tasks with failed attempts
# ======================================================================

def create_retry_jobs(
    session: Session, settings: Settings, qm_engine: QMEngine,
    blocked_owners: Collection[str] = (),
) -> None:
    """For pending tasks with fewer than ``max_attempts`` failed jobs and
    no successful or in-flight jobs: create a new job attempt with
    failure-specific retry strategies applied. Each task that wrote commits on its own."""

    pending_tasks = session.exec(
        select(ComputationTask).where(
            ComputationTask.status == TaskStatus.pending,
        )
    ).all()

    # A retry would recreate an archived molecule's deleted job folders.
    pending_tasks = [t for t in pending_tasks if not _molecule_is_archived(session, t)]

    # Don't create retry jobs for a project being archived/exported.
    paused = paused_project_names(session)
    if paused:
        pending_tasks = [t for t in pending_tasks if _task_project(session, t) not in paused]
    # Nor for an owner whose circuit breaker has tripped.
    pending_tasks = _unblocked(session, pending_tasks, blocked_owners)

    for task_id in [t.id for t in pending_tasks]:
        task = session.get(ComputationTask, task_id)
        if task is not None and task.status == TaskStatus.pending and _retry_task(session, task, settings, qm_engine):
            session.commit()


def _retry_task(session: Session, task: ComputationTask, settings: Settings, qm_engine: QMEngine) -> bool:
    """Create *task*'s next attempt if its current run calls for one; True when it wrote (or claimed)."""
    max_attempts = settings.pipeline.max_attempts
    jobs = session.exec(
        select(ComputationJob).where(ComputationJob.task_id == task.id)
    ).all()

    # Skip if there are still in-flight jobs (success is None)
    if any(j.success is None for j in jobs):
        return False

    current_run = _current_run(task, jobs)
    failed_count = sum(1 for j in current_run if j.success is False)
    success_count = sum(1 for j in current_run if j.success is True)

    # Skip if already succeeded or exhausted
    if success_count > 0 or failed_count >= max_attempts:
        return False

    # Find the most recent failed job (this run) to get failure info
    last_failed = max(
        (j for j in current_run if j.success is False),
        key=lambda j: j.attempt,
        default=None,
    )

    reason = (last_failed.fail_reason or "") if last_failed is not None else ""
    # A functional without third derivatives, an S1 that collapses or
    # dissociates, or a request SLURM can never run fails identically every time.
    if (
        reason == "['Excited Root']" or reason.startswith(UNSATISFIABLE)
        or any(f"'{c}'" in reason for c in DETERMINISTIC_CHECKS)
    ):
        task.status = TaskStatus.failed
        task.updated_at = datetime.now(timezone.utc)
        session.add(task)
        logger.info("Task %d failed deterministically (%s); not retried", task.id, reason)
        return True

    next_attempt = (task.retry_base or 0) + failed_count + 1
    # The paused list was read at the step's start; an archive may have begun since.
    if not _claim_task(session, task, TaskStatus.pending):
        return True  # its UPDATE still opened a write
    # Per task: retry-strategy application reads files from previous job
    # directories, which archive/cleanup may have removed. One raising
    # task must not discard every other task's retry job for the tick.
    if not _build_job(session, task, next_attempt, settings, qm_engine, last_failed):
        return True
    logger.info(
        "Created retry job (attempt %d) for task %d",
        next_attempt, task.id,
    )
    return True


# ======================================================================
# Step 6b -- Start newly created tasks (create their first job)
# ======================================================================

def start_new_tasks(
    session: Session,
    settings: Settings,
    qm_engine: QMEngine,
    blocked_owners: Collection[str] = (),
) -> None:
    """For tasks in ``created`` status: create the initial job, generate
    input files, and transition to ``pending``. Each task that wrote commits on its own."""

    tasks = session.exec(
        select(ComputationTask).where(
            ComputationTask.status == TaskStatus.created,
        ).order_by(ComputationTask.id.asc())  # type: ignore[union-attr]
    ).all()

    # Molecule.archived was set by archive_project but read by nothing in the
    # engine, so archiving a project removed its raw files while its tasks
    # kept being started and submitted -- and _create_job_for_task recreated
    # the very directory tree that had just been archived, then burned the
    # retry budget failing against it.
    tasks = [t for t in tasks if not _molecule_is_archived(session, t)]

    # Don't start new jobs for a project being archived/exported: a job
    # created here would recreate the comp_data tree the archive is deleting.
    paused = paused_project_names(session)
    if paused:
        tasks = [t for t in tasks if _task_project(session, t) not in paused]
    # Nor for an owner whose circuit breaker has tripped.
    tasks = _unblocked(session, tasks, blocked_owners)

    for task_id in [t.id for t in tasks]:
        task = session.get(ComputationTask, task_id)
        if task is not None and _start_task(session, task, settings, qm_engine):
            session.commit()


def _start_task(session: Session, task: ComputationTask, settings: Settings, qm_engine: QMEngine) -> bool:
    """Build *task*'s first job and mark it ``pending``; True when it wrote (or claimed)."""
    if not _claim_task(session, task, TaskStatus.created):
        return True  # an archive began since the paused list was read; its UPDATE still opened a write
    # Ensure task directory exists
    if not task.task_path:
        state = session.get(MoleculeState, task.state_id)
        if state is None:
            return True
        # Created with the job folder, inside its rollback.
        task_dir = (
            settings.comp_data_path
            / f"mol_{state.molecule_id}"
            / f"state_{state.id}_{state.description}"
            / "tasks"
            / f"{task.id}_{task.task_type.value}"
        )
        task.task_path = str(task_dir)

    # Per task, for the same reason as create_retry_jobs: one task that
    # raises here used to discard the whole step for every molecule.
    if not _build_job(session, task, (task.retry_base or 0) + 1, settings, qm_engine):
        return True

    task.status = TaskStatus.pending
    task.updated_at = datetime.now(timezone.utc)
    session.add(task)
    logger.info("Task %d started (type=%s)", task.id, task.task_type.value)
    return True


# ======================================================================
# Step 7 -- Submit pending (unsubmitted) jobs to the scheduler
# ======================================================================

# Submissions between squeue polls inside one tick. Small enough that jobs
# SLURM starts immediately stop counting against the cap quickly, large
# enough that a few hundred submissions do not become a few hundred
# subprocesses.
_QUEUE_RECHECK_EVERY = 5


def _pending_breakdown(scheduler: Scheduler) -> tuple[int, int]:
    """``(waiting_for_the_cluster, awaiting_the_scheduler)`` from *scheduler*.

    Falls back to treating every pending job as waiting for schedulers that
    cannot distinguish the two -- the conservative reading, and what the
    local test scheduler does.
    """
    getter = getattr(scheduler, "get_pending_breakdown", None)
    if getter is None:
        return scheduler.get_queue_length(), 0
    return getter()


def submit_pending_jobs(
    session: Session, scheduler: Scheduler, settings: Settings,
    blocked_owners: Collection[str] = (),
) -> None:
    """Find jobs that have been created (no slurm_jobid, no slurm_status)
    and submit them to the scheduler.

    Submission keeps going until the SLURM queue holds
    ``priority * queue_slots_per_priority`` of *our own* waiting jobs, where
    *priority* comes from the entrypoint that created the molecule. This is
    the only throttle in the system: the REST API accepts every submission
    immediately, entrypoints buffer in the database, and the queue depth is
    regulated here rather than at the door.

    The cap counts jobs that are *waiting*. A job SLURM starts straight away
    is running, not queued, and must stop counting against it: submission
    keeps going until squeue actually reports that many of our jobs in PD.
    On an idle partition that means the loop runs until the cluster is full
    and jobs finally begin to queue, which is the point.

    One tick is bounded by wall clock rather than by a job count, so a long
    submission run cannot starve status polling -- it yields and resumes on
    the next tick with the queue state it left behind.
    """

    from autodft.engine import submission_hold

    _release_claims(session)
    _resolve_paused_sbatch_failures(session, scheduler)
    held = submission_hold.read_state(settings.data_path)
    if held is not None:
        logger.info("Submission hold active (%s) -- not submitting jobs",
                    held.get("reason") or "no reason given")
        return

    # squeue counts our own waiting (PD) jobs only, so running work never
    # counts against the cap -- the limit controls how deep the backlog is
    # allowed to get, not how much may run.
    queue_len, unscheduled = _pending_breakdown(scheduler)
    if queue_len < 0:
        # -1 means squeue failed. Submitting blind would flood the partition,
        # so skip this tick; the jobs stay pending and go out on the next one.
        logger.warning("Queue length unknown; deferring submissions to the next tick")
        return

    slots = max(1, settings.pipeline.queue_slots_per_priority)

    # Ordered by priority descending so the scarce queue slots go to the
    # highest-priority work first. Because the cap scales with priority,
    # the first job that doesn't fit ends the loop: everything after it has
    # an equal or lower priority, hence an equal or lower cap.
    # Within a priority, jobs that unlock follow-ups (confsearch, optimisation,
    # ESD) go before singlepoints, so a retry never waits behind older singlepoints.
    singlepoint_types = [t for t in TaskType if t.value.startswith("singlepoint")]
    statement = (
        select(ComputationJob, col(Molecule.priority))
        .join(ComputationTask, col(ComputationTask.id) == col(ComputationJob.task_id))
        .join(MoleculeState, col(MoleculeState.id) == col(ComputationTask.state_id))
        .join(Molecule, col(Molecule.id) == col(MoleculeState.molecule_id))
        .where(
            col(ComputationJob.slurm_jobid).is_(None),
            col(ComputationJob.slurm_status).is_(None),
            col(ComputationJob.success).is_(None),
            col(Molecule.archived).is_(False),
        )
        .order_by(
            col(Molecule.priority).desc(),
            case((col(ComputationTask.task_type).in_(singlepoint_types), 1), else_=0),
            col(ComputationJob.id).asc(),
        )
    )
    # Don't submit a paused project's jobs: an archive/export is in flight and
    # a running calculation would race its file operations.
    paused = paused_project_names(session)
    if paused:
        statement = statement.where(col(Molecule.project_name).notin_(sorted(paused)))
    # Nor those of an owner whose circuit breaker has tripped.
    if blocked_owners:
        statement = statement.where(circuit_breaker.not_owned_by(session, blocked_owners))
    # Backstop only, disabled by default: the priority cap above is the
    # real throttle. A count limit here used to bind first and cap the fill
    # rate at one limit-full per tick.
    limit = settings.pipeline.max_submissions_per_tick
    if limit and limit > 0:
        statement = statement.limit(limit)
    rows = session.exec(statement).all()

    deadline = (
        monotonic() + max(1.0, settings.pipeline.max_submission_seconds_per_tick)
    )

    # A job that can't be submitted must be recorded as a failed attempt, not
    # skipped. Skipping left `success` NULL forever, so the retry path ignored
    # it, the task stayed `pending`, and the same warning was logged every
    # tick in perpetuity. Recording the failure lets max_attempts do its job.
    # Committed alone: a later commit must not carry it with a real submission.
    def _fail(job: ComputationJob, reason: str) -> None:
        logger.error("Job %d: %s", job.id, reason)
        job.success = False
        job.slurm_status = None
        job.fail_reason = reason[:500]
        job.time_end = datetime.now(timezone.utc)
        session.add(job)
        session.commit()

    # Between polls the count is incremented optimistically, which assumes
    # every submitted job sits in the queue. On an idle partition none of
    # them does -- they start immediately -- so re-reading the real waiting
    # count is what lets the loop keep going and actually fill the cluster.
    # One squeue per _QUEUE_RECHECK_EVERY submissions, not per submission:
    # the optimistic increment bounds the overshoot in between.
    since_poll = 0
    sbatch_calls = sbatch_failures = 0
    rejected: list[tuple[ComputationJob, str]] = []
    # Ids now: a commit below expires the rows, and a reset-task may delete one.
    job_ids = [job.id for job, _ in rows]

    for index, ((job, priority), job_id) in enumerate(zip(rows, job_ids)):
        if monotonic() >= deadline:
            logger.info(
                "Submission budget for this tick spent after %d job(s); "
                "%d left for the next one", index, len(rows) - index,
            )
            break

        if since_poll >= _QUEUE_RECHECK_EVERY:
            since_poll = 0
            fresh, unscheduled = _pending_breakdown(scheduler)
            if fresh >= 0:
                queue_len = fresh

        cap = max(1, priority) * slots
        if queue_len >= cap:
            logger.info(
                "%d job(s) waiting for the cluster (+%d not yet scheduled); "
                "cap for priority %d is %d -- stopping submission "
                "(%d job(s) left for a later tick)",
                queue_len, unscheduled, priority, cap, len(rows) - index,
            )
            break

        if not _claim(session, job_id):
            continue  # deleted or judged since the tick began (admin reset-task)

        if job.job_path is None:
            _fail(job, "No job_path recorded; input files were never generated")
            continue

        submit_script = Path(job.job_path) / "submit.cmd"
        if not submit_script.exists():
            _fail(job, f"Submit script missing: {submit_script}")
            continue

        adopted = None
        # Any failed sbatch may still have been accepted.
        if (job.fail_reason or "").startswith(SBATCH_FAILED):
            try:
                adopted = scheduler.find_submitted(submit_script)
            except Exception:  # noqa: BLE001 - resubmitting blind could run the job twice
                logger.exception("Job %d: SLURM lookup of an earlier sbatch failed; "
                                 "deferring to the next tick", job.id)
                job.slurm_status = None
                session.add(job)
                continue

        if adopted is not None:
            logger.warning("Job %d: adopting SLURM job %s from an sbatch that reported failure",
                           job.id, adopted)
            result = SubmitResult(success=True, job_id=adopted)
        else:
            result = scheduler.submit(submit_script, nice=settings.slurm.nice)
            sbatch_calls += 1
            sbatch_failures += not result.success
        if result.success:
            job.slurm_jobid = int(result.job_id)  # type: ignore[arg-type]
            job.slurm_status = SlurmStatus.PENDING
            job.time_start = datetime.now(timezone.utc)
            job.fail_reason = None
            session.add(job)
            # Commit immediately: sbatch has already run, so a later rollback
            # of this step would leave a real SLURM job with no row pointing
            # at it, and the next tick would re-select the row (slurm_jobid
            # IS NULL) and submit a second ORCA process into the same
            # directory -- two processes writing one output.out.
            session.commit()
            queue_len += 1
            since_poll += 1
            logger.info("Job %d submitted (slurm_jobid=%s)", job.id, result.job_id)
        elif result.rejected:
            rejected.append((job, result.error or ""))
        else:
            # sbatch failures are usually transient -- slurmctld restarting,
            # a socket timeout, a QOS submit limit that clears. Failing the
            # job here would burn a whole attempt (three ticks, 90 s, and the
            # task is dead) for a calculation that never ran. Record the
            # reason for visibility but leave success NULL so the next tick
            # retries; the per-tick submission cap bounds the retry rate.
            logger.error("Job %d: sbatch failed (will retry): %s", job.id, result.error)
            job.slurm_status = None
            job.fail_reason = f"{_SBATCH_RETRYING}: {result.error}"[:500]
            session.add(job)

    # Every sbatch of the tick failing points at the cluster, not at the scripts.
    outage = sbatch_calls >= 2 and sbatch_failures == sbatch_calls
    if outage:
        logger.warning("All %d sbatch calls failed this tick (every sbatch); "
                       "taken as a SLURM outage, no rejection is counted", sbatch_calls)
    for job, error in rejected:
        job.slurm_status = None
        # A rejection repeats on the same script, so a few cost an attempt.
        previous = _SBATCH_REJECTED.match(job.fail_reason or "")
        rejections = (int(previous.group(1)) if previous else 0) + (not outage)
        if rejections >= MAX_SBATCH_REJECTIONS:
            _fail(job, f"sbatch rejected {rejections} times: {error}")
            continue
        if rejections:
            logger.error("Job %d: sbatch rejected (%d/%d, will retry): %s",
                         job.id, rejections, MAX_SBATCH_REJECTIONS, error)
            job.fail_reason = (
                f"{SBATCH_FAILED} (rejected {rejections}/{MAX_SBATCH_REJECTIONS}, "
                f"retrying): {error}"
            )[:500]
        else:
            job.fail_reason = f"{_SBATCH_RETRYING}: {error}"[:500]
        session.add(job)

    session.flush()


def _claim(session: Session, job_id: int) -> bool:
    """Mark an unsent job SUBMITTING and commit; False when it is gone, no longer unsent or paused."""
    claimed = session.exec(
        update(ComputationJob)
        .where(
            col(ComputationJob.id) == job_id,
            col(ComputationJob.slurm_jobid).is_(None),
            col(ComputationJob.slurm_status).is_(None),
            col(ComputationJob.success).is_(None),
            _unpaused(ComputationJob.task_id),
        )
        .values(slurm_status=SlurmStatus.SUBMITTING.value)
        .execution_options(synchronize_session=False)
    ).rowcount
    session.commit()
    return bool(claimed)


def _resolve_paused_sbatch_failures(session: Session, scheduler: Scheduler) -> None:
    """Look up a paused project's jobs an sbatch that reported failure may have got accepted.

    The pause refuses their claim, so the adoption below never runs for them; the archive
    waits for these. Found: adopted. Not found: no longer possibly sent. Lookup failed: left.
    """
    paused = paused_project_names(session)
    if not paused:
        return
    jobs = session.exec(
        select(ComputationJob)
        .join(ComputationTask, col(ComputationTask.id) == col(ComputationJob.task_id))
        .join(MoleculeState, col(MoleculeState.id) == col(ComputationTask.state_id))
        .join(Molecule, col(Molecule.id) == col(MoleculeState.molecule_id))
        .where(
            col(Molecule.project_name).in_(sorted(paused)),
            col(ComputationJob.slurm_jobid).is_(None),
            col(ComputationJob.slurm_status).is_(None),
            col(ComputationJob.success).is_(None),
            col(ComputationJob.fail_reason).startswith(SBATCH_FAILED),
        )
    ).all()
    for job in jobs:
        try:
            adopted = (scheduler.find_submitted(Path(job.job_path) / "submit.cmd")
                       if job.job_path else None)
        except Exception:  # noqa: BLE001 - left possibly sent; the archive keeps waiting
            logger.exception("Job %d: SLURM lookup of an earlier sbatch failed", job.id)
            continue
        if adopted is not None:
            logger.warning("Job %d: adopting SLURM job %s from an sbatch that reported failure",
                           job.id, adopted)
            job.slurm_jobid = int(adopted)
            job.slurm_status = SlurmStatus.PENDING
            job.time_start = datetime.now(timezone.utc)
            job.fail_reason = None
        else:
            job.fail_reason = f"{_NOT_ACCEPTED} {job.fail_reason}"[:500]
        session.add(job)
        session.commit()


def _unpaused(task_id: ColumnElement[int] | int) -> ColumnElement[bool]:
    """SQL: task *task_id*'s project has no archive/export job running."""
    return ~exists().where(
        col(ComputationTask.id) == task_id,
        col(MoleculeState.id) == col(ComputationTask.state_id),
        col(Molecule.id) == col(MoleculeState.molecule_id),
        col(ProjectJob.qualified_name) == col(Molecule.project_name),
        ProjectJob.status == ProjectJobStatus.running,
    )


def _claim_task(session: Session, task: ComputationTask, status: TaskStatus) -> bool:
    """Take the write lock for building *task*'s job; False when it left *status* or its project is paused."""
    return bool(session.exec(
        update(ComputationTask)
        .where(col(ComputationTask.id) == task.id, ComputationTask.status == status,
               _unpaused(task.id))
        .values(updated_at=datetime.now(timezone.utc))
        .execution_options(synchronize_session=False)
    ).rowcount)


def _release_claims(session: Session) -> None:
    """Claims a crash or a failed step left: the next submission looks them up in SLURM first."""
    for job in session.exec(
        select(ComputationJob).where(
            ComputationJob.slurm_status == SlurmStatus.SUBMITTING.value,
            col(ComputationJob.slurm_jobid).is_(None),
            col(ComputationJob.success).is_(None),
        )
    ).all():
        logger.warning("Job %d: claimed for sbatch by an interrupted tick; looking it up first", job.id)
        job.slurm_status = None
        if not (job.fail_reason or "").startswith(SBATCH_FAILED):
            job.fail_reason = f"{_SBATCH_RETRYING}: interrupted during submission"
        session.add(job)
    session.flush()


# ======================================================================
# Helpers -- job creation & input file generation
# ======================================================================

def _current_run(task: ComputationTask, jobs: list[ComputationJob]) -> list[ComputationJob]:
    """*task*'s jobs since its last requeue; earlier runs no longer count."""
    base = task.retry_base or 0
    return [j for j in jobs if j.attempt > base]


def _molecule_is_archived(session: Session, task: ComputationTask) -> bool:
    """True when the task's molecule has been archived (raw files removed)."""
    from autodft.models.molecule import Molecule

    state = session.get(MoleculeState, task.state_id)
    if state is None:
        return False
    molecule = session.get(Molecule, state.molecule_id)
    return bool(molecule is not None and molecule.archived)


def paused_project_names(session: Session) -> set[str]:
    """Projects with a background archive/export job in flight.

    While such a job runs, the controller must not advance that project's
    tasks -- create followups, start or retry jobs, or submit them. A
    destructive archive is deleting the project's comp_data tree and a running
    calculation writing into it (or a new job recreating it) would race the
    ``rmtree``. The set is normally empty, so the per-task filtering below is
    skipped entirely in the common case.
    """
    return set(
        session.exec(
            select(ProjectJob.qualified_name).where(
                ProjectJob.status == ProjectJobStatus.running
            )
        ).all()
    )


def _unblocked(
    session: Session, tasks: list[ComputationTask], blocked_owners: Collection[str],
) -> list[ComputationTask]:
    """*tasks* without those of an owner whose circuit breaker has tripped."""
    if not blocked_owners:
        return tasks
    return [t for t in tasks
            if circuit_breaker.project_owner(_task_project(session, t)) not in blocked_owners]


def _task_project(session: Session, task: ComputationTask) -> Optional[str]:
    """The project name a task belongs to, via its state's molecule."""
    state = session.get(MoleculeState, task.state_id)
    if state is None:
        return None
    molecule = session.get(Molecule, state.molecule_id)
    return molecule.project_name if molecule is not None else None


def _build_job(
    session: Session,
    task: ComputationTask,
    attempt: int,
    settings: Settings,
    qm_engine: QMEngine | None,
    previous_failed_job: Optional[ComputationJob] = None,
) -> bool:
    """Build *task*'s job *attempt*; True on success.

    A build that raises costs no attempt but counts towards
    :data:`MAX_BUILD_FAILURES`, after which the task fails with the last error.
    A :class:`RetryWouldRepeat` fails it at once.
    """
    try:
        _create_job_for_task(session, task, attempt, settings,
                             qm_engine=qm_engine, previous_failed_job=previous_failed_job)
    except Exception as exc:  # noqa: BLE001 - one bad task must not stop the rest
        reason = None
        if isinstance(exc, RetryWouldRepeat):
            reason = str(exc)
        else:
            logger.exception("Failed to build job (attempt %d) for task %d", attempt, task.id)
            task.build_failures = (task.build_failures or 0) + 1
            if task.build_failures >= MAX_BUILD_FAILURES:
                reason = f"Job build failed {task.build_failures} times: {type(exc).__name__}: {exc}"
        if reason is not None:
            # The reason lives on a folderless failed job, as for _fail_task.
            session.add(ComputationJob(
                task_id=task.id, attempt=attempt, success=False,
                fail_reason=reason[:500], time_end=datetime.now(timezone.utc),
            ))
            task.status = TaskStatus.failed
            task.build_failures = 0
            logger.error("Task %d: %s", task.id, reason)
        task.updated_at = datetime.now(timezone.utc)
        session.add(task)
        return False
    if task.build_failures:
        task.build_failures = 0
        session.add(task)
    return True


def _create_job_for_task(
    session: Session,
    task: ComputationTask,
    attempt: int,
    settings: Settings,
    qm_engine: QMEngine | None = None,
    previous_failed_job: Optional[ComputationJob] = None,
) -> ComputationJob:
    """Create a :class:`ComputationJob` record for *task* and optionally
    generate its input files via *qm_engine*.

    All or nothing: if anything raises, the row is rolled back and the job
    folder removed before the exception propagates.

    Args:
        previous_failed_job: The most recent failed job for this task,
            used to apply failure-specific retry strategies.
    """

    # Determine job directory
    task_path = Path(task.task_path) if task.task_path else None
    if task_path is None:
        logger.error("Task %d has no task_path", task.id)
        raise ValueError(f"Task {task.id} has no task_path")

    # pysqlite sends no BEGIN for a SAVEPOINT, whose RELEASE would then commit
    # the job on its own; this write opens the real transaction first.
    task.updated_at = datetime.now(timezone.utc)
    session.add(task)
    session.flush()

    job_dir: Optional[Path] = None
    try:
        with session.begin_nested():
            job = ComputationJob(
                task_id=task.id,
                attempt=attempt,
            )
            session.add(job)
            session.flush()  # get job.id

            job_dir = task_path / f"job_{job.id}"
            job_dir.mkdir(parents=True, exist_ok=True)
            job.job_path = str(job_dir)
            session.add(job)

            # Generate input files if qm_engine is available
            if qm_engine is not None:
                _generate_job_files(
                    session, job, task, settings, qm_engine, attempt,
                    previous_failed_job=previous_failed_job,
                )

            session.flush()
    except BaseException:
        if job_dir is not None:
            shutil.rmtree(job_dir, ignore_errors=True)
        raise
    return job


def _restart_geometry(failed: ComputationJob, seed_xyz: str) -> str:
    """Where a retried optimisation starts: the failed attempt's last geometry.

    ORCA rewrites ``input.xyz`` every cycle and the job script copies it back,
    so a crash or stall resumes where it stopped instead of from the seed. A
    collapsed S1 (``Excited Root``), an SCF that failed at a possibly runaway
    geometry, a saddle point (``Imaginary Frequencies``: re-converging from it
    finds it again; the perturbation retry reads it itself) or an unreadable
    file keeps the seed.
    """
    reason = failed.fail_reason or ""
    if any(key in reason for key in ("Excited Root", "SCF Convergence", "Imaginary")) or not failed.job_path:
        return seed_xyz
    try:
        last = (Path(failed.job_path) / "input.xyz").read_text(encoding="utf-8")
    except OSError:
        return seed_xyz
    seed_atoms, last_atoms = _xyz_atoms(seed_xyz), _xyz_atoms(last)
    if not last_atoms or last_atoms != seed_atoms:
        return seed_xyz
    return last


def _xyz_atoms(xyz_data: str) -> Optional[list[str]]:
    """The element sequence of an XYZ block with numeric coordinates, else None."""
    lines = xyz_data.strip().splitlines()
    try:
        count = int(lines[0])
        rows = [line.split() for line in lines[2:2 + count]]
        if len(rows) != count or any(len(r) < 4 for r in rows):
            return None
        for r in rows:
            [float(v) for v in r[1:4]]
    except (ValueError, IndexError):
        return None
    return [r[0] for r in rows]


class JobInputError(ValueError):
    """Why a job's input cannot be built."""


class RetryWouldRepeat(Exception):
    """A retry would rerun the failed job unchanged; its task fails at once."""


class _RenderedJob(NamedTuple):
    input_text: str
    submit_text: str
    # The %maxcore --mem was sized from: the header's, else the stage's.
    mem_per_core: int
    # A rate job's inputs_json record and Hessians (file name -> source); None otherwise.
    rate_record: Optional[dict] = None
    hessians: Optional[dict[str, Path]] = None


def render_job_input(
    session: Session, settings: Settings, task: ComputationTask, state: MoleculeState, multiplicity: int,
    *, job: Optional[ComputationJob] = None, qm_engine: Optional[QMEngine] = None,
) -> tuple[str, str]:
    """(input.inp, submit.cmd) a fresh job of *task* gets; writes nothing.

    Without *job* the submit script names a placeholder job directory.
    Raises :class:`JobInputError` when the input cannot be built.
    """
    rendered = _render_job(session, settings, task, state, multiplicity, job, qm_engine)
    return rendered.input_text, rendered.submit_text


def _render_job(
    session: Session, settings: Settings, task: ComputationTask, state: MoleculeState, multiplicity: int,
    job: Optional[ComputationJob], qm_engine: Optional[QMEngine],
) -> _RenderedJob:
    qm_engine = qm_engine or OrcaParser(settings.orca)
    header = session.get(ComputationHeader, task.header_id)
    if header is None:
        raise JobInputError(f"Header {task.header_id} not found")
    header_text = header.header_text
    try:
        charge, _ = _get_job_charge_multiplicity(task.task_type, state)
    except ValueError as exc:
        raise JobInputError(f"Cannot resolve charge/multiplicity: {exc}") from exc

    metadata = json.loads(state.metadata_json) if state.metadata_json else {}
    if "advanced" in metadata:
        # Advanced settings render onto the method-only header before anything composes it.
        from autodft.qm.orca.advanced import render_settings, stage_for
        from autodft.qm.orca.blocks import is_tight_optimization

        stage = stage_for(task.task_type.value)
        floor = "Tight" if is_tight_optimization(task.task_type.value, metadata) else None
        header_text = render_settings(header_text, metadata["advanced"][stage], stage, geom_floor=floor)

    record: Optional[dict] = None
    hessians: Optional[dict[str, Path]] = None
    if task.task_type.value.startswith("esd_"):
        # Rate jobs are built from other tasks' results and Hessians.
        from autodft.engine.photophysics import rate_job_input
        from autodft.qm.orca.esd_inputs import EsdInputError

        try:
            header_text, record, hessians = rate_job_input(session, task, state, header_text)
        except (OSError, KeyError, ValueError) as exc:  # EsdInputError is a ValueError
            detail = str(exc) if isinstance(exc, EsdInputError) else f"{type(exc).__name__}: {exc}"
            raise JobInputError(f"ESD inputs: {detail}") from exc
    else:
        # Category tasks add their own blocks; every other type runs the
        # header verbatim.
        from autodft.qm.orca.blocks import HeaderConflict, compose_header

        if _runs_nbo(state, task):
            unavailable = categories.nbo_unavailable(settings)
            if unavailable:
                raise JobInputError(unavailable)
        try:
            header_text = compose_header(
                task.task_type.value, header_text, metadata, multiplicity=multiplicity,
            )
        except HeaderConflict as exc:
            raise JobInputError(str(exc)) from exc

    trailer = ""
    if task.task_type == TaskType.singlepoint_nmr:
        from autodft.qm.orca.blocks import nmr_trailer

        trailer = nmr_trailer(metadata)
    input_text = qm_engine.render_input(header_text, charge, multiplicity, trailer)

    # --- Gap 3 fix: Parse resources from header, fall back to config ---
    nprocs, mem_per_core = _parse_resources_from_header(header_text)
    computed = record.get("computed") if record is not None else None
    stage_config = _get_stage_config(settings, task.task_type, state, computed)
    if nprocs is None:
        nprocs = stage_config.default_nprocs
    if mem_per_core is None:
        mem_per_core = stage_config.default_mem_per_core

    if job is not None:
        job_path, job_name = Path(job.job_path), f"autodft_{job.id}_{task.task_type.value}"
    else:
        job_path, job_name = Path(task.task_path or ".") / "job_N", f"autodft_N_{task.task_type.value}"
    submit_text = qm_engine.render_submit_script(
        job_path=job_path,
        job_name=job_name,
        nprocs=nprocs,
        mem_per_core=mem_per_core,
        time_limit=stage_config.time_limit,
        partition=settings.slurm.partition,
        nice=settings.slurm.nice,
        keep_hessian=_keeps_hessian(state, task),
        keep_cubes=_keeps_cubes(state, task),
        runs_nbo=_runs_nbo(state, task),
        keep_wavefunction=_keeps_wavefunction(state, task),
        # Its log names how much memory per rank an MP2 response lacked.
        keep_mp2resp=task.task_type == TaskType.singlepoint_nmr,
        keep_esd_files=task.task_type.value.startswith("esd_"),
        extra_inputs=list(hessians) if hessians is not None else None,
    )
    return _RenderedJob(input_text, submit_text, mem_per_core, record, hessians)


def _generate_job_files(
    session: Session,
    job: ComputationJob,
    task: ComputationTask,
    settings: Settings,
    qm_engine: QMEngine,
    attempt: int,
    previous_failed_job: Optional[ComputationJob] = None,
) -> None:
    """Generate QM input + submit script for a job.

    Handles:
    - Correct charge/multiplicity for vertical excitation tasks
    - Resource parsing from ORCA headers (nprocs, maxcore)
    - Failure-specific retry strategies on attempts > 1
    """
    job_path = Path(job.job_path)

    # Fetch state for base charge / multiplicity
    state = session.get(MoleculeState, task.state_id)
    if state is None:
        return _fail_task(session, task, job, f"State {task.state_id} not found")

    # --- Gap 2 fix: Adjust charge/multiplicity for vertical excitations ---
    # A task whose charge/multiplicity can't be resolved must not silently
    # stall in `created` forever — fail it so it shows up in the dashboard.
    try:
        charge, multiplicity = _get_job_charge_multiplicity(task.task_type, state)
    except ValueError as exc:
        return _fail_task(session, task, job,
                          f"Cannot resolve charge/multiplicity: {exc}")

    try:
        rendered = _render_job(session, settings, task, state, multiplicity, job, qm_engine)
    except JobInputError as exc:
        return _fail_task(session, task, job, str(exc))

    # Fetch input geometry
    geom = session.get(MoleculeGeometry, task.input_geometry_id)
    if geom is None:
        return _fail_task(session, task, job, f"Input geometry {task.input_geometry_id} not found")

    xyz_data = geom.xyz_data
    if task.task_type == TaskType.optimization and previous_failed_job is not None:
        xyz_data = _restart_geometry(previous_failed_job, xyz_data)

    if rendered.hessians is not None:
        from autodft.engine.photophysics import copy_rate_hessians

        try:
            copy_rate_hessians(rendered.hessians, job_path)
        except OSError as exc:
            return _fail_task(session, task, job, f"ESD inputs: {type(exc).__name__}: {exc}")
        task.inputs_json = json.dumps(rendered.rate_record)
        session.add(task)

    job_path.mkdir(parents=True, exist_ok=True)
    (job_path / "input.inp").write_text(rendered.input_text, encoding="utf-8")
    (job_path / "input.xyz").write_text(xyz_data, encoding="utf-8")
    (job_path / "submit.cmd").write_text(rendered.submit_text, encoding="utf-8")

    # --- Gap 1 fix: Apply failure-specific retry strategies on retries ---
    if attempt > 1 and previous_failed_job is not None:
        _apply_retry_modifications(
            session, job_path, task, previous_failed_job, attempt,
            charge, multiplicity, settings, default_maxcore=rendered.mem_per_core,
        )


def _fail_task(
    session: Session, task: ComputationTask, job: ComputationJob, reason: str,
) -> None:
    """Mark a task and its job failed when input files cannot be generated.

    Setting only `task.status` here does not survive: start_new_tasks()
    overwrites it with `pending` immediately after this returns. Failing the
    *job* is what actually sticks, and it makes the task visible through the
    normal attempt accounting instead of leaving it stalled in `created`
    forever with nothing ever revisiting it.
    """
    logger.error("Task %d: %s", task.id, reason)
    job.success = False
    job.fail_reason = reason[:500]
    job.time_end = datetime.now(timezone.utc)
    session.add(job)
    session.flush()


def _get_job_charge_multiplicity(
    task_type: TaskType, state: MoleculeState,
) -> tuple[int, int]:
    """Return the correct (charge, multiplicity) for a job.

    For vertical excitation tasks, the charge/multiplicity differs from
    the state's base values:
    - vert_ox: charge + 1, altered multiplicity
    - vert_red: charge - 1, altered multiplicity
    - vert_spin_change: same charge, flip singlet <-> triplet
    """
    charge = state.charge
    multiplicity = state.multiplicity

    if task_type == TaskType.singlepoint_vert_ox:
        new_charge = charge + 1
        from autodft.engine.entrypoint_processor import calculate_altered_multiplicity
        return new_charge, calculate_altered_multiplicity(multiplicity, charge, new_charge)

    elif task_type == TaskType.singlepoint_vert_red:
        new_charge = charge - 1
        from autodft.engine.entrypoint_processor import calculate_altered_multiplicity
        return new_charge, calculate_altered_multiplicity(multiplicity, charge, new_charge)

    elif task_type == TaskType.singlepoint_vert_spin_change:
        # Flip singlet <-> triplet. Only these two are defined: the task is
        # the vertical partner of the S0/T1 pair. Returning the state's own
        # multiplicity for anything else (the previous behaviour) produced a
        # singlepoint byte-identical to the base one — it succeeded, cost a
        # full singlepoint per conformer, and made the derived reorganisation
        # energy a mislabelled adiabatic gap. Task creation is gated on the
        # same condition, so reaching this branch means the DB holds a task
        # that should not exist.
        if multiplicity == 1:
            return charge, 3
        if multiplicity == 3:
            return charge, 1
        raise ValueError(
            f"vert_spin_change is only defined for a singlet or triplet state; "
            f"state {state.id} has multiplicity {multiplicity}"
        )

    # SOC TDDFT and every ESD rate job run from the closed-shell singlet
    # reference, also on the T1 state.
    if task_type == TaskType.singlepoint_soc or task_type.value.startswith("esd_"):
        return charge, 1

    return charge, multiplicity


def _parse_resources_from_header(
    header_text: str,
) -> tuple[Optional[int], Optional[int]]:
    """Extract nprocs and maxcore from an ORCA header.

    Looks for ``%pal nprocs N end`` and ``%maxcore N`` patterns.

    Returns:
        Tuple of ``(nprocs, mem_per_core)``; either may be ``None``
        if not found in the header.
    """
    import re

    nprocs = None
    mem_per_core = None

    # Parse %pal block for nprocs
    pal_match = re.search(r"%pal(.*?)end", header_text, re.DOTALL | re.IGNORECASE)
    if pal_match:
        nprocs_match = re.search(r"nprocs\s+(\d+)", pal_match.group(1), re.IGNORECASE)
        if nprocs_match:
            nprocs = int(nprocs_match.group(1))

    # ORCA also accepts the route-line shorthand `! ... PAL16`. Without this
    # the header was read as "no core count given", nprocs fell back to the
    # per-stage config default, and a `! PAL32` job was allocated 16 cores
    # while ORCA spawned 32 ranks -- oversubscription plus a matching memory
    # under-request, with nothing warning about it.
    if nprocs is None:
        for line in header_text.splitlines():
            if not line.lstrip().startswith("!"):
                continue
            pal_short = re.search(r"\bPAL(\d+)\b", line, re.IGNORECASE)
            if pal_short:
                nprocs = int(pal_short.group(1))
                break

    # Parse %maxcore
    maxcore_match = re.search(r"%maxcore\s+(\d+)", header_text, re.IGNORECASE)
    if maxcore_match:
        mem_per_core = int(maxcore_match.group(1))

    return nprocs, mem_per_core


def _job_resources(input_text: str, submit_text: str) -> tuple[Optional[str], ...]:
    """(%maxcore, %pal nprocs, --ntasks-per-node, --mem) of a job's files."""
    import re

    found = (re.search(r"%maxcore\s+(\d+)", input_text, re.IGNORECASE),
             re.search(r"%pal\s+nprocs\s+(\d+)", input_text, re.IGNORECASE),
             re.search(r"#SBATCH --ntasks-per-node=(\d+)", submit_text),
             re.search(r"#SBATCH --mem=(\d+)", submit_text))
    return tuple(m.group(1) if m else None for m in found)


def _job_dir_resources(job_path: Optional[str]) -> Optional[tuple[Optional[str], ...]]:
    """:func:`_job_resources` of a job folder; None when its files can't be read."""
    if not job_path:
        return None
    try:
        return _job_resources((Path(job_path) / "input.inp").read_text(encoding="utf-8"),
                              (Path(job_path) / "submit.cmd").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return None


def _apply_retry_modifications(
    session: Session,
    job_path: Path,
    task: ComputationTask,
    failed_job: ComputationJob,
    attempt: int,
    charge: int,
    multiplicity: int,
    settings: Settings,
    default_maxcore: int = 1000,
) -> None:
    """Read the generated input/submit files and apply retry strategies.

    *default_maxcore* is the %maxcore that sized --mem for an input without one.
    """
    from autodft.qm.orca.retry import (
        FailureInfo,
        PerturbImaginaryMode,
        apply_retry_strategies,
        build_strategies,
    )

    input_file = job_path / "input.inp"
    submit_file = job_path / "submit.cmd"

    if not input_file.exists() or not submit_file.exists():
        logger.warning("Cannot apply retry strategies: files missing in %s", job_path)
        return

    input_content = input_file.read_text(encoding="utf-8")
    submit_content = submit_file.read_text(encoding="utf-8")

    # Replay every previous failure of this task, oldest first, not just the
    # most recent one. input.inp and submit.cmd are regenerated from the
    # header on every attempt, so applying only the latest failure's
    # strategies silently reverted earlier ones: a job that died on
    # 'Termination' (attempt 1 -> more cores) and then on 'Optimization
    # Convergence' (attempt 2) was rebuilt at the *original* core count for
    # attempt 3, dropping the resource increase on the attempt that needed
    # it most.
    prior_failures = session.exec(
        select(ComputationJob)
        .where(
            ComputationJob.task_id == task.id,
            col(ComputationJob.success).is_(False),
            ComputationJob.attempt > (task.retry_base or 0),
        )
        .order_by(col(ComputationJob.attempt).asc())
    ).all()
    if not prior_failures:
        prior_failures = [failed_job]
    # Missing cubes are a plotting hiccup: rerun unchanged, never escalate.
    prior_failures = [p for p in prior_failures if p.fail_reason != str([DENSITY_CUBES])]
    if not prior_failures:
        return

    all_strategies = build_strategies(settings)
    # Geometry perturbation is NOT replayable: it rewrites *xyzfile into an
    # inline *xyz block, so a second application silently no-ops, and running
    # it over the oldest failure first means the geometry that survives comes
    # from attempt 1 -- discarding the re-optimised structure the later
    # attempt produced. It reads files from the failed job's directory too,
    # which older attempts may no longer have. So: replay the text-only
    # strategies over every prior failure, then apply perturbation once,
    # against the most recent failure.
    replayable = [s for s in all_strategies if not isinstance(s, PerturbImaginaryMode)]
    perturbations = [s for s in all_strategies if isinstance(s, PerturbImaginaryMode)]

    applied: list[str] = []
    for prior in prior_failures:
        unchanged = (input_content, submit_content)
        failure = FailureInfo(
            fail_reason=prior.fail_reason or "",
            previous_job_path=prior.job_path or "",
            attempt=attempt - (task.retry_base or 0),
            charge=charge,
            multiplicity=multiplicity,
            default_maxcore=default_maxcore,
        )
        input_content, submit_content = apply_retry_strategies(
            task_type=task.task_type.value,
            failure=failure,
            input_content=input_content,
            submit_content=submit_content,
            strategies=replayable,
        )
        applied.append(f"attempt {prior.attempt}: {prior.fail_reason}")
    # At its cap the MP2 response retry would rerun the failed job's resources;
    # the replay's own step decides only when that job's files are gone.
    if MP2_RESPONSE_MEMORY in (prior_failures[-1].fail_reason or ""):
        ran = _job_dir_resources(prior_failures[-1].job_path)
        if (_job_resources(input_content, submit_content) == ran if ran is not None
                else (input_content, submit_content) == unchanged):
            raise RetryWouldRepeat(MP2_CAP_REACHED)

    if perturbations:
        latest = prior_failures[-1]
        input_content, submit_content = apply_retry_strategies(
            task_type=task.task_type.value,
            failure=FailureInfo(
                fail_reason=latest.fail_reason or "",
                previous_job_path=latest.job_path or "",
                attempt=attempt - (task.retry_base or 0),
                charge=charge,
                multiplicity=multiplicity,
                default_maxcore=default_maxcore,
            ),
            input_content=input_content,
            submit_content=submit_content,
            strategies=perturbations,
        )

    input_file.write_text(input_content, encoding="utf-8")
    submit_file.write_text(submit_content, encoding="utf-8")
    logger.info(
        "Applied retry strategies for job in %s (attempt %d) accumulating %s",
        job_path, attempt, applied,
    )


def _esd_role(state: MoleculeState) -> Optional[str]:
    """``"S1"`` / ``"T1"`` for an ESD state, else None."""
    metadata = json.loads(state.metadata_json) if state.metadata_json else {}
    return metadata.get("esd_role")


def _check_type(session: Session, task: ComputationTask) -> str:
    """The task type whose output checks apply; an ESD S1 optimisation gets the
    excited-state ones, a densities singlepoint the cube check, and a Reactivity
    singlepoint or vertical the Reactivity block checks plus the cube check."""
    if task.task_type == TaskType.optimization:
        state = session.get(MoleculeState, task.state_id)
        if state is not None and _esd_role(state) == "S1":
            return "optimization_excited"
    if task.task_type.value in categories.REACTIVITY_TYPES:
        state = session.get(MoleculeState, task.state_id)
        if state is not None and _is_reactivity(state, task):
            return "singlepoint_reactivity"
    if task.task_type == TaskType.singlepoint:
        state = session.get(MoleculeState, task.state_id)
        if state is not None and _keeps_cubes(state, task):
            return "singlepoint_densities"
    return task.task_type.value


def _imag_freq_tolerance(session: Session, task: ComputationTask) -> Optional[float]:
    """The imaginary-mode cutoff of an optimisation's state; None for other tasks."""
    if task.task_type != TaskType.optimization:
        return None
    state = session.get(MoleculeState, task.state_id)
    return results.imaginary_cutoff(state) if state is not None else None


def _keeps_hessian(state: MoleculeState, task: ComputationTask) -> bool:
    """Whether a job copies its Hessian back: every optimisation, so ESD can be added later (Extend)."""
    return task.task_type == TaskType.optimization


def _keeps_cubes(state: MoleculeState, task: ComputationTask) -> bool:
    """Whether a singlepoint's density cubes feed the Densities category."""
    if task.task_type != TaskType.singlepoint:
        return False
    metadata = json.loads(state.metadata_json) if state.metadata_json else {}
    return bool(metadata.get(categories.DENSITIES))


def _runs_nbo(state: MoleculeState, task: ComputationTask) -> bool:
    """Whether a singlepoint runs NBO: the NBO category's relaxed one, or a Reactivity one."""
    if _is_reactivity(state, task):
        return True
    if task.task_type != TaskType.singlepoint:
        return False
    metadata = json.loads(state.metadata_json) if state.metadata_json else {}
    return bool(metadata.get(categories.NBO))


def _keeps_wavefunction(state: MoleculeState, task: ComputationTask) -> bool:
    """Whether a job copies its ``.gbw`` back: Reactivity singlepoints."""
    return _is_reactivity(state, task)


def _is_reactivity(state: MoleculeState, task: ComputationTask) -> bool:
    if task.task_type.value not in categories.REACTIVITY_TYPES:
        return False
    metadata = json.loads(state.metadata_json) if state.metadata_json else {}
    return bool(metadata.get(categories.REACTIVITY))


def _get_stage_config(
    settings: Settings, task_type: TaskType, state: Optional[MoleculeState] = None,
    computed: Optional[dict] = None,
):
    """Return the :class:`StageConfig` for a given task type.

    *computed* is a built rate task's record, which says whether any channel runs HT.
    """
    if task_type.value.startswith("esd_"):
        from autodft.qm.orca import esd_inputs

        metadata = json.loads(state.metadata_json) if state is not None and state.metadata_json else {}
        if state is None or esd_inputs.is_fc(task_type.value, metadata, computed):
            return settings.pipeline.esd
        return settings.pipeline.esd_tddft
    elif task_type == TaskType.confsearch:
        return settings.pipeline.confsearch
    elif task_type == TaskType.optimization:
        if state is not None and _esd_role(state) == "S1":
            return settings.pipeline.excited_optimization
        return settings.pipeline.optimization
    elif task_type == TaskType.singlepoint_nmr and state is not None and \
            json.loads(state.metadata_json or "{}").get("nmr_method") == "dh":
        return settings.pipeline.nmr_dh
    else:
        # All singlepoint variants share the singlepoint config
        return settings.pipeline.singlepoint
