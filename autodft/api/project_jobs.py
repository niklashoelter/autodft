"""Durable, user-attached background jobs for project archive / export.

Archiving or exporting a project is minutes of file I/O over a network mount.
Run inside the HTTP request it blocked the response for the whole time and
died with the browser tab. This module moves that work onto a background
thread whose progress and outcome live in a :class:`ProjectJob` row, so the
website can be closed and the job still finishes and records what it did.

Three things this module guarantees:

* **Runs to completion off the request.** :func:`start_job` writes the row,
  spawns a daemon thread, and returns immediately. The thread always
  terminates the row (``successful`` or ``failed``) even if the body raises.
* **One operation per project, and computation paused meanwhile.** The
  ``running`` row is the lock. :func:`assert_no_active_job` refuses a second
  archive/export, a wipe, or a new submission on the same project (and
  :func:`start_job` a project a wipe is deleting, see :func:`wiping`), and the
  controller tick skips advancing that project's tasks (it reads
  :func:`active_job_project_names`). The destructive archive additionally
  waits until every job the project sent to SLURM is finished and judged and
  its task marked, so ``rmtree`` never races a job still writing files and
  every result is exported first.
* **No half-done ghost after a restart.** The thread lives in the controller
  process; a restart orphans it. :func:`reconcile_orphaned_jobs`, called once
  at startup, flips any still-``running`` row to ``failed`` rather than
  resuming it.

Threads open their own short-lived sessions and never hold one across the file
I/O -- the API and the controller share one SQLite connection pool (see
``db.py``), so a pooled connection held for minutes of copying would starve
the pipeline.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from time import monotonic, sleep
from typing import Optional

from sqlalchemy import exists, func, or_
from sqlmodel import Session, col, select

from autodft import accounts
from autodft.config import Settings
from autodft.db import get_session
from autodft.engine.state_machine import SBATCH_FAILED
from autodft.models.enums import (
    ProjectJobKind,
    ProjectJobStatus,
    TaskStatus,
)
from autodft.models.job import ARCHIVED_FAIL_REASON, ComputationJob
from autodft.models.molecule import Molecule
from autodft.models.project_job import ProjectJob
from autodft.models.state import MoleculeState
from autodft.models.task import ComputationTask

logger = logging.getLogger(__name__)


class JobInProgress(RuntimeError):
    """Raised when a project already has an active background job."""


class ProjectGone(LookupError):
    """Raised when a job is started for a project that no longer exists (e.g. reassigned meanwhile)."""


NOTHING_TO_EXPORT = "Nothing to export"
ARCHIVE_RUNNING = "Export and Delete is running for this project."
_ARCHIVE_RERUN = ("Re-running it is safe: it resumes, keeping the molecules already "
                  "archived and exporting and deleting only the rest.")


# How long the destructive archive waits for the project's sent jobs to finish
# and be judged before giving up. The pause stops *new* jobs, so the set only
# shrinks; if it is still non-empty after this, real long-running calculations
# are in flight and the archive aborts cleanly rather than deleting under them.
ARCHIVE_QUIESCENCE_TIMEOUT = 300.0
_QUIESCENCE_POLL_SECONDS = 3.0

# Serialises the check-and-insert in start_job so two requests cannot both
# create a job for the same project. In-process is all that is needed: the API
# and controller share one process on one host (the SQLite-on-NFS single-host
# rule), so there is never a second writer.
_CREATE_LOCK = threading.Lock()

# Projects a wipe is deleting; None: a database reset. start_job refuses them.
_wiping: list[Optional[str]] = []

# Cube builds and task-file downloads in flight per project; the archive waits for them.
_reads: dict[str, int] = {}
_READS_LOCK = threading.Lock()

# Live worker threads, for join_all() in tests and shutdown.
_threads: dict[int, threading.Thread] = {}


# ----------------------------------------------------------------------
# Queries -- the row as a lock, and as a status
# ----------------------------------------------------------------------


def active_job(session: Session, qualified_name: str) -> Optional[ProjectJob]:
    """The running job for *qualified_name*, or None."""
    return session.exec(
        select(ProjectJob)
        .where(
            ProjectJob.qualified_name == qualified_name,
            ProjectJob.status == ProjectJobStatus.running,
        )
        .order_by(col(ProjectJob.id).desc())
    ).first()


def active_job_project_names(session: Session) -> set[str]:
    """Every project with a job currently running.

    Read by the controller tick to skip advancing those projects' tasks.
    """
    return set(
        session.exec(
            select(ProjectJob.qualified_name).where(
                ProjectJob.status == ProjectJobStatus.running
            )
        ).all()
    )


def assert_no_active_job(session: Session, qualified_name: str) -> None:
    """Raise :class:`JobInProgress` if *qualified_name* has a running job."""
    existing = active_job(session, qualified_name)
    if existing is not None and existing.kind == ProjectJobKind.archive:
        raise JobInProgress(ARCHIVE_RUNNING)
    if existing is not None:
        raise JobInProgress(
            f"Project {qualified_name!r} has a {existing.kind.value} job in "
            f"progress. Wait for it to finish before starting another "
            f"operation on this project."
        )


def archive_running(session: Session, qualified_name: str) -> bool:
    """Whether *qualified_name*'s Export and Delete is running."""
    existing = active_job(session, qualified_name)
    return existing is not None and existing.kind == ProjectJobKind.archive


def begin_read(qualified_name: str) -> None:
    """Count a read of *qualified_name*'s job folders until :func:`end_read`.

    Raises:
        JobInProgress: while its archive runs.
    """
    with _CREATE_LOCK:  # an archive starting now waits for this read
        with get_session() as session:
            if archive_running(session, qualified_name):
                raise JobInProgress(ARCHIVE_RUNNING)
        with _READS_LOCK:
            _reads[qualified_name] = _reads.get(qualified_name, 0) + 1


def end_read(qualified_name: str) -> None:
    with _READS_LOCK:
        left = _reads.get(qualified_name, 0) - 1
        if left > 0:
            _reads[qualified_name] = left
        else:
            _reads.pop(qualified_name, None)


def reads_in_flight(qualified_name: str) -> int:
    with _READS_LOCK:
        return _reads.get(qualified_name, 0)


@contextmanager
def creation_lock():
    """Hold off :func:`start_job` while the caller checks and changes a project."""
    with _CREATE_LOCK:
        yield


@contextmanager
def wiping(qualified_name: Optional[str]):
    """Refuse while *qualified_name* (None: any project) has a running job; until the
    block exits, :func:`start_job` refuses it.

    Raises:
        JobInProgress: if a job is running.
    """
    with _CREATE_LOCK:
        with get_session() as session:
            if qualified_name is not None:
                assert_no_active_job(session, qualified_name)
            elif busy := active_job_project_names(session):
                raise JobInProgress(
                    f"A background archive/export job is running for {', '.join(sorted(busy))}. "
                    f"Wait for it to finish before resetting the database."
                )
        _wiping.append(qualified_name)
    try:
        yield
    finally:
        with _CREATE_LOCK:
            _wiping.remove(qualified_name)


def get_job(session: Session, job_id: int) -> Optional[ProjectJob]:
    return session.get(ProjectJob, job_id)


def list_jobs(session: Session, qualified_name: str, limit: int = 20) -> list[ProjectJob]:
    """Recent jobs for a project, newest first."""
    return list(
        session.exec(
            select(ProjectJob)
            .where(ProjectJob.qualified_name == qualified_name)
            .order_by(col(ProjectJob.id).desc())
            .limit(limit)
        ).all()
    )


def serialize(job: ProjectJob) -> dict:
    """A ProjectJob as a JSON-safe dict for the API."""
    return {
        "id": job.id,
        "qualified_name": job.qualified_name,
        "owner_id": job.owner_id,
        "kind": job.kind.value,
        "status": job.status.value,
        "params": _loads(job.params_json),
        "result": _loads(job.result_json),
        "error": job.error,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
    }


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _loads(text: Optional[str]):
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


# ----------------------------------------------------------------------
# Starting a job
# ----------------------------------------------------------------------


def start_job(
    *,
    owner_id: Optional[int],
    qualified_name: str,
    kind: ProjectJobKind,
    params: dict,
    settings: Settings,
) -> dict:
    """Record a job, spawn its worker thread, and return the row snapshot.

    Raises:
        JobInProgress: if the project already has a running job or is being wiped.
        ProjectGone: if *qualified_name* has no project row (legacy bare names excepted)
            or no molecules.
    """
    with _CREATE_LOCK:
        with get_session(settings) as session:
            if "/" in qualified_name and accounts.get_project(session, qualified_name) is None:
                raise ProjectGone(f"No project named {qualified_name!r}.")
            if None in _wiping:
                raise JobInProgress("The database is being reset.")
            if qualified_name in _wiping:
                raise JobInProgress(f"Project {qualified_name!r} is being wiped.")
            # A wipe may have emptied it since the caller looked.
            if session.exec(select(Molecule.id).where(Molecule.project_name == qualified_name)).first() is None:
                raise ProjectGone(f"Project {qualified_name!r} has no molecules.")
            assert_no_active_job(session, qualified_name)
            job = ProjectJob(
                qualified_name=qualified_name,
                owner_id=owner_id,
                kind=kind,
                status=ProjectJobStatus.running,
                params_json=json.dumps(params),
                started_at=datetime.now(timezone.utc),
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            job_id = job.id
            snapshot = serialize(job)

        thread = threading.Thread(
            target=_run_job,
            args=(job_id, kind, qualified_name, params, settings),
            name=f"autodft-projectjob-{job_id}",
            daemon=True,
        )
        _threads[job_id] = thread
        try:
            thread.start()
        except Exception:  # noqa: BLE001 - a thread that never starts must not
            # leave the row wedged in `running` (it would block the project
            # forever and reconcile only runs at startup).
            logger.exception("Could not start worker thread for project job %d", job_id)
            _finish(job_id, ProjectJobStatus.failed, error="Could not start the job thread.")
            _threads.pop(job_id, None)
            raise
    logger.info("Started project job %d: %s on %r", job_id, kind.value, qualified_name)
    return snapshot


def _run_job(
    job_id: int,
    kind: ProjectJobKind,
    qualified_name: str,
    params: dict,
    settings: Settings,
) -> None:
    """Worker-thread body: run the operation, then terminate the row."""
    try:
        result = _execute(kind, qualified_name, params, settings)
        _finish(job_id, ProjectJobStatus.successful, result=result)
        logger.info("Project job %d finished: %s on %r", job_id, kind.value, qualified_name)
    except Exception as exc:  # noqa: BLE001 - the outcome must be recorded
        logger.exception(
            "Project job %d (%s on %r) failed", job_id, kind.value, qualified_name
        )
        _finish(job_id, ProjectJobStatus.failed, error=f"{type(exc).__name__}: {exc}")
    finally:
        _threads.pop(job_id, None)


def _finish(
    job_id: int,
    status: ProjectJobStatus,
    *,
    result: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    """Write the terminal state of a job in its own short session."""
    with get_session() as session:
        job = session.get(ProjectJob, job_id)
        if job is None:
            logger.error("Project job %d vanished before it could be finished", job_id)
            return
        job.status = status
        job.finished_at = datetime.now(timezone.utc)
        if result is not None:
            job.result_json = json.dumps(result)
        if error is not None:
            job.error = error[:2000]
        session.add(job)
        session.commit()


# ----------------------------------------------------------------------
# The work itself
# ----------------------------------------------------------------------


def _execute(
    kind: ProjectJobKind,
    qualified_name: str,
    params: dict,
    settings: Settings,
) -> dict:
    """Run one operation and return its summary dict. No DB session is held
    across the file I/O -- each helper opens its own."""
    from autodft.extraction.extractor import PipelineExtractor
    from autodft.paths import project_file_stem, safe_subdirectory

    settings.ensure_directories()
    out_root = safe_subdirectory(settings.export_data_path, qualified_name)
    out_root.mkdir(parents=True, exist_ok=True)
    stem = project_file_stem(qualified_name)
    extractor = PipelineExtractor(qualified_name)
    all_conformers = bool(params.get("all_conformers", False))
    keep_gbw = bool(params.get("keep_gbw", False))

    if kind == ProjectJobKind.export_csv:
        target = out_root / f"{stem}.csv"
        if not extractor.export_summary_csv(target, all_conformers=all_conformers):
            return {"format": "csv", "message": NOTHING_TO_EXPORT, "downloadable": False}
        return {"format": "csv", "path": str(target), "downloadable": True}

    if kind == ProjectJobKind.export_json:
        target = out_root / f"{stem}.json"
        if not extractor.export_summary_json(target, all_conformers=all_conformers):
            return {"format": "json", "message": NOTHING_TO_EXPORT, "downloadable": False}
        return {"format": "json", "path": str(target), "downloadable": True}

    if kind == ProjectJobKind.export_files:
        summary = extractor.export_tree(settings.export_data_path, all_conformers, keep_gbw)
        # A directory, not a single file: reported by server path, not offered as a download.
        return {"format": "files", "path": summary["root"], **summary, "downloadable": False}

    if kind == ProjectJobKind.export_xlsx:
        from autodft.analysis.state_analysis import analyze_project, build_xlsx_bytes

        target = out_root / f"{stem}_state_analysis.xlsx"
        payload = analyze_project(qualified_name)
        target.write_bytes(build_xlsx_bytes(payload))
        return {"format": "xlsx", "path": str(target), "downloadable": True}

    if kind == ProjectJobKind.export_photophysics:
        from autodft.analysis import spectroscopy
        from autodft.extraction.extractor import write_photophysics

        target, data = write_photophysics(qualified_name, out_root, stem,
                                          payload=spectroscopy.full_payload(qualified_name))
        return {"format": "photophysics", "path": str(target), "json_path": str(data), "downloadable": True}

    if kind == ProjectJobKind.export_reactivity:
        from autodft.analysis.reactivity import reactivity_rows
        from autodft.extraction.extractor import write_reactivity

        rows = reactivity_rows(qualified_name, all_conformers)
        target, data = write_reactivity(qualified_name, out_root, stem, rows=rows)
        return {"format": "reactivity", "path": str(target), "json_path": str(data), "downloadable": True}

    if kind == ProjectJobKind.archive:
        from autodft.analysis import spectroscopy

        # Frozen NMR shifts are final; their reference must be in first.
        waiting = spectroscopy.pending_nmr_references(qualified_name)
        if waiting:
            raise ValueError(
                f"Project {qualified_name!r} has NMR molecules still waiting for their reference "
                f"({', '.join(waiting)}); archive it once the references have finished."
            )

        # Do not rmtree under a job still writing files or a read holding them open, nor
        # before a result is judged and exportable. The pause has stopped new jobs and reads.
        _wait_for_quiescence(qualified_name)
        fail_open_tasks(qualified_name)

        # The archive deletes the outputs the photophysics results are read from.
        frozen = spectroscopy.freeze(qualified_name, settings)

        summary = extractor.archive_project(
            export_root=settings.export_data_path,
            comp_root=settings.comp_data_path,
            all_conformers=all_conformers,
            keep_gbw=keep_gbw,
        )
        summary["downloadable"] = False
        if frozen:
            summary["photophysics_frozen"] = frozen
        return summary

    raise ValueError(f"Unknown project-job kind {kind!r}")


def _unsettled_count(qualified_name: str) -> int:
    """Jobs of *qualified_name* sent, or possibly sent, and not yet judged, plus pending tasks
    a judged success has not marked."""
    in_project = (
        select(ComputationTask.id)
        .join(MoleculeState, col(MoleculeState.id) == col(ComputationTask.state_id))
        .join(Molecule, col(Molecule.id) == col(MoleculeState.molecule_id))
        .where(Molecule.project_name == qualified_name)
    )
    with get_session() as session:
        unjudged = session.exec(
            select(func.count()).select_from(ComputationJob).where(
                col(ComputationJob.task_id).in_(in_project),
                # An sbatch that reported failure stays possibly sent until the controller looks it up.
                or_(col(ComputationJob.slurm_status).is_not(None),
                    col(ComputationJob.fail_reason).startswith(SBATCH_FAILED)),
                col(ComputationJob.success).is_(None),
            )
        ).one()
        # update_task_statuses has yet to mark these successful; the export plans only those.
        unmarked = session.exec(
            select(func.count()).select_from(ComputationTask).where(
                col(ComputationTask.id).in_(in_project),
                ComputationTask.status == TaskStatus.pending,
                exists().where(
                    col(ComputationJob.task_id) == col(ComputationTask.id),
                    col(ComputationJob.success).is_(True),
                    col(ComputationJob.attempt) > func.coalesce(ComputationTask.retry_base, 0),
                ),
            )
        ).one()
        return int(unjudged) + int(unmarked)


def fail_open_tasks(qualified_name: str) -> int:
    """Fail *qualified_name*'s created/pending tasks as archived; returns the count."""
    now = datetime.now(timezone.utc)
    with get_session() as session:
        tasks = session.exec(
            select(ComputationTask)
            .join(MoleculeState, col(MoleculeState.id) == col(ComputationTask.state_id))
            .join(Molecule, col(Molecule.id) == col(MoleculeState.molecule_id))
            .where(
                Molecule.project_name == qualified_name,
                col(ComputationTask.status).in_([TaskStatus.created, TaskStatus.pending]),
            )
        ).all()
        failed = 0
        for task in tasks:
            jobs = session.exec(
                select(ComputationJob).where(ComputationJob.task_id == task.id)
            ).all()
            if any(j.success is True and j.attempt > (task.retry_base or 0) for j in jobs):
                continue  # update_task_statuses marks it successful
            unsent = [j for j in jobs
                      if j.success is None and j.slurm_jobid is None and j.slurm_status is None]
            if not unsent:
                # The reason lives on a folderless failed job, as for a failed build.
                attempt = max([task.retry_base or 0, *(j.attempt for j in jobs)]) + 1
                unsent = [ComputationJob(task_id=task.id, attempt=attempt)]
            for job in unsent:
                job.success = False
                job.fail_reason = ARCHIVED_FAIL_REASON
                job.time_end = now
                session.add(job)
            task.status = TaskStatus.failed
            task.updated_at = now
            session.add(task)
            failed += 1
        session.commit()
    if failed:
        logger.info("Archive of %r failed %d open task(s)", qualified_name, failed)
    return failed


def _wait_for_quiescence(
    qualified_name: str,
    timeout: float = ARCHIVE_QUIESCENCE_TIMEOUT,
    poll: float = _QUIESCENCE_POLL_SECONDS,
) -> None:
    """Block until every sent job of the project is judged and its task marked, and no
    cube build or task-file download is open, or raise.

    Raises:
        RuntimeError: if some are still unsettled after *timeout*.
    """
    deadline = monotonic() + timeout
    while True:
        remaining = _unsettled_count(qualified_name)
        reads = reads_in_flight(qualified_name)
        if remaining == 0 and reads == 0:
            return
        if monotonic() >= deadline:
            raise RuntimeError(
                f"{remaining} computation job(s) and {reads} cube build(s) or download(s) "
                f"for {qualified_name!r} are still running or awaiting their results "
                f"after {timeout:.0f}s; the archive "
                f"was aborted so it would not delete files still in use or "
                f"results not yet exported. Retry once the project's calculations "
                f"and downloads have finished."
            )
        sleep(poll)


# ----------------------------------------------------------------------
# Startup reconciliation & test helpers
# ----------------------------------------------------------------------


def reconcile_orphaned_jobs(settings: Optional[Settings] = None) -> int:
    """Fail any job left ``running`` by a controller restart. Returns the count.

    Called once at controller startup. The worker threads do not survive a
    restart, so an orphan is marked failed (and its project unblocked) rather
    than retried; the user re-runs it.
    """
    with get_session(settings) as session:
        orphans = session.exec(
            select(ProjectJob).where(ProjectJob.status == ProjectJobStatus.running)
        ).all()
        for job in orphans:
            job.status = ProjectJobStatus.failed
            job.error = "Interrupted by a controller restart; the job did not finish. " + (
                _ARCHIVE_RERUN if job.kind == ProjectJobKind.archive else "Re-run it."
            )
            job.finished_at = datetime.now(timezone.utc)
            session.add(job)
        if orphans:
            session.commit()
            logger.warning(
                "Reconciled %d orphaned project job(s) to failed on startup", len(orphans)
            )
        return len(orphans)


def join_all(timeout: Optional[float] = None) -> None:
    """Wait for every live worker thread. For tests and shutdown."""
    for thread in list(_threads.values()):
        thread.join(timeout)
