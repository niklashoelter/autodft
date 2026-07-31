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
  archive/export, a wipe, or a new submission on the same project, and the
  controller tick skips advancing that project's tasks (it reads
  :func:`active_job_project_names`). The destructive archive additionally
  waits for the project's in-flight SLURM jobs to finish before it deletes
  anything, so ``rmtree`` never races a job still writing files.
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
from datetime import datetime, timezone
from time import monotonic, sleep
from typing import Optional

from sqlalchemy import func
from sqlmodel import Session, col, select

from autodft.config import Settings
from autodft.db import get_session
from autodft.models.enums import (
    TRANSIENT_SLURM_STATES,
    ProjectJobKind,
    ProjectJobStatus,
)
from autodft.models.job import ComputationJob
from autodft.models.molecule import Molecule
from autodft.models.project_job import ProjectJob
from autodft.models.state import MoleculeState
from autodft.models.task import ComputationTask

logger = logging.getLogger(__name__)


class JobInProgress(RuntimeError):
    """Raised when a project already has an active background job."""


# How long the destructive archive waits for the project's in-flight SLURM
# jobs to finish before giving up. The pause stops *new* jobs, so the set only
# shrinks; if it is still non-empty after this, real long-running calculations
# are in flight and the archive aborts cleanly rather than deleting under them.
ARCHIVE_QUIESCENCE_TIMEOUT = 300.0
_QUIESCENCE_POLL_SECONDS = 3.0

# Serialises the check-and-insert in start_job so two requests cannot both
# create a job for the same project. In-process is all that is needed: the API
# and controller share one process on one host (the SQLite-on-NFS single-host
# rule), so there is never a second writer.
_CREATE_LOCK = threading.Lock()

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
    if existing is not None:
        raise JobInProgress(
            f"Project {qualified_name!r} has a {existing.kind.value} job in "
            f"progress. Wait for it to finish before starting another "
            f"operation on this project."
        )


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
        JobInProgress: if the project already has a running job.
    """
    with _CREATE_LOCK:
        with get_session(settings) as session:
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

    if kind == ProjectJobKind.export_csv:
        target = out_root / f"{stem}.csv"
        extractor.export_summary_csv(target, all_conformers=all_conformers)
        return {"format": "csv", "path": str(target), "downloadable": True}

    if kind == ProjectJobKind.export_json:
        target = out_root / f"{stem}.json"
        extractor.export_summary_json(target, all_conformers=all_conformers)
        return {"format": "json", "path": str(target), "downloadable": True}

    if kind == ProjectJobKind.export_files:
        target = out_root / "files"
        count = extractor.export_calculation_files(target, all_conformers=all_conformers)
        # A directory, not a single file -- reported by server path, matching
        # the pre-job behaviour; not offered as a browser download.
        return {
            "format": "files",
            "path": str(target),
            "files_copied": count,
            "downloadable": False,
        }

    if kind == ProjectJobKind.export_xlsx:
        from autodft.analysis.state_analysis import analyze_project, build_xlsx_bytes

        target = out_root / f"{stem}_state_analysis.xlsx"
        payload = analyze_project(qualified_name)
        target.write_bytes(build_xlsx_bytes(payload))
        return {"format": "xlsx", "path": str(target), "downloadable": True}

    if kind == ProjectJobKind.archive:
        extensions = params.get("extensions") or [".inp", ".xyz", ".out"]
        # Do not rmtree under a job still writing files. The pause has already
        # stopped new jobs for this project; wait for what is left to finish.
        _wait_for_quiescence(qualified_name)
        summary = extractor.archive_project(
            export_root=settings.export_data_path,
            comp_root=settings.comp_data_path,
            extensions=extensions,
            all_conformers=all_conformers,
        )
        summary["downloadable"] = False
        return summary

    raise ValueError(f"Unknown project-job kind {kind!r}")


def _in_flight_job_count(qualified_name: str) -> int:
    """Jobs for *qualified_name* that SLURM has not finished with."""
    with get_session() as session:
        return int(
            session.exec(
                select(func.count())
                .select_from(ComputationJob)
                .join(ComputationTask, col(ComputationTask.id) == col(ComputationJob.task_id))
                .join(MoleculeState, col(MoleculeState.id) == col(ComputationTask.state_id))
                .join(Molecule, col(Molecule.id) == col(MoleculeState.molecule_id))
                .where(
                    Molecule.project_name == qualified_name,
                    col(ComputationJob.slurm_status).in_(sorted(TRANSIENT_SLURM_STATES)),
                )
            ).one()
        )


def _wait_for_quiescence(
    qualified_name: str,
    timeout: float = ARCHIVE_QUIESCENCE_TIMEOUT,
    poll: float = _QUIESCENCE_POLL_SECONDS,
) -> None:
    """Block until the project has no in-flight SLURM jobs, or raise.

    Raises:
        RuntimeError: if jobs are still running after *timeout*.
    """
    deadline = monotonic() + timeout
    while True:
        remaining = _in_flight_job_count(qualified_name)
        if remaining == 0:
            return
        if monotonic() >= deadline:
            raise RuntimeError(
                f"{remaining} computation job(s) for {qualified_name!r} are still "
                f"running after {timeout:.0f}s; the archive was aborted so it would "
                f"not delete files a job is still writing. Retry once the project's "
                f"calculations have finished."
            )
        sleep(poll)


# ----------------------------------------------------------------------
# Startup reconciliation & test helpers
# ----------------------------------------------------------------------


def reconcile_orphaned_jobs(settings: Optional[Settings] = None) -> int:
    """Fail any job left ``running`` by a controller restart. Returns the count.

    Called once at controller startup. The worker threads do not survive a
    restart and the destructive archive cannot be resumed blindly, so an
    orphan is marked failed (and its project unblocked) rather than retried.
    """
    with get_session(settings) as session:
        orphans = session.exec(
            select(ProjectJob).where(ProjectJob.status == ProjectJobStatus.running)
        ).all()
        for job in orphans:
            job.status = ProjectJobStatus.failed
            job.error = (
                "Interrupted by a controller restart; the job did not finish. Re-run it."
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
