"""Destructive maintenance operations: wipe a project, a molecule, or everything; re-run a task.

Separate from ``routes.py`` so the deletion logic can be read and tested on
its own -- these functions delete data that cannot be recovered.

Every operation comes in two halves: a ``preview_*`` that only counts, and a
``wipe_*`` that acts. The API layer always shows the preview first and
requires the caller to echo back an exact confirmation string.

Deletion order matters. The foreign keys form a cycle:

    computation_jobs      -> computation_tasks
    computation_tasks     -> molecule_states, molecule_geometries, itself
    molecule_geometries   -> molecule_states, computation_tasks
    molecule_states       -> molecules

so the task/geometry references are nulled out before either table is
deleted. SQLite runs with ``PRAGMA foreign_keys=ON`` (see ``db.py``), which
would otherwise reject the delete.

``computation_headers`` are never touched: they are shared across projects
and referenced by rows that may survive. ``job_results`` has no foreign key
and is deleted with its job, not by the cascade above.

Deleting the files is the slow half -- ~65 ms per file on this deployment's
network mount, so minutes for a real project. A project wipe and a database
reset therefore *stage* the directories (one rename each) and answer the
request immediately, leaving a background thread to do the unlinking. See
:class:`_BackgroundRemoval`.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Optional

from sqlmodel import Session, col, delete, func, select, update

from autodft.models.enums import TERMINAL_SLURM_STATES, TRANSIENT_SLURM_STATES, TaskStatus, TaskType
from autodft.api import project_jobs
from autodft.config import Settings
from autodft.db import get_session
from autodft.engine.state_machine import SBATCH_FAILED
from autodft.paths import safe_subdirectory
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.models.geometry import MoleculeGeometry
from autodft.models.job import ComputationJob
from autodft.models.molecule import Molecule
from autodft.models.result import JobResult
from autodft.models.state import MoleculeState
from autodft.models.task import ComputationTask

logger = logging.getLogger(__name__)

# Never wipeable. Mirrors PROTECTED_PROJECT_NAMES in routes.py.
PROTECTED_PROJECT_NAMES = {"default"}


def is_protected(name: str) -> bool:
    """Whether *name* names a project that may never be wiped.

    Only the *shared* default is protected -- ``admin/default``, and the
    bare ``default`` a database that predates the migration still uses.

    Two mistakes are possible here and this avoids both. Comparing the
    whole string stopped matching once the migration renamed ``default``
    to ``admin/default``, silently unprotecting it. Comparing only the bare
    segment over-corrected: ``project`` defaults to ``"default"`` on every
    submission, so each user acquires an ``alice/default`` that no one
    could ever wipe. The NMR reference project is always protected.
    """
    from autodft.accounts import ADMIN_USERNAME
    from autodft.paths import normalise_project_name

    from autodft.engine.nmr_references import is_reference_project

    if is_reference_project(name):
        return True

    normalised = normalise_project_name(name or "")
    owner, separator, bare = normalised.partition("/")
    if not separator:
        return normalised in PROTECTED_PROJECT_NAMES      # pre-migration
    return bare in PROTECTED_PROJECT_NAMES and owner == ADMIN_USERNAME

RESET_CONFIRMATION = "RESET THE DATABASE"


class WipeInProgress(RuntimeError):
    """Raised when a destructive operation is already running."""


# One destructive operation at a time, process-wide. Deleting a project is
# minutes of rmtree over the network mount, and the API serves each request
# on its own thread: a second wipe launched while the first was still running
# walked the same directories and blew up on paths the other had already
# removed -- which, under the old files-before-rows order, aborted the wipe
# with the files gone and the rows still there. Refuse the second one
# instead of letting two deleters race.
_EXCLUSIVE = threading.Lock()
_current_operation: Optional[str] = None

# Who holds the lock is per-thread state: the request thread acquires it and,
# once the files have been staged, hands ownership to the deleter thread.
_local = threading.local()


@contextmanager
def exclusive(label: str):
    """Hold the destructive-operation lock, or raise :class:`WipeInProgress`.

    The lock is normally released when the block exits. If the operation
    handed its file deletion to a background thread, that thread owns the
    lock instead and releases it when the last directory is gone -- so a
    second wipe is still refused while the first is only half-deleted.
    """
    global _current_operation
    if not _EXCLUSIVE.acquire(blocking=False):
        raise WipeInProgress(
            f"{_current_operation or 'Another destructive operation'} is still "
            f"running. Wait for it to finish before starting {label!r}."
        )
    _current_operation = label
    _local.holding = True
    _local.handed_off = False
    try:
        yield
    finally:
        handed_off = _local.handed_off
        _local.holding = False
        _local.handed_off = False
        if not handed_off:
            _current_operation = None
            _EXCLUSIVE.release()


def _release_exclusive() -> None:
    """Release the lock on behalf of the thread it was handed to."""
    global _current_operation
    _current_operation = None
    _EXCLUSIVE.release()


def current_operation() -> Optional[str]:
    """The destructive operation currently running, if any."""
    return _current_operation


# How long a wipe waits for scancel before deleting the files anyway.
SCANCEL_TIMEOUT_S = 60.0


def _live_slurm_ids(session: Session, job_ids: list[int]) -> list[str]:
    """SLURM ids of these jobs that SLURM may still have queued or running."""
    if not job_ids:
        return []
    return [
        str(j.slurm_jobid)
        for j in session.exec(
            select(ComputationJob).where(
                col(ComputationJob.id).in_(job_ids),
                col(ComputationJob.slurm_jobid).is_not(None),
                col(ComputationJob.slurm_status).in_(sorted(TRANSIENT_SLURM_STATES)),
            )
        ).all()
        if j.slurm_jobid is not None
    ]


def _cancel_scheduled_jobs(slurm_ids: list[str], scheduler) -> tuple[int, Optional[str]]:
    """scancel *slurm_ids*, with no transaction open; returns (cancelled, warning or None).

    Without this, wiping a project left its queued and running jobs alive,
    writing ORCA output into directories that had just been deleted.
    """
    if not slurm_ids or scheduler is None:
        return 0, None
    try:
        return scheduler.cancel_many(slurm_ids, timeout=SCANCEL_TIMEOUT_S), None
    except subprocess.TimeoutExpired:
        logger.error("scancel of %d job(s) gave no answer within %.0f s; deleting their files anyway",
                     len(slurm_ids), SCANCEL_TIMEOUT_S)
        problem = f"scancel gave no answer within {SCANCEL_TIMEOUT_S:.0f} s"
    except Exception:  # noqa: BLE001 - a failed scancel must not abort the wipe
        logger.exception("Failed to cancel %d job(s) before wiping", len(slurm_ids))
        problem = "scancel failed"
    return 0, f"{problem}; SLURM may still run job(s) {', '.join(slurm_ids)}: scancel them by hand."


def _refuse_mid_sbatch(session: Session, task_ids: Optional[list[int]] = None) -> None:
    """Raise ValueError if SLURM may have a job of *task_ids* (None: any) with no id to scancel.

    Claimed for sbatch, or sent by an sbatch that reported failure: deleting its row
    would leave SLURM running a job nothing tracks. Call under :func:`begin_write`.
    """
    if task_ids == []:
        return
    query = select(ComputationJob).where(
        col(ComputationJob.slurm_jobid).is_(None), col(ComputationJob.success).is_(None),
    )
    if task_ids is not None:
        query = query.where(col(ComputationJob.task_id).in_(task_ids))
    jobs = [j for j in session.exec(query).all() if not _never_sent(j)]
    if jobs:
        raise ValueError(
            f"{len(jobs)} job(s) may be in SLURM with no SLURM job id to cancel yet: "
            f"{describe_jobs(jobs)}. Retry once the controller has recorded them."
        )


# ----------------------------------------------------------------------
# Disk helpers
# ----------------------------------------------------------------------


# How long a *preview* may spend measuring files. A confirmation dialog
# wants a size to show before something irreversible, but this deployment
# stats at ~5 ms per file over the network mount -- 32 GB of comp_data took
# 5m14s to walk -- and the request holds a pooled database connection for
# every second of it. Measure what fits in the budget and say so: "at least
# 3.2 GB" answers what the dialog is actually asking. The exact figure is
# the on-demand walk in :class:`_DiskUsage`.
PREVIEW_MEASURE_SECONDS = 2.0


def _scan(path: Path, deadline: Optional[float]) -> tuple[int, int, bool]:
    """``(bytes, files, complete)`` under *path*; stops at *deadline*.

    ``os.scandir`` rather than ``Path.rglob``: rglob stats every entry to
    match the pattern, ``is_file()`` stats it again and ``stat()`` a third
    time, so it cost three round-trips per file on a mount where one is
    already milliseconds. scandir carries the type and the size along.

    *deadline* is a ``monotonic()`` value, or None for "walk it all".
    ``complete`` is False only when the deadline cut the walk short --
    unreadable entries are skipped either way.
    """
    try:
        if path.is_file():
            return path.stat().st_size, 1, True
        if not path.is_dir():
            return 0, 0, True
    except OSError:
        return 0, 0, True

    total = files = 0
    stack = [path]
    while stack:
        if deadline is not None and monotonic() >= deadline:
            return total, files, False
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:  # vanished mid-walk, broken symlink
                        continue
        except OSError:  # unreadable, or a race with another deleter
            continue
    return total, files, True


def _dir_size(path: Path) -> int:
    """Total size in bytes of everything under *path*, 0 if absent."""
    return _scan(path, None)[0]


def _measure(paths: list[Path], deadline: Optional[float]) -> tuple[int, bool]:
    """``(bytes, complete)`` across *paths*, sharing one deadline."""
    total = 0
    complete = True
    for path in paths:
        size, _, done = _scan(path, deadline)
        total += size
        complete = complete and done
    return total, complete


def _remove_tree(path: Path) -> tuple[bool, int]:
    """Delete a directory tree, returning ``(removed, bytes_freed)``.

    The size is accumulated during the walk that deletes. Measuring with
    ``_dir_size`` first and then calling ``shutil.rmtree`` traversed the tree
    twice, which on the network mount is most of the wall-clock cost of a
    wipe -- and a project wipe already walks it once for the preview.
    """
    if not path.exists() and not path.is_symlink():
        return False, 0

    if not path.is_dir() or path.is_symlink():
        size = path.stat().st_size if path.is_file() else 0
        path.unlink()
        return True, size

    freed = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for filename in files:
            entry = Path(root) / filename
            try:
                freed += entry.stat().st_size
            except OSError:  # broken symlink, or a race with another deleter
                pass
            try:
                entry.unlink()
            except FileNotFoundError:
                pass
        for dirname in dirs:
            try:
                (Path(root) / dirname).rmdir()
            except OSError:
                # Not empty (a symlinked dir, or something appeared): fall
                # back to the blunt instrument for this subtree.
                shutil.rmtree(Path(root) / dirname, ignore_errors=True)
    path.rmdir()
    logger.info("Deleted directory tree %s (%s)", path, human_bytes(freed))
    return True, freed


# ----------------------------------------------------------------------
# Staging + background removal
# ----------------------------------------------------------------------

# Where staged trees wait to be deleted. Dot-prefixed and never matching
# ``mol_*``, so nothing that scans the data directories picks it up.
_TRASH_DIRNAME = ".wipe-trash"

_batch_counter = itertools.count(1)


def _batch_name() -> str:
    """A trash sub-directory name unique to this operation."""
    return f"{os.getpid()}-{next(_batch_counter)}"


def _stage(paths: list[Path], batch: str) -> list[Path]:
    """Rename each path into a trash directory beside it; return the new paths.

    A rename is one metadata operation, so this returns in milliseconds where
    the deletion takes minutes. It also frees the *name* immediately, which
    matters more than the speed: molecule ids restart at 1 after a reset, and
    the worker runs in the same process as the API, so within a second of the
    reset returning it would otherwise be creating ``mol_1`` inside the very
    directory the deleter is still walking.

    A path that cannot be renamed is returned unstaged and deleted in place.
    """
    staged: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            trash = path.parent / _TRASH_DIRNAME / batch
            trash.mkdir(parents=True, exist_ok=True)
            destination = trash / path.name
            path.rename(destination)
            staged.append(destination)
        except OSError:
            logger.exception("Could not stage %s; it will be deleted in place", path)
            staged.append(path)
    return staged


def _stage_whole(root: Path, batch: str) -> list[Path]:
    """Stage a whole data directory with one rename, then recreate it empty.

    A reset stages everything, and renaming the children one at a time is
    ~24 ms each on this mount -- a minute of it for the full library, all of
    it inside the request. Moving the root itself costs one operation no
    matter how much is under it.
    """
    if not root.is_dir():
        return []
    mode = root.stat().st_mode
    staged = _stage([root], batch)
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, mode)
    except OSError:
        logger.warning("Recreated %s but could not restore its mode", root)
    return staged


def _stale_trash(*roots: Path, skip: str = "") -> list[Path]:
    """Batches left behind by an operation that died before finishing.

    Only one destructive operation runs at a time and the deleter holds the
    lock until it is done, so anything still here belongs to a dead process.
    Both possible locations are swept: a project wipe stages into
    ``comp_data/.wipe-trash``, a reset one level up. *skip* excludes the
    caller's own batch.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for trash in (root / _TRASH_DIRNAME, root.parent / _TRASH_DIRNAME):
            if trash in seen or not trash.is_dir():
                continue
            seen.add(trash)
            out += sorted(b for b in trash.iterdir() if b.name != skip)
    return out


class _BackgroundRemoval:
    """The slow half of a wipe: unlinking trees that are already staged.

    Runs on its own thread so the HTTP request returns as soon as the rows
    are committed and the directories are out of the way. It owns the
    exclusive lock for its lifetime, so a second wipe started while it is
    working is still refused with 409 rather than racing it.
    """

    def __init__(self, label: str, targets: list[Path]) -> None:
        self.label = label
        self.targets = list(targets)
        self.dirs_removed = 0
        self.bytes_freed = 0
        self.orphaned: list[str] = []
        self.background = False
        self.owns_lock = False
        self.finished = False
        self._thread = threading.Thread(
            target=self.run, name="autodft-wipe", daemon=True,
        )

    def run(self) -> None:
        try:
            for target in self.targets:
                try:
                    removed, size = _remove_tree(target)
                    self.dirs_removed += int(removed)
                    self.bytes_freed += size
                except OSError:
                    logger.exception("Could not remove %s; it is now orphaned", target)
                    self.orphaned.append(str(target))
            self._prune_trash()
        finally:
            self.finished = True
            logger.warning(
                "%s: deleted %d/%d staged tree(s), %s freed, %d orphaned",
                self.label, self.dirs_removed, len(self.targets),
                human_bytes(self.bytes_freed), len(self.orphaned),
            )
            if self.owns_lock:
                self.owns_lock = False
                _release_exclusive()

    def _prune_trash(self) -> None:
        """Remove the batch and trash directories once they are empty."""
        batches = {t.parent for t in self.targets if t.parent.parent.name == _TRASH_DIRNAME}
        for directory in sorted(batches) + sorted({b.parent for b in batches}):
            try:
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
            except OSError:
                pass

    def status(self) -> dict:
        """Progress, for the response body and the wipe-status endpoint."""
        pending = len(self.targets) - self.dirs_removed - len(self.orphaned)
        if not self.background:
            message = f"Freed {human_bytes(self.bytes_freed)}."
        elif self.finished:
            message = (
                f"Files deleted in the background; {human_bytes(self.bytes_freed)} freed."
            )
        else:
            message = (
                f"Files are already out of the way; {pending} directory tree(s) "
                f"are being deleted in the background."
            )
        return {
            "background": self.background,
            "state": "finished" if self.finished else "running",
            "label": self.label,
            "dirs_total": len(self.targets),
            "dirs_removed": self.dirs_removed,
            "dirs_pending": max(0, pending),
            "bytes_freed": self.bytes_freed,
            "freed_human": human_bytes(self.bytes_freed),
            "orphaned_dirs": list(self.orphaned),
            "message": message,
        }

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the deleter thread. For tests and shutdown."""
        if self._thread.is_alive():
            self._thread.join(timeout)


_removal: Optional[_BackgroundRemoval] = None


def _remove_in_background(
    label: str, targets: list[Path], *, background: bool,
) -> _BackgroundRemoval:
    """Delete *targets*, on a separate thread unless told otherwise."""
    global _removal, _current_operation

    removal = _BackgroundRemoval(label, targets)
    _removal = removal

    # Whatever was measured describes a tree that is about to shrink.
    _reset_disk_usage()

    if background and targets:
        removal.background = True
        if getattr(_local, "holding", False):
            # Hand the lock over: the operation is not finished until the
            # last file is gone, and nothing else may start before then.
            removal.owns_lock = True
            _local.handed_off = True
            _current_operation = f"{label} (deleting files)"
        removal._thread.start()
    else:
        removal.run()
    return removal


def removal_status() -> Optional[dict]:
    """Status of the most recent file removal, or None if there has been none."""
    return _removal.status() if _removal is not None else None


# ----------------------------------------------------------------------
# Start-up cleanup
# ----------------------------------------------------------------------

# What ``_batch_name`` produces: ``<pid>-<n>``.
_BATCH_RE = re.compile(r"([0-9]+)-[0-9]+")


def _pid_alive(pid: int) -> bool:
    """Whether *pid* may be running; unsure counts as alive."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (OSError, OverflowError):  # EPERM: someone else's; overflow: not ours to judge
        return True
    return True


def _dead_export_trash(export_root: Path) -> list[Path]:
    """``<export_root>/<owner>/.wipe-trash/<pid>-<n>`` batches whose process is gone.

    A project wipe stages its export folder there, and no wipe sweeps it.
    """
    out: list[Path] = []
    try:
        owners = sorted(export_root.iterdir()) if export_root.is_dir() else []
    except OSError:
        logger.exception("Start-up cleanup: could not list %s", export_root)
        return out
    for owner in owners:
        trash = owner / _TRASH_DIRNAME
        try:
            if owner.is_symlink() or trash.is_symlink() or not trash.is_dir():
                continue
            for batch in sorted(trash.iterdir()):
                match = _BATCH_RE.fullmatch(batch.name)
                if match and batch.is_dir() and not _pid_alive(int(match.group(1))):
                    out.append(batch)
        except OSError:
            logger.exception("Start-up cleanup: could not list %s", trash)
    return out


def _archived_job_paths(settings: Settings) -> dict[int, tuple[str, list[str]]]:
    """``{molecule_id: (project, job_paths)}`` for every archived molecule."""
    out: dict[int, tuple[str, list[str]]] = {}
    with get_session(settings) as session:
        rows = session.exec(
            select(Molecule.id, Molecule.project_name, ComputationJob.job_path)
            .outerjoin(MoleculeState, MoleculeState.molecule_id == Molecule.id)
            .outerjoin(ComputationTask, ComputationTask.state_id == MoleculeState.id)
            .outerjoin(ComputationJob, ComputationJob.task_id == ComputationTask.id)
            .where(Molecule.archived == True)  # noqa: E712
        ).all()
    for mol_id, project, job_path in rows:
        paths = out.setdefault(mol_id, (project, []))[1]
        if job_path:
            paths.append(job_path)
    return out


def _owns(mol_dir: Path, job_paths: list[str]) -> bool:
    """Whether a job path of the molecule lies under *mol_dir*."""
    root = mol_dir.resolve()
    return any(Path(p).resolve().is_relative_to(root) for p in job_paths)


def _clean_up(settings: Settings) -> None:
    for batch in _dead_export_trash(settings.export_data_path):
        try:
            _, freed = _remove_tree(batch)
            logger.warning("Start-up cleanup: removed wipe trash %s (%s)", batch, human_bytes(freed))
        except OSError:
            logger.exception("Start-up cleanup: could not remove %s", batch)
            continue
        try:
            if not any(batch.parent.iterdir()):
                batch.parent.rmdir()
        except OSError:
            logger.warning("Start-up cleanup: could not remove the emptied %s", batch.parent)

    # An archive killed during a molecule's delete leaves it flagged with its folder.
    # Deleted only when its export exists and its job paths agree with this comp_data.
    for mol_id, (project, job_paths) in _archived_job_paths(settings).items():
        mol_dir = settings.comp_data_path / f"mol_{mol_id}"
        try:
            if not (mol_dir.is_dir() or mol_dir.is_symlink()):
                continue
            if not safe_subdirectory(settings.export_data_path, project).is_dir():
                logger.warning("Start-up cleanup: kept %s, %r has no export folder", mol_dir, project)
                continue
            if not _owns(mol_dir, job_paths):
                logger.warning(
                    "Start-up cleanup: kept %s, no job of archived molecule %d lies in it",
                    mol_dir, mol_id,
                )
                continue
            with project_jobs.wiping(project):
                # A wipe may have handed the id to another molecule since the scan.
                with get_session(settings) as session:
                    mol = session.get(Molecule, mol_id)
                    still = mol is not None and mol.archived and mol.project_name == project
                if not still:
                    logger.warning("Start-up cleanup: kept %s, molecule %d is no longer archived", mol_dir, mol_id)
                    continue
                removed, freed = _remove_tree(mol_dir)
            if removed:
                logger.warning(
                    "Start-up cleanup: removed archived molecule %d's %s (%s)",
                    mol_id, mol_dir, human_bytes(freed),
                )
        except project_jobs.JobInProgress:
            logger.info("Start-up cleanup: %r has a job running; left %s", project, mol_dir)
        except (OSError, ValueError, RuntimeError):  # RuntimeError: a symlink loop
            logger.exception("Start-up cleanup: could not check or remove %s", mol_dir)


def clean_up_at_start(settings: Settings) -> threading.Thread:
    """Delete what killed wipes and archives left behind, on a daemon thread."""

    def run() -> None:
        try:
            _clean_up(settings)
        except Exception:  # noqa: BLE001 - never takes the controller down
            logger.exception("Start-up cleanup failed")

    thread = threading.Thread(target=run, name="autodft-startup-cleanup", daemon=True)
    thread.start()
    return thread


# ----------------------------------------------------------------------
# On-demand disk usage
# ----------------------------------------------------------------------


class _DiskUsage:
    """Measure the whole data directory, on request and off the request.

    This used to run inside the reset preview, which the admin page fetched
    on every render. That walk takes five minutes on this deployment's
    network mount; it ran inside a ``get_session()`` block, so it held one
    of the connection pool's fifteen slots for the whole time; and a sync
    handler is not cancelled when the operator navigates away, so each
    reload started another one. Fifteen reloads exhausted the pool and
    every request in the process -- dashboard, submissions, API-key auth --
    blocked behind it.

    So: asked for explicitly, run on its own thread, touching no database,
    and the answer is kept until someone asks for a fresh one. Progress is
    reported per top-level directory, because "212 / 639" is a thing an
    operator can wait for and a spinner is not.
    """

    def __init__(self, comp_root: Path, export_root: Path) -> None:
        self.comp_root = comp_root
        self.export_root = export_root
        self.comp_bytes = 0
        self.export_bytes = 0
        self.files = 0
        self.dirs_total = 0
        self.dirs_done = 0
        self.finished = False
        self.error: Optional[str] = None
        self.started = monotonic()
        self.measured_at: Optional[datetime] = None
        self._thread = threading.Thread(
            target=self.run, name="autodft-disk-usage", daemon=True,
        )

    def _walk(self, root: Path, field: str) -> None:
        """Total the bytes under *root* into ``self.<field>``.

        Accumulated per child rather than returned at the end, so the
        running total in :meth:`status` climbs while the walk is going.
        Four minutes of "0 B so far" reads like a hang.
        """
        try:
            children = sorted(root.iterdir()) if root.is_dir() else []
        except OSError:
            return
        # Counted here rather than up front so the total is right even when
        # the first root is huge and the second is empty.
        self.dirs_total += len(children)
        for child in children:
            size, files, _ = _scan(child, None)
            setattr(self, field, getattr(self, field) + size)
            self.files += files
            self.dirs_done += 1

    def run(self) -> None:
        try:
            self._walk(self.comp_root, "comp_bytes")
            self._walk(self.export_root, "export_bytes")
            self.measured_at = datetime.now(timezone.utc)
        except Exception as exc:  # noqa: BLE001 - reported, never raised at a thread
            logger.exception("Disk usage measurement failed")
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.finished = True
            logger.info(
                "Disk usage: %s in %d file(s), measured in %.1fs",
                human_bytes(self.comp_bytes + self.export_bytes),
                self.files, monotonic() - self.started,
            )

    def status(self) -> dict:
        total = self.comp_bytes + self.export_bytes
        return {
            "state": ("failed" if self.error else
                      "ready" if self.finished else "running"),
            "error": self.error,
            "dirs_total": self.dirs_total,
            "dirs_done": self.dirs_done,
            "files": self.files,
            "comp_data_bytes": self.comp_bytes,
            "comp_data_human": human_bytes(self.comp_bytes),
            "export_bytes": self.export_bytes,
            "export_human": human_bytes(self.export_bytes),
            "total_bytes": total,
            "total_human": human_bytes(total),
            "elapsed_seconds": round(monotonic() - self.started, 1),
            "measured_at": self.measured_at.isoformat() if self.measured_at else None,
        }

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the measuring thread. For tests and shutdown."""
        if self._thread.is_alive():
            self._thread.join(timeout)


_disk_usage: Optional[_DiskUsage] = None


def start_disk_usage(comp_root: Path, export_root: Path) -> dict:
    """Begin measuring, or report the walk already under way.

    Deliberately *not* guarded by :func:`exclusive`: this only reads, so it
    must not block a wipe or be blocked by one. A second request while a
    walk is running joins that walk rather than starting a competing stat
    storm on the same mount.
    """
    global _disk_usage

    if _disk_usage is not None and not _disk_usage.finished:
        return _disk_usage.status()

    usage = _DiskUsage(comp_root, export_root)
    _disk_usage = usage
    usage._thread.start()
    return usage.status()


def disk_usage_status() -> Optional[dict]:
    """The last measurement, or None if nobody has asked for one."""
    return _disk_usage.status() if _disk_usage is not None else None


def _reset_disk_usage() -> None:
    """Drop the cached measurement. For tests, and after a database reset."""
    global _disk_usage
    _disk_usage = None


def human_bytes(n: int) -> str:
    """Format a byte count for the confirmation dialog."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < step:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} PB"


# ----------------------------------------------------------------------
# Row collection
# ----------------------------------------------------------------------


def _project_molecule_ids(session: Session, name: str) -> list[int]:
    return [
        m for m in session.exec(
            select(Molecule.id).where(Molecule.project_name == name)
        ).all() if m is not None
    ]


def _queued_entrypoint_ids(session: Session, name: str) -> list[int]:
    """Entrypoint rows belonging to *name*.

    ``CalculationEntrypoint`` has no project column -- the project lives
    inside the ``request_metadata`` JSON -- so this filters in Python.
    Without it, wiping a project would leave its unprocessed submissions in
    the queue, and the controller would rebuild the project on the next tick.
    """
    out: list[int] = []
    for entry in session.exec(select(CalculationEntrypoint)).all():
        if entry.id is None:
            continue
        try:
            meta = json.loads(entry.request_metadata) if entry.request_metadata else {}
        except (ValueError, TypeError):
            continue
        if meta.get("project_name") == name:
            out.append(entry.id)
    return out


def _descendants(session: Session, molecule_ids: list[int]) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return (state_ids, task_ids, geometry_ids, job_ids) for these molecules."""
    if not molecule_ids:
        return [], [], [], []

    state_ids = [
        s for s in session.exec(
            select(MoleculeState.id).where(col(MoleculeState.molecule_id).in_(molecule_ids))
        ).all() if s is not None
    ]
    if not state_ids:
        return [], [], [], []

    task_ids = [
        t for t in session.exec(
            select(ComputationTask.id).where(col(ComputationTask.state_id).in_(state_ids))
        ).all() if t is not None
    ]
    geometry_ids = [
        g for g in session.exec(
            select(MoleculeGeometry.id).where(col(MoleculeGeometry.state_id).in_(state_ids))
        ).all() if g is not None
    ]
    job_ids = [
        j for j in session.exec(
            select(ComputationJob.id).where(col(ComputationJob.task_id).in_(task_ids))
        ).all() if j is not None
    ] if task_ids else []

    return state_ids, task_ids, geometry_ids, job_ids


def _job_result_count(session: Session, job_ids: list[int]) -> int:
    """How many ``job_results`` rows belong to *job_ids*."""
    if not job_ids:
        return 0
    return session.exec(
        select(func.count()).select_from(JobResult).where(col(JobResult.job_id).in_(job_ids))
    ).one()


def _delete_molecule_rows(session: Session, molecule_ids: list[int]) -> dict[str, int]:
    """Delete molecules and everything hanging off them. Returns row counts."""
    state_ids, task_ids, geometry_ids, job_ids = _descendants(session, molecule_ids)
    result_count = _delete_task_rows(session, task_ids, geometry_ids, job_ids)
    if state_ids:
        session.exec(delete(MoleculeState).where(col(MoleculeState.id).in_(state_ids)))
    if molecule_ids:
        session.exec(delete(Molecule).where(col(Molecule.id).in_(molecule_ids)))

    return {
        "molecules": len(molecule_ids),
        "states": len(state_ids),
        "tasks": len(task_ids),
        "geometries": len(geometry_ids),
        "jobs": len(job_ids),
        "job_results": result_count,
    }


def _delete_task_rows(
    session: Session, task_ids: list[int], geometry_ids: list[int], job_ids: list[int],
) -> int:
    """Delete these jobs (with their results), tasks and geometries; returns the result count."""
    result_count = _job_result_count(session, job_ids)

    if job_ids:
        session.exec(delete(JobResult).where(col(JobResult.job_id).in_(job_ids)))
        session.exec(delete(ComputationJob).where(col(ComputationJob.id).in_(job_ids)))

    # Break the task <-> geometry cycle before deleting either side.
    if task_ids:
        session.exec(
            update(ComputationTask)
            .where(col(ComputationTask.id).in_(task_ids))
            .values(input_geometry_id=None, output_geometry_id=None, depends_on_task_id=None)
        )
    if geometry_ids:
        session.exec(
            update(MoleculeGeometry)
            .where(col(MoleculeGeometry.id).in_(geometry_ids))
            .values(origin_task_id=None)
        )

    if task_ids:
        session.exec(delete(ComputationTask).where(col(ComputationTask.id).in_(task_ids)))
    if geometry_ids:
        session.exec(delete(MoleculeGeometry).where(col(MoleculeGeometry.id).in_(geometry_ids)))
    return result_count


# ----------------------------------------------------------------------
# Project wipe
# ----------------------------------------------------------------------


def preview_project_wipe(
    session: Session, name: str, comp_root: Path, export_root: Path,
) -> dict:
    """Count everything ``wipe_project`` would delete. Changes nothing."""
    molecule_ids = _project_molecule_ids(session, name)
    state_ids, task_ids, geometry_ids, job_ids = _descendants(session, molecule_ids)
    entrypoint_ids = _queued_entrypoint_ids(session, name)
    result_count = _job_result_count(session, job_ids)

    comp_dirs = [comp_root / f"mol_{mid}" for mid in molecule_ids]
    # Never build this by concatenation: `name` comes straight from the URL
    # and `..` would resolve to the data root.
    export_dir = safe_subdirectory(export_root, name)

    # One budget for both, so a large project bounds the request rather
    # than the request bounding on the project. See PREVIEW_MEASURE_SECONDS.
    deadline = monotonic() + PREVIEW_MEASURE_SECONDS
    comp_bytes, comp_done = _measure(comp_dirs, deadline)
    export_bytes, export_done = _measure([export_dir], deadline)

    return {
        "project": name,
        "exists": bool(molecule_ids or entrypoint_ids),
        "protected": is_protected(name),
        "rows": {
            "molecules": len(molecule_ids),
            "states": len(state_ids),
            "tasks": len(task_ids),
            "geometries": len(geometry_ids),
            "jobs": len(job_ids),
            "job_results": result_count,
            "queued_entrypoints": len(entrypoint_ids),
        },
        "files": {
            "comp_data_dirs": sum(1 for d in comp_dirs if d.exists()),
            "comp_data_bytes": comp_bytes,
            "comp_data_human": human_bytes(comp_bytes),
            "export_dir_exists": export_dir.exists(),
            "export_bytes": export_bytes,
            "export_human": human_bytes(export_bytes),
            "total_human": human_bytes(comp_bytes + export_bytes),
            # False when the time budget cut the walk short: every figure
            # above is then a lower bound, and the dialog must say so
            # rather than under-report what is about to be deleted.
            "measured_completely": comp_done and export_done,
        },
        "confirmation_required": name,
    }


def wipe_project(
    session: Session,
    name: str,
    comp_root: Path,
    export_root: Path,
    *,
    delete_exports: bool = True,
    scheduler=None,
    background: bool = True,
) -> dict:
    """Delete a project's DB rows, raw comp_data, and exported data.

    Irreversible. The caller is responsible for having confirmed.

    Rows are deleted and committed *before* the directory trees are removed,
    the same order ``reset_database`` already used. Removing the files first
    meant that anything raising partway through the rmtree -- most easily a
    second wipe racing this one through the same directories -- left an
    emptied filesystem with every row intact, after which the pipeline marked
    every job "Job path missing" en masse. Deleting rows first inverts that:
    a failure leaves orphaned directories, which the log names and the return
    value reports, and which nothing depends on.

    The file deletion is the slow half -- measured on this network mount,
    ~65 ms per file, so a real project is minutes. It runs with no
    transaction open and, unless *background* is false, on its own thread:
    the directories are renamed out of the way first, so by the time this
    returns the project is gone as far as anything else can tell.

    Checks and row deletes run under one write lock (:func:`begin_write`); scancel
    runs after the commit, before any file is removed.

    Raises:
        ValueError: if the project is protected, has nothing to delete, or has a job
            mid-sbatch (:func:`_refuse_mid_sbatch`).
    """
    if is_protected(name):
        raise ValueError(f"Project {name!r} is protected and cannot be wiped.")

    # Resolve the export directory before deleting anything: an unsafe name
    # must fail here, not halfway through an rmtree.
    export_dir = safe_subdirectory(export_root, name) if delete_exports else None

    begin_write(session)
    molecule_ids = _project_molecule_ids(session, name)
    entrypoint_ids = _queued_entrypoint_ids(session, name)
    if not molecule_ids and not entrypoint_ids:
        raise ValueError(f"Project {name!r} has no molecules and no queued entrypoints.")

    _, task_ids, _, job_ids = _descendants(session, molecule_ids)
    _refuse_mid_sbatch(session, task_ids)
    live = _live_slurm_ids(session, job_ids)

    counts = _delete_molecule_rows(session, molecule_ids)
    if entrypoint_ids:
        session.exec(
            delete(CalculationEntrypoint).where(col(CalculationEntrypoint.id).in_(entrypoint_ids))
        )
    counts["queued_entrypoints"] = len(entrypoint_ids)
    session.commit()

    # Stop the cluster writing into trees that are about to disappear.
    cancelled, scancel_warning = _cancel_scheduled_jobs(live, scheduler)

    # Everything below runs with no transaction open. The project has already
    # left the database, so a failure here leaves orphaned directories that
    # the log names -- recoverable -- rather than an inconsistent database.
    batch = _batch_name()
    stale = _stale_trash(comp_root, export_root)
    staged = _stage([comp_root / f"mol_{mid}" for mid in molecule_ids], batch)
    comp_dirs_staged = len(staged)

    export_staged = False
    if export_dir is not None:
        staged_export = _stage([export_dir], batch)
        export_staged = bool(staged_export)
        staged += staged_export

    label = f"wipe of project {name!r}"
    removal = _remove_in_background(label, stale + staged, background=background)

    logger.warning(
        "WIPED project %r: %s rows, %d comp_data dirs, %d job(s) cancelled",
        name, counts, comp_dirs_staged, cancelled,
    )
    return {
        "project": name,
        "wiped": True,
        "rows": counts,
        "comp_data_dirs_removed": comp_dirs_staged,
        "export_removed": export_staged,
        "jobs_cancelled": cancelled,
        "scancel_warning": scancel_warning,
        "orphaned_dirs": removal.orphaned,
        "bytes_freed": removal.bytes_freed,
        "freed_human": human_bytes(removal.bytes_freed),
        "file_removal": removal.status(),
    }


# ----------------------------------------------------------------------
# Single molecule wipe
# ----------------------------------------------------------------------


def preview_molecule_wipe(session: Session, molecule_id: int, comp_root: Path) -> Optional[dict]:
    """Count what ``wipe_molecule`` would delete, or None if it doesn't exist."""
    mol = session.get(Molecule, molecule_id)
    if mol is None:
        return None
    state_ids, task_ids, geometry_ids, job_ids = _descendants(session, [molecule_id])
    comp_dir = comp_root / f"mol_{molecule_id}"
    size, _, complete = _scan(comp_dir, monotonic() + PREVIEW_MEASURE_SECONDS)
    return {
        "molecule_id": molecule_id,
        "smiles": mol.smiles,
        "project": mol.project_name,
        "rows": {
            "states": len(state_ids),
            "tasks": len(task_ids),
            "geometries": len(geometry_ids),
            "jobs": len(job_ids),
            "job_results": _job_result_count(session, job_ids),
        },
        "files": {
            "comp_data_dir": str(comp_dir),
            "exists": comp_dir.exists(),
            "bytes": size,
            "human": human_bytes(size),
            "measured_completely": complete,
        },
        # The SMILES is what the user sees in the table, so that's what
        # they're asked to type back.
        "confirmation_required": mol.smiles,
    }


def wipe_molecule(
    session: Session, molecule_id: int, comp_root: Path, *, scheduler=None,
) -> dict:
    """Delete one molecule, its states/tasks/jobs, and its raw files.

    The project's exported CSV/JSON is left alone -- it may describe other
    molecules. Re-export after wiping to refresh it.

    Rows go first and are committed before the files are removed, for the
    reasons given on :func:`wipe_project`; checks and row deletes run under one
    write lock (:func:`begin_write`).
    """
    begin_write(session)
    mol = session.get(Molecule, molecule_id)
    if mol is None:
        raise ValueError(f"Molecule {molecule_id} does not exist.")

    smiles, project = mol.smiles, mol.project_name

    from autodft.engine.nmr_references import is_reference_project

    if is_reference_project(project):
        raise ValueError(
            "NMR reference molecules are managed by the pipeline and cannot be wiped."
        )

    _, task_ids, _, job_ids = _descendants(session, [molecule_id])
    _refuse_mid_sbatch(session, task_ids)
    live = _live_slurm_ids(session, job_ids)

    counts = _delete_molecule_rows(session, [molecule_id])
    session.commit()

    cancelled, scancel_warning = _cancel_scheduled_jobs(live, scheduler)

    removed, freed = _remove_tree(comp_root / f"mol_{molecule_id}")
    _reset_disk_usage()  # the cached total no longer describes the tree
    logger.warning(
        "WIPED molecule %d (%s) from project %r, %d job(s) cancelled",
        molecule_id, smiles, project, cancelled,
    )
    return {
        "molecule_id": molecule_id,
        "smiles": smiles,
        "project": project,
        "wiped": True,
        "rows": counts,
        "comp_data_removed": removed,
        "jobs_cancelled": cancelled,
        "scancel_warning": scancel_warning,
        "bytes_freed": freed,
        "freed_human": human_bytes(freed),
    }


# ----------------------------------------------------------------------
# Task re-run
# ----------------------------------------------------------------------

_SEED_LABEL = "seed_from_task_"


def begin_write(session: Session) -> None:
    """Take SQLite's write lock now, so what is checked cannot change before the writes.

    The submit loop claims a job (``SUBMITTING``) with a write, so a claim either
    precedes the check or waits for the commit.
    """
    if session.get_bind().dialect.name != "sqlite":
        return
    connection = session.connection()
    if not connection.connection.dbapi_connection.in_transaction:  # a write already holds it
        connection.exec_driver_sql("BEGIN IMMEDIATE")


def _never_sent(job: ComputationJob) -> bool:
    return (job.slurm_jobid is None and job.slurm_status is None
            and not (job.fail_reason or "").startswith(SBATCH_FAILED))


def unsettled_jobs(session: Session, task_ids: list[int]) -> list[ComputationJob]:
    """Unjudged jobs of *task_ids* that went, or may have gone, to SLURM."""
    if not task_ids:
        return []
    return [
        job for job in session.exec(
            select(ComputationJob).where(
                col(ComputationJob.task_id).in_(task_ids),
                col(ComputationJob.success).is_(None),
            )
        ).all()
        if not _never_sent(job)
    ]


def in_flight_jobs(session: Session, task_ids: list[int]) -> list[ComputationJob]:
    """Jobs of *task_ids* SLURM may be running: claimed for sbatch, possibly accepted by
    an sbatch that reported failure, or submitted and not finished."""
    return [
        job for job in unsettled_jobs(session, task_ids)
        if job.slurm_jobid is None or job.slurm_status not in TERMINAL_SLURM_STATES
    ]


def rerun_blocker(session: Session, task: ComputationTask) -> Optional[str]:
    """Why *task* may not re-run: its project is archived or has an archive/export job running."""
    state = session.get(MoleculeState, task.state_id)
    molecule = session.get(Molecule, state.molecule_id) if state is not None else None
    if molecule is None:
        return None
    if molecule.archived:
        return "project archived; nothing can re-run"
    try:
        project_jobs.assert_no_active_job(session, molecule.project_name)
    except project_jobs.JobInProgress as exc:
        return str(exc)
    return None


def describe_jobs(jobs: list[ComputationJob]) -> str:
    """Job ids with their SLURM id or state, for a refusal message."""
    return ", ".join(
        f"job {j.id} ({j.slurm_jobid or j.slurm_status or 'sbatch outcome unknown'})" for j in jobs
    )


def _input_task_ids(task: ComputationTask) -> set[int]:
    """The tasks an ESD rate task reads (its ``inputs_json`` roles)."""
    stored = json.loads(task.inputs_json) if task.inputs_json else {}
    return {
        role[key] for name in ("S0", "S1", "T1")
        if isinstance(role := stored.get(name), dict)
        for key in ("opt", "soc") if key in role
    }


def _geometry_source(geom: MoleculeGeometry) -> Optional[int]:
    """The task a geometry came from: its origin, or the S0* opt an ESD seed copies."""
    if geom.origin_task_id is not None:
        return geom.origin_task_id
    source = (geom.label or "").removeprefix(_SEED_LABEL)
    return int(source) if source != geom.label and source.isdigit() else None


def task_dependents(session: Session, task: ComputationTask) -> tuple[list[ComputationTask], list[int]]:
    """Tasks built on *task*'s results, recursively, and the geometries they and *task* made."""
    state = session.get(MoleculeState, task.state_id)
    state_ids = list(session.exec(
        select(MoleculeState.id).where(MoleculeState.molecule_id == state.molecule_id)
    ).all()) if state is not None else [task.state_id]
    tasks = session.exec(select(ComputationTask).where(col(ComputationTask.state_id).in_(state_ids))).all()
    geometries = session.exec(
        select(MoleculeGeometry).where(col(MoleculeGeometry.state_id).in_(state_ids))
    ).all()

    found = {task.id}
    while True:
        made = {g.id for g in geometries if _geometry_source(g) in found}
        more = {
            t.id for t in tasks if t.id not in found and (
                t.depends_on_task_id in found or t.input_geometry_id in made
                or _input_task_ids(t) & found
            )
        }
        if not more:
            break
        found |= more
    return [t for t in tasks if t.id in found and t.id != task.id], sorted(made)


def _reopen_esd(session: Session, molecule_id: int, removed: set[int], rate_states: set[int]) -> None:
    """Let ESD seed again (its S0* was removed) or join again (a rate task was removed)."""
    from autodft.engine import photophysics

    for s1 in session.exec(
        select(MoleculeState).where(
            MoleculeState.molecule_id == molecule_id,
            col(MoleculeState.metadata_json).contains('"esd_role": "S1"'),
        )
    ).all():
        metadata = json.loads(s1.metadata_json)
        t1 = photophysics.partner(session, s1, "T1")
        reseed = metadata.get("esd_seed", {}).get("task") in removed
        if not reseed and not {s1.id, t1.id if t1 else None} & rate_states:
            continue
        if reseed:
            metadata.pop("esd_seed", None)
        metadata.pop("esd_done", None)
        s1.metadata_json = json.dumps(metadata)
        session.add(s1)


def _reopen_s0_seeds(session: Session, molecule_id: int, removed: set[int]) -> None:
    """Let states seeded from a removed S0* seed again."""
    for state in session.exec(
        select(MoleculeState).where(
            MoleculeState.molecule_id == molecule_id,
            col(MoleculeState.metadata_json).contains('"s0_seed"'),
        )
    ).all():
        metadata = json.loads(state.metadata_json)
        if metadata["s0_seed"].get("task") in removed:
            metadata.pop("s0_seed")
            state.metadata_json = json.dumps(metadata)
            session.add(state)


def _under(path: Path, root: Path) -> bool:
    """Whether *path* lies strictly inside *root*."""
    path, root = path.resolve(), root.resolve()
    return path != root and path.is_relative_to(root)


def rerun_task(session: Session, task: ComputationTask, comp_root: Path) -> dict:
    """Return *task* to ``created`` with a fresh run: its jobs, every task built on
    its results and their jobs, results, geometries and folders are deleted.

    Checks and writes run under one write lock (:func:`begin_write`).

    Raises:
        ValueError: if SLURM may be running any of these jobs, or :func:`rerun_blocker` objects.
    """
    begin_write(session)
    session.refresh(task)
    blocker = rerun_blocker(session, task)
    if blocker:
        raise ValueError(blocker)
    dependents, geometry_ids = task_dependents(session, task)
    task_ids = [task.id, *(t.id for t in dependents)]
    live = in_flight_jobs(session, task_ids)
    if live:
        raise ValueError(f"{len(live)} job(s) may be in SLURM: {describe_jobs(live)}; "
                         "wait for them or scancel them first.")

    jobs = session.exec(select(ComputationJob).where(col(ComputationJob.task_id).in_(task_ids))).all()
    folders = [Path(t.task_path) for t in dependents if t.task_path]
    folders += [Path(j.job_path) for j in jobs if j.task_id == task.id and j.job_path]

    state = session.get(MoleculeState, task.state_id)
    if state is not None:
        _reopen_esd(
            session, state.molecule_id, set(task_ids),
            {t.state_id for t in dependents if t.task_type.value.startswith("esd_")},
        )
        _reopen_s0_seeds(session, state.molecule_id, set(task_ids))

    task.status = TaskStatus.created
    task.output_geometry_id = None
    task.retry_base = None
    task.build_failures = 0
    task.has_followups = task.task_type in (TaskType.confsearch, TaskType.optimization)
    task.updated_at = datetime.now(timezone.utc)
    session.add(task)
    session.flush()
    results = _delete_task_rows(session, task_ids[1:], geometry_ids, [j.id for j in jobs])
    session.commit()

    removed, orphaned = 0, []
    for folder in folders:
        if not _under(folder, comp_root):
            logger.warning("Not deleting %s: outside comp_data %s", folder, comp_root)
            continue
        try:
            removed += _remove_tree(folder)[0]
        except OSError:
            logger.exception("Could not delete %s", folder)
            orphaned.append(str(folder))
    _reset_disk_usage()
    logger.warning("Task %d re-run: %d dependent task(s), %d job(s) deleted",
                   task.id, len(dependents), len(jobs))
    return {
        "tasks": len(dependents), "jobs": len(jobs), "job_results": results,
        "geometries": len(geometry_ids), "folders_removed": removed, "orphaned_dirs": orphaned,
    }


# ----------------------------------------------------------------------
# Full reset
# ----------------------------------------------------------------------


def preview_database_reset(session: Session) -> dict:
    """Count everything in the database. Reads no files.

    Sizing the data directory used to happen here, which made rendering the
    admin page a five-minute walk of the whole network mount while holding
    a database connection. It is now :func:`start_disk_usage`, asked for by
    the operator when they want the number. The reset itself never needed
    it: it stages directories with a rename and deletes them in the
    background, so what is on disk does not change what it does.
    """
    def _count(model) -> int:
        # COUNT(*) in SQLite, not len() over every id fetched into Python.
        # Seven tables, ~23k rows, 2.3 s per render of the admin page.
        return session.exec(select(func.count()).select_from(model)).one()

    projects = session.exec(select(Molecule.project_name).distinct()).all()

    return {
        "projects": sorted(p for p in projects if p),
        "rows": {
            "molecules": _count(Molecule),
            "states": _count(MoleculeState),
            "tasks": _count(ComputationTask),
            "geometries": _count(MoleculeGeometry),
            "jobs": _count(ComputationJob),
            "entrypoints": _count(CalculationEntrypoint),
            "job_results": _count(JobResult),
        },
        "confirmation_required": RESET_CONFIRMATION,
    }


def reset_database(
    session: Session,
    comp_root: Path,
    export_root: Path,
    *,
    delete_files: bool = True,
    keep_headers: bool = True,
    scheduler=None,
    background: bool = True,
) -> dict:
    """Empty every pipeline table and optionally every data directory.

    Headers are kept by default: they are the user's saved methods, not
    results, and losing them silently would be a nasty surprise. When they
    are dropped the standard set is reseeded so submissions still work.

    As with :func:`wipe_project`, the directories are renamed out of the way
    and deleted on a background thread unless *background* is false. Staging
    is what makes that safe here: molecule ids restart at 1 after a reset, so
    a worker resuming while the old tree was still being walked would create
    ``mol_1`` straight into it.

    Checks and deletes run under one write lock (:func:`begin_write`).

    Raises:
        ValueError: if a job is mid-sbatch (:func:`_refuse_mid_sbatch`).
    """
    begin_write(session)
    _refuse_mid_sbatch(session)
    counts = {
        "jobs": len(session.exec(select(ComputationJob.id)).all()),
        "tasks": len(session.exec(select(ComputationTask.id)).all()),
        "geometries": len(session.exec(select(MoleculeGeometry.id)).all()),
        "states": len(session.exec(select(MoleculeState.id)).all()),
        "molecules": len(session.exec(select(Molecule.id)).all()),
        "entrypoints": len(session.exec(select(CalculationEntrypoint.id)).all()),
        "job_results": len(session.exec(select(JobResult.id)).all()),
    }

    live = _live_slurm_ids(
        session, [j for j in session.exec(select(ComputationJob.id)).all() if j is not None],
    )

    # Same cycle-breaking order as the per-project wipe.
    session.exec(delete(JobResult))
    session.exec(delete(ComputationJob))
    session.exec(update(ComputationTask).values(
        input_geometry_id=None, output_geometry_id=None, depends_on_task_id=None))
    session.exec(update(MoleculeGeometry).values(origin_task_id=None))
    session.exec(delete(ComputationTask))
    session.exec(delete(MoleculeGeometry))
    session.exec(delete(MoleculeState))
    session.exec(delete(Molecule))
    session.exec(delete(CalculationEntrypoint))

    headers_dropped = 0
    if not keep_headers:
        from autodft.models.header import ComputationHeader
        headers_dropped = len(session.exec(select(ComputationHeader.id)).all())
        session.exec(delete(ComputationHeader))

    session.commit()

    # Stop the cluster before the directories it is writing into disappear.
    cancelled, scancel_warning = _cancel_scheduled_jobs(live, scheduler)

    targets: list[Path] = []
    if delete_files:
        # After the commit, so a failure here leaves orphaned directories
        # rather than an emptied filesystem with every row intact -- which
        # made the pipeline mark every job "Job path missing" en masse.
        batch = _batch_name()
        data_root = comp_root.parent if comp_root.parent == export_root.parent else None
        for root in (comp_root, export_root):
            if not root.exists():
                continue
            # Guard against a config where comp_data_path or export_data_path
            # IS the data path: emptying it would unlink autodft.db from
            # under the open session.
            if data_root is not None and root == data_root:
                logger.error("Refusing to empty %s: it is the data root", root)
                continue
            targets += _stage_whole(root, batch)
            logger.warning("Emptied data directory %s", root)
        # Swept after staging: a batch inside comp_data went with the root.
        targets += _stale_trash(comp_root, export_root, skip=batch)

    removal = _remove_in_background("database reset", targets, background=background)

    if not keep_headers:
        from autodft.db import _seed_default_headers, get_engine
        _seed_default_headers(get_engine())

    logger.warning("DATABASE RESET: %s, headers_dropped=%d, %d job(s) cancelled",
                   counts, headers_dropped, cancelled)
    return {
        "reset": True,
        "rows": counts,
        "headers_dropped": headers_dropped,
        "files_deleted": delete_files,
        "jobs_cancelled": cancelled,
        "scancel_warning": scancel_warning,
        "bytes_freed": removal.bytes_freed,
        "freed_human": human_bytes(removal.bytes_freed),
        "file_removal": removal.status(),
    }
