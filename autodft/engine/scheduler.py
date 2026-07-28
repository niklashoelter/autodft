"""Scheduler abstraction: submit, monitor, and cancel computational jobs.

Provides a ``SlurmScheduler`` for production HPC clusters and a
``LocalScheduler`` for local testing without SLURM.
"""

from __future__ import annotations

import getpass
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SubmitResult:
    """Outcome of a job submission attempt."""

    success: bool
    job_id: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class Scheduler(ABC):
    """Interface that every job scheduler must implement."""

    @abstractmethod
    def submit(self, script_path: Path, nice: int = 0) -> SubmitResult:
        """Submit a job script and return the result."""
        ...

    @abstractmethod
    def get_status(self, job_id: str) -> str:
        """Return the current status string for *job_id*."""
        ...

    @abstractmethod
    def get_queue_length(self) -> int:
        """Return the number of jobs genuinely waiting in the queue."""
        ...

    def get_pending_breakdown(self) -> tuple[int, int]:
        """``(waiting_for_the_cluster, awaiting_the_scheduler)``.

        Schedulers that cannot tell the two apart report everything as
        waiting, which is the conservative reading.
        """
        return self.get_queue_length(), 0

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a running/pending job.  Return ``True`` on success."""
        ...

    def get_statuses(self, job_ids: list[str]) -> dict[str, str]:
        """Return ``{job_id: status}`` for many jobs at once.

        Default implementation falls back to one :meth:`get_status` call per
        job; schedulers that can answer in bulk should override it.
        """
        return {jid: self.get_status(jid) for jid in job_ids}

    def cancel_many(self, job_ids: list[str]) -> int:
        """Cancel many jobs, returning how many were cancelled."""
        return sum(1 for jid in job_ids if self.cancel(jid))


# ---------------------------------------------------------------------------
# SLURM implementation
# ---------------------------------------------------------------------------

class SlurmScheduler(Scheduler):
    """Production scheduler that delegates to SLURM CLI tools."""

    # sacct/scancel were always invoked directly while sbatch/squeue went
    # through `bash -l -c`, which re-sources the (NFS-homed) login profile on
    # every call. Measured on the controller: `bash -l -c true` takes 1.05 s
    # against 2.5 ms for a direct exec -- 400x, paid per submission. At a few
    # hundred submissions per tick that alone exceeds the loop interval.
    #
    # So: use the binary directly when it is on PATH, and only fall back to a
    # login shell when it isn't (which is the only case the wrapper was ever
    # buying anything).
    _LOGIN_SHELL = ["bash", "-l", "-c"]

    def __init__(self, partition: str = "CPU", nice: int = 1000) -> None:
        self.partition = partition
        self.nice = nice

    @staticmethod
    def _command(argv: list[str]) -> list[str]:
        """Return *argv* directly, or wrapped in a login shell if needed."""
        import shutil

        if shutil.which(argv[0]) is not None:
            return argv
        logger.debug("%s not on PATH; falling back to a login shell", argv[0])
        return SlurmScheduler._LOGIN_SHELL + [" ".join(str(a) for a in argv)]

    # -- submit ------------------------------------------------------------

    def submit(self, script_path: Path, nice: int | None = None) -> SubmitResult:
        """Run ``sbatch`` on *script_path* and parse the returned job ID.

        Args:
            script_path: Absolute path to the ``submit.cmd`` file.
            nice: SLURM nice value (overrides instance default if given).

        Returns:
            A :class:`SubmitResult` with the SLURM job ID on success.
        """
        nice_val = nice if nice is not None else self.nice
        script_path = Path(script_path)
        cmd = self._command(["sbatch", f"--nice={nice_val}", str(script_path)])
        env = os.environ.copy()

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, env=env,
                cwd=str(script_path.parent),
            )
            # Output: "Submitted batch job 12345\n"
            job_id = result.stdout.strip().split()[-1]
            if not job_id.isdigit():
                return SubmitResult(
                    success=False,
                    error=f"Could not parse job ID from sbatch output: {result.stdout}",
                )
            logger.info("Submitted SLURM job %s from %s", job_id, script_path)
            return SubmitResult(success=True, job_id=job_id)

        except subprocess.CalledProcessError as exc:
            logger.error("sbatch failed: %s", exc.stderr)
            return SubmitResult(success=False, error=f"{exc}: {exc.stderr}")

    # -- status ------------------------------------------------------------

    @staticmethod
    def _normalize_state(raw: str) -> str:
        """Reduce a raw ``sacct`` State field to a bare SLURM state name.

        ``sacct`` reports a job cancelled by a user as ``"CANCELLED by 12345"``,
        and when the State column is width-truncated that arrives as the single
        token ``"CANCELLED+"`` (the ``+`` marks truncation). Neither form equals
        the bare ``"CANCELLED"`` the state sets are keyed on, so an un-normalized
        value lands in *neither* TRANSIENT_SLURM_STATES nor TERMINAL_SLURM_STATES:
        the job is then never re-polled *and* never judged, wedging its task in
        ``pending`` forever. Take the first word and drop a trailing ``+`` so
        every cancelled-job spelling collapses to ``"CANCELLED"``.
        """
        if not raw:
            return raw
        return raw.split()[0].rstrip("+")

    def get_status(self, job_id: str) -> str:
        """Query ``sacct`` for the state of *job_id*.

        Returns one of the standard SLURM state strings
        (``PENDING``, ``RUNNING``, ``COMPLETED``, ``FAILED``, ``TIMEOUT``,
        ``CANCELLED``) or ``"UNKNOWN"`` on error.
        """
        env = os.environ.copy()
        cmd = ["sacct", "-j", str(job_id), "--format=JobID,State", "--noheader"]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, env=env,
            )
            for line in result.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == str(job_id):
                    state = self._normalize_state(parts[1])
                    logger.debug("SLURM job %s status: %s", job_id, state)
                    return state
            logger.warning("Job ID %s not found in sacct output", job_id)
            return "UNKNOWN"
        except subprocess.CalledProcessError as exc:
            logger.error("sacct query failed for job %s: %s", job_id, exc.stderr)
            return "UNKNOWN"

    # Job ids per sacct invocation. The command line has to stay well under
    # ARG_MAX and slurmdbd answers a batched query in roughly the time of a
    # single one.
    _SACCT_CHUNK = 500

    def get_statuses(self, job_ids: list[str]) -> dict[str, str]:
        """Query many jobs in one ``sacct`` call per chunk.

        One subprocess per in-flight job per tick does not scale: at 3000
        in-flight jobs that is 3000 sacct round-trips every loop, minutes of
        wall clock and enough load to degrade slurmdbd for everyone else on
        the cluster. Batching turns 3000 queries into 6.
        """
        out: dict[str, str] = {}
        if not job_ids:
            return out

        env = os.environ.copy()
        for start in range(0, len(job_ids), self._SACCT_CHUNK):
            chunk = [str(j) for j in job_ids[start:start + self._SACCT_CHUNK]]
            cmd = self._command([
                "sacct", "-j", ",".join(chunk), "--format=JobID,State", "--noheader",
            ])
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True, env=env,
                )
            except subprocess.CalledProcessError as exc:
                logger.error("Batched sacct query failed: %s", exc.stderr)
                continue

            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) < 2:
                    continue
                # Steps appear as "12345.batch" / "12345.0"; keep the parent.
                job_id = parts[0].split(".")[0]
                if job_id in chunk and job_id not in out:
                    out[job_id] = self._normalize_state(parts[1])

        missing = [j for j in job_ids if str(j) not in out]
        if missing:
            logger.warning("sacct returned no state for %d job(s)", len(missing))
        for job_id in missing:
            out[str(job_id)] = "UNKNOWN"
        return out

    # -- queue length ------------------------------------------------------

    # squeue reason codes meaning "slurmctld has not evaluated this job
    # yet", as opposed to "this job is waiting for the cluster". A job is
    # PENDING from the instant sbatch returns -- SLURM does not schedule on
    # submit, it schedules on its own cycle -- so counting raw PD made the
    # submission loop measure the jobs it had just submitted and stop after
    # one capful, on a completely idle partition.
    _UNSCHEDULED_REASONS = {"none", "null", "(null)", ""}

    def get_pending_breakdown(self) -> tuple[int, int]:
        """``(waiting_for_the_cluster, awaiting_the_scheduler)``.

        Both counts are our own jobs in the configured partition. The first
        is what the throttle means by "queued": slurmctld has looked at the
        job and cannot start it yet. The second is jobs sbatch has accepted
        but the scheduler has not considered.

        Returns ``(-1, -1)`` if squeue could not be queried.
        """
        env = os.environ.copy()
        user = getpass.getuser()
        # %r is the pending reason. `-h` suppresses the header.
        cmd = self._command([
            "squeue", "-t", "PD", "-p", self.partition, "-u", user,
            "-h", "-o", "%i|%r",
        ])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, env=env,
            )
        except (subprocess.CalledProcessError, ValueError) as exc:
            logger.error("Failed to query SLURM queue length: %s", exc)
            return -1, -1

        waiting = unscheduled = 0
        reasons: dict[str, int] = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            _, _, reason = line.partition("|")
            reason = reason.strip()
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason.lower() in self._UNSCHEDULED_REASONS:
                unscheduled += 1
            else:
                waiting += 1

        if reasons:
            # Logged so the reason vocabulary of this cluster is visible;
            # the set above is the only site-specific assumption here.
            logger.debug("Pending reasons: %s", reasons)
        return waiting, unscheduled

    def get_queue_length(self) -> int:
        """Our own jobs that are genuinely waiting for the cluster.

        Excludes jobs the scheduler has not evaluated yet -- see
        :meth:`get_pending_breakdown`.
        """
        waiting, _ = self.get_pending_breakdown()
        return waiting

    # -- cancel ------------------------------------------------------------

    def cancel(self, job_id: str) -> bool:
        """Cancel a SLURM job via ``scancel``."""
        env = os.environ.copy()
        cmd = ["scancel", str(job_id)]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            logger.info("Cancelled SLURM job %s", job_id)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("scancel failed for job %s: %s", job_id, exc.stderr)
            return False

    def cancel_many(self, job_ids: list[str]) -> int:
        """Cancel many jobs in one ``scancel`` call.

        Used by the destructive admin operations: wiping a project whose
        jobs are still queued left them running against directories that no
        longer exist, writing output into a deleted tree.
        """
        if not job_ids:
            return 0
        env = os.environ.copy()
        cmd = ["scancel", *(str(j) for j in job_ids)]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            logger.warning("Cancelled %d SLURM job(s)", len(job_ids))
            return len(job_ids)
        except subprocess.CalledProcessError as exc:
            logger.error("scancel failed for %d job(s): %s", len(job_ids), exc.stderr)
            return 0


# ---------------------------------------------------------------------------
# Local (subprocess) implementation -- for testing without SLURM
# ---------------------------------------------------------------------------

class LocalScheduler(Scheduler):
    """Run job scripts as local subprocesses for testing purposes.

    Submitted jobs are tracked in-memory; ``get_queue_length`` always
    returns 0 so the pipeline never throttles.
    """

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen] = {}
        self._counter: int = 0
        self._finished: dict[str, int] = {}  # job_id -> returncode

    def submit(self, script_path: Path, nice: int = 0) -> SubmitResult:
        """Execute the script in the background as a subprocess."""
        script_path = Path(script_path)
        if not script_path.exists():
            return SubmitResult(success=False, error=f"Script not found: {script_path}")

        self._counter += 1
        job_id = str(self._counter)

        try:
            proc = subprocess.Popen(
                ["bash", str(script_path)],
                cwd=str(script_path.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._processes[job_id] = proc
            logger.info("Local job %s started (PID %d)", job_id, proc.pid)
            return SubmitResult(success=True, job_id=job_id)
        except OSError as exc:
            return SubmitResult(success=False, error=str(exc))

    def get_status(self, job_id: str) -> str:
        """Check whether the subprocess is still running."""
        if job_id in self._finished:
            return "COMPLETED" if self._finished[job_id] == 0 else "FAILED"

        proc = self._processes.get(job_id)
        if proc is None:
            return "UNKNOWN"

        retcode = proc.poll()
        if retcode is None:
            return "RUNNING"

        # Process finished -- record and remove
        self._finished[job_id] = retcode
        del self._processes[job_id]
        return "COMPLETED" if retcode == 0 else "FAILED"

    def get_queue_length(self) -> int:
        """Local scheduler never has a queue backlog."""
        return 0

    def cancel(self, job_id: str) -> bool:
        """Terminate the subprocess if it is still running."""
        proc = self._processes.get(job_id)
        if proc is None:
            return False
        proc.terminate()
        self._finished[job_id] = -1
        del self._processes[job_id]
        logger.info("Terminated local job %s", job_id)
        return True
