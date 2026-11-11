"""Main pipeline worker -- the forever loop that drives the DFT pipeline.

Instantiate :class:`PipelineWorker` with a :class:`Settings` and a
:class:`Scheduler`, then call :meth:`run_forever` (or :meth:`tick` for
a single cycle, useful in tests).
"""

from __future__ import annotations

import logging
import time
from typing import Collection

from autodft.config import Settings
from autodft.db import get_session
from autodft.engine.entrypoint_processor import process_next_entrypoint
from autodft.engine.photophysics import advance_photophysics
from autodft.engine.s0_seed import advance_s0_seeds
from autodft.engine.scheduler import Scheduler
from autodft.engine.state_machine import (
    create_retry_jobs,
    process_finished_jobs,
    start_followup_tasks,
    start_new_tasks,
    submit_pending_jobs,
    update_running_jobs,
    update_task_statuses,
)
from autodft.engine import circuit_breaker
from autodft.qm.base import QMEngine

logger = logging.getLogger(__name__)


class PipelineWorker:
    """Orchestrates the eight-step pipeline cycle.

    Args:
        settings: Application configuration.
        scheduler: Job scheduler implementation (SLURM or Local).
        qm_engine: Quantum-mechanics engine for input/output handling.
    """

    def __init__(
        self,
        settings: Settings,
        scheduler: Scheduler,
        qm_engine: QMEngine,
    ) -> None:
        self.settings = settings
        self.scheduler = scheduler
        self.qm_engine = qm_engine
        # Set by _entrypoint_step(); False once the queue is drained.
        self._last_entrypoint_processed = False

    def run_forever(self) -> None:
        """Main loop -- runs until interrupted (e.g. ``KeyboardInterrupt``).

        Each iteration calls :meth:`tick` and then sleeps for the
        configured interval.
        """
        logger.info(
            "Pipeline worker starting (interval=%ds)",
            self.settings.pipeline.loop_interval_seconds,
        )
        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                logger.info("Pipeline worker interrupted -- shutting down")
                break
            except Exception:
                logger.exception("Error in pipeline tick")

            time.sleep(self.settings.pipeline.loop_interval_seconds)

    def _run_step(self, session, label: str, fn) -> bool:
        """Run one pipeline step, commit it, and contain its failures.

        Each step commits on its own so that progress is durable the moment
        it is made. Previously the whole tick shared one transaction that was
        committed only at the very end, which meant a single raising job --
        a truncated ORCA output, a missing ensemble file -- discarded every
        other molecule's progress from that tick and then raised again on the
        next tick, forever. Returns True if the step completed.
        """
        logger.debug("Step: %s", label)
        try:
            fn()
            session.commit()
            return True
        except Exception:  # noqa: BLE001 - one bad step must not stop the rest
            logger.exception("Pipeline step %r failed; rolling it back and continuing", label)
            session.rollback()
            return False

    def tick(self) -> None:
        """Execute one complete pipeline cycle.

        Steps are committed individually and are isolated from each other:
        a failure in one step is logged and rolled back, and the remaining
        steps still run.
        """
        logger.debug("--- Pipeline tick start ---")

        with get_session() as session:
            self._run_step(session, "1: update running job statuses",
                           lambda: update_running_jobs(session, self.scheduler))

            self._run_step(session, "2: process finished jobs",
                           lambda: process_finished_jobs(
                               session, self.qm_engine, self.settings.pipeline.max_attempts))

            self._run_step(session, "3: update task statuses",
                           lambda: update_task_statuses(
                               session, max_attempts=self.settings.pipeline.max_attempts))

            self._run_step(session, "4: start follow-up tasks",
                           lambda: start_followup_tasks(session, self.settings))

            self._run_step(session, "4b: advance photophysics",
                           lambda: advance_photophysics(session, self.settings))

            self._run_step(session, "4c: seed states from S0",
                           lambda: advance_s0_seeds(session))

            # An owner whose campaign is failing systematically gets no new
            # jobs and no expansion. Their submitted work keeps running; other
            # owners carry on.
            blocked = set(circuit_breaker.check(session, self.settings))
            for owner in sorted(blocked):
                logger.critical(
                    "Circuit breaker active for %s -- not creating or submitting their jobs", owner,
                )

            # Step 5: expand queued entrypoints. Each entrypoint commits
            # separately -- process_next_entrypoint() rolls the session back
            # when a SMILES can't be expanded, and without a commit per item
            # that rollback also discarded every entrypoint expanded before it
            # in the same tick (including their time_started marks, so they
            # were re-processed indefinitely).
            # Expansion is gated on the database, not on squeue. The REST API
            # accepts every submission the moment it arrives and parks it in
            # calculation_entrypoints; that table is the buffer, so a campaign
            # of any size can be handed over in one go without the submitting
            # script blocking. What we bound here is how much of that buffer
            # gets materialised into jobs and ORCA input directories ahead of
            # the queue, so the on-disk tree grows with what SLURM can
            # actually absorb.
            max_backlog = self.settings.pipeline.max_unsubmitted_jobs
            for _ in range(self.settings.pipeline.max_simultaneous_entrypoints):
                # Re-counted per entrypoint: one expansion adds a task per
                # requested state, so a single tick can cross the ceiling.
                backlog = self._unsubmitted_job_count(session, blocked)
                if max_backlog and backlog >= max_backlog:
                    logger.debug(
                        "%d job(s) awaiting submission (max %d); pausing entrypoint expansion",
                        backlog,
                        max_backlog,
                    )
                    break
                processed = self._run_step(
                    session, "5: process next entrypoint",
                    lambda: self._entrypoint_step(session, blocked),
                )
                if not processed or not self._last_entrypoint_processed:
                    break

            self._run_step(session, "6a: create retry jobs",
                           lambda: create_retry_jobs(session, self.settings, self.qm_engine, blocked))
            self._run_step(session, "6b: start new tasks",
                           lambda: start_new_tasks(session, self.settings, self.qm_engine, blocked))

            self._run_step(session, "7: submit pending jobs",
                           lambda: submit_pending_jobs(session, self.scheduler, self.settings, blocked))

        logger.debug("--- Pipeline tick complete ---")

    def _entrypoint_step(self, session, blocked_owners: Collection[str] = ()) -> None:
        """Process one entrypoint, recording whether there was one to process."""
        self._last_entrypoint_processed = process_next_entrypoint(
            session, self.settings, blocked_owners,
        )

    @staticmethod
    def _unsubmitted_job_count(session, blocked_owners: Collection[str] = ()) -> int:
        """Work that exists in the database but has not reached SLURM.

        Counts jobs created but not yet submitted, plus tasks that have not
        been turned into a job yet -- both become queue pressure within a
        tick or two, so both have to hold expansion back. An owner whose
        breaker tripped is left out, so their stalled backlog cannot hold
        back everyone else's expansion.
        """
        from sqlalchemy import func
        from sqlmodel import col, select

        from autodft.models.enums import TaskStatus
        from autodft.models.job import ComputationJob
        from autodft.models.molecule import Molecule
        from autodft.models.state import MoleculeState
        from autodft.models.task import ComputationTask

        unblocked = [circuit_breaker.not_owned_by(session, blocked_owners)] if blocked_owners else []
        # Archived molecules' leftovers never run, so they hold nothing back.
        jobs = session.exec(
            select(func.count()).select_from(ComputationJob)
            .join(ComputationTask, col(ComputationTask.id) == col(ComputationJob.task_id))
            .join(MoleculeState, col(MoleculeState.id) == col(ComputationTask.state_id))
            .join(Molecule, col(Molecule.id) == col(MoleculeState.molecule_id))
            .where(
                col(ComputationJob.slurm_jobid).is_(None),
                col(ComputationJob.slurm_status).is_(None),
                col(ComputationJob.success).is_(None),
                col(Molecule.archived).is_(False),
                *unblocked,
            )
        ).one()
        tasks = session.exec(
            select(func.count()).select_from(ComputationTask)
            .join(MoleculeState, col(MoleculeState.id) == col(ComputationTask.state_id))
            .join(Molecule, col(Molecule.id) == col(MoleculeState.molecule_id))
            .where(
                ComputationTask.status == TaskStatus.created,
                col(Molecule.archived).is_(False),
                *unblocked,
            )
        ).one()
        return int(jobs) + int(tasks)
