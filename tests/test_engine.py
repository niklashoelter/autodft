"""Tests for the engine: entrypoint expansion, state machine, retry chain.

Before these existed the whole of ``autodft/engine/`` was untested -- every
bug fixed in 0.2.0 lived in code no test executed. Each test here pins one
of those behaviours so it cannot regress silently.

The tests drive the real functions against an in-memory SQLite database with
stub scheduler / QM engines, so no ORCA, no SLURM and no filesystem outside
tmp_path are involved.
"""

from __future__ import annotations

import json
import re
import time

import pytest
from sqlmodel import Session, select

from autodft.config import Settings
from autodft.engine.entrypoint_processor import (
    calculate_altered_multiplicity,
    get_charge_and_multiplicity,
    validate_smiles,
)
from autodft.engine.state_machine import (
    _followups_were_expected,
    _get_job_charge_multiplicity,
    _parse_resources_from_header,
    process_finished_jobs,
    submit_pending_jobs,
)
from autodft.models import (
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeState,
    SlurmStatus,
    TaskStatus,
    TaskType,
)
from autodft.qm.base import QMResult


# ======================================================================
# Charge / multiplicity
# ======================================================================


class TestChargeAndMultiplicity:
    @pytest.mark.parametrize(
        "smiles,charge,multiplicity",
        [
            ("c1ccccc1", 0, 1),         # closed-shell neutral
            ("CCO", 0, 1),
            ("[O-][n+]1ccccc1", 0, 1),  # zwitterion, net neutral
            ("CC(=O)[O-]", -1, 1),      # anion
            ("[NH4+]", 1, 1),           # cation
            ("[CH3]", 0, 2),            # doublet radical
            ("F[S](F)(F)(F)F", 0, 2),   # SF5 radical, bracketed S
        ],
    )
    def test_derives_charge_and_multiplicity(self, smiles, charge, multiplicity):
        assert get_charge_and_multiplicity(smiles) == (charge, multiplicity)

    def test_electron_parity_is_consistent(self):
        """An odd electron count must not come back as a singlet."""
        from rdkit import Chem

        for smiles in ("[CH3]", "F[S](F)(F)(F)F", "C[C]1CC(C#N)C1"):
            mol = Chem.MolFromSmiles(smiles)
            electrons = (
                sum(a.GetAtomicNum() for a in mol.GetAtoms())
                + sum(a.GetTotalNumHs() for a in mol.GetAtoms())
                - Chem.GetFormalCharge(mol)
            )
            _, multiplicity = get_charge_and_multiplicity(smiles)
            # odd electrons -> even multiplicity, and vice versa
            assert (electrons % 2) == (multiplicity % 2 == 0)

    @pytest.mark.parametrize(
        "start,delta,expected",
        [
            (1, +1, 2),   # singlet oxidised -> doublet
            (1, -1, 2),   # singlet reduced  -> doublet
            (2, +1, 1),   # doublet oxidised -> singlet
            (2, -1, 1),   # doublet reduced  -> singlet
        ],
    )
    def test_altered_multiplicity(self, start, delta, expected):
        assert calculate_altered_multiplicity(start, 0, delta) == expected

    def test_round_trip_for_supported_references(self):
        """ox-then-back must return to the reference for singlets/doublets.

        These are the only reference states the pipeline supports. The rule
        is not an involution for higher spin, which is why T1 is refused for
        open-shell references rather than derived.
        """
        for start in (1, 2):
            oxidised = calculate_altered_multiplicity(start, 0, 1)
            assert calculate_altered_multiplicity(oxidised, 1, 0) == start


class TestValidateSmiles:
    def test_accepts_a_normal_molecule(self):
        result = validate_smiles("c1ccc(O)cc1")
        assert result["valid"] is True
        assert result["warning"] is None
        assert result["multiplicity"] == 1

    def test_rejects_unparseable(self):
        assert validate_smiles("not a smiles")["valid"] is False

    def test_rejects_single_atoms(self):
        # GOAT cannot run on a monomer.
        assert validate_smiles("[Fe+2]")["valid"] is False

    def test_warns_on_chemdraw_style_radicals(self):
        """`[C]` reads as every missing valence being an unpaired electron."""
        result = validate_smiles("FS(F)(F)(F)(F)C[C]c1ccccc1")
        assert result["valid"] is True
        assert result["multiplicity"] == 3
        assert result["warning"] is not None
        assert "unpaired" in result["warning"]


# ======================================================================
# Entrypoint expansion
# ======================================================================


def _settings(tmp_path) -> Settings:
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    settings.ensure_directories()
    return settings


def _queue(session: Session, smiles: str, **metadata):
    from autodft.models.entrypoint import CalculationEntrypoint

    priority = metadata.pop("priority", 10)
    meta = {"project_name": "t", "request_confsearch": True}
    meta.update(metadata)
    entry = CalculationEntrypoint(
        smiles=smiles,
        request_metadata=json.dumps(meta),
        priority=priority,
        header_confsearch="!GOAT XTB2\n",
        header_optimization="!B3LYP OPT\n",
        header_singlepoint="!B3LYP\n",
    )
    session.add(entry)
    session.commit()
    return entry


class TestStateCreation:
    """These need the real DB session the processor commits through."""

    def test_states_get_the_right_charge_and_multiplicity(self, engine, tmp_path, monkeypatch):
        from autodft.engine import entrypoint_processor as ep

        settings = _settings(tmp_path)
        with Session(engine) as session:
            _queue(session, "c1ccc(O)cc1", request_T1=True, request_ox=True, request_red=True)
            monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
            ep.process_next_entrypoint(session, settings)
            session.commit()

            states = {
                s.description: (s.charge, s.multiplicity)
                for s in session.exec(select(MoleculeState)).all()
            }

        assert states["S0"] == (0, 1)
        assert states["T1"] == (0, 3)
        assert states["ox"] == (1, 2)
        assert states["red"] == (-1, 2)

    def test_priority_reaches_the_molecule(self, engine, tmp_path, monkeypatch):
        """The submission throttle scales with priority, and a job can only
        reach its priority through the molecule row."""
        from autodft.engine import entrypoint_processor as ep

        settings = _settings(tmp_path)
        with Session(engine) as session:
            _queue(session, "CCO", priority=7)
            monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
            ep.process_next_entrypoint(session, settings)
            session.commit()

            assert session.exec(select(Molecule)).one().priority == 7

    def test_resubmission_raises_but_never_lowers_priority(
        self, engine, tmp_path, monkeypatch,
    ):
        from autodft.engine import entrypoint_processor as ep

        settings = _settings(tmp_path)
        with Session(engine) as session:
            monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
            for priority in (5, 9, 2):
                _queue(session, "CCO", priority=priority)
                ep.process_next_entrypoint(session, settings)
                session.commit()

            assert session.exec(select(Molecule)).one().priority == 9

    def test_t1_is_refused_for_an_open_shell_reference(self, engine, tmp_path, monkeypatch):
        """A doublet reference would otherwise get an impossible mult-3 T1."""
        from autodft.engine import entrypoint_processor as ep

        settings = _settings(tmp_path)
        with Session(engine) as session:
            entry = _queue(session, "C[C]1CC(C#N)C1", request_T1=True)
            monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
            ep.process_next_entrypoint(session, settings)
            session.commit()

            refreshed = session.get(type(entry), entry.id)
            assert refreshed.processing_error is not None
            assert "closed-shell" in refreshed.processing_error
            assert session.exec(select(MoleculeState)).all() == []

    def test_state_metadata_defaults_are_not_all_false(self, engine, tmp_path, monkeypatch):
        """An omitted key must not become False and kill the chain."""
        from autodft.engine import entrypoint_processor as ep

        settings = _settings(tmp_path)
        with Session(engine) as session:
            _queue(session, "CCO")  # nothing but project_name + confsearch
            monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
            ep.process_next_entrypoint(session, settings)
            session.commit()

            state = session.exec(select(MoleculeState)).first()
            metadata = json.loads(state.metadata_json)

        assert metadata["request_optimization"] is True
        assert metadata["request_singlepoint"] is True
        assert metadata["max_conformers_S0"] == 1


# ======================================================================
# Vertical excitations
# ======================================================================


class TestVerticalChargeMultiplicity:
    @staticmethod
    def _state(charge: int, multiplicity: int) -> MoleculeState:
        return MoleculeState(
            id=1, molecule_id=1, description="S0",
            charge=charge, multiplicity=multiplicity,
        )

    def test_vertical_ox_and_red_from_a_singlet(self):
        state = self._state(0, 1)
        assert _get_job_charge_multiplicity(TaskType.singlepoint_vert_ox, state) == (1, 2)
        assert _get_job_charge_multiplicity(TaskType.singlepoint_vert_red, state) == (-1, 2)

    def test_spin_flip_maps_singlet_and_triplet(self):
        assert _get_job_charge_multiplicity(
            TaskType.singlepoint_vert_spin_change, self._state(0, 1)) == (0, 3)
        assert _get_job_charge_multiplicity(
            TaskType.singlepoint_vert_spin_change, self._state(0, 3)) == (0, 1)

    def test_spin_flip_refuses_a_doublet(self):
        """Previously returned the state's own multiplicity, producing a
        singlepoint byte-identical to the base one."""
        with pytest.raises(ValueError, match="singlet or triplet"):
            _get_job_charge_multiplicity(
                TaskType.singlepoint_vert_spin_change, self._state(0, 2))

    def test_backward_leg_returns_to_the_reference(self):
        """ox state's vert_red must land back on the neutral singlet."""
        ox = self._state(1, 2)
        assert _get_job_charge_multiplicity(TaskType.singlepoint_vert_red, ox) == (0, 1)

    def test_base_tasks_keep_the_state_values(self):
        state = self._state(-1, 2)
        assert _get_job_charge_multiplicity(TaskType.singlepoint, state) == (-1, 2)
        assert _get_job_charge_multiplicity(TaskType.optimization, state) == (-1, 2)


# ======================================================================
# Resource parsing
# ======================================================================


class TestResourceParsing:
    def test_pal_block(self):
        header = "!B3LYP\n%pal nprocs 8 end\n%maxcore 1500\n"
        assert _parse_resources_from_header(header) == (8, 1500)

    def test_is_case_insensitive(self):
        header = "!B3LYP\n%PAL NPROCS 24 END\n%MaxCore 2000\n"
        assert _parse_resources_from_header(header) == (24, 2000)

    def test_multiline_pal_block(self):
        header = "!B3LYP\n%pal\n  nprocs 16\nend\n%maxcore 1000\n"
        assert _parse_resources_from_header(header) == (16, 1000)

    def test_route_line_pal_shorthand(self):
        """! PAL16 was invisible, so nprocs silently fell back to the config."""
        header = "! wB97X-D3 def2-TZVP PAL16\n%maxcore 4000\n"
        assert _parse_resources_from_header(header) == (16, 4000)

    def test_pal_block_wins_over_shorthand(self):
        header = "! B3LYP PAL4\n%pal nprocs 32 end\n%maxcore 1000\n"
        assert _parse_resources_from_header(header) == (32, 1000)

    def test_missing_values_are_none(self):
        assert _parse_resources_from_header("! B3LYP def2-SVP\n") == (None, None)


# ======================================================================
# Job lifecycle
# ======================================================================


class _StubEngine:
    """QM engine that returns a canned result or raises."""

    def __init__(self, result=None, raises=None):
        self.result = result
        self.raises = raises

    def check_output(self, job_path, task_type):
        if self.raises is not None:
            raise self.raises
        return self.result


class _StubScheduler:
    """*queues* models whether SLURM makes submitted jobs wait.

    False is an idle partition: every job starts at once, so the waiting
    count stays where it was. True is a busy one: each submission joins the
    queue. The distinction is the whole point of the throttle.
    """

    def __init__(self, succeed=True, queue_length=0, queues=False):
        self.succeed = succeed
        self.queue_length = queue_length
        self.queues = queues
        self.submitted = 0

    def get_queue_length(self):
        if self.queue_length < 0:
            return -1
        return self.queue_length + (self.submitted if self.queues else 0)

    def get_status(self, job_id):
        return SlurmStatus.RUNNING

    def submit(self, script, nice=0):
        self.submitted += 1

        class _Result:
            success = self.succeed
            job_id = "4242" if self.succeed else None
            error = None if self.succeed else "sbatch: command not found"
        return _Result()


@pytest.fixture()
def task_with_job(session, sample_task):
    job = ComputationJob(task_id=sample_task.id, attempt=1)
    session.add(job)
    session.commit()
    session.refresh(job)
    return sample_task, job


class TestProcessFinishedJobs:
    @pytest.mark.parametrize(
        "status", ["CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "COMPLETED"],
    )
    def test_every_non_running_status_is_judged(self, session, task_with_job, tmp_path, status):
        """These used to match neither the polling filter nor the terminal
        query, so the job kept success=NULL and its task never moved."""
        _, job = task_with_job
        job.slurm_status = status
        job.job_path = str(tmp_path)
        session.add(job)
        session.commit()

        process_finished_jobs(session, _StubEngine(QMResult(success=True, checks={})))
        session.refresh(job)
        assert job.success is not None

    @pytest.mark.parametrize("status", ["RUNNING", "PENDING"])
    def test_in_flight_jobs_are_left_alone(self, session, task_with_job, tmp_path, status):
        _, job = task_with_job
        job.slurm_status = status
        job.job_path = str(tmp_path)
        session.add(job)
        session.commit()

        process_finished_jobs(session, _StubEngine(QMResult(success=True, checks={})))
        session.refresh(job)
        assert job.success is None

    def test_a_raising_parser_fails_only_that_job(self, session, task_with_job, tmp_path):
        """An exception here used to unwind the whole pipeline tick."""
        _, job = task_with_job
        job.slurm_status = SlurmStatus.COMPLETED
        job.job_path = str(tmp_path)
        session.add(job)
        session.commit()

        process_finished_jobs(
            session, _StubEngine(raises=FileNotFoundError("input.finalensemble.xyz")),
        )
        session.refresh(job)
        assert job.success is False
        assert "Output parsing failed" in job.fail_reason


class TestSubmitPendingJobs:
    def test_missing_submit_script_counts_as_a_failed_attempt(
        self, session, task_with_job, tmp_path, monkeypatch,
    ):
        """Skipping left success NULL forever: no retry, no progress, and the
        same warning every tick."""
        _, job = task_with_job
        job.job_path = str(tmp_path)  # no submit.cmd inside
        session.add(job)
        session.commit()

        submit_pending_jobs(session, _StubScheduler(), Settings())
        session.refresh(job)
        assert job.success is False
        assert "Submit script missing" in job.fail_reason

    def test_sbatch_failure_is_recorded_but_retried(
        self, session, task_with_job, tmp_path,
    ):
        """sbatch failures are usually transient -- slurmctld restarting, a
        socket timeout, a QOS limit clearing. Failing the job would burn an
        attempt for a calculation that never ran, so the reason is recorded
        for visibility but success stays NULL and the next tick retries."""
        _, job = task_with_job
        job.job_path = str(tmp_path)
        (tmp_path / "submit.cmd").write_text("#!/bin/bash\n")
        session.add(job)
        session.commit()

        submit_pending_jobs(session, _StubScheduler(succeed=False), Settings())
        session.refresh(job)
        assert job.success is None
        assert "sbatch failed" in job.fail_reason
        assert job.slurm_jobid is None

    def test_a_good_submission_records_the_slurm_id(self, session, task_with_job, tmp_path):
        _, job = task_with_job
        job.job_path = str(tmp_path)
        (tmp_path / "submit.cmd").write_text("#!/bin/bash\n")
        session.add(job)
        session.commit()

        submit_pending_jobs(session, _StubScheduler(), Settings())
        session.refresh(job)
        assert job.slurm_jobid == 4242
        assert job.slurm_status == SlurmStatus.PENDING
        assert job.success is None


class TestPriorityThrottle:
    """Submission runs until the queue holds priority * slots waiting jobs."""

    @staticmethod
    def _ready(session, sample_task, tmp_path, priority):
        """A submittable job whose molecule carries *priority*."""
        state = session.get(MoleculeState, sample_task.state_id)
        molecule = session.get(Molecule, state.molecule_id)
        molecule.priority = priority
        session.add(molecule)
        job = ComputationJob(task_id=sample_task.id, attempt=1, job_path=str(tmp_path))
        (tmp_path / "submit.cmd").write_text("#!/bin/bash\n")
        session.add(job)
        session.commit()
        session.refresh(job)
        return job

    def test_a_full_queue_stops_submission(self, session, sample_task, tmp_path):
        job = self._ready(session, sample_task, tmp_path, priority=1)
        # priority 1 * 10 slots = 10; the queue already holds 10.
        submit_pending_jobs(session, _StubScheduler(queue_length=10), Settings())
        session.refresh(job)
        assert job.slurm_jobid is None
        assert job.success is None  # not a failure -- just deferred

    def test_room_below_the_cap_submits(self, session, sample_task, tmp_path):
        job = self._ready(session, sample_task, tmp_path, priority=1)
        submit_pending_jobs(session, _StubScheduler(queue_length=9), Settings())
        session.refresh(job)
        assert job.slurm_jobid == 4242

    def test_priority_scales_the_cap(self, session, sample_task, tmp_path):
        """The same queue depth that blocks priority 1 leaves room at 5."""
        job = self._ready(session, sample_task, tmp_path, priority=5)
        submit_pending_jobs(session, _StubScheduler(queue_length=30), Settings())
        session.refresh(job)
        assert job.slurm_jobid == 4242

    def test_archived_molecules_are_not_submitted(self, session, sample_task, tmp_path):
        """Archiving wipes the comp_data tree; submitting into it would burn
        the retry budget against directories that no longer exist."""
        job = self._ready(session, sample_task, tmp_path, priority=1)
        state = session.get(MoleculeState, sample_task.state_id)
        molecule = session.get(Molecule, state.molecule_id)
        molecule.archived = True
        session.add(molecule)
        session.commit()

        submit_pending_jobs(session, _StubScheduler(), Settings())
        session.refresh(job)
        assert job.slurm_jobid is None

    @staticmethod
    def _many_ready(session, sample_task, tmp_path, priority, count):
        """*count* submittable jobs, all on molecules at *priority*."""
        state = session.get(MoleculeState, sample_task.state_id)
        molecule = session.get(Molecule, state.molecule_id)
        molecule.priority = priority
        session.add(molecule)
        (tmp_path / "submit.cmd").write_text("#!/bin/bash\n")
        jobs = []
        for attempt in range(1, count + 1):
            job = ComputationJob(
                task_id=sample_task.id, attempt=attempt, job_path=str(tmp_path),
            )
            session.add(job)
            jobs.append(job)
        session.commit()
        return jobs

    def test_jobs_that_start_at_once_do_not_count_against_the_cap(
        self, session, sample_task, tmp_path,
    ):
        """The cap is on *waiting* jobs. On an idle partition nothing waits,
        so submission should keep going -- it used to stop after one cap-full
        per tick because every submitted job was counted as queued."""
        self._many_ready(session, sample_task, tmp_path, priority=1, count=40)
        scheduler = _StubScheduler(queue_length=0, queues=False)

        submit_pending_jobs(session, scheduler, Settings())

        assert scheduler.submitted == 40  # not 10

    def test_it_keeps_going_until_the_cluster_is_actually_full(
        self, session, sample_task, tmp_path,
    ):
        """The point of the throttle: submit until SLURM really has
        `priority * 10` of our jobs waiting. A count limit per tick used to
        bind first, so an idle partition was filled 100 jobs a minute no
        matter how much of it was free."""
        self._many_ready(session, sample_task, tmp_path, priority=1, count=250)
        scheduler = _StubScheduler(queue_length=0, queues=False)

        submit_pending_jobs(session, scheduler, Settings())

        assert scheduler.submitted == 250

    def test_one_tick_yields_on_wall_clock_not_on_a_job_count(
        self, session, sample_task, tmp_path,
    ):
        """Submission must not run so long that status polling starves.
        The bound is time, because that is what actually matters -- 20 slow
        sbatch calls can cost more than 2000 fast ones."""
        self._many_ready(session, sample_task, tmp_path, priority=1, count=60)

        class _SlowScheduler(_StubScheduler):
            def submit(self, script, nice=0):
                time.sleep(0.02)
                return super().submit(script, nice)

        settings = Settings()
        settings.pipeline.max_submission_seconds_per_tick = 0.1
        scheduler = _SlowScheduler(queue_length=0, queues=False)

        submit_pending_jobs(session, scheduler, settings)

        # Stopped early, but made real progress rather than stalling.
        assert 0 < scheduler.submitted < 60

    def test_the_count_backstop_still_applies_when_configured(
        self, session, sample_task, tmp_path,
    ):
        """Off by default, but an operator can still pin it."""
        self._many_ready(session, sample_task, tmp_path, priority=1, count=40)
        settings = Settings()
        settings.pipeline.max_submissions_per_tick = 7
        scheduler = _StubScheduler(queue_length=0, queues=False)

        submit_pending_jobs(session, scheduler, settings)

        assert scheduler.submitted == 7

    def test_a_partition_that_makes_jobs_wait_still_stops_at_the_cap(
        self, session, sample_task, tmp_path,
    ):
        """The other half: when submissions really do queue, the re-poll sees
        them and the cap holds."""
        self._many_ready(session, sample_task, tmp_path, priority=1, count=40)
        scheduler = _StubScheduler(queue_length=0, queues=True)

        submit_pending_jobs(session, scheduler, Settings())

        # 10 (the cap) plus at most one recheck interval of overshoot.
        assert 10 <= scheduler.submitted <= 10 + 5

    def test_jobs_the_scheduler_has_not_looked_at_do_not_count(
        self, session, sample_task, tmp_path,
    ):
        """A job is PENDING from the instant sbatch returns -- SLURM
        schedules on its own cycle, not on submit. Counting raw PD made the
        loop measure its own submissions and stop after one capful on a
        completely idle cluster."""
        self._many_ready(session, sample_task, tmp_path, priority=1, count=40)

        class _JustSubmitted(_StubScheduler):
            """Everything submitted shows up PD with no reason yet."""

            def get_pending_breakdown(self):
                return 0, self.submitted

            def get_queue_length(self):
                return 0

        scheduler = _JustSubmitted()
        submit_pending_jobs(session, scheduler, Settings())

        assert scheduler.submitted == 40

    def test_jobs_waiting_on_resources_do_count(
        self, session, sample_task, tmp_path,
    ):
        """The other side: once slurmctld says a job is waiting for the
        cluster, the cap applies."""
        self._many_ready(session, sample_task, tmp_path, priority=1, count=40)

        class _ClusterFull(_StubScheduler):
            def get_pending_breakdown(self):
                return self.submitted, 0

            def get_queue_length(self):
                return self.submitted

        scheduler = _ClusterFull()
        submit_pending_jobs(session, scheduler, Settings())

        assert 10 <= scheduler.submitted <= 15

    def test_a_scheduler_without_a_breakdown_still_works(
        self, session, sample_task, tmp_path,
    ):
        """LocalScheduler and any third-party one only expose a count."""
        self._many_ready(session, sample_task, tmp_path, priority=1, count=5)

        class _CountOnly:
            submitted = 0

            def get_queue_length(self):
                return 0

            def submit(self, script, nice=0):
                _CountOnly.submitted += 1

                class _R:
                    success = True
                    job_id = "1"
                    error = None
                return _R()

        scheduler = _CountOnly()
        submit_pending_jobs(session, scheduler, Settings())
        assert _CountOnly.submitted == 5

    def test_unknown_queue_length_defers_rather_than_floods(
        self, session, sample_task, tmp_path,
    ):
        """squeue returning -1 must not read as 'the queue is empty'."""
        job = self._ready(session, sample_task, tmp_path, priority=1)
        submit_pending_jobs(session, _StubScheduler(queue_length=-1), Settings())
        session.refresh(job)
        assert job.slurm_jobid is None
        assert job.fail_reason is None


class TestDeadEndDetection:
    def test_a_healthy_confsearch_is_not_flagged(self, session, sample_state, sample_header):
        """Guards the dead-end check against false positives: a confsearch
        that does spawn optimizations must stay successful."""
        from autodft.engine.state_machine import start_followup_tasks
        from autodft.models.geometry import MoleculeGeometry

        sample_state.metadata_json = json.dumps({
            "request_optimization": True, "max_conformers_S0": 2,
        })
        confsearch = ComputationTask(
            task_type=TaskType.confsearch, state_id=sample_state.id,
            header_id=sample_header.id, status=TaskStatus.successful,
            has_followups=True,
        )
        session.add_all([sample_state, confsearch])
        session.commit()
        session.refresh(confsearch)

        for i, energy in enumerate([-10.0, -9.0, -8.0]):
            session.add(MoleculeGeometry(
                state_id=sample_state.id, xyz_data="C 0 0 0\nH 1 0 0\n",
                energy=energy, origin_task_id=confsearch.id, label=f"conformer_{i}",
            ))
        session.commit()

        start_followup_tasks(session, Settings())
        session.refresh(confsearch)

        optimizations = session.exec(
            select(ComputationTask).where(
                ComputationTask.task_type == TaskType.optimization,
                ComputationTask.depends_on_task_id == confsearch.id,
            )
        ).all()
        assert len(optimizations) == 2                 # capped by max_conformers
        assert confsearch.status == TaskStatus.successful   # NOT flagged
        assert confsearch.has_followups is False

        # The cheapest conformers are the ones carried forward.
        carried = {session.get(MoleculeGeometry, o.input_geometry_id).energy
                   for o in optimizations}
        assert carried == {-10.0, -9.0}

    def test_a_confsearch_with_no_conformers_is_flagged(
        self, session, sample_state, sample_header,
    ):
        """The dead end that ended five real states with everything green."""
        from autodft.engine.state_machine import start_followup_tasks

        sample_state.metadata_json = json.dumps({"request_optimization": True})
        confsearch = ComputationTask(
            task_type=TaskType.confsearch, state_id=sample_state.id,
            header_id=sample_header.id, status=TaskStatus.successful,
            has_followups=True,
        )
        session.add_all([sample_state, confsearch])
        session.commit()
        session.refresh(confsearch)

        start_followup_tasks(session, Settings())   # no geometries exist
        session.refresh(confsearch)

        assert confsearch.status == TaskStatus.failed
        assert confsearch.has_followups is False

    def test_expected_followups_by_stage(self):
        confsearch = ComputationTask(task_type=TaskType.confsearch, state_id=1, header_id=1)
        optimization = ComputationTask(task_type=TaskType.optimization, state_id=1, header_id=1)
        singlepoint = ComputationTask(task_type=TaskType.singlepoint, state_id=1, header_id=1)

        assert _followups_were_expected(confsearch, {}) is True
        assert _followups_were_expected(optimization, {}) is True
        # Explicitly turning a stage off means zero follow-ups is correct.
        assert _followups_were_expected(confsearch, {"request_optimization": False}) is False
        assert _followups_were_expected(optimization, {"request_singlepoint": False}) is False
        # Singlepoints are the end of the chain.
        assert _followups_were_expected(singlepoint, {}) is False


# ======================================================================
# Retry chain
# ======================================================================


class TestRetryStrategies:
    def test_resource_increase_is_case_insensitive(self):
        """%MaxCore / %PAL silently no-op'd while submit.cmd got more cores."""
        from autodft.qm.orca.retry import FailureInfo, IncreaseResources

        strategy = IncreaseResources(nprocs=32)
        inp = "!M062X\n%MaxCore 1500\n%PAL NPROCS 8 END\n*xyzfile 0 1 input.xyz\n"
        # A memory failure so %maxcore is actually rewritten (case-insensitively).
        failure = FailureInfo(
            fail_reason="['Termination', 'OUT_OF_MEMORY']", previous_job_path="",
            attempt=2, charge=0, multiplicity=1,
        )
        new_input, _ = strategy.modify(inp, "", failure)
        assert "nprocs 32" in new_input.lower()
        assert "2500" in new_input  # raised to the ceiling, not left at 1500

    def test_maxcore_is_capped_at_the_ceiling(self):
        """A header asking for more than MAX_MAXCORE_MB is clamped down to it --
        the hard cap that stops retries ballooning to 129 GB jobs."""
        from autodft.qm.orca.retry import FailureInfo, IncreaseResources, MAX_MAXCORE_MB

        strategy = IncreaseResources(nprocs=32)
        inp = "!M062X\n%maxcore 8000\n%pal nprocs 8 end\n"
        submit = "#SBATCH --ntasks-per-node=8\n#SBATCH --mem=64400\n#SBATCH --time=1-00:00:00\n"
        failure = FailureInfo(
            fail_reason="['Termination', 'OUT_OF_MEMORY']", previous_job_path="",
            attempt=2, charge=0, multiplicity=1,
        )
        new_input, new_submit = strategy.modify(inp, submit, failure)
        assert f"%maxcore {MAX_MAXCORE_MB}" in new_input
        assert "8000" not in new_input
        assert f"--mem={32 * (MAX_MAXCORE_MB + 50)}" in new_submit

    def test_non_memory_failure_keeps_maxcore(self):
        """A timeout / SCF failure gets more cores and time, NOT more memory --
        throwing memory at a non-memory failure just re-ran the same job big."""
        from autodft.qm.orca.retry import FailureInfo, IncreaseResources

        strategy = IncreaseResources(nprocs=32)
        inp = "!M062X\n%maxcore 1000\n%pal nprocs 8 end\n"
        submit = "#SBATCH --ntasks-per-node=8\n#SBATCH --mem=8400\n#SBATCH --time=1-00:00:00\n"
        failure = FailureInfo(
            fail_reason="['Termination']", previous_job_path="", attempt=2,
            charge=0, multiplicity=1,
        )
        new_input, new_submit = strategy.modify(inp, submit, failure)
        assert "%maxcore 1000" in new_input       # per-rank memory untouched
        assert "nprocs 32" in new_input           # cores still escalate
        assert f"--mem={32 * (1000 + 50)}" in new_submit

    def test_orca_memory_message_triggers_escalation(self, tmp_path):
        """An ORCA out-of-memory message (no SLURM OOM state) still counts as
        a memory failure and raises %maxcore."""
        from autodft.qm.orca.retry import FailureInfo, IncreaseResources

        (tmp_path / "output.out").write_text(
            "ORCA aborting: insufficient memory to allocate the integrals\n",
            encoding="utf-8",
        )
        strategy = IncreaseResources(nprocs=32)
        inp = "!M062X\n%maxcore 1000\n%pal nprocs 8 end\n"
        failure = FailureInfo(
            fail_reason="['Termination']", previous_job_path=str(tmp_path),
            attempt=2, charge=0, multiplicity=1,
        )
        new_input, _ = strategy.modify(inp, "", failure)
        assert "%maxcore 2500" in new_input

    def test_memory_ceiling_reduces_ranks_not_per_rank_memory(self):
        """With %maxcore hard-capped at 2500, 32 ranks is 32 x 2550 = 81.6 GB;
        a smaller max_mem ceiling reduces the *rank count* to fit rather than
        the per-rank memory (clamping --mem alone would let ORCA allocate more
        per rank than SLURM granted)."""
        from autodft.qm.orca.retry import FailureInfo, IncreaseResources

        strategy = IncreaseResources(nprocs=32, max_mem_per_job_mb=64000)
        inp = "!M062X\n%maxcore 1500\n%pal nprocs 8 end\n"
        submit = "#SBATCH --ntasks-per-node=8\n#SBATCH --mem=12400\n#SBATCH --time=1-00:00:00\n"
        failure = FailureInfo(
            fail_reason="['Termination', 'OUT_OF_MEMORY']", previous_job_path="",
            attempt=2, charge=0, multiplicity=1,
        )
        new_input, new_submit = strategy.modify(inp, submit, failure)

        ranks = int(re.search(r"nprocs (\d+)", new_input).group(1))
        assert ranks == 64000 // (2500 + 50)   # reduced to fit the ceiling
        assert "%maxcore 2500" in new_input    # per-rank memory at the cap
        assert f"--ntasks-per-node={ranks}" in new_submit
        assert int(re.search(r"--mem=(\d+)", new_submit).group(1)) <= 64000

    def test_rank_count_never_drops_below_the_header(self):
        """Fitting the memory ceiling must not turn escalation into a downgrade."""
        from autodft.qm.orca.retry import FailureInfo, IncreaseResources

        strategy = IncreaseResources(nprocs=32, max_mem_per_job_mb=8000)
        inp = "!M062X\n%maxcore 16000\n%pal nprocs 12 end\n"
        failure = FailureInfo(
            fail_reason="['Termination', 'OUT_OF_MEMORY']", previous_job_path="",
            attempt=2, charge=0, multiplicity=1,
        )
        new_input, _ = strategy.modify(inp, "", failure)
        assert "nprocs 12" in new_input
        assert "%maxcore 2500" in new_input   # clamped down from 16000

    def test_tighten_convergence_is_not_a_noop_on_tightscf_headers(self):
        """Every shipped header contains TightSCF, and the old guard made the
        whole strategy do nothing for them."""
        from autodft.qm.orca.retry import FailureInfo, TightenConvergence

        inp = "!M062X def2-SVP TightSCF OPT FREQ\n%maxcore 1500\n*xyzfile 0 1 input.xyz\n"
        failure = FailureInfo(
            fail_reason="['Imaginary Frequencies']", previous_job_path="", attempt=2,
            charge=0, multiplicity=1,
        )
        new_input, _ = TightenConvergence().modify(inp, "", failure)
        assert new_input != inp
        assert "Convergence tight" in new_input

    def test_strategies_honour_the_retry_config(self):
        """[pipeline.retry] used to be read by nothing at all."""
        from autodft.qm.orca.retry import IncreaseResources, build_strategies

        settings = Settings()
        settings.pipeline.retry.increased_nprocs = 64
        settings.pipeline.retry.increased_mem_per_core = 7000

        resources = [s for s in build_strategies(settings) if isinstance(s, IncreaseResources)]
        assert resources[0].nprocs == 64
        assert resources[0].mem_per_core == 7000


# ======================================================================
# Regressions introduced by the 0.2.0 work itself
# ======================================================================


class TestSlurmStateClassification:
    """The first fix here over-corrected: inverting the terminal whitelist to
    "anything not RUNNING/PENDING/UNKNOWN" swept in states that mean the job
    is still going."""

    @pytest.mark.parametrize(
        "status", ["COMPLETING", "SUSPENDED", "CONFIGURING", "REQUEUED", "RESIZING"],
    )
    def test_transient_states_are_not_parsed(self, session, task_with_job, tmp_path, status):
        """A job in COMPLETING (epilog, node drain) parsed mid-write would be
        marked failed with ['Termination'] -- the exact reason that escalates
        its retry to 32 ranks for four days."""
        _, job = task_with_job
        job.slurm_status = status
        job.job_path = str(tmp_path)
        session.add(job)
        session.commit()

        process_finished_jobs(session, _StubEngine(QMResult(success=True, checks={})))
        session.refresh(job)
        assert job.success is None

    @pytest.mark.parametrize(
        "status",
        ["COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
         "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"],
    )
    def test_terminal_states_are_judged(self, session, task_with_job, tmp_path, status):
        _, job = task_with_job
        job.slurm_status = status
        job.job_path = str(tmp_path)
        session.add(job)
        session.commit()

        process_finished_jobs(session, _StubEngine(QMResult(success=True, checks={})))
        session.refresh(job)
        assert job.success is not None

    def test_transient_states_stay_in_the_polling_set(self):
        """Otherwise a job that passes through COMPLETING is never re-polled
        and keeps that status permanently."""
        from autodft.models.enums import TERMINAL_SLURM_STATES, TRANSIENT_SLURM_STATES

        assert "COMPLETING" in TRANSIENT_SLURM_STATES
        assert "UNKNOWN" in TRANSIENT_SLURM_STATES
        assert not (TRANSIENT_SLURM_STATES & TERMINAL_SLURM_STATES)


class TestConformerRankingScale:
    def test_never_compares_corrected_against_uncorrected(self):
        """e_combined = e_singlepoint + (G - E_el), and that correction is
        positive and large, so a conformer missing it would win every time."""
        from autodft.extraction.extractor import ConformerResult, PipelineExtractor

        corrected = ConformerResult(
            molecule_id=1, smiles="CCO", state="S0", conformer_index=1, opt_task_id=1,
            e_singlepoint=-500.50, e_correction=0.15, e_combined=-500.35,
        )
        uncorrected = ConformerResult(
            molecule_id=1, smiles="CCO", state="S0", conformer_index=2, opt_task_id=2,
            e_singlepoint=-500.48, e_correction=None, e_combined=None,
        )

        picked = PipelineExtractor._pick_reported_conformer([corrected, uncorrected])
        assert picked is corrected

    def test_falls_back_to_singlepoint_only_when_nothing_is_corrected(self):
        from autodft.extraction.extractor import ConformerResult, PipelineExtractor

        a = ConformerResult(molecule_id=1, smiles="CCO", state="S0", conformer_index=1,
                            opt_task_id=1, e_singlepoint=-500.20)
        b = ConformerResult(molecule_id=1, smiles="CCO", state="S0", conformer_index=2,
                            opt_task_id=2, e_singlepoint=-500.60)
        assert PipelineExtractor._pick_reported_conformer([a, b]) is b


class TestRetryReplayOrdering:
    def test_perturbation_uses_the_most_recent_failure(
        self, session, sample_task, sample_header, tmp_path, monkeypatch,
    ):
        """Replaying perturbation over every prior failure meant the geometry
        that survived came from attempt 1, discarding the later re-optimised
        structure -- and the second application silently no-ops, because the
        first already replaced *xyzfile with an inline block."""
        from autodft.engine import state_machine as sm
        from autodft.qm.orca import retry as retry_mod

        old_dir, new_dir = tmp_path / "job_1", tmp_path / "job_2"
        for d in (old_dir, new_dir):
            d.mkdir()
        (tmp_path / "input.inp").write_text("!M062X OPT FREQ\n%pal nprocs 4 end\n*xyzfile 0 1 input.xyz\n")
        (tmp_path / "submit.cmd").write_text("#SBATCH --ntasks-per-node=4\n")

        seen: list[str] = []

        class _Probe(retry_mod.PerturbImaginaryMode):
            def applies(self, failure, task_type):
                return True

            def modify(self, input_content, submit_content, failure):
                seen.append(failure.previous_job_path)
                return input_content, submit_content

        monkeypatch.setattr(retry_mod, "build_strategies", lambda settings=None: [_Probe()])

        for attempt, job_dir in ((1, old_dir), (2, new_dir)):
            session.add(ComputationJob(
                task_id=sample_task.id, attempt=attempt, success=False,
                fail_reason="['Imaginary Frequencies']", job_path=str(job_dir),
            ))
        session.commit()
        last = session.exec(
            select(ComputationJob).where(ComputationJob.attempt == 2)
        ).first()

        sm._apply_retry_modifications(
            session, tmp_path, sample_task, last, 3, 0, 1, Settings(),
        )

        # Applied exactly once, against the newest failure.
        assert seen == [str(new_dir)]


class TestCircuitBreaker:
    """max_attempts bounds retries per task; nothing bounded the campaign.
    A systematic error would fail every molecule in turn, each burning its
    escalated retry budget."""

    @staticmethod
    def _judged(session, sample_state, sample_header, failed: int, successful: int):
        for i in range(failed + successful):
            session.add(ComputationTask(
                task_type=TaskType.optimization,
                state_id=sample_state.id,
                header_id=sample_header.id,
                status=TaskStatus.failed if i < failed else TaskStatus.successful,
            ))
        session.commit()

    def test_ratio_counts_only_judged_tasks(self, session, sample_state, sample_header):
        from autodft.engine.circuit_breaker import recent_failure_ratio

        self._judged(session, sample_state, sample_header, failed=3, successful=7)
        session.add(ComputationTask(   # still running: says nothing either way
            task_type=TaskType.optimization, state_id=sample_state.id,
            header_id=sample_header.id, status=TaskStatus.pending,
        ))
        session.commit()

        ratio, failed, judged = recent_failure_ratio(session, window=100)
        assert (failed, judged) == (3, 10)
        assert ratio == pytest.approx(0.3)

    def test_below_threshold_does_not_trip(self, session, sample_state, sample_header, tmp_path):
        from autodft.engine.circuit_breaker import check

        settings = _settings(tmp_path)
        # 89 of 100 failed -- just under the 90% trip threshold.
        self._judged(session, sample_state, sample_header, failed=89, successful=11)
        assert check(session, settings) is None

    def test_above_threshold_trips_and_latches(self, session, sample_state, sample_header, tmp_path):
        from autodft.engine.circuit_breaker import check, read_state, reset

        settings = _settings(tmp_path)
        # 90 of the last 100 failed -- exactly the shipped 0.9 threshold, and
        # the comparison is inclusive so this trips.
        self._judged(session, sample_state, sample_header, failed=90, successful=10)

        tripped = check(session, settings)
        assert tripped is not None
        assert tripped["failed"] == 90

        # Latches: once submission stops nothing new is judged, so the ratio
        # cannot recover on its own and must be cleared deliberately.
        assert check(session, settings) is not None
        assert reset(settings.data_path) is True
        assert read_state(settings.data_path) is None

    def test_too_few_samples_never_trips(self, session, sample_state, sample_header, tmp_path):
        """Otherwise the first couple of failures in a fresh project would
        halt everything."""
        from autodft.engine.circuit_breaker import check

        settings = _settings(tmp_path)
        self._judged(session, sample_state, sample_header, failed=5, successful=0)  # 100%
        assert settings.pipeline.failure_breaker_min_samples > 5
        assert check(session, settings) is None

    def test_disabled_by_config(self, session, sample_state, sample_header, tmp_path):
        from autodft.engine.circuit_breaker import check

        settings = _settings(tmp_path)
        settings.pipeline.failure_breaker_enabled = False
        self._judged(session, sample_state, sample_header, failed=40, successful=0)
        assert check(session, settings) is None


# ======================================================================
# Entrypoint expansion backpressure
# ======================================================================


class TestExpansionBackpressure:
    """The REST API accepts everything immediately and parks it in
    calculation_entrypoints. Expansion, not submission, is what bounds how
    much of that buffer becomes jobs and ORCA input directories on disk."""

    def test_backlog_counts_unsubmitted_jobs_and_unstarted_tasks(
        self, session, sample_task,
    ):
        from autodft.engine.pipeline import PipelineWorker

        # sample_task is `created` with no job yet: it will become one.
        assert sample_task.status == TaskStatus.created
        assert PipelineWorker._unsubmitted_job_count(session) == 1

        session.add(ComputationJob(task_id=sample_task.id, attempt=1))
        session.commit()
        # Now the job exists too and has not reached SLURM, so both count.
        assert PipelineWorker._unsubmitted_job_count(session) == 2

        sample_task.status = TaskStatus.pending
        session.add(sample_task)
        session.commit()
        assert PipelineWorker._unsubmitted_job_count(session) == 1

    def test_expansion_pauses_above_the_ceiling(self, tmp_path, monkeypatch):
        """Needs a real on-disk database: tick() opens its own session."""
        from sqlalchemy import func

        from autodft.db import get_session, init_db, reset_engine
        from autodft.engine import entrypoint_processor as ep
        from autodft.engine.pipeline import PipelineWorker
        from autodft.models.entrypoint import CalculationEntrypoint
        from autodft.qm.orca.parser import OrcaParser

        settings = _settings(tmp_path)
        settings.pipeline.max_unsubmitted_jobs = 4  # one molecule's worth
        monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")

        reset_engine()
        try:
            init_db(settings)
            with get_session(settings) as session:
                for smiles in ("CCO", "CCC", "CCN", "CCF"):
                    _queue(session, smiles, request_T1=True, request_ox=True,
                           request_red=True, request_confsearch=False)

            PipelineWorker(
                settings=settings, scheduler=_StubScheduler(),
                qm_engine=OrcaParser(orca=settings.orca),
            ).tick()

            with get_session(settings) as session:
                # The first entrypoint yields 4 states -> 4 tasks, which is
                # the ceiling, so the other three stay queued for a later
                # tick rather than materialising input directories now.
                expanded = [
                    e for e in session.exec(select(CalculationEntrypoint)).all()
                    if e.time_started is not None
                ]
                assert len(expanded) == 1
                assert session.exec(
                    select(func.count()).select_from(Molecule)
                ).one() == 1
        finally:
            reset_engine()
