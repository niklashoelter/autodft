"""Durable background jobs for project archive / export.

Pins the properties the feature must have:

* a job runs to completion on a background thread and records its outcome,
  so closing the website does not lose it;
* the running row is a per-project lock -- a second archive/export, a wipe,
  or a submission on that project is refused while it runs;
* the controller skips advancing a paused project's tasks, and the
  destructive archive waits for in-flight SLURM jobs before deleting;
* a restart does not leave a ghost: a still-running row is reconciled to
  failed at startup.

Everything here runs against a throwaway file-backed SQLite database; the
production database is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autodft import accounts
from autodft.api import project_jobs, routes
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.models import (
    ComputationHeader,
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeState,
    ProjectJob,
    ProjectJobKind,
    ProjectJobStatus,
    TaskStatus,
    TaskType,
)
from autodft.engine import state_machine
from tests.test_export_paths import _seed_result
from tests.test_orca_parser import (
    ENERGY_OUTPUT, NORMAL_TERMINATION, OPTIMIZATION_CONVERGED, VIBRATIONAL_FREQUENCIES_CLEAN,
)

PROJECT = "owner/screening"


@pytest.fixture()
def env(tmp_path):
    """A throwaway DB with one user, one project, and one molecule."""
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    routes.set_active_settings(settings)
    with get_session(settings) as session:
        owner, key = accounts.create_user(session, "owner")
        accounts.get_or_create_project(session, owner, "screening")
        session.add(Molecule(smiles="CCO", project_name=PROJECT))
        session.commit()
        owner_id = owner.id
    yield {"settings": settings, "tmp_path": tmp_path, "owner_id": owner_id, "key": key}
    project_jobs.join_all(timeout=10)
    reset_engine()


def _make_task(session, project=PROJECT, status=TaskStatus.created):
    """Build a molecule -> state -> optimization task chain in *project*."""
    header = ComputationHeader(header_text="!B3LYP\n", description="t", validated=True)
    session.add(header)
    session.commit()
    session.refresh(header)

    mol = Molecule(smiles="CCC", project_name=project)
    session.add(mol)
    session.commit()
    session.refresh(mol)

    state = MoleculeState(
        molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
        optimization_header_id=header.id, singlepoint_header_id=header.id,
    )
    session.add(state)
    session.commit()
    session.refresh(state)

    task = ComputationTask(
        task_type=TaskType.optimization, status=status,
        state_id=state.id, header_id=header.id,
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


# ----------------------------------------------------------------------
# The job lifecycle
# ----------------------------------------------------------------------


class TestLifecycle:
    def test_a_job_runs_on_a_thread_and_records_success(self, env):
        job = project_jobs.start_job(
            owner_id=env["owner_id"], qualified_name=PROJECT,
            kind=ProjectJobKind.export_csv, params={}, settings=env["settings"],
        )
        assert job["status"] == "running"

        project_jobs.join_all(timeout=10)

        with get_session() as session:
            row = project_jobs.get_job(session, job["id"])
            assert row.status == ProjectJobStatus.successful
            assert row.finished_at is not None
            result = json.loads(row.result_json)
        assert result["format"] == "csv"
        assert result["downloadable"] is False

    def test_a_raising_body_is_recorded_as_failed(self, env, monkeypatch):
        from autodft.extraction.extractor import PipelineExtractor

        def _boom(self, *a, **k):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(PipelineExtractor, "export_summary_csv", _boom)

        job = project_jobs.start_job(
            owner_id=env["owner_id"], qualified_name=PROJECT,
            kind=ProjectJobKind.export_csv, params={}, settings=env["settings"],
        )
        project_jobs.join_all(timeout=10)

        with get_session() as session:
            row = project_jobs.get_job(session, job["id"])
        assert row.status == ProjectJobStatus.failed
        assert "disk on fire" in row.error
        # A failed job releases the project: nothing is left running.
        with get_session() as session:
            assert project_jobs.active_job(session, PROJECT) is None


# ----------------------------------------------------------------------
# One operation per project
# ----------------------------------------------------------------------


class TestExclusivity:
    def _park_running_job(self, owner_id):
        with get_session() as session:
            session.add(ProjectJob(
                qualified_name=PROJECT, owner_id=owner_id,
                kind=ProjectJobKind.archive, status=ProjectJobStatus.running,
            ))
            session.commit()

    def test_a_second_job_is_refused(self, env):
        self._park_running_job(env["owner_id"])
        with pytest.raises(project_jobs.JobInProgress):
            project_jobs.start_job(
                owner_id=env["owner_id"], qualified_name=PROJECT,
                kind=ProjectJobKind.export_csv, params={}, settings=env["settings"],
            )

    def test_a_project_that_no_longer_exists_gets_no_job(self, env):
        from sqlmodel import select

        with pytest.raises(project_jobs.ProjectGone):
            project_jobs.start_job(
                owner_id=env["owner_id"], qualified_name="owner/gone",
                kind=ProjectJobKind.export_csv, params={}, settings=env["settings"],
            )
        with get_session() as session:
            assert session.exec(select(ProjectJob)).all() == []

    def test_assert_no_active_job_raises_only_while_running(self, env):
        self._park_running_job(env["owner_id"])
        with get_session() as session:
            with pytest.raises(project_jobs.JobInProgress):
                project_jobs.assert_no_active_job(session, PROJECT)
            # A different project is unaffected.
            project_jobs.assert_no_active_job(session, "owner/other")


# ----------------------------------------------------------------------
# Restart reconciliation
# ----------------------------------------------------------------------


class TestReconcile:
    def test_a_running_row_is_failed_on_startup(self, env):
        with get_session() as session:
            session.add(ProjectJob(
                qualified_name=PROJECT, owner_id=env["owner_id"],
                kind=ProjectJobKind.export_files, status=ProjectJobStatus.running,
            ))
            session.commit()

        count = project_jobs.reconcile_orphaned_jobs(env["settings"])
        assert count == 1

        with get_session() as session:
            row = project_jobs.active_job(session, PROJECT)
            assert row is None  # nothing left running
            failed = session.exec(
                __import__("sqlmodel").select(ProjectJob)
            ).all()
        assert failed[0].status == ProjectJobStatus.failed
        assert "restart" in failed[0].error.lower()

    def test_an_interrupted_archive_may_be_re_run(self, env):
        with get_session() as session:
            session.add(ProjectJob(
                qualified_name=PROJECT, owner_id=env["owner_id"],
                kind=ProjectJobKind.archive, status=ProjectJobStatus.running,
            ))
            session.commit()
        project_jobs.reconcile_orphaned_jobs(env["settings"])
        with get_session() as session:
            row = session.exec(__import__("sqlmodel").select(ProjectJob)).one()
        assert "safe" in row.error and "resumes" in row.error

    def test_a_finished_row_is_left_alone(self, env):
        with get_session() as session:
            session.add(ProjectJob(
                qualified_name=PROJECT, owner_id=env["owner_id"],
                kind=ProjectJobKind.export_csv, status=ProjectJobStatus.successful,
            ))
            session.commit()
        assert project_jobs.reconcile_orphaned_jobs(env["settings"]) == 0


# ----------------------------------------------------------------------
# The controller tick honours the pause
# ----------------------------------------------------------------------


class TestTickPause:
    def test_a_paused_project_is_not_advanced(self, env):
        with get_session() as session:
            task = _make_task(session)
            session.add(ProjectJob(
                qualified_name=PROJECT, owner_id=env["owner_id"],
                kind=ProjectJobKind.archive, status=ProjectJobStatus.running,
            ))
            session.commit()
            task_id = task.id

            assert PROJECT in state_machine.paused_project_names(session)

            # start_new_tasks must leave the task in `created` and create no job.
            state_machine.start_new_tasks(session, env["settings"], qm_engine=None)
            session.commit()

            after = session.get(ComputationTask, task_id)
            assert after.status == TaskStatus.created
            jobs = session.exec(
                __import__("sqlmodel").select(ComputationJob)
                .where(ComputationJob.task_id == task_id)
            ).all()
            assert jobs == []

    def test_an_archive_started_mid_step_gets_no_new_job(self, env, monkeypatch):
        settings = env["settings"]
        with get_session() as session:
            created = _make_task(session)
            retry = _make_task(session, status=TaskStatus.pending)
            retry.task_path = str(env["tmp_path"] / f"task_{retry.id}")
            session.add(ComputationJob(task_id=retry.id, attempt=1, slurm_jobid=7,
                                       slurm_status="FAILED", success=False, fail_reason="['SCF']"))
            session.add(ProjectJob(qualified_name=PROJECT, owner_id=env["owner_id"],
                                   kind=ProjectJobKind.archive, status=ProjectJobStatus.running))
            session.commit()
            ids = [created.id, retry.id]
        # The step read its paused list before the archive started.
        monkeypatch.setattr(state_machine, "paused_project_names", lambda session: set())

        with get_session() as session:
            state_machine.create_retry_jobs(session, settings, qm_engine=None)
            state_machine.start_new_tasks(session, settings, qm_engine=None)
            session.commit()
            assert [len(_jobs_of(session, t)) for t in ids] == [0, 1]
            assert session.get(ComputationTask, created.id).status == TaskStatus.created

    def test_an_archive_started_mid_loop_gets_no_sbatch(self, env):
        from pathlib import Path

        from autodft.engine.scheduler import SubmitResult

        settings = env["settings"]
        with get_session() as session:
            tasks = [_make_task(session, status=TaskStatus.pending),
                     _make_task(session, status=TaskStatus.pending),
                     _make_task(session, project="owner/other", status=TaskStatus.pending)]
            jobs = []
            for task in tasks:
                path = env["tmp_path"] / f"job_{task.id}"
                path.mkdir()
                (path / "submit.cmd").write_text("#!/bin/bash\n")
                job = ComputationJob(task_id=task.id, attempt=1, job_path=str(path))
                session.add(job)
                session.commit()
                jobs.append(job.id)
            names = [f"job_{task.id}" for task in tasks]
        owner_id = env["owner_id"]

        class _Scheduler:
            calls: list[str] = []

            def get_queue_length(self):
                return 0

            def submit(self, script, nice=0):
                self.calls.append(Path(script).parent.name)
                if len(self.calls) == 1:  # the archive starts while this sbatch runs
                    with get_session() as api:
                        api.add(ProjectJob(qualified_name=PROJECT, owner_id=owner_id,
                                           kind=ProjectJobKind.archive,
                                           status=ProjectJobStatus.running))
                        api.commit()
                return SubmitResult(success=True, job_id=str(4000 + len(self.calls)))

        scheduler = _Scheduler()
        with get_session() as session:
            state_machine.submit_pending_jobs(session, scheduler, settings)
            session.commit()
            sent = [session.get(ComputationJob, j) for j in jobs]
            assert scheduler.calls == [names[0], names[2]]
            assert (sent[1].slurm_jobid, sent[1].slurm_status) == (None, None)

    def test_an_unpaused_project_is_advanced(self, env):
        with get_session() as session:
            task = _make_task(session)
            task_id = task.id
            # No ProjectJob row -> not paused.
            assert state_machine.paused_project_names(session) == set()

            state_machine.start_new_tasks(session, env["settings"], qm_engine=None)
            session.commit()

            after = session.get(ComputationTask, task_id)
            assert after.status == TaskStatus.pending


# ----------------------------------------------------------------------
# Archive waits for the cluster to go quiet
# ----------------------------------------------------------------------


class TestQuiescence:
    def test_it_aborts_while_a_job_is_still_running(self, env):
        with get_session() as session:
            task = _make_task(session)
            session.add(ComputationJob(
                task_id=task.id, attempt=1, slurm_jobid=42, slurm_status="RUNNING",
            ))
            session.commit()

        with pytest.raises(RuntimeError, match="still"):
            project_jobs._wait_for_quiescence(PROJECT, timeout=0.2, poll=0.02)

    def test_it_returns_once_the_jobs_are_judged(self, env):
        with get_session() as session:
            task = _make_task(session, status=TaskStatus.pending)
            session.add(ComputationJob(
                task_id=task.id, attempt=1, slurm_jobid=42, slurm_status="COMPLETED", success=False,
            ))
            session.add(ComputationJob(task_id=task.id, attempt=2))  # unsent: the archive fails it
            session.commit()
        project_jobs._wait_for_quiescence(PROJECT, timeout=0.2, poll=0.02)

    def test_it_waits_for_a_terminal_job_to_be_judged(self, env):
        with get_session() as session:
            task = _make_task(session, status=TaskStatus.pending)
            session.add(ComputationJob(
                task_id=task.id, attempt=1, slurm_jobid=42, slurm_status="COMPLETED",
            ))
            session.commit()
        with pytest.raises(RuntimeError, match="still"):
            project_jobs._wait_for_quiescence(PROJECT, timeout=0.2, poll=0.02)

    def test_it_waits_for_a_successful_job_to_mark_its_task(self, env):
        with get_session() as session:
            task = _make_task(session, status=TaskStatus.pending)
            session.add(ComputationJob(
                task_id=task.id, attempt=1, slurm_jobid=42, slurm_status="COMPLETED", success=True,
            ))
            session.commit()
            with pytest.raises(RuntimeError, match="still"):
                project_jobs._wait_for_quiescence(PROJECT, timeout=0.2, poll=0.02)
            state_machine.update_task_statuses(session)
            session.commit()
        project_jobs._wait_for_quiescence(PROJECT, timeout=0.2, poll=0.02)

    def test_a_job_that_finishes_during_the_wait_is_judged_and_exported(self, env, monkeypatch):
        from sqlmodel import select

        from autodft.models.result import JobResult

        _seed_result(env["tmp_path"])
        with get_session() as session:
            sp = session.exec(select(ComputationTask).where(
                ComputationTask.task_type == TaskType.singlepoint)).one()
            sp.status = TaskStatus.pending
            job = _jobs_of(session, sp.id)[0]
            job.slurm_jobid, job.slurm_status, job.success = 42, "RUNNING", None
            session.delete(session.exec(select(JobResult).where(JobResult.job_id == job.id)).one())
            (Path(job.job_path) / "output.out").write_text(ENERGY_OUTPUT + NORMAL_TERMINATION)
            session.add_all([sp, job])
            session.commit()
            job_id = job.id

        archive = _start_archive(env, monkeypatch)
        assert _status_after(archive, _finished(job_id)) == ProjectJobStatus.running
        assert _status_after(archive, _judge) == ProjectJobStatus.running
        _status_after(archive, state_machine.update_task_statuses)
        project_jobs.join_all(timeout=10)

        with get_session() as session:
            assert project_jobs.get_job(session, archive).status == ProjectJobStatus.successful
        csv = env["tmp_path"] / "export_data" / "owner" / "screening" / "screening.csv"
        assert "-230.98765" in csv.read_text()

    def test_an_opt_marked_during_the_wait_gets_no_followup(self, env, monkeypatch):
        from sqlmodel import select

        with get_session() as session:
            opt_id, job_id = _running_opt(session, env["tmp_path"])

        archive = _start_archive(env, monkeypatch)
        _status_after(archive, _finished(job_id))
        _status_after(archive, _judge)
        _status_after(archive, state_machine.update_task_statuses)
        project_jobs.join_all(timeout=10)

        with get_session() as session:
            assert project_jobs.get_job(session, archive).status == ProjectJobStatus.successful
            assert session.get(ComputationTask, opt_id).status == TaskStatus.successful
            state_machine.start_followup_tasks(session, env["settings"])
            session.commit()
            assert session.exec(select(ComputationTask).where(
                ComputationTask.depends_on_task_id == opt_id)).all() == []
            assert session.get(ComputationTask, opt_id).has_followups is False


class TestPossiblySent:
    """A job whose sbatch reported failure may be in SLURM; the archive waits for the lookup."""

    class _Scheduler:
        def __init__(self, found):
            self.found = found

        def get_queue_length(self):
            return 0

        def find_submitted(self, script):
            if isinstance(self.found, Exception):
                raise self.found
            return self.found

        def submit(self, script, nice=0):
            raise AssertionError("a paused project's job must not be sbatched")

    def _tick(self, env, found):
        def run(session):
            state_machine.submit_pending_jobs(session, self._Scheduler(found), env["settings"])
        return run

    def test_an_accepted_job_is_adopted_waited_for_and_exported(self, env, monkeypatch):
        with get_session() as session:
            opt_id, job_id = _running_opt(session, env["tmp_path"], slurm_jobid=None,
                                          slurm_status=None,
                                          fail_reason="sbatch failed (retrying): Socket timed out")

        archive = _start_archive(env, monkeypatch)
        assert _status_after(archive) == ProjectJobStatus.running
        assert _status_after(archive, self._tick(env, "777")) == ProjectJobStatus.running
        with get_session() as session:
            job = session.get(ComputationJob, job_id)
            assert (job.slurm_jobid, job.slurm_status, job.fail_reason) == (777, "PENDING", None)
        _status_after(archive, _finished(job_id))
        _status_after(archive, _judge)
        _status_after(archive, state_machine.update_task_statuses)
        project_jobs.join_all(timeout=10)

        with get_session() as session:
            assert project_jobs.get_job(session, archive).status == ProjectJobStatus.successful
            assert session.get(ComputationTask, opt_id).status == TaskStatus.successful
        manifests = (env["tmp_path"] / "export_data" / "owner" / "screening").rglob("molecule.json")
        files = [f for m in manifests for f in json.loads(m.read_text())["files"].values()]
        assert opt_id in {f["task_id"] for f in files}

    def test_a_job_slurm_lacks_is_failed_as_unsent_and_a_failed_lookup_waits(self, env, monkeypatch):
        with get_session() as session:
            opt_id, job_id = _running_opt(session, env["tmp_path"], slurm_jobid=None,
                                          slurm_status=None,
                                          fail_reason="sbatch failed (retrying): Socket timed out")

        archive = _start_archive(env, monkeypatch)
        lookup_down = self._tick(env, RuntimeError("squeue: Unable to contact slurm controller"))
        assert _status_after(archive, lookup_down) == ProjectJobStatus.running
        _status_after(archive, self._tick(env, None))
        project_jobs.join_all(timeout=10)

        with get_session() as session:
            assert project_jobs.get_job(session, archive).status == ProjectJobStatus.successful
            job = session.get(ComputationJob, job_id)
            assert (job.slurm_jobid, job.success, job.fail_reason) == (None, False, "project archived")
            assert session.get(ComputationTask, opt_id).status == TaskStatus.failed


def _running_opt(session, tmp_path, **job_fields):
    """A pending optimisation in PROJECT whose job SLURM runs, with its finished output on disk."""
    task = _make_task(session, status=TaskStatus.pending)
    path = tmp_path / "comp_data" / f"job_opt_{task.id}"
    path.mkdir(parents=True)
    (path / "submit.cmd").write_text("#!/bin/bash\n")
    (path / "input.xyz").write_text("3\n\nC 0 0 0\nC 1.5 0 0\nC 3 0 0\n")
    (path / "output.out").write_text(
        ENERGY_OUTPUT + VIBRATIONAL_FREQUENCIES_CLEAN + OPTIMIZATION_CONVERGED + NORMAL_TERMINATION)
    job = ComputationJob(task_id=task.id, attempt=1, job_path=str(path),
                         **{"slurm_jobid": 42, "slurm_status": "RUNNING", **job_fields})
    session.add(job)
    session.commit()
    return task.id, job.id


def _start_archive(env, monkeypatch) -> int:
    """Start PROJECT's archive with a fast quiescence poll; returns its id."""
    import time

    monkeypatch.setattr(project_jobs, "sleep", lambda _: time.sleep(0.01))
    return project_jobs.start_job(owner_id=env["owner_id"], qualified_name=PROJECT,
                                  kind=ProjectJobKind.archive, params={},
                                  settings=env["settings"])["id"]


def _status_after(archive_id, change=None):
    """Run the controller *change* and commit, then the archive's status a few polls later."""
    import time

    if change is not None:
        with get_session() as session:
            change(session)
            session.commit()
    time.sleep(0.2)
    with get_session() as session:
        return project_jobs.get_job(session, archive_id).status


def _finished(job_id):
    def run(session):
        session.get(ComputationJob, job_id).slurm_status = "COMPLETED"
    return run


def _judge(session):
    from autodft.qm.orca.parser import OrcaParser

    state_machine.process_finished_jobs(session, OrcaParser())


# ----------------------------------------------------------------------
# HTTP surface
# ----------------------------------------------------------------------


@pytest.fixture()
def client(env):
    with TestClient(create_app(env["settings"])) as c:
        yield c, {"X-AutoDFT-API-Key": env["key"]}, env


class TestHttp:
    def test_export_returns_202_and_a_job(self, client):
        c, headers, env = client
        r = c.post("/api/projects/owner:screening/export?format=csv", headers=headers)
        assert r.status_code == 202, r.text
        job = r.json()["job"]
        assert job["kind"] == "export_csv"

        project_jobs.join_all(timeout=10)
        detail = c.get(f"/api/jobs/{job['id']}", headers=headers)
        assert detail.status_code == 200
        assert detail.json()["status"] == "successful"

        listing = c.get("/api/projects/owner:screening/jobs", headers=headers)
        assert listing.status_code == 200
        assert listing.json()["jobs"][0]["id"] == job["id"]

    @pytest.mark.parametrize("path,body", [
        ("export?format=csv", None), ("state-analysis/export", None), ("archive", {"confirm": "ghost/p"}),
    ])
    def test_a_project_without_its_row_answers_404(self, client, path, body):
        c, _, env = client
        with get_session() as session:
            admin = accounts.get_user_by_username(session, "admin")
            key = accounts.rotate_api_key(session, admin)
            session.add(Molecule(smiles="CCO", project_name="ghost/p"))
            session.commit()
        r = c.post(f"/api/projects/ghost:p/{path}", headers={"X-AutoDFT-API-Key": key}, json=body)
        assert r.status_code == 404, r.text

    def test_download_streams_a_successful_csv(self, client):
        c, headers, env = client
        _seed_result(env["tmp_path"])
        r = c.post("/api/projects/owner:screening/export?format=csv", headers=headers)
        job_id = r.json()["job"]["id"]
        project_jobs.join_all(timeout=10)
        dl = c.get(f"/api/jobs/{job_id}/download", headers=headers)
        assert dl.status_code == 200
        expected = env["tmp_path"] / "export_data" / "owner" / "screening" / "screening.csv"
        assert dl.content == expected.read_bytes()

    def test_an_empty_export_offers_no_download(self, client):
        c, headers, env = client
        r = c.post("/api/projects/owner:screening/export?format=csv", headers=headers)
        job_id = r.json()["job"]["id"]
        project_jobs.join_all(timeout=10)
        dl = c.get(f"/api/jobs/{job_id}/download", headers=headers)
        assert dl.status_code == 400

    def test_a_running_job_blocks_a_second_export_and_a_submission(self, client):
        c, headers, env = client
        with get_session() as session:
            session.add(ProjectJob(
                qualified_name=PROJECT, owner_id=env["owner_id"],
                kind=ProjectJobKind.archive, status=ProjectJobStatus.running,
            ))
            session.commit()

        export = c.post("/api/projects/owner:screening/export?format=csv", headers=headers)
        assert export.status_code == 409, export.text

        submit = c.post(
            "/api/submit", headers=headers,
            json={"smiles": "CCO", "project": "screening"},
        )
        assert submit.status_code == 409, submit.text

    def test_a_stranger_cannot_read_the_job(self, client):
        c, headers, env = client
        with get_session() as session:
            stranger, stranger_key = accounts.create_user(session, "stranger")
            session.commit()
            job = ProjectJob(
                qualified_name=PROJECT, owner_id=env["owner_id"],
                kind=ProjectJobKind.export_csv, status=ProjectJobStatus.successful,
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            job_id = job.id

        r = c.get(f"/api/jobs/{job_id}", headers={"X-AutoDFT-API-Key": stranger_key})
        assert r.status_code == 404


# ----------------------------------------------------------------------
# Archive closes open tasks; archived leftovers make no work
# ----------------------------------------------------------------------


def _jobs_of(session, task_id):
    from sqlmodel import select
    return session.exec(select(ComputationJob).where(ComputationJob.task_id == task_id)).all()


class TestArchiveClosesOpenTasks:
    def test_archive_fails_open_tasks_and_no_jobs_follow(self, env):
        settings = env["settings"]
        with get_session() as session:
            created = _make_task(session)
            unsent = _make_task(session, status=TaskStatus.pending)
            session.add(ComputationJob(task_id=unsent.id, attempt=1))
            retry = _make_task(session, status=TaskStatus.pending)
            retry.task_path = str(env["tmp_path"] / f"task_{retry.id}")
            session.add(ComputationJob(task_id=retry.id, attempt=1, slurm_jobid=7,
                                       slurm_status="FAILED", success=False, fail_reason="['SCF']"))
            other = _make_task(session, project="owner/other")
            session.commit()
            ids = [created.id, unsent.id, retry.id]
            other_id = other.id

        job = project_jobs.start_job(owner_id=env["owner_id"], qualified_name=PROJECT,
                                     kind=ProjectJobKind.archive, params={}, settings=settings)
        project_jobs.join_all(timeout=10)

        with get_session() as session:
            assert project_jobs.get_job(session, job["id"]).status == ProjectJobStatus.successful
            for task_id in ids:
                assert session.get(ComputationTask, task_id).status == TaskStatus.failed
                assert _jobs_of(session, task_id)[-1].fail_reason == "project archived"
            assert all(j.success is False for j in _jobs_of(session, unsent.id))
            counts = {t: len(_jobs_of(session, t)) for t in ids}

            state_machine.create_retry_jobs(session, settings, qm_engine=None)
            state_machine.start_new_tasks(session, settings, qm_engine=None)
            session.commit()

            assert {t: len(_jobs_of(session, t)) for t in ids} == counts
            # A different project's open task is untouched, and still starts.
            assert session.get(ComputationTask, other_id).status == TaskStatus.pending

    def test_a_task_with_a_successful_job_is_left_to_succeed(self, env):
        with get_session() as session:
            task = _make_task(session, status=TaskStatus.pending)
            session.add(ComputationJob(task_id=task.id, attempt=1, slurm_jobid=7,
                                       slurm_status="COMPLETED", success=True))
            session.commit()
            task_id = task.id

        assert project_jobs.fail_open_tasks(PROJECT) == 0
        with get_session() as session:
            assert session.get(ComputationTask, task_id).status == TaskStatus.pending

    def test_a_success_before_a_requeue_does_not_spare_the_task(self, env):
        with get_session() as session:
            task = _make_task(session, status=TaskStatus.pending)
            task.retry_base = 1
            session.add(ComputationJob(task_id=task.id, attempt=1, slurm_jobid=7,
                                       slurm_status="COMPLETED", success=True))
            session.commit()
            task_id = task.id

        assert project_jobs.fail_open_tasks(PROJECT) == 1
        with get_session() as session:
            assert session.get(ComputationTask, task_id).status == TaskStatus.failed

    def test_archive_failures_do_not_trip_the_breaker(self, env):
        from autodft.engine import circuit_breaker

        with get_session() as session:
            for _ in range(3):
                _make_task(session)
        assert project_jobs.fail_open_tasks(PROJECT) == 3
        with get_session() as session:
            assert circuit_breaker.recent_failure_ratio(session, 100, "owner") == (0.0, 0, 0)


class TestArchivedLeftovers:
    def _archive(self, session, task):
        state = session.get(MoleculeState, task.state_id)
        mol = session.get(Molecule, state.molecule_id)
        mol.archived = True
        session.add(mol)
        session.commit()

    def test_an_archived_molecule_gets_no_job_and_no_backlog(self, env):
        from autodft.engine.pipeline import PipelineWorker

        settings = env["settings"]
        with get_session() as session:
            created = _make_task(session)
            retry = _make_task(session, status=TaskStatus.pending)
            retry.task_path = str(env["tmp_path"] / f"task_{retry.id}")
            session.add(ComputationJob(task_id=retry.id, attempt=1, slurm_jobid=7,
                                       slurm_status="FAILED", success=False, fail_reason="['SCF']"))
            unsent = _make_task(session, status=TaskStatus.pending)
            session.add(ComputationJob(task_id=unsent.id, attempt=1))
            session.commit()
            for task in (created, retry, unsent):
                self._archive(session, task)
            ids = [created.id, retry.id, unsent.id]

            assert PipelineWorker._unsubmitted_job_count(session) == 0

            state_machine.create_retry_jobs(session, settings, qm_engine=None)
            state_machine.start_new_tasks(session, settings, qm_engine=None)
            session.commit()

            assert [len(_jobs_of(session, t)) for t in ids] == [0, 1, 1]
            assert session.get(ComputationTask, created.id).status == TaskStatus.created

            # The same leftovers in a live molecule do count.
            live = _make_task(session)
            session.add(ComputationJob(task_id=live.id, attempt=1))
            session.commit()
            assert PipelineWorker._unsubmitted_job_count(session) == 2


class TestPendingNmrReferences:
    """The reference check reads every molecule's spectra, so it runs in the job, not the request."""

    @pytest.fixture()
    def pending(self, monkeypatch):
        import threading

        from autodft.analysis import spectroscopy

        threads = []

        def waiting(name):
            threads.append(threading.current_thread().name)
            return ["TMS"]

        monkeypatch.setattr(spectroscopy, "pending_nmr_references", waiting)
        return threads

    def test_the_request_returns_before_the_check(self, client, pending):
        c, headers, env = client
        r = c.post("/api/projects/owner:screening/archive", json={"confirm": PROJECT}, headers=headers)
        assert r.status_code == 202, r.text
        project_jobs.join_all(timeout=10)
        assert pending and all(t.startswith("autodft-projectjob-") for t in pending)

    def test_a_pending_reference_fails_the_archive_before_it_touches_anything(self, env, pending):
        from sqlmodel import select

        with get_session() as session:
            task_id = _make_task(session).id
        job = project_jobs.start_job(owner_id=env["owner_id"], qualified_name=PROJECT,
                                     kind=ProjectJobKind.archive, params={}, settings=env["settings"])
        project_jobs.join_all(timeout=10)

        with get_session() as session:
            row = project_jobs.get_job(session, job["id"])
            assert row.status == ProjectJobStatus.failed
            assert "waiting for their reference (TMS)" in row.error
            assert session.get(ComputationTask, task_id).status == TaskStatus.created
            assert not any(m.archived for m in session.exec(
                select(Molecule).where(Molecule.project_name == PROJECT)).all())


@pytest.mark.parametrize("kind,ext", [(ProjectJobKind.export_csv, "csv"), (ProjectJobKind.export_json, "json")])
def test_an_empty_export_reports_nothing_and_drops_the_stale_file(env, kind, ext):
    from autodft.paths import project_file_stem, safe_subdirectory

    settings = env["settings"]
    out = safe_subdirectory(settings.export_data_path, PROJECT)
    out.mkdir(parents=True, exist_ok=True)
    stale = out / f"{project_file_stem(PROJECT)}.{ext}"
    stale.write_text("old\n", encoding="utf-8")

    job = project_jobs.start_job(
        owner_id=env["owner_id"], qualified_name=PROJECT,
        kind=kind, params={}, settings=settings,
    )
    project_jobs.join_all(timeout=10)

    with get_session() as session:
        row = project_jobs.get_job(session, job["id"])
        result = json.loads(row.result_json)
        assert row.status == ProjectJobStatus.successful
    assert not stale.exists()
    assert result["message"] == "Nothing to export"
    assert result["downloadable"] is False and "path" not in result


# ----------------------------------------------------------------------
# Nothing else touches a project during Export and Delete; wipes and jobs exclude each other
# ----------------------------------------------------------------------

ARCHIVE_RUNNING = "Export and Delete is running for this project."


def _park(owner_id, kind=ProjectJobKind.archive) -> int:
    with get_session() as session:
        job = ProjectJob(qualified_name=PROJECT, owner_id=owner_id, kind=kind,
                         status=ProjectJobStatus.running)
        session.add(job)
        session.commit()
        return job.id


def _end(job_id, status):
    with get_session() as session:
        session.get(ProjectJob, job_id).status = status
        session.commit()


def _cco():
    from sqlmodel import select

    with get_session() as session:
        return session.exec(select(Molecule.id).where(Molecule.smiles == "CCO")).one()


class TestArchiveGuards:
    def _attempts(self, c, headers, mol_id):
        url = "/api/projects/owner:screening"
        return {
            "molecule wipe": lambda: c.post(f"/api/molecules/{mol_id}/wipe", headers=headers,
                                            json={"confirm": "CCO"}),
            "export": lambda: c.post(f"{url}/export?format=csv", headers=headers),
            "state-analysis export": lambda: c.post(f"{url}/state-analysis/export", headers=headers),
            "archive": lambda: c.post(f"{url}/archive", headers=headers, json={"confirm": PROJECT}),
            "project wipe": lambda: c.post(f"{url}/wipe", headers=headers, json={"confirm": PROJECT}),
        }

    @pytest.mark.parametrize("ends", [ProjectJobStatus.failed, ProjectJobStatus.successful])
    def test_each_answers_409_until_the_archive_ends(self, client, ends):
        c, headers, env = client
        mol_id = _cco()
        with get_session() as session:
            session.add(Molecule(smiles="CCC", project_name=PROJECT))
            session.commit()
        attempts = self._attempts(c, headers, mol_id)

        archive = _park(env["owner_id"])
        for name, attempt in attempts.items():
            r = attempt()
            assert (r.status_code, r.json()["detail"]) == (409, ARCHIVE_RUNNING), name
        with get_session() as session:
            assert session.get(Molecule, mol_id) is not None

        _end(archive, ends)
        expected = {"molecule wipe": 200, "export": 202, "state-analysis export": 202,
                    "archive": 202, "project wipe": 200}
        for name, attempt in attempts.items():
            assert attempt().status_code == expected[name], name
            project_jobs.join_all(timeout=10)

    def test_the_archive_message_wins_over_already_archived(self, client):
        c, headers, env = client
        with get_session() as session:
            session.get(Molecule, _cco()).archived = True
            session.commit()
        _park(env["owner_id"])
        for path, body in (("export?format=csv", None), ("archive", {"confirm": PROJECT})):
            r = c.post(f"/api/projects/owner:screening/{path}", headers=headers, json=body)
            assert (r.status_code, r.json()["detail"]) == (409, ARCHIVE_RUNNING), path


class TestWipesAndJobsExcludeEachOther:
    def test_a_running_export_refuses_both_wipes(self, client):
        c, headers, env = client
        mol_id = _cco()
        _park(env["owner_id"], ProjectJobKind.export_csv)
        for r in (c.post(f"/api/molecules/{mol_id}/wipe", headers=headers, json={"confirm": "CCO"}),
                  c.post("/api/projects/owner:screening/wipe", headers=headers,
                         json={"confirm": PROJECT})):
            assert r.status_code == 409 and "export_csv job in progress" in r.json()["detail"]
        with get_session() as session:
            assert session.get(Molecule, mol_id) is not None

    @pytest.mark.parametrize("route", ["project", "molecule", "reset"])
    def test_no_job_starts_while_a_wipe_runs(self, client, monkeypatch, route):
        from autodft.api import admin_ops

        c, headers, env = client
        mol_id = _cco()
        name = {"project": "wipe_project", "molecule": "wipe_molecule", "reset": "reset_database"}[route]
        real, refused = getattr(admin_ops, name), []

        def wipe(*args, **kwargs):
            for kind in (ProjectJobKind.export_csv, ProjectJobKind.archive):
                try:
                    project_jobs.start_job(owner_id=env["owner_id"], qualified_name=PROJECT,
                                           kind=kind, params={}, settings=env["settings"])
                except project_jobs.JobInProgress as exc:
                    refused.append(str(exc))
            return real(*args, **kwargs)

        monkeypatch.setattr(admin_ops, name, wipe)
        if route == "project":
            r = c.post("/api/projects/owner:screening/wipe", headers=headers, json={"confirm": PROJECT})
        elif route == "molecule":
            r = c.post(f"/api/molecules/{mol_id}/wipe", headers=headers, json={"confirm": "CCO"})
        else:
            with get_session() as session:
                admin = accounts.get_user_by_username(session, "admin")
                headers = {"X-AutoDFT-API-Key": accounts.rotate_api_key(session, admin)}
            r = c.post("/api/admin/reset-database", headers=headers,
                       json={"confirm": "RESET THE DATABASE", "delete_files": False})
        assert r.status_code == 200, r.text
        reason = "The database is being reset." if route == "reset" else f"Project {PROJECT!r} is being wiped."
        assert refused == [reason, reason]
        with get_session() as session:
            assert project_jobs.list_jobs(session, PROJECT) == []

    def test_a_job_checked_before_the_wipe_does_not_start_after_it(self, env):
        from sqlmodel import delete

        with get_session() as session:
            session.exec(delete(Molecule).where(Molecule.project_name == PROJECT))
            session.commit()
        with pytest.raises(project_jobs.ProjectGone, match="no molecules"):
            project_jobs.start_job(owner_id=env["owner_id"], qualified_name=PROJECT,
                                   kind=ProjectJobKind.export_csv, params={}, settings=env["settings"])
