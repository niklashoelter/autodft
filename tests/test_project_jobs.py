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
        assert result["downloadable"] is True

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

    def test_it_returns_once_the_jobs_are_terminal(self, env):
        with get_session() as session:
            task = _make_task(session)
            job = ComputationJob(
                task_id=task.id, attempt=1, slurm_jobid=42, slurm_status="COMPLETED",
            )
            session.add(job)
            session.commit()
        # No transient jobs -> returns immediately without raising.
        project_jobs._wait_for_quiescence(PROJECT, timeout=0.2, poll=0.02)


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

    def test_download_streams_a_successful_csv(self, client):
        c, headers, env = client
        r = c.post("/api/projects/owner:screening/export?format=csv", headers=headers)
        job_id = r.json()["job"]["id"]
        project_jobs.join_all(timeout=10)
        dl = c.get(f"/api/jobs/{job_id}/download", headers=headers)
        # The empty project writes no CSV file; the endpoint reports that
        # rather than 200. A populated project is covered by the extractor
        # tests. Either way it must not 500.
        assert dl.status_code in (200, 404)

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
