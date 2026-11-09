"""Archiving freezes the photophysics payload; archived molecules are served from it."""

from __future__ import annotations

import json
import shutil

import pytest
from sqlmodel import select

from autodft import accounts, categories
from autodft.analysis import spectroscopy
from autodft.api import admin_ops, project_jobs, routes
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.engine import nmr_references
from autodft.models import ComputationTask, Molecule, MoleculeState, ProjectJobKind, TaskStatus, TaskType
from tests.test_nbo_analysis import _conformer as _nbo_conformer, _state as _nbo_state  # noqa: F401 - helpers
from tests.test_nmr_analysis import _glyoxal, _state, _tms  # noqa: F401 - helpers
from tests.test_spectroscopy_analysis import project  # noqa: F401 - fixture


@pytest.fixture()
def settings(project):
    active = Settings()
    active.storage.data_path = str(project["tmp_path"])
    routes.set_active_settings(active)
    yield active
    routes.set_active_settings(None)


def _archive(settings, monkeypatch):
    monkeypatch.setattr(project_jobs, "_wait_for_quiescence", lambda name: None)
    return project_jobs._execute(ProjectJobKind.archive, "nho/p", {}, settings)


def _without_archived(payload):
    return [{k: v for k, v in m.items() if k != "archived"} for m in payload["molecules"]]


def test_archived_molecules_are_served_from_the_frozen_payload(project, settings, monkeypatch):
    project_jobs.fail_open_tasks("nho/p")  # the archive fails the pending UV/Vis task first
    summary = spectroscopy.analyze_spectra("nho/p", use_cache=False)
    detail = spectroscopy.analyze_spectra("nho/p", molecule_id=project["molecule"])
    result = _archive(settings, monkeypatch)
    assert result["photophysics_frozen"] == 1
    shutil.rmtree(project["tmp_path"] / "jobs")  # stands for the deleted comp_data
    after = spectroscopy.analyze_spectra("nho/p", use_cache=False)
    assert _without_archived(after) == _without_archived(summary)
    assert [m["archived"] for m in after["molecules"]] == [True]
    frozen = spectroscopy.analyze_spectra("nho/p", molecule_id=project["molecule"])
    assert _without_archived(frozen) == _without_archived(detail)
    assert frozen["molecules"][0]["uvvis"]["conformers"]


def test_the_cached_summary_switches_to_the_frozen_payload(project, settings, monkeypatch):
    project_jobs.fail_open_tasks("nho/p")
    live = spectroscopy.analyze_spectra("nho/p")
    _archive(settings, monkeypatch)
    shutil.rmtree(project["tmp_path"] / "jobs")
    assert _without_archived(spectroscopy.analyze_spectra("nho/p")) == _without_archived(live)


def test_a_second_archive_keeps_the_first_freeze(project, settings, monkeypatch):
    _archive(settings, monkeypatch)
    path = spectroscopy.frozen_dir("nho/p", settings) / f"mol_{project['molecule']}.json"
    before = path.read_text()
    shutil.rmtree(project["tmp_path"] / "jobs")
    assert spectroscopy.freeze("nho/p", settings) == 0
    assert path.read_text() == before


def test_an_archived_molecule_without_a_frozen_file_is_analysed_live(project, settings):
    with get_session() as session:
        mol = session.get(Molecule, project["molecule"])
        mol.archived = True
        session.add(mol)
        session.commit()
    assert spectroscopy.analyze_spectra("nho/p", use_cache=False)["molecules"][0]["uvvis"]["count"] == 1


def test_a_project_without_categories_freezes_nothing(project, settings, monkeypatch):
    with get_session() as session:
        mol = session.get(Molecule, project["molecule"])
        mol.project_name = "nho/other"
        session.add(mol)
        session.commit()
    result = _archive(settings, monkeypatch)
    assert "photophysics_frozen" not in result
    assert not spectroscopy.frozen_dir("nho/p", settings).exists()


def test_nbo_entry_survives_archiving(project, settings, monkeypatch):
    with get_session() as session:
        mol, state = _nbo_state(session, "nho/p", metadata={categories.NBO: True})
        _nbo_conformer(session, project["tmp_path"], state, 11, -100.0, [("C", -0.30, None), ("O", 0.10, None)])
        mol_id = mol.id

    before = spectroscopy.analyze_spectra("nho/p", molecule_id=mol_id, use_cache=False)
    _archive(settings, monkeypatch)
    shutil.rmtree(project["tmp_path"] / "jobs")
    after = spectroscopy.analyze_spectra("nho/p", molecule_id=mol_id)
    assert after["molecules"][0]["nbo"] == before["molecules"][0]["nbo"]
    assert after["molecules"][0]["nbo"]["states"][0]["extremes"]["most_negative"]["element"] == "C"


def test_the_full_payload_carries_every_detail(project, settings):
    payload = spectroscopy.full_payload("nho/p")
    assert [m["id"] for m in payload["molecules"]] == [project["molecule"]]
    assert payload["molecules"][0]["uvvis"]["conformers"]


def _add_flagged_molecule(smiles="CC=O"):
    with get_session() as session:
        mol = Molecule(smiles=smiles, project_name="nho/p")
        session.add(mol)
        session.commit()
        session.add(MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                                  metadata_json=json.dumps({categories.UVVIS: True})))
        session.commit()
        return mol.id


def test_freeze_leaves_no_tmp_file_behind(project, settings):
    spectroscopy.freeze("nho/p", settings)
    directory = spectroscopy.frozen_dir("nho/p", settings)
    assert sorted(p.name for p in directory.iterdir()) == [f"mol_{project['molecule']}.json"]


def test_a_truncated_frozen_file_does_not_break_the_project(project, settings, monkeypatch):
    other_id = _add_flagged_molecule()
    _archive(settings, monkeypatch)
    path = spectroscopy.frozen_dir("nho/p", settings) / f"mol_{project['molecule']}.json"
    path.write_text(path.read_text()[:100])

    summary = spectroscopy.analyze_spectra("nho/p", use_cache=False)
    assert other_id in [m["id"] for m in summary["molecules"]]

    payload = spectroscopy.full_payload("nho/p")
    assert other_id in [m["id"] for m in payload["molecules"]]


def test_a_frozen_file_of_another_molecule_is_ignored(project, settings, monkeypatch):
    """wipe_molecule leaves mol_<id>.json; SQLite reuses the id; the stale
    payload must not be served to the unrelated molecule that inherits it."""
    with get_session() as session:
        plain = session.exec(select(Molecule).where(Molecule.smiles == "CCO")).one()
        admin_ops.wipe_molecule(session, plain.id, settings.comp_data_path)
    _archive(settings, monkeypatch)
    flagged = project["molecule"]
    with get_session() as session:
        assert flagged == max(m.id for m in session.exec(select(Molecule)).all())
        admin_ops.wipe_molecule(session, flagged, settings.comp_data_path)
    with get_session() as session:
        newcomer = Molecule(smiles="CC", project_name="nho/p")
        session.add(newcomer)
        session.commit()
        session.add(MoleculeState(molecule_id=newcomer.id, description="S0", multiplicity=1,
                                  charge=0, metadata_json=json.dumps({})))
        session.commit()
        new_id = newcomer.id
    _archive(settings, monkeypatch)
    payload = spectroscopy.analyze_spectra("nho/p", use_cache=False)
    assert new_id == flagged
    assert payload["molecules"] == []


# ----------------------------------------------------------------------
# The archive route waits for pending NMR references (I1)
# ----------------------------------------------------------------------


@pytest.fixture()
def nmr_api(tmp_path):
    from fastapi.testclient import TestClient

    from autodft.api.app import create_app

    active = Settings()
    active.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(active)
    routes.set_active_settings(active)
    with get_session() as session:
        admin = accounts.get_user_by_username(session, "admin")
        key = accounts.rotate_api_key(session, admin)
        nho = accounts.get_user_by_username(session, "nho") or accounts.create_user(session, "nho")[0]
        accounts.get_or_create_project(session, nho, "p")
    with TestClient(create_app(active)) as client:
        yield client, {"X-AutoDFT-API-Key": key}, tmp_path
    project_jobs.join_all(timeout=10)
    routes.set_active_settings(None)
    reset_engine()


class TestArchiveWaitsForNmrReferences:
    def test_the_job_fails_while_the_reference_is_pending(self, nmr_api):
        client, headers, tmp_path = nmr_api
        with get_session() as session:
            _glyoxal(session, tmp_path)
            _, ref_state = _state(session, "C[Si](C)(C)C", nmr_references.REFERENCE_QUALIFIED,
                                  {categories.NMR: True})
            session.add(ComputationTask(task_type=TaskType.optimization, status=TaskStatus.pending,
                                        state_id=ref_state.id, header_id=3, has_followups=True))
            session.commit()

        r = client.post("/api/projects/nho:p/archive", headers=headers,
                        json={"confirm": "nho/p"})
        assert r.status_code == 202, r.text
        project_jobs.join_all(timeout=10)
        job = client.get(f"/api/jobs/{r.json()['job']['id']}", headers=headers).json()
        assert job["status"] == "failed"
        assert "C[Si](C)(C)C" in job["error"]
        with get_session() as session:
            assert not any(m.archived for m in session.exec(
                select(Molecule).where(Molecule.project_name == "nho/p")).all())

    def test_202_once_the_reference_is_successful(self, nmr_api):
        client, headers, tmp_path = nmr_api
        with get_session() as session:
            _glyoxal(session, tmp_path)
            _tms(session, tmp_path)

        r = client.post("/api/projects/nho:p/archive", headers=headers,
                        json={"confirm": "nho/p"})
        assert r.status_code == 202, r.text

    def test_202_for_a_project_without_nmr_molecules(self, nmr_api):
        client, headers, tmp_path = nmr_api
        with get_session() as session:
            mol = Molecule(smiles="CCO", project_name="nho/p")
            session.add(mol)
            session.commit()
            session.add(MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1,
                                      charge=0, metadata_json=json.dumps({})))
            session.commit()

        r = client.post("/api/projects/nho:p/archive", headers=headers,
                        json={"confirm": "nho/p"})
        assert r.status_code == 202, r.text
