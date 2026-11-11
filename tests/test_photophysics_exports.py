"""Raw-file exports and the photophysics workbook."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from openpyxl import load_workbook
from sqlmodel import select

from autodft import accounts
from autodft.analysis.photophysics_export import build_xlsx
from autodft.api import project_jobs, routes
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.extraction.extractor import (
    ConformerResult,
    PipelineExtractor,
    _task_key,
    _tree_files,
    _TreeJob,
)
from autodft.models import (
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeState,
    ProjectJobKind,
    TaskStatus,
    TaskType,
)
from tests.test_spectroscopy_analysis import project  # noqa: F401 - fixture


def _files(tmp_path, task_type, names, keep_gbw=False, cubes=False, part=None):
    job = tmp_path / "job"
    job.mkdir(exist_ok=True)
    for name in names:
        (job / name).write_text(name)
    task = ComputationTask(task_type=TaskType(task_type), state_id=1, header_id=1,
                           inputs_json=json.dumps({"part": part}) if part else None)
    planned = _TreeJob(1, task_type, _task_key(task), job, "S0/conf1", cubes)
    return [dest for src, dest in _tree_files(planned, keep_gbw) if (job / src).exists()]


@pytest.mark.parametrize("task_type,stem", [
    ("singlepoint", "sp"), ("singlepoint_vert_ox", "vert_ox"), ("singlepoint_vert_red", "vert_red"),
    ("singlepoint_vert_spin_change", "vert_spin_change"), ("singlepoint_uvvis", "uvvis"),
    ("singlepoint_nmr", "nmr"), ("singlepoint_soc", "singlepoint_soc"),
    ("esd_isc", "esd_isc"), ("esd_risc", "esd_risc"), ("esd_ic", "esd_ic"), ("esd_fluor", "esd_fluor"),
    ("esd_isc_t1s0", "esd_isc_t1s0"), ("esd_phosp", "esd_phosp"),
])
def test_job_types_export_input_and_output(tmp_path, task_type, stem):
    names = ("input.inp", "input.xyz", "output.out", "input.gbw", "initial.hess", "slurm-1.out")
    assert _files(tmp_path, task_type, names) == [f"{stem}.inp", f"{stem}.out"]


def test_the_optimisation_exports_its_final_geometry(tmp_path):
    names = ("input.inp", "input.xyz", "output.out", "input.hess")
    assert _files(tmp_path, "optimization", names) == ["opt.inp", "opt.out", "opt.xyz"]


def test_the_confsearch_exports_output_and_ensemble(tmp_path):
    names = ("input.inp", "output.out", "input.finalensemble.xyz")
    assert _files(tmp_path, "confsearch", names) == ["confsearch.out", "confsearch_ensemble.xyz"]


def test_an_ht_part_carries_its_number(tmp_path):
    assert _files(tmp_path, "esd_fluor", ("input.inp", "output.out"), part=3) == [
        "esd_fluor_part3.inp", "esd_fluor_part3.out",
    ]


@pytest.fixture()
def db(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    yield tmp_path
    reset_engine()


def test_singlepoint_cubes_are_copied_with_the_sp_prefix(tmp_path):
    names = ("input.inp", "input.xyz", "output.out", "ElDens.cube", "SpinDens.cube")
    assert _files(tmp_path, "singlepoint", names, cubes=True) == [
        "sp.inp", "sp.out", "sp_ElDens.cube", "sp_SpinDens.cube",
    ]


def test_cubes_are_not_copied_unless_requested(tmp_path):
    names = ("input.inp", "input.xyz", "output.out", "ElDens.cube")
    assert _files(tmp_path, "singlepoint", names) == ["sp.inp", "sp.out"]


def test_export_copies_singlepoint_cubes(db):
    with get_session() as session:
        mol = Molecule(smiles="Ccube", project_name="nho/p")
        session.add(mol)
        session.commit()
        state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                              metadata_json=json.dumps({"request_densities": True}))
        session.add(state)
        session.commit()
        opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                              state_id=state.id, header_id=1, has_followups=True)
        session.add(opt)
        session.commit()
        opt_dir = db / "jobs" / "opt"
        opt_dir.mkdir(parents=True)
        (opt_dir / "input.inp").write_text("i")
        session.add(ComputationJob(task_id=opt.id, attempt=1, job_path=str(opt_dir), success=True))
        session.commit()

        sp = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.successful,
                             state_id=state.id, header_id=1, depends_on_task_id=opt.id)
        session.add(sp)
        session.commit()
        sp_dir = db / "jobs" / "sp"
        sp_dir.mkdir(parents=True)
        (sp_dir / "input.inp").write_text("i")
        (sp_dir / "output.out").write_text("o")
        (sp_dir / "ElDens.cube").write_text("cube")
        session.add(ComputationJob(task_id=sp.id, attempt=1, job_path=str(sp_dir), success=True))
        session.commit()
        mol_id = mol.id

    PipelineExtractor("nho/p").export_tree(db / "export")
    names = sorted(p.name for p in (db / "export" / "nho" / "p" / "molecules" / str(mol_id) / "S0" / "conf1").iterdir())
    assert "sp_ElDens.cube" in names


def test_export_takes_the_conformer_the_csv_reports(db, monkeypatch):
    with get_session() as session:
        mol = Molecule(smiles="Cconf", project_name="nho/p")
        session.add(mol)
        session.commit()
        state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0)
        session.add(state)
        session.commit()
        opt_ids = []
        for name in ("first", "second"):
            opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                                  state_id=state.id, header_id=1, has_followups=True)
            session.add(opt)
            session.commit()
            job_dir = db / "jobs" / name
            job_dir.mkdir(parents=True)
            (job_dir / "input.inp").write_text(name)
            session.add(ComputationJob(task_id=opt.id, attempt=1, job_path=str(job_dir), success=True))
            session.commit()
            opt_ids.append(opt.id)
        mol_id = mol.id

    # The second conformer is the lower one.
    def lowest(self, session, mol, state, all_conformers):
        return [ConformerResult(molecule_id=mol.id, smiles=mol.smiles, state="S0",
                                conformer_index=2, opt_task_id=opt_ids[1])]
    monkeypatch.setattr(PipelineExtractor, "_extract_state_results", lowest)

    PipelineExtractor("nho/p").export_tree(db / "export")
    state_dir = db / "export" / "nho" / "p" / "molecules" / str(mol_id) / "S0"
    assert sorted(p.relative_to(state_dir).as_posix() for p in state_dir.rglob("*")) == ["conf2", "conf2/opt.inp"]


# ----------------------------------------------------------------------
# The photophysics workbook
# ----------------------------------------------------------------------


def _full_payload():
    return {
        "project": "nho/p",
        "temperature_k": 298.15,
        "molecules": [{
            "id": 1, "smiles": "c1ccccc1", "state_id": 10, "archived": False,
            "uvvis": {
                "count": 1, "pending": 0, "failed": 0, "unavailable": 0, "unweighted": 0,
                "weighting": "G", "peak": {"wavelength_nm": 300.0, "energy_ev": 4.13, "fosc": 0.5},
                "shortest_nm": 248.0,
                "conformers": [{
                    "conformer_index": 1, "opt_task_id": 5, "weight": 1.0,
                    "transitions": [
                        {"root": 1, "energy_ev": 4.13, "wavelength_nm": 300.0, "fosc": 0.5},
                        {"root": 2, "energy_ev": 5.0, "wavelength_nm": 248.0, "fosc": 0.1},
                    ],
                }],
            },
            "ir": {
                "count": 1, "pending": 0, "failed": 0, "unavailable": 0, "unweighted": 0,
                "weighting": "G", "peak": {"frequency_cm": 1650.0, "intensity_km_mol": 80.0},
                "conformers": [{
                    "conformer_index": 1, "opt_task_id": 5, "weight": 1.0,
                    "modes": [
                        {"frequency_cm": 1650.0, "intensity_km_mol": 80.0},
                        {"frequency_cm": 3050.0, "intensity_km_mol": 15.0},
                    ],
                }],
            },
            "nmr": {
                "equivalence": "topological",
                "reference": {"H": {"compound": "CHCl3", "status": "ok", "method_matches": True}},
                "nuclei": {"H": [
                    {"atoms": [0, 1], "count": 2, "shielding_ppm": 30.5, "shift_ppm": 7.26},
                    {"atoms": [2], "count": 1, "shielding_ppm": 28.0, "shift_ppm": 7.5},
                ]},
            },
            "esd": {
                "status": "done", "temperature_k": 298.15, "herzberg_teller": True,
                "delta_est_ev": 0.35, "delta_est_uks_ev": 0.30,
                "rates": {
                    "isc": {"status": "successful", "rate_s": 1.0e7, "e00_ev": 2.5,
                            "ht": True, "ht_reason": "SOCME 4.50 cm-1 < 5", "jobs": [
                        {"triplet": 1, "dele_cm": 1200.0, "socme_cm": 4.5, "rate_s": 1.0e7,
                         "fc_percent": 85.0, "ht_percent": 15.0, "k_squared": 1.2, "e00_cm": 20500.0,
                         "ht": True, "ht_reason": "SOCME 4.50 cm-1 < 5"},
                    ], "diagnostics": [
                        {"triplet": 1, "k_inlinew_100": 2.0e6, "k_inlinew_25": 1.0e5, "k_inlinew_250": 3.0e6,
                         "k_tcutfreq_300": 1.5e6, "width_sensitive": True, "soft_mode_dominated": False},
                    ]},
                    "risc": {"status": "successful", "rate_s": 2.0e5},
                    "ic": {"status": "successful", "rate_s": 5.0e9},
                    "fluorescence": {"status": "successful", "rate_s": 1.0e8},
                    "isc_t1_s0": {"status": "successful", "rate_s": 50.0},
                    "phosphorescence": {"status": "successful", "rate_s": 10.0},
                },
                "derived": {
                    "tau_s1_ns": 1.6, "phi_fluorescence": 0.6, "phi_isc": 0.35, "phi_ic": 0.05,
                    "tau_t1_us": 15.0, "phi_phosphorescence": 0.2, "phi_isc_t1_s0": 0.7, "phi_risc": 0.1,
                },
                "flags": [],
            },
            "nbo": {
                "states": [{
                    "state": "S0", "state_id": 10, "count": 1, "pending": 0, "failed": 0, "unavailable": 0,
                    "unweighted": 0, "weighting": "G",
                    "extremes": {
                        "most_negative": {"index": 1, "element": "O", "charge": -0.5},
                        "most_positive": {"index": 0, "element": "C", "charge": 0.3},
                    },
                    "atoms": [
                        {"index": 0, "element": "C", "charge": 0.3},
                        {"index": 1, "element": "O", "charge": -0.5},
                    ],
                }],
            },
        }],
    }


def _header(ws, row=1):
    return [c.value for c in ws[row]]


def _col(ws, name, row=1):
    return _header(ws, row).index(name) + 1


def test_build_xlsx_has_a_sheet_per_category_plus_sticks_and_esd_jobs():
    wb = load_workbook(BytesIO(build_xlsx(_full_payload())))
    assert wb.sheetnames == [
        "Summary", "UV-Vis", "UV-Vis sticks", "IR", "IR sticks", "NMR", "ESD", "ESD jobs",
        "ESD diagnostics", "NBO",
    ]

    assert _header(wb["UV-Vis"]) == [
        "mol_id", "smiles", "state_id", "count", "pending", "failed", "unavailable",
        "unweighted", "weighting", "peak_nm", "peak_eV", "peak_fosc", "shortest_nm",
    ]
    assert _header(wb["UV-Vis sticks"]) == [
        "mol_id", "state_id", "conformer", "weight", "root", "energy_eV", "wavelength_nm", "fosc",
    ]
    assert _header(wb["IR"]) == [
        "mol_id", "smiles", "state_id", "count", "pending", "failed", "unavailable", "unweighted",
        "weighting", "peak_cm_unscaled", "peak_intensity_km_mol",
    ]
    assert _header(wb["IR sticks"]) == [
        "mol_id", "state_id", "conformer", "weight", "frequency_cm_unscaled", "intensity_km_mol",
    ]
    assert _header(wb["NMR"]) == [
        "mol_id", "smiles", "state_id", "nucleus", "shift_ppm", "shielding_ppm", "count", "atoms_1based",
        "reference", "reference_delta_ppm", "reference_solvent", "reference_status", "method_matches", "equivalence",
        "anisotropy_ppm", "efg_cq_MHz", "efg_eta", "efg_vzz_au",
    ]
    assert _header(wb["ESD"]) == [
        "mol_id", "smiles", "state_id", "status", "temperature_K", "HT", "HT_scope", "HT_method", "dEST_eV", "dEST_UKS_eV",
        "k_ISC", "k_RISC", "k_IC", "k_F", "k_ISC_T1S0", "k_P", "tau_S1_ns", "phi_F",
        "phi_ISC", "phi_IC", "tau_T1_us", "phi_P", "phi_ISC_T1S0", "phi_RISC", "flags",
    ]
    assert _header(wb["ESD jobs"]) == [
        "mol_id", "state_id", "rate", "part", "triplet", "sublevel", "rate_s", "dele_cm", "socme_cm",
        "fc_percent", "ht_percent", "k_squared", "e00_cm", "HT", "HT_reason",
    ]
    assert _header(wb["ESD diagnostics"]) == [
        "mol_id", "state_id", "rate", "triplet", "k_FC_INLINEW100", "k_FC_INLINEW25", "k_FC_INLINEW250",
        "k_FC_TCUTFREQ300", "width_sensitive", "soft_mode_dominated",
    ]
    assert _header(wb["NBO"]) == [
        "mol_id", "smiles", "state_id", "state", "atom_1based", "element", "charge", "spin", "count", "weighting",
    ]

    sticks = wb["UV-Vis sticks"]
    assert sticks.cell(row=2, column=_col(sticks, "state_id")).value == 10
    assert sticks.cell(row=2, column=_col(sticks, "conformer")).value == 1

    esd = wb["ESD"]
    assert esd.cell(row=2, column=_col(esd, "state_id")).value == 10
    assert esd.cell(row=2, column=_col(esd, "k_ISC")).value == 1.0e7

    nmr = wb["NMR"]
    assert nmr.cell(row=2, column=_col(nmr, "state_id")).value == 10
    assert nmr.cell(row=2, column=_col(nmr, "shift_ppm")).value == 7.26

    jobs = wb["ESD jobs"]
    assert jobs.cell(row=2, column=_col(jobs, "state_id")).value == 10
    assert jobs.cell(row=2, column=_col(jobs, "rate")).value == "isc"
    assert jobs.cell(row=2, column=_col(jobs, "triplet")).value == 1
    assert jobs.cell(row=2, column=_col(jobs, "HT_reason")).value == "SOCME 4.50 cm-1 < 5"
    checks = wb["ESD diagnostics"]
    assert [c.value for c in checks[2]][1:] == [10, "isc", 1, 2.0e6, 1.0e5, 3.0e6, 1.5e6, True, False]

    nbo_sheet = wb["NBO"]
    assert nbo_sheet.cell(row=2, column=_col(nbo_sheet, "state_id")).value == 10
    assert nbo_sheet.cell(row=2, column=_col(nbo_sheet, "state")).value == "S0"
    nmr_sheet = wb["NMR"]
    assert nmr_sheet.cell(row=2, column=_col(nmr_sheet, "atoms_1based")).value == "1, 2"
    assert nbo_sheet.cell(row=2, column=_col(nbo_sheet, "atom_1based")).value == 1
    assert nbo_sheet.cell(row=2, column=_col(nbo_sheet, "charge")).value == 0.3
    assert nbo_sheet.cell(row=3, column=_col(nbo_sheet, "atom_1based")).value == 2
    assert nbo_sheet.cell(row=3, column=_col(nbo_sheet, "element")).value == "O"


def test_nbo_sheet_writes_each_states_own_state_id():
    """I3: a T1/ox row must not carry the S0 entry's id."""
    payload = {
        "project": "nho/p", "temperature_k": 298.15,
        "molecules": [{
            "id": 1, "smiles": "c1ccccc1", "state_id": 10, "archived": False,
            "nbo": {"states": [
                {"state": "S0", "state_id": 10, "count": 1, "pending": 0, "failed": 0,
                 "unavailable": 0, "unweighted": 0, "weighting": "G",
                 "atoms": [{"index": 0, "element": "C", "charge": 0.3}]},
                {"state": "ox", "state_id": 11, "count": 1, "pending": 0, "failed": 0,
                 "unavailable": 0, "unweighted": 0, "weighting": "G",
                 "atoms": [{"index": 0, "element": "C", "charge": 0.5}]},
            ]},
        }],
    }
    rows = list(load_workbook(BytesIO(build_xlsx(payload)))["NBO"].iter_rows(values_only=True))[1:]
    assert [r[2] for r in rows if r[3] == "S0"] == [10]
    assert [r[2] for r in rows if r[3] == "ox"] == [11]


def test_broken_geometries_get_their_own_sheet():
    payload = {"project": "nho/p", "molecules": [{
        "id": 1, "smiles": "CCCCF", "state_id": 10, "archived": False,
        "broken_bonds": [{"state": "T1", "state_id": 12, "conformer_index": 1, "opt_task_id": 99,
                          "detail": "dissociated: 2 fragments (C4H8F + H); C-H bonds 9 -> 8"}],
    }]}
    ws = load_workbook(BytesIO(build_xlsx(payload)))["Broken bonds"]
    assert _header(ws) == ["mol_id", "smiles", "state", "state_id", "conformer", "opt_task_id", "detail"]
    assert [c.value for c in ws[2]] == [
        1, "CCCCF", "T1", 12, 1, 99, "dissociated: 2 fragments (C4H8F + H); C-H bonds 9 -> 8",
    ]


def test_build_xlsx_on_an_empty_payload_has_only_the_summary_sheet():
    wb = load_workbook(BytesIO(build_xlsx({"project": "nho/p", "molecules": []})))
    assert wb.sheetnames == ["Summary"]


def test_job_kind_values_stay_short():
    assert max(len(k.value) for k in ProjectJobKind) <= 28


@pytest.fixture()
def settings(project):
    active = Settings()
    active.storage.data_path = str(project["tmp_path"])
    routes.set_active_settings(active)
    yield active
    routes.set_active_settings(None)


def test_execute_writes_the_workbook_and_the_full_json(project, settings):
    from autodft.analysis import spectroscopy

    full = spectroscopy.full_payload("nho/p")
    result = project_jobs._execute(ProjectJobKind.export_photophysics, "nho/p", {}, settings)
    assert result["format"] == "photophysics"
    assert result["downloadable"] is True

    xlsx_path = Path(result["path"])
    json_path = Path(result["json_path"])
    assert xlsx_path.name == "p_photophysics.xlsx"
    assert json_path.name == "p_photophysics.json"
    assert xlsx_path.is_file()
    assert json.loads(json_path.read_text()) == full


@pytest.fixture()
def client(project, settings):
    from fastapi.testclient import TestClient

    from autodft.api.app import create_app

    with get_session() as session:
        admin = accounts.get_user_by_username(session, "admin")
        key = accounts.rotate_api_key(session, admin)
        nho = accounts.get_user_by_username(session, "nho") or accounts.create_user(session, "nho")[0]
        accounts.get_or_create_project(session, nho, "p")
    with TestClient(create_app(settings)) as c:
        yield c, {"X-AutoDFT-API-Key": key}
    project_jobs.join_all(timeout=10)


def _archive_every_molecule(name):
    with get_session() as session:
        for mol in session.exec(select(Molecule).where(Molecule.project_name == name)).all():
            mol.archived = True
            session.add(mol)
        session.commit()


class TestPhotophysicsExportRoute:
    def test_it_returns_202(self, client):
        c, headers = client
        r = c.post("/api/projects/nho:p/export?format=photophysics", headers=headers)
        assert r.status_code == 202, r.text
        assert r.json()["job"]["kind"] == "export_photophysics"

    def test_it_is_allowed_on_an_archived_project_unlike_csv(self, client, project):
        c, headers = client
        _archive_every_molecule("nho/p")

        pp = c.post("/api/projects/nho:p/export?format=photophysics", headers=headers)
        assert pp.status_code == 202, pp.text

        csv = c.post("/api/projects/nho:p/export?format=csv", headers=headers)
        assert csv.status_code == 409, csv.text
