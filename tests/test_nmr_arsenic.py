"""75As: no reference; the shielding tensor and the EFG from an %eprnmr block after the geometry."""

from __future__ import annotations

import pytest

from autodft import categories
from autodft.analysis.nmr import method_fingerprint
from autodft.config import Settings
from autodft.db import get_session
from autodft.engine.state_machine import _create_job_for_task
from autodft.extraction import results
from autodft.models import ComputationTask, MoleculeGeometry, TaskStatus, TaskType
from autodft.qm.orca import blocks
from autodft.qm.orca.input_generator import render_orca_input
from autodft.qm.orca.parser import OrcaParser
from autodft.qm.orca.spectra_parser import Efg, parse_efg
from tests.test_nmr_analysis import B3LYP_NMR, FIXTURES, _detail, _job, _state, _task, _tms, db  # noqa: F401
from tests.test_uvvis_tasks import _s0_with_opt

ASME3 = (FIXTURES / "nmr_efg_asme3.out").read_text()
TRAILER = "%eprnmr\n  Nuclei = all As {fgrad}\nend\n"


def test_the_efg_is_parsed():
    assert parse_efg(ASME3) == [Efg(index=1, element="As", vzz_au=-3.6530845, cq_mhz=-269.520954, eta=0.004425)]


def test_no_efg_block_is_no_efg():
    assert parse_efg((FIXTURES / "nmr_glyoxal.out").read_text()) == []


def test_the_record_keeps_the_efg():
    record = results.extract("singlepoint_nmr", ASME3, B3LYP_NMR)
    assert [e.cq_mhz for e in results.efg(record)] == [-269.520954]
    assert len(results.shieldings(record)) == 13


class TestInput:
    def test_only_as_gets_the_block(self):
        assert blocks.nmr_trailer({"nmr_nuclei": ["H", "C", "As"]}) == TRAILER
        assert blocks.nmr_trailer({"nmr_nuclei": ["H", "C", "F"]}) == ""
        assert blocks.nmr_trailer(None) == ""

    def test_the_block_follows_the_geometry(self):
        text = render_orca_input("! B3LYP NMR", 0, 1, TRAILER)
        assert text == "! B3LYP NMR\n\n*xyzfile 0 1 input.xyz\n\n" + TRAILER + "\n"

    def test_without_a_trailer_the_input_is_unchanged(self):
        assert render_orca_input("! B3LYP NMR", 0, 1) == "! B3LYP NMR\n\n*xyzfile 0 1 input.xyz\n\n"

    def test_the_block_leaves_the_method_fingerprint_alone(self):
        plain = B3LYP_NMR + "*xyzfile 0 1 input.xyz\n"
        assert method_fingerprint(plain + "\n" + TRAILER) == method_fingerprint(plain)

    @pytest.mark.parametrize("nuclei, block", [(["H", "As"], True), (["H", "C"], False)])
    def test_the_nmr_job_input(self, session, tmp_path, nuclei, block):
        state, opt = _s0_with_opt(session, {categories.NMR: True, "nmr_nuclei": nuclei})
        task = ComputationTask(
            task_type=TaskType.singlepoint_nmr, status=TaskStatus.created, state_id=state.id,
            header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
            task_path=str(tmp_path / "nmr"),
        )
        session.add(task)
        session.commit()
        job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
        text = (tmp_path / "nmr" / f"job_{job.id}" / "input.inp").read_text()
        assert text.endswith("input.xyz\n\n" + TRAILER + "\n") is block
        assert ("%eprnmr" in text) is block

    def test_the_energy_singlepoint_gets_no_block(self, session, tmp_path):
        state, opt = _s0_with_opt(session, {categories.NMR: True, "nmr_nuclei": ["As"]})
        task = ComputationTask(
            task_type=TaskType.singlepoint, status=TaskStatus.created, state_id=state.id,
            header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
            task_path=str(tmp_path / "sp"),
        )
        session.add(task)
        session.commit()
        job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
        assert "%eprnmr" not in (tmp_path / "sp" / f"job_{job.id}" / "input.inp").read_text()


def test_as_is_a_valid_nucleus():
    metadata = {categories.NMR: True, "nmr_nuclei": ["H", "As"]}
    assert categories.rejection({"multiplicity": 1}, metadata, None, "! B3LYP\n") is None
    bad = categories.rejection({"multiplicity": 1}, {categories.NMR: True, "nmr_nuclei": ["Se"]}, None, "! B3LYP\n")
    assert "H, C, F, As" in bad


def _asme3(session, tmp_path):
    metadata = {categories.NMR: True, "nmr_nuclei": ["H", "C", "As"], "nmr_reference": "tms"}
    _, state = _state(session, "C[As](C)C", "nho/p", metadata)
    geom = MoleculeGeometry(state_id=state.id, xyz_data=(FIXTURES / "asme3.xyz").read_text())
    session.add(geom)
    session.commit()
    opt = _task(session, state, TaskType.optimization, geometry=geom.id)
    _job(session, opt, tmp_path, "a_opt", "G-E(el)                           ...      0.03000000 Eh")
    sp = _task(session, state, TaskType.singlepoint, parent=opt.id)
    _job(session, sp, tmp_path, "a_sp", "FINAL SINGLE POINT ENERGY      -2355.200000000")
    shield = _task(session, state, TaskType.singlepoint_nmr, parent=opt.id)
    _job(session, shield, tmp_path, "a_nmr", ASME3, B3LYP_NMR + "*xyzfile 0 1 input.xyz\n\n" + TRAILER)


def test_arsenic_reports_shielding_and_efg_without_a_reference(db):
    with get_session() as session:
        _asme3(session, db)
        _tms(session, db)
    shifts = _detail()
    (arsenic,) = shifts["nuclei"]["As"]
    assert arsenic["atoms"] == [1] and arsenic["shift_ppm"] is None
    assert arsenic["shielding_ppm"] == pytest.approx(1623.364)
    assert arsenic["anisotropy_ppm"] == pytest.approx(249.865)
    assert (arsenic["efg_cq_mhz"], arsenic["efg_eta"], arsenic["efg_vzz_au"]) == pytest.approx(
        (-269.520954, 0.004425, -3.6530845))
    assert "As" not in shifts["reference"]
    # The %eprnmr block does not make the TMS reference look like another method.
    assert shifts["reference"]["H"]["status"] == "ok" and shifts["reference"]["H"]["method_matches"] is True
    assert all(s["shift_ppm"] is not None and "efg_cq_mhz" not in s for s in shifts["nuclei"]["H"])
