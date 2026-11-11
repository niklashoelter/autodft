"""NMR shifts per molecule: Boltzmann-weighted, symmetry-averaged, referenced.

delta = sigma_ref - sigma + delta_ref, where sigma_ref is the reference compound's
mean shielding for that element at the same optimisation and singlepoint headers
and NMR method, and delta_ref its experimental shift vs TMS in the calculation's
solvent (see autodft.engine.nmr_references). Nuclei without a reference (As)
keep their shielding and add their electric field gradient.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Iterable, Optional

from sqlmodel import Session, col, select

from autodft import categories
from autodft.analysis.spectroscopy import Conformer, ensemble
from autodft.engine import nmr_references
from autodft.extraction import results as stored
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import ComputationHeader, ComputationTask, Molecule, MoleculeGeometry, MoleculeState, TaskStatus, TaskType
from autodft.qm.orca.advanced import effective_header, stored_method_settings
from autodft.qm.orca.spectra_parser import Efg, Shielding

_OPEN = (TaskStatus.created, TaskStatus.pending)

# Route keywords that do not change the NMR method.
_NEUTRAL_KEYWORDS = frozenset({
    "tightscf", "verytightscf", "normalscf", "loosescf", "slowconv", "veryslowconv", "keepdens",
})


def equivalence_classes(xyz: str) -> Optional[list[int]]:
    """Topological symmetry class of every atom of an XYZ geometry, in atom order.

    Bonds are perceived from the coordinates, so the classes never depend on
    how a SMILES numbered its atoms. None if RDKit cannot perceive them.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds

    from autodft.analysis.state_analysis import parse_xyz

    elements, coords = parse_xyz(xyz)
    if not elements:
        return None
    block = f"{len(elements)}\n\n" + "\n".join(
        f"{e} {x:.6f} {y:.6f} {z:.6f}" for e, (x, y, z) in zip(elements, coords)
    )
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromXYZBlock(block)
        rdDetermineBonds.DetermineConnectivity(mol)
        return list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    except Exception:  # noqa: BLE001 - RDKit raises several types on odd geometries
        return None


# Blocks retries and resource settings change without changing the method.
_NEUTRAL_BLOCKS = re.compile(r"%(?:pal|scf)\b.*?\bend\b|%maxcore\s+\S+", re.IGNORECASE | re.DOTALL)


def method_fingerprint(input_text: str) -> tuple[frozenset[str], str]:
    """What defines an NMR input's method: its ``!`` keywords and its ``%`` blocks.

    Parallel, memory and SCF-convergence settings are left out: retries change them.
    """
    words: set[str] = set()
    rest: list[str] = []
    for line in input_text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped.startswith("*"):
            break
        if stripped.startswith("!"):
            words.update(word.lower() for word in stripped[1:].split())
        elif stripped:
            rest.append(stripped)
    keywords = frozenset(w for w in words if w not in _NEUTRAL_KEYWORDS and not re.fullmatch(r"pal\d+", w))
    blocks = " ".join(_NEUTRAL_BLOCKS.sub(" ", " ".join(rest)).lower().split())
    return keywords, blocks


def reference_shieldings(
    session: Session, opt_header_id, sp_header_id, cache: Optional[dict] = None,
    method: Optional[str] = None, fingerprint: Optional[tuple[frozenset[str], str]] = None,
    choice: Optional[str] = None, settings: Optional[dict] = None,
) -> dict:
    """Per nucleus: the reference compound's mean shielding at these headers.

    Of the references at these headers, the one whose NMR input has
    *fingerprint* is taken, else the one stored with NMR *method*
    (``nmr_references.method_key``) and Advanced method *settings* (None:
    legacy). *cache*, when given, spares every molecule at the same method a
    repeat lookup. *choice* is the ``nmr_reference`` (chloroform or TMS for 1H/13C).
    """
    key = (opt_header_id, sp_header_id, method, fingerprint, choice, json.dumps(settings, sort_keys=True))
    if cache is not None and key in cache:
        return cache[key]
    from autodft.engine.entrypoint_processor import _canonicalize_smiles

    extractor = PipelineExtractor(nmr_references.REFERENCE_QUALIFIED)
    by_element = nmr_references.reference_compounds(choice)
    compounds = {
        smiles: _reference(session, extractor, _canonicalize_smiles(smiles), opt_header_id, sp_header_id,
                           method, fingerprint, settings)
        for smiles in sorted(set(by_element.values()))
    }
    out = {}
    for element, smiles in by_element.items():
        ref = compounds[smiles]
        values = [s.isotropic for s in ref.get("shieldings") or [] if s.element == element]
        out[element] = {
            "compound": smiles,
            "status": ref["status"],
            "molecule_id": ref.get("molecule_id"),
            "sigma_ppm": sum(values) / len(values) if values else None,
            "fingerprint": ref.get("fingerprint"),
        }
    if cache is not None:
        cache[key] = out
    return out


def molecule_nmr(
    session: Session, extractor: PipelineExtractor, state: MoleculeState,
    pool: list[Conformer], detail: bool, references: Optional[dict] = None,
    excluded: Iterable[Conformer] = (),
) -> dict:
    """NMR signals of one S0 state, Boltzmann-weighted over its conformers.

    The summary carries counts, references and the number of signals per
    nucleus; *detail* adds the signals and the conformer weights.
    """
    metadata = json.loads(state.metadata_json) if state.metadata_json else {}
    nuclei = categories.options(metadata).get("nmr_nuclei", list(categories.NMR_NUCLEI))

    items = []
    results: dict[int, tuple[list[Shielding], Optional[tuple[frozenset[str], str]], list[Efg]]] = {}
    for conformer in pool:
        status = _status(session, extractor, conformer, results)
        items.append((conformer, status, {}))
    # ensemble decides which conformers carry weight (data and energy in).
    out = ensemble(items, True, excluded=excluded)
    weight_of = {c["opt_task_id"]: c["weight"] for c in out["conformers"]}
    if not detail:
        del out["conformers"]
    out.update({"equivalence": "none", "reference": {}, "signals": {}})
    if detail:
        out["nuclei"] = {}
    have = [c for c, status, _ in items if status == "ok" and c.opt.id in weight_of]
    if not have:
        return out

    weights = [weight_of[c.opt.id] for c in have]
    first = results[have[0].opt.id]
    elements = [s.element for s in first[0]]
    sigma = [0.0] * len(elements)
    anisotropy = [0.0] * len(elements)
    for conformer, weight in zip(have, weights):
        for s in results[conformer.opt.id][0]:
            sigma[s.index] += weight * s.isotropic
            anisotropy[s.index] += weight * s.anisotropy
    efg = _weighted_efg([results[c.opt.id][2] for c in have], weights)

    classes = equivalence_classes(_geometry(session, have[0].opt.id))
    if classes is None or len(classes) != len(elements):
        classes = list(range(len(elements)))
    else:
        out["equivalence"] = "topological"

    fingerprint = first[1]
    reference = reference_shieldings(
        session, state.optimization_header_id, state.singlepoint_header_id, references,
        nmr_references.method_key(metadata), fingerprint, categories.options(metadata).get("nmr_reference"),
        stored_method_settings(metadata),
    )
    sp_header = session.get(ComputationHeader, state.singlepoint_header_id) if state.singlepoint_header_id else None
    sp_text = effective_header(sp_header.header_text, metadata, "singlepoint") if sp_header else None
    for element in nuclei:
        groups: dict[int, list[int]] = defaultdict(list)
        for index, symbol in enumerate(elements):
            if symbol == element:
                groups[classes[index]].append(index)
        if not groups:
            continue
        ref = reference.get(element)
        sigma_ref = ref["sigma_ppm"] if ref is not None and ref["status"] == "ok" else None
        delta_ref, solvent = nmr_references.reference_shift(
            ref["compound"], element, sp_text,
        ) if ref is not None else (0.0, None)
        signals = []
        for atoms in groups.values():
            mean = sum(sigma[i] for i in atoms) / len(atoms)
            signal = {
                "atoms": atoms, "count": len(atoms), "shielding_ppm": mean,
                "shift_ppm": sigma_ref - mean + delta_ref if sigma_ref is not None else None,
                "anisotropy_ppm": sum(anisotropy[i] for i in atoms) / len(atoms),
            }
            if all(i in efg for i in atoms):
                signal.update({key: sum(efg[i][key] for i in atoms) / len(atoms)
                               for key in ("efg_cq_mhz", "efg_eta", "efg_vzz_au")})
            signals.append(signal)
        signals.sort(key=lambda s: s["shielding_ppm"])
        out["signals"][element] = len(signals)
        if detail:
            out["nuclei"][element] = signals
        if ref is None:
            continue
        out["reference"][element] = {
            "compound": ref["compound"],
            "status": ref["status"],
            "molecule_id": ref["molecule_id"],
            "sigma_ppm": ref["sigma_ppm"],
            "delta_ref_ppm": delta_ref,
            "delta_ref_solvent": solvent,
            "method_matches": (
                ref["fingerprint"] == fingerprint
                if ref["fingerprint"] is not None and fingerprint is not None else None
            ),
        }
    return out


def _weighted_efg(per_conformer: list[list[Efg]], weights: list[float]) -> dict[int, dict]:
    """Per atom, the weighted EFG; only atoms every conformer has one for."""
    atoms = set.intersection(*({e.index for e in found} for found in per_conformer)) if per_conformer else set()
    out = {i: {"efg_cq_mhz": 0.0, "efg_eta": 0.0, "efg_vzz_au": 0.0} for i in atoms}
    for found, weight in zip(per_conformer, weights):
        for e in found:
            if e.index in out:
                out[e.index]["efg_cq_mhz"] += weight * e.cq_mhz
                out[e.index]["efg_eta"] += weight * e.eta
                out[e.index]["efg_vzz_au"] += weight * e.vzz_au
    return out


def _reference(session, extractor, canonical, opt_header_id, sp_header_id, method, fingerprint, settings) -> dict:
    molecule = session.exec(
        select(Molecule).where(
            Molecule.smiles == canonical,
            Molecule.project_name == nmr_references.REFERENCE_QUALIFIED,
        )
    ).first()
    if molecule is None:
        return {"status": "missing"}
    states = session.exec(
        select(MoleculeState).where(
            MoleculeState.molecule_id == molecule.id,
            MoleculeState.description == "S0",
            MoleculeState.optimization_header_id == opt_header_id,
            MoleculeState.singlepoint_header_id == sp_header_id,
        ).order_by(col(MoleculeState.id))
    ).all()
    found = [(state, _reference_state(session, extractor, molecule, state)) for state in states]
    # The jobs' own inputs decide first; the stored method covers references not run yet.
    for _, ref in found:
        if fingerprint is not None and ref["status"] == "ok" and ref["fingerprint"] == fingerprint:
            return ref
    for state, ref in found:
        metadata = json.loads(state.metadata_json or "{}")
        if nmr_references.method_key(metadata) == method and stored_method_settings(metadata) == settings:
            return ref
    return {"status": "missing", "molecule_id": molecule.id}


def _reference_state(session, extractor, molecule, state) -> dict:
    task = session.exec(
        select(ComputationTask)
        .where(ComputationTask.state_id == state.id,
               ComputationTask.task_type == TaskType.singlepoint_nmr)
        .order_by(col(ComputationTask.id).desc())
    ).first()
    if task is None:
        # No NMR job yet: still coming only while the optimisation is.
        opt = session.exec(
            select(ComputationTask)
            .where(ComputationTask.state_id == state.id,
                   ComputationTask.task_type == TaskType.optimization)
            .order_by(col(ComputationTask.id).desc())
        ).first()
        coming = opt is None or opt.status in _OPEN or (
            opt.status == TaskStatus.successful and opt.has_followups
        )
        return {"status": "pending" if coming else "failed", "molecule_id": molecule.id}
    if task.status in _OPEN:
        return {"status": "pending", "molecule_id": molecule.id}
    if task.status == TaskStatus.failed:
        return {"status": "failed", "molecule_id": molecule.id}
    shieldings, fingerprint, _ = _nmr_result(session, extractor, task.id)
    return {"status": "ok" if shieldings else "failed", "molecule_id": molecule.id,
            "shieldings": shieldings, "fingerprint": fingerprint}


def _status(session, extractor, conformer: Conformer, results: dict) -> str:
    """Where this conformer's NMR job stands; a readable result lands in *results*."""
    if conformer.opt.status in _OPEN:
        return "pending"
    task = session.exec(
        select(ComputationTask).where(
            ComputationTask.depends_on_task_id == conformer.opt.id,
            ComputationTask.task_type == TaskType.singlepoint_nmr,
        )
    ).first()
    if task is None:
        return "pending" if conformer.opt.has_followups else "failed"
    if task.status in _OPEN:
        return "pending"
    if task.status == TaskStatus.failed:
        return "failed"
    from autodft.analysis.state_analysis import parse_xyz

    shieldings, fingerprint, efg = _nmr_result(session, extractor, task.id)
    if not shieldings:
        return "unavailable"
    # Shieldings for other atoms than the optimised geometry's are unusable.
    elements, _ = parse_xyz(_geometry(session, conformer.opt.id))
    if [s.element for s in shieldings] != elements:
        return "unavailable"
    results[conformer.opt.id] = (shieldings, fingerprint, efg)
    return "ok"


def _nmr_result(
    session, extractor, task_id,
) -> tuple[list[Shielding], Optional[tuple[frozenset[str], str]], list[Efg]]:
    task = session.get(ComputationTask, task_id)
    record = stored.for_task(session, task) if task is not None else None
    if record is None:
        return [], None, []
    return stored.shieldings(record), stored.fingerprint(record), stored.efg(record)


def _geometry(session: Session, opt_task_id: int) -> str:
    task = session.get(ComputationTask, opt_task_id)
    geometry = session.get(MoleculeGeometry, task.output_geometry_id) if task and task.output_geometry_id else None
    return geometry.xyz_data if geometry is not None else ""
