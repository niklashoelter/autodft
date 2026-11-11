"""ESD work across a molecule's S0, S1 and T1 states.

Expansion creates the ESD S1 and T1 states without tasks. Once every S0
conformer is finished, the lowest intact one (S0*) seeds their optimisations and a
SOC singlepoint at S0*; each rate task follows once its two states'
optimisation and SOC singlepoint have succeeded.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, col, select

from autodft import categories
from autodft.config import Settings
from autodft.engine.state_machine import _create_singlepoint_task, paused_project_names
from autodft.extraction import connectivity, results
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import (
    ComputationHeader, ComputationTask, Molecule, MoleculeGeometry, MoleculeState, TaskStatus, TaskType,
)
from autodft.qm.orca import advanced, blocks, esd_inputs

logger = logging.getLogger(__name__)

# S0 work that decides S0*: nothing may still be open.
_S0_STAGES = (TaskType.confsearch, TaskType.optimization, TaskType.singlepoint)
_OPEN = (TaskStatus.created, TaskStatus.pending)
_RATE_TYPES = tuple(TaskType(rate) for rate in esd_inputs.RATES)


def advance_photophysics(session: Session, settings: Settings) -> None:
    """Move every ESD molecule on as far as its finished work allows."""
    paused = paused_project_names(session)
    s1_states = session.exec(
        select(MoleculeState).where(
            MoleculeState.description == "S1",
            col(MoleculeState.metadata_json).contains('"esd_role": "S1"'),
        )
    ).all()
    for s1 in s1_states:
        metadata = _metadata(s1)
        if metadata.get("esd_done"):
            continue
        molecule = session.get(Molecule, s1.molecule_id)
        if molecule is None or molecule.archived or molecule.project_name in paused:
            continue
        s0, t1 = partner(session, s1, "S0"), partner(session, s1, "T1")
        if s0 is None or t1 is None:
            logger.error("ESD state %d has no matching S0/T1 state", s1.id)
            continue
        try:
            if "esd_seed" not in metadata:
                _seed(session, molecule, s0, s1, t1)
            else:
                _join(session, s0, s1, t1)
        except SQLAlchemyError:
            raise  # the session is unusable; the step rolls back
        except Exception:  # noqa: BLE001 - one molecule must not stop the rest
            logger.exception("ESD step failed for molecule %d", molecule.id)
    session.flush()


def partner(session: Session, state: MoleculeState, description: str) -> Optional[MoleculeState]:
    """The *description* state submitted together with *state*: same molecule, same headers.

    "T1" is ESD's triplet, found by role: a requested T1 may sit beside it as "T1".
    """
    query = select(MoleculeState).where(
        MoleculeState.molecule_id == state.molecule_id,
        MoleculeState.confsearch_header_id == state.confsearch_header_id,
        MoleculeState.optimization_header_id == state.optimization_header_id,
        MoleculeState.singlepoint_header_id == state.singlepoint_header_id,
    )
    if description == "T1":
        query = query.where(col(MoleculeState.metadata_json).contains('"esd_role": "T1"'))
    else:
        query = query.where(MoleculeState.description == description)
    return session.exec(query).first()


def _seed(
    session: Session, molecule: Molecule,
    s0: MoleculeState, s1: MoleculeState, t1: MoleculeState,
) -> None:
    """Start S1, T1 and the S0* SOC singlepoint once every S0 conformer is finished."""
    if not s0_finished(session, s0):
        return
    opt, geometry, error = s0_star(session, molecule, s0)
    if geometry is None:
        _mark(session, s1, esd_seed={"error": error})
        logger.warning("Molecule %d: no S0 conformer to seed ESD from", molecule.id)
        return
    folder = PipelineExtractor(molecule.project_name).successful_job_path(session, opt.id)
    if folder is None or not (folder / "input.hess").exists():
        # Every rate job copies it: stop here, not after days of S1 and T1 work.
        _mark(session, s1, esd_seed={
            "error": f"The lowest S0 conformer's optimisation (task {opt.id}) kept no Hessian (input.hess).",
        })
        logger.warning("Molecule %d: S0* optimisation %d kept no Hessian", molecule.id, opt.id)
        return

    for state in (s1, t1):
        seed = MoleculeGeometry(
            state_id=state.id, xyz_data=geometry.xyz_data, label=f"seed_from_task_{opt.id}",
        )
        session.add(seed)
        session.flush()
        session.add(ComputationTask(
            task_type=TaskType.optimization,
            state_id=state.id,
            header_id=state.optimization_header_id,
            input_geometry_id=seed.id,
            has_followups=True,
            status=TaskStatus.created,
        ))
    _create_singlepoint_task(
        session, s0.id, s0.singlepoint_header_id, geometry.id, opt.id, TaskType.singlepoint_soc,
    )
    _mark(session, s1, esd_seed={"task": opt.id})
    logger.info("Molecule %d: ESD seeded from S0 optimisation %d", molecule.id, opt.id)


def s0_finished(session: Session, s0: MoleculeState) -> bool:
    """Whether *s0*'s conformer search, optimisations and energy singlepoints are all finished."""
    work = session.exec(
        select(ComputationTask).where(
            ComputationTask.state_id == s0.id,
            col(ComputationTask.task_type).in_(_S0_STAGES),
        )
    ).all()
    return bool(work) and not any(
        t.status in _OPEN or (t.status == TaskStatus.successful and t.has_followups)
        for t in work
    )


def s0_star(
    session: Session, molecule: Molecule, s0: MoleculeState,
) -> tuple[Optional[ComputationTask], Optional[MoleculeGeometry], Optional[str]]:
    """S0's lowest intact conformer (S0*): its optimisation and geometry, or why there is none."""
    extractor = PipelineExtractor(molecule.project_name)
    conformers = extractor.extract_state_results(session, molecule, s0)
    broken = results.connectivity_by_task(session, [c.opt_task_id for c in conformers])
    intact = [c for c in conformers if not connectivity.breaks(broken.get(c.opt_task_id))]
    pick = extractor._pick_reported_conformer(intact)
    opt = session.get(ComputationTask, pick.opt_task_id) \
        if pick is not None and pick.e_singlepoint is not None else None
    geometry = session.get(MoleculeGeometry, opt.output_geometry_id) \
        if opt is not None and opt.output_geometry_id is not None else None
    if geometry is None:
        return None, None, _no_seed(s0, conformers, intact)
    return opt, geometry, None


def _no_seed(s0: MoleculeState, conformers: list, intact: list) -> str:
    """Why no S0 conformer can seed another state."""
    if conformers and not intact:
        return "Every S0 conformer's optimised geometry lost bonds of the molecule."
    finished = [c for c in intact if c.e_singlepoint is not None]
    if finished and all(results.excludes(c.note) for c in finished):
        notes = "; ".join(f"conformer {c.conformer_index}: {c.note}" for c in finished)
        tolerance = results.imaginary_cutoff(s0) + 0.0  # -0.0 prints as 0
        return (f"Every finished S0 conformer has an imaginary mode below the imaginary-frequency "
                f"tolerance of {tolerance:g} cm-1 ({notes}).")
    return "No S0 conformer finished both its optimisation and its energy singlepoint."


def rate_inputs(
    session: Session, s0: MoleculeState, s1: MoleculeState, t1: MoleculeState,
) -> dict[str, tuple[Optional[ComputationTask], Optional[ComputationTask]]]:
    """Per state, its optimisation and SOC singlepoint (for S0, the seed conformer's)."""
    seed = _metadata(s1).get("esd_seed", {}).get("task")
    out = {}
    for name, state in (("S0", s0), ("S1", s1), ("T1", t1)):
        if name == "S0":
            opt = session.get(ComputationTask, seed) if seed is not None else None
        else:
            opt = session.exec(
                select(ComputationTask).where(
                    ComputationTask.state_id == state.id,
                    ComputationTask.task_type == TaskType.optimization,
                ).order_by(col(ComputationTask.id))
            ).first()
        soc = session.exec(
            select(ComputationTask).where(
                ComputationTask.depends_on_task_id == opt.id,
                ComputationTask.task_type == TaskType.singlepoint_soc,
            )
        ).first() if opt is not None else None
        out[name] = (opt, soc)
    return out


def input_status(pair: tuple[Optional[ComputationTask], Optional[ComputationTask]]) -> str:
    """``successful``, ``failed`` or ``open`` for one state's optimisation and SOC singlepoint."""
    opt, soc = pair
    if opt is None:
        return "open"
    if opt.status == TaskStatus.failed:
        return "failed"
    if soc is None:
        # Follow-ups consumed without a SOC task: it will never come.
        return "failed" if opt.status == TaskStatus.successful and not opt.has_followups else "open"
    if soc.status == TaskStatus.failed:
        return "failed"
    if opt.status == TaskStatus.successful and soc.status == TaskStatus.successful:
        return "successful"
    return "open"


def _join(session: Session, s0: MoleculeState, s1: MoleculeState, t1: MoleculeState) -> None:
    """Create each wanted rate task whose two states' inputs succeeded."""
    if "task" not in _metadata(s1)["esd_seed"]:
        _mark(session, s1, esd_done=True)
        return
    wanted = wanted_rates(_metadata(s1))
    states = {"S0": s0, "S1": s1, "T1": t1}
    inputs = rate_inputs(session, s0, s1, t1)
    existing = set(session.exec(
        select(ComputationTask.task_type).where(
            col(ComputationTask.state_id).in_([s1.id, t1.id]),
            col(ComputationTask.task_type).in_(_RATE_TYPES),
        )
    ).all())
    created = set(existing)
    for rate, (initial, final, _) in esd_inputs.RATES.items():
        task_type = TaskType(rate)
        if task_type in existing or rate not in wanted:
            continue
        status = {input_status(inputs[initial]), input_status(inputs[final])}
        if status != {"successful"}:
            # Failed or still open: skip it, unchanged, and re-check next tick.
            continue
        if any(results.broken_bonds(session, inputs[name][0]) for name in (initial, final)):
            continue  # a dissociated S1/T1 geometry gets no rates
        roles = {name: {"opt": inputs[name][0].id, "soc": inputs[name][1].id} for name in (initial, final)}
        plans = _part_plans(session, rate, states[initial], roles)
        stored = [{**roles, "part": k, "parts": len(plans), "plan": plan} for k, plan in enumerate(plans, 1)] \
            if len(plans) > 1 else [roles]
        for inputs_json in stored:
            session.add(ComputationTask(
                task_type=task_type,
                state_id=states[initial].id,
                header_id=states[initial].singlepoint_header_id,
                input_geometry_id=inputs[final][0].output_geometry_id,
                depends_on_task_id=inputs[initial][0].id,
                has_followups=False,
                status=TaskStatus.created,
                inputs_json=json.dumps(inputs_json),
            ))
        created.add(task_type)
        logger.info("Molecule %d: created %s (%d part%s)", s1.molecule_id, rate, len(stored),
                    "" if len(stored) == 1 else "s")
    if {TaskType(rate) for rate in wanted} <= created:
        _mark(session, s1, esd_done=True)


def wanted_rates(metadata: dict) -> tuple[str, ...]:
    """The rate task types an ESD S1 state asks for: IC only with ``esd_ic``."""
    return tuple(rate for rate in esd_inputs.RATES if rate != "esd_ic" or metadata.get("esd_ic"))


def rate_job_input(
    session: Session, task: ComputationTask, state: MoleculeState, sp_header: str,
) -> tuple[str, dict, dict[str, Path]]:
    """A rate job's input text, its ``inputs_json`` record and its Hessians (file name -> source).

    What went into it (energies, DELE, SOCME) is kept under the record's ``"computed"``.
    """
    stored = json.loads(task.inputs_json or "{}")
    roles = {name: stored[name] for name in ("S0", "S1", "T1") if name in stored}
    molecule = session.get(Molecule, state.molecule_id)
    extractor = PipelineExtractor(molecule.project_name if molecule is not None else "")
    data = _rate_data(session, roles)
    _, _, hessians = esd_inputs.RATES[task.task_type.value]
    sources = {}
    for filename, name in hessians.items():
        folder = extractor.successful_job_path(session, roles[name]["opt"])
        source = folder / "input.hess" if folder is not None else None
        if source is None or not source.exists():
            raise esd_inputs.EsdInputError(
                f"the {name} optimisation (task {roles[name]['opt']}) left no input.hess"
            )
        sources[filename] = source
    rate = task.task_type.value
    ht_header = _ht_header(session, state)
    if "part" in stored:
        split = esd_inputs.parts(rate, data, _metadata(state), sp_header, state.charge, ht_header)
        if len(split) != stored["parts"]:
            raise esd_inputs.EsdInputError(
                f"the rate now splits into {len(split)} parts, not the {stored['parts']} it was created with"
            )
        text, computed = split[stored["part"] - 1]
        if json.loads(json.dumps(computed["jobs"])) != stored["plan"]:
            raise esd_inputs.EsdInputError(
                f"part {stored['part']} no longer runs the jobs it was created with (its SOC inputs changed)"
            )
        roles = {**roles, **{key: stored[key] for key in ("part", "parts", "plan")}}
    else:
        text, computed = esd_inputs.build(rate, data, _metadata(state), sp_header, state.charge, ht_header)
    return text, {**roles, "computed": computed}, sources


def copy_rate_hessians(sources: dict[str, Path], job_path: Path) -> None:
    """Copy a rate job's Hessians (file name -> source) into *job_path*."""
    job_path.mkdir(parents=True, exist_ok=True)
    for filename, source in sources.items():
        shutil.copyfile(source, job_path / filename)


def _rate_data(session: Session, roles: dict) -> dict[str, esd_inputs.StateData]:
    """Each role's SOC singlepoint results, as the rate builders read them."""
    data = {}
    for name, ids in roles.items():
        soc_task = session.get(ComputationTask, ids["soc"])
        record = results.for_task(session, soc_task) if soc_task is not None else None
        if record is None:
            raise esd_inputs.EsdInputError(f"the {name} SOC singlepoint (task {ids['soc']}) has no output")
        state_data = results.state_data(record)
        if state_data is None:
            # StateData.parse's one message: fixed text, no arguments.
            raise esd_inputs.EsdInputError(
                f"{name} SOC singlepoint (task {ids['soc']}): "
                "the SOC singlepoint output lacks its roots or its SOC matrix"
            )
        data[name] = state_data
    return data


def _part_plans(session: Session, rate: str, state: MoleculeState, roles: dict) -> list[list]:
    """Each part's job records (``esd_inputs.parts``); one part when the inputs cannot be read."""
    header = session.get(ComputationHeader, state.singlepoint_header_id)
    if header is None:
        return [[]]
    try:
        data = _rate_data(session, roles)
        split = esd_inputs.parts(
            rate, data, _metadata(state), header.header_text, state.charge, _ht_header(session, state),
        )
    except (OSError, KeyError, ValueError):  # EsdInputError is a ValueError
        return [[]]  # one task; its job build reports the problem
    return [json.loads(json.dumps(computed["jobs"])) for _, computed in split]


def _ht_header(session: Session, state: MoleculeState) -> Optional[str]:
    """The optimisation header as a singlepoint, for HT jobs at ``esd_ht_method`` "opt"."""
    if categories.esd_settings(_metadata(state))["esd_ht_method"] != "opt":
        return None
    header = session.get(ComputationHeader, state.optimization_header_id)
    if header is None:
        raise esd_inputs.EsdInputError(f"optimisation header {state.optimization_header_id} not found")
    metadata = _metadata(state)
    if "advanced" in metadata:
        return advanced.render_settings(
            header.header_text, metadata["advanced"]["optimization"], "optimization", as_singlepoint=True,
        )
    return blocks.singlepoint_of(header.header_text)


def _metadata(state: MoleculeState) -> dict:
    return json.loads(state.metadata_json) if state.metadata_json else {}


def _mark(session: Session, state: MoleculeState, **values) -> None:
    """Record ESD progress in *state*'s metadata."""
    state.metadata_json = json.dumps({**_metadata(state), **values})
    session.add(state)
