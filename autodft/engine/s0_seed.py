"""T1, ox and red started from S0's lowest conformer instead of a conformer search of their own.

A ``seed_from_s0`` submission creates those states without tasks. Once every S0
conformer is finished, the lowest intact one (S0*) seeds each one's optimisation.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, col, select

from autodft.engine import photophysics
from autodft.engine.state_machine import paused_project_names
from autodft.models import ComputationTask, Molecule, MoleculeGeometry, MoleculeState, TaskStatus, TaskType

logger = logging.getLogger(__name__)

SEED_FROM_S0 = "seed_from_s0"


def advance_s0_seeds(session: Session) -> None:
    """Seed every waiting state whose S0 has finished."""
    paused = paused_project_names(session)
    waiting = session.exec(
        select(MoleculeState).where(
            col(MoleculeState.metadata_json).contains(f'"{SEED_FROM_S0}": true'),
            ~col(MoleculeState.metadata_json).contains('"s0_seed"'),
        )
    ).all()
    for state in waiting:
        molecule = session.get(Molecule, state.molecule_id)
        if molecule is None or molecule.archived or molecule.project_name in paused:
            continue
        s0 = photophysics.partner(session, state, "S0")
        if s0 is None:
            logger.error("State %d waits for S0 but has no matching S0 state", state.id)
            continue
        try:
            _seed(session, molecule, s0, state)
        except SQLAlchemyError:
            raise  # the session is unusable; the step rolls back
        except Exception:  # noqa: BLE001 - one molecule must not stop the rest
            logger.exception("Seeding state %d from S0 failed", state.id)
    session.flush()


def _seed(session: Session, molecule: Molecule, s0: MoleculeState, state: MoleculeState) -> None:
    if not photophysics.s0_finished(session, s0):
        return
    opt, geometry, error = photophysics.s0_star(session, molecule, s0)
    metadata = json.loads(state.metadata_json or "{}")
    if geometry is None:
        metadata["s0_seed"] = {"error": error}
        logger.warning("Molecule %d: no S0 conformer to seed %s from: %s", molecule.id, state.description, error)
    else:
        # Labelled as ESD's seeds are, so a rerun of S0* removes the seeded work too.
        seed = MoleculeGeometry(state_id=state.id, xyz_data=geometry.xyz_data, label=f"seed_from_task_{opt.id}")
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
        metadata["s0_seed"] = {"task": opt.id}
        logger.info("Molecule %d: %s seeded from S0 optimisation %d", molecule.id, state.description, opt.id)
    state.metadata_json = json.dumps(metadata)
    session.add(state)
