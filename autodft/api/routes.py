"""API endpoints and HTML dashboard routes."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from sqlmodel import col, func, select

from autodft.api.auth import COOKIE_NAME, issue_token
from autodft.api.identity import (
    Identity,
    current_identity,
    require_admin,
    require_molecule,
    resolve_project,
    scope_jobs,
    scope_molecules,
    scope_tasks,
    visible_entrypoints,
    visible_projects,
)

from autodft.db import get_session
from autodft.models import (
    CalculationEntrypoint,
    ComputationHeader,
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeState,
    ProjectJobKind,
    TaskStatus,
    TaskType,
)

router = APIRouter()

# Public router — accessible without authentication. Hosts the login
# form and the logout helper. Every other route lives on ``router`` and
# is gated by the auth middleware in ``autodft.api.app``.
public_router = APIRouter()

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


# ---------------------------------------------------------------------------
# Login / logout
# ---------------------------------------------------------------------------


@public_router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: Optional[str] = Query("/"), error: Optional[str] = None):
    """Render the login form."""
    return templates.TemplateResponse(
        request=request, name="login.html",
        context={"next": next or "/", "error": error},
    )


@public_router.post("/login")
def login_submit(request: Request,
                 password: str = Form(...),
                 username: str = Form(""),
                 next: str = Form("/")):
    """Sign in with a username and that account's API key.

    The key is the only credential; the browser exchanges it here for a
    session cookie so it is not re-sent on every request. Admin included
    — an operator who loses the admin key rotates it with
    ``autodft admin rotate-key``, which needs the machine, not a secret.
    """
    from autodft import accounts

    settings = get_active_settings()
    username = (username or "").strip().lower()

    resolved: Optional[str] = None
    if username:
        with get_session() as session:
            user = accounts.resolve_api_key(session, password)
            if user is not None and user.username == username:
                resolved = user.username

    if resolved is None:
        # One message for every failure: distinguishing "no such user"
        # from "wrong key" would turn the form into a username oracle.
        # Status stays 200 so the browser keeps the URL stable.
        return templates.TemplateResponse(
            request=request, name="login.html",
            context={"next": next, "error": "Incorrect username or key."},
            status_code=200,
        )

    token = issue_token(
        settings.session_secret(),
        settings.security.session_lifetime_seconds,
        resolved,
    )
    # Sanitise the redirect target — only allow same-origin paths so a
    # malicious ?next= can't bounce users off-site.
    target = next if next.startswith("/") and not next.startswith("//") else "/"
    resp = RedirectResponse(url=target, status_code=303)
    resp.set_cookie(
        COOKIE_NAME, token,
        max_age=settings.security.session_lifetime_seconds,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return resp


@public_router.get("/logout")
def logout(request: Request):
    """Clear the session cookie and redirect to the login page."""
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


# ---------------------------------------------------------------------------
# Active settings registry — set by autodft.api.app.create_app() so route
# handlers don't have to re-load a TOML to find the data_path / export dir.
# ---------------------------------------------------------------------------

_active_settings = None


def set_active_settings(settings) -> None:
    global _active_settings
    _active_settings = settings


def get_active_settings():
    """Return the settings registered by create_app(), or load defaults."""
    global _active_settings
    if _active_settings is None:
        from autodft.config import load_settings
        _active_settings = load_settings()
    return _active_settings


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class SubmitRequest(BaseModel):
    # Bounded: RDKit's SMILES parser overflows the C stack on very long
    # input and takes the whole process down with it -- and the dashboard
    # runs in a thread of the pipeline worker, so that kills the controller.
    smiles: str = Field(max_length=512)
    project: str = "default"
    # Accepted and IGNORED: the author recorded in request_metadata as
    # `project_author` is always the calling account's username. The field
    # survives only so pre-accounts scripts that still send it are not
    # rejected with a 422. See `_submission_owner`.
    author: str = "web"
    priority: int = 10
    request_t1: bool = False
    request_ox: bool = False
    request_red: bool = False
    skip_confsearch: bool = False
    request_optimization: bool = True
    request_singlepoint: bool = True
    # Singlepoint vertical excitations (ox / red / spin-flip). On by
    # default — matches the legacy `submit_to_db_*.py` scripts.
    request_singlepoint_vertical_excitations: bool = True
    # Per-state conformer limit. Default 1 per requested state (the
    # cheapest sensible run). max_conformers_S0 doubles as the global
    # fallback for any state whose specific limit isn't passed.
    max_conformers_S0: int = 1
    max_conformers_T1: int = 1
    max_conformers_ox: int = 1
    max_conformers_red: int = 1
    # Legacy single field, kept for backwards-compat. When set, it is
    # used as the per-state default if the more specific fields are
    # left at their default of 1.
    max_conformers: Optional[int] = None
    # For each slot you can pass either the raw header text OR the integer
    # ID of a stored ComputationHeader. ID takes precedence.
    header_confsearch: Optional[str] = None
    header_optimization: Optional[str] = None
    header_singlepoint: Optional[str] = None
    header_confsearch_id: Optional[int] = None
    header_optimization_id: Optional[int] = None
    header_singlepoint_id: Optional[int] = None

    @field_validator("project")
    @classmethod
    def _check_project(cls, value: str) -> str:
        # The project name becomes a directory name under the data root and
        # a path segment in several routes, so it is constrained here rather
        # than at each use site.
        from autodft.paths import validate_project_name

        return validate_project_name(value)


class HeaderCreate(BaseModel):
    header_text: str
    description: Optional[str] = None
    kind: Optional[str] = None  # "confsearch" | "optimization" | "singlepoint" | None
    validated: bool = False


class HeaderUpdate(BaseModel):
    header_text: Optional[str] = None
    description: Optional[str] = None
    kind: Optional[str] = None
    validated: Optional[bool] = None


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    """Render the single-page HTML dashboard."""
    return templates.TemplateResponse(request=request, name="dashboard.html")


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


def _reject_bad_project(name: str):
    """Return a 400 JSONResponse if *name* can't be used, else None.

    Applied to every route with a `{name}` path parameter: Starlette matches
    `[^/]+`, which includes `..`, and several of these routes build a
    filesystem path from it.
    """
    from autodft.paths import InvalidProjectName, validate_project_name

    try:
        validate_project_name(name)
    except InvalidProjectName as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    return None


@router.get("/api/overview")
def api_overview(identity: Identity = Depends(current_identity)):
    """Pipeline overview statistics, counting only the caller's work.

    Global counts here told every user exactly how much everyone else was
    running -- and, since the numbers are what the dashboard's front page
    is built from, made the whole page describe someone else's campaign.
    """
    with get_session() as session:
        molecule_count = session.exec(
            scope_molecules(
                select(func.count()).select_from(Molecule), session, identity,
            )
        ).one()

        task_counts = {}
        for status in TaskStatus:
            count = session.exec(
                scope_tasks(
                    select(func.count())
                    .select_from(ComputationTask)
                    .where(ComputationTask.status == status),
                    session, identity,
                )
            ).one()
            task_counts[status.value] = count

        job_counts: dict[str, int] = {}
        rows = session.exec(
            scope_jobs(
                select(ComputationJob.slurm_status, func.count())
                .select_from(ComputationJob),
                session, identity,
            ).group_by(ComputationJob.slurm_status)
        ).all()
        for slurm_status, count in rows:
            job_counts[slurm_status or "unknown"] = count

        queued = session.exec(
            select(CalculationEntrypoint)
            .where(col(CalculationEntrypoint.time_started).is_(None))
        ).all()
        queue_length = len(visible_entrypoints(queued, session, identity))

    return {
        "molecules": molecule_count,
        "tasks": task_counts,
        "jobs": job_counts,
        "queue_length": queue_length,
    }


@router.get("/api/molecules")
def api_molecules(
    project: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    identity: Identity = Depends(current_identity),
):
    """List molecules with optional project filter."""
    with get_session() as session:
        stmt = select(Molecule).order_by(col(Molecule.created_at).desc())
        if project:
            stmt = stmt.where(
                Molecule.project_name == resolve_project(session, identity, project)
            )
        stmt = scope_molecules(stmt, session, identity)
        stmt = stmt.offset(offset).limit(limit)
        molecules = session.exec(stmt).all()

        results = []
        for mol in molecules:
            state_count = session.exec(
                select(func.count())
                .select_from(MoleculeState)
                .where(MoleculeState.molecule_id == mol.id)
            ).one()
            results.append(
                {
                    "id": mol.id,
                    "smiles": mol.smiles,
                    "project_name": mol.project_name,
                    "created_at": mol.created_at.isoformat() if mol.created_at else None,
                    "state_count": state_count,
                }
            )
    return results


@router.get("/api/molecules/{molecule_id}")
def api_molecule_detail(
    molecule_id: int, identity: Identity = Depends(current_identity),
):
    """Single molecule detail with all states, tasks, and jobs."""
    with get_session() as session:
        mol = require_molecule(session, identity, molecule_id)
        if mol is None:
            return JSONResponse(status_code=404, content={"detail": "Molecule not found"})

        states_data = []
        for state in mol.states:
            tasks_data = []
            for task in state.tasks:
                jobs_data = [
                    {
                        "id": job.id,
                        "attempt": job.attempt,
                        "slurm_jobid": job.slurm_jobid,
                        "slurm_status": job.slurm_status,
                        "success": job.success,
                        "fail_reason": job.fail_reason,
                    }
                    for job in task.jobs
                ]
                tasks_data.append(
                    {
                        "id": task.id,
                        "task_type": task.task_type.value,
                        "status": task.status.value,
                        "created_at": task.created_at.isoformat() if task.created_at else None,
                        "jobs": jobs_data,
                    }
                )
            states_data.append(
                {
                    "id": state.id,
                    "description": state.description,
                    "multiplicity": state.multiplicity,
                    "charge": state.charge,
                    "tasks": tasks_data,
                }
            )

        return {
            "id": mol.id,
            "smiles": mol.smiles,
            "project_name": mol.project_name,
            "created_at": mol.created_at.isoformat() if mol.created_at else None,
            "states": states_data,
        }


@router.get("/api/tasks")
def api_tasks(
    status: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    identity: Identity = Depends(current_identity),
):
    """List tasks with optional status and type filters."""
    with get_session() as session:
        stmt = select(ComputationTask).order_by(
            col(ComputationTask.updated_at).desc()
        )
        stmt = scope_tasks(stmt, session, identity)
        if status:
            stmt = stmt.where(ComputationTask.status == TaskStatus(status))
        if type:
            stmt = stmt.where(ComputationTask.task_type == TaskType(type))
        stmt = stmt.limit(limit)
        tasks = session.exec(stmt).all()

        results = []
        for task in tasks:
            # Fetch the molecule SMILES through state relationship
            smiles = None
            if task.state and task.state.molecule:
                smiles = task.state.molecule.smiles

            results.append(
                {
                    "id": task.id,
                    "task_type": task.task_type.value,
                    "status": task.status.value,
                    "state_id": task.state_id,
                    "smiles": smiles,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "updated_at": task.updated_at.isoformat() if task.updated_at else None,
                }
            )
    return results


@router.get("/api/jobs")
def api_jobs(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    identity: Identity = Depends(current_identity),
):
    """List jobs with optional SLURM status filter."""
    with get_session() as session:
        stmt = select(ComputationJob).order_by(col(ComputationJob.id).desc())
        stmt = scope_jobs(stmt, session, identity)
        if status:
            stmt = stmt.where(ComputationJob.slurm_status == status)
        stmt = stmt.limit(limit)
        jobs = session.exec(stmt).all()

        results = []
        for job in jobs:
            results.append(
                {
                    "id": job.id,
                    "task_id": job.task_id,
                    "attempt": job.attempt,
                    "slurm_jobid": job.slurm_jobid,
                    "slurm_status": job.slurm_status,
                    "success": job.success,
                    "fail_reason": job.fail_reason,
                    "time_start": job.time_start.isoformat() if job.time_start else None,
                    "time_end": job.time_end.isoformat() if job.time_end else None,
                }
            )
    return results


@router.get("/api/queue")
def api_queue(identity: Identity = Depends(current_identity)):
    """Show the entrypoint queue (unstarted entries)."""
    with get_session() as session:
        stmt = (
            select(CalculationEntrypoint)
            .where(col(CalculationEntrypoint.time_started).is_(None))
            .order_by(CalculationEntrypoint.priority.desc(), CalculationEntrypoint.time_created)  # type: ignore[union-attr]
        )
        entries = visible_entrypoints(session.exec(stmt).all(), session, identity)

        results = []
        for entry in entries:
            results.append(
                {
                    "id": entry.id,
                    "smiles": entry.smiles,
                    "priority": entry.priority,
                    "time_created": entry.time_created.isoformat() if entry.time_created else None,
                    "processing_error": entry.processing_error,
                }
            )
    return results


@router.get("/api/projects")
def api_projects(identity: Identity = Depends(current_identity)):
    """List every distinct project name with summary counts."""
    with get_session() as session:
        names = session.exec(
            select(Molecule.project_name).distinct()
        ).all()
        allowed = visible_projects(session, identity)
        if allowed is not None:
            names = [n for n in names if n in set(allowed)]

        out = []
        for name in sorted(names):
            n_mol = session.exec(
                select(func.count()).select_from(Molecule).where(Molecule.project_name == name)
            ).one()
            # Tasks for molecules in this project
            n_tasks = session.exec(
                select(func.count())
                .select_from(ComputationTask)
                .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .join(Molecule, MoleculeState.molecule_id == Molecule.id)
                .where(Molecule.project_name == name)
            ).one()
            n_failed = session.exec(
                select(func.count())
                .select_from(ComputationTask)
                .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .join(Molecule, MoleculeState.molecule_id == Molecule.id)
                .where(Molecule.project_name == name, ComputationTask.status == TaskStatus.failed)
            ).one()
            n_succ = session.exec(
                select(func.count())
                .select_from(ComputationTask)
                .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .join(Molecule, MoleculeState.molecule_id == Molecule.id)
                .where(Molecule.project_name == name, ComputationTask.status == TaskStatus.successful)
            ).one()
            n_arch = session.exec(
                select(func.count())
                .select_from(Molecule)
                .where(Molecule.project_name == name, Molecule.archived == True)  # noqa: E712
            ).one()
            out.append({
                "name": name,
                "molecules": n_mol,
                "tasks_total": n_tasks,
                "tasks_failed": n_failed,
                "tasks_successful": n_succ,
                "archived": n_arch == n_mol and n_mol > 0,
                "protected": _is_protected(name),
            })
        return out


@router.get("/api/projects/{name}")
def api_project_detail(
    name: str, identity: Identity = Depends(current_identity),
):
    """Per-project view: molecules, submission progress, and success rate."""
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    with get_session() as session:
        name = resolve_project(session, identity, name)
    from autodft.extraction.extractor import PipelineExtractor

    extractor = PipelineExtractor(name)
    progress = extractor.get_submission_progress()
    success = extractor.get_success_rate()

    with get_session() as session:
        mols = session.exec(
            select(Molecule)
            .where(Molecule.project_name == name)
            .order_by(col(Molecule.created_at).desc())
        ).all()
        mol_ids = [m.id for m in mols if m.id is not None]

        # Two grouped queries instead of five COUNTs per molecule. At 3000
        # molecules the per-molecule version measured 9.1 s per request, on a
        # page the dashboard polls every 5 seconds per open tab -- each call
        # holding a threadpool slot and a SQLite connection while the worker
        # competes for the write lock.
        states_per_mol: dict[int, int] = {}
        tasks_per_mol: dict[int, dict[str, int]] = {}
        if mol_ids:
            for molecule_id, count in session.exec(
                select(MoleculeState.molecule_id, func.count())
                .where(col(MoleculeState.molecule_id).in_(mol_ids))
                .group_by(col(MoleculeState.molecule_id))
            ).all():
                states_per_mol[molecule_id] = count

            for molecule_id, status, count in session.exec(
                select(MoleculeState.molecule_id, ComputationTask.status, func.count())
                .select_from(ComputationTask)
                .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .where(col(MoleculeState.molecule_id).in_(mol_ids))
                .group_by(col(MoleculeState.molecule_id), col(ComputationTask.status))
            ).all():
                bucket = tasks_per_mol.setdefault(molecule_id, {})
                key = status.value if hasattr(status, "value") else str(status)
                bucket[key] = bucket.get(key, 0) + count

        mol_rows = []
        for m in mols:
            by_status = tasks_per_mol.get(m.id, {})
            n_tasks = sum(by_status.values())
            n_succ = by_status.get(TaskStatus.successful.value, 0)
            n_failed = by_status.get(TaskStatus.failed.value, 0)
            n_in_flight = (
                by_status.get(TaskStatus.created.value, 0)
                + by_status.get(TaskStatus.pending.value, 0)
            )
            mol_rows.append({
                "id": m.id,
                "smiles": m.smiles,
                "states": states_per_mol.get(m.id, 0),
                "tasks": n_tasks,
                "successful": n_succ,
                "failed": n_failed,
                "in_flight": n_in_flight,
                "done": n_in_flight == 0 and n_tasks > 0,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            })

    # Project-level completion summary. A project is "done" when every
    # task in every molecule has reached a terminal state (successful or
    # failed) — irrespective of which one. `in_flight_molecules` is the
    # count of molecules that still have at least one created/pending
    # task; `in_flight_tasks` is the total of those tasks across the
    # project. `status` is the single label the dashboard renders.
    in_flight_molecules = sum(1 for m in mol_rows if not m["done"])
    in_flight_tasks_total = sum(m["in_flight"] for m in mol_rows)
    total_mols = len(mol_rows)
    if total_mols == 0:
        status = "empty"
    elif in_flight_molecules == 0:
        # All terminal — check whether any are failed
        any_failed = any(m["failed"] > 0 for m in mol_rows)
        status = "complete_with_failures" if any_failed else "complete"
    else:
        status = "running"

    # Project-wide archive / protection flags.
    archived_project = total_mols > 0 and all(bool(getattr(m, "archived", False)) for m in mols)
    return {
        "name": name,
        "status": status,
        "archived": archived_project,
        "protected": _is_protected(name),
        "in_flight_molecules": in_flight_molecules,
        "in_flight_tasks": in_flight_tasks_total,
        "completed_molecules": total_mols - in_flight_molecules,
        "total_molecules": total_mols,
        "submission_progress": progress,
        "success_rate": success,
        "molecules": mol_rows,
    }


@router.get("/api/projects/{name}/molecules-detail")
def api_project_molecules_detail(
    name: str, identity: Identity = Depends(current_identity),
):
    """Per-molecule conformer-level breakdown for one project.

    Returns each molecule with an embedded 2-D SVG depiction, the list
    of states (S0 / T1 / ox / red), and per-conformer status for every
    task type spawned from that conformer's optimization. Used by the
    dashboard's "Project Overview → Molecules" subpage.

    Status values per task slot:
        "successful" | "failed" | "pending" | "created" | null (not requested).
    Each conformer = one ``optimization`` task; its dependent singlepoint
    family is looked up via ``ComputationTask.depends_on_task_id``.
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    with get_session() as session:
        name = resolve_project(session, identity, name)
    from autodft.models.geometry import MoleculeGeometry

    # Build the molecule list and gather all the related rows in a few
    # bulk queries — avoid the N+1 trap when a project has many mols.
    with get_session() as session:
        mols = session.exec(
            select(Molecule)
            .where(Molecule.project_name == name)
            .order_by(col(Molecule.id).asc())
        ).all()
        if not mols:
            return {"name": name, "molecules": []}

        mol_ids = [m.id for m in mols if m.id is not None]
        states = session.exec(
            select(MoleculeState).where(col(MoleculeState.molecule_id).in_(mol_ids))
        ).all()
        state_ids = [s.id for s in states if s.id is not None]

        tasks = (
            session.exec(
                select(ComputationTask).where(col(ComputationTask.state_id).in_(state_ids))
            ).all()
            if state_ids else []
        )

        # Look up the conformer index for every input geometry the
        # optimisation tasks reference, so we can render them in a
        # stable order.
        geom_ids = [t.input_geometry_id for t in tasks
                    if t.task_type == TaskType.optimization and t.input_geometry_id is not None]
        geoms = (
            session.exec(
                select(MoleculeGeometry).where(col(MoleculeGeometry.id).in_(geom_ids))
            ).all()
            if geom_ids else []
        )

    # Group helpers
    states_by_mol: dict[int, list[MoleculeState]] = {}
    for s in states:
        states_by_mol.setdefault(s.molecule_id, []).append(s)
    tasks_by_state: dict[int, list[ComputationTask]] = {}
    for t in tasks:
        tasks_by_state.setdefault(t.state_id, []).append(t)
    geom_by_id = {g.id: g for g in geoms}

    # task -> dependent tasks (singlepoint family hangs off the opt task)
    deps_by_parent: dict[int, list[ComputationTask]] = {}
    for t in tasks:
        if t.depends_on_task_id is not None:
            deps_by_parent.setdefault(t.depends_on_task_id, []).append(t)

    # Render the molecule list
    out_mols = []
    for m in mols:
        out_states = []
        for st in sorted(states_by_mol.get(m.id, []), key=_state_sort_key):
            opt_tasks = sorted(
                (t for t in tasks_by_state.get(st.id, []) if t.task_type == TaskType.optimization),
                key=lambda t: t.id,
            )
            conformers = []
            for idx, opt in enumerate(opt_tasks, start=1):
                deps = {d.task_type: d for d in deps_by_parent.get(opt.id, [])}
                conformers.append({
                    "index": idx,
                    "optimization":                  opt.status.value,
                    "singlepoint":                   _status_of(deps.get(TaskType.singlepoint)),
                    "singlepoint_vert_ox":           _status_of(deps.get(TaskType.singlepoint_vert_ox)),
                    "singlepoint_vert_red":          _status_of(deps.get(TaskType.singlepoint_vert_red)),
                    "singlepoint_vert_spin_change":  _status_of(deps.get(TaskType.singlepoint_vert_spin_change)),
                })
            # Confsearch status for the state — useful when no opt tasks exist yet.
            cs = next((t for t in tasks_by_state.get(st.id, [])
                       if t.task_type == TaskType.confsearch), None)
            out_states.append({
                "id": st.id,
                "description": st.description,
                "charge": st.charge,
                "multiplicity": st.multiplicity,
                "confsearch": _status_of(cs),
                "conformers": conformers,
            })
        out_mols.append({
            "id": m.id,
            "smiles": m.smiles,
            # Structure is rendered client-side via SmilesDrawer to keep
            # the backend free of the X11 / Cairo deps RDKit's Draw
            # module needs on a headless cluster.
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "states": out_states,
        })

    return {"name": name, "molecules": out_mols}


@router.get("/api/projects/{name}/state-analysis")
def api_project_state_analysis(
    name: str, identity: Identity = Depends(current_identity),
):
    """Per-molecule state analysis for one project.

    Returns triplet energies, redox free energies / E vs SCE in MeCN,
    and 4-point Marcus reorganisation energies for both
    ``lowest_energy`` and ``rmsd_matched`` conformer-selection modes.
    Solvation (MeCN) is detected from the optimisation / singlepoint
    header text — when not present, redox values are reported in
    ΔG only.

    Computation is delegated to ``autodft.analysis.state_analysis`` so
    the CSV/JSON exporters and this endpoint share the same energy
    extraction.
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    with get_session() as session:
        name = resolve_project(session, identity, name)
    from autodft.analysis.state_analysis import analyze_project
    return analyze_project(name)


@router.post("/api/projects/{name}/state-analysis/export")
def api_project_state_analysis_export(
    name: str, identity: Identity = Depends(current_identity),
):
    """Start building the state-analysis XLSX as a background job.

    Sheets: Summary, Lowest Energy, RMSD Matched, Conformers. Energies in
    Hartree (Eh); redox potentials in V vs SCE. Re-parsing every molecule's
    ORCA outputs takes seconds-to-minutes on a large project, so it runs off
    the request like the other exports: returns **202** with the job, then
    poll ``GET /api/projects/{name}/jobs`` and download via
    ``GET /api/jobs/{id}/download``. Works on archived projects too (the
    workbook is rebuilt from the frozen archive CSV).
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    with get_session() as session:
        name = resolve_project(session, identity, name)
    from autodft.api import project_jobs

    try:
        settings = get_active_settings()
        with get_session() as session:
            project_jobs.assert_no_active_job(session, name)
        job = project_jobs.start_job(
            owner_id=identity.user_id,
            qualified_name=name,
            kind=ProjectJobKind.export_xlsx,
            params={},
            settings=settings,
        )
        return JSONResponse(status_code=202, content={"project": name, "job": job})
    except project_jobs.JobInProgress as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    except Exception as exc:
        logging.getLogger(__name__).exception("Could not start XLSX export for project %r", name)
        return JSONResponse(
            status_code=500,
            content={"detail": f"XLSX export failed to start: {type(exc).__name__}: {exc}"},
        )


def _status_of(task: Optional[ComputationTask]) -> Optional[str]:
    return task.status.value if task is not None else None


_STATE_ORDER = {"S0": 0, "S1": 1, "T1": 2, "ox": 3, "red": 4}


def _state_sort_key(s: MoleculeState) -> tuple[int, str]:
    return (_STATE_ORDER.get(s.description, 99), s.description)


class ArchiveRequest(BaseModel):
    extensions: list[str] = [".inp", ".xyz", ".out"]
    all_conformers: bool = False


PROTECTED_PROJECT_NAMES = {"default"}


def _is_protected(name: str) -> bool:
    """Bare-name check; see admin_ops.is_protected for why."""
    from autodft.api.admin_ops import is_protected

    return is_protected(name)


def _project_is_archived(session, name: str) -> Optional[bool]:
    """Return True/False if every molecule in the project is archived,
    or None if the project doesn't exist."""
    rows = session.exec(
        select(Molecule.archived).where(Molecule.project_name == name)
    ).all()
    if not rows:
        return None
    return all(bool(r) for r in rows)


@router.post("/api/projects/{name}/archive")
def api_project_archive(
    name: str, body: ArchiveRequest, identity: Identity = Depends(current_identity),
):
    """Start a destructive archive of one project as a background job.

    Writes the CSV summary + the user-selected files into
    ``<export_data>/<name>/`` (CSV at the root, raw files under
    ``raw/``), then deletes every ``<comp_data>/mol_<id>/`` belonging
    to the project and flips ``Molecule.archived = True`` on every row.

    The work runs on a background thread so it survives the browser closing;
    poll ``GET /api/projects/{name}/jobs`` for progress. Returns **202** with
    the created job.

    Refuses with 4xx when:
    * the project is ``"default"`` (protected — never archivable)
    * the project doesn't exist (404)
    * the project is already archived (409)
    * the project already has a background job in flight (409)
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    with get_session() as session:
        name = resolve_project(session, identity, name)
    from autodft.api import project_jobs

    if _is_protected(name):
        return JSONResponse(
            status_code=409,
            content={"detail": f"Project {name!r} is protected and cannot be archived."},
        )

    try:
        settings = get_active_settings()

        with get_session() as session:
            state = _project_is_archived(session, name)
            if state is None:
                return JSONResponse(status_code=404, content={"detail": f"Project {name!r} has no molecules"})
            if state is True:
                return JSONResponse(status_code=409, content={"detail": f"Project {name!r} is already archived."})
            project_jobs.assert_no_active_job(session, name)

        job = project_jobs.start_job(
            owner_id=identity.user_id,
            qualified_name=name,
            kind=ProjectJobKind.archive,
            params={"extensions": body.extensions, "all_conformers": body.all_conformers},
            settings=settings,
        )
        return JSONResponse(status_code=202, content={"project": name, "job": job})
    except project_jobs.JobInProgress as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except Exception as exc:
        logging.getLogger(__name__).exception("Could not start archive for project %r", name)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Archive failed to start: {type(exc).__name__}: {exc}"},
        )


# Export format -> the background-job kind that produces it. `xlsx` is the
# state-analysis workbook; the rest come off the PipelineExtractor.
_EXPORT_KINDS = {
    "csv": ProjectJobKind.export_csv,
    "json": ProjectJobKind.export_json,
    "files": ProjectJobKind.export_files,
    "xlsx": ProjectJobKind.export_xlsx,
}


@router.post("/api/projects/{name}/export")
def api_project_export(
    name: str,
    format: str = Query("csv"),
    all_conformers: bool = Query(False),
    identity: Identity = Depends(current_identity),
):
    """Start an export for one project as a background job.

    Writes into ``<export_data>/<name>/``. Format: ``csv`` -> ``<name>.csv``;
    ``json`` -> ``<name>.json``; ``files`` -> raw ORCA files under
    ``<name>/files/``; ``xlsx`` -> the state-analysis workbook.

    The work runs on a background thread so it survives the browser closing;
    poll ``GET /api/projects/{name}/jobs`` and download the result via
    ``GET /api/jobs/{id}/download``. Returns **202** with the created job.

    ``csv`` / ``json`` / ``files`` need the on-disk ORCA outputs, so an
    archived project is refused (409); ``xlsx`` is built from the frozen
    archive CSV and is allowed. A project with a job already in flight is
    refused (409).
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    with get_session() as session:
        name = resolve_project(session, identity, name)
    from autodft.api import project_jobs

    kind = _EXPORT_KINDS.get(format)
    if kind is None:
        return JSONResponse(status_code=400, content={"detail": f"Unknown format {format!r}"})

    try:
        settings = get_active_settings()

        with get_session() as session:
            state = _project_is_archived(session, name)
            if state is None:
                return JSONResponse(status_code=404, content={"detail": f"Project {name!r} has no molecules"})
            if state is True and kind is not ProjectJobKind.export_xlsx:
                return JSONResponse(
                    status_code=409,
                    content={"detail": f"Project {name!r} is archived — source files are no longer on disk."},
                )
            project_jobs.assert_no_active_job(session, name)

        job = project_jobs.start_job(
            owner_id=identity.user_id,
            qualified_name=name,
            kind=kind,
            params={"all_conformers": all_conformers},
            settings=settings,
        )
        return JSONResponse(status_code=202, content={"project": name, "job": job})
    except project_jobs.JobInProgress as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    except Exception as exc:
        logging.getLogger(__name__).exception("Could not start export for project %r (format=%r)", name, format)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Export failed to start: {type(exc).__name__}: {exc}"},
        )


@router.get("/api/projects/{name}/jobs")
def api_project_jobs(name: str, identity: Identity = Depends(current_identity)):
    """Recent background jobs for one project, newest first.

    ``active`` is the job currently running (or null). Used by the dashboard
    to disable the project's controls and poll progress while one runs.
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    from autodft.api import project_jobs

    with get_session() as session:
        name = resolve_project(session, identity, name)
        jobs = [project_jobs.serialize(j) for j in project_jobs.list_jobs(session, name)]
    active = next((j for j in jobs if j["status"] == "running"), None)
    return {"project": name, "active": active, "jobs": jobs}


@router.get("/api/jobs/{job_id}")
def api_job_detail(job_id: int, identity: Identity = Depends(current_identity)):
    """One background job, scoped to a caller who may see its project."""
    from autodft.api import project_jobs

    with get_session() as session:
        job = project_jobs.get_job(session, job_id)
        if job is None or not _may_see_project(session, identity, job.qualified_name):
            return JSONResponse(status_code=404, content={"detail": f"Job {job_id} not found"})
        return project_jobs.serialize(job)


@router.get("/api/jobs/{job_id}/download")
def api_job_download(job_id: int, identity: Identity = Depends(current_identity)):
    """Stream the file a finished export produced.

    Only for successful jobs whose result is a single downloadable file
    (csv / json / xlsx). ``files`` exports and archives produce a directory
    tree on the server and answer 400 here.
    """
    from fastapi.responses import FileResponse

    from autodft.api import project_jobs

    with get_session() as session:
        job = project_jobs.get_job(session, job_id)
        if job is None or not _may_see_project(session, identity, job.qualified_name):
            return JSONResponse(status_code=404, content={"detail": f"Job {job_id} not found"})
        result = project_jobs.serialize(job)["result"] or {}
        status = job.status.value

    if status != "successful":
        return JSONResponse(status_code=409, content={"detail": f"Job {job_id} is {status}, not successful."})
    if not result.get("downloadable") or not result.get("path"):
        return JSONResponse(
            status_code=400,
            content={"detail": "This job did not produce a single downloadable file."},
        )
    path = Path(result["path"])
    if not path.is_file():
        return JSONResponse(status_code=404, content={"detail": "The produced file is no longer on disk."})
    return FileResponse(path, filename=path.name)


def _may_see_project(session, identity: Identity, qualified_name: str) -> bool:
    """Whether *identity* may read data for *qualified_name*."""
    if identity.is_admin:
        return True
    from autodft import accounts

    owner = accounts.owner_of(session, qualified_name)
    return owner is not None and owner.id == identity.user_id


@router.get("/api/entrypoints/failed")
def api_failed_entrypoints(identity: Identity = Depends(current_identity)):
    """Entrypoints that hit a processing error and were never expanded
    into molecules/tasks (e.g. unparseable SMILES). Surface these so the
    user can fix the input and resubmit.

    Filtered to the caller: a failed entrypoint carries the SMILES that was
    submitted, so an unfiltered list handed every user everyone else's
    structures.
    """
    with get_session() as session:
        stmt = (
            select(CalculationEntrypoint)
            .where(col(CalculationEntrypoint.processing_error).is_not(None))
            .order_by(col(CalculationEntrypoint.time_started).desc())
        )
        entries = visible_entrypoints(session.exec(stmt).all(), session, identity)
        return [
            {
                "id": e.id,
                "smiles": e.smiles,
                "priority": e.priority,
                "time_created": e.time_created.isoformat() if e.time_created else None,
                "time_started": e.time_started.isoformat() if e.time_started else None,
                "processing_error": e.processing_error,
            }
            for e in entries
        ]


@router.get("/api/headers")
def api_headers(
    kind: Optional[str] = Query(None),
    include_deleted: bool = Query(False),
):
    """List computation headers.

    Pass ``?kind=confsearch|optimization|singlepoint`` to filter the
    ``custom`` list strictly to headers tagged with that kind (untagged
    custom headers are NOT returned — tag them explicitly to make them
    appear in a slot's dropdown).

    Soft-deleted headers are hidden by default; pass
    ``?include_deleted=true`` to see them (used by the manager view).
    """
    from autodft.qm.orca.defaults import (
        DEFAULT_HEADER_CONFSEARCH,
        DEFAULT_HEADER_OPTIMIZATION,
        DEFAULT_HEADER_SINGLEPOINT,
    )

    defaults = [
        {
            "id": "default_confsearch",
            "label": "GOAT XTB2 (default)",
            "description": "GOAT conformer ensemble at GFN2-xTB; MAXEN 10, ENDIFF 0.2, RMSD 0.15.",
            "kind": "confsearch",
            "text": DEFAULT_HEADER_CONFSEARCH,
        },
        {
            "id": "default_optimization",
            "label": "wB97X-D3 / def2-TZVP TightOpt Freq (default)",
            "description": "Geometry optimisation + frequencies at wB97X-D3/def2-TZVP with RIJCOSX def2/J.",
            "kind": "optimization",
            "text": DEFAULT_HEADER_OPTIMIZATION,
        },
        {
            "id": "default_singlepoint",
            "label": "wB97X-D3 / def2-QZVPD KeepDens Freq (default)",
            "description": "Single-point at wB97X-D3/def2-QZVPD with RIJCOSX, KeepDens for downstream densities.",
            "kind": "singlepoint",
            "text": DEFAULT_HEADER_SINGLEPOINT,
        },
    ]

    with get_session() as session:
        stmt = select(ComputationHeader).order_by(ComputationHeader.id.desc())  # type: ignore[union-attr]
        if not include_deleted:
            stmt = stmt.where(ComputationHeader.deleted == False)  # noqa: E712
        if kind:
            # Strict match — untagged custom headers are intentionally
            # excluded from slot-filtered listings.
            stmt = stmt.where(ComputationHeader.kind == kind)
        headers = session.exec(stmt).all()
        db_headers = [
            {
                "id": h.id,
                "label": (h.description or f"Header #{h.id}")[:80],
                "description": h.description,
                "kind": h.kind,
                "validated": h.validated,
                "deleted": h.deleted,
                "text": h.header_text,
            }
            for h in headers
        ]

    return {"defaults": defaults, "custom": db_headers}


def _caller_id(session, identity: Identity):
    """The caller's user row id, or None when accounts are not set up."""
    from autodft import accounts

    user = accounts.get_user_by_username(session, identity.username)
    return None if user is None else user.id


def _may_change_header(session, identity: Identity, header) -> bool:
    """Whether *identity* may edit or remove *header*.

    Everyone uses every header; only its owner or admin changes one.
    """
    if identity.is_admin:
        return True
    owner_id = getattr(header, "owner_id", None)
    return owner_id is not None and owner_id == _caller_id(session, identity)


_HEADER_FORBIDDEN = {
    "detail": "This header belongs to another user. Create your own copy "
              "instead, or ask an administrator to change it."
}


@router.post("/api/headers")
def api_create_header(
    body: HeaderCreate, identity: Identity = Depends(current_identity),
):
    """Create a new custom computation header, owned by the caller.

    Anyone may create one, and anyone may *use* anyone's -- a method
    library is only useful shared. Only the owner (or admin) may later
    change or remove it, because editing a header silently changes what
    the next submission referencing it computes.
    """
    if body.kind and body.kind not in {"confsearch", "optimization", "singlepoint"}:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid kind {body.kind!r}"},
        )
    with get_session() as session:
        header = ComputationHeader(
            header_text=body.header_text,
            description=body.description,
            kind=body.kind,
            validated=body.validated,
            owner_id=_caller_id(session, identity),
        )
        session.add(header)
        session.commit()
        session.refresh(header)
        return {
            "id": header.id,
            "header_text": header.header_text,
            "description": header.description,
            "kind": header.kind,
            "validated": header.validated,
        }


@router.put("/api/headers/{header_id}")
def api_update_header(
    header_id: int, body: HeaderUpdate,
    identity: Identity = Depends(current_identity),
):
    """Update fields on an existing header. Owner or admin only."""
    if body.kind is not None and body.kind not in {"", "confsearch", "optimization", "singlepoint"}:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid kind {body.kind!r}"},
        )
    with get_session() as session:
        header = session.get(ComputationHeader, header_id)
        if header is None:
            return JSONResponse(status_code=404, content={"detail": "Header not found"})
        if not _may_change_header(session, identity, header):
            return JSONResponse(status_code=403, content=_HEADER_FORBIDDEN)
        if body.header_text is not None:
            header.header_text = body.header_text
        if body.description is not None:
            header.description = body.description
        if body.kind is not None:
            header.kind = body.kind or None
        if body.validated is not None:
            header.validated = body.validated
        session.add(header)
        session.commit()
        session.refresh(header)
        return {
            "id": header.id,
            "header_text": header.header_text,
            "description": header.description,
            "kind": header.kind,
            "validated": header.validated,
        }


@router.delete("/api/headers/{header_id}")
def api_delete_header(
    header_id: int, identity: Identity = Depends(current_identity),
):
    """Soft-delete a custom header. Owner or admin only.

    Refused only when an *in-flight* task (status ``created`` or
    ``pending``) still references it directly or through its state.
    Headers referenced by already-finished tasks (``successful`` /
    ``failed``) can be removed: the header row is kept in the table
    with ``deleted=True`` so historical FK pointers stay valid, but
    it disappears from listings and submission dropdowns.
    """
    from autodft.models.state import MoleculeState

    with get_session() as session:
        header = session.get(ComputationHeader, header_id)
        if header is None:
            return JSONResponse(status_code=404, content={"detail": "Header not found"})
        if not _may_change_header(session, identity, header):
            return JSONResponse(status_code=403, content=_HEADER_FORBIDDEN)

        # 1. Direct references on ComputationTask
        direct_inflight = session.exec(
            select(func.count())
            .select_from(ComputationTask)
            .where(
                ComputationTask.header_id == header_id,
                col(ComputationTask.status).in_([TaskStatus.created, TaskStatus.pending]),
            )
        ).one()

        # 2. Indirect references through MoleculeState -> ComputationTask
        indirect_inflight = session.exec(
            select(func.count())
            .select_from(ComputationTask)
            .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
            .where(
                col(ComputationTask.status).in_([TaskStatus.created, TaskStatus.pending]),
                (
                    (MoleculeState.confsearch_header_id == header_id)
                    | (MoleculeState.optimization_header_id == header_id)
                    | (MoleculeState.singlepoint_header_id == header_id)
                ),
            )
        ).one()

        blocking = direct_inflight + indirect_inflight
        if blocking:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": (
                        f"Header is still in use by {blocking} in-flight task(s) "
                        f"(created/pending). Wait for those to finish or fail, "
                        f"then delete."
                    ),
                },
            )

        if header.deleted:
            return {"id": header_id, "deleted": True, "already": True}

        header.deleted = True
        session.add(header)
        session.commit()
        return {"id": header_id, "deleted": True}


# ---------------------------------------------------------------------------
# Admin -- destructive maintenance
#
# Every wipe is two calls: a preview that only counts, then the wipe itself
# with the exact confirmation string echoed back. The confirmation is the
# project name / molecule SMILES / "RESET THE DATABASE", which is also what
# the preview response reports as `confirmation_required`.
# ---------------------------------------------------------------------------


class WipeRequest(BaseModel):
    confirm: str
    # Project wipes remove <export_data>/<project>/ as well by default.
    delete_exports: bool = True


class ResetRequest(BaseModel):
    confirm: str
    delete_files: bool = True
    keep_headers: bool = True


def _confirmation_error(expected: str, got: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "detail": (
                f"Confirmation mismatch. Type exactly {expected!r} to proceed "
                f"(received {got!r}). Nothing was deleted."
            )
        },
    )


def _admin_scheduler(settings):
    """A scheduler for the destructive routes to cancel jobs with.

    Wiping a project used to leave its queued and running jobs alive; they
    kept writing ORCA output into directories that had just been deleted.
    Returns None if SLURM isn't reachable -- a wipe must still work on a
    machine with no scheduler.
    """
    try:
        from autodft.engine.scheduler import SlurmScheduler

        return SlurmScheduler(
            partition=settings.slurm.partition, nice=settings.slurm.nice,
        )
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("No scheduler available; not cancelling jobs")
        return None


@router.get("/api/wipe-status")
def api_wipe_status(identity: Identity = Depends(current_identity)):
    """Whether a destructive operation is still deleting files.

    A project wipe and a database reset answer as soon as the rows are gone
    and the directories are renamed out of the way; the unlinking continues
    on a background thread, and until it finishes another wipe is refused.
    """
    from autodft.api import admin_ops

    operation = admin_ops.current_operation()
    if not identity.is_admin:
        # The label names the project being wiped, which may be someone
        # else's. A user only needs to know that they have to wait.
        return {"running": operation is not None, "operation": None,
                "file_removal": None}
    return {
        "running": operation is not None,
        "operation": operation,
        "file_removal": admin_ops.removal_status(),
    }


@router.get("/api/projects/{name}/wipe-preview")
def api_project_wipe_preview(
    name: str, identity: Identity = Depends(current_identity),
):
    """What a wipe of this project would delete. Read-only."""
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    with get_session() as session:
        name = resolve_project(session, identity, name)
    from autodft.api import admin_ops

    settings = get_active_settings()
    with get_session() as session:
        return admin_ops.preview_project_wipe(
            session, name, settings.comp_data_path, settings.export_data_path,
        )


@router.post("/api/projects/{name}/wipe")
def api_project_wipe(
    name: str, body: WipeRequest,
    identity: Identity = Depends(current_identity),
):
    """Permanently delete a project: DB rows, raw comp_data, exported data.

    Requires ``confirm`` to equal the project name exactly. Irreversible.

    Scoped, not admin-only: a user may wipe their own projects. The name
    is resolved through :func:`resolve_project`, so someone else's answers
    404 exactly as it does for a read.
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    with get_session() as session:
        name = resolve_project(session, identity, name)
    from autodft.api import admin_ops, project_jobs

    if body.confirm != name:
        return _confirmation_error(name, body.confirm)

    with get_session() as session:
        try:
            project_jobs.assert_no_active_job(session, name)
        except project_jobs.JobInProgress as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})

    settings = get_active_settings()
    try:
        with admin_ops.exclusive(f"wipe of project {name!r}"):
            with get_session() as session:
                return admin_ops.wipe_project(
                    session, name, settings.comp_data_path, settings.export_data_path,
                    delete_exports=body.delete_exports,
                    scheduler=_admin_scheduler(settings),
                )
    except admin_ops.WipeInProgress as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    except ValueError as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    except Exception as exc:
        logging.getLogger(__name__).exception("Wipe failed for project %r", name)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Wipe failed: {type(exc).__name__}: {exc}"},
        )


@router.get("/api/molecules/{molecule_id}/wipe-preview")
def api_molecule_wipe_preview(
    molecule_id: int, identity: Identity = Depends(current_identity),
):
    """What wiping this single calculation would delete. Read-only."""
    from autodft.api import admin_ops

    settings = get_active_settings()
    with get_session() as session:
        require_molecule(session, identity, molecule_id)
        preview = admin_ops.preview_molecule_wipe(
            session, molecule_id, settings.comp_data_path,
        )
    if preview is None:
        return JSONResponse(status_code=404, content={"detail": f"Molecule {molecule_id} not found"})
    return preview


@router.post("/api/molecules/{molecule_id}/wipe")
def api_molecule_wipe(
    molecule_id: int, body: WipeRequest,
    identity: Identity = Depends(current_identity),
):
    """Delete one molecule with its states, tasks, jobs and raw files.

    Requires ``confirm`` to equal the molecule's SMILES exactly. A user
    may wipe their own molecules; someone else's answers 404.
    """
    from autodft.api import admin_ops

    settings = get_active_settings()
    try:
        with admin_ops.exclusive(f"wipe of molecule {molecule_id}"):
            with get_session() as session:
                mol = require_molecule(session, identity, molecule_id)
                if body.confirm != mol.smiles:
                    return _confirmation_error(mol.smiles, body.confirm)
                return admin_ops.wipe_molecule(
                    session, molecule_id, settings.comp_data_path,
                    scheduler=_admin_scheduler(settings),
                )
    except admin_ops.WipeInProgress as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    except ValueError as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    except Exception as exc:
        logging.getLogger(__name__).exception("Wipe failed for molecule %d", molecule_id)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Wipe failed: {type(exc).__name__}: {exc}"},
        )


@router.get("/api/whoami")
def api_whoami(identity: Identity = Depends(current_identity)):
    """The caller, and what they own.

    The dashboard uses this to decide what to render: a user has no admin
    section, no other people's projects, and a locked author field.
    """
    from autodft import accounts

    with get_session() as session:
        user = accounts.get_user_by_username(session, identity.username)
        # What the caller *owns*, which is not the same as what they can
        # see: admin sees everything, and reporting that as ownership made
        # its own projects vanish from this list.
        owned = [] if user is None else accounts.projects_owned_by(session, user)
        names = sorted(p.name for p in owned)
    return {
        "username": identity.username,
        "is_admin": identity.is_admin,
        # Bare names: the owner is the caller, so the prefix is noise here.
        "projects": names,
    }


@router.get("/api/cluster")
def api_cluster_status():
    """Read-only pipeline health, for everyone who is signed in.

    Without this a user cannot tell "my jobs are stuck" from "the pipeline
    is halted" without asking an administrator. It exposes no one else's
    data: a queue depth and a breaker flag.
    """
    from autodft.engine import circuit_breaker

    settings = get_active_settings()
    state = circuit_breaker.read_state(settings.data_path)
    with get_session() as session:
        waiting = session.exec(
            select(func.count())
            .select_from(CalculationEntrypoint)
            .where(col(CalculationEntrypoint.time_started).is_(None))
        ).one()
    return {
        "breaker_tripped": state is not None,
        "queued_entrypoints": waiting,
    }


# ---------------------------------------------------------------------------
# User administration
# ---------------------------------------------------------------------------


class CreateUserRequest(BaseModel):
    username: str = Field(max_length=32)
    display_name: str = Field(default="", max_length=128)
    role: str = "user"


class ReassignRequest(BaseModel):
    owner: str = Field(max_length=32)


def _user_summary(session, user, counts: Optional[dict] = None) -> dict:
    """One account's row for the admin listing.

    *counts* is an optional ``{project_name: molecules}`` map so the
    listing can group once instead of running a COUNT per user.
    """
    from autodft import accounts

    projects = accounts.projects_owned_by(session, user)
    if counts is None:
        molecules = session.exec(
            select(func.count())
            .select_from(Molecule)
            .where(col(Molecule.project_name).in_([p.qualified_name for p in projects] or [""]))
        ).one()
    else:
        molecules = sum(counts.get(p.qualified_name, 0) for p in projects)
    return {
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role.value,
        "active": user.active,
        "api_key_prefix": user.api_key_prefix,
        "projects": [p.name for p in projects],
        "molecules": molecules,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_seen_at": user.last_seen_at.isoformat() if user.last_seen_at else None,
    }


@router.get("/api/admin/users")
def api_users(identity: Identity = Depends(current_identity)):
    """Every account, with what it owns."""
    require_admin(identity)
    from autodft.models import User

    with get_session() as session:
        users = session.exec(select(User).order_by(col(User.username))).all()
        counts = dict(session.exec(
            select(Molecule.project_name, func.count())
            .group_by(Molecule.project_name)
        ).all())
        return [_user_summary(session, u, counts) for u in users]


@router.post("/api/admin/users")
def api_create_user(
    body: CreateUserRequest, identity: Identity = Depends(current_identity),
):
    """Create an account and mint its API key.

    The key is in the response and nowhere else: only its hash is stored,
    so it can afterwards be rotated but never read back.
    """
    require_admin(identity)
    from autodft import accounts
    from autodft.models import UserRole

    try:
        role = UserRole(body.role)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Unknown role {body.role!r}. Use 'admin' or 'user'."},
        )

    with get_session() as session:
        try:
            user, key = accounts.create_user(
                session, body.username, display_name=body.display_name, role=role,
            )
        except (accounts.AccountError, ValueError) as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        summary = _user_summary(session, user)

    summary["api_key"] = key
    summary["api_key_warning"] = (
        "This key is shown once. Copy it now -- it cannot be recovered, "
        "only rotated."
    )
    return summary


@router.post("/api/admin/users/{username}/rotate-key")
def api_rotate_key(username: str, identity: Identity = Depends(current_identity)):
    """Issue a new key, invalidating the old one immediately."""
    require_admin(identity)
    from autodft import accounts

    with get_session() as session:
        user = accounts.get_user_by_username(session, username)
        if user is None:
            return JSONResponse(
                status_code=404, content={"detail": f"No user named {username!r}."},
            )
        key = accounts.rotate_api_key(session, user)
        summary = _user_summary(session, user)
    summary["api_key"] = key
    return summary


@router.post("/api/admin/users/{username}")
def api_update_user(
    username: str,
    active: Optional[bool] = Query(None),
    identity: Identity = Depends(current_identity),
):
    """Deactivate or reactivate an account.

    There is no delete: an account whose projects still hold hundreds of
    gigabytes should not disappear in one click. Wipe or reassign the
    projects first, deliberately, then deactivate.
    """
    require_admin(identity)
    from autodft import accounts

    with get_session() as session:
        user = accounts.get_user_by_username(session, username)
        if user is None:
            return JSONResponse(
                status_code=404, content={"detail": f"No user named {username!r}."},
            )
        if user.username == accounts.ADMIN_USERNAME and active is False:
            return JSONResponse(
                status_code=400,
                content={"detail": "The admin account cannot be deactivated."},
            )
        if active is not None:
            user.active = active
            session.add(user)
            session.commit()
            session.refresh(user)
        return _user_summary(session, user)


@router.post("/api/admin/projects/{name}/reassign")
def api_reassign_project(
    name: str, body: ReassignRequest,
    identity: Identity = Depends(current_identity),
):
    """Move a project to another owner, rewriting every reference to it."""
    require_admin(identity)
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    from autodft import accounts

    with get_session() as session:
        name = resolve_project(session, identity, name)
        new_owner = accounts.get_user_by_username(session, body.owner)
        if new_owner is None:
            return JSONResponse(
                status_code=404, content={"detail": f"No user named {body.owner!r}."},
            )
        try:
            project = accounts.reassign_project(session, name, new_owner)
        except accounts.AccountError as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        return {"project": project.qualified_name, "owner": new_owner.username}


@router.get("/api/admin/circuit-breaker")
def api_circuit_breaker_status():
    """Whether the global failure breaker has halted new submissions."""
    from autodft.engine import circuit_breaker

    settings = get_active_settings()
    state = circuit_breaker.read_state(settings.data_path)
    with get_session() as session:
        ratio, failed, judged = circuit_breaker.recent_failure_ratio(
            session, settings.pipeline.failure_breaker_window,
        )
    return {
        "tripped": state is not None,
        "state": state,
        "recent_failure_ratio": round(ratio, 4),
        "recent_failed": failed,
        "recent_judged": judged,
        "threshold": settings.pipeline.failure_breaker_ratio,
        "window": settings.pipeline.failure_breaker_window,
    }


@router.post("/api/admin/circuit-breaker/reset")
def api_circuit_breaker_reset(identity: Identity = Depends(current_identity)):
    """Clear the breaker so the pipeline resumes creating and submitting jobs.

    Deliberately manual: once submission stops no new tasks are judged, so
    the failure ratio cannot recover on its own. Fix the cause first.
    """
    require_admin(identity)
    from autodft.engine import circuit_breaker

    settings = get_active_settings()
    was_tripped = circuit_breaker.reset(settings.data_path)
    return {"reset": was_tripped, "tripped": False}


@router.get("/api/admin/reset-preview")
def api_reset_preview(identity: Identity = Depends(current_identity)):
    """Everything currently in the database. Read-only, and reads no files.

    Disk usage is *not* here: see ``/api/admin/disk-usage``. This endpoint
    is fetched every time the admin page renders, and measuring 32 GB
    across the network mount took five minutes per render.
    """
    require_admin(identity)
    from autodft.api import admin_ops

    with get_session() as session:
        return admin_ops.preview_database_reset(session)


@router.get("/api/admin/disk-usage")
def api_disk_usage(identity: Identity = Depends(current_identity)):
    """The last measurement, without starting one. ``null`` if never asked."""
    require_admin(identity)
    from autodft.api import admin_ops

    return {"usage": admin_ops.disk_usage_status()}


@router.post("/api/admin/disk-usage")
def api_disk_usage_measure(identity: Identity = Depends(current_identity)):
    """Start measuring the data directory, and report progress.

    Answers immediately: the walk runs on its own thread and the caller
    polls the GET above. A request arriving while a walk is under way joins
    it rather than starting a second stat storm on the same mount.
    """
    require_admin(identity)
    from autodft.api import admin_ops

    settings = get_active_settings()
    return {"usage": admin_ops.start_disk_usage(
        settings.comp_data_path, settings.export_data_path,
    )}


@router.post("/api/admin/reset-database")
def api_reset_database(
    body: ResetRequest, identity: Identity = Depends(current_identity),
):
    """Empty every pipeline table, and by default every data directory.

    Requires ``confirm`` to equal ``RESET THE DATABASE`` exactly. Saved
    headers are kept unless ``keep_headers`` is false, in which case the
    standard set is reseeded.
    """
    require_admin(identity)
    from autodft.api import admin_ops, project_jobs

    if body.confirm != admin_ops.RESET_CONFIRMATION:
        return _confirmation_error(admin_ops.RESET_CONFIRMATION, body.confirm)

    # A reset empties every project, so refuse while any project has a
    # background archive/export in flight.
    with get_session() as session:
        busy = project_jobs.active_job_project_names(session)
    if busy:
        return JSONResponse(
            status_code=409,
            content={"detail": (
                "A background archive/export job is running for "
                + ", ".join(sorted(busy))
                + ". Wait for it to finish before resetting the database."
            )},
        )

    settings = get_active_settings()
    try:
        with admin_ops.exclusive("database reset"):
            with get_session() as session:
                return admin_ops.reset_database(
                    session, settings.comp_data_path, settings.export_data_path,
                    delete_files=body.delete_files, keep_headers=body.keep_headers,
                    scheduler=_admin_scheduler(settings),
                )
    except admin_ops.WipeInProgress as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    except Exception as exc:
        logging.getLogger(__name__).exception("Database reset failed")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Reset failed: {type(exc).__name__}: {exc}"},
        )


class ValidateSmilesRequest(BaseModel):
    smiles: str = Field(max_length=512)


@router.post("/api/validate-smiles")
def api_validate_smiles(body: ValidateSmilesRequest):
    """Run RDKit validation on a SMILES string.

    Used by the dashboard's submission form to show live feedback as the
    user types, and by /api/submit before queuing. Returns a structured
    result rather than HTTP error codes so the frontend can show a hint
    inline.
    """
    from autodft.engine.entrypoint_processor import validate_smiles
    return validate_smiles(body.smiles)


def _reject_reason(body: SubmitRequest, smiles: str) -> Optional[tuple[str, dict]]:
    """Why *smiles* cannot be queued under *body*'s options, or None."""
    from autodft.engine.entrypoint_processor import validate_smiles

    check = validate_smiles(smiles)
    if not check["valid"]:
        return (check["error"] or "Invalid SMILES.", check)

    # The S0 -> T1 spin change is only defined from a closed-shell reference.
    # Refuse here rather than letting the controller build a state that is
    # either arithmetically impossible (odd electron count with multiplicity
    # 3) or a silent duplicate of S0.
    if body.request_t1 and check["multiplicity"] != 1:
        return (
            f"T1 requires a closed-shell reference, but {smiles!r} has "
            f"multiplicity {check['multiplicity']}. Resubmit without "
            f"request_t1 — ox and red remain available for open-shell "
            f"references.",
            check,
        )
    return None


@router.post("/api/submit")
def api_submit(
    body: SubmitRequest, identity: Identity = Depends(current_identity),
):
    """Submit a new molecule to the calculation queue.

    Accepting a submission is a single INSERT into ``calculation_entrypoints``
    -- it never waits on SLURM, on the queue depth, or on anything the worker
    is doing. Whether the work can start now is the controller's problem, not
    the caller's.
    """
    rejection = _reject_reason(body, body.smiles)
    if rejection is not None:
        detail, check = rejection
        return JSONResponse(
            status_code=400, content={"detail": detail, "validation": check},
        )

    from autodft.api import project_jobs

    with get_session() as session:
        project_name, author = _submission_owner(session, identity, body)
        try:
            project_jobs.assert_no_active_job(session, project_name)
        except project_jobs.JobInProgress as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        entry = _new_entrypoint(session, body, body.smiles, project_name, author)
        session.add(entry)
        session.commit()
        session.refresh(entry)

        return {
            "id": entry.id,
            "smiles": entry.smiles,
            "status": "queued",
            # Echoed because the caller sent a *bare* name and the server
            # qualified it. Without this a script cannot tell which
            # namespace its work landed in without a second request.
            "project": project_name,
            "author": author,
            "time_created": entry.time_created.isoformat() if entry.time_created else None,
        }


class SubmitBatchRequest(SubmitRequest):
    """A :class:`SubmitRequest` whose ``smiles`` is a list.

    ``smiles`` is inherited but unused; ``smiles_list`` carries the molecules.
    """

    smiles: str = ""
    smiles_list: list[str] = Field(default_factory=list, max_length=10000)


@router.post("/api/submit-batch")
def api_submit_batch(
    body: SubmitBatchRequest, identity: Identity = Depends(current_identity),
):
    """Queue many molecules under one set of options, in one transaction.

    The single-molecule endpoint is fine for a handful, but a library-sized
    campaign submitted one HTTP request at a time pays a SQLite commit per
    molecule and competes with the worker for the write lock on every one of
    them. Here the whole batch is one commit.

    Rejections do not fail the request: every SMILES is reported individually
    so the caller can log what was refused and keep the rest.
    """
    if not body.smiles_list:
        return JSONResponse(
            status_code=400,
            content={"detail": "smiles_list is empty; nothing to queue."},
        )

    accepted: list[tuple[str, CalculationEntrypoint]] = []
    rejected: list[dict] = []

    from autodft.api import project_jobs

    with get_session() as session:
        project_name, author = _submission_owner(session, identity, body)
        try:
            project_jobs.assert_no_active_job(session, project_name)
        except project_jobs.JobInProgress as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        for smiles in body.smiles_list:
            if len(smiles) > 512:
                # Matches the bound on SubmitRequest.smiles: RDKit's parser
                # overflows the C stack on very long input, and it runs in a
                # thread of the controller process.
                rejected.append({"smiles": smiles[:120], "detail": "SMILES too long (>512)."})
                continue
            rejection = _reject_reason(body, smiles)
            if rejection is not None:
                rejected.append({"smiles": smiles, "detail": rejection[0]})
                continue
            entry = _new_entrypoint(session, body, smiles, project_name, author)
            session.add(entry)
            accepted.append((smiles, entry))

        session.commit()

        return {
            "queued": [
                {"id": entry.id, "smiles": smiles} for smiles, entry in accepted
            ],
            "rejected": rejected,
            "project": project_name,
            "author": author,
            "counts": {"queued": len(accepted), "rejected": len(rejected)},
        }


def _submission_owner(session, identity: Identity, body: SubmitRequest) -> tuple[str, str]:
    """The qualified project and the author to record for this submission.

    The author is always the caller's username -- admin included. A
    provenance label anyone may set is not provenance, and an exemption
    for admin is an exemption for whoever holds the shared password. Work
    submitted for someone else is submitted with their key, or labelled
    in the project name. ``body.author`` is accepted and ignored so that
    pre-accounts scripts still post; the real value is echoed back.

    The project name in the body is *bare* and is qualified with the
    caller's namespace, so existing submit scripts keep working unchanged
    and a name someone else already uses is simply the caller's own
    project of that name.
    """
    from autodft import accounts

    user = accounts.get_user_by_username(session, identity.username)
    if user is None:
        # No account row: the shared-password admin on a database that has
        # not been bootstrapped. The project name cannot be qualified, but
        # the author is still the caller and not the body.
        return body.project, identity.username

    project = accounts.get_or_create_project(session, user, body.project)
    return project.qualified_name, user.username


def _new_entrypoint(
    session, body: SubmitRequest, smiles: str,
    project_name: Optional[str] = None, author: Optional[str] = None,
) -> CalculationEntrypoint:
    """Build (but do not add) the entrypoint row for one SMILES."""
    from autodft.qm.orca.defaults import (
        DEFAULT_HEADER_CONFSEARCH,
        DEFAULT_HEADER_OPTIMIZATION,
        DEFAULT_HEADER_SINGLEPOINT,
    )

    # If only the legacy `max_conformers` was supplied, apply it as a
    # blanket override to every state — preserves the old contract.
    legacy = body.max_conformers
    n_s0  = legacy if legacy is not None else body.max_conformers_S0
    n_t1  = legacy if legacy is not None else body.max_conformers_T1
    n_ox  = legacy if legacy is not None else body.max_conformers_ox
    n_red = legacy if legacy is not None else body.max_conformers_red

    request_metadata = {
        "project_name": project_name if project_name is not None else body.project,
        "project_author": author if author is not None else body.author,
        "request_S1": False,
        "request_T1": body.request_t1,
        "request_ox": body.request_ox,
        "request_red": body.request_red,
        "request_confsearch": not body.skip_confsearch,
        "request_optimization": body.request_optimization,
        "request_singlepoint": body.request_singlepoint,
        "request_singlepoint_vertical_excitations": body.request_singlepoint_vertical_excitations,
        "request_singlepoint_nbo": False,
        "max_conformers_S0": n_s0,
        "max_conformers_T1": n_t1,
        "max_conformers_ox": n_ox,
        "max_conformers_red": n_red,
    }

    # Resolve header IDs -> raw text (IDs win over raw text). Falls back
    # to the package defaults when neither is provided.
    def _resolve(text_in: Optional[str], id_in: Optional[int], default: Optional[str]) -> Optional[str]:
        if id_in is not None:
            row = session.get(ComputationHeader, id_in)
            if row is not None:
                return row.header_text
        if text_in:
            return text_in
        return default

    return CalculationEntrypoint(
        smiles=smiles,
        request_metadata=json.dumps(request_metadata),
        priority=body.priority,
        header_confsearch=_resolve(
            body.header_confsearch, body.header_confsearch_id,
            None if body.skip_confsearch else DEFAULT_HEADER_CONFSEARCH,
        ),
        header_optimization=_resolve(
            body.header_optimization, body.header_optimization_id,
            DEFAULT_HEADER_OPTIMIZATION,
        ),
        header_singlepoint=_resolve(
            body.header_singlepoint, body.header_singlepoint_id,
            DEFAULT_HEADER_SINGLEPOINT,
        ),
    )
