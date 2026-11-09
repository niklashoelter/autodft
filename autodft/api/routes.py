"""API endpoints and HTML dashboard routes."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlmodel import case, col, func, select

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
from autodft.models.enums import SLURM_STATUS_GROUPS

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
    fingerprint = ""
    if username:
        with get_session() as session:
            user = accounts.resolve_api_key(session, password)
            if user is not None and user.username == username:
                resolved = user.username
                fingerprint = accounts.key_fingerprint(user, settings.session_secret())

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
        fingerprint,
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
    # Unknown fields ignored, but kept so _tolerance_error can name strict_convergence.
    model_config = ConfigDict(extra="allow")

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
    # Opt-in categories (autodft.categories). Only submissions that set one
    # ever take its code path.
    request_esd: bool = False
    request_esd_ht: bool = False
    request_spec_uvvis: bool = False
    request_spec_ir: bool = False
    request_spec_nmr: bool = False
    request_densities: bool = False
    request_singlepoint_nbo: bool = False
    request_sp_reactivity: bool = False
    # Settings of the categories above; recorded only when the category is.
    uvvis_nroots: int = 20
    uvvis_tda: bool = False
    nmr_nuclei: list[str] = Field(default_factory=lambda: ["H", "C", "F"])
    nmr_method: str = "singlepoint"
    # Cores and MB per core of a dh NMR job (recommended 12 x 6000).
    nmr_dh_nprocs: int = 12
    nmr_dh_maxcore: int = 6000
    # 1H/13C reference compound: "chcl3" (default) or "tms".
    nmr_reference: str = "chcl3"
    esd_tn_window_ev: float = 0.3
    esd_temperature_k: float = 298.15
    esd_ic: bool = False
    # With request_esd_ht: "all" rates or only the S1 ones ("s1"); HT jobs at the
    # singlepoint ("sp") or the optimisation ("opt") level.
    esd_ht_scope: str = "all"
    esd_ht_method: str = "sp"
    # Every optimisation fails on any imaginary mode below this (cm^-1, -300 to 0).
    imag_freq_tolerance: float = 0.0
    # {confsearch|optimization|singlepoint: {setting: value}}; checked by advanced.resolve (400).
    advanced: Any = None
    density_eldens: bool = True
    density_spindens: bool = True
    density_grid: int = 75
    density_eldens_file: str = "ElDens.cube"
    density_spindens_file: str = "SpinDens.cube"
    nbo_keywords: str = ""
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
        job_groups = {
            group: sum(job_counts.get(state, 0) for state in states)
            for group, states in SLURM_STATUS_GROUPS.items()
        }

        queued = session.exec(
            select(CalculationEntrypoint)
            .where(col(CalculationEntrypoint.time_started).is_(None))
        ).all()
        queue_length = len(visible_entrypoints(queued, session, identity))

    return {
        "molecules": molecule_count,
        "tasks": task_counts,
        "jobs": job_counts,
        "job_groups": job_groups,
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

        from autodft.extraction import results as stored

        checks = stored.connectivity_by_task(
            session, [t.id for s in mol.states for t in s.tasks if t.task_type == TaskType.optimization],
        )
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
                        **({"connectivity": checks[task.id]} if task.id in checks else {}),
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
    for name, value, enum in (("status", status, TaskStatus), ("type", type, TaskType)):
        if value and value not in {e.value for e in enum}:
            return JSONResponse(status_code=400, content={
                "detail": f"{name} must be one of: {', '.join(e.value for e in enum)}."})
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
    response: Response,
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    identity: Identity = Depends(current_identity),
):
    """Newest jobs of a SLURM state or a SLURM_STATUS_GROUPS name; X-Total-Count counts all matches."""
    with get_session() as session:
        stmt = scope_jobs(select(ComputationJob), session, identity)
        counted = scope_jobs(select(func.count()).select_from(ComputationJob), session, identity)
        if status:
            match = col(ComputationJob.slurm_status).in_(SLURM_STATUS_GROUPS.get(status, {status}))
            stmt = stmt.where(match)
            counted = counted.where(match)
        jobs = session.exec(stmt.order_by(col(ComputationJob.id).desc()).limit(limit)).all()
        total = session.exec(counted).one()

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
    response.headers["X-Total-Count"] = str(total)
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

        from autodft.analysis import densities
        from autodft.extraction import results as stored

        checks = stored.connectivity_by_task(
            session, [t.id for t in tasks if t.task_type == TaskType.optimization],
        )

    # Group helpers
    states_by_mol: dict[int, list[MoleculeState]] = {}
    for s in states:
        states_by_mol.setdefault(s.molecule_id, []).append(s)
    tasks_by_state: dict[int, list[ComputationTask]] = {}
    for t in tasks:
        tasks_by_state.setdefault(t.state_id, []).append(t)

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
            # Only successful optimisations are numbered, as in the exports.
            numbers = stored.conformer_numbers(opt_tasks)
            conformers = []
            for opt in opt_tasks:
                deps = {d.task_type: d for d in deps_by_parent.get(opt.id, [])}
                conformers.append({
                    "index": numbers.get(opt.id),
                    "optimization":                  opt.status.value,
                    "singlepoint":                   _status_of(deps.get(TaskType.singlepoint)),
                    "singlepoint_vert_ox":           _status_of(deps.get(TaskType.singlepoint_vert_ox)),
                    "singlepoint_vert_red":          _status_of(deps.get(TaskType.singlepoint_vert_red)),
                    "singlepoint_vert_spin_change":  _status_of(deps.get(TaskType.singlepoint_vert_spin_change)),
                    "singlepoint_uvvis":             _status_of(deps.get(TaskType.singlepoint_uvvis)),
                    "singlepoint_nmr":                _status_of(deps.get(TaskType.singlepoint_nmr)),
                    # Every task, not one per type: an HT rate runs as several parts.
                    "esd": _combined_status([
                        d for d in deps_by_parent.get(opt.id, [])
                        if d.task_type == TaskType.singlepoint_soc or d.task_type.value.startswith("esd_")
                    ]),
                    # Only optimisations judged since the bond check existed carry one.
                    **({"connectivity": checks[opt.id]} if opt.id in checks else {}),
                    # Task ids, for the downloads and the lazily fetched energies and densities.
                    "tasks": {
                        TaskType.optimization.value: opt.id,
                        **{t.value: d.id for t, d in deps.items() if t in densities.SP_TYPES},
                    },
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


def _project_tasks(session, name: str, task_types, *extra):
    """``(id, task_type, state_id, depends_on_task_id)`` of one project's successful tasks of *task_types*."""
    return session.exec(
        select(
            ComputationTask.id, ComputationTask.task_type, ComputationTask.state_id,
            ComputationTask.depends_on_task_id,
        )
        .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
        .join(Molecule, MoleculeState.molecule_id == Molecule.id)
        .where(
            Molecule.project_name == name,
            ComputationTask.status == TaskStatus.successful,
            col(ComputationTask.task_type).in_(list(task_types)),
            *extra,
        )
    ).all()


@router.get("/api/projects/{name}/sp-energies")
def api_project_sp_energies(
    name: str, identity: Identity = Depends(current_identity),
):
    """E (Eh) of every successful singlepoint but NMR and UV/Vis; the relaxed one also G = E + G-E(el).

    G-E(el) is the raw ORCA correction of the successful optimisation the
    singlepoint depends on (no quasi-RRHO), as the extractor uses it; no
    correction, no G. Records are read in SQL; any re-parse waits until the
    session is closed.
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    from autodft.analysis import densities
    from autodft.extraction import results as stored

    with get_session() as session:
        name = resolve_project(session, identity, name)
        sps = _project_tasks(
            session, name, densities.SP_TYPES - {TaskType.singlepoint_nmr, TaskType.singlepoint_uvvis},
        )
        opts = {t.id for t in _project_tasks(session, name, [TaskType.optimization])}
        parents = {t.depends_on_task_id for t in sps if t.task_type == TaskType.singlepoint} & opts
        energies, pending = stored.stored_values(session, {t.id: t.task_type.value for t in sps}, ("energy",))
        corrections, pending_opts = stored.stored_values(
            session, dict.fromkeys(parents, TaskType.optimization.value), ("free_energy_correction",),
        )
    energies.update(stored.resolve_pending(pending, ("energy",)))
    corrections.update(stored.resolve_pending(pending_opts, ("free_energy_correction",)))

    def finite(found: Optional[dict], key: str) -> Optional[float]:
        value = (found or {}).get(key)
        return value if isinstance(value, (int, float)) and math.isfinite(value) else None

    out = {}
    for t in sps:
        energy = finite(energies.get(t.id), "energy")
        if energy is None:
            continue
        correction = (
            finite(corrections.get(t.depends_on_task_id), "free_energy_correction")
            if t.task_type == TaskType.singlepoint else None
        )
        out[t.id] = {"energy": energy, "gibbs": energy + correction if correction is not None else None}
    return {"name": name, "tasks": out}


@router.get("/api/projects/{name}/sp-densities")
def api_project_sp_densities(
    name: str, identity: Identity = Depends(current_identity),
):
    """Which successful singlepoints have density cubes to show.

    Only where densities were requested (molecule or state metadata) and the
    job folder holds the cube, or the .gbw/.densities/.densitiesinfo that
    orca_plot rebuilds it from; SpinDens only for an open-shell job. ``build``
    says which of them orca_plot must make first.
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    from autodft.analysis import densities
    from autodft.extraction import results as stored

    with get_session() as session:
        name = resolve_project(session, identity, name)
        states = {
            st.id: (st, m) for st, m in session.exec(
                select(MoleculeState, Molecule)
                .join(Molecule, MoleculeState.molecule_id == Molecule.id)
                .where(Molecule.project_name == name)
            ).all()
            if densities.requested(m, st)
        }
        opts = {state_id: densities.options(m, st) for state_id, (st, m) in states.items()}
        wanted = [t for t in _project_tasks(session, name, densities.SP_TYPES) if t.state_id in states]
        jobs = stored.latest_successful_jobs(session, [t.id for t in wanted])
    checks = [
        (t.id, Path(jobs[t.id].job_path), opts[t.state_id],
         densities.multiplicity(t.task_type, states[t.state_id][0]) > 1)
        for t in wanted if t.id in jobs and jobs[t.id].job_path
    ]

    found = densities.availability_many(checks)
    return {
        "name": name,
        "tasks": {task_id: a for task_id, a in found.items() if a["eldens"] or a["spindens"]},
    }


@router.post("/api/projects/{name}/densities/{task_id}/{kind}")
def api_project_density_cube(
    name: str, task_id: int, kind: str, identity: Identity = Depends(current_identity),
):
    """One singlepoint's ElDens or SpinDens cube, built with orca_plot first when only its densities are kept.

    POST, since it may build: the cube is cached in the job folder, which can
    take minutes on a fine grid. 503 while no build slot is free.
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    from autodft.analysis import densities

    if kind not in densities.KINDS:
        return JSONResponse(status_code=400, content={"detail": f"kind must be one of {', '.join(densities.KINDS)}."})
    with get_session() as session:
        name = resolve_project(session, identity, name)
    return _while_reading(name, lambda: _density_cube(name, task_id, kind))


def _density_cube(name: str, task_id: int, kind: str):
    from autodft.analysis import densities
    from autodft.extraction import results as stored

    with get_session() as session:
        task = next(iter(_project_tasks(session, name, densities.SP_TYPES, ComputationTask.id == task_id)), None)
        if task is None:
            return JSONResponse(status_code=404, content={"detail": f"No successful singlepoint {task_id} in {name}."})
        state = session.get(MoleculeState, task.state_id)
        molecule = session.get(Molecule, state.molecule_id)
        if not densities.requested(molecule, state):
            return JSONResponse(status_code=404, content={"detail": "Densities were not requested for this molecule."})
        if kind == "spindens" and densities.multiplicity(task.task_type, state) == 1:
            return JSONResponse(status_code=404, content={"detail": "A closed-shell singlepoint has no spin density."})
        opts = densities.options(molecule, state)
        if not opts.wanted[kind]:
            return JSONResponse(status_code=404, content={"detail": f"density_{kind} was off at submission."})
        job = stored.latest_successful_job(session, task.id)

    job_dir = Path(job.job_path) if job is not None and job.job_path else None
    if job_dir is None or not job_dir.is_dir():
        return JSONResponse(status_code=404, content={"detail": "The job folder is no longer on disk."})
    try:
        path = densities.ensure_cube(
            job_dir, kind, opts, densities.orca_plot_binary(get_active_settings().orca.path),
        )
    except densities.CubeBusy as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    except densities.CubeError as exc:
        return JSONResponse(status_code=500, content={"detail": str(exc)})
    if path.resolve().parent != job_dir.resolve():
        return JSONResponse(status_code=404, content={"detail": "The cube is not inside the job folder."})
    return _ReadResponse(path, media_type="text/plain", filename=f"{task_id}_{kind}.cube")


class _ReadResponse(FileResponse):
    """A file from a job folder; ends its read once sent or abandoned (see :func:`_while_reading`)."""

    release: Optional[Callable[[], None]] = None

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if self.release is not None:
                self.release()


def _while_reading(name: str, read: Callable[[], Response]) -> Response:
    """Run *read* as a read the project's archive waits for; 409 while the archive runs."""
    from autodft.api import project_jobs

    try:
        project_jobs.begin_read(name)
    except project_jobs.JobInProgress as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    try:
        response = read()
    except BaseException:
        project_jobs.end_read(name)
        raise
    if isinstance(response, _ReadResponse):
        response.release = lambda: project_jobs.end_read(name)
    else:
        project_jobs.end_read(name)
    return response


# What a task offers for download: its output, and an optimisation also its final geometry.
_TASK_FILES = {"out": "output.out", "xyz": "input.xyz"}


@router.get("/api/projects/{name}/tasks/{task_id}/files/{kind}")
def api_project_task_file(
    name: str, task_id: int, kind: str, identity: Identity = Depends(current_identity),
):
    """The .out of a successful optimisation or singlepoint, or the optimised .xyz, from its latest successful job.

    Named ``mol<id>_<state>_conf<k>_<stem>``, as in the export tree (``<state>__state<id>`` when shared).
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    if kind not in _TASK_FILES:
        return JSONResponse(status_code=400, content={"detail": f"kind must be one of {', '.join(_TASK_FILES)}."})
    with get_session() as session:
        name = resolve_project(session, identity, name)
    return _while_reading(name, lambda: _task_file(name, task_id, kind))


def _task_file(name: str, task_id: int, kind: str):
    from autodft.analysis import densities
    from autodft.extraction import results as stored
    from autodft.extraction.extractor import _TREE_STEMS

    with get_session() as session:
        task = next(iter(_project_tasks(
            session, name, densities.SP_TYPES | {TaskType.optimization}, ComputationTask.id == task_id,
        )), None)
        if task is None:
            return JSONResponse(
                status_code=404, content={"detail": f"No successful optimisation or singlepoint {task_id} in {name}."},
            )
        if kind == "xyz" and task.task_type != TaskType.optimization:
            return JSONResponse(status_code=404, content={"detail": "Only an optimisation has an output geometry."})
        state = session.get(MoleculeState, task.state_id)
        shared = stored.shared_descriptions(session.exec(
            select(MoleculeState).where(MoleculeState.molecule_id == state.molecule_id)
        ).all())
        numbers = stored.conformer_numbers(session.exec(
            select(ComputationTask).where(
                ComputationTask.state_id == task.state_id, ComputationTask.task_type == TaskType.optimization,
            )
        ).all())
        job = stored.latest_successful_job(session, task.id)

    opt_id = task.id if task.task_type == TaskType.optimization else task.depends_on_task_id
    conf = numbers.get(opt_id, 0)
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", state.description or "state")
    if (state.molecule_id, state.description) in shared:
        label = f"{label}__state{state.id}"
    filename = f"mol{state.molecule_id}_{label}_conf{conf}_{_TREE_STEMS[task.task_type.value]}.{kind}"
    job_dir = Path(job.job_path) if job is not None and job.job_path else None
    path = job_dir / _TASK_FILES[kind] if job_dir is not None else None
    if path is None or not path.is_file() or path.resolve().parent != job_dir.resolve():
        return JSONResponse(status_code=404, content={"detail": f"The {kind} file is no longer on disk."})
    return _ReadResponse(path, media_type="text/plain", filename=filename)


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


@router.get("/api/projects/{name}/methods")
def api_project_methods(
    name: str, identity: Identity = Depends(current_identity),
):
    """What a project ran with: headers, settings and their sources, cutoffs,
    the ORCA input of every job type, and the current pipeline config."""
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    from autodft.analysis.methods import project_methods

    with get_session() as session:
        name = resolve_project(session, identity, name)
        return project_methods(session, get_active_settings(), name)


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
    except project_jobs.ProjectGone as exc:
        return JSONResponse(status_code=404, content={"detail": str(exc)})
    except Exception as exc:
        logging.getLogger(__name__).exception("Could not start XLSX export for project %r", name)
        return JSONResponse(
            status_code=500,
            content={"detail": f"XLSX export failed to start: {type(exc).__name__}: {exc}"},
        )


@router.get("/api/projects/{name}/photophysics")
def api_project_photophysics(
    name: str, molecule_id: Optional[int] = None,
    identity: Identity = Depends(current_identity),
):
    """UV/Vis, IR and NMR for every molecule submitted with those categories.

    Without ``molecule_id``: one summary per molecule (counts, weighting,
    strongest band). With it: that molecule's sticks and Boltzmann weights
    (298.15 K); the dashboard broadens them.
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    with get_session() as session:
        name = resolve_project(session, identity, name)
    from autodft.analysis import spectroscopy

    return spectroscopy.analyze_spectra(name, molecule_id=molecule_id)


def _status_of(task: Optional[ComputationTask]) -> Optional[str]:
    return task.status.value if task is not None else None


def _combined_status(tasks: list[ComputationTask]) -> Optional[str]:
    """One status for several tasks: failed, else pending, else created, else successful."""
    if not tasks:
        return None
    statuses = {t.status for t in tasks}
    for status in (TaskStatus.failed, TaskStatus.pending, TaskStatus.created):
        if status in statuses:
            return status.value
    return TaskStatus.successful.value


_STATE_ORDER = {"S0": 0, "S1": 1, "T1": 2, "T1_esd": 3, "ox": 4, "red": 5}


def _state_sort_key(s: MoleculeState) -> tuple[int, str]:
    return (_STATE_ORDER.get(s.description, 99), s.description)


class ArchiveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The qualified project name, typed back as for a wipe.
    confirm: str = ""
    all_conformers: bool = False
    keep_gbw: bool = False


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
    """Start "Export and Delete" of one project as a background job.

    Writes the export tree into ``<export_data>/<name>/`` (summary files
    and ``molecules/``; see ``PipelineExtractor.export_tree``), then deletes
    every ``<comp_data>/mol_<id>/`` belonging to the project and flips
    ``Molecule.archived = True`` on every row.

    The work runs on a background thread so it survives the browser closing;
    poll ``GET /api/projects/{name}/jobs`` for progress. Returns **202** with
    the created job.

    Refuses with 4xx when:
    * ``confirm`` is not the project name exactly (400)
    * the project is ``"default"`` (protected — never archivable)
    * the project doesn't exist (404)
    * the project is already archived (409)
    * the project already has a background job in flight (409)

    The job fails, touching nothing, while an NMR molecule's reference is
    still pending -- the frozen shift would stay null forever.
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    with get_session() as session:
        name = resolve_project(session, identity, name)
    from autodft.api import project_jobs

    if body.confirm != name:
        return _confirmation_error(name, body.confirm)

    if _is_protected(name):
        return JSONResponse(
            status_code=409,
            content={"detail": f"Project {name!r} is protected and cannot be archived."},
        )

    try:
        settings = get_active_settings()

        with get_session() as session:
            project_jobs.assert_no_active_job(session, name)
            state = _project_is_archived(session, name)
            if state is None:
                return JSONResponse(status_code=404, content={"detail": f"Project {name!r} has no molecules"})
            if state is True:
                return JSONResponse(status_code=409, content={"detail": f"Project {name!r} is already archived."})

        job = project_jobs.start_job(
            owner_id=identity.user_id,
            qualified_name=name,
            kind=ProjectJobKind.archive,
            params={"all_conformers": body.all_conformers, "keep_gbw": body.keep_gbw},
            settings=settings,
        )
        return JSONResponse(status_code=202, content={"project": name, "job": job})
    except project_jobs.JobInProgress as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    except project_jobs.ProjectGone as exc:
        return JSONResponse(status_code=404, content={"detail": str(exc)})
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except Exception as exc:
        logging.getLogger(__name__).exception("Could not start archive for project %r", name)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Archive failed to start: {type(exc).__name__}: {exc}"},
        )


# Export format -> the background-job kind that produces it. `xlsx` is the
# state-analysis workbook; `photophysics` and `reactivity` are workbook + JSON;
# the rest come off the PipelineExtractor.
_EXPORT_KINDS = {
    "csv": ProjectJobKind.export_csv,
    "json": ProjectJobKind.export_json,
    "files": ProjectJobKind.export_files,
    "xlsx": ProjectJobKind.export_xlsx,
    "photophysics": ProjectJobKind.export_photophysics,
    "reactivity": ProjectJobKind.export_reactivity,
}


@router.post("/api/projects/{name}/export")
def api_project_export(
    name: str,
    format: str = Query("csv"),
    all_conformers: bool = Query(False),
    keep_gbw: bool = Query(False),
    identity: Identity = Depends(current_identity),
):
    """Start an export for one project as a background job.

    Writes into ``<export_data>/<name>/``. Format: ``files`` -> the export
    tree ("Export Project": summary files plus ``molecules/``, replacing a
    previous one; ``keep_gbw`` adds singlepoint ``.gbw`` files); ``csv`` ->
    ``<name>.csv``; ``json`` -> ``<name>.json``; ``xlsx`` -> the
    state-analysis workbook.

    The work runs on a background thread so it survives the browser closing;
    poll ``GET /api/projects/{name}/jobs`` and download the result via
    ``GET /api/jobs/{id}/download``. Returns **202** with the created job.

    ``csv`` / ``json`` / ``files`` need the on-disk ORCA outputs, so an
    archived project is refused (409); ``xlsx`` is built from the frozen
    archive CSV and is allowed. ``photophysics`` -> the photophysics workbook
    plus JSON, allowed for archived projects (served from the frozen
    payload). ``reactivity`` -> the reactivity workbook plus JSON, refused
    for archived projects like ``csv``. A project with a job already in
    flight is refused (409).
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
            project_jobs.assert_no_active_job(session, name)
            state = _project_is_archived(session, name)
            if state is None:
                return JSONResponse(status_code=404, content={"detail": f"Project {name!r} has no molecules"})
            if state is True and kind not in (ProjectJobKind.export_xlsx, ProjectJobKind.export_photophysics):
                return JSONResponse(
                    status_code=409,
                    content={"detail": f"Project {name!r} is archived — source files are no longer on disk."},
                )

        job = project_jobs.start_job(
            owner_id=identity.user_id,
            qualified_name=name,
            kind=kind,
            params={"all_conformers": all_conformers,
                    **({"keep_gbw": keep_gbw} if kind == ProjectJobKind.export_files else {})},
            settings=settings,
        )
        return JSONResponse(status_code=202, content={"project": name, "job": job})
    except project_jobs.JobInProgress as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    except project_jobs.ProjectGone as exc:
        return JSONResponse(status_code=404, content={"detail": str(exc)})
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
    from autodft.qm.orca.advanced import is_legacy_header

    defaults = [
        {
            "id": "default_confsearch",
            "label": "GOAT g-xTB (default)",
            "description": "GOAT conformer ensemble at g-xTB; MAXEN 10, ENDIFF 0.2, RMSD 0.15.",
            "kind": "confsearch",
            "text": DEFAULT_HEADER_CONFSEARCH,
        },
        {
            "id": "default_optimization",
            "label": "wB97X-D3 / def2-SVP (default)",
            "description": "Functional and basis wB97X-D3/def2-SVP; other settings are Advanced.",
            "kind": "optimization",
            "text": DEFAULT_HEADER_OPTIMIZATION,
        },
        {
            "id": "default_singlepoint",
            "label": "wB97X-D3 / def2-TZVP (default)",
            "description": "Functional and basis wB97X-D3/def2-TZVP; other settings are Advanced.",
            "kind": "singlepoint",
            "text": DEFAULT_HEADER_SINGLEPOINT,
        },
    ]
    for d in defaults:
        d["legacy"] = is_legacy_header(d["text"], d["kind"])

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
                "legacy": is_legacy_header(h.header_text, h.kind),
            }
            for h in headers
        ]

    return {"defaults": defaults, "custom": db_headers}


def _caller_id(session, identity: Identity):
    """The caller's user row id, or None when accounts are not set up."""
    from autodft import accounts

    user = accounts.get_user_by_username(session, identity.username)
    return None if user is None else user.id


@router.post("/api/headers")
def api_create_header(
    body: HeaderCreate, identity: Identity = Depends(current_identity),
):
    """Create a new custom computation header. Admin only; everyone uses it.

    Header text goes into the ORCA input verbatim.
    """
    require_admin(identity)
    if body.kind and body.kind not in {"confsearch", "optimization", "singlepoint"}:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid kind {body.kind!r}"},
        )
    with get_session() as session:
        header = ComputationHeader(
            header_text=body.header_text,
            description=body.description,
            kind=body.kind or None,
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
    """Update fields on an existing header. Admin only.

    A text or kind change to a header any state or task references is
    saved as a new header owned by the editor; the used one stays as it ran.
    """
    require_admin(identity)
    if body.kind is not None and body.kind not in {"", "confsearch", "optimization", "singlepoint"}:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid kind {body.kind!r}"},
        )
    with get_session() as session:
        header = session.get(ComputationHeader, header_id)
        if header is None:
            return JSONResponse(status_code=404, content={"detail": "Header not found"})
        changes_method = ((body.header_text is not None and body.header_text != header.header_text)
                          or (body.kind is not None and (body.kind or None) != header.kind))
        copied = changes_method and _header_in_use(session, header_id)
        if copied:
            # Unvalidated, as a create is.
            header = ComputationHeader(
                header_text=header.header_text, description=header.description, kind=header.kind,
                owner_id=_caller_id(session, identity),
            )
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
            "copied": copied,
        }


def _header_in_use(session, header_id: int) -> bool:
    """Whether any state or task (and so any job) references header *header_id*."""
    state = session.exec(select(MoleculeState.id).where(
        (MoleculeState.confsearch_header_id == header_id)
        | (MoleculeState.optimization_header_id == header_id)
        | (MoleculeState.singlepoint_header_id == header_id)
    ).limit(1)).first()
    task = session.exec(select(ComputationTask.id).where(ComputationTask.header_id == header_id).limit(1)).first()
    return state is not None or task is not None


@router.delete("/api/headers/{header_id}")
def api_delete_header(
    header_id: int, identity: Identity = Depends(current_identity),
):
    """Soft-delete a custom header. Admin only.

    Refused only when an *in-flight* task (status ``created`` or
    ``pending``) still references it directly or through its state.
    Headers referenced by already-finished tasks (``successful`` /
    ``failed``) can be removed: the header row is kept in the table
    with ``deleted=True`` so historical FK pointers stay valid, but
    it disappears from listings and submission dropdowns.
    """
    from autodft.models.state import MoleculeState

    require_admin(identity)
    with get_session() as session:
        header = session.get(ComputationHeader, header_id)
        if header is None:
            return JSONResponse(status_code=404, content={"detail": "Header not found"})

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
    model_config = ConfigDict(extra="forbid")

    confirm: str
    # Project wipes keep <export_data>/<project>/ unless asked.
    delete_exports: bool = False


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

    settings = get_active_settings()
    try:
        with admin_ops.exclusive(f"wipe of project {name!r}"), project_jobs.wiping(name):
            with get_session() as session:
                return admin_ops.wipe_project(
                    session, name, settings.comp_data_path, settings.export_data_path,
                    delete_exports=body.delete_exports,
                    scheduler=_admin_scheduler(settings),
                )
    except (admin_ops.WipeInProgress, project_jobs.JobInProgress) as exc:
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
    from autodft.api import admin_ops, project_jobs

    settings = get_active_settings()
    try:
        with admin_ops.exclusive(f"wipe of molecule {molecule_id}"):
            with get_session() as session:
                mol = require_molecule(session, identity, molecule_id)
                if body.confirm != mol.smiles:
                    return _confirmation_error(mol.smiles, body.confirm)
                project = mol.project_name
            with project_jobs.wiping(project), get_session() as session:
                return admin_ops.wipe_molecule(
                    session, molecule_id, settings.comp_data_path,
                    scheduler=_admin_scheduler(settings),
                )
    except (admin_ops.WipeInProgress, project_jobs.JobInProgress) as exc:
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
def api_cluster_status(identity: Identity = Depends(current_identity)):
    """Read-only pipeline health, for everyone who is signed in.

    Without this a user cannot tell "my jobs are stuck" from "the pipeline
    is halted" without asking an administrator. It exposes no one else's
    data: a queue depth and a breaker flag, the caller's own or, for admin, any owner's.
    """
    from autodft.engine import circuit_breaker, submission_hold

    settings = get_active_settings()
    if identity.is_admin:
        tripped = bool(circuit_breaker.tripped_owners(settings.data_path))
    else:
        tripped = circuit_breaker.read_state(settings.data_path, identity.username) is not None
    held = submission_hold.read_state(settings.data_path)
    with get_session() as session:
        waiting = session.exec(
            select(func.count())
            .select_from(CalculationEntrypoint)
            .where(col(CalculationEntrypoint.time_started).is_(None))
        ).one()
    return {
        "breaker_tripped": tripped,
        "submissions_held": held is not None,
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
    """Move a project to another owner, with its job rows and export directory."""
    require_admin(identity)
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    from autodft import accounts
    from autodft.api import project_jobs

    # Lock before session, as start_job does: no job starts between the check and the move.
    with project_jobs.creation_lock(), get_session() as session:
        name = resolve_project(session, identity, name)
        from autodft.engine import nmr_references

        if nmr_references.is_reference_project(name):
            return JSONResponse(
                status_code=409,
                content={"detail": "The NMR reference project is managed by the pipeline and cannot be reassigned."},
            )
        new_owner = accounts.get_user_by_username(session, body.owner)
        if new_owner is None:
            return JSONResponse(
                status_code=404, content={"detail": f"No user named {body.owner!r}."},
            )
        try:
            project_jobs.assert_no_active_job(session, name)
            project = accounts.reassign_project(
                session, name, new_owner, export_root=get_active_settings().export_data_path,
            )
        except (accounts.AccountError, project_jobs.JobInProgress) as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        return {"project": project.qualified_name, "owner": new_owner.username}


@router.get("/api/admin/circuit-breaker")
def api_circuit_breaker_status():
    """Each owner's failure breaker: the tripped ones and those with recent failures."""
    from autodft.engine import circuit_breaker

    settings = get_active_settings()
    window = settings.pipeline.failure_breaker_window
    tripped = circuit_breaker.tripped_owners(settings.data_path)
    owners = []
    with get_session() as session:
        for owner in sorted(circuit_breaker.owners(session) | set(tripped)):
            ratio, failed, judged = circuit_breaker.recent_failure_ratio(
                session, window, owner, since=circuit_breaker.reset_time(settings.data_path, owner),
            )
            if failed or owner in tripped:
                owners.append({
                    "owner": owner,
                    "tripped": owner in tripped,
                    "state": tripped.get(owner),
                    "recent_failure_ratio": round(ratio, 4),
                    "recent_failed": failed,
                    "recent_judged": judged,
                })
    return {
        "tripped": bool(tripped),
        "owners": owners,
        "threshold": settings.pipeline.failure_breaker_ratio,
        "window": window,
    }


class CircuitBreakerResetRequest(BaseModel):
    owner: Optional[str] = None


@router.post("/api/admin/circuit-breaker/reset")
def api_circuit_breaker_reset(
    body: Optional[CircuitBreakerResetRequest] = None,
    identity: Identity = Depends(current_identity),
):
    """Clear one owner's breaker, or every owner's, so their jobs are created and submitted again.

    Deliberately manual: once submission stops no new tasks are judged, so
    the failure ratio cannot recover on its own. Fix the cause first. A reset
    starts the owner's failure count afresh.
    """
    require_admin(identity)
    from autodft.engine import circuit_breaker

    settings = get_active_settings()
    try:
        with get_session() as session:
            cleared = circuit_breaker.reset(session, settings.data_path, body.owner if body else None)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    return {"reset": cleared, "tripped": sorted(circuit_breaker.tripped_owners(settings.data_path))}


class SubmissionHoldRequest(BaseModel):
    reason: str = Field(default="", max_length=500)


@router.get("/api/admin/submission-hold")
def api_submission_hold_status(identity: Identity = Depends(current_identity)):
    """Whether an admin has paused sbatch submission."""
    require_admin(identity)
    from autodft.engine import submission_hold

    state = submission_hold.read_state(get_active_settings().data_path)
    return {"held": state is not None, "state": state}


@router.post("/api/admin/submission-hold")
def api_submission_hold_set(
    body: SubmissionHoldRequest, identity: Identity = Depends(current_identity),
):
    """Stop submitting jobs; results are still processed and followups prepared."""
    require_admin(identity)
    from autodft.engine import submission_hold

    state = submission_hold.hold(
        get_active_settings().data_path, identity.username, body.reason.strip(),
    )
    return {"held": True, "state": state}


@router.post("/api/admin/submission-hold/release")
def api_submission_hold_release(identity: Identity = Depends(current_identity)):
    """Resume submission on the next tick."""
    require_admin(identity)
    from autodft.engine import submission_hold

    was_held = submission_hold.release(get_active_settings().data_path)
    return {"released": was_held, "held": False}


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

    settings = get_active_settings()
    try:
        # A reset empties every project: refused while any has an archive/export running.
        with admin_ops.exclusive("database reset"), project_jobs.wiping(None):
            with get_session() as session:
                return admin_ops.reset_database(
                    session, settings.comp_data_path, settings.export_data_path,
                    delete_files=body.delete_files, keep_headers=body.keep_headers,
                    scheduler=_admin_scheduler(settings),
                )
    except (admin_ops.WipeInProgress, project_jobs.JobInProgress, ValueError) as exc:
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


class ValidateSmilesBatchRequest(BaseModel):
    smiles_list: list[str] = Field(max_length=100)


@router.post("/api/validate-smiles-batch")
def api_validate_smiles_batch(body: ValidateSmilesBatchRequest):
    """``/api/validate-smiles`` for up to 100 SMILES, answered in order; a line over 512 characters is invalid."""
    from autodft.engine.entrypoint_processor import invalid_smiles, validate_smiles

    return [validate_smiles(s) if len(s) <= 512 else invalid_smiles("SMILES too long (>512).")
            for s in body.smiles_list]


@router.get("/api/advanced-defaults")
def api_advanced_defaults(identity: Identity = Depends(current_identity)):
    """The config defaults and choices of the Advanced settings, for the submit form."""
    from autodft.qm.orca import advanced

    return {"advanced": advanced.defaults(get_active_settings()), "choices": advanced.CHOICES,
            "imag_freq_tolerance": 0.0}


def _category_flags(body: SubmitRequest) -> dict:
    """The category part of *body*, keyed as in ``request_metadata``."""
    from autodft import categories

    return {
        categories.ESD: body.request_esd,
        categories.ESD_HT: body.request_esd_ht,
        categories.UVVIS: body.request_spec_uvvis,
        categories.IR: body.request_spec_ir,
        categories.NMR: body.request_spec_nmr,
        categories.DENSITIES: body.request_densities,
        categories.NBO: body.request_singlepoint_nbo,
        categories.REACTIVITY: body.request_sp_reactivity,
        "uvvis_nroots": body.uvvis_nroots,
        "uvvis_tda": body.uvvis_tda,
        "nmr_nuclei": body.nmr_nuclei,
        "nmr_method": body.nmr_method,
        "nmr_dh_nprocs": body.nmr_dh_nprocs,
        "nmr_dh_maxcore": body.nmr_dh_maxcore,
        "nmr_reference": body.nmr_reference,
        "esd_tn_window_ev": body.esd_tn_window_ev,
        "esd_temperature_k": body.esd_temperature_k,
        "esd_ic": body.esd_ic,
        "esd_ht_scope": body.esd_ht_scope,
        "esd_ht_method": body.esd_ht_method,
        "density_eldens": body.density_eldens,
        "density_spindens": body.density_spindens,
        "density_grid": body.density_grid,
        "density_eldens_file": body.density_eldens_file,
        "density_spindens_file": body.density_spindens_file,
        "nbo_keywords": body.nbo_keywords,
        "request_optimization": body.request_optimization,
        "request_singlepoint": body.request_singlepoint,
    }


def _resolved_headers(session, body: SubmitRequest) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """(confsearch, optimization, singlepoint) header text for *body*.

    A stored header's id wins over raw text, which wins over the default.
    """
    from autodft.qm.orca.defaults import (
        DEFAULT_HEADER_CONFSEARCH,
        DEFAULT_HEADER_OPTIMIZATION,
        DEFAULT_HEADER_SINGLEPOINT,
    )

    def _resolve(text_in: Optional[str], id_in: Optional[int], default: Optional[str]) -> Optional[str]:
        if id_in is not None:
            row = session.get(ComputationHeader, id_in)
            if row is not None:
                return row.header_text
        if text_in:
            return text_in
        return default

    return (
        _resolve(body.header_confsearch, body.header_confsearch_id,
                 None if body.skip_confsearch else DEFAULT_HEADER_CONFSEARCH),
        _resolve(body.header_optimization, body.header_optimization_id,
                 DEFAULT_HEADER_OPTIMIZATION),
        _resolve(body.header_singlepoint, body.header_singlepoint_id,
                 DEFAULT_HEADER_SINGLEPOINT),
    )


def _header_access_error(session, identity: Identity, body: SubmitRequest) -> Optional[JSONResponse]:
    """For a non-admin *body*: 400 for a deleted header (by id or text), 403 for header text not stored, else None."""
    from autodft.qm.orca.defaults import (
        DEFAULT_HEADER_CONFSEARCH,
        DEFAULT_HEADER_OPTIMIZATION,
        DEFAULT_HEADER_SINGLEPOINT,
    )

    if identity.is_admin:
        return None
    defaults = {DEFAULT_HEADER_CONFSEARCH, DEFAULT_HEADER_OPTIMIZATION, DEFAULT_HEADER_SINGLEPOINT}
    for slot in ("confsearch", "optimization", "singlepoint"):
        text = getattr(body, f"header_{slot}")
        header_id = getattr(body, f"header_{slot}_id")
        row = session.get(ComputationHeader, header_id) if header_id is not None else None
        if row is not None:
            if row.deleted:
                return JSONResponse(status_code=400, content={
                    "detail": f"header_{slot}_id: header #{row.id} was deleted; pick another."})
            continue
        if not text:
            continue
        rows = session.exec(select(ComputationHeader).where(ComputationHeader.header_text == text)).all()
        if any(not r.deleted for r in rows) or text in defaults:
            continue
        if rows:
            return JSONResponse(status_code=400, content={
                "detail": f"header_{slot}: header #{rows[0].id} with this text was deleted; pick another."})
        return JSONResponse(status_code=403, content={"detail": (
            f"header_{slot}: only the admin account adds header text; "
            f"pick a saved header (header_{slot}_id)."
        )})
    return None


def _priority_error(identity: Identity, body: SubmitRequest) -> Optional[JSONResponse]:
    """A 400 for a non-admin sending a non-default priority, or an admin priority outside 1..1000, else None."""
    if identity.is_admin:
        if not 1 <= body.priority <= 1000:
            return JSONResponse(status_code=400, content={"detail": "priority must be between 1 and 1000."})
    elif body.priority != SubmitRequest.model_fields["priority"].default:
        return JSONResponse(status_code=400, content={"detail": "Only admin can set priority."})
    return None


def _tolerance_error(body: SubmitRequest) -> Optional[JSONResponse]:
    """A 400 when *body*'s imag_freq_tolerance is out of range or it sends strict_convergence, else None."""
    from autodft.qm.orca.parser import IMAG_FREQ_TOLERANCE_MIN

    if "strict_convergence" in (body.model_extra or {}):
        return JSONResponse(status_code=400, content={"detail": (
            "strict_convergence was replaced by imag_freq_tolerance (cm-1, "
            f"{IMAG_FREQ_TOLERANCE_MIN:g} to 0; 0 fails any imaginary mode)."
        )})
    if IMAG_FREQ_TOLERANCE_MIN <= body.imag_freq_tolerance <= 0:
        return None
    return JSONResponse(status_code=400, content={"detail": (
        f"imag_freq_tolerance must be between {IMAG_FREQ_TOLERANCE_MIN:g} and 0 cm-1 "
        f"(got {body.imag_freq_tolerance:g})."
    )})


def _conformers_error(body: SubmitRequest) -> Optional[JSONResponse]:
    """A 400 for a negative conformer limit (0 asks for none), else None."""
    for name in ("max_conformers_S0", "max_conformers_T1", "max_conformers_ox", "max_conformers_red",
                 "max_conformers"):
        value = getattr(body, name)
        if value is not None and value < 0:
            return JSONResponse(status_code=400, content={"detail": f"{name} must be 0 or more (got {value})."})
    return None


def _submission_advanced(session, body: SubmitRequest, project: str, headers) -> tuple[Optional[dict], Optional[str]]:
    """*body*'s resolved ``advanced`` (None in legacy mode), or ``(None, error)``.

    Without ``advanced``, a submission to a project whose legacy S0 states use
    exactly *headers* stays legacy; every other one gets the config defaults.
    """
    from autodft.engine.entrypoint_processor import legacy_submission
    from autodft.qm.orca import advanced

    if body.advanced is None and legacy_submission(session, project, headers):
        return None, None
    return advanced.for_submission(body.advanced, headers, get_active_settings())


def _target_project(session, identity: Identity, body: SubmitRequest) -> str:
    """The qualified project *body* goes to, without creating it (see ``_submission_owner``)."""
    from autodft import accounts

    user = accounts.get_user_by_username(session, identity.username)
    return accounts.qualify(user.username, _bare_project(body)) if user is not None else body.project


def _bare_project(body: SubmitRequest) -> str:
    """*body*'s project name, or a 400 when it is not bare (an account qualifies it with its own name)."""
    from autodft import accounts

    try:
        return accounts.validate_project_name(body.project)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


def _reject_reason(body: SubmitRequest, smiles: str, headers,
                   advanced: Optional[dict] = None) -> tuple[Optional[str], dict]:
    """Why *smiles* cannot be queued under *body*'s options (None if it can),
    with the validation result."""
    from autodft import categories
    from autodft.engine.entrypoint_processor import validate_smiles

    check = validate_smiles(smiles)
    if not check["valid"]:
        return check["error"] or "Invalid SMILES.", check

    # A diradical is only ever calculated as a triplet -- see the matching
    # rule in the entrypoint processor. Refuse here so the caller gets a 400
    # instead of a molecule that fails later in the worker.
    if check["diradical"]:
        extra = [name for name, on in (
            ("request_t1", body.request_t1),
            ("request_ox", body.request_ox),
            ("request_red", body.request_red),
        ) if on]
        if extra:
            return (
                f"{smiles!r} is a diradical and is only calculated in the "
                f"triplet state. Resubmit without {', '.join(extra)}.",
                check,
            )

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

    return categories.rejection(check, _category_flags(body), headers[1], headers[2], advanced), check


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
    from autodft import categories
    from autodft.api import project_jobs

    bad = _priority_error(identity, body) or _tolerance_error(body) or _conformers_error(body)
    if bad is not None:
        return bad
    with get_session() as session:
        forbidden = _header_access_error(session, identity, body)
        if forbidden is not None:
            return forbidden
        headers = _resolved_headers(session, body)
        advanced, detail = _submission_advanced(session, body, _target_project(session, identity, body), headers)
        if detail is not None:
            return JSONResponse(status_code=400, content={"detail": detail})
        detail, check = _reject_reason(body, body.smiles, headers, advanced)
        if detail is not None:
            return JSONResponse(
                status_code=400, content={"detail": detail, "validation": check},
            )
        flags = _category_flags(body)
        if categories.NBO_NEEDED & categories.requested(flags):
            unavailable = categories.nbo_unavailable(get_active_settings())
            if unavailable:
                return JSONResponse(
                    status_code=400, content={"detail": unavailable, "validation": check},
                )
        project_name, author = _submission_owner(session, identity, body)
        conflict = categories.existing_conflict(
            session, project_name, check["canonical"], categories.requested(flags),
            {**categories.options(flags), categories.ESD_HT: bool(flags.get(categories.ESD_HT))}, advanced, headers,
        )
        if conflict:
            return JSONResponse(
                status_code=400, content={"detail": conflict, "validation": check},
            )
        try:
            project_jobs.assert_no_active_job(session, project_name)
        except project_jobs.JobInProgress as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        entry = _new_entrypoint(session, body, body.smiles, project_name, author, headers, advanced)
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


def _extend_scope(session, name: str) -> dict:
    """What a project's molecules already have, for Extend: headers, Advanced settings, categories, states, options."""
    from autodft import categories
    from autodft.extraction.results import imaginary_cutoff
    from autodft.qm.orca.advanced import stored_method_settings

    molecules = session.exec(select(Molecule).where(Molecule.project_name == name)).all()
    live = [m for m in molecules if not m.archived]
    ids = [m.id for m in live]
    states = session.exec(select(MoleculeState).where(col(MoleculeState.molecule_id).in_(ids))).all() if ids else []
    s0 = [st for st in states if st.description == "S0"]
    header_sets = {(st.confsearch_header_id, st.optimization_header_id, st.singlepoint_header_id) for st in s0}
    headers = None
    if len(header_sets) == 1:
        rows = [session.get(ComputationHeader, hid) if hid is not None else None for hid in next(iter(header_sets))]
        headers = dict(zip(("confsearch", "optimization", "singlepoint"),
                           ({"id": r.id, "text": r.header_text} if r else None for r in rows)))
    tolerances = {imaginary_cutoff(st) for st in s0}
    have: dict[str, int] = {}
    options: dict = {}
    first: dict = {}
    method_sets = []
    for st in s0:
        metadata = json.loads(st.metadata_json) if st.metadata_json else {}
        first = first or metadata
        settings = stored_method_settings(metadata)
        if settings not in method_sets:
            method_sets.append(settings)
        for key in categories.requested(metadata):
            have[key] = have.get(key, 0) + 1
            if key not in options:
                options[key] = {name: metadata.get(name, default)
                                for name, default in categories.OPTIONS.get(key, {}).items()}
                if key == categories.ESD:
                    options[key][categories.ESD_HT] = bool(metadata.get(categories.ESD_HT))
    counts: dict[str, int] = {}
    conformers: dict[str, int] = {"S0": first.get("max_conformers_S0", 1)}
    for st in states:
        metadata = json.loads(st.metadata_json) if st.metadata_json else {}
        # ESD's own triplet "T1" (ESD without request_T1) is the molecule's T1 as well.
        if st.description in ("T1", "ox", "red"):
            counts[st.description] = counts.get(st.description, 0) + 1
            if not metadata.get("esd_role"):
                conformers.setdefault(st.description, metadata.get(f"max_conformers_{st.description}", 1))
    n = len(live)

    def share(count: int) -> str:
        return "all" if n and count == n else "some" if count else "none"

    reason = None
    if not live:
        reason = "The project has no molecules left to extend (all archived)." if molecules else "The project is empty."
    elif headers is None:
        reason = "The project's molecules use more than one header set; extend is only possible with one."
    elif len(method_sets) > 1:
        reason = ("The project's molecules use more than one set of Advanced method settings; "
                  "extend is only possible with one.")
    return {
        "project": name,
        "molecules": n,
        "archived": len(molecules) - n,
        "extendable": reason is None,
        "reason": reason,
        "headers": headers,
        # Each extension takes its own S0's; None for a legacy project.
        "advanced": first.get("advanced"),
        # S0's cutoff, which each extension keeps; None when the molecules differ.
        "imag_freq_tolerance": next(iter(tolerances)) if len(tolerances) == 1 else None,
        "categories": {key: share(have.get(key, 0)) for key in categories.CATEGORIES},
        "options": options,
        "states": {d: share(counts.get(d, 0)) for d in ("T1", "ox", "red")},
        "max_conformers": conformers,
        "settings": {
            "request_singlepoint_vertical_excitations": first.get("request_singlepoint_vertical_excitations", True),
            "request_confsearch": bool(headers and headers["confsearch"]),
        },
        "submit_only": list(categories.SUBMIT_ONLY),
        "smiles": [m.smiles for m in live],
    }


@router.get("/api/projects/{name}/extend-info")
def api_extend_info(name: str, identity: Identity = Depends(current_identity)):
    """What the project's molecules already have, so Extend can offer only the rest."""
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    with get_session() as session:
        name = resolve_project(session, identity, name)
        return _extend_scope(session, name)


@router.post("/api/projects/{name}/extend")
def api_extend(name: str, body: SubmitBatchRequest, identity: Identity = Depends(current_identity)):
    """Queue an extension of the project's molecules (all, or ``smiles_list``) with *body*'s categories and states.

    Runs under the project's own headers; each molecule gains only what it lacks.
    """
    from autodft import categories
    from autodft.api import project_jobs
    from autodft.engine.entrypoint_processor import EXTEND_DELETED_HEADER

    bad = (_reject_bad_project(name) or _priority_error(identity, body) or _tolerance_error(body)
           or _conformers_error(body))
    if bad is not None:
        return bad
    if body.advanced is not None:
        return JSONResponse(status_code=400, content={
            "detail": "An extension runs at the project's own Advanced settings; send no advanced."})
    with get_session() as session:
        name = resolve_project(session, identity, name)
        scope = _extend_scope(session, name)
        if not scope["extendable"]:
            return JSONResponse(status_code=400, content={"detail": scope["reason"]})
        if not identity.is_admin and any(
            h and session.get(ComputationHeader, h["id"]).deleted for h in scope["headers"].values()
        ):
            return JSONResponse(status_code=400, content={"detail": EXTEND_DELETED_HEADER})
        try:
            project_jobs.assert_no_active_job(session, name)
        except project_jobs.JobInProgress as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        headers = tuple(h["text"] if h else None for h in (scope["headers"][k] for k in
                                                            ("confsearch", "optimization", "singlepoint")))
        advanced = scope["advanced"]
        wanted = categories.requested(_category_flags(body))
        states = {d for d, key in _EXTEND_STATES if getattr(body, key.lower())}
        known = set(scope["smiles"])
        accepted, rejected, skipped = [], [], []
        queued_now: set = set()
        for smiles in body.smiles_list or scope["smiles"]:
            if len(smiles) > 512:
                # As in submit-batch: RDKit overflows the C stack on very long input.
                rejected.append({"smiles": smiles[:120], "detail": "SMILES too long (>512)."})
                continue
            detail, check = _reject_reason(body, smiles, headers, advanced)
            if detail is None and check["canonical"] not in known:
                detail = f"{smiles} is not an unarchived molecule of {name}."
            added: set = set()
            if detail is None:
                detail, added = categories.extension(session, name, check["canonical"], wanted)
            if detail is not None:
                rejected.append({"smiles": smiles, "detail": detail})
                continue
            if not added and states <= _state_descriptions(session, name, check["canonical"]):
                skipped.append(smiles)
                continue
            if check["canonical"] in queued_now or _extend_waiting(session, name, check["canonical"], added, states):
                skipped.append(smiles)
                continue
            queued_now.add(check["canonical"])
            entry = _new_entrypoint(session, body, check["canonical"], name, identity.username, headers, advanced)
            entry.request_metadata = json.dumps({**json.loads(entry.request_metadata), "extend": True})
            accepted.append((smiles, entry))
        # Added only now: an add inside the loop autoflushes on the next query and holds the write lock.
        session.add_all(entry for _, entry in accepted)
        session.commit()
        return {
            "project": name,
            "queued": [{"id": entry.id, "smiles": smiles} for smiles, entry in accepted],
            "rejected": rejected,
            "skipped": skipped,
        }


_EXTEND_STATES = (("T1", "request_T1"), ("ox", "request_ox"), ("red", "request_red"))


def _extend_waiting(session, project: str, canonical: str, added: set, states: set) -> bool:
    """Whether an unprocessed extend entrypoint of *project* already covers *added* and *states*."""
    from autodft import categories

    meta = col(CalculationEntrypoint.request_metadata)
    for entry in session.exec(select(CalculationEntrypoint).where(
        CalculationEntrypoint.smiles == canonical, col(CalculationEntrypoint.time_started).is_(None),
        case((func.json_valid(meta), func.json_extract(meta, "$.project_name"))) == project,
    )).all():
        stored = json.loads(entry.request_metadata)
        have = {d for d, key in _EXTEND_STATES if stored.get(key)}
        if stored.get("extend") and added <= categories.requested(stored) and states <= have:
            return True
    return False


def _state_descriptions(session, project: str, canonical: str) -> set[str]:
    """The state descriptions *canonical* already has in *project*."""
    return set(session.exec(
        select(MoleculeState.description).join(Molecule, Molecule.id == MoleculeState.molecule_id)
        .where(Molecule.project_name == project, Molecule.smiles == canonical)
    ).all())


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
    bad = _priority_error(identity, body) or _tolerance_error(body) or _conformers_error(body)
    if bad is not None:
        return bad

    accepted: list[tuple[str, CalculationEntrypoint]] = []
    rejected: list[dict] = []

    from autodft import categories
    from autodft.api import project_jobs

    with get_session() as session:
        forbidden = _header_access_error(session, identity, body)
        if forbidden is not None:
            return forbidden
        headers = _resolved_headers(session, body)
        project_name, author = _submission_owner(session, identity, body)
        try:
            project_jobs.assert_no_active_job(session, project_name)
        except project_jobs.JobInProgress as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        flags = _category_flags(body)
        wanted = categories.requested(flags)
        if categories.NBO_NEEDED & wanted:
            unavailable = categories.nbo_unavailable(get_active_settings())
            if unavailable:
                return JSONResponse(status_code=400, content={"detail": unavailable})
        advanced, detail = _submission_advanced(session, body, project_name, headers)
        if detail is not None:
            return JSONResponse(status_code=400, content={"detail": detail})
        requested_options = {**categories.options(flags), categories.ESD_HT: bool(flags.get(categories.ESD_HT))}
        for smiles in body.smiles_list:
            if len(smiles) > 512:
                # Matches the bound on SubmitRequest.smiles: RDKit's parser
                # overflows the C stack on very long input, and it runs in a
                # thread of the controller process.
                rejected.append({"smiles": smiles[:120], "detail": "SMILES too long (>512)."})
                continue
            detail, check = _reject_reason(body, smiles, headers, advanced)
            if detail is None:
                detail = categories.existing_conflict(
                    session, project_name, check["canonical"], wanted, requested_options, advanced, headers,
                )
            if detail is not None:
                rejected.append({"smiles": smiles, "detail": detail})
                continue
            entry = _new_entrypoint(session, body, smiles, project_name, author, headers, advanced)
            accepted.append((smiles, entry))

        # Added only now: an add inside the loop autoflushes on the next query and holds the write lock.
        session.add_all(entry for _, entry in accepted)
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
    from autodft.engine import nmr_references

    reserved = nmr_references.reserved_name_error(body.project)
    if reserved:
        raise HTTPException(status_code=400, detail=reserved)

    from autodft import accounts

    user = accounts.get_user_by_username(session, identity.username)
    if user is None:
        # No account row: the shared-password admin on a database that has
        # not been bootstrapped. The project name cannot be qualified, but
        # the author is still the caller and not the body.
        if nmr_references.is_reference_project(body.project):
            raise HTTPException(status_code=400,
                                detail=nmr_references.reserved_name_error(nmr_references.REFERENCE_PROJECT))
        return body.project, identity.username

    project = accounts.get_or_create_project(session, user, _bare_project(body))
    return project.qualified_name, user.username


def _new_entrypoint(
    session, body: SubmitRequest, smiles: str,
    project_name: Optional[str] = None, author: Optional[str] = None,
    headers: Optional[tuple] = None, advanced: Optional[dict] = None,
) -> CalculationEntrypoint:
    """Build (but do not add) the entrypoint row for one SMILES; ``advanced`` None is legacy."""
    from autodft import categories

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
        "imag_freq_tolerance": body.imag_freq_tolerance,
        "max_conformers_S0": n_s0,
        "max_conformers_T1": n_t1,
        "max_conformers_ox": n_ox,
        "max_conformers_red": n_red,
    }
    # Only set categories are written, so an unflagged submission's row is
    # exactly what it always was.
    request_metadata.update(categories.snapshot(_category_flags(body)))
    if advanced is not None:
        request_metadata["advanced"] = advanced

    h_cs, h_opt, h_sp = headers if headers is not None else _resolved_headers(session, body)
    return CalculationEntrypoint(
        smiles=smiles,
        request_metadata=json.dumps(request_metadata),
        priority=body.priority,
        header_confsearch=h_cs,
        header_optimization=h_opt,
        header_singlepoint=h_sp,
    )
