"""Parsed job results, stored once when a job is judged successful.

Readers take the stored record and fall back to parsing the job's files, so
a job judged before records existed -- or on an older parser version -- reads
exactly as before.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from sqlmodel import Session, case, col, func, select

from autodft import categories
from autodft.db import get_session
from autodft.extraction import connectivity
from autodft.models import ComputationJob, ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType
from autodft.models.result import JobResult
from autodft.qm.orca import esd_parser
from autodft.qm.orca.esd_parser import Rate
from autodft.qm.orca.parser import IMAGINARY_FREQ_THRESHOLD, IMAGINARY_FREQ_THRESHOLD_S1, OrcaParser
from autodft.qm.orca.spectra_parser import (
    Efg, IRMode, NaturalCharge, Shielding, Transition, parse_absorption, parse_efg, parse_hirshfeld, parse_ir,
    parse_natural_charges, parse_polarizability, parse_quadrupole, parse_s2, parse_shieldings,
    parse_stability, parse_wiberg,
)

logger = logging.getLogger(__name__)

# Bump only when a parser changes what an existing key holds -- older records
# are then re-parsed. A new key needs no bump when its readers `require` it:
# a record without it is stale for them and falls back to parsing on its own.
# Pinned by tests/test_stored_results.py::test_extract_output_is_pinned.
PARSER_VERSION = 2

# The task types a reader ever asks for. parse() keeps nothing else, and the
# backfill's query skips them outright.
STORED_TYPES = frozenset(
    t.value for t in TaskType
    if t.value.startswith("esd_") or t.value in {
        "optimization", "singlepoint", "singlepoint_vert_ox", "singlepoint_vert_red",
        "singlepoint_vert_spin_change", "singlepoint_uvvis", "singlepoint_nmr", "singlepoint_soc",
    }
)


def extract(task_type: str, output: str, input_text: Optional[str] = None, ir: bool = False) -> dict:
    """Everything the analyses and exports read from one successful job."""
    record: dict = {
        "energy": (
            OrcaParser.extract_electronic_energy(output)
            if "FINAL SINGLE POINT ENERGY" in output else None
        ),
        "free_energy_correction": (
            OrcaParser.extract_free_energy_correction(output) if "G-E(el)" in output else None
        ),
    }
    if task_type == "optimization":
        record["imaginary"] = (
            OrcaParser.extract_imaginary_frequencies(output)
            if "VIBRATIONAL FREQUENCIES" in output else []
        )
        record["followed_root"] = esd_parser.followed_root(output)
        if ir:
            record["ir_modes"] = [asdict(m) for m in parse_ir(output)]
    elif task_type in categories.REACTIVITY_TYPES:
        charges = parse_natural_charges(output)
        record["npa_charges"] = [asdict(c) for c in charges] if charges else None
        record["reactivity"] = _reactivity(output)
    elif task_type == "singlepoint_uvvis":
        record["transitions"] = [asdict(t) for t in parse_absorption(output)]
    elif task_type == "singlepoint_nmr":
        from autodft.analysis.nmr import method_fingerprint

        record["shieldings"] = [asdict(s) for s in parse_shieldings(output)]
        record["efg"] = [asdict(e) for e in parse_efg(output)]
        if input_text is None:
            record["method_fingerprint"] = None
        else:
            keywords, blocks = method_fingerprint(input_text)
            record["method_fingerprint"] = [sorted(keywords), blocks]
    elif task_type == "singlepoint_soc":
        record["soc"] = _soc(output)
    elif task_type.startswith("esd_"):
        record["rates"] = [asdict(r) for r in esd_parser.rates(output)]
    return record


def _reactivity(output: str) -> Optional[dict]:
    """Reactivity's properties, each None when not printed; None without Hirshfeld or polarizability."""
    hirshfeld = parse_hirshfeld(output)
    polarizability = parse_polarizability(output)
    if not hirshfeld and polarizability is None:
        return None
    return {
        "hirshfeld": [asdict(c) for c in hirshfeld] or None,
        "polarizability": polarizability,
        "quadrupole": parse_quadrupole(output),
        "wiberg": parse_wiberg(output),
        "stability": parse_stability(output),
        "s2": parse_s2(output),
    }


def _soc(output: str) -> Optional[dict]:
    from autodft.qm.orca.esd_inputs import EsdInputError, StateData

    try:
        data = StateData.parse(output)
    except EsdInputError:
        return None
    return {
        "ground": data.ground,
        "singlets": sorted(data.singlets.items()),
        "triplets": sorted(data.triplets.items()),
        "socme": [[t, s, v] for (t, s), v in sorted(data.socme.items())],
    }


def parse_job(task_type: str, job_path: Path, wants_ir: bool) -> Optional[dict]:
    """Parse *job_path*'s files into a plain dict; None when it has no output."""
    output = _read(job_path / "output.out")
    if output is None:
        return None
    input_text = _read(job_path / "input.inp") if task_type == "singlepoint_nmr" else None
    return extract(task_type, output, input_text, ir=wants_ir)


def _upsert(
    session: Session, job_id: int, task_id: int, job_path: Optional[str],
    finished_at: Optional[datetime], record: dict,
) -> None:
    """Insert *job_id*'s row, or overwrite it with a fresh identity and parser version."""
    row = session.exec(select(JobResult).where(JobResult.job_id == job_id)).first()
    if row is None:
        row = JobResult(job_id=job_id, task_id=task_id, parser_version=PARSER_VERSION, data_json="")
    elif (row.task_id == task_id and row.job_path == job_path and _same_instant(row.finished_at, finished_at)
          and "connectivity" not in record):
        # Same job: a re-parse (backfill) must not drop the check made when it was judged.
        kept = json.loads(row.data_json or "{}").get("connectivity")
        if kept is not None:
            record = {**record, "connectivity": kept}
    row.task_id = task_id
    row.job_path = job_path
    row.finished_at = finished_at
    row.parser_version = PARSER_VERSION
    row.data_json = json.dumps(record)
    row.updated_at = datetime.now(timezone.utc)
    session.add(row)


def parse(session: Session, task: ComputationTask, job_path: Path) -> Optional[dict]:
    """The record :func:`save` keeps for a job of *task*; only reads, so it runs before the write lock."""
    task_type = task.task_type.value
    if task_type not in STORED_TYPES:
        return None
    # An unrelated pending row (e.g. a bad geometry from this same tick) must
    # not be autoflushed by our own queries -- that would raise from inside
    # this function's caller's log-and-ignore instead of at the normal flush.
    with session.no_autoflush:
        record = parse_job(task_type, job_path, _wants_ir(session, task_type, task.state_id))
        if record is None:
            return None
        if task_type == "optimization":
            try:
                found = _connectivity(session, task.state_id, job_path)
            except Exception:  # noqa: BLE001 - the record matters more than the check
                logger.warning("Connectivity check failed for %s; stored without it", job_path, exc_info=True)
                found = None
            if found is not None:
                record["connectivity"] = found
    return record


def save(session: Session, task: ComputationTask, job: ComputationJob, record: dict) -> None:
    """Keep *record*, from :func:`parse`, as *job*'s row."""
    with session.no_autoflush:
        _upsert(session, job.id, task.id, job.job_path, job.time_end, record)


def _connectivity(session: Session, state_id: int, job_path: Path) -> Optional[dict]:
    """The optimised geometry's bonds against the molecule's SMILES; only new jobs get one."""
    state = session.get(MoleculeState, state_id)
    molecule = session.get(Molecule, state.molecule_id) if state is not None else None
    xyz = _read(job_path / "input.xyz")
    if molecule is None or xyz is None:
        return None
    return connectivity.check(molecule.smiles, xyz)


def connectivity_by_task(session: Session, task_ids: Iterable[int]) -> dict[int, dict]:
    """Stored connectivity checks of these optimisations' latest successful jobs, by task id."""
    latest = latest_successful_jobs(session, task_ids)
    rows = {
        row.job_id: row for row in session.exec(
            select(JobResult).where(
                col(JobResult.job_id).in_([j.id for j in latest.values()]),
                col(JobResult.data_json).contains('"connectivity"'),
            )
        ).all()
    } if latest else {}
    out = {}
    for task_id, job in latest.items():
        row = rows.get(job.id)
        if not _same_job(row, job):
            continue
        found = json.loads(row.data_json).get("connectivity")
        if found is not None:
            out[task_id] = found
    return out


def broken_bonds(session: Session, task: Optional[ComputationTask]) -> Optional[str]:
    """How *task*'s optimised geometry lost bonds of its SMILES, or None (extra bonds alone do not count)."""
    if task is None or task.task_type != TaskType.optimization:
        return None
    found = connectivity_by_task(session, [task.id]).get(task.id)
    return found["detail"] if connectivity.breaks(found) else None


def latest_successful_job(session: Session, task_id: int) -> Optional[ComputationJob]:
    """The task's latest successful job, ordered as the extractor always did."""
    return session.exec(
        select(ComputationJob).where(
            ComputationJob.task_id == task_id,
            ComputationJob.success == True,  # noqa: E712
        ).order_by(col(ComputationJob.attempt).desc(), col(ComputationJob.id).desc())
    ).first()


def latest_successful_jobs(session: Session, task_ids: Iterable[int]) -> dict:
    """latest_successful_job() of many tasks at once, by task id.

    Rows of id, task_id, attempt, job_path and time_end, not ORM objects: a
    project's worth of those is most of the time spent.
    """
    latest: dict = {}
    for chunk in _chunks(sorted(set(task_ids)), 500):
        for job in session.exec(
            select(
                ComputationJob.id, ComputationJob.task_id, ComputationJob.attempt,
                ComputationJob.job_path, ComputationJob.time_end,
            ).where(
                col(ComputationJob.task_id).in_(chunk), ComputationJob.success == True,  # noqa: E712
            )
        ).all():
            best = latest.get(job.task_id)
            if best is None or (job.attempt, job.id) > (best.attempt, best.id):
                latest[job.task_id] = job
    return latest


def stored_values(
    session: Session, types: dict[int, str], keys: tuple[str, ...],
) -> tuple[dict[int, dict], list[tuple]]:
    """Scalar *keys* of many tasks' records, from task id -> task type, read in SQL.

    Current records matching their job answer here, without decoding them.
    The rest come back as ``pending`` for :func:`resolve_pending`, which may
    read output files and so belongs after the session is closed -- so do
    records SQLite's JSON cannot read (NaN or Infinity from json.dumps).
    """
    valid = func.json_valid(JobResult.data_json) == 1
    latest = latest_successful_jobs(session, types)
    rows: dict = {}
    for chunk in _chunks([j.id for j in latest.values()], 500):
        rows.update({
            row.job_id: row
            for row in session.exec(
                select(
                    JobResult.job_id, JobResult.task_id, JobResult.job_path, JobResult.finished_at,
                    JobResult.parser_version, valid.label("valid"),
                    *(case((valid, func.json_extract(JobResult.data_json, f"$.{key}")), else_=None).label(key)
                      for key in keys),
                ).where(col(JobResult.job_id).in_(chunk))
            ).all()
        })
    found: dict[int, dict] = {}
    # job id -> (task id, whether the record is current)
    decode: dict[int, tuple[int, bool]] = {}
    pending: list[tuple] = []
    for task_id, job in latest.items():
        row = rows.get(job.id)
        if not _same_job(row, job):
            pending.append((task_id, types[task_id], job.job_path, None, False))
        elif row.parser_version == PARSER_VERSION and row.valid:
            found[task_id] = {key: getattr(row, key) for key in keys}
        else:
            decode[job.id] = (task_id, row.parser_version == PARSER_VERSION)
    for chunk in _chunks(sorted(decode), 500):
        for job_id, data_json in session.exec(
            select(JobResult.job_id, JobResult.data_json).where(col(JobResult.job_id).in_(chunk))
        ).all():
            task_id, current = decode[job_id]
            pending.append((task_id, types[task_id], latest[task_id].job_path, data_json, current))
    return found, pending


def resolve_pending(pending: list[tuple], keys: tuple[str, ...]) -> dict[int, dict]:
    """The :func:`stored_values` the session could not answer, by :func:`resolve`; needs no session."""
    out = {}
    for task_id, task_type, job_path, data_json, current in pending:
        record = json.loads(data_json) if data_json is not None else None
        found = resolve(job_path, task_type, record, current)
        if found is not None:
            out[task_id] = {key: found.get(key) for key in keys}
    return out


def _same_instant(a: Optional[datetime], b: Optional[datetime]) -> bool:
    """Compare timestamps that may have lost their tzinfo in a SQLite round trip."""
    to_iso = lambda d: d.replace(tzinfo=None).isoformat() if d is not None else None  # noqa: E731
    return to_iso(a) == to_iso(b)


def for_job(
    session: Session, job: ComputationJob, task_type: str, require: Iterable[str] = (),
) -> Optional[dict]:
    """*job*'s record: stored, else parsed from its files; None without an output.

    A row is only served when it still belongs to this job -- same task, same
    job_path, same finished_at -- since SQLite recycles job ids after a wipe.
    A row that is stale (older version, or missing a key the caller
    `require`s) but still matches this job's identity is a fallback of last
    resort once the output is gone too, e.g. an archived project: served
    when it has every required key, else None.
    """
    return resolve(job.job_path, task_type, *stored(session, job), require)


def stored(session: Session, job: ComputationJob) -> tuple[Optional[dict], bool]:
    """*job*'s stored record if the row still belongs to it, and whether its parser version is current."""
    return _row_record(session.exec(select(JobResult).where(JobResult.job_id == job.id)).first(), job)


def _row_record(row: Optional[JobResult], job: ComputationJob) -> tuple[Optional[dict], bool]:
    if not _same_job(row, job):
        return None, False
    return json.loads(row.data_json), row.parser_version == PARSER_VERSION


def _same_job(row, job) -> bool:
    """Whether a JobResult (row) was parsed from this job (row), not one that reused its id."""
    return (
        row is not None
        and row.task_id == job.task_id
        and row.job_path == job.job_path
        and _same_instant(row.finished_at, job.time_end)
    )


def resolve(
    job_path: Optional[str], task_type: str, record: Optional[dict], current: bool, require: Iterable[str] = (),
) -> Optional[dict]:
    """:func:`for_job` from a :func:`stored` result; needs no session, reads the output when it must."""
    require = tuple(require)
    if record is not None and current and all(key in record for key in require):
        return record
    path = Path(job_path) if job_path else None
    output = _read(path / "output.out") if path is not None else None
    if output is None:
        return record if record is not None and all(key in record for key in require) else None
    input_text = _read(path / "input.inp") if task_type == "singlepoint_nmr" else None
    return extract(task_type, output, input_text, ir="ir_modes" in require)


def for_task(session: Session, task: ComputationTask, require: Iterable[str] = ()) -> Optional[dict]:
    """The record of *task*'s latest successful job, or None."""
    job = latest_successful_job(session, task.id)
    return for_job(session, job, task.task_type.value, require) if job is not None else None


def conformer_numbers(opts: Iterable[ComputationTask]) -> dict[int, int]:
    """Conformer number by task id: one state's successful optimisations by id, from 1."""
    done = sorted(t.id for t in opts if t.task_type == TaskType.optimization and t.status == TaskStatus.successful)
    return {task_id: number for number, task_id in enumerate(done, 1)}


def method_family(state: MoleculeState) -> tuple:
    """*state*'s method family: its confsearch, optimization and singlepoint header ids."""
    return (state.confsearch_header_id, state.optimization_header_id, state.singlepoint_header_id)


def shared_descriptions(states: Iterable[MoleculeState]) -> set[tuple[int, str]]:
    """``(molecule id, description)`` pairs that more than one of *states* holds."""
    seen: set[tuple[int, str]] = set()
    shared = set()
    for state in states:
        key = (state.molecule_id, state.description)
        (shared if key in seen else seen).add(key)
    return shared


def exclusion(session: Session, opt_task: ComputationTask, state: MoleculeState) -> Optional[str]:
    """Why exports leave *opt_task*'s conformer out, ``"unchecked"`` without a record, else None."""
    return record_exclusion(for_task(session, opt_task, require=("imaginary",)), state)


def imaginary_cutoff(state: MoleculeState) -> float:
    """The mode (cm-1) below which *state*'s optimisations fail and exports leave them out."""
    metadata = json.loads(state.metadata_json) if state.metadata_json else {}
    if "imag_freq_tolerance" in metadata:
        return float(metadata["imag_freq_tolerance"])
    # States from before the tolerance keep the old rule.
    if metadata.get("esd_role") == "S1" and not metadata.get("strict_convergence", True):
        return IMAGINARY_FREQ_THRESHOLD_S1
    return IMAGINARY_FREQ_THRESHOLD


def record_exclusion(record: Optional[dict], state: MoleculeState) -> Optional[str]:
    """exclusion() from an opt record already read."""
    cutoff = imaginary_cutoff(state)
    if record is None or "imaginary" not in record:
        return "unchecked"
    lowest = min(record["imaginary"], default=0.0)
    return f"imaginary mode {lowest:.1f} cm-1" if lowest < cutoff else None


def excludes(note: Optional[str]) -> bool:
    """Whether an exclusion() note drops the conformer; ``"unchecked"`` keeps it."""
    return note is not None and note.startswith("imaginary")


def ir_modes(record: dict) -> list[IRMode]:
    return [IRMode(**m) for m in record.get("ir_modes", [])]


def transitions(record: dict) -> list[Transition]:
    return [Transition(**t) for t in record.get("transitions", [])]


def shieldings(record: dict) -> list[Shielding]:
    return [Shielding(**s) for s in record.get("shieldings", [])]


def efg(record: dict) -> list[Efg]:
    return [Efg(**e) for e in record.get("efg", [])]


def natural_charges(record: dict) -> Optional[list[NaturalCharge]]:
    charges = record.get("npa_charges")
    return [NaturalCharge(**c) for c in charges] if charges is not None else None


def fingerprint(record: dict) -> Optional[tuple[frozenset[str], str]]:
    found = record.get("method_fingerprint")
    return (frozenset(found[0]), found[1]) if found is not None else None


def state_data(record: dict):
    """The SOC singlepoint's roots and SOC matrix, or None when it had none."""
    from autodft.qm.orca.esd_inputs import StateData

    soc = record.get("soc")
    if soc is None:
        return None
    return StateData(
        soc["ground"], dict(soc["singlets"]), dict(soc["triplets"]),
        {(t, s): v for t, s, v in soc["socme"]},
    )


def rates(record: dict) -> list[Rate]:
    return [Rate(**r) for r in record.get("rates", [])]


def _worklist(session: Session, project: Optional[str]) -> tuple[list[tuple], int]:
    """``(worklist, already_current)``: plain job tuples, and how many were skipped.

    A worklist entry is ``(job_id, job_path, finished_at, task_id, task_type,
    state_id)``. Successful jobs of a stored type that already have a
    current, matching record are counted but left out -- no ORM objects are
    kept, so the caller may hold the worklist across commits.
    """
    query = (
        select(
            ComputationJob.id, ComputationJob.job_path, ComputationJob.time_end,
            ComputationTask.id, ComputationTask.task_type, ComputationTask.state_id,
        )
        .join(ComputationTask, ComputationJob.task_id == ComputationTask.id)
        .where(
            ComputationJob.success == True,  # noqa: E712
            col(ComputationTask.task_type).in_([TaskType(t) for t in STORED_TYPES]),
        )
        .order_by(col(ComputationJob.id))
    )
    if project is not None:
        query = (
            query.join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
            .join(Molecule, MoleculeState.molecule_id == Molecule.id)
            .where(Molecule.project_name == project)
        )
    current = set(session.exec(
        select(JobResult.job_id, JobResult.job_path, JobResult.finished_at)
        .where(JobResult.parser_version == PARSER_VERSION)
    ).all())

    worklist = []
    already_current = 0
    for job_id, job_path, finished_at, task_id, task_type, state_id in session.exec(query).all():
        if (job_id, job_path, finished_at) in current:
            already_current += 1
            continue
        worklist.append((job_id, job_path, finished_at, task_id, task_type.value, state_id))
    return worklist, already_current


def _ir_wanted_by_state(session: Session, state_ids: Iterable[int]) -> dict[int, bool]:
    return {state_id: _wants_ir(session, "optimization", state_id) for state_id in state_ids}


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def backfill(
    project: Optional[str] = None, batch: int = 25,
    progress: Optional[Callable[[dict[str, int]], None]] = None,
) -> dict[str, int]:
    """Store records for successful jobs that have none, or an older parser's.

    Each chunk of *batch* jobs is read and parsed with no pending session
    changes, then written in one short transaction that re-checks the job is
    still there and still successful -- so the SQLite write lock is never
    held across a slow NFS read, and a job deleted mid-run gets no row.
    """
    counts = {"stored": 0, "current": 0, "missing_output": 0, "unreadable": 0}
    with get_session() as session:
        worklist, counts["current"] = _worklist(session, project)
        ir_wanted = _ir_wanted_by_state(
            session, {state_id for *_, task_type, state_id in worklist if task_type == "optimization"}
        )

    for chunk in _chunks(worklist, batch):
        parsed: list[tuple[int, int, Optional[str], Optional[datetime], dict]] = []
        for job_id, job_path, finished_at, task_id, task_type, state_id in chunk:
            if not job_path:
                counts["missing_output"] += 1
                continue
            try:
                record = parse_job(task_type, Path(job_path), ir_wanted.get(state_id, False))
            except (OSError, ValueError):
                logger.exception("Backfill: could not read job %d's output", job_id)
                counts["unreadable"] += 1
                continue
            if record is None:
                counts["missing_output"] += 1
                continue
            parsed.append((job_id, task_id, job_path, finished_at, record))

        if parsed:
            with get_session() as session:
                still_successful = set(session.exec(
                    select(ComputationJob.id).where(
                        col(ComputationJob.id).in_([p[0] for p in parsed]),
                        ComputationJob.success == True,  # noqa: E712
                    )
                ).all())
                for job_id, task_id, job_path, finished_at, record in parsed:
                    if job_id not in still_successful:
                        continue
                    _upsert(session, job_id, task_id, job_path, finished_at, record)
                    counts["stored"] += 1
                session.commit()

        if progress is not None:
            progress(dict(counts))

    return counts


def _wants_ir(session: Session, task_type: str, state_id: int) -> bool:
    if task_type != "optimization":
        return False
    state = session.get(MoleculeState, state_id)
    metadata = json.loads(state.metadata_json) if state is not None and state.metadata_json else {}
    return categories.IR in categories.requested(metadata)


def _read(path: Path) -> Optional[str]:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None
