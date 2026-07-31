"""Pipeline extraction: energy extraction, Excel/CSV export, file export, cleanup.

Ported from the old ``dft_pipeline_extraction`` package.  All database access
goes through SQLModel sessions; ORCA output parsing goes through
:class:`OrcaParser`.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sqlmodel import Session, col, select

from autodft.db import get_session
from autodft.models.enums import TaskStatus, TaskType
from autodft.models.geometry import MoleculeGeometry
from autodft.models.job import ComputationJob
from autodft.models.molecule import Molecule
from autodft.models.state import MoleculeState
from autodft.models.task import ComputationTask
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.qm.orca.parser import OrcaParser

logger = logging.getLogger(__name__)


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class ConformerResult:
    """Extracted energies for one conformer of one state."""
    molecule_id: int
    smiles: str
    state: str
    conformer_index: int
    opt_task_id: int
    e_singlepoint: Optional[float] = None
    e_correction: Optional[float] = None
    e_combined: Optional[float] = None
    e_vert_spin_change: Optional[float] = None
    e_vert_ox: Optional[float] = None
    e_vert_red: Optional[float] = None


# ======================================================================
# Main extractor
# ======================================================================

class PipelineExtractor:
    """Extract, summarise, and export results for a project.

    Usage::

        ext = PipelineExtractor("my_project")
        ext.export_summary_csv("results.csv")
        ext.export_calculation_files(Path("./export"))
    """

    def __init__(self, project_name: str) -> None:
        self.project_name = project_name
        self._parser = OrcaParser()

    # ------------------------------------------------------------------
    # Progress / success rate
    # ------------------------------------------------------------------

    def get_submission_progress(self) -> dict[str, int]:
        """Return ``{total, started}`` counts for entrypoints.

        Matching is on the parsed ``project_name`` rather than a LIKE over
        the whole JSON blob: `request_metadata.contains("test")` also counted
        project "test2", any molecule whose SMILES or author happened to
        contain "test", and treated % and _ as wildcards.
        """
        import json as _json

        with get_session() as session:
            rows = session.exec(
                select(CalculationEntrypoint.request_metadata,
                       CalculationEntrypoint.time_started)
            ).all()
        total = started = 0
        for metadata_json, time_started in rows:
            try:
                meta = _json.loads(metadata_json) if metadata_json else {}
            except (ValueError, TypeError):
                continue
            if meta.get("project_name") != self.project_name:
                continue
            total += 1
            if time_started is not None:
                started += 1
        return {"total": total, "started": started}

    def get_success_rate(self) -> dict[str, int]:
        """Return ``{total_molecules, successful_molecules}``.

        A molecule counts as successful when it has tasks and every one of
        them succeeded. Computed with two grouped queries rather than one
        state query plus one task query per molecule -- the per-molecule
        version loaded full ORM objects only to count them, and at a few
        thousand molecules dominated the project endpoint's response time.
        """
        from sqlmodel import func

        with get_session() as session:
            mol_ids = [
                m for m in session.exec(
                    select(Molecule.id).where(Molecule.project_name == self.project_name)
                ).all() if m is not None
            ]
            if not mol_ids:
                return {"total_molecules": 0, "successful_molecules": 0}

            counts: dict[int, dict[str, int]] = {}
            for molecule_id, status, count in session.exec(
                select(MoleculeState.molecule_id, ComputationTask.status, func.count())
                .select_from(ComputationTask)
                .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .where(col(MoleculeState.molecule_id).in_(mol_ids))
                .group_by(col(MoleculeState.molecule_id), col(ComputationTask.status))
            ).all():
                key = status.value if hasattr(status, "value") else str(status)
                counts.setdefault(molecule_id, {})[key] = count

        successful = sum(
            1 for by_status in counts.values()
            if by_status and set(by_status) == {TaskStatus.successful.value}
        )
        return {"total_molecules": len(mol_ids), "successful_molecules": successful}

    # ------------------------------------------------------------------
    # Energy extraction
    # ------------------------------------------------------------------

    def extract_results(self, all_conformers: bool = False) -> list[ConformerResult]:
        """Walk the DB and extract energies from ORCA output files.

        For each successful optimization, reads the ORCA output to get
        the free-energy correction, then looks at the dependent
        singlepoint tasks for electronic energies.

        Args:
            all_conformers: If ``False`` (default), only the lowest-energy
                conformer per state is included.

        Returns:
            List of :class:`ConformerResult` entries.
        """
        results: list[ConformerResult] = []

        with get_session() as session:
            molecules = session.exec(
                select(Molecule).where(Molecule.project_name == self.project_name)
            ).all()

            for mol in molecules:
                states = session.exec(
                    select(MoleculeState).where(MoleculeState.molecule_id == mol.id)
                ).all()

                for state in states:
                    state_results = self._extract_state_results(
                        session, mol, state, all_conformers,
                    )
                    results.extend(state_results)

        return results

    def _extract_state_results(
        self,
        session: Session,
        mol: Molecule,
        state: MoleculeState,
        all_conformers: bool,
    ) -> list[ConformerResult]:
        """Extract results for one molecule state."""
        # Get successful optimization tasks, ordered by task ID (stable conformer ordering)
        opt_tasks = session.exec(
            select(ComputationTask).where(
                ComputationTask.state_id == state.id,
                ComputationTask.task_type == TaskType.optimization,
                ComputationTask.status == TaskStatus.successful,
            ).order_by(ComputationTask.id.asc())  # type: ignore[union-attr]
        ).all()

        results: list[ConformerResult] = []
        for conf_idx, opt_task in enumerate(opt_tasks, 1):
            result = self._extract_conformer_energies(
                session, mol, state, opt_task, conf_idx,
            )
            if result is not None:
                results.append(result)

        # Keep the genuinely lowest-energy conformer, not the first one.
        # Selecting opt_tasks[0] (ordered by task id) picked the conformer
        # that GOAT ranked lowest at xTB level, which is frequently not the
        # lowest after DFT optimisation -- on real project data this shifted
        # a reported oxidation potential by ~100 mV. The comparison uses
        # e_combined (electronic + thermal correction), falling back to the
        # bare singlepoint when no correction is available.
        if not all_conformers and results:
            picked = self._pick_reported_conformer(results)
            results = [picked] if picked is not None else []

        return results

    @staticmethod
    def _pick_reported_conformer(
        results: list[ConformerResult],
    ) -> Optional[ConformerResult]:
        """Return the lowest-energy conformer, ranking on a single scale.

        ``e_combined = e_singlepoint + (G - E_el)`` and that correction is
        positive and large (~0.1-0.2 Ha), so ranking a mixed pool by
        "e_combined, else e_singlepoint" lets any conformer *missing* a
        thermal correction -- truncated output, an optimisation header
        without Freq -- beat every properly corrected one unconditionally.
        Prefer the corrected pool; fall back to bare electronic energies only
        when nothing has a correction.
        """
        if not results:
            return None
        corrected = [r for r in results if r.e_combined is not None]
        if corrected:
            return min(corrected, key=lambda r: r.e_combined)
        bare = [r for r in results if r.e_singlepoint is not None]
        if bare:
            return min(bare, key=lambda r: r.e_singlepoint)
        return results[0]

    def _extract_conformer_energies(
        self,
        session: Session,
        mol: Molecule,
        state: MoleculeState,
        opt_task: ComputationTask,
        conf_idx: int,
    ) -> Optional[ConformerResult]:
        """Extract all energies for a single conformer."""
        # Get the successful job path for optimization
        opt_job_path = self._get_successful_job_path(session, opt_task.id)
        if opt_job_path is None:
            return None

        # Free energy correction from optimization output
        opt_content = self._load_output(opt_job_path)
        e_correction = OrcaParser.extract_free_energy_correction(opt_content) if opt_content else None

        # Get dependent tasks
        dep_tasks = session.exec(
            select(ComputationTask).where(
                ComputationTask.depends_on_task_id == opt_task.id,
                ComputationTask.status == TaskStatus.successful,
            )
        ).all()

        e_sp = self._get_energy_from_task(session, dep_tasks, TaskType.singlepoint)
        e_vert_ox = self._get_energy_from_task(session, dep_tasks, TaskType.singlepoint_vert_ox)
        e_vert_red = self._get_energy_from_task(session, dep_tasks, TaskType.singlepoint_vert_red)
        e_vert_spin = self._get_energy_from_task(session, dep_tasks, TaskType.singlepoint_vert_spin_change)

        e_combined = None
        if e_sp is not None and e_correction is not None:
            e_combined = e_sp + e_correction

        return ConformerResult(
            molecule_id=mol.id,  # type: ignore[arg-type]
            smiles=mol.smiles,
            state=state.description,
            conformer_index=conf_idx,
            opt_task_id=opt_task.id,  # type: ignore[arg-type]
            e_singlepoint=e_sp,
            e_correction=e_correction,
            e_combined=e_combined,
            e_vert_spin_change=e_vert_spin,
            e_vert_ox=e_vert_ox,
            e_vert_red=e_vert_red,
        )

    def _get_energy_from_task(
        self,
        session: Session,
        dep_tasks: list[ComputationTask],
        task_type: TaskType,
    ) -> Optional[float]:
        """Find a dependent task by type and extract its electronic energy."""
        task = next((t for t in dep_tasks if t.task_type == task_type), None)
        if task is None:
            return None
        job_path = self._get_successful_job_path(session, task.id)
        if job_path is None:
            return None
        content = self._load_output(job_path)
        if content is None:
            return None
        return OrcaParser.extract_electronic_energy(content)

    def _get_successful_job_path(
        self, session: Session, task_id: int,
    ) -> Optional[Path]:
        """Return the job_path of the latest successful job for a task.

        Ordered explicitly: an unordered .first() is an arbitrary pick when a
        task has more than one successful attempt, and which one you get can
        change with the query plan.
        """
        job = session.exec(
            select(ComputationJob).where(
                ComputationJob.task_id == task_id,
                ComputationJob.success == True,  # noqa: E712
            ).order_by(ComputationJob.attempt.desc(), ComputationJob.id.desc())  # type: ignore[union-attr]
        ).first()
        if job and job.job_path:
            return Path(job.job_path)
        return None

    @staticmethod
    def _load_output(job_path: Path) -> Optional[str]:
        """Read output.out from a job directory."""
        output_file = job_path / "output.out"
        if output_file.exists():
            return output_file.read_text(encoding="utf-8", errors="replace")
        return None

    # ------------------------------------------------------------------
    # Export: CSV / JSON
    # ------------------------------------------------------------------

    def export_summary_csv(
        self, output_path: str | Path, all_conformers: bool = False,
    ) -> None:
        """Export an energy summary to CSV."""
        results = self.extract_results(all_conformers=all_conformers)
        if not results:
            logger.warning("No results to export for project '%s'", self.project_name)
            return

        path = Path(output_path)
        fields = [
            "molecule_id", "smiles", "state", "conformer_index", "opt_task_id",
            "e_singlepoint", "e_correction", "e_combined",
            "e_vert_spin_change", "e_vert_ox", "e_vert_red",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "molecule_id": r.molecule_id,
                    "smiles": r.smiles,
                    "state": r.state,
                    "conformer_index": r.conformer_index,
                    "opt_task_id": r.opt_task_id,
                    "e_singlepoint": r.e_singlepoint,
                    "e_correction": r.e_correction,
                    "e_combined": r.e_combined,
                    "e_vert_spin_change": r.e_vert_spin_change,
                    "e_vert_ox": r.e_vert_ox,
                    "e_vert_red": r.e_vert_red,
                })
        logger.info("Exported %d rows to %s", len(results), path)

    def export_summary_json(
        self, output_path: str | Path, all_conformers: bool = False,
    ) -> None:
        """Export an energy summary to JSON."""
        results = self.extract_results(all_conformers=all_conformers)
        path = Path(output_path)
        data = [
            {
                "molecule_id": r.molecule_id,
                "smiles": r.smiles,
                "state": r.state,
                "conformer_index": r.conformer_index,
                "opt_task_id": r.opt_task_id,
                "e_singlepoint": r.e_singlepoint,
                "e_correction": r.e_correction,
                "e_combined": r.e_combined,
                "e_vert_spin_change": r.e_vert_spin_change,
                "e_vert_ox": r.e_vert_ox,
                "e_vert_red": r.e_vert_red,
            }
            for r in results
        ]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Exported %d entries to %s", len(data), path)

    # ------------------------------------------------------------------
    # Export: raw calculation files
    # ------------------------------------------------------------------

    def export_calculation_files(
        self,
        dest_dir: str | Path,
        all_conformers: bool = False,
        additional_extensions: Optional[list[str]] = None,
    ) -> int:
        """Copy ORCA calculation files with standardized naming.

        File naming convention:
        ``<mol_id>/<state>/conf<N>_<task_type>_<file>.ext``

        Returns:
            Total number of files copied.
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        # Collect the copy plan under a short-lived session, then copy with no
        # session open. This runs on a background thread that shares the
        # SQLite connection pool with the controller (see db.py); holding a
        # pooled connection across minutes of NFS copies would starve the
        # pipeline's own writes.
        plan: list[tuple[Path, Path, int, str]] = []
        with get_session() as session:
            molecules = session.exec(
                select(Molecule).where(Molecule.project_name == self.project_name)
            ).all()

            for mol in molecules:
                states = session.exec(
                    select(MoleculeState).where(MoleculeState.molecule_id == mol.id)
                ).all()

                for state in states:
                    plan.extend(
                        self._plan_state_files(session, mol, state, dest, all_conformers)
                    )

        total_copied = 0
        for source_job_path, dest_subdir, conf_index, task_type in plan:
            total_copied += _copy_task_files(
                source_job_path=source_job_path,
                dest_dir=dest_subdir,
                conf_index=conf_index,
                task_type=task_type,
                additional_extensions=additional_extensions,
            )

        logger.info("Exported %d files to %s", total_copied, dest)
        return total_copied

    def _plan_state_files(
        self,
        session: Session,
        mol: Molecule,
        state: MoleculeState,
        dest: Path,
        all_conformers: bool,
    ) -> list[tuple[Path, Path, int, str]]:
        """Resolve which files to copy for one state.

        Returns a list of ``(source_job_path, dest_subdir, conf_index,
        task_type)`` -- everything the copy needs, gathered while the session
        is open so the copy itself can run without one.
        """
        # Get all successful tasks for this state
        tasks = session.exec(
            select(ComputationTask).where(
                ComputationTask.state_id == state.id,
                ComputationTask.status == TaskStatus.successful,
            )
        ).all()

        # Build conformer index from optimization tasks
        opt_tasks = sorted(
            [t for t in tasks if t.task_type == TaskType.optimization],
            key=lambda t: t.id,  # type: ignore[arg-type]
        )
        if not all_conformers and opt_tasks:
            opt_tasks = [opt_tasks[0]]

        opt_id_to_conf = {t.id: i + 1 for i, t in enumerate(opt_tasks)}
        plan: list[tuple[Path, Path, int, str]] = []

        for task in tasks:
            # Determine conformer index
            if task.task_type == TaskType.optimization:
                conf_idx = opt_id_to_conf.get(task.id)
            elif task.task_type == TaskType.confsearch:
                conf_idx = 0
            else:
                conf_idx = opt_id_to_conf.get(task.depends_on_task_id)

            if conf_idx is None:
                continue

            job_path = self._get_successful_job_path(session, task.id)
            if job_path is None:
                continue

            plan.append(
                (job_path, dest / str(mol.id) / state.description, conf_idx, task.task_type.value)
            )

        return plan

    # ------------------------------------------------------------------
    # Archive: filtered export + comp_data wipe + DB delete
    # ------------------------------------------------------------------

    def archive_project(
        self,
        export_root: str | Path,
        comp_root: str | Path,
        extensions: list[str],
        all_conformers: bool = False,
    ) -> dict[str, Any]:
        """Destructive one-shot archive of this project.

        Steps, in order:
          1. Write the energy summary CSV into ``<export_root>/<project>/``.
          2. Copy every file under ``<comp_root>/mol_<id>/`` whose suffix
             is in ``extensions`` into ``<export_root>/<project>/raw/``,
             preserving the directory structure.
          3. ``rm -rf`` each ``<comp_root>/mol_<id>/`` for this project.
          4. Delete the project's rows from the database (jobs → tasks
             → geometries → states → molecules → entrypoints).

        Returns a summary dict ``{molecules, files_copied, files_dropped,
        csv_path, files_root}``.

        Designed to be triggered from the dashboard's "Export all files"
        button. The caller is responsible for the user confirmation.
        """
        from autodft.models.entrypoint import CalculationEntrypoint

        export_root = Path(export_root)
        comp_root = Path(comp_root)

        # Normalise & sanity-check extensions (always include the leading dot,
        # never accept things that aren't extensions like "input.inp").
        clean_exts = {self._normalise_ext(e) for e in extensions}
        if not clean_exts:
            raise ValueError("At least one extension is required.")

        from autodft.paths import project_file_stem, safe_subdirectory

        project_export_dir = safe_subdirectory(export_root, self.project_name)
        project_export_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = project_export_dir / "raw"

        # 1) CSV summary first — if extraction fails we abort before
        #    touching anything destructive.
        csv_path = project_export_dir / f"{project_file_stem(self.project_name)}.csv"
        self.export_summary_csv(csv_path, all_conformers=all_conformers)

        # 2) Filtered copy of comp_data tree.
        files_copied = 0
        files_dropped = 0
        molecule_ids: list[int] = []

        with get_session() as session:
            molecules = session.exec(
                select(Molecule).where(Molecule.project_name == self.project_name)
            ).all()
            molecule_ids = [m.id for m in molecules if m.id is not None]

        for mol_id in molecule_ids:
            mol_dir = comp_root / f"mol_{mol_id}"
            if not mol_dir.is_dir():
                continue
            for src in mol_dir.rglob("*"):
                if not src.is_file():
                    continue
                if src.suffix not in clean_exts:
                    files_dropped += 1
                    continue
                rel = src.relative_to(comp_root)
                dest = raw_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                files_copied += 1

        # 3) Wipe comp_data/mol_<id> trees for this project.
        for mol_id in molecule_ids:
            mol_dir = comp_root / f"mol_{mol_id}"
            if mol_dir.is_dir():
                shutil.rmtree(mol_dir)

        # 4) Mark the project as archived. We deliberately KEEP every DB
        #    row so the Molecules subpage can still show the project's
        #    history; only the on-disk comp_data tree is gone.
        self._mark_project_archived()

        logger.info(
            "Archived project %r: %d molecules, %d files kept, %d dropped",
            self.project_name, len(molecule_ids), files_copied, files_dropped,
        )

        return {
            "molecules": len(molecule_ids),
            "files_copied": files_copied,
            "files_dropped": files_dropped,
            "csv_path": str(csv_path),
            "files_root": str(raw_dir),
            "extensions": sorted(clean_exts),
        }

    @staticmethod
    def _normalise_ext(ext: str) -> str:
        ext = ext.strip().lower()
        if not ext:
            return ""
        if not ext.startswith("."):
            ext = "." + ext
        return ext

    def _mark_project_archived(self) -> None:
        """Set ``Molecule.archived = True`` on every row in this project.

        Replaces the prior ``_delete_project_rows`` behaviour. Keeping
        the rows means archived projects still appear in selectors and
        the Molecules subpage; their on-disk data lives in the export
        directory and the DB acts as the searchable index.
        """
        with get_session() as session:
            mols = session.exec(
                select(Molecule).where(Molecule.project_name == self.project_name)
            ).all()
            for m in mols:
                m.archived = True
                session.add(m)
            session.commit()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_large_files(
        self,
        keep_extensions: Optional[list[str]] = None,
        dry_run: bool = False,
    ) -> int:
        """Delete large temporary files from successful job directories.

        Keeps only files with the specified extensions.

        Returns:
            Number of files deleted.
        """
        extensions = {".out", ".xyz", ".inp"}
        if keep_extensions:
            extensions.update(keep_extensions)

        deleted = 0
        with get_session() as session:
            jobs = session.exec(
                select(ComputationJob).where(ComputationJob.success == True)  # noqa: E712
            ).all()

            for job in jobs:
                if not job.job_path:
                    continue
                job_dir = Path(job.job_path)
                if not job_dir.is_dir():
                    continue
                for f in job_dir.iterdir():
                    if f.is_file() and f.suffix not in extensions:
                        if dry_run:
                            logger.info("Would delete: %s", f)
                        else:
                            f.unlink()
                            deleted += 1

        logger.info("Cleanup: deleted %d files (dry_run=%s)", deleted, dry_run)
        return deleted


# ======================================================================
# File copy helper (ported from old FileExporter)
# ======================================================================

# Map task_type -> list of (source_name, dest_suffix)
_FILE_MAP: dict[str, list[tuple[str, str]]] = {
    "optimization": [
        ("input.inp", "opt_input.inp"),
        ("input.xyz", "opt_geometry.xyz"),
        ("output.out", "opt_output.out"),
    ],
    "singlepoint": [
        ("input.inp", "sp_input.inp"),
        ("input.xyz", "sp_geometry.xyz"),
        ("output.out", "sp_output.out"),
    ],
    "singlepoint_vert_spin_change": [
        ("input.inp", "sp_vert_spin_change_input.inp"),
        ("input.xyz", "sp_vert_spin_change_geometry.xyz"),
        ("output.out", "sp_vert_spin_change_output.out"),
    ],
    "singlepoint_vert_ox": [
        ("input.inp", "sp_vert_ox_input.inp"),
        ("input.xyz", "sp_vert_ox_geometry.xyz"),
        ("output.out", "sp_vert_ox_output.out"),
    ],
    "singlepoint_vert_red": [
        ("input.inp", "sp_vert_red_input.inp"),
        ("input.xyz", "sp_vert_red_geometry.xyz"),
        ("output.out", "sp_vert_red_output.out"),
    ],
    "confsearch": [
        ("input.inp", "confsearch_input.inp"),
        ("output.out", "confsearch_output.out"),
        ("input.finalensemble.xyz", "confsearch_ensemble.xyz"),
    ],
}


def _copy_task_files(
    source_job_path: Path,
    dest_dir: Path,
    conf_index: int,
    task_type: str,
    additional_extensions: Optional[list[str]] = None,
) -> int:
    """Copy ORCA files with standardized naming. Returns files copied."""
    if not source_job_path.exists():
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    base = f"conf{conf_index}"
    copied = 0

    for src_name, dest_suffix in _FILE_MAP.get(task_type, []):
        src = source_job_path / src_name
        if src.exists():
            shutil.copy2(src, dest_dir / f"{base}_{dest_suffix}")
            copied += 1

    if additional_extensions:
        for ext in additional_extensions:
            for f in source_job_path.glob(f"*{ext}"):
                shutil.copy2(f, dest_dir / f"{base}_{f.name}")
                copied += 1

    return copied
