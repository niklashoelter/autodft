"""ORCA output file parser.

Reads ``output.out`` (and optionally ``input.finalensemble.xyz``) produced
by an ORCA calculation and extracts energies, frequencies, normal modes,
and conformer ensembles.  All regex patterns are preserved from the
original ``orca_output_processor.py``.
"""

import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

from autodft.qm.base import QMEngine, QMResult

__all__ = ["OrcaParser"]

logger = logging.getLogger(__name__)

# Imaginary modes softer than this (cm^-1) are treated as numerical noise
# rather than a saddle point. Hindered rotors and floppy side chains
# routinely produce -20 to -50 cm^-1 modes that no amount of re-optimisation
# removes; failing on them burns the whole retry budget for nothing.
IMAGINARY_FREQ_THRESHOLD = -50.0
# A legacy ESD S1 state submitted without strict convergence: its Hessian is
# numerical and floppy aryl/tBu/CF3 rotors land at -50 to -65 cm^-1.
IMAGINARY_FREQ_THRESHOLD_S1 = -100.0
# The lowest imag_freq_tolerance a submission may set (cm^-1).
IMAG_FREQ_TOLERANCE_MIN = -300.0


# Written into output.out by submit.cmd when its watchdog kills a hung ORCA.
STALL_MARKER = "AUTODFT STALLED:"
# A rank killed by SIGSEGV/SIGBUS, as mpirun/ORCA report it on stderr.
CRASH_RE = re.compile(r"Segmentation fault|Bus error")
# An NMR job whose MP2 response lacked memory per rank; its log names the shortfall.
MP2_RESPONSE_MEMORY = "MP2 Response Memory"
_MP2_SHORT_RE = re.compile(r"not enough memory to treat a single MO \(need ([\d.]+) MB more\)")
# A densities singlepoint's requested cubes are missing or empty.
DENSITY_CUBES = "Density Cubes"
# Blocks a Reactivity singlepoint must print, by the name its failure reports.
REACTIVITY_MARKERS = (("NBO", "NBO analysis completed"), ("Polarizability", "Isotropic polarizability"))
_PLOTS_RE = re.compile(r"^\s*%plots\b(.*?)^\s*end\b", re.IGNORECASE | re.MULTILINE | re.DOTALL)
_CUBE_FILE_RE = re.compile(r'\(\s*"([^"]+\.cube)"\s*\)', re.IGNORECASE)


def _esd_main_jobs(job_path: Path) -> Optional[int]:
    """ORCA jobs in *job_path*'s input before its FC diagnostics (all of them without any)."""
    from autodft.qm.orca.esd_inputs import DIAGNOSTICS_MARKER

    try:
        text = (Path(job_path) / "input.inp").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    main, marker, _ = text.partition(DIAGNOSTICS_MARKER)
    if marker and not main.strip():
        # An HT rate's FC tail with diagnostics only: every job must print its rate.
        return text.count("$new_job") + 1
    # The $new_job right before the marker closes the last main job.
    return main.count("$new_job") + (0 if marker else 1)


def _before_diagnostics(content: str, main_jobs: Optional[int]) -> Optional[str]:
    """*content* up to where the first diagnostic job starts; None if none started."""
    if main_jobs is None:
        return None
    start = re.search(rf"^.*JOB NUMBER\s+{main_jobs + 1}\b", content, re.MULTILINE)
    return content[:start.start()] if start else None


def _after_last(lines: list[str], title: str) -> list[str]:
    """The lines after the last one containing *title*; empty when none does."""
    starts = [i for i, line in enumerate(lines) if title in line]
    return lines[starts[-1] + 1:] if starts else []


def mp2_response_memory_short(job_path: Path) -> Optional[float]:
    """MB per rank an MP2 response in *job_path* said it lacked, or None."""
    found = [float(m) for log in Path(job_path).glob("*.mp2resp_*")
             for m in _MP2_SHORT_RE.findall(log.read_text(encoding="utf-8", errors="replace"))]
    return max(found) if found else None


class OrcaParser(QMEngine):
    """Concrete :class:`QMEngine` implementation for ORCA.

    The parser is stateless between calls -- every public method accepts a
    *job_path* so the same instance can be reused across many jobs.

    An optional :class:`autodft.config.OrcaConfig` can be injected at
    construction time to control which ORCA binary the generated submit
    script invokes, what extra args to pass, and whether to stage I/O
    through a per-job ``/tmp`` directory.
    """

    def __init__(self, orca: "OrcaConfig | None" = None) -> None:
        from autodft.config import OrcaConfig
        self.orca = orca if orca is not None else OrcaConfig()

    # ------------------------------------------------------------------
    # QMEngine interface
    # ------------------------------------------------------------------

    def check_output(
        self, job_path: Path, task_type: str, imag_freq_tolerance: Optional[float] = None,
    ) -> QMResult:
        """Run all checks on an ORCA output and return a :class:`QMResult`.

        Args:
            job_path: Directory containing ``output.out``.
            task_type: E.g. ``"optimization"``, ``"singlepoint"``,
                       ``"confsearch"``.
            imag_freq_tolerance: Fail on any mode below it (cm^-1); None: -50.

        Returns:
            Populated :class:`QMResult`.
        """
        job_path = Path(job_path)
        content = self._load_output(job_path)
        if STALL_MARKER in content:
            # Killed by the submit script's watchdog: a hang, not a result.
            return QMResult(success=False, checks={"Stalled": False})
        if "orca finished by error termination in plot" in content.lower():
            # Not a failure of the job, but its cube files are missing or empty.
            logger.warning("orca_plot failed in %s; cube files missing or empty", job_path)

        main_jobs = _esd_main_jobs(job_path) if task_type.startswith("esd_") else None
        main_content = _before_diagnostics(content, main_jobs)
        if main_content is not None:
            # FC diagnostics never cost the rate: whatever they do, judge the main jobs only.
            content = main_content
        checks: dict[str, bool] = {
            "Termination": main_content is not None or self._check_normal_termination(content),
            "Optimization Convergence": self._check_for_optimization_convergence(
                content, task_type
            ),
            "SCF Convergence": self._check_scf_convergence(content),
            "Imaginary Frequencies": self._check_for_imaginary_frequencies(
                content,
                IMAGINARY_FREQ_THRESHOLD if imag_freq_tolerance is None else imag_freq_tolerance,
            ),
            # ORCA prints its normal-termination banner even after an
            # internal module has given up, so the banner alone is not
            # evidence of a usable result.
            "No Error Banner": self._check_no_error_banner(content),
        }
        if task_type == "confsearch":
            # A conformer search that produced no ensemble has nothing for
            # the rest of the chain to consume. Scoring it successful left
            # the state with zero optimizations and no error recorded
            # anywhere -- the calculation simply stopped.
            checks["Conformer Ensemble"] = bool(self._parse_ensemble_table(content))
        if task_type == "singlepoint_uvvis":
            # A TDDFT that died after the SCF still terminates normally.
            from autodft.qm.orca.spectra_parser import parse_absorption

            checks["Absorption Spectrum"] = bool(parse_absorption(content))
        if task_type in ("singlepoint_densities", "singlepoint_reactivity"):
            # A Reactivity input without %plots asks for no cubes.
            checks[DENSITY_CUBES] = not self.missing_cubes(job_path)
        if task_type == "singlepoint_reactivity":
            # NBO and the response run after the SCF; either can die and ORCA still ends normally.
            for name, marker in REACTIVITY_MARKERS:
                checks[f"Reactivity: {name} missing"] = marker in content
        if task_type == "singlepoint_nmr":
            from autodft.qm.orca.spectra_parser import parse_shieldings

            checks["Shieldings"] = bool(parse_shieldings(content))
        if task_type == "singlepoint_soc" or task_type.startswith("esd_") \
                or task_type == "optimization_excited":
            from autodft.qm.orca import esd_parser

            if task_type == "singlepoint_soc":
                # An unstable reference shows up as negative roots.
                roots = [*esd_parser.excitation_energies(content, "SINGLETS").values(),
                         *esd_parser.excitation_energies(content, "TRIPLETS").values()]
                checks["Excited States"] = bool(roots) and min(roots) > 0
                checks["SOC Matrix"] = bool(esd_parser.socme(content))
            else:
                # A native B88 functional fails the same way on every retry.
                checks["LibXC Needed"] = not esd_parser.needs_libxc(content)
            if task_type.startswith("esd_"):
                found = esd_parser.rates(content)
                # One rate per main ORCA job; a missing sublevel would skew the sum or mean.
                checks["ESD Rate"] = bool(found) and all(math.isfinite(r.rate) for r in found) and (
                    main_jobs is None or len(found) == main_jobs
                )
            if task_type == "optimization_excited":
                # Below 0.1 eV the followed root has collapsed onto S0.
                gap = esd_parser.followed_root_energy(content)
                checks["Excited Root"] = gap is not None and gap >= 0.1 * esd_parser.EV_TO_EH
                # A converged S1 with a small gap is a bound state, not a dissociation.
                checks["S1 Stays Bound"] = checks["Optimization Convergence"] \
                    or not esd_parser.s1_dissociates(content)

        success = all(checks.values())
        if not success and task_type == "singlepoint_nmr" and mp2_response_memory_short(job_path) is not None:
            # More memory per rank fixes it, not more ranks: see IncreaseMP2ResponseMemory.
            return QMResult(success=False, checks={MP2_RESPONSE_MEMORY: False})
        if not success and not checks["Termination"] and self._crashed(job_path):
            # A segfault without normal termination says nothing about the input: resubmit unchanged.
            return QMResult(success=False, checks={"Crashed": False})

        energy = self.extract_electronic_energy(content)
        # Only optimisations have this block; a singlepoint never did anyway.
        free_energy_correction = (
            self.extract_free_energy_correction(content) if "G-E(el)" in content else None
        )

        conformers: Optional[list[str]] = None
        conformer_energies: Optional[list[float]] = None
        conformer_degeneracies: Optional[list[int]] = None
        if task_type == "confsearch":
            conformers = self.extract_conformer_ensemble(job_path, content)
            conformer_energies = self.extract_conformer_ensemble_energies(content)
            conformer_degeneracies = self.extract_conformer_ensemble_degeneracies(content)

        return QMResult(
            success=success,
            checks=checks,
            energy=energy,
            free_energy_correction=free_energy_correction,
            conformers=conformers,
            conformer_energies=conformer_energies,
            conformer_degeneracies=conformer_degeneracies,
        )

    def render_input(self, header: str, charge: int, multiplicity: int, trailer: str = "") -> str:
        """Delegated to :func:`input_generator.render_orca_input`."""
        from autodft.qm.orca.input_generator import render_orca_input

        return render_orca_input(header, charge, multiplicity, trailer)

    def render_submit_script(
        self,
        job_path: Path,
        job_name: str,
        nprocs: int,
        mem_per_core: int,
        time_limit: str,
        partition: str,
        nice: int = 0,
        keep_hessian: bool = False,
        keep_cubes: bool = False,
        runs_nbo: bool = False,
        keep_mp2resp: bool = False,
        keep_esd_files: bool = False,
        extra_inputs: Optional[list[str]] = None,
        keep_wavefunction: bool = False,
    ) -> str:
        """Delegated to :func:`input_generator.render_submit_script`.

        Threads through the :class:`OrcaConfig` injected at construction
        time so the resulting script uses the correct ORCA binary and
        optional MPI / NBO / tmp-dir settings.
        """
        from autodft.qm.orca.input_generator import render_submit_script

        return render_submit_script(
            job_path=job_path,
            job_name=job_name,
            nprocs=nprocs,
            mem_per_core=mem_per_core,
            time_limit=time_limit,
            partition=partition,
            nice=nice,
            orca_path=self.orca.path,
            orca_extra_args=self.orca.extra_args,
            nbo_exe=self.orca.nbo_exe,
            mpi_env=self.orca.mpi_env,
            tmp_dir=self.orca.tmp_dir,
            keep_wavefunction=keep_wavefunction or self.orca.keep_wavefunction,
            keep_densities=self.orca.keep_densities,
            keep_hessian=keep_hessian,
            keep_cubes=keep_cubes,
            runs_nbo=runs_nbo,
            keep_mp2resp=keep_mp2resp,
            keep_esd_files=keep_esd_files,
            extra_inputs=extra_inputs,
            stall_minutes=self.orca.stall_minutes,
            sync_minutes=self.orca.sync_output_minutes,
        )

    @staticmethod
    def missing_cubes(job_path: Path) -> list[str]:
        """The cube files input.inp's ``%plots`` block asks for that are missing or empty."""
        try:
            text = (Path(job_path) / "input.inp").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        names = [name for block in _PLOTS_RE.findall(text) for name in _CUBE_FILE_RE.findall(block)]
        missing = []
        for name in names:
            path = Path(job_path) / name
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(name)
        return missing

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    @staticmethod
    def _crashed(job_path: Path) -> bool:
        """Whether a process of this job died of SIGSEGV/SIGBUS."""
        return any(CRASH_RE.search(err.read_text(encoding="utf-8", errors="replace"))
                   for err in job_path.glob("slurm-*.err"))

    @staticmethod
    def _load_output(job_path: Path) -> str:
        """Read ``output.out`` from *job_path* and return its text."""
        output_file = job_path / "output.out"
        if not output_file.exists():
            logger.error("Output file does not exist: %s", output_file)
            return "Error: Output file not found."
        return output_file.read_text(encoding="utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_electronic_energy(content: str) -> Optional[float]:
        """Extract the last ``FINAL SINGLE POINT ENERGY`` value.

        Regex: ``FINAL SINGLE POINT ENERGY\\s+(-?\\d+\\.\\d+)``

        Returns:
            Energy in Hartree, or ``None`` if not found.
        """
        energy_pattern = r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)"
        matches = list(re.finditer(energy_pattern, content))
        if matches:
            energy = float(matches[-1].group(1))
            logger.debug("Extracted electronic energy: %f Hartree", energy)
            return energy
        logger.error("Electronic energy not found in the output file.")
        return None

    @staticmethod
    def extract_free_energy_correction(content: str) -> Optional[float]:
        """Extract the free-energy correction ``G - E(el)``.

        Regex: ``G-E\\(el\\)\\s*\\.{3,}\\s*([-+]?\\d+\\.\\d+)\\s*Eh``

        Returns:
            Correction in Hartree, or ``None`` if not found.
        """
        energy_pattern = r"G-E\(el\)\s*\.{3,}\s*([-+]?\d+\.\d+)\s*Eh"
        matches = list(re.finditer(energy_pattern, content))
        if matches:
            correction = float(matches[-1].group(1))
            logger.debug(
                "Extracted free energy correction (G-E(el)): %f Hartree", correction
            )
            return correction
        logger.error("Free energy correction (G-E(el)) not found in the output file.")
        return None

    # ------------------------------------------------------------------
    # Frequency / mode extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_frequencies(content: str) -> list[float]:
        """Parse the last VIBRATIONAL FREQUENCIES block, like ``parse_ir``.

        Returns:
            List of frequencies in cm^-1 (may be empty).
        """
        frequencies: list[float] = []
        for line in _after_last(content.splitlines(), "VIBRATIONAL FREQUENCIES"):
            # Skip decorative separator lines
            if re.match(r"-+", line):
                continue
            # Match frequency lines like "  6:     123.45"
            match = re.match(r"\s*\d+:\s*([-]?\d+\.\d+)", line)
            if match:
                frequencies.append(float(match.group(1)))
            # Stop at the next section
            elif "NORMAL MODES" in line:
                break
        return frequencies

    @classmethod
    def extract_imaginary_frequencies(cls, content: str) -> list[float]:
        """Return only the negative (imaginary) vibrational frequencies.

        Args:
            content: Full text of ``output.out``.

        Returns:
            List of imaginary frequencies in cm^-1 (empty if none).
        """
        frequencies = cls._extract_frequencies(content)
        if not frequencies:
            logger.warning("No vibrational frequencies found in the output file.")
            return []

        imag = [f for f in frequencies if f < 0]
        if not imag:
            logger.info("No imaginary frequencies found.")
        else:
            logger.debug("Extracted %d imaginary frequency(ies)", len(imag))
        return imag

    @classmethod
    def extract_imaginary_modes(
        cls, content: str
    ) -> list[np.ndarray]:
        """Extract Cartesian displacement vectors for imaginary modes.

        Args:
            content: Full text of ``output.out``.

        Returns:
            List of numpy arrays with shape ``(n_atoms, 3)``, one per
            imaginary mode (ordered by frequency, most negative first).
        """
        frequencies = cls._extract_frequencies(content)
        imaginary_indices = [i for i, freq in enumerate(frequencies) if freq < 0]
        if not imaginary_indices:
            return []
        return cls._extract_modes(content, imaginary_indices)

    @staticmethod
    def _extract_modes(
        content: str, mode_indices: list[int]
    ) -> list[np.ndarray]:
        """Parse the last NORMAL MODES section for specified mode indices.

        Args:
            content: Full text of ``output.out``.
            mode_indices: 0-based indices of the modes to extract.

        Returns:
            List of numpy arrays with shape ``(n_atoms, 3)``.
        """
        mode_vectors: dict[int, list[float]] = {}
        current_modes: list[int] = []

        # The last frequency block's modes; none when it was cut short.
        for line in _after_last(_after_last(content.splitlines(), "VIBRATIONAL FREQUENCIES"), "NORMAL MODES"):
            parts = line.strip().split()

            # Header line with mode column indices (e.g. "6 7 8 9 10 11")
            if len(parts) >= 3 and all(p.isdigit() for p in parts):
                current_modes = [int(p) for p in parts]
                for idx in current_modes:
                    if idx not in mode_vectors:
                        mode_vectors[idx] = []
                continue

            # Displacement data line (starts with integer row index)
            if parts and parts[0].isdigit():
                try:
                    floats = list(map(float, parts[1:]))
                    for i, val in enumerate(floats):
                        if i < len(current_modes):
                            mode_vectors[current_modes[i]].append(val)
                except ValueError:
                    continue
            else:
                continue

        if not mode_vectors:
            return []

        # Determine atom count from the first mode vector
        sample = next(iter(mode_vectors.values()))
        if len(sample) % 3 != 0:
            raise ValueError("Mode vector length is not a multiple of 3.")
        n_atoms = len(sample) // 3

        result: list[np.ndarray] = []
        for idx in mode_indices:
            if idx not in mode_vectors:
                continue
            vec = np.array(mode_vectors[idx]).reshape((n_atoms, 3))
            result.append(vec)
        return result

    # ------------------------------------------------------------------
    # Conformer ensemble extraction (GOAT jobs)
    # ------------------------------------------------------------------

    @classmethod
    def extract_conformer_ensemble(
        cls,
        job_path: Path,
        content: str,
        max_conformers: int = 20,
    ) -> Optional[list[str]]:
        """Extract conformers from a GOAT-type calculation.

        Parses the ``# Final ensemble info #`` table in the output and
        reads the corresponding structures from
        ``input.finalensemble.xyz``.

        Args:
            job_path: Directory containing the output files.
            content: Full text of ``output.out``.
            max_conformers: Upper limit on returned conformers.

        Returns:
            List of XYZ-format strings, or ``None`` if no ensemble data
            was found.
        """
        contributions = cls._parse_ensemble_table(content)
        if not contributions:
            logger.warning("No conformers found in the output file.")
            return None

        # Already sorted by energy; keep those contributing > 0.1 %.
        used = [c[0] for c in contributions if c[1] > 0.1][:max_conformers]

        ensemble_file = job_path / "input.finalensemble.xyz"
        conformers: list[str] = []
        for conf_index in used:
            conformers.append(
                cls._get_conformer_from_ensemble(ensemble_file, conf_index)
            )
        return conformers

    @classmethod
    def extract_conformer_ensemble_energies(
        cls,
        content: str,
        max_conformers: int = 20,
    ) -> Optional[list[float]]:
        """Per-conformer energies matching :meth:`extract_conformer_ensemble`.

        Same table, same sort and same >0.1 % contribution filter, so the
        returned list is index-aligned with the conformer XYZ blocks.
        """
        contributions = cls._parse_ensemble_table(content)
        if not contributions:
            return None
        return [c[2] for c in contributions if c[1] > 0.1][:max_conformers]

    @classmethod
    def extract_conformer_ensemble_degeneracies(
        cls,
        content: str,
        max_conformers: int = 20,
    ) -> Optional[list[Optional[int]]]:
        """GOAT's per-conformer degeneracy (``Degen.``), index-aligned like the energies."""
        contributions = cls._parse_ensemble_table(content)
        if not contributions:
            return None
        return [c[3] for c in contributions if c[1] > 0.1][:max_conformers]

    @staticmethod
    def _parse_ensemble_table(content: str) -> list[list]:
        """Parse the ``# Final ensemble info #`` table into [index, %total, energy, degeneracy].

        Sorted by energy ascending. Shared by the geometry and energy
        extractors so the two can never drift out of alignment.
        """
        contributions: list[list] = []
        start_reading = False

        for line in content.splitlines():
            if "# Final ensemble info #" in line:
                start_reading = True
                continue
            if start_reading:
                if "Conformer" in line or "----" in line:
                    continue
                parts = line.strip().split()
                if len(parts) == 5 and parts[0].isdigit():
                    contributions.append(
                        [int(parts[0]), float(parts[3]), float(parts[1]), _degeneracy(parts[2])]
                    )

        contributions.sort(key=lambda x: x[2])
        return contributions

    @staticmethod
    def _get_conformer_from_ensemble(
        ensemble_file: Path, conf_index: int
    ) -> str:
        """Read a single conformer from an XYZ ensemble file.

        Args:
            ensemble_file: Path to ``input.finalensemble.xyz``.
            conf_index: 0-based conformer index.

        Returns:
            The conformer block as an XYZ-format string.
        """
        lines = ensemble_file.read_text(encoding="utf-8").splitlines(keepends=True)
        n_atoms = int(lines[0])
        start = conf_index * (n_atoms + 2)
        end = (conf_index + 1) * (n_atoms + 2)
        return "".join(lines[start:end])

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_normal_termination(content: str) -> bool:
        """Return ``True`` if ORCA terminated normally.

        Regex: ``\\*\\*\\*\\*ORCA TERMINATED NORMALLY\\*\\*\\*\\*``
        """
        pattern = r"\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*"
        return re.search(pattern, content) is not None

    @staticmethod
    def _check_for_optimization_convergence(
        content: str, task_type: str
    ) -> bool:
        """Return ``True`` if the optimisation converged (or is not applicable).

        Only meaningful when *task_type* contains ``"optimization"``.

        Regex: ``OPTIMIZATION RUN DONE``
        """
        if "optimization" not in task_type:
            return True
        pattern = r"OPTIMIZATION RUN DONE"
        if re.search(pattern, content):
            logger.info("Optimization converged successfully.")
            return True
        logger.warning("Optimization did not converge.")
        return False

    @classmethod
    def _check_for_imaginary_frequencies(
        cls, content: str, threshold: float = IMAGINARY_FREQ_THRESHOLD,
    ) -> bool:
        """Return ``True`` if there are no *significant* imaginary frequencies.

        Modes softer than ``IMAGINARY_FREQ_THRESHOLD`` are numerical noise on
        floppy rotors, not saddle points. Failing on those meant a -42 cm^-1
        mode burned the full retry budget exactly like a -230 cm^-1 one.
        """
        # A singlepoint has no such block at all; nothing to warn about.
        if "VIBRATIONAL FREQUENCIES" not in content:
            return True
        significant = [
            f for f in cls.extract_imaginary_frequencies(content)
            if f < threshold
        ]
        if significant:
            logger.warning("Significant imaginary frequencies: %s", significant)
        return len(significant) == 0

    @staticmethod
    def _check_scf_convergence(content: str) -> bool:
        """Return ``False`` when ORCA reports the SCF failing to converge.

        Was hardcoded to ``True``, so a non-converged SCF passed every check
        and its last FINAL SINGLE POINT ENERGY was extracted as if valid.
        """
        for marker in (
            "SCF NOT CONVERGED AFTER",
            "SCF CONVERGENCE FAILED",
            "This wavefunction IS NOT FULLY CONVERGED",
        ):
            if marker.lower() in content.lower():
                logger.warning("SCF convergence failure detected: %r", marker)
                return False
        return True

    @staticmethod
    def _check_no_error_banner(content: str) -> bool:
        """Return ``False`` when ORCA printed a fatal error.

        ORCA can print ``****ORCA TERMINATED NORMALLY****`` after a module
        has already aborted -- a real GOAT run in this project ended with
        ``GOAT ERROR: No structure was left after collecting the data!`` and
        still terminated "normally" in 4.5 s, and was recorded as a success
        with no downstream calculations.
        """
        lowered = content.lower()

        # orca_plot can't build a spin density for a closed-shell reference and
        # aborts with this banner, but the calculation itself is fine -- only
        # this module's error termination is ignored.
        plot_error = "orca finished by error termination in plot"
        if plot_error in lowered:
            lowered = lowered.replace(plot_error, "")

        for marker in (
            "goat error",
            "orca finished by error termination",
            "orca terminated abnormally",
        ):
            if marker in lowered:
                logger.warning("ORCA error banner detected: %r", marker)
                return False

        # "Aborting the run" is also printed by the internal-coordinate
        # optimiser, which then recovers with "Trying Cartesian step" -- on
        # its own it would fail healthy jobs. Only treat it as fatal when the
        # optimiser really did give up.
        if "aborting the run" in lowered and "geometry optimization failed" in lowered:
            logger.warning("ORCA aborted the run and the optimisation failed")
            return False
        return True


def _degeneracy(text: str) -> Optional[int]:
    """GOAT's ``Degen.`` entry; None when it is not a positive number."""
    try:
        value = int(float(text))
    except (ValueError, OverflowError):
        return None
    return value if value >= 1 else None
