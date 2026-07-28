"""ORCA output file parser.

Reads ``output.out`` (and optionally ``input.finalensemble.xyz``) produced
by an ORCA calculation and extracts energies, frequencies, normal modes,
and conformer ensembles.  All regex patterns are preserved from the
original ``orca_output_processor.py``.
"""

import logging
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

    def check_output(self, job_path: Path, task_type: str) -> QMResult:
        """Run all checks on an ORCA output and return a :class:`QMResult`.

        Args:
            job_path: Directory containing ``output.out``.
            task_type: E.g. ``"optimization"``, ``"singlepoint"``,
                       ``"confsearch"``.

        Returns:
            Populated :class:`QMResult`.
        """
        job_path = Path(job_path)
        content = self._load_output(job_path)

        checks: dict[str, bool] = {
            "Termination": self._check_normal_termination(content),
            "Optimization Convergence": self._check_for_optimization_convergence(
                content, task_type
            ),
            "SCF Convergence": self._check_scf_convergence(content),
            "Imaginary Frequencies": self._check_for_imaginary_frequencies(content),
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

        success = all(checks.values())

        energy = self.extract_electronic_energy(content)
        free_energy_correction = self.extract_free_energy_correction(content)

        conformers: Optional[list[str]] = None
        conformer_energies: Optional[list[float]] = None
        if task_type == "confsearch":
            conformers = self.extract_conformer_ensemble(job_path, content)
            conformer_energies = self.extract_conformer_ensemble_energies(content)

        return QMResult(
            success=success,
            checks=checks,
            energy=energy,
            free_energy_correction=free_energy_correction,
            conformers=conformers,
            conformer_energies=conformer_energies,
        )

    def generate_input(
        self,
        job_path: Path,
        header: str,
        charge: int,
        multiplicity: int,
        xyz_data: str,
    ) -> None:
        """Delegated to :func:`input_generator.generate_orca_input`."""
        from autodft.qm.orca.input_generator import generate_orca_input, write_xyz_file

        generate_orca_input(job_path, header, charge, multiplicity, xyz_data)
        write_xyz_file(job_path, xyz_data)

    def generate_submit_script(
        self,
        job_path: Path,
        job_name: str,
        nprocs: int,
        mem_per_core: int,
        time_limit: str,
        partition: str,
        nice: int = 0,
    ) -> None:
        """Delegated to :func:`input_generator.generate_submit_script`.

        Threads through the :class:`OrcaConfig` injected at construction
        time so the resulting script uses the correct ORCA binary and
        optional MPI / NBO / tmp-dir settings.
        """
        from autodft.qm.orca.input_generator import generate_submit_script

        generate_submit_script(
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
            tmp_dir=self.orca.tmp_dir,
            keep_wavefunction=self.orca.keep_wavefunction,
            keep_densities=self.orca.keep_densities,
        )

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

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
        """Parse the VIBRATIONAL FREQUENCIES block.

        Returns:
            List of frequencies in cm^-1 (may be empty).
        """
        frequencies: list[float] = []
        in_freq_block = False
        for line in content.splitlines():
            if "VIBRATIONAL FREQUENCIES" in line:
                in_freq_block = True
                continue
            if in_freq_block:
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
        """Parse the NORMAL MODES section for specified mode indices.

        Args:
            content: Full text of ``output.out``.
            mode_indices: 0-based indices of the modes to extract.

        Returns:
            List of numpy arrays with shape ``(n_atoms, 3)``.
        """
        lines = content.splitlines()
        reading = False
        mode_vectors: dict[int, list[float]] = {}
        current_modes: list[int] = []

        for line in lines:
            if "NORMAL MODES" in line:
                reading = True
                continue
            if not reading:
                continue

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
        contributions: list[list] = []  # [index, %total, energy]
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
                    index = int(parts[0])
                    energy = float(parts[1])
                    total_pct = float(parts[3])
                    contributions.append([index, total_pct, energy])

        if not contributions:
            logger.warning("No conformers found in the output file.")
            return None

        # Sort by energy (ascending) and filter by contribution > 0.1%
        contributions.sort(key=lambda x: x[2])
        used = [c[0] for c in contributions if c[1] > 0.1]
        if len(used) > max_conformers:
            used = used[:max_conformers]

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

    @staticmethod
    def _parse_ensemble_table(content: str) -> list[list]:
        """Parse the ``# Final ensemble info #`` table into [index, %total, energy].

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
                        [int(parts[0]), float(parts[3]), float(parts[1])]
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
    def _check_for_imaginary_frequencies(cls, content: str) -> bool:
        """Return ``True`` if there are no *significant* imaginary frequencies.

        Modes softer than ``IMAGINARY_FREQ_THRESHOLD`` are numerical noise on
        floppy rotors, not saddle points. Failing on those meant a -42 cm^-1
        mode burned the full retry budget exactly like a -230 cm^-1 one.
        """
        significant = [
            f for f in cls.extract_imaginary_frequencies(content)
            if f < IMAGINARY_FREQ_THRESHOLD
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
