"""Abstract base class for quantum mechanics engines.

Defines the interface that every QM backend (ORCA, Gaussian, xTB, ...)
must implement so the rest of autodft can remain engine-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class QMResult:
    """Container for the outcome of a QM calculation check.

    Attributes:
        success: Whether every individual check passed.
        checks: Mapping of check name to pass/fail bool, e.g.
                 ``{"Termination": True, "Imaginary Frequencies": False}``.
        energy: Final electronic energy in Hartree (if available).
        free_energy_correction: G - E(el) correction in Hartree (if available).
        conformers: List of XYZ-format strings for conformer ensemble
                    results (GOAT-type jobs).
        conformer_energies: Per-conformer energies in Hartree, index-aligned
                    with ``conformers``. Without these every conformer
                    geometry is stored with the same scalar energy, which
                    makes the "keep the N lowest" ORDER BY an arbitrary tie.
        conformer_degeneracies: GOAT's degeneracy per conformer, index-aligned
                    with ``conformers``.
    """

    success: bool
    checks: dict[str, bool] = field(default_factory=dict)
    energy: Optional[float] = None
    free_energy_correction: Optional[float] = None
    conformers: Optional[list[str]] = None
    conformer_energies: Optional[list[float]] = None
    conformer_degeneracies: Optional[list[Optional[int]]] = None


class QMEngine(ABC):
    """Interface that every QM engine must satisfy."""

    @abstractmethod
    def check_output(
        self, job_path: Path, task_type: str, imag_freq_tolerance: Optional[float] = None,
    ) -> QMResult:
        """Analyse a completed calculation and return structured results.

        Args:
            job_path: Directory that contains the program output files.
            task_type: One of ``"optimization"``, ``"singlepoint"``,
                       ``"confsearch"``, etc.
            imag_freq_tolerance: An optimisation fails on any mode below it (cm^-1).

        Returns:
            A :class:`QMResult` summarising success and extracted data.
        """
        ...

    @abstractmethod
    def render_input(self, header: str, charge: int, multiplicity: int, trailer: str = "") -> str:
        """The program-specific input file's text; the geometry is read from ``input.xyz``.

        Args:
            header: Engine-specific header / route section.
            charge: Molecular charge.
            multiplicity: Spin multiplicity.
            trailer: Input that must follow the geometry.
        """
        ...

    @abstractmethod
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
        """The text of a cluster submit script (e.g. SLURM) for a job in *job_path*.

        Args:
            job_path: Target directory.
            job_name: Human-readable job name for the scheduler.
            nprocs: Number of CPU cores to request.
            mem_per_core: Memory per core in MB.
            time_limit: Wall-time string understood by the scheduler
                        (e.g. ``"2-00:00:00"``).
            partition: Scheduler partition / queue name.
            keep_hessian: Copy ``*.hess`` back from scratch when True.
            keep_cubes: Copy ``*.cube`` back from scratch when True.
            runs_nbo: The job runs NBO, which needs an unlimited stack.
            keep_mp2resp: Copy an NMR job's MP2 response logs back from scratch when True.
            keep_esd_files: Copy ESD's derivative Hessians, spectra and
                            correlation functions back from scratch when True.
            extra_inputs: Files in *job_path* to stage into scratch next to
                          the input file.
            keep_wavefunction: Copy ``*.gbw`` back from scratch when True.
        """
        ...
