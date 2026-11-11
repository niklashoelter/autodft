"""Render ORCA input files and SLURM submit scripts from Jinja2 templates.

All paths are handled via :class:`pathlib.Path`.  Templates live in
``autodft/qm/templates/``.
"""

import logging
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

__all__ = [
    "render_orca_input",
    "render_submit_script",
    "write_xyz_file",
]

logger = logging.getLogger(__name__)

# Resolve the templates directory relative to this file.
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def _get_jinja_env() -> Environment:
    """Return a Jinja2 :class:`Environment` pointing at the template dir."""
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
    )


# ------------------------------------------------------------------
# ORCA input file
# ------------------------------------------------------------------


def render_orca_input(header_text: str, charge: int, multiplicity: int, trailer: str = "") -> str:
    """The text of ``input.inp``; *trailer* follows the geometry."""
    return _get_jinja_env().get_template("orca_input.inp.j2").render(
        header_text=header_text.rstrip(),
        charge=charge,
        multiplicity=multiplicity,
        trailer=trailer,
    )


# ------------------------------------------------------------------
# SLURM submit script
# ------------------------------------------------------------------


def render_submit_script(
    job_path: Path,
    job_name: str,
    nprocs: int,
    mem_per_core: int,
    time_limit: str,
    partition: str,
    nice: int = 0,
    orca_path: str = "orca",
    orca_extra_args: str = "",
    nbo_exe: str | None = None,
    mpi_env: str = "",
    tmp_dir: str = "/tmp",
    keep_wavefunction: bool = False,
    keep_densities: bool = False,
    keep_hessian: bool = False,
    keep_cubes: bool = False,
    runs_nbo: bool = False,
    keep_mp2resp: bool = False,
    keep_esd_files: bool = False,
    extra_inputs: Optional[list[str]] = None,
    stall_minutes: int = 0,
    sync_minutes: int = 0,
) -> str:
    """The text of ``submit.cmd``.

    Args:
        job_path: The job directory the script runs in.
        job_name: SLURM job name (``--job-name``).
        nprocs: Number of tasks / cores (``--ntasks-per-node``).
        mem_per_core: Memory per core in MB. Total memory is computed as
                      ``nprocs * (mem_per_core + 50)`` to leave a small
                      buffer for ORCA overhead.
        time_limit: Wall-time string, e.g. ``"2-00:00:00"``.
        partition: SLURM partition name.
        nice: Optional SLURM nice value (skipped when 0).
        orca_path: Path to the ORCA binary (absolute path, or ``"orca"``
                   when a module system puts it on PATH).
        orca_extra_args: String passed as ORCA's second argument
                         (e.g. ``"--bind-to none"`` for MPI binding).
        nbo_exe: Optional NBO executable path; exported as NBOEXE.
        mpi_env: Optional script sourced before ORCA runs.
        tmp_dir: Parent directory for the per-job scratch directory.
                 Empty string disables the TMP_DIR copy-out pattern and
                 runs ORCA directly inside ``job_path``.
        keep_wavefunction: Copy ``*.gbw`` back from scratch when True.
        keep_densities: Copy ``*.densities``/``*.densitiesinfo``/``*.cube``
                        back from scratch when True.
        keep_hessian: Copy ``*.hess`` back from scratch when True.
        keep_cubes: Copy ``*.cube`` back from scratch when True.
        runs_nbo: The job runs NBO, which needs an unlimited stack.
        keep_mp2resp: Copy an NMR job's MP2 response logs back from scratch when True.
        keep_esd_files: Copy ESD's derivative Hessians, spectra and correlation
                        functions back from scratch when True.
        extra_inputs: Files in *job_path* to stage into scratch next to
                      ``input.inp``.
        stall_minutes: Stall watchdog threshold; 0 runs ORCA in the foreground.
        sync_minutes: Copy output.out back this often while the watchdog runs.
    """
    total_mem = nprocs * (mem_per_core + 50)

    env = _get_jinja_env()
    template = env.get_template("submit.cmd.j2")
    return template.render(
        job_name=job_name,
        job_path=str(job_path),
        nprocs=nprocs,
        total_mem=total_mem,
        time_limit=time_limit,
        partition=partition,
        nice=nice,
        orca_path=orca_path,
        orca_extra_args=orca_extra_args,
        nbo_exe=nbo_exe,
        mpi_env=mpi_env,
        tmp_dir=tmp_dir,
        keep_wavefunction=keep_wavefunction,
        keep_densities=keep_densities,
        keep_hessian=keep_hessian,
        keep_cubes=keep_cubes,
        runs_nbo=runs_nbo,
        keep_mp2resp=keep_mp2resp,
        keep_esd_files=keep_esd_files,
        extra_inputs=extra_inputs or [],
        stall_minutes=stall_minutes,
        sync_minutes=sync_minutes,
    )


# ------------------------------------------------------------------
# XYZ file
# ------------------------------------------------------------------


def write_xyz_file(job_path: Path, xyz_data: str) -> Path:
    """Write ``input.xyz`` into *job_path*.

    Args:
        job_path: Target directory.
        xyz_data: Full XYZ-format string (including the atom-count and
                  comment lines).

    Returns:
        Path to the written ``input.xyz``.
    """
    job_path = Path(job_path)
    job_path.mkdir(parents=True, exist_ok=True)

    xyz_path = job_path / "input.xyz"
    xyz_path.write_text(xyz_data, encoding="utf-8")
    logger.debug("Wrote XYZ file: %s", xyz_path)
    return xyz_path
