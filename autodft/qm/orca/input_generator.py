"""Generate ORCA input files and SLURM submit scripts using Jinja2 templates.

All paths are handled via :class:`pathlib.Path`.  Templates live in
``autodft/qm/templates/``.
"""

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

__all__ = [
    "generate_orca_input",
    "generate_submit_script",
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


def generate_orca_input(
    job_path: Path,
    header_text: str,
    charge: int,
    multiplicity: int,
    xyz_data: str,
) -> Path:
    """Render ``orca_input.inp.j2`` and write ``input.inp`` into *job_path*.

    Args:
        job_path: Target directory (will be created if it does not exist).
        header_text: ORCA header / route section (everything before the
                     geometry specification, typically starting with ``!``).
        charge: Molecular charge.
        multiplicity: Spin multiplicity.
        xyz_data: XYZ geometry data.  Only needed for ``write_xyz_file``;
                  the template uses ``*xyzfile`` to reference ``input.xyz``.

    Returns:
        Path to the written ``input.inp``.
    """
    job_path = Path(job_path)
    job_path.mkdir(parents=True, exist_ok=True)

    env = _get_jinja_env()
    template = env.get_template("orca_input.inp.j2")
    rendered = template.render(
        header_text=header_text.rstrip(),
        charge=charge,
        multiplicity=multiplicity,
    )

    input_path = job_path / "input.inp"
    input_path.write_text(rendered, encoding="utf-8")
    logger.debug("Wrote ORCA input file: %s", input_path)
    return input_path


# ------------------------------------------------------------------
# SLURM submit script
# ------------------------------------------------------------------


def generate_submit_script(
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
    tmp_dir: str = "/tmp",
    keep_wavefunction: bool = False,
    keep_densities: bool = False,
) -> Path:
    """Render ``submit.cmd.j2`` and write ``submit.cmd`` into *job_path*.

    Args:
        job_path: Target directory.
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
        tmp_dir: Parent directory for the per-job scratch directory.
                 Empty string disables the TMP_DIR copy-out pattern and
                 runs ORCA directly inside ``job_path``.
        keep_wavefunction: Copy ``*.gbw`` back from scratch when True.
        keep_densities: Copy ``*.densities``/``*.densitiesinfo``/``*.cube``
                        back from scratch when True.

    Returns:
        Path to the written ``submit.cmd``.
    """
    job_path = Path(job_path)
    job_path.mkdir(parents=True, exist_ok=True)

    total_mem = nprocs * (mem_per_core + 50)

    env = _get_jinja_env()
    template = env.get_template("submit.cmd.j2")
    rendered = template.render(
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
        tmp_dir=tmp_dir,
        keep_wavefunction=keep_wavefunction,
        keep_densities=keep_densities,
    )

    submit_path = job_path / "submit.cmd"
    submit_path.write_text(rendered, encoding="utf-8")
    logger.debug("Wrote SLURM submit script: %s", submit_path)
    return submit_path


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
