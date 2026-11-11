"""Submit molecules to the AutoDFT pipeline."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from autodft.cli.options import required_config_option
from autodft.db import get_session, init_db
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.qm.orca.defaults import (
    DEFAULT_HEADER_CONFSEARCH,
    DEFAULT_HEADER_OPTIMIZATION,
    DEFAULT_HEADER_SINGLEPOINT,
)
from autodft.qm.orca.parser import IMAG_FREQ_TOLERANCE_MIN

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(name="submit", help="Submit molecules for DFT calculations.")


def _read_header_file(path: Optional[Path]) -> Optional[str]:
    """Read header text from a file, or return None."""
    if path is None:
        return None
    if not path.exists():
        console.print(f"[red]Header file not found:[/red] {path}")
        raise typer.Exit(code=1)
    return path.read_text(encoding="utf-8")


def _check_reference_state(
    smiles: str, request_t1: bool, request_ox: bool = False,
    request_red: bool = False,
) -> None:
    """Refuse extra states the reference cannot support.

    A diradical is only calculated as a triplet; the S0 -> T1 spin change is
    only defined from a closed-shell singlet. Mirrors the guards in
    POST /api/submit so both entry paths behave the same.
    """
    from autodft.engine.entrypoint_processor import validate_smiles

    check = validate_smiles(smiles)
    if not check["valid"]:
        return

    if check["diradical"]:
        extra = [flag for flag, on in (
            ("--request-t1", request_t1),
            ("--request-ox", request_ox),
            ("--request-red", request_red),
        ) if on]
        if extra:
            console.print(
                f"[red]Diradicals are only calculated in the triplet state[/red] "
                f"— {smiles}. Drop {', '.join(extra)}."
            )
            raise typer.Exit(code=1)
    elif request_t1 and check["multiplicity"] != 1:
        console.print(
            f"[red]T1 requires a closed-shell reference[/red] — {smiles} has "
            f"multiplicity {check['multiplicity']}. Drop --request-t1; "
            f"ox / red still work for open-shell references."
        )
        raise typer.Exit(code=1)

    if check.get("warning"):
        console.print(f"[yellow]{check['warning']}[/yellow]")


def _category_options_to_flags(
    uvvis: bool, ir: bool, esd: bool, esd_ht: bool, nmr: bool,
    uvvis_nroots: int = 20, uvvis_tda: bool = False, nmr_nuclei: Optional[list] = None,
    esd_tn_window_ev: float = 0.3, esd_temperature_k: float = 298.15,
    nmr_method: str = "singlepoint", esd_ic: bool = False,
    nmr_dh_nprocs: int = 12, nmr_dh_maxcore: int = 6000, nmr_reference: str = "chcl3",
    esd_ht_scope: str = "all", esd_ht_method: str = "sp",
    densities: bool = False, density_eldens: bool = True, density_spindens: bool = True,
    density_grid: int = 75, density_eldens_file: str = "ElDens.cube",
    density_spindens_file: str = "SpinDens.cube",
    nbo: bool = False, nbo_keywords: str = "", reactivity: bool = False,
) -> dict:
    """CLI options as ``request_metadata`` category keys and settings."""
    from autodft import categories

    return {
        categories.UVVIS: uvvis,
        categories.IR: ir,
        categories.ESD: esd,
        categories.ESD_HT: esd_ht,
        categories.NMR: nmr,
        categories.DENSITIES: densities,
        categories.NBO: nbo,
        categories.REACTIVITY: reactivity,
        "uvvis_nroots": uvvis_nroots,
        "uvvis_tda": uvvis_tda,
        "nmr_nuclei": nmr_nuclei if nmr_nuclei is not None else ["H", "C", "F"],
        "esd_tn_window_ev": esd_tn_window_ev,
        "esd_temperature_k": esd_temperature_k,
        "nmr_method": nmr_method,
        "nmr_dh_nprocs": nmr_dh_nprocs,
        "nmr_dh_maxcore": nmr_dh_maxcore,
        "nmr_reference": nmr_reference,
        "esd_ic": esd_ic,
        "esd_ht_scope": esd_ht_scope,
        "esd_ht_method": esd_ht_method,
        "density_eldens": density_eldens,
        "density_spindens": density_spindens,
        "density_grid": density_grid,
        "density_eldens_file": density_eldens_file,
        "density_spindens_file": density_spindens_file,
        "nbo_keywords": nbo_keywords,
    }


# --<cs|opt|sp>-<setting> options of submit and submit-batch.
_ADVANCED_OPTIONS = (
    "opt_grid", "sp_grid", "opt_scf_convergence", "sp_scf_convergence", "opt_geom_convergence", "opt_ri",
    "sp_ri", "opt_aux_basis", "sp_aux_basis", "opt_solvation", "sp_solvation", "opt_solvent", "sp_solvent",
    "cs_nprocs", "opt_nprocs", "sp_nprocs", "cs_maxcore", "opt_maxcore", "sp_maxcore",
)
_STAGE_PREFIXES = {"cs": "confsearch", "opt": "optimization", "sp": "singlepoint"}


def _advanced_from_options(options: dict) -> Optional[dict]:
    """``advanced`` from the Advanced options given, or None if none was."""
    advanced: dict = {}
    for name in _ADVANCED_OPTIONS:
        if options[name] is not None:
            prefix, _, key = name.partition("_")
            advanced.setdefault(_STAGE_PREFIXES[prefix], {})[key] = options[name]
    return advanced or None


def _submission_advanced(project: str, headers: tuple, requested: Optional[dict], settings) -> Optional[dict]:
    """The resolved ``advanced`` (None in legacy mode); mirrors POST /api/submit."""
    from autodft.engine.entrypoint_processor import legacy_submission
    from autodft.qm.orca import advanced

    if requested is None:
        with get_session() as session:
            if legacy_submission(session, project, headers):
                return None
    resolved, error = advanced.for_submission(requested, headers, settings)
    if error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1)
    return resolved


def _check_categories(
    smiles: str, flags: dict,
    header_optimization: Optional[str], header_singlepoint: Optional[str],
    settings, advanced: Optional[dict] = None,
) -> None:
    """Refuse categories this molecule or these headers cannot run.

    Mirrors POST /api/submit. Adding categories to an existing molecule is
    refused at expansion, where the entrypoint then shows the reason.
    """
    from autodft import categories

    if not categories.requested(flags) and not flags.get(categories.ESD_HT):
        return

    from autodft.engine.entrypoint_processor import validate_smiles

    check = validate_smiles(smiles)
    if not check["valid"]:
        return
    reason = categories.rejection(check, flags, header_optimization, header_singlepoint, advanced)
    if reason:
        console.print(f"[red]{reason}[/red] — {smiles}")
        raise typer.Exit(code=1)
    if categories.NBO_NEEDED & categories.requested(flags):
        unavailable = categories.nbo_unavailable(settings)
        if unavailable:
            console.print(f"[red]{unavailable}[/red] — {smiles}")
            raise typer.Exit(code=1)


def _qualified_project(project: str, user: str) -> tuple[str, str]:
    """Resolve ``--project`` against ``--user``'s namespace.

    The CLI writes entrypoints straight to the database, so it has no
    request and no API key to identify it. Without this it would create
    rows under a bare project name that no longer resolves to an owner,
    leaving the work visible only to admin. The project is created if it
    does not exist, exactly as an API submission would.
    """
    from autodft.engine import nmr_references

    reserved = nmr_references.reserved_name_error(project)
    if reserved:
        console.print(f"[red]{reserved}[/red]")
        raise typer.Exit(code=1)

    from autodft import accounts
    from autodft.db import get_session

    with get_session() as session:
        account = accounts.get_user_by_username(session, user)
        if account is None:
            console.print(
                f"[red]No user named {user!r}. Create one on the admin page, "
                f"or pass --user with an existing name.[/red]"
            )
            raise typer.Exit(code=1)
        return accounts.get_or_create_project(session, account, project).qualified_name, account.username


def _build_request_metadata(
    project_name: str,
    project_author: str,
    request_t1: bool,
    request_ox: bool,
    request_red: bool,
    skip_confsearch: bool,
    request_vert_ex: bool,
    max_conformers_s0: int,
    max_conformers_t1: int,
    max_conformers_ox: int,
    max_conformers_red: int,
    category_flags: Optional[dict] = None,
    imag_freq_tolerance: float = 0.0,
    advanced: Optional[dict] = None,
    seed_from_s0: bool = False,
) -> str:
    """Build the request_metadata JSON string."""
    metadata = {
        "project_name": project_name,
        "project_author": project_author,
        "request_S1": False,
        "request_T1": request_t1,
        "request_ox": request_ox,
        "request_red": request_red,
        "request_confsearch": not skip_confsearch,
        "request_optimization": True,
        "request_singlepoint": True,
        "request_singlepoint_vertical_excitations": request_vert_ex,
        "request_singlepoint_nbo": False,
        "max_conformers_S0": max_conformers_s0,
        "max_conformers_T1": max_conformers_t1,
        "max_conformers_ox": max_conformers_ox,
        "max_conformers_red": max_conformers_red,
        "imag_freq_tolerance": imag_freq_tolerance,
    }

    if seed_from_s0:
        metadata["seed_from_s0"] = True

    from autodft import categories

    metadata.update(categories.snapshot(category_flags or {}))
    if advanced is not None:
        metadata["advanced"] = advanced

    return json.dumps(metadata)


@app.command()
def submit(
    smiles: str = typer.Option(..., "--smiles", help="SMILES string of the molecule"),
    project: str = typer.Option(..., "--project", help="Project name"),
    user: str = typer.Option(
        "admin", "--user",
        help="Account to submit as; the project lands in their namespace",
    ),
    priority: int = typer.Option(10, "--priority", help="Queue priority (higher = more urgent)"),
    request_t1: bool = typer.Option(False, "--request-t1", help="Request T1 state calculations"),
    request_ox: bool = typer.Option(False, "--request-ox", help="Request oxidation state calculations"),
    request_red: bool = typer.Option(False, "--request-red", help="Request reduction state calculations"),
    skip_confsearch: bool = typer.Option(
        False, "--skip-confsearch",
        help="Skip conformer search; use RDKit geometry directly for optimization",
    ),
    seed_from_s0: bool = typer.Option(
        False, "--seed-from-s0",
        help="T1/ox/red start from S0's lowest conformer once S0 is finished, without their own conformer search",
    ),
    no_vert_ex: bool = typer.Option(
        False, "--no-vert-ex",
        help="Disable singlepoint vertical excitations (ox/red/spin-flip). On by default.",
    ),
    header_confsearch: Optional[Path] = typer.Option(
        None, "--header-confsearch", help="ORCA header file for conformer search (default: GOAT g-xTB)"
    ),
    header_opt: Optional[Path] = typer.Option(
        None, "--header-opt", help="ORCA header file for optimization, method only (default: wB97X-D3 / def2-SVP)"
    ),
    header_sp: Optional[Path] = typer.Option(
        None, "--header-sp", help="ORCA header file for singlepoint, method only (default: wB97X-D3 / def2-TZVP)"
    ),
    max_conformers_s0: int = typer.Option(1, "--max-conformers-s0", help="Max conformers kept for S0"),
    max_conformers_t1: int = typer.Option(1, "--max-conformers-t1", help="Max conformers kept for T1"),
    max_conformers_ox: int = typer.Option(1, "--max-conformers-ox", help="Max conformers kept for ox"),
    max_conformers_red: int = typer.Option(1, "--max-conformers-red", help="Max conformers kept for red"),
    uvvis: bool = typer.Option(False, "--uvvis", help="UV/Vis absorption (TDDFT) on every S0 conformer"),
    ir: bool = typer.Option(False, "--ir", help="IR spectrum from the optimisation's frequencies"),
    esd: bool = typer.Option(
        False, "--esd",
        help="Excited-state dynamics: ISC/RISC/fluorescence/phosphorescence rates (IC with --esd-ic) from S1 and T1 seeded at the lowest S0",
    ),
    esd_ht: bool = typer.Option(
        False, "--esd-ht", help="Herzberg-Teller for fluorescence, phosphorescence and ISC channels with SOCME below 5 cm-1 (much more expensive: 3N-6 or more displaced TDDFT+SOC singlepoints per triplet sublevel; hours to days)"
    ),
    nmr: bool = typer.Option(
        False, "--nmr", help="NMR shifts (1H/13C/19F) vs automatically computed TMS / CFCl3"
    ),
    uvvis_nroots: int = typer.Option(20, "--uvvis-nroots", help="UV/Vis excited states (1-100)"),
    uvvis_tda: bool = typer.Option(False, "--uvvis-tda", help="Tamm-Dancoff approximation for UV/Vis"),
    nmr_nuclei: str = typer.Option(
        "H,C,F", "--nmr-nuclei", help="NMR nuclei to report, comma-separated (H, C, F, As; As: raw shielding and EFG)"
    ),
    esd_tn_window_ev: float = typer.Option(0.3, "--esd-tn-window", help="Include S1->Tn ISC for Tn up to this many eV above S1"),
    esd_temperature_k: float = typer.Option(298.15, "--esd-temperature", help="Temperature of the ESD rates (K)"),
    esd_ic: bool = typer.Option(
        False, "--esd-ic", help="Also compute the S1->S0 internal conversion rate (harmonic, often unreliable for large gaps)"
    ),
    esd_ht_scope: str = typer.Option("all", "--esd-ht-scope", help="With --esd-ht: all (every rate) or s1 (fluorescence and S1->Tn ISC only)"),
    esd_ht_method: str = typer.Option("sp", "--esd-ht-method", help="With --esd-ht: sp (singlepoint header level) or opt (optimisation header level, cheaper)"),
    imag_freq_tolerance: float = typer.Option(
        0.0, "--imag-freq-tolerance", min=IMAG_FREQ_TOLERANCE_MIN, max=0.0,
        help="Fail optimisations with an imaginary mode below this (cm-1, -300 to 0); e.g. -10 accepts noise modes",
    ),
    nmr_method: str = typer.Option(
        "singlepoint", "--nmr-method",
        help="NMR method: singlepoint (the singlepoint header's) or dh (revDSD-PBEP86-D4/2021 / pcSseg-3; H-Kr only)",
    ),
    nmr_dh_nprocs: int = typer.Option(12, "--nmr-dh-nprocs", help="Cores of a dh NMR job (recommended 12)"),
    nmr_dh_maxcore: int = typer.Option(6000, "--nmr-dh-maxcore", help="MB per core of a dh NMR job (recommended 6000)"),
    nmr_reference: str = typer.Option("chcl3", "--nmr-reference", help="1H/13C reference: chcl3 (default) or tms; 19F always CFCl3"),
    densities: bool = typer.Option(False, "--densities", help="Electron / spin density cube files from every state's energy singlepoint"),
    no_eldens: bool = typer.Option(False, "--no-eldens", help="Skip the electron density cube"),
    no_spindens: bool = typer.Option(False, "--no-spindens", help="Skip the spin density cube (open-shell jobs only)"),
    density_grid: int = typer.Option(75, "--density-grid", help="Cube grid points per axis (10-300)"),
    eldens_file: str = typer.Option("ElDens.cube", "--eldens-file", help="Electron density cube file name"),
    spindens_file: str = typer.Option("SpinDens.cube", "--spindens-file", help="Spin density cube file name"),
    nbo: bool = typer.Option(False, "--nbo", help="NBO 7 inside every state's energy singlepoint"),
    nbo_keywords: str = typer.Option("", "--nbo-keywords", help='Extra $NBO keywords, e.g. "BNDIDX"'),
    reactivity: bool = typer.Option(False, "--reactivity", help="Hirshfeld, %elprop polarizability/quadrupole, NBO with Wiberg indices and kept .gbw in every relaxed and vertical singlepoint"),
    opt_grid: Optional[str] = typer.Option(None, "--opt-grid", help="Optimisation grid: DEFGRID1-3 or omit (default: config [advanced])"),
    sp_grid: Optional[str] = typer.Option(None, "--sp-grid", help="Singlepoint grid: DEFGRID1-3 or omit (default: config [advanced])"),
    opt_scf_convergence: Optional[str] = typer.Option(None, "--opt-scf-convergence", help="Optimisation SCF convergence: Normal, Strong, Tight, VeryTight, Extreme or omit (default: config [advanced])"),
    sp_scf_convergence: Optional[str] = typer.Option(None, "--sp-scf-convergence", help="Singlepoint SCF convergence: Normal, Strong, Tight, VeryTight, Extreme or omit (default: config [advanced])"),
    opt_geom_convergence: Optional[str] = typer.Option(None, "--opt-geom-convergence", help="Geometry convergence: Loose, Normal, Tight, VeryTight or omit (default: config [advanced])"),
    opt_ri: Optional[str] = typer.Option(None, "--opt-ri", help="Optimisation RI: RIJCOSX, RI-J, RIJK, NoRI or omit (default: config [advanced])"),
    sp_ri: Optional[str] = typer.Option(None, "--sp-ri", help="Singlepoint RI: RIJCOSX, RI-J, RIJK, NoRI or omit (default: config [advanced])"),
    opt_aux_basis: Optional[str] = typer.Option(None, "--opt-aux-basis", help="Optimisation auxiliary basis: def2/J, def2/JK, AutoAux or omit (default: config [advanced])"),
    sp_aux_basis: Optional[str] = typer.Option(None, "--sp-aux-basis", help="Singlepoint auxiliary basis: def2/J, def2/JK, AutoAux or omit (default: config [advanced])"),
    opt_solvation: Optional[str] = typer.Option(None, "--opt-solvation", help="Optimisation solvation: CPCM, SMD or omit (default: config [advanced])"),
    sp_solvation: Optional[str] = typer.Option(None, "--sp-solvation", help="Singlepoint solvation: CPCM, SMD or omit (default: config [advanced])"),
    opt_solvent: Optional[str] = typer.Option(None, "--opt-solvent", help="Optimisation solvent (ORCA name) (default: config [advanced])"),
    sp_solvent: Optional[str] = typer.Option(None, "--sp-solvent", help="Singlepoint solvent (ORCA name) (default: config [advanced])"),
    cs_nprocs: Optional[int] = typer.Option(None, "--cs-nprocs", help="Conformer search cores (1-64) (default: config [advanced])"),
    opt_nprocs: Optional[int] = typer.Option(None, "--opt-nprocs", help="Optimisation cores (1-64) (default: config [advanced])"),
    sp_nprocs: Optional[int] = typer.Option(None, "--sp-nprocs", help="Singlepoint cores (1-64) (default: config [advanced])"),
    cs_maxcore: Optional[int] = typer.Option(None, "--cs-maxcore", help="Conformer search MB per core (250-16000) (default: config [advanced])"),
    opt_maxcore: Optional[int] = typer.Option(None, "--opt-maxcore", help="Optimisation MB per core (250-16000) (default: config [advanced])"),
    sp_maxcore: Optional[int] = typer.Option(None, "--sp-maxcore", help="Singlepoint MB per core (250-16000) (default: config [advanced])"),
    config: str = required_config_option(),
) -> None:
    """Submit a single molecule by SMILES string."""
    options = dict(locals())  # the parameters, for _advanced_from_options
    from autodft.cli.options import cli_settings

    settings = cli_settings(config)
    init_db(settings)

    _check_reference_state(smiles, request_t1, request_ox, request_red)
    flags = _category_options_to_flags(
        uvvis, ir, esd, esd_ht, nmr, uvvis_nroots, uvvis_tda,
        nmr_nuclei=[n.strip() for n in nmr_nuclei.split(",") if n.strip()],
        esd_tn_window_ev=esd_tn_window_ev, esd_temperature_k=esd_temperature_k,
        nmr_method=nmr_method, esd_ic=esd_ic, esd_ht_scope=esd_ht_scope, esd_ht_method=esd_ht_method,
        nmr_dh_nprocs=nmr_dh_nprocs, nmr_dh_maxcore=nmr_dh_maxcore, nmr_reference=nmr_reference,
        densities=densities, density_eldens=not no_eldens, density_spindens=not no_spindens,
        density_grid=density_grid, density_eldens_file=eldens_file, density_spindens_file=spindens_file,
        nbo=nbo, nbo_keywords=nbo_keywords, reactivity=reactivity,
    )
    qualified, author = _qualified_project(project, user)

    # Use custom headers if provided, otherwise use defaults
    h_cs = _read_header_file(header_confsearch) if header_confsearch else (None if skip_confsearch else DEFAULT_HEADER_CONFSEARCH)
    h_opt = _read_header_file(header_opt) if header_opt else DEFAULT_HEADER_OPTIMIZATION
    h_sp = _read_header_file(header_sp) if header_sp else DEFAULT_HEADER_SINGLEPOINT
    advanced = _submission_advanced(qualified, (h_cs, h_opt, h_sp), _advanced_from_options(options), settings)
    _check_categories(smiles, flags, h_opt, h_sp, settings, advanced)

    request_metadata = _build_request_metadata(
        project_name=qualified,
        project_author=author,
        request_t1=request_t1,
        request_ox=request_ox,
        request_red=request_red,
        skip_confsearch=skip_confsearch,
        request_vert_ex=not no_vert_ex,
        max_conformers_s0=max_conformers_s0,
        max_conformers_t1=max_conformers_t1,
        max_conformers_ox=max_conformers_ox,
        max_conformers_red=max_conformers_red,
        category_flags=flags,
        imag_freq_tolerance=imag_freq_tolerance,
        advanced=advanced,
        seed_from_s0=seed_from_s0,
    )

    entry = CalculationEntrypoint(
        smiles=smiles,
        request_metadata=request_metadata,
        priority=priority,
        header_confsearch=h_cs,
        header_optimization=h_opt,
        header_singlepoint=h_sp,
    )

    with get_session() as session:
        session.add(entry)
        session.commit()
        session.refresh(entry)
        entry_id = entry.id

    cs_info = "skip" if skip_confsearch else ("custom" if header_confsearch else "default")
    console.print(f"[green]Submitted[/green] {smiles} (entry #{entry_id}, project={project}, confsearch={cs_info})")
    logger.info("Submitted SMILES=%s entry_id=%d project=%s", smiles, entry_id, project)


@app.command("submit-batch")
def submit_batch(
    file: Path = typer.Option(..., "--file", help="CSV file with SMILES (one per line or with header)"),
    project: str = typer.Option(..., "--project", help="Project name"),
    user: str = typer.Option(
        "admin", "--user",
        help="Account to submit as; the project lands in their namespace",
    ),
    priority: int = typer.Option(10, "--priority", help="Queue priority (higher = more urgent)"),
    request_t1: bool = typer.Option(False, "--request-t1", help="Request T1 state calculations"),
    request_ox: bool = typer.Option(False, "--request-ox", help="Request oxidation state calculations"),
    request_red: bool = typer.Option(False, "--request-red", help="Request reduction state calculations"),
    skip_confsearch: bool = typer.Option(
        False, "--skip-confsearch",
        help="Skip conformer search; use RDKit geometry directly for optimization",
    ),
    seed_from_s0: bool = typer.Option(
        False, "--seed-from-s0",
        help="T1/ox/red start from S0's lowest conformer once S0 is finished, without their own conformer search",
    ),
    no_vert_ex: bool = typer.Option(
        False, "--no-vert-ex",
        help="Disable singlepoint vertical excitations (ox/red/spin-flip). On by default.",
    ),
    header_confsearch: Optional[Path] = typer.Option(
        None, "--header-confsearch", help="ORCA header file for conformer search (default: GOAT g-xTB)"
    ),
    header_opt: Optional[Path] = typer.Option(
        None, "--header-opt", help="ORCA header file for optimization, method only (default: wB97X-D3 / def2-SVP)"
    ),
    header_sp: Optional[Path] = typer.Option(
        None, "--header-sp", help="ORCA header file for singlepoint, method only (default: wB97X-D3 / def2-TZVP)"
    ),
    max_conformers_s0: int = typer.Option(1, "--max-conformers-s0", help="Max conformers kept for S0"),
    max_conformers_t1: int = typer.Option(1, "--max-conformers-t1", help="Max conformers kept for T1"),
    max_conformers_ox: int = typer.Option(1, "--max-conformers-ox", help="Max conformers kept for ox"),
    max_conformers_red: int = typer.Option(1, "--max-conformers-red", help="Max conformers kept for red"),
    uvvis: bool = typer.Option(False, "--uvvis", help="UV/Vis absorption (TDDFT) on every S0 conformer"),
    ir: bool = typer.Option(False, "--ir", help="IR spectrum from the optimisation's frequencies"),
    esd: bool = typer.Option(
        False, "--esd",
        help="Excited-state dynamics: ISC/RISC/fluorescence/phosphorescence rates (IC with --esd-ic) from S1 and T1 seeded at the lowest S0",
    ),
    esd_ht: bool = typer.Option(
        False, "--esd-ht", help="Herzberg-Teller for fluorescence, phosphorescence and ISC channels with SOCME below 5 cm-1 (much more expensive: 3N-6 or more displaced TDDFT+SOC singlepoints per triplet sublevel; hours to days)"
    ),
    nmr: bool = typer.Option(
        False, "--nmr", help="NMR shifts (1H/13C/19F) vs automatically computed TMS / CFCl3"
    ),
    uvvis_nroots: int = typer.Option(20, "--uvvis-nroots", help="UV/Vis excited states (1-100)"),
    uvvis_tda: bool = typer.Option(False, "--uvvis-tda", help="Tamm-Dancoff approximation for UV/Vis"),
    nmr_nuclei: str = typer.Option(
        "H,C,F", "--nmr-nuclei", help="NMR nuclei to report, comma-separated (H, C, F, As; As: raw shielding and EFG)"
    ),
    esd_tn_window_ev: float = typer.Option(0.3, "--esd-tn-window", help="Include S1->Tn ISC for Tn up to this many eV above S1"),
    esd_temperature_k: float = typer.Option(298.15, "--esd-temperature", help="Temperature of the ESD rates (K)"),
    esd_ic: bool = typer.Option(
        False, "--esd-ic", help="Also compute the S1->S0 internal conversion rate (harmonic, often unreliable for large gaps)"
    ),
    esd_ht_scope: str = typer.Option("all", "--esd-ht-scope", help="With --esd-ht: all (every rate) or s1 (fluorescence and S1->Tn ISC only)"),
    esd_ht_method: str = typer.Option("sp", "--esd-ht-method", help="With --esd-ht: sp (singlepoint header level) or opt (optimisation header level, cheaper)"),
    imag_freq_tolerance: float = typer.Option(
        0.0, "--imag-freq-tolerance", min=IMAG_FREQ_TOLERANCE_MIN, max=0.0,
        help="Fail optimisations with an imaginary mode below this (cm-1, -300 to 0); e.g. -10 accepts noise modes",
    ),
    nmr_method: str = typer.Option(
        "singlepoint", "--nmr-method",
        help="NMR method: singlepoint (the singlepoint header's) or dh (revDSD-PBEP86-D4/2021 / pcSseg-3; H-Kr only)",
    ),
    nmr_dh_nprocs: int = typer.Option(12, "--nmr-dh-nprocs", help="Cores of a dh NMR job (recommended 12)"),
    nmr_dh_maxcore: int = typer.Option(6000, "--nmr-dh-maxcore", help="MB per core of a dh NMR job (recommended 6000)"),
    nmr_reference: str = typer.Option("chcl3", "--nmr-reference", help="1H/13C reference: chcl3 (default) or tms; 19F always CFCl3"),
    densities: bool = typer.Option(False, "--densities", help="Electron / spin density cube files from every state's energy singlepoint"),
    no_eldens: bool = typer.Option(False, "--no-eldens", help="Skip the electron density cube"),
    no_spindens: bool = typer.Option(False, "--no-spindens", help="Skip the spin density cube (open-shell jobs only)"),
    density_grid: int = typer.Option(75, "--density-grid", help="Cube grid points per axis (10-300)"),
    eldens_file: str = typer.Option("ElDens.cube", "--eldens-file", help="Electron density cube file name"),
    spindens_file: str = typer.Option("SpinDens.cube", "--spindens-file", help="Spin density cube file name"),
    nbo: bool = typer.Option(False, "--nbo", help="NBO 7 inside every state's energy singlepoint"),
    nbo_keywords: str = typer.Option("", "--nbo-keywords", help='Extra $NBO keywords, e.g. "BNDIDX"'),
    reactivity: bool = typer.Option(False, "--reactivity", help="Hirshfeld, %elprop polarizability/quadrupole, NBO with Wiberg indices and kept .gbw in every relaxed and vertical singlepoint"),
    opt_grid: Optional[str] = typer.Option(None, "--opt-grid", help="Optimisation grid: DEFGRID1-3 or omit (default: config [advanced])"),
    sp_grid: Optional[str] = typer.Option(None, "--sp-grid", help="Singlepoint grid: DEFGRID1-3 or omit (default: config [advanced])"),
    opt_scf_convergence: Optional[str] = typer.Option(None, "--opt-scf-convergence", help="Optimisation SCF convergence: Normal, Strong, Tight, VeryTight, Extreme or omit (default: config [advanced])"),
    sp_scf_convergence: Optional[str] = typer.Option(None, "--sp-scf-convergence", help="Singlepoint SCF convergence: Normal, Strong, Tight, VeryTight, Extreme or omit (default: config [advanced])"),
    opt_geom_convergence: Optional[str] = typer.Option(None, "--opt-geom-convergence", help="Geometry convergence: Loose, Normal, Tight, VeryTight or omit (default: config [advanced])"),
    opt_ri: Optional[str] = typer.Option(None, "--opt-ri", help="Optimisation RI: RIJCOSX, RI-J, RIJK, NoRI or omit (default: config [advanced])"),
    sp_ri: Optional[str] = typer.Option(None, "--sp-ri", help="Singlepoint RI: RIJCOSX, RI-J, RIJK, NoRI or omit (default: config [advanced])"),
    opt_aux_basis: Optional[str] = typer.Option(None, "--opt-aux-basis", help="Optimisation auxiliary basis: def2/J, def2/JK, AutoAux or omit (default: config [advanced])"),
    sp_aux_basis: Optional[str] = typer.Option(None, "--sp-aux-basis", help="Singlepoint auxiliary basis: def2/J, def2/JK, AutoAux or omit (default: config [advanced])"),
    opt_solvation: Optional[str] = typer.Option(None, "--opt-solvation", help="Optimisation solvation: CPCM, SMD or omit (default: config [advanced])"),
    sp_solvation: Optional[str] = typer.Option(None, "--sp-solvation", help="Singlepoint solvation: CPCM, SMD or omit (default: config [advanced])"),
    opt_solvent: Optional[str] = typer.Option(None, "--opt-solvent", help="Optimisation solvent (ORCA name) (default: config [advanced])"),
    sp_solvent: Optional[str] = typer.Option(None, "--sp-solvent", help="Singlepoint solvent (ORCA name) (default: config [advanced])"),
    cs_nprocs: Optional[int] = typer.Option(None, "--cs-nprocs", help="Conformer search cores (1-64) (default: config [advanced])"),
    opt_nprocs: Optional[int] = typer.Option(None, "--opt-nprocs", help="Optimisation cores (1-64) (default: config [advanced])"),
    sp_nprocs: Optional[int] = typer.Option(None, "--sp-nprocs", help="Singlepoint cores (1-64) (default: config [advanced])"),
    cs_maxcore: Optional[int] = typer.Option(None, "--cs-maxcore", help="Conformer search MB per core (250-16000) (default: config [advanced])"),
    opt_maxcore: Optional[int] = typer.Option(None, "--opt-maxcore", help="Optimisation MB per core (250-16000) (default: config [advanced])"),
    sp_maxcore: Optional[int] = typer.Option(None, "--sp-maxcore", help="Singlepoint MB per core (250-16000) (default: config [advanced])"),
    config: str = required_config_option(),
) -> None:
    """Submit molecules from a CSV file."""
    options = dict(locals())  # the parameters, for _advanced_from_options
    from autodft.cli.options import cli_settings

    settings = cli_settings(config)
    init_db(settings)

    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(code=1)
    qualified, author = _qualified_project(project, user)
    flags = _category_options_to_flags(
        uvvis, ir, esd, esd_ht, nmr, uvvis_nroots, uvvis_tda,
        nmr_nuclei=[n.strip() for n in nmr_nuclei.split(",") if n.strip()],
        esd_tn_window_ev=esd_tn_window_ev, esd_temperature_k=esd_temperature_k,
        nmr_method=nmr_method, esd_ic=esd_ic, esd_ht_scope=esd_ht_scope, esd_ht_method=esd_ht_method,
        nmr_dh_nprocs=nmr_dh_nprocs, nmr_dh_maxcore=nmr_dh_maxcore, nmr_reference=nmr_reference,
        densities=densities, density_eldens=not no_eldens, density_spindens=not no_spindens,
        density_grid=density_grid, density_eldens_file=eldens_file, density_spindens_file=spindens_file,
        nbo=nbo, nbo_keywords=nbo_keywords, reactivity=reactivity,
    )

    h_cs = _read_header_file(header_confsearch) if header_confsearch else (None if skip_confsearch else DEFAULT_HEADER_CONFSEARCH)
    h_opt = _read_header_file(header_opt) if header_opt else DEFAULT_HEADER_OPTIMIZATION
    h_sp = _read_header_file(header_sp) if header_sp else DEFAULT_HEADER_SINGLEPOINT
    advanced = _submission_advanced(qualified, (h_cs, h_opt, h_sp), _advanced_from_options(options), settings)

    request_metadata = _build_request_metadata(
        project_name=qualified,
        project_author=author,
        request_t1=request_t1,
        request_ox=request_ox,
        request_red=request_red,
        skip_confsearch=skip_confsearch,
        request_vert_ex=not no_vert_ex,
        max_conformers_s0=max_conformers_s0,
        max_conformers_t1=max_conformers_t1,
        max_conformers_ox=max_conformers_ox,
        max_conformers_red=max_conformers_red,
        category_flags=flags,
        imag_freq_tolerance=imag_freq_tolerance,
        advanced=advanced,
        seed_from_s0=seed_from_s0,
    )

    smiles_list: list[str] = []
    with open(file, newline="", encoding="utf-8") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        sniffer = csv.Sniffer()
        try:
            has_header = sniffer.has_header(sample)
        except csv.Error:
            has_header = False

        reader = csv.reader(fh)
        if has_header:
            header_row = next(reader)
            smiles_col = 0
            for i, col_name in enumerate(header_row):
                if col_name.strip().lower() == "smiles":
                    smiles_col = i
                    break
            for row in reader:
                if row and row[smiles_col].strip():
                    smiles_list.append(row[smiles_col].strip())
        else:
            for row in reader:
                if row and row[0].strip():
                    smiles_list.append(row[0].strip())

    if not smiles_list:
        console.print("[yellow]No SMILES found in file.[/yellow]")
        raise typer.Exit(code=1)

    # Check every row before writing any of them, so a batch either goes in
    # whole or not at all.
    for smi in smiles_list:
        _check_reference_state(smi, request_t1, request_ox, request_red)
        _check_categories(smi, flags, h_opt, h_sp, settings, advanced)

    submitted = 0
    with get_session() as session:
        for smi in smiles_list:
            entry = CalculationEntrypoint(
                smiles=smi,
                request_metadata=request_metadata,
                priority=priority,
                header_confsearch=h_cs,
                header_optimization=h_opt,
                header_singlepoint=h_sp,
            )
            session.add(entry)
            submitted += 1
        session.commit()

    console.print(
        f"[green]Submitted {submitted} molecule(s)[/green] from {file} (project={project})"
    )
    logger.info("Batch submitted %d molecules from %s project=%s", submitted, file, project)
