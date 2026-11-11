"""Spectra from ORCA output: TDDFT absorption and IR tables.

Only the plain electric-dipole absorption table is read. With DOSOC, ORCA also
prints SOC-corrected and velocity-gauge tables in the same row format; the
parser keys on the exact title and stops at the first table's end. With
``triplets true`` the plain table also lists spin-forbidden rows
(``0-1A -> 1-3A``, fosc 0); only spin-allowed rows are kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

EV_TO_CM = 8065.543937

_ABS_TITLE = "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
# ORCA 6: "  0-1A  ->  3-1A    6.022348   48573.5   205.9   0.240844161 ..."
# Groups: initial multiplicity, final root, final multiplicity, eV, cm-1, nm, fosc.
_ABS_ROW_6 = re.compile(
    r"^\s*\d+-(\d+)\w*\s+->\s+(\d+)-(\d+)\w*\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([-\d.eE+]+)"
)
# ORCA 5: "   3   48573.5    205.9   0.240844161 ..."
_ABS_ROW_5 = re.compile(r"^\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([-\d.eE+]+)\s")
# "  6:    245.98   0.000004    0.02  0.000005  ( ... )"
_IR_ROW = re.compile(r"^\s*(\d+):\s+(-?[\d.]+)\s+[-\d.eE+]+\s+([-\d.eE+]+)\s")


@dataclass(frozen=True)
class Transition:
    root: int
    energy_ev: float
    energy_cm: float
    wavelength_nm: float
    fosc: float


@dataclass(frozen=True)
class IRMode:
    mode: int
    frequency_cm: float
    intensity_km_mol: float


def parse_absorption(content: str) -> list[Transition]:
    """Transitions from the first plain electric-dipole absorption table."""
    lines = content.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == _ABS_TITLE), None)
    if start is None:
        return []

    rows: list[Transition] = []
    seen_row = False
    for line in lines[start + 1:]:
        match6 = _ABS_ROW_6.match(line)
        if match6:
            seen_row = True
            mult_from, root, mult_to, ev, cm, nm, fosc = match6.groups()
            if mult_to == mult_from:
                rows.append(Transition(int(root), float(ev), float(cm), float(nm), float(fosc)))
            continue
        match5 = _ABS_ROW_5.match(line)
        if match5:
            seen_row = True
            root, cm, nm, fosc = match5.groups()
            rows.append(Transition(int(root), float(cm) / EV_TO_CM, float(cm), float(nm), float(fosc)))
            continue
        if seen_row and not line.strip():
            break
    return rows


def parse_ir(content: str) -> list[IRMode]:
    """Modes from the last IR SPECTRUM table."""
    lines = content.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "IR SPECTRUM"]
    if not starts:
        return []

    modes: list[IRMode] = []
    for line in lines[starts[-1] + 1:]:
        match = _IR_ROW.match(line)
        if match:
            modes.append(IRMode(int(match.group(1)), float(match.group(2)), float(match.group(3))))
            continue
        if modes and not line.strip():
            break
    return modes


_SHIELDING_TITLE = "CHEMICAL SHIELDING SUMMARY (ppm)"
# "      2       C           67.627        182.501 "
_SHIELDING_ROW = re.compile(r"^\s*(\d+)\s+([A-Z][a-z]?)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")


@dataclass(frozen=True)
class Shielding:
    index: int  # 0-based ORCA atom index
    element: str
    isotropic: float
    anisotropy: float


def parse_shieldings(content: str) -> list[Shielding]:
    """Isotropic shieldings (ppm) from the last shielding summary.

    Double hybrids print three -- SCF, unrelaxed and relaxed MP2 -- and the
    last is the final result.
    """
    lines = content.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == _SHIELDING_TITLE]
    if not starts:
        return []

    rows: list[Shielding] = []
    for line in lines[starts[-1] + 1:]:
        match = _SHIELDING_ROW.match(line)
        if match:
            rows.append(Shielding(
                int(match.group(1)), match.group(2), float(match.group(3)), float(match.group(4)),
            ))
            continue
        if rows:
            break
    return rows


# " Nucleus   1As: A  : Isotope=   75 I=  1.5 P= 91.6668 MHz/au**3"
_EFG_NUCLEUS = re.compile(r"^\s*Nucleus\s+(\d+)\s*([A-Z][a-z]?)\s*:\s*A\s*:")
_EFG_VTOT = re.compile(r"^\s*V\(Tot\)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)")
_EFG_CQ = re.compile(r"^\s*e\*\*2qQ\s+=\s+(-?[\d.]+)\s+MHz")
_EFG_ETA = re.compile(r"^\s*eta\s+=\s+(-?[\d.]+)")


@dataclass(frozen=True)
class Efg:
    index: int  # 0-based ORCA atom index
    element: str
    vzz_au: float  # largest-magnitude principal component of the total EFG
    cq_mhz: float  # quadrupole coupling constant e**2qQ/h
    eta: float


def parse_efg(content: str) -> list[Efg]:
    """Electric field gradients from the %eprnmr ``fgrad`` output; the last per nucleus."""
    found: dict[int, Efg] = {}
    nucleus = vzz = cq = None
    for line in content.splitlines():
        match = _EFG_NUCLEUS.match(line)
        if match:
            nucleus, vzz, cq = (int(match.group(1)), match.group(2)), None, None
            continue
        if nucleus is None:
            continue
        if (match := _EFG_VTOT.match(line)):
            vzz = float(match.group(3))
        elif (match := _EFG_CQ.match(line)):
            cq = float(match.group(1))
        elif (match := _EFG_ETA.match(line)) and vzz is not None and cq is not None:
            found[nucleus[0]] = Efg(nucleus[0], nucleus[1], vzz, cq, float(match.group(1)))
            nucleus = None
    return [found[i] for i in sorted(found)]


_NPA_TITLE = "Summary of Natural Population Analysis:"
# "    C  1    0.30354      1.99995     3.66683    0.02968     5.69646" -- the
# open-shell total block adds a Natural Spin Density column at the end.
# NBO 7's fixed-width format (1x,2x,a2,i3,...) runs element and number
# together from atom 100 on, e.g. "    C100    0.30354 ...".
_NPA_ROW = re.compile(
    r"^\s*([A-Z][a-z]?)\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)"
    r"(?:\s+(-?[\d.]+))?\s*$"
)
_ATOM_COUNT_RE = re.compile(r"Number of atoms\s*\.+\s*(\d+)")


@dataclass(frozen=True)
class NaturalCharge:
    index: int  # 0-based: the NBO atom number - 1
    element: str
    charge: float
    spin: Optional[float] = None  # Natural Spin Density; only for open shell


def parse_natural_charges(content: str) -> list[NaturalCharge]:
    """Atoms from the first ``Summary of Natural Population Analysis`` block.

    Stops at that block's ``====`` line, so the total row and any later
    (alpha/beta) summaries are never read. When ORCA's own atom count is
    printed and disagrees with what was parsed, ``[]`` is returned instead
    of a silently truncated list.
    """
    lines = content.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == _NPA_TITLE), None)
    if start is None:
        return []

    rows: list[NaturalCharge] = []
    for line in lines[start + 1:]:
        if line.strip().startswith("===="):
            break
        match = _NPA_ROW.match(line)
        if match:
            element, atom_no, charge, _core, _valence, _rydberg, _total, spin = match.groups()
            rows.append(NaturalCharge(
                int(atom_no) - 1, element, float(charge), float(spin) if spin is not None else None,
            ))
    count_match = _ATOM_COUNT_RE.search(content)
    if count_match is not None and len(rows) != int(count_match.group(1)):
        return []
    return rows


_HIRSHFELD_TITLE = "HIRSHFELD ANALYSIS"
# "   0 Br  -0.055841    0.000000"
_HIRSHFELD_ROW = re.compile(r"^\s*(\d+)\s+([A-Z][a-z]?)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")
_POLAR_ISO_RE = re.compile(r"^Isotropic polarizability\s*:\s*(-?[\d.]+)", re.MULTILINE)
_POLAR_TITLE = "STATIC POLARIZABILITY TENSOR"
_POLAR_TENSOR_RE = re.compile(
    r"The raw cartesian tensor \(atomic units\):[ \t]*\n"
    + r"[ \t]*(-?[\d.]+)[ \t]+(-?[\d.]+)[ \t]+(-?[\d.]+)[ \t]*\n" * 3
)
_QUAD_TITLE = "QUADRUPOLE MOMENT"
# "  TOT     -136.108961855  -161.573828523 ... (a.u.)": XX YY ZZ XY XZ YZ.
_QUAD_TOT_RE = re.compile(r"^\s*TOT\s+" + r"(-?[\d.]+)\s+" * 6 + r"\(a\.u\.\)", re.MULTILINE)
_QUAD_ISO_RE = re.compile(r"^Isotropic quadrupole\s*:\s*(-?[\d.]+)", re.MULTILINE)
_ORIGIN_RE = re.compile(r"^Choice of electric origin\s*\.\.\.\s*(.+?)\s*$", re.MULTILINE)
_WIBERG_TITLE = "Wiberg bond index matrix in the NAO basis:"
WIBERG_MIN = 0.05
# "     Atom    1       2       3 ..." and "   1. Br  0.0000  1.0417  0.0585 ..."
_WIBERG_COLS = re.compile(r"^\s*Atom((?:\s+\d+)+)\s*$")
_WIBERG_ROW = re.compile(r"^\s*(\d+)\.\s*[A-Z][a-z]?((?:\s+\d+\.\d+)+)\s*$")
_STABLE_RE = re.compile(
    r"^(The stability analysis shows that the wavefunction is stable"
    r"|The stability analysis indicates that the wavefunction is unstable"
    r"|.*The wavefunction is trivially stable)",
    re.MULTILINE,
)
_STAB_EIGEN_TITLE = "The eigenvalues of the stability matrix:"
_STAB_EIGEN_ROW = re.compile(r"^\s*E\(\s*\d+\)\s*=\s*(-?[\d.]+)\s*Eh")
_S2_RE = re.compile(r"^Expectation value of <S\*\*2>\s*:\s*(-?[\d.]+)", re.MULTILINE)


@dataclass(frozen=True)
class AtomCharge:
    index: int  # 0-based ORCA atom index
    element: str
    charge: float
    spin: Optional[float] = None


def parse_hirshfeld(content: str) -> list[AtomCharge]:
    """Atoms from the last Hirshfeld table; ``[]`` when ORCA's atom count disagrees."""
    lines = content.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == _HIRSHFELD_TITLE]
    if not starts:
        return []

    rows: list[AtomCharge] = []
    for line in lines[starts[-1] + 1:]:
        if line.strip().startswith("TOTAL"):
            break
        match = _HIRSHFELD_ROW.match(line)
        if match:
            rows.append(AtomCharge(
                int(match.group(1)), match.group(2), float(match.group(3)), float(match.group(4)),
            ))
    count_match = _ATOM_COUNT_RE.search(content)
    if count_match is not None and len(rows) != int(count_match.group(1)):
        return []
    return rows


def parse_polarizability(content: str) -> Optional[dict]:
    """The last static dipole polarizability (a.u.): ``isotropic`` and the 3x3 ``tensor`` (None if cut short)."""
    found = _POLAR_ISO_RE.findall(content)
    if not found:
        return None
    start = content.rfind(_POLAR_TITLE)
    match = _POLAR_TENSOR_RE.search(content, start) if start >= 0 else None
    values = [float(v) for v in match.groups()] if match else None
    tensor = [values[0:3], values[3:6], values[6:9]] if values else None
    return {"isotropic": float(found[-1]), "tensor": tensor}


def parse_quadrupole(content: str) -> Optional[dict]:
    """The last quadrupole moment: total ``tensor`` (a.u.), electric ``origin`` and ``isotropic``."""
    start = content.rfind(_QUAD_TITLE)
    if start < 0:
        return None
    block = content[start:]
    tot = _QUAD_TOT_RE.search(block)
    if tot is None:
        return None
    xx, yy, zz, xy, xz, yz = (float(v) for v in tot.groups())
    origins = _ORIGIN_RE.findall(content[:start])
    iso = _QUAD_ISO_RE.search(block)
    return {
        "tensor": [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
        "origin": origins[-1] if origins else None,
        "isotropic": float(iso.group(1)) if iso else None,
    }


def parse_wiberg(content: str) -> Optional[list[tuple[int, int, float]]]:
    """Atom pairs (0-based, i < j) with a Wiberg index >= 0.05.

    Reads the first NAO-basis matrix -- the total density's; open shell
    prints alpha and beta matrices after it.
    """
    lines = content.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == _WIBERG_TITLE), None)
    if start is None:
        return None

    pairs: list[tuple[int, int, float]] = []
    columns: list[int] = []
    for line in lines[start + 1:]:
        header = _WIBERG_COLS.match(line)
        if header:
            columns = [int(c) for c in header.group(1).split()]
            continue
        row = _WIBERG_ROW.match(line)
        if row and columns:
            i = int(row.group(1))
            for j, value in zip(columns, (float(v) for v in row.group(2).split())):
                if i < j and value >= WIBERG_MIN:
                    pairs.append((i - 1, j - 1, value))
            continue
        if line.strip() and not line.strip().startswith("----"):
            break
    return pairs


def parse_stability(content: str) -> Optional[dict]:
    """The last stability analysis: ``stable`` and its ``lowest_eigenvalue`` (Eh), if printed."""
    verdicts = _STABLE_RE.findall(content)
    if not verdicts:
        return None
    lines = content.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == _STAB_EIGEN_TITLE]
    eigenvalues: list[float] = []
    for line in lines[starts[-1] + 1:] if starts else []:
        match = _STAB_EIGEN_ROW.match(line)
        if not match:
            break
        eigenvalues.append(float(match.group(1)))
    return {
        "stable": "unstable" not in verdicts[-1],
        "lowest_eigenvalue": min(eigenvalues) if eigenvalues else None,
    }


def parse_s2(content: str) -> Optional[float]:
    """The last <S**2>; ORCA prints it for open shell only."""
    found = _S2_RE.findall(content)
    return float(found[-1]) if found else None
