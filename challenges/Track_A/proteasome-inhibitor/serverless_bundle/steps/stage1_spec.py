"""
stage1_spec — scientific specification, identity and validation of Stage 0/1 (molecule + Hartree-Fock).

Shared by the Serverless worker (``step1_meanfield``) and the local orchestrator (``proteasome_challenge.hf_stage``)
so both sides agree on what a Stage 1 calculation *is*.  Nothing here knows any molecule by name, runs an SCF, or
touches the network.  PySCF is never imported at module level.

Three distinct notions are kept apart:

``stage1_param_digest``   SCIENTIFIC IDENTITY of the *requested* calculation (this module).
artifact SHA-256          INTEGRITY of a persisted file (``hashlib`` over its bytes).
numerical tolerances      AGREEMENT of independently computed results (``validate_scf_state`` thresholds).
None of them substitutes for another.

Canonicalization policy (``stage1-spec/1``)
-------------------------------------------
The digest is SHA-256 over compact, key-sorted UTF-8 JSON of::

    {"schema": "stage1-spec/1",
     "molecule": {"geometry": [[symbol, x, y, z], ...], "units": u, "charge": c, "spin": s},
     "hf": {"method", "basis", "conv_tol", "conv_tol_grad", "max_cycle", "init_guess", "level_shift", "damp"}}

* Atom ORDER is part of the identity (fragment indices follow it).  Element symbols are normalised to standard
  capitalisation.  Coordinates are parsed to IEEE doubles and written with ``repr(float)`` (shortest round-trip form),
  so ``1.0`` and ``1.000`` are one value; ``-0.0`` is written as ``0.0``.  Coordinates are NEVER converted between
  units, rotated, translated or otherwise modified: the same molecule written in bohr and in angstrom has two
  identities by design.
* NOT part of the identity: molecule ``name``, the XYZ comment line, ``source_path``, ``verbose`` and the software
  version (a version is provenance of an execution, not part of the request).
* ``basis`` is lower-cased; ``method`` is upper-cased; the initial-guess alias ``1e`` is canonicalised to ``hcore``.
  Numeric options are stored as floats / ints.

PySCF convention: ``spin = N_alpha - N_beta = 2S``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

__all__ = [
    "SPEC_SCHEMA", "RESULT_SCHEMA", "Stage1SpecError", "Stage1ValidationError",
    "MoleculeSpec", "HFConfig", "Stage1Spec", "parse_xyz",
    "SUPPORTED_METHODS", "SUPPORTED_INIT_GUESSES", "DEFERRED_METHODS",
    "spec_from_pyscf", "scf_diagnostics", "validate_scf_state", "build_result_record",
    "sha256_bytes", "sha256_file", "diff_specs", "check_basis_available",
]

SPEC_SCHEMA = "stage1-spec/1"
RESULT_SCHEMA = "stage1-hf-result/1"

SUPPORTED_METHODS = ("RHF",)
DEFERRED_METHODS = ("UHF", "ROHF", "RKS", "UKS", "ROKS", "DFT", "DF-HF", "DFHF")
SUPPORTED_INIT_GUESSES = ("minao", "atom", "huckel", "mod_huckel", "hcore")
_INIT_GUESS_ALIASES = {"1e": "hcore"}

_MAX_XYZ_BYTES = 5_000_000
_MIN_SEPARATION_ANGSTROM = 0.1
_BOHR_PER_ANGSTROM = 1.8897261246257702
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_BASIS_RE = re.compile(r"^[a-z0-9][a-z0-9+*()\-_,.]{0,63}$")
_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$")
_SYMBOL_RE = re.compile(r"^[A-Za-z]{1,2}$")
_UNIT_ALIASES = {"angstrom": "angstrom", "ang": "angstrom", "bohr": "bohr", "au": "bohr", "a.u.": "bohr"}

_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split()
_ATOMIC_NUMBER = {s: i + 1 for i, s in enumerate(_SYMBOLS)}


class Stage1SpecError(ValueError):
    """The requested molecule / HF configuration is invalid or unsupported."""


class Stage1ValidationError(RuntimeError):
    """A computed (or persisted) mean-field state failed scientific validation."""


# ─────────────────────────────────────────────────────────────────────────────
#  Hashing helpers
# ─────────────────────────────────────────────────────────────────────────────

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  XYZ parsing
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_symbol(tok: str, where: str) -> str:
    if not _SYMBOL_RE.match(tok):
        raise Stage1SpecError(
            f"{where}: {tok!r} is not an element symbol (atomic numbers, labels such as 'C1' and ghost "
            "atoms are not supported; use plain symbols like 'C', 'Cl')")
    sym = tok[0].upper() + tok[1:].lower()
    if sym not in _ATOMIC_NUMBER:
        raise Stage1SpecError(f"{where}: unsupported element {tok!r} (supported: H through Rn)")
    return sym


def _parse_coordinate(tok: str, where: str) -> float:
    if not _NUM_RE.match(tok):
        raise Stage1SpecError(f"{where}: coordinate {tok!r} is not a finite decimal number")
    value = float(tok.replace("d", "e").replace("D", "e"))
    if not math.isfinite(value):
        raise Stage1SpecError(f"{where}: coordinate {tok!r} is not finite")
    return value + 0.0          # -0.0 -> 0.0


def parse_xyz(text: str) -> tuple[str, tuple[tuple[str, float, float, float], ...]]:
    """Parse standard XYZ text -> ``(comment, ((symbol, x, y, z), ...))``.  Strict: no silent repair."""
    if not isinstance(text, str) or not text.strip():
        raise Stage1SpecError("geometry is empty; supply XYZ text (atom count, comment line, then one atom per line)")
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    try:
        n_atoms = int(lines[0].strip())
    except (ValueError, IndexError) as exc:
        raise Stage1SpecError(
            "XYZ line 1 must be the integer atom count (the geometry has no header line)") from exc
    if n_atoms < 1:
        raise Stage1SpecError(f"XYZ atom count must be >= 1, got {n_atoms}")
    if len(lines) < 2 + n_atoms:
        raise Stage1SpecError(
            f"XYZ header declares {n_atoms} atoms but only {max(len(lines) - 2, 0)} atom records follow")
    if len(lines) > 2 + n_atoms:
        raise Stage1SpecError(
            f"XYZ header declares {n_atoms} atoms but {len(lines) - 2} records follow; fix the header or remove "
            "the extra lines")
    atoms = []
    for i, line in enumerate(lines[2:2 + n_atoms], start=1):
        where = f"XYZ atom record {i}"
        toks = line.split()
        if len(toks) != 4:
            raise Stage1SpecError(f"{where}: expected 'symbol x y z', got {len(toks)} fields: {line.strip()!r}")
        sym = _normalize_symbol(toks[0], where)
        atoms.append((sym, *(_parse_coordinate(t, where) for t in toks[1:])))
    return lines[1], tuple(atoms)


def _hill_formula(symbols) -> str:
    counts: dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    order = ([e for e in ("C", "H") if e in counts] + sorted(e for e in counts if e not in ("C", "H"))
             if "C" in counts else sorted(counts))
    return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order)


# ─────────────────────────────────────────────────────────────────────────────
#  MoleculeSpec  (Stage 0)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MoleculeSpec:
    """Validated molecular specification.  Exactly one of ``geometry`` (inline XYZ text) or ``path`` (XYZ file).

    A ``path`` is read ONCE at construction and the text is embedded, so the specification is self-contained and can
    be transported to Serverless without assuming that the local path exists there.  ``source_path`` is recorded as
    information only.
    """
    name: str
    units: str
    charge: int
    spin: int
    geometry: Optional[str] = None
    path: Optional[str] = None
    source_path: Optional[str] = field(default=None, compare=False)
    atoms: tuple = field(default=(), init=False, repr=False, compare=False)
    comment: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not _NAME_RE.match(self.name):
            raise Stage1SpecError(
                f"molecule name {self.name!r} is invalid; use 1-64 characters from [A-Za-z0-9_.-], "
                "starting with a letter or digit")
        units = _UNIT_ALIASES.get(str(self.units).strip().lower()) if isinstance(self.units, str) else None
        if units is None:
            raise Stage1SpecError(f"units {self.units!r} not supported; use 'angstrom' or 'bohr' (no default is assumed)")
        for label, value in (("charge", self.charge), ("spin", self.spin)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise Stage1SpecError(f"{label} must be an integer, got {value!r}")
        if self.spin < 0:
            raise Stage1SpecError(f"spin = N_alpha - N_beta must be >= 0, got {self.spin}")
        if (self.geometry is None) == (self.path is None):
            raise Stage1SpecError("give exactly one of geometry= (inline XYZ text) or path= (XYZ file)")
        text, source = self.geometry, self.source_path
        if self.path is not None:
            p = Path(self.path).expanduser()
            try:
                if p.stat().st_size > _MAX_XYZ_BYTES:
                    raise Stage1SpecError(f"XYZ file {p} is larger than {_MAX_XYZ_BYTES} bytes")
                text = p.read_text(encoding="utf-8")
            except OSError as exc:
                raise Stage1SpecError(f"cannot read XYZ file {str(p)!r}: {exc.strerror or exc}") from exc
            except UnicodeDecodeError as exc:
                raise Stage1SpecError(f"XYZ file {str(p)!r} is not valid UTF-8 text") from exc
            source = str(p.resolve())
        comment, atoms = parse_xyz(text)
        self._check_geometry(atoms, units)
        n_el = sum(_ATOMIC_NUMBER[a[0]] for a in atoms) - self.charge
        if n_el < 1:
            raise Stage1SpecError(f"charge {self.charge} leaves {n_el} electrons; need at least 1")
        if self.spin > n_el or (n_el - self.spin) % 2:
            raise Stage1SpecError(
                f"{n_el} electrons (charge {self.charge}) are inconsistent with spin = N_alpha - N_beta = {self.spin}: "
                "N_electrons - spin must be even and non-negative")
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "geometry", text)
        object.__setattr__(self, "path", None)
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "comment", comment)

    @staticmethod
    def _check_geometry(atoms, units: str) -> None:
        limit = _MIN_SEPARATION_ANGSTROM * (_BOHR_PER_ANGSTROM if units == "bohr" else 1.0)
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                d = math.dist(atoms[i][1:], atoms[j][1:])
                if d < limit:
                    raise Stage1SpecError(
                        f"atoms {i + 1} ({atoms[i][0]}) and {j + 1} ({atoms[j][0]}) are coincident or closer than "
                        f"{_MIN_SEPARATION_ANGSTROM} angstrom ({d:.4g} {units}); the geometry is malformed")

    # ── derived quantities ───────────────────────────────────────────────────
    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def formula(self) -> str:
        return _hill_formula(a[0] for a in self.atoms)

    @property
    def n_electrons(self) -> int:
        return sum(_ATOMIC_NUMBER[a[0]] for a in self.atoms) - self.charge

    def canonical_geometry(self) -> list:
        return [[s, x, y, z] for s, x, y, z in self.atoms]

    def geometry_digest(self) -> str:
        """SHA-256 identity of the atoms (order, symbols, coordinates) and units; independent of name/charge/spin."""
        blob = _canonical_json({"schema": SPEC_SCHEMA, "geometry": self.canonical_geometry(), "units": self.units})
        return sha256_bytes(blob)

    def describe(self) -> str:
        return (f"{self.name}: {self.formula}, {self.n_atoms} atoms, {self.n_electrons} electrons, "
                f"charge {self.charge:+d}, spin {self.spin} (2S), units {self.units}, "
                f"geometry digest {self.geometry_digest()[:16]}...")

    def to_dict(self) -> dict:
        return {"name": self.name, "units": self.units, "charge": self.charge, "spin": self.spin,
                "geometry": self.geometry, "source_path": self.source_path}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MoleculeSpec":
        return cls(name=d["name"], units=d["units"], charge=d["charge"], spin=d["spin"],
                   geometry=d["geometry"], source_path=d.get("source_path"))


# ─────────────────────────────────────────────────────────────────────────────
#  HFConfig  (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def check_basis_available(basis: str, symbols) -> list[str]:
    """Problems PySCF reports for ``basis`` on ``symbols`` (empty = fine).  Needs PySCF; returns [] if it is absent
    (the worker re-checks authoritatively).  ECPs are reported as unsupported."""
    try:
        from pyscf.gto import basis as _b
    except ImportError:
        return []
    problems = []
    for sym in sorted(set(symbols)):
        try:
            _b.load(basis, sym)
        except Exception:
            problems.append(f"basis {basis!r} is not available for element {sym}")
            continue
        try:
            if _b.load_ecp(basis, sym):
                problems.append(f"basis {basis!r} carries an effective core potential for {sym}; ECPs are not supported")
        except Exception:
            pass
    return problems


def _finite(value: Any, label: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None,
            exclusive_max: bool = False, exclusive_min: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage1SpecError(f"{label} must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise Stage1SpecError(f"{label} must be finite, got {value!r}")
    if minimum is not None and (v < minimum or (exclusive_min and v == minimum)):
        raise Stage1SpecError(f"{label} must be {'>' if exclusive_min else '>='} {minimum}, got {v}")
    if maximum is not None and (v > maximum or (exclusive_max and v == maximum)):
        raise Stage1SpecError(f"{label} must be {'<' if exclusive_max else '<='} {maximum}, got {v}")
    return v + 0.0


@dataclass(frozen=True)
class HFConfig:
    """Validated Hartree-Fock configuration.  Defaults are the established Proteasome Challenge calculation.

    Standard: ``method``, ``basis``.  Advanced: ``conv_tol``, ``conv_tol_grad`` (None = PySCF default sqrt(conv_tol)),
    ``max_cycle``, ``init_guess``, ``level_shift``, ``damp``.  ``verbose`` is not scientific and is not part of the
    identity.
    """
    method: str = "RHF"
    basis: str = "sto-3g"
    conv_tol: float = 1e-9
    max_cycle: int = 200
    init_guess: str = "minao"
    level_shift: float = 0.0
    damp: float = 0.0
    conv_tol_grad: Optional[float] = None
    verbose: int = 3

    def __post_init__(self):
        if not isinstance(self.method, str) or not self.method.strip():
            raise Stage1SpecError(f"method must be a string, got {self.method!r}")
        method = self.method.strip().upper()
        if method in DEFERRED_METHODS:
            raise Stage1SpecError(
                f"reference method {self.method!r} is not supported in this release (deferred: "
                f"{', '.join(DEFERRED_METHODS)}); the downstream EWF/solver contracts are only qualified for a "
                "closed-shell restricted reference. Supported: " + ", ".join(SUPPORTED_METHODS))
        if method not in SUPPORTED_METHODS:
            raise Stage1SpecError(f"unknown reference method {self.method!r}; supported: {', '.join(SUPPORTED_METHODS)}")
        if not isinstance(self.basis, str):
            raise Stage1SpecError(f"basis must be a basis-set name string, got {type(self.basis).__name__}")
        basis = self.basis.strip().lower()
        if not _BASIS_RE.match(basis):
            raise Stage1SpecError(f"basis {self.basis!r} is not a valid basis-set name (e.g. 'sto-3g', '6-31g*', 'cc-pvdz')")
        conv_tol = _finite(self.conv_tol, "conv_tol", minimum=0.0, maximum=1e-2, exclusive_min=True)
        tol_grad = (None if self.conv_tol_grad is None
                    else _finite(self.conv_tol_grad, "conv_tol_grad", minimum=0.0, exclusive_min=True))
        if isinstance(self.max_cycle, bool) or not isinstance(self.max_cycle, int) or not 1 <= self.max_cycle <= 100_000:
            raise Stage1SpecError(f"max_cycle must be an integer in [1, 100000], got {self.max_cycle!r}")
        if isinstance(self.verbose, bool) or not isinstance(self.verbose, int) or not 0 <= self.verbose <= 9:
            raise Stage1SpecError(f"verbose must be an integer in [0, 9], got {self.verbose!r}")
        guess = self.init_guess.strip().lower() if isinstance(self.init_guess, str) else None
        guess = _INIT_GUESS_ALIASES.get(guess, guess)
        if guess not in SUPPORTED_INIT_GUESSES:
            raise Stage1SpecError(f"init_guess {self.init_guess!r} not supported; choose from {SUPPORTED_INIT_GUESSES}")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "conv_tol", conv_tol)
        object.__setattr__(self, "conv_tol_grad", tol_grad)
        object.__setattr__(self, "init_guess", guess)
        object.__setattr__(self, "level_shift", _finite(self.level_shift, "level_shift", minimum=0.0))
        object.__setattr__(self, "damp", _finite(self.damp, "damp", minimum=0.0, maximum=1.0, exclusive_max=True))

    def canonical(self) -> dict:
        return {"method": self.method, "basis": self.basis, "conv_tol": self.conv_tol,
                "conv_tol_grad": self.conv_tol_grad, "max_cycle": self.max_cycle,
                "init_guess": self.init_guess, "level_shift": self.level_shift, "damp": self.damp}

    def to_dict(self) -> dict:
        return {**self.canonical(), "verbose": self.verbose}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HFConfig":
        return cls(**{k: d[k] for k in ("method", "basis", "conv_tol", "max_cycle", "init_guess", "level_shift",
                                        "damp", "conv_tol_grad", "verbose") if k in d})

    @property
    def effective_conv_tol_grad(self) -> float:
        return self.conv_tol_grad if self.conv_tol_grad is not None else math.sqrt(self.conv_tol)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage1Spec  (molecule + HF) and its digest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stage1Spec:
    molecule: MoleculeSpec
    config: HFConfig

    def __post_init__(self):
        mol, cfg = self.molecule, self.config
        if cfg.method == "RHF" and mol.spin != 0:      # MoleculeSpec already guarantees electron/spin parity
            raise Stage1SpecError(
                f"RHF requires a closed-shell singlet (spin = 0) but spin = {mol.spin} was requested. PySCF would "
                "silently substitute ROHF; that is refused. Open-shell references (UHF/ROHF) are not supported "
                "in this release.")

    def canonical(self) -> dict:
        m = self.molecule
        return {"schema": SPEC_SCHEMA,
                "molecule": {"geometry": m.canonical_geometry(), "units": m.units, "charge": m.charge, "spin": m.spin},
                "hf": self.config.canonical()}

    def digest(self) -> str:
        """``stage1_param_digest``: scientific identity of the requested calculation (see the module docstring)."""
        return sha256_bytes(_canonical_json(self.canonical()))

    @property
    def tag(self) -> str:
        return f"{self.molecule.name}__{self.digest()[:12]}"

    def to_dict(self) -> dict:
        return {"schema": SPEC_SCHEMA, "molecule": self.molecule.to_dict(), "config": self.config.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Stage1Spec":
        if d.get("schema") != SPEC_SCHEMA:
            raise Stage1SpecError(f"unsupported Stage 1 specification schema {d.get('schema')!r}")
        return cls(MoleculeSpec.from_dict(d["molecule"]), HFConfig.from_dict(d["config"]))


def diff_specs(a: Stage1Spec, b: Stage1Spec) -> list[str]:
    """Human-readable list of the scientific components in which two specifications differ."""
    ca, cb = a.canonical(), b.canonical()
    out = []
    for sect in ("molecule", "hf"):
        for key in sorted(set(ca[sect]) | set(cb[sect])):
            if ca[sect].get(key) != cb[sect].get(key):
                va, vb = ca[sect].get(key), cb[sect].get(key)
                if key == "geometry":
                    out.append("geometry (atoms/coordinates)")
                else:
                    out.append(f"{key}: {va!r} vs {vb!r}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Reconstructing a specification from a PySCF state (actual parameters)
# ─────────────────────────────────────────────────────────────────────────────

def _atoms_from_pyscf(mol) -> list[tuple[str, float, float, float]]:
    raw = mol.atom
    atoms = []
    if isinstance(raw, str):
        for rec in [r for r in re.split(r"[\n;]", raw) if r.strip()]:
            toks = rec.replace(",", " ").split()
            if len(toks) != 4:
                raise Stage1SpecError(f"cannot recover the geometry from mol.atom record {rec!r}")
            atoms.append((_normalize_symbol(toks[0], "mol.atom"), *(_parse_coordinate(t, "mol.atom") for t in toks[1:])))
    else:
        for entry in raw:
            sym, rest = entry[0], entry[1:]
            xyz = rest[0] if len(rest) == 1 else rest
            atoms.append((_normalize_symbol(str(sym), "mol.atom"), *(float(c) + 0.0 for c in xyz)))
    return atoms


def spec_from_pyscf(mol, mf, name: str) -> Stage1Spec:
    """The ACTUAL specification embodied by a PySCF ``mol``/``mf`` pair (what was really computed)."""
    if not isinstance(mol.basis, str):
        raise Stage1SpecError("the PySCF molecule uses a non-string basis; it cannot be expressed as an HFConfig")
    units = _UNIT_ALIASES.get(str(mol.unit).strip().lower())
    if units is None:
        raise Stage1SpecError(f"PySCF molecule uses unit {mol.unit!r}")
    atoms = _atoms_from_pyscf(mol)
    geometry = f"{len(atoms)}\n\n" + "\n".join(f"{s} {x!r} {y!r} {z!r}" for s, x, y, z in atoms) + "\n"
    molecule = MoleculeSpec(name=name, geometry=geometry, units=units, charge=int(mol.charge), spin=int(mol.spin))
    cls = type(mf).__name__
    if cls != "RHF":
        raise Stage1SpecError(f"the mean-field object is {cls}, not RHF")
    guess = mf.init_guess
    if not isinstance(guess, str):
        raise Stage1SpecError("the mean-field initial guess is not a named guess; it cannot be recorded")
    tol_grad = None if mf.conv_tol_grad is None else float(mf.conv_tol_grad)
    config = HFConfig(method="RHF", basis=mol.basis, conv_tol=float(mf.conv_tol), max_cycle=int(mf.max_cycle),
                      init_guess=guess, level_shift=float(mf.level_shift), damp=float(mf.damp),
                      conv_tol_grad=tol_grad, verbose=int(getattr(mol, "verbose", 3)))
    return Stage1Spec(molecule, config)


# ─────────────────────────────────────────────────────────────────────────────
#  Independent scientific validation of a mean-field state
# ─────────────────────────────────────────────────────────────────────────────
#
# Thresholds (all scale-aware; recorded with every diagnostic):
#   orthonormality   max|C^T S C - 1| <= max(1e-9, 100 * eps * cond(S))
#       The eigen-solver error grows with cond(S); diffuse / large bases are looser than STO-3G, tight bases stay at 1e-9.
#   electron trace   |tr(D S) - N| <= N * max(1e-9, 100 * eps * cond(S))
#   energy           |E_elec(D) + E_nuc - E_tot| <= max(10 * conv_tol, 1e-8) Ha
#       E_elec is recomputed from the stored density with a fresh Fock build; converged PySCF energies agree to ~conv_tol.
#   stationarity     ||orbital gradient||_2 <= max(10 * conv_tol_grad_eff, 1e-8)
#       The gradient is PySCF's own SCF convergence quantity 2*F_vo; 10x allows the one-cycle lag between the final
#       density and the last Fock build.  This is the Fock-density commutator in the MO basis.
# These do not touch HF_TOL / _FLOAT_TOL, which compare independently produced energies.

def scf_diagnostics(mf, mol, config: HFConfig, *, stationarity: bool = True) -> dict:
    """Numerical diagnostics of a converged RHF state.  ``stationarity=True`` performs one Fock build."""
    import numpy as np

    eps = float(np.finfo(float).eps)
    C = np.asarray(mf.mo_coeff, dtype=float)
    occ = np.asarray(mf.mo_occ, dtype=float)
    e_mo = np.asarray(mf.mo_energy, dtype=float)
    S = np.asarray(mf.get_ovlp(), dtype=float)
    ev = np.linalg.eigvalsh(S)
    cond = float(ev.max() / ev.min()) if ev.min() > 0 else float("inf")
    ortho_tol = max(1e-9, 100.0 * eps * cond)
    n_el = int(mol.nelectron)
    D = (C * occ) @ C.T
    n_occ = int(np.count_nonzero(occ > 0.5))
    homo = float(e_mo[n_occ - 1]) if n_occ > 0 else None
    lumo = float(e_mo[n_occ]) if n_occ < e_mo.size else None
    diag: dict[str, Any] = {
        "energy_total": float(mf.e_tot), "converged": bool(mf.converged),
        "scf_cycles": (int(mf.cycles) if getattr(mf, "cycles", None) is not None else None),
        "n_ao": int(mol.nao), "n_mo": int(C.shape[1]), "n_electrons": n_el,
        "n_alpha": int(mol.nelec[0]), "n_beta": int(mol.nelec[1]),
        "n_occupied": n_occ, "n_virtual": int(e_mo.size - n_occ),
        "occupation_sum": float(occ.sum()), "occupation_values": sorted({round(float(v), 10) for v in occ}),
        "homo_energy": homo, "lumo_energy": lumo,
        "homo_lumo_gap": (lumo - homo) if (homo is not None and lumo is not None) else None,
        "aufbau_satisfied": bool(homo is None or lumo is None or homo <= lumo + 1e-8),
        "overlap_condition_number": cond,
        "orthonormality_residual": float(np.max(np.abs(C.T @ S @ C - np.eye(C.shape[1])))),
        "orthonormality_tolerance": ortho_tol,
        "electron_trace_error": float(abs(np.einsum("ij,ji->", D, S) - n_el)),
        "electron_trace_tolerance": n_el * max(1e-9, 100.0 * eps * cond),
        "thresholds": {"energy": max(10.0 * config.conv_tol, 1e-8),
                       "stationarity": max(10.0 * config.effective_conv_tol_grad, 1e-8)},
    }
    if stationarity:
        h1 = mf.get_hcore()
        fock = h1 + mf.get_veff(mol, D)
        e_elec = 0.5 * float(np.einsum("ij,ji->", D, h1 + fock))
        diag["energy_recomputed_error"] = float(abs(e_elec + float(mf.energy_nuc()) - float(mf.e_tot)))
        f_mo = C.T @ fock @ C
        grad = 2.0 * f_mo[n_occ:, :n_occ]
        diag["orbital_gradient_norm"] = float(np.linalg.norm(grad)) if grad.size else 0.0
        diag["orbital_gradient_max"] = float(np.max(np.abs(grad))) if grad.size else 0.0
    return diag


def validate_scf_state(mf, mol, spec: Stage1Spec, *, stationarity: bool = True) -> dict:
    """Validate a mean-field state against the specification; returns the diagnostics or raises.

    Checks: restricted closed-shell RHF object (no density fitting / ECP / symmetry), the ACTUAL charge, spin, basis and
    SCF settings reproduce ``spec.digest()``, finite energy, convergence, electron count, occupations, orthonormality and
    (optionally) energy consistency and SCF stationarity.
    """
    import numpy as np

    problems: list[str] = []
    if type(mf).__name__ != "RHF" or not type(mf).__module__.startswith("pyscf.scf"):
        problems.append(f"mean-field object is {type(mf).__module__}.{type(mf).__name__}, expected a plain pyscf RHF")
    if getattr(mf, "with_df", None) is not None:
        problems.append("density fitting is active; density-fitted HF is not supported")
    if getattr(mol, "has_ecp", None) and mol.has_ecp():
        problems.append("an effective core potential is present; ECPs are not supported")
    if getattr(mol, "symmetry", False):
        problems.append("point-group symmetry is active; it is not part of the specification")
    if problems:
        raise Stage1ValidationError("; ".join(problems))
    if np.asarray(mf.mo_coeff).ndim != 2 or np.asarray(mf.mo_occ).ndim != 1:
        raise Stage1ValidationError("orbital arrays are not restricted-reference shaped")
    try:
        actual = spec_from_pyscf(mol, mf, spec.molecule.name)
    except Stage1SpecError as exc:
        raise Stage1ValidationError(f"cannot recover the actual specification: {exc}") from exc
    if actual.digest() != spec.digest():
        raise Stage1ValidationError(
            "the computed state does not correspond to the requested specification; differs in "
            + "; ".join(diff_specs(spec, actual)))
    diag = scf_diagnostics(mf, mol, spec.config, stationarity=stationarity)
    if not np.isfinite(diag["energy_total"]):
        problems.append(f"total energy is not finite: {diag['energy_total']!r}")
    if not diag["converged"]:
        problems.append("SCF did not converge")
    if diag["n_electrons"] != spec.molecule.n_electrons:
        problems.append(f"electron count {diag['n_electrons']} != expected {spec.molecule.n_electrons}")
    if (diag["n_alpha"], diag["n_beta"]) != (diag["n_electrons"] // 2,) * 2:
        problems.append(f"alpha/beta electrons ({diag['n_alpha']}, {diag['n_beta']}) are not closed-shell")
    if not set(diag["occupation_values"]) <= {0.0, 2.0}:
        problems.append(f"occupations {diag['occupation_values']} are not {{0, 2}}")
    if abs(diag["occupation_sum"] - diag["n_electrons"]) > 1e-8 or 2 * diag["n_occupied"] != diag["n_electrons"]:
        problems.append("occupation numbers do not sum to the electron count")
    if diag["orthonormality_residual"] > diag["orthonormality_tolerance"]:
        problems.append(f"orbitals are not orthonormal in the AO overlap metric (residual "
                        f"{diag['orthonormality_residual']:.2e} > {diag['orthonormality_tolerance']:.2e})")
    if diag["electron_trace_error"] > diag["electron_trace_tolerance"]:
        problems.append(f"tr(DS) differs from the electron count by {diag['electron_trace_error']:.2e}")
    if stationarity:
        if diag["energy_recomputed_error"] > diag["thresholds"]["energy"]:
            problems.append(f"energy recomputed from the density differs from E_tot by "
                            f"{diag['energy_recomputed_error']:.2e} Ha (> {diag['thresholds']['energy']:.1e})")
        if diag["orbital_gradient_norm"] > diag["thresholds"]["stationarity"]:
            problems.append(f"SCF is not stationary: orbital gradient {diag['orbital_gradient_norm']:.2e} > "
                            f"{diag['thresholds']['stationarity']:.1e}")
    diag["passed"] = not problems
    diag["problems"] = problems
    if problems:
        raise Stage1ValidationError("; ".join(problems))
    return diag


# ─────────────────────────────────────────────────────────────────────────────
#  Result record (provenance sidecar)
# ─────────────────────────────────────────────────────────────────────────────

def build_result_record(spec: Stage1Spec, mol, mf, diagnostics: Mapping[str, Any], *, mf_sha256: str,
                        mf_size: int, execution: Mapping[str, Any]) -> dict:
    """The complete Stage 1 provenance record.  Flat legacy keys (``stage``, ``system``, ``basis``, ``hf_energy`` ...)
    are kept so older readers of the sidecar still work; every value is the ACTUAL one, never a challenge default."""
    digest = spec.digest()
    return {
        "schema": RESULT_SCHEMA,
        "stage": "hf",
        "run_name": spec.tag,
        "system": spec.molecule.name,
        "reference": spec.config.method,
        "basis": spec.config.basis,
        "charge": int(mol.charge),
        "spin": int(mol.spin),
        "hf_energy": float(mf.e_tot),
        "hf_converged": bool(mf.converged),
        "n_ao": int(mol.nao),
        "n_electrons": int(mol.nelectron),
        "stage1_param_digest": digest,
        "geometry_digest": spec.molecule.geometry_digest(),
        "spec": spec.to_dict(),
        "execution": dict(execution),
        "diagnostics": dict(diagnostics),
        "artifact": {"mf_data_pkl_sha256": mf_sha256, "mf_data_pkl_bytes": int(mf_size)},
    }
