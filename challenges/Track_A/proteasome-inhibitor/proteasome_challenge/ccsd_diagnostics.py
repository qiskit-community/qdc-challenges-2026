"""
ccsd_diagnostics — coupled-cluster amplitude diagnostics for embedded fragments.

This module is deliberately **self-contained** (numpy + stdlib only, no
relative imports).  The same file is shipped byte-for-byte in three places:

* ``proteasome_challenge/ccsd_diagnostics.py``                       (notebook side)
* ``serverless_bundle/quantum_fragment_methods/.../classical_zoo/``  (Stage 3 function)
* ``ewf_sqd_serverless/quantum_fragment_methods/.../classical_zoo/`` (dev copy)

A unit test asserts the three copies are identical, so the diagnostics computed
inside the Stage 3 CCSD solve and the ones validated on the notebook side can
never drift apart.

Why diagnostics are computed inside the solve
---------------------------------------------
``pyscf.cc.CCSD.kernel()`` returns ``(e_corr, t1, t2)`` and the baseline EWF
needs the same solve for its RDMs.  The diagnostics are therefore extracted
from the converged amplitudes of **that** solve; no second correlated
calculation is ever launched to obtain them.

Amplitude representation (read this before comparing any numbers)
-----------------------------------------------------------------
The embedded-cluster CCSD wrapper runs a *closed-shell restricted* HF on the
fragment Hamiltonian and then ``pyscf.cc.CCSD`` (RCCSD).  The amplitudes are
therefore **spin-adapted spatial-orbital** tensors in the canonical RHF MO
basis of that re-run SCF, occupied orbitals first:

* ``t1[i, a]``        shape ``(nocc, nvir)``
* ``t2[i, j, a, b]``  shape ``(nocc, nocc, nvir, nvir)`` — the alpha-beta
  amplitude ``t(i_alpha j_beta -> a_alpha b_beta)``.

Conventions used here
~~~~~~~~~~~~~~~~~~~~~
``N_corr = 2 * nocc``            correlated electrons (no frozen core in the
                                 embedded cluster Hamiltonian).

``T1  = ||t1||_F / sqrt(N_corr)`` Lee & Taylor, Int. J. Quantum Chem. 36(S23),
                                 199 (1989).  Identical to
                                 ``pyscf.cc.ccsd.get_t1_diagnostic``.

``D1  = max(sqrt(lambda_max(t1 t1^T)), sqrt(lambda_max(t1^T t1)))``
                                 Janssen & Nielsen, Chem. Phys. Lett. 290, 423
                                 (1998): the spectral (2-)norm of the singles
                                 matrix, built from the *restricted spatial*
                                 ``t1`` with no extra spin factor.  Identical
                                 to ``pyscf.cc.ccsd.get_d1_diagnostic``.  Values
                                 from codes that rescale ``t1`` differently are
                                 not directly comparable.

``max|t1|``, ``max|t2|``         largest absolute entries of the native
                                 restricted tensors above.

``t2_rms = ||t2||_F / sqrt(nocc^2 nvir^2)``
                                 Same normalization as ``mp2_t2_rms``.

MP2-comparable doubles
~~~~~~~~~~~~~~~~~~~~~~
``mp2_diagnostics.compute_fragment_mp2_diagnostics`` evaluates
``t_ij^ab = <ij||ab>/D`` on spatial indices, i.e. the **same-spin
(antisymmetrized)** amplitude, not the alpha-beta amplitude that RCCSD stores.
Comparing ``max|t2_alpha_beta|`` with ``mp2_max_abs_t2`` would silently mix two
representations.  To permit a like-for-like comparison this module also
reports

    t2_ss[i, j, a, b] = t2[i, j, a, b] - t2[i, j, b, a]

(exactly the relation that maps the MP2 alpha-beta amplitude ``(ia|jb)/D`` onto
the MP2 antisymmetrized amplitude) as ``ccsd_max_abs_t2_mp2_convention`` and
``ccsd_t2_rms_mp2_convention``.  Both are tagged in ``orbital_reference``.

D2
~~
Not implemented.  PySCF exposes ``get_d2_diagnostic`` (Nielsen & Janssen,
Chem. Phys. Lett. 310, 568 (1999)) but no independent validation reference
exists in this repository; no approximation is substituted.

Missing values
~~~~~~~~~~~~~~
A diagnostic that cannot be evaluated is ``None`` with a human-readable reason
in ``record["unavailable"]``.  ``0.0`` is never used as a placeholder.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping

import numpy as np

SCHEMA_VERSION = "ccsd-diagnostics/1"

#: Representation tag attached to every record.
AMPLITUDE_REPRESENTATION = "restricted_closed_shell_spatial_orbital"

#: How ``ccsd_t2_rms`` is normalised.
T2_RMS_CONVENTION = "||t2||_F / sqrt(nocc^2 * nvir^2); native alpha-beta tensor t2[i,j,a,b]"

#: Per-record statuses.
STATUS_AVAILABLE = "available"
STATUS_UNAVAILABLE = "unavailable"
STATUS_NOT_APPLICABLE = "not_applicable"

#: Fields required by the result schema (task specification, section 2).
CORE_FIELDS: tuple[str, ...] = (
    "ccsd_t1_diagnostic",
    "ccsd_d1_diagnostic",
    "ccsd_max_abs_t1",
    "ccsd_max_abs_t2",
    "ccsd_t2_rms",
    "ccsd_converged",
    "ccsd_correlation_energy",
    "hamiltonian_hash",
    "orbital_reference",
    "n_correlated_electrons",
)

#: Numeric fields that must be ``None`` (never 0) when unavailable.
NUMERIC_AMPLITUDE_FIELDS: tuple[str, ...] = (
    "ccsd_t1_diagnostic",
    "ccsd_d1_diagnostic",
    "ccsd_max_abs_t1",
    "ccsd_max_abs_t2",
    "ccsd_t2_rms",
    "ccsd_max_abs_t2_mp2_convention",
    "ccsd_t2_rms_mp2_convention",
)

#: Every key a record carries.
RECORD_FIELDS: tuple[str, ...] = CORE_FIELDS + (
    "ccsd_max_abs_t2_mp2_convention",
    "ccsd_t2_rms_mp2_convention",
    "ccsd_d2_diagnostic",
    "schema_version",
    "diagnostics_status",
    "amplitude_representation",
    "t2_rms_convention",
    "unavailable",
    "n_occ",
    "n_vir",
)

D2_UNAVAILABLE_REASON = (
    "D2 not implemented: no independently validated implementation is available "
    "in this repository (PySCF get_d2_diagnostic, Nielsen & Janssen 1999, has "
    "not been validated here); no approximation is substituted."
)

_CONSISTENCY_TOL = 1.0e-9


class DiagnosticRepresentationError(ValueError):
    """Raised when amplitudes are not in the supported restricted representation."""


# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonian identity
# ─────────────────────────────────────────────────────────────────────────────

def hamiltonian_hash(h1e: np.ndarray, h2e: np.ndarray) -> str:
    """SHA-256 identity of one embedded Hamiltonian.

    Definition (identical to the Stage 3 single-fragment entrypoint and to the
    Stage 2 dump layout): ``sha256(heff.tobytes() + eris.tobytes())`` over the
    arrays exactly as stored in ``ewf_dumpfile.h5`` (float64, C order).
    """
    return hashlib.sha256(np.asarray(h1e).tobytes() + np.asarray(h2e).tobytes()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Pure formulas
# ─────────────────────────────────────────────────────────────────────────────

def _check_t1(t1: Any, nocc: int | None = None, nvir: int | None = None) -> np.ndarray:
    if isinstance(t1, (tuple, list)):
        raise DiagnosticRepresentationError(
            "t1 is a tuple/list (unrestricted alpha/beta amplitudes); only restricted "
            "closed-shell spatial-orbital t1 of shape (nocc, nvir) is supported."
        )
    a = np.asarray(t1)
    if a.ndim != 2:
        raise DiagnosticRepresentationError(
            f"t1 must be 2-D (nocc, nvir) restricted spatial amplitudes; got ndim={a.ndim}."
        )
    if not np.issubdtype(a.dtype, np.floating):
        raise DiagnosticRepresentationError(f"t1 must be real floating point; got {a.dtype}.")
    if nocc is not None and a.shape[0] != nocc:
        raise DiagnosticRepresentationError(f"t1.shape[0]={a.shape[0]} != nocc={nocc}.")
    if nvir is not None and a.shape[1] != nvir:
        raise DiagnosticRepresentationError(f"t1.shape[1]={a.shape[1]} != nvir={nvir}.")
    if not np.all(np.isfinite(a)):
        raise DiagnosticRepresentationError("t1 contains non-finite values.")
    return a.astype(np.float64, copy=False)


def _check_t2(t2: Any, nocc: int, nvir: int) -> np.ndarray:
    if isinstance(t2, (tuple, list)):
        raise DiagnosticRepresentationError(
            "t2 is a tuple/list (unrestricted spin blocks); only the restricted "
            "(nocc, nocc, nvir, nvir) alpha-beta tensor is supported."
        )
    a = np.asarray(t2)
    if a.shape != (nocc, nocc, nvir, nvir):
        raise DiagnosticRepresentationError(
            f"t2 must have restricted shape {(nocc, nocc, nvir, nvir)}; got {a.shape}."
        )
    if not np.issubdtype(a.dtype, np.floating):
        raise DiagnosticRepresentationError(f"t2 must be real floating point; got {a.dtype}.")
    if not np.all(np.isfinite(a)):
        raise DiagnosticRepresentationError("t2 contains non-finite values.")
    return a.astype(np.float64, copy=False)


def t1_diagnostic(t1: np.ndarray, n_correlated_electrons: int | None = None) -> float:
    """Lee–Taylor T1 = ||t1||_F / sqrt(N_corr).

    For closed-shell restricted spatial ``t1`` the default is
    ``N_corr = 2 * t1.shape[0]``; an explicit different value is rejected so a
    wrong electron count cannot be applied silently.
    """
    a = _check_t1(t1)
    n_default = 2 * a.shape[0]
    n = n_default if n_correlated_electrons is None else int(n_correlated_electrons)
    if n != n_default:
        raise DiagnosticRepresentationError(
            f"n_correlated_electrons={n} is inconsistent with restricted closed-shell "
            f"t1 (expected 2*nocc={n_default})."
        )
    return float(math.sqrt(float(np.sum(a * a)) / n))


def d1_diagnostic(t1: np.ndarray) -> float:
    """Janssen–Nielsen D1: spectral norm of the restricted singles matrix.

    Built independently of PySCF: the larger of sqrt(lambda_max(t1 t1^T)) and
    sqrt(lambda_max(t1^T t1)) (the two share their non-zero eigenvalues, so the
    result equals the largest singular value of ``t1``).
    """
    a = _check_t1(t1)
    lam_oo = float(np.linalg.eigvalsh(a @ a.T)[-1])
    lam_vv = float(np.linalg.eigvalsh(a.T @ a)[-1])
    return float(math.sqrt(max(lam_oo, lam_vv, 0.0)))


def t2_same_spin(t2: np.ndarray) -> np.ndarray:
    """MP2-convention (antisymmetrized, same-spin) doubles from alpha-beta ``t2``.

    ``t_ss[i,j,a,b] = t2[i,j,a,b] - t2[i,j,b,a]``.
    """
    a = np.asarray(t2, dtype=np.float64)
    if a.ndim != 4:
        raise DiagnosticRepresentationError("t2 must be 4-D.")
    return a - np.swapaxes(a, 2, 3)


def t2_rms(t2: np.ndarray) -> float:
    """RMS doubles amplitude = ||t2||_F / sqrt(t2.size) (nocc^2 nvir^2 elements)."""
    a = np.asarray(t2, dtype=np.float64)
    return float(np.linalg.norm(a) / math.sqrt(a.size))


# ─────────────────────────────────────────────────────────────────────────────
# Record construction
# ─────────────────────────────────────────────────────────────────────────────

def _blank_record(status: str, reason: str | None) -> dict[str, Any]:
    rec: dict[str, Any] = {k: None for k in RECORD_FIELDS}
    rec["schema_version"] = SCHEMA_VERSION
    rec["diagnostics_status"] = status
    rec["amplitude_representation"] = AMPLITUDE_REPRESENTATION
    rec["t2_rms_convention"] = T2_RMS_CONVENTION
    rec["unavailable"] = {"ccsd_d2_diagnostic": D2_UNAVAILABLE_REASON}
    if reason is not None:
        for f in NUMERIC_AMPLITUDE_FIELDS:
            rec["unavailable"][f] = reason
    return rec


def orbital_reference(
    *,
    mo_energy: Iterable[float] | None = None,
    nocc: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reference-orbital / representation metadata attached to every record."""
    ref: dict[str, Any] = {
        "representation": AMPLITUDE_REPRESENTATION,
        "orbital_basis": "canonical_rhf_mos_of_rerun_scf_on_embedded_hamiltonian",
        "orbital_ordering": "occupied_first",
        "spin_treatment": "closed_shell_restricted_spin_adapted",
        "t1_index_order": "t1[i,a]",
        "t2_index_order": "t2[i,j,a,b] = t(i_alpha j_beta -> a_alpha b_beta)",
        "frozen_core": False,
        "scf_homo_ha": None,
        "scf_lumo_ha": None,
        "scf_gap_ha": None,
    }
    if mo_energy is not None and nocc is not None:
        e = np.asarray(list(mo_energy), dtype=np.float64)
        if e.ndim == 1 and 0 < nocc < e.size:
            ref["scf_homo_ha"] = float(e[nocc - 1])
            ref["scf_lumo_ha"] = float(e[nocc])
            ref["scf_gap_ha"] = float(e[nocc] - e[nocc - 1])
    if extra:
        ref.update(dict(extra))
    return ref


def unavailable_record(
    reason: str,
    *,
    status: str = STATUS_UNAVAILABLE,
    converged: bool | None = None,
    correlation_energy: float | None = None,
    hamiltonian_hash_value: str | None = None,
    n_correlated_electrons: int | None = None,
    nocc: int | None = None,
    nvir: int | None = None,
    orbital_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Explicit 'no diagnostics' record.  Numeric amplitude fields are ``None``."""
    if status not in (STATUS_UNAVAILABLE, STATUS_NOT_APPLICABLE):
        raise ValueError(f"status must be unavailable/not_applicable, got {status!r}")
    if not reason or not str(reason).strip():
        raise ValueError("an unavailable record must carry a non-empty reason")
    rec = _blank_record(status, str(reason))
    rec["ccsd_converged"] = None if converged is None else bool(converged)
    rec["ccsd_correlation_energy"] = None if correlation_energy is None else float(correlation_energy)
    rec["hamiltonian_hash"] = hamiltonian_hash_value
    rec["n_correlated_electrons"] = None if n_correlated_electrons is None else int(n_correlated_electrons)
    rec["n_occ"] = None if nocc is None else int(nocc)
    rec["n_vir"] = None if nvir is None else int(nvir)
    rec["orbital_reference"] = dict(orbital_ref) if orbital_ref is not None else None
    return rec


def compute_ccsd_diagnostics(
    t1: Any,
    t2: Any,
    *,
    nocc: int,
    nvir: int,
    converged: bool,
    correlation_energy: float,
    hamiltonian_hash_value: str | None,
    orbital_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full diagnostic record from converged restricted amplitudes.

    Never raises for missing/incompatible/non-converged inputs: those yield an
    explicit *unavailable* record (with a reason) so a diagnostics problem can
    never abort the baseline energy calculation.
    """
    n_corr = 2 * int(nocc)
    common = dict(
        converged=converged,
        correlation_energy=correlation_energy,
        hamiltonian_hash_value=hamiltonian_hash_value,
        n_correlated_electrons=n_corr,
        nocc=nocc,
        nvir=nvir,
        orbital_ref=orbital_ref,
    )
    if t1 is None or t2 is None:
        return unavailable_record(
            "CCSD amplitudes (t1/t2) were not retained by the solver; a fresh "
            "user-executed calculation is required.", **common)
    if not converged:
        return unavailable_record(
            "CCSD did not converge; amplitudes are not a stationary solution and "
            "are not reported.", **common)
    try:
        t1a = _check_t1(t1, nocc, nvir)
        t2a = _check_t2(t2, nocc, nvir)
    except DiagnosticRepresentationError as exc:
        return unavailable_record(f"incompatible amplitude representation: {exc}", **common)

    t2ss = t2_same_spin(t2a)
    rec = _blank_record(STATUS_AVAILABLE, None)
    rec["unavailable"] = {"ccsd_d2_diagnostic": D2_UNAVAILABLE_REASON}
    rec.update({
        "ccsd_t1_diagnostic": t1_diagnostic(t1a, n_corr),
        "ccsd_d1_diagnostic": d1_diagnostic(t1a),
        "ccsd_max_abs_t1": float(np.max(np.abs(t1a))),
        "ccsd_max_abs_t2": float(np.max(np.abs(t2a))),
        "ccsd_t2_rms": t2_rms(t2a),
        "ccsd_max_abs_t2_mp2_convention": float(np.max(np.abs(t2ss))),
        "ccsd_t2_rms_mp2_convention": t2_rms(t2ss),
        "ccsd_converged": True,
        "ccsd_correlation_energy": float(correlation_energy),
        "hamiltonian_hash": hamiltonian_hash_value,
        "orbital_reference": dict(orbital_ref) if orbital_ref is not None else orbital_reference(),
        "n_correlated_electrons": n_corr,
        "n_occ": int(nocc),
        "n_vir": int(nvir),
    })
    return rec


def not_applicable_record(
    solver: str,
    *,
    converged: bool | None = None,
    correlation_energy: float | None = None,
    hamiltonian_hash_value: str | None = None,
    nocc: int | None = None,
    nvir: int | None = None,
) -> dict[str, Any]:
    """Record for fragments not treated by CCSD (e.g. FCI)."""
    return unavailable_record(
        f"not applicable: fragment solved with {solver}, which has no coupled-cluster amplitudes.",
        status=STATUS_NOT_APPLICABLE,
        converged=converged,
        correlation_energy=correlation_energy,
        hamiltonian_hash_value=hamiltonian_hash_value,
        n_correlated_electrons=None if nocc is None else 2 * int(nocc),
        nocc=nocc,
        nvir=nvir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation (used on load; never recomputes anything)
# ─────────────────────────────────────────────────────────────────────────────

def validate_record(rec: Mapping[str, Any]) -> list[str]:
    """Return a list of schema/consistency problems (empty list = valid)."""
    problems: list[str] = []
    if not isinstance(rec, Mapping):
        return ["record is not a mapping"]
    if rec.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version {rec.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    missing = [k for k in RECORD_FIELDS if k not in rec]
    if missing:
        problems.append(f"missing fields: {missing}")
        return problems
    status = rec["diagnostics_status"]
    if status not in (STATUS_AVAILABLE, STATUS_UNAVAILABLE, STATUS_NOT_APPLICABLE):
        problems.append(f"unknown diagnostics_status {status!r}")
        return problems
    unavailable = rec["unavailable"] or {}
    if status == STATUS_AVAILABLE:
        for f in NUMERIC_AMPLITUDE_FIELDS:
            v = rec[f]
            if v is None or not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                problems.append(f"{f} must be a finite number when status=available")
            elif v < 0:
                problems.append(f"{f} must be non-negative")
        if rec["ccsd_converged"] is not True:
            problems.append("status=available requires ccsd_converged=True")
        if not rec["hamiltonian_hash"]:
            problems.append("status=available requires hamiltonian_hash")
        if not isinstance(rec["orbital_reference"], Mapping):
            problems.append("status=available requires orbital_reference metadata")
        n = rec["n_correlated_electrons"]
        if not isinstance(n, int) or n <= 0 or n % 2:
            problems.append("n_correlated_electrons must be a positive even integer")
        if not problems:
            t1, d1, m1 = rec["ccsd_t1_diagnostic"], rec["ccsd_d1_diagnostic"], rec["ccsd_max_abs_t1"]
            fro = t1 * math.sqrt(n)
            if d1 > fro + _CONSISTENCY_TOL:
                problems.append("inconsistent: D1 (spectral norm) exceeds ||t1||_F")
            if m1 > d1 + _CONSISTENCY_TOL:
                problems.append("inconsistent: max|t1| exceeds D1")
        if rec["ccsd_d2_diagnostic"] is not None:
            problems.append("ccsd_d2_diagnostic must be None (D2 not implemented)")
    else:
        for f in NUMERIC_AMPLITUDE_FIELDS:
            if rec[f] is not None:
                problems.append(f"{f} must be None when status={status} (never a 0 placeholder)")
            elif not unavailable.get(f):
                problems.append(f"{f} unavailable without a reason")
    return problems


def assert_valid(rec: Mapping[str, Any]) -> None:
    problems = validate_record(rec)
    if problems:
        raise ValueError("invalid CCSD diagnostic record: " + "; ".join(problems))
