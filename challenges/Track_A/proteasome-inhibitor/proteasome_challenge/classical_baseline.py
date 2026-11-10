"""
classical_baseline — contract for the all-classical (mixed FCI/CCSD) EWF baseline.

The baseline is **not** an all-CCSD calculation: the frozen map assigns FCI to
the designated small embedded fragments and CCSD to the others, and contains
**zero** SQD/TrimSQD assignments.  It has two jobs:

1. the all-classical molecular reference energy;
2. the source of CCSD diagnostics (taken from the amplitudes of the very same
   CCSD solves, see :mod:`proteasome_challenge.ccsd_diagnostics`) that inform
   which fragments are sent to quantum treatment.

This module is pure (stdlib + numpy/h5py on demand): it builds, validates and
tabulates baseline records.  It never submits a job.

Record contract (``checkpoints/stage3/<ligand>_classical_baseline.json``)
--------------------------------------------------------------------------
::

    schema_version     "classical-baseline/1"
    kind               "all_classical_ewf_baseline"
    ligand
    identity           cache_key, assignments_sha256, hamiltonian_set_sha256,
                       hamiltonians{fid: sha256}, solver_config{...},
                       diagnostics_schema, bundle_api
    software           {pyscf, numpy, scipy}   (compatibility metadata)
    solver_counts      n_fci, n_ccsd, n_sqd(=0), n_trimsqd(=0)
    energies           e_hf, e_corr, e_total
    solver_assignments {fid: "FCI"|"CCSD"}
    fragments          [{fragment_id, solver, n_orb, energy_ha, e_corr_ha,
                         e_corr_cumulant_ha, converged, hamiltonian_hash,
                         ccsd_diagnostics: <ccsd_diagnostics record>}]
    diagnostics_sha256 digest of the per-fragment diagnostic records
    created_utc

Cache identity
--------------
``cache_key = sha256(ligand, assignments_sha256, hamiltonian_set_sha256,
solver_config, diagnostics_schema)``.  Any change to a fragment Hamiltonian,
to the frozen solver map, to the solver tolerances, or to the diagnostic schema
makes an existing baseline *incompatible*: :func:`validate_baseline_record`
rejects it instead of reusing it.  Software versions are recorded but not part
of the key (they are reported, so a version drift is visible).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import ccsd_diagnostics as cd

SCHEMA_VERSION = "classical-baseline/1"
KIND = "all_classical_ewf_baseline"
CLASSICAL_SOLVERS = frozenset({"FCI", "CCSD"})
CONSISTENCY_TOL = 1.0e-6          # Ha, same tolerance as Challenge.validate_full_ewf

NOT_RETAINED_REASON = (
    "diagnostics unavailable: this baseline was produced before CCSD amplitudes were "
    "retained (bundle API < 3). Amplitudes cannot be reconstructed from energies; a fresh "
    "user-executed baseline run is required."
)


class BaselineError(ValueError):
    """The baseline record is missing, malformed, inconsistent, or stale."""


def _digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Identity
# ─────────────────────────────────────────────────────────────────────────────

def hamiltonian_hashes_from_dumpfile(h5_path: Path | str) -> dict[int, str]:
    """``{fragment_id: sha256(heff.tobytes()+eris.tobytes())}`` from the Stage 2 dump.

    Pure file read (no chemistry).  Same definition as
    :func:`proteasome_challenge.ccsd_diagnostics.hamiltonian_hash`.
    """
    import h5py

    out: dict[int, str] = {}
    with h5py.File(str(h5_path), "r") as hf:
        for key in hf:
            if not key.startswith("fragment_"):
                continue
            g = hf[key]
            out[int(key.split("_", 1)[1])] = cd.hamiltonian_hash(g["heff"][:], g["eris"][:])
    if not out:
        raise BaselineError(f"No fragment_* groups found in {h5_path}.")
    return dict(sorted(out.items()))


def hamiltonian_set_digest(hashes: Mapping[int, str]) -> str:
    return _digest({str(int(k)): v for k, v in sorted(hashes.items())})


def assignments_digest(assignments: Mapping[int, str]) -> str:
    return _digest({str(int(k)): str(v) for k, v in sorted(assignments.items())})


def assert_classical_assignments(assignments: Mapping[int, str]) -> None:
    """The baseline may contain only FCI and CCSD assignments."""
    bad = {int(k): v for k, v in assignments.items() if v not in CLASSICAL_SOLVERS}
    if bad:
        raise BaselineError(
            "The all-classical baseline must contain zero SQD/TrimSQD assignments and only "
            f"FCI/CCSD; found {dict(sorted(bad.items()))}.")
    if not assignments:
        raise BaselineError("Empty solver-assignment map.")


def default_solver_config(fci_threshold: int = 10, sqd_threshold: int = 20) -> dict:
    """Solver configuration requested for the baseline.

    Must equal what ``step3_solve.run_solve`` echoes back as ``solver_config``
    (the CCSD tolerances are the ``CCSD()`` constructor defaults).
    """
    return {
        "fci_threshold": int(fci_threshold),
        "sqd_threshold": int(sqd_threshold),
        "use_threshold_fallback": False,
        "ccsd": {"conv_tol": 1e-7, "max_cycle": 50, "diis_space": 6},
    }


def compute_identity(
    ligand: str,
    hamiltonians: Mapping[int, str],
    assignments: Mapping[int, str],
    solver_config: Mapping[str, Any],
    *,
    bundle_api: int,
) -> dict:
    assert_classical_assignments(assignments)
    if set(int(k) for k in hamiltonians) != set(int(k) for k in assignments):
        raise BaselineError(
            "Hamiltonian set and solver-assignment map cover different fragments "
            f"({sorted(set(hamiltonians) ^ set(assignments))}).")
    a_sha = assignments_digest(assignments)
    h_sha = hamiltonian_set_digest(hamiltonians)
    cfg = json.loads(json.dumps(solver_config, default=str))
    cache_key = _digest({
        "ligand": ligand, "assignments_sha256": a_sha, "hamiltonian_set_sha256": h_sha,
        "solver_config": cfg, "diagnostics_schema": cd.SCHEMA_VERSION,
    })
    return {
        "cache_key": cache_key,
        "assignments_sha256": a_sha,
        "hamiltonian_set_sha256": h_sha,
        "hamiltonians": {str(int(k)): v for k, v in sorted(hamiltonians.items())},
        "solver_config": cfg,
        "diagnostics_schema": cd.SCHEMA_VERSION,
        "bundle_api": int(bundle_api),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 structural reference (no calculation, no paid job)
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURAL_SCHEMA_VERSION = "structural-reference/1"
STRUCTURAL_KIND = "stage2_structural_reference"
STRUCTURAL_PROVENANCE = (
    "Structural view of the Stage 2 embedded fragments (identity, Hamiltonian hashes, orbital and electron "
    "counts) read from the Stage 2 dump. NO classical baseline was computed: it carries no energies and no "
    "CCSD amplitudes, and it is not a valid classical-baseline record.")


def fragment_dims_from_dumpfile(h5_path: Path | str) -> dict[int, tuple[int, int]]:
    """``{fragment_id: (norb, nocc)}`` from the Stage 2 dump attributes.  Pure file read."""
    import h5py

    out: dict[int, tuple[int, int]] = {}
    with h5py.File(str(h5_path), "r") as hf:
        for key in hf:
            if not key.startswith("fragment_"):
                continue
            attrs = hf[key].attrs
            if "norb" not in attrs or "nocc" not in attrs:
                raise BaselineError(f"{key}: the Stage 2 dump lacks the norb/nocc attributes.")
            out[int(key.split("_", 1)[1])] = (int(attrs["norb"]), int(attrs["nocc"]))
    if not out:
        raise BaselineError(f"No fragment_* groups found in {h5_path}.")
    return dict(sorted(out.items()))


def build_structural_reference(
    ligand: str, *, identity: Mapping[str, Any], assignments: Mapping[int, str],
    dims: Mapping[int, tuple[int, int]],
) -> dict:
    """The per-fragment structure a protocol / manifest / plan binds to, WITHOUT running the baseline.

    Same ``identity`` (hence the same ``cache_key``) the paid baseline would carry for these Stage 2
    Hamiltonians.  Energies are deliberately absent; CCSD amplitudes are recorded as unavailable with a reason.
    """
    assert_classical_assignments(assignments)
    ham = {int(k): v for k, v in identity["hamiltonians"].items()}
    if set(ham) != {int(k) for k in assignments} or set(ham) != {int(k) for k in dims}:
        raise BaselineError("Hamiltonians, solver assignments and Stage 2 dimensions cover different fragments.")
    rows = []
    for fid in sorted(ham):
        norb, nocc = dims[fid]
        nvir = norb - nocc
        solver = assignments[fid]
        if solver == "CCSD":
            rec = cd.unavailable_record(
                "structural view only: no classical baseline was computed for this run.",
                hamiltonian_hash_value=ham[fid], n_correlated_electrons=2 * nocc, nocc=nocc, nvir=nvir)
        else:
            rec = cd.not_applicable_record(solver, hamiltonian_hash_value=ham[fid], nocc=nocc, nvir=nvir)
        problems = cd.validate_record(rec)
        if problems:
            raise BaselineError(f"Fragment {fid}: invalid structural diagnostic record: {problems}")
        rows.append({"fragment_id": fid, "solver": solver, "n_orb": int(norb), "hamiltonian_hash": ham[fid],
                     "ccsd_diagnostics": rec})
    return {
        "schema_version": STRUCTURAL_SCHEMA_VERSION, "kind": STRUCTURAL_KIND, "ligand": ligand,
        "identity": dict(identity), "provenance": STRUCTURAL_PROVENANCE, "baseline_computed": False,
        "solver_assignments": {str(k): v for k, v in sorted(assignments.items())}, "fragments": rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Building a record from a Stage 3 result
# ─────────────────────────────────────────────────────────────────────────────

def _by_id(mapping: Mapping, fid: int):
    if int(fid) in mapping:
        return mapping[int(fid)]
    return mapping.get(str(int(fid)))


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def build_baseline_record(
    result: Mapping[str, Any],
    *,
    ligand: str,
    expected_identity: Mapping[str, Any],
    assignments: Mapping[int, str],
    expected_bundle_api: int,
    now: _dt.datetime | None = None,
) -> dict:
    """Validate a Stage 3 result and convert it into a persisted baseline record."""
    assert_classical_assignments(assignments)
    if not isinstance(result, Mapping) or result.get("bundle_api") != expected_bundle_api:
        raise BaselineError(
            f"Stage 3 result lacks bundle_api={expected_bundle_api}; the deployed function is "
            "stale and does not emit CCSD diagnostics. Redeploy the bundle.")
    for key in ("n_sqd", "n_trimsqd"):
        if int(result.get(key, -1)) != 0:
            raise BaselineError(
                f"Baseline result reports {key}={result.get(key)!r}; the all-classical "
                "baseline must contain zero SQD/TrimSQD fragments.")
    used = result.get("solver_assignments_used")
    if not isinstance(used, Mapping):
        raise BaselineError(
            "Stage 3 result did not echo solver_assignments_used; it was not run from the "
            "frozen assignment map (threshold routing is not accepted for the baseline).")
    if {int(k): v for k, v in used.items()} != {int(k): v for k, v in assignments.items()}:
        raise BaselineError("The assignment map used by Stage 3 differs from the frozen map.")
    if result.get("use_threshold_fallback") is not False:
        raise BaselineError("Baseline must be run with use_threshold_fallback=False.")
    reported_cfg = json.loads(json.dumps(result.get("solver_config"), default=str))
    if reported_cfg != expected_identity["solver_config"]:
        raise BaselineError(
            "The solver configuration reported by Stage 3 differs from the requested one "
            f"({reported_cfg!r} vs {expected_identity['solver_config']!r}).")

    for k in ("e_hf", "e_corr", "e_total"):
        if not _finite(result.get(k)):
            raise BaselineError(f"Stage 3 result has non-finite {k}: {result.get(k)!r}")
    e_hf, e_corr, e_total = (float(result[k]) for k in ("e_hf", "e_corr", "e_total"))
    if abs(e_total - (e_hf + e_corr)) > CONSISTENCY_TOL:
        raise BaselineError(
            f"Inconsistent energies: e_total={e_total!r} != e_hf+e_corr={e_hf + e_corr!r}.")

    summaries = result.get("fragment_summaries") or {}
    recon = result.get("fragments") or {}
    expected_h = {int(k): v for k, v in expected_identity["hamiltonians"].items()}
    if {int(k) for k in summaries} != set(expected_h):
        raise BaselineError(
            "fragment_summaries do not cover exactly the expected fragments "
            f"(difference: {sorted({int(k) for k in summaries} ^ set(expected_h))}).")

    rows: list[dict] = []
    cumulant_sum = 0.0
    for fid in sorted(expected_h):
        s = _by_id(summaries, fid)
        r = _by_id(recon, fid) or {}
        solver = s.get("solver")
        if solver != assignments[fid]:
            raise BaselineError(
                f"Fragment {fid}: solved by {solver!r} but the frozen map assigns "
                f"{assignments[fid]!r} (silent fallback?).")
        h = s.get("hamiltonian_hash")
        if h != expected_h[fid]:
            raise BaselineError(
                f"Fragment {fid}: solver-side Hamiltonian hash does not match the Stage 2 "
                "dump. The baseline was computed on a different embedded Hamiltonian.")
        nocc = s.get("n_occ")
        nvir = None if nocc is None or s.get("n_orb") is None else int(s["n_orb"]) - int(nocc)
        if solver == "CCSD":
            rec = s.get("ccsd_diagnostics")
            if not isinstance(rec, Mapping):
                rec = cd.unavailable_record(
                    "the Stage 3 result carries no CCSD diagnostic record for this fragment.",
                    converged=s.get("converged"), correlation_energy=s.get("e_corr_ha"),
                    hamiltonian_hash_value=h,
                    n_correlated_electrons=None if nocc is None else 2 * int(nocc),
                    nocc=nocc, nvir=nvir)
            rec = dict(rec)
            problems = cd.validate_record(rec)
            if problems:
                raise BaselineError(f"Fragment {fid}: invalid diagnostic record: {problems}")
            if rec["hamiltonian_hash"] not in (None, h):
                raise BaselineError(
                    f"Fragment {fid}: diagnostic record Hamiltonian hash differs from the solve.")
        else:
            rec = cd.not_applicable_record(
                solver, converged=s.get("converged"), correlation_energy=s.get("e_corr_ha"),
                hamiltonian_hash_value=h, nocc=nocc, nvir=nvir)
        e_c = r.get("e_corr")
        if _finite(e_c):
            cumulant_sum += float(e_c)
        rows.append({
            "fragment_id": fid,
            "solver": solver,
            "n_orb": int(s["n_orb"]),
            "energy_ha": float(s["energy_ha"]),
            "e_corr_ha": float(s["e_corr_ha"]),
            "e_corr_cumulant_ha": float(e_c) if _finite(e_c) else None,
            "converged": bool(s["converged"]),
            "hamiltonian_hash": h,
            "ccsd_diagnostics": rec,
        })
    if all(x["e_corr_cumulant_ha"] is not None for x in rows) and abs(cumulant_sum - e_corr) > CONSISTENCY_TOL:
        raise BaselineError(
            f"Σ fragment cumulant e_corr = {cumulant_sum!r} does not match e_corr = {e_corr!r}.")

    n_fci = sum(1 for x in rows if x["solver"] == "FCI")
    n_ccsd = sum(1 for x in rows if x["solver"] == "CCSD")
    now = now or _dt.datetime.now(_dt.timezone.utc)
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "ligand": ligand,
        "identity": dict(expected_identity),
        "software": dict(result.get("software") or {}),
        "solver_counts": {"n_fci": n_fci, "n_ccsd": n_ccsd, "n_sqd": 0, "n_trimsqd": 0},
        "energies": {"e_hf": e_hf, "e_corr": e_corr, "e_total": e_total},
        "solver_assignments": {str(k): v for k, v in sorted(assignments.items())},
        "fragments": rows,
        "created_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    record["diagnostics_sha256"] = diagnostics_digest(record)
    return record


def diagnostics_digest(record: Mapping[str, Any]) -> str:
    """Digest of the per-fragment diagnostic records (what the final selection binds to)."""
    return _digest([
        {"fragment_id": f["fragment_id"], "ccsd_diagnostics": f["ccsd_diagnostics"]}
        for f in sorted(record["fragments"], key=lambda x: x["fragment_id"])
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Validation of a stored record (no recomputation)
# ─────────────────────────────────────────────────────────────────────────────

def validate_baseline_record(
    record: Mapping[str, Any], *, expected_identity: Mapping[str, Any] | None = None,
) -> dict:
    """Structural + provenance validation of a stored baseline.  Raises BaselineError.

    With ``expected_identity`` (computed from the *current* Stage 2 dump, frozen
    map and solver config) a stale record is rejected.
    """
    if not isinstance(record, Mapping) or record.get("kind") != KIND:
        raise BaselineError("Not a classical-baseline record.")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise BaselineError(
            f"Baseline schema_version={record.get('schema_version')!r} (expected {SCHEMA_VERSION!r}).")
    for k in ("identity", "energies", "solver_assignments", "fragments", "solver_counts",
              "diagnostics_sha256"):
        if k not in record:
            raise BaselineError(f"Baseline record is missing {k!r}.")
    assert_classical_assignments({int(k): v for k, v in record["solver_assignments"].items()})
    sc = record["solver_counts"]
    if sc.get("n_sqd") != 0 or sc.get("n_trimsqd") != 0:
        raise BaselineError("Baseline records SQD/TrimSQD fragments; it is not all-classical.")
    for f in record["fragments"]:
        if f["solver"] not in CLASSICAL_SOLVERS:
            raise BaselineError(f"Fragment {f['fragment_id']}: solver {f['solver']!r} is not FCI/CCSD.")
        problems = cd.validate_record(f["ccsd_diagnostics"])
        if problems:
            raise BaselineError(f"Fragment {f['fragment_id']}: invalid diagnostics: {problems}")
    if diagnostics_digest(record) != record["diagnostics_sha256"]:
        raise BaselineError("Diagnostics digest mismatch: the stored diagnostics were altered.")
    ident = record["identity"]
    recomputed = compute_identity(
        record["ligand"],
        {int(k): v for k, v in ident["hamiltonians"].items()},
        {int(k): v for k, v in record["solver_assignments"].items()},
        ident["solver_config"], bundle_api=ident["bundle_api"])["cache_key"]
    if recomputed != ident["cache_key"]:
        raise BaselineError("Baseline cache_key is inconsistent with its own contents.")
    if expected_identity is not None:
        for k, label in (("hamiltonian_set_sha256", "embedded Hamiltonians"),
                         ("assignments_sha256", "frozen solver-assignment map"),
                         ("cache_key", "cache identity")):
            if ident.get(k) != expected_identity.get(k):
                raise BaselineError(
                    f"STALE classical baseline: the {label} changed since it was computed "
                    f"({k} mismatch). It is not reused; a fresh user-executed baseline is required.")
        if ident.get("solver_config") != expected_identity.get("solver_config"):
            raise BaselineError(
                "STALE classical baseline: the solver configuration changed since it was computed.")
    return dict(record)


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

DIAGNOSTIC_TABLE_COLUMNS = [
    "fragment_id", "classical_solver", "diagnostics_status",
    "ccsd_t1_diagnostic", "ccsd_d1_diagnostic", "ccsd_max_abs_t1",
    "ccsd_max_abs_t2", "ccsd_t2_rms",
    "ccsd_max_abs_t2_mp2_convention", "ccsd_t2_rms_mp2_convention",
    "ccsd_converged", "ccsd_correlation_energy", "n_correlated_electrons",
    "hamiltonian_hash", "unavailable_reason",
]


def diagnostics_rows(record: Mapping[str, Any]) -> list[dict]:
    """One flat row per fragment.  Unavailable values are ``None`` (never 0)."""
    rows = []
    for f in sorted(record["fragments"], key=lambda x: x["fragment_id"]):
        d = f["ccsd_diagnostics"]
        reasons = d.get("unavailable") or {}
        reason = reasons.get("ccsd_t1_diagnostic")
        rows.append({
            "fragment_id": int(f["fragment_id"]),
            "classical_solver": f["solver"],
            "diagnostics_status": d["diagnostics_status"],
            "ccsd_t1_diagnostic": d["ccsd_t1_diagnostic"],
            "ccsd_d1_diagnostic": d["ccsd_d1_diagnostic"],
            "ccsd_max_abs_t1": d["ccsd_max_abs_t1"],
            "ccsd_max_abs_t2": d["ccsd_max_abs_t2"],
            "ccsd_t2_rms": d["ccsd_t2_rms"],
            "ccsd_max_abs_t2_mp2_convention": d["ccsd_max_abs_t2_mp2_convention"],
            "ccsd_t2_rms_mp2_convention": d["ccsd_t2_rms_mp2_convention"],
            "ccsd_converged": d["ccsd_converged"],
            "ccsd_correlation_energy": d["ccsd_correlation_energy"],
            "n_correlated_electrons": d["n_correlated_electrons"],
            "hamiltonian_hash": f["hamiltonian_hash"],
            "unavailable_reason": reason,
        })
    return rows


def legacy_baseline_status(path: Path | str) -> dict:
    """Inspect a pre-diagnostics classical EWF checkpoint WITHOUT using its numbers as diagnostics.

    Returns ``{"present", "diagnostics_available", "n_ccsd_fragments", "reason"}``.
    """
    p = Path(path)
    if not p.is_file():
        return {"present": False, "diagnostics_available": False, "n_ccsd_fragments": 0, "reason": None}
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    summ = data.get("fragment_summaries") or {}
    n_ccsd = sum(1 for s in summ.values() if isinstance(s, Mapping) and s.get("solver") == "CCSD")
    has = any(isinstance(s, Mapping) and isinstance(s.get("ccsd_diagnostics"), Mapping)
              for s in summ.values())
    return {
        "present": True,
        "diagnostics_available": bool(has),
        "n_ccsd_fragments": n_ccsd,
        "reason": None if has else NOT_RETAINED_REASON,
        "e_total_ha": data.get("e_total"),
    }


def ranks(values: Iterable[float | None]) -> list[float | None]:
    """Average ranks (1 = smallest) for finite values; ``None`` stays ``None``."""
    vals = list(values)
    idx = [i for i, v in enumerate(vals) if v is not None and math.isfinite(v)]
    order = sorted(idx, key=lambda i: vals[i])
    out: list[float | None] = [None] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out
