"""
final_selection — the FINAL, user-frozen, CCSD-informed quantum-fragment selection.

Why a separate artifact
-----------------------
The MP2-only 4/6/9 panel (:mod:`proteasome_challenge.experiment_selection`) is a
*provisional experimental proposal*.  The final selection is created only after
the all-classical EWF baseline and its CCSD diagnostics exist, and only by an
explicit, user-approved freeze.  This module defines the manifest contract and
its validation.  It contains **no scoring and no thresholds**: the fragment IDs,
roles and rationale are supplied by the user and recorded verbatim.

Contract (``schema_version`` = ``"1.0"``)
-----------------------------------------
::

    selection_stage            "final_ccsd_informed"
    frozen                     true  (a manifest with frozen != true is rejected)
    authorizes_sqd             true  (only here, only when frozen)
    automatic_thresholds       false (recorded; no T1/D1 threshold routes solvers)
    ligand
    approved_by, approved_at_utc
    selection_basis            free-text, non-empty (the user's scientific reasoning)
    selected_fragment_ids      sorted list of ints
    sqd_calibration_fragment_id  one of selected_fragment_ids (Stage E target)
    fragments                  one entry per selected id:
                                 fragment_id, role, rationale,
                                 resource_requirements{...},
                                 classical_solver_in_baseline,
                                 mp2_max_abs_t2, ccsd_diagnostics_summary{...}
    source_provenance          baseline_cache_key, baseline_e_total_ha,
                               diagnostics_sha256, assignments_sha256,
                               hamiltonian_set_sha256,
                               preliminary_manifest_digest (or null)
    manifest_digest            sha256 of the canonical content above

Staleness
---------
A final manifest is bound to the exact baseline it was reviewed against.  If the
baseline cache key or the CCSD-diagnostics digest changes, the manifest is
*stale* and :func:`check_not_stale` raises.  Downstream artifacts (promotion
records, tuning grids, fragment results, full-EWF results) carry the manifest
digest, so changing the selection invalidates them too.

Size-based policy manifest (``schema_version`` = ``"2.0"``)
-----------------------------------------------------------
The participant workflow selects quantum fragments with a deterministic,
transparent orbital-count rule — every fragment with ``n_orb >= min_n_orb`` —
instead of user-reviewed diagnostics.  The v1 contract above is unchanged and
still validates; v2 is a *second* accepted contract, dispatched on
``selection_stage``.  v2 carries **no** diagnostic fields
(``mp2_max_abs_t2``, ``ccsd_diagnostics_summary``, ``diagnostics_sha256`` and
``preliminary_manifest_digest`` are absent, never filled with placeholders)::

    schema_version             "2.0"
    selection_stage            "final_size_based"
    frozen                     true
    authorizes_sqd             true
    ligand
    selection_policy           {rule: "min_spatial_orbitals", field: "n_orb",
                                comparison: ">=", min_n_orb: int,
                                description: str}
    selected_fragment_ids      sorted list of ints
    sqd_calibration_fragment_id  derived, not chosen: smallest n_orb, ties by
                                 lowest fragment_id (single-fragment Stage E
                                 compatibility target; see ``calibration_rule``)
    calibration_rule           str
    fragments                  one entry per selected id:
                                 fragment_id, n_orb, role, rationale,
                                 resource_requirements{n_orb, n_elec},
                                 classical_solver_in_baseline
    source_provenance          catalog_sha256 (sha256 of the ligand's sorted
                               (fragment_id, n_orb) pairs), assignments_sha256
                               (frozen FCI/CCSD solver map)
    manifest_digest            sha256 of the canonical content above

v2 has no timestamp and no approver: it is a pure function of the catalog and
the frozen solver map, so regenerating it yields a byte-identical file.
Staleness is :func:`check_size_manifest_current` — the rule is re-applied to the
current catalog and the catalog / assignment digests are re-compared.  It needs
neither the classical baseline nor any CCSD diagnostics.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "1.0"
SELECTION_STAGE_FINAL = "final_ccsd_informed"

SCHEMA_VERSION_SIZE = "2.0"
SELECTION_STAGE_SIZE = "final_size_based"
SIZE_POLICY_RULE = "min_spatial_orbitals"
SIZE_ROLE = "size_selected_trimsqd_candidate"
CALIBRATION_RULE = "smallest n_orb among selected fragments; ties broken by lowest fragment_id"

_SIZE_REQUIRED_KEYS = {
    "schema_version", "selection_stage", "frozen", "authorizes_sqd", "ligand",
    "selection_policy", "selected_fragment_ids", "sqd_calibration_fragment_id",
    "calibration_rule", "fragments", "source_provenance", "manifest_digest",
}
_SIZE_REQUIRED_PROVENANCE = {"catalog_sha256", "assignments_sha256"}
_SIZE_REQUIRED_FRAGMENT_KEYS = {
    "fragment_id", "n_orb", "role", "rationale", "resource_requirements",
    "classical_solver_in_baseline",
}

_REQUIRED_KEYS = {
    "schema_version", "selection_stage", "frozen", "authorizes_sqd",
    "automatic_thresholds", "ligand", "approved_by", "approved_at_utc",
    "selection_basis", "selected_fragment_ids", "sqd_calibration_fragment_id",
    "fragments", "source_provenance", "manifest_digest",
}
_REQUIRED_PROVENANCE = {
    "baseline_cache_key", "baseline_e_total_ha", "diagnostics_sha256",
    "assignments_sha256", "hamiltonian_set_sha256", "preliminary_manifest_digest",
}
_REQUIRED_FRAGMENT_KEYS = {
    "fragment_id", "role", "rationale", "resource_requirements",
    "classical_solver_in_baseline", "mp2_max_abs_t2", "ccsd_diagnostics_summary",
}


class FinalSelectionError(ValueError):
    """The final selection is missing, malformed, unfrozen, or stale."""


def canonical_digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    return canonical_digest({k: v for k, v in manifest.items() if k != "manifest_digest"})


def _clean(v: Any) -> Any:
    """JSON-safe scalar: NaN/inf -> None (never a 0 placeholder)."""
    try:
        import numpy as np  # local: keep module importable without numpy side effects
        if isinstance(v, np.generic):
            v = v.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def build_final_manifest(
    *,
    ligand: str,
    entries: Sequence[Mapping[str, Any]],
    approved_by: str,
    selection_basis: str,
    sqd_calibration_fragment_id: int,
    baseline_identity: Mapping[str, Any],
    baseline_e_total_ha: float,
    diagnostics_sha256: str,
    fragment_context: Mapping[int, Mapping[str, Any]],
    preliminary_manifest_digest: str | None = None,
    now: _dt.datetime | None = None,
) -> dict:
    """Assemble (but do not write) a frozen final manifest from user-supplied entries.

    ``entries``: ``[{"fragment_id": int, "role": str, "rationale": str}, ...]``.
    ``fragment_context[fid]`` supplies read-only context from the catalog /
    baseline: ``resource_requirements``, ``classical_solver_in_baseline``,
    ``mp2_max_abs_t2`` and ``ccsd_diagnostics_summary``.  Nothing here ranks,
    scores, or filters fragments.
    """
    if not str(approved_by).strip():
        raise FinalSelectionError("approved_by must be a non-empty string (who approves the freeze).")
    if not str(selection_basis).strip():
        raise FinalSelectionError(
            "selection_basis must be a non-empty scientific rationale written by the user.")
    if not entries:
        raise FinalSelectionError("The final selection must contain at least one fragment.")

    ids: list[int] = []
    frags: list[dict] = []
    for e in entries:
        try:
            fid = int(e["fragment_id"])
        except (KeyError, TypeError, ValueError):
            raise FinalSelectionError(f"Selection entry lacks an integer fragment_id: {e!r}")
        role = str(e.get("role", "")).strip()
        why = str(e.get("rationale", "")).strip()
        if not role:
            raise FinalSelectionError(f"Fragment {fid}: an experimental 'role' is required.")
        if not why:
            raise FinalSelectionError(f"Fragment {fid}: a per-fragment 'rationale' is required.")
        if fid in ids:
            raise FinalSelectionError(f"Fragment {fid} appears more than once in the selection.")
        ctx = fragment_context.get(fid)
        if ctx is None:
            raise FinalSelectionError(f"Fragment {fid} is not part of the reviewed baseline/catalog.")
        ids.append(fid)
        frags.append({
            "fragment_id": fid,
            "role": role,
            "rationale": why,
            "resource_requirements": {k: _clean(v) for k, v in dict(ctx["resource_requirements"]).items()},
            "classical_solver_in_baseline": ctx["classical_solver_in_baseline"],
            "mp2_max_abs_t2": _clean(ctx.get("mp2_max_abs_t2")),
            "ccsd_diagnostics_summary": {k: _clean(v) for k, v in dict(ctx["ccsd_diagnostics_summary"]).items()},
        })

    cal = int(sqd_calibration_fragment_id)
    if cal not in ids:
        raise FinalSelectionError(
            f"sqd_calibration_fragment_id={cal} must be one of the selected fragments {sorted(ids)}.")

    order = sorted(range(len(ids)), key=lambda i: ids[i])
    now = now or _dt.datetime.now(_dt.timezone.utc)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_stage": SELECTION_STAGE_FINAL,
        "frozen": True,
        "authorizes_sqd": True,
        "automatic_thresholds": False,
        "ligand": str(ligand),
        "approved_by": str(approved_by).strip(),
        "approved_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_basis": str(selection_basis).strip(),
        "selected_fragment_ids": sorted(ids),
        "sqd_calibration_fragment_id": cal,
        "fragments": [frags[i] for i in order],
        "source_provenance": {
            "baseline_cache_key": baseline_identity["cache_key"],
            "baseline_e_total_ha": float(baseline_e_total_ha),
            "diagnostics_sha256": str(diagnostics_sha256),
            "assignments_sha256": baseline_identity["assignments_sha256"],
            "hamiltonian_set_sha256": baseline_identity["hamiltonian_set_sha256"],
            "preliminary_manifest_digest": preliminary_manifest_digest,
        },
    }
    manifest["manifest_digest"] = manifest_digest(manifest)
    return manifest


def select_by_orbital_count(
    records: Sequence[Mapping[str, Any]], *, ligand: str, min_n_orb: int,
) -> list[dict]:
    """The size rule: every ``ligand`` fragment with ``n_orb >= min_n_orb``.

    ``records`` are catalog rows (``ligand``, ``fragment_id``, ``n_orb`` and
    optionally ``n_elec``).  Pure and deterministic; returns rows sorted by
    ``fragment_id``.  No diagnostic, ranking or hand-picked id is involved.
    """
    rows = [dict(r) for r in records if str(r["ligand"]) == str(ligand)]
    return sorted((r for r in rows if int(r["n_orb"]) >= int(min_n_orb)),
                  key=lambda r: int(r["fragment_id"]))


def catalog_digest(records: Sequence[Mapping[str, Any]], ligand: str) -> str:
    """sha256 of the ligand's sorted ``(fragment_id, n_orb)`` pairs."""
    pairs = sorted((int(r["fragment_id"]), int(r["n_orb"]))
                   for r in records if str(r["ligand"]) == str(ligand))
    return canonical_digest(pairs)


def calibration_fragment_id(selected: Sequence[Mapping[str, Any]]) -> int:
    """Derived single-fragment Stage E target: smallest ``n_orb``, ties by lowest id."""
    if not selected:
        raise FinalSelectionError("No fragment satisfies the size rule; nothing to select.")
    return int(min(selected, key=lambda r: (int(r["n_orb"]), int(r["fragment_id"])))["fragment_id"])


def build_size_manifest(
    *,
    ligand: str,
    catalog_records: Sequence[Mapping[str, Any]],
    min_n_orb: int,
    assignments: Mapping[int, str],
    assignments_sha256: str,
) -> dict:
    """Assemble the deterministic size-based manifest (v2).  Nothing is written."""
    min_n_orb = int(min_n_orb)
    chosen = select_by_orbital_count(catalog_records, ligand=ligand, min_n_orb=min_n_orb)
    if not chosen:
        raise FinalSelectionError(
            f"No {ligand} fragment has n_orb >= {min_n_orb}; the size rule selects nothing.")
    why = f"n_orb >= {min_n_orb}: selected by the fragment-size rule"
    frags = []
    for r in chosen:
        fid = int(r["fragment_id"])
        solver = assignments.get(fid)
        if solver not in ("FCI", "CCSD"):
            raise FinalSelectionError(
                f"Fragment {fid}: the frozen baseline assigns {solver!r}, expected FCI or CCSD.")
        n_elec = r.get("n_elec")
        frags.append({
            "fragment_id": fid,
            "n_orb": int(r["n_orb"]),
            "role": SIZE_ROLE,
            "rationale": why,
            "resource_requirements": {
                "n_orb": int(r["n_orb"]),
                "n_elec": None if n_elec is None or not math.isfinite(float(n_elec)) else int(n_elec),
            },
            "classical_solver_in_baseline": solver,
        })
    manifest = {
        "schema_version": SCHEMA_VERSION_SIZE,
        "selection_stage": SELECTION_STAGE_SIZE,
        "frozen": True,
        "authorizes_sqd": True,
        "ligand": str(ligand),
        "selection_policy": {
            "rule": SIZE_POLICY_RULE, "field": "n_orb", "comparison": ">=",
            "min_n_orb": min_n_orb,
            "description": f"all {ligand} fragments with at least {min_n_orb} spatial orbitals",
        },
        "selected_fragment_ids": [f["fragment_id"] for f in frags],
        "sqd_calibration_fragment_id": calibration_fragment_id(chosen),
        "calibration_rule": CALIBRATION_RULE,
        "fragments": frags,
        "source_provenance": {
            "catalog_sha256": catalog_digest(catalog_records, ligand),
            "assignments_sha256": str(assignments_sha256),
        },
    }
    manifest["manifest_digest"] = manifest_digest(manifest)
    return manifest


def _validate_size_manifest(manifest: Mapping[str, Any]) -> dict:
    if str(manifest.get("schema_version")) != SCHEMA_VERSION_SIZE:
        raise FinalSelectionError(
            f"Size-based selection schema_version={manifest.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION_SIZE!r}).")
    missing = _SIZE_REQUIRED_KEYS - set(manifest)
    if missing:
        raise FinalSelectionError(f"Size-based selection is missing keys: {sorted(missing)}")
    if manifest["frozen"] is not True or manifest["authorizes_sqd"] is not True:
        raise FinalSelectionError("The size-based selection is not frozen; regenerate it.")
    if manifest_digest(manifest) != manifest["manifest_digest"]:
        raise FinalSelectionError(
            "Final selection does not match its manifest_digest (edited after freezing). "
            "Regenerate it.")
    pol = manifest["selection_policy"]
    if not isinstance(pol, Mapping) or pol.get("rule") != SIZE_POLICY_RULE or "min_n_orb" not in pol:
        raise FinalSelectionError(f"selection_policy must be the {SIZE_POLICY_RULE!r} rule.")
    prov = manifest["source_provenance"]
    if not isinstance(prov, Mapping) or _SIZE_REQUIRED_PROVENANCE - set(prov):
        raise FinalSelectionError(
            f"source_provenance is incomplete (need {sorted(_SIZE_REQUIRED_PROVENANCE)}).")
    try:
        ids = [int(x) for x in manifest["selected_fragment_ids"]]
        min_n = int(pol["min_n_orb"])
    except (TypeError, ValueError):
        raise FinalSelectionError("selected_fragment_ids must be ints and min_n_orb an int.")
    if not ids or len(set(ids)) != len(ids):
        raise FinalSelectionError("selected_fragment_ids must be a non-empty list without duplicates.")
    frag_ids = []
    for f in manifest["fragments"]:
        if _SIZE_REQUIRED_FRAGMENT_KEYS - set(f):
            raise FinalSelectionError(
                f"fragment entry lacks keys {sorted(_SIZE_REQUIRED_FRAGMENT_KEYS - set(f))}")
        if int(f["n_orb"]) < min_n:
            raise FinalSelectionError(
                f"Fragment {f['fragment_id']} has n_orb={f['n_orb']} < {min_n}; it violates the size rule.")
        frag_ids.append(int(f["fragment_id"]))
    if sorted(frag_ids) != sorted(ids):
        raise FinalSelectionError("fragments[] does not match selected_fragment_ids.")
    cal = int(manifest["sqd_calibration_fragment_id"])
    if cal not in ids:
        raise FinalSelectionError("sqd_calibration_fragment_id is not among selected_fragment_ids.")
    out = dict(manifest)
    out["selected_fragment_ids"] = sorted(ids)
    out["sqd_calibration_fragment_id"] = cal
    return out


def check_size_manifest_current(
    manifest: Mapping[str, Any], *, catalog_records: Sequence[Mapping[str, Any]],
    assignments_sha256: str,
) -> None:
    """Raise unless the size rule, applied to the *current* catalog, reproduces ``manifest``."""
    ligand = manifest["ligand"]
    min_n = int(manifest["selection_policy"]["min_n_orb"])
    prov = manifest["source_provenance"]
    now_ids = [int(r["fragment_id"])
               for r in select_by_orbital_count(catalog_records, ligand=ligand, min_n_orb=min_n)]
    if now_ids != sorted(int(i) for i in manifest["selected_fragment_ids"]):
        raise FinalSelectionError(
            f"The size-based selection is STALE: the catalog now selects {now_ids}. Regenerate it.")
    if prov["catalog_sha256"] != catalog_digest(catalog_records, ligand):
        raise FinalSelectionError(
            "The size-based selection is STALE: the fragment catalog changed since it was created. "
            "Regenerate it.")
    if prov["assignments_sha256"] != assignments_sha256:
        raise FinalSelectionError(
            "The size-based selection is STALE: the frozen solver-assignment map changed. "
            "Regenerate it.")


def is_size_based(manifest: Mapping[str, Any]) -> bool:
    return manifest.get("selection_stage") == SELECTION_STAGE_SIZE


def validate_final_manifest(manifest: Mapping[str, Any]) -> dict:
    """Structural validation.  Returns the manifest with ids coerced to int.

    Accepts the v1 user-frozen CCSD-informed contract and the v2 size-based
    contract; anything else (e.g. a preliminary MP2 proposal) is rejected.
    """
    if not isinstance(manifest, Mapping):
        raise FinalSelectionError("Final selection manifest is not a JSON object.")
    stage = manifest.get("selection_stage")
    if stage == SELECTION_STAGE_SIZE:
        return _validate_size_manifest(manifest)
    if stage != SELECTION_STAGE_FINAL:
        raise FinalSelectionError(
            f"selection_stage={stage!r}: this is not a final CCSD-informed selection "
            f"(nor a {SELECTION_STAGE_SIZE!r} one). "
            "A preliminary MP2 proposal can never be used as the final selection.")
    if str(manifest.get("schema_version")) != SCHEMA_VERSION:
        raise FinalSelectionError(
            f"Final selection schema_version={manifest.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION!r}).")
    missing = _REQUIRED_KEYS - set(manifest)
    if missing:
        raise FinalSelectionError(f"Final selection is missing keys: {sorted(missing)}")
    if manifest["frozen"] is not True or manifest["authorizes_sqd"] is not True:
        raise FinalSelectionError(
            "The final selection is not frozen. Review the CCSD diagnostics and call "
            "freeze_final_selection(..., confirm=True).")
    if manifest["automatic_thresholds"] is not False:
        raise FinalSelectionError("automatic_thresholds must be false: no threshold may route solvers.")
    if manifest_digest(manifest) != manifest["manifest_digest"]:
        raise FinalSelectionError(
            "Final selection does not match its manifest_digest (edited after freezing). "
            "Re-freeze it.")
    prov = manifest["source_provenance"]
    if not isinstance(prov, Mapping) or _REQUIRED_PROVENANCE - set(prov):
        raise FinalSelectionError(
            f"source_provenance is incomplete (need {sorted(_REQUIRED_PROVENANCE)}).")
    try:
        ids = [int(x) for x in manifest["selected_fragment_ids"]]
    except (TypeError, ValueError):
        raise FinalSelectionError("selected_fragment_ids must be a list of ints.")
    if not ids or len(set(ids)) != len(ids):
        raise FinalSelectionError("selected_fragment_ids must be a non-empty list without duplicates.")
    frag_ids = []
    for f in manifest["fragments"]:
        if _REQUIRED_FRAGMENT_KEYS - set(f):
            raise FinalSelectionError(
                f"fragment entry lacks keys {sorted(_REQUIRED_FRAGMENT_KEYS - set(f))}")
        frag_ids.append(int(f["fragment_id"]))
    if sorted(frag_ids) != sorted(ids):
        raise FinalSelectionError("fragments[] does not match selected_fragment_ids.")
    cal = int(manifest["sqd_calibration_fragment_id"])
    if cal not in ids:
        raise FinalSelectionError("sqd_calibration_fragment_id is not among selected_fragment_ids.")
    out = dict(manifest)
    out["selected_fragment_ids"] = sorted(ids)
    out["sqd_calibration_fragment_id"] = cal
    return out


def write_final_manifest(manifest: Mapping[str, Any], path: Path | str) -> None:
    validate_final_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def load_final_manifest(path: Path | str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FinalSelectionError(
            "No final quantum-fragment selection exists.\n"
            "Quantum execution (SQD/TrimSQD calibration, tuning grid, promotion, EWF-SQD) is "
            "blocked until a selection is created with Challenge.select_fragments_by_size() "
            "(Stage D) or, for the research workflow, "
            "Challenge.freeze_final_selection(..., confirm=True). The preliminary MP2 panel is "
            "NOT used as a fallback.")
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return validate_final_manifest(raw)


def check_not_stale(
    manifest: Mapping[str, Any], *, baseline_cache_key: str, diagnostics_sha256: str,
) -> None:
    """Raise unless the manifest was frozen against exactly this baseline/diagnostics."""
    prov = manifest["source_provenance"]
    if prov["baseline_cache_key"] != baseline_cache_key:
        raise FinalSelectionError(
            "The final selection is STALE: it was frozen against a different classical "
            "baseline (cache key mismatch). Re-review the diagnostics and re-freeze.")
    if prov["diagnostics_sha256"] != diagnostics_sha256:
        raise FinalSelectionError(
            "The final selection is STALE: the CCSD diagnostics changed since it was frozen. "
            "Re-review the diagnostics and re-freeze.")
