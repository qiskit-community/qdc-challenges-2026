"""
Multi-fragment TrimSQD promotion manifest (schema 1.0, and 1.1 for a shared-source protocol).

The single-fragment ``promoted_configuration.json`` (one ``promote_configuration`` record) is
unchanged and stays the format of Stage E experiments and historical checkpoints.  This module adds a
SEPARATE artifact, ``checkpoints/promotion_manifest.json``, that holds one validated TrimSQD
configuration per selected fragment.  Nothing here reads or writes the legacy file.

Identities are the repository's existing ones, not new ones:

* ``final_selection.manifest_digest``      -- the frozen fragment selection,
* ``baseline.cache_key`` (+ the Hamiltonian-set and assignment digests) -- the classical baseline,
* ``hamiltonian_sha256``                   -- the embedded Hamiltonian of each fragment
                                              (must equal the baseline's ``hamiltonian_hash``),
* ``counts_hash``                          -- SHA-256 of the measurement file,
* ``config_digest``                        -- ``Challenge._config_digest`` of the equivalent legacy
                                              record, so a manifest entry and ``solve_molecule``'s
                                              saved ``promoted_digests`` agree.

Status
------
``"draft"``     any subset of the selected fragments; explicitly labelled and NEVER accepted as a
                complete promotion.
``"complete"``  exactly the selected fragments, each once, every entry valid.  Only
                :func:`finalize` produces it, and only :func:`validate_manifest` with
                ``require_complete=True`` accepts it for an experiment.

Schema 1.1 (Task 4B.2) is the default workflow: ONE fragment is optimized in Stage E, its
configuration is frozen as a shared protocol (:mod:`proteasome_challenge.protocol_transfer`) and
applied to the others.  A 1.1 manifest embeds that protocol and marks every entry ``origin``:

``"optimized"``    the single source fragment: its entry carries the validated tuning result and the
                   source acquisition settings,
``"transferred"``  every other fragment: the protocol's configuration, NO validated result, NO
                   energy of its own, and no acquisition settings (those belong to the measurement
                   plan).  A transferred entry is never presented as independently optimized.

Schema 1.0 (independently configured fragments, ``origin`` reported as ``"independent"``) keeps
validating exactly as before.  Configuration completeness, measurement availability and scientific
validation are reported separately by :func:`provenance_report`.

Evidence is recorded, never invented: ``measurement`` and ``validated_result`` are ``None`` for a
fragment without them.  Structure can therefore be validated for all 11 fragments without
measurements; :func:`provenance_report` and ``require_measurements=True`` state what is genuinely
evidenced.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from proteasome_challenge import protocol_transfer as pt
from proteasome_challenge.final_selection import canonical_digest

SCHEMA_VERSION = "1.0"
SCHEMA_SHARED = "1.1"
ORIGIN_INDEPENDENT = "independent"      # reported for 1.0 entries; never stored
ORIGIN_OPTIMIZED = "optimized"
ORIGIN_TRANSFERRED = "transferred"
KIND = "multi_fragment_promotion_manifest"
DECLARED_GRID_LABEL = "declared_reference"   # protocol declared from the literature, no tuning executed
SOLVER = "TrimSQD"
STATUS_DRAFT = "draft"
STATUS_COMPLETE = "complete"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_TOP_KEYS = {"schema_version", "kind", "status", "ligand", "solver", "final_selection",
             "baseline", "entries", "missing_fragment_ids", "manifest_digest"}
_ENTRY_KEYS = {"fragment_id", "solver", "grid_label", "config_digest", "sqd_config", "trim_config",
               "n_shots", "n_reps", "qpu_backend", "grid_energy_ha", "hamiltonian_sha256",
               "validated_result", "measurement", "rationale"}
_SHARED_TOP_EXTRA = {"shared_protocol"}
_SHARED_ENTRY_EXTRA = {"origin", "protocol_digest"}
_RESULT_KEYS = {"source", "file", "sha256", "grid_label", "hamiltonian_sha256"}
_MEASUREMENT_KEYS = {"counts_hash", "counts_source", "artifact"}


class PromotionManifestError(ValueError):
    """The promotion manifest is malformed, inconsistent, stale, or not complete."""


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────

def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _hex(v: Any, what: str) -> str:
    if not isinstance(v, str) or not _HEX64.fullmatch(v):
        raise PromotionManifestError(f"{what} must be a 64-character lowercase SHA-256, got {v!r}")
    return v


def _assert_finite(obj: Any, where: str) -> None:
    """Reject NaN / inf anywhere inside a configuration value."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    if isinstance(obj, (int, float)):
        if not math.isfinite(obj):
            raise PromotionManifestError(f"{where} is not finite: {obj!r}")
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            _assert_finite(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_finite(v, f"{where}[{i}]")
    else:
        raise PromotionManifestError(f"{where} has unsupported type {type(obj).__name__}")


def seal(manifest: Mapping[str, Any]) -> dict:
    """Return a copy with entries sorted by fragment id and ``manifest_digest`` recomputed."""
    out = copy.deepcopy(dict(manifest))
    out["entries"] = sorted(out["entries"], key=lambda e: e["fragment_id"])
    out.pop("manifest_digest", None)
    out["manifest_digest"] = canonical_digest(out)
    return out


def _selected_ids(selection: Mapping[str, Any]) -> list[int]:
    ids = selection.get("selected_fragment_ids")
    if not isinstance(ids, Sequence) or isinstance(ids, str) or not ids \
            or not all(_is_int(i) for i in ids) or len(set(ids)) != len(ids):
        raise PromotionManifestError("the frozen selection has no valid selected_fragment_ids")
    return sorted(int(i) for i in ids)


def _baseline_view(baseline: Mapping[str, Any]) -> dict:
    ident = baseline["identity"]
    return {
        "cache_key": ident["cache_key"],
        "hamiltonian_set_sha256": ident["hamiltonian_set_sha256"],
        "assignments_sha256": ident["assignments_sha256"],
        "hamiltonians": {int(f["fragment_id"]): f["hamiltonian_hash"] for f in baseline["fragments"]},
    }


# ── construction ─────────────────────────────────────────────────────────────────────────────────

def new_manifest(*, ligand: str, selection: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict:
    """An empty DRAFT bound to this frozen selection and classical baseline."""
    if selection.get("ligand") != ligand:
        raise PromotionManifestError(
            f"the frozen selection is for {selection.get('ligand')!r}, not {ligand!r}")
    ids = _selected_ids(selection)
    b = _baseline_view(baseline)
    return seal({
        "schema_version": SCHEMA_VERSION, "kind": KIND, "status": STATUS_DRAFT,
        "ligand": ligand, "solver": SOLVER,
        "final_selection": {"manifest_digest": selection["manifest_digest"],
                            "selected_fragment_ids": ids},
        "baseline": {k: b[k] for k in ("cache_key", "hamiltonian_set_sha256", "assignments_sha256")},
        "entries": [], "missing_fragment_ids": ids,
    })


def entry_from_promoted_record(
    record: Mapping[str, Any], *, config_digest: str,
    validated_result: Mapping[str, Any] | None = None,
    measurement: Mapping[str, Any] | None = None,
) -> dict:
    """Convert one ``promote_configuration`` record into a manifest entry.

    The record must be a TrimSQD promotion.  ``validated_result`` / ``measurement`` are passed in
    only when they genuinely exist; they are never derived or defaulted here.
    """
    method = record.get("solver_method") or record.get("solver_mode")
    if str(method).lower() != "trimsqd":
        raise PromotionManifestError(
            f"fragment {record.get('fragment_id')}: the manifest requires TrimSQD, got {method!r}")
    return {
        "fragment_id": record.get("fragment_id"),
        "solver": SOLVER,
        "grid_label": record.get("selected_grid_label") or record.get("grid_label"),
        "config_digest": config_digest,
        "sqd_config": copy.deepcopy(record.get("sqd_config")),
        "trim_config": copy.deepcopy(record.get("trim_config")),
        "n_shots": record.get("n_shots"), "n_reps": record.get("n_reps"),
        "qpu_backend": record.get("qpu_backend"),
        "grid_energy_ha": record.get("grid_energy_ha"),
        "hamiltonian_sha256": record.get("hamiltonian_sha256"),
        "validated_result": None if validated_result is None else dict(validated_result),
        "measurement": None if measurement is None else dict(measurement),
        "rationale": record.get("rationale"),
    }


def with_entry(manifest: Mapping[str, Any], entry: Mapping[str, Any], *, replace: bool = False) -> dict:
    """Add (or, with ``replace=True``, replace) one fragment's entry.  The result is always a DRAFT.

    Re-adding an identical entry is a no-op; a different entry for a fragment already present is a
    conflict and is refused unless ``replace=True``.
    """
    fid = entry.get("fragment_id")
    if not _is_int(fid):
        raise PromotionManifestError(f"entry fragment_id must be an integer, got {fid!r}")
    out = copy.deepcopy(dict(manifest))
    present = {e["fragment_id"]: e for e in out["entries"]}
    if fid in present and present[fid] != dict(entry) and not replace:
        raise PromotionManifestError(
            f"fragment {fid} already has a different promotion in the manifest; pass replace=True "
            "to replace it deliberately")
    if fid in present and present[fid] == dict(entry):
        return out
    out["entries"] = [e for e in out["entries"] if e["fragment_id"] != fid] + [copy.deepcopy(dict(entry))]
    out["status"] = STATUS_DRAFT
    have = {e["fragment_id"] for e in out["entries"]}
    out["missing_fragment_ids"] = [i for i in out["final_selection"]["selected_fragment_ids"] if i not in have]
    return seal(out)


def finalize(
    manifest: Mapping[str, Any], *, selection: Mapping[str, Any], baseline: Mapping[str, Any],
    config_digest_fn: Callable[[dict], str] | None = None, require_measurements: bool = False,
) -> dict:
    """Mark a draft COMPLETE iff it already covers every selected fragment and validates."""
    probe = copy.deepcopy(dict(manifest))
    probe["status"] = STATUS_COMPLETE
    probe["missing_fragment_ids"] = []
    probe = seal(probe)
    validate_manifest(probe, selection=selection, baseline=baseline, require_complete=True,
                      config_digest_fn=config_digest_fn, require_measurements=require_measurements)
    return probe


def transferred_entry(protocol: Mapping[str, Any], fragment_id: int, *,
                      config_digest_fn: Callable[[dict], str],
                      measurement: Mapping[str, Any] | None = None) -> dict:
    """The entry of a fragment that RECEIVES the shared protocol (never independently optimized)."""
    row = next((r for r in protocol["targets"] if r["fragment_id"] == fragment_id), None)
    if row is None or row["role"] != "transferred":
        raise PromotionManifestError(f"fragment {fragment_id} is not a transferred target of the protocol")
    tr = protocol["transferable"]
    entry = {
        "fragment_id": int(fragment_id), "solver": SOLVER, "grid_label": protocol["source"]["grid_label"],
        "config_digest": "0" * 64, "sqd_config": copy.deepcopy(tr["sqd_config"]),
        "trim_config": copy.deepcopy(tr["trim_config"]),
        "n_shots": None, "n_reps": None, "qpu_backend": None, "grid_energy_ha": None,
        "hamiltonian_sha256": row["hamiltonian_sha256"], "validated_result": None,
        "measurement": None if measurement is None else dict(measurement), "rationale": None,
        "origin": ORIGIN_TRANSFERRED, "protocol_digest": protocol["protocol_digest"],
    }
    entry["config_digest"] = config_digest_fn(entry_to_promoted_record(
        entry, ligand=protocol["ligand"], final_selection_digest=protocol["final_selection_digest"]))
    return entry


def shared_manifest(
    protocol: Mapping[str, Any], *, source_entry: Mapping[str, Any], selection: Mapping[str, Any],
    baseline: Mapping[str, Any], config_digest_fn: Callable[[dict], str],
) -> dict:
    """A DRAFT schema-1.1 manifest: the optimized source plus one transferred entry per other target.

    ``source_entry`` is :func:`entry_from_promoted_record` of the Stage E promotion the protocol was
    frozen from; it is the only entry that may carry a validated result and acquisition settings.
    """
    if source_entry.get("fragment_id") != protocol["source"]["fragment_id"]:
        raise PromotionManifestError("source_entry is not the protocol's source fragment")
    src = dict(copy.deepcopy(dict(source_entry)), origin=ORIGIN_OPTIMIZED, protocol_digest=protocol["protocol_digest"])
    entries = [src] + [transferred_entry(protocol, r["fragment_id"], config_digest_fn=config_digest_fn)
                       for r in protocol["targets"] if r["role"] == "transferred"]
    have = {e["fragment_id"] for e in entries}
    ids = _selected_ids(selection)
    b = _baseline_view(baseline)
    return seal({
        "schema_version": SCHEMA_SHARED, "kind": KIND, "status": STATUS_DRAFT,
        "ligand": protocol["ligand"], "solver": SOLVER,
        "final_selection": {"manifest_digest": selection["manifest_digest"], "selected_fragment_ids": ids},
        "baseline": {k: b[k] for k in ("cache_key", "hamiltonian_set_sha256", "assignments_sha256")},
        "entries": entries, "missing_fragment_ids": [i for i in ids if i not in have],
        "shared_protocol": copy.deepcopy(dict(protocol)),
    })


# ── validation ───────────────────────────────────────────────────────────────────────────────────

def _validate_entry(e: Any, baseline_ham: Mapping[int, str], position: int,
                    config_digest_fn, ligand: str, selection_digest: str,
                    require_measurements: bool, protocol: Mapping[str, Any] | None = None) -> int:
    if not isinstance(e, Mapping):
        raise PromotionManifestError(f"entries[{position}] is not an object")
    fid = e.get("fragment_id")
    if not _is_int(fid):
        raise PromotionManifestError(f"entries[{position}]: fragment_id must be an integer, got {fid!r}")
    where = f"fragment {fid}"
    keys = _ENTRY_KEYS | (_SHARED_ENTRY_EXTRA if protocol is not None else set())
    if set(e) != keys:
        raise PromotionManifestError(
            f"{where}: entry keys differ from the schema (missing {sorted(keys - set(e))}, "
            f"unexpected {sorted(set(e) - keys)})")
    origin = e.get("origin") if protocol is not None else None
    transferred = origin == ORIGIN_TRANSFERRED
    if protocol is not None:
        if origin not in (ORIGIN_OPTIMIZED, ORIGIN_TRANSFERRED):
            raise PromotionManifestError(f"{where}: origin must be 'optimized' or 'transferred', got {origin!r}")
        src_id = protocol["source"]["fragment_id"]
        if (origin == ORIGIN_OPTIMIZED) != (fid == src_id):
            raise PromotionManifestError(
                f"{where}: origin {origin!r} contradicts the protocol, whose single optimized source is "
                f"fragment {src_id}")
        if e["protocol_digest"] != protocol["protocol_digest"]:
            raise PromotionManifestError(f"{where}: protocol_digest differs from the manifest's shared protocol")
        tr = protocol["transferable"]
        if e["sqd_config"] != tr["sqd_config"] or e["trim_config"] != tr["trim_config"]:
            raise PromotionManifestError(
                f"{where}: its configuration differs from the shared protocol's (accidental parameter "
                "divergence); a transferred configuration is the protocol's, unchanged")
        if e["grid_label"] != protocol["source"]["grid_label"]:
            raise PromotionManifestError(f"{where}: grid_label must be the source's {protocol['source']['grid_label']!r}")
        if fid == src_id and e["config_digest"] != protocol["source"]["config_digest"]:
            raise PromotionManifestError(f"{where}: the source entry's config_digest differs from the protocol's")
        if transferred:
            for k in ("n_shots", "n_reps", "qpu_backend", "grid_energy_ha", "validated_result"):
                if e[k] is not None:
                    raise PromotionManifestError(
                        f"{where}: a transferred entry carries no {k} of its own (it was not independently "
                        "optimized or acquired); got a value")
    if e["solver"] != SOLVER:
        raise PromotionManifestError(f"{where}: solver must be {SOLVER!r}, got {e['solver']!r}")
    if not isinstance(e["grid_label"], str) or not e["grid_label"]:
        raise PromotionManifestError(f"{where}: grid_label must be a non-empty string")
    for key in ("sqd_config", "trim_config"):
        if not isinstance(e[key], Mapping) or not e[key]:
            raise PromotionManifestError(f"{where}: {key} is missing or empty")
        _assert_finite(e[key], f"{where}.{key}")
    tr = e["trim_config"].get("trim_ratio")
    if not _is_number(tr) or not (0 < tr <= 1):
        raise PromotionManifestError(f"{where}: trim_config['trim_ratio'] must satisfy 0 < x <= 1, got {tr!r}")
    if not transferred:
        for key in ("n_shots", "n_reps"):
            if not _is_int(e[key]) or e[key] < 1:
                raise PromotionManifestError(f"{where}: {key} must be an integer >= 1, got {e[key]!r}")
        if not isinstance(e["qpu_backend"], str) or not e["qpu_backend"]:
            raise PromotionManifestError(f"{where}: qpu_backend must be a non-empty string")
        if e["grid_label"] == DECLARED_GRID_LABEL:
            # A literature/reference protocol declared WITHOUT tuning: no measured grid energy exists, and
            # none may be fabricated.  It is also never accompanied by a validated result.
            if e["grid_energy_ha"] is not None:
                raise PromotionManifestError(
                    f"{where}: a declared reference protocol was not tuned and carries no grid_energy_ha "
                    f"(got {e['grid_energy_ha']!r}; a placeholder would imply a measurement)")
            if e["validated_result"] is not None:
                raise PromotionManifestError(f"{where}: a declared reference protocol has no validated result")
        elif not _is_number(e["grid_energy_ha"]) or not math.isfinite(e["grid_energy_ha"]):
            raise PromotionManifestError(f"{where}: grid_energy_ha must be a finite number, got {e['grid_energy_ha']!r}")
    _hex(e["config_digest"], f"{where}: config_digest")
    ham = _hex(e["hamiltonian_sha256"], f"{where}: hamiltonian_sha256")
    if fid not in baseline_ham:
        raise PromotionManifestError(f"{where}: not part of the classical baseline")
    if ham != baseline_ham[fid]:
        raise PromotionManifestError(
            f"{where}: Hamiltonian mismatch -- entry {ham[:12]}... differs from the classical "
            f"baseline's embedded Hamiltonian {baseline_ham[fid][:12]}...")

    res = e["validated_result"]
    if res is not None:
        if not isinstance(res, Mapping) or set(res) != _RESULT_KEYS:
            raise PromotionManifestError(f"{where}: validated_result must carry exactly {sorted(_RESULT_KEYS)}")
        _hex(res["sha256"], f"{where}: validated_result.sha256")
        if res["grid_label"] != e["grid_label"]:
            raise PromotionManifestError(f"{where}: validated_result belongs to grid point {res['grid_label']!r}")
        if res["hamiltonian_sha256"] is not None and res["hamiltonian_sha256"] != ham:
            raise PromotionManifestError(f"{where}: validated_result was produced for a different Hamiltonian")
    meas = e["measurement"]
    if meas is not None:
        if not isinstance(meas, Mapping) or set(meas) != _MEASUREMENT_KEYS:
            raise PromotionManifestError(f"{where}: measurement must carry exactly {sorted(_MEASUREMENT_KEYS)}")
        _hex(meas["counts_hash"], f"{where}: measurement.counts_hash")
        if meas["counts_source"] not in ("live", "frozen"):
            raise PromotionManifestError(f"{where}: measurement.counts_source must be 'live' or 'frozen'")
    if require_measurements and (meas is None or (res is None and not transferred)):
        raise PromotionManifestError(
            f"{where}: genuine provenance is required but the entry has "
            f"{'no measurement' if meas is None else 'no validated result'}")

    if config_digest_fn is not None:
        legacy = entry_to_promoted_record(e, ligand=ligand, final_selection_digest=selection_digest)
        if config_digest_fn(legacy) != e["config_digest"]:
            raise PromotionManifestError(f"{where}: config_digest does not match the entry's configuration")
    return fid


def validate_manifest(
    manifest: Any, *, selection: Mapping[str, Any], baseline: Mapping[str, Any],
    require_complete: bool = True,
    config_digest_fn: Callable[[dict], str] | None = None,
    require_measurements: bool = False,
) -> dict:
    """Validate ``manifest`` against the current frozen selection and classical baseline.

    ``require_complete=True`` (experiment mode) accepts only ``status == "complete"`` with every
    selected fragment exactly once.  ``False`` also accepts an explicitly labelled draft.  Returns
    a deep copy; raises :class:`PromotionManifestError` on any violation.
    """
    if not isinstance(manifest, Mapping):
        raise PromotionManifestError("the promotion manifest is not a JSON object")
    if manifest.get("kind") != KIND or manifest.get("schema_version") not in (SCHEMA_VERSION, SCHEMA_SHARED):
        raise PromotionManifestError(
            f"not a {KIND} schema {SCHEMA_VERSION} or {SCHEMA_SHARED} (kind={manifest.get('kind')!r}, "
            f"schema_version={manifest.get('schema_version')!r})")
    shared = manifest["schema_version"] == SCHEMA_SHARED
    top_keys = _TOP_KEYS | (_SHARED_TOP_EXTRA if shared else set())
    if set(manifest) != top_keys:
        raise PromotionManifestError(
            f"manifest keys differ from the schema (missing {sorted(top_keys - set(manifest))}, "
            f"unexpected {sorted(set(manifest) - top_keys)})")
    if canonical_digest({k: v for k, v in manifest.items() if k != "manifest_digest"}) \
            != manifest["manifest_digest"]:
        raise PromotionManifestError("manifest_digest does not match the content: the file was edited")

    status = manifest["status"]
    if status not in (STATUS_DRAFT, STATUS_COMPLETE):
        raise PromotionManifestError(f"status must be 'draft' or 'complete', got {status!r}")
    if require_complete and status != STATUS_COMPLETE:
        raise PromotionManifestError(
            "the promotion manifest is a DRAFT and cannot be used as a complete multi-fragment "
            f"promotion (missing fragments: {manifest['missing_fragment_ids']})")
    if manifest["solver"] != SOLVER:
        raise PromotionManifestError(f"manifest solver must be {SOLVER!r}, got {manifest['solver']!r}")
    if manifest["ligand"] != selection.get("ligand"):
        raise PromotionManifestError(
            f"manifest ligand {manifest['ligand']!r} differs from the frozen selection's "
            f"{selection.get('ligand')!r}")

    expected = _selected_ids(selection)
    fs = manifest["final_selection"]
    if not isinstance(fs, Mapping) or fs.get("manifest_digest") != selection["manifest_digest"] \
            or fs.get("selected_fragment_ids") != expected:
        raise PromotionManifestError(
            "the manifest was built for a different frozen fragment selection (digest or fragment "
            "ids differ); it is stale")
    b = _baseline_view(baseline)
    mb = manifest["baseline"]
    if not isinstance(mb, Mapping) or set(mb) != {"cache_key", "hamiltonian_set_sha256", "assignments_sha256"}:
        raise PromotionManifestError("manifest baseline identity is malformed")
    for key in ("cache_key", "hamiltonian_set_sha256", "assignments_sha256"):
        if mb[key] != b[key]:
            raise PromotionManifestError(
                f"the manifest was built against a different classical baseline ({key} differs); it is stale")

    protocol = None
    if shared:
        try:
            protocol = pt.validate_protocol(manifest["shared_protocol"], selection=selection, baseline=baseline)
        except pt.ProtocolTransferError as exc:
            raise PromotionManifestError(f"shared protocol: {exc}") from exc
    entries = manifest["entries"]
    if not isinstance(entries, list):
        raise PromotionManifestError("entries must be a list")
    ids = [_validate_entry(e, b["hamiltonians"], i, config_digest_fn, manifest["ligand"],
                           selection["manifest_digest"], require_measurements, protocol)
           for i, e in enumerate(entries)]
    if protocol is not None:
        stray = sorted(set(ids) - set(protocol["target_fragment_ids"]))
        if stray:
            raise PromotionManifestError(f"fragment(s) {stray} are not targets of the shared protocol")
        if sum(1 for e in entries if e["origin"] == ORIGIN_OPTIMIZED) != 1 \
                and (status == STATUS_COMPLETE or protocol["source"]["fragment_id"] in ids):
            raise PromotionManifestError("a shared-protocol manifest has exactly one optimized source entry")
        if status == STATUS_COMPLETE and protocol["source"]["fragment_id"] not in ids:
            raise PromotionManifestError("the optimized source fragment has no entry")
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise PromotionManifestError(f"duplicate or conflicting entries for fragment(s) {dup}")
    unexpected = sorted(set(ids) - set(expected))
    if unexpected:
        raise PromotionManifestError(
            f"fragment(s) {unexpected} are not in the frozen selection {expected}")
    missing = [i for i in expected if i not in ids]
    if manifest["missing_fragment_ids"] != missing:
        raise PromotionManifestError(
            f"missing_fragment_ids {manifest['missing_fragment_ids']} is inconsistent with the "
            f"entries (actually missing: {missing})")
    if status == STATUS_COMPLETE and missing:
        raise PromotionManifestError(f"a manifest cannot be 'complete': fragment(s) {missing} are missing")
    if require_complete and missing:
        raise PromotionManifestError(f"missing required configurations for fragment(s) {missing}")
    return copy.deepcopy(dict(manifest))


# ── reporting and adaptation ─────────────────────────────────────────────────────────────────────

def provenance_report(manifest: Mapping[str, Any]) -> list[dict]:
    """Per selected fragment: whether a configuration, a validated result and a measurement exist."""
    by_id = {e["fragment_id"]: e for e in manifest["entries"]}
    shared = manifest.get("schema_version") == SCHEMA_SHARED
    rows = []
    for fid in manifest["final_selection"]["selected_fragment_ids"]:
        e = by_id.get(fid)
        origin = (e["origin"] if shared else ORIGIN_INDEPENDENT) if e else None
        # Configuration completeness, measurement availability and validation are three facts.
        # A transferred fragment is validated only through its source, so it never reports its own.
        rows.append({
            "fragment_id": fid,
            "origin": origin,
            # True only for a protocol declared from the literature/reference WITHOUT any tuning run.
            "declared_reference": bool(e and e["grid_label"] == DECLARED_GRID_LABEL),
            "configured": e is not None,
            "validated_result": bool(e and e["validated_result"]),
            "measurement": bool(e and e["measurement"]),
            "genuine_provenance": bool(e and e["measurement"]
                                       and (e["validated_result"] or origin == ORIGIN_TRANSFERRED)),
            "protocol_digest": (e["protocol_digest"] if e and shared else None),
        })
    return rows


def entry_to_promoted_record(entry: Mapping[str, Any], *, ligand: str, final_selection_digest: str) -> dict:
    """The legacy ``promote_configuration`` record equivalent to one entry (for ``solve_molecule``)."""
    meas = entry.get("measurement") or {}
    return {
        "selected_grid_label": entry["grid_label"], "grid_label": entry["grid_label"],
        "ligand": ligand, "fragment_id": entry["fragment_id"],
        "solver_method": "trimsqd", "solver_mode": "trimsqd",
        "sqd_config": copy.deepcopy(entry["sqd_config"]),
        "trim_config": copy.deepcopy(entry["trim_config"]),
        "n_shots": entry["n_shots"], "n_reps": entry["n_reps"], "qpu_backend": entry["qpu_backend"],
        "grid_energy_ha": entry["grid_energy_ha"],
        "measurement_artifact": meas.get("artifact"),
        "hamiltonian_sha256": entry["hamiltonian_sha256"],
        "counts_source": meas.get("counts_source"), "counts_hash": meas.get("counts_hash"),
        "rationale": entry["rationale"],
        "final_selection_digest": final_selection_digest,
        "baseline_hamiltonian_hash": entry["hamiltonian_sha256"],
    }


def write_manifest(manifest: Mapping[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def read_manifest(path: Path | str) -> dict:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise PromotionManifestError(f"cannot read the promotion manifest {path}: {exc}") from exc
