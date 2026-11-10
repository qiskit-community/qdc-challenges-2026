"""
Single-source TrimSQD protocol transfer (Task 4B.2, schema 1.0).

Intended workflow::

    optimize ONE representative fragment in Stage E  ->  promote ONE configuration
    ->  freeze its transferable algorithmic parameters  ->  apply the SAME protocol to every
    TrimSQD fragment of Stage F  ->  reconstruct the full EWF energy.

Fragments are not tuned independently.  This module is pure (no network, no chemistry); the client
(`protocol_client.ProtocolTransferMixin`), the entrypoint and ``steps/step3_solve.run_solve`` use it
or mirror its checks, and ``tests/test_protocol_transfer.py`` asserts they agree.

Parameter classes
-----------------
A  transferable algorithmic   ``APPLIED_SQD_KEYS`` / ``APPLIED_TRIM_KEYS`` (+ the inert, recorded
                              ``carryover_threshold``).  Frozen in the protocol, identical for
                              every target.  The caps (``max_dim``, ``max_strings_per_trim``,
                              ``max_carryover``) are expressed in strings per spin sector, so they
                              are meaningful for any fragment size: no per-fragment rescaling exists.
B  fragment-dependent         Hamiltonian, ``n_orb``, electron / occupied-orbital counts, qubit
   physical                   count, RDMs, energies.  Never part of the protocol; each target keeps
                              its own and Step 3 checks them against the classical baseline.
C  measurement-execution      ``qpu_backend``, ``n_shots``, ``n_reps``, live/replay mode and the
                              counts artifact.  They describe how a fragment's counts were (or will
                              be) acquired and live in the measurement plan.  The source record's
                              copies are kept under ``source.acquisition`` as provenance only.

Deterministic compatibility adaptations are an explicit, registered mechanism.  The registry is
EMPTY: no adaptation is scientifically needed or defined, ``adaptations`` is always ``[]`` and any
requested rule is refused.  A target the protocol cannot be applied to unchanged is an error, never
silently retuned.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from proteasome_challenge.final_selection import canonical_digest

SCHEMA_VERSION = "1.0"
KIND = "shared_trimsqd_protocol"
SOLVER = "TrimSQD"
PROTOCOL_API = 1                    # capability flag echoed by the deployed entrypoint

APPLIED_SQD_KEYS = ("classical_backend", "symmetrize_spin", "n_batches", "iterations",
                    "samples_per_batch", "max_dim", "energy_tol", "occupancies_tol", "seed")
INERT_SQD_KEYS = ("carryover_threshold",)       # recorded; TrimSQD never passes it (mutually exclusive)
APPLIED_TRIM_KEYS = ("trim_ratio", "max_strings_per_trim", "max_carryover", "sector_trim")
FRAGMENT_DEPENDENT_KEYS = frozenset({
    "fragment_id", "n_orb", "norb", "nocc", "n_occ", "nelec", "n_elec", "n_qubits", "counts",
    "counts_hash", "counts_file", "hamiltonian_sha256", "heff", "eris", "qpu_backend", "backend",
    "n_shots", "n_reps", "lucj"})
ADAPTATION_RULES: dict[str, Any] = {}           # name -> rule; deliberately empty (see module docstring)

_HEX64 = re.compile(r"[0-9a-f]{64}")
_TOP_KEYS = {"schema_version", "kind", "ligand", "solver", "final_selection_digest",
             "baseline_cache_key", "source", "transferable", "protocol_digest", "applied_config",
             "applied_config_digest", "adaptations", "target_fragment_ids", "targets", "complete",
             "document_digest"}
_SOURCE_KEYS = {"fragment_id", "grid_label", "config_digest", "promotion_digest", "hamiltonian_sha256",
                "grid_energy_ha", "validated_result", "acquisition"}
_ACQ_KEYS = {"qpu_backend", "n_shots", "n_reps", "counts_hash", "counts_source"}
_TARGET_KEYS = {"fragment_id", "role", "hamiltonian_sha256", "n_orb", "n_occ", "n_elec",
                "effective_config_digest"}
_VALIDATED_KEYS = {"source", "file", "sha256", "grid_label", "hamiltonian_sha256"}


class ProtocolTransferError(ValueError):
    """The shared protocol is malformed, stale, inconsistent, or cannot be applied unchanged."""


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────

def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _hex(v: Any, what: str) -> str:
    if not isinstance(v, str) or not _HEX64.fullmatch(v):
        raise ProtocolTransferError(f"{what} must be a 64-character lowercase SHA-256, got {v!r}")
    return v


def solver_config_hash(applied: Mapping[str, Any]) -> str:
    """The SQD solvers' own ``_config_hash`` convention (what ``sqd_config_hash`` records)."""
    return hashlib.sha256(json.dumps(dict(applied), sort_keys=True, default=str).encode()).hexdigest()


def seal(protocol: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(dict(protocol))
    out.pop("document_digest", None)
    out["document_digest"] = canonical_digest(out)
    return out


# ── transferable configuration ───────────────────────────────────────────────────────────────────

def validate_transferable(sqd_config: Any, trim_config: Any) -> tuple[dict, dict]:
    """Return clean copies of the class-A settings, or raise with the offending key named."""
    if not isinstance(sqd_config, Mapping) or not isinstance(trim_config, Mapping):
        raise ProtocolTransferError("the promotion carries no sqd_config / trim_config to transfer")
    for where, cfg, allowed in (("sqd_config", sqd_config, set(APPLIED_SQD_KEYS) | set(INERT_SQD_KEYS)),
                                ("trim_config", trim_config, set(APPLIED_TRIM_KEYS))):
        for k in cfg:
            if k in FRAGMENT_DEPENDENT_KEYS:
                raise ProtocolTransferError(
                    f"{where}[{k!r}] is fragment-dependent or a measurement-execution setting; it is not "
                    "part of a transferable protocol")
            if k not in allowed:
                raise ProtocolTransferError(
                    f"{where}[{k!r}] is not a recognised transferable setting (allowed: {sorted(allowed)}); "
                    "its application to every solver could not be verified")
    for where, cfg, keys in (("sqd_config", sqd_config, APPLIED_SQD_KEYS),
                             ("trim_config", trim_config, APPLIED_TRIM_KEYS)):
        missing = [k for k in keys if k not in cfg]
        if missing:
            raise ProtocolTransferError(
                f"{where} lacks required transferable setting(s) {missing}; defaults would differ "
                "silently between fragments, so every one must be explicit")
    s, t = dict(sqd_config), dict(trim_config)
    if s["classical_backend"] != "python":
        raise ProtocolTransferError(
            f"sqd_config['classical_backend'] must be 'python' (the SBD backend needs its own, "
            f"non-transferable configuration), got {s['classical_backend']!r}")
    for where, cfg, key in (("sqd_config", s, "symmetrize_spin"), ("trim_config", t, "sector_trim")):
        if not isinstance(cfg[key], bool):
            raise ProtocolTransferError(f"{where}[{key!r}] must be a boolean, got {cfg[key]!r}")
    for where, cfg, key in (("sqd_config", s, "n_batches"), ("sqd_config", s, "iterations"),
                            ("sqd_config", s, "samples_per_batch"), ("sqd_config", s, "max_dim"),
                            ("trim_config", t, "max_strings_per_trim"), ("trim_config", t, "max_carryover")):
        if not _is_int(cfg[key]) or cfg[key] < 1:
            raise ProtocolTransferError(f"{where}[{key!r}] must be an integer >= 1, got {cfg[key]!r}")
    if not _is_int(s["seed"]) or s["seed"] < 0:
        raise ProtocolTransferError(f"sqd_config['seed'] must be an integer >= 0, got {s['seed']!r}")
    for key in ("energy_tol", "occupancies_tol", *[k for k in INERT_SQD_KEYS if k in s]):
        if not _is_num(s[key]) or s[key] <= 0:
            raise ProtocolTransferError(f"sqd_config[{key!r}] must be a finite number > 0, got {s[key]!r}")
    if not _is_num(t["trim_ratio"]) or not (0 < t["trim_ratio"] <= 1):
        raise ProtocolTransferError(f"trim_config['trim_ratio'] must satisfy 0 < x <= 1, got {t['trim_ratio']!r}")
    return copy.deepcopy(s), copy.deepcopy(t)


def applied_view(sqd_config: Mapping[str, Any], trim_config: Mapping[str, Any]) -> dict:
    """The part of the protocol a TrimSQD solver records about itself (``metadata['sqd_config']``)."""
    view = {k: sqd_config[k] for k in APPLIED_SQD_KEYS}
    view["trim"] = {k: trim_config[k] for k in APPLIED_TRIM_KEYS}
    return view


def resolve_adaptations(requested: Sequence[str] | None) -> list[str]:
    """Only registered, named rules may adapt a protocol; none are registered."""
    names = list(requested or [])
    unknown = [n for n in names if n not in ADAPTATION_RULES]
    if unknown:
        raise ProtocolTransferError(
            f"adaptation rule(s) {unknown} are not approved (registered rules: "
            f"{sorted(ADAPTATION_RULES) or 'none'}); the protocol is transferred unchanged or not at all")
    return names


def protocol_digest(sqd_config: Mapping[str, Any], trim_config: Mapping[str, Any],
                    adaptations: Sequence[str] = ()) -> str:
    """Digest of the transferable settings only: no fragment id, counts or acquisition setting."""
    return canonical_digest({"solver": SOLVER, "sqd_config": dict(sqd_config),
                             "trim_config": dict(trim_config), "adaptations": list(adaptations)})


def effective_config_digest(sqd_config: Mapping[str, Any], trim_config: Mapping[str, Any],
                            adaptations: Sequence[str] = ()) -> str:
    """Digest of the parameters one fragment's solver effectively runs with (equal for all targets)."""
    return canonical_digest({"solver": SOLVER, "applied": applied_view(sqd_config, trim_config),
                             "adaptations": list(adaptations)})


# ── per-target physical inputs ───────────────────────────────────────────────────────────────────

def baseline_targets(baseline: Mapping[str, Any]) -> dict[int, dict]:
    out = {}
    for f in baseline["fragments"]:
        d = f.get("ccsd_diagnostics") or {}
        out[int(f["fragment_id"])] = {
            "hamiltonian_sha256": f["hamiltonian_hash"], "n_orb": int(f["n_orb"]),
            "n_occ": d.get("n_occ"), "n_elec": d.get("n_correlated_electrons")}
    return out


def check_target_compatibility(sqd_config: Mapping[str, Any], fid: int, n_orb: Any, n_occ: Any,
                               n_elec: Any) -> None:
    """Raise unless the frozen settings apply UNCHANGED to this fragment's physical inputs."""
    where = f"fragment {fid}"
    if not (_is_int(n_orb) and _is_int(n_occ) and _is_int(n_elec)):
        raise ProtocolTransferError(
            f"{where}: n_orb / n_occ / n_elec are not established by the classical baseline "
            f"({n_orb!r}, {n_occ!r}, {n_elec!r}); the protocol cannot be checked against this fragment")
    if n_orb < 1 or not (1 <= n_occ <= n_orb):
        raise ProtocolTransferError(f"{where}: inconsistent orbital counts n_orb={n_orb}, n_occ={n_occ}")
    if n_elec != 2 * n_occ:
        raise ProtocolTransferError(
            f"{where}: n_elec={n_elec} is not 2*n_occ={2 * n_occ} (open shell). The solvers run the "
            "closed-shell sector (n_occ, n_occ) and symmetrize_spin assumes equal spin sectors, so the "
            "frozen settings are incompatible; no adaptation rule is approved for this case")


# ── construction ─────────────────────────────────────────────────────────────────────────────────

def _selected_ids(selection: Mapping[str, Any]) -> list[int]:
    ids = selection.get("selected_fragment_ids")
    if not isinstance(ids, Sequence) or isinstance(ids, str) or not ids \
            or not all(_is_int(i) for i in ids) or len(set(ids)) != len(ids):
        raise ProtocolTransferError("the frozen selection has no valid selected_fragment_ids")
    return sorted(int(i) for i in ids)


def transfer_protocol(
    source_promotion: Mapping[str, Any], target_fragment_ids: Sequence[int], *,
    selection: Mapping[str, Any], baseline: Mapping[str, Any], config_digest: str,
    validated_result: Mapping[str, Any] | None = None, adaptations: Sequence[str] = (),
) -> dict:
    """Freeze ONE promoted Stage E configuration as the protocol for ``target_fragment_ids``.

    ``source_promotion`` is a ``promote_configuration`` record of any fragment of the frozen
    selection (the source is selectable; nothing here names a particular fragment).
    ``config_digest`` is that record's legacy ``Challenge._config_digest``.
    Raises :class:`ProtocolTransferError` before producing anything when the source is invalid or
    stale, a target is outside the selection or the baseline, or the settings do not apply unchanged.
    """
    if not isinstance(source_promotion, Mapping):
        raise ProtocolTransferError("the source promotion is missing or not a record")
    rec = source_promotion
    method = str(rec.get("solver_method") or rec.get("solver_mode")).lower()
    if method != "trimsqd":
        raise ProtocolTransferError(f"the source promotion must be TrimSQD, got {method!r}")
    sid = rec.get("fragment_id")
    if not _is_int(sid):
        raise ProtocolTransferError(f"the source promotion has no integer fragment_id ({sid!r})")
    if rec.get("ligand") != selection.get("ligand"):
        raise ProtocolTransferError(
            f"the source promotion is for {rec.get('ligand')!r}, not {selection.get('ligand')!r}")
    if rec.get("final_selection_digest") != selection.get("manifest_digest"):
        raise ProtocolTransferError(
            "the source promotion was made under a different frozen selection (digest mismatch); "
            "it is stale -- promote again")
    selected = _selected_ids(selection)
    if sid not in selected:
        raise ProtocolTransferError(f"source fragment {sid} is not in the frozen selection {selected}")
    base = baseline_targets(baseline)
    if sid not in base:
        raise ProtocolTransferError(f"source fragment {sid} is not part of the classical baseline")
    ham = _hex(rec.get("hamiltonian_sha256"), f"source fragment {sid} hamiltonian_sha256")
    if ham != base[sid]["hamiltonian_sha256"]:
        raise ProtocolTransferError(
            f"source fragment {sid}: its Hamiltonian differs from the classical baseline's; the "
            "promotion was validated on a different problem")
    if rec.get("baseline_hamiltonian_hash") not in (None, ham):
        raise ProtocolTransferError(f"source fragment {sid}: baseline_hamiltonian_hash disagrees with it")
    _hex(config_digest, "config_digest")

    if isinstance(target_fragment_ids, (str, bytes)) or not isinstance(target_fragment_ids, Sequence) \
            or not target_fragment_ids or not all(_is_int(i) for i in target_fragment_ids):
        raise ProtocolTransferError("target_fragment_ids must be a non-empty list of integers")
    targets = sorted(int(i) for i in target_fragment_ids)
    if len(set(targets)) != len(targets):
        raise ProtocolTransferError(f"duplicate target fragment ids in {list(target_fragment_ids)}")
    outside = sorted(set(targets) - set(selected))
    if outside:
        raise ProtocolTransferError(f"target fragment(s) {outside} are not in the frozen selection {selected}")
    off_base = sorted(set(targets) - set(base))
    if off_base:
        raise ProtocolTransferError(f"target fragment(s) {off_base} are not part of the classical baseline")

    adapt = resolve_adaptations(adaptations)
    sqd, trim = validate_transferable(rec.get("sqd_config"), rec.get("trim_config"))
    eff = effective_config_digest(sqd, trim, adapt)
    rows = []
    for fid in targets:
        b = base[fid]
        check_target_compatibility(sqd, fid, b["n_orb"], b["n_occ"], b["n_elec"])
        rows.append({"fragment_id": fid, "role": "source" if fid == sid else "transferred",
                     "hamiltonian_sha256": b["hamiltonian_sha256"], "n_orb": b["n_orb"],
                     "n_occ": int(b["n_occ"]), "n_elec": int(b["n_elec"]), "effective_config_digest": eff})
    view = applied_view(sqd, trim)
    return seal({
        "schema_version": SCHEMA_VERSION, "kind": KIND, "ligand": selection["ligand"], "solver": SOLVER,
        "final_selection_digest": selection["manifest_digest"],
        "baseline_cache_key": baseline["identity"]["cache_key"],
        "source": {
            "fragment_id": int(sid),
            "grid_label": rec.get("selected_grid_label") or rec.get("grid_label"),
            "config_digest": config_digest,
            "promotion_digest": canonical_digest(dict(rec)),
            "hamiltonian_sha256": ham,
            "grid_energy_ha": rec.get("grid_energy_ha"),
            "validated_result": None if validated_result is None else dict(validated_result),
            # how the SOURCE counts were acquired: provenance only, never transferred
            "acquisition": {"qpu_backend": rec.get("qpu_backend"), "n_shots": rec.get("n_shots"),
                            "n_reps": rec.get("n_reps"), "counts_hash": rec.get("counts_hash"),
                            "counts_source": rec.get("counts_source")},
        },
        "transferable": {"sqd_config": sqd, "trim_config": trim},
        "protocol_digest": protocol_digest(sqd, trim, adapt),
        "applied_config": view, "applied_config_digest": solver_config_hash(view),
        "adaptations": adapt,
        "target_fragment_ids": targets, "targets": rows,
        "complete": targets == selected,
    })


# ── validation ───────────────────────────────────────────────────────────────────────────────────

def validate_protocol(protocol: Any, *, selection: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict:
    """Validate a protocol against the current frozen selection and classical baseline."""
    if not isinstance(protocol, Mapping):
        raise ProtocolTransferError("the shared protocol is not a JSON object")
    if protocol.get("kind") != KIND or protocol.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolTransferError(
            f"not a {KIND} schema {SCHEMA_VERSION} (kind={protocol.get('kind')!r}, "
            f"schema_version={protocol.get('schema_version')!r})")
    if set(protocol) != _TOP_KEYS:
        raise ProtocolTransferError(
            f"protocol keys differ from the schema (missing {sorted(_TOP_KEYS - set(protocol))}, "
            f"unexpected {sorted(set(protocol) - _TOP_KEYS)})")
    if canonical_digest({k: v for k, v in protocol.items() if k != "document_digest"}) \
            != protocol["document_digest"]:
        raise ProtocolTransferError("document_digest does not match the content: the protocol was edited")
    if protocol["solver"] != SOLVER:
        raise ProtocolTransferError(f"protocol solver must be {SOLVER!r}, got {protocol['solver']!r}")
    if protocol["ligand"] != selection.get("ligand"):
        raise ProtocolTransferError("the protocol is for a different ligand than the frozen selection")
    selected = _selected_ids(selection)
    if protocol["final_selection_digest"] != selection.get("manifest_digest"):
        raise ProtocolTransferError(
            "the protocol was built for a different frozen fragment selection; it is stale")
    if protocol["baseline_cache_key"] != baseline["identity"]["cache_key"]:
        raise ProtocolTransferError(
            "the protocol was built against a different classical baseline; it is stale")

    tr = protocol["transferable"]
    if not isinstance(tr, Mapping) or set(tr) != {"sqd_config", "trim_config"}:
        raise ProtocolTransferError("transferable must carry exactly sqd_config and trim_config")
    sqd, trim = validate_transferable(tr["sqd_config"], tr["trim_config"])
    adapt = resolve_adaptations(protocol["adaptations"] if isinstance(protocol["adaptations"], list) else ["?"])
    if protocol["protocol_digest"] != protocol_digest(sqd, trim, adapt):
        raise ProtocolTransferError("protocol_digest does not match the transferable settings")
    view = applied_view(sqd, trim)
    if protocol["applied_config"] != view or protocol["applied_config_digest"] != solver_config_hash(view):
        raise ProtocolTransferError("applied_config / applied_config_digest do not match the transferable settings")

    src = protocol["source"]
    if not isinstance(src, Mapping) or set(src) != _SOURCE_KEYS:
        raise ProtocolTransferError(f"source must carry exactly {sorted(_SOURCE_KEYS)}")
    base = baseline_targets(baseline)
    sid = src["fragment_id"]
    if not _is_int(sid) or sid not in selected or sid not in base:
        raise ProtocolTransferError(f"source fragment {sid!r} is not part of the frozen selection / baseline")
    for k in ("config_digest", "promotion_digest"):
        _hex(src[k], f"source.{k}")
    if _hex(src["hamiltonian_sha256"], "source.hamiltonian_sha256") != base[sid]["hamiltonian_sha256"]:
        raise ProtocolTransferError(f"source fragment {sid}: Hamiltonian differs from the classical baseline")
    acq = src["acquisition"]
    if not isinstance(acq, Mapping) or set(acq) != _ACQ_KEYS:
        raise ProtocolTransferError(f"source.acquisition must carry exactly {sorted(_ACQ_KEYS)}")
    vr = src["validated_result"]
    if vr is not None:
        if not isinstance(vr, Mapping) or set(vr) != _VALIDATED_KEYS:
            raise ProtocolTransferError(f"source.validated_result must carry exactly {sorted(_VALIDATED_KEYS)}")
        if vr["grid_label"] != src["grid_label"] or vr["hamiltonian_sha256"] not in (None, src["hamiltonian_sha256"]):
            raise ProtocolTransferError("source.validated_result does not belong to the source grid point / Hamiltonian")

    ids = protocol["target_fragment_ids"]
    rows = protocol["targets"]
    if not isinstance(ids, list) or not ids or not all(_is_int(i) for i in ids) or ids != sorted(set(ids)):
        raise ProtocolTransferError("target_fragment_ids must be a sorted list of unique integers")
    if sorted(set(ids) - set(selected)):
        raise ProtocolTransferError(
            f"target fragment(s) {sorted(set(ids) - set(selected))} are not in the frozen selection {selected}")
    if not isinstance(rows, list) or [r.get("fragment_id") if isinstance(r, Mapping) else None for r in rows] != ids:
        raise ProtocolTransferError("targets must list exactly target_fragment_ids, in order")
    eff = effective_config_digest(sqd, trim, adapt)
    for r in rows:
        fid = r["fragment_id"]
        if set(r) != _TARGET_KEYS:
            raise ProtocolTransferError(f"fragment {fid}: target keys differ from the schema")
        b = base.get(fid)
        if b is None:
            raise ProtocolTransferError(f"fragment {fid}: not part of the classical baseline")
        if r["role"] != ("source" if fid == sid else "transferred"):
            raise ProtocolTransferError(f"fragment {fid}: role {r['role']!r} is wrong")
        for k in ("hamiltonian_sha256", "n_orb", "n_occ", "n_elec"):
            if r[k] != b[k]:
                raise ProtocolTransferError(
                    f"fragment {fid}: {k} {r[k]!r} differs from the classical baseline's {b[k]!r}")
        if r["effective_config_digest"] != eff:
            raise ProtocolTransferError(
                f"fragment {fid}: its effective configuration differs from the protocol's "
                "(accidental parameter divergence)")
        check_target_compatibility(sqd, fid, r["n_orb"], r["n_occ"], r["n_elec"])
    if protocol["complete"] is not (ids == selected):
        raise ProtocolTransferError("complete does not reflect whether every selected fragment is a target")
    return copy.deepcopy(dict(protocol))


# ── application evidence ─────────────────────────────────────────────────────────────────────────

def expected_applied(protocol: Mapping[str, Any]) -> tuple[dict, str]:
    """(what each TrimSQD solver must record about itself, its ``sqd_config_hash``)."""
    return copy.deepcopy(dict(protocol["applied_config"])), protocol["applied_config_digest"]


def check_applied(protocol: Mapping[str, Any], fragment_summaries: Mapping[Any, Mapping[str, Any]],
                  quantum_protocol: Mapping[str, Any] | None) -> None:
    """Client-side proof that a Step 3 result applied THIS protocol to every quantum fragment.

    Raises :class:`ProtocolTransferError` when the echo is missing or differs, when a TrimSQD
    fragment has no application record, or when its recorded effective configuration differs.
    """
    if not isinstance(quantum_protocol, Mapping):
        raise ProtocolTransferError("the result carries no quantum_protocol: the protocol was not applied")
    for k in ("protocol_digest", "document_digest", "applied_config_digest"):
        want = protocol["document_digest" if k == "document_digest" else k]
        if quantum_protocol.get(k) != want:
            raise ProtocolTransferError(f"the result applied a different protocol ({k} differs)")
    if quantum_protocol.get("source_fragment_id") != protocol["source"]["fragment_id"]:
        raise ProtocolTransferError("the result names a different source fragment")
    if quantum_protocol.get("adaptations") != protocol["adaptations"]:
        raise ProtocolTransferError("the result applied different adaptations than the protocol")
    targets = {r["fragment_id"]: r for r in protocol["targets"]}
    seen = []
    for key, s in fragment_summaries.items():
        fid = int(key)
        if s.get("solver") not in ("SQD", "TrimSQD"):
            continue
        if s.get("solver") != SOLVER:
            raise ProtocolTransferError(f"fragment {fid}: solver {s.get('solver')!r} ran under a TrimSQD protocol")
        p = s.get("protocol")
        if not isinstance(p, Mapping) or p.get("verified") is not True:
            raise ProtocolTransferError(f"fragment {fid}: a quantum fragment without a verified protocol record")
        t = targets.get(fid)
        if t is None:
            raise ProtocolTransferError(f"fragment {fid}: ran a quantum solver but is not a protocol target")
        if (p.get("protocol_digest") != protocol["protocol_digest"]
                or p.get("effective_config_digest") != t["effective_config_digest"]
                or p.get("applied_config_digest") != protocol["applied_config_digest"]
                or s.get("sqd_config_hash") != protocol["applied_config_digest"]
                or p.get("role") != t["role"]):
            raise ProtocolTransferError(
                f"fragment {fid}: the recorded effective configuration differs from the protocol")
        if s.get("hamiltonian_hash") != t["hamiltonian_sha256"] or s.get("n_orb") != t["n_orb"] \
                or s.get("n_occ") != t["n_occ"]:
            raise ProtocolTransferError(f"fragment {fid}: solved a Hamiltonian that is not the target's")
        seen.append(fid)
    if sorted(seen) != sorted(quantum_protocol.get("applied_fragment_ids") or []):
        raise ProtocolTransferError("applied_fragment_ids does not list the quantum fragments that ran")
    if not seen:
        raise ProtocolTransferError("the result ran no quantum fragment under the protocol")


# ── job payload ──────────────────────────────────────────────────────────────────────────────────

def job_payload(protocol: Mapping[str, Any]) -> dict:
    """The ``shared_protocol`` argument of a Stage 3 job (JSON-safe; Step 3 re-derives every digest)."""
    tr = protocol["transferable"]
    return {
        "schema_version": protocol["schema_version"], "kind": protocol["kind"],
        "protocol_digest": protocol["protocol_digest"], "document_digest": protocol["document_digest"],
        "applied_config": copy.deepcopy(protocol["applied_config"]),
        "applied_config_digest": protocol["applied_config_digest"],
        "adaptations": list(protocol["adaptations"]),
        "source_fragment_id": protocol["source"]["fragment_id"],
        "sqd_config": copy.deepcopy(tr["sqd_config"]), "trim_config": copy.deepcopy(tr["trim_config"]),
        "targets": {str(r["fragment_id"]): {k: r[k] for k in sorted(_TARGET_KEYS - {"fragment_id"})}
                    for r in protocol["targets"]},
    }
