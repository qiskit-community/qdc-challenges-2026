"""
Fragment-specific measurement plan (schema 1.0).

A plan states, for every quantum fragment, WHERE its measurements come from:

``live``    acquire new QPU measurements (the container samples the QPU for that fragment),
``replay``  process an existing, hash-pinned counts artifact (the QPU is never touched),

and a plan holding both is ``mixed``.  The mode is explicit for every fragment: there is no default
and no inference from "counts happen to exist".

Identity bound into every spec (nothing new is invented -- these are the repository's identities):

* ``hamiltonian_sha256``  the embedded Hamiltonian, equal to the classical baseline's
                          ``hamiltonian_hash`` for that fragment,
* ``n_orb`` / ``n_qubits`` the counts width: ``n_qubits == 2 * n_orb``,
* ``backend``             the backend the counts were (or will be) acquired on,
* ``counts_sha256``       SHA-256 of the artifact FILE (the same ``counts_hash`` the promotion record
                          and the promotion manifest use), pinned at planning time for a replay,
* the frozen selection digest and the classical baseline cache key.

This module is pure (no network, no chemistry).  The Serverless client, entrypoint and Step 3 use it
or mirror its checks, and ``tests/test_measurement_routing.py`` asserts they agree.

Artifacts are write-once.  A live acquisition is stored under ``checkpoints/counts/runs/`` with a
unique name and a ``*.provenance.json`` sidecar; nothing here ever overwrites or deletes a counts file.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence  # noqa: F401

from proteasome_challenge.final_selection import canonical_digest

SCHEMA_VERSION = "1.0"
KIND = "measurement_plan"
SIDECAR_KIND = "measurement_artifact"
MODES = ("live", "replay")
EXECUTIONS = ("live", "replay", "mixed")
SOLVERS = ("TrimSQD", "SQD")
LOCATIONS = ("frozen", "checkpoint", "run")
MEASUREMENT_PLAN_API = 1            # capability flag echoed by the deployed entrypoint

_HEX64 = re.compile(r"[0-9a-f]{64}")
_PLAN_KEYS = {"schema_version", "kind", "ligand", "execution", "final_selection_digest",
              "baseline_cache_key", "specs", "plan_digest"}
_SPEC_KEYS = {"fragment_id", "mode", "solver", "hamiltonian_sha256", "n_orb", "n_qubits", "backend",
              "live", "replay"}
_LIVE_KEYS = {"n_shots", "n_reps", "sqd_config", "trim_config"}
_REPLAY_KEYS = {"artifact", "counts_sha256", "n_shots_observed", "provenance", "acquired_from"}
_ARTIFACT_KEYS = {"name", "location"}


def _challenge_error():
    from proteasome_challenge import ChallengeError      # imported lazily: the package imports this module
    return ChallengeError


class MeasurementPlanError(_challenge_error(), ValueError):
    """The measurement plan or a measurement artifact is malformed, inconsistent or unsafe."""


# ── small helpers ────────────────────────────────────────────────────────────────────────────────

def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _hex(v: Any, what: str) -> str:
    if not isinstance(v, str) or not _HEX64.fullmatch(v):
        raise MeasurementPlanError(f"{what} must be a 64-character lowercase SHA-256, got {v!r}")
    return v


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def n_qubits_for(n_orb: int) -> int:
    """Measured bits for an ``n_orb``-orbital fragment (alpha + beta spin orbitals)."""
    return 2 * int(n_orb)


def spec_digest(spec: Mapping[str, Any]) -> str:
    return canonical_digest(dict(spec))


def seal(plan: Mapping[str, Any]) -> dict:
    """Copy with specs sorted by fragment id, ``execution`` derived and ``plan_digest`` recomputed."""
    out = copy.deepcopy(dict(plan))
    out["specs"] = sorted(out["specs"], key=lambda s: s["fragment_id"])
    out["execution"] = execution_of(out["specs"])
    out.pop("plan_digest", None)
    out["plan_digest"] = canonical_digest(out)
    return out


def execution_of(specs: Sequence[Mapping[str, Any]]) -> str:
    modes = {s.get("mode") for s in specs}
    if not modes or not modes <= set(MODES):
        raise MeasurementPlanError(f"a plan needs at least one spec with mode in {MODES}, got {sorted(map(str, modes))}")
    return "mixed" if len(modes) == 2 else modes.pop()


def safe_name(name: Any) -> str:
    """A bare ``*.npy`` file name: no directories, no traversal."""
    if (not isinstance(name, str) or not name or Path(name).name != name or name in (".", "..")
            or "\\" in name or not name.endswith(".npy")):
        raise MeasurementPlanError(f"artifact name must be a bare '*.npy' file name, got {name!r}")
    return name


def names_fragment(name: str, fragment_id: int) -> bool:
    """True when the file name carries THIS fragment's token (``frag<id>`` / ``fragment_<id>``)."""
    return re.search(rf"(?:frag|fragment_){int(fragment_id)}(?!\d)", name) is not None


def sidecar_path(artifact: Path | str) -> Path:
    p = Path(artifact)
    return p.with_name(p.stem + ".provenance.json")


# ── counts validation (mirrors serverless_bundle/steps/step3_solve.py::_check_counts_file) ───────

def check_counts_file(path: Path | str, n_orb: int) -> dict:
    """Raise unless ``path`` holds usable counts for an ``n_orb``-orbital fragment.

    Returns ``{"n_shots", "n_outcomes", "n_bits"}``.  Accepts bit-string keys of width ``2*n_orb`` or
    integer keys that fit in ``2*n_orb`` bits (``np.save`` round trips can turn the former into the
    latter).
    """
    import numpy as np

    path = Path(path)
    width = n_qubits_for(n_orb)
    try:
        raw = np.load(path, allow_pickle=True).item()
    except Exception as exc:
        raise MeasurementPlanError(f"measurement counts {path.name} cannot be read ({type(exc).__name__}: {exc})") from exc
    if not isinstance(raw, dict) or not raw:
        raise MeasurementPlanError(f"measurement counts {path.name} are empty or not a counts dict")
    try:
        values = [int(v) for v in raw.values()]
    except Exception as exc:
        raise MeasurementPlanError(f"measurement counts {path.name} have non-numeric values") from exc
    if sum(values) <= 0 or any(v <= 0 for v in values):
        raise MeasurementPlanError(f"measurement counts {path.name} have non-positive shot counts")
    keys = list(raw)
    if all(isinstance(k, str) for k in keys):
        widths = {len(k) for k in keys}
        if widths != {width} or any(set(k) - {"0", "1"} for k in keys):
            raise MeasurementPlanError(
                f"measurement counts {path.name} are not {width}-bit strings for a {n_orb}-orbital "
                f"fragment (bit widths found: {sorted(widths)}); counts of another fragment?")
    elif not all(isinstance(k, (int, np.integer)) for k in keys):
        raise MeasurementPlanError(f"measurement counts {path.name} have unsupported key types")
    elif max(int(k) for k in keys).bit_length() > width:
        raise MeasurementPlanError(f"measurement counts {path.name} contain outcomes wider than {width} bits")
    return {"n_shots": sum(values), "n_outcomes": len(values), "n_bits": width}


# ── write-once storage ───────────────────────────────────────────────────────────────────────────

def write_new(path: Path | str, data: bytes) -> bool:
    """Create ``path`` exclusively.  ``True`` if created, ``False`` if it already holds these exact
    bytes (an idempotent retry); raises if it exists with different content.  Never overwrites."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as fh:
            fh.write(data)
        return True
    except FileExistsError:
        if sha256_bytes(path.read_bytes()) == sha256_bytes(data):
            return False
        raise MeasurementPlanError(
            f"{path.name} already exists with different content; measurement artifacts are write-once "
            "and are never overwritten") from None


def write_sidecar_once(artifact: Path | str, record: Mapping[str, Any]) -> Path:
    """Write the provenance sidecar next to ``artifact`` once; an existing one must describe the same bytes."""
    side = sidecar_path(artifact)
    if side.exists():
        existing = read_sidecar(artifact)
        if existing.get("counts_sha256") != record.get("counts_sha256"):
            raise MeasurementPlanError(f"{side.name} already describes different counts; refusing to replace it")
        return side
    write_new(side, json.dumps(dict(record), indent=2, sort_keys=True, default=float).encode("utf-8"))
    return side


def read_sidecar(artifact: Path | str) -> dict | None:
    side = sidecar_path(artifact)
    if not side.is_file():
        return None
    try:
        rec = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MeasurementPlanError(f"unreadable provenance sidecar {side.name}: {exc}") from exc
    if not isinstance(rec, dict) or rec.get("kind") != SIDECAR_KIND:
        raise MeasurementPlanError(f"{side.name} is not a {SIDECAR_KIND} record")
    return rec


# ── spec construction ────────────────────────────────────────────────────────────────────────────

def _backend(v: Any, fid: int) -> str:
    if not isinstance(v, str) or not v.strip():
        raise MeasurementPlanError(
            f"fragment {fid}: a backend identity is required (non-empty string), got {v!r}. Counts without a "
            "provenance sidecar carry no backend: declare it (replay_backends={fragment_id: 'ibm_...'}).")
    return v.strip()


def live_spec(*, fragment_id: int, hamiltonian_sha256: str, n_orb: int, backend: str, solver: str = "TrimSQD",
              n_shots: int, n_reps: int, sqd_config: Mapping | None = None,
              trim_config: Mapping | None = None) -> dict:
    """Spec that acquires NEW measurements for ``fragment_id``.

    ``sqd_config`` / ``trim_config`` only shape the by-product energy of the acquisition job; the
    configuration used for the final reconstruction is fragment-specific and applied elsewhere.
    """
    return {
        "fragment_id": int(fragment_id), "mode": "live", "solver": solver,
        "hamiltonian_sha256": hamiltonian_sha256, "n_orb": int(n_orb), "n_qubits": n_qubits_for(n_orb),
        "backend": _backend(backend, fragment_id),
        "live": {"n_shots": n_shots, "n_reps": n_reps,
                 "sqd_config": None if sqd_config is None else dict(sqd_config),
                 "trim_config": None if trim_config is None else dict(trim_config)},
        "replay": None,
    }


PROVENANCES = ("sidecar", "stage3_record", "declared")


def replay_spec(*, fragment_id: int, hamiltonian_sha256: str, n_orb: int, backend: str | None,
                artifact: Mapping[str, str], path: Path | str, solver: str = "TrimSQD",
                backend_source: str = "declared") -> dict:
    """Spec that replays the existing ``artifact`` (found at ``path``), pinned by its SHA-256.

    The file is validated now (readable, ``2*n_orb`` bits) and, when a provenance sidecar exists, its
    fragment, Hamiltonian, width and backend are cross-checked.  Without a sidecar the backend must
    be supplied by the caller and the spec records where it came from (``backend_source``:
    ``"stage3_record"`` = the saved result of the job that produced the counts, or ``"declared"``).
    """
    if backend_source not in PROVENANCES[1:]:
        raise MeasurementPlanError(f"backend_source must be one of {PROVENANCES[1:]}, got {backend_source!r}")
    path = Path(path)
    name = safe_name(artifact.get("name"))
    if artifact.get("location") not in LOCATIONS:
        raise MeasurementPlanError(f"artifact location must be one of {LOCATIONS}, got {artifact.get('location')!r}")
    if not names_fragment(name, fragment_id):
        raise MeasurementPlanError(
            f"artifact {name!r} does not carry fragment {fragment_id}'s token; it will not be assigned "
            "to this fragment")
    if not path.is_file():
        raise MeasurementPlanError(f"fragment {fragment_id}: counts artifact {name} does not exist")
    sha = sha256_file(path)
    stats = check_counts_file(path, n_orb)
    side = read_sidecar(path)
    declared = backend
    acquired = None
    if side is not None:
        _check_sidecar(side, fragment_id, hamiltonian_sha256, n_orb, sha, name)
        if declared is not None and _backend(declared, fragment_id) != side["backend"]:
            raise MeasurementPlanError(
                f"fragment {fragment_id}: declared backend {declared!r} differs from the artifact's "
                f"recorded backend {side['backend']!r}")
        declared = side["backend"]
        acquired = {"job_id": side.get("job_id"), "utc": side.get("stored_utc")}
    return {
        "fragment_id": int(fragment_id), "mode": "replay", "solver": solver,
        "hamiltonian_sha256": hamiltonian_sha256, "n_orb": int(n_orb), "n_qubits": n_qubits_for(n_orb),
        "backend": _backend(declared, fragment_id),
        "live": None,
        "replay": {"artifact": {"name": name, "location": artifact["location"]}, "counts_sha256": sha,
                   "n_shots_observed": stats["n_shots"],
                   "provenance": "sidecar" if side is not None else backend_source, "acquired_from": acquired},
    }


def _check_sidecar(side: Mapping, fid: int, ham: str, n_orb: int, sha: str, name: str) -> None:
    if side.get("fragment_id") != int(fid):
        raise MeasurementPlanError(
            f"{name}: provenance says the counts belong to fragment {side.get('fragment_id')}, not {fid}")
    if side.get("hamiltonian_sha256") != ham:
        raise MeasurementPlanError(f"{name}: the counts were acquired for a different embedded Hamiltonian")
    if side.get("n_bits") != n_qubits_for(n_orb):
        raise MeasurementPlanError(f"{name}: the counts have {side.get('n_bits')} bits, expected {n_qubits_for(n_orb)}")
    if side.get("counts_sha256") != sha:
        raise MeasurementPlanError(f"{name}: the file no longer matches its recorded SHA-256; it was altered")


def new_plan(*, ligand: str, selection: Mapping[str, Any], baseline: Mapping[str, Any],
             specs: Sequence[Mapping[str, Any]]) -> dict:
    return seal({
        "schema_version": SCHEMA_VERSION, "kind": KIND, "ligand": ligand, "execution": "replay",
        "final_selection_digest": selection["manifest_digest"],
        "baseline_cache_key": baseline["identity"]["cache_key"],
        "specs": [copy.deepcopy(dict(s)) for s in specs],
    })


# ── validation ───────────────────────────────────────────────────────────────────────────────────

def baseline_view(baseline: Mapping[str, Any]) -> dict[int, tuple[str, int]]:
    """``{fragment_id: (hamiltonian_hash, n_orb)}`` from a classical baseline record."""
    return {int(f["fragment_id"]): (f["hamiltonian_hash"], int(f["n_orb"])) for f in baseline["fragments"]}


def validate_spec(s: Any, view: Mapping[int, tuple[str, int]], position: int = 0) -> int:
    if not isinstance(s, Mapping):
        raise MeasurementPlanError(f"specs[{position}] is not an object")
    fid = s.get("fragment_id")
    if not _is_int(fid):
        raise MeasurementPlanError(f"specs[{position}]: fragment_id must be an integer, got {fid!r}")
    where = f"fragment {fid}"
    if set(s) != _SPEC_KEYS:
        raise MeasurementPlanError(f"{where}: spec keys differ from the schema (missing "
                                   f"{sorted(_SPEC_KEYS - set(s))}, unexpected {sorted(set(s) - _SPEC_KEYS)})")
    if s["mode"] not in MODES:
        raise MeasurementPlanError(f"{where}: measurement mode must be one of {MODES}, got {s['mode']!r}")
    if s["solver"] not in SOLVERS:
        raise MeasurementPlanError(f"{where}: solver must be one of {SOLVERS}, got {s['solver']!r}")
    if fid not in view:
        raise MeasurementPlanError(f"{where}: not part of the classical baseline")
    ham, n_orb = view[fid]
    if _hex(s["hamiltonian_sha256"], f"{where}: hamiltonian_sha256") != ham:
        raise MeasurementPlanError(
            f"{where}: Hamiltonian mismatch -- spec {s['hamiltonian_sha256'][:12]}... differs from the "
            f"classical baseline's embedded Hamiltonian {ham[:12]}...")
    if s["n_orb"] != n_orb:
        raise MeasurementPlanError(f"{where}: n_orb {s['n_orb']!r} differs from the baseline's {n_orb}")
    if s["n_qubits"] != n_qubits_for(n_orb):
        raise MeasurementPlanError(
            f"{where}: n_qubits {s['n_qubits']!r} must equal 2*n_orb = {n_qubits_for(n_orb)}")
    _backend(s["backend"], fid)
    if s["mode"] == "live":
        live = s["live"]
        if s["replay"] is not None or not isinstance(live, Mapping) or set(live) != _LIVE_KEYS:
            raise MeasurementPlanError(f"{where}: a live spec carries exactly a 'live' section {sorted(_LIVE_KEYS)}")
        for k in ("n_shots", "n_reps"):
            if not _is_int(live[k]) or live[k] < 1:
                raise MeasurementPlanError(f"{where}: live.{k} must be an integer >= 1, got {live[k]!r}")
        for k in ("sqd_config", "trim_config"):
            if live[k] is not None:
                if not isinstance(live[k], Mapping):
                    raise MeasurementPlanError(f"{where}: live.{k} must be an object or null")
                _assert_finite(live[k], f"{where}: live.{k}")
        tr = (live["trim_config"] or {}).get("trim_ratio")
        if tr is not None and (not isinstance(tr, (int, float)) or isinstance(tr, bool) or not (0 < tr <= 1)):
            raise MeasurementPlanError(f"{where}: live.trim_config['trim_ratio'] must satisfy 0 < x <= 1, got {tr!r}")
    else:
        rep = s["replay"]
        if s["live"] is not None or not isinstance(rep, Mapping) or set(rep) != _REPLAY_KEYS:
            raise MeasurementPlanError(f"{where}: a replay spec carries exactly a 'replay' section {sorted(_REPLAY_KEYS)}")
        art = rep["artifact"]
        if not isinstance(art, Mapping) or set(art) != _ARTIFACT_KEYS or art["location"] not in LOCATIONS:
            raise MeasurementPlanError(f"{where}: replay.artifact must be {{name, location in {LOCATIONS}}}")
        if not names_fragment(safe_name(art["name"]), fid):
            raise MeasurementPlanError(f"{where}: artifact {art['name']!r} does not carry this fragment's token")
        _hex(rep["counts_sha256"], f"{where}: replay.counts_sha256")
        if rep["provenance"] not in PROVENANCES:
            raise MeasurementPlanError(f"{where}: replay.provenance must be one of {PROVENANCES}")
        if not _is_int(rep["n_shots_observed"]) or rep["n_shots_observed"] < 1:
            raise MeasurementPlanError(f"{where}: replay.n_shots_observed must be a positive integer")
    return fid


def _assert_finite(obj: Any, where: str) -> None:
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    if isinstance(obj, (int, float)):
        if not math.isfinite(obj):
            raise MeasurementPlanError(f"{where} is not finite: {obj!r}")
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            _assert_finite(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_finite(v, f"{where}[{i}]")
    else:
        raise MeasurementPlanError(f"{where} has unsupported type {type(obj).__name__}")


def validate_plan(plan: Any, *, selection: Mapping[str, Any], baseline: Mapping[str, Any],
                  require_full_selection: bool = False) -> dict:
    """Validate ``plan`` against the frozen selection and classical baseline; return a deep copy.

    ``require_full_selection=True`` (full-EWF experiment) demands exactly the selected fragments, each
    once.  ``False`` accepts any non-empty subset of them (e.g. "sample fragments 2 and 3 now").
    """
    if not isinstance(plan, Mapping):
        raise MeasurementPlanError("the measurement plan is not an object")
    if plan.get("kind") != KIND or plan.get("schema_version") != SCHEMA_VERSION:
        raise MeasurementPlanError(f"not a {KIND} schema {SCHEMA_VERSION} (kind={plan.get('kind')!r}, "
                                   f"schema_version={plan.get('schema_version')!r})")
    if set(plan) != _PLAN_KEYS:
        raise MeasurementPlanError(f"plan keys differ from the schema (missing {sorted(_PLAN_KEYS - set(plan))}, "
                                   f"unexpected {sorted(set(plan) - _PLAN_KEYS)})")
    if canonical_digest({k: v for k, v in plan.items() if k != "plan_digest"}) != plan["plan_digest"]:
        raise MeasurementPlanError("plan_digest does not match the content: the plan was edited")
    if plan["ligand"] != selection.get("ligand"):
        raise MeasurementPlanError(f"plan ligand {plan['ligand']!r} differs from the frozen selection's "
                                   f"{selection.get('ligand')!r}")
    if plan["final_selection_digest"] != selection["manifest_digest"]:
        raise MeasurementPlanError("the plan was built for a different frozen fragment selection; it is stale")
    if plan["baseline_cache_key"] != baseline["identity"]["cache_key"]:
        raise MeasurementPlanError("the plan was built against a different classical baseline; it is stale")
    specs = plan["specs"]
    if not isinstance(specs, list) or not specs:
        raise MeasurementPlanError("a plan needs at least one fragment spec")
    view = baseline_view(baseline)
    ids = [validate_spec(s, view, i) for i, s in enumerate(specs)]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise MeasurementPlanError(f"duplicate fragment id(s) {dup} in the measurement plan")
    selected = sorted(int(i) for i in selection["selected_fragment_ids"])
    outside = sorted(set(ids) - set(selected))
    if outside:
        raise MeasurementPlanError(f"fragment(s) {outside} are not in the frozen selection {selected}")
    if plan["execution"] != execution_of(specs):
        raise MeasurementPlanError(f"execution {plan['execution']!r} does not match the specs' modes "
                                   f"({execution_of(specs)!r})")
    if require_full_selection:
        missing = [i for i in selected if i not in ids]
        if missing:
            raise MeasurementPlanError(f"the plan does not cover selected fragment(s) {missing}")
    return copy.deepcopy(dict(plan))


def check_against_manifest(plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Cross-check replay specs with a promotion manifest's genuine measurement records.

    For every replay fragment that the manifest holds a measurement for, the pinned counts hash and
    the backend must agree; a disagreement means the plan would replay different data than the
    configuration was promoted on.
    """
    entries = {e["fragment_id"]: e for e in manifest.get("entries", [])}
    for s in plan["specs"]:
        e = entries.get(s["fragment_id"])
        if e is None or s["mode"] != "replay" or not e.get("measurement"):
            continue
        fid = s["fragment_id"]
        if e["measurement"]["counts_hash"] != s["replay"]["counts_sha256"]:
            raise MeasurementPlanError(
                f"fragment {fid}: the replayed counts differ from the measurement the promotion "
                "manifest was validated on")
        if e["qpu_backend"] != s["backend"]:
            raise MeasurementPlanError(
                f"fragment {fid}: backend {s['backend']!r} differs from the promotion's {e['qpu_backend']!r}")


def job_payload(specs: Sequence[Mapping[str, Any]], remote_names: Mapping[int, str]) -> dict:
    """The JSON-safe ``measurement_plan`` argument sent to the Stage 3 function (string keys)."""
    out = {}
    for s in specs:
        fid = int(s["fragment_id"])
        rep = s["replay"]
        out[str(fid)] = {
            "mode": s["mode"], "solver": s["solver"], "n_qubits": s["n_qubits"], "backend": s["backend"],
            "hamiltonian_sha256": s["hamiltonian_sha256"],
            "counts_file": remote_names.get(fid) if s["mode"] == "replay" else None,
            "counts_sha256": rep["counts_sha256"] if rep else None,
        }
    return out


def bind_acquired(spec: Mapping[str, Any], *, artifact: Mapping[str, str], path: Path | str) -> dict:
    """Turn a live spec into the replay spec of the artifact it produced (provenance from its sidecar)."""
    return replay_spec(fragment_id=spec["fragment_id"], hamiltonian_sha256=spec["hamiltonian_sha256"],
                       n_orb=spec["n_orb"], backend=spec["backend"], artifact=artifact, path=path,
                       solver=spec["solver"])
