"""Notebook-facing, current-run SQD/TrimSQD promotion.

This module contains no QPU or Serverless execution. Existing protocol-transfer,
promotion-manifest, and Challenge validators remain the source of truth.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from proteasome_challenge import promotion_manifest as pm
from proteasome_challenge import protocol_transfer as pt


class TunedPromotionError(RuntimeError):
    """Refusal to promote stale or mismatched scientific evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TunedPromotionError(message)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".promotion_", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _verify_grid(fragment_lab: Any, ligand: str, fid: int, rows: list[dict],
                 source_hash: str) -> tuple[Path, dict[str, dict]]:
    path = Path(fragment_lab._grid_results_path(ligand, fid)).resolve()
    _require(path.is_file(), "Current-run tuning grid not persisted")
    data = json.loads(path.read_text(encoding="utf-8"))
    _require(data.get("ligand") == ligand and int(data.get("fragment_id", -1)) == fid,
             "Tuning results belong to a different molecule or fragment")
    saved = data.get("tuning_results")
    defs = {p["label"]: p for p in fragment_lab.load_tuning_grid()}
    _require(isinstance(saved, list) and isinstance(rows, list), "Tuning grid must contain result rows")
    _require(len(saved) == len(rows) == len(defs) == 8, "Exactly eight tuning results required")
    by_disk = {r["label"]: r for r in saved}
    by_mem = {r["label"]: r for r in rows}
    _require(len(by_disk) == len(by_mem) == 8 and set(defs) == set(by_disk) == set(by_mem),
             "Tuning labels are incomplete or duplicated")
    for label, stored in by_disk.items():
        current = by_mem[label]
        _require(stored.get("hamiltonian_sha256") == current.get("hamiltonian_sha256") == source_hash,
                 f"{label}: Hamiltonian mismatch")
        for item in (stored, current):
            _require(math.isfinite(float(item["energy_ha"])), f"{label}: missing/nonfinite energy")
        _require(float(stored["energy_ha"]) == float(current["energy_ha"]),
                 f"{label}: saved energy differs from kernel")
        for field in ("solver_method", "sqd_config", "trim_config"):
            _require(stored.get(field) == current.get(field), f"{label}: {field} mismatch")
            if defs[label].get(field) is not None:
                _require(stored.get(field) == defs[label][field],
                         f"{label}: {field} differs from the approved grid")
    return path, by_disk


def _preserve_previous(path: Path, existing: dict, replacement: dict) -> None:
    """Only permitted replacement: declared -> measured or backend metadata only.

    FREEZE_PROTOCOL=True in the notebook is the single write authorization.
    """
    old = existing["shared_protocol"]
    new = replacement["shared_protocol"]
    old_source, new_source = old["source"], new["source"]
    declared = old_source["grid_label"] == "declared_reference"
    _require(old["target_fragment_ids"] == new["target_fragment_ids"],
             "Refusing to change the previously selected target fragments")
    _require(old["protocol_digest"] == new["protocol_digest"],
             "Refusing to change transferable scientific settings")
    _require(old_source["hamiltonian_sha256"] == new_source["hamiltonian_sha256"],
             "Existing source Hamiltonian differs")
    _require(existing["baseline"] == replacement["baseline"],
             "Reference baseline identity differs")
    if not declared:
        _require(old_source["grid_label"] == new_source["grid_label"], "Different selected grid label")
        _require(old_source["grid_energy_ha"] == new_source["grid_energy_ha"], "Different tuning energy")
        _require(old_source["validated_result"] == new_source["validated_result"],
                 "Different persisted tuning evidence")
        for key in ("counts_hash", "n_shots", "n_reps", "counts_source"):
            _require(old_source["acquisition"][key] == new_source["acquisition"][key],
                     f"Cannot correct a changed acquisition {key}")
        _require(old["transferable"] == new["transferable"], "Changed solver configuration")
        _require(old_source["acquisition"]["qpu_backend"] != new_source["acquisition"]["qpu_backend"],
                 "No backend correction needed")
    backup = path.with_name(
        f"{path.stem}.{'declared' if declared else 'prior_backend'}.{existing['manifest_digest']}.json")
    if backup.exists():
        _require(backup.read_bytes() == path.read_bytes(), "Existing archive conflicts with current file")
    else:
        # Exclusive create; preserve the old bytes exactly.
        fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(path.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            backup.unlink(missing_ok=True)
            raise


def freeze_tuned_protocol(*, lab: Any, fragment_lab: Any, run_ctx: Any, ligand: str,
                          fragment_id: int, fragment_ids: list[int], grid_results: list[dict],
                          selected_label: str, ccsd_hash: str, qpu_backend: str,
                          n_shots: int, n_reps: int) -> tuple[dict, dict]:
    """Promote a measured grid and seal it using existing schema 1.1 validators.

    ``qpu_backend`` comes from the notebook's configured scientific acquisition
    setting, not an ad-hoc promotion override. It must agree with the original
    source acquisition record when that record provides a backend identity.
    """
    rid = run_ctx.run_id
    _require(fragment_lab.run_id == rid, "Fragment laboratory does not belong to current run")
    _require(isinstance(qpu_backend, str) and qpu_backend.strip(),
             "QPU_BACKEND must be set in the notebook")
    _require(run_ctx.manifest is None, "RunManifest already frozen; cannot replace protocol")
    _require(not run_ctx.registry.list_tasks(rid), "Fragment tasks exist; protocol is immutable")
    run_dir = Path(run_ctx.paths.run_dir).resolve()
    _require(not (run_dir / "shared_protocol.json").exists(),
             "Run-local shared_protocol.json already exists")
    fid = int(fragment_id)
    selection = lab.load_final_selection()
    baseline = lab.reference_baseline(ligand)
    ids = sorted(int(x) for x in fragment_ids)
    _require(selection.get("ligand") == ligand
             and ids == sorted(selection["selected_fragment_ids"]), "Final selection mismatch")
    base_hashes = {int(f["fragment_id"]): f["hamiltonian_hash"] for f in baseline["fragments"]}
    _require(fid in ids and base_hashes.get(fid) == ccsd_hash,
             "CCSD and Stage 2 source Hamiltonians disagree")
    grid_path, by_label = _verify_grid(fragment_lab, ligand, fid, grid_results, ccsd_hash)
    _require(grid_path.is_relative_to(run_dir), "Grid results are not scoped to the current run")
    _require(selected_label in by_label, "Selected configuration missing from measured grid")
    chosen = by_label[selected_label]
    _require(chosen.get("solver_method") == "trimsqd", "Hybrid production requires TrimSQD")
    counts_path = fragment_lab.resolve_counts(ligand, fid)
    _require(counts_path is not None, "Current-run source measurement file missing")
    counts_path = Path(counts_path).resolve()
    _require(counts_path.is_file() and counts_path.is_relative_to(run_dir),
             "Counts file is not authenticated within the current run")
    counts_hash = _sha256(counts_path)
    grid_hash = _sha256(grid_path)
    # Preserve the backend recorded in the actual source-measurement result,
    # where available. Replay/tuning job backends are not acquisition proof.
    # Do not silently promote a notebook setting that conflicts with an
    # independently recorded acquisition backend.
    result_path = (Path(fragment_lab.ckpt_dir) / "stage3" /
                   f"{ligand}_frag{fid}_both.json")
    if result_path.is_file():
        measured_result = json.loads(result_path.read_text(encoding="utf-8"))
        recorded = measured_result.get("acquisition")
        if isinstance(recorded, dict):
            recorded_backend = recorded.get("qpu_backend") or recorded.get("backend")
            recorded_counts = recorded.get("counts_hash")
            if recorded_backend and recorded_counts == counts_hash:
                _require(recorded_backend == qpu_backend.strip(),
                         "Notebook QPU_BACKEND disagrees with recorded source acquisition")
    source = {
        "selected_grid_label": selected_label, "grid_label": selected_label,
        "ligand": ligand, "fragment_id": fid, "solver_method": "trimsqd", "solver_mode": "trimsqd",
        "sqd_config": dict(chosen["sqd_config"]), "trim_config": dict(chosen["trim_config"]),
        "n_shots": int(n_shots), "n_reps": int(n_reps),
        "qpu_backend": qpu_backend.strip(),
        "grid_energy_ha": float(chosen["energy_ha"]), "measurement_artifact": str(counts_path),
        "hamiltonian_sha256": ccsd_hash,
        "counts_source": fragment_lab._counts_source(ligand, fid),
        "counts_hash": counts_hash,
        "rationale": f"Eight measured grid points, run {rid}, artifact sha256={grid_hash}",
        "final_selection_digest": selection["manifest_digest"],
        "baseline_hamiltonian_hash": ccsd_hash,
    }
    validated = {"source": "tuning_results", "file": grid_path.name, "sha256": grid_hash,
                 "grid_label": selected_label, "hamiltonian_sha256": ccsd_hash}
    measurement = {"counts_hash": counts_hash, "counts_source": source["counts_source"],
                   "artifact": counts_path.name}
    digest = lab._config_digest(source)
    protocol = pt.transfer_protocol(source, ids, selection=selection, baseline=baseline,
                                    config_digest=digest, validated_result=validated)
    entry = pm.entry_from_promoted_record(source, config_digest=digest,
                                          validated_result=validated, measurement=measurement)
    candidate = pm.shared_manifest(protocol, source_entry=entry, selection=selection,
                                   baseline=baseline, config_digest_fn=lab._config_digest)
    candidate = pm.finalize(candidate, selection=selection, baseline=baseline,
                            config_digest_fn=lab._config_digest)
    path = Path(lab.promotion_manifest_path)
    if path.is_file():
        existing = lab.load_promotion_manifest(require_complete=True)
        if existing["manifest_digest"] != candidate["manifest_digest"]:
            _preserve_previous(path, existing, candidate)
            _write_atomic(path, candidate)
    else:
        _write_atomic(path, candidate)
    result = lab.load_promotion_manifest(require_complete=True)
    _require(result["manifest_digest"] == candidate["manifest_digest"],
             "The persisted manifest does not match the validated candidate")
    return result, lab.load_shared_protocol(require_complete=True)
