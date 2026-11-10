"""
Client side of single-source protocol transfer (Task 4B.2): the ``Challenge`` mixin.

Notebook-level flow (nothing here submits a job or touches the QPU)::

    promoted = lab.promote_configuration(label, results, fragment_id=SOURCE)   # Stage E, ONE fragment
    manifest = lab.transfer_promoted_protocol(promoted)                        # source + 10 transferred
    protocol = lab.load_shared_protocol()
    lab.shared_protocol_report()                                               # optimized vs transferred
    # Stage F job (via _submit_planned_stage3(..., protocol=protocol)); then
    lab.verify_protocol_result(protocol, result)

The protocol is stored INSIDE ``checkpoints/promotion_manifest.json`` (schema 1.1) -- the Task 4A
manifest, extended, not a second promotion system.  ``promoted_configuration.json`` is never written.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from proteasome_challenge import (SELECTED_LIGAND, ChallengeError, _coerce_fragment_id, _sha256_file)
from proteasome_challenge import protocol_transfer as pt


class ProtocolTransferMixin:
    def _protocol_context(self, ligand: str | None):
        ligand = ligand or SELECTED_LIGAND
        return ligand, self.load_final_selection(), self.reference_baseline(ligand)

    def _validated_and_measurement(self, ligand: str, record: dict) -> tuple[dict | None, dict | None]:
        """The evidence that exists on disk for the SOURCE promotion (never invented)."""
        fid = _coerce_fragment_id(record.get("fragment_id"), "record fragment_id")
        label = record.get("selected_grid_label") or record.get("grid_label")
        validated = None
        grid_p = self._grid_results_path(ligand, fid)
        if grid_p.is_file():
            rows = json.loads(grid_p.read_text(encoding="utf-8")).get("tuning_results", [])
            row = next((r for r in rows if r.get("label") == label), None)
            if row is not None:
                validated = {"source": "tuning_results", "file": grid_p.name, "sha256": _sha256_file(grid_p),
                             "grid_label": label, "hamiltonian_sha256": row.get("hamiltonian_sha256")}
        measurement = None
        if record.get("counts_hash"):
            art = record.get("measurement_artifact")
            measurement = {"counts_hash": record["counts_hash"], "counts_source": record.get("counts_source"),
                           "artifact": Path(art).name if art else None}
        return validated, measurement

    def transfer_promoted_protocol(self, source: dict | None = None, target_fragment_ids=None, *,
                                   ligand: str | None = None, overwrite: bool = False,
                                   adaptations=()) -> dict:
        """Freeze ONE promoted Stage E configuration as the protocol for every TrimSQD fragment.

        ``source``  a ``promote_configuration`` record (any fragment of the frozen selection; pass
                    ``persist=False`` records freely), or ``None`` for ``promoted_configuration.json``.
        ``target_fragment_ids``  default: the whole frozen selection.
        Writes ``promotion_manifest.json`` (schema 1.1): the source entry is ``optimized``, every
        other target ``transferred`` (configuration only; no result or measurement is implied).
        An existing manifest is kept unless ``overwrite=True``.  Raises before writing when the
        source is missing/stale, a target is outside the selection, or the settings do not apply
        unchanged.
        """
        from proteasome_challenge import promotion_manifest as pm
        ligand, sel, base = self._protocol_context(ligand)
        if source is None:
            source = self.load_promoted_configuration()
        if not isinstance(source, dict):
            raise ChallengeError("source must be a promote_configuration record or None")
        ids = sel["selected_fragment_ids"] if target_fragment_ids is None else list(target_fragment_ids)
        if self.promotion_manifest_path.is_file() and not overwrite:
            raise ChallengeError(
                f"{self.promotion_manifest_path.name} already exists; load it, or pass overwrite=True.")
        try:
            digest = self._config_digest(source)
            validated, measurement = self._validated_and_measurement(ligand, source)
            protocol = pt.transfer_protocol(source, ids, selection=sel, baseline=base, config_digest=digest,
                                            validated_result=validated, adaptations=adaptations)
            src_entry = pm.entry_from_promoted_record(source, config_digest=digest,
                                                      validated_result=validated, measurement=measurement)
            manifest = pm.shared_manifest(protocol, source_entry=src_entry, selection=sel, baseline=base,
                                          config_digest_fn=self._config_digest)
            if protocol["complete"]:
                manifest = pm.finalize(manifest, selection=sel, baseline=base,
                                       config_digest_fn=self._config_digest)
            else:
                pm.validate_manifest(manifest, selection=sel, baseline=base, require_complete=False,
                                     config_digest_fn=self._config_digest)
        except (pt.ProtocolTransferError, pm.PromotionManifestError) as exc:
            raise ChallengeError(str(exc)) from exc
        pm.write_manifest(manifest, self.promotion_manifest_path)
        return manifest

    def load_shared_protocol(self, *, require_complete: bool = True) -> dict:
        """The validated shared protocol of ``promotion_manifest.json`` (schema 1.1 only)."""
        manifest = self.load_promotion_manifest(require_complete=require_complete)
        if manifest.get("schema_version") != "1.1":
            raise ChallengeError(
                "promotion_manifest.json holds independently configured fragments (schema 1.0), not a "
                "shared protocol; call transfer_promoted_protocol() to create one.")
        return manifest["shared_protocol"]

    def shared_protocol_report(self) -> list[dict]:
        """Per selected fragment: origin (optimized/transferred), configured, validated, measurement."""
        return self.promotion_manifest_report()

    def shared_protocol_payload(self, protocol: dict | None = None) -> dict:
        """The validated ``shared_protocol`` job argument."""
        from proteasome_challenge import promotion_manifest as pm
        protocol = self.load_shared_protocol(require_complete=False) if protocol is None else protocol
        ligand, sel, base = self._protocol_context(protocol.get("ligand") if isinstance(protocol, dict) else None)
        try:
            return pt.job_payload(pt.validate_protocol(protocol, selection=sel, baseline=base))
        except pt.ProtocolTransferError as exc:
            raise ChallengeError(str(exc)) from exc

    def verify_protocol_result(self, protocol: dict, result: dict) -> None:
        """Fail closed unless a Stage 3 ``result`` proves the protocol reached every quantum fragment."""
        if not isinstance(result, dict):
            raise ChallengeError("the Stage 3 result is not an object")
        if result.get("shared_protocol_api") != pt.PROTOCOL_API:
            raise ChallengeError(
                "the result does not carry shared_protocol_api; the deployed bundle did not apply the "
                "shared protocol (redeploy the bundle).")
        try:
            pt.check_applied(protocol, result.get("fragment_summaries") or {}, result.get("quantum_protocol"))
        except pt.ProtocolTransferError as exc:
            raise ChallengeError(str(exc)) from exc
