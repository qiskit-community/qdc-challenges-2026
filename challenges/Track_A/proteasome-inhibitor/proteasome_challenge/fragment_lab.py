"""Current-run single-fragment CCSD, SQD, and TrimSQD laboratory."""

import hashlib
import json
from pathlib import Path

from . import Challenge, ChallengeError, SQD_THRESHOLD


class RunScopedFragmentLab(Challenge):
    """Use verified run-scoped EWF inputs with the existing Stage 3 API."""

    def __init__(self, parent, run_ctx, mf_ref, ewf_ref,
                 selected_ids, catalog, ligand):

        if run_ctx is None or not getattr(run_ctx, "run_id", None):
            raise ChallengeError("A current RunContext is required.")

        if any(ref is None or ref.run_id != run_ctx.run_id
               for ref in (mf_ref, ewf_ref)):
            raise ChallengeError("Stage 1/2 references must match the run ID.")

        if catalog is None or catalog.empty or not {
            "fragment_id", "n_orb"
        } <= set(catalog.columns):
            raise ChallengeError("Current Stage 2 catalog is required.")

        if catalog["fragment_id"].duplicated().any():
            raise ChallengeError("Duplicate Stage 2 fragment IDs.")

        selected = [int(fid) for fid in selected_ids]
        eligible = {
            int(fid)
            for fid, norb in zip(catalog["fragment_id"], catalog["n_orb"])
            if int(norb) >= SQD_THRESHOLD
        }

        if (not selected or len(selected) != len(set(selected))
                or set(selected) != eligible):
            raise ChallengeError(
                "Selection does not match the current Stage 2 catalog."
            )

        super().__init__(
            notebook_dir=parent.notebook_dir,
            allow_unverified_bundle=parent.allow_unverified_bundle,
        )

        self.run_id = str(run_ctx.run_id)
        self._artifact_store = run_ctx.store
        self._trial_dir = (
            Path(run_ctx.store.root) / "runs" / self.run_id
        ).resolve()

        self._source = parent
        self._source_ligand = str(ligand)
        self._selected_ids = frozenset(selected)
        self._source_refs = {
            "mf_data.pkl": mf_ref,
            "ewf_dumpfile.h5": ewf_ref,
        }

        self.ckpt_dir = self._trial_dir / "single_fragment_lab"

        for suffix, ref in self._source_refs.items():
            expected = f"{self._run_name(ligand)}__{suffix}"
            if ref.artifact_name != expected:
                raise ChallengeError(
                    f"Unexpected artifact name: {ref.artifact_name}"
                )
            self._verified_file(ref)

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _verified_file(self, ref):
        path = Path(ref.uri).resolve()

        if not path.is_relative_to(self._trial_dir) or not path.is_file():
            raise ChallengeError(
                f"Current-run artifact unavailable: {path}"
            )

        with path.open("rb") as stream:
            actual = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()

        if actual != ref.file_sha256:
            raise ChallengeError(
                f"Artifact SHA-256 mismatch: {path.name}"
            )

        return path

    def _resolve_quantum_fragment(self, fragment_id, caller):
        if fragment_id is None:
            raise ChallengeError(
                f"{caller}: select an explicit fragment ID."
            )

        fid = int(fragment_id)

        if fid not in self._selected_ids:
            raise ChallengeError(
                f"{caller}: fragment {fid} is not selected."
            )

        return fid

    def _ensure_stage3_files(self, ligand, fn):
        if ligand != self._source_ligand:
            raise ChallengeError(f"Unexpected ligand: {ligand}")

        for suffix, ref in self._source_refs.items():
            self._upload_file_once(
                self._verified_file(ref),
                f"{self._run_name(ligand)}__{suffix}",
                fn,
            )

    def _assert_bundle_deployed(self):
        self._source._assert_bundle_deployed()

    def resolve_counts(self, ligand, fragment_id):
        if ligand != self._source_ligand:
            raise ChallengeError(f"Unexpected ligand: {ligand}")

        path = self._live_counts_path(ligand, fragment_id)
        return path if path.is_file() else None

    def load_archived_results(self):
        return {}

    def _load_fragment_checkpoint(self, ligand, fragment_id, method):
        if ligand != self._source_ligand:
            raise ChallengeError(f"Unexpected ligand: {ligand}")

        path = (
            self.ckpt_dir / "stage3"
            / f"{ligand}_frag{int(fragment_id)}_{method}.json"
        )

        if not path.is_file():
            raise ChallengeError(
                f"No current-run {method} checkpoint for fragment "
                f"{fragment_id}."
            )

        result = json.loads(path.read_text(encoding="utf-8"))

        self._check_result_fragment(
            result, int(fragment_id), "current-run checkpoint"
        )
        self._check_method_used(result, method)

        if not result.get("hamiltonian_sha256"):
            raise ChallengeError(
                "Saved result lacks a Hamiltonian SHA-256."
            )

        return result



    def publish_ccsd_artifact(self, fragment_id, job_id):
        """Publish a verified CCSD checkpoint to the run artifact store."""
        fid = self._resolve_quantum_fragment(
            fragment_id, "publish_ccsd_artifact"
        )
        path = self.ckpt_dir / "stage3" / (
            f"{self._source_ligand}_frag{fid}_ccsd.json"
        )

        if not path.is_file():
            raise ChallengeError(f"CCSD checkpoint missing: {path}")

        result = json.loads(path.read_text())
        self._check_result_fragment(result, fid, "CCSD artifact")
        self._check_method_used(result, "ccsd")

        energy = self.energy_of(result, "ccsd")
        ham_hash = result.get("hamiltonian_sha256")

        if energy is None or not isinstance(ham_hash, str) or len(ham_hash) != 64:
            raise ChallengeError("Incomplete CCSD artifact metadata.")

        ref = self._artifact_store.publish(
            path,
            kind="fragment_result",
            run_id=self.run_id,
            artifact_name=f"{self._source_ligand}_frag{fid}_ccsd.json",
            producer_version="fragment_lab/1",
            meta={
                "schema": "fragment_ccsd/1",
                "run_id": self.run_id,
                "ligand": self._source_ligand,
                "fragment_id": fid,
                "solver": "CCSD",
                "energy_ha": energy,
                "hamiltonian_sha256": ham_hash,
                "serverless_job_id": str(job_id),
                "stage1_sha256": self._source_refs["mf_data.pkl"].file_sha256,
                "stage2_sha256": self._source_refs["ewf_dumpfile.h5"].file_sha256,
                "stage2_artifact_name": self._source_refs["ewf_dumpfile.h5"].artifact_name,
            },
        )
        return ref

    def _collect_fragment(self, job, record, poll_seconds):
        result = super()._collect_fragment(job, record, poll_seconds)
        if record["method"] == "ccsd":
            ref = self.publish_ccsd_artifact(
                record["fragment_id"], record["job_id"]
            )
            print(f"CCSD artifact published: {ref.uri}")
        return result


def prepare_fragment_lab(
    parent, run_ctx, mf_ref, ewf_ref, selected_ids, catalog, ligand
):
    """Initialize a current-run fragment laboratory."""
    return RunScopedFragmentLab(
        parent, run_ctx, mf_ref, ewf_ref,
        selected_ids, catalog, ligand,
    )
