"""
Client side of fragment-specific measurement routing (Task 4B.1): the ``Challenge`` mixin.

Public notebook interface (all methods of ``Challenge``)::

    plan = lab.build_measurement_plan(live=[2, 3], replay=[1, 9], backend="ibm_pittsburgh",
                                      n_shots=100_000, n_reps=1)       # explicit mode per fragment
    lab.show_measurement_plan(plan)                                    # table; nothing submitted
    jobs = lab.submit_measurement_plan(plan, confirm_live=True, allow_second_qpu=True)   # prints job ids
    out  = lab.retrieve_measurements(plan)                             # resumable, per fragment
    lab.measurement_status(plan)                                       # what is submitted / acquired
    done = lab.plan_from_acquired(plan)                                # live specs -> replay of their artifacts
    lab.check_measurement_plan(done)                                   # against the promotion manifest

Rules enforced here (the container re-checks them in ``step3_solve.run_solve``):

* the mode of every fragment is explicit; counts are never inferred from "a file happens to exist";
* a replay never samples: no sampling is recorded, the budget gate is not consulted, and the
  artifact is re-hashed against the plan right before it is uploaded;
* a live acquisition is stored as a NEW write-once artifact under ``checkpoints/counts/runs/`` with a
  provenance sidecar.  The legacy ``..._counts_frag<id>.npy`` files and the frozen counts are never
  written, replaced or deleted by this API.
"""
from __future__ import annotations

import base64
import gzip
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from proteasome_challenge import (BASIS, DEFAULT_N_REPS, DEFAULT_N_SHOTS, FCI_THRESHOLD,
                                  SELECTED_LIGAND, SQD_THRESHOLD, ChallengeError, resolve_qpu_backend)
from proteasome_challenge import measurement_plan as mp

_METHOD_OF = {"TrimSQD": "trimsqd", "SQD": "sqd"}
_BAD_END = ("ERROR", "FAILED", "CANCELED", "CANCELLED")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fid(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise ChallengeError(f"fragment ids must be integers, got {v!r}")
    return int(v)


class MeasurementRoutingMixin:
    # ── locations ───────────────────────────────────────────────────────────────────────────────

    @property
    def measurements_dir(self) -> Path:
        return self.ckpt_dir / "measurements"

    @property
    def counts_runs_dir(self) -> Path:
        """Write-once home of every NEW live acquisition (never the legacy per-fragment path)."""
        return self.ckpt_dir / "counts" / "runs"

    def _artifact_dir(self, location: str) -> Path:
        return {"frozen": self.frozen_counts_dir, "checkpoint": self.ckpt_dir / "counts",
                "run": self.counts_runs_dir}[location]

    def _artifact_path(self, artifact: dict) -> Path:
        return self._artifact_dir(artifact["location"]) / mp.safe_name(artifact["name"])

    def _meas_job_path(self, ligand: str, fid: int, plan_digest: str) -> Path:
        return self.measurements_dir / "jobs" / f"{ligand}_frag{int(fid)}_{plan_digest[:12]}.job.json"

    # ── plan construction ───────────────────────────────────────────────────────────────────────

    def _plan_context(self, ligand: str | None) -> tuple[str, dict, dict]:
        ligand = ligand or SELECTED_LIGAND
        return ligand, self.load_final_selection(), self.reference_baseline(ligand)

    def _replay_candidates(self, ligand: str, fid: int) -> list[dict]:
        found = []
        legacy = self._live_counts_path(ligand, fid)
        if legacy.is_file():
            found.append({"name": legacy.name, "location": "checkpoint"})
        frozen = self._frozen_counts_path(ligand, fid)
        if frozen.is_file():
            found.append({"name": frozen.name, "location": "frozen"})
        if self.counts_runs_dir.is_dir():
            for p in sorted(self.counts_runs_dir.glob(f"{ligand}_frag{int(fid)}_*.npy")):
                if mp.names_fragment(p.name, fid):
                    found.append({"name": p.name, "location": "run"})
        return found

    def _recorded_backend(self, ligand: str, fid: int, ham: str) -> str | None:
        """Backend of the QPU job that produced this fragment's counts, from a saved Stage 3 result.

        Accepted only when that result names a QPU job id (a replay job has none) and the same
        embedded Hamiltonian.
        """
        found = set()
        for method in ("both", "trimsqd", "sqd"):
            p = self.ckpt_dir / "stage3" / f"{ligand}_frag{int(fid)}_{method}.json"
            if not p.is_file():
                continue
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if rec.get("qpu_job_id") and rec.get("hamiltonian_sha256") == ham and rec.get("qpu_backend"):
                found.add(rec["qpu_backend"])
        if len(found) > 1:
            raise ChallengeError(f"fragment {fid}: saved results disagree on the acquisition backend {sorted(found)}")
        return found.pop() if found else None

    def build_measurement_plan(self, ligand: str | None = None, *, live=(), replay=None,
                               backend: str | None = None, replay_backends: dict | None = None,
                               n_shots: int = DEFAULT_N_SHOTS, n_reps: int = DEFAULT_N_REPS,
                               sqd_config: dict | None = None, trim_config: dict | None = None,
                               solver: str = "TrimSQD", require_full_selection: bool = False) -> dict:
        """Build and validate a sealed measurement plan.  Submits and writes nothing.

        ``live``    fragment ids to acquire on ``backend`` (default ``IBM_QUANTUM_BACKEND``).
        ``replay``  fragment ids, or ``{fragment_id: artifact_name | {name, location}}``.  A bare id is
                    resolved only when exactly ONE counts artifact exists for it (legacy checkpoint
                    file, frozen file, or a run artifact); otherwise name the artifact.
        ``replay_backends``  ``{fragment_id: backend}`` for artifacts that carry no provenance (neither
                    a sidecar nor a saved Stage 3 result of the acquiring job).
        Every fragment must be in the frozen selection and appear once; its Hamiltonian hash and
        ``2*n_orb`` qubit count come from the classical baseline.
        """
        ligand, sel, base = self._plan_context(ligand)
        view = mp.baseline_view(base)
        live_ids = [_fid(i) for i in live]
        if replay is None:
            replay_map: dict = {}
        elif isinstance(replay, dict):
            replay_map = {_fid(k): v for k, v in replay.items()}
        else:
            replay_list = [_fid(i) for i in replay]
            replay_map = {i: None for i in replay_list}
            if len(replay_map) != len(replay_list):
                raise ChallengeError(f"duplicate fragment ids in replay={replay_list}")
        both = sorted(set(live_ids) & set(replay_map))
        if both or len(set(live_ids)) != len(live_ids):
            raise ChallengeError(
                f"a fragment has exactly one measurement mode: {both or sorted(i for i in live_ids if live_ids.count(i) > 1)} "
                "appear more than once in live/replay")
        for i in [*live_ids, *replay_map]:
            if i not in view:
                raise ChallengeError(f"fragment {i} is not part of the classical baseline")
        specs = []
        if live_ids:
            # require=True: refuse to plan live fragments when no backend is configured.
            live_backend = resolve_qpu_backend(backend, require=backend is None)
        for i in live_ids:
            ham, n_orb = view[i]
            self._validate_solver_settings(sqd_config, trim_config)
            specs.append(mp.live_spec(fragment_id=i, hamiltonian_sha256=ham, n_orb=n_orb,
                                      backend=live_backend, solver=solver, n_shots=int(n_shots),
                                      n_reps=int(n_reps), sqd_config=sqd_config, trim_config=trim_config))
        for i, art in replay_map.items():
            ham, n_orb = view[i]
            if art is None:
                cands = self._replay_candidates(ligand, i)
                if not cands:
                    raise ChallengeError(
                        f"fragment {i}: replay was requested but no counts artifact exists "
                        "(replay never samples). Acquire it with a live plan, or name the file.")
                if len(cands) > 1:
                    raise ChallengeError(
                        f"fragment {i}: {len(cands)} counts artifacts exist "
                        f"({[c['name'] for c in cands]}); choose one with replay={{{i}: '<name>'}}.")
                art = cands[0]
            elif isinstance(art, str):
                art = self._locate_artifact(ligand, i, art)
            path = self._artifact_path(art)
            side = mp.read_sidecar(path) if path.is_file() else None
            recorded = self._recorded_backend(ligand, i, ham)
            declared = (replay_backends or {}).get(i)
            if declared is not None and recorded is not None and declared != recorded:
                raise ChallengeError(
                    f"fragment {i}: declared backend {declared!r} contradicts the backend {recorded!r} "
                    "recorded for the job that acquired these counts")
            chosen = declared or recorded
            source = "stage3_record" if (declared is None and recorded is not None) else "declared"
            if side is None and chosen is None:
                raise ChallengeError(
                    f"fragment {i}: the counts artifact {art['name']} has no recorded backend. Pass "
                    f"replay_backends={{{i}: '<backend>'}} (the backend it was acquired on).")
            specs.append(mp.replay_spec(fragment_id=i, hamiltonian_sha256=ham, n_orb=n_orb,
                                        backend=chosen, artifact=art, path=path, solver=solver,
                                        backend_source=source))
        if not specs:
            raise ChallengeError("an empty measurement plan: give live=[...] and/or replay=[...]")
        plan = mp.new_plan(ligand=ligand, selection=sel, baseline=base, specs=specs)
        return mp.validate_plan(plan, selection=sel, baseline=base,
                                require_full_selection=require_full_selection)

    def _locate_artifact(self, ligand: str, fid: int, name: str) -> dict:
        hits = [{"name": mp.safe_name(name), "location": loc} for loc in ("run", "checkpoint", "frozen")
                if (self._artifact_dir(loc) / mp.safe_name(name)).is_file()]
        if not hits:
            raise ChallengeError(f"fragment {fid}: counts artifact {name!r} was not found")
        if len(hits) > 1:
            raise ChallengeError(f"fragment {fid}: {name!r} exists in several locations "
                                 f"{[h['location'] for h in hits]}; pass {{'name':..., 'location':...}}")
        return hits[0]

    def _validated_plan(self, plan: Any, *, require_full_selection: bool = False) -> dict:
        if not isinstance(plan, dict) or "ligand" not in plan:
            raise ChallengeError("expected a measurement plan from build_measurement_plan()")
        _, sel, base = self._plan_context(plan["ligand"])
        return mp.validate_plan(plan, selection=sel, baseline=base,
                                require_full_selection=require_full_selection)

    def show_measurement_plan(self, plan: dict) -> pd.DataFrame:
        """Validate ``plan`` and return (and print) one row per fragment.  Contacts nothing."""
        plan = self._validated_plan(plan)
        rows = []
        for s in plan["specs"]:
            rep = s["replay"] or {}
            live = s["live"] or {}
            rows.append({
                "fragment": s["fragment_id"], "mode": s["mode"], "solver": s["solver"],
                "n_orb": s["n_orb"], "qubits": s["n_qubits"], "backend": s["backend"],
                "shots": live.get("n_shots"), "artifact": (rep.get("artifact") or {}).get("name"),
                "counts_sha256": (rep.get("counts_sha256") or "")[:12],
                "provenance": rep.get("provenance"),
            })
        df = pd.DataFrame(rows)
        print(f"Measurement plan {plan['plan_digest'][:12]}  execution={plan['execution']}  "
              f"fragments={len(rows)}  ligand={plan['ligand']}")
        print(df.to_string(index=False))
        return df

    def check_measurement_plan(self, plan: dict, manifest: dict | None = None) -> None:
        """Cross-check replay specs against a promotion manifest (counts hash and backend)."""
        plan = self._validated_plan(plan)
        if manifest is None:
            manifest = self.load_promotion_manifest(require_complete=False)
        mp.check_against_manifest(plan, manifest)

    # ── Stage 3 submission with a plan ──────────────────────────────────────────────────────────

    def _submit_planned_stage3(self, ligand: str, *, label: str, plan: dict, specs: list[dict],
                               reconstruct: bool, fragment_ids=None, quantum_fragment_ids=None,
                               solver_assignments=None, partial_assignments=False,
                               allow_sampling: bool = False, allow_second_qpu: bool = False,
                               sqd_config=None, trim_config=None, protocol: dict | None = None):
        """Submit ONE Stage 3 job whose quantum fragments are ``specs`` of the sealed ``plan``.

        Raises before any network call when the plan is invalid, when specs disagree on solver /
        backend / sampling settings, when a replay artifact no longer matches its pinned SHA-256,
        or when a live spec is not authorised.  With ``protocol`` (Task 4B.2) the SAME frozen
        TrimSQD settings are sent for every fragment: the job's ``sqd_config`` / ``trim_config`` are the
        protocol's, a live spec that carries different settings is refused, and every spec must be a
        TrimSQD target of the protocol.  ``will_sample`` is true iff some spec is live; then
        ``allow_sampling`` and the QPU budget gate apply, and one sampling is recorded after the
        gateway accepted the job.  A replay-only job never samples and never touches the gate.
        """
        plan = self._validated_plan(plan)
        if not specs:
            raise ChallengeError("no fragment specs to submit")
        ids = [s["fragment_id"] for s in specs]
        if len(set(ids)) != len(ids):
            raise ChallengeError(f"duplicate fragment ids in the job: {ids}")
        planned = {s["fragment_id"]: s for s in plan["specs"]}
        for s in specs:
            if planned.get(s["fragment_id"]) != s:
                raise ChallengeError(f"fragment {s['fragment_id']}: the spec is not the plan's spec")
        solvers = {s["solver"] for s in specs}
        if len(solvers) != 1:
            raise ChallengeError(f"one job runs one quantum solver; the specs name {sorted(solvers)}")
        live = [s for s in specs if s["mode"] == "live"]
        backends = {s["backend"] for s in live} or {s["backend"] for s in specs}
        if live and len(backends) != 1:
            raise ChallengeError(f"live fragments in one job must share a backend, got {sorted(backends)}")
        # When no backend can be inferred from the specs (replay-only job), require=True
        # ensures a missing IBM_QUANTUM_BACKEND env-var raises before the job is submitted.
        _has_live = bool(live)
        qpu_backend = backends.pop() if len(backends) == 1 else resolve_qpu_backend(None, require=_has_live)
        shot_cfg = {(s["live"]["n_shots"], s["live"]["n_reps"],
                     json.dumps([s["live"]["sqd_config"], s["live"]["trim_config"]], sort_keys=True))
                    for s in live}
        if len(shot_cfg) > 1:
            raise ChallengeError("live fragments in one job must share n_shots / n_reps / solver settings; "
                                 "submit them as separate jobs")
        n_shots, n_reps = (next(iter(live))["live"]["n_shots"], next(iter(live))["live"]["n_reps"]) \
            if live else (DEFAULT_N_SHOTS, DEFAULT_N_REPS)
        if live:
            sqd_config = live[0]["live"]["sqd_config"]
            trim_config = live[0]["live"]["trim_config"]
        shared_payload = None
        if protocol is not None:
            shared_payload = self.shared_protocol_payload(protocol)
            if solvers != {"TrimSQD"}:
                raise ChallengeError(f"a shared TrimSQD protocol cannot run {sorted(solvers)} fragments")
            outside = sorted(set(ids) - {int(k) for k in shared_payload["targets"]})
            if outside:
                raise ChallengeError(f"fragment(s) {outside} are not targets of the shared protocol "
                                     f"{sorted(int(k) for k in shared_payload['targets'])}")
            for s in live:
                if s["live"]["sqd_config"] != shared_payload["sqd_config"] \
                        or s["live"]["trim_config"] != shared_payload["trim_config"]:
                    raise ChallengeError(
                        f"fragment {s['fragment_id']}: the live measurement spec carries solver settings "
                        "that differ from the shared protocol; rebuild the plan with the protocol's "
                        "settings (nothing is silently retuned)")
            sqd_config, trim_config = shared_payload["sqd_config"], shared_payload["trim_config"]

        uploads, remote = [], {}
        for s in specs:
            if s["mode"] != "replay":
                continue
            art = s["replay"]["artifact"]
            src = self._artifact_path(art)
            if not src.is_file():
                raise ChallengeError(f"fragment {s['fragment_id']}: counts artifact {art['name']} has disappeared")
            sha = s["replay"]["counts_sha256"]
            if mp.sha256_file(src) != sha:
                raise ChallengeError(
                    f"fragment {s['fragment_id']}: {art['name']} changed after the plan was made "
                    "(SHA-256 differs); it will not be replayed. Rebuild the plan.")
            remote[s["fragment_id"]] = f"{self._run_name(ligand)}__plan{plan['plan_digest'][:12]}_frag{s['fragment_id']}_{sha[:12]}.npy"
            uploads.append((src, remote[s["fragment_id"]]))

        will_sample = bool(live)
        if will_sample:
            if not allow_sampling:
                raise ChallengeError(
                    f"[{label}] fragment(s) {[s['fragment_id'] for s in live]} are planned LIVE, so this "
                    "call would sample the QPU, but live sampling was not confirmed "
                    "(confirm_live=True). Nothing was submitted.")
            self._check_qpu_budget(allow_second=allow_second_qpu)
        self._assert_bundle_deployed()

        kw: dict[str, Any] = dict(
            run_name=self._run_name(ligand), basis=BASIS,
            fci_threshold=FCI_THRESHOLD, sqd_threshold=SQD_THRESHOLD,
            qpu_backend=qpu_backend, n_shots=int(n_shots), n_reps=int(n_reps),
            solver_mode=_METHOD_OF[next(iter(solvers))],
            sqd_config=dict(sqd_config) if sqd_config else None,
            trim_config=dict(trim_config) if trim_config else None,
            reconstruct=bool(reconstruct),
            measurement_plan=mp.job_payload(specs, remote),
        )
        if shared_payload is not None:
            kw["shared_protocol"] = shared_payload
        if fragment_ids is not None:
            kw["fragment_ids"] = [int(i) for i in fragment_ids]
        if quantum_fragment_ids is not None:
            kw["quantum_fragment_ids"] = [int(i) for i in quantum_fragment_ids]
        if solver_assignments is not None:
            kw["solver_assignments"] = {str(int(k)): str(v) for k, v in solver_assignments.items()}
            kw["use_threshold_fallback"] = bool(partial_assignments)
        print(f"Submitting [{label}] {ligand}: fragments={sorted(ids)} "
              f"modes={ {s['fragment_id']: s['mode'] for s in specs} } backend={qpu_backend} "
              f"live QPU sampling={'YES' if will_sample else 'no'}")
        job = self._dispatch_stage3(ligand, kw, uploads)
        print(f"  job_id: {job.job_id}")
        if will_sample:
            self._record_qpu_sampling()
        return job

    # ── per-fragment jobs: submit / retrieve / status ───────────────────────────────────────────

    def _select_specs(self, plan: dict, fragments) -> list[dict]:
        if fragments is None:
            return list(plan["specs"])
        ids = [_fid(i) for i in fragments]
        if len(set(ids)) != len(ids):
            raise ChallengeError(f"duplicate fragment ids in fragments={ids}")
        by_id = {s["fragment_id"]: s for s in plan["specs"]}
        missing = sorted(set(ids) - set(by_id))
        if missing:
            raise ChallengeError(f"fragment(s) {missing} are not in the measurement plan {sorted(by_id)}")
        return [by_id[i] for i in ids]

    def _read_meas_record(self, path: Path) -> dict | None:
        if not path.is_file():
            return None
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ChallengeError(f"Unreadable measurement job record {path}: {exc}. Nothing was submitted.") from exc
        if not isinstance(rec, dict) or not rec.get("job_id") or rec.get("kind") != "measurement_job":
            raise ChallengeError(f"{path} is not a measurement job record. Nothing was submitted.")
        return rec

    def submit_measurement_plan(self, plan: dict, *, fragments=None, confirm_live: bool = False,
                                allow_second_qpu: bool = False, repeat: bool = False) -> dict:
        """Submit one single-fragment Stage 3 job per selected fragment; return ``{fragment_id: record}``.

        Each job follows ITS fragment's mode: a live fragment samples the QPU (needs
        ``confirm_live=True``; the QPU budget gate applies, so several live fragments need
        ``allow_second_qpu=True``, the organizer authorisation); a replay fragment uploads its
        pinned artifact and never samples.  The Serverless job id is written to
        ``checkpoints/measurements/jobs/`` and printed as soon as the gateway accepts the job.

        Resumable: a fragment with a job record is never submitted again -- a ``submitted`` one is
        only re-attached to, a ``collected`` one is skipped.  ``repeat=True`` deliberately
        re-acquires a collected live fragment; its previous record is kept, and the new counts are
        stored as another artifact.  Everything is validated before the first submission.
        """
        plan = self._validated_plan(plan)
        specs = self._select_specs(plan, fragments)
        ligand = plan["ligand"]
        todo, out = [], {}
        for s in specs:
            path = self._meas_job_path(ligand, s["fragment_id"], plan["plan_digest"])
            rec = self._read_meas_record(path)
            if rec is not None and rec["spec_digest"] != mp.spec_digest(s):
                raise ChallengeError(f"fragment {s['fragment_id']}: {path.name} belongs to a different spec; "
                                     "nothing was submitted")
            if rec is not None and rec["status"] == "collected" and repeat and s["mode"] == "live":
                rec = None                      # kept on disk; archived right before the new submission
            if rec is not None:
                out[s["fragment_id"]] = rec
                print(f"fragment {s['fragment_id']}: already {rec['status']} (job_id={rec['job_id']}) -- "
                      "not submitted again")
            else:
                todo.append((s, path))
        new_live = [s for s, _ in todo if s["mode"] == "live"]
        if new_live:
            if not confirm_live:
                raise ChallengeError(
                    f"fragment(s) {[s['fragment_id'] for s in new_live]} are planned LIVE; sampling the QPU "
                    "needs confirm_live=True. Nothing was submitted.")
            self._check_qpu_budget(allow_second=allow_second_qpu)
            if len(new_live) > 1 and not allow_second_qpu:
                raise ChallengeError(
                    f"{len(new_live)} live sampling jobs were requested; more than one needs the organizer "
                    "authorisation allow_second_qpu=True. Nothing was submitted.")
        for s, path in todo:                      # explicit replay specs are re-checked in the submission
            job = self._submit_planned_stage3(
                ligand, label=f"fragment {s['fragment_id']} {s['mode']}", plan=plan, specs=[s],
                reconstruct=False, fragment_ids=[s["fragment_id"]],
                solver_assignments={s["fragment_id"]: s["solver"]}, partial_assignments=True,
                allow_sampling=s["mode"] == "live", allow_second_qpu=allow_second_qpu)
            rec = {"kind": "measurement_job", "schema_version": "1.0", "ligand": ligand,
                   "fragment_id": s["fragment_id"], "plan_digest": plan["plan_digest"],
                   "spec_digest": mp.spec_digest(s), "mode": s["mode"], "solver": s["solver"],
                   "backend": s["backend"], "job_id": str(job.job_id), "status": "submitted",
                   "live_qpu_sampling": s["mode"] == "live", "submitted_utc": _utc()}
            old = self._read_meas_record(path)     # a collected record being re-acquired: keep it
            if old is not None:
                path.rename(path.with_name(path.name[:-len(".json")] + f".collected-{old['job_id']}.json"))
            self._write_json(path, rec)        # the only handle to a possibly QPU-consuming job
            self._fragment_jobs[rec["job_id"]] = job
            out[s["fragment_id"]] = rec
        return out

    def retrieve_measurements(self, plan: dict, *, fragments=None, poll_seconds: int = 30) -> dict:
        """Collect the jobs of :meth:`submit_measurement_plan`; return ``{fragment_id: summary}``.

        Per fragment: wait for the job, validate the result against the plan, and for a live
        fragment store the returned counts as a NEW write-once artifact with a provenance sidecar.
        Fragments that are already ``collected`` are skipped; a failed fragment does not stop the
        others, and the call raises at the end listing every failure.  A job that ended in ERROR
        has its record renamed ``*.failed-<job id>.json`` (a new submission is then possible); a
        refused result keeps its record (the QPU time is spent and its output is the evidence).
        """
        plan = self._validated_plan(plan)
        ligand, out, failures = plan["ligand"], {}, {}
        for s in self._select_specs(plan, fragments):
            fid = s["fragment_id"]
            path = self._meas_job_path(ligand, fid, plan["plan_digest"])
            rec = self._read_meas_record(path)
            if rec is None:
                failures[fid] = "no submitted job -- run submit_measurement_plan first"
                continue
            if rec["status"] == "collected":
                out[fid] = rec["summary"]
                print(f"fragment {fid}: already collected -- skipped")
                continue
            try:
                job = self._reconnect_fragment_job(rec, path)
                print(f"Retrieving fragment {fid} ({s['mode']}) job_id={rec['job_id']}")
                try:
                    result = self._wait_and_collect({f"fragment {fid}": job}, poll_seconds)[f"fragment {fid}"]
                except ChallengeError:
                    try:
                        bad = str(job.status()).upper() in _BAD_END
                    except Exception:
                        bad = False
                    if bad and path.is_file():
                        path.rename(path.with_name(path.name[:-len(".json")] + f".failed-{rec['job_id']}.json"))
                        self._fragment_jobs.pop(rec["job_id"], None)
                        print(f"  fragment {fid}: the job ended unsuccessfully; record kept as *.failed-"
                              f"{rec['job_id']}.json. Re-run submit_measurement_plan to resubmit it.")
                    raise
                summary = self._finish_measurement(plan, s, rec, result)
            except ChallengeError as exc:
                failures[fid] = str(exc)
                continue
            rec.update(status="collected", summary=summary, collected_utc=_utc())
            self._write_json(path, rec)
            self._fragment_jobs.pop(rec["job_id"], None)
            out[fid] = summary
        if failures:
            lines = "\n".join(f"  fragment {k}: {v}" for k, v in sorted(failures.items()))
            raise ChallengeError(
                f"Measurement retrieval incomplete ({len(failures)} of {len(out) + len(failures)} failed):\n"
                f"{lines}\nCollected fragments {sorted(out)} are saved and are skipped on the next call.")
        return out

    def _finish_measurement(self, plan: dict, spec: dict, rec: dict, result: dict) -> dict:
        """Validate one fragment's Stage 3 result against its spec, store counts, return the summary."""
        fid, ligand = spec["fragment_id"], plan["ligand"]
        self._validate_bundle_api(result, f"measurement fragment {fid}")
        if result.get("measurement_plan_api") != mp.MEASUREMENT_PLAN_API:
            raise ChallengeError(
                f"fragment {fid}: the result does not acknowledge measurement_plan_api="
                f"{mp.MEASUREMENT_PLAN_API}; the deployed Stage 3 function ignores measurement plans. "
                "Redeploy it (organizer). The job record is kept.")
        self._check_result_fragment(result, fid, f"measurement fragment {fid}")
        self._check_method_used(result, _METHOD_OF[spec["solver"]])
        m = result.get("measurement") or {}
        if m.get("mode") != spec["mode"] or m.get("backend") != spec["backend"] \
                or m.get("n_qubits") != spec["n_qubits"]:
            raise ChallengeError(f"fragment {fid}: the job reports measurement {m!r}, which is not the "
                                 f"planned {spec['mode']} on {spec['backend']} ({spec['n_qubits']} qubits)")
        if result.get("hamiltonian_sha256") != spec["hamiltonian_sha256"]:
            raise ChallengeError(f"fragment {fid}: the job solved a different embedded Hamiltonian than the plan")
        summary = {"fragment_id": fid, "mode": spec["mode"], "job_id": rec["job_id"],
                   "backend": spec["backend"], "qpu_job_id": result.get("qpu_job_id"),
                   "method_used": result.get("method_used"), "energy_ha": result.get("energy_ha"),
                   "sqd_energy_ha": result.get("sqd_energy_ha"),
                   "trimsqd_energy_ha": result.get("trimsqd_energy_ha")}
        if spec["mode"] == "replay":
            if m.get("counts_sha256") != spec["replay"]["counts_sha256"]:
                raise ChallengeError(f"fragment {fid}: the job consumed different counts than the planned artifact")
            summary.update(artifact=spec["replay"]["artifact"], counts_sha256=m["counts_sha256"])
        else:
            if not result.get("counts_b64"):
                raise ChallengeError(f"fragment {fid}: the live job returned no counts; nothing to store")
            data = gzip.decompress(base64.b64decode(result["counts_b64"]))
            sha = mp.sha256_bytes(data)
            if sha != m.get("counts_sha256"):
                raise ChallengeError(f"fragment {fid}: the returned counts do not match the job's own SHA-256")
            art = self._store_acquired(ligand, spec, rec, result, data, sha, plan["plan_digest"])
            summary.update(artifact=art, counts_sha256=sha)
        self._write_json(self.measurements_dir / "results" / f"{ligand}_frag{fid}_{rec['job_id']}.json",
                         {**summary, "plan_digest": plan["plan_digest"]})
        return summary

    def _store_acquired(self, ligand: str, spec: dict, rec: dict, result: dict, data: bytes, sha: str,
                        plan_digest: str) -> dict:
        """Write the live counts as a NEW artifact + provenance sidecar; never touch an existing file."""
        fid = spec["fragment_id"]
        tmp = self.measurements_dir / "incoming" / f"{ligand}_frag{fid}_{rec['job_id']}.npy"
        mp.write_new(tmp, data)
        try:
            mp.check_counts_file(tmp, spec["n_orb"])
        except mp.MeasurementPlanError as exc:
            raise ChallengeError(f"fragment {fid}: the returned counts are unusable ({exc}); "
                                 f"they were kept at {tmp}") from exc
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        name = f"{ligand}_frag{fid}_{stamp}_{rec['job_id']}.npy"
        target = self.counts_runs_dir / name
        mp.write_new(target, data)
        mp.write_sidecar_once(target, {
            "kind": mp.SIDECAR_KIND, "schema_version": mp.SCHEMA_VERSION, "ligand": ligand,
            "fragment_id": fid, "hamiltonian_sha256": spec["hamiltonian_sha256"],
            "n_bits": spec["n_qubits"], "counts_sha256": sha, "backend": spec["backend"],
            "job_id": result.get("qpu_job_id"), "serverless_job_id": rec["job_id"],
            "solver": spec["solver"], "n_shots_requested": spec["live"]["n_shots"],
            "n_reps": spec["live"]["n_reps"], "plan_digest": plan_digest, "stored_utc": _utc()})
        tmp.unlink()
        return {"name": name, "location": "run"}

    def plan_from_acquired(self, plan: dict) -> dict:
        """New sealed plan where every live fragment with a collected acquisition becomes a replay of
        that artifact (hash-pinned, with its sidecar provenance).  Other specs are unchanged, so a
        partially acquired plan stays ``mixed``."""
        plan = self._validated_plan(plan)
        _, sel, base = self._plan_context(plan["ligand"])
        specs = []
        for s in plan["specs"]:
            rec = self._read_meas_record(self._meas_job_path(plan["ligand"], s["fragment_id"], plan["plan_digest"]))
            if s["mode"] == "live" and rec is not None and rec["status"] == "collected":
                art = rec["summary"]["artifact"]
                s = mp.bind_acquired(s, artifact=art, path=self._artifact_path(art))
            specs.append(s)
        return mp.validate_plan(mp.new_plan(ligand=plan["ligand"], selection=sel, baseline=base, specs=specs),
                                selection=sel, baseline=base)

    def measurement_status(self, plan: dict | None = None, ligand: str | None = None) -> pd.DataFrame:
        """Table of job records (all of them, or those of ``plan``) with job ids and artifacts."""
        ligand = plan["ligand"] if plan else (ligand or SELECTED_LIGAND)
        jobs = self.measurements_dir / "jobs"
        rows = []
        if plan is not None:
            plan = self._validated_plan(plan)
            for s in plan["specs"]:
                rec = self._read_meas_record(self._meas_job_path(ligand, s["fragment_id"], plan["plan_digest"]))
                art = ((rec or {}).get("summary") or {}).get("artifact") or (s["replay"] or {}).get("artifact")
                rows.append({"fragment": s["fragment_id"], "mode": s["mode"],
                             "status": rec["status"] if rec else "not submitted",
                             "job_id": rec["job_id"] if rec else None,
                             "artifact": (art or {}).get("name")})
        elif jobs.is_dir():
            for p in sorted(jobs.glob(f"{ligand}_frag*.job.json")):
                rec = self._read_meas_record(p)
                rows.append({"fragment": rec["fragment_id"], "mode": rec["mode"], "status": rec["status"],
                             "job_id": rec["job_id"],
                             "artifact": ((rec.get("summary") or {}).get("artifact") or {}).get("name")})
        df = pd.DataFrame(rows, columns=["fragment", "mode", "status", "job_id", "artifact"])
        print(df.to_string(index=False) if len(df) else "No measurement jobs recorded.")
        return df

    def measurement_artifacts(self, ligand: str | None = None, fragment_id: int | None = None) -> pd.DataFrame:
        """Every counts artifact on disk (frozen, legacy checkpoint, run) with its SHA-256 and provenance."""
        ligand = ligand or SELECTED_LIGAND
        rows = []
        ids = [fragment_id] if fragment_id is not None else range(0, 200)
        for loc, d in (("frozen", self.frozen_counts_dir), ("checkpoint", self.ckpt_dir / "counts"),
                       ("run", self.counts_runs_dir)):
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.npy")):
                fid = next((i for i in ids if mp.names_fragment(p.name, i)), None)
                if fid is None:
                    continue
                side = mp.read_sidecar(p)
                rows.append({"fragment": fid, "location": loc, "name": p.name, "sha256": mp.sha256_file(p),
                             "backend": (side or {}).get("backend"), "job_id": (side or {}).get("job_id"),
                             "provenance": "sidecar" if side else "none"})
        return pd.DataFrame(rows, columns=["fragment", "location", "name", "sha256", "backend", "job_id", "provenance"])
