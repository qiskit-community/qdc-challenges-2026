"""The notebook's event client.

``connect(team, service, backend)`` opens a team's campaign and returns a
:class:`Campaign`, the one object the notebook talks to: the exercise
checks of Parts 1 and 2 (:mod:`tcfs.checks`), job registration and
scoring, frontier cards and the running score.

Until the event endpoint exists the grader runs in-process: a
:class:`~tcfs.grader_server.GraderServer` with the job anchor
:class:`~tcfs.anchors.RuntimeMetadataAnchor` (job metadata read from the
IBM service, never results), the team's tables (the pinned demo instance
or a handout) and a hash-chained ledger under ``./qdc_ledger/<team>/``.
The team side is :class:`~tcfs.grader_client.RemoteCampaign` over an
:class:`~tcfs.grader_client.InProcessTransport`, i.e. the event protocol
with the network replaced by a function call. Setting ``QDC_GRADER_URL``
selects the remote grader, whose transport is provided at the event.

``TCFS_RATE_LIMIT_MIN`` (default 20) sets the local registration pace; the
organizer QA run sets it to 0.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path

import numpy as np

from . import checks
from . import hardware as hw
from .instances import DensestKSubgraph

POINTS = {"part1": 20, "part2": 65, "part3": 15}
# Part 2 = Exercises 2a-2d (checks.EX2_POINTS) + the campaign, 55 (S_best + 0.1 E)
PART2_SPLIT = {"exercises": sum(checks.EX2_POINTS.values()),
               "campaign": POINTS["part2"] - sum(checks.EX2_POINTS.values())}
DEFAULT_BACKEND = "ibm_phoenix"
LEDGER_ROOT = "qdc_ledger"
DEMO_CACHE = hw.CALIBRATION_DIR / "demo_instance_106.json"
DEMO_INSTANCE_ID = "demo106"   # the id of the demo instance on the grading server
ALPHA = 0.10
USAGE_WAIT_S = 120.0          # how long submit_job waits for finalized usage
USAGE_POLL_S = 5.0
DEMO_LINE = ("Instance: the demo instance, for practice only; it is not an "
             "event instance. At the event, pass your handout file to "
             "event.connect(..., handout=\"<file>\").")
CONTROL_NOT_REGISTERED = ("the control job is not registered yet: run the cell "
                          "that registers it again when your next slot opens, "
                          "then run this cell again. No card was filed.")
CONTROL_MISSING = ("file_card needs the control job, a job registered with "
                   "campaign.submit_job: campaign.file_card(axis, hypothesis, "
                   "control=...). No card was filed.")
TREATMENT_NOT_REGISTERED = ("the treatment job is not registered yet; run this "
                            "cell again when your next slot opens. The card "
                            "stays open.")
NO_CARD = ("no card was filed, so there is nothing to close: file the card "
           "with campaign.file_card first.")


class NotRegistered(RuntimeError):
    """A job argument that is not a registered job of the team: a refusal
    report, None, an unknown id or number. ``Campaign.file_card`` raises it
    to stop the cell before its treatment job is submitted. Jupyter shows
    it as one line, without a traceback."""

    def _render_traceback_(self):                # IPython: one line
        return [f"NotRegistered: {self}"]
# QPU usage per job on ibm_phoenix, fitted to the eleven R1 jobs of 2026-09-30
# (hardware_runs/r1_jobs.json; usage is billed in whole seconds): 1 x 1,024
# shots 3 s; 2 x 32,768 shots 4 s; 4 x 32,768 and 12 x 8,192 shots 5 s;
# 24 x 4,096 shots 6 s. usage ~ 2.9 s + 14.2 us per execution + 0.083 s per
# circuit, for p = 1 circuits of the demo instance (5.9 us scheduled).
USAGE_FIT = {"per_job_s": 2.9, "per_execution_s": 14.2e-6,
             "per_circuit_s": 0.083, "reference_duration_s": 5.9e-6}


def job_usage_estimate(n_circuits: int, shots_per_circuit: int,
                       circuit_duration_s: float | None = None) -> float:
    """Estimated QPU seconds of one Sampler job on ibm_phoenix
    (``USAGE_FIT``). A circuit longer than the p = 1 reference adds its
    extra scheduled duration to every execution."""
    f = USAGE_FIT
    per_exec = f["per_execution_s"]
    if circuit_duration_s is not None:
        per_exec += max(0.0, float(circuit_duration_s) - f["reference_duration_s"])
    return float(f["per_job_s"] + per_exec * int(n_circuits) * int(shots_per_circuit)
                 + f["per_circuit_s"] * int(n_circuits))

# The paper's Table I parameters on the training instance, computed once with
# optimize.run_penalty_experiments(training_instance(), n_grid=61) (2026-09-30):
# - "ensemble" = experiments 1-2, the coherent <H> optimum (beta*, gamma*),
#   stored as (gamma, beta). The <H> landscape is flat to 1e-6 along gamma
#   near the optimum while Gamma(gamma) is steep, so gamma is the point of
#   that flat optimum where Gamma equals the paper's 11.5189 (<H> =
#   -0.3343367, maximum -0.3343364; the grid run itself gave gamma =
#   -0.0617324, Gamma 11.5224).
# - "single" = experiment 4, the CVaR_{1/Gamma_3}-optimal single basis
#   circuit (gamma, theta, beta). Its P(feasible) is 0.1312 against the
#   paper's 0.1325 (the authors' released notebook gives 0.1316); P(optimal)
#   0.0320 against 0.0317.
TABLE_I = {
    "ensemble": (-0.06167112138141628, 0.2135801448887747),
    "single": {"gamma": 1.4885289663507832, "theta": 4.977064747733083,
               "beta": 0.4353775206304279},
}


def training_instance() -> DensestKSubgraph:
    """The paper's 12-node, 3-regular instance (k = 4, 18 edges, lam = 4)."""
    return DensestKSubgraph.random_regular(n=12, d=3, k=4, seed=42)


# ------------------------------------------------------------ demo instance
def build_demo_instance(milp: bool = True, time_limit: float = 120.0) -> dict:
    """The pinned 106-qubit demo instance, built fresh: ``hw.instance_from_
    backend(hw.phoenix_snapshot_backend(), n=106, swap_layers=2, seed=0)``,
    OPT by ``hw.milp_optimum`` (skipped with ``milp=False``) and B by the
    seeded ``hw.baseline_B_mc``. Returns the cache record."""
    inst, layout = hw.instance_from_backend(hw.phoenix_snapshot_backend(),
                                            n=106, swap_layers=2, seed=0)
    inst = DensestKSubgraph(n=inst.n, k=inst.k, edges=inst.edges,
                            name="phoenix-demo-106")
    rec = {"what": "pinned 106-qubit demo instance of the notebook "
                   "(in-process grader tables)",
           "instance_id": DEMO_INSTANCE_ID,
           "source": "hw.instance_from_backend(hw.phoenix_snapshot_backend(), "
                     "n=106, swap_layers=2, seed=0)",
           "instance": hw.instance_to_dict(inst),
           "layout": hw.layout_to_dict(layout), "alpha": ALPHA,
           "baseline": hw.baseline_B_mc(inst, ALPHA, samples=200_000, seed=2026),
           "baseline_method": "hw.baseline_B_mc(inst, alpha=0.10, "
                              "samples=200000, seed=2026)"}
    if milp:
        m = hw.milp_optimum(inst, time_limit=time_limit)
        rec.update(opt=float(m["value"]), opt_proven=bool(m["proven"]),
                   opt_method="hw.milp_optimum (HiGHS)")
    return rec


def write_demo_cache(path=None) -> Path:
    """Write :func:`build_demo_instance` to ``calibration/
    demo_instance_106.json`` (organizer helper)."""
    path = DEMO_CACHE if path is None else Path(path)
    Path(path).write_text(json.dumps(build_demo_instance(), indent=1) + "\n")
    return Path(path)


def demo_instance(path=None):
    """``(inst, layout, opt, baseline)`` of the pinned demo instance, read
    from ``calibration/demo_instance_106.json``. ``inst.instance_id`` is the
    file's ``"instance_id"``, ``DEMO_INSTANCE_ID`` ("demo106") by default:
    the id under which the grading server holds the demo instance."""
    d = json.loads(Path(DEMO_CACHE if path is None else path).read_text())
    inst = hw.instance_from_dict(d["instance"],
                                 instance_id=d.get("instance_id", DEMO_INSTANCE_ID))
    return (inst, hw.layout_from_dict(d["layout"]),
            float(d["opt"]), float(d["baseline"]))


def _handout_tables(handout):
    """(inst, layout, opt, baseline) from a handout dict or JSON path
    (``organizer/generate_instances.py`` format; OPT and B from the private
    record when present, otherwise computed here)."""
    d = handout if isinstance(handout, dict) else json.loads(Path(handout).read_text())
    inst = hw.instance_from_dict(d["instance"])
    layout = hw.layout_from_dict(d["layout"])
    main = d.get("main") or {}
    opt = d.get("opt", main.get("OPT"))
    base = d.get("baseline", main.get("B"))
    if opt is None or base is None:
        print(f"Computing OPT (MILP) and B for the handout instance "
              f"(n = {inst.n}); this takes up to two minutes.")
        opt = hw.milp_optimum(inst, time_limit=120)["value"] if opt is None else opt
        base = hw.baseline_B_mc(inst, ALPHA) if base is None else base
    return inst, layout, float(opt), float(base)


# ----------------------------------------------------------------- connect
def connect(team: str, service=None, backend=None, handout=None,
            budget_s: float = 1800.0, *, anchor=None, ledger_root=None,
            rate_limit_min: float | None = None, verbose: bool = True):
    """Open the team's campaign and return a :class:`Campaign`.

    ``service`` is the ``QiskitRuntimeService`` (the job anchor reads job
    metadata through it), ``backend`` the assigned backend. ``handout``: a
    per-team handout (dict or JSON path); default the pinned demo instance.
    ``budget_s``: the campaign's QPU-time wallet. ``anchor`` replaces the
    runtime anchor (tests: ``MockAnchor()``, with ``service=None,
    backend=None``). The ledger lives in ``<ledger_root>/<team>/``
    (default ``./qdc_ledger/<team>/``), created on demand. The registration
    pace is ``rate_limit_min`` minutes, default ``TCFS_RATE_LIMIT_MIN`` or
    20. Prints three lines: backend and queue, budget, grader; without a
    handout a fourth line says that the campaign runs on the demo instance,
    which is for practice and is not an event instance."""
    from .anchors import RuntimeMetadataAnchor
    from .grader_client import InProcessTransport, RemoteCampaign
    from .grader_server import GraderServer

    if os.getenv("QDC_GRADER_URL"):
        raise NotImplementedError(
            "QDC_GRADER_URL is set, but the transport to the remote grader is "
            "provided at the event; unset QDC_GRADER_URL to use the in-process "
            "grader")
    team = str(team)
    backend_name = _backend_name(backend) or DEFAULT_BACKEND
    if handout is None:
        inst, layout, opt, base = demo_instance()
    else:
        inst, layout, opt, base = _handout_tables(handout)
    if rate_limit_min is None:
        rate_limit_min = float(os.getenv("TCFS_RATE_LIMIT_MIN", "20"))
    ledger_dir = Path(LEDGER_ROOT if ledger_root is None else ledger_root) / team
    ledger_dir.mkdir(parents=True, exist_ok=True)
    if anchor is None:
        if service is None:
            raise ValueError("connect needs service= (a QiskitRuntimeService) "
                             "to read job metadata, or an explicit anchor=")
        anchor = RuntimeMetadataAnchor(service)
    server = GraderServer({team: {"instance": hw.instance_to_dict(inst),
                                  "opt": opt, "baseline": base}},
                          backend_name, anchor=anchor, ledger_dir=ledger_dir,
                          rate_limit_min=rate_limit_min, wallet_s=budget_s)
    client = RemoteCampaign(InProcessTransport(server), team, inst, backend_name)
    camp = Campaign(team=team, server=server, client=client, instance=inst,
                    layout=layout, backend=backend, service=service,
                    ledger_dir=ledger_dir, backend_name=backend_name,
                    demo=handout is None)
    if verbose:
        camp._print_connect(budget_s, rate_limit_min)
    return camp


def _backend_name(backend) -> str | None:
    if backend is None:
        return None
    name = getattr(backend, "name", None)
    name = name() if callable(name) else name
    return None if name is None else str(name)


def _job_id(job) -> str:
    jid = getattr(job, "job_id", None)
    return str(jid() if callable(jid) else job)


def _probe_value(i: int) -> float:
    """A fixed, generic angle for parameter i (sorted by name): a circuit
    bound at these values identifies its parametrized structure."""
    return 0.1 + 0.6180339887498949 * (i + 1)


def pubs_fingerprint(pubs, shots=None) -> list | None:
    """What a ``sampler.run(pubs, shots=shots)`` call submits, per PUB:
    ``circuit`` (:func:`~tcfs.fingerprint.circuit_fingerprint` of the
    circuit, its free parameters bound at fixed probe values), ``values``
    (sha256 of the parameter names and values, rounded to 1e-9; None
    without parameters) and ``shots`` (the PUB's, else ``shots``). None if
    the PUBs cannot be read."""
    import hashlib

    try:
        from qiskit import QuantumCircuit
        from qiskit.primitives.containers.sampler_pub import SamplerPub

        from .fingerprint import circuit_fingerprint

        single = isinstance(pubs, (QuantumCircuit, SamplerPub)) or (
            isinstance(pubs, tuple) and pubs and isinstance(pubs[0], QuantumCircuit))
        out = []
        for item in ([pubs] if single else list(pubs)):
            pub = SamplerPub.coerce(tuple(item) if isinstance(item, list) else item)
            qc = pub.circuit
            params = sorted(qc.parameters, key=lambda p: p.name)
            values = None
            if params:
                qc = qc.assign_parameters({p: _probe_value(i)
                                           for i, p in enumerate(params)})
                arr = np.round(np.asarray(pub.parameter_values.as_array(params),
                                          dtype=float), 9) + 0.0
                h = hashlib.sha256(json.dumps([p.name for p in params]).encode())
                h.update(str(arr.shape).encode())
                h.update(arr.tobytes())
                values = h.hexdigest()
            n_shots = pub.shots if pub.shots is not None else shots
            out.append({"circuit": circuit_fingerprint(qc), "values": values,
                        "shots": None if n_shots is None else int(n_shots)})
        return out
    except Exception:                            # noqa: BLE001 - never block a run
        return None


def _same_submission(before: list, now: list) -> bool:
    """Two :func:`pubs_fingerprint` lists describe the same submission:
    same circuits and parameter values PUB by PUB, and the same shots
    wherever both calls give them (a call that leaves the shots out, only
    to get the job back, is not a change)."""
    if len(before) != len(now):
        return False
    for b, n in zip(before, now):
        if b.get("circuit") != n.get("circuit") or b.get("values") != n.get("values"):
            return False
        if None not in (b.get("shots"), n.get("shots")) and b["shots"] != n["shots"]:
            return False
    return True


def _report(d):
    from .nb import Report

    return Report(d)


FAILED_STATES = ("ERROR", "FAILED", "CANCELLED")


def _failed_status(job):
    """``(status, reason)`` when ``job`` ended in a terminal failure
    (``FAILED_STATES``; ``reason`` is the first line of
    ``job.error_message()``, cut at about 200 characters, or ""), else
    None. None as well when the status cannot be read."""
    try:
        status = job.status()
    except Exception:                            # noqa: BLE001 - network, fakes
        return None
    status = str(getattr(status, "name", status)).upper()
    if status not in FAILED_STATES:
        return None
    reason = ""
    get = getattr(job, "error_message", None)
    if callable(get):
        try:
            reason = get() or ""
        except Exception:                        # noqa: BLE001
            reason = ""
    lines = [ln.strip() for ln in str(reason).splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    if len(first) > 200:
        return status, first[:200].rstrip() + "..."
    return status, first.rstrip(".")


# ---------------------------------------------------------------- Campaign
class Campaign:
    """One team's campaign (created by :func:`connect`).

    Attributes: ``team``, ``instance`` (the 106-qubit campaign instance),
    ``layout`` (its ``HardwareLayout`` on the backend), ``training`` (the
    12-qubit instance of Part 1), ``backend``, ``service``, ``target`` (the
    backend's Target, or the pinned phoenix snapshot without a backend),
    ``ledger_dir``, ``server`` and ``client``."""

    def __init__(self, team, server, client, instance, layout, backend,
                 service, ledger_dir, backend_name, demo=False):
        self.team, self.server, self.client = team, server, client
        self.instance, self.layout = instance, layout
        self.backend, self.service = backend, service
        self.backend_name = backend_name
        self.demo = bool(demo)
        self.ledger_dir = Path(ledger_dir)
        self.training = training_instance()
        self._target = None
        self._jobs_in_memory: dict = {}

    # ------------------------------------------------------------ plumbing
    @property
    def target(self):
        if self._target is None:
            tgt = getattr(self.backend, "target", None)
            self._target = tgt if tgt is not None else \
                hw.phoenix_snapshot_backend().target
        return self._target

    @property
    def jobs_file(self) -> Path:
        """``<ledger>/jobs.json``: ``{tag: job_id}`` of ``run_once``."""
        return self.ledger_dir / "jobs.json"

    @property
    def job_records_file(self) -> Path:
        """``<ledger>/job_records.json``: ``{job_id: {tag, submitted,
        backend, pubs, options}}`` of the jobs ``run_once`` submitted
        (``pubs``: :func:`pubs_fingerprint` of what was submitted)."""
        return self.ledger_dir / "job_records.json"

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            d = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return {}
        return d if isinstance(d, dict) else {}

    def _write_json(self, path: Path, d: dict) -> None:
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1) + "\n")
        tmp.replace(path)

    def _load_jobs_file(self) -> dict:
        return {str(tag): str(v["job_id"] if isinstance(v, dict) else v)
                for tag, v in self._read_json(self.jobs_file).items()}

    def _print_connect(self, budget_s, rate_limit_min) -> None:
        if self.backend is None:
            print(f"Backend: {self.backend_name} (not connected; no queue "
                  "information).")
        else:
            try:
                pending = self.backend.status().pending_jobs
                queue = f"{pending} jobs in the queue"
            except Exception:                    # noqa: BLE001 - metadata only
                queue = "queue length unavailable"
            print(f"Backend: {self.backend_name}, {queue}.")
            self._warn_faulty()
        view = self.client.registered_jobs()
        left = view.get("qpu_remaining_s", budget_s)
        print(f"Campaign budget: {budget_s:.0f} s of QPU time, {left:.1f} s "
              f"left. A job of one 106-qubit circuit with 2^15 shots uses "
              f"about {job_usage_estimate(1, 2 ** 15):.1f} s.")
        pace = ("no registration limit" if not rate_limit_min else
                f"one registration per {rate_limit_min:g} min")
        print(f"Grader: in-process, ledger {self.ledger_dir}/, {pace}; team "
              f"{self.team}, instance n = {self.instance.n}, k = "
              f"{self.instance.k}.")
        if self.demo:
            print(DEMO_LINE)

    def _warn_faulty(self) -> None:
        try:
            bad_q, bad_e = hw.faulty_from_target(self.backend.target)
        except Exception:                        # noqa: BLE001
            return
        used = set(self.layout.physical)
        q = sorted(bad_q & used)
        e = sorted(x for x in bad_e if x[0] in used and x[1] in used)
        if q or e:
            print(f"Note: the backend reports faulty qubits {q} / couplers "
                  f"{e} inside your layout; tell the organizers.")

    # ------------------------------------------------------------ exercises
    def check(self, exercise: str, *answer):
        """Run the check of ``exercise``, print its PASS / FAIL line and
        record the result in the team's ledger. Part 1 ("1a" .. "1d") runs
        on the training instance: "1a" the basis circuit (or the function
        ``basis_circuit``), "1b" the function ``fourier_lcu``, "1c" the ISA
        circuit, "1d" ``(job, evaluate_counts, cvar)`` (the job's result is
        downloaded here, and the job is verified with the IBM Quantum
        service through the grader's job anchor: a job the service does not
        know fails). Part 2 ("2a" .. "2d") runs on the campaign instance:
        "2a" the function ``my_score``, "2b" ``(job_c, theta_c, stats)``,
        "2c" ``(job_s, thetas_sweep, scores_s)``, "2d" ``my_plan``; the jobs
        of 2b and 2c are verified in the same way and must be scored
        registrations of the team (until they are, nothing is recorded).

        ``check("1d", evaluate_counts, cvar)``, without the job, tests the
        two functions on reference data only, and ``check("2b", theta_c,
        stats)`` the angle only: they print PASS or FAIL and record no
        points. Returns the ``CheckResult``; never raises."""
        part2 = str(exercise).strip().lower() in checks.EX2_POINTS
        res = checks.check(exercise, self.instance if part2 else self.training,
                           *answer, target=None if part2 else self.target,
                           assigned_backend=self.backend_name,
                           verify_job=self.server.anchor.verify,
                           registered=self._registered, budget=self.budget)
        print(res.message)
        res.shown = True
        if res.exercise in checks.EX_POINTS and res.record:
            try:
                resp = self._record_check(res)
                if not resp.get("ok"):
                    print(f"(the result was not recorded: {resp.get('error')})")
            except Exception as exc:             # noqa: BLE001 - never raise
                print(f"(the result was not recorded: {exc})")
        return res

    def _record_check(self, res) -> dict:
        """Send a check result to the grader: the ``part1`` operation for
        Exercises 1a-1d, ``part2`` for 2a-2d (same payload)."""
        if res.exercise in checks.EX1_POINTS:
            return self.client.record_part1(res.exercise, res.ok, res.points,
                                            res.message)
        return self.client.transport.send({
            "op": "part2", "team": self.team, "exercise": str(res.exercise),
            "ok": bool(res.ok), "points": float(res.points),
            "message": None if res.message is None else str(res.message)[:1000]})

    # ------------------------------------------------------------ run once
    def run_once(self, tag: str, sampler, pubs, **run_kwargs):
        """Submit ``sampler.run(pubs, **run_kwargs)`` the first time ``tag``
        is used and remember ``{tag: job_id}`` in ``<ledger>/jobs.json``.
        With a known tag nothing is submitted: the job is fetched with
        ``service.job(job_id)`` (or, without a service, the job object kept
        in memory) - so re-running a cell spends neither QPU time nor a
        registration slot. ``forget(tag)`` drops the entry.

        A job is recognized by its tag only. The job record keeps a
        fingerprint of what was submitted (:func:`pubs_fingerprint`: per
        PUB the circuit, the parameter values and the shots); when a known
        tag is used with other PUBs, the earlier job is still returned and
        one line says that it ran different circuits, parameter values or
        shots (a re-transpilation after a recalibration can change the
        circuit slightly): use a new tag for new PUBs.

        A known tag whose job ended in a terminal failure (status ERROR,
        FAILED or CANCELLED) is submitted again: one line gives the status
        and the first line of ``job.error_message()``, the PUBs passed now
        are submitted as for a new tag, the tag then holds the new job, and
        the new job record lists the failed job ids under
        ``earlier_attempts``. A failed job was never registered, so it used
        no registration slot. When the status cannot be read, the known
        job is returned as before."""
        tag = str(tag)
        if not tag:
            raise ValueError("run_once needs a non-empty tag")
        known = self._load_jobs_file()
        if tag not in known:
            return self._submit_new(tag, sampler, pubs, run_kwargs)
        job_id = known[tag]
        if self.service is not None:
            job = self.service.job(job_id)
        elif tag in self._jobs_in_memory:
            job = self._jobs_in_memory[tag]
        else:
            raise RuntimeError(f"tag {tag!r} holds job {job_id}, but there "
                               "is no service to fetch it: connect with "
                               "service=..., or campaign.forget(tag)")
        failed = _failed_status(job)
        if failed is not None:
            status, reason = failed
            head = (f"job {job_id} (tag {tag}) ended with status {status}"
                    + (f": {reason}" if reason else ""))
            print(head + ("" if head.endswith(".") else ".")
                  + " Submitting the PUBs again under the same tag.")
            return self._submit_new(tag, sampler, pubs, run_kwargs,
                                    failed_id=job_id)
        rec = self._read_json(self.job_records_file).get(job_id)
        before = rec.get("pubs") if isinstance(rec, dict) else None
        now = pubs_fingerprint(pubs, run_kwargs.get("shots")) if before else None
        if before and now is not None and not _same_submission(before, now):
            print(f"reusing job {job_id} (tag {tag}), but that job ran "
                  "different circuits, parameter values or shots than the "
                  f"PUBs passed now: use a new tag, or "
                  f"campaign.forget(\"{tag}\") to submit again")
        else:
            print(f"reusing job {job_id} (tag {tag}); "
                  f"campaign.forget(\"{tag}\") to submit again")
        return job

    def _submit_new(self, tag: str, sampler, pubs, run_kwargs: dict,
                    failed_id: str | None = None):
        """``sampler.run(pubs, **run_kwargs)``; the tag then holds the new
        job (``jobs.json``) and its record is written (``job_records.json``;
        with ``failed_id``, the failed job it replaces is appended to
        ``earlier_attempts``)."""
        job = sampler.run(pubs, **run_kwargs)
        job_id = _job_id(job)
        known = self._load_jobs_file()
        known[tag] = job_id
        self._write_json(self.jobs_file, known)
        record = {"tag": tag,
                  "submitted": datetime.datetime.now().isoformat(timespec="seconds"),
                  "backend": self.backend_name,
                  "pubs": pubs_fingerprint(pubs, run_kwargs.get("shots"))}
        try:
            from .grader_client import options_dict

            record["options"] = options_dict(getattr(sampler, "options", None))
        except (TypeError, ValueError):
            record["options"] = None
        records = self._read_json(self.job_records_file)
        if failed_id is not None:
            old = records.get(failed_id)
            earlier = old.get("earlier_attempts") if isinstance(old, dict) else None
            record["earlier_attempts"] = list(earlier or []) + [failed_id]
        records[job_id] = record
        self._write_json(self.job_records_file, records)
        self._jobs_in_memory[tag] = job
        print(f"submitted {job_id} (tag {tag})")
        return job

    def forget(self, tag: str) -> None:
        """Remove ``tag`` from ``jobs.json`` (the next ``run_once`` with this
        tag submits a new job)."""
        known = self._load_jobs_file()
        job_id = known.pop(str(tag), None)
        self._jobs_in_memory.pop(str(tag), None)
        self._write_json(self.jobs_file, known)
        if job_id is None:
            print(f"tag {tag} is not known")
        else:
            print(f"forgot tag {tag} (job {job_id})")

    def _options_for(self, job_id: str):
        rec = self._read_json(self.job_records_file).get(str(job_id))
        return None if not isinstance(rec, dict) else rec.get("options")

    # ---------------------------------------------------------- campaign
    def _view(self) -> dict:
        view = self.client.registered_jobs()
        if not view.get("ok"):
            raise RuntimeError(f"grader: {view.get('error')}")
        return view

    def _registered(self, job_id: str):
        for row in self._view()["jobs"]:
            if row["job_id"] == job_id:
                return row
        return None

    @staticmethod
    def _expand_circuits(circuits) -> list:
        """One bound circuit per parameter set: circuits pass through; a PUB
        ``(circuit, parameter_values[, shots])`` expands to its parameter
        sets in C order (the order of ``BitArray.get_counts(loc)``)."""
        from qiskit import QuantumCircuit
        from qiskit.primitives.containers.sampler_pub import SamplerPub

        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        out = []
        for i, item in enumerate(circuits):
            if isinstance(item, QuantumCircuit):
                if item.parameters:
                    raise ValueError(
                        f"circuit {i} has unbound parameters "
                        f"{sorted(p.name for p in item.parameters)}: pass the "
                        "PUB (circuit, parameter_values) instead")
                out.append(item)
                continue
            pub = SamplerPub.coerce(tuple(item) if isinstance(item, list) else item)
            params = list(pub.circuit.parameters)
            if not params:
                out.append(pub.circuit)
                continue
            values = pub.parameter_values.as_array(params)
            for loc in (np.ndindex(pub.shape) if pub.shape else [()]):
                out.append(pub.circuit.assign_parameters(
                    dict(zip(params, np.asarray(values[loc], dtype=float)))))
        return out

    def submit_job(self, job, circuits, lcu=None, note: str = "", *,
                   counts_list=None, weights=None, options=None,
                   wait_s: float = USAGE_WAIT_S, poll_s: float = USAGE_POLL_S):
        """Register a finished job with the grader and print its score.

        ``job``: the runtime job (anything with ``job_id()`` and
        ``result()``); its result is downloaded unless ``counts_list`` is
        given. ``circuits``: the job's ISA circuits, one per PUB, or the PUBs
        themselves when a PUB carries several parameter sets (each set is
        one circuit of the declaration, in the order of the PUB's values).
        ``lcu``: the declaration. ``None`` declares a single circuit, or,
        for several circuits or parameter sets, one distribution that pools
        them all with equal weights. ``{"kind": "penalty", "gamma": g}``
        declares the Fourier-LCU ensemble of the penalty at ``g``: the
        circuits, or the parameter sets of the PUB, are the branches b = 0,
        ..., n in this order (theta_b = 2 pi b / (n+1)), and the grader
        recomputes the weights q_b = |c_b| / ||c||_1 itself; a subset of
        branches is declared with ``"branches": [b, ...]``, one index per
        circuit. ``note``: a short label. Execution options are taken from
        ``options``, else from the ``run_once`` record of the job. While
        the service has not finalized the job's usage the registration is
        retried every ``poll_s`` seconds for up to ``wait_s``.

        Prints one line: job number, id, Q, F, S (0.7 Q + 0.2 F; the E term
        enters at the close), CVaR, usage and budget left. Returns the
        score dict (with ``seq`` and ``job_id``), or the refusal. A job that
        is registered already is not registered again: its stored score is
        returned."""
        job_id = _job_id(job)
        known = self._registered(job_id)
        if known is not None and known["status"] in ("scored", "void"):
            if known["status"] == "scored":
                print("(registered earlier) " + self._score_line(known))
            else:
                print(f"job #{known['seq']} {job_id} was registered earlier and "
                      f"is void: {known.get('void_reason')}")
            return _report(known)
        from qiskit import QuantumCircuit

        items = [circuits] if isinstance(circuits, QuantumCircuit) else list(circuits)
        n_pubs = len(items)
        if counts_list is None:
            from .grader_client import RemoteCampaign

            try:
                result = job.result()
            except Exception as exc:             # noqa: BLE001 - failed job
                msg = " ".join(str(exc).split())[:200]
                print(f"Registration not attempted: the result of job {job_id} "
                      f"cannot be read ({type(exc).__name__}: {msg}).")
                return _report({"ok": False, "job_id": job_id,
                                "error": f"result unavailable: {msg}"})
            counts_list = RemoteCampaign.from_runtime_job(job, result=result,
                                                          expand=True)
            n_pubs = len(result)                 # the job's own PUB count
        bound = self._expand_circuits(items)
        if len(bound) != len(counts_list):
            raise ValueError(f"the job holds {len(counts_list)} parameter sets "
                             f"but {len(bound)} circuits were given: pass one "
                             "circuit per PUB, or the PUBs themselves")
        if options is None:
            options = self._options_for(job_id)
        kwargs = {"lcu": lcu, "weights": weights, "options": options,
                  "note": note or None, "n_pubs": min(n_pubs, len(bound))}
        if known is not None:                    # registered, not revealed
            resp = self.client.resume_job(bound, counts_list, job_id, **kwargs)
        else:
            deadline = time.time() + float(wait_s)
            while True:
                resp = self.client.submit_job(bound, counts_list, job_id, **kwargs)
                if resp.get("ok") or not resp.get("retry") \
                        or time.time() + float(poll_s) > deadline:
                    break
                time.sleep(float(poll_s))
        if not resp.get("ok"):
            print(f"Registration refused: {resp.get('error')}")
            return _report(resp)
        out = {**resp, "job_id": job_id, "note": note or None}
        print(self._score_line(out))
        return _report(out)

    @staticmethod
    def _score_line(r: dict) -> str:
        cv = r.get(f"CVaR_{ALPHA:.2f}")
        note = f" ({r['note']})" if r.get("note") else ""
        left = r.get("qpu_remaining_s")
        tail = "" if left is None else f"  budget left {left:.1f} s"
        return (f"job #{r['seq']} {r['job_id']}{note}: Q {r['Q']:.4f}  F "
                f"{r['F']:.4f}  S {r['S']:.4f}  CVaR {cv:.3f}  usage "
                f"{r['usage_s']:.1f} s{tail}")

    def _job_ref(self, ref) -> tuple[str, int]:
        """(job_id, seq) of a registered job of this team, given as the dict
        ``submit_job`` returned, a job number, a job id or the job itself.
        Anything else raises :class:`NotRegistered`: a refusal report,
        None, a dict with neither ``job_id`` nor ``seq``, an id or a number
        the grader does not know."""
        rows = self._view()["jobs"]
        key = ref
        if isinstance(ref, dict):
            key = ref.get("job_id") if ref.get("job_id") is not None else ref.get("seq")
        if key is None or isinstance(key, bool):
            raise NotRegistered("not a registered job")
        if isinstance(key, (int, np.integer)):
            for row in rows:
                if row["seq"] == int(key):
                    return row["job_id"], row["seq"]
            raise NotRegistered(f"no registered job #{key}")
        jid = _job_id(key)
        for row in rows:
            if row["job_id"] == jid:
                return jid, row["seq"]
        raise NotRegistered(f"job {jid} is not registered")

    # ---------------------------------------------------------------- cards
    def _filed_card(self, axis: str, control_job):
        """The team's card on ``axis`` with this control job, or None."""
        st = self.client.standing()
        for c in st.get("part3", {}).get("cards", []) if st.get("ok") else []:
            if c.get("axis") == axis and c.get("control_job") == control_job:
                return c
        return None

    def file_card(self, axis: str, hypothesis: str, control=None):
        """File a frontier card (Part 3) before the treatment job is
        submitted. The grader compares the filing time with the creation
        time of the treatment job: a card filed more than 120 s after the
        treatment job was submitted, or after it was registered, is not
        valid. ``control``: the registered control job, as the dict
        ``submit_job`` returned, a job number or a job id. Prints one line;
        returns the grader's answer (with ``card_id``). A card that was
        filed earlier on this axis with the same control job is returned as
        it is (re-running the cell files nothing new; the hypothesis is
        fixed at the first filing).

        If ``control`` is not a registered job of the team (for example the
        report of a registration refused by the 20-minute rule) nothing is
        filed and :class:`NotRegistered` is raised, which stops the cell
        before its treatment job is submitted."""
        if control is None:
            raise NotRegistered(CONTROL_MISSING)
        try:
            job_id, seq = self._job_ref(control)
        except NotRegistered:
            raise NotRegistered(CONTROL_NOT_REGISTERED) from None
        ctl = f"; control job #{seq} {job_id}"
        earlier = self._filed_card(axis, job_id)
        if earlier is not None:
            print(f"{earlier['card_id']} was filed earlier on axis {axis}{ctl}; "
                  "its hypothesis is fixed at filing.")
            return _report({"ok": True, "card_id": earlier["card_id"],
                            "axis": axis, "control_job": job_id,
                            "filed_earlier": True})
        resp = self.client.file_card(axis, hypothesis, control_job=job_id)
        if not resp.get("ok"):
            print(f"Card refused: {resp.get('error')}")
            return _report(resp)
        print(f"{resp['card_id']} filed on axis {axis} at {resp['t']}{ctl}.")
        for w in resp.get("warnings", []):
            print(f"Note: {w}")
        return _report({**resp, "axis": axis, "control_job": job_id})

    def close_card(self, card, treatment, result: str = ""):
        """Name the treatment job of ``card`` (the dict ``file_card``
        returned, or its id) and a result line, then print the card's
        verdict: both scores, the gain and whether the card is valid.
        Returns the verdict. If ``treatment`` is not a registered job of the
        team, nothing is written and one line says so: the card stays open
        and the cell can be run again when the next slot opens."""
        card_id = card.get("card_id") if isinstance(card, dict) else card
        if not card_id:
            print(NO_CARD)
            return _report({"ok": False, "error": NO_CARD})
        card_id = str(card_id)
        try:
            job_id, _ = self._job_ref(treatment)
        except NotRegistered:
            print(TREATMENT_NOT_REGISTERED)
            return _report({"ok": False, "card_id": card_id,
                            "error": TREATMENT_NOT_REGISTERED})
        fields = {"treatment_job": job_id}
        if result:
            fields["result"] = str(result)
        resp = self.client.update_card(card_id, **fields)
        if not resp.get("ok"):
            print(f"Card update refused: {resp.get('error')}")
            return _report(resp)
        st = self.client.standing()
        verdict = next((c for c in st.get("part3", {}).get("cards", [])
                        if c["card_id"] == card_id), {"card_id": card_id})
        if "gain" in verdict:
            line = (f"{card_id}: treatment S {verdict['S_treatment']:.4f} vs "
                    f"control S {verdict['S_control']:.4f}, gain "
                    f"{verdict['gain']:+.4f}; ")
        else:
            line = f"{card_id}: "
        if verdict.get("valid"):
            line += "the card is valid."
        else:
            line += "the card is not valid: " + "; ".join(
                verdict.get("reasons", ["unknown"])) + "."
        print(line)
        return _report(verdict)

    # ----------------------------------------------------------- the score
    def scoreboard(self):
        """Print all registered jobs (one row each) and the budget; returns
        the rows."""
        from .nb import ReportList

        view = self._view()
        rows = view["jobs"]
        if not rows:
            print("No registered jobs yet.")
        else:
            import pandas as pd

            cv = f"CVaR_{ALPHA:.2f}"
            cols = ["seq", "job_id", "note", "status", "Q", "F", "S", cv,
                    "usage_s"]
            df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
            with pd.option_context("display.float_format", "{:.4f}".format):
                print(df.to_string(index=False))
        wait = view.get("next_slot_in_s", 0.0)
        print(f"QPU time: {view['qpu_spent_s']:.1f} s of {view['wallet_s']:.0f} s "
              f"spent; next registration slot "
              f"{'now' if wait <= 0 else f'in {wait / 60:.1f} min'}.")
        return ReportList(rows)

    def budget(self) -> tuple[float, float, float]:
        """``(budget, spent, left)`` in QPU seconds."""
        view = self._view()
        return (float(view["wallet_s"]), float(view["qpu_spent_s"]),
                float(view["qpu_remaining_s"]))

    def summary(self):
        """Print and return the current points: Part 1 (recorded checks),
        Part 2 (recorded checks, and the campaign: best job and E), Part 3
        (cards, V, X and delta, the gain in S that earns the full X).
        Read-only; callable any number of times."""
        st = self.client.standing()
        if not st.get("ok"):
            print(f"Grader: {st.get('error')}")
            return _report(st)
        p1, p2, p3 = st["part1"], st["part2"], st["part3"]
        per = ", ".join(f"{ex} {v['points']:g}/{checks.EX1_POINTS[ex]}"
                        for ex, v in p1["per_exercise"].items())
        print(f"Part 1  {p1['points']:5.1f} / {POINTS['part1']}  ({per})")
        per2 = ", ".join(f"{ex} {v['points']:g}/{checks.EX2_POINTS[ex]}"
                         for ex, v in p2["per_exercise"].items())
        best = ("no scored job yet; points start with the first scored job"
                if p2["best_job"] is None else
                f"best job {p2['best_job']}: Q {p2['Q']:.4f}, F {p2['F']:.4f}")
        print(f"Part 2  {p2['points']:5.1f} / {POINTS['part2']}  ({per2}; "
              f"campaign {p2['campaign_points']:.1f} / {PART2_SPLIT['campaign']}: "
              f"{best}; E {p2['E']:.4f})")
        cards = ", ".join(f"{c['card_id']} {'valid' if c['valid'] else 'not valid'}"
                          for c in p3["cards"]) or "no cards"
        if not any(c["valid"] for c in p3["cards"]):
            cards += "; points start with the first valid card"
        print(f"Part 3  {p3['points']:5.1f} / {POINTS['part3']}  (V {p3['V']:.2f}, "
              f"X {p3['X']:.2f} with delta {p3['delta']:g}, E {p3['E']:.4f}; "
              f"{cards})")
        print(f"Total   {st['total']:5.1f} / {sum(POINTS.values())}   QPU time "
              f"{st['qpu_spent_s']:.1f} s of {st['wallet_s']:.0f} s")
        return _report({"part1": p1["points"], "part2": p2["points"],
                        "part2_exercises": p2["exercise_points"],
                        "part2_campaign": p2["campaign_points"],
                        "part3": p3["points"], "total": st["total"],
                        "qpu_spent_s": st["qpu_spent_s"], "standing": st})
