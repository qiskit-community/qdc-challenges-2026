"""Mock grader for the MVP notebook (challenge spec Secs. 3-5).

Part-1 checks are tolerance-banded numerical comparisons against reference
values the grader computes itself (in the real event these live server-side;
here they are computed lazily and cached in-process). Part-2 submissions are
executed on a local Aer simulator against a budget ledger, and scored with
the spec Sec. 4 metric

    Q = clip((CVaR_alpha(f_lam) - B) / (OPT - B), 0, 1),  alpha = 0.10
    F = weighted feasible fraction
    score_frac = 0.7 Q + 0.2 F + 0.1 E   (E = unspent budget at close)

The grader recomputes every objective value from raw bitstrings; nothing
self-reported enters the score. Independent-reimplementation of the
references (anti-collusion with the notebook code) is scheduled for the
red-team milestone; for the MVP the references come from ``tcfs``
itself with an independent-formula spot check where cheap (ex1a uses an
explicit DFT loop, not np.fft).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import diagonal_lcu as dlcu
from . import exact, schur_weyl as sw
from .cvar import cvar
from .instances import DensestKSubgraph

ALPHA_SCORE = 0.10
_REF_CACHE: dict = {}

# QPU-usage model (IBM docs, "Workload usage"): usage ~= a fixed per-job
# overhead + (circuit duration + reset + repetition delay) per execution.
# The docs' quick figure, 350 us per execution, assumes the 250 us default
# repetition delay of Eagle / Heron / the Nighthawk r1 template. The target,
# ``ibm_phoenix`` (Nighthawk r2), defaults to a **1 us** repetition delay
# (range 0-2 ms), so the per-execution time is the circuit itself: the p = 1
# 106-qubit basis circuit (2 swap layers, level-1 ISA, measurement included)
# schedules to 6.6 us on the phoenix Target snapshot of 2026-09-29
# (``calibration/phoenix_backend_facts.json``); reset is 0.4 us. A 2^15-shot
# single-circuit job is therefore ~ 2.3 s, not 13.5 s, and E is dominated by
# the job count, not the shot count. Tested against ``job.usage()`` on
# 2026-09-30 (eleven R1 jobs on ibm_phoenix, ``calibration/r1_phoenix.json``):
# the jobs were billed 3-6 whole seconds (48 s in total), 1.7 times this model
# in total and 1.5 to 2.2 times per job (the model has no per-circuit term).
# The measured fit is ``tcfs.event.USAGE_FIT`` (2.9 s per job + 14.2 us per
# execution + 0.083 s per circuit), which the notebook's estimates use. The
# constants below stay as the *mock* campaign's model (tests and the redteam
# scripts pin them), so ``USAGE_MODEL_CALIBRATED`` stays False. Live jobs are
# charged their real ``job.usage()`` by the anchor, never a model.
USAGE_OVERHEAD_S = 2.0
PHOENIX_DEFAULT_REP_DELAY_S = 1e-6
PHOENIX_RESET_S = 4e-7
PHOENIX_P1_CIRCUIT_S = 6.6e-6           # 106 q, 2 swap layers, level 1, incl. measure
USAGE_PER_EXEC_S = PHOENIX_P1_CIRCUIT_S + PHOENIX_RESET_S + PHOENIX_DEFAULT_REP_DELAY_S
USAGE_MODEL_CALIBRATED = False
USAGE_MODEL_SOURCE = ("phoenix Target snapshot 2026-09-29: circuit 6.6 us + reset "
                      "0.4 us + default rep_delay 1 us per execution; 2 s per job; "
                      "low by a factor 1.7 in total (1.5-2.2 per job) against "
                      "job.usage() of the eleven R1 jobs (2026-09-30); the "
                      "measured fit is tcfs.event.USAGE_FIT")


def per_exec_seconds(circuit_duration_s: float,
                     rep_delay_s: float = PHOENIX_DEFAULT_REP_DELAY_S,
                     reset_s: float = PHOENIX_RESET_S) -> float:
    """Seconds the QPU is busy per execution of one circuit: its scheduled
    duration (``circuit.estimate_duration(target)``, measurement included)
    plus the reset of ``init_qubits`` and the repetition delay."""
    return float(circuit_duration_s) + float(reset_s) + float(rep_delay_s)


def usage_estimate(n_circuits: int, shots_per_circuit: int,
                   per_exec_s: float = USAGE_PER_EXEC_S,
                   overhead_s: float = USAGE_OVERHEAD_S) -> float:
    """Estimated QPU seconds for one job of n_circuits x shots executions."""
    return overhead_s + per_exec_s * n_circuits * shots_per_circuit


def usage_estimate_isa(isa_circuits, shots_per_circuit: int, target,
                       rep_delay_s: float = PHOENIX_DEFAULT_REP_DELAY_S,
                       reset_s: float = PHOENIX_RESET_S,
                       overhead_s: float = USAGE_OVERHEAD_S) -> float:
    """Duration-based estimate for scheduled (ISA) circuits on ``target``:
    every circuit's own ``estimate_duration`` enters, so a deeper circuit
    costs more QPU time than a shallower one at the same shot count."""
    total = float(overhead_s)
    for qc in isa_circuits:
        d = float(qc.estimate_duration(target))
        total += per_exec_seconds(d, rep_delay_s, reset_s) * shots_per_circuit
    return total


def _ok(msg: str) -> tuple[bool, str]:
    return True, f"PASS - {msg}"


def _fail(msg: str) -> tuple[bool, str]:
    return False, f"FAIL - {msg}"


# ------------------------------------------------------------------- Part 1a
def grade_ex1a(inst: DensestKSubgraph, gamma_angle: float, c: np.ndarray,
               Gamma: float) -> tuple[bool, str]:
    """Check the DFT coefficients of the cardinality penalty at gamma_angle.

    Independent reference: explicit-loop DFT of e^{i*gamma*lam*(w-k)^2}.
    """
    n, lam, k = inst.n, inst.lam, inst.k
    m1 = n + 1
    ref = np.zeros(m1, dtype=complex)
    for j in range(m1):
        for w in range(m1):
            ref[j] += np.exp(1j * gamma_angle * lam * (w - k) ** 2) \
                * np.exp(-2j * np.pi * j * w / m1)
    ref /= m1
    c = np.asarray(c)
    if c.shape != ref.shape:
        return _fail(f"expected {m1} coefficients, got {c.shape}")
    if not np.allclose(c, ref, atol=1e-8):
        return _fail("coefficients do not match the DFT of the penalty phases")
    if abs(np.linalg.norm(c) - 1.0) > 1e-8:
        return _fail(f"Parseval violated: ||c||_2 = {np.linalg.norm(c):.6f}")
    if not np.isclose(Gamma, np.abs(c).sum() ** 2, rtol=1e-8):
        return _fail("Gamma != ||c||_1^2")
    if Gamma > m1 + 1e-9:
        return _fail(f"Gamma = {Gamma:.4f} exceeds the bound n+1 = {m1}")
    return _ok(f"coefficients valid, ||c||_2 = 1, Gamma = {Gamma:.4f} <= {m1}")


# ------------------------------------------------------------------- Part 1b
def _reference_rows(inst: DensestKSubgraph) -> dict:
    key = ("penalty", inst.name)
    if key not in _REF_CACHE:
        from .optimize import run_penalty_experiments

        rows, info = run_penalty_experiments(inst)
        _REF_CACHE[key] = {"rows": {r["experiment"][0]: r for r in rows},
                           "info": info}
    return _REF_CACHE[key]


# The paper's published Table I (rows 1, 2, 4) is a second valid reference:
# the protocol's landscape is multi-modal, so the recomputed reference and
# the paper can land on different-but-legitimate local optima (e.g. the
# Exp.-3 Gamma lineage: 12.9972 recomputed vs 12.5041 published). A
# submission passes a field if it falls inside the ENVELOPE of the two
# references, widened by the band — and only fields that are meaningful for
# a row are checked at all, one-sided-up (capped at the theoretical limit)
# for that row's own optimization objective.
PAPER_TABLE_I = {
    "1": {"exp_H": -0.3344, "P_feasible": 0.5898, "P_optimal": 0.0068},
    "2": {"exp_H": -16.9382, "Gamma": 11.5189, "CVaR": 2.2987,
          "P_feasible": 0.1592, "P_optimal": 0.0016},
    "4": {"exp_H": -24.2318, "Gamma": 12.5041, "CVaR": 3.3969,
          "P_feasible": 0.1325, "P_optimal": 0.0317},
}

# Third reference: the authors' own executed notebook outputs (their released
# repo, notebooks/experiments_dks_lcu_penalty.ipynb, seed-42 instance), at
# full precision. Their pooled sampler pairs |c_j| with theta_{N-j} (the
# conjugate-target decomposition; Gamma is identical), so row 2 differs from
# our pairing in the third decimal — the envelope must span both.
AUTHOR_TABLE_I = {
    "1": {"exp_H": -0.334337, "P_feasible": 0.589534, "P_optimal": 0.006815},
    "2": {"exp_H": -17.467421, "Gamma": 11.521114, "CVaR": 2.299064,
          "P_feasible": 0.157529, "P_optimal": 0.001650},
    "4": {"exp_H": -24.092795, "Gamma": 12.504190, "CVaR": 3.399969,
          "P_feasible": 0.131606, "P_optimal": 0.031987},
}

# (field, band, mode) per row; mode "up" = one-sided above the envelope
# floor, capped at the theoretical limit; "band" = two-sided envelope.
# Row 1 optimizes exp_H; rows 2 is an evaluation at row 1's optimum (all
# fields two-sided; exp_H loose since it inherits the optimum's branch);
# row 4 optimizes CVaR, so its exp_H is unconstrained garbage-guarded only,
# and its Gamma inherits the multi-modal Exp.-3 lineage (checked only for
# plausibility within [1, n+1]).
_ROW_POLICY = {
    "1": (("exp_H", 0.15, "up"), ("P_feasible", 0.03, "band"),
          ("P_optimal", 0.005, "up")),
    "2": (("exp_H", 0.50, "band"), ("Gamma", 0.05, "band"),
          ("CVaR", 0.15, "band"), ("P_feasible", 0.03, "band"),
          ("P_optimal", 0.01, "band")),
    "4": (("Gamma", None, "range"), ("CVaR", 0.15, "up"),
          ("P_feasible", 0.05, "band"), ("P_optimal", 0.01, "up")),
}


def _check_rows(inst: DensestKSubgraph, rows: dict) -> tuple[bool, str]:
    """Envelope check of experiment rows 1, 2, 4 (recomputed reference,
    the paper's Table I, or the authors' executed notebook; beating all
    three on a row's own objective passes up to the theoretical limit)."""
    ref = _reference_rows(inst)["rows"]
    opt = inst.brute_force()[0]
    flam_min = float(inst.penalized_table().min())
    for expno, policy in _ROW_POLICY.items():
        if expno not in rows:
            return _fail(f"missing experiment {expno}")
        sub, rr = rows[expno], ref[expno]
        pp, aa = PAPER_TABLE_I[expno], AUTHOR_TABLE_I[expno]
        for field, band, mode in policy:
            try:
                val = float(sub[field])
            except (KeyError, TypeError, ValueError):
                return _fail(f"exp {expno}: missing or non-numeric {field}")
            if not np.isfinite(val):
                return _fail(f"exp {expno}: {field} is not finite")
            refs = [rr[field], pp[field], aa[field]]
            if field.startswith("P_"):
                cap = 1.0
            elif field == "Gamma":
                cap = inst.n + 1.0
            else:
                cap = opt
            if mode == "range":  # row-4 Gamma: any legitimate LCU overhead
                lo, hi = 1.0, inst.n + 1.0 + 1e-9
            elif mode == "up":
                lo, hi = min(refs) - band, cap + 1e-9
            else:
                lo, hi = min(refs) - band, min(max(refs) + band, cap + 1e-9)
            if field == "exp_H" and mode != "up":
                lo = max(lo, flam_min)
            if not lo <= val <= hi:
                return _fail(f"exp {expno}: {field} = {val:.4f} outside "
                             f"band [{lo:.4f}, {hi:.4f}]")
        if "exp_H" in sub and not flam_min - 1e-9 <= float(sub["exp_H"]) \
                <= opt + 1e-9:
            return _fail(f"exp {expno}: exp_H = {sub['exp_H']} is not a "
                         f"possible expectation value of f_lambda")
    return _ok("experiments 1, 2, 4 within tolerance bands "
               "(reference or published values)")


def _rows_from_params(inst: DensestKSubgraph, params: dict) -> dict:
    """Recompute rows 1, 2, 4 from the submitted PARAMETERS with the
    grader's own engine - the table itself is never trusted."""
    from .optimize import _row, penalty_pooled_dist

    b1, g1 = (float(v) for v in params["opt_coherent"])
    d1 = np.abs(exact.coherent_penalty_state(inst, b1, g1)) ** 2
    p2, G2 = penalty_pooled_dist(inst, b1, g1)
    b4, g4, t4 = (float(v) for v in params["opt_basis"])
    G3 = float(params["Gamma3"])
    d4 = np.abs(exact.penalty_basis_state(inst, b4, g4, t4)) ** 2
    return {"1": _row("1", inst, d1),
            "2": _row("2", inst, p2, Gamma=G2, eta=1 / G2),
            "4": _row("4", inst, d4, Gamma=G3, eta=1 / G3)}


def grade_ex1b(inst: DensestKSubgraph, warm_start_fn, sampler_fn,
               params: dict) -> tuple[bool, str]:
    """Part 1b: the participant's warm start and ancilla-free sampler are
    exercised directly on instances the notebook never shows, and the
    Table-I rows are RECOMPUTED from the submitted optimizer parameters
    (``params`` = the ``info`` dict of the penalty protocol: keys
    ``opt_coherent`` (beta, gamma), ``opt_basis`` (beta, gamma, theta),
    ``Gamma3``). Nothing self-reported enters the grade.
    """
    # 1. warm start on two unseen instances (exact amplitudes)
    for n_, k_, seed_ in ((10, 3, 5), (8, 3, 8)):
        inst_ = DensestKSubgraph.random_regular(n_, 3, k_, seed=seed_)
        try:
            psi = np.asarray(warm_start_fn(inst_), dtype=float)
        except Exception as e:  # noqa: BLE001
            return _fail(f"warm start raised on n={n_}: {e!r}")
        want = exact.warm_start_state(inst_)
        if psi.shape != want.shape:
            return _fail(f"warm start: expected {want.shape}, got {psi.shape}")
        if not np.allclose(psi, want, atol=1e-10):
            return _fail(f"warm start amplitudes wrong on n={n_}, k={k_} "
                         "(hint: statevector index b encodes x_i = (b>>i)&1)")
    # 2. sampler on a synthetic 5-branch problem (statistical, seeded)
    rng = np.random.default_rng(2026)
    m, d = 5, 32
    dists = rng.dirichlet(np.ones(d) * 0.3, size=m)
    q = rng.dirichlet(np.ones(m))
    pooled = q @ dists
    shots = 100_000
    try:
        samples = np.asarray(sampler_fn(np.random.default_rng(1), q, dists,
                                        shots))
    except Exception as e:  # noqa: BLE001
        return _fail(f"sampler raised: {e!r}")
    if samples.shape != (shots,) or not np.issubdtype(samples.dtype,
                                                      np.integer):
        return _fail(f"sampler must return {shots} integer indices, got "
                     f"shape {samples.shape} dtype {samples.dtype}")
    if samples.min() < 0 or samples.max() >= d:
        return _fail("sampler returned an out-of-range index")
    emp = np.bincount(samples, minlength=d) / shots
    tv = 0.5 * np.abs(emp - pooled).sum()
    direct = rng.choice(d, size=shots, p=pooled)
    tv_direct = 0.5 * np.abs(np.bincount(direct, minlength=d) / shots
                             - pooled).sum()
    if tv > 1.5 * tv_direct + 2e-3:
        return _fail(f"sampler distribution is off: TV = {tv:.4f} vs "
                     f"{tv_direct:.4f} for direct sampling (branch weights "
                     "q_j ignored, or wrong per-branch distribution?)")
    # 3. the table, recomputed from the submitted parameters
    try:
        rows = _rows_from_params(inst, params)
    except (KeyError, TypeError, ValueError) as e:
        return _fail(f"params must carry opt_coherent, opt_basis, Gamma3: "
                     f"{e!r}")
    ok, msg = _check_rows(inst, rows)
    if not ok:
        return ok, msg
    return _ok("warm start exact on unseen instances; sampler matches the "
               "pooled distribution; " + msg[len("PASS - "):])


# ------------------------------------------------------------------- Part 1c
def grade_ex1c(cvar_fn) -> tuple[bool, str]:
    """Property-test a weighted CVaR implementation against this package's
    (itself validated against exact quantile enumeration in the test suite).
    """
    rng = np.random.default_rng(2026)
    for trial in range(200):
        m = rng.integers(2, 40)
        v = np.round(rng.normal(0, 5, m), 3)
        w = rng.exponential(1.0, m)
        alpha = float(rng.uniform(0.02, 1.0))
        want = cvar(v, w, alpha)
        got = float(cvar_fn(v, w, alpha))
        if not np.isclose(got, want, rtol=1e-9, atol=1e-9):
            return _fail(f"trial {trial}: cvar({alpha=:.3f}) = {got:.6f}, "
                         f"expected {want:.6f}")
    # boundary semantics
    if not np.isclose(float(cvar_fn([1.0, 2.0], [1.0, 1.0], 1.0)), 1.5):
        return _fail("alpha = 1 must recover the mean")
    if not np.isclose(float(cvar_fn([1.0, 2.0], [1.0, 3.0], 0.5)), 2.0):
        return _fail("upper tail must be filled by the best samples first")
    return _ok("weighted CVaR matches reference on 200 random cases")


# ------------------------------------------------------------------- Part 1d
def grade_ex1d(inst: DensestKSubgraph, a_beta_fn, params7, eta: float,
               reference_row: dict | None = None) -> tuple[bool, str]:
    """Part 1d: the participant's coefficient density is property-tested
    against the reference on fresh Haar samples - at n = 12 AND at an odd
    n (chi has period 4 pi there: the half-integer-m trap) - and the XY
    single-basis-circuit row is RECOMPUTED from the submitted angles
    ``params7`` = (gamma, theta, chi) (alpha = 0 at p = 1) at tail level
    ``eta`` (= 1 / Gamma_3 of the penalty track), then banded against the
    cross-comparable Table II row 7.

    The reference row is held inside the grader (the authors' executed
    notebook value, reproduced by our engine to < 2e-4); ``reference_row``
    is an organizer override for non-default instances.
    """
    rng = np.random.default_rng(1729)
    for n_ in (12, 7):
        al, th, ch = sw.haar_euler_samples(rng, 300)
        for beta_ in (0.37, 1.1, -0.6):
            try:
                got = np.asarray(a_beta_fn(n_, beta_, al, th, ch))
            except Exception as e:  # noqa: BLE001
                return _fail(f"a_beta raised at n={n_}: {e!r}")
            want = sw.a_beta(n_, beta_, al, th, ch)
            if got.shape != want.shape or not np.allclose(got, want,
                                                          atol=1e-9):
                hint = " (odd n: is chi treated with period 4*pi?)" \
                    if n_ % 2 else ""
                return _fail(f"a_beta wrong at n={n_}, beta={beta_}{hint}")
    try:
        g7, t7, c7 = (float(v) for v in params7)
    except (TypeError, ValueError):
        return _fail("params7 must be (gamma, theta, chi)")
    dist = np.abs(exact.xy_basis_state(inst, g7, 0.0, t7, c7)) ** 2
    met = exact.distribution_metrics(inst, dist)
    row = {"CVaR": cvar(inst.penalized_table(), dist, float(eta)),
           "P_feasible": met["P_feasible"], "P_optimal": met["P_optimal"]}
    ref = reference_row or {"CVaR": 3.400571, "P_feasible": 0.131174,
                            "P_optimal": 0.032035}
    opt = inst.brute_force()[0]
    for field, band in (("CVaR", 0.2), ("P_feasible", 0.05),
                        ("P_optimal", 0.02)):
        val = float(row[field])
        lo = ref[field] - band
        if field == "P_feasible":
            hi = ref[field] + band
        else:
            hi = (1.0 if field.startswith("P_") else opt) + 1e-9
        if not lo <= val <= hi:
            return _fail(f"recomputed {field} = {val:.4f} outside "
                         f"[{lo:.4f}, {hi:.4f}]")
    return _ok("a_beta matches the reference at n = 12 and n = 7; XY "
               "single-basis-circuit row (recomputed from your angles) "
               "within tolerance bands")


# --------------------------------------------------------------- Part 2 mock
# ---------------------------------------------------------- declared weights
MIN_SHOTS_PER_SUBMISSION = 1000   # the score is a distribution property


def declared_weights(inst: DensestKSubgraph, lcu: dict | None,
                     n_circuits: int) -> np.ndarray:
    """Branch weights the grader expects for a declared LCU construction.

    Spec Sec. 4: *for standard constructions the grader independently
    recomputes q_j from the declared LCU coefficients and rejects
    mismatches; for custom ansaetze, uniform weights over jobs.*

    ``lcu`` forms (weights returned normalised to sum 1):

    - ``None`` - custom ansatz: uniform weights over the submitted circuits.
    - ``{"kind": "penalty", "gamma": g, "branches": [j, ...]}`` - the DFT-LCU
      of the quadratic cardinality penalty (Part 1a): q_j = |c_j| / Gamma
      over the submitted DFT branch indices (default: all n+1, DFT order).
    - ``{"kind": "window", "gamma": g, "lam": lam_ind, "lo": lo, "hi": hi,
      "branches": [...]}`` - the indicator-window penalty (the Hamming-weight
      window variant, unscored kit capability since 2026-09-29).
    - ``{"kind": "xy", "beta": b, "beta0": b0, "alphas": [...],
      "thetas": [...], "chis": [...]}`` - the Schur-Weyl XY ensemble:
      circuits importance-sampled at ``beta0`` from |a_beta0| carry weights
      |a_beta(g)| / |a_beta0(g)| (uniform when beta == beta0; the paper's
      Exp.-3 re-weighting otherwise). ``beta0`` defaults to ``beta``.
    """
    if lcu is None:
        return np.full(n_circuits, 1.0 / n_circuits)
    kind = lcu.get("kind")
    if kind in ("penalty", "window"):
        g = float(lcu["gamma"])
        if kind == "penalty":
            phases = dlcu.cardinality_penalty_phases(g, inst.lam, inst.k, inst.n)
        else:
            phases = dlcu.indicator_window_phases(
                g, float(lcu["lam"]), int(lcu["lo"]), int(lcu["hi"]), inst.n)
        q = dlcu.branch_probabilities(dlcu.fourier_coefficients(phases))
        branches = np.asarray(lcu.get("branches", np.arange(inst.n + 1)),
                              dtype=int)
        if (len(branches) != n_circuits or branches.min() < 0
                or branches.max() > inst.n):
            raise ValueError(f"lcu['branches'] must list one DFT index in "
                             f"[0, {inst.n}] per submitted circuit")
        w = q[branches]
    elif kind == "xy":
        b = float(lcu["beta"])
        b0 = float(lcu.get("beta0", b))
        al, th, ch = (np.asarray(lcu[key], dtype=float)
                      for key in ("alphas", "thetas", "chis"))
        if not (len(al) == len(th) == len(ch) == n_circuits):
            raise ValueError("lcu alphas/thetas/chis must have one entry per "
                             "submitted circuit")
        w = np.abs(sw.a_beta(inst.n, b, al, th, ch))
        w = w / np.abs(sw.a_beta(inst.n, b0, al, th, ch))
    else:
        raise ValueError(f"unknown LCU kind {kind!r} (penalty | window | xy "
                         "| None for a custom ansatz)")
    if not np.all(np.isfinite(w)) or w.sum() <= 0:
        raise ValueError("declared LCU puts no weight on the submitted branches")
    return w / w.sum()


def _two_qubit_gate_count(qc) -> int:
    skip = {"measure", "barrier", "reset", "delay"}
    return sum(1 for ci in qc.data
               if ci.operation.num_qubits >= 2 and ci.operation.name not in skip)


class MockCampaign:
    """Grader-executed submissions against a QPU-time ledger (spec Sec. 5).

    The wallet is denominated in **QPU seconds** - the resource the event
    actually allocates (30 QPU-minutes per person) - and shots are derived
    from it through the usage model above (at the event the grader debits
    the job's real ``job.usage()``). MVP stand-in: executes submitted
    circuits on a local Aer backend (optionally noisy), recomputes
    everything from raw counts, debits the ledger *before* execution
    (consumed time stays consumed even if scoring fails - as on the real
    service for user-side failures), validates the qubit region, and
    returns the Sec. 4 score components. The part score is the best over
    submissions of the *combined* 0.7 Q + 0.2 F metric; E is the unspent
    QPU-time fraction at close. OPT and B stay private inside this object.

    Grader hardening (M3):

    - the ledger is append-only and hash-chained (:mod:`tcfs.ledger`);
    - declared weights are **recomputed** from the declared LCU
      (:func:`declared_weights`) and mismatches are rejected before any
      debit; custom ansaetze (``lcu=None``) get uniform weights;
    - every score is cross-checked against the authors' released metrics
      code when it is checked out (:mod:`tcfs.oracle`);
    - a scored submission needs at least ``MIN_SHOTS_PER_SUBMISSION``
      executions (the metric is a distribution property, not a lottery);
    - review flags (no two-qubit gates, near-deterministic feasibility) are
      recorded for the human-review channel - the metric cannot and does
      not try to block classical preparation (design doc Sec. 7);
    - ``window=(lo, hi)`` scores the Hamming-weight window variant (unscored
      kit capability since 2026-09-29): f - lam_ind * 1[weight outside the
      window] with F = in-window mass.

    The spec's rate limit (one scored submission per 20 minutes per team)
    is enforced on a *mock clock*: the rehearsal advances it with
    ``wait(minutes)``; at the event you simply wait. ``grant`` models the
    design-review stamp (budget top-up).
    """

    RATE_LIMIT_MINUTES = 20

    def __init__(self, inst: DensestKSubgraph, qpu_budget_s: float,
                 ledger_path: str | Path, seed: int = 11,
                 noise_model=None, alpha: float = ALPHA_SCORE,
                 per_exec_s: float = USAGE_PER_EXEC_S,
                 overhead_s: float = USAGE_OVERHEAD_S,
                 opt_value: float | None = None,
                 baseline: float | None = None,
                 window: tuple[int, int] | None = None,
                 lam_ind: float | None = None):
        from .ledger import Ledger

        self.inst = inst
        self.alpha = alpha
        self.budget = float(qpu_budget_s)          # QPU seconds
        self.per_exec_s, self.overhead_s = per_exec_s, overhead_s
        self.window = None if window is None else (int(window[0]), int(window[1]))
        if self.window is not None:
            lo, hi = self.window
            if not 0 <= lo <= hi <= inst.n:
                raise ValueError("window must satisfy 0 <= lo <= hi <= n")
            self.lam_ind = (1.0 + sum(abs(w) for _, _, w in inst.edges)
                            if lam_ind is None else float(lam_ind))
        else:
            self.lam_ind = None
        if opt_value is None or baseline is None:
            if inst.n > 20:
                raise ValueError("n > 20: pass opt_value (e.g. from "
                                 "tcfs.hardware.milp_optimum) and baseline "
                                 "(tcfs.hardware.baseline_B_mc) explicitly")
            if self.window is None:
                opt_value = inst.brute_force()[0] if opt_value is None else opt_value
                baseline = inst.baseline_B(alpha) if baseline is None else baseline
            else:
                f_ind = self._penalized(inst.objective_table(),
                                        inst.hamming_table())
                opt_value = float(f_ind.max()) if opt_value is None else opt_value
                baseline = (cvar(f_ind, inst.warm_start_probs(), alpha)
                            if baseline is None else baseline)
        self._opt, self._B = float(opt_value), float(baseline)
        from qiskit_aer import AerSimulator

        # local execution is statevector Aer (n <= 24); beyond that only
        # the QPU executes - score its counts with ``submit_counts``
        self._backend = AerSimulator(seed_simulator=seed,
                                     noise_model=noise_model)
        self._submissions = []  # (Q, F) per scored submission
        self._clock_min = 0.0          # mock wall clock (minutes)
        self._last_attempt_min = None
        self.ledger = Ledger(ledger_path)
        self._anchor = len(self.ledger)   # in-memory tail anchor

    # ------------------------------------------------------ objective
    def _penalized(self, f, ham):
        if self.window is None:
            return f - self.inst.lam * (ham - self.inst.k) ** 2
        lo, hi = self.window
        return f - self.lam_ind * ((ham < lo) | (ham > hi))

    def _feasible(self, ham):
        if self.window is None:
            return ham == self.inst.k
        lo, hi = self.window
        return (ham >= lo) & (ham <= hi)

    # mock clock / budget top-ups --------------------------------------
    def wait(self, minutes: float) -> None:
        """Advance the rehearsal clock (stand-in for real waiting)."""
        self._clock_min += float(minutes)

    def grant(self, qpu_seconds: float, reason: str) -> None:
        """Budget top-up in QPU seconds (design-review stamp)."""
        self.budget += float(qpu_seconds)
        self._append(kind="grant", grant_s=float(qpu_seconds), reason=reason)

    def usage_estimate(self, n_circuits: int, shots_per_circuit: int) -> float:
        return usage_estimate(n_circuits, shots_per_circuit,
                              self.per_exec_s, self.overhead_s)

    # ledger -------------------------------------------------------------
    def _append(self, **fields):
        entry = self.ledger.append(t_min=self._clock_min, **fields)
        self._anchor = entry["seq"] + 1
        return entry

    def _entries(self):
        return self.ledger.entries()

    def verify_ledger(self) -> int:
        """Re-walk the hash chain against the in-memory tail anchor."""
        return self.ledger.verify(expected_length=self._anchor)

    @property
    def spent(self) -> float:
        """QPU seconds consumed so far."""
        return float(sum(e.get("usage_s", 0.0) for e in self._entries()
                         if e.get("kind") == "debit"))

    @property
    def remaining(self) -> float:
        return self.budget - self.spent

    @property
    def shots_spent(self) -> int:
        return int(sum(e.get("shots", 0) for e in self._entries()
                       if e.get("kind") == "debit"))

    # validation ---------------------------------------------------------
    def _check_rate_limit(self):
        if self._last_attempt_min is not None:
            since = self._clock_min - self._last_attempt_min
            if since < self.RATE_LIMIT_MINUTES:
                left = self.RATE_LIMIT_MINUTES - since
                raise RuntimeError(
                    f"rate limit: 1 scored submission per "
                    f"{self.RATE_LIMIT_MINUTES} min - next allowed in "
                    f"{left:.0f} min (rehearsal: camp.wait({left:.0f}))")

    def _check_weights(self, weights, n_circuits, lcu):
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n_circuits or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("one nonnegative weight per circuit required")
        expected = declared_weights(self.inst, lcu, n_circuits)
        if not np.allclose(weights / weights.sum(), expected, rtol=1e-4,
                           atol=1e-6):
            what = ("a custom ansatz (lcu=None) must use uniform weights"
                    if lcu is None else
                    f"declared weights do not match the declared "
                    f"{lcu.get('kind')} LCU (grader recomputes q_j)")
            raise ValueError(f"declared-weight mismatch: {what}")
        return expected

    def _check_shots(self, n_circuits, shots_per_circuit):
        if shots_per_circuit * n_circuits < MIN_SHOTS_PER_SUBMISSION:
            raise ValueError(f"a scored submission needs at least "
                             f"{MIN_SHOTS_PER_SUBMISSION} executions in total "
                             "(the metric is a distribution property)")

    # scoring core -------------------------------------------------------
    def _score_counts(self, counts_list, weights, shots_per_circuit):
        from .hardware import bits_from_counts

        n = self.inst.n
        values, wts, feas_w = [], [], 0.0
        for counts, w in zip(counts_list, weights):
            bits, cnt = bits_from_counts(counts, n)
            f, ham = self.inst.evaluate_bits(bits)
            values.append(self._penalized(f, ham))
            sw_ = w * cnt / shots_per_circuit
            wts.append(sw_)
            feas_w += float(sw_[self._feasible(ham)].sum())
        values, wts = np.concatenate(values), np.concatenate(wts)
        cv = cvar(values, wts, self.alpha)
        Q = float(np.clip((cv - self._B) / (self._opt - self._B), 0, 1))
        F = float(feas_w / wts.sum())
        return cv, Q, F, self._oracle(counts_list, weights, shots_per_circuit,
                                      cv, F)

    def _oracle(self, counts_list, weights, shots_per_circuit, cv, F) -> str:
        from . import oracle

        if self.window is not None:
            return "n/a (window objective)"
        if not oracle.available():
            return "unavailable"
        pooled: dict = {}
        for counts, w in zip(counts_list, weights):
            for key, c in counts.items():
                key = key.replace(" ", "")
                pooled[key] = pooled.get(key, 0.0) + w * c / shots_per_circuit
        ref = oracle.author_metrics(self.inst, pooled, self.alpha)
        if (abs(ref["cvar"] - cv) > 1e-7 * max(1.0, abs(cv))
                or abs(ref["F"] - F) > 1e-9):
            raise RuntimeError(f"grader integrity: kit (CVaR {cv:.6f}, F {F:.6f})"
                               f" and author oracle (CVaR {ref['cvar']:.6f}, "
                               f"F {ref['F']:.6f}) disagree")
        return "agree"

    def _finish(self, debit, cv, Q, F, oracle_status, flags, cost):
        if F > 0.9:
            flags = list(flags) + ["near-deterministic feasibility (F > 0.9) "
                                   "-> review channel"]
        self._append(kind="score", ref=debit["seq"], cvar=cv, Q=Q, F=F,
                     oracle=oracle_status, review_flags=flags)
        self._submissions.append((Q, F))
        return {"CVaR_0.10": cv, "Q": Q, "F": F, "usage_s": cost,
                "qpu_spent_s": self.spent, "qpu_remaining_s": self.remaining,
                "clock_min": self._clock_min, "oracle": oracle_status,
                "review_flags": flags}

    # public scoring entry points ----------------------------------------
    def submit(self, circuits, weights, shots_per_circuit: int,
               lcu: dict | None = None) -> dict:
        """Score one submission: measured circuits + declared weights (+ the
        LCU they come from, see :func:`declared_weights`).

        Returns the public score components; raises before any debit if
        the submission is malformed, over budget, too soon or too small.
        """
        n = self.inst.n
        if n > 24:
            raise RuntimeError("n > 24: the mock grader cannot simulate this "
                               "circuit - run it on the QPU and score the "
                               "counts with submit_counts()")
        self._check_rate_limit()
        self._check_shots(len(circuits), shots_per_circuit)
        for i, qc in enumerate(circuits):  # spec Sec. 5: validate the region
            if qc.num_qubits != n or qc.num_clbits != n:
                raise ValueError(
                    f"circuit {i} must act on and measure exactly {n} "
                    f"qubits, got {qc.num_qubits}q/{qc.num_clbits}c")
        weights = self._check_weights(weights, len(circuits), lcu)
        cost = self.usage_estimate(len(circuits), shots_per_circuit)   # form first, then budget
        if cost > self.remaining + 1e-9:
            raise RuntimeError(f"QPU budget exceeded: this job needs "
                               f"~{cost:.1f} s, you have {self.remaining:.1f} s")
        flags = []
        if all(_two_qubit_gate_count(qc) == 0 for qc in circuits):
            flags.append("no two-qubit gates in any circuit (classical-"
                         "preparation pattern) -> review channel")

        # debit BEFORE execution: consumed time stays consumed even if
        # scoring raises (a 'score' entry follows on success)
        self._last_attempt_min = self._clock_min
        debit = self._append(kind="debit", usage_s=cost,
                             shots=int(shots_per_circuit * len(circuits)),
                             n_circuits=len(circuits), source="aer",
                             lcu=None if lcu is None else lcu.get("kind"))

        from qiskit.transpiler import generate_preset_pass_manager

        pm = generate_preset_pass_manager(optimization_level=1,
                                          backend=self._backend)
        counts_list = [self._backend.run(pm.run(qc), shots=shots_per_circuit)
                       .result().get_counts() for qc in circuits]
        cv, Q, F, oracle_status = self._score_counts(counts_list, weights,
                                                     shots_per_circuit)
        return self._finish(debit, cv, Q, F, oracle_status, flags, cost)

    def submit_counts(self, counts_list, weights, shots_per_circuit: int,
                      usage_s: float, lcu: dict | None = None,
                      job_id: str | None = None) -> dict:
        """Score counts obtained elsewhere (the QPU): same rate limit, same
        ledger, same weight rule, debited with the job's ACTUAL usage
        (``job.usage()``). At the event the grader retrieves the counts
        itself from ``job_id``; here they are supplied and flagged as such."""
        self._check_rate_limit()
        self._check_shots(len(counts_list), shots_per_circuit)
        weights = self._check_weights(weights, len(counts_list), lcu)
        cost = float(usage_s)
        if cost > self.remaining + 1e-9:
            raise RuntimeError(f"QPU budget exceeded: job used {cost:.1f} s, "
                               f"you had {self.remaining:.1f} s")
        flags = ["counts supplied client-side (event grader retrieves its "
                 "own from the job id)"] if job_id is None else []
        self._last_attempt_min = self._clock_min
        debit = self._append(kind="debit", usage_s=cost,
                             shots=int(shots_per_circuit * len(counts_list)),
                             n_circuits=len(counts_list), source="counts",
                             job_id=job_id,
                             lcu=None if lcu is None else lcu.get("kind"))
        cv, Q, F, oracle_status = self._score_counts(counts_list, weights,
                                                     shots_per_circuit)
        return self._finish(debit, cv, Q, F, oracle_status, flags, cost)

    def submit_job(self, job_id: str, weights, lcu: dict | None = None,
                   service=None, fetch=None, usage_tries: int = 30,
                   poll_s: float = 2.0) -> dict:
        """Grader-side retrieval (spec Sec. 5): the grader fetches the job
        itself, reads the counts and the job's ACTUAL usage, and scores -
        nothing about shots or usage is taken from the team. Refuses before
        any debit if the usage is not finalized yet (retry later).

        ``fetch(job_id) -> job`` overrides the service lookup (tests, or a
        pre-authenticated service); otherwise ``service.job`` is used.
        """
        self._check_rate_limit()
        if fetch is None:
            if service is None:
                from qiskit_ibm_runtime import QiskitRuntimeService
                service = QiskitRuntimeService()
            fetch = service.job
        job = fetch(job_id)
        counts_list = _counts_from_job(job)
        if len(counts_list) != len(weights):
            raise ValueError(f"job {job_id} holds {len(counts_list)} circuits, "
                             f"{len(weights)} weights declared")
        shots = [int(sum(c.values())) for c in counts_list]
        if len(set(shots)) != 1:
            raise ValueError("all circuits in a scored job must use the same "
                             "number of shots")
        shots_per_circuit = shots[0]
        self._check_shots(len(counts_list), shots_per_circuit)
        weights = self._check_weights(weights, len(counts_list), lcu)
        used = _poll_usage(job, usage_tries, poll_s)
        if used is None:
            raise RuntimeError(f"job {job_id}: usage not finalized by the "
                               "service yet - resubmit the job id later")
        cost = float(used)
        if cost > self.remaining + 1e-9:
            raise RuntimeError(f"QPU budget exceeded: job used {cost:.1f} s, "
                               f"you had {self.remaining:.1f} s")
        backend = getattr(job, "backend", None)
        backend = backend() if callable(backend) else backend
        self._last_attempt_min = self._clock_min
        debit = self._append(kind="debit", usage_s=cost,
                             shots=int(shots_per_circuit * len(counts_list)),
                             n_circuits=len(counts_list), source="job",
                             job_id=job_id,
                             backend=getattr(backend, "name", str(backend)),
                             lcu=None if lcu is None else lcu.get("kind"))
        cv, Q, F, oracle_status = self._score_counts(counts_list, weights,
                                                     shots_per_circuit)
        return self._finish(debit, cv, Q, F, oracle_status, [], cost)

    def close(self, part_points: float = 55.0) -> dict:
        """Leaderboard close: the part score is the best over scored
        submissions of 0.7 Q + 0.2 F, plus 0.1 E for the unspent QPU-time
        fraction, times ``part_points``: by default the campaign term of
        Part 2, 55 of its 65 points (``grader_server.PART2_CAMPAIGN_POINTS``;
        Exercises 2a-2d give the other 10). With no scored submission the
        part score is 0: E rewards frugal work, not no work."""
        E = max(0.0, self.remaining / self.budget)
        if self._submissions:
            Q, F = max(self._submissions, key=lambda qf: 0.7 * qf[0]
                       + 0.2 * qf[1])
            frac = 0.7 * Q + 0.2 * F + 0.1 * E
        else:
            Q, F, frac = 0.0, 0.0, 0.0
        out = {"Q": Q, "F": F, "E": E, "part_score": part_points * frac}
        self._append(kind="close", part_points=part_points, **out)
        return out


def _counts_from_job(job) -> list[dict]:
    """One counts dict per pub of a Sampler job result (first classical
    register of each pub)."""
    out = []
    for pub in job.result():
        data = pub.data
        fields = (list(data.keys()) if hasattr(data, "keys")
                  else list(vars(data).keys()))
        regs = [getattr(data, f) for f in fields
                if hasattr(getattr(data, f), "get_counts")]
        if not regs:
            raise ValueError("pub result without a classical register")
        out.append(regs[0].get_counts())
    return out


def _poll_usage(job, tries: int, poll_s: float):
    """``job.usage()`` reads 0/None until the service finalizes accounting."""
    for i in range(max(1, tries)):
        used = job.usage()
        if used:
            return float(used)
        if i + 1 < tries and poll_s > 0:
            time.sleep(poll_s)
    return None
