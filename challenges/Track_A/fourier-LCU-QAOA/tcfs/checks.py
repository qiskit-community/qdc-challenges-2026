"""The exercise checks of the notebook (Parts 1 and 2).

Pure functions, no network. Each check takes an instance (the training
instance in Part 1, the campaign instance in Part 2) and a participant's
answer and returns a :class:`CheckResult` whose message
starts with ``PASS`` or ``FAIL``. A check never raises on a wrong answer:
participant code that raises (a scaffold's ``NotImplementedError``
included) becomes a one-line FAIL that says where to see the traceback;
what the participant's code prints is held back, so the verdict is the
only line; and a wrong answer is compared with known wrong variants of the
reference so that the message names the mistake. A variant is named only
when it reproduces the answer on every test case, and when two explanations
reproduce it and the circuit does not tell them apart, the message gives
both.

- :func:`check_1a`  the basis circuit (paper Fig. 3) as a parametrized
  ``QuantumCircuit`` that measures the n qubits into one n-bit register:
  output distribution against ``exact`` at three fixed random parameter
  sets, then the two-qubit structure (every interaction on an edge of the
  graph, every edge used).
- :func:`check_1b`  the Fourier LCU of the penalty phase: angles (modulo
  2 pi), reconstruction, weights, sampling overhead, on the training
  instance at three values of gamma and on a second instance.
- :func:`check_1c`  the ISA circuit: target validity, one n-bit register,
  two-qubit gate count against ``CZ_BAR``, then equivalence on the measured
  qubits (simulated on the qubits connected to them).
- :func:`check_1d`  ``evaluate_counts`` and ``cvar`` on reference data (the
  training instance, and a 7-node instance with lam = 5 and an edge of
  weight 2) and on the counts of the participant's job; the job's shape;
  with a job anchor, the service's record of the job.
  :func:`check_1d_functions` runs the reference-data part alone.

Part 2 (the campaign instance; ``EX2_POINTS``):

- :func:`check_2a`  ``my_score(counts, alpha=0.10)`` against
  ``hw.metrics_from_counts`` on three seeded reference counts dicts of the
  instance, at alpha = 0.10 and 0.25.
- :func:`check_2b`  the re-centred angle ``theta_c`` (its ideal mean
  weight is k - shift) and its job: one circuit, at least 1,000 shots, a
  measured mean weight closer to k than the baseline's, registered.
  :func:`check_2b_angle` runs the angle part alone.
- :func:`check_2c`  the theta sweep: one PUB of at least four settings,
  the values of theta, the participant's scores of every setting against
  the kit's, registered.
- :func:`check_2d`  the QPU-time and slot plan: rows as ``nb.plan`` reads
  them, every job registrable, within the budget that is left.

A job of Exercises 2b and 2c must be a scored registration of the team
(``registered``, a lookup the caller provides); until it is, the check
records nothing.

The diagnoses are tables (``VARIANTS_1A``, ``_PHASES_1B``,
``_value_diagnoses``, ``_CVAR_VARIANTS``, ``_cvar_variants_2a``,
``_f_variants_2a``): a new case is one entry.
"""

from __future__ import annotations

import contextlib
import functools
import io
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression

from . import exact
from .cvar import cvar as _kit_cvar
from .instances import DensestKSubgraph

CZ_BAR = 60                  # measured on ibm_phoenix 2026-09-30: 51-57 at level 3
EX1_POINTS = {"1a": 5, "1b": 5, "1c": 4, "1d": 6}
EX2_POINTS = {"2a": 2, "2b": 3, "2c": 3, "2d": 2}
EX_POINTS = {**EX1_POINTS, **EX2_POINTS}
ALPHA = 0.10
MIN_SHOTS_1D = 1000
MAX_SIM_QUBITS = 22          # widest reduced ISA circuit the 1c check simulates
_NOT_GATES = ("barrier", "delay")
_PARAM_NAMES = ("gamma", "theta", "beta")
_TERMINAL = ("ERROR", "CANCELLED", "FAILED")

# fixed "random" parameter sets (gamma, theta, beta) of the circuit checks
_PARAMS_1A = ((0.7312, 2.4172, -0.3791), (-1.2653, 5.0213, 0.6437),
              (1.9387, 0.9121, 1.1829))
_PARAMS_1C = ((0.5419, 1.3377, -0.8123), (-2.0714, 4.4486, 0.2961))
_GAMMAS_1B = (0.37, 0.9, 1.63)
_GAMMA_1B_SECOND = 0.9       # gamma of the 1b case on the second instance
TOL_1A = 1e-9
TOL_1C = 1e-6
TOL_VARIANT = 1e-7
TOL_1B = 1e-5                # float32 / complex64 answers round at ~1e-7
RTOL_GAMMA_1B = 1e-6
_CVAR_CONFIRM = 5            # further cases a named cvar mistake must match
# Part 2
ALPHA_2A = 0.25              # the second alpha of the 2a check
SIZE_2A = 1500               # distinct bitstrings per reference counts dict
TOL_CVAR_2 = 1e-6            # |CVaR - kit| of 2a and 2c
TOL_F_2 = 1e-9               # |F - kit| of 2a and 2c
TOL_WEIGHT_2B = 0.05         # |ideal <W>(theta_c) - (k - shift)|
MIN_SHOTS_2 = 1000           # per parameter set, Exercises 2b and 2c
MIN_SETTINGS_2C = 4          # values of theta in the sweep (the exercise asks for 8)
BUDGET_S = 1800.0            # the campaign budget when 2d is given none
SLOT_MIN = 20                # one registration per 20 minutes at the event

_FIX_1A = "build the circuit as QuantumCircuit(inst.n) and end it with qc.measure_all()"
_FIX_1C = ("call qc.measure_all() on the basis circuit before transpiling, not "
           "on the ISA circuit")

# a 7-node instance without symmetry (bit-order mistakes change its values),
# with lam = 5 and an edge of weight 2 (code that fixes lam = 4 or ignores
# the weights fails on it); the second instance of 1b and 1d
ASYMMETRIC_INSTANCE = DensestKSubgraph(
    n=7, k=3, edges=((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0),
                     (1, 2, 2.0), (2, 3, 1.0), (4, 5, 1.0)), name="asymmetric-7")


# ------------------------------------------------------------------ result
@dataclass(repr=False)
class CheckResult:
    """Outcome of one exercise check: ``ok``, ``points`` (all or nothing),
    ``message`` (starts with PASS or FAIL) and the ``exercise`` name.
    ``record`` is False for results that carry no points and are not
    recorded: the functions-only check of Exercise 1d, the angle-only check
    of Exercise 2b, a job that is not a scored registration yet (2b, 2c)
    and a call with the wrong arguments."""

    ok: bool
    points: float
    message: str
    exercise: str = ""
    shown: bool = field(default=False, compare=False)
    record: bool = field(default=True, compare=False)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return (f"CheckResult(ok={self.ok}, points={self.points:g}, "
                f"exercise={self.exercise!r})")

    def _ipython_display_(self):
        # a result already printed by Campaign.check is not shown twice
        if not self.shown:
            print(self.message)


def _result(ex: str, ok: bool, text: str) -> CheckResult:
    top = EX_POINTS[ex]
    pts = float(top if ok else 0)
    head = "PASS" if ok else "FAIL"
    return CheckResult(bool(ok), pts,
                       f"{head} - Exercise {ex} ({pts:g}/{top} points): {text}",
                       ex)


CALL_1D = 'campaign.check("1d", job, evaluate_counts, cvar)'
CALL_1D_FUNCTIONS = 'campaign.check("1d", evaluate_counts, cvar)'
CALL_2A = 'campaign.check("2a", my_score)'
CALL_2B = 'campaign.check("2b", job_c, theta_c, stats)'
CALL_2B_ANGLE = 'campaign.check("2b", theta_c, stats)'
CALL_2C = 'campaign.check("2c", job_s, thetas_sweep, scores_s)'
CALL_2D = 'campaign.check("2d", my_plan)'
_CALLS = {"1a": 'campaign.check("1a", basis_circuit)',
          "1b": 'campaign.check("1b", fourier_lcu)',
          "1c": 'campaign.check("1c", isa)', "1d": CALL_1D,
          "2a": CALL_2A, "2b": CALL_2B, "2c": CALL_2C, "2d": CALL_2D}


def _preview(ex: str, what: str, ok: bool, text: str) -> CheckResult:
    """Result of a partial check that carries no points and is not
    recorded (Exercise 1d, functions only; Exercise 2b, angle only)."""
    head = "PASS" if ok else "FAIL"
    return CheckResult(bool(ok), 0.0, f"{head} - Exercise {ex}, {what} "
                       f"(no points recorded): {text}", ex, record=False)


def _functions_result(ok: bool, text: str) -> CheckResult:
    """Result of the functions-only check of Exercise 1d (no points, not
    recorded)."""
    return _preview("1d", "functions only", ok, text)


def _usage(ex: str, text: str) -> CheckResult:
    """FAIL for a call with the wrong arguments (nothing checked, nothing
    recorded)."""
    return CheckResult(False, 0.0, f"FAIL - Exercise {ex}: {text}", ex,
                       record=False)


class _Fail(Exception):
    """A FAIL with a participant-facing message (never leaves this module)."""

    def __init__(self, text: str):
        super().__init__(text)
        self.text = text


class _NotRecorded(_Fail):
    """A FAIL that records nothing: the job of Exercise 2b or 2c is not a
    scored registration of the team (yet)."""


def _one_line(exc: BaseException, limit: int = 200) -> str:
    text = " ".join(str(exc).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _call(label: str, fn, *args, call: str | None = None):
    """Call participant code with its printed output held back; an
    exception becomes a one-line FAIL that says where to see the full
    traceback (``call``: the call to run in a cell of its own)."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args)
    except NotImplementedError:
        raise _Fail(f"{label} is not implemented yet (it raises "
                    "NotImplementedError).") from None
    except Exception as exc:                     # noqa: BLE001 - participant code
        raise _Fail(f"{label} raised {type(exc).__name__}: "
                    f"{_one_line(exc) or 'no message'}; call {call or label} in "
                    "a cell of its own to see the full traceback.") from None


def _guarded(ex: str, fail=None):
    """Turn a ``_Fail`` (or any exception) raised by the check into a FAIL
    result built by ``fail(text)`` (default: the exercise's points line);
    a ``_NotRecorded`` becomes a FAIL line that records nothing. Warnings
    raised while checking are not shown."""
    fail = fail or (lambda text: _result(ex, False, text))

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    return fn(*args, **kwargs)
                except _NotRecorded as f:
                    return _usage(ex, f.text)
                except _Fail as f:
                    return fail(f.text)
                except Exception as exc:         # noqa: BLE001 - never raise
                    return fail("the answer could not be evaluated "
                                f"({type(exc).__name__}: {_one_line(exc)}).")
        return wrapper
    return deco


def _fmt(values) -> str:
    return "(" + ", ".join(f"{v:g}" for v in values) + ")"


# ---------------------------------------------------- circuit utilities
def _named_parameters(circuit: QuantumCircuit) -> dict:
    return {p.name: p for p in circuit.parameters}


def _register_problem(circuit: QuantumCircuit, n: int, fix: str) -> str | None:
    """A message unless the circuit has exactly one classical register, of
    n bits, and no classical bit outside it."""
    regs = list(circuit.cregs)
    in_regs = sum(r.size for r in regs)
    if len(regs) == 1 and regs[0].size == n and circuit.num_clbits == n:
        return None
    parts = [f"{r.name} ({r.size} bits)" for r in regs]
    loose = circuit.num_clbits - in_regs
    if loose > 0:
        parts.append(f"{loose} classical bit{'s' if loose > 1 else ''} outside "
                     "any register")
    desc = ("the classical register" + ("s " if len(regs) > 1 else " ")
            if regs else "") + " and ".join(parts)
    return (f"the circuit has {desc}; measure the n = {n} qubits into one "
            f"register of {n} bits: {fix}.")


def _measurement_positions(circuit: QuantumCircuit, n: int, fix: str = _FIX_1A):
    """``({qubit: bit position}, problem)``: which output bit each measured
    qubit lands in, for a circuit that measures n qubits into one n-bit
    classical register (``measure_all`` on ``QuantumCircuit(n)``)."""
    qmap = {}
    for ci in circuit.data:
        if ci.operation.name != "measure":
            continue
        q = circuit.find_bit(ci.qubits[0]).index
        if q in qmap:
            return None, f"qubit {q} is measured twice; measure every qubit once."
        qmap[q] = ci.clbits[0]
    if not qmap:
        return None, f"the circuit has no measurements: {fix}."
    problem = _register_problem(circuit, n, fix)
    if problem:
        return None, problem
    clbits = list(qmap.values())
    if len(set(clbits)) != len(clbits):
        return None, ("two qubits are measured into the same classical bit; "
                      "use qc.measure_all().")
    if len(qmap) != n:
        return None, (f"the circuit measures {len(qmap)} qubits; it must "
                      f"measure the n = {n} instance qubits: {fix}.")
    return {q: circuit.find_bit(c).registers[0][1] for q, c in qmap.items()}, None


def _to_positions(probs: np.ndarray, sub_index: dict, pos: dict, n: int):
    """Marginal of ``probs`` (over the simulated qubits, little-endian) on
    the measured qubits, re-indexed so that bit pos[q] is qubit q."""
    idx = np.arange(len(probs))
    out = np.zeros(len(probs), dtype=np.int64)
    for q, p in pos.items():
        out |= ((idx >> sub_index[q]) & 1) << p
    return np.bincount(out, weights=probs, minlength=1 << n)


def _bound_distribution(circuit, params: dict, values, pos: dict, n: int):
    """Measured-bit distribution of an n-qubit parametrized circuit at
    ``values`` = (gamma, theta, beta)."""
    from qiskit.quantum_info import Statevector

    g, t, b = values
    binding = {params[name]: v for name, v in zip(_PARAM_NAMES, (g, t, b))
               if name in params}
    bound = _flatten(circuit.assign_parameters(binding))
    bare = bound.remove_final_measurements(inplace=False)
    try:
        probs = Statevector(bare).probabilities()
    except Exception as exc:                     # noqa: BLE001 - participant circuit
        raise _Fail("the circuit cannot be simulated as a unitary followed by "
                    "measurements (measure only at the end, no resets or "
                    f"classical conditions): {_one_line(exc)}") from None
    return _to_positions(probs, {q: q for q in range(n)}, pos, n)


def _reference_1a(inst, values) -> np.ndarray:
    g, t, b = values
    return np.abs(exact.penalty_basis_state(inst, b, g, t)) ** 2


def _wide_gates(qc: QuantumCircuit) -> list:
    return [ci for ci in qc.data if len(ci.qubits) > 2
            and ci.operation.name not in _NOT_GATES + ("measure",)]


def _flatten(circuit: QuantumCircuit, max_levels: int = 20) -> QuantumCircuit:
    """Decompose composite gates on more than two qubits (e.g. a
    ``PauliEvolutionGate`` of the cost operator, whose Z terms commute, so
    its synthesis is exact, wrapped in custom gates to any depth) until no
    gate acts on more than two qubits or nothing decomposes further."""
    qc = circuit
    for _ in range(max_levels):
        wide = Counter(ci.operation.name for ci in _wide_gates(qc))
        if not wide:
            break
        new = qc.decompose(gates_to_decompose=sorted(wide))
        if (Counter(ci.operation.name for ci in _wide_gates(new)) == wide
                and len(new.data) == len(qc.data)):
            break                                # nothing left to decompose
        qc = new
    return qc


def _interaction_counts(circuit: QuantumCircuit, pos: dict | None = None):
    """(Counter of node pairs of the two-qubit gates, names of gates on more
    than two qubits) after :func:`_flatten`. ``pos`` maps a qubit to the
    output bit, i.e. the node, it is measured into."""
    flat = _flatten(circuit)
    pairs, wide = Counter(), set()
    for ci in flat.data:
        name = ci.operation.name
        if name in _NOT_GATES or name == "measure":
            continue
        qs = [flat.find_bit(q).index for q in ci.qubits]
        if len(qs) == 2:
            a, b = ((pos.get(q, q) for q in qs) if pos else qs)
            pairs[(min(a, b), max(a, b))] += 1
        elif len(qs) > 2:
            wide.add(name)
    return pairs, wide


def _edge_set(inst) -> set:
    return {(min(i, j), max(i, j)) for i, j, _ in inst.edges}


def _structure_problem(inst, circuit, pos: dict | None = None) -> str | None:
    pairs, wide = _interaction_counts(circuit, pos)
    edges = _edge_set(inst)
    if wide:
        return (f"gates {sorted(wide)} act on more than two qubits even after "
                "decomposition; build the cost layer from two-qubit gates.")
    extra, missing = sorted(set(pairs) - edges), sorted(edges - set(pairs))
    parts = []
    if extra:
        parts.append(f"two-qubit gates act on pairs that are not edges of the "
                     f"graph: {extra[:6]}{' ...' if len(extra) > 6 else ''}")
    if missing:
        parts.append(f"edges without a two-qubit gate: {missing[:6]}"
                     f"{' ...' if len(missing) > 6 else ''}")
    if parts:
        return "; ".join(parts) + "."
    return None


# ------------------------------------------------------------ Exercise 1a
def _variant_circuit(inst, g, t, b, init="ok", cost="ok", theta="ok",
                     mixer="ok", order="ok", double_edge=None,
                     t0_scale=1.0) -> QuantumCircuit:
    """The reference basis circuit (bound, unmeasured) with known wrong
    variants of its layers: ``init``, ``cost``, ``theta``, ``mixer``, the
    layer ``order``, one edge applied twice (``double_edge``) and the
    mixer's theta_init scaled by ``t0_scale``."""
    n, t0 = inst.n, inst.theta_init
    t0m = t0 * t0_scale
    qc = QuantumCircuit(n)

    def init_layer():
        angle = {"ok": t0, "half": t0 / 2, "minus": -t0}.get(init)
        if angle is not None:
            qc.ry(angle, range(n))
        elif init == "hadamard":
            qc.h(range(n))

    def cost_layer():
        for i, j, w in inst.edges:
            a = g * w / 2
            zz, z = {"ok": (a, -a), "no_rz": (a, None), "full_angle": (2 * a, -a),
                     "full_angle_all": (2 * a, -2 * a), "half_all": (a / 2, -a / 2),
                     "sign": (-a, a), "rz_sign": (a, a)}[cost]
            for _ in range(2 if double_edge == (min(i, j), max(i, j)) else 1):
                qc.rzz(zz, i, j)
                if z is not None:
                    qc.rz(z, i)
                    qc.rz(z, j)

    def theta_layer():
        factor = {"ok": 1.0, "minus": -1.0, "double": 2.0}.get(theta)
        if factor is not None:
            qc.rz(factor * t, range(n))

    def mixer_layer():
        if mixer == "x":
            qc.rx(2 * b, range(n))
            return
        first, angle, last = {"ok": (-t0m, -2 * b, t0m),
                              "reversed": (t0m, -2 * b, -t0m),
                              "sign": (-t0m, 2 * b, t0m),
                              "half": (-t0m, -b, t0m),
                              "paper": (-t0m, b, t0m)}[mixer]
        qc.ry(first, range(n))
        qc.rz(angle, range(n))
        qc.ry(last, range(n))

    layers = {"ok": (init_layer, cost_layer, theta_layer, mixer_layer),
              "mixer_first": (init_layer, mixer_layer, cost_layer,
                              theta_layer)}[order]
    for layer in layers:
        layer()
    return qc


def _reverse_bits(probs: np.ndarray, n: int) -> np.ndarray:
    idx = np.arange(1 << n)
    rev = np.zeros_like(idx)
    for i in range(n):
        rev |= ((idx >> i) & 1) << (n - 1 - i)
    out = np.zeros_like(probs)
    out[rev] = probs
    return out


def _mod(**mods):
    """Builder of the distribution of a :func:`_variant_circuit`."""
    def build(inst, values):
        from qiskit.quantum_info import Statevector

        g, t, b = values
        return Statevector(_variant_circuit(inst, g, t, b, **mods)).probabilities()
    return build


def _permuted(order):
    """Builder of the reference with the parameter values permuted: the
    layer of slot s (cost, LCU phase, mixer) uses ``values[order[s]]``."""
    def build(inst, values):
        return _reference_1a(inst, tuple(values[i] for i in order))
    return build


def _bitorder(inst, values):
    return _reverse_bits(_reference_1a(inst, values), inst.n)


def _by_signs(theta: int, beta: int):
    """Decide between two explanations from the signs of the circuit's
    rz(theta) and mixer rz(beta) coefficients: "main" when they read
    (theta, beta), "alt" when both read the opposite, None otherwise."""
    def decide(readings):
        known = [(readings[k], want) for k, want in (("theta", theta),
                                                      ("beta", beta))
                 if readings[k] is not None]
        if known and all(got == want for got, want in known):
            return "main"
        if known and all(got == -want for got, want in known):
            return "alt"
        return None
    return decide


def _by_init(sign: int):
    """Decide from the sign of the initial ry(+-theta_init)."""
    def decide(readings):
        if readings["init"] is None:
            return None
        return "main" if readings["init"] == sign else "alt"
    return decide


@dataclass(frozen=True)
class Variant1a:
    """A known wrong basis circuit: ``build(inst, (gamma, theta, beta))``
    gives its measured distribution. ``text`` names the mistake; ``pauli``
    replaces it when the cost layer is a ``PauliEvolutionGate``. When a
    second mistake gives the same distribution, ``alt`` names it, ``decide``
    picks one from the participant's circuit ("main", "alt" or None) and
    ``both`` is the message when the circuit cannot tell them apart.
    ``{cost_fix}`` in a text is the fix of the cost layer."""

    name: str
    build: Callable
    text: str
    pauli: str | None = None
    alt: str | None = None
    both: str | None = None
    decide: Callable | None = None


_CF_GATES = "use rzz(gamma*w/2, i, j) and rz(-gamma*w/2) on i and on j"
_CF_PAULI = ("build it as PauliEvolutionGate(cost_op, time=gamma) with cost_op "
             "= -ising_objective of Step 1 (to_ising returns the operator to "
             "minimize, and PauliEvolutionGate(op, time=gamma) applies "
             "exp(-i gamma op))")
_SIGNS = ("the signs of the cost layer, of rz(theta) and of rz(-2*beta) with "
          "the table above")


def _swap(order, text):
    return Variant1a(f"swap{order}", _permuted(order), text)


VARIANTS_1A = (
    Variant1a(
        "mixer_reversed", _mod(mixer="reversed"),
        "the three mixer gates are in the wrong order: in circuit order apply "
        "ry(-theta_init), then rz(-2*beta), then ry(theta_init) on every qubit.",
        alt="the initial state uses ry(-theta_init): start with ry(theta_init) "
            "on every qubit, theta_init = inst.theta_init.",
        both="the three mixer gates are in the wrong order, or the initial "
             "state uses ry(-theta_init) (both give this output): start with "
             "ry(theta_init), and apply the mixer as ry(-theta_init), then "
             "rz(-2*beta), then ry(theta_init).",
        decide=_by_init(+1)),
    Variant1a(
        "mixer_sign", _mod(mixer="sign"),
        "the mixer rotates the wrong way: its middle gate is rz(-2*beta), not "
        "rz(2*beta).",
        alt="the cost layer and the rz(theta) layer both have the wrong sign: "
            "in the cost layer {cost_fix}, and use rz(theta).",
        both="the mixer's middle gate has the wrong sign, or else the cost "
             "layer and the rz(theta) layer both do (both give this output): "
             f"compare {_SIGNS}.",
        decide=_by_signs(+1, +1)),
    Variant1a(
        "mixer_half", _mod(mixer="half"),
        "the mixer angle is off by a factor of 2: its middle gate is "
        "rz(-2*beta), not rz(-beta).",
        alt="the mixer's middle gate is rz(beta), as the paper's Fig. 3 writes "
            "it, and the cost layer and the rz(theta) layer both have the wrong "
            "sign: this notebook uses rz(-2*beta), a rescaled beta (the table "
            "above is the convention to follow); in the cost layer {cost_fix}, "
            "and use rz(theta).",
        both="the mixer's middle gate must be rz(-2*beta): the output matches "
             "rz(-beta), or rz(beta) (as the paper's Fig. 3 writes it) with the "
             "cost layer and the rz(theta) layer both of the wrong sign; follow "
             "the table above.",
        decide=_by_signs(+1, -1)),
    Variant1a(
        "mixer_paper", _mod(mixer="paper"),
        "the mixer's middle gate is rz(beta), as the paper's Fig. 3 writes it; "
        "this notebook writes it rz(-2*beta), a rescaled beta, and the check "
        "and the kit's angles follow the table above: use rz(-2*beta).",
        alt="the mixer's middle gate is rz(-beta), off by a factor of 2, and "
            "the cost layer and the rz(theta) layer both have the wrong sign: "
            "use rz(-2*beta); in the cost layer {cost_fix}, and use rz(theta).",
        both="the mixer's middle gate must be rz(-2*beta): the output matches "
             "rz(beta), as the paper's Fig. 3 writes it, or rz(-beta) with the "
             "cost layer and the rz(theta) layer both of the wrong sign; follow "
             "the table above.",
        decide=_by_signs(+1, +1)),
    Variant1a(
        "mixer_x", _mod(mixer="x"),
        "the mixer is the standard rx mixer; the warm-start mixer is "
        "ry(-theta_init), rz(-2*beta), ry(theta_init) on every qubit."),
    Variant1a(
        "mixer_first", _mod(order="mixer_first"),
        "the output is the initial state at every parameter set: the mixer "
        "comes last, after the cost layer and rz(theta) (the mixer alone "
        "leaves the initial state unchanged)."),
    Variant1a(
        "cost_no_rz", _mod(cost="no_rz"),
        "the cost layer lacks its single-qubit terms: for every edge (i, j, w) "
        "apply rz(-gamma*w/2) on i and on j next to rzz(gamma*w/2, i, j).",
        pauli="the cost operator lacks its single-qubit Z terms: {cost_fix}."),
    Variant1a(
        "cost_full_angle", _mod(cost="full_angle"),
        "the rzz angle is gamma*w; it must be gamma*w/2, because x_i x_j = "
        "(1 - Z_i - Z_j + Z_i Z_j)/4.",
        pauli="the ZZ terms of the cost operator are twice too large: "
              "{cost_fix}."),
    Variant1a(
        "cost_full_angle_all", _mod(cost="full_angle_all"),
        "the cost-layer angles are twice too large (or every edge is applied "
        "twice, as a loop over each node's neighbours does): {cost_fix}, once "
        "per edge of inst.edges.",
        pauli="the cost layer is twice too strong (time=2*gamma, or an "
              "operator counted twice): {cost_fix}."),
    Variant1a(
        "cost_half_all", _mod(cost="half_all"),
        "the cost-layer angles are half as large as they should be: "
        "{cost_fix}.",
        pauli="the cost layer is half as strong as it should be: {cost_fix}."),
    Variant1a(
        "cost_sign", _mod(cost="sign"),
        "the cost layer has the wrong sign (it applies exp(+i gamma H1)): "
        "{cost_fix}.",
        alt="the rz(theta) layer and the mixer both have the wrong sign: use "
            "rz(theta), and rz(-2*beta) in the mixer.",
        both="the cost layer has the wrong sign, or else the rz(theta) layer "
             f"and the mixer both do (both give this output): compare {_SIGNS}.",
        decide=_by_signs(+1, -1)),
    Variant1a(
        "cost_rz_sign", _mod(cost="rz_sign"),
        "the single-qubit terms of the cost layer have the wrong sign: use "
        "rz(-gamma*w/2) on i and on j.",
        pauli="the single-qubit Z terms of the cost operator have the wrong "
              "sign: {cost_fix}.",
        alt="the rzz terms of the cost layer, the rz(theta) layer and the "
            "mixer all have the wrong sign: use rzz(gamma*w/2, i, j), "
            "rz(theta) and rz(-2*beta).",
        both="the single-qubit terms of the cost layer have the wrong sign, or "
             "else its rzz terms, the rz(theta) layer and the mixer all do "
             "(both give this output): compare the signs with the table above.",
        decide=_by_signs(+1, -1)),
    Variant1a(
        "theta_missing", _mod(theta="missing"),
        "the LCU phase layer has no effect on the measured output: apply "
        "rz(theta) to every qubit after the cost layer and before the mixer "
        "(placed after the mixer it only changes phases)."),
    Variant1a(
        "theta_sign", _mod(theta="minus"),
        "the LCU phase layer has the wrong sign: use rz(theta), which "
        "multiplies |x> by exp(i theta W(x)) up to a global phase.",
        alt="the cost layer and the mixer both have the wrong sign: in the "
            "cost layer {cost_fix}, and use rz(-2*beta) in the mixer.",
        both="the rz(theta) layer has the wrong sign, or else the cost layer "
             f"and the mixer both do (both give this output): compare {_SIGNS}.",
        decide=_by_signs(-1, -1)),
    Variant1a(
        "theta_double", _mod(theta="double"),
        "the LCU phase layer is rz(2*theta); it must be rz(theta) on every "
        "qubit."),
    Variant1a(
        "init_missing", _mod(init="missing"),
        "the initial state is missing: start with ry(theta_init) on every "
        "qubit, theta_init = inst.theta_init."),
    Variant1a(
        "init_hadamard", _mod(init="hadamard"),
        "the initial state is the uniform superposition; the warm start is "
        "ry(theta_init) on every qubit, theta_init = inst.theta_init."),
    Variant1a(
        "init_half", _mod(init="half"),
        "the initial-state angle is halved: the initial state is "
        "ry(theta_init) on every qubit with theta_init = 2*arcsin(sqrt(k/n)) "
        "= inst.theta_init, so that each qubit reads 1 with probability k/n."),
    Variant1a(
        "init_half_everywhere", _mod(init="half", t0_scale=0.5),
        "theta_init is halved, in the initial state and in the mixer: "
        "theta_init = 2*arcsin(sqrt(k/n)) = inst.theta_init."),
    _swap((0, 2, 1), "theta and beta are swapped: theta is the angle of the "
                     "LCU phase layer rz(theta), beta the mixer angle in "
                     "rz(-2*beta)."),
    _swap((1, 0, 2), "gamma and theta are swapped: gamma is the angle of the "
                     "cost layer, theta the angle of the LCU phase layer "
                     "rz(theta)."),
    _swap((2, 1, 0), "gamma and beta are swapped: gamma is the angle of the "
                     "cost layer, beta the mixer angle in rz(-2*beta)."),
    _swap((1, 2, 0), "the parameters are permuted: the cost layer uses theta, "
                     "the LCU phase layer beta and the mixer gamma; gamma "
                     "belongs in the cost layer, theta in rz(theta), beta in "
                     "rz(-2*beta)."),
    _swap((2, 0, 1), "the parameters are permuted: the cost layer uses beta, "
                     "the LCU phase layer gamma and the mixer theta; gamma "
                     "belongs in the cost layer, theta in rz(theta), beta in "
                     "rz(-2*beta)."),
    Variant1a(
        "bitorder", _bitorder,
        "the output is the reference with the qubit order reversed: node i "
        "must be qubit i and be measured into classical bit i (use "
        "qc.measure_all() and index qubits by node number)."),
)


def _matches(inst, build, dists) -> bool:
    for values, got in zip(_PARAMS_1A, dists):
        if np.max(np.abs(got - build(inst, values))) > TOL_VARIANT:
            return False
    return True


def _coefficient_sign(circuit: QuantumCircuit, name: str) -> int | None:
    """Sign of the coefficient of Parameter ``name`` in the circuit's rz /
    phase gates (+1, -1), or None when there is no such gate, the gates
    disagree or the angle is not linear in the parameter."""
    signs = set()
    for ci in circuit.data:
        if ci.operation.name not in ("rz", "p", "u1", "phase"):
            continue
        for p in ci.operation.params:
            if not isinstance(p, ParameterExpression) or len(p.parameters) != 1:
                continue
            (par,) = p.parameters
            if par.name != name:
                continue
            try:
                c = float(p.gradient(par))
            except (TypeError, ValueError):
                return None
            if c:
                signs.add(1 if c > 0 else -1)
    return signs.pop() if len(signs) == 1 else None


def _init_sign(circuit: QuantumCircuit, t0: float) -> int | None:
    """+1 / -1 when the first gate on every qubit is ry(+theta_init) /
    ry(-theta_init), None otherwise."""
    first = {}
    for ci in circuit.data:
        if ci.operation.name in _NOT_GATES:
            continue
        for q in ci.qubits:
            first.setdefault(circuit.find_bit(q).index, ci)
    signs = set()
    for ci in first.values():
        if ci.operation.name != "ry" or len(ci.qubits) != 1:
            return None
        try:
            a = float(ci.operation.params[0])
        except (TypeError, ValueError):
            return None
        if abs(a - t0) < 1e-6:
            signs.add(1)
        elif abs(a + t0) < 1e-6:
            signs.add(-1)
        else:
            return None
    return signs.pop() if len(signs) == 1 else None


def _uses_pauli_evolution(circuit: QuantumCircuit, depth: int = 0) -> bool:
    from qiskit.circuit.library import PauliEvolutionGate

    for ci in circuit.data:
        op = ci.operation
        if isinstance(op, PauliEvolutionGate):
            return True
        if depth < 5 and len(ci.qubits) > 2 and getattr(op, "definition", None) \
                is not None and _uses_pauli_evolution(op.definition, depth + 1):
            return True
    return False


def _variant_message(v: Variant1a, circuit, inst) -> str:
    pauli = _uses_pauli_evolution(circuit)
    main = v.pauli if (pauli and v.pauli) else v.text
    if v.alt is not None:
        readings = {"theta": _coefficient_sign(circuit, "theta"),
                    "beta": _coefficient_sign(circuit, "beta"),
                    "init": _init_sign(circuit, inst.theta_init)}
        choice = v.decide(readings) if v.decide else None
        main = {"main": main, "alt": v.alt}.get(choice, v.both)
    return main.format(cost_fix=_CF_PAULI if pauli else _CF_GATES)


def _duplicate_edge(inst, dists, pairs: Counter) -> str | None:
    """An edge whose cost gates are applied twice: the edges with more
    two-qubit gates than the others are tried as that variant."""
    edges = sorted(_edge_set(inst))
    used = [pairs[e] for e in edges if pairs.get(e)]
    if not used:
        return None
    base = min(used)
    for e in [e for e in edges if pairs.get(e, 0) > base][:6]:
        if _matches(inst, _mod(double_edge=e), dists):
            return (f"the cost gates of edge {e} are applied twice: apply each "
                    "edge of inst.edges once.")
    return None


def _diagnose_1a(inst, circuit, dists, pos) -> str | None:
    for v in VARIANTS_1A:
        if _matches(inst, v.build, dists):
            return _variant_message(v, circuit, inst)
    pairs, _ = _interaction_counts(circuit, pos)
    return _duplicate_edge(inst, dists, pairs)


@_guarded("1a")
def check_1a(inst: DensestKSubgraph, circuit) -> CheckResult:
    """Exercise 1a: ``circuit`` (or the function ``basis_circuit`` itself,
    called with ``inst``) is the basis circuit of paper Fig. 3 with
    parameters ``gamma``, ``theta``, ``beta`` that measures the n qubits
    into one n-bit register (``measure_all``).

    Deciding test: the measured-bit distribution equals
    |exact.penalty_basis_state(inst, beta, gamma, theta)|^2 within 1e-9 at
    three fixed random parameter sets. Structure (after decomposing
    composite gates such as a ``PauliEvolutionGate``, qubits mapped to the
    node they are measured into): every two-qubit interaction acts on an
    edge of the graph and every edge is used. On a mismatch the known wrong
    variants (``VARIANTS_1A``) are simulated and the one that matches is
    named."""
    if callable(circuit) and not isinstance(circuit, QuantumCircuit):
        circuit = _call("basis_circuit(inst)", circuit, inst)
    if circuit is None:
        raise _Fail("no circuit: basis_circuit(inst) returned None.")
    if not isinstance(circuit, QuantumCircuit):
        raise _Fail("basis_circuit(inst) must return a QuantumCircuit, got "
                    f"{type(circuit).__name__}.")
    n = inst.n
    if circuit.num_qubits != n:
        raise _Fail(f"the circuit has {circuit.num_qubits} qubits; it needs "
                    f"one qubit per node, n = {n} (QuantumCircuit(inst.n)).")
    params = _named_parameters(circuit)
    if set(params) != set(_PARAM_NAMES):
        missing = [p for p in _PARAM_NAMES if p not in params]
        extra = sorted(set(params) - set(_PARAM_NAMES))
        text = (f"the circuit has the parameters {sorted(params)}; it needs "
                "exactly Parameter(\"gamma\"), Parameter(\"theta\") and "
                "Parameter(\"beta\").")
        if missing:
            text += f" Missing: {', '.join(missing)}."
            if "theta" in missing:
                text += " (theta is the angle of the LCU phase layer rz(theta).)"
        if extra:
            text += f" Not expected: {', '.join(extra)}."
        raise _Fail(text)
    pos, problem = _measurement_positions(circuit, n, _FIX_1A)
    if problem:
        raise _Fail(problem)
    dists = [_bound_distribution(circuit, params, v, pos, n) for v in _PARAMS_1A]
    diffs = [float(np.max(np.abs(d - _reference_1a(inst, v))))
             for d, v in zip(dists, _PARAMS_1A)]
    if max(diffs) <= TOL_1A:
        problem = _structure_problem(inst, circuit, pos)
        if problem:
            raise _Fail("the output distribution is correct, but the "
                        "two-qubit structure is not the cost layer: " + problem)
        return _result("1a", True, "the circuit reproduces the reference "
                       "distribution at three fixed (gamma, theta, beta); "
                       "every two-qubit interaction acts on an edge and all "
                       f"{len(inst.edges)} edges are used.")
    text = _diagnose_1a(inst, circuit, dists, pos)
    if text:
        raise _Fail(text)
    worst = int(np.argmax(diffs))
    hint = _structure_problem(inst, circuit, pos)
    raise _Fail(f"the output distribution differs from the reference (largest "
                f"difference {diffs[worst]:.3g} at (gamma, theta, beta) = "
                f"{_fmt(_PARAMS_1A[worst])}). "
                + (f"Structure: {hint} " if hint else "")
                + "Compare each of the four layers with the table above.")


# ------------------------------------------------------------ Exercise 1b
def _as_array(x):
    """``np.asarray`` for numbers, objects converted to complex; None when
    ``x`` is not numeric."""
    if x is None:
        return None
    try:
        arr = np.asarray(x)
        if arr.dtype.kind in "USV":
            return None
        if arr.dtype == object or arr.dtype.kind not in "biufc":
            arr = arr.astype(complex)
    except (TypeError, ValueError):
        return None
    return arr


def _angles_equal(a, b, tol: float = TOL_1B) -> bool:
    """Equal modulo 2 pi, entry by entry."""
    d = np.angle(np.exp(1j * (np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
    return bool(np.all(np.abs(d) <= tol))


def _vector(x, name: str, m: int, where: str, real: bool = False,
            complex_text: str | None = None) -> np.ndarray:
    """``x`` as a vector of n+1 = m numbers (real ones for ``real``)."""
    arr = _as_array(x)
    if arr is None:
        raise _Fail(f"{name} must be an array of {m} numbers ({where}).")
    arr = arr.ravel()
    if arr.shape != (m,):
        raise _Fail(f"{name} has {arr.size} entries; it needs n+1 = {m}, one "
                    f"per b = 0..n ({where}).")
    if not real:
        return arr.astype(complex)
    if np.iscomplexobj(arr):
        if np.max(np.abs(arr.imag)) > TOL_1B:
            raise _Fail(complex_text or f"{name} must be real numbers; the "
                        f"values returned are complex ({where}).")
        arr = arr.real
    return arr.astype(float)


def _label_1b(x, m: int, want_t) -> str | None:
    """What a returned value looks like: "thetas", "c", "weights", "Gamma"
    or None."""
    arr = _as_array(x)
    if arr is None:
        return None
    if arr.size == 1:
        return "Gamma"
    flat = arr.ravel()
    if flat.size != m:
        return None
    if np.iscomplexobj(flat) and np.max(np.abs(flat.imag)) > TOL_1B:
        return "c"
    re = flat.real.astype(float)
    if _angles_equal(re, want_t):
        return "thetas"
    if np.all(re >= -TOL_1B) and abs(re.sum() - 1.0) <= 1e-4:
        return "weights"
    return None


def _order_problem_1b(out, m: int, want_t) -> str | None:
    """A message when the returned values are the four results in another
    order (each recognized kind once, at least one misplaced)."""
    labels = [_label_1b(x, m, want_t) for x in out]
    expected = ("thetas", "c", "weights", "Gamma")
    known = [lab for lab in labels if lab is not None]
    if len(set(known)) != len(known) or all(
            lab is None or lab == e for lab, e in zip(labels, expected)):
        return None
    got = ", ".join(lab or "?" for lab in labels)
    return ("fourier_lcu returns its values in the wrong order: return thetas, "
            f"c, weights, Gamma, in this order (the values returned look like "
            f"{got}).")


def _diagnose_thetas(thetas, want_t, m: int, where: str) -> str:
    n = m - 1
    b = np.arange(m)
    if np.allclose(thetas, 2 * np.pi * b / n, rtol=0, atol=TOL_1B):
        return ("thetas include the end point 2*pi: theta_b = 2*pi*b/(n+1), "
                "e.g. 2*np.pi*np.arange(n+1)/(n+1) (np.linspace(0, 2*np.pi, "
                "n+1) spaces them by 2*pi/n).")
    k = np.rint(np.mod(thetas, 2 * np.pi) * m / (2 * np.pi)).astype(int) % m
    if sorted(k) == list(range(m)) and _angles_equal(thetas, want_t[k]):
        return ("thetas hold the right angles in another order: list them for "
                "b = 0..n, theta_b = 2*pi*b/(n+1), with c and the weights in "
                "the same order.")
    if np.allclose(thetas, np.degrees(want_t), rtol=0, atol=1e-3):
        return "thetas are in degrees; use radians: theta_b = 2*pi*b/(n+1)."
    if np.allclose(thetas, b, rtol=0, atol=TOL_1B):
        return "thetas are the indices b; the angles are theta_b = 2*pi*b/(n+1)."
    return (f"thetas must be theta_b = 2*pi*b/(n+1) for b = 0..n, in radians "
            f"({where}; angles equal modulo 2*pi are accepted).")


def _phases_1b(inst, g):
    """Wrong phases u(W) and their messages (the right one is
    exp(i gamma lam (W - k)^2), W = 0..n)."""
    W = np.arange(inst.n + 1)
    lam, k = inst.lam, inst.k
    return (
        (np.exp(1j * g * (W - k) ** 2), "the phase lacks the factor lam: "
         "u(W) = exp(i*gamma*lam*(W-k)**2), lam = inst.lam."),
        (np.exp(1j * g * lam * (W - k)), "(W - k) is not squared: u(W) = "
         "exp(i*gamma*lam*(W-k)**2)."),
        (np.exp(1j * g * lam * (W - k) ** 2 / 2), "the phase is "
         "gamma*lam*(W-k)^2/2; it is gamma*lam*(W-k)^2, without the 1/2."),
        (np.exp(1j * g * lam * (W + 1 - k) ** 2), "W runs over 1..n+1; it runs "
         "over 0..n, W = np.arange(n+1)."),
    )


_PHASES_1B = _phases_1b          # the table of wrong phases (see above)


def _diagnose_1b(recon: np.ndarray, inst, g: float) -> str:
    m = inst.n + 1
    W = np.arange(m)
    u = np.exp(1j * g * inst.lam * (W - inst.k) ** 2)
    cands = (("u", u), ("conj", np.conj(u)), ("rev", u[(-W) % m]),
             ("conj_rev", np.conj(u)[(-W) % m]))
    for label, v in cands:
        for scale in (1.0, float(m), 1.0 / m):
            if not np.allclose(recon, scale * v, rtol=0, atol=TOL_1B * max(1.0, scale)):
                continue
            once = "" if scale == 1 else " Divide np.fft.fft(u) by n+1 exactly once."
            if label == "u":
                if scale > 1:
                    return ("the coefficients are n+1 times too large: divide "
                            "np.fft.fft(u) by n+1.")
                return ("the coefficients are n+1 times too small: divide by "
                        "n+1 only once.")
            if label == "conj":
                return ("the coefficients reproduce the complex conjugate "
                        "exp(-i gamma lam (W-k)^2): the phase needs a plus "
                        "sign, u = np.exp(1j*gamma*lam*(W-k)**2)." + once)
            if label == "rev":
                return ("the sum reproduces u at -W: the coefficients are "
                        "c_b = (1/(n+1)) sum_W u(W) exp(-i theta_b W), i.e. "
                        "np.fft.fft(u) / (n+1), not np.fft.ifft." + once)
            return ("the coefficients belong to the conjugate phase with "
                    "reversed index: use u = np.exp(1j*gamma*lam*(W-k)**2) "
                    "and c = np.fft.fft(u) / (n+1).")
    for v, text in _PHASES_1B(inst, g):
        if np.allclose(recon, v, rtol=0, atol=TOL_1B):
            return text
    err = float(np.max(np.abs(recon - u)))
    return ("sum_b c_b exp(i theta_b W) does not reproduce u(W) = "
            f"exp(i gamma lam (W-k)^2) (largest error {err:.3g}). The "
            "coefficients are c_b = (1/(n+1)) sum_W u(W) exp(-i theta_b W), "
            "which is np.fft.fft(u) / (n+1).")


def _gamma_value(x, where: str) -> float:
    arr = _as_array(x)
    if arr is None or arr.size != 1:
        raise _Fail(f"Gamma must be one number, ||c||_1^2 ({where}).")
    G = complex(arr.reshape(-1)[0])
    if abs(G.imag) > 1e-9:
        raise _Fail("Gamma must be a real number, ||c||_1^2.")
    return float(G.real)


def _check_1b_case(inst, g: float, fn, where: str) -> float:
    """One call ``fn(inst, g)``; returns Gamma or raises ``_Fail``."""
    m = inst.n + 1
    W = np.arange(m)
    want_t = 2.0 * np.pi * W / m
    out = _call(f"fourier_lcu(inst, {g})", fn, inst, g)
    if isinstance(out, dict):
        raise _Fail("fourier_lcu returned a dict; return the four values in "
                    "this order: return thetas, c, weights, Gamma.")
    if not isinstance(out, (tuple, list)) or len(out) != 4:
        raise _Fail("fourier_lcu must return four values: thetas, c, "
                    "weights, Gamma.")
    problem = _order_problem_1b(out, m, want_t)
    if problem:
        raise _Fail(problem)
    thetas = _vector(out[0], "thetas", m, where, real=True,
                     complex_text="thetas must be real angles; the first "
                                  "value returned is complex (c goes second).")
    c = _vector(out[1], "c", m, where)
    if not _angles_equal(thetas, want_t):
        raise _Fail(_diagnose_thetas(thetas, want_t, m, where))
    u = np.exp(1j * g * inst.lam * (W - inst.k) ** 2)
    recon = np.exp(1j * np.outer(W, want_t)) @ c
    if not np.allclose(recon, u, rtol=0, atol=TOL_1B):
        raise _Fail(f"{where}: " + _diagnose_1b(recon, inst, g))
    q = np.abs(c) / np.abs(c).sum()
    wts = _vector(out[2], "weights", m, where, real=True,
                  complex_text="weights must be real probabilities: q_b = "
                               "|c_b| / sum_b |c_b|.")
    if not np.allclose(wts, q, rtol=0, atol=TOL_1B):
        if np.allclose(wts, np.abs(c), rtol=0, atol=TOL_1B):
            raise _Fail("weights are |c_b|; divide them by ||c||_1 = sum_b "
                        "|c_b| so that they sum to 1.")
        if np.allclose(wts, np.abs(c) ** 2, rtol=0, atol=TOL_1B):
            raise _Fail("weights are |c_b|^2; the paper samples branch b "
                        "with q_b = |c_b| / ||c||_1.")
        if np.any(wts < 0):
            raise _Fail("weights must be nonnegative: q_b = |c_b| / ||c||_1.")
        if abs(wts.sum() - 1.0) > 1e-6:
            raise _Fail(f"weights sum to {wts.sum():.6g}; q_b = |c_b| / "
                        "||c||_1 sums to 1.")
        raise _Fail("weights must be q_b = |c_b| / ||c||_1.")
    G = _gamma_value(out[3], where)
    want_G = float(np.abs(c).sum() ** 2)
    close = functools.partial(np.isclose, rtol=RTOL_GAMMA_1B, atol=1e-9)
    if not close(G, want_G):
        if close(G, np.abs(c).sum()):
            raise _Fail("Gamma is ||c||_1; the sampling overhead is its "
                        "square, Gamma = ||c||_1^2.")
        if close(G, float((np.abs(c) ** 2).sum())) or close(
                G, float(np.sqrt((np.abs(c) ** 2).sum()))):
            raise _Fail("Gamma uses the 2-norm, which is 1 by Parseval's "
                        "identity; Gamma = ||c||_1^2 = (sum_b |c_b|)^2.")
        raise _Fail(f"Gamma = {G:.10g} {where}; it must be ||c||_1^2 = "
                    f"{want_G:.10g}.")
    if G > m + 1e-6:
        raise _Fail(f"Gamma = {G:.10g} exceeds the bound n+1 = {m} ({where}).")
    return G


@_guarded("1b")
def check_1b(inst: DensestKSubgraph, fn) -> CheckResult:
    """Exercise 1b: ``fn(inst, gamma) -> (thetas, c, weights, Gamma)`` on
    the training instance at gamma in {0.37, 0.9, 1.63} and on a second
    instance (``ASYMMETRIC_INSTANCE``: n = 7, k = 3, lam = 5): shapes n+1;
    theta_b = 2 pi b/(n+1) modulo 2 pi; sum_b c_b e^{i theta_b W} =
    e^{+i gamma lam (W-k)^2} for all W (on failure: wrong order of the
    returned values, complex conjugate, missing 1/(n+1), ifft for fft,
    missing lam, (W-k) not squared, ...); weights = |c| / ||c||_1;
    Gamma = ||c||_1^2 <= n+1. Tolerances admit float32 answers."""
    if not callable(fn):
        raise _Fail("pass the function fourier_lcu itself, not its result.")
    gammas_out = []
    for g in _GAMMAS_1B:
        gammas_out.append(_check_1b_case(inst, g, fn, f"at gamma = {g}"))
    second = ASYMMETRIC_INSTANCE
    where = (f"on a second instance (n = {second.n}, k = {second.k}, lam = "
             f"{second.lam:g}) at gamma = {_GAMMA_1B_SECOND}")
    try:
        _check_1b_case(second, _GAMMA_1B_SECOND, fn, where)
    except _Fail as f:
        text = f.text if "second instance" in f.text else f"{where}: {f.text}"
        raise _Fail(f"{text.rstrip('.')}. The function passes on the training "
                    "instance: use inst.n, inst.k and inst.lam of the instance "
                    "passed in, not fixed numbers.") from None
    m = inst.n + 1
    return _result("1b", True, "sum_b c_b exp(i theta_b W) = "
                   "exp(i gamma lam (W-k)^2) for every W at gamma = "
                   f"{', '.join(str(g) for g in _GAMMAS_1B)} and on a second "
                   "instance; weights q_b = |c_b|/||c||_1; Gamma = "
                   f"{', '.join(f'{G:.4f}' for G in gammas_out)} <= n+1 = {m}.")


# ------------------------------------------------------------ Exercise 1c
def _is_two_qubit(ci) -> bool:
    return len(ci.qubits) == 2 and ci.operation.name not in _NOT_GATES + ("measure",)


def _isa_distribution(bound: QuantumCircuit, n: int, pos: dict | None = None
                      ) -> np.ndarray:
    """Measured-bit distribution of a (wide) bound ISA circuit, simulated on
    the measured qubits and the qubits connected to them through
    multi-qubit gates; operations on other qubits cannot change the
    measured bits and are dropped. Unmeasured kept qubits are traced out."""
    from qiskit.quantum_info import Statevector

    if pos is None:
        pos, problem = _measurement_positions(bound, n, _FIX_1C)
        if problem:
            raise _Fail(problem)
    parent: dict = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ops = []
    for ci in bound.data:
        if ci.operation.name in _NOT_GATES:
            continue
        qs = [bound.find_bit(q).index for q in ci.qubits]
        for q in qs:
            parent.setdefault(q, q)
        for q in qs[1:]:
            parent[find(q)] = find(qs[0])
        ops.append((ci, qs))
    roots = {find(q) for q in pos}
    keep = sorted(q for q in parent if find(q) in roots)
    if len(keep) > MAX_SIM_QUBITS:
        raise _Fail(f"the measured qubits interact with {len(keep)} qubits; "
                    f"the check simulates at most {MAX_SIM_QUBITS}. Transpile "
                    "the 12-qubit basis circuit with optimization_level=3 so "
                    "that it stays on few qubits.")
    index = {q: i for i, q in enumerate(keep)}
    sub = QuantumCircuit(len(keep), global_phase=bound.global_phase)
    measured, started = set(), set()
    for ci, qs in ops:
        if qs[0] not in index:
            continue
        name = ci.operation.name
        if name == "measure":
            measured.add(qs[0])
            continue
        if measured.intersection(qs):
            raise _Fail(f"{name} acts on qubits {qs} after their measurement; "
                        "measure only at the end.")
        if ci.clbits or getattr(ci.operation, "condition", None):
            raise _Fail("classically conditioned operations are not allowed.")
        if name == "reset":
            if started.intersection(qs):
                raise _Fail("a reset inside the circuit is not allowed.")
            continue
        started.update(qs)
        sub.append(ci.operation, [sub.qubits[index[q]] for q in qs])
    probs = Statevector(sub).probabilities()
    return _to_positions(probs, index, pos, n)


@_guarded("1c")
def check_1c(inst: DensestKSubgraph, isa, target=None,
             reference_circuit=None) -> CheckResult:
    """Exercise 1c: ``isa`` is the transpiled, still parametrized basis
    circuit. ``hw.check_isa`` against ``target`` must be clean; it must
    measure the n qubits into one n-bit register; it may hold at most
    ``CZ_BAR`` two-qubit gates; and its measured distribution (simulated on
    the measured qubits and those connected to them) must equal the
    reference at two random parameter sets (the exact reference, or the
    distribution of ``reference_circuit`` when given)."""
    from . import hardware as hw

    if isa is None:
        raise _Fail("no ISA circuit yet (isa is None): transpile the basis "
                    "circuit qc for your backend.")
    if not isinstance(isa, QuantumCircuit):
        raise _Fail(f"pass the transpiled QuantumCircuit, got "
                    f"{type(isa).__name__}.")
    params = _named_parameters(isa)
    if not params:
        raise _Fail("the ISA circuit has no free parameters: transpile the "
                    "parametrized basis circuit once and bind the values in "
                    "the PUB.")
    if set(params) != set(_PARAM_NAMES):
        raise _Fail(f"the ISA circuit has the parameters {sorted(params)}; it "
                    "must keep gamma, theta and beta of the basis circuit.")
    if target is not None:
        findings = hw.check_isa([isa], target)
        if findings:
            first = findings[0].split(": ", 1)[-1]
            more = f" (and {len(findings) - 1} more)" if len(findings) > 1 else ""
            raise _Fail(f"not an ISA circuit for the backend: {first}{more}. "
                        "Transpile it with generate_preset_pass_manager("
                        "optimization_level=3, backend=backend).")
    n = inst.n
    pos, problem = _measurement_positions(isa, n, _FIX_1C)
    if problem:
        raise _Fail(problem)
    two_q = sum(1 for ci in isa.data if _is_two_qubit(ci))
    depth2 = isa.depth(_is_two_qubit)
    if two_q > CZ_BAR:
        raise _Fail(f"a valid ISA circuit, but it has {two_q} two-qubit gates; "
                    f"the bar is {CZ_BAR}. Transpile with optimization_level=3 "
                    "and several values of seed_transpiler, and keep the "
                    "circuit with the fewest cz.")
    ref_params = ref_pos = None
    if reference_circuit is not None:
        ref_params = _named_parameters(reference_circuit)
        ref_pos, problem = _measurement_positions(reference_circuit, n, _FIX_1A)
        if problem:
            raise _Fail("reference circuit: " + problem)
    for g, t, b in _PARAMS_1C:
        bound = isa.assign_parameters({params["gamma"]: g, params["theta"]: t,
                                       params["beta"]: b})
        got = _isa_distribution(bound, n, pos)
        if reference_circuit is None:
            ref = _reference_1a(inst, (g, t, b))
        else:
            ref = _bound_distribution(reference_circuit, ref_params, (g, t, b),
                                      ref_pos, n)
        d = float(np.max(np.abs(got - ref)))
        if d > TOL_1C:
            raise _Fail("the ISA circuit is not equivalent to the basis "
                        f"circuit: at (gamma, theta, beta) = {_fmt((g, t, b))} "
                        f"its output distribution differs by {d:.3g}. "
                        "Transpile the circuit that passed Exercise 1a and "
                        "keep measure_all() before transpiling.")
    return _result("1c", True, "valid ISA circuit, equivalent to the basis "
                   "circuit at two fixed parameter sets; "
                   f"{two_q} two-qubit gates (bar {CZ_BAR}), two-qubit depth "
                   f"{depth2}.")


# ------------------------------------------------------------ Exercise 1d
def _random_counts(n: int, k: int, size: int, seed: int) -> dict:
    """Seeded counts dict of ``size`` distinct n-bit strings (bit density
    k / n, so feasible strings occur) with counts 1..40."""
    rng = np.random.default_rng(seed)
    keys = []
    while len(keys) < size:
        bits = (rng.random(n) < k / n).astype(int)
        key = "".join(str(b) for b in bits[::-1])
        if key not in keys:
            keys.append(key)
    return {key: int(c) for key, c in zip(keys, rng.integers(1, 41, size))}


def _synthetic_1d(inst):
    yield ("synthetic counts on the training instance", inst,
           _random_counts(inst.n, inst.k, 60, 2026))
    a = ASYMMETRIC_INSTANCE
    yield ("synthetic counts on a 7-node instance with lam = 5", a,
           _random_counts(a.n, a.k, 40, 7))


def _evaluate(label: str, fn, inst, counts):
    out = _call(f"evaluate_counts ({label})", fn, inst, counts,
                call="evaluate_counts(inst, counts)")
    if not isinstance(out, (tuple, list)) or len(out) != 3:
        raise _Fail("evaluate_counts must return three arrays: values, "
                    "feasible, shots.")
    try:
        values = np.asarray(out[0], dtype=float).ravel()
        feasible = np.asarray(out[1]).ravel()
        shots = np.asarray(out[2], dtype=float).ravel()
    except (TypeError, ValueError):
        raise _Fail("evaluate_counts must return three numeric arrays "
                    "(values, feasible, shots).") from None
    m = len(counts)
    for name, arr in (("values", values), ("feasible", feasible),
                      ("shots", shots)):
        if arr.shape != (m,):
            raise _Fail(f"evaluate_counts returned {arr.size} {name} for {m} "
                        f"bitstrings ({label}): one entry per key of the "
                        "counts dictionary, in its order.")
    return values, feasible, shots


def _value_diagnoses(inst, bits, f, w):
    """Wrong values of f_lambda and their messages ("{label}" is filled in):
    the table of known evaluate_counts mistakes."""
    lam, k = inst.lam, inst.k
    fr, wr = inst.evaluate_bits(bits[:, ::-1])
    yield (fr - lam * (wr - k) ** 2, "bit order ({label}): Qiskit bitstrings are "
           "little-endian, so x[i] = bitstring[-1 - i]; reverse the string "
           "before reading node i.")
    yield (f, "values are f(x) ({label}); the challenge scores f_lambda(x) = "
           "f(x) - lam*(W - k)^2.")
    yield (f + lam * (w - k) ** 2, "the penalty has the wrong sign ({label}): "
           "f_lambda(x) = f(x) - lam*(W - k)^2.")
    yield (-(f - lam * (w - k) ** 2), "the values have the wrong sign "
           "({label}): f_lambda(x) = f(x) - lam*(W - k)^2, larger is better.")
    yield (f - (w - k) ** 2, "the penalty lacks the factor lam = inst.lam "
           "({label}).")
    if lam != 4.0:
        yield (f - 4.0 * (w - k) ** 2, "the penalty uses lam = 4, the value of "
               "the training instance ({label}); use inst.lam of the instance "
               "passed in.")
    if k != 4:
        yield (f - lam * (w - 4) ** 2, "the penalty uses k = 4, the value of "
               "the training instance ({label}); use inst.k of the instance "
               "passed in.")
    if any(wt != 1.0 for *_, wt in inst.edges):
        unit = DensestKSubgraph(n=inst.n, k=k, edges=tuple(
            (i, j, 1.0) for i, j, _ in inst.edges))
        f1, _ = unit.evaluate_bits(bits)
        yield (f1 - lam * (w - k) ** 2, "the edge weights are ignored "
               "({label}): f(x) sums w * x_i * x_j over the edges (i, j, w) of "
               "inst.edges.")
    yield (f - lam * (w - k), "the penalty is lam*(W - k) ({label}); it is "
           "lam*(W - k)^2.")
    yield (f - lam * np.abs(w - k), "the penalty is lam*|W - k| ({label}); it "
           "is lam*(W - k)^2.")
    yield (2 * f - lam * (w - k) ** 2, "each edge is counted twice ({label}): "
           "sum over inst.edges once.")


def _same_rows(got, want) -> bool:
    """The (value, feasible, shots) rows agree as a multiset: consistent
    entries in another order."""
    try:
        g = np.column_stack([np.round(got[0], 6), got[1].astype(float),
                             np.round(got[2], 6)])
        w = np.column_stack([np.round(want[0], 6), want[1].astype(float),
                             np.round(want[2], 6)])
    except (TypeError, ValueError):
        return False
    if g.shape != w.shape:
        return False
    g = g[np.lexsort(g.T[::-1])]
    w = w[np.lexsort(w.T[::-1])]
    return bool(np.allclose(g, w, rtol=0, atol=1e-6))


def _compare_evaluate(label: str, fn, inst, counts) -> tuple:
    from .hardware import bits_from_counts

    values, feasible, shots = _evaluate(label, fn, inst, counts)
    bits, cnt = bits_from_counts(counts, inst.n)
    f, w = inst.evaluate_bits(bits)
    flam = f - inst.lam * (w - inst.k) ** 2
    ref_feas = w == inst.k
    try:
        feas = feasible.astype(bool)
    except (TypeError, ValueError):
        raise _Fail("feasible must be booleans (W == k).") from None
    if (np.allclose(values, flam, rtol=0, atol=1e-9)
            and np.array_equal(feas, ref_feas) and np.allclose(shots, cnt)):
        return values, feas, shots
    if _same_rows((values, feas, shots), (flam, ref_feas, cnt)):
        return values, feas, shots             # consistent, in another order
    if not np.allclose(values, flam, rtol=0, atol=1e-9):
        for cand, text in _value_diagnoses(inst, bits, f, w):
            if np.allclose(values, cand, rtol=0, atol=1e-9):
                raise _Fail(text.format(label=label))
        i = int(np.argmax(np.abs(values - flam) > 1e-9))
        key = list(counts)[i]
        raise _Fail(f"values differ from f_lambda(x) ({label}): bitstring "
                    f"'{key}' gives {values[i]:g}, expected {flam[i]:g}.")
    if not np.array_equal(feas, ref_feas):
        raise _Fail(f"feasible must be True exactly when W(x) = k, W the "
                    f"number of ones in the bitstring ({label}).")
    raise _Fail(f"the third array must hold the shot count of each "
                f"bitstring, the values of the counts dictionary ({label}).")


def _cvar_cases():
    yield [1.0, 2.0], [1.0, 3.0], 0.5
    yield [1.0, 2.0], [1.0, 3.0], 0.8
    yield [3.0, 1.0, 2.0], [1.0, 1.0, 2.0], 1.0
    rng = np.random.default_rng(1729)
    for _ in range(40):
        m = int(rng.integers(2, 30))
        v = rng.integers(-20, 8, m).astype(float)          # ties included
        w = rng.exponential(1.0, m) * float(rng.choice([1.0, 7.0, 1000.0]))
        yield v.tolist(), w.tolist(), float(np.round(rng.uniform(0.03, 0.9), 4))


def _desc(v, w):
    v, w = np.asarray(v, float), np.asarray(w, float)
    order = np.argsort(v, kind="stable")[::-1]
    return v[order], w[order]


def _cv_boundary_full(v, w, a):
    vs, ws = _desc(v, w)
    ws = ws / ws.sum()
    j = min(int(np.searchsorted(np.cumsum(ws), a, side="left")), len(vs) - 1)
    return float(np.dot(vs[:j + 1], ws[:j + 1]) / ws[:j + 1].sum())


def _cv_unnormalized(v, w, a):
    vs, raw = _desc(v, w)
    before = np.concatenate([[0.0], np.cumsum(raw)[:-1]])
    return float((vs * np.clip(a - before, 0.0, raw)).sum() / a)


def _cv_unpermuted(v, w, a):
    vs = np.sort(np.asarray(v, float))[::-1]
    ws = np.asarray(w, float) / np.sum(w)
    before = np.cumsum(ws) - ws
    return float((vs * np.clip(a - before, 0.0, ws)).sum() / a)


def _cv_no_max(v, w, a):
    vs, ws = _desc(v, w)
    ws = ws / ws.sum()
    before = np.cumsum(ws) - ws
    return float((vs * np.minimum(ws, a - before)).sum() / a)


# known wrong CVaR implementations (value function, message)
_CVAR_VARIANTS = (
    (lambda v, w, a: _kit_cvar(v, w, a, tail="lower"),
     "the tail holds the worst values: CVaR averages the best (largest) "
     "values, so sort in descending order."),
    (_cv_boundary_full,
     "the sample that crosses alpha counts fully; it must count only with the "
     "part of its mass that is still missing, alpha - (mass before it)."),
    (_cv_unnormalized,
     "the weights are not normalized: divide them by their sum before "
     "accumulating up to alpha."),
    (lambda v, w, a: float(np.dot(v, w) / np.sum(w)),
     "the result is the mean of all samples, not the average over the best "
     "alpha-fraction."),
    (_cv_unpermuted,
     "the values are sorted but the weights are not: reorder both with the "
     "same order, order = np.argsort(values)[::-1]."),
    (_cv_no_max,
     "the mass taken from a sample goes negative after alpha is reached: take "
     "min(w, max(alpha - mass_before, 0))."),
    (lambda v, w, a: float(_kit_cvar(v, w, a) * a),
     "the sum is not divided by alpha: CVaR is (sum of value * mass taken) / "
     "alpha."),
    (lambda v, w, a: float(_kit_cvar(v, np.ones(len(v)), a)),
     "the weights are ignored: each value counts with its weight (its shot "
     "count)."),
    (lambda v, w, a: float(np.max(v)),
     "cvar returns the single best value; it must average the best "
     "alpha-fraction of the samples."),
)


def _cvar_value(fn, v, w, alpha) -> float:
    got = _call(f"cvar(values, weights, {alpha:g})", fn, list(v), list(w), alpha,
                call="cvar(values, weights, alpha)")
    try:
        return float(got)
    except (TypeError, ValueError):
        raise _Fail("cvar must return one number.") from None


def _confirm_cases():
    """Cases without ties (a wrong algorithm may depend on how ties are
    sorted) that a named cvar mistake must also reproduce."""
    rng = np.random.default_rng(4242)
    for _ in range(_CVAR_CONFIRM):
        m = int(rng.integers(4, 25))
        v = np.round(rng.normal(0.0, 5.0, m), 6)
        w = rng.exponential(1.0, m) * float(rng.choice([1.0, 7.0, 1000.0]))
        yield v.tolist(), w.tolist(), float(np.round(rng.uniform(0.05, 0.9), 4))


def _name_cvar_mistake(fn, cases, i: int, got: float) -> str | None:
    """The known mistake that reproduces ``got`` on case i and the
    participant's cvar on the ``_CVAR_CONFIRM`` tie-free cases, or None."""
    close = functools.partial(np.isclose, rtol=1e-9, atol=1e-9)
    v, w, a = cases[i]
    cands = [(f, t) for f, t in _CVAR_VARIANTS if close(got, f(v, w, a))]
    if not cands:
        return None
    others = list(_confirm_cases())
    outs = [_cvar_value(fn, *c) for c in others]
    for f, text in cands:
        if all(close(o, f(*c)) for o, c in zip(outs, others)):
            return text
    return None


def _check_cvar(fn) -> None:
    cases = list(_cvar_cases())
    for i, (v, w, alpha) in enumerate(cases):
        got = _cvar_value(fn, v, w, alpha)
        want = _kit_cvar(v, w, alpha)
        if np.isclose(got, want, rtol=1e-9, atol=1e-9):
            continue
        case = (f"cvar({[round(x, 3) for x in v][:5]}{' ...' if len(v) > 5 else ''}, "
                f"weights, alpha={alpha:g}) returned {got:.6g}, expected "
                f"{want:.6g}")
        text = _name_cvar_mistake(fn, cases, i, got) if alpha < 1.0 else None
        if text:
            raise _Fail(f"{case}: {text}")
        raise _Fail(f"{case}. Sort by value (best first), normalize the "
                    "weights and take from each sample min(w, max(alpha - "
                    "mass_before, 0)).")


def job_facts(job, result=None, assigned_backend=None, n=None):
    """``(record, counts)`` of a finished Sampler job for :func:`check_1d`.

    ``record``: ``job_id``, ``backend``, ``assigned_backend``, ``status``,
    ``n_pubs``, ``sets_per_pub`` and ``shots_per_set``; ``counts``: one
    counts dict per parameter set (PUB by PUB, parameter sets in C order).
    ``result`` defaults to ``job.result()``; with ``n`` every PUB must hold
    n-bit outcomes. Raises ``_Fail`` with a participant-facing message if
    the job cannot be read."""
    jid = job.job_id() if callable(getattr(job, "job_id", None)) else None
    try:
        status = job.status()
    except Exception as exc:                     # noqa: BLE001 - network, fakes
        raise _Fail(f"the job's status cannot be read: {_one_line(exc)}") from None
    status = str(getattr(status, "name", status)).upper()
    if status in _TERMINAL:
        raise _Fail(f"{'job ' + str(jid) if jid else 'the job'} ended with "
                    f"status {status} and will not finish: look at "
                    "job.error_message(), then submit the job again (a new "
                    "run_once tag, or campaign.forget(tag) first).")
    if status != "DONE":
        raise _Fail(f"the job has status {status}; wait until it is DONE "
                    "(job.status()) and run the check again.")
    try:
        backend = job.backend()
        backend = getattr(backend, "name", backend)
        backend = backend() if callable(backend) else backend
    except Exception:                            # noqa: BLE001
        backend = None
    try:
        result = job.result() if result is None else result
    except Exception as exc:                     # noqa: BLE001
        raise _Fail(f"the job's result cannot be read: {_one_line(exc)}") from None
    counts, sets, shots = [], [], []
    for i, pub in enumerate(result):
        data = pub.data
        names = list(data.keys()) if hasattr(data, "keys") else list(vars(data))
        regs = [nm for nm in names if hasattr(getattr(data, nm), "get_counts")]
        if not regs:
            raise _Fail(f"PUB {i} of the job has no classical register.")
        if "meas" in regs:
            reg = "meas"
        elif len(regs) == 1:
            reg = regs[0]
        else:
            raise _Fail(f"PUB {i} of the job has the classical registers "
                        f"{regs}; measure into one register with "
                        "qc.measure_all().")
        ba = getattr(data, reg)
        width = getattr(ba, "num_bits", None)
        if n is not None and width is not None and int(width) != int(n):
            raise _Fail(f"PUB {i} of the job has {width}-bit outcomes (register "
                        f"{reg!r}); the circuit must measure the n = {n} qubits "
                        f"into one register of {n} bits: QuantumCircuit(inst.n) "
                        "and qc.measure_all().")
        shape = tuple(getattr(ba, "shape", ()) or ())
        locs = list(np.ndindex(shape)) if shape else [None]
        sets.append(len(locs))
        for loc in locs:
            c = ba.get_counts() if loc is None else ba.get_counts(loc)
            counts.append(dict(c))
            shots.append(int(sum(c.values())))
    record = {"job_id": jid, "backend": None if backend is None else str(backend),
              "assigned_backend": assigned_backend, "status": status,
              "n_pubs": len(sets), "sets_per_pub": sets, "shots_per_set": shots}
    return record, counts


def _anchor_facts(verify_job, job_id) -> dict:
    """What the IBM Quantum service reports about the job (the job anchor's
    metadata record), or ``{"error": ...}`` when it cannot say."""
    if not job_id:
        return {"error": "the job has no id"}
    try:
        a = verify_job(str(job_id))
    except Exception as exc:                     # noqa: BLE001 - network, 404
        return {"error": _one_line(exc) or type(exc).__name__}
    return dict(a) if a else {"error": "no record"}


def _check_anchor(rec: dict, jid: str) -> None:
    a = rec.get("anchor")
    if a is None:
        return
    if "error" in a:
        raise _Fail(f"job {jid} could not be verified with the IBM Quantum "
                    f"service ({a['error']}): pass the job that sampler.run "
                    "returned for your assigned backend.")
    assigned = rec.get("assigned_backend")
    if assigned and a.get("backend") != assigned:
        raise _Fail(f"the IBM Quantum service reports that job {jid} ran on "
                    f"{a.get('backend')!r}; run it on your assigned backend "
                    f"{assigned!r}.")
    status = str(a.get("status") or "").upper()
    if status and status != "DONE":
        raise _Fail(f"the IBM Quantum service reports status {status} for job "
                    f"{jid}; the check needs a job that is DONE.")
    prim = a.get("primitive_id")
    if prim is not None and prim not in ("sampler", "executor"):
        raise _Fail(f"job {jid} is not a Sampler job (the service reports "
                    f"{prim!r}): run it with the Sampler.")
    sets = list(rec.get("sets_per_pub") or [])
    if a.get("n_pubs") is not None and int(a["n_pubs"]) not in (
            rec.get("n_pubs"), sum(sets)):
        raise _Fail(f"the IBM Quantum service reports {a['n_pubs']} PUBs for "
                    f"job {jid}, but the job passed here holds "
                    f"{rec.get('n_pubs')} PUBs with {sum(sets)} parameter sets: "
                    "pass the job itself.")
    total = sum(rec.get("shots_per_set") or [])
    if a.get("shots") is not None and int(a["shots"]) != total:
        raise _Fail(f"the IBM Quantum service reports {a['shots']} shots for "
                    f"job {jid}, but the job passed here holds {total}: pass "
                    "the job itself.")


def _check_job_basics(rec: dict) -> str:
    """What every job of a check must satisfy: DONE, on the assigned
    backend, and (with a job anchor) the service's record of it. Returns
    the job id for the messages; raises ``_Fail``."""
    jid = rec.get("job_id") or "the job"
    status = str(rec.get("status", "")).upper()
    if status in _TERMINAL:
        raise _Fail(f"job {jid} ended with status {status} and will not "
                    "finish: submit the job again.")
    if status != "DONE":
        raise _Fail(f"job {jid} has status {rec.get('status')}; wait until it "
                    "is DONE.")
    assigned = rec.get("assigned_backend")
    if assigned and rec.get("backend") is None:
        raise _Fail(f"the backend of job {jid} cannot be read (a local "
                    "simulator job?): run the job with Sampler(mode=backend) "
                    f"on your assigned backend {assigned!r}.")
    if assigned and rec.get("backend") != assigned:
        raise _Fail(f"job {jid} ran on {rec.get('backend')!r}; run it on your "
                    f"assigned backend {assigned!r}.")
    _check_anchor(rec, jid)
    return jid


def _check_job_record(rec: dict, inst) -> None:
    jid = _check_job_basics(rec)
    if rec.get("n_pubs") != 2:
        raise _Fail(f"job {jid} has {rec.get('n_pubs')} PUBs; the exercise "
                    "asks for two: the ensemble (n+1 parameter sets) and the "
                    "single trained circuit.")
    sets = list(rec.get("sets_per_pub") or [])
    if len(sets) != 2 or sets[0] != inst.n + 1:
        raise _Fail(f"the first PUB of job {jid} holds {sets[0] if sets else 0} "
                    f"parameter sets; it must hold the n+1 = {inst.n + 1} "
                    "branches of the ensemble.")
    if sets[1] != 1:
        raise _Fail(f"the second PUB of job {jid} holds {sets[1]} parameter "
                    "sets; it must hold one, the single trained circuit.")
    shots = list(rec.get("shots_per_set") or [])
    if not shots or min(shots) < MIN_SHOTS_1D:
        raise _Fail(f"job {jid} has {min(shots) if shots else 0} shots in a "
                    f"parameter set; use at least {MIN_SHOTS_1D} per parameter "
                    "set.")


def _check_functions_1d(inst, evaluate_counts, cvar) -> None:
    """The reference-data part of the 1d check (raises ``_Fail``)."""
    if not callable(evaluate_counts) or not callable(cvar):
        raise _Fail("pass the functions evaluate_counts and cvar themselves.")
    for label, tinst, tcounts in _synthetic_1d(inst):
        _compare_evaluate(label, evaluate_counts, tinst, tcounts)
    _check_cvar(cvar)


@_guarded("1d", fail=lambda text: _functions_result(
    False, text.rstrip(".") + f". The job part is checked by {CALL_1D}."))
def check_1d_functions(inst: DensestKSubgraph, evaluate_counts,
                       cvar) -> CheckResult:
    """Exercise 1d before the job is done: ``evaluate_counts`` and ``cvar``
    on the reference data of :func:`check_1d` only (no job). The result
    carries no points and is not recorded (``record=False``); the full
    check, which records the points, needs the job as well."""
    _check_functions_1d(inst, evaluate_counts, cvar)
    return _functions_result(True, "evaluate_counts and cvar agree with the "
                             "reference on synthetic data. The job part is "
                             f"checked by {CALL_1D}, which records the "
                             "points.")


@_guarded("1d")
def check_1d(inst: DensestKSubgraph, job_record, counts, evaluate_counts,
             cvar) -> CheckResult:
    """Exercise 1d: ``evaluate_counts(inst, counts) -> (values, feasible,
    shots)`` and ``cvar(values, weights, alpha)`` on seeded synthetic data
    (the training instance and ``ASYMMETRIC_INSTANCE``, which exposes
    bit-order mistakes and code that fixes lam or ignores the weights; CVaR
    cases with a fractional boundary sample), then the job: DONE, on the
    assigned backend, two PUBs (n+1 parameter sets and one), at least 1,000
    shots per parameter set, the service's record of the job when
    ``job_record["anchor"]`` is present, and both functions on every counts
    dict of the job against ``hw.metrics_from_counts``. ``job_record`` and
    ``counts`` come from :func:`job_facts`. Entries of ``evaluate_counts``
    in another consistent order are accepted."""
    from .hardware import metrics_from_counts

    _check_functions_1d(inst, evaluate_counts, cvar)
    if job_record is None or counts is None:
        raise _Fail("evaluate_counts and cvar are correct on synthetic data; "
                    "no job was given: run the job cell and pass the job.")
    _check_job_record(job_record, inst)
    counts = list(counts)
    for i, c in enumerate(counts):
        label = f"parameter set {i} of job {job_record.get('job_id')}"
        values, _, shots = _compare_evaluate(label, evaluate_counts, inst, c)
        got = _call(f"cvar on {label}", cvar, values, shots, ALPHA,
                    call="cvar(values, shots, 0.1)")
        try:
            got = float(got)
        except (TypeError, ValueError):
            raise _Fail("cvar must return one number.") from None
        kit = metrics_from_counts(inst, c, ALPHA)
        if not np.isclose(got, kit["CVaR"], rtol=1e-9, atol=1e-9):
            raise _Fail(f"cvar on {label} returned {got:.6g}, expected "
                        f"{kit['CVaR']:.6g}.")
    shots = job_record.get("shots_per_set") or [0]
    verified = "; verified with the IBM Quantum service" \
        if job_record.get("anchor") else ""
    return _result("1d", True, "evaluate_counts and cvar agree with the "
                   "reference on synthetic data and on the "
                   f"{len(counts)} counts dictionaries of job "
                   f"{job_record.get('job_id')} (ran on "
                   f"{job_record.get('backend')}, 2 PUBs, at least "
                   f"{min(shots)} shots per parameter set{verified}).")


# ================================================================ Part 2
def _real(x) -> float | None:
    """``x`` as a float when it is one real number (a Python or NumPy
    scalar, or an array of one element; a complex number with zero
    imaginary part), else None. Booleans and strings are not numbers."""
    if x is None or isinstance(x, (bool, np.bool_, str, bytes)):
        return None
    try:
        arr = np.asarray(x)
    except Exception:                            # noqa: BLE001 - participant object
        return None
    if arr.size != 1 or arr.dtype.kind not in "iufc":
        return None
    v = arr.reshape(-1)[0]
    if arr.dtype.kind == "c":
        if v.imag != 0:
            return None
        v = v.real
    return float(v)


def _kind(x) -> str:
    """A short description of a participant's value for a message."""
    if x is None:
        return "None"
    if isinstance(x, np.ndarray):
        return f"an array of shape {x.shape}"
    if isinstance(x, (list, tuple)):
        return f"a {type(x).__name__} of {_plural(len(x), 'value')}"
    return f"a {type(x).__name__}"


def _close_pair(got, cvar_value: float, f_value: float) -> bool:
    return (abs(got[0] - cvar_value) <= TOL_CVAR_2
            and abs(got[1] - f_value) <= TOL_F_2)


def _registration(registered, jid: str):
    """The grader's row of job ``jid`` when it is a scored registration of
    the team, else ``_NotRecorded``. ``registered`` maps a job id to the
    row of the grader's jobs view (None when unknown); without it the test
    is skipped."""
    if registered is None:
        return None
    try:
        row = registered(str(jid))
    except Exception as exc:                     # noqa: BLE001 - grader, network
        raise _NotRecorded(f"the grader cannot confirm the registration of job "
                           f"{jid} ({_one_line(exc)}); run the cell again."
                           ) from None
    if not row:
        raise _NotRecorded(f"job {jid} is not registered yet: register it with "
                           "campaign.submit_job (if your next slot is later, "
                           "run the cell again when it opens).")
    seq = row.get("seq")
    if row.get("status") == "void":
        raise _NotRecorded(f"job {jid} (job #{seq}) is void ("
                           f"{row.get('void_reason') or 'no reason given'}): run "
                           "the job again with a new tag, register it and check "
                           "the new job.")
    if row.get("status") != "scored":
        raise _NotRecorded(f"job {jid} (job #{seq}) is registered but not scored "
                           "yet: run campaign.submit_job with this job again to "
                           "finish its registration, then this cell.")
    return row


def _job_text(rec: dict, shape: str) -> str:
    verified = "; verified with the IBM Quantum service" if rec.get("anchor") else ""
    return f"job {rec.get('job_id')} (ran on {rec.get('backend')}, {shape}{verified})"


# ------------------------------------------------------------ Exercise 2a
# (mean Hamming weight - k, sd of the weight) of the three reference dicts
_SHAPES_2A = ((0.0, 2.0), (3.0, 3.0), (-1.5, 1.5))


def _instance_seed(inst) -> int:
    """A seed fixed by the instance (n, k, lam and the weighted edges)."""
    import hashlib

    body = repr((int(inst.n), int(inst.k), float(inst.lam),
                 sorted((int(i), int(j), float(w)) for i, j, w in inst.edges)))
    return int(hashlib.sha256(body.encode()).hexdigest()[:12], 16)


def _reference_counts_2a(inst) -> list[dict]:
    """The three reference counts dicts of the 2a check: ``inst.n``-bit
    strings with Hamming weights drawn around k (``_SHAPES_2A``), ``SIZE_2A``
    distinct strings each (fewer on a small instance), 1 to 6 shots each,
    so that a feasible fraction of shots and one of distinct strings
    differ; seeded from the instance, so the check is reproducible."""
    rng = np.random.default_rng(_instance_seed(inst))
    n, k = inst.n, inst.k
    size = min(SIZE_2A, 2 ** min(n, 30) // 4)
    out = []
    for offset, sd in _SHAPES_2A:
        counts: dict = {}
        for _ in range(50 * size):
            if len(counts) >= size:
                break
            w = int(np.clip(np.rint(rng.normal(k + offset, sd)), 0, n))
            bits = np.zeros(n, dtype=np.uint8)
            bits[rng.choice(n, w, replace=False)] = 1
            key = "".join("1" if b else "0" for b in bits[::-1])
            if key not in counts:
                counts[key] = int(rng.integers(1, 7))
        out.append(counts)
    return out


def _score_pair(out, call: str) -> tuple:
    """What ``my_score`` returned, as ``(CVaR, F)``; raises ``_Fail`` when
    it is not a pair of two real numbers."""
    fix = "my_score returns two numbers: return tail, F"
    if out is None:
        raise _Fail(f"{call} returned None; end my_score with return tail, F.")
    if isinstance(out, Mapping):
        raise _Fail(f"{call} returned a dict; {fix}.")
    if isinstance(out, (str, bytes)):
        raise _Fail(f"{call} returned a string; {fix}.")
    if _real(out) is not None:
        raise _Fail(f"{call} returned one number; {fix} (the CVaR_alpha and the "
                    "feasible fraction).")
    try:
        items = list(np.ravel(out)) if isinstance(out, np.ndarray) else list(out)
    except TypeError:
        raise _Fail(f"{call} returned {_kind(out)}; {fix}.") from None
    if len(items) != 2:
        raise _Fail(f"{call} returned {_plural(len(items), 'value')}; {fix}.")
    vals = [_real(x) for x in items]
    for pos, x, v in zip(("first", "second"), items, vals):
        if v is None:
            raise _Fail(f"{call} returned {_kind(x)} as its {pos} value; {fix}.")
    return float(vals[0]), float(vals[1])


def _same_instance(a, b) -> bool:
    return ((a.n, a.k, float(a.lam), sorted(a.edges))
            == (b.n, b.k, float(b.lam), sorted(b.edges)))


def _uses_training(fn, inst) -> bool:
    """True when ``fn`` scores 12-bit counts of the training instance as
    the kit does: my_score uses the 12-qubit instance of Part 1 in place of
    the campaign instance ``inst``."""
    from .event import training_instance
    from .hardware import metrics_from_counts

    small = training_instance()
    if _same_instance(small, inst):
        return False
    counts = _random_counts(small.n, small.k, 60, 2026)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            got = _score_pair(fn(counts), "my_score")
    except Exception:                            # noqa: BLE001 - participant code
        return False
    want = metrics_from_counts(small, counts, ALPHA)
    return _close_pair(got, want["CVaR"], want["F"])


def _training_text(inst) -> str:
    return ("my_score uses the 12-qubit training instance of Part 1: use inst106 "
            "(inst106 = campaign.instance) in your evaluate_counts; its "
            f"bitstrings have {inst.n} bits.")


def _my_score(fn, inst, counts: dict, alpha) -> tuple:
    """``fn(counts)`` (``alpha`` None) or ``fn(counts, alpha=alpha)``, its
    printed output held back, as ``(CVaR, F)``; raises ``_Fail``."""
    call = "my_score(counts)" if alpha is None else f"my_score(counts, alpha={alpha:g})"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = fn(counts) if alpha is None else fn(counts, alpha=alpha)
    except NotImplementedError:
        raise _Fail("my_score is not implemented yet (it raises "
                    "NotImplementedError).") from None
    except Exception as exc:                     # noqa: BLE001 - participant code
        if _uses_training(fn, inst):
            raise _Fail(_training_text(inst)) from None
        raise _Fail(f"{call} raised {type(exc).__name__}: "
                    f"{_one_line(exc) or 'no message'}; call {call} in a cell of "
                    "its own to see the full traceback.") from None
    return _score_pair(out, call)


def _cvar_variants_2a(flam, cnt, alpha):
    """Wrong CVaR values of my_score and their messages."""
    fix = "tail = cvar(values, shots, alpha)"
    yield (float(np.dot(flam, cnt) / cnt.sum()),
           "is the mean of f_lambda over all shots; CVaR_alpha averages only "
           f"the best alpha-fraction of the shots: {fix}")
    yield (_kit_cvar(flam, cnt, alpha, tail="lower"),
           "averages the worst alpha-fraction of the shots (the lower tail); "
           f"CVaR averages the best (largest) values: {fix}")
    yield (_kit_cvar(flam, np.ones_like(cnt), alpha),
           f"counts every distinct bitstring once; weight each value by its shots: {fix}")


def _f_variants_2a(feas, cnt):
    """Wrong feasible fractions of my_score and their messages."""
    fix = "F = shots[feasible].sum() / shots.sum()"
    yield (float(feas.sum() / len(feas)),
           "is the fraction of distinct bitstrings that are feasible; F is the "
           f"fraction of the shots: {fix}")
    yield (float(cnt[feas].sum()),
           f"is the number of feasible shots, not a fraction: divide it by all shots, {fix}")
    yield (float(feas.sum()),
           f"is the number of feasible bitstrings, not a fraction of the shots: {fix}")


def _diagnose_2a(inst, fn, counts: dict, got: tuple, alpha: float, i: int) -> str:
    """The message for a my_score result that differs from the kit's on
    reference counts dict ``i``: a named mistake when one reproduces it."""
    from .hardware import bits_from_counts

    bits, cnt = bits_from_counts(counts, inst.n)
    f, w = inst.evaluate_bits(bits)
    flam = f - inst.lam * (w - inst.k) ** 2
    feas = w == inst.k
    want_cv, want_F = _kit_cvar(flam, cnt, alpha), float(cnt[feas].sum() / cnt.sum())
    cv, F = got
    a = f"{alpha:.2f}"
    where = (f"on reference counts {i + 1} of your instance ({len(counts)} distinct "
             f"bitstrings, {int(cnt.sum())} shots) at alpha = {a}")
    if abs(cv - want_F) <= TOL_F_2 and abs(F - want_cv) <= TOL_CVAR_2:
        return (f"{where}, my_score returned ({cv:.6g}, {F:.6g}): the CVaR and F "
                "are swapped; my_score returns (CVaR_alpha, F), the CVaR first: "
                "return tail, F.")
    parts, unnamed = [], False
    if abs(cv - want_cv) > TOL_CVAR_2:
        text = next((t for v, t in _cvar_variants_2a(flam, cnt, alpha)
                     if abs(cv - v) <= TOL_CVAR_2), None)
        unnamed |= text is None
        parts.append(f"the CVaR_{a} returned ({cv:.6g}) {text}" if text else
                     f"the CVaR_{a} returned is {cv:.6g}, the kit's {want_cv:.6g}")
    if abs(F - want_F) > TOL_F_2:
        text = next((t for v, t in _f_variants_2a(feas, cnt)
                     if abs(F - v) <= TOL_F_2), None)
        unnamed |= text is None
        parts.append(f"the F returned ({F:.6g}) {text}" if text else
                     f"the F returned is {F:.6g}, the kit's {want_F:.6g}")
    if unnamed and _uses_training(fn, inst):
        return _training_text(inst)
    out = f"{where}: " + "; ".join(parts) + "."
    if unnamed:
        out += (" Compute values, feasible and shots with your evaluate_counts("
                "inst106, counts), then tail = cvar(values, shots, alpha) and "
                "F = shots[feasible].sum() / shots.sum().")
    return out


@_guarded("2a")
def check_2a(inst: DensestKSubgraph, my_score) -> CheckResult:
    """Exercise 2a: ``my_score(counts, alpha=0.10) -> (CVaR_alpha of
    f_lambda, feasible fraction F)`` for one counts dict of the campaign
    instance ``inst``. Deciding test: agreement with
    ``hw.metrics_from_counts`` (CVaR within ``TOL_CVAR_2``, F within
    ``TOL_F_2``) on the three reference counts dicts of
    :func:`_reference_counts_2a` at the default alpha, and on the first at
    alpha = ``ALPHA_2A``. Named mistakes: not a pair of two numbers, CVaR
    and F swapped, alpha ignored, the mean, the lower tail or unweighted
    values for the CVaR, F over distinct bitstrings or as a count, and the
    12-qubit instance in place of the campaign instance."""
    from .hardware import metrics_from_counts

    if not callable(my_score):
        raise _Fail(f"pass the function my_score itself, not its result: {CALL_2A}.")
    cases = _reference_counts_2a(inst)
    first = None
    for i, counts in enumerate(cases):
        want = metrics_from_counts(inst, counts, ALPHA)
        got = _my_score(my_score, inst, counts, None)
        if not _close_pair(got, want["CVaR"], want["F"]):
            raise _Fail(_diagnose_2a(inst, my_score, counts, got, ALPHA, i))
        first = got if first is None else first
    want = metrics_from_counts(inst, cases[0], ALPHA_2A)
    got = _my_score(my_score, inst, cases[0], ALPHA_2A)
    if not _close_pair(got, want["CVaR"], want["F"]):
        if (abs(got[0] - first[0]) <= TOL_CVAR_2
                and abs(want["CVaR"] - first[0]) > TOL_CVAR_2):
            raise _Fail(f"my_score ignores alpha: my_score(counts, alpha="
                        f"{ALPHA_2A:g}) returned the CVaR at alpha = {ALPHA:.2f} "
                        f"({got[0]:.6g}) on reference counts 1; the kit's "
                        f"CVaR_{ALPHA_2A:.2f} is {want['CVaR']:.6g}. Pass alpha "
                        "on to cvar: tail = cvar(values, shots, alpha).")
        raise _Fail(_diagnose_2a(inst, my_score, cases[0], got, ALPHA_2A, 0))
    sizes = sorted({len(c) for c in cases})
    size = str(sizes[0]) if len(sizes) == 1 else f"{sizes[0]} to {sizes[-1]}"
    return _result("2a", True, "my_score agrees with hw.metrics_from_counts on "
                   "three reference counts dictionaries of your instance "
                   f"({size} distinct bitstrings each) at alpha = {ALPHA:.2f}, "
                   f"and at alpha = {ALPHA_2A:.2f}.")


# ------------------------------------------------------------ Exercise 2b
_THETA_C = ("theta_c = hw.theta_for_weight(inst106, gamma, beta, "
            "target=inst106.k - shift)")
_STATS = ("pass the stats that nb.weight_histogram(inst106, counts, ideal=info) "
          "returned in Step 3")
_ANGLE_TAIL = f" The points are recorded by {CALL_2B} after the job."


def _stats_like(x) -> bool:
    """The Report of ``nb.weight_histogram`` (a dict) or a ``hw.WeightStats``."""
    return isinstance(x, Mapping) or (
        isinstance(x, tuple) and hasattr(x, "mean") and not callable(x.mean))


def _measured_mean(stats) -> float:
    """The measured mean Hamming weight in ``stats``: the Report of
    ``nb.weight_histogram`` or a dict with ``mean_W``, a ``hw.WeightStats``,
    or the mean itself."""
    if stats is None:
        raise _Fail(f"stats has no measured mean (stats is None): {_STATS}.")
    if isinstance(stats, Mapping):
        if "mean_W" not in stats:
            if "exp_W" in stats:
                raise _Fail("stats holds the ideal moments of hw.lagrangian_"
                            f"layers (exp_W), not a measured mean: {_STATS}.")
            raise _Fail(f"stats has no measured mean (no key \"mean_W\"): {_STATS}.")
        value = _real(stats["mean_W"])
    elif _stats_like(stats):
        value = _real(stats.mean)
    else:
        value = _real(stats)
        if value is None:
            raise _Fail(f"stats must hold the measured mean Hamming weight, got "
                        f"{_kind(stats)}: {_STATS}.")
    if value is None or not np.isfinite(value):
        raise _Fail(f"the measured mean in stats is not a finite number: {_STATS}.")
    return float(value)


def _angle_value(theta_c) -> float:
    if theta_c is None:
        raise _Fail(f"theta_c is None: {_THETA_C}.")
    v = _real(theta_c)
    if v is None:
        raise _Fail(f"theta_c must be one real number, got {_kind(theta_c)}: "
                    f"{_THETA_C}.")
    if not np.isfinite(v):
        raise _Fail(f"theta_c is not a finite number ({v}): {_THETA_C}.")
    return v


def _same_angle(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(np.angle(np.exp(1j * (a - b))))) <= tol


def _angle_2b(inst, theta_c, stats) -> dict:
    """The angle test of Exercise 2b (raises ``_Fail``): with the kit's
    transfer ``layers, info = hw.lagrangian_layers(inst)``, ``(gamma, theta,
    beta) = layers[0]`` and ``shift = mean_W - info["exp_W"]``, the ideal
    mean weight at ``theta_c`` is k - shift within ``TOL_WEIGHT_2B``.
    Returns the numbers of the messages; ``note`` is a clause when
    ``theta_c`` is the crossing of the target with the smaller ideal <f>."""
    from . import hardware as hw

    mean = _measured_mean(stats)
    tc = _angle_value(theta_c)
    layers, info = hw.lagrangian_layers(inst)
    gamma, theta, beta = (float(x) for x in layers[0])
    ideal = float(info["exp_W"])
    k = inst.k
    shift = mean - ideal
    target = k - shift
    at_c = hw.layer_expectations(inst, gamma, tc, beta)
    got = float(at_c["exp_W"])
    facts = {"mean": mean, "theta": theta, "theta_c": tc, "shift": shift,
             "target": target, "exp_W": got, "note": ""}
    aim = f"k - shift = {k} - ({shift:+.2f}) = {target:.2f}"
    if abs(got - target) <= TOL_WEIGHT_2B:
        try:
            best = hw.theta_for_weight(inst, gamma, beta, target)
        except ValueError:
            return facts
        f_c = float(at_c["exp_f"])
        f_best = float(hw.layer_expectations(inst, gamma, best, beta)["exp_f"])
        if f_c < f_best - 1e-6 and not _same_angle(tc, best, 1e-6):
            facts["note"] = (f" (the other crossing of the target, theta = "
                             f"{best:.4f}, has the larger ideal <f>, {f_best:.2f} "
                             f"against {f_c:.2f}; hw.theta_for_weight returns it)")
        return facts
    if _same_angle(tc, theta):
        raise _Fail(f"theta_c equals the baseline theta ({theta:.4f}), so nothing "
                    f"was changed; aim at {aim}: {_THETA_C}.")
    if abs(got - (k + shift)) <= TOL_WEIGHT_2B:
        raise _Fail(f"the ideal mean weight at theta_c is {got:.2f} = k + shift: "
                    "the shift has the wrong sign. The measured mean weight lies "
                    f"{shift:+.2f} from the ideal one, so aim at {aim}: {_THETA_C}.")
    if abs(got - k) <= TOL_WEIGHT_2B:
        raise _Fail(f"the ideal mean weight at theta_c is {got:.2f} = k: the "
                    f"shift was not applied; aim at {aim}: {_THETA_C}.")
    raise _Fail(f"the ideal mean weight at theta_c = {tc:.4f} is {got:.2f}; the "
                f"target is {aim} (shift = the measured mean weight {mean:.2f} "
                f"minus the ideal {ideal:.2f}): {_THETA_C}.")


@_guarded("2b", fail=lambda text: _preview(
    "2b", "angle only", False, text.rstrip(".") + "." + _ANGLE_TAIL))
def check_2b_angle(inst: DensestKSubgraph, theta_c, stats) -> CheckResult:
    """Exercise 2b before the job: the angle test of :func:`check_2b`
    alone. The result carries no points and is not recorded."""
    a = _angle_2b(inst, theta_c, stats)
    return _preview("2b", "angle only", True,
                    f"shift = {a['shift']:+.2f}, theta {a['theta']:.4f} -> theta_c "
                    f"{a['theta_c']:.4f}, whose ideal mean weight {a['exp_W']:.2f} "
                    f"equals the target k - shift = {a['target']:.2f}{a['note']}."
                    + _ANGLE_TAIL)


@_guarded("2b")
def check_2b(inst: DensestKSubgraph, job_record, counts, theta_c, stats,
             registered=None) -> CheckResult:
    """Exercise 2b: the angle test (:func:`_angle_2b`), then the job
    (``job_record`` and ``counts`` from :func:`job_facts`): DONE, on the
    assigned backend, the service's record when ``job_record["anchor"]``
    is present; one PUB with one circuit and at least ``MIN_SHOTS_2``
    shots; a measured mean Hamming weight closer to k than the baseline
    mean in ``stats``; and a scored registration of the team
    (``registered``, see :func:`_registration`; until then nothing is
    recorded)."""
    from .hardware import weight_stats

    a = _angle_2b(inst, theta_c, stats)
    if job_record is None or counts is None:
        raise _Fail(f"theta_c is correct; no job was given: run the job cell and "
                    f"pass the job, {CALL_2B}.")
    jid = _check_job_basics(job_record)
    sets = list(job_record.get("sets_per_pub") or [])
    if job_record.get("n_pubs") != 1:
        raise _Fail(f"job {jid} has {job_record.get('n_pubs')} PUBs; the exercise "
                    "asks for one, the circuit with theta_c: campaign.run_once("
                    "\"recentred\", sampler, [(isa_c,)], shots=2**15).")
    if sets != [1]:
        raise _Fail(f"the PUB of job {jid} holds {sets[0] if sets else 0} parameter "
                    "sets; it must hold one circuit, built with theta_c in its "
                    "layer tuple, and no parameter values.")
    shots = int((job_record.get("shots_per_set") or [0])[0])
    if shots < MIN_SHOTS_2:
        raise _Fail(f"job {jid} has {shots} shots; use at least {MIN_SHOTS_2} (the "
                    "exercise uses 2**15).")
    mean_c = float(weight_stats(inst, list(counts)[0]).mean)
    base, k = a["mean"], inst.k
    if not abs(mean_c - k) < abs(base - k):
        raise _Fail(f"the measured mean Hamming weight of job {jid} is "
                    f"{mean_c:.2f}, no closer to k = {k} than the baseline's "
                    f"{base:.2f}: check that the job ran the circuit built with "
                    "theta_c (qc_c, isa_c and a new tag), and the sign of the "
                    "shift.")
    row = _registration(registered, jid)
    reg = f"; registered as job #{row['seq']}" if row else ""
    return _result("2b", True, f"theta_c = {a['theta_c']:.4f} gives the ideal mean "
                   f"weight {a['exp_W']:.2f} = k - shift{a['note']}; "
                   f"{_job_text(job_record, f'one circuit, {shots} shots')} has the "
                   f"measured mean weight {mean_c:.2f}, closer to k = {k} than the "
                   f"baseline's {base:.2f}{reg}.")


# ------------------------------------------------------------ Exercise 2c
_SCORES_2C = ("scores_s = [my_score(data_s.c.get_counts(i)) for i in "
              "range(len(thetas_sweep))]")
_PUB_2C = "pub_s = (isa_s, {\"theta\": thetas_sweep}, 4096)"


def _sweep_values(thetas, m: int) -> np.ndarray:
    """``thetas_sweep`` as m finite, pairwise different floats."""
    arr = None
    if thetas is not None and not isinstance(thetas, (str, bytes, Mapping)):
        try:
            arr = np.asarray(thetas, dtype=float).ravel()
        except (TypeError, ValueError):
            arr = None
    if arr is None:
        raise _Fail(f"thetas_sweep must be the {m} values of theta of the PUB, got "
                    f"{_kind(thetas)}.")
    if arr.size != m:
        raise _Fail(f"thetas_sweep holds {_plural(arr.size, 'value')}; the PUB of the "
                    f"job holds {m} parameter sets: pass the values of theta of the "
                    "PUB, in its order.")
    if not np.all(np.isfinite(arr)):
        raise _Fail("thetas_sweep holds values that are not finite numbers: pass the "
                    "values of theta of the PUB.")
    srt = np.sort(arr)
    same = np.diff(srt) <= 1e-12
    if np.any(same):
        raise _Fail(f"thetas_sweep holds the value {srt[int(np.argmax(same))]:.6g} "
                    f"more than once; a sweep needs {m} different values of theta, "
                    "e.g. np.linspace(theta_c - 0.20, theta_c + 0.20, 8).")
    return arr


def _score_pairs(scores, m: int) -> list:
    """``scores_s`` as m pairs of floats (raises ``_Fail``)."""
    bad = (scores is None or isinstance(scores, (str, bytes, Mapping))
           or _real(scores) is not None)
    items = None
    if not bad:
        try:
            items = list(scores)
        except TypeError:
            items = None
    if items is None:
        raise _Fail(f"scores_s must be a list of {m} pairs (CVaR_0.10, F), one per "
                    f"setting, got {_kind(scores)}: {_SCORES_2C}.")
    if len(items) != m:
        entries = "1 entry" if len(items) == 1 else f"{len(items)} entries"
        raise _Fail(f"scores_s holds {entries}; the job has {m} settings, one pair "
                    f"(CVaR_0.10, F) per setting: {_SCORES_2C}.")
    pairs = []
    for i, x in enumerate(items):
        try:
            pairs.append(_score_pair(x, f"scores_s[{i}]"))
        except _Fail:
            raise _Fail(f"scores_s[{i}] is {_kind(x)}, not a pair of two numbers: "
                        f"each entry is my_score of one setting, {_SCORES_2C}."
                        ) from None
    return pairs


def _merged(counts_list) -> dict:
    out: dict = {}
    for c in counts_list:
        for key, v in c.items():
            out[key] = out.get(key, 0) + v
    return out


def _diagnose_2c(inst, counts, pairs, kit, thetas, bad) -> str:
    """The message for scores that differ from the kit's per-setting
    scores: pairs in the other order, reverse order, the pooled counts."""
    from .hardware import metrics_from_counts

    def all_close(ps, ks):
        return all(_close_pair(p, k["CVaR"], k["F"]) for p, k in zip(ps, ks))

    fix = ("scores_s[i] = my_score(data_s.c.get_counts(i)), in the order of "
           "thetas_sweep")
    if all_close([(p[1], p[0]) for p in pairs], kit):
        return ("every pair of scores_s is (F, CVaR_0.10); each pair is "
                f"(CVaR_0.10, F), as my_score returns it: {fix}.")
    if all_close(pairs, kit[::-1]):
        return ("scores_s is in reverse order: scores_s[i] is the score of setting "
                "i, whose counts are data_s.c.get_counts(i) and whose theta is "
                "thetas_sweep[i].")
    pooled = metrics_from_counts(inst, _merged(counts), ALPHA)
    if all_close(pairs, [pooled] * len(pairs)):
        return (f"every entry of scores_s is the score of the pooled counts of all "
                f"{len(pairs)} settings (CVaR_0.10 = {pooled['CVaR']:.2f}, F = "
                f"{pooled['F']:.4f}): score each setting on its own counts, {fix}.")
    i = bad[0]
    p, k = pairs[i], kit[i]
    if _close_pair((p[1], p[0]), k["CVaR"], k["F"]):
        return (f"scores_s[{i}] is (F, CVaR_0.10); each pair is (CVaR_0.10, F), "
                "as my_score returns it.")
    return (f"scores_s[{i}] = ({p[0]:.6g}, {p[1]:.6g}), but the counts of setting "
            f"{i} (theta = {thetas[i]:.4f}) give CVaR_0.10 = {k['CVaR']:.6g} and "
            f"F = {k['F']:.6g} (hw.metrics_from_counts): {fix}.")


@_guarded("2c")
def check_2c(inst: DensestKSubgraph, job_record, counts, thetas_sweep, scores,
             registered=None) -> CheckResult:
    """Exercise 2c: the sweep job (``job_record`` and ``counts`` from
    :func:`job_facts`): DONE, on the assigned backend, the service's
    record; one PUB with m >= ``MIN_SETTINGS_2C`` parameter sets of at
    least ``MIN_SHOTS_2`` shots each. ``thetas_sweep``: m finite, pairwise
    different values. ``scores``: m pairs (CVaR_0.10, F) that agree with
    ``hw.metrics_from_counts`` of the counts of each setting, in order
    (named: the wrong length, pairs in the other order, the pooled counts,
    reverse order). Then a scored registration of the team
    (``registered``; until then nothing is recorded)."""
    from .hardware import metrics_from_counts

    if job_record is None or counts is None:
        raise _Fail(f"no job was given: run the job cell and pass the job, {CALL_2C}.")
    jid = _check_job_basics(job_record)
    sets = list(job_record.get("sets_per_pub") or [])
    if job_record.get("n_pubs") != 1:
        raise _Fail(f"job {jid} has {job_record.get('n_pubs')} PUBs; the sweep is one "
                    f"PUB that carries all values of theta: {_PUB_2C}.")
    m = sets[0] if sets else 0
    if m < MIN_SETTINGS_2C:
        raise _Fail(f"the PUB of job {jid} holds {_plural(m, 'parameter set')}; a "
                    f"sweep needs at least {MIN_SETTINGS_2C} values of theta (the "
                    f"exercise uses eight): {_PUB_2C}.")
    shots = [int(s) for s in (job_record.get("shots_per_set") or [0])]
    if min(shots) < MIN_SHOTS_2:
        raise _Fail(f"job {jid} has {min(shots)} shots in a setting; use at least "
                    f"{MIN_SHOTS_2} per setting (the exercise uses 4096).")
    th = _sweep_values(thetas_sweep, m)
    pairs = _score_pairs(scores, m)
    counts = list(counts)
    kit = [metrics_from_counts(inst, c, ALPHA) for c in counts]
    bad = [i for i, (p, k) in enumerate(zip(pairs, kit))
           if not _close_pair(p, k["CVaR"], k["F"])]
    if bad:
        raise _Fail(_diagnose_2c(inst, counts, pairs, kit, th, bad))
    row = _registration(registered, jid)
    reg = f"; registered as job #{row['seq']}" if row else ""
    best = int(np.argmax([k["CVaR"] for k in kit]))
    each = (f"{shots[0]} shots each" if len(set(shots)) == 1
            else f"at least {min(shots)} shots each")
    return _result("2c", True, f"{_job_text(job_record, f'one PUB of {m} settings, {each}')}"
                   "; your scores agree with the kit's for every setting; the best "
                   f"setting by CVaR_0.10 is theta = {th[best]:.4f} (CVaR_0.10 = "
                   f"{kit[best]['CVaR']:.2f}, F = {kit[best]['F']:.4f}){reg}.")


# ------------------------------------------------------------ Exercise 2d
_PLAN_ROW = ("each row is (label, circuits, shots) or (label, circuits, shots, "
             "repeat), or a dict with the keys label, circuits, shots and repeat")
_PLAN = "my_plan = [(label, circuits, shots, repeat), ...]"


def _whole(v) -> int | None:
    """``v`` as an int when it is an integer >= 1 (a whole float counts)."""
    if isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v) if v >= 1 else None
    if (isinstance(v, (float, np.floating)) and np.isfinite(v)
            and float(v).is_integer() and v >= 1):
        return int(v)
    return None


def _plan_row(item, i: int) -> tuple:
    """Row ``i`` of the plan as ``(label, circuits, shots, repeat)``, with
    the defaults of ``nb.plan`` for a dict (circuits 1, shots 2^15, repeat
    1); raises ``_Fail`` for a row of the wrong shape."""
    if isinstance(item, Mapping):
        unknown = sorted(set(map(str, item)) - {"label", "circuits", "shots", "repeat"})
        if unknown:
            raise _Fail(f"row {i} has the keys {unknown}, which nb.plan does not "
                        f"read: {_PLAN_ROW}.")
        label = item.get("label")
        values = (item.get("circuits", 1), item.get("shots", 2 ** 15),
                  item.get("repeat", 1))
    elif isinstance(item, (tuple, list)):
        if len(item) not in (3, 4):
            raise _Fail(f"row {i} holds {_plural(len(item), 'value')}: {_PLAN_ROW}.")
        label, values = item[0], (item[1], item[2], item[3] if len(item) == 4 else 1)
    else:
        raise _Fail(f"row {i} is {_kind(item)}: {_PLAN_ROW}.")
    if not isinstance(label, str) or not label.strip():
        raise _Fail(f"row {i} needs a label, a short text that names the kind of "
                    f"job: {_PLAN_ROW}.")
    out = []
    for name, v in zip(("circuits", "shots", "repeat"), values):
        whole = _whole(v)
        if whole is None:
            raise _Fail(f"row {i} (\"{label.strip()}\"): {name} must be an integer "
                        f">= 1, got {v!r}.")
        out.append(whole)
    return (label.strip(), *out)


def _plan_rows(plan) -> list:
    if plan is None:
        raise _Fail(f"my_plan is None: write one row per kind of job, {_PLAN}.")
    if isinstance(plan, Mapping):
        if "rows" in plan and "qpu_s" in plan:
            raise _Fail(f"pass my_plan, the list of rows, not what nb.plan "
                        f"returned: {CALL_2D}.")
        raise _Fail(f"my_plan must be a list of rows, got one dict: put the rows "
                    f"in a list, {_PLAN}.")
    if isinstance(plan, (str, bytes)):
        raise _Fail(f"my_plan must be a list of rows, got a string: {_PLAN}.")
    if isinstance(plan, tuple) and plan and isinstance(plan[0], str):
        raise _Fail(f"my_plan is one row; the plan is a list of rows: {_PLAN}.")
    try:
        items = list(plan)
    except TypeError:
        raise _Fail(f"my_plan must be a list of rows, got {_kind(plan)}: "
                    f"{_PLAN}.") from None
    return [_plan_row(item, i) for i, item in enumerate(items, 1)]


def _budget_left(budget) -> float:
    """The QPU seconds left: ``budget`` is the ``(budget, spent, left)`` of
    ``Campaign.budget``, a callable that returns it, or None (a fresh
    campaign of ``BUDGET_S``)."""
    if budget is None:
        return BUDGET_S
    if callable(budget):
        budget = budget()
    return float(tuple(budget)[2])


@_guarded("2d")
def check_2d(plan, budget=None) -> CheckResult:
    """Exercise 2d: ``plan`` is a list of rows as ``nb.plan`` reads them,
    ``(label, circuits, shots[, repeat])`` or dicts; each row has a
    non-empty label and integers circuits, shots, repeat >= 1; every
    planned job has at least ``grader.MIN_SHOTS_PER_SUBMISSION`` shots in
    total (it can be registered); the plan is not empty and its QPU time
    (``event.job_usage_estimate``) fits the budget that is left
    (``budget``, see :func:`_budget_left`)."""
    from .event import job_usage_estimate
    from .grader import MIN_SHOTS_PER_SUBMISSION

    rows = _plan_rows(plan)
    if not rows:
        raise _Fail("the plan is empty: add one row per kind of job you intend to "
                    "run, (label, circuits, shots, repeat).")
    for i, (label, c, s, _) in enumerate(rows, 1):
        if c * s < MIN_SHOTS_PER_SUBMISSION:
            raise _Fail(f"row {i} (\"{label}\"): a job of {_plural(c, 'circuit')} x "
                        f"{s} shots has {c * s} shots in total; a registered job "
                        f"needs at least {MIN_SHOTS_PER_SUBMISSION}.")
    left = _budget_left(budget)
    qpu = float(sum(r * job_usage_estimate(c, s) for _, c, s, r in rows))
    jobs = int(sum(r for *_, r in rows))
    if qpu > left + 1e-9:
        raise _Fail(f"the plan needs {qpu:.1f} s of QPU time, and {left:.1f} s of "
                    "your budget are left: plan fewer jobs, circuits or shots.")
    return _result("2d", True, f"{_plural(jobs, 'job')}, {qpu:.1f} s of QPU time, "
                   f"{_plural(jobs, 'registration slot')} of {SLOT_MIN} minutes "
                   f"({jobs * SLOT_MIN / 60:.1f} h); {left - qpu:.1f} s of your "
                   f"budget remain after the plan ({left:.1f} s left now).")


# --------------------------------------------------------------- dispatch
def _job_like(x) -> bool:
    """A primitive job: callable ``result`` and ``status``."""
    return (not isinstance(x, (str, bytes, dict))
            and callable(getattr(x, "result", None))
            and callable(getattr(x, "status", None)))


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _usage_1d(answer) -> CheckResult | None:
    """A readable FAIL for a call of the 1d check with the wrong arguments,
    or None when the arguments have the shape of the full check."""
    n = len(answer)
    if n == 3:
        job, ev, cv = answer
        if job is None:
            return _usage("1d", "no job was given (job is None): run the job "
                          f"cell first, then {CALL_1D}; {CALL_1D_FUNCTIONS} "
                          "tests the two functions before the job is done.")
        if _job_like(job):
            if (getattr(ev, "__name__", "") == "cvar"
                    and getattr(cv, "__name__", "") == "evaluate_counts"):
                return _usage("1d", "evaluate_counts and cvar are swapped: "
                              f"{CALL_1D}.")
            return None
        if callable(job) and _job_like(cv):
            return _usage("1d", f"the job goes first: {CALL_1D}.")
        if isinstance(job, str):
            return _usage("1d", "pass the job itself, not its id: fetch it "
                          f"with job = service.job(\"{job[:40]}\"), then "
                          f"{CALL_1D}.")
        return _usage("1d", "the first argument must be the job that "
                      "sampler.run (or campaign.run_once) returned, got "
                      f"{type(job).__name__}: {CALL_1D}.")
    if n == 2 and _job_like(answer[0]):
        return _usage("1d", f"the check needs both functions: {CALL_1D}.")
    return _usage("1d", f"the check takes (job, evaluate_counts, cvar), or "
                  "(evaluate_counts, cvar) to test the two functions before "
                  f"the job is done; got {_plural(n, 'argument')}. Use "
                  f"{CALL_1D}.")


def _counts_like(x) -> bool:
    """A counts dictionary (bitstring keys)."""
    return isinstance(x, Mapping) and len(x) > 0 and all(
        isinstance(key, str) and key.strip()
        and set(key.replace(" ", "")) <= {"0", "1"} for key in list(x)[:20])


def _float_array(x):
    if x is None or isinstance(x, (str, bytes, Mapping)) or _job_like(x):
        return None
    try:
        return np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None


def _usage_job(ex: str, job, name: str, call: str) -> CheckResult:
    """The FAIL line for a first argument that is not the job."""
    if job is None:
        extra = (f"; {CALL_2B_ANGLE} tests theta_c before the job is done"
                 if ex == "2b" else "")
        return _usage(ex, f"no job was given ({name} is None): run the job cell "
                      f"first, then {call}{extra}.")
    if isinstance(job, str):
        return _usage(ex, "pass the job itself, not its id: fetch it with "
                      f"{name} = service.job(\"{job[:40]}\"), then {call}.")
    if _counts_like(job):
        return _usage(ex, "the first argument is a counts dictionary; pass the "
                      f"job itself, the one that campaign.run_once returned: {call}.")
    return _usage(ex, "the first argument must be the job that sampler.run (or "
                  f"campaign.run_once) returned, got {type(job).__name__}: {call}.")


def _usage_2b(answer) -> CheckResult | None:
    """A readable FAIL for a call of the 2b check with the wrong arguments,
    or None when the arguments have the shape of the full check."""
    n = len(answer)
    if n == 3:
        job, a, b = answer
        if _job_like(job):
            if _stats_like(a) and not _stats_like(b):
                return _usage("2b", f"theta_c and stats are swapped: {CALL_2B}.")
            return None
        if _job_like(a) or _job_like(b):
            return _usage("2b", f"the job goes first: {CALL_2B}.")
        return _usage_job("2b", job, "job_c", CALL_2B)
    if n == 2 and _job_like(answer[0]):
        return _usage("2b", f"the check needs theta_c and stats as well: {CALL_2B}.")
    if n == 2 and _stats_like(answer[0]):
        return _usage("2b", f"theta_c goes first: {CALL_2B_ANGLE} before the job, "
                      f"{CALL_2B} after it.")
    return _usage("2b", "the check takes (job_c, theta_c, stats), or (theta_c, "
                  "stats) to test the angle before the job is done; got "
                  f"{_plural(n, 'argument')}. Use {CALL_2B}.")


def _usage_2c(answer) -> CheckResult | None:
    """A readable FAIL for a call of the 2c check with the wrong arguments,
    or None when the arguments have the shape of the check."""
    n = len(answer)
    if n == 3:
        job, a, b = answer
        if _job_like(job):
            pa, pb = _float_array(a), _float_array(b)
            if (pa is not None and pa.ndim == 2 and pa.shape[-1] == 2
                    and pb is not None and pb.ndim == 1):
                return _usage("2c", f"thetas_sweep and scores_s are swapped: {CALL_2C}.")
            return None
        if _job_like(a) or _job_like(b):
            return _usage("2c", f"the job goes first: {CALL_2C}.")
        return _usage_job("2c", job, "job_s", CALL_2C)
    if n and _job_like(answer[0]):
        return _usage("2c", "the check needs the values of theta and your scores "
                      f"as well: {CALL_2C}.")
    return _usage("2c", "the check takes (job_s, thetas_sweep, scores_s); got "
                  f"{_plural(n, 'argument')}. Use {CALL_2C}.")


def _check_part2(ex: str, inst, answer, assigned_backend, verify_job,
                 registered, budget) -> CheckResult:
    """Dispatch of Exercises 2a .. 2d (see :func:`check`)."""
    if ex in ("2a", "2d") and len(answer) != 1:
        return _usage(ex, f"the check takes one answer, got "
                      f"{_plural(len(answer), 'argument')}: {_CALLS[ex]}.")
    if ex == "2a":
        return check_2a(inst, answer[0])
    if ex == "2d":
        return check_2d(answer[0], budget)
    if ex == "2b" and len(answer) == 2 and not (
            _job_like(answer[0]) or _stats_like(answer[0])):
        return check_2b_angle(inst, *answer)
    usage = (_usage_2b if ex == "2b" else _usage_2c)(answer)
    if usage is not None:
        return usage
    job, second, third = answer

    @_guarded(ex)
    def _run():
        if ex == "2b":
            _angle_2b(inst, second, third)         # before the job is read
        record, counts = job_facts(job, assigned_backend=assigned_backend,
                                   n=inst.n)
        if verify_job is not None:
            record["anchor"] = _anchor_facts(verify_job, record.get("job_id"))
        if ex == "2b":
            return check_2b(inst, record, counts, second, third, registered)
        return check_2c(inst, record, counts, second, third, registered)
    return _run()


def check(exercise: str, inst: DensestKSubgraph, *answer, target=None,
          assigned_backend=None, verify_job=None, registered=None,
          budget=None) -> CheckResult:
    """Dispatch to the check of ``exercise``: "1a" .. "1d" (``inst`` is the
    training instance) or "2a" .. "2d" (``inst`` is the campaign instance).
    For "1d" the answer is ``(job, evaluate_counts, cvar)``, whose job
    result is read here, or ``(evaluate_counts, cvar)`` alone: the
    functions-only check (:func:`check_1d_functions`), which records no
    points. For "2b" it is ``(job_c, theta_c, stats)``, or ``(theta_c,
    stats)`` alone: the angle-only check (:func:`check_2b_angle`), which
    records no points; for "2c" ``(job_s, thetas_sweep, scores_s)``; for
    "2a" ``my_score`` and for "2d" ``my_plan``. ``verify_job`` (job id ->
    metadata dict, e.g. a job anchor's ``verify``) checks a job against
    the IBM Quantum service; a job it cannot verify fails. ``registered``
    (job id -> the grader's row of a registered job of the team, or None)
    is the registration test of 2b and 2c; ``budget`` (the ``(budget,
    spent, left)`` of ``Campaign.budget``, or a callable returning it) is
    the budget of 2d. Arguments of the wrong shape give a one-line FAIL
    that shows the right call. Never raises."""
    ex = str(exercise).strip().lower()
    if ex not in EX_POINTS:
        return CheckResult(False, 0.0, f"FAIL - unknown exercise {exercise!r}: "
                           f"use one of {', '.join(EX_POINTS)}.", "",
                           record=False)
    if ex in EX2_POINTS:
        return _check_part2(ex, inst, answer, assigned_backend, verify_job,
                            registered, budget)
    if ex == "1d":
        if len(answer) == 2 and all(callable(a) and not _job_like(a)
                                    for a in answer):
            return check_1d_functions(inst, *answer)
        usage = _usage_1d(answer)
        if usage is not None:
            return usage
    elif len(answer) != 1:
        return _usage(ex, f"the check takes one answer, got "
                      f"{_plural(len(answer), 'argument')}: {_CALLS[ex]}.")
    if ex == "1a":
        return check_1a(inst, answer[0])
    if ex == "1b":
        return check_1b(inst, answer[0])
    if ex == "1c":
        return check_1c(inst, answer[0], target)
    job, evaluate_counts, cvar = answer

    @_guarded("1d")
    def _run():
        record, counts = job_facts(job, assigned_backend=assigned_backend,
                                   n=inst.n)
        if verify_job is not None:
            record["anchor"] = _anchor_facts(verify_job, record.get("job_id"))
        return check_1d(inst, record, counts, evaluate_counts, cvar)
    return _run()
