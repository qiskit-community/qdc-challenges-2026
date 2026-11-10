"""Helpers for the QDC 2026 verifiability challenge (Lab 1).

These are the parts that are tedious, memory-dangerous, or easy to get silently
wrong. Everything conceptual stays in the notebook.

Sections
--------
Registers and counts   add_meas_register
Device calibration     native_two_qubit_gate, edge_error, device_defects
Stabilizer sampling    sample_stabilizers
Classical DFE          inverse_payload, dfe_from_counts
Hardware DFE           stabilizer_bases, collect_program_info,
                       non_markovian_mask, check_mask,
                       fidelity_from_evs
Plotting               PALETTE, plot_postselection, plot_fidelity_vs_checks,
                       plot_bound
"""

from __future__ import annotations

from dataclasses import replace
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister
from qiskit.quantum_info import Clifford, Pauli, PauliList

# Ordered by section, matching the docstring above, rather than alphabetically.
__all__ = [  # noqa: RUF022
    "add_meas_register",
    "native_two_qubit_gate",
    "edge_error",
    "device_defects",
    "sample_stabilizers",
    "inverse_payload",
    "dfe_from_counts",
    "stabilizer_bases",
    "collect_program_info",
    "non_markovian_mask",
    "check_mask",
    "fidelity_from_evs",
    "PALETTE",
    "plot_postselection",
    "plot_fidelity_vs_checks",
    "plot_bound",
]


# ===========================================================================
# Registers and counts
# ===========================================================================
def add_meas_register(checked_circuit, layout, name="meas"):
    """Return a copy of ``checked_circuit`` with a payload measurement register.

    ``meas[k]`` measures the physical qubit ``layout[k]``, i.e. virtual qubit ``k``
    of the target circuit. The existing check register is left untouched.
    """
    circ = checked_circuit.circuit.copy()
    creg = ClassicalRegister(len(layout), name=name)
    circ.add_register(creg)
    circ.measure(layout, creg)
    return replace(checked_circuit, circuit=circ)


# ===========================================================================
# Device calibration
# ===========================================================================
TWO_Q_CANDIDATES = ("cz", "ecr", "cx")


def native_two_qubit_gate(backend, candidates=TWO_Q_CANDIDATES):
    """Name of the backend's native two-qubit gate."""
    names = set(backend.target.operation_names)
    for name in candidates:
        if name in names:
            return name
    raise ValueError(f"none of {candidates} is native; Target has {sorted(names)}")


def edge_error(backend, two_q_name, a, b):
    """Error of the native 2Q gate on edge ``(a, b)``, or ``None`` if uncalibrated.

    The ``Target`` is keyed by *directed* qargs, so both orientations are tried.
    """
    props_map = backend.target[two_q_name]
    for key in ((a, b), (b, a)):
        if key in props_map:
            props = props_map[key]
            return None if props is None else props.error
    return None


def _readout_error(backend, qubit):
    try:
        props = backend.target["measure"][(qubit,)]
    except (KeyError, TypeError):
        return None
    return None if props is None else props.error


def device_defects(backend, ro_threshold=5e-2, tq_threshold=5e-2, verbose=True):
    """Qubits and couplers to keep out of a layout, read off the backend calibration.

    A qubit is excluded when its readout error is at or above ``ro_threshold``; a
    coupler when its two-qubit gate error is at or above ``tq_threshold``. Anything
    uncalibrated is excluded too, on the grounds that an error you cannot read is
    one you cannot trust.

    Returns
    -------
    dict
        ``bad_qubits`` and ``bad_edges`` (directed pairs, both orientations), plus
        ``clean_edges``: a list of ``[a, b]`` pairs ready to hand straight to
        ``mapomatic.matching_layouts``. Also the observed medians and the
        thresholds used.

    Pruning the coupling map before the search is cheaper than discarding checks
    afterwards, but it is only safe while the surviving graph still contains the
    payload chain. The long paths of a coupling map share qubits, and each qubit
    removed cuts every path through it, so an over-aggressive cut can leave no
    embedding at all -- which is why the summary prints how much of the device each
    threshold costs.
    """
    two_q = native_two_qubit_gate(backend)
    n = backend.num_qubits

    ro = {q: _readout_error(backend, q) for q in range(n)}
    bad_qubits = {q for q, e in ro.items() if e is None or e >= ro_threshold}

    edges = [tuple(e) for e in backend.coupling_map.get_edges()]
    tq = {pair: edge_error(backend, two_q, *pair) for pair in edges}
    bad_edges = {p for p, e in tq.items() if e is None or e >= tq_threshold}

    clean_edges = [
        [a, b]
        for (a, b) in edges
        if a not in bad_qubits and b not in bad_qubits and (a, b) not in bad_edges
    ]

    ro_known = [e for e in ro.values() if e is not None]
    tq_known = [e for e in tq.values() if e is not None]

    if verbose:
        print(f"--- {backend.name} calibration, excluding errors >= threshold ---")
        if ro_known:
            print(
                f"  readout : median {np.median(ro_known):.4f}  max {max(ro_known):.4f}"
                f"   threshold {ro_threshold:g}"
            )
        if tq_known:
            print(
                f"  {two_q:<8}: median {np.median(tq_known):.5f}  max {max(tq_known):.5f}"
                f"   threshold {tq_threshold:g}"
            )
        print(
            f"  qubits kept : {n - len(bad_qubits):>4}/{n} "
            f"({100 * (n - len(bad_qubits)) / n:.0f}%)"
        )
        print(
            f"  edges  kept : {len(clean_edges):>4}/{len(edges)} "
            f"({100 * len(clean_edges) / max(len(edges), 1):.0f}%)"
        )
        if not clean_edges:
            print("  WARNING: nothing survived; raise the thresholds")

    return {
        "two_q_name": two_q,
        "bad_qubits": bad_qubits,
        "bad_edges": bad_edges,
        "clean_edges": clean_edges,
        "median_readout": float(np.median(ro_known)) if ro_known else None,
        "median_edge": float(np.median(tq_known)) if tq_known else None,
        "ro_threshold": ro_threshold,
        "tq_threshold": tq_threshold,
    }


# ===========================================================================
# Stabilizer sampling
# ===========================================================================
def sample_stabilizers(circuit, num_stabilizers, seed):
    """Sample random non-identity elements of ``circuit``'s stabilizer group.

    The group has ``2**num_qubits`` elements, so it must never be enumerated: at
    61 qubits a ``PauliList`` of all of them is 3e20 bytes. Instead, a uniformly
    random **non-empty subset of the generators** is a uniformly random
    non-identity group element (the generators are independent, so subset and
    element are in bijection).

    The products are formed with ``Pauli.compose``, not by XOR-ing the symplectic
    arrays: XOR gives the right operators with the **wrong phases**, and DFE reads
    the sign of each expectation value.

    Returns
    -------
    (PauliList, np.ndarray)
        ``stabilizers`` and ``masks``, where ``masks[i]`` is the 0/1 indicator of
        which generators multiply to ``stabilizers[i]``. Keep ``masks``: it is what
        lets one simulation per check count serve all stabilizers at once.
    """
    n = circuit.num_qubits
    if not num_stabilizers < 2**n:
        raise ValueError(f"num_stabilizers must be < 2**{n}")

    generators = [Pauli(label) for label in Clifford(circuit).to_labels(mode="S")]

    rng = np.random.default_rng(seed)
    masks, seen = [], set()
    while len(masks) < num_stabilizers:
        mask = rng.integers(0, 2, size=n, dtype=np.uint8)
        if not mask.any():  # skip the identity
            continue
        key = mask.tobytes()
        if key in seen:  # sample without replacement
            continue
        seen.add(key)
        masks.append(mask)

    masks = np.array(masks, dtype=np.int64)
    stabilizers = PauliList(
        [
            reduce(
                lambda a, b: a.compose(b), (generators[j] for j in np.flatnonzero(m))
            )
            for m in masks
        ]
    )
    return stabilizers, masks


# ===========================================================================
# Classical DFE on the Aer stabilizer simulator
# ===========================================================================
def inverse_payload(circuit, noise_instructions=()):
    """Basis change that maps every stabilizer generator onto a ``Z`` operator.

    All ``n`` generators commute, so a single rotation measures the whole group at
    once: the inverse payload Clifford sends generator ``q`` to ``Z_q``, which makes
    payload bit ``q`` the eigenvalue of generator ``q``. One simulation per check
    count then serves all stabilizers, instead of one per (check count, stabilizer).

    ``noise_instructions`` is the set of gate names your Aer noise model touches
    (``aer_nm.noise_instructions``). The basis change must be noiseless for the
    estimate to mean anything, so it is checked rather than assumed.
    """
    inv = Clifford(circuit).adjoint().to_circuit()
    overlap = set(inv.count_ops()) & set(noise_instructions)
    if overlap:
        raise ValueError(f"basis change would pick up gate noise on {sorted(overlap)}")
    return inv


def dfe_from_counts(counts, circuit, stab_masks, ps_fn, name="meas"):
    """Fidelity estimates from one counts dict of a basis-changed circuit.

    Parameters
    ----------
    counts : dict[str, int]
        Qiskit counts from running the basis-changed circuit.
    circuit : QuantumCircuit
        The circuit those counts came from, i.e. the one with the inverse payload
        and the generator measurement already appended.
    stab_masks : np.ndarray
        The ``masks`` returned by :func:`sample_stabilizers`, of shape
        ``(num_stabilizers, num_generators)`` and 0/1 valued: ``stab_masks[i, q]``
        says whether generator ``q`` appears in stabilizer ``i``. Its second
        dimension must match the width of the ``name`` register.
    ps_fn : Callable[[str], np.ndarray]
        One shot's outcome to its syndrome vector, with the shot kept exactly when
        every entry is zero -- that is ``checked_circuit.get_postselection_method()``.
    name : str
        Classical register holding the generator eigenvalue. "meas" by default.

    Returns
    -------
    (float, float, float)
        ``F_raw``   -- mean <G> over all shots.
        ``F_ps``    -- mean <G> over the shots that pass every check.
        ``F_group`` -- P(every generator bit read 0) on the kept shots. Averaging
        <G> over the *whole* group collapses to exactly this probability, so it is
        a free cross-check on the sampled-stabilizer estimate ``F_ps``.
        ``F_ps`` and ``F_group`` are ``nan`` when no shot survives.
    """

    meas = next(creg for creg in circuit.cregs if creg.name == name)
    meas_clbits = np.array([circuit.find_bit(cb).index for cb in meas])

    keys = list(counts)
    weights = np.array([counts[k] for k in keys], dtype=float)
    # Qiskit counts strings run from high clbit index (left) to clbit 0 (right),
    # so reversing the de-spaced string indexes bits by global clbit.
    bits = np.array(
        [
            (np.frombuffer(k.replace(" ", "").encode(), np.uint8) - ord("0"))[::-1]
            for k in keys
        ]
    )
    s = bits[:, meas_clbits].astype(np.int64)  # s[:, q] = eigenvalue bit of generator q

    kept = weights * np.array([not ps_fn(k).any() for k in keys])
    signs = 1 - 2 * ((s @ stab_masks.T) % 2)  # +-1 for every (outcome, stabilizer) pair

    f_raw = float(np.mean((weights @ signs) / weights.sum()))
    if kept.sum() == 0:
        return f_raw, float("nan"), float("nan")
    f_ps = float(np.mean((kept @ signs) / kept.sum()))
    f_group = float(kept[~s.any(axis=1)].sum() / kept.sum())
    return f_raw, f_ps, f_group


# ===========================================================================
# Hardware DFE: Executor + TREX plumbing
# ===========================================================================
def stabilizer_bases(stabilizers, layout, num_device_qubits):
    """Measurement-basis array for the sampled stabilizers.

    ``bases[i, 0, q]`` is the basis in which physical qubit ``q`` is measured for
    stabilizer ``i``: Z=1, X=2, Y=3, with 1 (Z) on every qubit outside the payload.
    """
    bases = np.ones((len(stabilizers), 1, num_device_qubits), dtype=np.uint8)
    bases[:, 0, layout] = stabilizers.z + 2 * stabilizers.x
    return bases


def collect_program_info(
    program, circuit, layout, stabilizers, checked_circuits, item=-1
):
    """Plain-data description of one ``QuantumProgram`` item, for the Step 6 grader.

    Call it once, after the program is built::

        program_info = utils.collect_program_info(
            program, circuit, layout, stabilizers, reduced_checked_circuits
        )

    ``item`` picks which program item to describe, defaulting to the last, which
    carries the most checks. The four extra arguments are what the program does
    not know about itself: the target ``circuit`` (so the grader can verify that
    your stabilizers really stabilize it), the payload ``layout``, the
    ``stabilizers`` in submission order, and the ``checked_circuits`` the items
    were built from, for that item's ancillas.

    What comes back is plain data plus two circuits, which is all that can be
    submitted for grading. The ``samplomatic`` objects are read here and never
    passed on: the grading server does not have that package.
    """
    it = program.items[item]
    cc = list(checked_circuits)[item]
    (spec,) = it.samplex.inputs().get_specs()
    keys = list(it.samplex_arguments.keys())
    key = keys[0] if len(keys) == 1 else None
    return {
        "template": it.circuit,
        "circuit": circuit,
        "layout": [int(q) for q in layout],
        "check_qubits": [int(q) for q in cc.check_qubits],
        "stabilizers": [str(G) for G in stabilizers],
        "basis_array": None if key is None else np.asarray(it.samplex_arguments[key]),
        "input_name": str(spec.name),
        "argument_key": None if key is None else str(key),
        "shape": [int(v) for v in it.shape],
        "shots": int(program.shots),
    }


def non_markovian_mask(out, selector):
    """Per-shot keep-mask from the non-Markovian error checks alone."""
    return selector.compute_mask(
        {name: out[name] for name in out if "." not in name}, "edge"
    )


def check_mask(out):
    """Per-shot keep-mask from the spacetime Pauli checks alone.

    Returns an all-True mask when the variant has no check register, so that
    ``nm & checks`` is always well defined.
    """
    if "checks_c" not in out:
        # No check register at all, so every shot passes. Take the shape from
        # "meas", which carries the shots axis: "measurement_flips.meas" is per
        # randomization, not per shot, so it would be one short and only work when
        # this mask is broadcast against a correctly shaped one.
        return np.ones(out["meas"].shape[:-1], dtype=bool)
    return ~(out["checks_c"] ^ out["measurement_flips.checks_c"]).any(axis=-1)


def fidelity_from_evs(e):
    """Fidelity and its uncertainty from per-stabilizer ``(mean, sem)`` rows.

    Two independent error sources, added in quadrature: the finite stabilizer
    sample (we averaged ``n`` of the ``2**num_qubits - 1`` group elements) and shot
    noise (the per-observable ``sem``).
    """
    e = np.asarray(e, dtype=float)
    n = len(e)
    mean = float(e[:, 0].mean())
    var_sample = e[:, 0].std(ddof=1) ** 2 / n if n > 1 else 0.0
    var_shots = (e[:, 1] ** 2).sum() / n**2
    return np.abs(mean), float(np.sqrt(var_sample + var_shots))


# ===========================================================================
# Plotting
# ===========================================================================
#: Categorical slots in fixed order (never cycled), plus ink and status colors.
PALETTE = {
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    "muted": "#898781",
    "critical": "#d03b3b",
    "ideal": "#0b0b0b",
}


def _xs(x, label, ys):
    """Resolve the x values for one curve: None, a shared list, or a per-label dict."""
    if x is None:
        return list(range(len(ys)))
    if isinstance(x, dict):
        return list(x.get(label, range(len(ys))))
    return list(x)


def _finish(ax, xlabel, ylabel, legend=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="best", fontsize=9)


def plot_postselection(series, x=None, xlabel="Number of checks", title=None):
    """Post-selection rate vs. check count. ``series`` is ``{label: rates}``.

    ``x`` sets the check counts explicitly, for a run that covered only some of
    them (Step 6 runs ``reduced_checks``). It may also be a ``{label: xs}`` dict
    when the curves cover different check counts.
    """
    _fig, ax = plt.subplots(figsize=(8, 4))
    for (label, rates), color in zip(series.items(), PALETTE["series"]):
        xs = _xs(x, label, rates)
        ax.plot(xs, rates, "o-", lw=1.5, color=color, label=label)
    ax.set_ylim(0, 1.05)
    if title:
        ax.set_title(title)
    _finish(ax, xlabel, "Post-selection rate", legend=len(series) > 1)
    plt.tight_layout()
    plt.show()


def plot_fidelity_vs_checks(
    curves, rates=None, x=None, reference=None, xlabel="Number of checks", title=None
):
    """Fidelity vs. check count, with post-selection rate in a second panel.

    ``curves`` is ``{label: values}`` or ``{label: (values, errors)}``, and ``x``
    sets the check counts explicitly (a list, or ``{label: xs}`` when curves cover
    different check counts). ``reference`` is an optional ``(label, values)`` pair
    drawn as a muted dotted line, for a comparison curve that is chrome rather than
    a series (e.g. the simulated result behind a hardware measurement). Fidelity and
    post-selection rate are different measures, so they get their own panels rather
    than one twinned axis, where the crossing point would be an artifact of the two
    ranges.
    """
    if rates is None:
        _fig, ax = plt.subplots(figsize=(8, 4))
        ax2 = None
    else:
        _fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, height_ratios=[2, 1]
        )

    for (label, value), color in zip(curves.items(), PALETTE["series"]):
        ys, errs = value if isinstance(value, tuple) else (value, None)
        xs = _xs(x, label, ys)
        if errs is None:
            ax.plot(xs, ys, "o-", lw=1.5, color=color, label=label)
        else:
            ax.errorbar(
                xs, ys, yerr=errs, fmt="o-", lw=1.5, capsize=3, color=color, label=label
            )
    if reference is not None:
        ref_label, ref_ys = reference
        ax.plot(
            _xs(None, ref_label, ref_ys),
            ref_ys,
            ":",
            lw=1.5,
            color=PALETTE["muted"],
            label=ref_label,
        )
    # ax.axhline(1.0, color=PALETTE["ideal"], lw=1, ls=":", alpha=0.4, label="ideal")
    ax.set_ylim(auto=True)
    if title:
        ax.set_title(title)
    _finish(ax, xlabel if ax2 is None else "", "DFE fidelity")

    if ax2 is not None:
        for (label, r), color in zip(rates.items(), PALETTE["series"][1:]):
            ax2.plot(_xs(x, label, r), r, "o-", lw=1.5, color=color, label=label)
        ax2.set_ylim(0, 1.05)
        _finish(ax2, xlabel, "Shots kept", legend=True)

    plt.tight_layout()
    plt.show()


def plot_bound(reduced_checks, f_lower, hardware, xlabel="Number of checks"):
    """The Step 9 lower bound against the Clifford fidelity it is built from.

    ``hardware`` is an optional ``(x, values, errors)`` triple for the hardware DFE
    points, which exist only at the check counts you ran on the QPU.
    """
    _fig, ax = plt.subplots(figsize=(8, 4))

    xs = reduced_checks
    ys, errs = hardware

    ax.plot(
        xs,
        f_lower,
        "s-",
        lw=1.5,
        color=PALETTE["series"][1],
        label="T-doped fidelity lower bound",
    )

    ax.errorbar(
        xs,
        ys,
        yerr=errs,
        fmt="^",
        ms=8,
        capsize=3,
        lw=1.5,
        color=PALETTE["series"][2],
        label="Clifford DFE, post-selected (QPU)",
    )

    _finish(ax, xlabel, "Fidelity")
    plt.tight_layout()
    plt.show()
