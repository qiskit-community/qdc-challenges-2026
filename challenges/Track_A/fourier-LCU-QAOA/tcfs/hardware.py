"""Utility-scale pathway (paper Sec. VI C).

- ``select_region`` / ``instance_from_backend``: grow the campaign instance
  from a backend's own coupling map (compact, error-aware region), with the
  edge set extended by ``swap_layers`` layers of a swap network (the paper's
  App. F builds its 106-node instance this way: lattice + 3 SWAP layers).
- ``hardware_basis_circuit``: the Fig. 3 basis circuit in *position space* -
  cost sub-layers interleaved with the swap layers, RZ(theta) layer, warm-start
  mixer, measurements mapped back to logical bits - so every two-qubit gate is
  on a native edge and no routing is needed. Exactly equivalent to
  ``exact.penalty_basis_state`` up to the tracked permutation (tests).
- ``hardware_basis_circuit_p``: the depth-p version (odd layers run the swap
  network mirrored, so every layer reaches every edge and sigma returns to
  the identity after each pair of layers); ``swap_last`` orders each swap
  pair's cost gate right before its swap for two-qubit block resynthesis.
- ``resynth_pass_manager`` / ``circuit_cost``: level-1 transpilation plus
  two-qubit block resynthesis, and the CZ / depth / duration cost table.
- ``bits_from_counts`` / ``metrics_from_counts``: scoring from raw counts at
  any n (no 2^n tables).
- ``layer_expectations`` / ``lightcone_expectations``: exact <H>, <f>, <W>
  of the basis circuit at any n via the light cone of a diagonal phase +
  product mixer (the paper's "analytic p = 1" transfer). O(n * deg^2),
  vectorized. ``layer_expectations`` takes the layer order (gamma, theta,
  beta); ``lightcone_expectations`` takes (beta, gamma, theta).
- ``lagrangian_layers`` / ``theta_for_weight`` / ``weight_stats``: the
  shipped parameter transfer of the campaign: theta is read as the
  multiplier of the cardinality constraint, solved so that the ideal mean
  Hamming weight equals a target, and (gamma, beta) maximize the ideal <f>
  under that constraint; ``weight_stats`` reads mean, spread and feasible
  fraction of the Hamming weight from measured counts; ``theta_profile``
  gives the ideal <W>, <f>, Var W as functions of theta.
- ``transfer_parameters``: grid + COBYLA over the light-cone <H> (kept for
  the tests; superseded for the campaign by ``lagrangian_layers``).
- ``milp_optimum``: exact OPT via scipy's HiGHS MILP (organizer side).
- ``baseline_B_mc``: seeded Monte-Carlo B at scales where enumeration fails.
- ``qpy_bytes`` / ``circuits_from_qpy``: the scored-submission wire format.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from dataclasses import dataclass
from math import erf, sqrt

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, qpy

from . import exact
from .cvar import cvar
from .instances import DensestKSubgraph


# ------------------------------------------------------------- region growth
def _edge_errors(backend) -> dict:
    err = {}
    target = getattr(backend, "target", None)
    if target is not None:
        for gate in ("cz", "ecr", "cx", "rzz"):
            if gate in target.operation_names:
                for qs, props in target[gate].items():
                    e = getattr(props, "error", None)
                    e = 1e-2 if e is None else float(e)
                    key = frozenset(qs)
                    err[key] = min(err.get(key, 1.0), e)
    return err


def faulty_from_target(target, threshold: float = 0.5) -> tuple[set, set]:
    """``(qubits, edges)`` whose Target error rate is >= ``threshold``:
    qubits by any single-qubit gate or the measurement, edges by any
    two-qubit gate. Empty sets when ``target`` is None."""
    bad_q, bad_e = set(), set()
    if target is None:
        return bad_q, bad_e
    for name in target.operation_names:
        try:
            items = target[name].items()
        except Exception:  # noqa: BLE001 - globally defined instructions
            continue
        for qs, props in items:
            e = getattr(props, "error", None)
            if qs is None or e is None or e < threshold:
                continue
            if len(qs) == 1:
                bad_q.add(int(qs[0]))
            elif len(qs) == 2:
                bad_e.add((int(qs[0]), int(qs[1])))
    return bad_q, bad_e


def select_region(backend, n: int, start: int | None = None,
                  exclude=()) -> list[int]:
    """A compact, connected n-qubit region grown from a central qubit,
    preferring (region-adjacency, low two-qubit error) at every step.
    ``exclude``: physical qubits that may not be used (other teams)."""
    G = nx.Graph(list(backend.coupling_map.get_edges()))
    G.remove_nodes_from([q for q in exclude if q in G])
    # never touch a qubit/edge the primitives' check_faulty would reject
    props = getattr(backend, "properties", None)
    try:
        props = props() if callable(props) else props
        for q in (props.faulty_qubits() if props is not None else []):
            if q in G:
                G.remove_node(q)
        for gate in (props.faulty_gates() if props is not None else []):
            if len(gate.qubits) == 2 and G.has_edge(*gate.qubits):
                G.remove_edge(*gate.qubits)
    except Exception:  # noqa: BLE001 - fake backends without properties()
        pass
    # ... and nothing the Target itself reports as dead: a 1q-gate or
    # readout error >= 0.5 marks a qubit, a 2q error >= 0.5 an edge
    # (ibm_phoenix lists such entries while properties().faulty_* is empty)
    bad_q, bad_e = faulty_from_target(getattr(backend, "target", None))
    G.remove_nodes_from([q for q in bad_q if q in G])
    G.remove_edges_from([e for e in bad_e if G.has_edge(*e)])
    err = _edge_errors(backend)
    e_of = lambda a, b: err.get(frozenset((a, b)), 1e-2)  # noqa: E731
    if start is None:
        center = nx.center(G)
        start = min(center, key=lambda q: np.mean([e_of(q, v) for v in G[q]]))
    region, inside = [start], {start}
    while len(region) < n:
        cands = {}
        for q in region:
            for v in G[q]:
                if v in inside:
                    continue
                c = cands.setdefault(v, [0, 1.0])
                c[0] += 1
                c[1] = min(c[1], e_of(q, v))
        if not cands:
            raise ValueError("coupling map too small / disconnected for n")
        best = min(cands, key=lambda v: (-cands[v][0], cands[v][1]))
        region.append(best)
        inside.add(best)
    return region


@dataclass
class HardwareLayout:
    physical: list            # position p -> physical qubit
    native_edges: list        # (p, q) position pairs that are native edges
    swap_layers: list         # list of layers; each a list of (p, q) to swap
    backend_name: str = ""


def _matchings(native_edges, n_layers, rng):
    """Greedy maximal matchings, penalizing reuse of the previous layer."""
    layers, prev = [], set()
    for _ in range(n_layers):
        order = list(native_edges)
        rng.shuffle(order)
        order.sort(key=lambda e: e in prev)          # unused edges first
        used, layer = set(), []
        for a, b in order:
            if a in used or b in used:
                continue
            layer.append((a, b))
            used.update((a, b))
        layers.append(layer)
        prev = set(layer)
    return layers


def instance_from_backend(backend, n: int = 106, swap_layers: int = 0,
                          seed: int = 0, k: int | None = None,
                          weighted: bool = False, start: int | None = None,
                          name: str | None = None, exclude=()):
    """(instance, layout): the instance's edges are the native edges of the
    region plus every logical pair brought adjacent by the swap network."""
    rng = np.random.default_rng(seed)
    physical = select_region(backend, n, start, exclude)
    pos = {q: p for p, q in enumerate(physical)}
    G = nx.Graph(list(backend.coupling_map.get_edges()))
    native = sorted({(min(pos[a], pos[b]), max(pos[a], pos[b]))
                     for a, b in G.subgraph(physical).edges()})
    layers = _matchings(native, swap_layers, rng)
    sigma = list(range(n))                     # position -> logical
    edges = set()
    for stage in range(swap_layers + 1):
        for a, b in native:
            i, j = sigma[a], sigma[b]
            edges.add((min(i, j), max(i, j)))
        if stage < swap_layers:
            for a, b in layers[stage]:
                sigma[a], sigma[b] = sigma[b], sigma[a]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i, j in sorted(edges):
        w = float(np.round(rng.uniform(0.5, 1.5), 3)) if weighted else 1.0
        g.add_edge(i, j, weight=w)
    inst = DensestKSubgraph.from_graph(
        g, k=k if k is not None else n // 3,
        name=name or f"{getattr(backend, 'name', 'backend')}-n{n}-swap{swap_layers}-seed{seed}")
    layout = HardwareLayout(physical=physical, native_edges=native,
                            swap_layers=layers,
                            backend_name=getattr(backend, "name", ""))
    return inst, layout


# ---------------------------------------------------------- circuit builder
def _edge_colouring(edges) -> list:
    """Greedy proper edge colouring (<= max degree + 1 classes): gates in
    one class act on disjoint qubits, so emitting class by class keeps the
    DAG depth at #classes per stage instead of a serial chain."""
    classes, used = [], []            # used[c] = set of qubits in class c
    for a, b in sorted(edges, key=lambda e: (max(e), min(e))):
        for c, qs in enumerate(used):
            if a not in qs and b not in qs:
                classes[c].append((a, b))
                qs.update((a, b))
                break
        else:
            classes.append([(a, b)])
            used.append({a, b})
    return classes


def hardware_basis_circuit(inst: DensestKSubgraph, layout: HardwareLayout,
                           gamma, theta=0.0, beta=0.0, measure: bool = True,
                           xy=None):
    """Fig. 3 basis circuit in position space (all 2q gates on native edges).

    ``xy=(alpha, theta_xy, chi)`` builds the Fig. 5 XY-basis circuit instead:
    the same warm start and cost/swap network, then the collective rotation
    R(alpha, theta_xy, chi)^(x)n in place of the RZ layer + warm-start mixer
    (theta and beta are ignored). This is the circuit behind the paper's best
    hardware result (Exp. 5).

    Returns (circuit, sigma) with sigma[p] = logical qubit sitting at
    position p at the end; measurements write position p to clbit sigma[p],
    so counts bitstrings are in LOGICAL order (clbit i <-> x_i).
    """
    n = inst.n
    t = inst.theta_init
    qc = QuantumCircuit(n, n if measure else 0, name="hw_basis")
    qc.ry(t, range(n))
    W = {(i, j): w for i, j, w in inst.edges}
    sigma = list(range(n))
    colours = _edge_colouring(layout.native_edges)   # depth = #colours/stage
    missing = _cost_network(qc, W, sigma, colours, layout.swap_layers, gamma)
    if missing:
        raise ValueError(f"{len(missing)} instance edges are not reachable "
                         "with this layout's swap network")
    if xy is None:
        _penalty_tail(qc, n, t, theta, beta)
    else:
        _xy_tail(qc, n, xy)
    if measure:
        for p in range(n):
            qc.measure(p, sigma[p])
    return qc, sigma


def hardware_basis_circuit_p(inst: DensestKSubgraph, layout: HardwareLayout,
                             layers, measure: bool = True, xy=None,
                             swap_last: bool = False):
    """Depth-p basis circuit in position space (all 2q gates native).

    ``layers``: a sequence of penalty layers ``(gamma, theta, beta)`` in
    circuit order. Each layer is one full cost layer e^{-i gamma H1} (cost
    sub-layers on the native edges interleaved with the swap network), the
    RZ LCU layer V(theta) with its global phase theta*n/2, and the
    warm-start mixer at beta - the same blocks as ``hardware_basis_circuit``
    after the single warm start.

    Swap network direction: layer 0, 2, 4, ... runs the swap layers forward,
    layer 1, 3, 5, ... runs them *mirrored* (swap layers in reverse order).
    A mirrored pass revisits the forward pass's placements in reverse, so it
    reaches every instance edge and undoes the permutation: after every even
    number of layers sigma is the identity. (A second *forward* pass does
    not work: starting from the permuted placement it brings a different set
    of logical pairs together - on the 106-qubit / 2-swap-layer seed-0
    instance only 230 of 397 edges on the phoenix snapshot, 225 of 385 on
    the FakeNighthawk template.) Every layer must reach every instance edge,
    otherwise ``ValueError`` names the layer and the number of missing
    edges.

    ``xy=(alpha, theta_xy, chi)``: the LAST layer's RZ layer + warm-start
    mixer are replaced by the collective rotation R(alpha, theta_xy, chi)
    (as in ``hardware_basis_circuit``); only the last tuple's gamma is read
    (its theta and beta are ignored, so it may be given as ``(gamma,)``).
    Earlier layers stay penalty basis layers.

    ``swap_last=True`` reorders the gates of every cost sub-layer that is
    followed by a swap layer: first the cost gates on native edges that are
    not a pair of that swap layer (colour by colour), then, per swap pair
    (a, b), its cost gate (if the pair currently holds an unapplied instance
    edge) immediately followed by ``swap(a, b)``. Same unitary (all cost
    gates are diagonal and every cost gate on a or b still precedes the
    swap), but cost + swap on one pair form a single two-qubit block, which
    ``resynth_pass_manager`` synthesizes in <= 3 CZ instead of 2 + 3.

    One layer with ``swap_last=False`` reproduces ``hardware_basis_circuit``
    gate for gate (same operations, order, parameters, global phase, sigma).

    The circuit declares its depth in ``metadata = {"tcfs": {"p": p}}``
    (kept by transpilation and QPY); the grader client reports it, and the
    frontier cards of axes D and T compare it.

    Returns (circuit, sigma) with sigma[p] = logical qubit at position p at
    the end; measurements write position p to clbit sigma[p] (counts are in
    logical order).
    """
    layers = list(layers)
    if not layers:
        raise ValueError("layers must contain at least one (gamma, theta, "
                         "beta) layer")
    n_layers = len(layers)
    n = inst.n
    t = inst.theta_init
    name = "hw_basis" if n_layers == 1 else f"hw_basis_p{n_layers}"
    qc = QuantumCircuit(n, n if measure else 0, name=name)
    qc.ry(t, range(n))
    W = {(i, j): w for i, j, w in inst.edges}
    sigma = list(range(n))
    colours = _edge_colouring(layout.native_edges)
    forward = list(layout.swap_layers)
    mirrored = forward[::-1]
    for li, layer in enumerate(layers):
        last_xy = xy is not None and li == n_layers - 1
        gamma = layer[0]
        missing = _cost_network(qc, W, sigma, colours,
                                forward if li % 2 == 0 else mirrored, gamma,
                                swap_last=swap_last)
        if missing:
            raise ValueError(
                f"layer {li} ({'forward' if li % 2 == 0 else 'mirrored'} swap "
                f"network): {len(missing)} instance edges are not reachable "
                "with this layout's swap network")
        if last_xy:
            _xy_tail(qc, n, xy)
        else:
            _, theta, beta = layer
            _penalty_tail(qc, n, t, theta, beta)
    if measure:
        for p in range(n):
            qc.measure(p, sigma[p])
    qc.metadata = {"tcfs": {"p": n_layers}}
    return qc, sigma


def _cost_network(qc, W, sigma, colours, swap_layers, gamma,
                  swap_last: bool = False) -> set:
    """One full cost layer e^{-i gamma H1}: len(swap_layers) + 1 cost
    sub-layers interleaved with ``swap_layers`` (in the order given).
    Updates ``sigma`` (position -> logical) in place and returns the set of
    instance edges it did not reach (empty = complete)."""
    applied = set()
    for stage in range(len(swap_layers) + 1):
        swaps = swap_layers[stage] if stage < len(swap_layers) else ()
        _cost_stage(qc, W, sigma, colours, applied, gamma, swaps, swap_last)
    return set(W) - applied


def _cost_stage(qc, W, sigma, colours, applied, gamma, swaps=(),
                swap_last: bool = False) -> None:
    """Cost gates at the current placement on every native edge whose
    logical pair is a not-yet-applied instance edge (colour by colour), then
    the swap layer ``swaps``. With ``swap_last`` the cost gate of each native
    swap pair is deferred to just before its swap (see
    ``hardware_basis_circuit_p``)."""

    def cost(a, b):
        key = (min(sigma[a], sigma[b]), max(sigma[a], sigma[b]))
        if key in W and key not in applied:
            w = W[key]
            qc.rzz(gamma * w / 2, a, b)
            qc.rz(-gamma * w / 2, a)
            qc.rz(-gamma * w / 2, b)
            qc.global_phase -= gamma * w / 4
            applied.add(key)

    deferred = set()
    if swap_last and swaps:
        native = {frozenset(e) for colour in colours for e in colour}
        deferred = {frozenset(e) for e in swaps} & native
    for colour in colours:
        for a, b in colour:
            if frozenset((a, b)) not in deferred:
                cost(a, b)
    for a, b in swaps:
        if frozenset((a, b)) in deferred:
            cost(a, b)
        qc.swap(a, b)
        sigma[a], sigma[b] = sigma[b], sigma[a]


def _penalty_tail(qc, n, t, theta, beta) -> None:
    """RZ LCU layer V(theta) (with its global phase theta*n/2) followed by
    the warm-start mixer at beta."""
    qc.rz(theta, range(n))
    qc.global_phase += theta * n / 2
    qc.ry(-t, range(n))
    qc.rz(-2 * beta, range(n))
    qc.ry(t, range(n))


def _xy_tail(qc, n, xy) -> None:
    """Collective rotation R(alpha, theta_xy, chi)^(x)n, circuit order
    rz(chi), ry(theta_xy), rz(alpha)."""
    a_xy, t_xy, c_xy = xy
    qc.rz(c_xy, range(n))
    qc.ry(t_xy, range(n))
    qc.rz(a_xy, range(n))


def logical_statevector(qc: QuantumCircuit, sigma: list) -> np.ndarray:
    """Statevector of an unmeasured position-space circuit, re-ordered so
    that index bit i is logical qubit i (for tests / small n)."""
    from qiskit.quantum_info import Statevector

    psi = np.asarray(Statevector(qc).data)
    n = len(sigma)
    # amplitude index b has bit p = value at position p; logical bit
    # sigma[p] takes that value
    idx = np.arange(1 << n)
    out_idx = np.zeros_like(idx)
    for p in range(n):
        out_idx |= ((idx >> p) & 1) << sigma[p]
    out = np.zeros_like(psi)
    out[out_idx] = psi
    return out


# ------------------------------------------------------ counts -> metrics
def bits_from_counts(counts: dict, n: int):
    """(bits[m, n] uint8 with bits[:, i] = x_i, counts[m]) from a Qiskit
    counts dict (bitstring char at position -1-i is clbit i)."""
    keys = list(counts)
    arr = np.frombuffer("".join(k.replace(" ", "") for k in keys).encode(),
                        dtype=np.uint8).reshape(len(keys), n) - 48
    bits = arr[:, ::-1].astype(np.uint8)
    return bits, np.array([counts[k] for k in keys], dtype=float)


def metrics_from_counts(inst: DensestKSubgraph, counts: dict,
                        alpha: float = 0.10, opt_value: float | None = None,
                        baseline: float | None = None) -> dict:
    bits, cnt = bits_from_counts(counts, inst.n)
    f, w = inst.evaluate_bits(bits)
    flam = f - inst.lam * (w - inst.k) ** 2
    feas = w == inst.k
    out = {"CVaR": cvar(flam, cnt, alpha), "F": float(cnt[feas].sum() / cnt.sum()),
           "best_feasible": float(f[feas].max()) if feas.any() else float("nan"),
           "mean_flam": float(np.dot(flam, cnt) / cnt.sum()),
           "shots": int(cnt.sum())}
    if opt_value is not None and baseline is not None:
        out["Q"] = float(np.clip((out["CVaR"] - baseline)
                                 / (opt_value - baseline), 0, 1))
    return out


# --------------------------------------------------- light-cone expectations
class _LightCone:
    """Precomputed neighbourhood structure for a fixed instance."""

    def __init__(self, inst: DensestKSubgraph):
        n = inst.n
        adj = [dict() for _ in range(n)]
        for i, j, w in inst.edges:
            adj[i][j] = w
            adj[j][i] = w
        self.inst, self.adj = inst, adj
        dmax = max((len(a) for a in adj), default=0)
        # singles: weights of neighbours, padded with zeros (E(0) = 1)
        self.A1 = np.zeros((n, dmax))
        for a in range(n):
            self.A1[a, :len(adj[a])] = list(adj[a].values())
        # pairs with overlapping closed neighbourhoods
        pairs, Awts, Bwts, wab, is_edge = [], [], [], [], []
        for a in range(n):
            near = set(adj[a])
            for j in adj[a]:
                near |= set(adj[j])
            for b in sorted(near):
                if b <= a:
                    continue
                N = (set(adj[a]) | set(adj[b])) - {a, b}
                pairs.append((a, b))
                Awts.append([adj[a].get(j, 0.0) for j in N])
                Bwts.append([adj[b].get(j, 0.0) for j in N])
                wab.append(adj[a].get(b, 0.0))
                is_edge.append(b in adj[a])
        jmax = max((len(x) for x in Awts), default=0)
        P = len(pairs)
        self.pairs = np.array(pairs, dtype=int).reshape(P, 2)
        self.A = np.zeros((P, jmax))
        self.B = np.zeros((P, jmax))
        for p in range(P):
            self.A[p, :len(Awts[p])] = Awts[p]
            self.B[p, :len(Bwts[p])] = Bwts[p]
        self.wab = np.array(wab)
        self.is_edge = np.array(is_edge, dtype=bool)   # membership, not sign
        # edge weights for the objective
        self.edge_w = np.array([w for _, _, w in inst.edges])

    def expectations(self, beta: float, gamma: float, theta: float,
                     M=None) -> dict:
        """Exact moments after warm start -> cost(gamma) -> e^{i theta w}
        -> M^(x)n. ``M`` defaults to the warm-start mixer at ``beta``; the
        XY-basis circuit is M = R(alpha, theta_xy, chi), theta = 0."""
        inst, n = self.inst, self.inst.n
        t = inst.theta_init
        c, s = np.cos(t / 2), np.sin(t / 2)
        psi0 = np.array([c, s])
        M = exact.ws_mixer_matrix(inst, beta) if M is None else np.asarray(M)

        def E(alpha):  # E_j[e^{i alpha x_j}], x_j ~ Bernoulli(s^2)
            return c ** 2 + s ** 2 * np.exp(1j * alpha)

        # ---- singles: rho[s, s'] = psi0[s] psi0[s'] e^{i theta (s-s')} K(s-s')
        K1 = {d: np.prod(E(-gamma * d * self.A1), axis=1) for d in (-1, 0, 1)}
        rho1 = np.zeros((n, 2, 2), dtype=complex)
        for sa in (0, 1):
            for sb in (0, 1):
                d = sa - sb
                rho1[:, sa, sb] = psi0[sa] * psi0[sb] * np.exp(1j * theta * d) * K1[d]
        rho1 = np.einsum("ij,njk,lk->nil", M, rho1, M.conj())
        x1 = rho1[:, 1, 1].real

        # ---- overlapping pairs: rho[(sa,sb),(sa',sb')]
        P = len(self.pairs)
        K2 = {}
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                K2[(da, db)] = np.prod(E(-gamma * (da * self.A + db * self.B)),
                                       axis=1) if P else np.zeros(0)
        rho2 = np.zeros((P, 4, 4), dtype=complex)
        for sa in (0, 1):
            for sb in (0, 1):
                for ta in (0, 1):
                    for tb in (0, 1):
                        amp = psi0[sa] * psi0[sb] * psi0[ta] * psi0[tb]
                        ph = np.exp(1j * theta * ((sa + sb) - (ta + tb))) \
                            * np.exp(-1j * gamma * self.wab * (sa * sb - ta * tb))
                        rho2[:, 2 * sa + sb, 2 * ta + tb] = amp * ph * K2[(sa - ta, sb - tb)]
        MM = np.kron(M, M)
        rho2 = np.einsum("ij,pjk,lk->pil", MM, rho2, MM.conj())
        xx_near = rho2[:, 3, 3].real if P else np.zeros(0)

        # ---- assemble: <W>, <W^2>, <f>
        W1 = float(x1.sum())
        # sum over ordered pairs a != b of <x_a x_b>: independent pairs
        # contribute x1[a] x1[b]; near pairs their exact value
        all_pairs = W1 ** 2 - float(np.dot(x1, x1))
        near_prod = float(np.sum(x1[self.pairs[:, 0]] * x1[self.pairs[:, 1]])) if P else 0.0
        sum_xx = all_pairs - 2 * near_prod + 2 * float(xx_near.sum())
        W2 = W1 + sum_xx
        exp_f = float(np.dot(self.wab[self.is_edge], xx_near[self.is_edge])) if P else 0.0
        exp_pen = W2 - 2 * inst.k * W1 + inst.k ** 2
        return {"exp_H": exp_f - inst.lam * exp_pen, "exp_f": exp_f,
                "exp_W": W1, "var_W": W2 - W1 ** 2, "exp_penalty": exp_pen,
                "x1": x1}


def lightcone_expectations(inst: DensestKSubgraph, beta: float, gamma: float,
                           theta: float, _cache: dict = {}) -> dict:
    """Exact <H>, <f>, <W>, Var(W) of the single LCU basis circuit at any n.

    The angles come in the order (beta, gamma, theta), not in the order of
    the layer tuples (gamma, theta, beta): use :func:`layer_expectations`,
    which takes the layer order, e.g. ``layer_expectations(inst,
    *layers[0])``."""
    key = id(inst)
    if key not in _cache or _cache[key][0] is not inst:
        _cache.clear()
        _cache[key] = (inst, _LightCone(inst))
    return _cache[key][1].expectations(beta, gamma, theta)


def layer_expectations(inst: DensestKSubgraph, gamma: float, theta: float,
                       beta: float) -> dict:
    """Exact moments of the one-layer basis circuit at any n, with the
    angles in the order of a layer tuple ``(gamma, theta, beta)``:
    ``layer_expectations(inst, *layers[0])`` evaluates the first layer of
    ``lagrangian_layers`` or of any list passed to
    ``hardware_basis_circuit_p``.

    Returns a dict: ``exp_H`` (<f_lam>), ``exp_f`` (<f>), ``exp_W`` (mean
    Hamming weight), ``var_W`` (its variance), ``exp_penalty`` (<(W - k)^2>)
    and ``x1`` (the probability that each qubit reads 1). The values are
    those of the ideal (noiseless) circuit, computed from the light cone of
    each qubit and each pair of qubits in milliseconds."""
    return lightcone_expectations(inst, beta, gamma, theta)


def lightcone_expectations_xy(inst: DensestKSubgraph, gamma: float,
                              alpha: float, theta: float, chi: float,
                              _cache: dict = {}) -> dict:
    """Exact <H>, <f>, <W>, Var(W) of the single XY-basis circuit at any n."""
    from .schur_weyl import su2_matrix

    key = id(inst)
    if key not in _cache or _cache[key][0] is not inst:
        _cache.clear()
        _cache[key] = (inst, _LightCone(inst))
    return _cache[key][1].expectations(0.0, gamma, 0.0,
                                       M=su2_matrix(alpha, theta, chi))


def transfer_objective(e: dict, k: int, lam: float, alpha: float = 0.10,
                       mu: float = 1.0, nu: float = 1.0,
                       window: tuple[int, int] | None = None) -> float:
    """Tail-aware moment proxy for the scored CVaR_alpha:

        <f>  -  mu (<W> - k)^2  -  nu * lam * max(0, sigma_W - sigma_alpha)^2,

    sigma_alpha = 1 / (alpha sqrt(2 pi)) being the Hamming-weight spread at
    which the feasible mass P(W = k) falls to alpha. Rationale (verified on
    n = 12-24 against the exact CVaR): maximizing <H> = <f> - lam <(W-k)^2>
    with the dominance lambda buys feasibility mass with objective and lands
    BELOW the warm-start baseline (Q = 0); maximizing <f> alone inflates W
    (f grows with W); keeping <W> at k recovers ~0.3-0.6 of the direct
    CVaR optimum where the feasible mass exceeds alpha, and the spread term
    stops the W distribution from widening past alpha at utility scale.

    ``window=(lo, hi)`` switches to the Hamming-weight window variant
    (unscored kit capability since 2026-09-29), objective
    f - lam_ind 1[1^T x outside [lo, hi]], for which "feasible" is a band
    rather than a point. The two W terms are replaced by

        <f> - mu dist(<W>, [lo, hi])^2
            - nu lam (sigma_alpha / alpha)^2 max(0, alpha - P_win)^2,

    with dist(.) = 0 inside the window and P_win the normal-approximation
    in-window mass P(lo - 1/2 <= W <= hi + 1/2) at mean <W>, std sigma_W
    (half-integer edges = the usual continuity correction for the integer
    W). Both terms vanish on a distribution that is centred in the window
    and puts at least alpha of its mass there, so nothing is paid for a
    spread that is already tail-safe - only the CVaR_alpha-relevant deficit
    is charged, which is what makes a *band* different from the equality
    case (a wide W is fine if the band is wide).

    Scale: the mass deficit max(0, alpha - P_win)^2 is dimensionless and at
    most alpha^2, so on its own it would be invisible next to <f> and the
    dist^2 term. Multiplying by (sigma_alpha / alpha)^2 normalizes it to
    the *relative* shortfall (1 - P_win/alpha)^2 scaled by sigma_alpha^2,
    i.e. a total loss of the feasible mass (P_win = 0) costs exactly what
    the equality-case spread term charges for a distribution one
    sigma_alpha too wide. The two branches therefore price an equally bad
    tail equally, and nu keeps its meaning across both."""
    sig_alpha = 1.0 / (alpha * np.sqrt(2 * np.pi))
    sig = np.sqrt(max(e["var_W"], 0.0))
    if window is None:
        return (e["exp_f"] - mu * (e["exp_W"] - k) ** 2
                - nu * lam * max(0.0, sig - sig_alpha) ** 2)
    lo, hi = window
    mean_W = e["exp_W"]
    dist = max(lo - mean_W, mean_W - hi, 0.0)
    s = max(float(sig), 1e-12)
    phi = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))  # noqa: E731
    p_win = phi((hi + 0.5 - mean_W) / s) - phi((lo - 0.5 - mean_W) / s)
    return (e["exp_f"] - mu * dist ** 2
            - nu * lam * (sig_alpha / alpha) ** 2
            * max(0.0, alpha - p_win) ** 2)


def transfer_parameters(inst: DensestKSubgraph, n_grid: int = 9,
                        objective: str = "tail", alpha: float = 0.10,
                        mu: float = 1.0, nu: float = 1.0,
                        gamma_max: float | None = None, maxiter: int = 200,
                        mode: str = "penalty",
                        window: tuple[int, int] | None = None):
    """Optimize the basis-circuit angles on the exact light-cone moments
    (the paper's analytic p = 1 transfer, extended to the LCU angle theta),
    with no simulator. Superseded for the campaign by
    :func:`lagrangian_layers`; kept for the tests and the earlier
    calibration runs. ``objective`` = "tail" (default, see
    ``transfer_objective``) or "exp_H" (the plain expectation value).
    ``mode`` = "penalty": returns ((beta, gamma, theta), value);
    ``mode`` = "xy": optimizes the XY-basis circuit and returns
    ((gamma, alpha, theta_xy, chi), value). The final rz(alpha) commutes
    with the Z measurement, so alpha is fixed at 0 and chi is periodic in
    2 pi for the measured distribution: a 3-angle search (n_grid = 9 is
    ~15 s at 106 qubits). ``window=(lo, hi)`` optimizes the Hamming-weight
    window variant (unscored kit capability since 2026-09-29) instead of
    the equality objective (see
    ``transfer_objective``); it applies to both modes and is ignored by
    ``objective="exp_H"``, whose <H> is the equality Hamiltonian."""
    from .optimize import grid_then_cobyla
    from .schur_weyl import su2_matrix

    lc = _LightCone(inst)
    if gamma_max is None:
        gamma_max = np.pi
    if mode == "penalty":
        bounds = [(-np.pi / 2, np.pi / 2), (-gamma_max, gamma_max), (0.0, 2 * np.pi)]
        moments = lambda p: lc.expectations(p[0], p[1], p[2])  # noqa: E731
    elif mode == "xy":
        bounds = [(-gamma_max, gamma_max), (0.0, np.pi), (0.0, 2 * np.pi)]
        moments = lambda p: lc.expectations(  # noqa: E731
            0.0, p[0], 0.0, M=su2_matrix(0.0, p[1], p[2]))
    else:
        raise ValueError("mode must be 'penalty' or 'xy'")
    if objective == "exp_H":
        fun = lambda p: moments(p)["exp_H"]  # noqa: E731
    elif objective == "tail":
        fun = lambda p: transfer_objective(  # noqa: E731
            moments(p), inst.k, inst.lam, alpha, mu, nu, window)
    else:
        raise ValueError("objective must be 'tail' or 'exp_H'")
    x, v = grid_then_cobyla(fun, bounds, n_grid=n_grid, maxiter=maxiter)
    if mode == "xy":
        x = (x[0], 0.0, x[1], x[2])            # (gamma, alpha=0, theta, chi)
    return tuple(float(a) for a in x), float(v)


# ------------------------------------------ Lagrangian transfer (v0.4)
# The theta dependence of the p = 1 moments is a trigonometric polynomial:
# V(theta) multiplies |x> by e^{i theta W(x)}, so a one-qubit reduced
# density matrix before the mixer picks up e^{i theta (s - s')} and a
# two-qubit one e^{i theta ((s_a + s_b) - (t_a + t_b))}. Hence <W>(theta) has
# degree 1 and <f>(theta), <W^2>(theta), Var W(theta) have degree <= 2 in
# theta, and five light-cone evaluations at theta = 2 pi j / 5 determine all
# of them exactly (DFT interpolation).
_TRIG_THETAS = 2.0 * np.pi * np.arange(5) / 5.0
_TRIG_ORDERS = np.array([0, 1, 2, -2, -1])          # numpy DFT order, 5 points
LAGRANGIAN_GAMMAS = tuple(float(g) for g in -np.linspace(0.35, 1.25, 7))
LAGRANGIAN_BETAS = tuple(float(b) for b in np.linspace(0.15, 0.75, 7))
LAGRANGIAN_MAXITER = 30
LAGRANGIAN_CACHE = "lagrangian_106.json"            # in CALIBRATION_DIR


def _lightcone_of(inst: DensestKSubgraph, _cache: dict = {}) -> "_LightCone":
    """One ``_LightCone`` per instance object (the same identity-keyed
    cache as :func:`lightcone_expectations`)."""
    key = id(inst)
    if key not in _cache or _cache[key][0] is not inst:
        _cache.clear()
        _cache[key] = (inst, _LightCone(inst))
    return _cache[key][1]


class _ThetaProfile:
    """Exact <W>, <f>, Var W of the p = 1 basis circuit as functions of
    theta at fixed (gamma, beta), from five light-cone evaluations."""

    def __init__(self, lc: "_LightCone", gamma: float, beta: float):
        rows = []
        for t in _TRIG_THETAS:
            e = lc.expectations(float(beta), float(gamma), float(t))
            rows.append((e["exp_W"], e["exp_f"], e["var_W"]))
        self.gamma, self.beta = float(gamma), float(beta)
        self.coef = np.fft.fft(np.asarray(rows), axis=0) / len(_TRIG_THETAS)

    def moments(self, theta) -> np.ndarray:
        """Columns <W>, <f>, Var W at ``theta`` (scalar or array)."""
        th = np.atleast_1d(np.asarray(theta, dtype=float))
        return np.real(np.exp(1j * np.outer(th, _TRIG_ORDERS)) @ self.coef)

    def weight_range(self) -> tuple[float, float]:
        """(min, max) of <W>(theta) over theta (a first-order trigonometric
        polynomial: mean +- twice the modulus of its first harmonic)."""
        a0, amp = float(np.real(self.coef[0, 0])), 2.0 * float(abs(self.coef[1, 0]))
        return a0 - amp, a0 + amp

    def roots(self, target: float) -> list[float]:
        """Every theta in [0, 2 pi) with <W>(theta) = target (exact: <W> is
        a0 + A cos(theta + phi), so there are 0, 1 or 2 roots)."""
        a0 = float(np.real(self.coef[0, 0]))
        c1 = complex(self.coef[1, 0])
        amp, phi = 2.0 * abs(c1), float(np.angle(c1))
        if amp <= 1e-12:
            return []
        r = (float(target) - a0) / amp
        if abs(r) > 1.0 + 1e-12:
            return []
        base = float(np.arccos(np.clip(r, -1.0, 1.0)))
        return sorted({round(float((s * base - phi) % (2 * np.pi)), 12)
                       for s in (1.0, -1.0)})

    def best_root(self, target: float):
        """(theta, (<W>, <f>, Var W)) of the root with the larger <f>, or
        None when no theta reaches the target."""
        best = None
        for t in self.roots(target):
            m = self.moments(t)[0]
            if best is None or m[1] > best[1][1]:
                best = (t, m)
        return best


def _instance_key(inst: DensestKSubgraph) -> str:
    """sha256 over (n, k, sorted weighted edges): the identity used by the
    calibration caches."""
    import hashlib

    body = json.dumps({"n": int(inst.n), "k": int(inst.k),
                       "edges": [[int(i), int(j), float(w)]
                                 for i, j, w in sorted(inst.edges)]},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def _lagrangian_args(target, gammas, betas, maxiter) -> dict:
    return {"target": float(target),
            "gammas": [round(float(g), 12) for g in gammas],
            "betas": [round(float(b), 12) for b in betas],
            "maxiter": int(maxiter)}


def _load_lagrangian_cache(inst, args, path=None):
    path = CALIBRATION_DIR / LAGRANGIAN_CACHE if path is None else Path(path)
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if d.get("instance_key") != _instance_key(inst) or d.get("args") != args:
        return None
    info = dict(d["info"])
    info["source"] = f"calibration/{Path(path).name}"
    return [tuple(float(a) for a in layer) for layer in d["layers"]], info


def lagrangian_layers(inst: DensestKSubgraph, target_weight=None, gammas=None,
                      betas=None, maxiter: int = LAGRANGIAN_MAXITER,
                      use_cache: bool = True, _memo: dict = {}):
    """The shipped p = 1 parameter transfer of the campaign. The paper reads
    the LCU angle theta as the Lagrange multiplier of the cardinality
    constraint (Sec. V); this function uses that reading and solves theta so
    that the ideal mean Hamming weight equals the target (a rule of the kit,
    not of the paper).

    1. ``target = inst.k`` unless ``target_weight`` is given.
    2. For every (gamma, beta) of the grid (defaults ``gammas =
       -linspace(0.35, 1.25, 7)``, ``betas = linspace(0.15, 0.75, 7)``)
       every theta in [0, 2 pi) with ideal
       <W>(theta) = target is found and the root with the larger ideal <f>
       is kept; a pair with no root is skipped. The roots are exact: <W> is
       a first-order trigonometric polynomial in theta, and five light-cone
       evaluations fix <W>, <f> and Var W as functions of theta.
    3. The pair with the largest <f> is refined locally (Nelder-Mead on
       (gamma, beta), theta re-solved at every step, at most ``maxiter``
       iterations); the refined point replaces the grid point only if its
       <f> is larger.

    Returns ``(layers, info)``: ``layers = [(gamma, theta, beta)]`` (the
    kit's layer order, ready for ``hardware_basis_circuit_p``) and ``info``
    with ``gamma``, ``theta``, ``beta``, ``exp_f``, ``exp_W``, ``sd_W``,
    ``var_W``, ``p_feasible`` (the Gaussian estimate 1 / sqrt(2 pi Var W)),
    ``target``, ``grid`` (one row per grid pair, ``theta`` None where no
    root exists), ``grid_best``, ``refined``, ``seconds`` and ``source``.

    Results are memoized per instance and arguments; the pinned demo
    instance ships its result in ``calibration/lagrangian_106.json``, which
    is used when the instance and the arguments match (``use_cache``).
    Raises ``ValueError`` when no grid pair reaches the target weight."""
    import time as _time

    from scipy.optimize import minimize

    target = float(inst.k if target_weight is None else target_weight)
    gammas = LAGRANGIAN_GAMMAS if gammas is None else tuple(
        float(g) for g in np.ravel(gammas))
    betas = LAGRANGIAN_BETAS if betas is None else tuple(
        float(b) for b in np.ravel(betas))
    args = _lagrangian_args(target, gammas, betas, maxiter)
    memo_key = (_instance_key(inst), json.dumps(args, sort_keys=True))
    if memo_key in _memo:
        layers, info = _memo[memo_key]
        return list(layers), dict(info)
    if use_cache:
        cached = _load_lagrangian_cache(inst, args)
        if cached is not None:
            _memo[memo_key] = cached
            return list(cached[0]), dict(cached[1])

    t0 = _time.time()
    lc = _lightcone_of(inst)
    grid, best = [], None
    for g in gammas:
        for b in betas:
            hit = _ThetaProfile(lc, g, b).best_root(target)
            row = {"gamma": g, "beta": b, "theta": None, "exp_f": None,
                   "exp_W": None, "sd_W": None}
            if hit is not None:
                th, (w, f, v) = hit
                row.update(theta=float(th), exp_f=float(f), exp_W=float(w),
                           sd_W=float(np.sqrt(max(v, 0.0))))
                if best is None or f > best["exp_f"]:
                    best = row
            grid.append(row)
    if best is None:
        raise ValueError(f"no (gamma, beta) on the grid reaches an ideal mean "
                         f"Hamming weight of {target:g}")

    def neg_f(x):
        hit = _ThetaProfile(lc, x[0], x[1]).best_root(target)
        return np.inf if hit is None else -float(hit[1][1])

    dg = 0.5 * (abs(gammas[1] - gammas[0]) if len(gammas) > 1 else 0.1)
    db = 0.5 * (abs(betas[1] - betas[0]) if len(betas) > 1 else 0.1)
    x0 = np.array([best["gamma"], best["beta"]])
    res = minimize(neg_f, x0, method="Nelder-Mead",
                   options={"maxiter": int(maxiter), "xatol": 1e-4,
                            "fatol": 1e-6,
                            "initial_simplex": [x0, x0 + [dg, 0.0],
                                                x0 + [0.0, db]]})
    gamma, beta = float(best["gamma"]), float(best["beta"])
    refined = bool(np.isfinite(res.fun) and -res.fun > best["exp_f"])
    if refined:
        gamma, beta = float(res.x[0]), float(res.x[1])
    th, (w, f, v) = _ThetaProfile(lc, gamma, beta).best_root(target)
    var_w = max(float(v), 1e-12)
    info = {"gamma": gamma, "theta": float(th), "beta": beta,
            "exp_f": float(f), "exp_W": float(w), "var_W": float(v),
            "sd_W": float(np.sqrt(var_w)),
            "p_feasible": float(1.0 / np.sqrt(2.0 * np.pi * var_w)),
            "target": target, "grid": grid,
            "grid_best": {k: best[k] for k in ("gamma", "beta", "theta",
                                                "exp_f")},
            "refined": refined, "refine_iterations": int(res.nit),
            "seconds": float(_time.time() - t0), "source": "computed"}
    layers = [(gamma, float(th), beta)]
    _memo[memo_key] = (layers, info)
    return list(layers), dict(info)


def write_lagrangian_cache(inst: DensestKSubgraph, path=None, **kwargs) -> Path:
    """Compute :func:`lagrangian_layers` (no cache) and write it to
    ``calibration/lagrangian_106.json`` (organizer helper)."""
    path = CALIBRATION_DIR / LAGRANGIAN_CACHE if path is None else Path(path)
    layers, info = lagrangian_layers(inst, use_cache=False, **kwargs)
    target = info["target"]
    gammas = kwargs.get("gammas")
    betas = kwargs.get("betas")
    args = _lagrangian_args(
        target, LAGRANGIAN_GAMMAS if gammas is None else np.ravel(gammas),
        LAGRANGIAN_BETAS if betas is None else np.ravel(betas),
        kwargs.get("maxiter", LAGRANGIAN_MAXITER))
    info = {k: v for k, v in info.items() if k != "source"}
    body = {"what": "hw.lagrangian_layers result for the pinned demo instance",
            "instance": {"name": inst.name, "n": inst.n, "k": inst.k,
                         "edges": len(inst.edges)},
            "instance_key": _instance_key(inst), "args": args,
            "layers": [[float(a) for a in layer] for layer in layers],
            "info": info}
    Path(path).write_text(json.dumps(body, indent=1) + "\n")
    return Path(path)


def theta_for_weight(inst: DensestKSubgraph, gamma: float, beta: float,
                     target: float) -> float:
    """The LCU angle theta in [0, 2 pi) at which the ideal mean Hamming
    weight of the p = 1 basis circuit equals ``target`` (step 2 of
    :func:`lagrangian_layers` for one (gamma, beta)): of the at most two
    roots, the one with the larger ideal <f>. Raises ``ValueError`` naming
    the reachable range of <W> when no theta reaches the target."""
    prof = _ThetaProfile(_lightcone_of(inst), gamma, beta)
    hit = prof.best_root(target)
    if hit is None:
        lo, hi = prof.weight_range()
        raise ValueError(f"no theta gives an ideal mean Hamming weight of "
                         f"{float(target):g} at gamma = {float(gamma):g}, "
                         f"beta = {float(beta):g}: reachable range "
                         f"[{lo:.2f}, {hi:.2f}]")
    return float(hit[0])


def theta_profile(inst: DensestKSubgraph, gamma: float, beta: float,
                  thetas) -> dict:
    """Ideal <W>, <f> and Var W of the p = 1 basis circuit at fixed
    (gamma, beta) as functions of the LCU angle theta, exact at any n (the
    five-point trigonometric interpolation behind :func:`lagrangian_layers`;
    read-only). Returns arrays ``theta``, ``exp_W``, ``exp_f``, ``var_W``."""
    th = np.asarray(thetas, dtype=float).reshape(-1)
    m = _ThetaProfile(_lightcone_of(inst), gamma, beta).moments(th)
    return {"theta": th, "exp_W": m[:, 0], "exp_f": m[:, 1], "var_W": m[:, 2]}


class WeightStats(tuple):
    """``(mean, sd, feasible)`` of the measured Hamming weight: its mean,
    standard deviation and the fraction of shots with weight k."""

    __slots__ = ()
    _fields = ("mean", "sd", "feasible")

    def __new__(cls, mean, sd, feasible):
        return super().__new__(cls, (float(mean), float(sd), float(feasible)))

    mean = property(lambda self: self[0])
    sd = property(lambda self: self[1])
    feasible = property(lambda self: self[2])

    def __repr__(self) -> str:
        return (f"WeightStats(mean={self[0]:.4f}, sd={self[1]:.4f}, "
                f"feasible={self[2]:.4f})")


def weight_stats(inst: DensestKSubgraph, counts: dict) -> WeightStats:
    """Mean and standard deviation of the Hamming weight W of the measured
    bitstrings and the feasible fraction P(W = k), from a counts dict (or
    a list of counts dicts, pooled with equal shot weight)."""
    if isinstance(counts, (list, tuple)):
        merged: dict = {}
        for c in counts:
            for key, v in c.items():
                merged[key] = merged.get(key, 0) + v
        counts = merged
    bits, cnt = bits_from_counts(counts, inst.n)
    w = bits.sum(axis=1).astype(float)
    total = cnt.sum()
    mean = float(np.dot(w, cnt) / total)
    sd = float(np.sqrt(np.dot((w - mean) ** 2, cnt) / total))
    return WeightStats(mean, sd, float(cnt[w == inst.k].sum() / total))


# ------------------------------------------------------------ OPT and B
def milp_optimum(inst: DensestKSubgraph, time_limit: float = 120.0,
                 window: tuple[int, int] | None = None):
    """Exact densest-k-subgraph optimum via HiGHS (scipy.optimize.milp).

    ``window=(lo, hi)`` replaces 1^T x = k by lo <= 1^T x <= hi (the
    Hamming-weight window variant, unscored kit capability since 2026-09-29;
    with a dominant lam_ind the optimum of the indicator objective is the
    best in-window f, which is what this returns).

    Linearization y_e = x_i x_j via y_e <= x_i, y_e <= x_j and
    y_e >= x_i + x_j - 1 (exact for weights of either sign). Returns a dict:
    ``value`` (objective of the returned x), ``x``, ``proven`` (HiGHS status
    0 = optimal), ``upper_bound`` (the solver's dual bound - an absolute
    upper bound on OPT when the time limit hit), ``gap`` (HiGHS's *relative*
    MIP gap), ``status``, ``message``."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    n, E = inst.n, list(inst.edges)
    m = len(E)
    c = np.concatenate([np.zeros(n), -np.array([w for _, _, w in E])])
    A = lil_matrix((3 * m + 1, n + m))
    for e, (i, j, _) in enumerate(E):
        A[3 * e, n + e] = 1          # y_e - x_i <= 0
        A[3 * e, i] = -1
        A[3 * e + 1, n + e] = 1      # y_e - x_j <= 0
        A[3 * e + 1, j] = -1
        A[3 * e + 2, n + e] = 1      # y_e - x_i - x_j >= -1
        A[3 * e + 2, i] = -1
        A[3 * e + 2, j] = -1
    A[3 * m, :n] = 1
    lo, hi = (inst.k, inst.k) if window is None else window
    lb = np.concatenate([np.tile([-np.inf, -np.inf, -1.0], m), [lo]])
    ub = np.concatenate([np.tile([0.0, 0.0, np.inf], m), [hi]])
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lb, ub),
               integrality=np.concatenate([np.ones(n), np.zeros(m)]),
               bounds=Bounds(0, 1), options={"time_limit": time_limit})
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    x = np.round(res.x[:n]).astype(np.int8)
    dual = getattr(res, "mip_dual_bound", None)
    return {"value": float(inst.objective_of_bitstring(x)), "x": x,
            "status": int(res.status), "proven": res.status == 0,
            "upper_bound": float(-dual) if dual is not None else float("nan"),
            "gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
            "message": res.message}


def baseline_B_mc(inst: DensestKSubgraph, alpha: float = 0.10,
                  samples: int = 200_000, seed: int = 2026,
                  window: tuple[int, int] | None = None,
                  lam_ind: float | None = None) -> float:
    """Monte-Carlo CVaR_alpha of the warm start (B) at any n: of f_lam, or
    of the Hamming-weight window variant (unscored kit capability since
    2026-09-29), f - lam_ind * 1[weight outside ``window``]
    (lam_ind defaults to the dominance value 1 + sum |w|)."""
    rng = np.random.default_rng(seed)
    bits = (rng.random((samples, inst.n)) < inst.k / inst.n).astype(np.uint8)
    f, w = inst.evaluate_bits(bits)
    if window is None:
        val = f - inst.lam * (w - inst.k) ** 2
    else:
        lo, hi = window
        lam_ind = (1.0 + sum(abs(x) for _, _, x in inst.edges)
                   if lam_ind is None else float(lam_ind))
        val = f - lam_ind * ((w < lo) | (w > hi))
    return float(cvar(val, np.ones(samples), alpha))


# ------------------------------------------------------- JSON round trip
def instance_to_dict(inst: DensestKSubgraph) -> dict:
    return {"n": inst.n, "k": inst.k, "name": inst.name,
            "edges": [[int(i), int(j), float(w)] for i, j, w in inst.edges]}


def instance_from_dict(d: dict, instance_id: str | None = None) -> DensestKSubgraph:
    """The instance of an :func:`instance_to_dict` record. ``instance_id``
    (default: the record's own ``"instance_id"``, else "") is attached to
    the instance."""
    if instance_id is None:
        instance_id = d.get("instance_id", "")
    return DensestKSubgraph(n=int(d["n"]), k=int(d["k"]),
                            edges=tuple(sorted((int(i), int(j), float(w))
                                               for i, j, w in d["edges"])),
                            name=d.get("name", "instance"),
                            instance_id=str(instance_id or ""))


def layout_to_dict(layout: HardwareLayout) -> dict:
    return {"physical": [int(q) for q in layout.physical],
            "native_edges": [[int(a), int(b)] for a, b in layout.native_edges],
            "swap_layers": [[[int(a), int(b)] for a, b in layer]
                            for layer in layout.swap_layers],
            "backend_name": layout.backend_name}


def layout_from_dict(d: dict) -> HardwareLayout:
    return HardwareLayout(physical=list(d["physical"]),
                          native_edges=[tuple(e) for e in d["native_edges"]],
                          swap_layers=[[tuple(e) for e in layer]
                                       for layer in d["swap_layers"]],
                          backend_name=d.get("backend_name", ""))


def load_handout(path) -> tuple[DensestKSubgraph, HardwareLayout]:
    """Read a handout file ({"instance_id", "backend", "alpha", "instance",
    "layout"}, plus "team" in the files of ``organizer/generate_instances.py``)
    and return ``(inst, layout)`` - the pair every utility-scale entry point
    here takes. At the event this replaces the ``instance_from_backend``
    call: the organizers ship the graph and the physical layout, so every
    team scores the same instance on the same qubits.

    ``inst.instance_id`` is the handout's top-level ``"instance_id"`` (the
    id the grading server knows the instance by), else the ``"instance_id"``
    of its ``"instance"`` record, else "". The demo cache
    ``calibration/demo_instance_106.json`` is a handout too (id
    ``"demo106"``)."""
    with open(path, "r") as fh:
        d = json.load(fh)
    inst = instance_from_dict(d["instance"], instance_id=d.get("instance_id"))
    return inst, layout_from_dict(d["layout"])


# --------------------------------------------------------------- wire format
# QPY policy (plan Sec. 12.4 item 9): the kit writes QPY format version 17
# (qiskit 2.5) and refuses anything newer before loading - qiskit 2.6 writes
# 18, which 2.5 cannot read, and a loader crash must never look like a
# malformed submission. The hash that identifies a submission is the sha256
# of the *received bytes*: ``qpy.dump`` is not byte-deterministic for
# circuits carrying a layout, so re-dumping never reproduces a hash.
KIT_QPY_VERSION = 17
QPY_PREFACE = b"QISKIT"


def qpy_bytes(circuits, version: int = KIT_QPY_VERSION) -> bytes:
    """Serialize ``circuits`` in the kit's pinned QPY format version."""
    buf = io.BytesIO()
    qpy.dump(list(circuits), buf, version=version)
    return buf.getvalue()


def qpy_info(data: bytes) -> dict:
    """Header facts of a QPY payload without loading it: ``sha256`` of the
    bytes, ``qpy_version``, ``writer_qiskit`` (the qiskit version that wrote
    it), ``num_programs`` and ``n_bytes``. Raises ``ValueError`` on a payload
    that is not QPY at all."""
    import hashlib
    import struct

    from qiskit.qpy import formats

    data = bytes(data)
    if len(data) < formats.FILE_HEADER_V10_SIZE or not data.startswith(QPY_PREFACE):
        raise ValueError("not a QPY payload (missing 'QISKIT' preface)")
    hdr = formats.FILE_HEADER_V10._make(
        struct.unpack(formats.FILE_HEADER_V10_PACK,
                      data[:formats.FILE_HEADER_V10_SIZE]))
    return {"sha256": hashlib.sha256(data).hexdigest(),
            "qpy_version": int(hdr.qpy_version),
            "writer_qiskit": f"{hdr.major_version}.{hdr.minor_version}."
                             f"{hdr.patch_version}",
            "num_programs": int(hdr.num_programs), "n_bytes": len(data)}


def circuits_from_qpy(data: bytes, max_version: int = KIT_QPY_VERSION) -> list:
    """Load circuits from QPY bytes, refusing a format newer than the kit's
    pin *before* the loader runs (``ValueError`` names both versions)."""
    info = qpy_info(data)
    if info["qpy_version"] > max_version:
        raise ValueError(
            f"QPY format version {info['qpy_version']} (written by qiskit "
            f"{info['writer_qiskit']}) is newer than the kit's pin "
            f"{max_version}: re-export with qiskit 2.5 or "
            f"qpy.dump(..., version={max_version})")
    return list(qpy.load(io.BytesIO(data)))


# ------------------------------------------------------- phoenix snapshot
CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "calibration"


def load_phoenix_snapshot(path=None) -> tuple:
    """``(target, facts)`` of ``ibm_phoenix`` as pinned on 2026-09-29
    (``calibration/phoenix_target_snapshot.pkl``, metadata only - no job was
    run). The Target carries the coupling map, gate durations and error
    rates of that day; use it for instance growth, transpilation and the
    duration-based cost table when no account is at hand. At the event the
    live ``service.backend("ibm_phoenix").target`` replaces it."""
    import pickle

    path = CALIBRATION_DIR / "phoenix_target_snapshot.pkl" if path is None else Path(path)
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return d["target"], d["facts"]


class SnapshotBackend:
    """The slice of ``BackendV2`` the kit and the notebook read from a
    backend - ``target``, ``coupling_map``, ``name``, ``num_qubits``,
    ``default_rep_delay``, ``rep_delay_range`` - served from a pinned Target
    snapshot. It cannot run anything: it is for growing instances,
    transpiling and costing circuits without an account."""

    def __init__(self, target, facts: dict):
        self.target, self.facts = target, dict(facts)
        self.coupling_map = target.build_coupling_map()
        self.name = f"{facts.get('backend', 'snapshot')} (snapshot {facts.get('snapshot_utc', '')[:10]})"
        self.num_qubits = target.num_qubits
        self.default_rep_delay = facts.get("default_rep_delay_s")
        self.rep_delay_range = tuple(facts.get("rep_delay_range_s", ()))
        self.dt = target.dt

    def run(self, *_, **__):
        raise RuntimeError("a Target snapshot cannot execute circuits: use "
                           "the QPU (or Aer at n <= 24 in the mock grader)")


def phoenix_snapshot_backend(path=None) -> SnapshotBackend:
    """``SnapshotBackend`` over :func:`load_phoenix_snapshot`."""
    return SnapshotBackend(*load_phoenix_snapshot(path))


def check_isa(circuits, target) -> list[str]:
    """Findings (empty = fine) for circuits that are not ISA for ``target``:
    an operation the target lacks, or a gate on qubits / an edge the target
    does not support. Mirrors the runtime's ``validate_isa_circuits`` but
    returns reviewer-facing strings instead of raising."""
    findings = []
    names = set(target.operation_names)
    for i, qc in enumerate(circuits):
        if qc.num_qubits > target.num_qubits:
            findings.append(f"circuit {i}: {qc.num_qubits} qubits > target's "
                            f"{target.num_qubits}")
            continue
        seen = set()
        for ci in qc.data:
            name = ci.operation.name
            if name == "barrier":
                continue
            qargs = tuple(qc.find_bit(q).index for q in ci.qubits)
            if name not in names:
                seen.add(f"'{name}' is not a target operation")
                continue
            if not target.instruction_supported(name, qargs):
                seen.add(f"'{name}' on {list(qargs)} is not supported")
        findings.extend(f"circuit {i}: {msg}" for msg in sorted(seen))
    return findings


# ------------------------------------------------ compilation and its cost
def compact_global_phase(phase):
    """An equivalent short form of a symbolic global phase.

    With a free ``Parameter`` in the circuit, every block that the
    resynthesis rewrites adds a constant to the symbolic global phase, which
    grows to thousands of nested terms (about 41,000 characters for the
    106-qubit one-layer circuit with a free ``theta``). QPY cannot carry
    such a phase: the reader of qiskit 2.5 fails on the circuit header
    ("binary parsing error ... register_type"), and so does the IBM Quantum
    service, which refuses the job. The phases of the kit's circuits are
    linear in the free parameters, so the phase is rebuilt as
    ``c + sum_p a_p * p`` with ``c`` reduced modulo 2 pi. Returns ``phase``
    unchanged when it is a number, and ``0.0`` when it is not linear in its
    parameters (a global phase has no effect on measured samples)."""
    from qiskit.circuit import ParameterExpression

    if not isinstance(phase, ParameterExpression):
        return phase
    params = sorted(phase.parameters, key=lambda p: p.name)
    zero = {p: 0.0 for p in params}
    c0 = float(phase.bind(zero))
    coeffs = []
    for p in params:
        a = float(phase.bind({**zero, p: 1.0})) - c0
        coeffs.append(float(f"{a:.12g}"))        # drop the rounding of a long sum
    probe = {p: 0.37 + 0.61 * i for i, p in enumerate(params)}
    want = float(phase.bind(probe))
    got = c0 + sum(a * probe[p] for a, p in zip(coeffs, params))
    if abs(want - got) > 1e-6 * max(1.0, abs(want)):
        return 0.0
    out = float(c0 % (2.0 * np.pi))
    for a, p in zip(coeffs, params):
        if a != 0.0:
            out = out + a * p
    return out


def _compact_phase_pass():
    from qiskit.transpiler.basepasses import TransformationPass

    class CompactGlobalPhase(TransformationPass):
        """Rewrite a symbolic global phase in its short equivalent form
        (:func:`compact_global_phase`); a numeric phase is left alone."""

        def run(self, dag):
            dag.global_phase = compact_global_phase(dag.global_phase)
            return dag

    return CompactGlobalPhase()


def resynth_pass_manager(target, initial_layout, seed_transpiler: int = 1):
    """Level-1 preset pass manager followed by two-qubit block resynthesis:
    ``Collect2qBlocks`` -> ``ConsolidateBlocks(force_consolidate=True)`` ->
    ``UnitarySynthesis`` -> ``Optimize1qGatesDecomposition``, all for
    ``target`` (appended as the preset's ``post_optimization`` stage, so a
    single ``.run`` keeps the layout bookkeeping of the level-1 run).

    Every maximal run of gates on one qubit pair is replaced by its optimal
    exact synthesis (<= 3 CZ). Paired with ``hardware_basis_circuit_p(...,
    swap_last=True)`` this merges each swap with the cost gate on the same
    pair (2 + 3 CZ -> <= 3 CZ). Exact (approximation degree 1): the output
    is equivalent to the plain level-1 output up to global phase.

    A last pass writes a symbolic global phase in its short form
    (:func:`compact_global_phase`): without it a circuit with a free
    parameter cannot be serialized, and the service refuses the job."""
    from qiskit.transpiler import PassManager, generate_preset_pass_manager
    from qiskit.transpiler.passes import (Collect2qBlocks, ConsolidateBlocks,
                                          Optimize1qGatesDecomposition,
                                          UnitarySynthesis)

    pm = generate_preset_pass_manager(optimization_level=1, target=target,
                                      initial_layout=initial_layout,
                                      seed_transpiler=seed_transpiler)
    extra = [Collect2qBlocks(),
             ConsolidateBlocks(force_consolidate=True, target=target),
             UnitarySynthesis(target=target),
             Optimize1qGatesDecomposition(target=target),
             _compact_phase_pass()]
    if pm.post_optimization is None:
        pm.post_optimization = PassManager(extra)
    else:
        pm.post_optimization.append(extra)
    return pm


# layered two-qubit error used by the notebook's fidelity proxy
PROXY_ERR_2Q = 3.3e-3


def circuit_cost(isa: QuantumCircuit, target, err_2q: float = PROXY_ERR_2Q) -> dict:
    """Cost-table row of an ISA circuit: ``cz`` (two-qubit gate count - all
    CZ on a CZ-native target such as ibm_phoenix), ``depth_2q`` (two-qubit
    depth), ``depth``, ``duration_us`` (``estimate_duration`` on
    ``target``) and ``fidelity_proxy`` = (1 - err_2q)^cz."""
    two_q = lambda ci: len(ci.qubits) == 2 and ci.operation.name != "barrier"  # noqa: E731
    cz = sum(1 for ci in isa.data if two_q(ci))
    return {"cz": int(cz),
            "depth_2q": int(isa.depth(two_q)),
            "depth": int(isa.depth()),
            "duration_us": float(1e6 * isa.estimate_duration(target, unit="s")),
            "fidelity_proxy": float((1.0 - err_2q) ** cz)}
