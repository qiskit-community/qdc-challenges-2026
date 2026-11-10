"""Gate-level CP-SAT scheduling of folded Trotter steps for the Iceberg-encoded 2+1D MFIM.

Follows Methods C of Froland et al., arXiv:2607.24947, on top of `iceberg_layout.Placement`
(used unmodified).

Classes
-------
ParamA
    The schedule parameters A = {o, l, w}: block orientation, logical relabeling, and connection
    choice per edge.
TrotterStep
    One Trotter step for a fixed A: its units (inter-block gadgets, frame toggles, local hosts) and
    their individual gates. ``TrotterStep.schedule()`` runs the gate-level CP-SAT;
    ``TrotterStep.trotter_order()`` gives the logical product formula the step implements.
GateSchedule
    The CP-SAT result: a time slot for every gate. Builds encoded circuits, checks them exactly
    (``verify``, Pauli propagation; ``expectations``, statevector) and prints the schedule.
ScheduleOptimizer
    Coordinate descent over A, scoring each candidate by its gate-level depth.
UnencodedBaseline
    Unencoded circuits with the same Trotter order: exact statevector, quimb MPS, and a version
    transpiled for hardware.

Module-level functions hold the circuit helpers shared with the workflow notebook (state prep,
syndrome extraction, measurement) and small utilities.

Physical sites of a block, used throughout. These are fixed positions; which *role* occupies s1
and s3 depends on the frame::

      s0 - s1 - s2        compute configuration, frame 0:
           |                s0 = d0, s1 = d1, s2 = az, s3 = d2, s4 = d3, s5 = ax
           s3             frame 1 swaps the roles on s1 and s3 (d1 <-> d2)
           |
      s5 - s4

Run ``python iceberg_schedule.py`` for a quick self-test on the two-block toy lattice.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from itertools import combinations, zip_longest

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Measure
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Pauli, Statevector, StabilizerState
from qiskit.transpiler import CouplingMap

from . import iceberg_device as device
from . import iceberg_layout as lay
from .iceberg_block import X_SUPPORTS, Z_SUPPORTS

#: Filter for ``QuantumCircuit.depth``: count only two-qubit operations.
TWO_Q = lambda instr: len(instr.qubits) == 2

#: ``ortools.sat.python.cp_model``, imported on first use by `_cp_model`.
cp_model = None


def _cp_model():
    """The CP-SAT module, loaded lazily.

    Only `TrotterStep.schedule` and `ScheduleOptimizer.relabel` solve CP-SAT models, so OR-Tools
    is imported when one of them runs rather than at module import. That keeps the rest of the
    module -- circuit helpers, `ParamA`, `GateSchedule`, `UnencodedBaseline` -- usable where
    OR-Tools is not installed.
    """
    global cp_model
    if cp_model is None:
        try:
            from ortools.sat.python import cp_model as module
        except ModuleNotFoundError as err:          # re-raise with the install hint
            raise ModuleNotFoundError(
                "CP-SAT scheduling needs OR-Tools, which is not installed in this environment. "
                "Install it with `pip install 'ortools>=9.10'` (see requirements.txt). The rest "
                "of iceberg_schedule.py does not need it."
            ) from err
        cp_model = module
    return cp_model



# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def snake_order(lattice):
    """Lattice sites in snake order: rows top to bottom, alternating left-to-right and
    right-to-left. Used as the 1D qubit order for MPS simulations."""
    rows = sorted({r for r, c in lattice.sites})
    out = []
    for k, r in enumerate(rows):
        cols = sorted(c for rr, c in lattice.sites if rr == r)
        for c in (cols if k % 2 == 0 else reversed(cols)):
            out.append(lattice.sites[(r, c)])
    return out


def device_coupling_map(topology=None):
    """Bidirectional Qiskit CouplingMap of the device (default: built-in ibm_boston)."""
    edges = [list(e) for e in device.as_topology(topology).edges]
    return CouplingMap(edges + [e[::-1] for e in edges])


def transpiled_stats(qc, coupling, optimization_level=2, seed=1):
    """(2q depth, CZ count) after transpiling to cz/rz/sx/x on the device with the layout fixed.

    SWAPs are decomposed first: otherwise the transpiler may absorb them into a qubit relabeling,
    which makes a final syndrome round look cheap and a mid-circuit one expensive.
    """
    qc = qc.decompose(gates_to_decompose=["swap"])
    tq = transpile(qc, basis_gates=["cz", "rz", "sx", "x", "measure", "reset"],
                   coupling_map=coupling, initial_layout=list(range(qc.num_qubits)),
                   optimization_level=optimization_level, seed_transpiler=seed)
    return tq.depth(TWO_Q), tq.count_ops().get("cz", 0)


def max_diff(a, b):
    """Largest |a[k] - b[k]| over the keys of ``a``."""
    return max(abs(a[k] - b[k]) for k in a)


def compare(ref, enc, stab):
    """(max |reference - encoded| over all observables, smallest stabilizer expectation)."""
    return max_diff(ref, enc), min(min(v) for v in stab.values())


def fanout_forward(path, m):
    """Forward half of an inter-block CNOT fanout.

    Two arms march inward to ``path[m]``: arm a is ``path[0] -> path[1] -> ... -> path[m]``, arm b
    is ``path[-1] -> path[-2] -> ... -> path[m]``. Their gates are interleaved.

    Returns
    -------
    list of (control, target)
    """
    L = len(path)
    arms = ([(path[i], path[i + 1]) for i in range(0, m)],
            [(path[i], path[i - 1]) for i in range(L - 1, m, -1)])
    return [g for pair in zip_longest(*arms) for g in pair if g is not None]


def alap_layers(gates):
    """ALAP 2q-layer index of each gate in a forward fanout, and the fanout's depth.

    Returns
    -------
    (layers, depth)
    """
    # ASAP on the reversed list is ALAP on the forward list
    free, rev = {}, []
    for c, t in reversed(gates):
        layer = max(free.get(c, 0), free.get(t, 0))
        rev.append(layer)
        free[c] = free[t] = layer + 1
    depth = max(free.values())
    return [depth - 1 - x for x in reversed(rev)], depth


def best_m(path):
    """R_Z position minimizing the fanout depth (ties: most central, then lowest index)."""
    L = len(path)
    return min(range(1, L - 1),
               key=lambda m: (alap_layers(fanout_forward(path, m))[1], abs(m - (L - 1) / 2), m))


_PLACEMENT_CACHE = {}


def placement_for(A, block_qubits, topology=None, bridges=lay.BRIDGES):
    """The (unmodified) Placement class, built with A's orientations via ``az_qubits``."""
    key = (tuple(sorted(A.orient.items())), tuple(sorted(block_qubits)), id(topology))
    if key not in _PLACEMENT_CACHE:
        forks = lay.fork_candidates(block_qubits, topology)
        az = {b: forks[b][A.orient[b]] for b in block_qubits}
        _PLACEMENT_CACHE[key] = lay.Placement.from_qubits(
            placement=block_qubits, bridges=bridges, az_qubits=az, topology=topology)
    return _PLACEMENT_CACHE[key]


# ---------------------------------------------------------------------------
# Parameterization and units
# ---------------------------------------------------------------------------


def _connection_name(connection):
    """Short name of a physical connection between two blocks, e.g. "via bridge 91"."""
    if connection.bridge is None:
        return f"direct {connection.a_qubit}-{connection.b_qubit}"
    return f"via bridge {connection.bridge}"


@dataclass
class ParamA:
    """The schedule parameters A = {o, l, w} of Methods C.

    Attributes
    ----------
    orient : dict
        ``block -> o_b`` in {0, 1}. Index into ``lay.fork_candidates()[block]`` of the fork
        leaf that carries ``az`` (the other leaf becomes ``d0``).
    relabel : dict
        ``block -> l_b`` in {0, 1}. Lattice site ``(b, l)`` is carried by code logical qubit
        ``(b, l XOR l_b)``.
    conn : dict
        ``inter-block lattice edge -> w_e``. Index into ``place.connection_routes(u, v)``
        (shortest first). Every edge whose blocks have more than one connection gets an explicit
        entry (0 = shortest) when built by ``default(lattice, place)`` or `ScheduleOptimizer`.
        Edges with a single connection have no entry; for those, 0 is the only possible value.
    conn_names : dict
        ``edge -> [name of each connection option]`` in index order, e.g.
        ``["via bridge 91", "via bridge 115"]``. Display only: when present, the printout shows
        the chosen connection instead of its index. Filled by ``default(lattice, place)``.
    """
    orient: dict
    relabel: dict
    conn: dict
    conn_names: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def default(cls, lattice, place=None):
        """All-zero A for ``lattice``: default orientations, no relabeling, shortest routes.

        Parameters
        ----------
        place : lay.Placement, optional
            If given, ``conn`` lists every inter-block edge that has more than one connection,
            with its default choice 0, and ``conn_names`` records what each choice is. Without
            it, ``conn`` is empty, which means the same thing.
        """
        blocks = lattice.blocks()
        conn, names = {}, {}
        if place is not None:
            for u, v in lattice.inter_edges():
                edge = tuple(sorted((u, v)))
                # relabel is all 0 here, so lattice sites are their own code qubits
                options = list(place.connection_routes(*edge))   # Connections, shortest first
                if len(options) > 1:
                    conn[edge] = 0
                    names[edge] = [_connection_name(c) for c in options]
        return cls(orient={b: 0 for b in blocks}, relabel={b: 0 for b in blocks}, conn=conn,
                   conn_names=names)

    def copy(self):
        """Independent copy (the three dicts are copied, so edits do not alias)."""
        return ParamA(dict(self.orient), dict(self.relabel), dict(self.conn),
                      dict(self.conn_names))

    def key(self):
        """Hashable form of A, used to cache evaluations."""
        return (tuple(sorted(self.orient.items())), tuple(sorted(self.relabel.items())),
                tuple(sorted(self.conn.items())))

    def code_qubit(self, site):
        """Code logical qubit ``(b, l')`` carrying lattice site ``(b, l)``, with l' = l XOR l_b."""
        b, l = site
        return (b, l ^ self.relabel.get(b, 0))

    def __repr__(self):
        # o and l as bit strings in block order; w as "coupling: chosen connection"
        o = "".join(str(self.orient[b]) for b in sorted(self.orient))
        l = "".join(str(self.relabel[b]) for b in sorted(self.relabel))
        w = ", ".join(
            f"{u}-{v}: {self.conn_names[(u, v)][k] if (u, v) in self.conn_names else k}"
            for (u, v), k in sorted(self.conn.items()))
        return f"ParamA(o={o}, l={l}, w={{{w}}})"



@dataclass
class Unit:
    """One logical piece of a Trotter step: an inter-block gadget, a frame toggle, or a local host.

    Attributes
    ----------
    name : str
        ``"G<u>-<v>"`` for a gadget on lattice edge (u, v), ``"T<b>"`` for block b's toggle,
        ``"L<b>.<f>"`` for a local host of block b at frame f.
    kind : str
        ``'gadget'`` (inter-block RZZ fanout), ``'toggle'`` (SWAP(d1, d2) = frame flip),
        or ``'local'`` (a CNOT pair that only hosts single-qubit folds).
    frames : dict
        ``block -> frame`` each touched block must be in while the unit runs (empty for toggles).
    block : int or None
        The block a toggle or local unit acts on.
    edge : tuple or None
        Gadget only: the lattice-site edge it implements.
    pair : tuple or None
        Toggle: physical ``(s1, s3)``. Local: physical ``(s3, s4)``.
    route : lay.Route or None
        Gadget only: the fanout path chosen by ``A.conn``.
    m : int or None
        Gadget only: index into ``route.path`` of the qubit that receives the central R_Z.
    """
    name: str
    kind: str
    frames: dict = field(default_factory=dict)
    block: int | None = None
    edge: tuple | None = None
    pair: tuple | None = None
    route: object = None
    m: int | None = None


@dataclass
class Gate:
    """One gate of a Trotter step.

    Attributes
    ----------
    id : int
        Index into ``TrotterStep.gates``.
    kind : str
        ``'cx'``, ``'rz'`` or ``'rx'``.
    qubits : tuple
        ``(control, target)`` for a CX, ``(qubit,)`` for a rotation (physical indices).
    angle : str or None
        For rotations, which coupling sets the angle: ``'zz'`` (J), ``'x'`` (gx) or ``'z'``
        (gz). The numeric angle is only filled in when the step is appended to a circuit.
    owner : str
        Name of the unit or parity group the gate belongs to.
    """
    id: int
    kind: str
    qubits: tuple
    angle: str | None = None
    owner: str = ""


def noncommuting(g, h):
    """True if gates g and h do not commute.

    * CX vs CX: they fail to commute only if one's control is the other's target.
    * rotation vs rotation: only RZ and RX on the same qubit fail to commute.
    * RZ vs CX: fails only on the CX target. RX vs CX: only on the CX control.
    """
    if g.kind == "cx" and h.kind == "cx":
        return g.qubits[0] == h.qubits[1] or g.qubits[1] == h.qubits[0]
    if g.kind != "cx" and h.kind != "cx":
        return g.qubits == h.qubits and g.kind != h.kind
    cx, r = (g, h) if g.kind == "cx" else (h, g)
    q = r.qubits[0]
    return q == (cx.qubits[1] if r.kind == "rz" else cx.qubits[0])


# ---------------------------------------------------------------------------
# One Trotter step for a fixed A
# ---------------------------------------------------------------------------


class TrotterStep:
    """One Trotter step for a fixed parameterization A: its units and their individual gates.

    **Units.** One gadget per inter-block lattice edge (route from ``A.conn``, R_Z position from
    `best_m`), one frame toggle per block, and a local host for any (block, frame) that no gadget
    visits (only if some logical qubit has no inter-block edge).

    **Frame structure.** Every block starts the step at f=0, runs all its f=0 gadgets, toggles
    once (SWAP(d1, d2), hosting the intra-block RZZ), then runs its f=1 gadgets and exits at f=1.
    Odd-numbered steps run backwards. Some A admit no such ordering; see `is_consistent`.

    **Gates.** Each gadget arm's first and last CNOT (far -> near) is pulled into a **parity
    group** keyed by (block, frame, far, near), so gadgets of the same block and frame that start
    from the same support pair share one CNOT pair. For each (block, frame), the group with the
    most members hosts the folded R_Zbar_(b,f) (on near) and R_Xbar_(b,1-f) (on far): in frame f
    a support pair's ZZ parity is Zbar_(b,f) and its XX parity is Xbar_(b,1-f).

    Parameters
    ----------
    A : ParamA
    lattice : lay.LogicalLattice
        In lattice-site labels; ``A.relabel`` maps them to code qubits.
    block_qubits : dict
        ``block -> its six physical qubits`` (e.g. a subset of ``lay.PLACEMENT``).
    topology : optional
        Device connectivity; None means the built-in ibm_boston map.

    Attributes
    ----------
    A, lattice, place
        The parameterization, lattice, and (unmodified) Placement built with A's orientations.
    units : dict
        ``unit name -> Unit``.
    gates : list of Gate
        Every gate of the step; ``gates[i].id == i``.
    unit_gates : dict
        ``unit name -> [gate ids]`` in reference order. A gadget's list excludes its arms' first
        and last CNOTs, which belong to parity groups.
    groups : dict
        ``group name -> {"key", "C", "U", "folds", "members"}``: the shared opening and closing
        CX ids, the folded rotation ids (host group only), and the member gadget names.
    unit_groups : dict
        ``unit name -> [group names it belongs to]``.

    Raises
    ------
    ValueError
        If some coupling has no canonical route under A.
    """

    def __init__(self, A, lattice, block_qubits, topology=None):
        self.A = A
        self.lattice = lattice
        self.block_qubits = block_qubits
        self.topology = topology
        self.place = placement_for(A, block_qubits, topology)
        self.units = self._build_units()
        self.gates, self.unit_gates, self.groups, self.unit_groups = self._build_gates()

    # -- construction -----------------------------------------------------------

    def _build_units(self):
        """Gadgets, toggles and local hosts of the step."""
        A, lattice, place = self.A, self.lattice, self.place
        units = {}
        has_frame = {(b, f): False for b in lattice.blocks() for f in (0, 1)}
        for u, v in lattice.inter_edges():
            edge = tuple(sorted((u, v)))
            cu, cv = A.code_qubit(edge[0]), A.code_qubit(edge[1])
            routes = list(place.connection_routes(cu, cv).values())
            if not routes:
                raise ValueError(f"no canonical route for {edge} (code {cu}-{cv}) under {A}")
            r = routes[A.conn.get(edge, 0) % len(routes)]
            # code qubit (b, l) is only reachable at frame l, so this is also the frame requirement
            frames = {r.u[0]: r.frames[0], r.v[0]: r.frames[1]}
            for b, f in frames.items():
                has_frame[(b, f)] = True
            name = f"G{edge[0]}-{edge[1]}"
            units[name] = Unit(name, "gadget", frames=frames, edge=edge, route=r, m=best_m(r.path))
        for b in lattice.blocks():
            blk = place.blocks[b]
            units[f"T{b}"] = Unit(f"T{b}", "toggle", block=b,
                                  pair=(blk.qubits["d1"], blk.qubits["d2"]))      # (s1, s3)
            # deg(v) >= 2 guarantees a host gadget in each frame; otherwise add a local host
            for f in (0, 1):
                if not has_frame[(b, f)]:
                    units[f"L{b}.{f}"] = Unit(f"L{b}.{f}", "local", frames={b: f}, block=b,
                                              pair=(blk.qubits["d2"], blk.qubits["d3"]))  # (s3, s4)
        return units

    def _build_gates(self):
        """Expand the units into gates: parity groups first, then each unit's own gates."""
        code_sites = self.code_sites
        intra = {u[0] for u, v in self.lattice.intra_edges()}
        gates, unit_gates, groups, unit_groups = [], {}, {}, {}

        def add(kind, qubits, owner, angle=None):
            gates.append(Gate(len(gates), kind, tuple(qubits), angle, owner))
            return len(gates) - 1

        # group membership: each gadget has one arm per block side, (far, near) = its first CNOT
        members = {}
        for name, t in sorted(self.units.items()):
            if t.kind != "gadget":
                continue
            p, L = t.route.path, len(t.route.path)
            for b, far, near in ((t.route.u[0], p[0], p[1]), (t.route.v[0], p[L - 1], p[L - 2])):
                key = (b, t.frames[b], far, near)
                members.setdefault(key, []).append(name)
                unit_groups.setdefault(name, []).append(f"P{key}")

        # one host group per (block, frame): the one with most members
        host = {}
        for key, ms in sorted(members.items()):
            bf = key[:2]
            if bf not in host or len(ms) > len(members[host[bf]]):
                host[bf] = key

        # group gates: C, then the host's folds, then U
        for key, ms in sorted(members.items()):
            b, f, far, near = key
            gname = f"P{key}"
            C = add("cx", (far, near), gname)
            folds = []
            if host[(b, f)] == key:
                if (b, f) in code_sites:
                    folds.append(add("rz", (near,), gname, "z"))      # Zbar_(b,f)
                if (b, 1 - f) in code_sites:
                    folds.append(add("rx", (far,), gname, "x"))       # Xbar_(b,1-f)
            U = add("cx", (far, near), gname)
            groups[gname] = dict(key=key, C=C, U=U, folds=folds, members=ms)

        # unit gates, in reference order
        for name, t in sorted(self.units.items()):
            if t.kind == "gadget":
                p, m, L = t.route.path, t.m, len(t.route.path)
                arm_starts = {(p[0], p[1]), (p[L - 1], p[L - 2])}   # owned by the parity groups
                fwd = [g for g in fanout_forward(p, m) if g not in arm_starts]
                seq = [add("cx", g, name) for g in fwd]
                seq.append(add("rz", (p[m],), name, "zz"))
                seq += [add("cx", g, name) for g in reversed(fwd)]
            elif t.kind == "toggle":
                s1, s3 = t.pair
                seq = [add("cx", (s1, s3), name)]
                if t.block in intra:
                    seq.append(add("rz", (s3,), name, "zz"))         # Zbar_0 Zbar_1
                seq += [add("cx", (s3, s1), name), add("cx", (s1, s3), name)]
            else:  # local host for a frame with no gadget
                b, f = t.block, t.frames[t.block]
                s3, s4 = t.pair
                seq = [add("cx", (s3, s4), name)]
                if (b, f) in code_sites:
                    seq.append(add("rz", (s4,), name, "z"))
                if (b, 1 - f) in code_sites:
                    seq.append(add("rx", (s3,), name, "x"))
                seq.append(add("cx", (s3, s4), name))
            unit_gates[name] = seq
            unit_groups.setdefault(name, [])
        return gates, unit_gates, groups, unit_groups

    # -- basic properties -------------------------------------------------------

    @property
    def code_sites(self):
        """Code logical qubits that carry a lattice site."""
        return {self.A.code_qubit(s) for s in self.lattice.nodes()}

    @property
    def block_to_index(self):
        """``block -> index`` into a frames list (the order of ``place.blocks``)."""
        return {b: i for i, b in enumerate(self.place.blocks)}

    @property
    def n_cx(self):
        """Number of CNOTs in the step."""
        return sum(g.kind == "cx" for g in self.gates)

    def _group_gate_ids(self, gname):
        """Gate ids of a parity group in reference order: C, folds, U."""
        g = self.groups[gname]
        return [g["C"], *g["folds"], g["U"]]

    # -- frame structure --------------------------------------------------------

    def toggle_order_graph(self):
        """Digraph with an edge b -> c meaning "block b must toggle before block c".

        A gadget that needs block b at f=1 and block c at f=0 must run after b's toggle and before
        c's. A cycle means A admits no consistent one-toggle ordering.
        """
        g = nx.DiGraph()
        g.add_nodes_from(self.lattice.blocks())
        for t in self.units.values():
            if t.kind == "gadget":
                (b, fb), (c, fc) = t.frames.items()
                if fb == 1 and fc == 0:
                    g.add_edge(b, c)
                elif fb == 0 and fc == 1:
                    g.add_edge(c, b)
        return g

    def is_consistent(self):
        """True if A admits the one-toggle-per-block ordering (toggle-order graph is acyclic)."""
        return nx.is_directed_acyclic_graph(self.toggle_order_graph())

    def frame_precedences(self):
        """Pairs (i, j) of unit names whose order is forced by the one-toggle-per-block structure:
        every f=0 unit of block b precedes toggle T_b, which precedes every f=1 unit of b."""
        pairs = set()
        for name, t in self.units.items():
            for b, f in t.frames.items():
                tog = f"T{b}"
                pairs.add((name, tog) if f == 0 else (tog, name))
        for b in self.lattice.blocks():
            tog = f"T{b}"
            pre = [i for (i, j) in pairs if j == tog]
            post = [j for (i, j) in pairs if i == tog]
            pairs |= {(i, j) for i in pre for j in post}
        return pairs

    # -- the logical product formula ----------------------------------------------

    def trotter_order(self, n_steps):
        """The Trotter order this step implements, as an explicit list of logical terms.

        Forward (even-indexed) steps: Rx on every site carried by code label 1, then every ZZ
        coupling and every Rz field (these commute; the ZZ terms are grouped into layers of
        couplings that share no site), then Rx on every code-label-0 site. Reverse (odd-indexed)
        steps are the mirror image. Which sites carry code label l depends on ``A.relabel``.
        `GateSchedule.verify` proves the encoded circuit equals this sequence up to swaps of
        commuting terms.

        Returns
        -------
        list of list of tuple
            One list per step, terms in application order: ``("x", site)``, ``("z", site)`` or
            ``("zz", u, v)``.
        """
        nodes = list(self.lattice.nodes())
        x_terms = {l: [("x", s) for s in nodes if self.A.code_qubit(s)[1] == l] for l in (0, 1)}
        line = nx.line_graph(nx.Graph(list(self.lattice.edges)))
        color = nx.greedy_color(line, strategy="largest_first")
        edge_index = {frozenset(e): k for k, e in enumerate(self.lattice.edges)}
        zz = sorted(self.lattice.edges,
                    key=lambda e: (color[e] if e in color else color[e[::-1]],
                                   edge_index[frozenset(e)]))
        middle = [("zz", u, v) for u, v in zz] + [("z", s) for s in nodes]
        steps = []
        for step in range(n_steps):
            forward = x_terms[1] + middle + x_terms[0]
            steps.append(forward if step % 2 == 0 else list(reversed(forward)))
        return steps

    # -- gate-level CP-SAT ------------------------------------------------------

    def schedule(self, horizon=None, time_limit=60.0, workers=8, seed=0, verbose=False):
        """Gate-level CP-SAT: minimum-CNOT-depth time slots for every gate of the step.

        The circuit must be a commutation-reordering of a reference circuit that runs the units
        one after another in some order (integer positions ``p``, chosen by CP-SAT), with each
        parity group's opening CNOT C canceled against its closing CNOT U between consecutive
        members. Constraints:

        1. inside a unit or group, non-commuting gates keep their reference order;
        2. a group's C (and folds) precede, and its U follows, every member gate they do not
           commute with;
        3. two conflicting units are ordered by ``p``, and their non-commuting gates follow it;
        4. a unit conflicting with a group lies entirely before or after all its members;
        5. two groups whose gates do not commute are ordered as wholes;
        6. each qubit hosts at most one CNOT per layer.

        Times live on a doubled grid: a CX in layer k at 2k, a rotation between layers at 2k+1.

        Parameters
        ----------
        horizon : int, optional
            Maximum CNOT layers; defaults to the CNOT count (any valid order fits).
        time_limit : float
            Seconds; the best solution so far is returned if the limit is hit.
        workers, seed : int
            CP-SAT parallel search threads and random seed. With ``workers > 1`` the solver may
            return a different (equally deep) schedule on each run; ``workers=1`` is deterministic.
        verbose : bool
            Print CP-SAT's search log.

        Returns
        -------
        GateSchedule or None
            None if A admits no one-toggle ordering or no solution was found.
        """
        if not self.is_consistent():
            return None
        gates = self.gates
        if horizon is None:
            horizon = self.n_cx
        cp_model = _cp_model()
        model = cp_model.CpModel()
        T, var = {}, {}
        for g in gates:
            if g.kind == "cx":
                var[g.id] = model.NewIntVar(0, horizon - 1, f"L{g.id}")
                T[g.id] = 2 * var[g.id]
            else:
                var[g.id] = model.NewIntVar(-1, horizon - 1, f"u{g.id}")
                T[g.id] = 2 * var[g.id] + 1
        depth = model.NewIntVar(1, horizon, "depth")
        for g in gates:
            if g.kind == "cx":
                model.Add(var[g.id] + 1 <= depth)

        def order(i, j, lit=None):
            """Gate i strictly before gate j (only when ``lit`` is true, if given)."""
            c = model.Add(T[i] < T[j])
            if lit is not None:
                c.OnlyEnforceIf(lit)

        # (6) one CNOT per qubit per layer
        per_qubit = {}
        for g in gates:
            if g.kind == "cx":
                for q in g.qubits:
                    per_qubit.setdefault(q, []).append(var[g.id])
        for vs in per_qubit.values():
            if len(vs) > 1:
                model.AddAllDifferent(vs)

        # (1) reference order inside units and groups
        for seq in (list(self.unit_gates.values())
                    + [self._group_gate_ids(gn) for gn in self.groups]):
            for a, i in enumerate(seq):
                for j in seq[a + 1:]:
                    if noncommuting(gates[i], gates[j]):
                        order(i, j)
        for g in self.groups.values():
            order(g["C"], g["U"])   # identical CNOTs commute, but C opens the window

        # (2) group window around member gates
        for g in self.groups.values():
            head = [g["C"], *g["folds"]]
            for mname in g["members"]:
                for x in self.unit_gates[mname]:
                    for h in head:
                        if noncommuting(gates[h], gates[x]):
                            order(h, x)
                    if noncommuting(gates[g["U"]], gates[x]):
                        order(x, g["U"])

        # full gate set of a unit = its own gates + the gates of the groups it belongs to
        full = {u: list(seq) + [i for gn in self.unit_groups[u] for i in self._group_gate_ids(gn)]
                for u, seq in self.unit_gates.items()}

        def conflict(ids_a, ids_b):
            """True if some gate in ids_a does not commute with some gate in ids_b."""
            return any(noncommuting(gates[i], gates[j]) for i in ids_a for j in ids_b
                       if set(gates[i].qubits) & set(gates[j].qubits))

        # position of each unit in the reference sequence (a permutation, so acyclic)
        names = sorted(self.unit_gates)
        pos = {u: model.NewIntVar(0, len(names) - 1, f"p_{u}") for u in names}
        model.AddAllDifferent(list(pos.values()))
        fixed = self.frame_precedences()

        # (3) unit-unit order
        qubits_of = {u: {q for i in full[u] for q in gates[i].qubits} for u in names}
        for a, b in combinations(names, 2):
            if not (qubits_of[a] & qubits_of[b]) or not conflict(full[a], full[b]):
                continue
            if (a, b) in fixed or (b, a) in fixed:   # frame order decides it
                first, second = (a, b) if (a, b) in fixed else (b, a)
                lits = [(first, second, None)]
            else:
                o = model.NewBoolVar(f"o_{a}_{b}")
                lits = [(a, b, o), (b, a, o.Not())]
            for x, y, lit in lits:
                c = model.Add(pos[x] < pos[y])
                if lit is not None:
                    c.OnlyEnforceIf(lit)
                for i in self.unit_gates[x]:
                    for j in self.unit_gates[y]:
                        if noncommuting(gates[i], gates[j]):
                            order(i, j, lit)

        # (4) unit vs group it is not a member of
        for gname, g in self.groups.items():
            gids = self._group_gate_ids(gname)
            gq = {q for i in gids for q in gates[i].qubits}
            for u in names:
                if u in g["members"] or not (qubits_of[u] & gq) or not conflict(full[u], gids):
                    continue
                after = model.NewBoolVar(f"a_{u}_{gname}")
                for mname in g["members"]:
                    model.Add(pos[u] > pos[mname]).OnlyEnforceIf(after)
                    model.Add(pos[u] < pos[mname]).OnlyEnforceIf(after.Not())
                for i in self.unit_gates[u]:
                    for h in gids:
                        if noncommuting(gates[i], gates[h]):
                            order(h, i, after)
                            order(i, h, after.Not())

        # (5) group vs group: (4) keeps member spans disjoint, so one member of each decides
        for a, b in combinations(sorted(self.groups), 2):
            ga, gb = self.groups[a], self.groups[b]
            ia, ib = self._group_gate_ids(a), self._group_gate_ids(b)
            if not conflict(ia, ib):
                continue
            q = model.NewBoolVar(f"q_{a}_{b}")
            model.Add(pos[ga["members"][0]] < pos[gb["members"][0]]).OnlyEnforceIf(q)
            model.Add(pos[ga["members"][0]] > pos[gb["members"][0]]).OnlyEnforceIf(q.Not())
            for i in ia:
                for j in ib:
                    if noncommuting(gates[i], gates[j]):
                        order(i, j, q)
                        order(j, i, q.Not())

        model.Minimize(depth)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = workers
        solver.parameters.random_seed = seed
        solver.parameters.log_search_progress = verbose
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        times = {g.id: solver.Value(T[g.id]) for g in gates}
        return GateSchedule(self, times, solver.Value(depth), solver.StatusName(status),
                            solver.WallTime())


# ---------------------------------------------------------------------------
# A scheduled step: circuits and exactness checks
# ---------------------------------------------------------------------------


def _cx_conj(p, c, t):
    """Conjugate a Pauli ``p = (x, z, sign)`` (bitmasks over qubits, Y = x & z) by CX(c -> t)
    (Aaronson-Gottesman update; CX is self-inverse, so this serves both directions)."""
    x, z, sign = p
    xc, xt, zc, zt = (x >> c) & 1, (x >> t) & 1, (z >> c) & 1, (z >> t) & 1
    if xc & zt & (xt ^ zc ^ 1):
        sign = -sign
    x ^= xc << t
    z ^= zt << c
    return x, z, sign


def _to_pauli(p, qubits):
    """Bitmask Pauli -> qiskit Pauli on the compact register ``qubits`` (physical indices)."""
    x, z, sign = p
    n = len(qubits)
    label = ["I"] * n
    for k, q in enumerate(qubits):
        xb, zb = (x >> q) & 1, (z >> q) & 1
        label[n - 1 - k] = "Y" if xb and zb else "X" if xb else "Z" if zb else "I"
    return Pauli(("-" if sign < 0 else "") + "".join(label))


def _compact(qc):
    """Drop idle wires, measurements and barriers. Returns (circuit, physical -> compact index)."""
    dag = circuit_to_dag(qc)
    idle = set(dag.idle_wires(ignore=["barrier", "measure"]))
    active = [q for q in qc.qubits if q not in idle]
    pos = {q: i for i, q in enumerate(active)}
    out = QuantumCircuit(len(active))
    for ins in qc.data:
        if ins.operation.name in ("barrier", "measure"):
            continue
        out.append(ins.operation, [pos[q] for q in ins.qubits])
    return out, {qc.find_bit(q).index: i for q, i in pos.items()}


@dataclass
class GateSchedule:
    """A time slot for every gate of a TrotterStep (the gate-level CP-SAT result).

    Attributes
    ----------
    step : TrotterStep
        The step that was scheduled.
    time : dict
        ``gate id -> time`` on a doubled grid: a CX in layer k sits at 2k, a rotation between
        layers k and k+1 at 2k+1 (so -1 means before the first layer).
    depth : int
        Number of CNOT layers in the step.
    status : str
        CP-SAT status name (``'OPTIMAL'``, ``'FEASIBLE'``, or ``'broken'`` for `without_gate`).
    wall_time : float
        Solver wall time in seconds.
    """
    step: TrotterStep
    time: dict
    depth: int
    status: str
    wall_time: float

    @property
    def n_cx(self):
        """Number of CNOTs in the scheduled step."""
        return sum(self.step.gates[i].kind == "cx" for i in self.time)

    def sequence(self):
        """Gate ids in execution order (sorted by time)."""
        return sorted(self.time, key=lambda i: (self.time[i], i))

    def without_gate(self, gate_id):
        """A deliberately broken copy with one gate removed (for negative controls)."""
        return GateSchedule(self.step, {i: t for i, t in self.time.items() if i != gate_id},
                            self.depth, "broken", 0.0)

    # -- circuits ---------------------------------------------------------------

    def append_step(self, qc, frames, delta_t, J, gx, gz, reverse=False, barriers=True):
        """Append one Trotter step to ``qc`` (on the full device register).

        Gates are appended in time order; ``reverse=True`` plays the list backwards (odd steps).
        Rotation angles are ``-delta_t * (J | gx | gz)``: each term c*P of H gives
        exp(-i * delta_t/2 * c * P), the paper's convention (Methods C, U_e(dt) = e^{-i dt/2 ZZ}).

        Parameters
        ----------
        frames : list of int
            Per-block frame, indexed by ``step.block_to_index``. Must be all 0 (forward) or all 1
            (reverse). Every block toggles once, so all frames flip; updated in place.

        Returns
        -------
        (qc, frames)
        """
        b2i = self.step.block_to_index
        start = 1 if reverse else 0
        for b in self.step.lattice.blocks():
            if frames[b2i[b]] != start:
                raise ValueError(f"block {b} must start at frame {start}")
        angle = {"zz": -delta_t * J, "x": -delta_t * gx, "z": -delta_t * gz}
        seq = self.sequence()
        for i in (reversed(seq) if reverse else seq):
            g = self.step.gates[i]
            if g.kind == "cx":
                qc.cx(*g.qubits)
            elif g.kind == "rz":
                qc.rz(angle[g.angle], g.qubits[0])
            else:
                qc.rx(angle[g.angle], g.qubits[0])
        for b in self.step.lattice.blocks():
            frames[b2i[b]] ^= 1
        if barriers:
            qc.barrier()
        return qc, frames

    # def circuit(self, bubble_sites, n_steps, delta_t, J, gx, gz, rounds=0, measure=False,
    #             reset_ancillas=True, barriers=False):
    #     """Encoded circuit: state prep -> bubble -> n_steps steps (alternating forward / reverse),
    #     with ``rounds`` syndrome rounds (one after every ``n_steps // rounds`` steps).

    #     Parameters
    #     ----------
    #     bubble_sites : list of lattice sites
    #         Sites starting in logical |0>; all others start in |1>.
    #     rounds : int
    #         Number of syndrome-extraction rounds (0 = none).
    #     measure : bool
    #         Append the final data measurement (needed for sampling, not for `expectations`).
    #     reset_ancillas : bool
    #         Reset the measured ancillas after each round (see `syndrome_extraction`).

    #     Returns
    #     -------
    #     (qc, frames)
    #     """
    #     step, place = self.step, self.step.place
    #     qc, frames = init_circuit(len(place.blocks), rounds, place.topology.n_qubits)
    #     qc = prep_logical_zero_blocks(qc, place, barriers=barriers)
    #     qc = create_bubble(qc, step, bubble_sites, frames, step.block_to_index, barriers=barriers)
    #     r = 0
    #     for k in range(n_steps):
    #         qc, frames = self.append_step(qc, frames, delta_t, J, gx, gz, reverse=(k % 2 == 1),
    #                                       barriers=barriers)
    #         if rounds and (k + 1) % (n_steps // rounds) == 0 and r < rounds:
    #             qc = syndrome_extraction(qc, place, r, barriers=barriers,
    #                                      reset_ancillas=reset_ancillas)
    #             r += 1
    #     if measure:
    #         qc = final_measurements(qc, place, frames)
    #     return qc, frames

    def describe(self):
        """Print the schedule: one line per CNOT layer, and a "1q" line for the rotations in the
        slot between two layers (rotations show their coupling type, e.g. ``rz[zz](98)``)."""
        by_time = {}
        for i, t in self.time.items():
            by_time.setdefault(t, []).append(self.step.gates[i])
        for t in sorted(by_time):
            items = [f"cx{g.qubits}" if g.kind == "cx" else f"{g.kind}[{g.angle}]({g.qubits[0]})"
                     for g in by_time[t]]
            label = f"layer {t // 2:2d}" if t % 2 == 0 else "   1q   "
            print(f"{label}: " + ", ".join(items))

    # -- exactness checks -------------------------------------------------------

    def expectations(self, qc, frames):
        """Noiseless logical expectation values of an encoded, measurement-free circuit
        (statevector on its active qubits; small lattices only).

        Returns
        -------
        (values, stabilizers)
            ``values``: ``("Z", site)``, ``("X", site)``, ``("ZZ", u, v)`` -> expectation, as from
            `UnencodedBaseline.exact_expectations`. ``stabilizers``: block -> (<S_X>, <S_Z>).
        """
        small, pos = _compact(qc)
        state = Statevector(small)
        n = small.num_qubits
        step = self.step
        place, A = step.place, step.A
        b2i = step.block_to_index

        def ev(pauli, phys):
            label = ["I"] * n
            for q in phys:
                label[n - 1 - pos[q]] = pauli
            return float(np.real(state.expectation_value(Pauli("".join(label)))))

        def support(kind, site):
            b, l = A.code_qubit(site)
            roles = (Z_SUPPORTS if kind == "Z" else X_SUPPORTS)[l][0]
            return [place.blocks[b].physical(r, frames[b2i[b]]) for r in roles]

        out = {}
        for s in step.lattice.nodes():
            out[("Z", s)] = ev("Z", support("Z", s))
            out[("X", s)] = ev("X", support("X", s))
        for u, v in step.lattice.edges:
            # within a block Zbar_0 Zbar_1 = Z_d1 Z_d2: the shared d0 cancels
            out[("ZZ", u, v)] = ev("Z", set(support("Z", u)) ^ set(support("Z", v)))
        stab = {b: (ev("X", blk.stabilizer_support()), ev("Z", blk.stabilizer_support()))
                for b, blk in place.blocks.items()}
        return out, stab

    def verify(self, n_steps, delta_t=1.0, J=0.1, gx=0.2, gz=0.3):
        """Exact, angle-independent check of ``n_steps`` steps by Pauli propagation.

        Every rotation is pushed back through the CNOTs before it, which rewrites the circuit as
        Pauli rotations on the initial code space followed by one CNOT network. Then:

        1. each rotation must be exactly one intended logical term (Zbar, Xbar or Zbar Zbar on the
           right sites, sign +1) that does not leave the code space;
        2. the CNOT network must act as the logical identity at the final frames and return every
           ancilla and bridge qubit to |0>;
        3. the term sequence must equal `TrotterStep.trotter_order` up to swaps of commuting terms.

        The couplings only label the rotations, so they must be distinct. With an even
        ``n_steps``, a window-closing CNOT cancels against its mirror in the next step, so deleting
        it leaves a correct circuit; use an odd ``n_steps`` for broken-schedule controls.

        Returns
        -------
        dict
            ``{"ok", "rotations_ok", "network_ok", "order_ok", "n_rotations", "problems"}``.
        """
        step = self.step
        place, A, lattice = step.place, step.A, step.lattice
        b2i = step.block_to_index
        qc = QuantumCircuit(place.topology.n_qubits)
        frames = [0] * len(place.blocks)
        for s in range(n_steps):
            qc, frames = self.append_step(qc, frames, delta_t, J, gx, gz, reverse=(s % 2 == 1),
                                          barriers=False)
        # must match the angles in append_step, so each rotation can be told apart by its angle
        kind_of = {round(-delta_t * J, 12): "zz", round(-delta_t * gx, 12): "x",
                   round(-delta_t * gz, 12): "z"}
        assert len(kind_of) == 3, "use three distinct couplings"
        ops = [(i.operation.name, [qc.find_bit(q).index for q in i.qubits],
                i.operation.params[0] if i.operation.params else None) for i in qc.data]
        qubits = sorted({q for _, qs, _ in ops for q in qs}
                        | {q for b in place.blocks.values() for q in b.physical_qubits()})
        data = {q for b in place.blocks.values() for q in b.stabilizer_support()}
        spare = [q for q in qubits if q not in data]            # ancillas + bridges, start in |0>

        # 1. push every rotation back to the start (process the circuit backwards)
        held = []
        for k in range(len(ops) - 1, -1, -1):
            name, qs, theta = ops[k]
            if name == "cx":
                held = [(_cx_conj(p, *qs), kd, j) for p, kd, j in held]
            else:
                q = qs[0]
                p = (1 << q, 0, 1) if name == "rx" else (0, 1 << q, 1)
                held.append((p, kind_of[round(theta, 12)], k))
        held.reverse()

        def rep(kind, site, frame_of):
            b, l = A.code_qubit(site)
            roles = (Z_SUPPORTS if kind == "Z" else X_SUPPORTS)[l][0]
            bits = 0
            for r in roles:
                bits |= 1 << place.blocks[b].physical(r, frame_of(b))
            return _to_pauli((bits, 0, 1) if kind == "X" else (0, bits, 1), qubits)

        nodes = list(lattice.nodes())
        Zr = {s: rep("Z", s, lambda b: 0) for s in nodes}
        Xr = {s: rep("X", s, lambda b: 0) for s in nodes}
        stab_gens = []
        for b in place.blocks.values():
            bits = sum(1 << q for q in b.stabilizer_support())
            stab_gens += [_to_pauli((bits, 0, 1), qubits), _to_pauli((0, bits, 1), qubits)]
        spare_z = [_to_pauli((0, 1 << q, 1), qubits) for q in spare]
        problems = []

        def logical_of(P):
            """P = sign * (logical Pauli) * (stabilizers) * (Z on spare), or None."""
            lab = {}
            for s in nodes:
                ax, az = not P.commutes(Zr[s]), not P.commutes(Xr[s])
                if ax or az:
                    lab[s] = "Y" if ax and az else "X" if ax else "Z"
            L = _to_pauli((0, 0, 1), qubits)
            for s, c in lab.items():
                L = (L.dot(Xr[s]) if c == "X" else L.dot(Zr[s]) if c == "Z"
                     else L.dot(Xr[s].dot(Zr[s])) * 1j)
            R = P.dot(L.adjoint())
            if any(not R.commutes(g) for g in stab_gens) or any(
                    R.x[k] and qubits[k] in spare for k in range(len(qubits))):
                return None
            return lab, R

        # a residual in <S_X, S_Z, Z_spare> has a definite sign on the initial code state
        prep = QuantumCircuit(len(qubits))
        pos = {q: k for k, q in enumerate(qubits)}
        for b in place.blocks.values():
            d0, d1, d2, d3 = (pos[b.physical(r, 0)] for r in ("d0", "d1", "d2", "d3"))
            prep.h(d0)
            prep.cx(d0, d1)
            prep.cx(d1, d2)
            prep.cx(d2, d3)
        code0 = StabilizerState(prep)

        def residual_sign(R):
            v = code0.expectation_value(R)
            return int(round(np.real(v))) if abs(abs(v) - 1) < 1e-9 else None

        expect = {"z": [{s: "Z"} for s in nodes], "x": [{s: "X"} for s in nodes],
                  "zz": [{u: "Z", v: "Z"} for u, v in lattice.edges]}
        seq = []
        for P, kd, k in held:
            dec = logical_of(_to_pauli(P, qubits))
            sgn = residual_sign(dec[1]) if dec else None
            lab = dec[0] if dec else None
            if lab is None or sgn != 1 or lab not in expect[kd]:
                problems.append(f"rotation {k} ({ops[k][0]} on {ops[k][1]}, {kd}): acts as "
                                f"{lab} with sign {sgn}")
            seq.append((kd, tuple(sorted(lab.items())) if lab else None))

        # 2. the CNOT network must be the logical identity and restore the spare qubits
        cx_ops = [qs for name, qs, _ in ops if name == "cx"]

        def forward(P):
            for c, t in cx_ops:
                P = _cx_conj(P, c, t)
            return P

        def bits_of(Pq):
            x = sum(1 << qubits[k] for k in range(len(qubits)) if Pq.x[k])
            z = sum(1 << qubits[k] for k in range(len(qubits)) if Pq.z[k])
            return x, z, 1

        final = lambda b: frames[b2i[b]]
        Zf = {s: rep("Z", s, final) for s in nodes}
        Xf = {s: rep("X", s, final) for s in nodes}
        network_ok = True
        for s in nodes:
            for P0, Pf in ((Zr[s], Zf[s]), (Xr[s], Xf[s])):
                out = _to_pauli(forward(bits_of(P0)), qubits)
                if residual_sign(out.dot(Pf.adjoint())) != 1:
                    network_ok = False
                    problems.append(f"CNOT network does not map a logical of {s} correctly")
        for P0 in spare_z + stab_gens:
            if residual_sign(_to_pauli(forward(bits_of(P0)), qubits)) != 1:
                network_ok = False
                problems.append("CNOT network does not restore a stabilizer / spare qubit")

        # 3. compare the term sequence with the Trotter order, up to commuting reorderings
        def as_label(term):
            if term[0] == "zz":
                return ("zz", tuple(sorted({term[1]: "Z", term[2]: "Z"}.items())))
            return (term[0], ((term[1], term[0].upper()),))

        ref = [as_label(t) for st in step.trotter_order(n_steps) for t in st]
        order_ok = sorted(map(str, seq)) == sorted(map(str, ref))

        def occurrences(lst):
            count, out = {}, []
            for item in lst:
                count[item] = count.get(item, 0) + 1
                out.append((item, count[item]))
            return out

        def anticommute(a, b):
            da, db = dict(a[1]), dict(b[1])
            return sum(1 for s in set(da) & set(db) if da[s] != db[s]) % 2 == 1

        if order_ok:
            rpos = {o: i for i, o in enumerate(occurrences(ref))}
            enc = occurrences(seq)
            for i in range(len(enc)):
                for j in range(i + 1, len(enc)):
                    if anticommute(enc[i][0], enc[j][0]) and rpos[enc[i]] > rpos[enc[j]]:
                        order_ok = False
                        problems.append(f"order of {enc[i][0]} and {enc[j][0]} differs")
                        break
                if not order_ok:
                    break
        else:
            problems.append("the multiset of logical terms differs from the Trotter order")

        rotations_ok = not any(p.startswith("rotation") for p in problems)
        return dict(ok=rotations_ok and network_ok and order_ok, rotations_ok=rotations_ok,
                    network_ok=network_ok, order_ok=order_ok, n_rotations=len(seq),
                    problems=problems[:10])


# ---------------------------------------------------------------------------
# Choosing A
# ---------------------------------------------------------------------------


class ScheduleOptimizer:
    """Coordinate descent over A = {o, l, w}, scoring each A by its gate-level CP-SAT depth.

    Not every relabeling l admits the one-toggle ordering. `relabel` finds feasible labelings with
    a small CP-SAT model, and is used both for the starting point and to *repair* l after each
    l-move (on the full Fig. 2a lattice no bare single flip of l is feasible). Moves on o_b and
    w_e change only that coordinate. A move is kept if it lowers (depth, CNOT count).

    Parameters
    ----------
    lattice : lay.LogicalLattice
    block_qubits : dict
        ``block -> its six physical qubits``.
    topology : optional
        Device connectivity; None means the built-in ibm_boston map.
    time_limit : float
        Seconds per gate-level CP-SAT solve.
    workers, seed : int
        Passed to `TrotterStep.schedule`.

    Attributes
    ----------
    cache : dict
        ``A.key() -> (score, step, gsched)`` for every A evaluated so far.
    """

    def __init__(self, lattice, block_qubits, topology=None, time_limit=10.0, workers=8, seed=0):
        self.lattice = lattice
        self.block_qubits = block_qubits
        self.topology = topology
        self.time_limit = time_limit
        self.workers = workers
        self.seed = seed
        self.cache = {}

    def relabel(self, fixed=None, near=None, seed=None, time_limit=10.0):
        """A relabeling l admitting the one-toggle ordering, from a small CP-SAT model.

        Booleans l_b, an integer toggle position tau_b per block, and for each inter-block edge
        "if the code labels differ, the block whose qubit sits at f=1 toggles first".

        Parameters
        ----------
        fixed : dict, optional
            ``{block: l_b}`` values to impose.
        near : dict, optional
            A current labeling; if given, minimize the number of l_b that differ from it.
        seed : int, optional
            Otherwise, None minimizes the number of mixed-frame edges; an int uses random
            positive weights instead (different feasible starts for restarts).

        Returns
        -------
        dict or None
            ``{block: l_b}``, or None if infeasible.
        """
        blocks = self.lattice.blocks()
        rng = random.Random(seed)
        cp_model = _cp_model()
        model = cp_model.CpModel()
        L = {b: model.NewBoolVar(f"l_{b}") for b in blocks}
        tau = {b: model.NewIntVar(0, len(blocks), f"tau_{b}") for b in blocks}
        mixed = []
        for u, v in self.lattice.inter_edges():
            (b, lu), (c, lv) = u, v
            xb, xc = model.NewBoolVar(""), model.NewBoolVar("")      # code labels
            model.Add(xb != L[b]) if lu else model.Add(xb == L[b])
            model.Add(xc != L[c]) if lv else model.Add(xc == L[c])
            model.Add(tau[b] < tau[c]).OnlyEnforceIf([xb, xc.Not()])
            model.Add(tau[c] < tau[b]).OnlyEnforceIf([xb.Not(), xc])
            mix = model.NewBoolVar("")
            model.AddBoolXOr([xb, xc, mix.Not()])                  # mix = xb XOR xc
            mixed.append(mix)
        for b, val in (fixed or {}).items():
            model.Add(L[b] == val)
        if near is not None:
            model.Minimize(sum(L[b] if near[b] == 0 else 1 - L[b] for b in blocks))
        elif seed is None:
            model.Minimize(sum(mixed))
        else:
            model.Minimize(sum(rng.randint(1, 10) * x for x in mixed))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = 1          # deterministic: same labeling every run
        solver.parameters.random_seed = 0 if seed is None else seed
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        return {b: solver.Value(L[b]) for b in blocks}

    def initial(self, seed=None):
        """Starting A: default orientations, shortest routes (listed explicitly in ``conn`` for
        every edge with a choice), and l from `relabel` (raises if none)."""
        base = ParamA.default(self.lattice)
        A0 = ParamA.default(self.lattice,
                            placement_for(base, self.block_qubits, self.topology))
        relabel = self.relabel(seed=seed)
        if relabel is None:
            raise RuntimeError("no relabeling admits a one-toggle-per-block ordering")
        A0.relabel = relabel
        return A0

    def evaluate(self, A):
        """Score A by its gate-level depth.

        Returns
        -------
        (score, step, gsched)
            ``score = (depth, n_cx)``, or ``(inf, inf)`` if A is infeasible.
        """
        key = A.key()
        if key in self.cache:
            return self.cache[key]
        step, gsched = None, None
        try:
            step = TrotterStep(A, self.lattice, self.block_qubits, self.topology)
        except ValueError:   # some coupling has no route under A
            pass
        if step is not None and step.is_consistent():
            gsched = step.schedule(time_limit=self.time_limit, workers=self.workers,
                                   seed=self.seed)
        score = (gsched.depth, gsched.n_cx) if gsched else (math.inf, math.inf)
        self.cache[key] = (score, step, gsched)
        return self.cache[key]

    def coordinates(self, A):
        """``(label, attribute, key, values)`` for every o_b, every l_b, and every w_e that has
        more than one connection."""
        coords = [(f"o[{b}]", "orient", b, (0, 1)) for b in sorted(A.orient)]
        coords += [(f"l[{b}]", "relabel", b, (0, 1)) for b in sorted(A.relabel)]
        place = placement_for(A, self.block_qubits, self.topology)
        for u, v in self.lattice.inter_edges():
            edge = tuple(sorted((u, v)))
            n = len(place.connection_routes(A.code_qubit(edge[0]), A.code_qubit(edge[1])))
            if n > 1:
                coords.append((f"w[{edge}]", "conn", edge, tuple(range(n))))
        return coords

    def descend(self, A0=None, max_sweeps=5, verbose=True):
        """One coordinate-descent run.

        Returns
        -------
        (A, step, gsched, history)
            The best A, its TrotterStep and GateSchedule, and the best score after each
            coordinate.
        """
        A = (A0 or self.initial()).copy()
        best_score, best_step, best_sched = self.evaluate(A)
        if best_sched is None:
            raise RuntimeError(f"initial {A} is infeasible")
        history = [best_score]
        if verbose:
            print(f"start: depth {best_score[0]}, {best_score[1]} CNOTs  {A}")
        for sweep in range(max_sweeps):
            improved = False
            for name, attr, key, values in self.coordinates(A):
                current = getattr(A, attr).get(key, 0)
                for val in values:
                    if val == current:
                        continue
                    trial = A.copy()
                    if attr == "relabel":
                        repaired = self.relabel(fixed={key: val}, near=A.relabel)
                        if repaired is None:
                            continue
                        trial.relabel = repaired
                    else:
                        getattr(trial, attr)[key] = val
                    score, step, gsched = self.evaluate(trial)
                    if score < best_score:
                        extra = ""
                        if attr == "relabel":
                            others = sorted(b for b in A.relabel
                                            if b != key and trial.relabel[b] != A.relabel[b])
                            extra = f" (repair also flipped l of {others})" if others else ""
                        A, best_score, best_step, best_sched = trial, score, step, gsched
                        current = val
                        improved = True
                        if verbose:
                            print(f"  sweep {sweep}: {name} -> {val}{extra}: depth {score[0]}, "
                                  f"{score[1]} CNOTs")
                history.append(best_score)
            if not improved:
                break
        if verbose:
            print(f"final: depth {best_score[0]}, {best_score[1]} CNOTs  {A} "
                  f"({len(self.cache)} evaluations)")
        return A, best_step, best_sched, history

    def optimize(self, n_starts=1, max_sweeps=5, verbose=True):
        """Best A over ``n_starts`` descents: start 0 uses the fewest-mixed-edges labeling,
        the others random feasible labelings.

        Returns
        -------
        (A, step, gsched)
        """
        best = None
        for k in range(n_starts):
            if verbose and n_starts > 1:
                print(f"--- start {k}")
            A, step, gsched, hist = self.descend(self.initial(seed=None if k == 0 else k),
                                                 max_sweeps, verbose)
            if best is None or hist[-1] < best[0]:
                best = (hist[-1], A, step, gsched)
        return best[1:]
