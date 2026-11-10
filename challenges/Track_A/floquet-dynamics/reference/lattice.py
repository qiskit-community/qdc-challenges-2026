"""Heavy-hex lattice generation and balanced 3-edge-colouring.

The paper's Floquet circuit lives on a heavy-hex lattice whose bonds are split
into three edge-colour classes (``E1, E2, E3``); the ``ZZ`` rotations within one
class commute and run in parallel (Eq. 1). This module builds heavy-hex patches
of an *arbitrary* qubit count so participants can validate finite-size scaling,
and returns a proper 3-edge-colouring of the bonds.

Construction
------------
Heavy-hex is the *subdivision* of a honeycomb (hexagonal) lattice: every bond of
a degree-3 honeycomb gets a mid-bond "flag" qubit (degree 2). We build the
honeycomb in its brick-wall form, then subdivide. The subdivision of any graph
is bipartite, so by Konig's theorem the bonds are edge-colourable with exactly
``max_degree = 3`` colours -- computed here with
``rustworkx.graph_bipartite_edge_color``.

Why full patches (not arbitrary-N carving)
-------------------------------------------
A subdivided-honeycomb hexagon is a **12-qubit loop**, so any attempt to carve a
connected sub-patch of <= ~30 qubits severs every plaquette and leaves a *tree* --
which has no frustration and none of the loop structure that makes the physics
interesting. Heavy-hex therefore comes in **discrete sizes** (one per ``rows x
cols`` brick-wall patch). The size knob :func:`heavy_hex` snaps a requested count
to the nearest such patch; :func:`heavy_hex_ladder` exposes the two-row family
(18, 25, 32, ... qubits) whose thermodynamic limit the paper's scaling targets.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import rustworkx as rx

Edge = tuple[int, int]


@dataclass(frozen=True)
class HeavyHexLattice:
    """A heavy-hex qubit patch with a balanced 3-edge-colouring.

    Attributes
    ----------
    n_qubits : int
        Number of qubits (nodes), labelled ``0 .. n_qubits-1``.
    edges : tuple[Edge, ...]
        Undirected bonds ``(i, j)`` with ``i < j``. Each carries a ``ZZ`` rotation.
    edge_colors : tuple[int, ...]
        Colour in ``{0, 1, 2}`` for each edge, aligned with ``edges``.
    positions : dict[int, tuple[float, float]]
        2-D layout coordinates for plotting.
    """

    n_qubits: int
    edges: tuple[Edge, ...]
    edge_colors: tuple[int, ...]
    positions: dict[int, tuple[float, float]] = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_colors(self) -> int:
        return len(set(self.edge_colors))

    def color_groups(self) -> list[list[Edge]]:
        """Edges partitioned by colour: ``[E1, E2, E3]``.

        Every group is a set of disjoint bonds, so all its ``ZZ`` rotations can be
        applied in a single parallel layer.
        """
        groups: dict[int, list[Edge]] = {}
        for e, c in zip(self.edges, self.edge_colors):
            groups.setdefault(c, []).append(e)
        return [groups[c] for c in sorted(groups)]

    def degrees(self) -> list[int]:
        deg = [0] * self.n_qubits
        for i, j in self.edges:
            deg[i] += 1
            deg[j] += 1
        return deg

    def validate(self) -> None:
        """Assert the patch is a legal heavy-hex colouring; raise otherwise."""
        assert all(0 <= i < self.n_qubits and 0 <= j < self.n_qubits for i, j in self.edges)
        assert max(self.degrees()) <= 3, "heavy-hex is degree <= 3"
        assert self.n_colors <= 3, f"expected <= 3 colours, got {self.n_colors}"
        # proper colouring: no qubit touches two same-colour bonds
        incident: dict[int, list[int]] = {q: [] for q in range(self.n_qubits)}
        for (i, j), c in zip(self.edges, self.edge_colors):
            incident[i].append(c)
            incident[j].append(c)
        for q, cols in incident.items():
            assert len(cols) == len(set(cols)), f"qubit {q} has a colour clash"
        # connected
        g = _to_pygraph(self.n_qubits, self.edges)
        assert rx.is_connected(g), "patch is not connected"


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #

def heavy_hex_patch(rows: int, cols: int) -> HeavyHexLattice:
    """A full heavy-hex patch from a ``rows x cols`` brick-wall honeycomb.

    ``rows >= 2`` gives genuine degree-3 heavy-hex structure with hexagonal
    plaquettes (``rows == 1`` is a chain). Qubit count grows in coarse steps; use
    :func:`heavy_hex` to snap a target size to the nearest patch.
    """
    n, edges, pos = _build_subdivided_honeycomb(rows, cols)
    return _colored(n, edges, pos)


def heavy_hex_ladder(cols: int) -> HeavyHexLattice:
    """A heavy-hex *ladder* (two rows of hexagons) -- the geometry whose
    thermodynamic limit the paper's scaling analysis targets.

    Sizes: ``cols = 2, 3, 4, ...`` -> ``18, 25, 32, ...`` qubits, each column
    adding one hexagonal plaquette. This is the recommended finite-size-scaling
    family: fix the two-row cross-section and grow the length.
    """
    return heavy_hex_patch(2, cols)


def heavy_hex_paper(n_qubits: int) -> HeavyHexLattice:
    """Older provisional 21 / 28 / 35 qubit family by automatic coloring.

    The paper *does* draw the exact colored small patches in Fig. E1. Use
    :func:`paper_colored_lattice` for a paper-circuit reproduction. This older
    constructor reproduces the counts and a similar structure from first
    principles: take the width-two heavy-hex ladder and keep its **2-core**
    (iteratively drop degree-1 flag qubits, which are the dangling half-bonds left
    at the open ends of a ladder carved from a device). The 2-core of the
    ``cols = 3, 4, 5`` ladders lands on exactly ``21, 28, 35`` qubits with ``2, 3,
    4`` closed heavy-hex loops respectively -- a clean monotone +7 family, all of
    max degree 3, all width two.

    .. warning::
       **Provisional.** Its automatic coloring changes the driven circuit and
       is incompatible with the versioned Fig. E1 reference, even at 21 qubits.

    Parameters
    ----------
    n_qubits : int
        One of ``21``, ``28``, ``35``.
    """
    cols = {21: 3, 28: 4, 35: 5}.get(n_qubits)
    if cols is None:
        raise ValueError(
            f"heavy_hex_paper supports the paper's sizes 21/28/35, got {n_qubits}. "
            "Use heavy_hex_ladder for the clean 18/25/32 family or heavy_hex(n) to "
            "snap an arbitrary count."
        )
    n, edges, pos = _build_subdivided_honeycomb(2, cols)
    alive, adj = _two_core(n, edges)
    order = sorted(alive)
    remap = {old: i for i, old in enumerate(order)}
    core_edges = sorted({_ordered(remap[v], remap[w]) for v in alive for w in adj[v]})
    core_pos = {remap[v]: pos[v] for v in alive}
    return _colored(len(alive), core_edges, core_pos)


# Fig. E1 of arXiv:2607.24937v1, with its *ordered* colour layers.  The
# automatically coloured heavy_hex_paper(21) has the same uncoloured graph up
# to relabelling, but its [5, 7, 10] layer order defines a different drive.
PAPER_GROUPS_21: tuple[tuple[Edge, ...], ...] = (
    ((3, 4), (9, 12), (10, 16), (11, 20), (13, 14), (17, 18)),
    ((0, 1), (2, 3), (4, 5), (6, 7), (8, 11), (12, 13),
     (14, 15), (16, 17), (18, 19)),
    ((1, 2), (5, 6), (7, 8), (0, 9), (4, 10), (15, 16), (19, 20)),
)

# The remaining small patches are transcribed from the same Fig. E1.  The
# 21-qubit drawing is the induced 0..20 portion of the 28-qubit drawing.
PAPER_GROUPS_28: tuple[tuple[Edge, ...], ...] = (
    PAPER_GROUPS_21[0] + ((21, 23), (24, 25), (22, 27)),
    PAPER_GROUPS_21[1] + ((23, 24), (25, 26)),
    PAPER_GROUPS_21[2] + ((14, 21), (18, 22), (26, 27)),
)

PAPER_GROUPS_35: tuple[tuple[Edge, ...], ...] = (
    ((0, 1), (3, 4), (5, 9), (6, 13), (10, 11), (16, 19),
     (17, 23), (18, 27), (20, 21), (24, 25), (28, 30),
     (29, 34), (31, 32)),
    ((2, 3), (7, 8), (9, 10), (11, 12), (13, 14), (15, 18),
     (19, 20), (21, 22), (23, 24), (25, 26), (30, 31),
     (32, 33)),
    ((1, 2), (0, 5), (4, 6), (8, 9), (12, 13), (14, 15),
     (7, 16), (11, 17), (22, 23), (26, 27), (21, 28),
     (25, 29), (33, 34)),
)

# Fig. A1 prints the ibm_boston physical qubit labels.  Keep those labels in
# the transcription, then relabel them consecutively for logical circuits.
PAPER_PHYSICAL_LABELS_51: tuple[int, ...] = (
    29, 30, 31, 32, 33, 38, 39, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 57, 58, 59, 67, 68, 69, 70, 71, 72, 73, 74, 75,
    78, 79, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99,
    107, 108, 109, 110, 111, 112, 113, 114, 115,
)
_PAPER_GROUPS_51_PHYSICAL: tuple[tuple[Edge, ...], ...] = (
    ((29, 30), (32, 33), (38, 49), (39, 53), (50, 51),
     (57, 67), (58, 71), (59, 75), (68, 69), (72, 73),
     (78, 89), (79, 93), (90, 91), (94, 95), (97, 107),
     (98, 111), (99, 115), (108, 109), (112, 113)),
    ((31, 32), (47, 48), (49, 50), (51, 52), (53, 54),
     (55, 59), (67, 68), (69, 70), (71, 72), (73, 74),
     (87, 88), (89, 90), (91, 92), (93, 94), (95, 99),
     (107, 108), (109, 110), (111, 112), (113, 114)),
    ((30, 31), (29, 38), (33, 39), (48, 49), (52, 53),
     (54, 55), (47, 57), (51, 58), (70, 71), (74, 75),
     (69, 78), (73, 79), (88, 89), (92, 93), (87, 97),
     (91, 98), (110, 111), (114, 115)),
)
_PAPER_51_TO_LOGICAL = {physical: logical for logical, physical
                        in enumerate(PAPER_PHYSICAL_LABELS_51)}
PAPER_GROUPS_51: tuple[tuple[Edge, ...], ...] = tuple(
    tuple((_PAPER_51_TO_LOGICAL[i], _PAPER_51_TO_LOGICAL[j])
          for i, j in group) for group in _PAPER_GROUPS_51_PHYSICAL
)

# Fig. A1 (lower panel): the 51-qubit patch plus two further loop rows.  The
# edge colours were read from the figure's vector strokes; they differ from
# the 51-qubit colouring on the shared qubits.
PAPER_PHYSICAL_LABELS_74: tuple[int, ...] = PAPER_PHYSICAL_LABELS_51 + (
    118, 119, 127, 128, 129, 130, 131, 132, 133, 134, 135,
    137, 138, 139, 147, 148, 149, 150, 151, 152, 153, 154, 155,
)
_PAPER_GROUPS_74_PHYSICAL: tuple[tuple[Edge, ...], ...] = (
    ((29, 30), (31, 32), (38, 49), (39, 53), (47, 48), (51, 58),
     (54, 55), (68, 69), (70, 71), (72, 73), (74, 75), (78, 89),
     (87, 88), (90, 91), (92, 93), (97, 107), (98, 111), (109, 118),
     (112, 113), (114, 115), (119, 133), (129, 130), (131, 138),
     (134, 135), (137, 147), (150, 151), (152, 153), (154, 155)),
    ((29, 38), (32, 33), (47, 57), (48, 49), (50, 51), (52, 53),
     (58, 71), (59, 75), (67, 68), (69, 78), (73, 74), (79, 93),
     (89, 90), (91, 98), (94, 95), (99, 115), (108, 109), (110, 111),
     (113, 114), (118, 129), (127, 128), (130, 131), (132, 133),
     (135, 139), (138, 151), (147, 148), (149, 150)),
    ((30, 31), (33, 39), (49, 50), (51, 52), (53, 54), (55, 59),
     (57, 67), (69, 70), (71, 72), (73, 79), (87, 97), (88, 89),
     (91, 92), (93, 94), (95, 99), (107, 108), (109, 110), (111, 112),
     (113, 119), (127, 137), (128, 129), (131, 132), (133, 134),
     (139, 155), (148, 149), (151, 152), (153, 154)),
)
_PAPER_74_TO_LOGICAL = {physical: logical for logical, physical
                        in enumerate(PAPER_PHYSICAL_LABELS_74)}
PAPER_GROUPS_74: tuple[tuple[Edge, ...], ...] = tuple(
    tuple((_PAPER_74_TO_LOGICAL[i], _PAPER_74_TO_LOGICAL[j])
          for i, j in group) for group in _PAPER_GROUPS_74_PHYSICAL
)

# Fig. A1 drawing rows: (physical labels, x offset, y).
_PAPER_A1_ROWS = (
    ((29, 30, 31, 32, 33), 2., 0.),
    ((38, 39), 2., -1.),
    ((47, 48, 49, 50, 51, 52, 53, 54, 55), 0., -2.),
    ((57, 58, 59), 0., -3.),
    ((67, 68, 69, 70, 71, 72, 73, 74, 75), 0., -4.),
    ((78, 79), 2., -5.),
    ((87, 88, 89, 90, 91, 92, 93, 94, 95), 0., -6.),
    ((97, 98, 99), 0., -7.),
    ((107, 108, 109, 110, 111, 112, 113, 114, 115), 0., -8.),
    ((118, 119), 2., -9.),
    ((127, 128, 129, 130, 131, 132, 133, 134, 135), 0., -10.),
    ((137, 138, 139), 0., -11.),
    ((147, 148, 149, 150, 151, 152, 153, 154, 155), 0., -12.),
)


def _paper_a1_positions(physical_labels: tuple[int, ...]) -> dict[int, tuple[float, float]]:
    """Logical-site plotting coordinates following the Fig. A1 drawing."""
    physical_positions = {}
    for row, x0, y in _PAPER_A1_ROWS:
        step = 4. if len(row) in (2, 3) else 1.
        for i, physical in enumerate(row):
            physical_positions[physical] = (x0 + i * step, y)
    return {logical: physical_positions[physical]
            for logical, physical in enumerate(physical_labels)}

# An explicit graph isomorphism from the provisional helper's vertex labels
# to Fig. E1 labels.  It is used only to reuse plotting coordinates; the
# authoritative edges and colours above define the circuit.
_OLD_TO_PAPER_21 = (
    0, 12, 14, 2, 4, 16, 18, 6, 8, 20, 9,
    13, 3, 10, 17, 7, 11, 1, 15, 5, 19,
)


def paper_colored_21() -> HeavyHexLattice:
    """The 21-site Fig. E1 circuit, preserving the paper's layer order.

    Unlike :func:`heavy_hex_paper`, this constructor has a transcription of
    the actual ordered bonds.  It does not assert that the provisional 28/35
    colourings are paper exact.
    """
    old = heavy_hex_paper(21)
    groups = PAPER_GROUPS_21
    assert tuple(map(len, groups)) == (6, 9, 7)
    paper_edges = {_ordered(*edge) for group in groups for edge in group}
    mapped_old = {_ordered(_OLD_TO_PAPER_21[i], _OLD_TO_PAPER_21[j])
                  for i, j in old.edges}
    assert len(paper_edges) == 22 and paper_edges == mapped_old
    inverse = {paper: old_q for old_q, paper in enumerate(_OLD_TO_PAPER_21)}
    ordered_edges = tuple(_ordered(*edge) for group in groups for edge in group)
    lattice = HeavyHexLattice(
        n_qubits=21,
        edges=ordered_edges,
        edge_colors=tuple(c for c, group in enumerate(groups) for _ in group),
        positions={q: old.positions[inverse[q]] for q in range(21)},
    )
    lattice.validate()
    assert sum(degree == 2 for degree in lattice.degrees()) == 19
    return lattice


def paper_colored_lattice(n_qubits: int) -> HeavyHexLattice:
    """Ordered paper geometry: Fig. E1 (21/28/35), Fig. A1 (51/74)."""
    if n_qubits == 21:
        return paper_colored_21()
    if n_qubits == 28:
        groups = PAPER_GROUPS_28
        positions = {q: (float(q), 0.) for q in range(9)}
        positions.update({9: (0., -1.), 10: (4., -1.), 11: (8., -1.)})
        positions.update({q: (float(q - 12), -2.) for q in range(12, 21)})
        positions.update({21: (2., -3.), 22: (6., -3.)})
        positions.update({q: (float(q - 21), -4.) for q in range(23, 28)})
    elif n_qubits == 35:
        groups = PAPER_GROUPS_35
        positions = {q: (float(q + 2), 0.) for q in range(5)}
        positions.update({5: (2., -1.), 6: (6., -1.)})
        positions.update({q: (float(q - 7), -2.) for q in range(7, 16)})
        positions.update({16: (0., -3.), 17: (4., -3.), 18: (8., -3.)})
        positions.update({q: (float(q - 19), -4.) for q in range(19, 28)})
        positions.update({28: (2., -5.), 29: (6., -5.)})
        positions.update({q: (float(q - 28), -6.) for q in range(30, 35)})
    elif n_qubits == 51:
        groups = PAPER_GROUPS_51
        positions = _paper_a1_positions(PAPER_PHYSICAL_LABELS_51)
    elif n_qubits == 74:
        groups = PAPER_GROUPS_74
        positions = _paper_a1_positions(PAPER_PHYSICAL_LABELS_74)
    else:
        raise ValueError("Paper layer transcriptions exist for 21, 28, 35, 51, and 74")
    edges = tuple(_ordered(*edge) for group in groups for edge in group)
    assert len(set(edges)) == len(edges)
    lattice = HeavyHexLattice(
        n_qubits=n_qubits,
        edges=edges,
        edge_colors=tuple(c for c, group in enumerate(groups) for _ in group),
        positions=positions,
    )
    lattice.validate()
    assert sum(degree == 2 for degree in lattice.degrees()) == {28: 24, 35: 29, 51: 41, 74: 58}[n_qubits]
    return lattice


def nearest_patch_dims(
    n_qubits: int, max_rows: int = 6, max_cols: int = 60
) -> tuple[int, int, int]:
    """``(rows, cols, actual_qubits)`` of the full patch closest to ``n_qubits``.

    Ties break toward the *larger* patch, then toward *fewer rows* (ladder-like),
    so the two-row family is preferred wherever it is as close as any other.
    """
    best: tuple[tuple[int, int, int, int], int, int, int] | None = None
    for rows in range(2, max_rows + 1):
        for cols in range(2, max_cols + 1):
            n = _build_subdivided_honeycomb(rows, cols)[0]
            if n > n_qubits + 60:
                break
            key = (abs(n - n_qubits), 0 if n >= n_qubits else 1, rows, cols)
            if best is None or key < best[0]:
                best = (key, rows, cols, n)
    assert best is not None
    return best[1], best[2], best[3]


def heavy_hex(n_qubits: int) -> HeavyHexLattice:
    """Heavy-hex patch closest to ``n_qubits`` qubits (the finite-size knob).

    Heavy-hex patches exist only at discrete sizes (18, 25, 31, 32, 34, ...), so a
    requested count is **snapped** to the nearest full patch -- which, unlike an
    arbitrary carved sub-patch, keeps its hexagonal plaquettes. When the snap
    changes the size, a warning reports the actual qubit count. Use
    :func:`heavy_hex_ladder` for a clean monotone scaling family.
    """
    if n_qubits < 2:
        raise ValueError("n_qubits must be >= 2")

    rows, cols, n = nearest_patch_dims(n_qubits)
    if n != n_qubits:
        warnings.warn(
            f"heavy_hex({n_qubits}) snapped to the nearest full patch: {n} qubits "
            f"(rows={rows}, cols={cols}). Heavy-hex comes in discrete sizes; "
            f"carving an arbitrary count would destroy the hexagonal plaquettes.",
            stacklevel=2,
        )
    return heavy_hex_patch(rows, cols)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _build_subdivided_honeycomb(
    rows: int, cols: int
) -> tuple[int, list[Edge], dict[int, tuple[float, float]]]:
    """Brick-wall honeycomb on a ``(rows+1) x (cols+1)`` vertex grid, then
    subdivide every bond with a flag qubit -> heavy-hex."""
    site = {}
    for r in range(rows + 1):
        for c in range(cols + 1):
            site[(r, c)] = len(site)
    pos: dict[int, tuple[float, float]] = {
        i: (c * 2.0, -r * 2.0) for (r, c), i in site.items()
    }

    # Honeycomb (brick wall): all horizontal bonds; vertical bonds only where
    # (r + c) is even -> every interior vertex has degree 3.
    site_edges: list[Edge] = []
    for r in range(rows + 1):
        for c in range(cols):
            site_edges.append((site[(r, c)], site[(r, c + 1)]))
    for r in range(rows):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                site_edges.append((site[(r, c)], site[(r + 1, c)]))

    # Subdivide: insert a flag qubit in the middle of each site-site bond.
    n = len(site)
    edges: list[Edge] = []
    for u, v in site_edges:
        flag = n
        n += 1
        pos[flag] = ((pos[u][0] + pos[v][0]) / 2.0, (pos[u][1] + pos[v][1]) / 2.0)
        edges.append(_ordered(u, flag))
        edges.append(_ordered(flag, v))
    return n, edges, pos


def _two_core(
    n: int, edges: list[Edge]
) -> tuple[set[int], dict[int, set[int]]]:
    """The 2-core of ``(n, edges)``: iteratively remove every degree-<=1 node until
    only nodes of degree >= 2 remain. Returns the surviving node set and the pruned
    adjacency. Used to strip the dangling flag qubits off a ladder's open ends so
    the result is a clean loops-only heavy-hex subgraph (see :func:`heavy_hex_paper`)."""
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    alive = set(range(n))
    changed = True
    while changed:
        changed = False
        for v in list(alive):
            if len(adj[v]) <= 1:
                for w in adj[v]:
                    adj[w].discard(v)
                adj[v].clear()
                alive.discard(v)
                changed = True
    return alive, adj


def _colored(
    n: int, edges: list[Edge], pos: dict[int, tuple[float, float]]
) -> HeavyHexLattice:
    edges = [_ordered(i, j) for i, j in edges]
    g = _to_pygraph(n, edges)
    coloring = rx.graph_bipartite_edge_color(g)  # {edge_index: color}
    # Align colours to our `edges` order via the graph's own edge list.
    color_by_pair = {
        _ordered(u, v): coloring[ei] for ei, (u, v) in enumerate(g.edge_list())
    }
    edge_colors = tuple(color_by_pair[e] for e in edges)
    lat = HeavyHexLattice(
        n_qubits=n,
        edges=tuple(edges),
        edge_colors=edge_colors,
        positions=pos,
    )
    lat.validate()
    return lat


def _to_pygraph(n: int, edges: list[Edge]) -> rx.PyGraph:
    g = rx.PyGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from([(i, j, None) for i, j in edges])
    return g


def _ordered(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)
