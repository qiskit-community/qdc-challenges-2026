"""Visualization helpers for [[4,2,2]] Iceberg code blocks on heavy-hex connectivity.

Reproduces the single-block diagrams shown in the insets of Fig. 1c) and in Fig. 6a)
of Froland et al., arXiv:2607.24947 (see doc/Paper_Device_422.tex).

A block uses six physical qubits: four data qubits d0-d3 and two ancillas a_x, a_z
used for syndrome readout and logical operations. The six sites form a fixed shape on
the heavy-hex lattice (one degree-3 vertex, two degree-2 vertices); the "compute" and
"syndrome" configurations differ only in which qubit sits on which site.
"""

import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .iceberg_block import ANCILLA_ROLES, CONFIGURATIONS, SITE_EDGES, SITE_POS

# Direction to nudge each site's label so it clears both the node and its edges.
LABEL_OFFSET = {
    0: (0.0, 0.34),
    1: (0.0, 0.34),
    2: (0.0, 0.34),
    3: (-0.36, 0.12),
    4: (-0.36, 0.20),
    5: (0.0, 0.34),
}

ANCILLA_COLOR = {"az": "#c0392b", "ax": "#2471a3"}
DATA_COLOR = "black"
EDGE_COLOR = "black"
BRIDGE_COLOR = "#1e8449"
UNUSED_COLOR = "#b3b6b7"

# Logical-graph nodes: ordinary sites, and true-vacuum bubble sites (dark red, white label).
LOGICAL_NODE_COLOR = "#d6dbdf"
LOGICAL_NODE_EDGE = "#5d6d7e"
BUBBLE_COLOR = "#922b21"
BUBBLE_EDGE = "#641e16"

# Rounded halo tracing each code block, as in Fig. 1b.
BLOCK_HALO_COLOR = "#ccd9e8"
BLOCK_HALO_WIDTH = 18.0

NODE_SIZE = 340
EDGE_WIDTH = 2.5
FONT_SIZE = 14

# Whole-device views need smaller marks than a single block.
DEVICE_NODE_SIZE = 70
DEVICE_EDGE_WIDTH = 1.4
DEVICE_FONT_SIZE = 7


def build_block(configuration="compute"):
    """Return a graph of one 6-qubit [[4,2,2]] block in the given configuration.

    Nodes are named 'd0'-'d3', 'ax', 'az' and carry 'pos', 'site' and 'kind'
    attributes; 'kind' is either 'data' or 'ancilla'.
    """
    if configuration not in CONFIGURATIONS:
        raise ValueError(
            "configuration must be one of %s, got %r"
            % (sorted(CONFIGURATIONS), configuration)
        )
    roles = CONFIGURATIONS[configuration]

    G = nx.Graph(configuration=configuration)
    for site, qubit in roles.items():
        G.add_node(
            qubit,
            pos=SITE_POS[site],
            site=site,
            kind="ancilla" if qubit in ANCILLA_ROLES else "data",
        )
    G.add_edges_from((roles[u], roles[v]) for u, v in SITE_EDGES)
    return G


def _math_label(qubit):
    """'d0' -> '$d_{0}$', 'az' -> '$a_{z}$'."""
    return "$%s_{%s}$" % (qubit[0], qubit[1:])


def plot_block(G, ax=None, title=None, labels=True):
    """Draw a single code block. Returns the matplotlib axes."""
    pos = nx.get_node_attributes(G, "pos")
    if ax is None:
        _, ax = plt.subplots(figsize=(2.8, 2.8))

    node_color = [ANCILLA_COLOR.get(q, DATA_COLOR) for q in G.nodes]
    nx.draw_networkx_edges(G, pos, ax=ax, width=EDGE_WIDTH, edge_color=EDGE_COLOR)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=NODE_SIZE, node_color=node_color)

    if labels:
        for qubit in G.nodes:
            x, y = pos[qubit]
            dx, dy = LABEL_OFFSET[G.nodes[qubit]["site"]]
            ax.text(
                x + dx,
                y + dy,
                _math_label(qubit),
                fontsize=FONT_SIZE,
                ha="center",
                va="center",
            )

    if title is None:
        title = G.graph.get("configuration", "").capitalize()
    if title:
        ax.set_title(title, fontsize=FONT_SIZE)

    ax.set_aspect("equal")
    ax.margins(0.22)
    ax.axis("off")
    return ax


def plot_configurations(configurations=("compute", "syndrome")):
    """Draw the compute and syndrome configurations side by side. Returns the figure."""
    fig, axes = plt.subplots(1, len(configurations), figsize=(2.8 * len(configurations), 3.0))
    axes = [axes] if len(configurations) == 1 else list(axes)
    for ax, configuration in zip(axes, configurations):
        plot_block(build_block(configuration), ax=ax)
    fig.tight_layout()
    return fig


def _device_node_color(attrs):
    if attrs["kind"] == "bridge":
        return BRIDGE_COLOR
    if attrs["kind"] == "unused":
        return UNUSED_COLOR
    return ANCILLA_COLOR.get(attrs["role"], DATA_COLOR)


def _draw_block_halos(ax, placement, pos, color=BLOCK_HALO_COLOR,
                      width=BLOCK_HALO_WIDTH):
    """Trace each block's five edges in a thick round-capped line.

    Because a block is a tree, thick round-capped segments along its edges merge into
    the rounded blob the paper uses to outline code blocks - no hull geometry needed.
    """
    for block in placement.blocks.values():
        for u, v in block.physical_edges():
            (x0, y0), (x1, y1) = pos[u], pos[v]
            ax.plot(
                [x0, x1], [y0, y1], color=color, linewidth=width,
                solid_capstyle="round", solid_joinstyle="round", zorder=0,
            )


def plot_placement(placement, ax=None, label_blocks=True, configuration="compute",
                   halos=True, heron=None):
    """Draw the whole device with each qubit colored by its role - Fig. 1b style.

    Data qubits black, ``az`` red, ``ax`` blue, bridge qubits green, unused gray.
    Blocks are set apart two ways: a rounded halo around each block's six qubits, and
    edge weight - only *intra-block* links are bold black, while inter-block,
    block-to-bridge and unused links are all thin gray.

    ``configuration`` selects which role labeling to color by, since the ancillas sit
    on different physical qubits in the compute and syndrome configurations.

    ``heron`` is forwarded to :meth:`Placement.physical_graph`: ``None`` auto-detects
    the heavy-hex lattice, ``True`` insists on it (so the plot is laid out as a real
    heavy-hex device even when the topology came from a bare ``CouplingMap``), ``False``
    omits the geometry and falls back to a computed layout.
    """
    graph = placement.physical_graph(heron=heron)
    pos = nx.get_node_attributes(graph, "pos")
    if len(pos) < graph.number_of_nodes():
        # No heavy-hex geometry: this placement is on a topology other than ibm_boston.
        pos = nx.spring_layout(graph, seed=0)
    if ax is None:
        _, ax = plt.subplots(figsize=(13.0, 11.0))

    if configuration == "syndrome":
        for _, attrs in graph.nodes(data=True):
            if attrs["kind"] == "block":
                attrs["role"] = attrs["syndrome_role"]

    if halos:
        _draw_block_halos(ax, placement, pos)

    active = [q for q, a in graph.nodes(data=True) if a["kind"] != "unused"]
    idle = [q for q, a in graph.nodes(data=True) if a["kind"] == "unused"]
    intra = [(u, v) for u, v, a in graph.edges(data=True) if a["kind"] == "intra_block"]
    rest = [(u, v) for u, v, a in graph.edges(data=True) if a["kind"] != "intra_block"]

    faint = nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=rest,
        width=DEVICE_EDGE_WIDTH, edge_color=UNUSED_COLOR,
    )
    bold = nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=intra,
        width=DEVICE_EDGE_WIDTH * 2.2, edge_color=EDGE_COLOR,
    )
    dim_nodes = nx.draw_networkx_nodes(
        graph, pos, ax=ax, nodelist=idle,
        node_size=DEVICE_NODE_SIZE * 0.6, node_color=UNUSED_COLOR,
    )
    lit_nodes = nx.draw_networkx_nodes(
        graph, pos, ax=ax, nodelist=active, node_size=DEVICE_NODE_SIZE,
        node_color=[_device_node_color(graph.nodes[q]) for q in active],
    )
    # Halos are Line2D (default zorder 2) and would otherwise cover the collections.
    for artist, order in ((faint, 1), (bold, 1.5), (dim_nodes, 3), (lit_nodes, 3)):
        if artist is not None:
            artist.set_zorder(order)

    if label_blocks:
        for index, block in placement.blocks.items():
            xs = [pos[q][0] for q in block.physical_qubits()]
            ys = [pos[q][1] for q in block.physical_qubits()]
            ax.text(
                sum(xs) / len(xs), sum(ys) / len(ys), str(index),
                fontsize=DEVICE_FONT_SIZE + 2, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="black", lw=0.8),
            )

    ax.set_aspect("equal")
    ax.margins(0.03)
    ax.axis("off")
    return ax


def logical_positions(placement, separation=1.1):
    """Map ``(b, l)`` -> plotting position, from each block's physical centroid.

    The paper's Fig. 2a schematic positions are not yet extracted, so this derives a
    device-faithful layout instead: each block sits at the centroid of its six physical
    qubits, and its two logical qubits are offset along the block's longer axis.
    """
    geometry = _device_layout()
    pos = {}
    for index, block in placement.blocks.items():
        xs = [geometry[q]["pos"][0] for q in block.physical_qubits()]
        ys = [geometry[q]["pos"][1] for q in block.physical_qubits()]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        horizontal = (max(xs) - min(xs)) >= (max(ys) - min(ys))
        shift = separation / 2.0
        offsets = ((-shift, 0.0), (shift, 0.0)) if horizontal else ((0.0, shift), (0.0, -shift))
        for l, (dx, dy) in zip((0, 1), offsets):
            pos[(index, l)] = (cx + dx, cy + dy)
    return pos


def _device_layout():
    from . import iceberg_device

    return iceberg_device.layout()


def plot_logical_graph(graph, bubble_sites=(), pos=None, ax=None, labels=True, highlight=(),
                       show_missing=True):
    """Draw a logical connectivity graph on ``(b, l)`` nodes.

    Positions come from the ``grid_pos`` node attribute when present, so the default
    Fig. 2a graph lands on its square lattice; otherwise they fall back to each block's
    physical centroid. Intra-block edges are drawn thick, inter-block edges thin, and
    the four square-lattice links the device cannot provide are dashed gray when
    ``show_missing``. ``highlight`` is an extra edge list drawn in color on top - handy
    for a route's endpoints or a candidate 1D chain. Nodes listed in ``bubble_sites``
    (the true-vacuum bubble, as ``(b, l)`` lattice sites) are drawn in dark red.
    """
    bubble = {tuple(s) for s in bubble_sites}
    unknown = bubble - set(graph.nodes)
    if unknown:
        raise ValueError("bubble_sites not in the graph: %s" % sorted(unknown))
    placement = graph.graph.get("placement")
    if pos is None:
        grid = nx.get_node_attributes(graph, "grid_pos")
        if len(grid) == graph.number_of_nodes():
            pos = grid
        elif placement is not None:
            pos = logical_positions(placement)
        else:
            raise ValueError("pass pos=..., or a graph with grid_pos / placement")
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 8.5))

    intra = [(u, v) for u, v, a in graph.edges(data=True) if a.get("kind") == "intra"]
    inter = [(u, v) for u, v, a in graph.edges(data=True) if a.get("kind") != "intra"]
    missing = [
        (u, v) for u, v in graph.graph.get("missing_edges", ())
        if u in pos and v in pos
    ]

    if show_missing and missing:
        nx.draw_networkx_edges(
            graph, pos, ax=ax, edgelist=missing, width=1.2,
            edge_color=UNUSED_COLOR, style="dashed",
        )
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=inter, width=1.8, edge_color="#f39c12"
    )
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=intra, width=5.0, edge_color="#e67e22"
    )
    if highlight:
        nx.draw_networkx_edges(
            graph, pos, ax=ax, edgelist=list(highlight), width=3.0, edge_color="#2471a3"
        )
    nodes = list(graph.nodes)
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, nodelist=nodes, node_size=620,
        node_color=[BUBBLE_COLOR if n in bubble else LOGICAL_NODE_COLOR for n in nodes],
        edgecolors=[BUBBLE_EDGE if n in bubble else LOGICAL_NODE_EDGE for n in nodes],
    )
    if labels:
        # one call per label color: white text on the dark bubble nodes
        for group, color in (([n for n in nodes if n not in bubble], "black"),
                             ([n for n in nodes if n in bubble], "white")):
            if group:
                nx.draw_networkx_labels(
                    graph, pos, ax=ax, labels={n: "%d,%d" % n for n in group},
                    font_size=DEVICE_FONT_SIZE + 1, font_color=color,
                )

    ax.set_aspect("equal")
    ax.margins(0.06)
    ax.axis("off")
    return ax


if __name__ == "__main__":
    from . import iceberg_layout

    plot_configurations()
    plt.savefig("block_configurations.png", dpi=200, bbox_inches="tight")

    place = iceberg_layout.Placement.from_qubits()
    plot_placement(place)
    plt.savefig("placement.png", dpi=160, bbox_inches="tight")

    plot_logical_graph(place.logical_graph())
    plt.savefig("logical_graph.png", dpi=160, bbox_inches="tight")
    print("wrote block_configurations.png, placement.png, logical_graph.png")



### Plotting Results 

def site_values(obs_k, field="O_k_star", k=None, reps=(0, 1)):
    """{site: value} from one ORP result (obs_k keyed by (site, rep)), averaging the representatives.

    field="O_k_star"        : the ORP value (default)
    field="O_k", k=0        : parity postselection only ("raw")
    field="O_k", k=<level>  : any fixed cut level
    """
    pick = (lambda d: d[field]) if k is None else (lambda d: d[field][k])
    sites = sorted({site for site, _ in obs_k})
    return {s: float(np.mean([pick(obs_k[(s, r)]) for r in reps if (s, r) in obs_k])) for s in sites}


def plot_bubble_grid(lattice, panels, times, bubble_sites=(), cmap="coolwarm", node_size=110,
                     title=None):
    """Fig. 20-style panels: <Zbar> on every logical qubit at its grid position.

    panels : {row label: [ {site: <Zbar>} for each time ]}, e.g. {"$g_z = 0.5$": [...], "$g_z = 1.5$": [...]}
    times  : the time of each column (t = n_steps * delta_t)
    bubble_sites : drawn with a black outline (the initial true-vacuum bubble)
    """
    pos = lattice.positions()
    rows = list(panels)
    fig, axes = plt.subplots(len(rows), len(times), figsize=(2.2 * len(times), 2.3 * len(rows)), dpi=300,
                             squeeze=False, layout="constrained")
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    bubble = set(bubble_sites)
    for i, row in enumerate(rows):
        for j, t in enumerate(times):
            ax = axes[i][j]
            for u, v in lattice.edges:            # couplings, drawn faintly
                (x0, y0), (x1, y1) = pos[u], pos[v]
                ax.plot([x0, x1], [y0, y1], color="#c8c8c8", lw=0.8, zorder=1)
            sites = list(lattice.nodes())
            vals = [panels[row][j].get(s, np.nan) for s in sites]
            xy = np.array([pos[s] for s in sites])
            ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap=cmap, norm=norm, s=node_size, zorder=2,
                       edgecolors=["black" if s in bubble else "white" for s in sites],
                       linewidths=[1.4 if s in bubble else 0.6 for s in sites], plotnonfinite=True)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if i == 0:
                ax.set_title(f"t = {t:g}", fontsize=10)
            if j == 0:
                ax.set_ylabel(row, fontsize=10)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.8, pad=0.02)
    cb.set_label(r"$\langle \bar Z_n \rangle$")
    if title:
        fig.suptitle(title, fontsize=11)
    return fig, axes



def cross_section(lattice, row=None, col=None):
    """Lattice sites along one grid row (left to right) or one grid column (top to bottom)."""
    if row is not None:
        return [lattice.sites[(r, c)] for (r, c) in sorted(lattice.sites) if r == row]
    return [lattice.sites[(r, c)] for (r, c) in sorted(lattice.sites) if c == col]


def site_errors(obs_k, reps=(0, 1)):
    """{site: sigma} at each estimator's k*, averaged over the representatives (they share shots,
    so this is a conservative error for the representative-averaged value)."""
    sites = sorted({site for site, _ in obs_k})
    return {s: float(np.mean([np.sqrt(obs_k[(s, r)]["sig2_k"][obs_k[(s, r)]["k_star"]])
                              for r in reps if (s, r) in obs_k])) for s in sites}

def unencoded_errors(unenc, n_shots):
    """{site: sigma} for unencoded <Z> values: binomial error sqrt((1 - <Z>^2) / n_shots)."""
    return {s: float(np.sqrt((1 - z**2) / n_shots)) for s, z in unenc.items()}

ENC_COLOR, UNENC_COLOR, MPS_COLOR = "#2a78d6", "#8c8c8c", "black"


def plot_cross_sections(lattice, panels, row, col, title=None):
    """Fig. 3b-style plot: <Zbar_n> along one grid row and one grid column, in separate panels.

    panels : {panel label: {"encoded": {site: <Z>}, "encoded_err": {site: sigma},
                            "unencoded": {site: <Z>}, "unencoded_err": {site: sigma},
                            "mps": {site: <Z>}}}
             (any entry may be omitted)
    One row of subplots per panel label; left column = row cross-section, right = column cross-section.
    """
    sections = [(f"row {row}", cross_section(lattice, row=row)),
                (f"column {col}", cross_section(lattice, col=col))]
    fig, axes = plt.subplots(len(panels), 2, figsize=(10, 2.8 * len(panels)), dpi=300, squeeze=False,
                             layout="constrained", sharey=True)
    for i, (label, data) in enumerate(panels.items()):
        for j, (name, sites) in enumerate(sections):
            ax, x = axes[i, j], np.arange(len(sites))
            if "mps" in data:
                ax.plot(x, [data["mps"][s] for s in sites], color=MPS_COLOR, lw=1.5)
            for key, color, marker in (("unencoded", UNENC_COLOR, "s"), ("encoded", ENC_COLOR, "o")):
                if key in data:
                    err = data.get(f"{key}_err")
                    ax.errorbar(x, [data[key][s] for s in sites],
                                yerr=[err[s] for s in sites] if err else None,
                                color=color, lw=1.2, marker=marker, ms=4, capsize=2)
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in sites], rotation=45, ha="right", fontsize=8)
            ax.set_ylim(-1.1, 1.1)
            ax.grid(color="#e5e5e5", lw=0.8)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(name, fontsize=10)
            if i == len(panels) - 1:
                ax.set_xlabel(r"Logical qubit $(b, l_b)$")
        axes[i, 0].set_ylabel(f"{label}\n" + r"$\langle \bar Z_n \rangle$")
    handles = [Line2D([], [], color=ENC_COLOR, marker="o", ms=4, label="encoded (ORP)"),
               Line2D([], [], color=UNENC_COLOR, marker="s", ms=4, label="unencoded"),
               Line2D([], [], color=MPS_COLOR, label="MPS")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=11)
    return fig, axes



def abs_errors(values, mps, n_list, n_used=None):
    """|<Z>_meas - <Z>_MPS| for every site and every time, flattened (NaNs dropped).

    values : list of {site: <Z>}, one per entry of n_list
    mps    : {n_steps: {site: <Z>}}
    n_used : the n_steps to include (default: all of n_list)
    """
    n_used = n_list if n_used is None else n_used
    errs = np.array([abs(values[j][s] - mps[n][s])
                     for j, n in enumerate(n_list) if n in n_used for s in values[j]])
    return errs[np.isfinite(errs)]


def plot_error_cdfs(panels, title=None):
    """Fig. 3c-style CDFs of the per-qubit, per-time absolute error, one panel per label.

    panels : {panel label: {"encoded": errors, "unencoded": errors}} (either may be omitted)
    Each median is drawn as a dotted vertical line and quoted in the legend.
    """
    styles = {"encoded": ("R = 1", ENC_COLOR, "-"), "unencoded": ("unenc", UNENC_COLOR, "--")}
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3), squeeze=False,
                             layout="constrained", sharey='all', sharex=True, dpi = 300)
    for ax, (label, data) in zip(axes[0, :], panels.items()):
        handles = []
        for key, (name, color, ls) in styles.items():
            if key not in data:
                continue
            errs = np.sort(data[key])
            ax.step(errs, 100 * np.arange(1, len(errs) + 1) / len(errs), where="post",
                    color=color, ls=ls, lw=2)
            med = np.median(errs)
            ax.axvline(med, color=color, ls=":", lw=1.5)
            handles.append(Line2D([], [], color=color, ls=ls, lw=2, label=f"{name}: {med:.3f}"))
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.set_xlabel(r"$|\langle \bar Z \rangle_{\mathrm{meas}} - \langle \bar Z \rangle_{\mathrm{MPS}}|$")
        ax.grid(color="#e5e5e5", lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(handles=handles, title=f"{label}\nmedian |err|", loc="upper left", fontsize=8,
                  title_fontsize=8, frameon=False)
    axes[0, 0].set_ylabel("CDF (%)")
    if title:
        fig.suptitle(title, fontsize=11)
    

