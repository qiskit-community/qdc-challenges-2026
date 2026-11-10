"""Figures in the style of Barron et al. (arXiv:2607.25998), so results can be compared
with the paper's Figs. 1-3. Plotting only; no QPU calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from reference import FloquetResult

# Paper palette: sage for mitigated, gray for unmitigated, coral and navy for roles.
SAGE, GRAY, CORAL, NAVY, PURPLE = "#7FB5A1", "#8C8C8C", "#E2795B", "#34345C", "#6A5D9E"


def use_paper_style() -> None:
    """Serif type, inward ticks and light legends, as in the paper's figures."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 11, "legend.fontsize": 9, "legend.framealpha": 0.9,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "axes.linewidth": 0.8, "lines.linewidth": 1.0, "lines.markersize": 6,
        "errorbar.capsize": 2, "savefig.dpi": 150, "figure.dpi": 110,
    })


def _mark(ax, x, y, err, color, marker, label, dashed=True):
    ax.errorbar(x, y, yerr=err, fmt=marker, color=color, mec=color, mfc=color, ls="--" if dashed else "none",
                lw=1.0, label=label)


def heron_coordinates(backend) -> dict[int, tuple[int, int]] | None:
    """Grid (row, column) of each qubit on a 156-qubit Heron, or None for other devices.

    Heron has 8 rows of 16 qubits joined by 4 bridge qubits between rows, alternately at
    columns 3, 7, 11, 15 and 1, 5, 9, 13. The coordinates are checked against the
    backend's coupling map, so a device with a different structure falls back.
    """
    if backend.num_qubits != 156:
        return None
    bridges = ([3, 7, 11, 15], [1, 5, 9, 13])
    coords = {}
    for q in range(backend.num_qubits):
        block, r = divmod(q, 20)
        coords[q] = (2 * block, r) if r < 16 else (2 * block + 1, bridges[block % 2][r - 16])
    for a, b in backend.coupling_map.get_edges():
        if abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1]) != 1:
            return None
    return coords

def _floquet_error_snapshot(backend) -> dict:
    """Calibrated X/SX, readout and RZZ errors from a fractional-gate backend."""
    target = backend.target
    if "rzz" not in target.operation_names:
        raise ValueError("The backend target does not expose RZZ gates.")

    qubits = {q: {"x": np.nan, "sx": np.nan, "readout": np.nan}
              for q in range(backend.num_qubits)}
    for gate, field in (("x", "x"), ("sx", "sx"), ("measure", "readout")):
        if gate not in target.operation_names:
            continue
        for qargs, props in target[gate].items():
            if qargs is not None and len(qargs) == 1 and props is not None and props.error is not None:
                qubits[qargs[0]][field] = float(props.error)

    samples = {}
    for qargs, props in target["rzz"].items():
        if qargs is None or len(qargs) != 2:
            continue
        edge = (min(qargs), max(qargs))
        value = np.nan if props is None or props.error is None else float(props.error)
        samples.setdefault(edge, []).append(value)
    edges = {}
    for edge, values in samples.items():
        known = [value for value in values if np.isfinite(value)]
        edges[edge] = float(np.mean(known)) if known else np.nan
    if not edges:
        raise ValueError("The backend target has no RZZ couplers.")
    return {"qubits": qubits, "edges": edges}


def plot_floquet_layouts(backend, layout: Sequence[int], groups: Sequence[Sequence[tuple[int, int]]]):
    """Full-device and selected Floquet maps with identical marker and line sizes."""
    coords = heron_coordinates(backend)
    if coords is None:
        raise ValueError("plot_floquet_layouts draws 156-qubit Heron devices only.")

    snapshot = _floquet_error_snapshot(backend)
    qubits, edges = snapshot["qubits"], snapshot["edges"]
    layout = list(layout)
    selected_qubits = set(layout)
    if len(selected_qubits) != len(layout) or not selected_qubits <= set(qubits):
        raise ValueError("The selected layout must contain distinct device qubits.")

    selected_edges = {
        (min(layout[i], layout[j]), max(layout[i], layout[j]))
        for group in groups for i, j in group
    }
    if not selected_edges <= set(edges):
        raise ValueError("The selected layout contains a bond without RZZ support.")

    def norm(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        cap = float(np.percentile(values, 95)) if len(values) else 0.0
        return Normalize(0, max(cap, 1e-6), clip=True)

    readout_norm = norm([v["readout"] for v in qubits.values()])
    rzz_norm = norm(list(edges.values()))
    readout_cmap = rzz_cmap = plt.get_cmap("RdBu")

    marker_size = 52
    gate_width = 2.2
    gate_outline_width = 3.2
    background_width = 0.8

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    device_edges = (
        {(min(a, b), max(a, b)) for a, b in backend.coupling_map.get_edges()}
        | set(edges)
    )
    panels = (
        (axes[0], set(qubits), set(edges), "(a) Full device"),
        (
            axes[1], selected_qubits, selected_edges,
            f"(b) Selected Floquet layout: {len(selected_qubits)} qubits, "
            f"{len(selected_edges)} bonds",
        ),
    )

    for ax, shown, mapped, title in panels:
        for a, b in sorted(device_edges):
            (ra, ca), (rb, cb) = coords[a], coords[b]

            if (a, b) not in mapped:
                ax.plot(
                    [ca, cb], [ra, rb], color="#CFCFCF",
                    lw=background_width, zorder=0,
                )
                continue

            value = edges.get((a, b), np.nan)
            color = rzz_cmap(rzz_norm(value)) if np.isfinite(value) else "#A6A6A6"
            ax.plot(
                [ca, cb], [ra, rb], color="#606060",
                lw=gate_outline_width, zorder=1,
            )
            ax.plot(
                [ca, cb], [ra, rb], color=color,
                lw=gate_width, zorder=2,
            )

        for q, (r, c) in coords.items():
            if q in shown:
                value = qubits[q]["readout"]
                color = readout_cmap(readout_norm(value)) if np.isfinite(value) else "#CFCFCF"
                ax.scatter(
                    c, r, s=marker_size, color=color,
                    edgecolor="#707070", lw=0.4, zorder=3,
                )
                label_color = (
                    "white" if np.isfinite(value) and readout_norm(value) > 0.55
                    else "black"
                )
                ax.text(
                    c, r, str(q), ha="center", va="center",
                    fontsize=4.2, color=label_color, zorder=4,
                )
            else:
                ax.scatter(
                    c, r, s=28, facecolor="white",
                    edgecolor="#A6A6A6", lw=0.8, zorder=2,
                )

        ax.set(
            xlim=(-0.8, 15.8),
            ylim=(max(r for r, _ in coords.values()) + 0.8, -0.8),
            aspect="equal",
        )
        ax.axis("off")
        ax.set_title(title, fontsize=10)

    fig.suptitle(f"{backend.name}: calibrated Floquet layouts", fontsize=12)
    fig.subplots_adjust(bottom=0.16, top=0.94, wspace=0.15)

    for left, scale, cmap, label in (
        (0.10, readout_norm, readout_cmap, "Readout error (95th percentile cap)"),
        (0.55, rzz_norm, rzz_cmap, "RZZ error (95th percentile cap)"),
    ):
        bar_ax = fig.add_axes([left, 0.075, 0.35, 0.015])
        bar = fig.colorbar(
            ScalarMappable(norm=scale, cmap=cmap),
            cax=bar_ax, orientation="horizontal",
        )
        bar.ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        bar.set_label(label)

    return fig

def plot_floquet_error_cdfs(backend, layout: Sequence[int], groups: Sequence[Sequence[tuple[int, int]]],
                            xlim=(8e-5, 0.2)):
    """Full-device and selected-layout error CDFs with matching axes and colors."""
    snapshot = _floquet_error_snapshot(backend)
    qubits, edges = snapshot["qubits"], snapshot["edges"]
    layout = list(layout)
    selected_qubits = set(layout)
    if len(selected_qubits) != len(layout) or not selected_qubits <= set(qubits):
        raise ValueError("The selected layout must contain distinct device qubits.")
    selected_edges = {(min(layout[i], layout[j]), max(layout[i], layout[j]))
                      for group in groups for i, j in group}
    if not selected_edges <= set(edges):
        raise ValueError("The selected layout contains a bond without RZZ support.")

    fig, axes = plt.subplots(1, 2, figsize=(8., 4.), sharex=True, sharey=True)
    panels = ((axes[0], list(qubits), list(edges), "(a) Full device"),
              (axes[1], layout, sorted(selected_edges),
               f"(b) Selected Floquet layout: {len(selected_qubits)} qubits, {len(selected_edges)} bonds"))
    for ax, shown, mapped, title in panels:
        series = {
            "1Q gates (SX/X)": [qubits[q][gate] for q in shown for gate in ("sx", "x")],
            "RZZ gates": [edges[edge] for edge in mapped],
            "Readout": [qubits[q]["readout"] for q in shown],
        }
        for label, data in series.items():
            values = np.asarray(data, dtype=float)
            values = np.sort(np.clip(values[np.isfinite(values)], 1e-5, None))
            if not len(values):
                continue
            line, = ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=label)
            ax.axvline(np.median(values), color=line.get_color(), ls="--", lw=0.9)
        ax.set(xscale="log", ylabel="Cumulative probability", ylim=(0, 1.02), xlim=xlim)
        ax.set_title(title, fontsize=10)
    axes[0].legend(loc="upper left")
    axes[1].set_xlabel("Error rate")
    fig.suptitle(f"{backend.name}: calibrated errors (dashed: medians)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig

def plot_cycle_folded(run, sample_cycles=None):
    cycles = run["cycles"]
    factors = run["factors"]

    if sample_cycles is None:
        sample_cycles = np.unique(
            cycles[[0, len(cycles) // 2, len(cycles) - 1]]
        )

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    for row, observable in enumerate(("M", "Mz2")):
        values = run["values"][observable]
        trace_axis, factor_axis = axes[row]

        for column, factor in enumerate(factors):
            trace_axis.plot(
                cycles, values[:, column], marker="s", linestyle="-",
                markeredgecolor="black", markeredgewidth=1.0,
                label=f"factor {factor}",
            )

        for cycle in sample_cycles:
            matching = np.flatnonzero(cycles == cycle)
            if len(matching):
                factor_axis.plot(
                    factors, values[matching[0], :], marker="s", linestyle="-",
                    markeredgecolor="black", markeredgewidth=1.0,
                    label=f"cycle {cycle}",
                )

        trace_axis.set(xlabel="Floquet cycle", ylabel=observable)
        factor_axis.set(xlabel="Cycle-fold factor", ylabel=observable)
        trace_axis.legend()
        factor_axis.legend()

    fig.suptitle(f"{run['n_qubits']}-qubit TREX measurements")
    fig.tight_layout()
    plt.show()
    return fig
def plot_zne_extrapolation(
    reference: Sequence[float],
    mitigated_values: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    reference_cycles: Sequence[int] | None = None,
    measured_cycles: Sequence[int] | None = None,
):
    """Compare exact magnetization with zero-noise extrapolated traces.

    Explicit cycle labels let a complete reference and sparsely measured traces
    retain their physical horizontal coordinates. ``measured_cycles`` includes
    cycle 0; a trace with one fewer point gets the reference's known initial
    value there. Without labels, the archived dense 0, 1, ... grid is retained.
    """
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("reference must be a non-empty one-dimensional trace")
    if len(mitigated_values) != len(labels):
        raise ValueError("mitigated_values and labels must have the same length")

    def cycle_labels(values, name):
        try:
            values = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain nonnegative integer labels") from exc
        if (values.ndim != 1 or values.size == 0
                or not np.all(np.isfinite(values))
                or np.any(values < 0) or np.any(values != np.floor(values))
                or np.any(np.diff(values) <= 0)):
            raise ValueError(
                f"{name} must be a non-empty, strictly increasing sequence "
                "of nonnegative integer labels"
            )
        return values.astype(int)

    reference_cycles = cycle_labels(
        np.arange(reference.size) if reference_cycles is None else reference_cycles,
        "reference_cycles",
    )
    if reference_cycles.size != reference.size:
        raise ValueError("reference_cycles and reference must have the same length")
    cycles = cycle_labels(
        reference_cycles if measured_cycles is None else measured_cycles,
        "measured_cycles",
    )
    if measured_cycles is not None and cycles[0] != 0:
        raise ValueError("measured_cycles must include cycle 0 as its first label")

    curves = []
    for label, curve in zip(labels, mitigated_values):
        curve = np.asarray(curve, dtype=float)
        if curve.ndim != 1:
            raise ValueError(f"{label} must be a one-dimensional trace")
        if curve.size == cycles.size - 1:
            if (cycles[0] != 0 or reference_cycles[0] != 0
                    or not np.isfinite(reference[0])):
                raise ValueError(
                    f"{label} omits cycle 0, but no cycle-zero reference is available"
                )
            curve = np.r_[reference[0], curve]
        elif curve.size != cycles.size:
            raise ValueError(
                f"{label} has {curve.size} values; expected "
                f"{cycles.size - 1} or {cycles.size}"
            )
        curves.append(curve)

    colors = (SAGE, CORAL, NAVY, PURPLE)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(reference_cycles, reference, color="black", label="Reference", lw=2.5)

    for index, (label, curve) in enumerate(zip(labels, curves)):
        ax.plot(cycles, curve, color=colors[index % len(colors)],
                marker="s", linestyle="none", 
                markeredgecolor="black", markeredgewidth=1.0,
                label=label)

    ax.set(xlabel="Floquet cycle", ylabel="Magnetization",
           xlim=(0, max(reference_cycles[-1], cycles[-1])))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_mitigated_magnetization(
    m_mit: Sequence[float],
    *,
    reference: FloquetResult,
    measured_cycles: Sequence[int],
) -> Figure:
    """Plot first-pass mitigated full magnetization against an exact reference.

    Args:
        m_mit: Linear-ZNE magnetization values in measured-cycle order, optionally
            omitting the known cycle-zero value.
        reference: Exact result exposing ``magnetization`` and ``cycles``.
        measured_cycles: Physical measured-cycle labels, including zero first.

    Returns:
        Figure with the exact classical reference and first-pass linear-ZNE
        trace. A missing cycle-zero estimate is supplied by the reference.

    Input and cycle-label errors are reported by ``plot_zne_extrapolation``.
    This creates a figure but leaves displaying or saving it to the notebook.
    """
    values = np.asarray(m_mit, dtype=float)
    fig = plot_zne_extrapolation(
        reference.magnetization, [values], ["First-pass linear ZNE"],
        reference_cycles=reference.cycles, measured_cycles=measured_cycles,
    )
    ax = fig.axes[0]
    ax.lines[0].set_label("Exact classical reference")
    ax.legend()
    return fig


def plot_multi_size_zne(
    runs_by_size,
    extrapolated_by_size,
    selected_method,
    ref21,
    paper_51q,
    n_cycles=30,
    references_by_size=None,
):
    """Plot measured TREX/ZNE traces and the supplied targets at every size."""
    references = {21: ref21, 51: paper_51q}
    references.update(references_by_size or {})
    sizes = sorted(runs_by_size)
    if not sizes:
        raise ValueError("runs_by_size must contain at least one run")

    fig, axes = plt.subplots(
        len(sizes), 2, figsize=(14, 3.25 * len(sizes)),
        sharex=True, squeeze=False,
    )
    colors = {"linear": "C0", "richardson": "C1", "exp": "C2"}

    for row, n in enumerate(sizes):
        run = runs_by_size[n]
        cycles = np.asarray(run["cycles"])
        factor_one = tuple(run["factors"]).index(1)
        for col, (observable, target_attr) in enumerate(
            (("M", "magnetization"), ("Mz2", "magnetization_z2"))
        ):
            ax = axes[row, col]
            raw = np.asarray(run["values"][observable])[:, factor_one]
            ax.plot(np.r_[0, cycles], np.r_[1.0, raw], marker="s", linestyle="none",
                    markeredgecolor="black", markeredgewidth=1.0, color="0.6",
                    label="TREX, factor 1")
            for name, traces in extrapolated_by_size[n].items():
                chosen = name == selected_method
                ax.plot(
                    np.r_[0, cycles], np.r_[1.0, traces[observable]],
                    marker="s", markeredgecolor="black", markeredgewidth=1.0, 
                    linestyle="none", ms=6 if chosen else 4,
                    color=colors.get(name),
                    label=name + (" (selected)" if chosen else ""),
                )
            target = references.get(n)
            if target is not None:
                t_cycles = np.asarray(target.cycles, dtype=float)
                t_values = np.asarray(getattr(target, target_attr), dtype=float)
                # Digitized paper traces hold NaN at unplotted cycles, which
                # would break the line; connect the finite points instead.
                finite = np.isfinite(t_values)
                is_paper = n == 51
                ax.plot(t_cycles[finite], t_values[finite],
                        "k--" if is_paper else "k:", lw=1.8,
                        label=("digitized data from the paper" if is_paper
                               else "classical reference"))
            ax.set(ylabel=f"{n}q  {observable}", xlim=(0, n_cycles))
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(observable)
            if row == len(sizes) - 1:
                ax.set_xlabel("Floquet cycle")
            if col == 0:
                ax.legend(fontsize=8, ncol=2)

    fig.suptitle("TREX and ZNE across system sizes")
    fig.tight_layout()
    return fig
