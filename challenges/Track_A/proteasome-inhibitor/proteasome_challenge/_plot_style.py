"""
proteasome_challenge._plot_style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Restrained plotting utilities for the QDC 2026 Proteasome laboratory.

Palette (colour-blind safe subset of Wong 2011):
  BLUE   #0072B2   ixazomib / SQD
  ORANGE #D55E00   bortezomib / TrimSQD
  GRAY   #8A8F98   reference / CCSD / neutral elements

Usage in the notebook::

    from proteasome_challenge._plot_style import apply, BLUE, ORANGE, GRAY, ligand_color, method_color
    apply()                   # call once in the setup cell
    fig, ax = plt.subplots()
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Palette ──────────────────────────────────────────────────────────────────
BLUE   = "#0072B2"
ORANGE = "#D55E00"
GRAY   = "#8A8F98"

LIGAND_COLORS  = {"ixazomib": BLUE,   "bortezomib": ORANGE}
METHOD_COLORS  = {"sqd":      BLUE,   "trimsqd":    ORANGE, "ccsd": GRAY, "fci": GRAY}
METHOD_MARKERS = {"sqd":      "o",    "trimsqd":    "s",    "ccsd": "^",  "fci": "^"}


def ligand_color(ligand: str) -> str:
    return LIGAND_COLORS.get(ligand.lower(), GRAY)


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method.lower(), GRAY)


def method_marker(method: str) -> str:
    return METHOD_MARKERS.get(method.lower(), "D")


# ── Global rcParams ───────────────────────────────────────────────────────────
_RC = {
    # Figure
    "figure.figsize":         (8.0, 3.8),
    "figure.dpi":             120,
    "figure.facecolor":       "white",
    "axes.facecolor":         "white",
    "savefig.facecolor":      "white",
    # Font
    "font.family":            "sans-serif",
    "font.size":              11,
    "axes.titlesize":         11,
    "axes.labelsize":         10,
    "xtick.labelsize":        9,
    "ytick.labelsize":        9,
    "legend.fontsize":        9,
    # Lines and markers
    "lines.linewidth":        1.6,
    "lines.markersize":       6,
    "patch.linewidth":        1.4,
    # Axes
    "axes.linewidth":         0.8,
    "axes.grid":              True,
    "grid.color":             "#DDDDDD",
    "grid.linewidth":         0.6,
    "grid.alpha":             1.0,
    # Ticks
    "xtick.direction":        "out",
    "ytick.direction":        "out",
    "xtick.major.width":      0.8,
    "ytick.major.width":      0.8,
    # Spines — keep only bottom and left
    "axes.spines.top":        False,
    "axes.spines.right":      False,
    # Color cycle (BLUE, ORANGE, GRAY, then muted variants)
    "axes.prop_cycle":        mpl.cycler(
        color=[BLUE, ORANGE, GRAY, "#56B4E9", "#E69F00", "#009E73"]
    ),
    # Legend
    "legend.frameon":         True,
    "legend.framealpha":      0.9,
    "legend.edgecolor":       "#CCCCCC",
    # Layout
    "figure.constrained_layout.use": True,
}


def apply() -> None:
    """Apply the lab's rcParams globally.  Call once in the notebook setup cell."""
    mpl.rcParams.update(_RC)


def subplots(*args, **kwargs):
    """Thin wrapper around ``plt.subplots`` that returns (fig, ax) with the lab style."""
    kwargs.setdefault("figsize", _RC["figure.figsize"])
    return plt.subplots(*args, **kwargs)


def finalize(fig: mpl.figure.Figure, ax, title: str | None = None, caption: str | None = None) -> None:
    """Tighten layout, optionally set a title/caption, then show."""
    if title:
        ax.set_title(title, pad=6)
    if caption:
        fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=8.5, color="#555555")
    plt.show()
