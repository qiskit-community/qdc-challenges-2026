"""Classical reference for the QDC 2026 Floquet-dynamics challenge.

An exact statevector baseline for the Floquet mixed-field Ising magnet on
heavy-hex lattices -- the ground truth participants compare their error-mitigated
Heron r3 results against.

Quick start
-----------
    from reference import FloquetSimulator, paper_colored_lattice

    # the paper's Fig. E1 colouring of the 21-qubit patch, 30 Floquet cycles
    result = FloquetSimulator(paper_colored_lattice(21)).run(30)
    print(result.magnetization[:2], result.magnetization_z2[:2])
"""

from __future__ import annotations

from .analysis import (
    LateCycleFit,
    ScalingFit,
    dominant_period,
    fit_amplitude_scaling,
    fit_late_cycle,
    fit_late_cycle_anchored,
    oscillation_amplitude,
    prethermal_value,
)
from .backend import Backend, get_backend
from .lattice import (
    HeavyHexLattice,
    heavy_hex,
    heavy_hex_ladder,
    heavy_hex_paper,
    paper_colored_21,
    paper_colored_lattice,
    PAPER_GROUPS_21,
    PAPER_GROUPS_28,
    PAPER_GROUPS_35,
    PAPER_GROUPS_51,
    PAPER_GROUPS_74,
    PAPER_PHYSICAL_LABELS_51,
    PAPER_PHYSICAL_LABELS_74,
    heavy_hex_patch,
    nearest_patch_dims,
)
from .make_figure3 import load_paper_51q, load_results
from .plotting import (
    figure3,
    figure3_comparison,
    figure3_comparison_partial,
    plot_amplitude_scaling,
    plot_dynamics,
    plot_finite_size,
    plot_lattice,
)
from .simulator import FloquetParams, FloquetResult, FloquetSimulator, run_reference

__all__ = [
    "Backend",
    "get_backend",
    "HeavyHexLattice",
    "heavy_hex",
    "heavy_hex_ladder",
    "heavy_hex_paper",
    "paper_colored_21",
    "paper_colored_lattice",
    "PAPER_GROUPS_21",
    "PAPER_GROUPS_28",
    "PAPER_GROUPS_35",
    "PAPER_GROUPS_51",
    "PAPER_GROUPS_74",
    "PAPER_PHYSICAL_LABELS_51",
    "PAPER_PHYSICAL_LABELS_74",
    "heavy_hex_patch",
    "nearest_patch_dims",
    "FloquetParams",
    "FloquetResult",
    "FloquetSimulator",
    "run_reference",
    "ScalingFit",
    "LateCycleFit",
    "dominant_period",
    "fit_amplitude_scaling",
    "fit_late_cycle",
    "fit_late_cycle_anchored",
    "oscillation_amplitude",
    "prethermal_value",
    "figure3",
    "figure3_comparison",
    "figure3_comparison_partial",
    "load_paper_51q",
    "load_results",
    "plot_amplitude_scaling",
    "plot_dynamics",
    "plot_finite_size",
    "plot_lattice",
]
