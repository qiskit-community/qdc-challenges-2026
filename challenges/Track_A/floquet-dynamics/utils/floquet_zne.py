"""Shared input validation for Floquet ZNE exercises.

The notebook retains the participant's fitter, model definitions and constraints.
The grader supplies independent organizer fallbacks for failed exercises.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _zne_data(
    run: Mapping[str, Any],
    chosen_factors: Sequence[int],
    observable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select noise-factor columns and validate a run's measured fit inputs.

    ``run['values'][observable]`` and ``run['stds'][observable]`` use rows for
    cycles and columns in ``run['factors']`` order. At least two distinct chosen
    factors must exist, measured values must be finite, and the selected standard
    errors must have the same shape as the selected values.

    Returns floating-point factor coordinates, values, and standard errors in
    the requested factor order without changing cycle order or the input run.
    Raises ``ValueError`` for invalid factors, values, or incompatible shapes.
    """
    available = tuple(run["factors"])
    chosen = tuple(chosen_factors)
    if len(chosen) < 2 or len(set(chosen)) != len(chosen):
        raise ValueError("Choose at least two distinct noise factors")
    if any(factor not in available for factor in chosen):
        raise ValueError("A chosen factor is missing from the run")
    columns = [available.index(factor) for factor in chosen]
    values = np.asarray(run["values"][observable], dtype=float)[:, columns]
    if not np.isfinite(values).all():
        raise ValueError("Cannot extrapolate non-finite measurements")
    stds = np.asarray(run["stds"][observable], dtype=float)[:, columns]
    if stds.shape != values.shape:
        raise ValueError("Standard errors and measurements must have the same shape")
    return np.asarray(chosen, dtype=float), values, stds
