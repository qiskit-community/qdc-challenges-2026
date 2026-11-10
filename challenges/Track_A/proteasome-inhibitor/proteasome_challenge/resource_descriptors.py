"""
resource_descriptors — pre-execution computational resource descriptors.

All quantities are derived analytically from the embedded-cluster orbital
counts (n_orb, n_occ, n_vir) stored in the Stage 2 HDF5 dumpfiles and
propagated through ``fragment_diagnostics.csv``.  No FCI, CCSD, SQD, or
QPU run is required.

Exported functions
------------------
compute_resource_descriptors(n_orb, n_occ, n_vir)
    Compute all nine resource fields for one fragment.

enrich_catalog(df)
    Add resource-descriptor columns to a fragment catalog DataFrame in place.

Descriptor definitions
-----------------------
``n_alpha``, ``n_beta``
    Spin-up and spin-down electron counts.  The embedded clusters are
    closed-shell (RHF reference), so n_alpha = n_beta = n_occ.

``n_occ``, ``n_vir``
    Occupied and virtual spatial-orbital counts (passed through from the
    diagnostics; no new computation needed).

``fci_dimension``
    Exact number of spin-orbital Slater determinants in the FCI space::

        D = C(n_orb, n_alpha) * C(n_orb, n_beta)
            = C(n_orb, n_occ)^2

    Computed as a Python ``int`` (arbitrary precision) so overflow is
    impossible even for n_orb = 25.

``log10_fci_dimension``
    log10(D), stored as float64.  Uses ``math.log10(comb(...))`` on the
    exact integer ``fci_dimension``; never approximated by Stirling or
    summation of log-gammas.

``fci_vector_bytes``
    Memory to store one FCI state vector of float64 coefficients::

        fci_vector_bytes = D * 8

    Exact integer (Python int).

``ccsd_t2_elements``
    Number of elements in the CCSD doubles tensor t_{ijab}, using the
    full (non-antisymmetrized, non-symmetric) spatial-orbital convention::

        ccsd_t2_elements = n_occ^2 * n_vir^2

    This matches the convention used by PySCF / Vayesta, where ``t2`` has
    shape (n_occ, n_occ, n_vir, n_vir).  Exact integer.

``ccsd_t2_bytes``
    Memory to store the doubles tensor at float64::

        ccsd_t2_bytes = ccsd_t2_elements * 8

    Exact integer.

``sqd_n_qubits``
    Number of qubits required for Jordan–Wigner encoding of the fermionic
    Fock space::

        sqd_n_qubits = 2 * n_orb

    One qubit per spin-orbital, before any circuit optimisation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "compute_resource_descriptors",
    "enrich_catalog",
]

# Bytes per float64
_F64_BYTES: int = 8


def compute_resource_descriptors(
    n_orb: int,
    n_occ: int,
    n_vir: int,
) -> dict[str, Any]:
    """Return the nine resource-descriptor fields for a single fragment.

    Parameters
    ----------
    n_orb : int
        Number of spatial orbitals in the embedded cluster.
    n_occ : int
        Number of occupied spatial orbitals (closed-shell: n_alpha = n_beta = n_occ).
    n_vir : int
        Number of virtual spatial orbitals (n_vir = n_orb - n_occ).

    Returns
    -------
    dict with keys:
        n_alpha, n_beta, n_occ, n_vir,
        fci_dimension (Python int),
        log10_fci_dimension (float),
        fci_vector_bytes (Python int),
        ccsd_t2_elements (Python int),
        ccsd_t2_bytes (Python int),
        sqd_n_qubits (int)

    Raises
    ------
    ValueError
        If any argument is non-positive or inconsistent
        (n_occ + n_vir ≠ n_orb, or n_occ > n_orb).
    """
    n_orb = int(n_orb)
    n_occ = int(n_occ)
    n_vir = int(n_vir)

    if n_orb <= 0:
        raise ValueError(f"n_orb must be positive, got {n_orb}")
    if n_occ <= 0:
        raise ValueError(f"n_occ must be positive, got {n_occ}")
    if n_vir < 0:
        raise ValueError(f"n_vir must be non-negative, got {n_vir}")
    if n_occ + n_vir != n_orb:
        raise ValueError(
            f"Inconsistent orbital counts: n_occ={n_occ} + n_vir={n_vir} "
            f"≠ n_orb={n_orb}"
        )
    if n_occ > n_orb:
        raise ValueError(f"n_occ={n_occ} > n_orb={n_orb}")

    # Closed-shell spin occupations
    n_alpha: int = n_occ
    n_beta:  int = n_occ

    # FCI dimension — exact Python int (no overflow possible)
    alpha_dim: int = math.comb(n_orb, n_alpha)
    fci_dimension: int = alpha_dim * alpha_dim   # comb(n_orb, n_occ)^2

    # log10 of the exact integer
    log10_fci_dimension: float = math.log10(fci_dimension)

    # Single state-vector storage
    fci_vector_bytes: int = fci_dimension * _F64_BYTES

    # CCSD doubles tensor — spatial orbital, PySCF/Vayesta convention
    # t2[i, j, a, b]  i,j ∈ [0, n_occ)  a,b ∈ [0, n_vir)
    ccsd_t2_elements: int = n_occ * n_occ * n_vir * n_vir
    ccsd_t2_bytes:    int = ccsd_t2_elements * _F64_BYTES

    # Jordan–Wigner qubit count (unreduced, one qubit per spin-orbital)
    sqd_n_qubits: int = 2 * n_orb

    return {
        "n_alpha":             n_alpha,
        "n_beta":              n_beta,
        "n_occ":               n_occ,
        "n_vir":               n_vir,
        "fci_dimension":       fci_dimension,
        "log10_fci_dimension": log10_fci_dimension,
        "fci_vector_bytes":    fci_vector_bytes,
        "ccsd_t2_elements":    ccsd_t2_elements,
        "ccsd_t2_bytes":       ccsd_t2_bytes,
        "sqd_n_qubits":        sqd_n_qubits,
    }


# Column names written to the catalog (order matters for display)
RESOURCE_COLUMNS: list[str] = [
    "n_alpha",
    "n_beta",
    "n_occ",
    "n_vir",
    "fci_dimension",
    "log10_fci_dimension",
    "fci_vector_bytes",
    "ccsd_t2_elements",
    "ccsd_t2_bytes",
    "sqd_n_qubits",
]


def enrich_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Add resource-descriptor columns to *df* and return it.

    The DataFrame must contain ``n_orb``, ``n_occ``, and ``n_vir``.
    Rows where any of those three are NaN are left with NaN resource
    columns so the merge never silently drops fragment rows.

    Parameters
    ----------
    df : pd.DataFrame
        Fragment catalog with at least ``n_orb``, ``n_occ``, ``n_vir``.

    Returns
    -------
    The same DataFrame with RESOURCE_COLUMNS added (or replaced if they
    already exist).  The DataFrame is modified in place AND returned.
    """
    required = {"n_orb", "n_occ", "n_vir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"enrich_catalog: missing columns {sorted(missing)} in DataFrame"
        )

    # Read input orbital-count columns before adding any new columns,
    # so that initialising RESOURCE_COLUMNS with NaN does not overwrite
    # n_occ / n_vir (which are in RESOURCE_COLUMNS as pass-through values).
    _n_orb = df["n_orb"].tolist()
    _n_occ = df["n_occ"].tolist()
    _n_vir = df["n_vir"].tolist()

    # Initialise all resource columns with NaN (rows with missing orbital data
    # keep NaN so the merge never silently drops fragment rows).
    for col in RESOURCE_COLUMNS:
        df[col] = np.nan

    for i, idx in enumerate(df.index):
        try:
            n_orb = _n_orb[i]
            n_occ = _n_occ[i]
            n_vir = _n_vir[i]
            # NaN check (handles float NaN, None, and numpy NaN)
            if any(v is None or (isinstance(v, float) and v != v)
                   for v in (n_orb, n_occ, n_vir)):
                continue
            d = compute_resource_descriptors(int(n_orb), int(n_occ), int(n_vir))
            for col in RESOURCE_COLUMNS:
                df.at[idx, col] = d[col]
        except (ValueError, TypeError):
            pass   # leave NaN for this row; never break catalog loading

    # Restore integer dtype for integer columns (pandas stores int as float
    # when mixed with NaN unless we use Int64 nullable type)
    _int_cols = [
        "n_alpha", "n_beta", "n_occ", "n_vir",
        "fci_dimension", "fci_vector_bytes",
        "ccsd_t2_elements", "ccsd_t2_bytes", "sqd_n_qubits",
    ]
    for col in _int_cols:
        if col in df.columns and df[col].notna().all():
            try:
                df[col] = df[col].astype("Int64")
            except Exception:
                pass

    return df
