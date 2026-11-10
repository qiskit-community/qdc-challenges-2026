"""Load the stored reference traces as :class:`FloquetResult` objects.

* :func:`load_results` reads a trace CSV such as
  ``traces_paper_colored_21q_v1.csv`` (columns: n_qubits, n_z2, cycle,
  magnetization, magnetization_z2), one result per system size.
* :func:`load_paper_51q` reads the 51-qubit experiment digitized from the
  paper's Fig. 3 (``paper_fig3_51q.csv``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .simulator import FloquetParams, FloquetResult

HERE = Path(__file__).parent


def load_results(csv_path: Path) -> list[FloquetResult]:
    """Read the traces CSV into one FloquetResult per system size."""
    rows = np.genfromtxt(csv_path, delimiter=",", names=True)
    has_z2 = "magnetization_z2" in rows.dtype.names
    results = []
    for n in np.unique(rows["n_qubits"]):
        sel = rows["n_qubits"] == n
        cycles = rows["cycle"][sel]
        mags = rows["magnetization"][sel]
        order = np.argsort(cycles)
        if has_z2:
            mags_z2 = rows["magnetization_z2"][sel][order]
            n_z2 = int(rows["n_z2"][sel][0]) if "n_z2" in rows.dtype.names else 0
        else:  # legacy CSV without the coordination-two column
            mags_z2 = np.full(order.size, np.nan)
            n_z2 = 0
        results.append(
            FloquetResult(
                n_qubits=int(n),
                cycles=cycles[order].astype(int),
                magnetization=mags[order],
                magnetization_z2=mags_z2,
                n_z2=n_z2,
                params=FloquetParams(),
                device="csv",
            )
        )
    return sorted(results, key=lambda r: r.n_qubits)


def load_paper_51q(csv_path: Path | None = None) -> tuple[FloquetResult, float]:
    """Load the 51q experimental trace digitized from the paper's Fig. 3.

    Returns ``(result, amplitude)`` where ``result`` is a :class:`FloquetResult`
    (``device='paper-51q'``) holding the digitized full ``M`` (panel a) and
    coordination-two ``M_{z=2}`` (panel b) with ``NaN`` at cycles the paper does not
    plot, and ``amplitude`` is the digitized panel-(c) fitted amplitude at ``N=51``.
    These values are **approximate** (see :mod:`reference.digitize_paper_51q`).
    """
    csv_path = csv_path or HERE / "paper_fig3_51q.csv"

    # Parse by hand: the header comments contain commas (which trip genfromtxt),
    # and empty cells mark cycles the paper does not plot -> NaN.
    def cell(s):
        return float(s) if s.strip() else np.nan

    cyc, m_col, mz2_col = [], [], []
    amp = 0.0128                        # panel-c amplitude, from header comment
    for line in csv_path.read_text().splitlines():
        if line.startswith("#"):
            if "amplitude at N=51" in line:
                amp = float(line.rstrip().rstrip(".").split("=")[-1])
            continue
        if line.startswith("cycle") or not line.strip():
            continue
        c, m, v = (line.split(",") + ["", ""])[:3]
        cyc.append(int(c))
        m_col.append(cell(m))
        mz2_col.append(cell(v))
    order = np.argsort(cyc)
    result = FloquetResult(
        n_qubits=51,
        cycles=np.array(cyc)[order].astype(int),
        magnetization=np.array(m_col)[order],
        magnetization_z2=np.array(mz2_col)[order],
        n_z2=41,                       # Fig. A1 graph: 56 bonds, ten degree-three sites
        params=FloquetParams(),
        device="paper-51q",
    )
    return result, amp

