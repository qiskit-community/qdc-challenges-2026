"""
proteasome_challenge.diagnostic_analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stage C: CCSD-informed fragment assessment.

Pure table / plotting helpers.  Nothing in this module submits a job, imports a
chemistry package, or fabricates a value: every function consumes

* a *saved, validated* classical-baseline record
  (:func:`proteasome_challenge.classical_baseline.validate_baseline_record`), and
* the fragment catalog (embedded MP2 characterization),

and returns tables or figures.  When CCSD diagnostics are unavailable the plot
functions draw an explicit *missing-results* panel instead of any placeholder data.

Conventions
-----------
* MP2 doubles amplitudes (``mp2_max_abs_t2`` / ``mp2_t2_rms``) are the antisymmetrized
  same-spin amplitudes ``<ij||ab>/D``.  CCSD amplitudes are restricted spin-adapted
  alpha-beta amplitudes.  Analysis A therefore compares MP2 only with the
  ``*_mp2_convention`` CCSD columns (``t2_ss = t2 - t2.swapaxes(2, 3)``), never with the
  raw alpha-beta ``ccsd_max_abs_t2``.
* T1 and D1 are CCSD-amplitude diagnostics; they are *not* derived from MP2.
* Published thresholds (T1 > 0.02 etc.) are never applied here as solver-routing
  rules.  Population summaries use descriptive, user-adjustable parameters only.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from proteasome_challenge import classical_baseline as cb

MISSING_RESULTS_TEXT = "CCSD diagnostics not available"

CATALOG_COLUMNS = [
    "fragment_id", "center_element", "chemical_environment",
    "n_orb", "n_occ", "n_vir", "n_elec",
    "mp2_max_abs_t2", "mp2_t2_rms", "hf_gap_ha",
    "fci_dimension", "log10_fci_dimension", "sqd_n_qubits",
]

NUMERIC_DIAGNOSTIC_COLUMNS = [
    "ccsd_t1_diagnostic", "ccsd_d1_diagnostic", "ccsd_max_abs_t1",
    "ccsd_max_abs_t2", "ccsd_t2_rms",
    "ccsd_max_abs_t2_mp2_convention", "ccsd_t2_rms_mp2_convention",
    "ccsd_correlation_energy",
]

#: Column order for the notebook's fragment-resolved table.
DISPLAY_COLUMNS = [
    "fragment_id", "center_element", "chemical_environment",
    "n_orb", "n_elec",
    "mp2_max_abs_t2", "mp2_t2_rms",
    "ccsd_t1_diagnostic", "ccsd_d1_diagnostic",
    "ccsd_max_abs_t1", "ccsd_max_abs_t2",
    "ccsd_converged", "classical_solver", "diagnostics_status",
    "log10_fci_dimension", "sqd_n_qubits",
]


# ─────────────────────────────────────────────────────────────────────────────
# Table
# ─────────────────────────────────────────────────────────────────────────────

def merged_table(baseline: Mapping[str, Any], catalog: pd.DataFrame, ligand: str) -> pd.DataFrame:
    """Join the MP2 catalog with the baseline's CCSD diagnostics on ``fragment_id``.

    Unavailable / not-applicable diagnostics are ``NaN`` (never 0).  A fragment present
    in the baseline but absent from the catalog (or vice versa) is a provenance error.
    """
    cat = catalog.copy()
    if "ligand" in cat.columns:
        cat = cat[cat["ligand"] == ligand]
    cols = [c for c in CATALOG_COLUMNS if c in cat.columns]
    missing_cols = {"fragment_id", "n_orb", "n_elec"} - set(cols)
    if missing_cols:
        raise ValueError(f"catalog lacks required columns {sorted(missing_cols)}")
    cat = cat[cols].copy()
    cat["fragment_id"] = cat["fragment_id"].astype(int)

    rows = pd.DataFrame(cb.diagnostics_rows(baseline))
    cat_ids, base_ids = set(cat["fragment_id"]), set(rows["fragment_id"])
    if cat_ids != base_ids:
        raise ValueError(
            "baseline and catalog fragment sets differ: "
            f"only in baseline {sorted(base_ids - cat_ids)}, only in catalog {sorted(cat_ids - base_ids)}")

    table = cat.merge(rows, on="fragment_id", how="inner", validate="one_to_one")
    for c in NUMERIC_DIAGNOSTIC_COLUMNS:
        table[c] = pd.to_numeric(table[c], errors="coerce")      # None -> NaN, never 0
    for c in ("mp2_max_abs_t2", "mp2_t2_rms"):
        if c in table.columns:
            table[c] = pd.to_numeric(table[c], errors="coerce")
    return table.sort_values("fragment_id").reset_index(drop=True)


def display_table(table: pd.DataFrame) -> pd.DataFrame:
    """Subset / order the merged table for display."""
    return table[[c for c in DISPLAY_COLUMNS if c in table.columns]]


def available_mask(table: pd.DataFrame) -> pd.Series:
    return table["diagnostics_status"] == "available"


# ─────────────────────────────────────────────────────────────────────────────
# Analyses A-D (tables)
# ─────────────────────────────────────────────────────────────────────────────

def doubles_response(table: pd.DataFrame) -> pd.DataFrame:
    """Analysis A: MP2 vs CCSD maximum doubles amplitude in the *same* convention.

    ``ratio`` = max|t2_CCSD (same-spin)| / max|t2_MP2|.  Rows lacking either value are
    dropped, not zero-filled.
    """
    m = available_mask(table) & table["mp2_max_abs_t2"].notna() \
        & table["ccsd_max_abs_t2_mp2_convention"].notna()
    out = table.loc[m, ["fragment_id", "center_element", "n_orb",
                        "mp2_max_abs_t2", "ccsd_max_abs_t2_mp2_convention"]].copy()
    out = out.rename(columns={"ccsd_max_abs_t2_mp2_convention": "ccsd_max_abs_t2_same_spin"})
    out["ratio_ccsd_over_mp2"] = out["ccsd_max_abs_t2_same_spin"] / out["mp2_max_abs_t2"]
    return out.reset_index(drop=True)


def rank_comparison(table: pd.DataFrame, *, top_k: int = 5) -> pd.DataFrame:
    """Analysis D: rank each fragment by MP2 hardness, CCSD T1, D1 and CCSD max doubles.

    Ranks (1 = smallest, average ranks for ties) are computed over fragments for which
    *all* four quantities exist, so the rankings are directly comparable.  The result lists
    rank differences; large differences are scientifically interesting disagreements
    between *different* diagnostics, not numerical failures.
    """
    need = ["mp2_max_abs_t2", "ccsd_t1_diagnostic", "ccsd_d1_diagnostic",
            "ccsd_max_abs_t2_mp2_convention"]
    m = available_mask(table) & table[need].notna().all(axis=1)
    sub = table.loc[m, ["fragment_id", "center_element", "n_orb"] + need].reset_index(drop=True)
    if sub.empty:
        return sub.assign(rank_mp2=[], rank_t1=[], rank_d1=[], rank_ccsd_t2=[],
                          max_rank_spread=[], in_top_k_of=[])
    sub["rank_mp2"] = cb.ranks(sub["mp2_max_abs_t2"])
    sub["rank_t1"] = cb.ranks(sub["ccsd_t1_diagnostic"])
    sub["rank_d1"] = cb.ranks(sub["ccsd_d1_diagnostic"])
    sub["rank_ccsd_t2"] = cb.ranks(sub["ccsd_max_abs_t2_mp2_convention"])
    rk = sub[["rank_mp2", "rank_t1", "rank_d1", "rank_ccsd_t2"]]
    sub["max_rank_spread"] = rk.max(axis=1) - rk.min(axis=1)
    n = len(sub)
    k = max(1, min(int(top_k), n))
    cutoff = n - k + 1                                   # top-k largest values have rank >= cutoff
    names = {"rank_mp2": "MP2", "rank_t1": "T1", "rank_d1": "D1", "rank_ccsd_t2": "CCSD-t2"}
    sub["in_top_k_of"] = [
        ", ".join(names[c] for c in names if sub.at[i, c] >= cutoff) or "-"
        for i in range(n)
    ]
    return sub.sort_values("max_rank_spread", ascending=False).reset_index(drop=True)


def rank_correlations(table: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation (Pearson on average ranks) between MP2 hardness and CCSD quantities."""
    cmp = rank_comparison(table)
    pairs = [("rank_mp2", "rank_t1"), ("rank_mp2", "rank_d1"),
             ("rank_mp2", "rank_ccsd_t2"), ("rank_t1", "rank_d1")]
    rows = []
    for a, b in pairs:
        rho = float("nan")
        if len(cmp) >= 3 and cmp[a].std() > 0 and cmp[b].std() > 0:
            rho = float(np.corrcoef(cmp[a], cmp[b])[0, 1])
        rows.append({"pair": f"{a[5:]} vs {b[5:]}", "n_fragments": len(cmp), "spearman_rho": rho})
    return pd.DataFrame(rows)


def population_summary(
    table: pd.DataFrame, *, large_n_orb: int = 20, mp2_split: float | None = None,
    carbon_element: str = "C",
) -> pd.DataFrame:
    """Analysis C: re-examine the apparent MP2/FCI-dimension populations with CCSD diagnostics.

    Groups are *descriptive*, defined by user-adjustable parameters (``large_n_orb``,
    ``mp2_split``; default split = median MP2 maximum amplitude of the large fragments),
    and are not classes of strong or weak correlation.  Groups:

    * large, high-MP2, carbon-centred
    * large, high-MP2, other centre
    * large, low-MP2
    * smaller (n_orb < large_n_orb)

    Per group: number of fragments, MP2 and CCSD medians/maxima over the fragments for
    which CCSD diagnostics exist (``n_with_diagnostics``).
    """
    t = table.copy()
    large = t["n_orb"] >= large_n_orb
    if mp2_split is None:
        vals = t.loc[large, "mp2_max_abs_t2"].dropna()
        mp2_split = float(vals.median()) if len(vals) else float("nan")
    high = t["mp2_max_abs_t2"] > mp2_split
    carbon = t["center_element"].astype(str) == carbon_element

    def label(i: int) -> str:
        if not large.iloc[i]:
            return f"smaller (n_orb < {large_n_orb})"
        if not high.iloc[i]:
            return "large, low-MP2"
        return "large, high-MP2, carbon-centred" if carbon.iloc[i] else "large, high-MP2, other centre"

    t["population"] = [label(i) for i in range(len(t))]
    rows = []
    for name, g in t.groupby("population", sort=True):
        av = g[available_mask(g)]
        rows.append({
            "population": name,
            "n_fragments": len(g),
            "fragment_ids": ", ".join(str(i) for i in g["fragment_id"]),
            "n_with_diagnostics": len(av),
            "median_mp2_max_abs_t2": g["mp2_max_abs_t2"].median(),
            "median_ccsd_t1": av["ccsd_t1_diagnostic"].median() if len(av) else float("nan"),
            "max_ccsd_t1": av["ccsd_t1_diagnostic"].max() if len(av) else float("nan"),
            "median_ccsd_d1": av["ccsd_d1_diagnostic"].median() if len(av) else float("nan"),
            "max_ccsd_d1": av["ccsd_d1_diagnostic"].max() if len(av) else float("nan"),
            "mp2_split_used": mp2_split,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def missing_results_figure(message: str, title: str = ""):
    """An explicit empty panel stating why no data are shown (no synthetic data)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 2.4))
    ax.axis("off")
    ax.text(0.5, 0.62, MISSING_RESULTS_TEXT, ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.30, message[:300], ha="center", va="center", fontsize=9, wrap=True, color="#555555")
    if title:
        ax.set_title(title)
    return fig


def _need(table: pd.DataFrame | None, columns: list[str], title: str):
    """Return (data, None) or (None, missing_figure)."""
    if table is None or len(table) == 0:
        return None, missing_results_figure("The baseline has not been loaded.", title)
    m = available_mask(table)
    for c in columns:
        m &= table[c].notna()
    if not m.any():
        return None, missing_results_figure(
            "No CCSD-treated fragment has these diagnostics available in the saved baseline.", title)
    return table[m], None


def plot_doubles_response(table: pd.DataFrame | None):
    """Analysis A scatter: max|t2| MP2 vs CCSD (same-spin convention), log-log."""
    import matplotlib.pyplot as plt
    from proteasome_challenge import _plot_style as ps
    title = "A. MP2 vs CCSD maximum doubles amplitude (same-spin convention)"
    d, miss = _need(table, ["mp2_max_abs_t2", "ccsd_max_abs_t2_mp2_convention"], title)
    if miss is not None:
        return miss
    fig, ax = ps.subplots()
    ax.scatter(d["mp2_max_abs_t2"], d["ccsd_max_abs_t2_mp2_convention"], c=ps.BLUE, s=36)
    lo = float(min(d["mp2_max_abs_t2"].min(), d["ccsd_max_abs_t2_mp2_convention"].min()))
    hi = float(max(d["mp2_max_abs_t2"].max(), d["ccsd_max_abs_t2_mp2_convention"].max()))
    ax.plot([lo, hi], [lo, hi], color=ps.GRAY, lw=1, ls="--", label="equal")
    for _, r in d.iterrows():
        ax.annotate(str(int(r["fragment_id"])), (r["mp2_max_abs_t2"], r["ccsd_max_abs_t2_mp2_convention"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("max |t2|  MP2  (antisymmetrized)")
    ax.set_ylabel("max |t2|  CCSD (same-spin, t2 − t2ᵀ)")
    ax.set_title(title); ax.legend()
    return fig


def plot_mp2_vs_t1_d1(table: pd.DataFrame | None):
    """Analysis B: MP2 hardness against CCSD T1 and D1 (two panels).

    No threshold lines are drawn: published values are context-dependent and are not
    solver-routing rules.
    """
    import matplotlib.pyplot as plt
    from proteasome_challenge import _plot_style as ps
    title = "B. MP2 hardness vs CCSD T1 / D1"
    d, miss = _need(table, ["mp2_max_abs_t2", "ccsd_t1_diagnostic", "ccsd_d1_diagnostic"], title)
    if miss is not None:
        return miss
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, col, lab in zip(axes, ["ccsd_t1_diagnostic", "ccsd_d1_diagnostic"], ["T1", "D1"]):
        ax.scatter(d["mp2_max_abs_t2"], d[col], c=ps.BLUE, s=36)
        for _, r in d.iterrows():
            ax.annotate(str(int(r["fragment_id"])), (r["mp2_max_abs_t2"], r[col]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("max |t2|  MP2"); ax.set_ylabel(f"CCSD {lab}")
        ax.set_title(f"MP2 hardness vs {lab}")
    fig.suptitle(title, fontsize=11)
    return fig


def plot_populations(table: pd.DataFrame | None, *, color_by: str = "ccsd_t1_diagnostic"):
    """Analysis C: the original MP2 / FCI-dimension map, coloured by a CCSD diagnostic.

    Fragments without the diagnostic are drawn as hollow grey markers (not as zero).
    """
    import matplotlib.pyplot as plt
    from proteasome_challenge import _plot_style as ps
    title = "C. Exploratory MP2 populations viewed through CCSD diagnostics"
    if table is None or len(table) == 0:
        return missing_results_figure("The baseline has not been loaded.", title)
    if not available_mask(table).any():
        return missing_results_figure(
            "No CCSD diagnostics in the saved baseline; the MP2-only map is shown in Stage A.", title)
    fig, ax = ps.subplots()
    have = table[color_by].notna() & available_mask(table)
    no = table[~have]
    yes = table[have]
    ax.scatter(no["log10_fci_dimension"], no["mp2_max_abs_t2"], facecolors="none",
               edgecolors=ps.GRAY, s=36, label="no CCSD diagnostic (FCI-treated)")
    sc = ax.scatter(yes["log10_fci_dimension"], yes["mp2_max_abs_t2"], c=yes[color_by],
                    cmap="viridis", s=48, edgecolors="black", linewidths=0.4)
    fig.colorbar(sc, ax=ax, label=color_by.replace("ccsd_", "CCSD ").replace("_diagnostic", ""))
    for _, r in table.iterrows():
        ax.annotate(str(int(r["fragment_id"])), (r["log10_fci_dimension"], r["mp2_max_abs_t2"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("log10 FCI dimension"); ax.set_ylabel("max |t2|  MP2")
    ax.set_title(title); ax.legend(loc="best")
    return fig


def plot_rank_disagreement(table: pd.DataFrame | None, *, top_k: int = 5):
    """Analysis D: rank of MP2 hardness against rank of T1 and of D1."""
    import matplotlib.pyplot as plt
    from proteasome_challenge import _plot_style as ps
    title = "D. Diagnostic rankings: MP2 vs T1 / D1"
    cmp = rank_comparison(table, top_k=top_k) if table is not None and len(table) else None
    if cmp is None or cmp.empty:
        return missing_results_figure("Rankings need MP2, T1, D1 and CCSD doubles for the same fragments.", title)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    n = len(cmp)
    for ax, col, lab in zip(axes, ["rank_t1", "rank_d1"], ["T1", "D1"]):
        ax.scatter(cmp["rank_mp2"], cmp[col], c=ps.BLUE, s=36)
        ax.plot([1, n], [1, n], color=ps.GRAY, lw=1, ls="--")
        for _, r in cmp.iterrows():
            ax.annotate(str(int(r["fragment_id"])), (r["rank_mp2"], r[col]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("rank by MP2 max |t2|"); ax.set_ylabel(f"rank by CCSD {lab}")
    fig.suptitle(title, fontsize=11)
    return fig


def is_finite(x: Any) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))
