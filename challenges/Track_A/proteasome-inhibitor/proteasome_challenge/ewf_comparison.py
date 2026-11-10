"""
proteasome_challenge.ewf_comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Organizer-facing comparison of the classical EWF baseline with a hybrid
EWF-TrimSQD result.  Pure post-processing of two saved JSON artifacts:
no chemistry, no job submission, no checkpoint is written or modified.

Quantities
----------
* **Solver energy** (``energy_ha``): total energy of the embedded fragment as
  returned by its solver.  Comparable between CCSD and TrimSQD only for the
  *same* embedded Hamiltonian (verified through the Hamiltonian hash).
* **EWF contribution** (``e_corr`` in the hybrid result, ``e_corr_cumulant_ha``
  in the baseline): the fragment's term of the cumulant energy
  ``E_corr = sum_F (e1_pc + e22_pc)`` built by ``step4_reconstruct`` from the
  fragment's *own* RDMs and projector.  This is the additive quantity.
* **Raw cluster correlation energy** (``e_corr_ha``): NOT additive over
  fragments (it double counts the embedding), and for SQD / TrimSQD it is not
  reported by the solver; the recorded ``0.0`` is a placeholder default and is
  never used here.

Reconciliation
--------------
Because ``E_corr`` is a sum of per-fragment terms, 
``dE_total - sum_F dContribution_F`` must vanish up to floating-point error
(``E_HF`` is identical in both runs).  The residual is reported, never
redistributed.  The tolerance is the project convention
``classical_baseline.CONSISTENCY_TOL`` (1e-6 Ha); summation of 42 terms of
magnitude ~1e-2 Ha carries rounding of ~1e-14 Ha.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .classical_baseline import CONSISTENCY_TOL

REPORT_SCHEMA = "ewf-comparison/1"
COMPARISON_SCHEMA = "ewf-fragment-comparison/1"
QUANTUM_SOLVERS = ("SQD", "TrimSQD")
QUANTUM_SOLVER = "TrimSQD"                    # the only quantum solver this comparison supports
CLASSICAL_SOLVERS = ("FCI", "CCSD")
FROZEN_QUANTUM_PANEL = (1, 2, 3, 4, 6, 8, 9, 13, 17, 19, 22)   # n_orb >= 20, frozen in Stage D
EXPECTED_BASELINE_COUNTS = {"CCSD": 23, "FCI": 19}
EXPECTED_HYBRID_COUNTS = {"CCSD": 12, "FCI": 19, "TrimSQD": 11}

QUANTUM_CONVERGENCE_UNVERIFIED = (
    "unverified: SQD/TrimSQD report no convergence flag; the recorded True is a pipeline default"
)
CLASSICAL_RDM_NOT_RECORDED = "not recorded in the classical baseline record"

SOLVER_ECORR_UNAVAILABLE = (
    "unavailable: SQD/TrimSQD report no correlation energy; the recorded 0.0 "
    "is a default placeholder, not a physical value"
)

ROW_COLUMNS = (
    "fragment_id", "n_orb", "n_elec", "hamiltonian_hash",
    "classical_solver", "classical_energy_ha",
    "quantum_solver", "quantum_energy_ha", "solver_energy_diff_ha",
    "classical_ewf_contribution_ha", "quantum_ewf_contribution_ha",
    "ewf_contribution_diff_ha",
    "classical_solver_e_corr_ha", "quantum_solver_e_corr_ha", "quantum_solver_e_corr_status",
    "classical_rdm_integrated", "rdm_integrated", "rdm_validation",
    "converged_recorded", "quantum_convergence_status",
    "hamiltonian_sha256", "measurement_sha256", "sqd_config_hash", "seed", "qpu_job_id",
)


class ComparisonError(ValueError):
    """The two artifacts are not comparable."""


class IncompleteComparisonError(ComparisonError):
    """Strict mode: the hybrid result is not the complete 11-fragment experiment."""


def _by_id(mapping: Mapping, fid: int):
    if int(fid) in mapping:
        return mapping[int(fid)]
    return mapping.get(str(int(fid)))


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _canon(name: str) -> str:
    return str(name).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Per-fragment extraction (Task 2)
# ─────────────────────────────────────────────────────────────────────────────

def _need(rec: Mapping, key: str, fid: int, where: str):
    """Return rec[key] or raise: a missing value is never replaced by a default."""
    v = rec.get(key) if isinstance(rec, Mapping) else None
    if v is None:
        raise ComparisonError(f"fragment {fid}: {where} lacks required field '{key}'")
    return v


def compare_fragments(
    classical: Mapping[str, Any],
    hybrid: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    mode: str = "partial",
    tol: float = CONSISTENCY_TOL,
) -> dict[str, Any]:
    """Extract the per-fragment comparison of a classical EWF record and a hybrid EWF record.

    Parameters
    ----------
    classical : the validated baseline record (``classical-baseline/1``).
    hybrid    : a ``{ligand}_full_ewf_sqd.json`` style full-EWF result.
    manifest  : optional frozen final-selection manifest.  When given, its
                ``manifest_digest`` must equal the hybrid's ``final_selection_digest``
                and its ``selected_fragment_ids`` define the panel.  Without it
                the frozen 11-fragment panel ``FROZEN_QUANTUM_PANEL`` is used.
    mode      : ``"partial"`` (default) returns an explicitly labelled partial
                comparison for inspection; ``"complete"`` additionally requires the
                full 11-fragment TrimSQD experiment and raises
                ``IncompleteComparisonError`` otherwise.

    Solver energies come from ``fragment_summaries[f].energy_ha``; EWF
    contributions from ``fragments[f].e_corr`` (hybrid) and
    ``e_corr_cumulant_ha`` (classical).  The solver ``e_corr_ha`` of a quantum
    fragment is a placeholder default and is never used.  ``hamiltonian_hash`` is
    the only Hamiltonian identity compared; the SQD-side ``hamiltonian_sha256``
    uses a different hashing scheme and is carried as provenance only.

    Raises ``ComparisonError`` on any incompatibility or missing datum.
    """
    if mode not in ("partial", "complete"):
        raise ValueError("mode must be 'partial' or 'complete'")

    # ── molecular input / reference calculation ─────────────────────────────
    ident = classical.get("identity") or {}
    if hybrid.get("baseline_cache_key") != ident.get("cache_key") or ident.get("cache_key") is None:
        raise ComparisonError("hybrid baseline_cache_key does not match the baseline identity cache_key")
    if manifest is not None and hybrid.get("final_selection_digest") != manifest.get("manifest_digest"):
        raise ComparisonError("hybrid final_selection_digest does not match the frozen selection manifest")
    e_hf_c = (classical.get("energies") or {}).get("e_hf")
    if not (_finite(e_hf_c) and _finite(hybrid.get("e_hf"))):
        raise ComparisonError("e_hf missing or non-finite")
    if abs(float(e_hf_c) - float(hybrid["e_hf"])) > tol:
        raise ComparisonError("E_HF differs between baseline and hybrid: different molecular input")

    # ── fragment sets ───────────────────────────────────────────────────────
    base_frags = {int(f["fragment_id"]): f for f in classical.get("fragments") or []}
    summaries = hybrid.get("fragment_summaries") or {}
    contribs = hybrid.get("fragments") or {}
    ids = set(base_frags)
    if not ids or ids != {int(k) for k in summaries} or ids != {int(k) for k in contribs}:
        raise ComparisonError("fragment sets of baseline and hybrid differ")

    # ── valid baseline solver assignments ───────────────────────────────────
    base_solver = {fid: _canon(f.get("solver")) for fid, f in base_frags.items()}
    bad = sorted(fid for fid, s in base_solver.items() if s not in CLASSICAL_SOLVERS)
    if bad:
        raise ComparisonError(f"baseline fragments {bad} are not classical (FCI/CCSD)")
    base_counts = {k: sum(1 for s in base_solver.values() if s == k) for k in sorted(set(base_solver.values()))}
    if base_counts != EXPECTED_BASELINE_COUNTS:
        raise ComparisonError(f"baseline composition {base_counts} is not the frozen {EXPECTED_BASELINE_COUNTS}")

    # ── Hamiltonian identity (backend scheme only), orbital counts ──────────
    base_h = {int(k): v for k, v in (ident.get("hamiltonians") or {}).items()}
    for fid, f in base_frags.items():
        s, c = _by_id(summaries, fid), _by_id(contribs, fid)
        hc = _need(f, "hamiltonian_hash", fid, "baseline")
        if hc != _need(s, "hamiltonian_hash", fid, "hybrid fragment_summaries"):
            raise ComparisonError(f"fragment {fid}: Hamiltonian hash differs between baseline and hybrid")
        if base_h and base_h.get(fid) != hc:
            raise ComparisonError(f"fragment {fid}: baseline fragment hash differs from its identity map")
        n = int(_need(f, "n_orb", fid, "baseline"))
        if n != int(_need(s, "n_orb", fid, "hybrid fragment_summaries")) or n != int(
                _need(c, "n_orb", fid, "hybrid fragments")):
            raise ComparisonError(f"fragment {fid}: orbital count differs")

    # ── actual (not merely requested) hybrid solver identities ──────────────
    actual = {fid: _canon(_need(_by_id(summaries, fid), "solver", fid, "hybrid fragment_summaries")) for fid in ids}
    reconstructed = {fid: _canon(_need(_by_id(contribs, fid), "solver", fid, "hybrid fragments")) for fid in ids}
    disagree = sorted(fid for fid in ids if actual[fid] != reconstructed[fid])
    if disagree:
        raise ComparisonError(f"solver recorded by step 3 and step 4 differ for fragments {disagree}")
    unknown = sorted(fid for fid, s in actual.items() if s not in CLASSICAL_SOLVERS + QUANTUM_SOLVERS)
    if unknown:
        raise ComparisonError(f"unrecognised hybrid solver for fragments {unknown}")
    quantum_ids = sorted(fid for fid, s in actual.items() if s in QUANTUM_SOLVERS)
    not_trim = [fid for fid in quantum_ids if actual[fid] != QUANTUM_SOLVER]
    if not_trim:
        raise ComparisonError(f"fragments {not_trim} were solved by SQD; this comparison is for {QUANTUM_SOLVER}")
    declared = sorted(int(x) for x in hybrid.get("quantum_fragment_ids") or [])
    if declared != quantum_ids:
        raise ComparisonError(
            f"declared quantum_fragment_ids {declared} disagree with solvers actually recorded {quantum_ids}")
    requested = hybrid.get("solver_assignments_used") or {}
    mismatched = [fid for fid in ids
                  if _by_id(requested, fid) is not None and _canon(_by_id(requested, fid)) != actual[fid]]
    if mismatched:
        raise ComparisonError(f"requested and recorded solver differ for fragments {mismatched}")
    changed = sorted(fid for fid in ids if fid not in quantum_ids and actual[fid] != base_solver[fid])
    if changed:
        raise ComparisonError(f"classical fragments changed solver without quantum treatment: {changed}")

    panel = sorted(int(x) for x in manifest["selected_fragment_ids"]) if manifest else list(FROZEN_QUANTUM_PANEL)
    outside = sorted(set(quantum_ids) - set(panel))
    if outside:
        raise ComparisonError(f"quantum fragments {outside} are outside the frozen panel {panel}")
    not_ccsd = sorted(fid for fid in quantum_ids if base_solver[fid] != "CCSD")
    if not_ccsd:
        raise ComparisonError(f"quantum fragments {not_ccsd} were not CCSD fragments in the baseline")

    # ── extraction ──────────────────────────────────────────────────────────
    rows = []
    for fid in quantum_ids:
        f, s, c = base_frags[fid], _by_id(summaries, fid), _by_id(contribs, fid)
        n_orb = int(f["n_orb"])
        n_occ = int(_need(s, "n_occ", fid, "hybrid fragment_summaries"))
        if not 0 < n_occ <= n_orb:
            raise ComparisonError(f"fragment {fid}: n_occ={n_occ} is inconsistent with n_orb={n_orb}")
        for label, v in (("classical energy_ha", f.get("energy_ha")), ("hybrid energy_ha", s.get("energy_ha")),
                         ("classical EWF contribution", f.get("e_corr_cumulant_ha")),
                         ("hybrid EWF contribution", c.get("e_corr"))):
            if not _finite(v):
                raise ComparisonError(f"fragment {fid}: {label} missing or non-finite")
        if c.get("rdm_integrated") is not True:
            raise ComparisonError(
                f"fragment {fid}: rdm_integrated is not True, so fragments[{fid}].e_corr is the "
                "step-4 fallback to the solver placeholder, not a cumulant contribution")
        e_c, e_q = float(f["energy_ha"]), float(s["energy_ha"])
        k_c, k_q = float(f["e_corr_cumulant_ha"]), float(c["e_corr"])
        rows.append({
            "fragment_id": fid,
            "n_orb": n_orb,
            "n_elec": 2 * n_occ,
            "hamiltonian_hash": f["hamiltonian_hash"],
            "classical_solver": base_solver[fid],
            "classical_energy_ha": e_c,
            "quantum_solver": actual[fid],
            "quantum_energy_ha": e_q,
            "solver_energy_diff_ha": e_q - e_c,
            "classical_ewf_contribution_ha": k_c,
            "quantum_ewf_contribution_ha": k_q,
            "ewf_contribution_diff_ha": k_q - k_c,
            "classical_solver_e_corr_ha": f.get("e_corr_ha"),
            "quantum_solver_e_corr_ha": None,
            "quantum_solver_e_corr_status": SOLVER_ECORR_UNAVAILABLE,
            "classical_rdm_integrated": None,
            "rdm_integrated": True,
            "rdm_validation": "integrated; flag only (no RDM hash / trace / N-electron check stored)",
            "converged_recorded": s.get("converged"),
            "quantum_convergence_status": QUANTUM_CONVERGENCE_UNVERIFIED,
            "hamiltonian_sha256": s.get("hamiltonian_sha256"),
            "measurement_sha256": s.get("measurement_sha256"),
            "sqd_config_hash": s.get("sqd_config_hash"),
            "seed": s.get("seed"),
            "qpu_job_id": s.get("qpu_job_id"),
        })

    counts = {k: sum(1 for s in actual.values() if s == k) for k in sorted(set(actual.values()))}
    missing = sorted(set(panel) - set(quantum_ids))
    complete = (not missing and sorted(panel) == list(FROZEN_QUANTUM_PANEL)
                and counts == EXPECTED_HYBRID_COUNTS and len(ids) == 42)
    composition = {
        "baseline_counts": base_counts,
        "hybrid_counts": counts,
        "quantum_fragment_ids": quantum_ids,
        "baseline_solvers_of_quantum_fragments": sorted({base_solver[f] for f in quantum_ids}),
        "frozen_panel": panel,
        "panel_complete": not missing,
        "panel_missing": missing,
        "panel_extra": outside,
        "is_19_12_11": counts == EXPECTED_HYBRID_COUNTS and len(ids) == 42,
        "status": "complete" if complete else f"partial ({len(quantum_ids)} of {len(panel)})",
    }
    out = {
        "schema": COMPARISON_SCHEMA,
        "mode": mode,
        "complete": complete,
        "label": ("COMPLETE 11-fragment EWF-TrimSQD comparison" if complete else
                  f"PARTIAL comparison: {len(quantum_ids)} of {len(panel)} panel fragments; "
                  f"missing {missing}; NOT a valid complete experiment"),
        "ligand": classical.get("ligand"),
        "baseline_cache_key": ident.get("cache_key"),
        "final_selection_digest": hybrid.get("final_selection_digest"),
        "n_fragments": len(ids),
        "composition": composition,
        "rows": rows,
        "notes": [
            "Solver energies: fragment_summaries[f].energy_ha. EWF contributions: fragments[f].e_corr "
            "(hybrid) and e_corr_cumulant_ha (classical); raw cluster energies are not additive.",
            SOLVER_ECORR_UNAVAILABLE,
            QUANTUM_CONVERGENCE_UNVERIFIED,
            f"classical RDM integration: {CLASSICAL_RDM_NOT_RECORDED}",
            "hamiltonian_hash (backend scheme) is the identity compared; hamiltonian_sha256 is provenance only.",
        ],
    }
    if mode == "complete":
        require_complete(out)
    return out


def require_complete(comparison: Mapping[str, Any]) -> Mapping[str, Any]:
    """Strict gate: pass only for the complete 19 FCI / 12 CCSD / 11 TrimSQD experiment."""
    if comparison.get("schema") != COMPARISON_SCHEMA:
        raise IncompleteComparisonError("not an ewf-fragment-comparison record")
    comp = comparison.get("composition") or {}
    problems = []
    if comp.get("panel_missing"):
        problems.append(f"panel fragments without a TrimSQD result: {comp['panel_missing']}")
    if list(comp.get("frozen_panel") or []) != list(FROZEN_QUANTUM_PANEL):
        problems.append(f"panel is not the frozen {list(FROZEN_QUANTUM_PANEL)}")
    if comp.get("hybrid_counts") != EXPECTED_HYBRID_COUNTS:
        problems.append(f"composition {comp.get('hybrid_counts')} is not {EXPECTED_HYBRID_COUNTS}")
    if comp.get("baseline_counts") != EXPECTED_BASELINE_COUNTS:
        problems.append(f"baseline composition {comp.get('baseline_counts')} is not {EXPECTED_BASELINE_COUNTS}")
    if [r["fragment_id"] for r in comparison.get("rows") or []] != list(FROZEN_QUANTUM_PANEL):
        problems.append("rows do not cover exactly the frozen panel")
    if not comparison.get("complete"):
        problems.append("comparison is labelled partial")
    if problems:
        raise IncompleteComparisonError("; ".join(problems))
    return comparison


# ─────────────────────────────────────────────────────────────────────────────
# Report construction
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    baseline: Mapping[str, Any],
    hybrid: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
    *,
    tol: float = CONSISTENCY_TOL,
) -> dict[str, Any]:
    """Per-fragment comparison (``compare_fragments``) plus molecular-energy totals.

    Numerical disagreements (reconciliation residual, drift of unchanged
    fragments) are reported as flags, never raised and never smoothed over.
    """
    cmp_ = compare_fragments(baseline, hybrid, manifest=manifest, mode="partial", tol=tol)
    base_frags = {int(f["fragment_id"]): f for f in baseline["fragments"]}
    contribs = hybrid["fragments"]
    quantum_ids = cmp_["composition"]["quantum_fragment_ids"]
    composition = cmp_["composition"]

    # ── per-fragment contribution shifts, all fragments ─────────────────────
    deltas: dict[int, float] = {}
    for fid, f in base_frags.items():
        cb = f.get("e_corr_cumulant_ha")
        cq = _by_id(contribs, fid).get("e_corr")
        if not (_finite(cb) and _finite(cq)):
            raise ComparisonError(f"fragment {fid}: EWF contribution missing or non-finite")
        deltas[fid] = float(cq) - float(cb)

    sum_delta = math.fsum(deltas.values())
    sum_abs = math.fsum(abs(v) for v in deltas.values())
    d_total = float(hybrid["e_total"]) - float(baseline["energies"]["e_total"])
    residual = d_total - sum_delta
    unchanged_dev = max((abs(deltas[f]) for f in deltas if f not in quantum_ids), default=0.0)
    internal = {
        "baseline_sum_contributions_minus_e_corr": math.fsum(
            float(f["e_corr_cumulant_ha"]) for f in base_frags.values()) - float(baseline["energies"]["e_corr"]),
        "hybrid_sum_contributions_minus_e_corr": math.fsum(
            float(_by_id(contribs, f)["e_corr"]) for f in base_frags) - float(hybrid["e_corr"]),
        "baseline_e_total_minus_hf_plus_corr": float(baseline["energies"]["e_total"])
            - float(baseline["energies"]["e_hf"]) - float(baseline["energies"]["e_corr"]),
        "hybrid_e_total_minus_hf_plus_corr": float(hybrid["e_total"]) - float(hybrid["e_hf"]) - float(hybrid["e_corr"]),
    }

    totals = {
        "e_total_classical_ha": float(baseline["energies"]["e_total"]),
        "e_total_hybrid_ha": float(hybrid["e_total"]),
        "e_hf_ha": float(hybrid["e_hf"]),
        "net_shift_ha": d_total,
        "sum_contribution_shifts_ha": sum_delta,
        "sum_abs_contribution_shifts_ha": sum_abs,
        "sum_abs_quantum_contribution_shifts_ha": math.fsum(abs(deltas[f]) for f in quantum_ids),
        "reconciliation_residual_ha": residual,
        "tolerance_ha": float(tol),
        "additive_within_tolerance": abs(residual) <= tol,
        "max_unchanged_fragment_contribution_shift_ha": unchanged_dev,
        "unchanged_fragments_within_tolerance": unchanged_dev <= tol,
        "internal_consistency_ha": internal,
        "internal_consistency_ok": all(abs(v) <= tol for v in internal.values()),
    }

    return {
        "schema": REPORT_SCHEMA,
        "ligand": cmp_["ligand"],
        "baseline_cache_key": cmp_["baseline_cache_key"],
        "final_selection_digest": cmp_["final_selection_digest"],
        "n_fragments": cmp_["n_fragments"],
        "label": cmp_["label"],
        "composition": composition,
        "totals": totals,
        "rows": cmp_["rows"],
        "notes": cmp_["notes"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Loading and export
# ─────────────────────────────────────────────────────────────────────────────

def compare_from_files(baseline_path, hybrid_path, manifest_path=None, *, tol: float = CONSISTENCY_TOL) -> dict:
    """Read the three saved artifacts (read-only) and build the report."""
    load = lambda p: json.loads(Path(p).read_text())
    manifest = load(manifest_path) if manifest_path else None
    return build_report(load(baseline_path), load(hybrid_path), manifest, tol=tol)


def write_csv(report: Mapping[str, Any], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_COLUMNS)
        w.writeheader()
        for r in report["rows"]:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in ROW_COLUMNS})
    return path


def write_json(report: Mapping[str, Any], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figures (validated report only; nothing is filled in)
# ─────────────────────────────────────────────────────────────────────────────

def _bar(report, key, title, ylabel, color, path=None):
    import matplotlib
    if path is not None:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    rows = [r for r in report["rows"] if _finite(r.get(key))]
    skipped = len(report["rows"]) - len(rows)
    if not rows:
        raise ComparisonError(f"no finite values for {key}; nothing to plot")
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(rows) + 2.0), 3.8))
    xs = range(len(rows))
    ax.bar(xs, [1e3 * float(r[key]) for r in rows], color=color)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"F{r['fragment_id']}" for r in rows])
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_ylabel(ylabel)
    comp = report["composition"]
    ax.set_title(f"{title} [{comp['status']}]")
    if skipped:
        ax.annotate(f"{skipped} fragment(s) without a value omitted", xy=(0.01, 0.97),
                    xycoords="axes fraction", va="top", fontsize=8)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def plot_solver_energy_differences(report, path=None):
    """Bar chart of E_solver(TrimSQD) - E_solver(CCSD) per quantum fragment."""
    from ._plot_style import ORANGE
    return _bar(report, "solver_energy_diff_ha",
                "Standalone solver energy: TrimSQD - classical",
                "E(TrimSQD) - E(classical)  [mHa]", ORANGE, path)


def plot_ewf_contribution_differences(report, path=None):
    """Bar chart of the reconstructed EWF contribution shift per quantum fragment."""
    from ._plot_style import BLUE
    return _bar(report, "ewf_contribution_diff_ha",
                "Reconstructed EWF contribution shift: hybrid - classical",
                "dE contribution  [mHa]", BLUE, path)


def write_report(report: Mapping[str, Any], outdir) -> dict[str, Path]:
    """Write CSV, JSON and both figures under ``outdir`` (never under checkpoints/)."""
    outdir = Path(outdir)
    if "checkpoints" in outdir.resolve().parts:
        raise ComparisonError("refusing to write report files inside a checkpoints directory")
    import matplotlib.pyplot as plt
    out = {
        "csv": write_csv(report, outdir / "ewf_comparison_fragments.csv"),
        "json": write_json(report, outdir / "ewf_comparison_summary.json"),
    }
    for name, fn in (("solver_png", plot_solver_energy_differences),
                     ("contribution_png", plot_ewf_contribution_differences)):
        p = outdir / ("ewf_solver_energy_diff.png" if name == "solver_png" else "ewf_contribution_diff.png")
        fig = fn(report, p)
        plt.close(fig)
        out[name] = p
    return out


def format_totals(report: Mapping[str, Any]) -> str:
    t, c = report["totals"], report["composition"]
    return "\n".join([
        f"composition      : {c['hybrid_counts']}  ({c['status']}; 19/12/11 = {c['is_19_12_11']})",
        f"E_total classical: {t['e_total_classical_ha']:.10f} Ha",
        f"E_total hybrid   : {t['e_total_hybrid_ha']:.10f} Ha",
        f"net shift        : {t['net_shift_ha']:+.10f} Ha",
        f"sum |shifts|     : {t['sum_abs_contribution_shifts_ha']:.10f} Ha",
        f"residual         : {t['reconciliation_residual_ha']:+.3e} Ha (tol {t['tolerance_ha']:.0e}; "
        f"additive={t['additive_within_tolerance']})",
    ])


if __name__ == "__main__":  # organizer CLI: python -m proteasome_challenge.ewf_comparison
    import argparse
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--ligand", default="ixazomib")
    ap.add_argument("--out", default=str(root / "reports"))
    a = ap.parse_args()
    s3 = root / "checkpoints" / "stage3"
    rep = compare_from_files(s3 / f"{a.ligand}_classical_baseline.json",
                             s3 / f"{a.ligand}_full_ewf_sqd.json",
                             root / "checkpoints" / "final_experiment_selection.json")
    paths = write_report(rep, a.out)
    print(format_totals(rep))
    for k, v in paths.items():
        print(f"{k}: {v}")
