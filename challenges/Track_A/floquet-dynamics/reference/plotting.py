"""Plotting helpers that reproduce the paper's key figures.

* :func:`plot_dynamics`        -> Fig. 3a style: M vs. Floquet cycle, prethermal line.
* :func:`plot_finite_size`     -> Fig. 3b style: M vs. cycle for several sizes.
* :func:`plot_amplitude_scaling` -> Fig. 3c style: amplitude vs. N with the fit.
* :func:`plot_lattice`         -> the heavy-hex patch, bonds coloured by class.
* :func:`figure3`              -> the combined Fig. 3 (the challenge's final result).

All functions accept an optional ``ax`` and return it, so they compose into a
multi-panel figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .analysis import (
    ScalingFit,
    fit_amplitude_scaling,
    fit_late_cycle,
    fit_late_cycle_anchored,
    prethermal_value,
)
from .lattice import HeavyHexLattice
from .simulator import FloquetResult

_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

# The paper reports two magnetizations (App. B, p19): the full-lattice
# ``M = (1/N) sum_q <Z_q>`` and the *coordination-two* ``M_{z=2}``, averaged over
# only the degree-2 qubits. Fig. 3a shows M for the 51q system; the finite-size
# scaling in Fig. 3b/c is done on ``M_{z=2}``. Select with ``observable=``.
_OBSERVABLES = {
    "M": ("magnetization", r"$M = \frac{1}{N}\sum_q \langle Z_q\rangle$"),
    "Mz2": ("magnetization_z2", r"$M_{z=2}$"),
}


def _series(result: FloquetResult, observable: str) -> np.ndarray:
    attr, _ = _OBSERVABLES[observable]
    return getattr(result, attr)


def _amplitude_samples(results, observable, window):
    """Keep identifiable late-cycle fits together with their size and uncertainty."""
    sizes, amplitudes, errors = [], [], []
    for result in results:
        fitted = fit_late_cycle(result.cycles, _series(result, observable), window=window)
        if not fitted.valid:
            continue
        variance = fitted.pcov[2, 2]
        if not np.isfinite(variance) or variance < 0:
            continue
        sizes.append(result.n_qubits)
        amplitudes.append(fitted.amplitude)
        errors.append(np.sqrt(variance))
    return tuple(np.asarray(values, dtype=float) for values in (sizes, amplitudes, errors))


def plot_dynamics(result: FloquetResult, ax=None, label=None, color=None,
                  show_prethermal=True, observable="M"):
    """Magnetization vs. Floquet cycle for a single system (paper Fig. 3a).

    ``observable`` selects the full-lattice ``M`` (default, as in Fig. 3a) or the
    coordination-two ``M_{z=2}``. The x-axis starts at the first driven cycle
    ``n = 1`` (the paper's convention); ``n = 0`` is the trivial initial state and
    is not shown.
    """
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    y = _series(result, observable)
    lbl = label if label is not None else f"{result.n_qubits} qubits"
    m = result.cycles >= 1
    ax.plot(
        result.cycles[m], y[m],
        marker="o", ms=4, lw=1.5, color=color, label=lbl,
    )
    if show_prethermal:
        pv = prethermal_value(y)
        ax.axhline(pv, ls="--", lw=1, color="0.5", label=f"prethermal value ~ {pv:.3f}")
    ax.set_xlabel("Floquet cycle $n$")
    ax.set_ylabel("Magnetization  " + _OBSERVABLES[observable][1])
    ax.set_title("Prethermal Floquet dynamics")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    return ax


def plot_finite_size(results: list[FloquetResult], ax=None, observable="Mz2",
                     window=(15, 30), show_fit=True, ylim=(0.20, 0.40),
                     ytick=0.02):
    """Overlay magnetization vs. cycle for several system sizes (paper Fig. 3b).

    Defaults to ``M_{z=2}`` -- the coordination-two magnetization the paper uses
    for its finite-size scaling -- over the paper's ``window`` of Floquet cycles
    (Fig. 3b spans ``n = 15 .. 30``). With ``show_fit`` the Eq. 3 damped-cosine
    fit ``C e^{-gamma n} + A cos(2 pi n / T + phi)`` is drawn over each trace (a
    solid line), the same fit whose amplitude ``A`` scales in Fig. 3c.

    Axis scaling follows the paper's Fig. 3b: x is fixed to ``window`` with ticks
    every 3 cycles and y carries the paper's ``ytick`` (0.02) spacing. The y
    *range* defaults to ``(0.20, 0.40)`` -- the paper's tick scale, widened at the
    bottom because with the provisional lattice the absolute ``M_{z=2}`` sits ~0.1
    below the paper's 0.32-0.40 window (the offset is in the full ``M`` too, so it
    is a lattice-connectivity issue, not an ``M_{z=2}`` normalization one). Pass
    ``ylim=(0.32, 0.40)`` to force the paper's exact window once the geometry is confirmed, or ``ylim=None`` to autoscale.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator

    ax = ax or plt.subplots(figsize=(6, 4))[1]
    lo, hi = window
    grid = np.linspace(lo, hi, 200)
    ordered = sorted(results, key=lambda r: r.n_qubits)
    size_handles = []
    fitted_any = False
    unavailable_sizes = []
    for k, res in enumerate(ordered):
        color = _COLORS[k % len(_COLORS)]
        y = _series(res, observable)
        m = (res.cycles >= lo) & (res.cycles <= hi)
        ax.plot(res.cycles[m], y[m], marker="o", ms=4, ls="none", color=color)
        if show_fit:
            fit = fit_late_cycle(res.cycles, y, window=window)
            if fit.valid:
                mu = fit.model(grid)
                sd = fit.model_std(grid)
                ax.plot(grid, mu, "-", lw=1.3, color=color, alpha=0.9)
                ax.fill_between(grid, mu - sd, mu + sd, color=color, alpha=0.2, lw=0)
                fitted_any = True
            else:
                unavailable_sizes.append(res.n_qubits)
        size_handles.append(
            Line2D([], [], marker="o", ls="-", color=color, ms=5,
                   label=f"{res.n_qubits} qubits")
        )
    # Paper Fig. 3b axis scaling: ticks at the window bounds every 3 cycles, y
    # ticks every 0.02. Pad the x-limits by half a cycle so the boundary markers
    # (n = lo, hi) are not clipped by the spine.
    ax.set_xlim(lo - 0.6, hi + 0.6)
    ax.xaxis.set_major_locator(MultipleLocator(3))
    ax.yaxis.set_major_locator(MultipleLocator(ytick))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Floquet cycle $N_c$")
    ax.set_ylabel("Magnetization " + _OBSERVABLES[observable][1])
    ax.set_title("Finite-size dependence")
    ax.grid(alpha=0.3)

    # Two legends: system sizes (by colour), and what each glyph means. The band
    # is the +/-1 sigma uncertainty of the Eq. 3 fit, not measurement scatter
    # (the exact reference has none).
    size_leg = ax.legend(handles=size_handles, frameon=True, framealpha=0.95,
                         edgecolor="0.7", fontsize=9, ncol=2, loc="upper right")
    ax.add_artist(size_leg)
    if fitted_any:
        glyph_handles = [
            Line2D([], [], marker="o", ls="none", color="0.35",
                   label="exact " + _OBSERVABLES[observable][1]),
            Line2D([], [], ls="-", color="0.35", label="Eq. 3 fit"),
            Patch(facecolor="0.35", alpha=0.2, label=r"$\pm1\sigma$ fit band"),
        ]
        ax.legend(handles=glyph_handles, frameon=True, framealpha=0.95,
                  edgecolor="0.7", fontsize=8, loc="lower left")
    if unavailable_sizes:
        ax.text(0.02, 0.02, "fit unavailable: " + ", ".join(f"{n}q" for n in unavailable_sizes),
                transform=ax.transAxes, fontsize=8, color="0.5")
    return ax


def plot_amplitude_scaling(results: list[FloquetResult], ax=None,
                           fit: ScalingFit | None = None, observable="Mz2",
                           window=(15, 30), ytick=0.015):
    """Oscillation amplitude vs. system size with the ``a e^{-sqrt(N/xi)}+c`` fit
    (paper Fig. 3c). Defaults to the coordination-two ``M_{z=2}`` amplitude.

    The amplitude of each size is the ``|A|`` of the Eq. 3 late-cycle fit over
    ``window`` (paper: "extracted from the late-cycle fits in (b)"), and its error
    bar is the ``1 sigma`` on ``A`` from that fit's covariance (paper: "one-
    standard-deviation uncertainties from the weighted nonlinear least-squares
    fits"). y ticks follow the paper's 0.015 spacing.
    """
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    from matplotlib.ticker import MultipleLocator

    results = sorted(results, key=lambda r: r.n_qubits)
    sizes, amps, aerr = _amplitude_samples(results, observable, window)
    if sizes.size:
        ax.errorbar(sizes, amps, yerr=aerr, fmt="o", ms=6, color="#1f77b4",
                    capsize=3, lw=1, label="fitted amplitude $|A|$ (Eq. 3)")
    if sizes.size >= 3:
        fit = fit or fit_amplitude_scaling(sizes, amps)
        grid = np.linspace(sizes.min(), sizes.max(), 200)
        ax.plot(grid, fit.model(grid), "-", lw=1.5, color="#d62728",
                label=rf"scaling fit: $c={fit.c:.3f}$")
        ax.axhline(fit.c, ls=":", lw=1, color="0.5",
                   label="thermodynamic offset" if fit.survives_thermodynamic_limit else "offset ~ 0")
    else:
        ax.text(0.5, 0.5, "need >= 3 valid amplitudes\nfor scaling fit",
                ha="center", va="center", transform=ax.transAxes, color="0.5")
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(MultipleLocator(ytick))
    ax.set_xlabel("System size $N_q$")
    ax.set_ylabel("Oscillation amplitude $A$")
    ax.set_title("Finite-size scaling")
    ax.grid(alpha=0.3)
    if sizes.size:
        ax.legend(frameon=False, fontsize=9)
    return ax


def _ratio_on_common(ref: FloquetResult, hw: FloquetResult, attr: str,
                     window=None):
    """Reference/hardware ratio at the cycles both traces share.

    Cycles are integer-indexed, so we match by value (not position) -- the
    hardware run may cover fewer or different cycles. ``window`` (inclusive)
    restricts to a cycle range; division by ~0 hardware yields ``nan`` rather
    than blowing up.
    """
    rc = np.asarray(ref.cycles)
    hc = np.asarray(hw.cycles)
    common = np.intersect1d(rc, hc)
    if window is not None:
        lo, hi = window
        common = common[(common >= lo) & (common <= hi)]
    if common.size == 0:
        return common, common
    rmap = dict(zip(rc.tolist(), np.asarray(getattr(ref, attr), dtype=float)))
    hmap = dict(zip(hc.tolist(), np.asarray(getattr(hw, attr), dtype=float)))
    r = np.array([rmap[c] for c in common], dtype=float)
    h = np.array([hmap[c] for c in common], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(h) > 1e-9, r / h, np.nan)
    return common, ratio


def _empty_ratio_note(ax):
    ax.text(0.5, 0.5, "ratio appears after a\nhardware run is submitted",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=8, color="0.5")


_PAPER51_COLOR = "#c2185b"   # the paper's 51q QESEM-Extrapolated pink/crimson
# Size palette for the comparison figure: deliberately skips red, which is too
# close to the crimson 51q target (red is reused only for the panel-c fit line).
_CMP_PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#17becf", "#9467bd", "#8c564b"]
_PARTIAL_AMPLITUDES = {28: (0.021, 0.001), 35: (0.013, 0.001)}


def figure3_comparison(reference_results: list[FloquetResult],
                       hardware_results: list[FloquetResult] | None = None,
                       paper_51q: FloquetResult | None = None,
                       paper_51q_amplitude: float | None = None,
                       savepath: str | None = None, window=(15, 30),
                       *, _partial: bool = False):
    """Paper Fig. 3 (a, b, c) with the classical **reference**, the digitized
    **51q experiment** from the paper (the participant's target), and the
    participant's own **hardware** overlaid, with a target/hardware ratio strip
    beneath each panel.

    ``reference_results`` is a list of :class:`FloquetResult`, one per classical
    size (21/28/35). ``paper_51q`` is the 51q experiment digitized from the
    paper's Fig. 3 (see :func:`reference.make_figure3.load_paper_51q`); it is drawn
    as a distinct crimson target series in all three panels and is **approximate**
    (pixel-digitized). ``hardware_results`` may be ``None`` (before any run) or
    cover any subset of sizes; where a hardware size has a target -- our reference
    for 21/28/35, or the paper 51q for 51 -- the ratio strip shows
    ``target / hardware`` (closer to the ``1.0`` guide line is better -- this is
    what grading rewards). Sizes are colour-coded consistently across the panels;
    targets are lines / circles and hardware uses black-outlined markers:
    upward triangles for 28q, filled plus signs for 35q, diamonds for 51q,
    and squares for other sizes.

    * (a) full-lattice ``M`` vs. cycle, all sizes overlaid;
    * (b) coordination-two ``M_{z=2}`` over ``window`` with the Eq. 3 fit;
      Target markers are shown only at the cycles the hardware measured; target
      fit curves use the full trace. Hardware traces are fitted (dashed) with
      :func:`fit_late_cycle_anchored`, seeded by the matching target's fit.
    * (c) fitted oscillation amplitude ``|A|`` vs. size with the scaling fit.
      Reference error bars are formal fit errors; hardware error bars are
      bootstrap 1-sigma, and the hardware amplitudes get their own Eq. 4 fit.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    ref = sorted(reference_results, key=lambda r: r.n_qubits)
    hw = sorted(hardware_results or [], key=lambda r: r.n_qubits)
    sizes = [r.n_qubits for r in ref]
    color_sizes = sorted(set(sizes) | ({r.n_qubits for r in hw} if _partial else set()))
    if paper_51q is not None:   # 51q hardware shares the digitized experiment's crimson
        color_sizes = [n for n in color_sizes if n != paper_51q.n_qubits]
    cmap = {n: _CMP_PALETTE[i % len(_CMP_PALETTE)]
            for i, n in enumerate(color_sizes)}
    # Ratio targets by size: our statevector reference for 21/28/35, and the
    # digitized paper 51q for 51.
    target_by_size = {r.n_qubits: r for r in ref}
    if paper_51q is not None:
        target_by_size[paper_51q.n_qubits] = paper_51q

    def color_for(size):
        return cmap.get(size, _PAPER51_COLOR)

    def hardware_marker(size):
        return {28: "^", 35: "P", 51: "D"}.get(int(size), "s")

    # Hardware Eq. 3 fits: the sparse (even-cycle) traces are fitted from the
    # matching target's dense fit, with bootstrap error bars (panel b curves and
    # panel c amplitudes).
    # A size without its own trace target is anchored on the nearest-size one.
    hw_fits = {}
    anchors = {n: fit_late_cycle(t.cycles, _series(t, "Mz2"), window=window)
               for n, t in target_by_size.items()}
    anchors = {n: fit for n, fit in anchors.items() if fit.valid}
    for hwr in hw:
        anchor = (anchors[min(anchors, key=lambda n: abs(n - hwr.n_qubits))]
                  if anchors else None)
        y_hw = _series(hwr, "Mz2")
        hw_fits[hwr.n_qubits] = (
            fit_late_cycle_anchored(hwr.cycles, y_hw, anchor, window=window)
            if anchor is not None else fit_late_cycle(hwr.cycles, y_hw, window=window))
    # Panel (b) shows target markers only where hardware was measured.
    hw_cycles = (np.unique(np.concatenate([np.asarray(r.cycles) for r in hw]))
                 if hw else None)

    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.08, wspace=0.28)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[0, 2])
    # Ratio strips share each panel's x-axis so they sit directly beneath it.
    rxa = fig.add_subplot(gs[1, 0], sharex=axa)
    rxb = fig.add_subplot(gs[1, 1], sharex=axb)
    rxc = fig.add_subplot(gs[1, 2], sharex=axc)
    for main, ratio in ((axa, rxa), (axb, rxb), (axc, rxc)):
        main.tick_params(labelbottom=False)
        ratio.axhline(1.0, ls="--", lw=1, color="0.5")
        ratio.set_ylabel("target / hw", fontsize=9)
    ratio_devs = []   # |ratio - 1| across all panels -> one shared symmetric range

    def _overlay_hw(main_ax, ratio_ax, attr, win):
        """Draw hardware traces and ratios wherever targets are available."""
        drew = False
        for hwr in hw:
            tgt = target_by_size.get(hwr.n_qubits)
            if tgt is None and not _partial:
                continue
            c = color_for(hwr.n_qubits)
            y = np.asarray(getattr(hwr, attr), dtype=float)
            mh = hwr.cycles >= (win[0] if win else 1)
            if win:
                mh &= hwr.cycles <= win[1]
            mh &= np.isfinite(y)
            main_ax.plot(hwr.cycles[mh], y[mh], marker=hardware_marker(hwr.n_qubits),
                         ms=4, ls="none", mfc=c, mec="black", mew=0.7, color=c)
            if tgt is None:
                continue
            xr, yr = _ratio_on_common(tgt, hwr, attr, window=win)
            good = np.isfinite(yr)
            if win is None:
                good &= xr >= 1
            if np.any(good):
                ratio_ax.plot(xr[good], yr[good], marker=hardware_marker(hwr.n_qubits),
                              ms=3, ls="none", mfc=c, mec="black", mew=0.7, color=c)
                ratio_devs.append(np.abs(yr[good] - 1.0))
                drew = True
        return drew

    # -- (a) full magnetization M vs. cycle -------------------------------- #
    for r in ref:
        m = r.cycles >= 1
        axa.plot(r.cycles[m], r.magnetization[m], "-", lw=1.5,
                 color=cmap[r.n_qubits])
    if paper_51q is not None:
        y = np.asarray(paper_51q.magnetization, dtype=float)
        m = (paper_51q.cycles >= 1) & np.isfinite(y)
        axa.plot(paper_51q.cycles[m], y[m], marker="o", ms=4, lw=1.5,
                 color=_PAPER51_COLOR, mec="black", mew=0.7)
    drew_ratio_a = _overlay_hw(axa, rxa, "magnetization", None)
    axa.set_ylabel(r"$M = \frac{1}{N}\sum_q \langle Z_q\rangle$")
    axa.set_title("(a) Full magnetization")
    axa.grid(alpha=0.3)
    ya0, ya1 = axa.get_ylim()
    axa.set_ylim(ya0, ya1 + 0.28 * (ya1 - ya0))   # headroom for the legend
    rxa.set_xlabel("Floquet cycle $N_c$")
    if not drew_ratio_a:
        _empty_ratio_note(rxa)

    # -- (b) coordination-two M_{z=2} over the late-cycle window ----------- #
    lo, hi = window
    grid = np.linspace(lo, hi, 200)
    unavailable_fits = []

    def _mz2_markers_and_fit(res, color):
        y = _series(res, "Mz2")
        m = (res.cycles >= lo) & (res.cycles <= hi) & np.isfinite(y)
        if hw_cycles is not None:
            m &= np.isin(res.cycles, hw_cycles)
        axb.plot(res.cycles[m], y[m], marker="o", ms=4, ls="none", color=color,
                 mec="black", mew=0.7)
        fit = fit_late_cycle(res.cycles, y, window=window)
        if not fit.valid:
            unavailable_fits.append(f"{res.n_qubits}q")
            return
        mu, sd = fit.model(grid), fit.model_std(grid)
        axb.plot(grid, mu, "-", lw=1.3, color=color, alpha=0.9)
        axb.fill_between(grid, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)

    for r in ref:
        _mz2_markers_and_fit(r, cmap[r.n_qubits])
    if paper_51q is not None:
        _mz2_markers_and_fit(paper_51q, _PAPER51_COLOR)
    if unavailable_fits:
        axb.text(0.02, 0.02, "fit unavailable: " + ", ".join(unavailable_fits),
                 transform=axb.transAxes, fontsize=8, color="0.5")
    drew_ratio_b = _overlay_hw(axb, rxb, "magnetization_z2", window)
    for size, fit in hw_fits.items():
        if fit.valid:
            axb.plot(grid, fit.model(grid), "--", lw=1.1, color=color_for(size), alpha=0.9)
    axb.set_xlim(lo - 0.6, hi + 0.6)
    axb.xaxis.set_major_locator(MultipleLocator(3))
    axb.set_ylabel(r"Magnetization $M_{z=2}$")
    axb.set_title("(b) Coordination-two $M_{z=2}$")
    axb.grid(alpha=0.3)
    # Headroom at the top so the (paper-style) size legend clears the data.
    yb0, yb1 = axb.get_ylim()
    axb.set_ylim(yb0, yb1 + 0.45 * (yb1 - yb0))
    rxb.set_xlabel("Floquet cycle $N_c$")
    rxb.xaxis.set_major_locator(MultipleLocator(3))
    if not drew_ratio_b:
        _empty_ratio_note(rxb)

    # -- (c) oscillation amplitude vs. system size ------------------------- #
    sizes_arr, ref_amps, ref_err = _amplitude_samples(ref, "Mz2", window)
    if sizes_arr.size:
        axc.errorbar(sizes_arr, ref_amps, yerr=ref_err, fmt="o", ms=6,
                     color="0.2", mec="black", mew=0.7,
                     capsize=3, lw=1, label="reference $|A|$")
    if sizes_arr.size >= 3 and not _partial:
        sfit = fit_amplitude_scaling(sizes_arr, ref_amps)
        g = np.linspace(sizes_arr.min(), sizes_arr.max(), 200)
        axc.plot(g, sfit.model(g), "-", lw=1.5, color="#d62728",
                 label=rf"scaling fit: $c={sfit.c:.3f}$")
    # target amplitude by size: reference for 21/28/35, paper 51q for 51.
    target_amp_by_size = dict(zip(sizes_arr.astype(int), ref_amps))
    if _partial:
        extra = [(n, amp, se) for n, (amp, se) in _PARTIAL_AMPLITUDES.items()]
        axc.errorbar([item[0] for item in extra],
                     [item[1] for item in extra],
                     yerr=[item[2] for item in extra], fmt="o", ms=6,
                     color="0.2", mec="black", mew=0.7, capsize=3, lw=1,
                     label="28/35q amplitude targets")
        target_amp_by_size.update((n, amp) for n, amp, _ in extra)
    if (paper_51q is not None and paper_51q_amplitude is not None
            and np.isfinite(paper_51q_amplitude) and paper_51q_amplitude >= 0):
        axc.plot([paper_51q.n_qubits], [paper_51q_amplitude], marker="*", ms=13,
                 ls="none", color=_PAPER51_COLOR, label="paper 51q $|A|$ (digitized)")
        target_amp_by_size[paper_51q.n_qubits] = paper_51q_amplitude
    if _partial and len(target_amp_by_size) >= 3:
        target_sizes = np.array(sorted(target_amp_by_size), dtype=float)
        target_amps = np.array([target_amp_by_size[int(n)] for n in target_sizes])
        sfit = fit_amplitude_scaling(target_sizes, target_amps)
        grid_sizes = np.linspace(target_sizes.min(), target_sizes.max(), 200)
        axc.plot(grid_sizes, sfit.model(grid_sizes), "-", lw=1.5,
                 color="#d62728", label=rf"target scaling fit: $c={sfit.c:.3f}$")
    hw_amp_by_size = {n: fit.amplitude for n, fit in hw_fits.items()
                      if fit.valid and np.isfinite(fit.amplitude_err)}
    hw_amp_err_by_size = {n: hw_fits[n].amplitude_err for n in hw_amp_by_size}
    if hw_amp_by_size:
        hs = np.array(sorted(hw_amp_by_size), dtype=float)
        ha = np.array([hw_amp_by_size[int(s)] for s in hs])
        he = np.array([hw_amp_err_by_size[int(s)] for s in hs])
        for i, s in enumerate(hs):
            axc.errorbar(
                [s], [ha[i]], yerr=[he[i]],
                fmt=hardware_marker(s), ms=7, ls="none",
                mfc="#1f77b4", mec="black", mew=0.7, color="#1f77b4",
                capsize=3,
                label="hardware $|A|$ (bootstrap 1$\\sigma$)" if i == 0 else "_nolegend_",
            )
        if hs.size >= 3:
            hfit = fit_amplitude_scaling(hs, ha, sigma=np.maximum(he, 1e-4))
            grid_sizes = np.linspace(hs.min(), hs.max(), 200)
            axc.plot(grid_sizes, hfit.model(grid_sizes), "--", lw=1.5,
                     color="#1f77b4", label=rf"hardware scaling fit: $c={hfit.c:.3f}$")
        cr = [s for s in hs if int(s) in target_amp_by_size]
        if cr:
            cr = np.array(cr, dtype=float)
            rat = np.array([target_amp_by_size[int(s)] / hw_amp_by_size[int(s)]
                            if hw_amp_by_size[int(s)] else np.nan for s in cr])
            for s, value in zip(cr, rat):
                rxc.plot(
                    [s], [value], marker=hardware_marker(s), ms=5, ls="none",
                    mfc="0.2", mec="black", mew=0.7, color="0.2",
                )
            ratio_devs.append(np.abs(rat - 1.0))
        else:
            _empty_ratio_note(rxc)
    else:
        _empty_ratio_note(rxc)
    axc.set_ylim(bottom=0.0)
    axc.set_ylim(0.0, axc.get_ylim()[1] * 1.30)   # headroom for the legend
    axc.set_ylabel("Oscillation amplitude $|A|$")
    axc.set_title("(c) Amplitude scaling")
    axc.grid(alpha=0.3)
    if axc.get_legend_handles_labels()[0]:
        axc.legend(frameon=True, framealpha=0.95, edgecolor="0.7", fontsize=8,
                   loc="upper right")
    else:
        axc.text(0.5, 0.5, "amplitudes unavailable", ha="center", va="center",
                 transform=axc.transAxes, color="0.5")
    rxc.set_xlabel("System size $N_q$")

    # One paper-style legend on (b): classical sizes, the digitized 51q target,
    # and a glyph for each hardware size.
    # (The colours carry across all three panels, so (a) needs no second copy.)
    legend_handles = [
        Line2D([], [], color=cmap[n], lw=2, marker="o", ms=5,
               mec="black", mew=0.7, label=f"{n}q sim.")
        for n in sizes
    ]
    if paper_51q is not None:
        legend_handles.append(
            Line2D([], [], color=_PAPER51_COLOR, lw=2, marker="o", ms=5,
                   mec="black", mew=0.7,
                   label=f"{paper_51q.n_qubits}q exp. (digitized)")
        )
    legend_handles.extend(
        Line2D(
            [], [], color=color_for(n), marker=hardware_marker(n), ls="none",
            mfc=color_for(n), mec="black", mew=0.7, ms=6,
            label=f"{n}q hardware",
        )
        for n in sorted({r.n_qubits for r in hw})
    )
    for ax in (axa, axb):
        handles = legend_handles
        if ax is axb and any(fit.valid for fit in hw_fits.values()):
            handles = legend_handles + [Line2D([], [], ls="--", color="0.35",
                                               label="hardware Eq. 3 fit")]
        ax.legend(handles=handles, frameon=True, framealpha=0.95,
                  edgecolor="0.7", fontsize=8, loc="upper right", ncol=2)

    # One shared symmetric y-range for all three ratio strips, centred on 1.0
    # (perfect agreement) with equal room above and below, so the panels are
    # directly comparable. Falls back to +/-10% when there is no hardware yet.
    finite = np.concatenate([d[np.isfinite(d)] for d in ratio_devs]) \
        if ratio_devs else np.array([])
    dev = float(finite.max()) * 1.15 if finite.size else 0.10
    dev = max(dev, 1e-3)
    for ratio in (rxa, rxb, rxc):
        ratio.set_ylim(1.0 - dev, 1.0 + dev)

    title = ("Fig. 3 - partial reference, amplitude targets & hardware" if _partial
             else "Fig. 3 - classical reference & paper 51q target vs. hardware")
    fig.suptitle(title, fontsize=13)
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def figure3_comparison_partial(reference_results: list[FloquetResult],
                               hardware_results: list[FloquetResult] | None = None,
                               paper_51q: FloquetResult | None = None,
                               paper_51q_amplitude: float | None = None,
                               savepath: str | None = None, window=(15, 30)):
    """Figure 3 with all hardware sizes and reported 28/35q amplitude targets.

    Pass the available trace references (typically only the 21q result). The
    28/35q hardware traces appear in panels (a) and (b), while their supplied
    amplitude targets, 0.021 +/- 0.001 and 0.013 +/- 0.001, appear in (c).
    Ratio strips use only sizes with a matching target.
    """
    return figure3_comparison(
        reference_results, hardware_results=hardware_results,
        paper_51q=paper_51q, paper_51q_amplitude=paper_51q_amplitude,
        savepath=savepath, window=window, _partial=True,
    )


def plot_lattice(lattice: HeavyHexLattice, ax=None):
    """Draw the heavy-hex patch with bonds coloured by their edge-colour class."""
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    palette = ["#1f77b4", "#d62728", "#2ca02c"]
    pos = lattice.positions
    for (i, j), c in zip(lattice.edges, lattice.edge_colors):
        xs = [pos[i][0], pos[j][0]]
        ys = [pos[i][1], pos[j][1]]
        ax.plot(xs, ys, "-", lw=2.5, color=palette[c % 3], zorder=1)
    xs = [pos[q][0] for q in range(lattice.n_qubits)]
    ys = [pos[q][1] for q in range(lattice.n_qubits)]
    ax.scatter(xs, ys, s=90, color="0.15", zorder=2)
    ax.set_title(f"Heavy-hex patch: {lattice.n_qubits} qubits, {lattice.n_edges} bonds")
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def figure3(results: list[FloquetResult], savepath: str | None = None):
    """Assemble the paper's Fig. 3 -- the challenge's final classical result.

    Panels: (a) dynamics for the largest size, (b) finite-size overlay,
    (c) amplitude scaling with the thermodynamic-limit fit.
    """
    results = sorted(results, key=lambda r: r.n_qubits)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    # (a) full-lattice M for the largest system (paper Fig. 3a);
    # (b, c) coordination-two M_{z=2} finite-size scaling (paper Fig. 3b/c).
    plot_dynamics(results[-1], ax=axes[0], observable="M")
    plot_finite_size(results, ax=axes[1], observable="Mz2")
    if len(results) >= 3:
        plot_amplitude_scaling(results, ax=axes[2], observable="Mz2")
    else:
        axes[2].text(0.5, 0.5, "need >= 3 sizes\nfor scaling fit",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].axis("off")
    fig.suptitle("Fig. 3 - Error-free classical reference (statevector)", fontsize=13)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
