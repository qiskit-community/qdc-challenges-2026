"""Post-processing: subharmonic period, oscillation amplitude, finite-size fit.

These turn raw ``M(n)`` traces into the quantities the paper reports: the
period-~4 subharmonic response and the oscillation amplitude vs. system size
(paper Fig. 3b/c), including the ``A(N_q) = a exp(-sqrt(N_q/xi)) + c`` fit whose
offset ``c`` decides whether the signal survives the thermodynamic limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dominant_period(
    magnetization: np.ndarray, skip: int = 4, window: int = 8
) -> float:
    """Estimate the oscillation period (in Floquet cycles) of the late-time signal.

    The prethermal plateau also *decays* slowly, and that envelope dominates a
    naive FFT (it looks like one very long "period"). We first subtract a moving
    average of length ``window`` to kill the envelope, trim the convolution edges,
    then take the FFT of what remains -- so the reported period is the genuine
    subharmonic oscillation (the paper's response peaks near period ~4).
    """
    tail = np.asarray(magnetization[skip:], dtype=float)
    if tail.size < 2 * window:
        return float("nan")
    resid = tail - _moving_mean(tail, window)
    half = window // 2
    resid = resid[half:-half] if half else resid          # drop edge artifacts
    resid = resid - resid.mean()
    spec = np.abs(np.fft.rfft(resid)) ** 2
    freqs = np.fft.rfftfreq(resid.size, d=1.0)
    spec[0] = 0.0                                          # ignore DC
    k = int(np.argmax(spec))
    if freqs[k] == 0.0:
        return float("nan")
    return float(1.0 / freqs[k])


def oscillation_amplitude(
    magnetization: np.ndarray, skip: int = 6, window: int = 8
) -> float:
    """Late-time oscillation amplitude = half the peak-to-peak of the detrended tail.

    Detrended exactly as :func:`dominant_period` (subtract the slow envelope, then
    drop the convolution edges) so the value reflects the steady subharmonic
    oscillation on the plateau, not the initial transient or edge artifacts. This
    is the quantity fit vs. system size in Fig. 3c.
    """
    tail = np.asarray(magnetization[skip:], dtype=float)
    if tail.size < 2 * window:
        return float("nan")
    resid = tail - _moving_mean(tail, window)
    half = window // 2
    resid = resid[half:-half] if half else resid
    return float(0.5 * (resid.max() - resid.min()))


def prethermal_value(magnetization: np.ndarray, skip: int = 6) -> float:
    """Mean magnetization over the late-time (prethermal) window."""
    tail = np.asarray(magnetization[skip:], dtype=float)
    return float(tail.mean()) if tail.size else float("nan")


@dataclass
class LateCycleFit:
    """Fit of the paper's Eq. 3 to the late-cycle ``M_{z=2}`` trace:

    ``M_{z=2}(N_c) = C e^{-gamma N_c} + A cos(2 pi N_c / T + phi)``

    -- a slowly-decaying prethermal plateau (``C``, ``gamma``) plus a subharmonic
    oscillation (amplitude ``A``, period ``T``, phase ``phi``). ``A`` is the
    quantity plotted vs. system size in Fig. 3c.
    """

    C: float
    gamma: float
    A: float
    T: float
    phi: float
    window: tuple[int, int]      # (first, last) cycle used in the fit, inclusive
    cycles: np.ndarray           # the in-window cycle indices actually fitted
    values: np.ndarray           # the in-window data actually fitted
    pcov: np.ndarray             # 5x5 parameter covariance (order C, gamma, A, T, phi)
    valid: bool = True
    failure_reason: str | None = None
    boot: np.ndarray | None = None   # (n_boot, 5) bootstrap parameter draws, if any

    def model(self, n):
        n = np.asarray(n, dtype=float)
        return self.C * np.exp(-self.gamma * n) + self.A * np.cos(
            2.0 * np.pi * n / self.T + self.phi
        )

    def _jacobian(self, n):
        """Rows are d(model)/d(param) at each ``n`` (param order C, gamma, A, T, phi)."""
        n = np.asarray(n, dtype=float)
        e = np.exp(-self.gamma * n)
        theta = 2.0 * np.pi * n / self.T + self.phi
        cos, sin = np.cos(theta), np.sin(theta)
        return np.stack([
            e,                                          # d/dC
            -n * self.C * e,                            # d/dgamma
            cos,                                        # d/dA
            self.A * sin * (2.0 * np.pi * n / self.T ** 2),  # d/dT
            -self.A * sin,                              # d/dphi
        ], axis=-1)

    def model_std(self, n):
        """1-sigma standard deviation of the fitted model at ``n``, propagated from
        the parameter covariance ``pcov`` via ``sqrt(J Sigma J^T)``.

        This is the *fit* uncertainty (how tightly Eq. 3's five parameters are
        pinned by the exact trace), not measurement scatter -- the reference has
        none. Bootstrapped fits use the spread of the bootstrap curves instead of
        the linearized covariance. An unavailable fit or covariance produces
        ``nan`` rather than a misleading zero-width band."""
        n = np.asarray(n, dtype=float)
        if self.valid and self.boot is not None and len(self.boot):
            C, gamma, A, T, phi = (self.boot[:, k, None] for k in range(5))
            curves = C * np.exp(-gamma * n.ravel()) + A * np.cos(
                2.0 * np.pi * n.ravel() / T + phi)
            return np.std(curves, axis=0).reshape(n.shape)
        cov = np.array(self.pcov, dtype=float)
        if not self.valid or cov.shape != (5, 5) or not np.all(np.isfinite(cov)):
            return np.full_like(n, np.nan, dtype=float)
        J = self._jacobian(n)                       # (..., 5)
        var = np.einsum("...i,ij,...j->...", J, cov, J)
        return np.sqrt(np.clip(var, 0.0, None))

    @property
    def amplitude(self) -> float:
        return abs(self.A) if self.valid else float("nan")

    @property
    def amplitude_err(self) -> float:
        """1-sigma uncertainty on ``|A|``: half the 16-84 % bootstrap interval when
        bootstrapped (robust to the heavy tails of sparse fits), otherwise the
        formal ``sqrt(pcov[A, A])``."""
        if not self.valid:
            return float("nan")
        if self.boot is not None and len(self.boot):
            lo, hi = np.percentile(np.abs(self.boot[:, 2]), [16.0, 84.0])
            return float(0.5 * (hi - lo))
        variance = float(self.pcov[2, 2])
        return float(np.sqrt(variance)) if np.isfinite(variance) and variance >= 0 else float("nan")


def fit_late_cycle(
    cycles: np.ndarray,
    magnetization: np.ndarray,
    window: tuple[int, int] = (15, 30),
) -> LateCycleFit:
    """Fit Eq. 3 to the ``M_{z=2}`` trace over ``window`` (inclusive), returning a
    :class:`LateCycleFit`.

    The default window ``(15, 30)`` is the late-cycle range shown in the paper's
    Fig. 3b (its x-axis runs 15 -> 30); it is used here as the fit window. The 5-parameter fit
    is under-determined from a handful of points if seeded badly, so we multi-start
    over a small grid of period/phase guesses and keep the lowest-residual fit.
    Uniform sparse samples restrict the period to the Nyquist branch (at least
    twice the sample spacing). Fits with insufficient data or unidentifiable
    parameters are returned with ``valid=False`` and a ``failure_reason``.
    """
    from scipy.optimize import curve_fit

    cycles = np.asarray(cycles, dtype=float)
    mags = np.asarray(magnetization, dtype=float)
    lo, hi = window

    def unavailable(reason, x=None, y=None):
        return LateCycleFit(
            C=np.nan, gamma=np.nan, A=np.nan, T=np.nan, phi=np.nan,
            window=(lo, hi), cycles=np.asarray([] if x is None else x, dtype=float),
            values=np.asarray([] if y is None else y, dtype=float),
            pcov=np.full((5, 5), np.nan), valid=False, failure_reason=reason,
        )

    if cycles.ndim != 1 or mags.ndim != 1 or cycles.shape != mags.shape:
        return unavailable("cycles and magnetization must be aligned one-dimensional arrays")
    if (not np.all(np.isfinite(cycles)) or np.any(cycles < 0)
            or np.any(cycles != np.floor(cycles)) or np.any(np.diff(cycles) <= 0)):
        return unavailable("cycle labels must be nonnegative integers and strictly increasing")
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        return unavailable("fit window must be finite and increasing")
    sel = (cycles >= lo) & (cycles <= hi)
    x, y = cycles[sel], mags[sel]
    if x.size < 6:
        return unavailable("at least six samples are required in the fit window", x, y)
    if not np.all(np.isfinite(y)):
        return unavailable("fit-window magnetization contains non-finite samples", x, y)

    spacing = np.diff(x)
    uniform = np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-10)
    dense = uniform and np.isclose(spacing[0], 1.0)
    min_period = max(2.0, 2.0 * spacing[0]) if uniform else 2.0
    if min_period >= 30.0:
        return unavailable("sample spacing leaves no identifiable period below 30 cycles", x, y)

    def f(n, C, gamma, A, T, phi):
        return C * np.exp(-gamma * n) + A * np.cos(2.0 * np.pi * n / T + phi)

    C0 = float(np.clip(y.mean(), -1.0, 1.0))
    A0 = float(np.clip(0.5 * (y.max() - y.min()), 1e-4, 1.0))
    # T from the detrended-FFT estimate, plus a few fixed fallbacks (the drive
    # sits near a period-4 subharmonic), each tried at several phases.
    T_est = dominant_period(mags) if dense and np.all(np.isfinite(mags)) else np.nan
    starts = (T_est, 4.0, 3.0, 5.0, 6.0, 8.0) if dense else (min_period, 4.0, 5.0, 6.0, 8.0, 12.0)
    T_guesses = list(dict.fromkeys(float(t) for t in starts
                                  if np.isfinite(t) and min_period <= t <= 30.0))
    phi_guesses = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    bounds = ([-1.0, 0.0, 0.0, min_period, -2.0 * np.pi],
              [1.0, 1.0, 1.0, 30.0, 2.0 * np.pi])

    best_popt, best_pcov, best_sse = None, None, np.inf
    for T0 in T_guesses:
        for phi0 in phi_guesses:
            p0 = (C0, 0.02, A0, float(T0), float(phi0))
            try:
                popt, pcov = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=20000)
            except Exception:
                continue
            sse = float(np.sum((f(x, *popt) - y) ** 2))
            if np.isfinite(sse) and sse < best_sse:
                best_sse, best_popt, best_pcov = sse, popt, pcov
    if best_popt is None:
        return unavailable("nonlinear fit did not converge", x, y)

    C, gamma, A, T, phi = (float(v) for v in best_popt)
    fitted = LateCycleFit(C=C, gamma=gamma, A=A, T=T, phi=phi,
                         window=(lo, hi), cycles=x, values=y,
                         pcov=np.asarray(best_pcov, dtype=float))
    if uniform and not dense and np.isclose(T, min_period, rtol=1e-5, atol=1e-3):
        return unavailable("period at the sampling limit leaves amplitude and phase unidentifiable", x, y)
    if fitted.pcov.shape != (5, 5) or not np.all(np.isfinite(fitted.pcov)):
        return unavailable("fit covariance is non-finite", x, y)
    jacobian = fitted._jacobian(x)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if np.count_nonzero(singular_values > singular_values[0] * 1e-10) < 5:
        return unavailable("fit parameters are unidentifiable (rank-deficient Jacobian)", x, y)
    return fitted


def fit_late_cycle_anchored(
    cycles: np.ndarray,
    magnetization: np.ndarray,
    anchor: LateCycleFit,
    window: tuple[int, int] = (15, 30),
    *,
    period_tol: float = 0.15,
    n_boot: int = 400,
    seed: int = 2026,
) -> LateCycleFit:
    """Eq. 3 fit seeded by a known ``anchor`` fit, with bootstrap uncertainties.

    Sparse traces (e.g. even cycles only in ``window``: 8 points for 5
    parameters) leave the free multi-start fit of :func:`fit_late_cycle` badly
    conditioned -- its formal covariance can blow up by orders of magnitude.
    Here the nonlinear least-squares fit starts from the ``anchor`` parameters
    (typically the dense fit of the exact reference at the same size), and the
    period is held within ``period_tol`` of the anchor's period and above
    the sampling (Nyquist) limit; ``|A|`` is capped at the larger of the data's
    peak-to-peak and twice the anchor amplitude. Uncertainties come from a residual bootstrap:
    the fitted curve plus residuals resampled with replacement (inflated by
    ``sqrt(n / (n - 5))`` for the fitted degrees of freedom), refit ``n_boot``
    times. ``boot`` holds the draws; ``pcov`` is their covariance. An invalid
    anchor falls back to :func:`fit_late_cycle`.
    """
    from scipy.optimize import curve_fit

    if not anchor.valid:
        return fit_late_cycle(cycles, magnetization, window=window)
    free = fit_late_cycle(cycles, magnetization, window=window)
    if free.cycles.size < 6 or not np.all(np.isfinite(free.values)):
        return free   # input validation / too few samples: same failure as the free fit
    x, y = free.cycles, free.values
    lo, hi = window
    spacing = np.diff(x)
    uniform = np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-10)
    nyquist = 2.0 * spacing[0] if uniform else 2.0

    def f(n, C, gamma, A, T, phi):
        return C * np.exp(-gamma * n) + A * np.cos(2.0 * np.pi * n / T + phi)

    T_lo = max(anchor.T * (1.0 - period_tol), 1.05 * nyquist)
    T_hi = max(anchor.T * (1.0 + period_tol), T_lo * 1.05)
    phi_a = float((anchor.phi + np.pi) % (2.0 * np.pi) - np.pi)
    # The oscillation cannot be much larger than the data's own peak-to-peak.
    A_hi = float(max(np.ptp(y), 2.0 * abs(anchor.A), 1e-3))
    bounds = ([-1.0, 0.0, 0.0, T_lo, phi_a - 2.0 * np.pi],
              [1.0, 1.0, A_hi, T_hi, phi_a + 2.0 * np.pi])
    T0 = float(np.clip(anchor.T, T_lo, T_hi))
    base = [float(np.clip(anchor.C, -1.0, 1.0)), float(np.clip(anchor.gamma, 0.0, 1.0)),
            float(np.clip(abs(anchor.A), 1e-4, 0.99 * A_hi)), T0]

    def best_fit(values, starts):
        best, best_sse = None, np.inf
        for p0 in starts:
            try:
                popt, _ = curve_fit(f, x, values, p0=p0, bounds=bounds, maxfev=20000)
            except Exception:
                continue
            sse = float(np.sum((f(x, *popt) - values) ** 2))
            if np.isfinite(sse) and sse < best_sse:
                best, best_sse = popt, sse
        return best

    # Anchor phase first; quarter-turn offsets guard against a shifted phase.
    popt = best_fit(y, [base + [phi_a + d] for d in (0.0, 0.5 * np.pi, -0.5 * np.pi, np.pi)])
    if popt is None:
        return LateCycleFit(C=np.nan, gamma=np.nan, A=np.nan, T=np.nan, phi=np.nan,
                            window=(lo, hi), cycles=x, values=y,
                            pcov=np.full((5, 5), np.nan), valid=False,
                            failure_reason="anchored nonlinear fit did not converge")

    fitted_y = f(x, *popt)
    residuals = (y - fitted_y) * np.sqrt(x.size / max(x.size - 5, 1))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pb = best_fit(fitted_y + rng.choice(residuals, residuals.size, replace=True),
                      [list(popt)])
        if pb is not None:
            draws.append(pb)
    boot = np.asarray(draws, dtype=float).reshape(-1, 5)
    if len(boot) < max(10, n_boot // 2):
        return LateCycleFit(C=np.nan, gamma=np.nan, A=np.nan, T=np.nan, phi=np.nan,
                            window=(lo, hi), cycles=x, values=y,
                            pcov=np.full((5, 5), np.nan), valid=False,
                            failure_reason="too few bootstrap refits converged")
    C, gamma, A, T, phi = (float(v) for v in popt)
    return LateCycleFit(C=C, gamma=gamma, A=A, T=T, phi=phi, window=(lo, hi),
                        cycles=x, values=y, pcov=np.cov(boot, rowvar=False), boot=boot)


@dataclass
class ScalingFit:
    a: float
    xi: float
    c: float
    sizes: np.ndarray
    amplitudes: np.ndarray

    def model(self, n):
        n = np.asarray(n, dtype=float)
        return self.a * np.exp(-np.sqrt(n / self.xi)) + self.c

    @property
    def survives_thermodynamic_limit(self) -> bool:
        """A nonzero offset ``c`` => the oscillation persists as N -> infinity."""
        return self.c > 1e-3


def fit_amplitude_scaling(
    sizes: np.ndarray, amplitudes: np.ndarray, sigma: np.ndarray | None = None
) -> ScalingFit:
    """Fit ``A(N) = a exp(-sqrt(N/xi)) + c`` (paper Eq. 4), optionally weighted
    by the amplitude uncertainties ``sigma``."""
    from scipy.optimize import curve_fit

    sizes = np.asarray(sizes, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)

    def f(n, a, xi, c):
        return a * np.exp(-np.sqrt(n / xi)) + c

    p0 = (float(amplitudes[0]), float(sizes.mean()), float(amplitudes.min()))
    bounds = ([0.0, 1e-3, 0.0], [np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(f, sizes, amplitudes, p0=p0, bounds=bounds, maxfev=20000,
                            sigma=sigma, absolute_sigma=sigma is not None)
    except Exception:
        popt = p0
    return ScalingFit(a=popt[0], xi=popt[1], c=popt[2], sizes=sizes, amplitudes=amplitudes)


def _moving_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size < window:
        return np.full_like(x, x.mean())
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")
