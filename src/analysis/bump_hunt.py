"""Bump-hunt analysis: background fitting and Z-score calculation.

Implements a standard HEP falling-spectrum background fit to an invariant-mass
distribution with proper Poisson errors, and computes a local and global
significance (Z-score) for an excess using a sliding signal window.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True)
class BumpHuntResult:
    """Container for bump-hunt output."""
    z_score: float  # Local Z-score
    p_value: float  # Local p-value
    global_z_score: float # Estimated global Z-score
    signal_count: float
    background_estimate: float
    fit_params: Tuple[float, ...] = ()


def _hep_background(x: np.ndarray, p0: float, p1: float, p2: float) -> np.ndarray:
    """Evaluate a standard dijet background model.
    f(x) = p0 * (1 - x/sqrt(s))^p1 * (x/sqrt(s))^p2
    Here we scale x by 13000 (LHC center of mass energy approx)
    """
    xs = np.clip(x / 13000.0, 1e-5, 0.999)
    return p0 * ((1 - xs) ** p1) * (xs ** p2)


def bump_hunt(
    masses: np.ndarray,
    *,
    mass_window: Optional[Tuple[float, float]] = None,
    num_bins: int = 50,
    window_bins: int = 3,
) -> BumpHuntResult:
    """Run a bump-hunt on an invariant-mass distribution.

    Parameters
    ----------
    masses : np.ndarray
        1-D array of reconstructed invariant masses (GeV).
    mass_window : tuple[float, float] | None
        ``(low, high)`` edges of the signal window in GeV. If *None*, the
        script performs a sliding window scan to find the largest local excess.
    num_bins : int
        Number of histogram bins.
    window_bins : int
        Number of bins for the sliding window if mass_window is None (default 3).

    Returns
    -------
    BumpHuntResult
        Z-score (local), p-value (local), global Z-score, observed signal count, and background estimate.
    """
    masses = np.asarray(masses, dtype=np.float64)
    if len(masses) < 10:
        warnings.warn("Too few events for meaningful bump-hunt.", stacklevel=2)
        return BumpHuntResult(
            z_score=0.0, p_value=1.0, global_z_score=0.0, signal_count=0.0, background_estimate=0.0
        )

    counts, bin_edges = np.histogram(masses, bins=num_bins)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Proper Poisson uncertainties for the fit
    sigma = np.sqrt(counts)
    sigma[counts == 0] = 1.0  # Prevent division by zero

    # Exclude signal window from the fit if specified
    if mass_window is not None:
        lo, hi = mass_window
        mask = (bin_centres < lo) | (bin_centres > hi)
    else:
        # If no window is specified, we fit the whole spectrum first to identify anomalies.
        # Ideally, we should iteratively mask out regions, but for simplicity we fit all.
        mask = np.ones(len(bin_centres), dtype=bool)

    # Initial guess for curve_fit
    p0_guess = [np.sum(counts), 5.0, -5.0]
    
    try:
        fit_params, _ = optimize.curve_fit(
            _hep_background,
            bin_centres[mask],
            counts[mask].astype(float),
            p0=p0_guess,
            sigma=sigma[mask],
            absolute_sigma=True,
            maxfev=10000
        )
    except (RuntimeError, optimize.OptimizeWarning):
        warnings.warn("Background fit failed; returning null result.", stacklevel=2)
        return BumpHuntResult(
            z_score=0.0, p_value=1.0, global_z_score=0.0, signal_count=0.0, background_estimate=0.0
        )

    bg_estimate_all = _hep_background(bin_centres, *fit_params)
    bg_estimate_all = np.maximum(bg_estimate_all, 1e-3)  # strictly positive bg

    # Calculate significance
    if mass_window is not None:
        lo, hi = mass_window
        win = (bin_centres >= lo) & (bin_centres <= hi)
        trials = 1
    else:
        # Sliding window scan
        best_z = -np.inf
        best_win = None
        for i in range(len(bin_centres) - window_bins + 1):
            win_idx = np.zeros(len(bin_centres), dtype=bool)
            win_idx[i : i + window_bins] = True
            
            n_o = counts[win_idx].sum()
            n_b = bg_estimate_all[win_idx].sum()
            val = (n_o - n_b) / np.sqrt(n_b) if n_b > 0 else 0
            
            if val > best_z:
                best_z = val
                best_win = win_idx
                
        win = best_win
        trials = (len(bin_centres) - window_bins + 1) / window_bins

    n_obs = float(counts[win].sum())
    n_bg = float(bg_estimate_all[win].sum())

    if n_bg <= 0:
        z_score = 0.0
        p_value = 1.0
    else:
        # Local limit Z ≈ (N_obs - N_bg) / sqrt(N_bg)
        z_score = (n_obs - n_bg) / np.sqrt(n_bg)
        p_value = float(stats.norm.sf(abs(z_score)))

    # Global significance approximation (Look-Elsewhere Effect)
    global_p = min(1.0, trials * p_value)
    global_z = float(stats.norm.isf(global_p))

    return BumpHuntResult(
        z_score=round(z_score, 4),
        p_value=round(p_value, 6),
        global_z_score=round(global_z, 4) if np.isfinite(global_z) else 0.0,
        signal_count=round(n_obs, 1),
        background_estimate=round(n_bg, 1),
        fit_params=tuple(float(p) for p in fit_params),
    )
