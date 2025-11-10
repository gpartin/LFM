# -*- coding: utf-8 -*-
"""
Dissipation rate estimation utilities.

Provides robust estimation of dissipation rate γ from energy time series
with trimming and least-squares fallback.
"""
import numpy as np
from typing import Optional


def estimate_dissipation_rate(
    times: np.ndarray,
    energy_series: np.ndarray,
    trim_fraction: float = 0.1,
    use_fallback: bool = True
) -> Optional[float]:
    """
    Estimate dissipation rate γ from energy decay.
    
    For damped systems with E(t) ≈ E₀ exp(-2γt), estimates γ from the
    time derivative of log(E).
    
    Automatically detects and excludes initial growth/transient phases.
    Uses trimming to exclude initial transients and final noise regions.
    Falls back to least-squares exponential fit if derivative method fails.
    
    Args:
        times: Time array
        energy_series: Energy values over time
        trim_fraction: Fraction to trim from start/end (default: 0.1 = 10%)
        use_fallback: If True, use LS fit when derivative method fails
    
    Returns:
        Estimated γ value, or None if estimation fails
    """
    if len(times) < 10:
        return None
    
    # Detect initial growth phase (e.g., Gaussian spreading in strong damping)
    # Find energy peak and start analysis from there
    peak_idx = np.argmax(energy_series)
    if peak_idx > len(energy_series) * 0.2:  # Peak after 20% - use post-peak data
        times = times[peak_idx:]
        energy_series = energy_series[peak_idx:]
    
    # Trim transients
    n = len(times)
    i_start = int(n * trim_fraction)
    i_end = n - int(n * trim_fraction)
    
    if i_end <= i_start + 2:
        return None
    
    t_trim = times[i_start:i_end]
    E_trim = energy_series[i_start:i_end]
    
    # Avoid log of negative or zero
    if np.any(E_trim <= 0):
        return None
    
    # Method 1: Derivative of log(E)
    # E(t) ≈ E₀ exp(-2γt) → d/dt[ln(E)] ≈ -2γ → γ ≈ -0.5 * d/dt[ln(E)]
    try:
        log_E = np.log(E_trim)
        d_log_E = np.gradient(log_E, t_trim)
        gamma_est = -0.5 * np.mean(d_log_E)
        
        if gamma_est > 0 and np.isfinite(gamma_est):
            return float(gamma_est)
    except Exception:
        pass
    
    # Method 2: Least-squares exponential fit (fallback)
    if use_fallback:
        try:
            log_E = np.log(E_trim)
            # Fit: log(E) = a + b*t → γ = -b/2
            coeffs = np.polyfit(t_trim, log_E, deg=1)
            gamma_est = -0.5 * coeffs[0]  # coeffs[0] is slope
            
            if gamma_est > 0 and np.isfinite(gamma_est):
                return float(gamma_est)
        except Exception:
            pass
    
    return None
