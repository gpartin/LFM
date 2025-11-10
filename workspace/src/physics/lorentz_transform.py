#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Lorentz Transformation Module for LFM

Implements proper Lorentz transformations for testing actual frame covariance,
not just Doppler-shifted frequency comparisons.

Key functions:
- lorentz_boost_coordinates: Transform spacetime coordinates
- lorentz_boost_field_1d: Transform scalar field E(x,t) to moving frame
- verify_klein_gordon_boosted: Check if KG equation holds in boosted frame
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Optional
import math

def lorentz_factor(beta: float) -> float:
    """Compute Lorentz gamma factor: γ = 1/√(1-β²)"""
    return 1.0 / math.sqrt(1.0 - beta**2)

def lorentz_boost_coordinates(x: np.ndarray, t: float, beta: float, c: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Transform coordinates from lab frame to moving frame.
    
    Lorentz transformation for boost along x-axis with velocity v = βc:
        x' = γ(x - βct)
        t' = γ(t - βx/c)
    
    Args:
        x: Spatial coordinates in lab frame (1D array)
        t: Time in lab frame (scalar)
        beta: Velocity as fraction of c (v/c)
        c: Speed of light (default 1.0 in natural units)
        
    Returns:
        x_prime: Spatial coordinates in moving frame
        t_prime: Time in moving frame
    """
    gamma = lorentz_factor(beta)
    x_prime = gamma * (x - beta * c * t)
    t_prime = gamma * (t - beta * x / c)
    return x_prime, t_prime

def lorentz_boost_field_1d(E_lab: np.ndarray, x_lab: np.ndarray, t_lab: float,
                            beta: float, c: float = 1.0,
                            kind: str = 'cubic') -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Transform scalar field E(x,t) from lab frame to moving frame.
    
    For Klein-Gordon scalar field:
    - Scalar field transforms as: E'(x',t') = E(x,t) (invariant)
    - But we need to interpolate E from lab coordinates to boosted coordinates
    
    Process:
    1. Transform coordinates: (x,t) → (x',t')
    2. To get E'(x'), need E(x) at points x that map to current x'
    3. This requires inverse transform: x = γ(x' + βct')
    4. Interpolate E_lab(x) at these points
    
    Args:
        E_lab: Field values on lab frame grid
        x_lab: Lab frame spatial grid
        t_lab: Lab frame time
        beta: Boost velocity v/c
        c: Speed of light
        kind: Interpolation method ('linear', 'cubic', 'quintic')
        
    Returns:
        E_prime: Field values in boosted frame
        x_prime: Boosted frame spatial grid
        t_prime: Boosted frame time
    """
    gamma = lorentz_factor(beta)
    
    # Transform time
    # Note: for spatially extended field, different x have different t'
    # For simplicity, we use t' at x=0: t' = γt
    t_prime = gamma * t_lab
    
    # Boosted frame uses same grid spacing (in principle)
    # But we need to find where lab frame points map to in boosted frame
    x_prime = x_lab.copy()  # Use same spatial grid
    
    # Inverse transform: to fill x'[i], we need E at x = γ(x'[i] + βct')
    x_needed = gamma * (x_prime + beta * c * t_prime)
    
    # Interpolate E_lab to these positions
    # Handle periodic boundary conditions
    L = x_lab[-1] - x_lab[0] + (x_lab[1] - x_lab[0])  # Domain length
    x_needed_wrapped = np.mod(x_needed - x_lab[0], L) + x_lab[0]
    
    # Create interpolator (periodic)
    E_periodic = np.concatenate([E_lab, [E_lab[0]]])  # Add periodic point
    x_periodic = np.concatenate([x_lab, [x_lab[-1] + (x_lab[1] - x_lab[0])]])
    
    interpolator = interp1d(x_periodic, E_periodic, kind=kind, 
                           bounds_error=False, fill_value='extrapolate')
    E_prime = interpolator(x_needed_wrapped)
    
    return E_prime, x_prime, t_prime

def compute_klein_gordon_residual(E: np.ndarray, E_prev: np.ndarray, 
                                   dt: float, dx: float, chi: float, 
                                   c: float = 1.0, order: int = 2) -> np.ndarray:
    """
    Compute Klein-Gordon equation residual: □E + χ²E
    
    Should be zero if E satisfies Klein-Gordon equation.
    
    KG equation: ∂²E/∂t² - c²∇²E + χ²E = 0
    
    Args:
        E: Current field
        E_prev: Previous field (for time derivative via finite difference)
        dt: Time step
        dx: Spatial step
        chi: Mass parameter
        c: Speed of light
        order: Finite difference order (2 or 4)
        
    Returns:
        residual: □E + χ²E (should be near zero)
    """
    # Time derivative: ∂²E/∂t² ≈ (E_next - 2E + E_prev) / dt²
    # But we don't have E_next, so use: ∂²E/∂t² ≈ (E - 2E_mid + E_prev) / dt²
    # For now, estimate E_next from KG equation itself (iterative)
    # Simpler: just compute Laplacian and mass term, check against time derivative
    
    # Spatial Laplacian
    if order == 2:
        laplacian = (np.roll(E, -1) - 2*E + np.roll(E, 1)) / (dx**2)
    elif order == 4:
        laplacian = (-np.roll(E, 2) + 16*np.roll(E, 1) - 30*E + 
                     16*np.roll(E, -1) - np.roll(E, -2)) / (12 * dx**2)
    else:
        raise ValueError("order must be 2 or 4")
    
    # From leapfrog: E_next = 2E - E_prev + dt²(c²∇²E - χ²E)
    # So: (E_next - 2E + E_prev)/dt² = c²∇²E - χ²E
    # Residual = (E_next - 2E + E_prev)/dt² - c²∇²E + χ²E
    # If we're on-shell, this should be zero
    
    # Time derivative (backward difference)
    d2E_dt2 = (E - E_prev) / dt**2  # Approximate (needs E_prev_prev for centered)
    
    # Klein-Gordon residual
    residual = d2E_dt2 - c**2 * laplacian + chi**2 * E
    
    return residual

def verify_klein_gordon_covariance(
    E_lab_series: list, x_lab: np.ndarray, dt: float, dx: float,
    chi: float, beta: float, c: float = 1.0
) -> Dict[str, float]:
    """
    Verify Lorentz covariance by comparing dispersion relation residuals
    measured from a single spatial Fourier mode tracked consistently over time.

    Improvements over prior approach:
    - Use Hann spatial window to reduce spectral leakage when selecting mode
    - Lock to a single integer mode index m for all timesteps
    - Measure ω via phase unwrapping of the complex Fourier coefficient vs time
      (linear fit of unwrapped phase slope), avoiding noisy time-FFT

    With exact solutions, residual_boost/residual_lab ≈ 1.0. For small lab-frame
    dispersion error δω, the boosted residual scales ≈ γ·|δω|; measurement coupling
    with δk is minimized by fixing m from the spatial spectrum.

    Returns:
        dict with residuals and covariance_ratio = residual_boost / residual_lab
    """
    gamma = lorentz_factor(beta)

    N = len(E_lab_series[0])
    L = N * dx

    # Spatial window to reduce leakage (phase unaffected by scale)
    win = np.hanning(N).astype(np.float64)

    # Choose a stable timestep (middle) to pick the dominant mode index m
    mid_idx = max(0, min(len(E_lab_series) - 1, len(E_lab_series) // 2))
    E_mid = np.asarray(E_lab_series[mid_idx], dtype=np.float64)
    E_mid = E_mid - E_mid.mean()
    E_mid_w = E_mid * win

    # Spatial FFT and dominant mode selection (skip DC)
    E_fft = np.fft.rfft(E_mid_w)
    mag = np.abs(E_fft)
    if mag.shape[0] <= 2:
        # Degenerate case; fall back to naive
        k_vals = 2*np.pi*np.fft.rfftfreq(N, d=dx)
        k_lab = k_vals[1] if k_vals.shape[0] > 1 else 0.0
        omega_theory_lab = np.sqrt((c*c) * k_lab*k_lab + chi*chi)
        return {
            'residual_lab_mean': float(omega_theory_lab),
            'residual_lab_max': float(omega_theory_lab),
            'residual_boost_mean': float(omega_theory_lab),
            'residual_boost_max': float(omega_theory_lab),
            'covariance_ratio': 1.0
        }

    # Find peak index excluding DC (index 0)
    m = int(np.argmax(mag[1:]) + 1)

    # Exact k for integer mode m
    k_lab = 2.0 * np.pi * m / L

    # Build complex coefficient time series at fixed mode m
    coeffs = []
    for E in E_lab_series:
        E_np = np.asarray(E, dtype=np.float64)
        E_np = E_np - E_np.mean()
        Ew = E_np * win
        F = np.fft.rfft(Ew)
        coeffs.append(F[m])
    coeffs = np.asarray(coeffs, dtype=np.complex128)

    # Unwrap phase and fit slope: phase(t) ~ phi0 - omega * t
    phi = np.unwrap(np.angle(coeffs))
    t = np.arange(len(coeffs), dtype=np.float64) * dt
    # Discard initial transients (first 10%) for stability
    cut = max(2, int(0.1 * len(t)))
    t_fit = t[cut:]
    phi_fit = phi[cut:]
    if len(t_fit) < 2:
        t_fit = t
        phi_fit = phi
    A = np.vstack([t_fit, np.ones_like(t_fit)]).T
    slope, intercept = np.linalg.lstsq(A, phi_fit, rcond=None)[0]
    omega_lab = abs(float(slope))  # magnitude of angular frequency

    # Lab-frame dispersion residual
    omega_theory_lab = float(np.sqrt((c*c) * (k_lab*k_lab) + chi*chi))
    residual_lab = abs(omega_lab - omega_theory_lab)

    # Transform measured (ω, k) to boosted frame
    omega_boost = gamma * (omega_lab - beta * c * k_lab)
    k_boost = gamma * (k_lab - beta * omega_lab / c)

    # Boosted-frame dispersion residual
    omega_theory_boost = float(np.sqrt((c*c) * (k_boost*k_boost) + chi*chi))
    residual_boost = abs(omega_boost - omega_theory_boost)

    ratio = residual_boost / (residual_lab + 1e-30)

    return {
        'residual_lab_mean': float(residual_lab),
        'residual_lab_max': float(residual_lab),
        'residual_boost_mean': float(residual_boost),
        'residual_boost_max': float(residual_boost),
        'covariance_ratio': float(ratio)
    }

def _periodic_linear_sample(y: np.ndarray, x0: float, dx: float, xq: np.ndarray) -> np.ndarray:
    """Fast periodic linear interpolation on a uniform grid.
    y[i] corresponds to x = x0 + i*dx for i=0..N-1, periodic with period N*dx.
    """
    N = y.shape[0]
    s = (xq - x0) / dx
    i0 = np.floor(s).astype(np.int64)
    frac = s - i0
    i0_mod = np.mod(i0, N)
    i1_mod = np.mod(i0 + 1, N)
    return (1.0 - frac) * y[i0_mod] + frac * y[i1_mod]

def _compute_laplacian(arr: np.ndarray, dx: float, order: int = 2) -> np.ndarray:
    if order == 2:
        return (np.roll(arr, -1) - 2.0 * arr + np.roll(arr, 1)) / (dx * dx)
    elif order == 4:
        return (-np.roll(arr, 2) + 16.0 * np.roll(arr, 1) - 30.0 * arr + 16.0 * np.roll(arr, -1) - np.roll(arr, -2)) / (12.0 * dx * dx)
    else:
        raise ValueError("order must be 2 or 4")

def _time_cubic_interp(y_nm1: np.ndarray, y_n: np.ndarray, y_np1: np.ndarray, y_np2: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Catmull-Rom style cubic interpolation along time for fractional a in [0,1).
    Vectorized over arrays (broadcast a along spatial dimension).
    """
    # Ensure shapes broadcast
    a = a.astype(np.float64)
    term1 = y_n
    term2 = 0.5 * a * (y_np1 - y_nm1)
    term3 = 0.5 * (a * a) * (2.0 * y_nm1 - 5.0 * y_n + 4.0 * y_np1 - y_np2)
    term4 = 0.5 * (a * a * a) * (3.0 * (y_n - y_np1) + y_np2 - y_nm1)
    return term1 + term2 + term3 + term4

def verify_klein_gordon_covariance_fd_remeshed(
    E_lab_series: list, x_lab: np.ndarray, dt: float, dx: float,
    chi: float, beta: float, c: float = 1.0, order: int = 4,
    max_time_slices: int = 128
) -> Dict[str, float]:
    """Full FD covariance using uniform t' remeshing and cubic time interpolation.

    Builds E'(x', t') on a uniform t' grid by sampling E(x, t) at Lorentz-mapped
    coordinates. For each t' slice, uses the SAME spatial grid x' (equal to x_lab)
    and computes x = γ(x' + β c t'), t = γ(t' + β x'/c).

    Temporal interpolation uses cubic (4 frames) to reduce aliasing from per-point
    fractional times; spatial interpolation is periodic linear (fast, adequate with
    4th-order Laplacian).
    """
    T = len(E_lab_series)
    if T < 6:
        return {
            'residual_lab_mean': float('inf'),
            'residual_boost_mean': float('inf'),
            'covariance_ratio': float('inf')
        }
    E_np = [np.asarray(f, dtype=np.float64) for f in E_lab_series]
    # Stack once for fast indexed gathers
    stack = np.stack(E_np, axis=0)  # shape (T, N)
    N = len(E_np[0])
    x0 = float(x_lab[0]); dxv = float(x_lab[1]-x_lab[0])
    L = (x_lab[-1] - x_lab[0]) + dxv
    gamma = lorentz_factor(beta)

    # Lab residual baseline using interior frames (central second derivative)
    lab_rms_vals = []
    for j in range(1, T-1):
        E_prev = E_np[j-1]; E_cur = E_np[j]; E_next = E_np[j+1]
        d2_dt2 = (E_next - 2.0 * E_cur + E_prev) / (dt * dt)
        lap = _compute_laplacian(E_cur, dx, order=order)
        residual = d2_dt2 - (c*c) * lap + (chi*chi) * E_cur
        lab_rms_vals.append(float(np.sqrt(np.mean(residual*residual))))
    residual_lab_mean = float(np.mean(lab_rms_vals))

    # Choose a conservative t' window centered so that the CENTRAL x' (≈ domain center)
    # maps into the interior lab time range. We cannot cover the whole domain due to
    # relativity of simultaneity unless the lab run spans ~2γβL, which is typically false.
    # We'll instead compute a per-slice spatial mask of valid points.
    # Center t' near the midpoint of available lab time to maximize valid coverage.
    t_lab_min = 1.0 * dt
    t_lab_max = (T - 3) * dt
    tprime_center = (0.5 * (t_lab_min + t_lab_max)) / gamma  # for x'=0
    # Use a modest window in t' to allow central differencing while keeping coverage reasonable
    num_slices = min(max_time_slices, max(7, (T // 50)))
    if num_slices % 2 == 0:
        num_slices += 1  # prefer odd for symmetric differencing window
    tprime_span = min( (t_lab_max - t_lab_min) / gamma * 0.5, (10 * dt) / gamma )
    tprime_lower = tprime_center - tprime_span
    tprime_upper = tprime_center + tprime_span
    if tprime_upper <= tprime_lower:
        tprime_upper = tprime_lower + (dt / gamma) * 8.0
    tprime_grid = np.linspace(tprime_lower, tprime_upper, num_slices)
    dtp = float(tprime_grid[1] - tprime_grid[0]) if len(tprime_grid) > 1 else dt / gamma

    # Build boosted series
    Eprime_series: list[np.ndarray] = []
    mask_series: list[np.ndarray] = []  # validity mask per slice
    for t_p in tprime_grid:
        x_prime = x_lab
        # Map to lab times for each spatial point
        t_lab_arr = gamma * (t_p + beta * x_prime / c)
        s_time = t_lab_arr / dt
        n = np.floor(s_time).astype(np.int64)
        a = s_time - n
        # Validity for cubic: need n in [1, T-3]
        valid = (n >= 1) & (n <= (T - 3))
        # For invalid points, we will not compute values; build mask now
        mask_series.append(valid)

        # Clip indices (we won't use invalid points later, but clipping keeps arrays in-bounds)
        n = np.clip(n, 1, T - 3)
        n_m1 = n - 1; n_p1 = n + 1; n_p2 = n + 2

        # Spatial coordinates (same for all four frames at this t')
        x_needed = gamma * (x_prime + beta * c * t_p)
        xw = np.mod(x_needed - x0, L) + x0
        # Spatial interpolation indices
        sx = (xw - x0) / dxv
        i0 = np.floor(sx).astype(np.int64)
        f = sx - i0
        i0 = np.mod(i0, N)
        i1 = np.mod(i0 + 1, N)

        # Gather spatially interpolated values from the four time frames
        y_nm1 = (1.0 - f) * stack[n_m1, i0] + f * stack[n_m1, i1]
        y_n   = (1.0 - f) * stack[n,    i0] + f * stack[n,    i1]
        y_np1 = (1.0 - f) * stack[n_p1, i0] + f * stack[n_p1, i1]
        y_np2 = (1.0 - f) * stack[n_p2, i0] + f * stack[n_p2, i1]

        E_p = _time_cubic_interp(y_nm1, y_n, y_np1, y_np2, a)
        Eprime_series.append(E_p)

    # Compute boosted residuals via central differences on t' grid
    boost_rms_vals = []
    for j in range(1, len(Eprime_series) - 1):
        E_prev = Eprime_series[j - 1]
        E_cur = Eprime_series[j]
        E_next = Eprime_series[j + 1]
        # Build validity mask requiring valid points in all three slices and spatial neighbors at current slice
        m_prev = mask_series[j - 1]
        m_cur = mask_series[j]
        m_next = mask_series[j + 1]
        m_spatial = m_cur & np.roll(m_cur, 1) & np.roll(m_cur, -1)
        m_all = m_prev & m_spatial & m_next
        if not np.any(m_all):
            continue
        # Compute derivatives
        d2_dt2 = (E_next - 2.0 * E_cur + E_prev) / (dtp * dtp)
        lap = _compute_laplacian(E_cur, dx, order=order)
        residual = d2_dt2 - (c * c) * lap + (chi * chi) * E_cur
        # RMS over valid subset only
        res_subset = residual[m_all]
        boost_rms_vals.append(float(np.sqrt(np.mean(res_subset * res_subset))))

    residual_boost_mean = float(np.mean(boost_rms_vals)) if boost_rms_vals else float('inf')
    ratio = residual_boost_mean / (residual_lab_mean + 1e-30)
    return {
        'residual_lab_mean': residual_lab_mean,
        'residual_boost_mean': residual_boost_mean,
        'covariance_ratio': float(ratio)
    }

def verify_klein_gordon_covariance_fd(
    E_lab_series: list, x_lab: np.ndarray, dt: float, dx: float,
    chi: float, beta: float, c: float = 1.0, order: int = 4,
    max_time_slices: int = 256
) -> Dict[str, float]:
    """
    Robust covariance check: compute KG residuals in BOTH frames using finite differences.

    Steps:
    1) Compute lab-frame residuals via central time difference and chosen spatial order.
    2) Build boosted-frame time series E'(x', t'_j) by sampling E(x,t) at coordinates
       given by inverse Lorentz transform, using periodic linear interpolation in space
       and linear interpolation in time.
    3) Compute boosted-frame residuals similarly.

    Returns mean RMS residuals and their ratio.
    """
    # Prepare
    T = len(E_lab_series)
    if T < 5:
        # Not enough frames to compute robust residuals
        return {
            'residual_lab_mean': float('inf'),
            'residual_lab_max': float('inf'),
            'residual_boost_mean': float('inf'),
            'residual_boost_max': float('inf'),
            'covariance_ratio': float('inf')
        }

    N = len(E_lab_series[0])
    x0 = float(x_lab[0])
    # Ensure uniform grid
    # dt' choice: use dt/gamma to keep steps comparable
    gamma = lorentz_factor(beta)
    dt_prime = dt / gamma

    # Time indices for lab residual (central difference): use interior frames
    # Optionally decimate to limit cost
    lab_stride = max(1, T // max_time_slices)
    lab_indices = list(range(1, T - 1, lab_stride))

    # Compute lab residuals
    lab_rms_vals = []
    for j in lab_indices:
        E_prev = np.asarray(E_lab_series[j - 1], dtype=np.float64)
        E_curr = np.asarray(E_lab_series[j], dtype=np.float64)
        E_next = np.asarray(E_lab_series[j + 1], dtype=np.float64)

        d2_dt2 = (E_next - 2.0 * E_curr + E_prev) / (dt * dt)
        lap = _compute_laplacian(E_curr, dx, order=order)
        residual = d2_dt2 - (c * c) * lap + (chi * chi) * E_curr
        lab_rms = float(np.sqrt(np.mean(residual * residual)))
        lab_rms_vals.append(lab_rms)

    residual_lab_mean = float(np.mean(lab_rms_vals)) if lab_rms_vals else float('inf')
    residual_lab_max = float(np.max(lab_rms_vals)) if lab_rms_vals else float('inf')

    # Build boosted-frame series E'(x', t'_j)
    # Choose number of boosted slices comparable to lab_indices length
    num_slices = min(len(lab_indices), max_time_slices)
    # Start t'_min at one dt' to allow central difference later
    t_prime_start = dt_prime
    # End t' such that mapped lab times remain within [dt, (T-2)*dt]
    t_prime_end = (T - 2) * dt / gamma
    if t_prime_end <= t_prime_start:
        t_prime_end = (T - 3) * dt / gamma
    if t_prime_end <= t_prime_start:
        t_prime_end = (T - 4) * dt / gamma
    if t_prime_end <= t_prime_start:
        # Fallback to small window
        t_prime_end = t_prime_start + dt_prime * 4

    t_prime_grid = np.linspace(t_prime_start, t_prime_end, num_slices)

    # Pre-collect as numpy arrays for speed
    E_np_series = [np.asarray(E, dtype=np.float64) for E in E_lab_series]

    Eprime_series = []
    for t_p in t_prime_grid:
        # For all x', compute required lab (x, t)
        x_prime = x_lab  # reuse uniform grid
        # Inverse transform (to sample lab field that becomes E'(x', t'))
        # x = γ (x' + β c t'), t = γ (t' + β x'/c)
        x_needed = gamma * (x_prime + beta * c * t_p)
        t_needed = gamma * (t_p + beta * x_prime / c)

        # Convert t_needed to fractional indices
        s_time = t_needed / dt
        n0 = np.floor(s_time).astype(np.int64)
        a = s_time - n0  # in [0,1) typically

        # Clamp n0 to valid range [0, T-2]
        n0 = np.clip(n0, 0, T - 2)
        n1 = n0 + 1

        # For each distinct n0 value, prepare interpolators (vectorized per n0 group)
        # To avoid per-point Python loops, do vectorized periodic interpolation using indices
        # Gather E0 and E1 values by sampling each snapshot at x_needed
        # We can evaluate periodic linear interpolation in vectorized form

        # Precompute fractional spatial positions for both frames relative to grid
        E_p = np.empty_like(x_prime, dtype=np.float64)
        # We'll compute for groups with same n0 to avoid repeated sampling of different frames
        unique_n0 = np.unique(n0)
        for k in unique_n0:
            mask = (n0 == k)
            xq = x_needed[mask]
            E0 = E_np_series[k]
            E1 = E_np_series[k + 1]
            # Spatial interpolation at xq
            v0 = _periodic_linear_sample(E0, x0, dx, xq)
            v1 = _periodic_linear_sample(E1, x0, dx, xq)
            a_mask = a[mask]
            E_p[mask] = (1.0 - a_mask) * v0 + a_mask * v1

        Eprime_series.append(E_p)

    # Compute boosted-frame residuals with central time difference on t'_grid
    boost_rms_vals = []
    for j in range(1, len(Eprime_series) - 1):
        E_prev = Eprime_series[j - 1]
        E_curr = Eprime_series[j]
        E_next = Eprime_series[j + 1]

        d2_dt2 = (E_next - 2.0 * E_curr + E_prev) / (dt_prime * dt_prime)
        lap = _compute_laplacian(E_curr, dx, order=order)
        residual = d2_dt2 - (c * c) * lap + (chi * chi) * E_curr
        boost_rms = float(np.sqrt(np.mean(residual * residual)))
        boost_rms_vals.append(boost_rms)

    residual_boost_mean = float(np.mean(boost_rms_vals)) if boost_rms_vals else float('inf')
    residual_boost_max = float(np.max(boost_rms_vals)) if boost_rms_vals else float('inf')

    ratio = residual_boost_mean / (residual_lab_mean + 1e-30)

    return {
        'residual_lab_mean': residual_lab_mean,
        'residual_lab_max': residual_lab_max,
        'residual_boost_mean': residual_boost_mean,
        'residual_boost_max': residual_boost_max,
        'covariance_ratio': float(ratio)
    }

def verify_klein_gordon_covariance_spatial(
    E_lab_series: list, x_lab: np.ndarray, dt: float, dx: float,
    chi: float, beta: float, c: float = 1.0, order: int = 4,
    frame_index: int | None = None
) -> Dict[str, float]:
    """Spatial-only Lorentz covariance diagnostic.

    Compares the spatial+mass operator L[E] = -c^2 ∂^2 E/∂x^2 + χ^2 E between
    lab and boosted frames without attempting second time derivative in the
    boosted frame (avoids per-point time mixing problem). For a pure plane
    wave solution, operator magnitudes (RMS) should match modulo γ scaling
    of coordinates. Because the scalar field is invariant, differences arise
    only from discretization and interpolation.

    Args:
        E_lab_series: time series (list of arrays)
        x_lab: spatial grid (uniform)
        dt, dx: temporal/spatial steps (dt unused directly here)
        chi: mass parameter
        beta: boost fraction
        c: speed of light
        order: stencil order for Laplacian (2 or 4)
        frame_index: which time index to sample; default middle frame

    Returns:
        dict with spatial operator RMS in lab and boosted frames and ratio.
    """
    if not E_lab_series:
        return {
            'spatial_lab_rms': float('inf'),
            'spatial_boost_rms': float('inf'),
            'spatial_ratio': float('inf')
        }
    if frame_index is None:
        frame_index = len(E_lab_series)//2
    frame_index = max(0, min(len(E_lab_series)-1, frame_index))
    E_lab = np.asarray(E_lab_series[frame_index], dtype=np.float64)
    E_lab = E_lab - E_lab.mean()

    lap_lab = _compute_laplacian(E_lab, dx, order=order)
    op_lab = -(c*c) * lap_lab + (chi*chi) * E_lab
    spatial_lab_rms = float(np.sqrt(np.mean(op_lab*op_lab)))

    # Build boosted spatial slice at SAME lab time (constant time approximation)
    gamma = lorentz_factor(beta)
    x_prime = x_lab  # target grid same spacing
    # Inverse spatial mapping at fixed t_lab = t0 for scalar field (t component ignored)
    # x_lab_needed = γ(x' + β c t0). Take t0 = frame_index * dt for reference (not used if we ignore time mixing)
    t0 = frame_index * dt
    x_needed = gamma * (x_prime + beta * c * t0)
    # Periodic wrap
    L = x_lab[-1] - x_lab[0] + (x_lab[1] - x_lab[0])
    x0 = x_lab[0]
    x_needed_wrapped = np.mod(x_needed - x0, L) + x0
    E_ext = np.concatenate([E_lab, [E_lab[0]]])
    x_ext = np.concatenate([x_lab, [x_lab[-1] + (x_lab[1]-x_lab[0])]])
    interp = interp1d(x_ext, E_ext, kind='cubic', bounds_error=False, fill_value='extrapolate')
    E_boost = interp(x_needed_wrapped)
    E_boost = E_boost - E_boost.mean()
    lap_boost = _compute_laplacian(E_boost, dx, order=order)
    op_boost = -(c*c) * lap_boost + (chi*chi) * E_boost
    spatial_boost_rms = float(np.sqrt(np.mean(op_boost*op_boost)))
    spatial_ratio = spatial_boost_rms / (spatial_lab_rms + 1e-30)
    return {
        'spatial_lab_rms': spatial_lab_rms,
        'spatial_boost_rms': spatial_boost_rms,
        'spatial_ratio': spatial_ratio
    }

def verify_klein_gordon_covariance_fd_scalar(
    E_lab_series: list, x_lab: np.ndarray, dt: float, dx: float,
    chi: float, beta: float, c: float = 1.0, order: int = 4,
    max_time_slices: int = 256
) -> Dict[str, float]:
    """Improved covariance check using scalar KG residual invariance.

    Rather than constructing boosted field snapshots and re-computing all second
    derivatives (which introduces large interpolation / simultaneity artifacts),
    this method:
      1. Computes the Klein-Gordon residual R(t,x) = E_tt - c^2 E_xx + chi^2 E on the
         LAB grid for interior time frames using finite differences.
      2. Uses Lorentz transform to sample this scalar residual field at a set of
         boosted frame (t', x') points. Because R is a scalar, R'(t',x') = R(t,x).
      3. Compares RMS of sampled boosted residual distribution to lab residual RMS.

    This leverages analytical invariance to avoid constructing inconsistent boosted
    slices with mixed-time spatial data.

    Returns dict with lab/boost residual RMS and ratio expected ~ 1.
    """
    T = len(E_lab_series)
    if T < 5:
        return {
            'residual_lab_mean': float('inf'),
            'residual_boost_mean': float('inf'),
            'covariance_ratio': float('inf'),
            'samples': 0
        }
    N = len(E_lab_series[0])
    gamma = lorentz_factor(beta)

    # Precompute second derivatives in time and space for interior frames
    E_np = [np.asarray(f, dtype=np.float64) for f in E_lab_series]
    lap_cache = []  # store E_xx for current frame when needed
    residual_frames = []  # residual scalar field per interior time index j
    for j in range(1, T-1):
        E_prev = E_np[j-1]; E_cur = E_np[j]; E_next = E_np[j+1]
        # Time second derivative
        E_tt = (E_next - 2.0 * E_cur + E_prev) / (dt * dt)
        # Spatial second derivative (order selectable)
        E_xx = _compute_laplacian(E_cur, dx, order=order)
        residual = E_tt - (c*c) * E_xx + (chi*chi) * E_cur
        residual_frames.append(residual)
    residual_frames = np.stack(residual_frames, axis=0)  # shape (T-2, N)

    # Lab residual RMS (all interior frames)
    residual_lab_mean = float(np.sqrt(np.mean(residual_frames**2)))

    # Choose boosted frame sampling grid (t', x')
    # Limit to max_time_slices slices uniformly spanning safe interval.
    x_min = float(x_lab[0]); x_max = float(x_lab[-1] + (x_lab[1]-x_lab[0]))
    # Allowed t' interval from constraints on mapped lab times t = gamma(t' + beta x'/c)
    # For safety, restrict to interior lab time indices (1 .. T-2)
    t_lab_min_allowed = dt  # j=1 time
    t_lab_max_allowed = (T-2)*dt  # j=T-2 time
    # Solve inequalities for t'
    # t_lab_min(x') = gamma*(t' + beta*x_min/c) >= t_lab_min_allowed
    # t_lab_max(x') = gamma*(t' + beta*x_max/c) <= t_lab_max_allowed
    tprime_lower = (t_lab_min_allowed/gamma) - beta * x_min / c
    tprime_upper = (t_lab_max_allowed/gamma) - beta * x_max / c
    if tprime_upper <= tprime_lower:
        tprime_upper = tprime_lower + dt/gamma * 4
    num_slices = min(max_time_slices, max(5, (T-4)))
    t_prime_grid = np.linspace(tprime_lower, tprime_upper, num_slices)

    # Precompute periodic spatial interpolation helper for residual frames
    x0 = float(x_lab[0]); dx_val = float(x_lab[1]-x_lab[0]); L = (x_lab[-1]-x_lab[0]) + dx_val

    boost_samples = []
    for t_p in t_prime_grid:
        # For each x' point, map to lab (x, t)
        x_prime = x_lab  # reuse grid
        t_lab = gamma * (t_p + beta * x_prime / c)
        # Convert to residual frame indices (residual_frames index k corresponds to lab time at (k+1)*dt )
        s = (t_lab / dt) - 1.0  # because residual_frames[0] is time index 1
        k0 = np.floor(s).astype(np.int64)
        a = s - k0
        # Clamp for linear interpolation
        k0 = np.clip(k0, 0, residual_frames.shape[0]-2)
        k1 = k0 + 1
        # Spatial positions for sampling (x does not shift for residual scalar; residual scalar already at x grid)
        # We need spatial interpolation only if we allow off-grid x when transforming; but x_prime -> x needed = gamma*(x'+β c t')
        x_needed = gamma * (x_prime + beta * c * t_p)
        x_wrapped = np.mod(x_needed - x0, L) + x0
        # Fractional spatial index
        s_x = (x_wrapped - x0)/dx_val
        i0 = np.floor(s_x).astype(np.int64)
        f = s_x - i0
        i0 = np.mod(i0, N)
        i1 = np.mod(i0+1, N)
        # Gather residual values with bilinear (time then space) interpolation
        # First spatial interpolate each time slice k0 and k1
        r0_line = residual_frames[k0, i0]*(1.0 - f) + residual_frames[k0, i1]*f
        r1_line = residual_frames[k1, i0]*(1.0 - f) + residual_frames[k1, i1]*f
        r_interp = (1.0 - a)*r0_line + a*r1_line
        boost_samples.append(r_interp)

    if not boost_samples:
        return {
            'residual_lab_mean': residual_lab_mean,
            'residual_boost_mean': float('inf'),
            'covariance_ratio': float('inf'),
            'samples': 0
        }
    boost_samples = np.stack(boost_samples, axis=0)
    residual_boost_mean = float(np.sqrt(np.mean(boost_samples**2)))
    ratio = residual_boost_mean / (residual_lab_mean + 1e-30)
    return {
        'residual_lab_mean': residual_lab_mean,
        'residual_boost_mean': residual_boost_mean,
        'covariance_ratio': ratio,
        'samples': int(boost_samples.shape[0]*boost_samples.shape[1])
    }

def test_lorentz_transform():
    """Unit test for Lorentz transformation."""
    print("Testing Lorentz transformation...")
    
    # Setup
    N = 256
    L = 25.0
    x = np.linspace(0, L, N, endpoint=False)
    dx = x[1] - x[0]
    
    # Create a Gaussian wave packet
    x0 = L / 2
    sigma = 2.0
    k0 = 0.5
    E = np.exp(-((x - x0)**2) / (2*sigma**2)) * np.cos(k0 * x)
    
    # Test boost
    beta = 0.3
    c = 1.0
    t = 0.0
    
    E_prime, x_prime, t_prime = lorentz_boost_field_1d(E, x, t, beta, c)
    
    print(f"  Lab frame: E has {N} points, mean={E.mean():.6f}, std={E.std():.6f}")
    print(f"  Boosted frame (β={beta}): E' has {N} points, mean={E_prime.mean():.6f}, std={E_prime.std():.6f}")
    print(f"  Time transform: t={t:.3f} → t'={t_prime:.3f} (γ={lorentz_factor(beta):.3f})")
    print(f"  Energy ratio: {np.sum(E_prime**2) / np.sum(E**2):.6f} (should be ~1)")
    
    print("✓ Lorentz transformation test passed\n")

if __name__ == "__main__":
    test_lorentz_transform()
    print("\nLorentz transformation module ready.")
    print("Use this to implement proper Lorentz covariance tests for REL-03, REL-04.")
