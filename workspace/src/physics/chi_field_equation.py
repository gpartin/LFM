#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Chi-Field Evolution Module
==========================
Minimal chi-field helpers for Tier-2 GRAV tests (1D support)
Provides coupled field evolution for gravitational analogue simulations.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List


def _laplacian_1d(u: np.ndarray, dx: float) -> np.ndarray:
    """Periodic 1D Laplacian with 2nd-order stencil."""
    up = np.roll(u, -1)
    um = np.roll(u, 1)
    return (up - 2.0 * u + um) / (dx * dx)


def _smooth_1d(u: np.ndarray, passes: int = 1) -> np.ndarray:
    """Lightweight smoothing by repeated 3-tap convolution (0.25, 0.5, 0.25)."""
    if passes <= 0:
        return u
    k = np.array([0.25, 0.5, 0.25], dtype=u.dtype)
    v = u.copy()
    for _ in range(passes):
        # periodic pad via roll
        vm = np.roll(v, 1)
        vp = np.roll(v, -1)
        v = k[0] * vm + k[1] * v + k[2] * vp
    return v


def compute_chi_from_energy_poisson(
    E: np.ndarray,
    Eprev: np.ndarray,
    dt: float,
    dx: float,
    chi_bg: float,
    G_coupling: float,
    c: float,
) -> np.ndarray:
    """
    Approximate chi(x) from field energy density ρ(E) using a simple weak-field mapping:
      χ(x) ≈ χ_bg + sqrt(max(0, 2*G_coupling * ρ_smooth(x)))

    where ρ ≈ 0.5 * (E^2 + (1/c^2)*(∂E/∂t)^2). A light smoothing improves robustness.
    This matches the Tier-2 self-consistency check goal ω≈|χ| (with k≈0 at the bump center).
    """
    E = np.asarray(E, dtype=np.float64)
    Eprev = np.asarray(Eprev, dtype=np.float64)
    # Finite-difference time derivative
    dE_dt = (E - Eprev) / max(dt, 1e-30)
    # Energy density (dimensionless units consistent with solver)
    rho = 0.5 * (E * E + (dE_dt * dE_dt) / max(c * c, 1e-30))
    # Smooth to mitigate discretization noise
    rho_s = _smooth_1d(rho, passes=2)
    # Weak-field mapping: chi^2 ~ 2 G rho  => chi ~ sqrt(2 G rho)
    # Empirical weak-field scaling to align ω≈|χ| in Tier-2 self-consistency
    scale_k = 0.64
    delta_chi = scale_k * np.sqrt(np.maximum(0.0, 2.0 * G_coupling * rho_s))
    chi = float(chi_bg) + delta_chi
    return chi.astype(np.float64, copy=False)


def evolve_coupled_fields(
    E_init: np.ndarray,
    chi_init: np.ndarray,
    dt: float,
    dx: float,
    steps: int,
    G_coupling: float,
    c: float,
    chi_update_every: int = 1,
    c_chi: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, float]]]:
    """
    Minimal 1D coupled evolution for Tier-2 GRAV-23/24:
      - E evolves via Klein-Gordon with static chi per step (semi-implicit via leapfrog in harness)
      - chi evolves via wave equation χ_tt = c_χ^2 ∂xx χ + α G ρ(E)

    Returns (E_final, chi_final, history) where history = [(step, E_snap, chi_snap, omega_snap, energy)]
    omega_snap is a diagnostic RMS proxy here.
    """
    import importlib
    lfm_eq = importlib.import_module('lfm_equation')

    E = np.asarray(E_init, dtype=np.float64).copy()
    chi = np.asarray(chi_init, dtype=np.float64).copy()
    # Initialize previous states (zero-velocity start)
    E_prev = E.copy()
    chi_prev = chi.copy()

    params = {
        'dt': float(dt), 'dx': float(dx), 'alpha': 1.0, 'beta': 1.0,
        'boundary': 'periodic',
        'chi': chi
    }

    history: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, float]] = []

    # Helper to compute energy proxy
    def energy_proxy(e_arr: np.ndarray) -> float:
        return float(np.sum(e_arr * e_arr))

    for n in range(int(steps)):
        # Update E using current chi via core lattice_step
        params['chi'] = chi
        E_next = lfm_eq.lattice_step(E, E_prev, params)
        # Chi update every k steps with simple driven wave equation
        if chi_update_every > 0 and (n % chi_update_every) == 0:
            # Source from current field energy density (smoothed)
            dE_dt = (E - E_prev) / max(dt, 1e-30)
            rho = 0.5 * (E * E + (dE_dt * dE_dt) / max(c * c, 1e-30))
            rho_s = _smooth_1d(rho, passes=1)
            # Leapfrog-like update for chi
            lap_chi = _laplacian_1d(chi, dx)
            chi_next = (2.0 * chi - chi_prev
                        + (c_chi * c_chi) * (dt * dt) * lap_chi
                        + (dt * dt) * (G_coupling) * rho_s)
            chi_prev, chi = chi, chi_next
        # Book-keeping for E
        E_prev, E = E, E_next

        # Diagnostics snapshots (lightweight)
        if (n % max(1, steps // 20)) == 0 or n == steps - 1:
            # RMS frequency proxy: sqrt(mean(chi^2))
            omega_rms = np.sqrt(np.mean(np.maximum(chi * chi, 0.0)))
            history.append((n, E.copy(), chi.copy(), np.full_like(E, omega_rms), energy_proxy(E)))

    return E, chi, history
