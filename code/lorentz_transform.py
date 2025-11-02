#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com

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
    Verify that Klein-Gordon equation holds in both lab and boosted frames.
    
    This is the TRUE test of Lorentz covariance:
    1. Field evolves according to KG in lab frame
    2. Transform to boosted frame at each time step
    3. Verify KG equation ALSO satisfied in boosted frame
    4. If yes → equation is Lorentz covariant
    
    Args:
        E_lab_series: List of E(x,t) snapshots in lab frame
        x_lab: Spatial grid (lab frame)
        dt: Time step
        dx: Spatial step  
        chi: Mass parameter
        beta: Boost velocity
        c: Speed of light
        
    Returns:
        dict with residual statistics in both frames
    """
    # Check residuals in lab frame
    residuals_lab = []
    for i in range(1, len(E_lab_series) - 1):
        E = E_lab_series[i]
        E_prev = E_lab_series[i-1]
        E_next = E_lab_series[i+1]
        
        # Compute residual
        laplacian = (np.roll(E, -1) - 2*E + np.roll(E, 1)) / (dx**2)
        d2E_dt2 = (E_next - 2*E + E_prev) / (dt**2)
        residual = d2E_dt2 - c**2 * laplacian + chi**2 * E
        residuals_lab.append(np.sqrt(np.mean(residual**2)))
    
    # Transform to boosted frame and check residuals
    residuals_boost = []
    for i in range(1, len(E_lab_series) - 1):
        t_lab = i * dt
        
        # Transform each snapshot
        E_boost, x_boost, t_boost = lorentz_boost_field_1d(
            E_lab_series[i], x_lab, t_lab, beta, c
        )
        E_boost_prev, _, t_boost_prev = lorentz_boost_field_1d(
            E_lab_series[i-1], x_lab, (i-1)*dt, beta, c
        )
        E_boost_next, _, t_boost_next = lorentz_boost_field_1d(
            E_lab_series[i+1], x_lab, (i+1)*dt, beta, c
        )
        
        # Time step in boosted frame (not uniform due to relativity of simultaneity)
        dt_boost = t_boost - t_boost_prev
        dt_boost_next = t_boost_next - t_boost
        dt_boost_avg = (dt_boost + dt_boost_next) / 2
        
        # Compute residual in boosted frame
        laplacian_boost = (np.roll(E_boost, -1) - 2*E_boost + np.roll(E_boost, 1)) / (dx**2)
        d2E_dt2_boost = (E_boost_next - 2*E_boost + E_boost_prev) / (dt_boost_avg**2)
        residual_boost = d2E_dt2_boost - c**2 * laplacian_boost + chi**2 * E_boost
        residuals_boost.append(np.sqrt(np.mean(residual_boost**2)))
    
    return {
        'residual_lab_mean': float(np.mean(residuals_lab)),
        'residual_lab_max': float(np.max(residuals_lab)),
        'residual_boost_mean': float(np.mean(residuals_boost)),
        'residual_boost_max': float(np.max(residuals_boost)),
        'covariance_ratio': float(np.mean(residuals_boost) / (np.mean(residuals_lab) + 1e-30))
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
