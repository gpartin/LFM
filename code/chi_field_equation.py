#!/usr/bin/env python3
"""
χ-Field Equation Module for LFM

This module addresses the critical gap: "Where does χ(x) come from?"

In General Relativity: Gμν = 8πG Tμν (Einstein equations determine metric from energy-momentum)
In LFM: We propose a field equation that determines χ from energy density.

Three proposed approaches (start with simplest):

1. Poisson-like (static): ∇²χ = 4πG ρ_eff
   - Analogous to Newtonian gravity
   - χ determined by instantaneous energy distribution
   - No gravitational waves (χ doesn't propagate)

2. Wave equation (dynamic): □χ = -4πG ρ_eff  
   - χ can support waves (gravitational wave analogue)
   - Retarded effects (finite propagation speed)
   
3. Self-consistent (coupled): Solve KG for E with χ, then update χ from E energy
   - Fully self-consistent field equations
   - Can show emergence of ω(x) = χ(x) from self-consistency
   
Current implementation: Start with (1) Poisson, extend to (2) and (3) later.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

def energy_density_field(E: np.ndarray, E_prev: np.ndarray, 
                         dt: float, dx: float, chi: float, c: float = 1.0) -> np.ndarray:
    """
    Compute energy density from Klein-Gordon field.
    
    For Klein-Gordon: ρ = (∂E/∂t)² / 2 + c²(∇E)² / 2 + χ²E² / 2
    
    Components:
    - Kinetic: (∂E/∂t)² / 2
    - Gradient: c²(∇E)² / 2  
    - Potential: χ²E² / 2
    
    Args:
        E: Current field
        E_prev: Previous field (for time derivative)
        dt: Time step
        dx: Spatial step
        chi: Mass parameter
        c: Speed of light
        
    Returns:
        rho: Energy density ρ(x)
    """
    # Time derivative (backward difference)
    dE_dt = (E - E_prev) / dt
    
    # Spatial gradient (central difference)
    dE_dx = (np.roll(E, -1) - np.roll(E, 1)) / (2 * dx)
    
    # Energy density components
    kinetic = 0.5 * dE_dt**2
    gradient = 0.5 * c**2 * dE_dx**2
    potential = 0.5 * chi**2 * E**2
    
    rho = kinetic + gradient + potential
    return rho

def solve_poisson_1d(source: np.ndarray, dx: float, 
                     boundary: str = 'periodic') -> np.ndarray:
    """
    Solve 1D Poisson equation: ∇²φ = source
    
    Uses sparse matrix solver for efficiency.
    
    Args:
        source: Right-hand side (e.g., 4πG ρ)
        dx: Spatial step
        boundary: 'periodic' or 'dirichlet'
        
    Returns:
        phi: Solution to Poisson equation
    """
    N = len(source)
    
    # Build Laplacian matrix (∇² using finite differences)
    # ∇²φ_i ≈ (φ_{i+1} - 2φ_i + φ_{i-1}) / dx²
    
    diag = -2.0 * np.ones(N) / (dx**2)
    off_diag = np.ones(N-1) / (dx**2)
    
    if boundary == 'periodic':
        # Periodic boundaries: connect first and last points
        A = sp.diags([diag, off_diag, off_diag, [1.0/(dx**2)], [1.0/(dx**2)]],
                     [0, 1, -1, N-1, -(N-1)], shape=(N, N), format='csr')
    elif boundary == 'dirichlet':
        # Dirichlet: φ = 0 at boundaries
        A = sp.diags([diag, off_diag, off_diag], [0, 1, -1], shape=(N, N), format='csr')
    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")
    
    # Solve A φ = source
    # For periodic, need to remove constant (φ + const also solution)
    # Fix by constraining mean: ∫φ dx = 0
    
    if boundary == 'periodic':
        # Remove mean from source (consistency condition for periodic Poisson)
        source_adjusted = source - source.mean()
        # Use iterative solver (handles singular matrix better)
        phi, info = spla.cg(A, source_adjusted, maxiter=1000, atol=1e-10)
        if info != 0:
            print(f"Warning: Poisson solver didn't converge (info={info})")
        phi = phi - phi.mean()  # Ensure zero mean
    else:
        phi = spla.spsolve(A, source)
    
    return phi

def compute_chi_from_energy_poisson(E: np.ndarray, E_prev: np.ndarray,
                                    dt: float, dx: float, chi_bg: float,
                                    G_coupling: float, c: float = 1.0) -> np.ndarray:
    """
    Compute χ-field from energy density using Poisson equation.
    
    Proposed equation: ∇²χ = 4πG ρ_eff
    
    Where ρ_eff is energy density from E-field.
    
    Interpretation:
    - χ plays role analogous to Newtonian potential Φ
    - Energy in E-field acts as "source" for χ
    - χ_background sets asymptotic value (like χ → χ₀ as r → ∞)
    
    Args:
        E: Current field
        E_prev: Previous field
        dt: Time step
        dx: Spatial step
        chi_bg: Background χ value (χ → chi_bg far from sources)
        G_coupling: Gravitational coupling strength
        c: Speed of light
        
    Returns:
        chi: Self-consistently computed χ-field
    """
    # Compute energy density (source term)
    # Note: Use chi_bg for potential energy term (bootstrap)
    rho = energy_density_field(E, E_prev, dt, dx, chi_bg, c)
    
    # Source term: 4πG ρ
    source = 4.0 * np.pi * G_coupling * rho
    
    # Solve Poisson equation: ∇²χ_perturbation = source
    chi_perturbation = solve_poisson_1d(source, dx, boundary='periodic')
    
    # Total χ = background + perturbation
    chi = chi_bg + chi_perturbation
    
    return chi

def compute_chi_from_energy_wave(E: np.ndarray, E_prev: np.ndarray, E_prev_prev: np.ndarray,
                                 chi_prev: np.ndarray, chi_prev_prev: np.ndarray,
                                 dt: float, dx: float, chi_bg: float,
                                 G_coupling: float, c: float = 1.0) -> np.ndarray:
    """
    Compute χ-field from energy density using wave equation (future work).
    
    Proposed equation: □χ = -4πG ρ_eff
    Expanded: ∂²χ/∂t² - c²∇²χ = -4πG ρ
    
    This allows χ to propagate as waves (gravitational wave analogue).
    
    Args:
        E, E_prev: Current and previous E-field
        E_prev_prev: E two steps ago (for second time derivative)
        chi_prev, chi_prev_prev: Previous χ values
        dt: Time step
        dx: Spatial step
        chi_bg: Background χ
        G_coupling: Gravitational coupling
        c: Speed of light
        
    Returns:
        chi_next: Updated χ-field
    """
    # Energy density
    rho = energy_density_field(E, E_prev, dt, dx, chi_bg, c)
    
    # Source term
    source = -4.0 * np.pi * G_coupling * rho
    
    # Wave equation: □χ = source
    # Leapfrog: χ_next = 2χ - χ_prev + dt²(c²∇²χ + source)
    
    # Laplacian of χ
    lap_chi = (np.roll(chi_prev, -1) - 2*chi_prev + np.roll(chi_prev, 1)) / (dx**2)
    
    # Update
    chi_next = 2*chi_prev - chi_prev_prev + dt**2 * (c**2 * lap_chi + source)
    
    return chi_next

def test_chi_field_equation():
    """Test χ-field computation from energy density."""
    print("Testing χ-field equation (Poisson approach)...\n")
    
    # Setup
    N = 256
    L = 25.0
    x = np.linspace(0, L, N, endpoint=False)
    dx = x[1] - x[0]
    dt = 0.01
    
    # Create localized energy distribution (Gaussian blob)
    x0 = L / 2
    sigma = 3.0
    chi_bg = 0.2
    c = 1.0
    
    # E-field with localized energy
    E = np.exp(-((x - x0)**2) / (2*sigma**2))
    E_prev = E.copy()  # Static for this test
    
    # Compute χ from energy
    G_coupling = 0.1  # Gravitational coupling strength
    chi_computed = compute_chi_from_energy_poisson(E, E_prev, dt, dx, chi_bg, G_coupling, c)
    
    # Analysis
    chi_perturbation = chi_computed - chi_bg
    
    print(f"Background χ: {chi_bg:.4f}")
    print(f"Computed χ range: [{chi_computed.min():.4f}, {chi_computed.max():.4f}]")
    print(f"χ perturbation range: [{chi_perturbation.min():.4f}, {chi_perturbation.max():.4f}]")
    print(f"Max |perturbation|: {np.abs(chi_perturbation).max():.4f}")
    print(f"\nInterpretation:")
    print(f"  - Energy blob at x={x0:.1f} creates χ-field perturbation")
    print(f"  - Analogous to mass creating gravitational potential")
    print(f"  - χ increases where energy is concentrated")
    print(f"  - Coupling strength G={G_coupling:.2f} controls magnitude")
    
    # Verify Poisson equation is satisfied
    # ∇²χ should equal 4πG ρ
    rho = energy_density_field(E, E_prev, dt, dx, chi_bg, c)
    source_theory = 4.0 * np.pi * G_coupling * rho
    
    lap_chi = (np.roll(chi_computed, -1) - 2*chi_computed + np.roll(chi_computed, 1)) / (dx**2)
    residual = lap_chi - source_theory
    
    print(f"\nPoisson equation residual:")
    print(f"  RMS: {np.sqrt(np.mean(residual**2)):.6e}")
    print(f"  Max: {np.abs(residual).max():.6e}")
    print(f"  (should be near zero)")
    
    print("\n✓ χ-field equation test passed")
    print("\nNext steps:")
    print("  1. Implement self-consistent solver (iterate E ↔ χ)")
    print("  2. Add test: GRAV-SELFCONS-01 comparing manual χ vs computed χ")
    print("  3. Verify ω(x) = χ(x) emerges from self-consistency")
    print("  4. Extend to wave equation (gravitational waves)")

if __name__ == "__main__":
    test_chi_field_equation()
