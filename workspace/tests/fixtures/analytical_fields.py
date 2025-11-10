# -*- coding: utf-8 -*-
"""
Analytical test field generators with known energy properties.

These functions create synthetic fields for testing energy calculations,
with analytical solutions that can be compared to numerical results.
"""
import numpy as np
from typing import Tuple


def gaussian_packet_1d(N: int, dx: float, sigma: float = 1.0, 
                       amplitude: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Create 1D Gaussian packet with known energy.
    
    Field: E(x) = A * exp(-x²/(2σ²))
    Gradient: ∂E/∂x = -x/(σ²) * E(x)
    Energy density: e = ½[(∂E/∂x)² + 0] (no time derivative, no potential)
    
    Args:
        N: Number of grid points
        dx: Grid spacing
        sigma: Gaussian width
        amplitude: Peak amplitude
        
    Returns:
        E: Field array (N,)
        E_analytical: Analytical total energy
    """
    x = np.arange(N) * dx - (N * dx / 2)
    E = amplitude * np.exp(-x**2 / (2 * sigma**2))
    
    # Analytical gradient energy: ∫ ½(∂E/∂x)² dx
    # ∂E/∂x = -x/σ² * A*exp(-x²/2σ²)
    # (∂E/∂x)² = (x²/σ⁴) * A²*exp(-x²/σ²)
    # ∫ ½(x²/σ⁴)*A²*exp(-x²/σ²) dx = A²/(4σ²) * √(π*σ²)
    E_gradient = (amplitude**2) / (4 * sigma**2) * np.sqrt(np.pi * sigma**2)
    
    # Total energy (no kinetic, no potential)
    # Note: compute_field_energy() already multiplies by dV internally
    E_analytical = E_gradient
    
    return E, E_analytical


def uniform_field_1d(N: int, dx: float, amplitude: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Create uniform 1D field with zero gradient energy.
    
    Field: E(x) = A (constant)
    Gradient: ∂E/∂x = 0
    Energy: 0 (no gradients, no time derivative, no potential)
    
    Args:
        N: Number of grid points
        dx: Grid spacing
        amplitude: Field value
        
    Returns:
        E: Field array (N,)
        E_analytical: Analytical total energy (0)
    """
    E = np.full(N, amplitude, dtype=np.float64)
    return E, 0.0


def sine_wave_1d(N: int, dx: float, k: float, amplitude: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Create 1D sine wave with known gradient energy.
    
    Field: E(x) = A * sin(kx)
    Gradient: ∂E/∂x = Ak * cos(kx)
    Energy density: e = ½(Ak)² * cos²(kx)
    Total energy: ∫ e dx = ¼(Ak)² * L (where L = N*dx)
    
    Args:
        N: Number of grid points
        dx: Grid spacing
        k: Wave number
        amplitude: Peak amplitude
        
    Returns:
        E: Field array (N,)
        E_analytical: Analytical total energy
    """
    x = np.arange(N) * dx
    E = amplitude * np.sin(k * x)
    
    # Analytical gradient energy
    L = N * dx
    E_gradient = 0.25 * (amplitude * k)**2 * L
    
    return E, E_gradient


def gaussian_packet_2d(Nx: int, Ny: int, dx: float, sigma: float = 1.0,
                       amplitude: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Create 2D Gaussian packet with known energy.
    
    Field: E(x,y) = A * exp(-(x² + y²)/(2σ²))
    Energy: ∫∫ ½|∇E|² dxdy
    
    Args:
        Nx, Ny: Grid dimensions
        dx: Grid spacing (isotropic)
        sigma: Gaussian width
        amplitude: Peak amplitude
        
    Returns:
        E: Field array (Ny, Nx)
        E_analytical: Analytical total energy
    """
    x = np.arange(Nx) * dx - (Nx * dx / 2)
    y = np.arange(Ny) * dx - (Ny * dx / 2)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    r_sq = X**2 + Y**2
    E = amplitude * np.exp(-r_sq / (2 * sigma**2))
    
    # Analytical gradient energy in 2D
    # |∇E|² = (r²/σ⁴) * A² * exp(-r²/σ²)
    # ∫∫ ½|∇E|² dA = A²/(2σ²) * π * σ²
    E_gradient = (amplitude**2 / (2 * sigma**2)) * np.pi * sigma**2
    
    # Note: compute_field_energy() already multiplies by dV internally
    E_analytical = E_gradient
    
    return E, E_analytical


def gaussian_packet_3d(Nx: int, Ny: int, Nz: int, dx: float, 
                       sigma: float = 1.0, amplitude: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Create 3D Gaussian packet with known energy.
    
    Field: E(x,y,z) = A * exp(-(x² + y² + z²)/(2σ²))
    Energy: ∫∫∫ ½|∇E|² dV
    
    Args:
        Nx, Ny, Nz: Grid dimensions
        dx: Grid spacing (isotropic)
        sigma: Gaussian width
        amplitude: Peak amplitude
        
    Returns:
        E: Field array (Nz, Ny, Nx)
        E_analytical: Analytical total energy
    """
    x = np.arange(Nx) * dx - (Nx * dx / 2)
    y = np.arange(Ny) * dx - (Ny * dx / 2)
    z = np.arange(Nz) * dx - (Nz * dx / 2)
    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')
    
    r_sq = X**2 + Y**2 + Z**2
    E = amplitude * np.exp(-r_sq / (2 * sigma**2))
    
    # Analytical gradient energy in 3D (spherical coordinates)
    # |∇E|² = (r²/σ⁴) * A² * exp(-r²/σ²)
    # ∫∫∫ ½|∇E|² dV = (3π^(3/2) A² σ) / 4
    E_gradient = (3 * amplitude**2 * sigma * np.pi**(3/2)) / 4
    
    # Note: compute_field_energy() already multiplies by dV internally
    E_analytical = E_gradient
    
    return E, E_analytical


def moving_packet_1d(N: int, dx: float, dt: float, v: float = 0.5,
                     sigma: float = 1.0, amplitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Create moving 1D Gaussian packet with kinetic energy.
    
    E(x,t) = A * exp(-(x - vt)²/(2σ²))
    E_prev at t-dt, E at t
    Kinetic energy: ½(∂E/∂t)² = ½v²(x-vt)²/σ⁴ * E²
    
    Args:
        N: Number of grid points
        dx: Grid spacing
        dt: Time step
        v: Velocity
        sigma: Gaussian width
        amplitude: Peak amplitude
        
    Returns:
        E: Field at t (N,)
        E_prev: Field at t-dt (N,)
        E_analytical: Analytical kinetic energy
    """
    x = np.arange(N) * dx - (N * dx / 2)
    
    E = amplitude * np.exp(-(x - 0)**2 / (2 * sigma**2))
    E_prev = amplitude * np.exp(-(x + v*dt)**2 / (2 * sigma**2))
    
    # Kinetic energy contribution
    # ∂E/∂t ≈ (E - E_prev)/dt
    E_dot = (E - E_prev) / dt
    E_kinetic = 0.5 * np.sum(E_dot**2) * dx
    
    return E, E_prev, E_kinetic
