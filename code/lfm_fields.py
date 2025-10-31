#!/usr/bin/env python3
"""
LFM Field Initialization Utilities
==================================
Standard field initialization patterns for test scenarios.

Provides:
- Gaussian fields (1D, 2D, 3D)
- Wave packets (modulated Gaussians)
- Traveling wave initialization
"""

import math
import numpy as np


def gaussian_field(shape, center=None, width=1.0, amplitude=1.0, xp=None):
    """Create N-dimensional Gaussian field.
    
    Args:
        shape: Tuple of grid dimensions, e.g., (N,) or (Nx, Ny) or (Nx, Ny, Nz)
        center: Center position (defaults to grid center). 
                Tuple of coordinates or None.
        width: Gaussian width in grid cells (standard deviation)
        amplitude: Peak amplitude at center
        xp: Backend module (np or cp). If None, uses NumPy.
    
    Returns:
        Gaussian field as ndarray with specified shape
    
    Examples:
        >>> # 1D Gaussian
        >>> field_1d = gaussian_field((128,), width=10.0)
        
        >>> # 2D Gaussian at custom center
        >>> field_2d = gaussian_field((64, 64), center=(32, 16), width=5.0)
        
        >>> # 3D Gaussian on GPU
        >>> import cupy as cp
        >>> field_3d = gaussian_field((32, 32, 32), xp=cp)
    """
    if xp is None:
        xp = np
    
    ndim = len(shape)
    
    if center is None:
        center = tuple((n - 1) / 2.0 for n in shape)
    
    # Create coordinate axes
    axes = [xp.arange(n, dtype=xp.float64) for n in shape]
    
    # Compute squared radial distance from center
    r_squared = xp.zeros(shape, dtype=xp.float64)
    
    for i, (ax, c) in enumerate(zip(axes, center)):
        # Create proper broadcasting shape for this axis
        ax_shape = [1] * ndim
        ax_shape[i] = shape[i]
        ax_broadcast = ax.reshape(ax_shape)
        
        # Add contribution from this dimension
        r_squared += (ax_broadcast - c) ** 2
    
    return amplitude * xp.exp(-r_squared / (2.0 * width ** 2))


def wave_packet(shape, kvec, center=None, width=1.0, amplitude=1.0, phase=0.0, xp=None):
    """Create modulated Gaussian wave packet.
    
    Constructs a Gaussian envelope multiplied by a plane wave:
        Ψ(r) = A exp(-|r-r₀|²/2σ²) cos(k·(r-r₀) + φ)
    
    Args:
        shape: Grid dimensions tuple
        kvec: Wave vector. Scalar for 1D, array/list for multi-D
        center: Center position (defaults to grid center)
        width: Gaussian envelope width (standard deviation)
        amplitude: Peak amplitude
        phase: Phase offset in radians
        xp: Backend module (np or cp)
    
    Returns:
        Real-valued wave packet field
    
    Examples:
        >>> # 1D wave packet
        >>> packet_1d = wave_packet((256,), kvec=0.5, width=20.0)
        
        >>> # 2D diagonal wave
        >>> packet_2d = wave_packet((128, 128), kvec=[0.3, 0.3], width=15.0)
        
        >>> # 3D packet with phase shift
        >>> packet_3d = wave_packet((64, 64, 64), kvec=[0.2, 0, 0], 
        ...                         phase=np.pi/4, width=10.0)
    """
    if xp is None:
        xp = np
    
    # Create Gaussian envelope
    envelope = gaussian_field(shape, center, width, amplitude, xp)
    
    # Prepare wave vector
    ndim = len(shape)
    if not hasattr(kvec, '__len__'):
        kvec = [kvec] + [0.0] * (ndim - 1)
    
    if center is None:
        center = tuple((n - 1) / 2.0 for n in shape)
    
    # Compute phase k·r
    axes = [xp.arange(n, dtype=xp.float64) for n in shape]
    phase_field = xp.zeros(shape, dtype=xp.float64)
    
    for i, (ax, k, c) in enumerate(zip(axes, kvec, center)):
        ax_shape = [1] * ndim
        ax_shape[i] = shape[i]
        ax_broadcast = ax.reshape(ax_shape)
        phase_field += k * (ax_broadcast - c)
    
    return envelope * xp.cos(phase_field + phase)


def traveling_wave_init(E0, kvec, omega, dt, xp=None):
    """Create E and E_prev for traveling wave with correct initial velocity.
    
    For leapfrog time integration, we need both E(t=0) and E(t=-dt).
    This function creates initial conditions for a wave propagating with
    dispersion ω = ω(k).
    
    The wave at t=-dt is phase-shifted: E(t=-dt) = E₀ cos(-ωdt)
    For smooth packets, this approximates the traveling wave.
    
    Args:
        E0: Initial field at t=0
        kvec: Wave vector (determines propagation direction)
        omega: Angular frequency (rad/s)
        dt: Time step
        xp: Backend module
    
    Returns:
        Tuple (E, E_prev) for use in leapfrog integration
    
    Example:
        >>> E0 = wave_packet((128,), kvec=0.5, width=20.0)
        >>> omega = np.sqrt(0.5**2 + 0.1**2)  # ω² = c²k² + χ²
        >>> E, E_prev = traveling_wave_init(E0, 0.5, omega, dt=0.01)
    """
    if xp is None:
        xp = np
    
    # Phase shift for time t = -dt
    phase_shift = -omega * dt
    
    # For smooth initial conditions, approximate by scaling
    E_prev = E0 * xp.cos(phase_shift)
    
    return E0, E_prev


def plane_wave_1d(N, k, amplitude=1.0, phase=0.0, xp=None):
    """Create 1D plane wave: A cos(kx + φ).
    
    Args:
        N: Grid size
        k: Wave number (in units of 2π/N)
        amplitude: Wave amplitude
        phase: Phase offset
        xp: Backend module
    
    Returns:
        1D plane wave array
    """
    if xp is None:
        xp = np
    
    x = xp.arange(N, dtype=xp.float64)
    return amplitude * xp.cos(k * x + phase)


def gaussian_bump_3d(shape, center, width, amplitude=1.0, xp=None):
    """Create 3D Gaussian bump (convenience wrapper for 3D case).
    
    Args:
        shape: 3-tuple (Nx, Ny, Nz)
        center: 3-tuple (cx, cy, cz) of center coordinates
        width: Gaussian width
        amplitude: Peak amplitude
        xp: Backend module
    
    Returns:
        3D Gaussian field
    """
    if len(shape) != 3:
        raise ValueError("gaussian_bump_3d requires 3D shape")
    return gaussian_field(shape, center, width, amplitude, xp)


def zero_mean_field(field, xp=None):
    """Remove mean from field (useful for controlling energy).
    
    Args:
        field: Input field array
        xp: Backend module (auto-detected if None)
    
    Returns:
        Zero-mean field
    """
    if xp is None:
        if hasattr(field, '__array_interface__'):
            xp = np
        else:
            try:
                import cupy as cp
                if isinstance(field, cp.ndarray):
                    xp = cp
                else:
                    xp = np
            except ImportError:
                xp = np
    
    return field - xp.mean(field)


def normalize_energy(E, E_prev, target_energy, dt, dx, c, chi, xp=None):
    """Scale field to achieve target total energy.
    
    Args:
        E: Current field
        E_prev: Previous field
        target_energy: Desired total energy
        dt: Time step
        dx: Spatial step
        c: Speed of light
        chi: Mass parameter
        xp: Backend module
    
    Returns:
        Tuple (E_scaled, E_prev_scaled)
    """
    if xp is None:
        if hasattr(E, '__array_interface__'):
            xp = np
        else:
            try:
                import cupy as cp
                if isinstance(E, cp.ndarray):
                    xp = cp
                else:
                    xp = np
            except ImportError:
                xp = np
    
    # Compute current energy (simple approximation)
    dE_dt = (E - E_prev) / dt
    kinetic = 0.5 * xp.sum(dE_dt ** 2) * (dx ** E.ndim)
    potential = 0.5 * (chi ** 2) * xp.sum(E ** 2) * (dx ** E.ndim)
    current_energy = kinetic + potential
    
    if abs(current_energy) < 1e-30:
        return E, E_prev
    
    # Scale factor
    scale = math.sqrt(abs(target_energy) / abs(current_energy))
    
    return E * scale, E_prev * scale
