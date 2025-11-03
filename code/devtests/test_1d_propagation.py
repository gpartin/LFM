# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""Minimal 1D wave propagation test to debug GRAV-12 issue."""
import numpy as np
import math

# Import core solver
from lfm_equation import lattice_step

# Simple 1D test: traveling wave packet
N = 256
dx = 1.0
dt = 0.01
c = 1.0
alpha = c**2
beta = 1.0

# Wave parameters
kx = 0.1 * np.pi / dx  # Long wavelength
omega = math.sqrt(c**2 * kx**2)  # Free wave dispersion

# Initial packet: Gaussian envelope × cos wave
x = np.arange(N, dtype=np.float64)
x_center = 20.0
sigma = 15.0
env = np.exp(-((x - x_center)**2) / (2.0 * sigma**2))

# Traveling wave initialization with velocity
cos_spatial = np.cos(kx * x)
sin_spatial = np.sin(kx * x)
amplitude = 0.02

E0 = amplitude * env * cos_spatial
E_dot = amplitude * env * omega * sin_spatial
Eprev0 = E0 - dt * E_dot

print(f"1D wave test: N={N}, dx={dx}, dt={dt}, c={c}")
print(f"Wave: kx={kx:.6f}, omega={omega:.6f}, wavelength={2*np.pi/kx:.2f} cells")
print(f"Packet: center={x_center:.1f}, sigma={sigma:.1f}")
print(f"Initial E0 range: [{E0.min():.6f}, {E0.max():.6f}]")
print(f"Initial velocity (E0-Eprev)/dt range: [{((E0-Eprev0)/dt).min():.6f}, {((E0-Eprev0)/dt).max():.6f}]")
print(f"Peak position at t=0: x={np.argmax(np.abs(E0))}")

# Setup params for lattice_step
# Test with chi field (uniform background)
chi_bg = 0.05
chi_field = np.full(N, chi_bg, dtype=np.float64)
params = dict(
    dt=dt, dx=dx, alpha=alpha, beta=beta,
    chi=chi_field,  # Uniform chi field
    boundary="periodic"
)
print(f"Chi field: shape={chi_field.shape}, range=[{chi_field.min():.6f}, {chi_field.max():.6f}]")

# Evolve
E, Ep = E0.copy(), Eprev0.copy()
steps = 500
track_positions = []

for n in range(steps):
    E_next = lattice_step(E, Ep, params)
    Ep, E = E, E_next
    
    # Track peak position
    x_peak = np.argmax(np.abs(E))
    track_positions.append(x_peak)
    
    if n % 100 == 0:
        print(f"Step {n}: peak at x={x_peak}, amplitude={np.abs(E[x_peak]):.6f}")

print(f"\nFinal peak position: x={track_positions[-1]} (started at x={track_positions[0]})")
print(f"Travel distance: {track_positions[-1] - track_positions[0]} cells in {steps} steps")
print(f"Expected travel (v=c): {c * dt * steps / dx:.1f} cells")

# Check if packet moved
moved = (track_positions[-1] != track_positions[0])
print(f"\n{'✓ PASS' if moved else '✗ FAIL'}: Packet {'moved' if moved else 'did not move'}")
