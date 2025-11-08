#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
Algorithm 2: Gradient-Adaptive Culling (Moderate)
==================================================

HYPOTHESIS: Cells with negligible field AND gradient don't need updates (flat → stable).

APPROACH:
- Track cells where |∇E| > grad_threshold AND |E| > field_threshold
- Include 1-cell halo around active cells (immediate neighbors)
- Dynamically update mask each timestep based on actual field state
- More responsive than fixed radius, adapts to wave propagation

PHYSICS SAFETY:
- Gradient threshold catches propagating waves (∇E ≠ 0)
- Field threshold catches standing waves (E ≠ 0)
- 1-cell halo ensures neighbors of active cells updated
- Threshold tuning critical: too high → miss physics, too low → no speedup

EXPECTED:
- Speedup: 3-8x (depends on wave localization)
- Energy error: 10⁻⁴ to 10⁻⁵ (threshold-dependent)
- Memory: Same as baseline

RISK LEVEL: MEDIUM
- Threshold choice affects physics accuracy
- Gradient computation adds overhead
- Could miss slow-building resonances if thresholds too aggressive
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.lfm_equation import energy_total
from core.lfm_backend import pick_backend, to_numpy

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def compute_gradient_magnitude(E, dx, xp):
    """
    Compute |∇E| using central differences.
    
    Returns magnitude of gradient at each point.
    """
    # Central differences
    gx = (xp.roll(E, -1, 2) - xp.roll(E, 1, 2)) / (2 * dx)
    gy = (xp.roll(E, -1, 1) - xp.roll(E, 1, 1)) / (2 * dx)
    gz = (xp.roll(E, -1, 0) - xp.roll(E, 1, 0)) / (2 * dx)
    
    grad_mag = xp.sqrt(gx**2 + gy**2 + gz**2)
    return grad_mag


def create_adaptive_mask(E, grad_mag, field_threshold, grad_threshold, halo_size, xp):
    """
    Create mask based on field and gradient thresholds.
    
    Active if: |E| > field_thresh OR |∇E| > grad_thresh
    Then expand by halo_size cells.
    """
    # Initial mask: significant field or gradient
    mask = (xp.abs(E) > field_threshold) | (grad_mag > grad_threshold)
    
    # Expand by halo (dilate mask)
    for _ in range(halo_size):
        mask_expanded = xp.array(mask, copy=True)
        for axis in range(3):
            mask_expanded |= xp.roll(mask, 1, axis)
            mask_expanded |= xp.roll(mask, -1, axis)
        mask = mask_expanded
    
    return mask


def laplacian_masked(E, dx, mask, xp):
    """Compute Laplacian only on masked cells (order-2 3D stencil)"""
    lap = xp.zeros_like(E)
    
    Exn1 = xp.roll(E, -1, 2); Exp1 = xp.roll(E, 1, 2)
    Eyn1 = xp.roll(E, -1, 1); Eyp1 = xp.roll(E, 1, 1)
    Ezn1 = xp.roll(E, -1, 0); Ezp1 = xp.roll(E, 1, 0)
    
    lap[mask] = ((Exp1[mask] + Exn1[mask] + Eyp1[mask] + 
                  Eyn1[mask] + Ezp1[mask] + Ezn1[mask] - 6 * E[mask]) / (dx * dx))
    
    return lap


def lattice_step_masked(E, E_prev, chi, c, dt, dx, gamma, mask, xp):
    """Verlet step with masked updates"""
    E_next = xp.array(E, copy=True)
    
    lap = laplacian_masked(E, dx, mask, xp)
    
    c2 = c * c
    dt2 = dt * dt
    chi2 = chi * chi if xp.isscalar(chi) else chi * chi
    
    term_wave = c2 * lap
    term_mass = -chi2 * E
    
    E_next[mask] = ((2.0 - gamma) * E[mask] - 
                    (1.0 - gamma) * E_prev[mask] +
                    dt2 * (term_wave[mask] + term_mass[mask]))
    
    return E_next


def create_chi_field_orbit(shape, chi_earth, chi_moon, earth_pos, moon_pos, xp):
    """Create chi field with Earth and Moon"""
    chi = xp.zeros(shape, dtype=np.float64)
    Nz, Ny, Nx = shape
    
    z, y, x = xp.meshgrid(
        xp.arange(Nz), xp.arange(Ny), xp.arange(Nx),
        indexing='ij'
    )
    
    r_earth = xp.sqrt((x - earth_pos[0])**2 + (y - earth_pos[1])**2 + (z - earth_pos[2])**2)
    chi[r_earth < 3.0] = chi_earth
    
    r_moon = xp.sqrt((x - moon_pos[0])**2 + (y - moon_pos[1])**2 + (z - moon_pos[2])**2)
    chi[r_moon < 1.5] = chi_moon
    
    return chi


def update_moon_position_kinematic(moon_pos, moon_vel, earth_pos, k_acc, dt):
    """Update moon using kinematic gravity"""
    r_vec = moon_pos - earth_pos
    r = np.linalg.norm(r_vec)
    r_hat = r_vec / (r + 1e-10)
    a_mag = k_acc / (r**2 + 1.0)
    acc = -a_mag * r_hat
    moon_vel = moon_vel + acc * dt
    moon_pos = moon_pos + moon_vel * dt
    return moon_pos, moon_vel


def run_algorithm2_gradient_adaptive(
    grid_size=128,
    steps=300,
    use_gpu=True,
    seed=42,
    field_threshold=1e-6,
    grad_threshold=1e-6,
    halo_size=1,
    output_dir=None
):
    """
    Run Algorithm 2: Gradient-Adaptive Culling
    
    Args:
        field_threshold: Minimum |E| to consider active
        grad_threshold: Minimum |∇E| to consider active
        halo_size: Cells to expand mask (1 = immediate neighbors)
    """
    np.random.seed(seed)
    
    print("="*60)
    print("ALGORITHM 2: Gradient-Adaptive Culling (Moderate)")
    print("="*60)
    
    xp, on_gpu = pick_backend(use_gpu and HAS_CUPY)
    device_name = "GPU (CuPy)" if on_gpu else "CPU (NumPy)"
    print(f"Device: {device_name}")
    
    # Grid setup
    Nx = Ny = Nz = grid_size
    shape = (Nz, Ny, Nx)
    dx = 1.0
    dt = 0.05
    
    print(f"Grid: {Nx}×{Ny}×{Nz} = {Nx*Ny*Nz:,} cells")
    print(f"Steps: {steps}")
    print(f"Field threshold: {field_threshold:.1e}")
    print(f"Gradient threshold: {grad_threshold:.1e}")
    print(f"Halo size: {halo_size} cells")
    
    # Physics
    alpha = 1.0
    beta = 1.0
    c = np.sqrt(alpha / beta)
    gamma_damp = 0.0
    
    chi_earth = 50.0
    chi_moon = chi_earth / 81.0
    k_acc = 0.20
    
    earth_pos = np.array([Nz//2, Ny//2, Nx//2], dtype=np.float64)
    moon_pos = np.array([Nz//2, Ny//2 + 32, Nx//2], dtype=np.float64)
    moon_vel = np.array([0.0, 0.0, 1.86], dtype=np.float64)
    
    print(f"\nPhysics:")
    print(f"  chi_earth={chi_earth:.2f}, chi_moon={chi_moon:.3f}")
    print(f"  k_acc={k_acc}, c={c:.2f}")
    
    # Initialize fields
    E = xp.zeros(shape, dtype=np.float64)
    E_prev = xp.zeros(shape, dtype=np.float64)
    chi = create_chi_field_orbit(shape, chi_earth, chi_moon, earth_pos, moon_pos, xp)
    
    # Initial energy
    E0_energy = energy_total(
        to_numpy(E), to_numpy(E_prev), dt, dx, c,
        to_numpy(chi) if hasattr(chi, 'get') else chi
    )
    
    print(f"Initial energy: {E0_energy:.6e}")
    
    # Storage
    trajectory = [moon_pos.copy()]
    energies = [E0_energy]
    active_cell_counts = []
    
    print("\nStarting simulation...")
    t_start = time.perf_counter()
    
    for step in range(steps):
        # Update chi field
        chi = create_chi_field_orbit(shape, chi_earth, chi_moon, earth_pos, moon_pos, xp)
        
        # Compute gradient magnitude
        grad_mag = compute_gradient_magnitude(E, dx, xp)
        
        # Create adaptive mask based on current field state
        mask = create_adaptive_mask(E, grad_mag, field_threshold, grad_threshold, halo_size, xp)
        active_count = int(xp.sum(mask))
        active_cell_counts.append(active_count)
        
        # Masked lattice step
        E_next = lattice_step_masked(E, E_prev, chi, c, dt, dx, gamma_damp, mask, xp)
        
        # Swap
        E_prev = E
        E = E_next
        
        # Update moon
        moon_pos, moon_vel = update_moon_position_kinematic(
            moon_pos, moon_vel, earth_pos, k_acc, dt
        )
        trajectory.append(moon_pos.copy())
        
        # Energy monitoring
        if (step + 1) % 30 == 0 or step == steps - 1:
            E_current = energy_total(
                to_numpy(E), to_numpy(E_prev), dt, dx, c,
                to_numpy(chi) if hasattr(chi, 'get') else chi
            )
            energies.append(E_current)
            drift = abs((E_current - E0_energy) / (abs(E0_energy) + 1e-30))
            active_pct = 100.0 * active_count / (Nx * Ny * Nz)
            
            print(f"  Step {step+1:3d}/{steps}: E={E_current:.6e}, |ΔE/E₀|={drift:.2e}, active={active_pct:.1f}%")
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    
    # Final metrics
    E_final = energies[-1]
    energy_drift = abs((E_final - E0_energy) / (abs(E0_energy) + 1e-30))
    mean_active_pct = 100.0 * np.mean(active_cell_counts) / (Nx * Ny * Nz)
    
    memory_mb = (Nx * Ny * Nz * 8 * 3) / (1024**2)
    
    print("\n" + "="*60)
    print("ALGORITHM 2 RESULTS")
    print("="*60)
    print(f"Total time: {elapsed:.3f} s")
    print(f"Time per step: {elapsed/steps:.6f} s")
    print(f"Energy drift: {energy_drift:.6e}")
    print(f"Mean active cells: {mean_active_pct:.1f}%")
    print(f"Memory: {memory_mb:.1f} MB")
    
    success = energy_drift < 1e-4
    print(f"\nEnergy conservation: {'✓ PASS' if success else '✗ FAIL'} (< 1e-4)")
    
    results = {
        "algorithm": "algorithm2_gradient_adaptive",
        "device": device_name,
        "grid_size": grid_size,
        "steps": steps,
        "elapsed_time_s": elapsed,
        "time_per_step_s": elapsed / steps,
        "energy_initial": float(E0_energy),
        "energy_final": float(E_final),
        "energy_drift": float(energy_drift),
        "energy_drift_pass": success,
        "mean_active_pct": float(mean_active_pct),
        "memory_mb": memory_mb,
        "trajectory": [pos.tolist() for pos in trajectory],
        "energies": [float(e) for e in energies],
        "active_cell_counts": [int(c) for c in active_cell_counts],
        "params": {
            "field_threshold": field_threshold,
            "grad_threshold": grad_threshold,
            "halo_size": halo_size,
            "chi_earth": chi_earth,
            "chi_moon": chi_moon,
            "k_acc": k_acc,
            "dx": dx,
            "dt": dt,
            "seed": seed
        }
    }
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "algorithm2_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "results"
    
    print("Running Algorithm 2 (3 trials)...\n")
    
    trials = []
    for trial in range(3):
        print(f"\n{'#'*60}")
        print(f"# TRIAL {trial + 1}/3")
        print(f"{'#'*60}\n")
        
        result = run_algorithm2_gradient_adaptive(
            grid_size=128,
            steps=300,
            use_gpu=True,
            seed=42 + trial,
            field_threshold=1e-6,
            grad_threshold=1e-6,
            halo_size=1,
            output_dir=output_dir if trial == 0 else None
        )
        trials.append(result)
        
        print(f"\nTrial {trial+1}: {result['elapsed_time_s']:.3f}s, drift={result['energy_drift']:.2e}")
    
    # Statistics
    times = [t['elapsed_time_s'] for t in trials]
    drifts = [t['energy_drift'] for t in trials]
    
    print("\n" + "="*60)
    print("ALGORITHM 2 SUMMARY (3 trials)")
    print("="*60)
    print(f"Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
    print(f"Energy drift: {np.mean(drifts):.2e} ± {np.std(drifts):.2e}")
    print(f"All pass: {all(t['energy_drift_pass'] for t in trials)}")
    
    summary = {
        "algorithm2_summary": {
            "n_trials": 3,
            "mean_time_s": float(np.mean(times)),
            "std_time_s": float(np.std(times)),
            "mean_drift": float(np.mean(drifts)),
            "std_drift": float(np.std(drifts)),
            "all_pass": all(t['energy_drift_pass'] for t in trials)
        },
        "trials": trials
    }
    
    summary_file = output_dir / "algorithm2_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print("\nAlgorithm 2 complete! ✓")
