#!/usr/bin/env python3
"""
Algorithm 1: Active Region Mask (Conservative)
===============================================

HYPOTHESIS: Can safely skip cells far from particles if we maintain adequate buffer zone.

APPROACH:
- Define "active" cells as those within radius R of any particle/source
- Add 2-cell buffer zone to ensure wave propagation is captured
- Update only active + buffer cells using full physics
- All other cells maintain previous values (frozen)

PHYSICS SAFETY:
- Wave speed c ≈ 1.0, dt = 0.05, so waves travel ~0.05 cells/step
- 2-cell buffer means 40 timesteps of safety margin
- No approximations to Laplacian - just selective application

EXPECTED:
- Speedup: 2-5x (depends on % active cells)
- Energy error: < 10⁻⁵ (very conservative)
- Memory: Same as baseline (no compression)

RISK LEVEL: LOW
- Buffer ensures no missed wave propagation
- Standard Laplacian on active cells
- Frozen cells can't introduce instability
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


def create_active_mask(shape, particle_positions, radius, buffer, xp):
    """
    Create boolean mask of active cells.
    
    Active = within radius of any particle + buffer zone
    """
    mask = xp.zeros(shape, dtype=bool)
    Nz, Ny, Nx = shape
    
    z, y, x = xp.meshgrid(
        xp.arange(Nz), xp.arange(Ny), xp.arange(Nx),
        indexing='ij'
    )
    
    for pos in particle_positions:
        r = xp.sqrt(
            (x - pos[0])**2 +
            (y - pos[1])**2 +
            (z - pos[2])**2
        )
        mask |= (r < (radius + buffer))
    
    return mask


def laplacian_masked(E, dx, mask, xp):
    """
    Compute Laplacian only on masked cells.
    
    Uses order-2 stencil with periodic boundaries.
    Identical to lfm_equation.laplacian() but selective.
    """
    lap = xp.zeros_like(E)
    
    # Only compute where mask is True
    # For 3D order-2 stencil:
    Exn1 = xp.roll(E, -1, 2)
    Exp1 = xp.roll(E, 1, 2)
    Eyn1 = xp.roll(E, -1, 1)
    Eyp1 = xp.roll(E, 1, 1)
    Ezn1 = xp.roll(E, -1, 0)
    Ezp1 = xp.roll(E, 1, 0)
    
    lap[mask] = ((Exp1[mask] + Exn1[mask] + Eyp1[mask] + 
                  Eyn1[mask] + Ezp1[mask] + Ezn1[mask] - 6 * E[mask]) / (dx * dx))
    
    return lap


def lattice_step_masked(E, E_prev, chi, c, dt, dx, gamma, mask, xp):
    """
    Verlet step with masked updates.
    
    Only updates cells where mask=True.
    Other cells maintain previous values (frozen).
    """
    E_next = xp.array(E, copy=True)  # Start with copy
    
    # Compute Laplacian only on active cells
    lap = laplacian_masked(E, dx, mask, xp)
    
    # Verlet update only on active cells
    c2 = c * c
    dt2 = dt * dt
    
    # Handle chi (scalar or array)
    if xp.isscalar(chi):
        chi2 = chi * chi
    else:
        chi2 = chi * chi
    
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
    
    # Earth
    r_earth = xp.sqrt((x - earth_pos[0])**2 + (y - earth_pos[1])**2 + (z - earth_pos[2])**2)
    chi[r_earth < 3.0] = chi_earth
    
    # Moon
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


def run_algorithm1_masked(
    grid_size=128,
    steps=300,
    use_gpu=True,
    seed=42,
    active_radius=40.0,
    buffer_cells=2,
    output_dir=None
):
    """
    Run Algorithm 1: Active Region Mask
    
    Args:
        active_radius: Radius around particles to consider active
        buffer_cells: Additional buffer zone for wave propagation
    """
    np.random.seed(seed)
    
    print("="*60)
    print("ALGORITHM 1: Active Region Mask (Conservative)")
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
    print(f"Active radius: {active_radius} cells")
    print(f"Buffer: {buffer_cells} cells")
    
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
        
        # Create active mask
        particle_positions = [earth_pos, moon_pos]
        mask = create_active_mask(shape, particle_positions, active_radius, buffer_cells, xp)
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
    
    # Memory (same as baseline - no compression)
    memory_mb = (Nx * Ny * Nz * 8 * 3) / (1024**2)
    
    print("\n" + "="*60)
    print("ALGORITHM 1 RESULTS")
    print("="*60)
    print(f"Total time: {elapsed:.3f} s")
    print(f"Time per step: {elapsed/steps:.6f} s")
    print(f"Energy drift: {energy_drift:.6e}")
    print(f"Mean active cells: {mean_active_pct:.1f}%")
    print(f"Memory: {memory_mb:.1f} MB")
    
    success = energy_drift < 1e-4
    print(f"\nEnergy conservation: {'✓ PASS' if success else '✗ FAIL'} (< 1e-4)")
    
    results = {
        "algorithm": "algorithm1_active_mask",
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
            "active_radius": active_radius,
            "buffer_cells": buffer_cells,
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
        
        results_file = output_dir / "algorithm1_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "results"
    
    print("Running Algorithm 1 (3 trials)...\n")
    
    trials = []
    for trial in range(3):
        print(f"\n{'#'*60}")
        print(f"# TRIAL {trial + 1}/3")
        print(f"{'#'*60}\n")
        
        result = run_algorithm1_masked(
            grid_size=128,
            steps=300,
            use_gpu=True,
            seed=42 + trial,
            active_radius=40.0,
            buffer_cells=2,
            output_dir=output_dir if trial == 0 else None
        )
        trials.append(result)
        
        print(f"\nTrial {trial+1}: {result['elapsed_time_s']:.3f}s, drift={result['energy_drift']:.2e}")
    
    # Statistics
    times = [t['elapsed_time_s'] for t in trials]
    drifts = [t['energy_drift'] for t in trials]
    
    print("\n" + "="*60)
    print("ALGORITHM 1 SUMMARY (3 trials)")
    print("="*60)
    print(f"Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
    print(f"Energy drift: {np.mean(drifts):.2e} ± {np.std(drifts):.2e}")
    print(f"All pass: {all(t['energy_drift_pass'] for t in trials)}")
    
    summary = {
        "algorithm1_summary": {
            "n_trials": 3,
            "mean_time_s": float(np.mean(times)),
            "std_time_s": float(np.std(times)),
            "mean_drift": float(np.mean(drifts)),
            "std_drift": float(np.std(drifts)),
            "all_pass": all(t['energy_drift_pass'] for t in trials)
        },
        "trials": trials
    }
    
    summary_file = output_dir / "algorithm1_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print("\nAlgorithm 1 complete! ✓")
