#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
Algorithm 1: Active Field Mask (Wave Packet Version)
====================================================

HYPOTHESIS: Can safely skip cells with negligible field values.

APPROACH:
- Define "active" cells as |E| > threshold OR |E_prev| > threshold
- Add 3-cell buffer zone to capture wave propagation
- Update only active + buffer cells using full physics
- Frozen cells maintain previous values

EXPECTED:
- Speedup depends on active fraction
- Energy error: < 5e-4 (same as baseline)
- Conservative approach - large buffer ensures safety
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.lfm_equation import energy_total, laplacian
from core.lfm_backend import pick_backend, to_numpy

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def create_gaussian_wave_packet(shape, center, width, amplitude, k_vec, omega, dt, xp):
    """Create Gaussian wave packet (same as baseline)"""
    Nz, Ny, Nx = shape
    z, y, x = xp.meshgrid(xp.arange(Nz), xp.arange(Ny), xp.arange(Nx), indexing='ij')
    dz, dy, dx = z - center[0], y - center[1], x - center[2]
    r_sq = dz**2 + dy**2 + dx**2
    envelope = amplitude * xp.exp(-r_sq / (2 * width**2))
    k_dot_x = k_vec[0]*z + k_vec[1]*y + k_vec[2]*x
    E = envelope * xp.cos(k_dot_x)
    E_prev = envelope * xp.cos(k_dot_x + omega * dt)
    return E, E_prev


def create_active_mask_from_field(E, E_prev, threshold, buffer_cells, xp):
    """
    Create boolean mask where field is significant.
    
    Active = |E| > threshold OR |E_prev| > threshold
    Then dilate by buffer_cells to capture wave front.
    """
    # Core active region
    mask = (xp.abs(E) > threshold) | (xp.abs(E_prev) > threshold)
    
    # Dilate mask by buffer_cells using 6-neighborhood rolls (backend-agnostic)
    def dilate6(m):
        return (
            m
            | xp.roll(m, 1, 0) | xp.roll(m, -1, 0)
            | xp.roll(m, 1, 1) | xp.roll(m, -1, 1)
            | xp.roll(m, 1, 2) | xp.roll(m, -1, 2)
        )

    for _ in range(max(0, int(buffer_cells))):
        mask = dilate6(mask)
    
    return mask


def laplacian_full(E, dx, xp):
    """Delegate to canonical laplacian implementation from core.lfm_equation."""
    # core.lfm_equation.laplacian expects numpy/cupy array and dx
    return laplacian(E, dx)


def lattice_step_masked(E, E_prev, chi, c, dt, dx, gamma, mask, xp):
    """Verlet step with masked updates."""
    E_next = xp.array(E, copy=True)
    
    # Compute canonical Laplacian on full grid, then use mask for updates
    lap = laplacian_full(E, dx, xp)
    
    # Verlet update
    c2 = c * c
    dt2 = dt * dt
    
    # Compute chi squared term
    if xp.isscalar(chi):
        chi2 = chi * chi
    else:
        chi2 = chi * chi
    
    term_mass = -chi2 * E
    term_wave = c2 * lap
    
    # Update only masked cells (ensure proper broadcasting)
    update_value = ((2.0 - gamma) * E - 
                    (1.0 - gamma) * E_prev +
                    dt2 * (term_wave + term_mass))
    
    # Apply mask
    E_next = xp.where(mask, update_value, E)
    
    return E_next


def measure_active_fraction(mask):
    """Measure what fraction of grid is active."""
    if hasattr(mask, 'get'):
        mask_np = mask.get()
    else:
        mask_np = mask
    
    active_cells = np.sum(mask_np)
    total_cells = mask_np.size
    return active_cells / total_cells, active_cells, total_cells


def run_algorithm1_wave(
    grid_size=128,
    steps=300,
    use_gpu=True,
    seed=42,
    threshold=1e-6,
    buffer_cells=3,
    output_dir=None
):
    """Run Algorithm 1 on wave packet."""
    np.random.seed(seed)
    
    print("="*60)
    print("ALGORITHM 1: Active Field Mask (Wave Packet)")
    print("="*60)
    
    xp, on_gpu = pick_backend(use_gpu and HAS_CUPY)
    device_name = "GPU (CuPy)" if on_gpu else "CPU (NumPy)"
    print(f"Device: {device_name}")
    
    # Grid setup
    Nx = Ny = Nz = grid_size
    shape = (Nz, Ny, Nx)
    dx = 1.0
    dt = 0.05
    
    print(f"Grid: {Nx}x{Ny}x{Nz} = {Nx*Ny*Nz:,} cells")
    print(f"Steps: {steps}")
    print(f"Field threshold: {threshold:.2e}")
    print(f"Buffer: {buffer_cells} cells")
    
    # Physics
    alpha = 1.0
    beta = 1.0
    c = np.sqrt(alpha / beta)
    gamma_damp = 0.0
    
    # Wave packet (same as baseline)
    center = np.array([Nz//2, Ny//2, Nx//2], dtype=np.float64)
    width = 20.0
    amplitude = 0.01
    k_mag = 2*np.pi / 32.0
    k_vec = np.array([0.0, 0.0, k_mag])
    omega = c * k_mag
    
    print(f"\nWave Packet:")
    print(f"  Width: {width:.1f} cells")
    print(f"  Amplitude: {amplitude:.2f}")
    print(f"  Wavelength: {2*np.pi/k_mag:.1f} cells")
    
    # Initialize fields
    E, E_prev = create_gaussian_wave_packet(shape, center, width, amplitude, k_vec, omega, dt, xp)
    chi = xp.zeros(shape, dtype=np.float64)
    
    # Initial energy
    E0_energy = energy_total(
        to_numpy(E), to_numpy(E_prev), dt, dx, c,
        to_numpy(chi) if hasattr(chi, 'get') else chi
    )
    
    print(f"\nInitial energy: {E0_energy:.6e}")
    
    # Storage
    energies = [E0_energy]
    active_fractions = []
    
    print("\nStarting simulation...")
    t_start = time.perf_counter()
    
    for step in range(steps):
        # Create active mask based on current field
        mask = create_active_mask_from_field(E, E_prev, threshold, buffer_cells, xp)
        frac, active, total = measure_active_fraction(mask)
        active_fractions.append(frac)
        
        # Masked lattice step
        E_next = lattice_step_masked(E, E_prev, chi, c, dt, dx, gamma_damp, mask, xp)
        
        # Swap fields
        E_prev = E
        E = E_next
        
        # Monitoring
        if (step + 1) % 30 == 0 or step == steps - 1:
            E_current = energy_total(
                to_numpy(E), to_numpy(E_prev), dt, dx, c,
                to_numpy(chi) if hasattr(chi, 'get') else chi
            )
            energies.append(E_current)
            drift = abs((E_current - E0_energy) / (abs(E0_energy) + 1e-30))
            
            print(f"  Step {step+1:3d}/{steps}: E={E_current:.6e}, |dE/E0|={drift:.2e}, active={frac*100:.1f}%")
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    
    # Final metrics
    E_final = energies[-1]
    energy_drift = abs((E_final - E0_energy) / (abs(E0_energy) + 1e-30))
    frac_mean = np.mean(active_fractions)
    
    # Memory (estimate)
    bytes_per_cell = 8
    fields_count = 3
    memory_mb = (Nx * Ny * Nz * bytes_per_cell * fields_count) / (1024**2)
    
    print("\n" + "="*60)
    print("ALGORITHM 1 RESULTS")
    print("="*60)
    print(f"Total time: {elapsed:.3f} s")
    print(f"Time per step: {elapsed/steps:.6f} s")
    print(f"Energy drift: {energy_drift:.6e}")
    print(f"Mean active cells: {frac_mean*100:.1f}%")
    print(f"Memory estimate: {memory_mb:.1f} MB")
    
    success = energy_drift < 5e-4
    print(f"\nEnergy conservation: {'PASS' if success else 'FAIL'} (< 5e-4)")
    
    # Baseline comparison (load from results)
    baseline_file = Path(__file__).parent / "results" / "baseline_wave_summary.json"
    if baseline_file.exists():
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline = json.load(f)
        baseline_time = baseline['baseline_wave_summary']['mean_time_s']
        speedup = baseline_time / elapsed
        print(f"Speedup vs baseline: {speedup:.2f}x ({baseline_time:.3f}s -> {elapsed:.3f}s)")
    else:
        speedup = None
        print("Baseline not found - cannot compute speedup")
    
    # Results
    results = {
        "algorithm": "algorithm1_active_mask_wave",
        "device": device_name,
        "grid_size": grid_size,
        "total_cells": Nx * Ny * Nz,
        "steps": steps,
        "elapsed_time_s": elapsed,
        "time_per_step_s": elapsed / steps,
        "energy_initial": float(E0_energy),
        "energy_final": float(E_final),
        "energy_drift": float(energy_drift),
        "energy_drift_pass": success,
        "memory_mb": memory_mb,
        "active_fraction_mean": float(frac_mean),
        "speedup_vs_baseline": float(speedup) if speedup else None,
        "params": {
            "threshold": threshold,
            "buffer_cells": buffer_cells,
            "wave_width": width,
            "wave_amplitude": amplitude,
            "dx": dx,
            "dt": dt,
            "seed": seed
        }
    }
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "algorithm1_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "results"
    
    print("Running Algorithm 1 (Active Field Mask) - 3 trials\n")
    
    trials = []
    for trial in range(3):
        print(f"\n{'#'*60}")
        print(f"# TRIAL {trial + 1}/3")
        print(f"{'#'*60}\n")
        
        result = run_algorithm1_wave(
            grid_size=128,
            steps=300,
            use_gpu=True,
            seed=42 + trial,
            threshold=1e-6,
            buffer_cells=3,
            output_dir=output_dir if trial == 0 else None
        )
        trials.append(result)
        
        print(f"\nTrial {trial+1} complete: {result['elapsed_time_s']:.3f}s, drift={result['energy_drift']:.2e}, active={result['active_fraction_mean']*100:.1f}%")
    
    # Summary
    times = [t['elapsed_time_s'] for t in trials]
    drifts = [t['energy_drift'] for t in trials]
    actives = [t['active_fraction_mean'] for t in trials]
    speedups = [t['speedup_vs_baseline'] for t in trials if t['speedup_vs_baseline']]
    
    print("\n" + "="*60)
    print("ALGORITHM 1 SUMMARY (3 trials)")
    print("="*60)
    print(f"Time: {np.mean(times):.3f}s +/- {np.std(times):.3f}s")
    print(f"Energy drift: {np.mean(drifts):.2e} +/- {np.std(drifts):.2e}")
    print(f"Active fraction: {np.mean(actives)*100:.1f}% +/- {np.std(actives)*100:.1f}%")
    if speedups:
        print(f"Speedup: {np.mean(speedups):.2f}x +/- {np.std(speedups):.2f}x")
    print(f"All trials pass: {all(t['energy_drift_pass'] for t in trials)}")
    
    # Save summary
    summary = {
        "algorithm1_summary": {
            "n_trials": 3,
            "mean_time_s": float(np.mean(times)),
            "std_time_s": float(np.std(times)),
            "mean_drift": float(np.mean(drifts)),
            "std_drift": float(np.std(drifts)),
            "mean_active_fraction": float(np.mean(actives)),
            "std_active_fraction": float(np.std(actives)),
            "mean_speedup": float(np.mean(speedups)) if speedups else None,
            "all_pass": all(t['energy_drift_pass'] for t in trials)
        },
        "trials": trials
    }
    
    summary_file = output_dir / "algorithm1_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print("\nAlgorithm 1 complete!")

