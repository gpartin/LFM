#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
Baseline Benchmark - Wave Packet Propagation
=============================================

NEW test case: Propagating Gaussian wave packet in 3D
This excites the E-field (unlike kinematic orbit) so field-based optimizations can work.

Test case: Gaussian wave packet
Grid: 128³ cells
Steps: 300 timesteps
Physics: Free-field wave propagation (chi=0 everywhere)

The wave packet starts localized and spreads as it propagates.
Field-based optimizations should be able to skip "empty" regions.

Metrics measured:
1. Energy conservation: |ΔE/E₀|
2. Wall-clock time: Total runtime
3. Memory usage: Peak allocation
4. Active cell percentage: How much of grid has significant E-field
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
import numpy as np

# Add project paths to import core and performance modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))  # core/*
sys.path.insert(0, str(Path(__file__).parents[2]))  # workspace root (for performance/*)

from core.lfm_equation import lattice_step, energy_total
from core.lfm_backend import pick_backend, to_numpy

# Try GPU first
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def create_gaussian_wave_packet(shape, center, width, amplitude, k_vec, omega, dt, xp):
    """
    Create a Gaussian wave packet: E(x,t) = A * exp(-r²/2σ²) * cos(k·x - ωt)
    
    Properly initialized for leapfrog time integration.
    
    Args:
        shape: (Nz, Ny, Nx) grid shape
        center: (z0, y0, x0) packet center
        width: σ - Gaussian width
        amplitude: A - peak amplitude
        k_vec: (kz, ky, kx) - wave vector (momentum)
        omega: ω - angular frequency
        dt: time step
        xp: numpy or cupy
        
    Returns:
        E(t=0): Field at t=0
        E(t=-dt): Field at t=-dt (for leapfrog)
    """
    Nz, Ny, Nx = shape
    
    # Create coordinate grids
    z, y, x = xp.meshgrid(
        xp.arange(Nz), xp.arange(Ny), xp.arange(Nx), 
        indexing='ij'
    )
    
    # Displacements from center
    dz = z - center[0]
    dy = y - center[1]
    dx = x - center[2]
    r_sq = dz**2 + dy**2 + dx**2
    
    # Gaussian envelope
    envelope = amplitude * xp.exp(-r_sq / (2 * width**2))
    
    # Phase: k·x
    k_dot_x = k_vec[0]*z + k_vec[1]*y + k_vec[2]*x
    
    # E(t=0) = A * exp(-r²/2σ²) * cos(k·x)
    E = envelope * xp.cos(k_dot_x)
    
    # E(t=-dt) = A * exp(-r²/2σ²) * cos(k·x + ω*dt)
    # This gives proper ∂E/∂t for propagating packet
    E_prev = envelope * xp.cos(k_dot_x + omega * dt)
    
    return E, E_prev


def measure_active_fraction(E, threshold=1e-6):
    """
    Measure what fraction of grid has significant field.
    
    Active cells: |E| > threshold
    
    This tells us how much speedup is theoretically possible.
    """
    if hasattr(E, 'get'):  # CuPy array
        E_np = E.get()
    else:
        E_np = E
    
    active_cells = np.sum(np.abs(E_np) > threshold)
    total_cells = E_np.size
    fraction = active_cells / total_cells
    
    return fraction, active_cells, total_cells


def run_baseline_wave_packet(
    grid_size: int = 128,
    steps: int = 300,
    use_gpu: bool = True,
    seed: int = 42,
    output_dir: str | Path | None = None,
    backend: str = "baseline",
    dt_override: float | None = None,
    width_override: float | None = None,
    precision: str = "float64",
):
    """
    Run baseline simulation with full lattice updates.
    
    Returns:
        dict with metrics and field snapshots
    """
    np.random.seed(seed)
    
    print("="*60)
    print("BASELINE: Full Lattice Update (Wave Packet)")
    print("="*60)
    
    # Setup backend
    xp, on_gpu = pick_backend(use_gpu and HAS_CUPY)
    device_name = "GPU (CuPy)" if on_gpu else "CPU (NumPy)"
    print(f"Device: {device_name}")
    
    # Grid setup
    Nx = Ny = Nz = grid_size
    shape = (Nz, Ny, Nx)
    dx = 1.0
    dt = float(dt_override) if dt_override is not None else 0.05
    
    print(f"Grid: {Nx}×{Ny}×{Nz} = {Nx*Ny*Nz:,} cells")
    print(f"Steps: {steps}")
    print(f"dx={dx}, dt={dt}")
    
    # Physics parameters
    alpha = 1.0
    beta = 1.0
    c = np.sqrt(alpha / beta)
    gamma_damp = 0.0  # No damping - test pure wave propagation
    
    # Gaussian wave packet parameters
    center = np.array([Nz//2, Ny//2, Nx//2], dtype=np.float64)
    width = float(width_override) if width_override is not None else 20.0  # Wider packet for numerical stability
    amplitude = 0.01  # Very small amplitude to minimize nonlinear effects
    # Wave vector pointing in +x direction
    k_mag = 2*np.pi / 32.0  # Wavelength ~ 32 cells (well-resolved)
    k_vec = np.array([0.0, 0.0, k_mag])  # Wave vector
    # Dispersion relation for Klein-Gordon: ω² = c²k²
    omega = c * k_mag
    
    print(f"\nGaussian Wave Packet:")
    print(f"  Center: ({center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f})")
    print(f"  Width: {width:.1f} cells")
    print(f"  Amplitude: {amplitude:.2f}")
    print(f"  Wavelength: {2*np.pi/k_mag:.1f} cells")
    print(f"  Frequency: omega={omega:.4f}")
    print(f"  Direction: +x")
    print(f"  Propagation speed: c={c:.2f}")
    
    # Initialize fields
    E, E_prev = create_gaussian_wave_packet(shape, center, width, amplitude, k_vec, omega, dt, xp)
    
    # Chi field (uniform, no gravity)
    chi = xp.zeros(shape, dtype=np.float64)
    
    # Parameters dict for lattice_step
    params = {
        "dt": dt,
        "dx": dx,
        "alpha": alpha,
        "beta": beta,
        "chi": chi,
        "gamma_damp": gamma_damp,
        "boundary": "periodic",
        "stencil_order": 2,
        "precision": precision,
    }
    
    # Measure initial energy
    E0_energy = energy_total(
        to_numpy(E), to_numpy(E_prev), dt, dx, c, 
        to_numpy(chi) if hasattr(chi, 'get') else chi
    )
    
    # Measure initial active fraction
    frac0, active0, total0 = measure_active_fraction(E, threshold=1e-6)
    
    print(f"\nInitial energy: {E0_energy:.6e}")
    print(f"Initial active cells: {active0:,} / {total0:,} ({frac0*100:.2f}%)")
    
    # Storage
    energies = [E0_energy]
    active_fractions = [frac0]
    
    # Timing
    print("\nStarting simulation...")
    t_start = time.perf_counter()
    
    # Main loop
    use_fused = (backend.lower() == "fused") and on_gpu
    if use_fused:
        try:
            # Lazy import to avoid CuPy dependency when unused
            from performance.optimizations.fused_tiled_kernel import fused_verlet_step
        except Exception as e:
            print(f"[WARN] Fused backend requested but unavailable ({e}); falling back to baseline.")
            use_fused = False

    for step in range(steps):
        # Lattice update (baseline or fused GPU path)
        if use_fused:
            E_next = fused_verlet_step(E, E_prev, chi, dt, dx, c, gamma_damp)
        else:
            E_next = lattice_step(E, E_prev, params)
        
        # Swap fields
        E_prev = E
        E = E_next
        
        # Monitoring (every 30 steps)
        if (step + 1) % 30 == 0 or step == steps - 1:
            E_current = energy_total(
                to_numpy(E), to_numpy(E_prev), dt, dx, c,
                to_numpy(chi) if hasattr(chi, 'get') else chi
            )
            energies.append(E_current)
            drift = abs((E_current - E0_energy) / (abs(E0_energy) + 1e-30))
            
            frac, active, total = measure_active_fraction(E, threshold=1e-6)
            active_fractions.append(frac)
            
            print(f"  Step {step+1:3d}/{steps}: E={E_current:.6e}, |dE/E0|={drift:.2e}, active={frac*100:.1f}%")
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    
    # Final metrics
    E_final = energies[-1]
    energy_drift = abs((E_final - E0_energy) / (abs(E0_energy) + 1e-30))
    frac_final, active_final, total_final = measure_active_fraction(E, threshold=1e-6)
    
    # Memory usage (estimate)
    bytes_per_cell = 8  # float64
    fields_count = 3  # E, E_prev, chi
    memory_mb = (Nx * Ny * Nz * bytes_per_cell * fields_count) / (1024**2)
    
    print("\n" + "="*60)
    print("BASELINE RESULTS (Wave Packet)")
    print("="*60)
    print(f"Total time: {elapsed:.3f} s")
    print(f"Time per step: {elapsed/steps:.6f} s")
    print(f"Energy drift: {energy_drift:.6e}")
    print(f"Final active cells: {active_final:,} / {total_final:,} ({frac_final*100:.2f}%)")
    print(f"Memory estimate: {memory_mb:.1f} MB")
    
    # Check success
    success = energy_drift < 1e-4
    print(f"\nEnergy conservation: {'PASS' if success else 'FAIL'} (< 1e-4)")
    
    # Theoretical speedup if we only updated active cells
    theoretical_speedup = 1.0 / np.mean(active_fractions)
    print(f"Theoretical speedup (if only active cells): {theoretical_speedup:.2f}x")
    
    # Prepare results
    results = {
        "algorithm": "baseline_full_update_wave",
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
        "active_fraction_initial": float(frac0),
        "active_fraction_final": float(frac_final),
        "active_fraction_mean": float(np.mean(active_fractions)),
        "theoretical_speedup": float(theoretical_speedup),
        "speedup_vs_baseline": 1.0,  # By definition
        "params": {
            "wave_center": center.tolist(),
            "wave_width": width,
            "wave_amplitude": amplitude,
            "wave_k_vec": k_vec.tolist(),
            "wave_omega": omega,
            "dx": dx,
            "dt": dt,
            "seed": seed
        }
    }
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "baseline_wave_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Save active fraction time series
        frac_file = output_dir / "baseline_wave_active_fractions.csv"
        with open(frac_file, 'w', encoding='utf-8') as f:
            f.write("step,active_fraction\n")
            for i, frac in enumerate(active_fractions):
                step_num = i * 30 if i > 0 else 0
                f.write(f"{step_num},{frac:.6f}\n")
        print(f"Active fractions saved to: {frac_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wave packet baseline/fused benchmark")
    parser.add_argument("--grid", type=int, default=128, help="Grid size N (uses N^3)")
    parser.add_argument("--steps", type=int, default=300, help="Number of time steps")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--backend", choices=["baseline", "fused"], default="baseline", help="Time-stepping backend")
    parser.add_argument("--dt", type=float, default=None, help="Override time step dt")
    parser.add_argument("--width", type=float, default=None, help="Override Gaussian width (sigma)")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64", help="Simulation precision")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials to run")
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "results"), help="Output directory for first trial + summary")
    args = parser.parse_args()

    output_dir = Path(args.out)

    print("Running wave packet benchmark...\n")

    trials = []
    for trial in range(max(1, args.trials)):
        print(f"\n{'#'*60}")
        print(f"# TRIAL {trial + 1}/{max(1, args.trials)}")
        print(f"{'#'*60}\n")

        result = run_baseline_wave_packet(
            grid_size=args.grid,
            steps=args.steps,
            use_gpu=args.gpu,
            seed=42 + trial,
            output_dir=output_dir if trial == 0 else None,  # Only save first trial
            backend=args.backend,
            dt_override=args.dt,
            width_override=args.width,
            precision=args.precision,
        )
        trials.append(result)

        print(f"\nTrial {trial+1} complete: {result['elapsed_time_s']:.3f}s, drift={result['energy_drift']:.2e}, active={result['active_fraction_mean']*100:.1f}%")

    # Compute statistics
    times = [t['elapsed_time_s'] for t in trials]
    drifts = [t['energy_drift'] for t in trials]
    actives = [t['active_fraction_mean'] for t in trials]

    print("\n" + "="*60)
    label = "FUSED" if args.backend == "fused" else "BASELINE"
    print(f"{label} SUMMARY ({len(trials)} trials, Wave Packet)")
    print("="*60)
    print(f"Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
    print(f"Energy drift: {np.mean(drifts):.2e} ± {np.std(drifts):.2e}")
    print(f"Active fraction: {np.mean(actives)*100:.1f}% ± {np.std(actives)*100:.1f}%")
    print(f"All trials pass: {all(t['energy_drift_pass'] for t in trials)}")

    # Save summary
    summary = {
        f"{label.lower()}_wave_summary": {
            "n_trials": len(trials),
            "mean_time_s": float(np.mean(times)),
            "std_time_s": float(np.std(times)),
            "mean_drift": float(np.mean(drifts)),
            "std_drift": float(np.std(drifts)),
            "mean_active_fraction": float(np.mean(actives)),
            "std_active_fraction": float(np.std(actives)),
            "all_pass": all(t['energy_drift_pass'] for t in trials)
        },
        "trials": trials
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / ("fused_wave_summary.json" if args.backend == "fused" else "baseline_wave_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")
    print("\nWave packet benchmark complete!")
