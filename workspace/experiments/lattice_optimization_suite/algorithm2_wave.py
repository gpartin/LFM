#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
Algorithm 2: Gradient-Adaptive Culling (Wave Packet)
====================================================

Detect significant dynamics via gradient magnitude and field amplitude,
add a small halo, and update only those regions.

Target (Gate 0): energy drift < 5e-4; any speedup vs baseline.
"""

import sys
import json
import time
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


def gaussian_packet(shape, center, width, amplitude, k_vec, omega, dt, xp):
    Nz, Ny, Nx = shape
    z, y, x = xp.meshgrid(xp.arange(Nz), xp.arange(Ny), xp.arange(Nx), indexing='ij')
    dz, dy, dx = z - center[0], y - center[1], x - center[2]
    r2 = dz*dz + dy*dy + dx*dx
    env = amplitude * xp.exp(-r2 / (2*width*width))
    kdot = k_vec[0]*z + k_vec[1]*y + k_vec[2]*x
    E = env * xp.cos(kdot)
    E_prev = env * xp.cos(kdot + omega*dt)
    return E, E_prev


def grad_mag(E, dx, xp):
    # Central differences with periodic boundaries
    gx = (xp.roll(E, -1, 2) - xp.roll(E, 1, 2)) / (2*dx)
    gy = (xp.roll(E, -1, 1) - xp.roll(E, 1, 1)) / (2*dx)
    gz = (xp.roll(E, -1, 0) - xp.roll(E, 1, 0)) / (2*dx)
    return xp.sqrt(gx*gx + gy*gy + gz*gz)


def dilate6(mask, xp):
    return (
        mask
        | xp.roll(mask, 1, 0) | xp.roll(mask, -1, 0)
        | xp.roll(mask, 1, 1) | xp.roll(mask, -1, 1)
        | xp.roll(mask, 1, 2) | xp.roll(mask, -1, 2)
    )


def adaptive_mask(E, E_prev, dx, amp_thresh, grad_thresh, halo, xp):
    m_amp = (xp.abs(E) > amp_thresh) | (xp.abs(E_prev) > amp_thresh)
    g = grad_mag(E, dx, xp)
    m_grad = g > grad_thresh
    mask = m_amp | m_grad
    for _ in range(max(0, int(halo))):
        mask = dilate6(mask, xp)
    return mask


def lattice_step_masked(E, E_prev, chi, c, dt, dx, gamma, mask, xp):
    lap = laplacian(E, dx)
    c2 = c*c
    dt2 = dt*dt
    chi2 = chi*chi if not xp.isscalar(chi) else (chi*chi)
    term = c2*lap - chi2*E
    update = (2.0 - gamma)*E - (1.0 - gamma)*E_prev + dt2*term
    return xp.where(mask, update, E)


def run_algorithm2_wave(
    grid_size=128,
    steps=300,
    use_gpu=True,
    seed=42,
    amp_thresh=1e-6,
    grad_thresh=1e-7,
    halo=2,
    output_dir=None,
):
    np.random.seed(seed)
    xp, on_gpu = pick_backend(use_gpu and HAS_CUPY)
    device_name = "GPU (CuPy)" if on_gpu else "CPU (NumPy)"

    Nx = Ny = Nz = grid_size
    shape = (Nz, Ny, Nx)
    dx = 1.0
    dt = 0.05

    alpha = 1.0
    beta = 1.0
    c = np.sqrt(alpha/beta)
    gamma_damp = 0.0

    center = np.array([Nz//2, Ny//2, Nx//2], dtype=np.float64)
    width = 20.0
    amplitude = 0.01
    k_mag = 2*np.pi/32.0
    k_vec = np.array([0.0, 0.0, k_mag])
    omega = c*k_mag

    E, E_prev = gaussian_packet(shape, center, width, amplitude, k_vec, omega, dt, xp)
    chi = xp.zeros(shape, dtype=np.float64)

    E0 = energy_total(to_numpy(E), to_numpy(E_prev), dt, dx, c, to_numpy(chi) if hasattr(chi,'get') else chi)

    energies = [E0]
    active_fracs = []

    t0 = time.perf_counter()
    for step in range(steps):
        mask = adaptive_mask(E, E_prev, dx, amp_thresh, grad_thresh, halo, xp)
        frac = float((mask.get() if hasattr(mask,'get') else mask).mean())
        active_fracs.append(frac)

        E_next = lattice_step_masked(E, E_prev, chi, c, dt, dx, gamma_damp, mask, xp)
        E_prev, E = E, E_next

        if (step+1) % 30 == 0 or step == steps-1:
            Ec = energy_total(to_numpy(E), to_numpy(E_prev), dt, dx, c, to_numpy(chi) if hasattr(chi,'get') else chi)
            energies.append(Ec)
    t1 = time.perf_counter()

    drift = abs((energies[-1]-E0)/(abs(E0)+1e-30))
    elapsed = t1 - t0
    mean_active = float(np.mean(active_fracs))

    # Compare to baseline
    baseline_file = Path(__file__).parent / "results" / "baseline_wave_summary.json"
    if baseline_file.exists():
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline = json.load(f)
        baseline_time = baseline['baseline_wave_summary']['mean_time_s']
        speedup = baseline_time / elapsed
    else:
        baseline_time = None
        speedup = None

    results = {
        "algorithm": "algorithm2_gradient_adaptive_wave",
        "device": device_name,
        "grid_size": grid_size,
        "steps": steps,
        "elapsed_time_s": elapsed,
        "energy_drift": float(drift),
        "active_fraction_mean": mean_active,
        "speedup_vs_baseline": float(speedup) if speedup else None,
        "params": {
            "amp_thresh": amp_thresh,
            "grad_thresh": grad_thresh,
            "halo": halo,
            "width": width,
            "amplitude": amplitude,
        }
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir/"algorithm2_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    print("\n=== ALGORITHM 2 RESULTS ===")
    print(f"Time: {elapsed:.3f}s, drift={drift:.2e}, active={mean_active*100:.1f}%")
    if speedup:
        print(f"Speedup vs baseline: {speedup:.2f}x")

    return results


if __name__ == "__main__":
    out = Path(__file__).parent / "results"
    trials = []
    for i in range(3):
        r = run_algorithm2_wave(output_dir=out, seed=42+i)
        trials.append(r)
    times = [t['elapsed_time_s'] for t in trials]
    drifts = [t['energy_drift'] for t in trials]
    fracs = [t['active_fraction_mean'] for t in trials]
    speedups = [t['speedup_vs_baseline'] for t in trials if t['speedup_vs_baseline']]
    summary = {
        "algorithm2_summary": {
            "n_trials": 3,
            "mean_time_s": float(np.mean(times)),
            "std_time_s": float(np.std(times)),
            "mean_drift": float(np.mean(drifts)),
            "std_drift": float(np.std(drifts)),
            "mean_active_fraction": float(np.mean(fracs)),
            "std_active_fraction": float(np.std(fracs)),
            "mean_speedup": float(np.mean(speedups)) if speedups else None,
        },
        "trials": trials,
    }
    with open(out/"algorithm2_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print("\nALGORITHM 2 SUMMARY")
    print(f"Time: {summary['algorithm2_summary']['mean_time_s']:.3f}s ± {summary['algorithm2_summary']['std_time_s']:.3f}s")
    print(f"Drift: {summary['algorithm2_summary']['mean_drift']:.2e}")
    print(f"Active: {summary['algorithm2_summary']['mean_active_fraction']*100:.1f}%")
    if summary['algorithm2_summary']['mean_speedup']:
        print(f"Speedup: {summary['algorithm2_summary']['mean_speedup']:.2f}x")
    print("Algorithm 2 complete!")
