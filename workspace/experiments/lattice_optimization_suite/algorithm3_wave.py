#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm 3: Wavefront Prediction (Wave Packet)
===============================================

Predict where the wave will be next using physics (speed c) and only
update those regions. Define sources by amplitude or time-derivative,
then expand by a conservative front radius ~ c·dt with a safety factor.

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


def dilate6(mask, xp):
    return (
        mask
        | xp.roll(mask, 1, 0) | xp.roll(mask, -1, 0)
        | xp.roll(mask, 1, 1) | xp.roll(mask, -1, 1)
        | xp.roll(mask, 1, 2) | xp.roll(mask, -1, 2)
    )


def signal_mask(E, E_prev, dt, amp_thresh, rate_thresh, xp):
    dEdt = (E - E_prev) / dt
    return (xp.abs(E) > amp_thresh) | (xp.abs(dEdt) > rate_thresh)


def predict_front_mask(src_mask, c, dt, safety, xp):
    # Convert physical radius to integer halo steps (Manhattan approx)
    r = c * dt * safety
    steps = max(1, int(np.ceil(r)))
    mask = src_mask
    for _ in range(steps):
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


def run_algorithm3_wave(
    grid_size=128,
    steps=300,
    use_gpu=True,
    seed=42,
    amp_thresh=1e-6,
    rate_thresh=1e-5,
    safety=2.0,
    persistence=True,
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

    prev_mask = xp.zeros(shape, dtype=bool)

    t0 = time.perf_counter()
    for step in range(steps):
        src = signal_mask(E, E_prev, dt, amp_thresh, rate_thresh, xp)
        mask = predict_front_mask(src, c, dt, safety, xp)
        if persistence:
            mask = mask | prev_mask
        frac = float((mask.get() if hasattr(mask,'get') else mask).mean())
        active_fracs.append(frac)

        E_next = lattice_step_masked(E, E_prev, chi, c, dt, dx, gamma_damp, mask, xp)
        prev_mask = mask
        E_prev, E = E, E_next

        if (step+1) % 30 == 0 or step == steps-1:
            Ec = energy_total(to_numpy(E), to_numpy(E_prev), dt, dx, c, to_numpy(chi) if hasattr(chi,'get') else chi)
            energies.append(Ec)
    t1 = time.perf_counter()

    drift = abs((energies[-1]-E0)/(abs(E0)+1e-30))
    elapsed = t1 - t0
    mean_active = float(np.mean(active_fracs))

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
        "algorithm": "algorithm3_wavefront_prediction_wave",
        "device": device_name,
        "grid_size": grid_size,
        "steps": steps,
        "elapsed_time_s": elapsed,
        "energy_drift": float(drift),
        "active_fraction_mean": mean_active,
        "speedup_vs_baseline": float(speedup) if speedup else None,
        "params": {
            "amp_thresh": amp_thresh,
            "rate_thresh": rate_thresh,
            "safety": safety,
            "persistence": bool(persistence),
            "width": width,
            "amplitude": amplitude,
        }
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir/"algorithm3_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    print("\n=== ALGORITHM 3 RESULTS ===")
    print(f"Time: {elapsed:.3f}s, drift={drift:.2e}, active={mean_active*100:.1f}%")
    if speedup:
        print(f"Speedup vs baseline: {speedup:.2f}x")

    return results


if __name__ == "__main__":
    out = Path(__file__).parent / "results"
    trials = []
    for i in range(3):
        r = run_algorithm3_wave(output_dir=out, seed=42+i)
        trials.append(r)
    times = [t['elapsed_time_s'] for t in trials]
    drifts = [t['energy_drift'] for t in trials]
    fracs = [t['active_fraction_mean'] for t in trials]
    speedups = [t['speedup_vs_baseline'] for t in trials if t['speedup_vs_baseline']]
    summary = {
        "algorithm3_summary": {
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
    with open(out/"algorithm3_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print("\nALGORITHM 3 SUMMARY")
    print(f"Time: {summary['algorithm3_summary']['mean_time_s']:.3f}s ± {summary['algorithm3_summary']['std_time_s']:.3f}s")
    print(f"Drift: {summary['algorithm3_summary']['mean_drift']:.2e}")
    print(f"Active: {summary['algorithm3_summary']['mean_active_fraction']*100:.1f}%")
    if summary['algorithm3_summary']['mean_speedup']:
        print(f"Speedup: {summary['algorithm3_summary']['mean_speedup']:.2f}x")
    print("Algorithm 3 complete!")
