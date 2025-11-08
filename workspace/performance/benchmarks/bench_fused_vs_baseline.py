# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
Compare baseline lattice_step vs fused RawKernel (GPU) for a given N^3.

Usage:
  py -3 workspace\performance\benchmarks\bench_fused_vs_baseline.py --N 256 --steps 150
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from core.lfm_equation import lattice_step, energy_total
from core.lfm_backend import pick_backend, to_numpy

sys.path.insert(0, str(Path(__file__).parents[1] / "optimizations"))
from fused_tiled_kernel import fused_verlet_step


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


def run(N: int, steps: int) -> dict:
    xp, on_gpu = pick_backend(True)
    device = "GPU (CuPy)" if on_gpu else "CPU (NumPy)"

    Nx = Ny = Nz = N
    shape = (Nz, Ny, Nx)
    dx = 1.0
    dt = 0.05
    alpha = 1.0
    beta = 1.0
    c = np.sqrt(alpha/beta)
    gamma_damp = 0.0

    center = np.array([Nz//2, Ny//2, Nx//2], dtype=np.float64)
    width = max(8.0, N/16.0)
    amplitude = 0.01
    k_mag = 2*np.pi / max(16.0, N/8.0)
    k_vec = np.array([0.0, 0.0, k_mag])
    omega = c*k_mag

    E, E_prev = gaussian_packet(shape, center, width, amplitude, k_vec, omega, dt, xp)
    chi = xp.zeros(shape, dtype=np.float64)

    params = {
        "dt": dt,
        "dx": dx,
        "alpha": alpha,
        "beta": beta,
        "chi": chi,
        "gamma_damp": gamma_damp,
        "boundary": "periodic",
        "stencil_order": 2,
        "precision": "float64",
    }

    # Baseline
    Eb, Eb_prev = E.copy(), E_prev.copy()
    t0 = time.perf_counter()
    for _ in range(steps):
        Eb_next = lattice_step(Eb, Eb_prev, params)
        Eb_prev, Eb = Eb, Eb_next
    t1 = time.perf_counter()

    # Fused
    Ef, Ef_prev = E.copy(), E_prev.copy()
    t2 = time.perf_counter()
    for _ in range(steps):
        Ef_next = fused_verlet_step(Ef, Ef_prev, chi, dt, dx, c, gamma_damp)
        Ef_prev, Ef = Ef, Ef_next
    t3 = time.perf_counter()

    # Energy drift (reference baseline energy)
    E0 = energy_total(to_numpy(E), to_numpy(E_prev), dt, dx, c, to_numpy(chi) if hasattr(chi,'get') else chi)
    Eb_final = energy_total(to_numpy(Eb), to_numpy(Eb_prev), dt, dx, c, to_numpy(chi) if hasattr(chi,'get') else chi)
    Ef_final = energy_total(to_numpy(Ef), to_numpy(Ef_prev), dt, dx, c, to_numpy(chi) if hasattr(chi,'get') else chi)

    return {
        "N": N,
        "steps": steps,
        "device": device,
        "baseline_time_s": t1 - t0,
        "baseline_ms_per_step": (t1 - t0) * 1000.0 / steps,
        "fused_time_s": t3 - t2,
        "fused_ms_per_step": (t3 - t2) * 1000.0 / steps,
        "speedup": (t1 - t0) / (t3 - t2),
        "E0": float(E0),
        "Eb_final": float(Eb_final),
        "Ef_final": float(Ef_final),
        "baseline_drift": float(abs((Eb_final - E0)/(abs(E0)+1e-30))),
        "fused_drift": float(abs((Ef_final - E0)/(abs(E0)+1e-30))),
        "L2_relative_error": float(np.linalg.norm(to_numpy(Eb - Ef)) / (np.linalg.norm(to_numpy(Eb)) + 1e-12)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--outdir", type=str, default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Comparing baseline vs fused kernel at N={args.N}^3, steps={args.steps}...")
    r = run(args.N, args.steps)
    print(f"Baseline: {r['baseline_ms_per_step']:.3f} ms/step, drift={r['baseline_drift']:.2e}")
    print(f"Fused   : {r['fused_ms_per_step']:.3f} ms/step, drift={r['fused_drift']:.2e}")
    print(f"Speedup : {r['speedup']:.2f}x")
    print(f"L2 rel error vs baseline: {r['L2_relative_error']:.3e}")

    with open(outdir/"bench_fused_vs_baseline.json", 'w', encoding='utf-8') as f:
        json.dump(r, f, indent=2)
    print(f"Saved: {outdir/'bench_fused_vs_baseline.json'}")


if __name__ == "__main__":
    main()
