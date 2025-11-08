# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
3D Scaling Benchmark for LFM
----------------------------

Runs lattice_step on cubic domains (N^3) with configurable precision and steps.
Records wall time and energy drift to JSON/CSV under performance/benchmarks/results.

Usage (Windows):
  py -3 workspace\performance\benchmarks\bench_scale_3d.py --sizes 128,256 --steps 150 --precision float32 --gpu
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import List
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from core.lfm_equation import lattice_step, energy_total
from core.lfm_backend import pick_backend, to_numpy


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


def run_case(N: int, steps: int, precision: str, use_gpu: bool) -> dict:
    xp, on_gpu = pick_backend(use_gpu)
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
    chi = xp.zeros(shape, dtype=np.float32 if precision=='float32' else np.float64)

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

    E0 = energy_total(to_numpy(E), to_numpy(E_prev), dt, dx, c, to_numpy(chi) if hasattr(chi,'get') else chi)

    t0 = time.perf_counter()
    for _ in range(steps):
        E_next = lattice_step(E, E_prev, params)
        E_prev, E = E, E_next
    t1 = time.perf_counter()

    E_final = energy_total(to_numpy(E), to_numpy(E_prev), dt, dx, c, to_numpy(chi) if hasattr(chi,'get') else chi)
    drift = abs((E_final - E0)/(abs(E0)+1e-30))

    return {
        "N": N,
        "steps": steps,
        "precision": precision,
        "device": device,
        "elapsed_s": t1 - t0,
        "time_per_step_ms": (t1 - t0) * 1000.0 / steps,
        "energy_initial": float(E0),
        "energy_final": float(E_final),
        "energy_drift": float(drift),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="128,256", help="Comma-separated sizes, e.g. 128,256,384")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--precision", type=str, choices=["float32", "float64"], default="float32")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--outdir", type=str, default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    sizes: List[int] = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("LFM 3D Scaling Benchmark")
    print("="*60)
    print(f"Sizes      : {sizes}")
    print(f"Steps      : {args.steps}")
    print(f"Precision  : {args.precision}")
    print(f"Backend    : {'GPU' if args.gpu else 'CPU'}")

    results = []
    for N in sizes:
        print(f"\nRunning N={N}^3 ...")
        r = run_case(N, args.steps, args.precision, args.gpu)
        print(f"  Time: {r['elapsed_s']:.3f}s  ({r['time_per_step_ms']:.3f} ms/step)  Drift: {r['energy_drift']:.2e}")
        results.append(r)

    # Save JSON
    js = {
        "config": {
            "sizes": sizes,
            "steps": args.steps,
            "precision": args.precision,
            "backend": "GPU" if args.gpu else "CPU",
        },
        "results": results,
    }
    with open(outdir/"bench_scale_3d.json", 'w', encoding='utf-8') as f:
        json.dump(js, f, indent=2)

    # Save CSV
    import csv
    with open(outdir/"bench_scale_3d.csv", 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(["N", "steps", "precision", "device", "elapsed_s", "time_per_step_ms", "energy_drift"])
        for r in results:
            w.writerow([r['N'], r['steps'], r['precision'], r['device'], f"{r['elapsed_s']:.6f}", f"{r['time_per_step_ms']:.6f}", f"{r['energy_drift']:.6e}"])

    print(f"\nSaved: {outdir/'bench_scale_3d.json'}")
    print(f"Saved: {outdir/'bench_scale_3d.csv'}")


if __name__ == "__main__":
    main()
