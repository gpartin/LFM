# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
Fused Backend Performance Benchmark
====================================

Measures baseline vs fused backend performance across representative workloads
to validate speedup claims and accuracy (P1 gate: drift <1e-4).

Outputs CSV with: grid_size, backend, mean_time_ms, std_ms, speedup, drift, 
                  timestamp, hardware_id

Usage:
    python fused_backend_benchmark.py --output results.csv
    python fused_backend_benchmark.py --quick  # Fast smoke test
"""
from __future__ import annotations

import argparse
import time
import platform
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from core.lfm_backend import pick_backend, to_numpy
from core.lfm_equation import lattice_step

try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False


def get_hardware_id() -> str:
    """Get reproducible hardware identifier for results."""
    if HAS_CUPY:
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
            cuda_ver = cp.cuda.runtime.runtimeGetVersion()
            return f"{gpu_name}_CUDA{cuda_ver}"
        except Exception:
            pass
    return f"CPU_{platform.processor()}"


def create_gaussian_packet(shape: Tuple[int, int, int], xp) -> Tuple:
    """Create 3D Gaussian wave packet with momentum."""
    Nz, Ny, Nx = shape
    z, y, x = xp.meshgrid(xp.arange(Nz), xp.arange(Ny), xp.arange(Nx), indexing='ij')
    center = xp.array([Nz//2, Ny//2, Nx//2], dtype=xp.float64)
    
    dz, dy, dx_arr = z - center[0], y - center[1], x - center[2]
    r2 = dz*dz + dy*dy + dx_arr*dx_arr
    
    width = max(6.0, Nx / 16.0)
    amp = 0.01
    env = amp * xp.exp(-r2 / (2 * width * width), dtype=xp.float64)
    
    k = 2 * np.pi / max(16.0, Nx / 8.0)
    phase = k * x
    
    E = env * xp.cos(phase, dtype=xp.float64)
    
    # E_prev for rightward propagation
    dt_init = 0.05
    omega = k  # Approximate for small k
    E_prev = env * xp.cos(phase + omega * dt_init, dtype=xp.float64)
    
    return E, E_prev


def energy_discrete(E, E_prev, dt: float, dx: float, xp) -> float:
    """Compute discrete Klein-Gordon energy."""
    Et = (E - E_prev) / dt
    
    gx = (xp.roll(E, -1, 2) - xp.roll(E, 1, 2)) / (2 * dx)
    gy = (xp.roll(E, -1, 1) - xp.roll(E, 1, 1)) / (2 * dx)
    gz = (xp.roll(E, -1, 0) - xp.roll(E, 1, 0)) / (2 * dx)
    
    c = 1.0  # Natural units
    dens = 0.5 * (Et**2 + (c**2) * (gx**2 + gy**2 + gz**2))
    
    return float(xp.sum(dens) * (dx**3))


def benchmark_workload(
    workload_name: str,
    shape: Tuple[int, int, int],
    steps: int,
    backend: str,
    trials: int = 3
) -> Dict:
    """
    Benchmark a single workload configuration.
    
    Returns:
        Dict with mean_time_ms, std_ms, drift, hardware
    """
    xp, on_gpu = pick_backend(True)
    
    if not on_gpu and backend == "fused":
        # Skip fused on CPU
        return {
            "workload": workload_name,
            "grid_size": f"{shape[0]}x{shape[1]}x{shape[2]}",
            "backend": backend,
            "mean_time_ms": None,
            "std_ms": None,
            "speedup": None,
            "drift": None,
            "trials": 0,
            "hardware": get_hardware_id(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "skipped_cpu"
        }
    
    # Setup
    E, E_prev = create_gaussian_packet(shape, xp)
    chi = xp.zeros(shape, dtype=xp.float64)
    
    dt, dx = 0.05, 1.0
    c = 1.0
    
    params = {
        "dt": dt,
        "dx": dx,
        "alpha": c * c,
        "beta": 1.0,
        "chi": chi,
        "gamma_damp": 0.0,
        "boundary": "periodic",
        "stencil_order": 2,
        "precision": "float64",
        "backend": backend
    }
    
    # Initial energy
    E0 = energy_discrete(E, E_prev, dt, dx, xp)
    
    # Timing trials
    times = []
    for trial in range(trials):
        E_trial, E_prev_trial = E.copy(), E_prev.copy()
        
        # Warmup (1 step)
        _ = lattice_step(E_trial, E_prev_trial, params)
        if on_gpu:
            cp.cuda.Stream.null.synchronize()
        
        # Timed run
        t0 = time.perf_counter()
        for _ in range(steps):
            E_next = lattice_step(E_trial, E_prev_trial, params)
            E_prev_trial, E_trial = E_trial, E_next
        
        if on_gpu:
            cp.cuda.Stream.null.synchronize()
        
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000 / steps)  # ms per step
    
    # Final energy drift
    E_final = energy_discrete(E_trial, E_prev_trial, dt, dx, xp)
    drift = abs(E_final - E0) / (abs(E0) + 1e-30)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        "workload": workload_name,
        "grid_size": f"{shape[0]}x{shape[1]}x{shape[2]}",
        "backend": backend,
        "mean_time_ms": float(mean_time),
        "std_ms": float(std_time),
        "speedup": None,  # Computed later
        "drift": float(drift),
        "trials": trials,
        "hardware": get_hardware_id(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "completed"
    }


def compute_speedups(results: List[Dict]) -> List[Dict]:
    """Compute speedup = baseline_time / fused_time for each workload."""
    workloads = {}
    for r in results:
        key = (r["workload"], r["grid_size"])
        if key not in workloads:
            workloads[key] = {}
        workloads[key][r["backend"]] = r
    
    # Update speedups
    for key, backends in workloads.items():
        if "baseline" in backends and "fused" in backends:
            baseline = backends["baseline"]
            fused = backends["fused"]
            
            if baseline["mean_time_ms"] and fused["mean_time_ms"]:
                speedup = baseline["mean_time_ms"] / fused["mean_time_ms"]
                fused["speedup"] = float(speedup)
    
    return results


def write_csv(filepath: Path, results: List[Dict]):
    """Write results to CSV."""
    import csv
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "workload", "grid_size", "backend", "mean_time_ms", "std_ms",
            "speedup", "drift", "trials", "hardware", "timestamp", "status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Results written to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Fused Backend Performance Benchmark")
    parser.add_argument("--output", type=str, default="fused_benchmark_results.csv",
                       help="Output CSV filename")
    parser.add_argument("--quick", action="store_true",
                       help="Quick smoke test (fewer steps/trials)")
    parser.add_argument("--trials", type=int, default=3,
                       help="Number of trials per configuration")
    args = parser.parse_args()
    
    # Workload definitions
    if args.quick:
        workloads = [
            ("wave_packet_64", (64, 64, 64), 10),
            ("wave_packet_128", (128, 128, 128), 5),
        ]
        trials = 2
    else:
        workloads = [
            ("wave_packet_64", (64, 64, 64), 50),
            ("wave_packet_128", (128, 128, 128), 30),
            ("wave_packet_256", (256, 256, 256), 20),
            ("gravity_sim_64", (64, 64, 64), 100),  # Earth-Moon analogue
        ]
        trials = args.trials
    
    print(f"Fused Backend Benchmark")
    print(f"=======================")
    print(f"Hardware: {get_hardware_id()}")
    print(f"Workloads: {len(workloads)}")
    print(f"Backends: baseline, fused")
    print(f"Trials per config: {trials}")
    print()
    
    results = []
    
    for workload_name, shape, steps in workloads:
        print(f"Running: {workload_name} {shape} × {steps} steps...")
        
        # Baseline
        result_baseline = benchmark_workload(workload_name, shape, steps, "baseline", trials)
        results.append(result_baseline)
        if result_baseline["mean_time_ms"]:
            print(f"  Baseline: {result_baseline['mean_time_ms']:.3f} ± {result_baseline['std_ms']:.3f} ms/step, drift={result_baseline['drift']:.2e}")
        
        # Fused
        result_fused = benchmark_workload(workload_name, shape, steps, "fused", trials)
        results.append(result_fused)
        if result_fused["mean_time_ms"]:
            print(f"  Fused:    {result_fused['mean_time_ms']:.3f} ± {result_fused['std_ms']:.3f} ms/step, drift={result_fused['drift']:.2e}")
            if result_baseline["mean_time_ms"]:
                speedup = result_baseline["mean_time_ms"] / result_fused["mean_time_ms"]
                result_fused["speedup"] = speedup
                print(f"  Speedup:  {speedup:.2f}×")
        else:
            print(f"  Fused:    {result_fused['status']}")
        
        print()
    
    # Compute speedups
    results = compute_speedups(results)
    
    # Write CSV
    output_path = Path(__file__).parent / args.output
    write_csv(output_path, results)
    
    # Summary
    print("\nSummary")
    print("-------")
    fused_results = [r for r in results if r["backend"] == "fused" and r["speedup"]]
    if fused_results:
        speedups = [r["speedup"] for r in fused_results]
        print(f"Speedup range: {min(speedups):.2f}× to {max(speedups):.2f}×")
        print(f"Mean speedup: {np.mean(speedups):.2f}×")
        
        # Check accuracy: fused should match baseline (relative difference <1e-6)
        baseline_results = [r for r in results if r["backend"] == "baseline"]
        drift_diffs = []
        for fused_r in fused_results:
            baseline_r = next((b for b in baseline_results if b["workload"] == fused_r["workload"] and b["grid_size"] == fused_r["grid_size"]), None)
            if baseline_r and baseline_r["drift"] is not None and fused_r["drift"] is not None:
                rel_diff = abs(fused_r["drift"] - baseline_r["drift"]) / (abs(baseline_r["drift"]) + 1e-30)
                drift_diffs.append(rel_diff)
        
        if drift_diffs:
            max_drift_diff = max(drift_diffs)
            print(f"Max drift difference (fused vs baseline): {max_drift_diff:.2e}")
            p1_pass = max_drift_diff < 1e-4
            print(f"P1 accuracy gate (drift match <1e-4): {'PASS ✓' if p1_pass else 'FAIL ✗'}")
        else:
            print("Could not compute drift differences")
    else:
        print("No fused results to summarize (GPU not available?)")


if __name__ == "__main__":
    main()
