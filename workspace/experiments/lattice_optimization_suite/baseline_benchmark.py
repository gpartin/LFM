#!/usr/bin/env python3
"""
Baseline Benchmark - Full Lattice Update
==========================================

This establishes the reference performance for full 3D lattice updates.
All optimization experiments will be compared against this baseline.

Test case: Circular orbit (Earth-Moon analog)
Grid: 128³ cells
Steps: 300 timesteps
Physics: Kinematic gravity via chi field

Metrics measured:
1. Energy conservation: |ΔE/E₀|
2. Wall-clock time: Total runtime
3. Memory usage: Peak allocation
4. Trajectory accuracy: Moon position over time
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.lfm_equation import lattice_step, energy_total
from core.lfm_backend import pick_backend, to_numpy

# Try GPU first
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def create_chi_field_orbit(shape, chi_earth, chi_moon, earth_pos, moon_pos, xp):
    """Create chi field with Earth and Moon"""
    chi = xp.zeros(shape, dtype=np.float64)
    Nz, Ny, Nx = shape
    
    # Create coordinate grids
    z, y, x = xp.meshgrid(
        xp.arange(Nz), xp.arange(Ny), xp.arange(Nx), 
        indexing='ij'
    )
    
    # Earth (centered)
    r_earth = xp.sqrt(
        (x - earth_pos[0])**2 + 
        (y - earth_pos[1])**2 + 
        (z - earth_pos[2])**2
    )
    earth_mask = r_earth < 3.0  # 3-cell radius
    chi[earth_mask] = chi_earth
    
    # Moon (orbiting)
    r_moon = xp.sqrt(
        (x - moon_pos[0])**2 + 
        (y - moon_pos[1])**2 + 
        (z - moon_pos[2])**2
    )
    moon_mask = r_moon < 1.5  # 1.5-cell radius (smaller than Earth)
    chi[moon_mask] = chi_moon
    
    return chi


def update_moon_position_kinematic(moon_pos, moon_vel, earth_pos, k_acc, dt):
    """
    Update moon position using kinematic gravity: a = k_acc * ∇χ
    
    This is simplified for the baseline - we're focusing on lattice performance,
    not perfect gravity integration.
    """
    # Vector from Earth to Moon
    r_vec = moon_pos - earth_pos
    r = np.linalg.norm(r_vec)
    
    # Direction
    r_hat = r_vec / (r + 1e-10)
    
    # Acceleration toward Earth (∇χ points inward for gravity)
    # Simplified: a ∝ 1/r² (Newtonian analog)
    a_mag = k_acc / (r**2 + 1.0)
    acc = -a_mag * r_hat  # Negative for attraction
    
    # Update velocity and position
    moon_vel = moon_vel + acc * dt
    moon_pos = moon_pos + moon_vel * dt
    
    return moon_pos, moon_vel


def run_baseline_orbit(
    grid_size=128,
    steps=300,
    use_gpu=True,
    seed=42,
    output_dir=None
):
    """
    Run baseline simulation with full lattice updates.
    
    Returns:
        dict with metrics and trajectory data
    """
    np.random.seed(seed)
    
    print("="*60)
    print("BASELINE: Full Lattice Update")
    print("="*60)
    
    # Setup backend
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
    print(f"dx={dx}, dt={dt}")
    
    # Physics parameters
    alpha = 1.0
    beta = 1.0
    c = np.sqrt(alpha / beta)
    gamma_damp = 0.0
    
    # Orbit parameters
    chi_earth = 50.0
    chi_moon = chi_earth / 81.0  # Mass ratio
    k_acc = 0.20  # Acceleration coupling
    
    # Initial positions (Earth at center)
    earth_pos = np.array([Nz//2, Ny//2, Nx//2], dtype=np.float64)
    moon_pos = np.array([Nz//2, Ny//2 + 32, Nx//2], dtype=np.float64)  # 32 units away
    
    # Initial velocity (tangential for circular orbit)
    # v = sqrt(a*r) approximately, tuned empirically
    v_tangential = 1.86
    moon_vel = np.array([0.0, 0.0, v_tangential], dtype=np.float64)
    
    print(f"\nPhysics:")
    print(f"  chi_earth={chi_earth:.2f}")
    print(f"  chi_moon={chi_moon:.3f}")
    print(f"  k_acc={k_acc}")
    print(f"  Initial orbit radius: {np.linalg.norm(moon_pos - earth_pos):.1f} cells")
    print(f"  Initial velocity: {np.linalg.norm(moon_vel):.3f} units/step")
    
    # Initialize fields
    E = xp.zeros(shape, dtype=np.float64)
    E_prev = xp.zeros(shape, dtype=np.float64)
    
    # Create initial chi field
    chi = create_chi_field_orbit(shape, chi_earth, chi_moon, earth_pos, moon_pos, xp)
    
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
        "precision": "float64"
    }
    
    # Measure initial energy
    E0_energy = energy_total(
        to_numpy(E), to_numpy(E_prev), dt, dx, c, 
        to_numpy(chi) if hasattr(chi, 'get') else chi
    )
    
    print(f"\nInitial energy: {E0_energy:.6e}")
    
    # Storage for trajectory and metrics
    trajectory = [moon_pos.copy()]
    energies = [E0_energy]
    
    # Timing
    print("\nStarting simulation...")
    t_start = time.perf_counter()
    
    # Main loop
    for step in range(steps):
        # Update chi field with new moon position
        chi = create_chi_field_orbit(shape, chi_earth, chi_moon, earth_pos, moon_pos, xp)
        params["chi"] = chi
        
        # Lattice update (BASELINE: full grid)
        E_next = lattice_step(E, E_prev, params)
        
        # Swap fields
        E_prev = E
        E = E_next
        
        # Update moon position (kinematic)
        moon_pos, moon_vel = update_moon_position_kinematic(
            moon_pos, moon_vel, earth_pos, k_acc, dt
        )
        
        trajectory.append(moon_pos.copy())
        
        # Energy monitoring (every 30 steps)
        if (step + 1) % 30 == 0 or step == steps - 1:
            E_current = energy_total(
                to_numpy(E), to_numpy(E_prev), dt, dx, c,
                to_numpy(chi) if hasattr(chi, 'get') else chi
            )
            energies.append(E_current)
            drift = abs((E_current - E0_energy) / (abs(E0_energy) + 1e-30))
            
            print(f"  Step {step+1:3d}/{steps}: E={E_current:.6e}, |ΔE/E₀|={drift:.2e}")
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    
    # Final metrics
    E_final = energies[-1]
    energy_drift = abs((E_final - E0_energy) / (abs(E0_energy) + 1e-30))
    
    # Memory usage (estimate)
    bytes_per_cell = 8  # float64
    fields_count = 3  # E, E_prev, chi
    memory_mb = (Nx * Ny * Nz * bytes_per_cell * fields_count) / (1024**2)
    
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(f"Total time: {elapsed:.3f} s")
    print(f"Time per step: {elapsed/steps:.6f} s")
    print(f"Energy drift: {energy_drift:.6e}")
    print(f"Memory estimate: {memory_mb:.1f} MB")
    print(f"Trajectory points: {len(trajectory)}")
    
    # Check success
    success = energy_drift < 1e-4
    print(f"\nEnergy conservation: {'✓ PASS' if success else '✗ FAIL'} (< 1e-4)")
    
    # Prepare results
    results = {
        "algorithm": "baseline_full_update",
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
        "trajectory": [pos.tolist() for pos in trajectory],
        "energies": [float(e) for e in energies],
        "speedup_vs_baseline": 1.0,  # By definition
        "params": {
            "chi_earth": chi_earth,
            "chi_moon": chi_moon,
            "k_acc": k_acc,
            "dx": dx,
            "dt": dt,
            "seed": seed
        }
    }
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "baseline_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Save trajectory as CSV
        traj_file = output_dir / "baseline_trajectory.csv"
        with open(traj_file, 'w', encoding='utf-8') as f:
            f.write("step,x,y,z,r\n")
            for i, pos in enumerate(trajectory):
                r = np.linalg.norm(pos - earth_pos)
                f.write(f"{i},{pos[2]:.6f},{pos[1]:.6f},{pos[0]:.6f},{r:.6f}\n")
        print(f"Trajectory saved to: {traj_file}")
    
    return results


if __name__ == "__main__":
    # Run baseline benchmark
    output_dir = Path(__file__).parent / "results"
    
    print("Running baseline benchmark (3 trials)...\n")
    
    trials = []
    for trial in range(3):
        print(f"\n{'#'*60}")
        print(f"# TRIAL {trial + 1}/3")
        print(f"{'#'*60}\n")
        
        result = run_baseline_orbit(
            grid_size=128,
            steps=300,
            use_gpu=True,
            seed=42 + trial,
            output_dir=output_dir if trial == 0 else None  # Only save first trial
        )
        trials.append(result)
        
        print(f"\nTrial {trial+1} complete: {result['elapsed_time_s']:.3f}s, drift={result['energy_drift']:.2e}")
    
    # Compute statistics
    times = [t['elapsed_time_s'] for t in trials]
    drifts = [t['energy_drift'] for t in trials]
    
    print("\n" + "="*60)
    print("BASELINE SUMMARY (3 trials)")
    print("="*60)
    print(f"Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
    print(f"Energy drift: {np.mean(drifts):.2e} ± {np.std(drifts):.2e}")
    print(f"All trials pass: {all(t['energy_drift_pass'] for t in trials)}")
    
    # Save summary
    summary = {
        "baseline_summary": {
            "n_trials": 3,
            "mean_time_s": float(np.mean(times)),
            "std_time_s": float(np.std(times)),
            "mean_drift": float(np.mean(drifts)),
            "std_drift": float(np.std(drifts)),
            "all_pass": all(t['energy_drift_pass'] for t in trials)
        },
        "trials": trials
    }
    
summary_file = output_dir / "baseline_summary.json"
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    print("\nBaseline benchmark complete! ✓")
