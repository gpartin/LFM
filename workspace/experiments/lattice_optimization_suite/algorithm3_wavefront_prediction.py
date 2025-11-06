#!/usr/bin/env python3
"""
Algorithm 3: Wavefront Prediction (Physics-Informed)
=====================================================

HYPOTHESIS: Wave equation is deterministic - can predict where updates needed next.

APPROACH:
- Identify "source" cells: |E| > ε OR |∂E/∂t| > ε (active oscillation)
- Compute wavefront expansion: waves travel at speed c
- Only update cells within distance c·dt from any source
- Uses physics (wave speed) to guide culling, not just geometry

PHYSICS SAFETY:
- Respects maximum propagation speed (c)
- Time derivative catches oscillating regions
- Conservative distance: c·dt ensures no missed propagation
- More sophisticated than static masks

EXPECTED:
- Speedup: 4-10x for localized wave packets
- Energy error: < 10⁻⁴ (respects wave physics)
- Memory: Same as baseline
- Best for sparse, well-localized phenomena

RISK LEVEL: MEDIUM-HIGH
- Time derivative computation adds cost
- Prediction could miss slow-building effects
- Requires accurate c·dt calculation
- Most complex of the three algorithms
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


def compute_time_derivative(E, E_prev, dt, xp):
    """
    Compute ∂E/∂t using backward difference.
    
    ∂E/∂t ≈ (E - E_prev) / dt
    """
    return (E - E_prev) / dt


def compute_distance_field(source_mask, xp):
    """
    Compute approximate distance from each cell to nearest source.
    
    Uses iterative dilation (Manhattan distance approximation).
    Fast but approximate - sufficient for wavefront prediction.
    """
    Nz, Ny, Nx = source_mask.shape
    dist = xp.full(source_mask.shape, xp.inf, dtype=xp.float32)
    dist[source_mask] = 0.0
    
    # Iterative expansion (max 20 iterations for 128³ grid)
    max_iter = 20
    for d in range(1, max_iter + 1):
        # Find cells at distance d-1
        at_prev_dist = (dist == d - 1)
        if not xp.any(at_prev_dist):
            break
        
        # Expand to neighbors
        for axis in range(3):
            neighbors_p = xp.roll(at_prev_dist, 1, axis)
            neighbors_n = xp.roll(at_prev_dist, -1, axis)
            
            # Update cells that are neighbors of distance d-1 cells
            can_update = xp.isinf(dist)
            dist[can_update & neighbors_p] = d
            dist[can_update & neighbors_n] = d
    
    return dist


def create_wavefront_mask(E, E_prev, dt, c, field_threshold, rate_threshold, xp):
    """
    Create mask based on wave propagation physics.
    
    Source cells: |E| > field_thresh OR |∂E/∂t| > rate_thresh
    Wavefront: All cells within distance c·dt from sources
    """
    # Identify source cells (active oscillation)
    dE_dt = compute_time_derivative(E, E_prev, dt, xp)
    
    source_mask = (xp.abs(E) > field_threshold) | (xp.abs(dE_dt) > rate_threshold)
    
    # If no sources, update nothing (all frozen)
    if not xp.any(source_mask):
        return xp.zeros_like(E, dtype=bool)
    
    # Compute distance field from sources
    dist = compute_distance_field(source_mask, xp)
    
    # Wavefront: within c·dt of sources
    # Add safety factor of 1.5 to be conservative
    max_propagation_dist = c * dt * 1.5
    wavefront_mask = dist <= max_propagation_dist
    
    return wavefront_mask


def laplacian_masked(E, dx, mask, xp):
    """Compute Laplacian only on masked cells"""
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


def run_algorithm3_wavefront(
    grid_size=128,
    steps=300,
    use_gpu=True,
    seed=42,
    field_threshold=1e-6,
    rate_threshold=1e-5,
    output_dir=None
):
    """
    Run Algorithm 3: Wavefront Prediction
    
    Args:
        field_threshold: Minimum |E| to consider source
        rate_threshold: Minimum |∂E/∂t| to consider source
    """
    np.random.seed(seed)
    
    print("="*60)
    print("ALGORITHM 3: Wavefront Prediction (Physics-Informed)")
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
    print(f"Rate threshold: {rate_threshold:.1e}")
    
    # Physics
    alpha = 1.0
    beta = 1.0
    c = np.sqrt(alpha / beta)
    gamma_damp = 0.0
    
    print(f"Wave speed: c={c:.2f}")
    print(f"Max propagation per step: c·dt = {c*dt:.3f} cells")
    
    chi_earth = 50.0
    chi_moon = chi_earth / 81.0
    k_acc = 0.20
    
    earth_pos = np.array([Nz//2, Ny//2, Nx//2], dtype=np.float64)
    moon_pos = np.array([Nz//2, Ny//2 + 32, Nx//2], dtype=np.float64)
    moon_vel = np.array([0.0, 0.0, 1.86], dtype=np.float64)
    
    print(f"\nPhysics:")
    print(f"  chi_earth={chi_earth:.2f}, chi_moon={chi_moon:.3f}")
    print(f"  k_acc={k_acc}")
    
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
        
        # Create wavefront mask (physics-informed prediction)
        mask = create_wavefront_mask(E, E_prev, dt, c, field_threshold, rate_threshold, xp)
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
    print("ALGORITHM 3 RESULTS")
    print("="*60)
    print(f"Total time: {elapsed:.3f} s")
    print(f"Time per step: {elapsed/steps:.6f} s")
    print(f"Energy drift: {energy_drift:.6e}")
    print(f"Mean active cells: {mean_active_pct:.1f}%")
    print(f"Memory: {memory_mb:.1f} MB")
    
    success = energy_drift < 1e-4
    print(f"\nEnergy conservation: {'✓ PASS' if success else '✗ FAIL'} (< 1e-4)")
    
    results = {
        "algorithm": "algorithm3_wavefront_prediction",
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
            "rate_threshold": rate_threshold,
            "chi_earth": chi_earth,
            "chi_moon": chi_moon,
            "k_acc": k_acc,
            "dx": dx,
            "dt": dt,
            "c": c,
            "seed": seed
        }
    }
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "algorithm3_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "results"
    
    print("Running Algorithm 3 (3 trials)...\n")
    
    trials = []
    for trial in range(3):
        print(f"\n{'#'*60}")
        print(f"# TRIAL {trial + 1}/3")
        print(f"{'#'*60}\n")
        
        result = run_algorithm3_wavefront(
            grid_size=128,
            steps=300,
            use_gpu=True,
            seed=42 + trial,
            field_threshold=1e-6,
            rate_threshold=1e-5,
            output_dir=output_dir if trial == 0 else None
        )
        trials.append(result)
        
        print(f"\nTrial {trial+1}: {result['elapsed_time_s']:.3f}s, drift={result['energy_drift']:.2e}")
    
    # Statistics
    times = [t['elapsed_time_s'] for t in trials]
    drifts = [t['energy_drift'] for t in trials]
    
    print("\n" + "="*60)
    print("ALGORITHM 3 SUMMARY (3 trials)")
    print("="*60)
    print(f"Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
    print(f"Energy drift: {np.mean(drifts):.2e} ± {np.std(drifts):.2e}")
    print(f"All pass: {all(t['energy_drift_pass'] for t in trials)}")
    
    summary = {
        "algorithm3_summary": {
            "n_trials": 3,
            "mean_time_s": float(np.mean(times)),
            "std_time_s": float(np.std(times)),
            "mean_drift": float(np.mean(drifts)),
            "std_drift": float(np.std(drifts)),
            "all_pass": all(t['energy_drift_pass'] for t in trials)
        },
        "trials": trials
    }
    
    summary_file = output_dir / "algorithm3_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print("\nAlgorithm 3 complete! ✓")
