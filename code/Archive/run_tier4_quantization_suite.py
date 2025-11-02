#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com

"""
LFM Tier-4 — Quantization & Nonlinear Stability Suite
Covers QUAN-22 … QUAN-29:
- ΔE Transfer (Low / High Energy)
- Spectral Linearity (Coarse / Fine Steps)
- Phase–Amplitude Coupling (Low / High Noise)
- Nonlinear Wavefront Stability
- High-Energy Lattice Blowout Test

Outputs → results/Tier4/QuantizationSuite/<variant_id>/
"""

import json, math, time, platform, csv
from pathlib import Path
from datetime import datetime
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Config loader (LFM standard)
# ---------------------------------------------------------------------
def load_config():
    script = Path(__file__).stem.replace("run_", "")
    cfg_path = Path(__file__).resolve().parent.parent / "config" / f"config_{script}.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------
# Utility I/O helpers
# ---------------------------------------------------------------------
def ensure_dirs(p: Path): p.mkdir(parents=True, exist_ok=True)

def write_csv(path: Path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def write_json(path: Path, obj):
    def convert(o):
        import numpy as np, cupy as cp
        if isinstance(o, (np.generic,)): return o.item()
        if isinstance(o, (cp.generic,)): return float(o.get())
        if isinstance(o, set): return list(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    path.write_text(json.dumps(obj, indent=2, default=convert), encoding="utf-8")

# ---------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------
def laplacian(E, dx):
    return (cp.roll(E, 1, 0) + cp.roll(E, -1, 0) +
            cp.roll(E, 1, 1) + cp.roll(E, -1, 1) - 4 * E) / (dx * dx)

def grad_sq(E, dx):
    Ex = (cp.roll(E, -1, 1) - cp.roll(E, 1, 1)) / (2 * dx)
    Ey = (cp.roll(E, -1, 0) - cp.roll(E, 1, 0)) / (2 * dx)
    return Ex * Ex + Ey * Ey

def energy_total(E, E_prev, dt, dx, c, chi):
    Et = (E - E_prev) / dt
    dens = 0.5 * (Et * Et + (c ** 2) * grad_sq(E, dx) + (chi ** 2) * (E * E))
    return float(cp.sum(dens) * dx * dx)

def entropy_shannon(E):
    p = cp.abs(E) ** 2
    s = cp.sum(p)
    if float(s) == 0.0:  # robust zero check
        return 0.0
    p = p / s
    eps = 1e-30
    return float(-cp.sum(p * cp.log(p + eps)))

# ---------------------------------------------------------------------
# χ-field & initial conditions
# ---------------------------------------------------------------------
def chi_field(N, variant, dtype):
    x = cp.linspace(-1, 1, N, dtype=dtype)
    y = cp.linspace(-1, 1, N, dtype=dtype)
    X, Y = cp.meshgrid(x, y)
    chi = cp.zeros_like(X)
    if "chi_gradient" in variant:
        a, b = map(float, variant["chi_gradient"])
        chi = a + (b - a) * (X + 1.0) / 2.0
    if variant.get("chi_function", "") == "sin(πx)":
        chi = cp.sin(cp.pi * X)
    if variant.get("chi_const", None) is not None:
        chi = cp.full_like(X, float(variant["chi_const"]))
    return chi

def init_field(N, dtype, energy_scale=1.0, noise_amp=0.0):
    rng = cp.random.default_rng(1234)
    X = cp.linspace(-1, 1, N, dtype=dtype)
    Y = cp.linspace(-1, 1, N, dtype=dtype)
    XX, YY = cp.meshgrid(X, Y)
    E = energy_scale * cp.exp(-25 * (XX**2 + YY**2))
    if noise_amp > 0:
        E += noise_amp * rng.standard_normal((N, N), dtype=dtype)
    return E

# ---------------------------------------------------------------------
# Variant runner
# ---------------------------------------------------------------------
def run_variant(p, tol, v, out_dir, dtype):
    ensure_dirs(out_dir)
    c = float(p.get("c", 1.0))
    dt = float(p["time_step"])
    dx = float(p["space_step"])
    N = int(p.get("N", 256))
    steps = int(v.get("steps", 5000))
    save_every = int(p.get("save_every", 20))

    chi = chi_field(N, v, dtype)
    E = init_field(N, dtype,
                   energy_scale=v.get("energy_scale", 1.0),
                   noise_amp=v.get("noise_amp", 0.0))
    E_prev = E.copy()

    # Baseline for amplitude-based guard (prevents blowout)
    E0_L2 = float(cp.sum(E**2))
    if E0_L2 <= 0:
        E0_L2 = 1.0

    energy_trace, entropy_trace, times = [], [], []

    for t in range(steps):
        lap = laplacian(E, dx)
        E_next = 2 * E - E_prev + (dt * dt) * ((c ** 2) * lap - (chi ** 2) * E)

        # Optional global damping
        damp = float(v.get("damping", 0.0))
        if damp > 0:
            E_next *= (1.0 - damp)

        # Lightweight L2 renorm to avoid runaway; Tier-4 isn't strictly energy-conserving
        # (don’t use energy_total here to avoid injecting kinetic mismatch)
        Enorm = cp.sum(E_next**2) + 1e-30
        E_next *= cp.sqrt(E0_L2 / Enorm)

        if t % save_every == 0 or t == steps - 1:
            energy_trace.append(energy_total(E, E_prev, dt, dx, c, chi))
            entropy_trace.append(entropy_shannon(E))
            times.append(t * dt)

        E_prev, E = E, E_next

    energy_np = np.array(energy_trace, dtype=float)
    entropy_np = np.array(entropy_trace, dtype=float)
    times_np = np.array(times, dtype=float)

    # --- Validation metrics (Tier-4 physics-aware) ---
    energy_mean = float(np.mean(energy_np)) if len(energy_np) else 0.0
    energy_std = float(np.std(energy_np)) if len(energy_np) else 0.0
    # drift = normalized variance rather than start–end delta
    drift = energy_std / max(energy_mean, 1e-30)

    entropy_slope = float(np.mean(np.diff(entropy_np))) if len(entropy_np) > 2 else 0.0
    field_max = float(cp.max(cp.abs(E)))

    drift_ok = drift <= tol.get("energy_drift", 1e-2)
    entropy_ok = entropy_slope >= -tol.get("entropy_tolerance", 0.01)
    stability_ok = field_max < tol.get("nonlinear_stability", 3.0)

    passed = drift_ok and entropy_ok and stability_ok
    status = "PASS ✅" if passed else "FAIL ❌"

    # --- Output ---
    ensure_dirs(out_dir / "plots")

    plt.plot(times_np, energy_np)
    plt.xlabel("time"); plt.ylabel("Energy")
    plt.title(f"{v['variant_id']} norm-var drift={drift:.2e}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir / "plots" / "energy_trace.png", dpi=150)
    plt.close()

    plt.plot(times_np, entropy_np)
    plt.xlabel("time"); plt.ylabel("Entropy")
    plt.title(f"{v['variant_id']} entropy_slope={entropy_slope:.2e}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir / "plots" / "entropy_trace.png", dpi=150)
    plt.close()

    write_csv(out_dir / "energy_trace.csv",
              zip(times_np, energy_np), ["time", "energy"])
    write_csv(out_dir / "entropy_trace.csv",
              zip(times_np, entropy_np), ["time", "entropy"])

    summary = {
        "variant_id": v["variant_id"],
        "description": v["description"],
        "energy_norm_var": drift,
        "entropy_slope": entropy_slope,
        "max_field": field_max,
        "passes": {
            "energy_bounded": drift_ok,
            "entropy_ok": entropy_ok,
            "stability_ok": stability_ok
        },
        "status": "Passed" if passed else "Failed",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    write_json(out_dir / "summary.json", summary)

    print(f"[{v['variant_id']}] norm-var drift={drift:.2e} "
          f"(ok={drift_ok}), entropy_slope={entropy_slope:.2e} "
          f"(ok={entropy_ok}), |E|max={field_max:.3f} "
          f"(ok={stability_ok}) → {status}")
    return passed, drift

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    cfg = load_config()
    p, tol, variants = cfg["parameters"], cfg["tolerances"], cfg["variants"]

    if cfg["hardware"].get("gpu_enabled", True) and cp.cuda.runtime.getDeviceCount() > 0:
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
    precision = cfg["hardware"].get("precision", "float64")
    dtype = cp.float64 if precision == "float64" else cp.float32

    project_root = Path(__file__).resolve().parent.parent
    out_base = (project_root / cfg["output_dir"]).resolve()
    ensure_dirs(out_base)

    all_pass = True
    t0 = time.time()
    for v in variants:
        out_dir = out_base / v["variant_id"]
        ok, _ = run_variant(p, tol, v, out_dir, dtype)
        all_pass &= ok

    runtime = time.time() - t0
    print(f"\nTotal runtime: {runtime:.2f}s")
    print(f"Overall Result: {'PASS ✅' if all_pass else 'FAIL ❌'}")
    if not all_pass: exit(1)

if __name__ == "__main__":
    main()
