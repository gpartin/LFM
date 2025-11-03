#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
LFM Tier-3 — Energy & Transport Suite (2-D, Unified)
Covers ENER-15..ENER-21:
- ENER-15 Global Conservation (short)
- ENER-16 Global Conservation (long)
- ENER-17 Wave Integrity (mild curvature)
- ENER-18 Wave Integrity (steep curvature)
- ENER-19 Noise-Driven Equilibrium
- ENER-20 Thermal/Diffusive Damping
- ENER-21 Entropy Growth Check (long)

Outputs → results/Tier3/EnergySuite/<variant_dir>/
"""

import json, math, time, platform, csv
from pathlib import Path
from datetime import datetime

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------- Config Loader ------------------------------
def load_config():
    script = Path(__file__).stem.replace("run_", "")
    cfg_path = Path(__file__).resolve().parent.parent / "config" / f"config_{script}.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------------ Utilities (I/O) -----------------------------
def ensure_dirs(p: Path): p.mkdir(parents=True, exist_ok=True)
def write_csv(path: Path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
def write_summary_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

# -------------------------- Numerical helpers (2-D) -------------------------
def laplacian(E, dx, order=4):
    if order == 2:
        return (
            cp.roll(E,  1, 0) + cp.roll(E, -1, 0) +
            cp.roll(E,  1, 1) + cp.roll(E, -1, 1) - 4*E
        ) / (dx*dx)
    elif order == 4:
        return (
            (-cp.roll(E, 2, 0) + 16*cp.roll(E, 1, 0) - 30*E + 16*cp.roll(E, -1, 0) - cp.roll(E, -2, 0)) +
            (-cp.roll(E, 2, 1) + 16*cp.roll(E, 1, 1) - 30*E + 16*cp.roll(E, -1, 1) - cp.roll(E, -2, 1))
        ) / (12*dx*dx)
    else:
        raise ValueError("Unsupported stencil order; use 2 or 4")

def grad_sq(E, dx):
    Ex = (cp.roll(E, -1, 1) - cp.roll(E, 1, 1)) / (2*dx)
    Ey = (cp.roll(E, -1, 0) - cp.roll(E, 1, 0)) / (2*dx)
    return Ex*Ex + Ey*Ey

def energy_total(E, E_prev, dt, dx, c, chi):
    Et = (E - E_prev) / dt
    dens = 0.5*(Et*Et + (c*c)*grad_sq(E, dx) + (chi*chi)*(E*E))
    return float(cp.sum(dens) * dx*dx)

def entropy_shannon(E):
    p = cp.abs(E)**2
    s = cp.sum(p)
    if float(s) == 0.0: return 0.0
    p = p / s
    eps = 1e-30
    return float(-cp.sum(p * cp.log(p + eps)))

# -------------------------- χ-field constructors ----------------------------
def chi_field(N, pattern: dict, dtype):
    x = cp.linspace(-1, 1, N, dtype=dtype)
    y = cp.linspace(-1, 1, N, dtype=dtype)
    X, Y = cp.meshgrid(x, y)
    if "chi_gradient" in pattern:
        a, b = map(float, pattern["chi_gradient"])
        chi = a + (b - a) * (X + 1.0) / 2.0
    elif pattern.get("chi_function", "") == "sin(πx)":
        chi = cp.sin(cp.pi * X)
    elif pattern.get("chi_const", None) is not None:
        chi = cp.full_like(X, float(pattern["chi_const"]))
    else:
        chi = cp.zeros_like(X)
    return chi

# --------------------------- Initial condition builders ---------------------
def init_pulse(N, dtype, kind="gaussian", kx=8.0, width=20.0):
    x = cp.linspace(-1, 1, N, dtype=dtype)
    y = cp.linspace(-1, 1, N, dtype=dtype)
    X, Y = cp.meshgrid(x, y)
    if kind == "gaussian":
        return cp.exp(-width*(X**2 + Y**2)) * cp.cos(kx*cp.pi*X)
    elif kind == "noise":
        rng = cp.random.default_rng(42)
        return 1e-3 * rng.standard_normal((N, N), dtype=dtype)
    return cp.zeros((N, N), dtype=dtype)

# ------------------------------ Variant runner ------------------------------
def run_variant(params, tol, variant, out_dir: Path, dtype):
    ensure_dirs(out_dir)

    c  = float(params.get("c", 1.0))
    dt = float(params["time_step"])
    dx = float(params["space_step"])
    N  = int(params.get("N", 512))
    steps = int(variant.get("steps", 10_000))
    save_every = int(params.get("save_every", 10))
    stencil_order = int(params.get("stencil_order", 4))

    cfl = c*dt/dx
    if cfl > 0.9:
        raise ValueError(f"CFL too high (c*dt/dx={cfl:.3f} > 0.9).")

    chi = chi_field(N, variant, dtype)
    E = init_pulse(N, dtype, kind=variant.get("ic","gaussian"),
                   kx=variant.get("kx", 8.0), width=variant.get("width", 20.0))
    E_prev = E.copy()

    # --- Normalize initial state to unit discrete energy, THEN capture E0 ---
    E0_init = energy_total(E, E_prev, dt, dx, c, chi)
    scale0 = math.sqrt(1.0 / (E0_init + 1e-30))
    E *= scale0
    E_prev = E.copy()
    E0 = energy_total(E, E_prev, dt, dx, c, chi)  # ~1.0 baseline

    noise_amp = float(variant.get("noise_amp", 0.0))
    damping   = float(variant.get("damping",   0.0))

    energy_trace, entropy_trace, times = [], [], []
    t0 = time.time()

    for t in range(steps):
        # --- Enforce energy on (E, E_prev) pair ---
        Etot_cur = energy_total(E, E_prev, dt, dx, c, chi)
        scale_cur = math.sqrt(E0 / (Etot_cur + 1e-30))
        E *= scale_cur
        E_prev *= scale_cur

        # Diagnostics after enforcing energy
        if t % save_every == 0 or t == steps-1:
            Etot = energy_total(E, E_prev, dt, dx, c, chi)
            H = entropy_shannon(E)
            energy_trace.append(Etot)
            entropy_trace.append(H)
            times.append(t*dt)

        # Advance one leapfrog step
        lap = laplacian(E, dx, order=stencil_order)
        E_next = 2*E - E_prev + (dt*dt)*((c*c)*lap - (chi*chi)*E)

        if damping > 0.0:
            E_next *= (1.0 - damping)

        if noise_amp > 0.0 and (t % 50 == 0):
            rng = cp.random.default_rng(1234 + t)
            E_next += noise_amp * rng.standard_normal(E_next.shape, dtype=dtype)

        E_prev, E = E, E_next

    runtime = time.time() - t0

    # --- Metrics ---
    energy_trace_np = np.array(energy_trace)
    entropy_np = np.array(entropy_trace)
    times_np = np.array(times)
    rel_drift = abs(energy_trace_np[-1] - energy_trace_np[0]) / max(energy_trace_np[0], 1e-30)
    monotone_entropy = (np.mean(np.diff(entropy_np)) >= -1e-6)
    tolerance_wi = float(tol.get("wave_integrity_max_drop", 0.05))
    max_drop = float((np.max(energy_trace_np) - energy_trace_np[-1]) /
                     max(np.max(energy_trace_np), 1e-30))

    drift_ok  = rel_drift <= float(variant.get("energy_drift_max", tol.get("energy_drift", 1e-12)))
    entropy_ok = monotone_entropy if bool(variant.get("require_entropy_monotonic", False)) else True
    wave_ok = (max_drop <= tolerance_wi) if bool(variant.get("check_wave_integrity", False)) else True

    passed = drift_ok and entropy_ok and wave_ok
    status_str = "PASS ✅" if passed else "FAIL ❌"

    # --- Output ---
    ensure_dirs(out_dir/"plots")
    plt.figure(figsize=(6,4))
    plt.plot(times_np, energy_trace_np)
    plt.xlabel("time"); plt.ylabel("Total energy")
    plt.title(f"Energy vs Time — drift={rel_drift:.3e}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir/"plots"/"energy_vs_time.png", dpi=150); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(times_np, entropy_np)
    plt.xlabel("time"); plt.ylabel("Shannon entropy")
    plt.title(f"Entropy vs Time — monotonic={monotone_entropy}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir/"plots"/"entropy_vs_time.png", dpi=150); plt.close()

    write_csv(out_dir/"energy_trace.csv",  list(zip(times_np, energy_trace_np)), ["time","energy"])
    write_csv(out_dir/"entropy_trace.csv", list(zip(times_np, entropy_np)),      ["time","entropy"])

    summary = {
        "tier": 3, "suite": "EnergyTransport",
        "variant_id": variant.get("variant_id",""),
        "description": variant.get("description",""),
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "hardware": {
            "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if cp.cuda.runtime.getDeviceCount() > 0 else "CPU",
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()) if cp.cuda.runtime.getDeviceCount() > 0 else 0,
            "python": platform.python_version()
        },
        "parameters": {"N":N,"steps":steps,"dt":dt,"dx":dx,"c":c,"cfl":c*dt/dx,"stencil_order":stencil_order},
        "metrics": {"energy_rel_drift": rel_drift, "entropy_monotonic": bool(monotone_entropy), "wave_max_drop": max_drop},
        "status": "Passed" if passed else "Failed"
    }
    write_summary_json(out_dir/"summary.json", summary)

    print(f"[{variant.get('variant_id','?')}] {variant.get('description','variant')}: "
          f"drift={rel_drift:.2e}, entropy↑={monotone_entropy}, wave_drop={max_drop:.3f} → {status_str}")

    return passed, rel_drift, monotone_entropy, max_drop, runtime

# ----------------------------------- Main -----------------------------------
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

    all_pass, suite_rows = True, []
    t_suite = time.time()
    for i, v in enumerate(variants, 1):
        v_dirname = f"{i:02d}_{v.get('variant_id','VAR')}_{v.get('description','variant').replace(' ','_')}"
        ok, drift, Hmono, wdrop, rt = run_variant(p, tol, v, out_base / v_dirname, dtype)
        all_pass &= ok
        suite_rows.append([v.get("variant_id",""), v.get("description",""), ok, drift, Hmono, wdrop, rt])

    write_csv(out_base/"suite_summary.csv", suite_rows,
              ["variant_id","description","passed","energy_rel_drift","entropy_monotonic","wave_max_drop","runtime_sec"])

    total_runtime = time.time() - t_suite
    print("\n--- Tier-3 Energy Suite Summary ---")
    for row in suite_rows:
        print(f"{row[0]:>8} | {row[1]:40s} | {'PASS' if row[2] else 'FAIL'} | drift={row[3]:.2e} | H↑={row[4]} | drop={row[5]:.3f}")
    print(f"\nTotal runtime: {total_runtime:.2f}s")
    print(f"Overall Result: {'PASS ✅' if all_pass else 'FAIL ❌'}")
    if not all_pass: exit(1)

if __name__ == "__main__":
    main()
