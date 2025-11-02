#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com

"""
LFM Tier-1 — Dispersion Curve Diagnostic (4th-Order Stencil, Continuum)
Sweeps multiple k-fractions to measure ω(k) vs theory √(c²k²+χ²).

Outputs → results/Tier1/DispersionCurve_Continuum/
"""

import json, math, time
from pathlib import Path
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------- Config Loader ------------------------------
def load_config():
    script = Path(__file__).stem.replace("run_", "")
    cfg_path = Path(__file__).resolve().parent.parent / "config" / f"config_{script}.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

# --------------------------- Spatial Laplacian helper -----------------------
def laplacian(E, dx, order=2):
    if order == 2:
        return (cp.roll(E, -1) - 2*E + cp.roll(E, 1)) / dx**2
    elif order == 4:
        # 4th-order central difference: (-1, 16, -30, 16, -1)/(12 dx^2)
        return (-cp.roll(E, 2) + 16*cp.roll(E, 1) - 30*E + 16*cp.roll(E, -1) - cp.roll(E, -2)) / (12*dx**2)
    else:
        raise ValueError("Unsupported stencil order; use 2 or 4")

# ---------------------------- Standing-wave init ----------------------------
def init_field(N, k0, dx):
    x = cp.arange(N) * dx
    return cp.cos(k0 * x)  # real standing wave so center oscillates

# -------------------------- Frequency measurement --------------------------
def measure_frequency(N, steps, dx, dt, alpha, beta, chi, save_every, k_frac, stencil_order):
    c = math.sqrt(alpha / beta)
    k0 = k_frac * (math.pi / dx)  # fraction of Nyquist
    omega_theory = math.sqrt((c * k0) ** 2 + chi ** 2)

    E_prev = init_field(N, k0, dx)
    E = E_prev.copy()
    trace = []

    for n in range(steps):
        lap = laplacian(E, dx, order=stencil_order)
        E_next = 2 * E - E_prev + (dt ** 2) * (c ** 2 * lap - chi ** 2 * E)

        if n % save_every == 0:
            trace.append(float(cp.asnumpy(E[N // 2])))

        E_prev, E = E, E_next

    data = np.asarray(trace, dtype=float)
    window = np.hanning(len(data))
    yf = np.abs(np.fft.rfft(data * window))
    xf = np.fft.rfftfreq(len(data), d=save_every * dt)

    peak_idx = 1 + np.argmax(yf[1:]) if len(yf) > 1 else 0
    f_peak = xf[peak_idx] if peak_idx < len(xf) else 0.0
    omega_meas = 2.0 * math.pi * f_peak
    return k0, omega_meas, omega_theory

# ----------------------------------- Main -----------------------------------
def main():
    cfg = load_config()
    p = cfg["parameters"]

    N          = p["grid_points"]
    steps      = p["steps"]
    dx         = p["dx"]
    dt         = p["dt"]
    alpha      = p["alpha"]
    beta       = p["beta"]
    chi        = p["chi"]
    save_every = p["save_every"]
    k_list     = p.get("k_sweep", [p["k_fraction"]])
    stencil_order = int(p.get("stencil_order", 4))  # default to 4th-order

    c = math.sqrt(alpha / beta)
    cfl = c * dt / dx
    if cfl > 0.9:
        raise ValueError(f"CFL too high (c*dt/dx={cfl:.3f} > 0.9). Reduce dt or increase dx.")

    project_root = Path(__file__).resolve().parent.parent
    outdir = (project_root / cfg["output_dir"]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for k_frac in k_list:
        print(f"Running k_fraction = {k_frac:.3f}  (stencil_order={stencil_order})")
        k0, omega_meas, omega_theory = measure_frequency(
            N, steps, dx, dt, alpha, beta, chi, save_every, k_frac, stencil_order
        )
        rows.append((k0, omega_meas, omega_theory))
    print(f"\nCompleted sweep in {time.time() - t0:.2f}s")

    arr = np.array(rows, dtype=float)
    k_vals, omega_meas, omega_theory = arr.T
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs(omega_meas - omega_theory) / np.where(omega_theory == 0, 1.0, omega_theory)

    np.savetxt(outdir / "dispersion_curve.csv",
               np.column_stack([k_vals, omega_meas, omega_theory, err]),
               header="k, omega_meas, omega_theory, rel_error")

    plt.figure(figsize=(6,4))
    plt.plot(k_vals, omega_theory, "k--", label="Theory √(c²k²+χ²)")
    plt.plot(k_vals, omega_meas, "o-", label="Measured (standing wave)")
    plt.xlabel("k"); plt.ylabel("ω"); plt.grid(True)
    plt.title("Tier-1 Dispersion Curve — Continuum (4th-Order)")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "dispersion_curve.png", dpi=150); plt.close()

    max_err = float(np.max(err))
    tol = float(cfg["tolerances"]["phase_error_max"])
    print(f"Output saved to {outdir}")
    print(f"\nMax relative error across sweep: {max_err*100:.3f}%  (tol {tol*100:.2f}%)")
    print(f"Overall Result: {'PASS ✅' if max_err <= tol else 'FAIL ❌'}")

if __name__ == "__main__":
    main()
