#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com

"""
Quick Regression Test — LFM Core Equation (v1.6, Taylor-corrected)
------------------------------------------------------------------
Purpose:
  Verify numerical stability & energy behavior of lfm_equation.py after changes.
  Uses a 1-D Gaussian pulse and a physically consistent E_prev via a
  Taylor backstep: E_prev = E0 - c*dt*∂E/∂x  (periodic gradient).

What's new vs v1.4:
  • Taylor-corrected E_prev initialization (fixes artificial energy loss).
  • Manual evolution loop to allow custom E_prev (uses lattice_step).
  • Clear CFL printout and checkpoint trend line.
  • Optional legacy comparison run (advance/E_prev=E0) for A/B verification.

PASS if |drift| < 1e-3 and values are finite.
"""

import math, time, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import core pieces
from lfm_equation import lattice_step, core_metrics, energy_total

# Ensure results directory exists
RESULTS_DIR = Path(__file__).parent.parent / "results" / "Tests" / "diagnostics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------- Config ---------------------------------
nx = 200
x = np.linspace(-10.0, 10.0, nx)
E0 = np.exp(-x**2)

# Optional χ field (kept scalar here; toggle as needed)
USE_FIELD_CHI = False
CHI_VALUE = 0.0
if USE_FIELD_CHI:
    # tiny gradient demo (kept off by default)
    CHI_VALUE = 0.01

params = {
    "dt": 0.01,
    "dx": 0.1,
    "alpha": 1.0,
    "beta": 1.0,
    "chi": CHI_VALUE,
    "gamma_damp": 0.0,
    "boundary": "periodic",
    "stencil_order": 2,
    "energy_monitor_every": 10,
    "debug": {
        "enable_diagnostics": True,
        "energy_tol": 1e-3,
        "profile_steps": 50,
        "diagnostics_path": str(RESULTS_DIR / "diagnostics_core.csv"),
    },
}

STEPS = 200
CHECKPOINT_EVERY = 50
SHOW_PLOTS = True
COMPARE_LEGACY = False  # set True to also run the old initialization for A/B

# ----------------------- Helper: periodic gradient -------------------
def periodic_dx(arr: np.ndarray, dx: float) -> np.ndarray:
    """Central periodic spatial gradient."""
    return (np.roll(arr, -1) - np.roll(arr, 1)) / (2.0 * dx)

# ----------------------- Taylor-corrected E_prev ---------------------
c = math.sqrt(params["alpha"] / params["beta"])
dt = params["dt"]
dx = params["dx"]
cfl = c * dt / dx

E_prev = E0 - c * dt * periodic_dx(E0, dx)  # ← key fix

# -------------- Manual evolution loop (so we can pass E_prev) --------
print("=== LFM Core Regression Test (v1.6) ===")
print(f"Grid: {nx}   χ-mode: {'field' if USE_FIELD_CHI else 'scalar'}")
print(f"dt={dt:.3g}, dx={dx:.3g}, steps={STEPS}")
print(f"CFL = {cfl:.3f}   (3D limit ≈ 1/sqrt(3) ≈ 0.577)")

E = E0.copy()
dbg = params["debug"]
E0_energy = energy_total(E, E_prev, dt, dx, c, params["chi"])

print("\nRunning solver with Taylor-corrected E_prev …")
drift_samples = []
t0 = time.time()
for n in range(STEPS):
    E_next = lattice_step(E, E_prev, params)
    E_prev, E = E, E_next

    # Lightweight drift sample via core_metrics
    if (n + 1) % params["energy_monitor_every"] == 0:
        met = core_metrics(E, E_prev, params, E0_energy, {
            "enable": False, "energy_tol": dbg.get("energy_tol", 1e-3),
            "check_nan": True, "edge_band": 0, "checksum_stride": 4096
        })
        if (n + 1) % CHECKPOINT_EVERY == 0:
            drift_samples.append((n + 1, met["drift"]))
            print(f"  step {n+1:4d} | drift={met['drift']:+.3e} | max|E|={met['max_abs']:.3e}")
elapsed = time.time() - t0

# Final diagnostics
met_final = core_metrics(E, E_prev, params, E0_energy, {
    "enable": False, "energy_tol": dbg.get("energy_tol", 1e-3),
    "check_nan": True, "edge_band": 0, "checksum_stride": 4096
})

print("\n=== Regression Summary (Taylor) ===")
print(f"Time elapsed: {elapsed:.3f} s  (avg {(elapsed/max(1,STEPS))*1000:.3f} ms/step)")
print(f"Energy drift: {met_final['drift']:+.3e}")
print(f"Max |E|     : {met_final['max_abs']:.3e}")
print(f"CFL ratio   : {cfl:.3f}")
print(f"Grad ratio  : {met_final.get('grad_ratio','N/A')}")
print(f"Diagnostics : {RESULTS_DIR / 'diagnostics_core.csv'}")

print("\nDrift trend (sampled checkpoints):")
for step_i, d in drift_samples:
    print(f"  step {step_i:4d}: drift={d:+.3e}")

PASS = (abs(met_final["drift"]) < 1e-3) and np.isfinite(met_final["max_abs"])
if PASS:
    print("\n✅ PASS — Stable energy and numerics verified.")
else:
    print("\n❌ FAIL — Energy drift or numeric instability detected.")
    # (we still continue if user wants legacy comparison/plots)

# ----------------------- Optional legacy comparison -------------------
if COMPARE_LEGACY:
    from lfm_equation import advance  # uses E_prev = E0 internally
    print("\n[Legacy A/B] Running legacy init path (advance with E_prev=E0)…")
    t1 = time.time()
    E_legacy = advance(E0, params, steps=STEPS, save_every=0)
    t_legacy = time.time() - t1

    # Build a comparable drift metric for legacy end-state
    # (We can't recover legacy's internal E_prev easily; compute drift against E0)
    en_legacy = energy_total(np.asarray(E_legacy), np.asarray(E0), dt, dx, c, params["chi"])
    drift_legacy = (en_legacy - E0_energy) / (abs(E0_energy) + 1e-30)

    print("[Legacy A/B] Summary:")
    print(f"  Time        : {t_legacy:.3f} s")
    print(f"  Drift (end) : {drift_legacy:+.3e}")
    print(f"  Max |E|     : {np.max(np.abs(E_legacy)):.3e}")

# ----------------------------- Plots ----------------------------------
if SHOW_PLOTS:
    plt.figure(figsize=(8, 4))
    plt.plot(x, E0, label="Initial", lw=1.6)
    plt.plot(x, np.asarray(E), label="Final (Taylor init)", lw=1.2)
    if COMPARE_LEGACY:
        plt.plot(x, np.asarray(E_legacy), label="Final (legacy init)", lw=1.0, ls="--")
    plt.xlabel("x")
    plt.ylabel("E amplitude")
    plt.title("LFM Core — 1D Regression")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Exit non-zero on fail for CI/regression harnesses
if not PASS:
    sys.exit(1)
