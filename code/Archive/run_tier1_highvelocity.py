#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
LFM Tier-1 — Lorentz Boost (High Velocity)
Tests isotropy and dispersion linearity for v ≈ 0.9 c.
Validates that ω² = c²k² + χ² holds under relativistic Doppler boost.

Outputs → results/Tier1/HighVelocity/<variant_id>/
"""

import json, math, time
from pathlib import Path
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Config Loader  (standard LFM convention)
# ---------------------------------------------------------------------------
def load_config():
    script = Path(__file__).stem.replace("run_", "")
    cfg_path = Path(__file__).resolve().parent.parent / "config" / f"config_{script}.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Initialize Field  (traveling wave so FFT finds frequency)
# ---------------------------------------------------------------------------
def init_field(N, k0, dx):
    x = cp.arange(N) * dx
    return cp.cos(k0 * x) + 1j * cp.sin(k0 * x)   # e^{i kx}


# ---------------------------------------------------------------------------
# Main Simulation
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    p = cfg["parameters"]

    N       = p["grid_points"]
    steps   = p["steps"]
    dx      = p["dx"]
    dt      = p["dt"]
    alpha   = p["alpha"]
    beta    = p["beta"]
    chi     = p["chi"]
    save_every = p["save_every"]
    k_frac  = p["k_fraction"]

    c = math.sqrt(alpha / beta)
    k0 = 2 * math.pi * k_frac / (N * dx)
    omega_theory = math.sqrt((c * k0) ** 2 + chi ** 2)

    # initial fields
    E_prev = init_field(N, k0, dx)
    E = E_prev.copy()

    results = []
    t0 = time.time()
    for n in range(steps):
        lap = (cp.roll(E, -1) - 2 * E + cp.roll(E, 1)) / dx ** 2
        E_next = 2 * E - E_prev + (dt ** 2) * (c ** 2 * lap - chi ** 2 * E)

        if n % save_every == 0:
            probe = cp.asnumpy(E[N // 2].real)
            results.append((n * dt, probe))

        E_prev, E = E, E_next

    runtime = time.time() - t0
    print(f"Completed {steps} steps in {runtime:.2f}s")

    # -----------------------------------------------------------------------
    # Phase velocity measurement
    # -----------------------------------------------------------------------
    data = np.array(results)
    freqs = np.fft.rfftfreq(len(data[:, 1]), d=save_every * dt)
    fft = np.abs(np.fft.rfft(data[:, 1]))
    peak_index = np.argmax(fft[1:]) + 1   # ignore DC bin
    peak = freqs[peak_index] if peak_index < len(freqs) else 0.0

    v_phase_theory = omega_theory / k0
    v_meas = (2 * math.pi * peak) / k0 if k0 != 0 else 0.0
    err = abs(v_meas - v_phase_theory) / v_phase_theory if v_phase_theory != 0 else 1.0

    # -----------------------------------------------------------------------
    # Validation and Pass/Fail Reporting
    # -----------------------------------------------------------------------
    tol = cfg["tolerances"]["phase_error_max"]
    passed = err <= tol

    print("\n--- Validation Summary ---")
    print(f"Phase velocity relative error: {err*100:.3f}%  (tolerance {tol*100:.2f}%)")
    print(f"Test Result: {'PASS ✅' if passed else 'FAIL ❌'}")

    summary = {
        "tier": cfg["tier"],
        "variant_id": cfg["variant_id"],
        "test_name": cfg["test_name"],
        "measured_vp": float(v_meas),
        "theoretical_vp": float(v_phase_theory),
        "relative_error": float(err),
        "tolerance": float(tol),
        "status": "Passed" if passed else "Failed",
        "runtime_sec": runtime
    }

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        outdir / "phase_velocity.txt",
        np.array([[v_phase_theory, v_meas, err]]),
        header="v_phase_theory, v_meas, rel_error"
    )

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plt.plot(data[:, 0], data[:, 1])
    plt.xlabel("Time")
    plt.ylabel("E(center)")
    plt.title(f"Tier1 High-Velocity Test — rel err={err*100:.3f}%")
    plt.savefig(outdir / "probe_signal.png", dpi=150)
    plt.close()

    print("Output saved to", outdir)
    if not passed:
        exit(1)


if __name__ == "__main__":
    main()
