#!/usr/bin/env python3
"""
Tier-2 Dispersion & Isotropy Validation — LFM 3-D (v1.3)

Purpose
-------
Validate the canonical dispersion ω^2 = c^2 k^2 + χ^2 in 3-D for
both serial (core) and threaded parallel backends, with minimal
runtime and strong diagnostics.

What's new vs v1.2
------------------
• Robust, theory-guided ω estimator: scans a narrow band around ω_th
  and picks the peak response (DFT-at-selected-freqs), not a blunt FFT.
• Zoomed diagnostic plots around the expected low frequency.
• Same traveling-wave initialization; clearer comments.
• Unchanged timing + lightweight drift sampling.

Notes
-----
– Traveling wave init:
    φ = k·x,  ω_th = sqrt(c^2|k|^2 + χ^2)
    E(t0)      = A·sin(φ)
    E(t0−dt)   = A·sin(φ − ω_th·dt)     # equivalent to adding correct velocity
– With N=64 & k = (2π/N)/dx along an axis, ω_th ≈ 0.0982 rad/s (f≈0.0156 Hz),
  i.e., the peak sits very close to DC; generic FFT plots can hide it.
"""

import math, time, csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lfm_equation import advance, core_metrics, energy_total
from lfm_parallel import run_lattice

# ----------------------------
# Config
# ----------------------------
N = 64
dx = 1.0
dt = 0.04
steps = 300
alpha, beta = 1.0, 1.0
chi = 0.0
amplitude = 1e-1          # small enough for linear regime
SHOW_PLOTS = True         # set False to suppress figures

# Ensure results directory exists
RESULTS_DIR = Path(__file__).parent / "results" / "Tests" / "diagnostics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

dirs = {
    "x": np.array([1,0,0], float),
    "y": np.array([0,1,0], float),
    "z": np.array([0,0,1], float),
    "d": np.array([1,1,1], float) / math.sqrt(3),
}

params_base = dict(
    dt=dt, dx=dx, alpha=alpha, beta=beta, chi=chi,
    boundary="periodic",
    debug={"enable_diagnostics": False},
)

# --------------------------------
# ω measurement utilities
# --------------------------------
def measure_omega_targeted(series, dt, omega_hint, span=0.30, ngrid=401):
    """
    Scan ω in [ω_hint*(1-span), ω_hint*(1+span)] and pick the maximal
    response |Σ x(t) e^{-i ω t}| after mean-removal + Hann window.
    Returns (omega_hat, freqs_Hz, power, best_idx).
    """
    x = np.asarray(series, dtype=float)
    t = np.arange(len(x)) * dt
    x = x - x.mean()
    w = np.hanning(len(x))
    xw = x * w

    wmin = max(1e-9, omega_hint * (1.0 - span))
    wmax = omega_hint * (1.0 + span)
    omegas = np.linspace(wmin, wmax, ngrid)

    # Compute complex projection for each test ω
    # (same as a DFT evaluated at arbitrary continuous frequencies)
    proj = np.array([np.vdot(xw, np.exp(-1j * w_ * t)) for w_ in omegas])
    power = np.abs(proj)
    idx = int(np.argmax(power))
    omega_hat = float(omegas[idx])
    freqs_Hz = omegas / (2.0 * math.pi)
    return omega_hat, freqs_Hz, power, idx

# --------------------------------
# Main run
# --------------------------------
results = []
probe = (N//2, N//2, N//2)
coords = np.indices((N, N, N), dtype=float)
c = math.sqrt(alpha / beta)

for label, kdir in dirs.items():
    print(f"\n=== Direction {label} ===")
    kfrac = 2.0 / N                            # moderate wavelength
    kvec  = kfrac * kdir * math.pi / dx
    omega_th = math.sqrt((c**2) * (np.linalg.norm(kvec)**2) + chi**2)

    # Traveling-wave init (adds correct initial velocity via time shift)
    phase  = (kvec[0]*coords[0] + kvec[1]*coords[1] + kvec[2]*coords[2])
    E0     = amplitude * np.sin(phase)
    Eprev0 = amplitude * np.sin(phase - dt * omega_th)

    # ---------- SERIAL ----------
    print("  Running serial …")
    params_s = dict(params_base)
    E = np.copy(E0); E_prev = np.copy(Eprev0)
    series_s = []
    t0 = time.time()

    # energy baseline for diagnostics
    E0_energy = energy_total(E, E_prev, dt, dx, c, chi)

    for n in range(steps):
        E_next = advance(E, params_s, 1)
        E_prev, E = E, E_next
        series_s.append(float(E[probe]))
        # lightweight drift sample
        if (n+1) % 20 == 0:
            met = core_metrics(E, E_prev, params_s, E0_energy, {
                "enable": False, "energy_tol": 1e-4, "check_nan": True,
                "edge_band": 0, "checksum_stride": 4096
            })
            print(f"    [serial] step {n+1:4d}  drift={met['drift']:+.3e}  "
                  f"max|E|={met['max_abs']:.3e}")

    t_serial = time.time() - t0
    w_s, f_s, pow_s, i_s = measure_omega_targeted(series_s, dt, omega_th)

    # ---------- PARALLEL ----------
    print("  Running parallel (threads 2x2x2) …")
    params_p = dict(params_base)
    E = np.copy(E0); E_prev = np.copy(Eprev0)
    series_p = []
    t0 = time.time()

    E0_energy_p = E0_energy  # identical init
    for n in range(steps):
        E_next = run_lattice(E, params_p, 1, tiles=(2,2,2))
        E_prev, E = E, E_next
        series_p.append(float(E[probe]))
        if (n+1) % 20 == 0:
            met = core_metrics(E, E_prev, params_p, E0_energy_p, {
                "enable": False, "energy_tol": 1e-4, "check_nan": True,
                "edge_band": 0, "checksum_stride": 4096
            })
            print(f"    [par]   step {n+1:4d}  drift={met['drift']:+.3e}  "
                  f"max|E|={met['max_abs']:.3e}")

    t_parallel = time.time() - t0
    w_p, f_p, pow_p, i_p = measure_omega_targeted(series_p, dt, omega_th)

    # ---------- Compare ----------
    rel_err_s = abs(w_s - omega_th) / max(omega_th, 1e-30)
    rel_err_p = abs(w_p - omega_th) / max(omega_th, 1e-30)
    backend_gap = abs(w_s - w_p) / max(omega_th, 1e-30)

    results.append({
        "dir": label,
        "omega_meas_serial": w_s,
        "omega_meas_parallel": w_p,
        "omega_theory": omega_th,
        "rel_err_serial": rel_err_s,
        "rel_err_parallel": rel_err_p,
        "backend_gap": backend_gap,
        "serial_t": t_serial,
        "parallel_t": t_parallel,
    })

    # ---------- Visualization ----------
    if SHOW_PLOTS:
        plt.figure(figsize=(11,3.2))
        # Time series
        plt.subplot(1,3,1)
        plt.plot(series_s, lw=1)
        plt.title(f"{label.upper()} Probe (serial)")
        plt.xlabel("step"); plt.ylabel("E(probe)")
        # Zoomed spectrum (serial)
        plt.subplot(1,3,2)
        plt.semilogy(f_s, pow_s, lw=1)
        plt.axvline(omega_th/(2*math.pi), ls="--", lw=1, color="k")
        plt.axvline(w_s/(2*math.pi), ls=":", lw=1, color="k")
        plt.xlim(max(0, f_s[0]), min(f_s[-1], 5*omega_th/(2*math.pi)))
        plt.title("DFT scan (serial)")
        plt.xlabel("Hz")
        # Zoomed spectrum (parallel)
        plt.subplot(1,3,3)
        plt.semilogy(f_p, pow_p, lw=1)
        plt.axvline(omega_th/(2*math.pi), ls="--", lw=1, color="k")
        plt.axvline(w_p/(2*math.pi), ls=":", lw=1, color="k")
        plt.xlim(max(0, f_p[0]), min(f_p[-1], 5*omega_th/(2*math.pi)))
        plt.title("DFT scan (parallel)")
        plt.xlabel("Hz")
        plt.tight_layout()
        plt.show()

# --------------------------------
# Summary
# --------------------------------
print("\n=== Dispersion Results ===")
for r in results:
    print(
        f"{r['dir']:>3}  ω_th={r['omega_theory']:.6f}  "
        f"ω_s={r['omega_meas_serial']:.6f} (Δ={r['rel_err_serial']*100:.3f}%)  "
        f"ω_p={r['omega_meas_parallel']:.6f} (Δ={r['rel_err_parallel']*100:.3f}%)  "
        f"gap(s|p)={r['backend_gap']*100:.3f}%  "
        f"t_s={r['serial_t']:.2f}s  t_p={r['parallel_t']:.2f}s"
    )

iso  = np.std([r["omega_meas_serial"]   for r in results]) / np.mean([r["omega_meas_serial"]   for r in results])
isoP = np.std([r["omega_meas_parallel"] for r in results]) / np.mean([r["omega_meas_parallel"] for r in results])
print(f"\nIsotropy CoV  serial={iso*100:.3f}%   parallel={isoP*100:.3f}%")

with open(RESULTS_DIR / "dispersion_results_3d.csv", "w", newline="") as f:
    fields = [
        "dir","omega_theory","omega_meas_serial","omega_meas_parallel",
        "rel_err_serial","rel_err_parallel","backend_gap","serial_t","parallel_t"
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(results)
print(f"\nWrote {RESULTS_DIR / 'dispersion_results_3d.csv'}")
