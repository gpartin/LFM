#!/usr/bin/env python3
"""
Tier-2 — Gravity Analogue (v3.0, canonical χ-gradient, serial vs parallel)

Goal
----
Validate the LFM redshift analogue implied by the canonical dispersion:
    ω^2 = c^2 k^2 + χ(x)^2
by comparing *measured* local frequencies at two probe locations with
different χ(x) values, for both serial and threaded-parallel backends.

What this script does
---------------------
• Builds a smooth 3-D χ(x) field: χ(r) = χ0 * exp( - r^2 / (2σ^2) ).
• Initializes a small-amplitude traveling wave packet (Gaussian envelope).
• Runs two identical experiments:
    1) Serial (advance)
    2) Parallel threads (run_lattice, tiles=2x2x2)
• Records probe time-series at:
    A) near center (high χ)       B) outer region (low χ)
• Estimates ω via Hann-windowed FFT for each probe/backend.
• Compares measured ω with theoretical   ω_th(x) = sqrt(c^2 k^2 + χ(x)^2)
  and also reports the *redshift ratio*   ω_A / ω_B  vs  ω_th_A / ω_th_B
• Logs lightweight energy-drift/health every 20 steps using core_metrics().
• Writes results to: gravity_analogue_results.csv
• Optional plots of probe traces & spectra.

Requirements
------------
Relies on your verified kernels:
  - lfm_equation.py v1.4  (advance, core_metrics, energy_total)
  - lfm_parallel.py v1.3  (run_lattice)

Physics knobs
-------------
This test is linear-regime by design (small amplitude, modest steps).
It’s not a stress or long-term conservation test; that can be Tier-2b.

"""

from __future__ import annotations
import math, time, csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from lfm_equation import advance, core_metrics, energy_total
from lfm_parallel import run_lattice

# ----------------------------
# Config
# ----------------------------
N = 96                     # lattice size per axis (3-D)
dx = 1.0
dt = 0.04
steps = 600               # moderate; enough cycles for clean FFT
alpha, beta = 1.0, 1.0
chi0 = 0.25               # peak curvature at center (tunable)
sigma = 18.0              # width of χ well (grid units)
amplitude = 1e-2          # keep linear
kfrac = 2.0 / N           # spatial frequency (units of π/dx)
wave_dir = np.array([1, 0, 0], float)   # along +x (simple & isotropy-agnostic)
SHOW_PLOTS = True

params_base = dict(
    dt=dt, dx=dx, alpha=alpha, beta=beta,
    boundary="periodic",
    debug={"enable_diagnostics": False},   # we sample with core_metrics()
)

# Probe positions
PROBE_A = (N//2, N//2, N//2)               # near χ peak
PROBE_B = (N//2, N//2, int(0.85*N))        # low-χ ring (periodic makes this safe)

# ----------------------------
# Helpers
# ----------------------------
def hann_fft_freq(series, dt):
    """Return (omega_peak, freqs, spectrum) using Hann window on zero-mean series."""
    x = np.asarray(series, float)
    x = x - x.mean()
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), dt)
    mag = np.abs(X)
    idx = int(np.argmax(mag))
    return 2 * math.pi * f[idx], f, mag

def radial_chi_field(N, chi0, sigma, center=None):
    """χ(r) = chi0 * exp( - r^2 / (2σ^2) ) centered at `center` (defaults to box center)."""
    if center is None:
        center = np.array([N/2, N/2, N/2], float)
    z, y, x = np.indices((N, N, N), dtype=float)
    r2 = (x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2
    return chi0 * np.exp(- r2 / (2.0 * sigma * sigma))

def gaussian_packet(N, kvec, amplitude, env_sigma, center=None):
    """Gaussian-enveloped plane wave sin(k·r) at t=0 (no phase)."""
    if center is None:
        center = np.array([N/2, N/2, N/2], float)
    z, y, x = np.indices((N, N, N), dtype=float)
    phase = (kvec[0]*x + kvec[1]*y + kvec[2]*z)
    r2 = (x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2
    env = np.exp(- r2 / (2.0 * env_sigma * env_sigma))
    return amplitude * env * np.sin(phase)

def local_omega_theory(c, k_mag, chi_val):
    return math.sqrt((c*c) * (k_mag*k_mag) + chi_val*chi_val)

@dataclass
class BackendResult:
    label: str                  # "serial" | "parallel"
    series_A: list[float]
    series_B: list[float]
    wA: float
    wB: float
    t_elapsed: float

# ----------------------------
# Build χ(x) and initial fields
# ----------------------------
c = math.sqrt(alpha / beta)
kvec = (kfrac * math.pi / dx) * wave_dir
k_mag = float(np.linalg.norm(kvec))

chi_field = radial_chi_field(N, chi0=chi0, sigma=sigma)
E0 = gaussian_packet(N, kvec, amplitude=amplitude, env_sigma=sigma*0.9)

# To impart forward travel, estimate ω using χ at the *envelope center*.
chi_center = float(chi_field[PROBE_A])
omega_init = local_omega_theory(c, k_mag, chi_center)

# Simple traveling-start (one-step back in time with center ω).
# For variable χ this is approximate but adequate for small dt and a mild well.
Eprev0 = gaussian_packet(N, kvec, amplitude=amplitude, env_sigma=sigma*0.9)
Eprev0 = Eprev0 * np.cos(dt * omega_init) - (E0) * np.sin(0)  # phase-shift proxy

# ----------------------------
# Experiment runners
# ----------------------------
def run_backend(label: str, use_parallel: bool) -> BackendResult:
    params = dict(params_base)
    params["chi"] = chi_field  # inject curvature field

    # Baselines for drift sampling
    E = np.copy(E0)
    Ep = np.copy(Eprev0)
    E0_energy = energy_total(E, Ep, dt, dx, c, chi_field)

    series_A, series_B = [], []
    t0 = time.time()

    for n in range(steps):
        if use_parallel:
            E_next = run_lattice(E, params, 1, backend="thread", tiles=(2, 2, 2))
        else:
            E_next = advance(E, params, 1)
        Ep, E = E, E_next
        series_A.append(float(E[PROBE_A]))
        series_B.append(float(E[PROBE_B]))

        # Light diagnostics every 20 steps
        if (n + 1) % 20 == 0:
            met = core_metrics(E, Ep, params, E0_energy, {
                "enable": False, "energy_tol": 1e-4, "check_nan": True,
                "edge_band": 0, "checksum_stride": 4096
            })
            tag = "par" if use_parallel else "serial"
            print(f"    [{tag}] step {n+1:4d}  drift={met['drift']:+.3e}  max|E|={met['max_abs']:.3e}")

    elapsed = time.time() - t0
    wA, fA, sA = hann_fft_freq(series_A, dt)
    wB, fB, sB = hann_fft_freq(series_B, dt)

    # Optional quick plots
    if SHOW_PLOTS:
        fig = plt.figure(figsize=(11, 3.0))
        fig.suptitle(f"{label.capitalize()} — Gravity Analogue (χ well)", y=1.02, fontsize=12)
        plt.subplot(1, 3, 1)
        plt.plot(series_A, lw=1, label="Probe A (high χ)")
        plt.plot(series_B, lw=1, label="Probe B (low χ)")
        plt.xlabel("step"); plt.ylabel("E(probe)"); plt.legend(loc="best", fontsize=8)
        plt.subplot(1, 3, 2)
        plt.semilogy(fA, sA, lw=1); plt.axvline(wA/(2*math.pi), ls="--", lw=1)
        plt.xlabel("Hz"); plt.title("FFT — Probe A")
        plt.subplot(1, 3, 3)
        plt.semilogy(fB, sB, lw=1); plt.axvline(wB/(2*math.pi), ls="--", lw=1)
        plt.xlabel("Hz"); plt.title("FFT — Probe B")
        plt.tight_layout()
        plt.show()

    return BackendResult(label=label, series_A=series_A, series_B=series_B,
                         wA=wA, wB=wB, t_elapsed=elapsed)

# ----------------------------
# Run both backends
# ----------------------------
print("\n=== Tier-2 Gravity Analogue — Starting ===")
print(f"Grid: {N}^3   c·dt/dx = {c*dt/dx:.3f}   CFL_3D limit ≈ {1/math.sqrt(3):.3f}")
print(f"χ0 = {chi0:.3f},  σ = {sigma:.1f},  kfrac = {kfrac:.5f},  |k| = {k_mag:.5f}")

print("\nRunning serial …")
res_serial = run_backend("serial", use_parallel=False)

print("\nRunning parallel (threads 2x2x2) …")
res_parallel = run_backend("parallel", use_parallel=True)

# ----------------------------
# Local theory (per-probe) & summary
# ----------------------------
chi_A = float(chi_field[PROBE_A])
chi_B = float(chi_field[PROBE_B])
wA_th = local_omega_theory(c, k_mag, chi_A)
wB_th = local_omega_theory(c, k_mag, chi_B)
ratio_th = wA_th / wB_th

def pct(x): return 100.0 * x

print("\n=== Results (per backend) ===")
for res in (res_serial, res_parallel):
    ratio_meas = res.wA / res.wB if res.wB != 0 else float("nan")
    err_A = abs(res.wA - wA_th) / wA_th
    err_B = abs(res.wB - wB_th) / wB_th
    err_ratio = abs(ratio_meas - ratio_th) / ratio_th
    print(
        f"[{res.label:8s}]  ωA={res.wA:.6e} (th {wA_th:.6e}, Δ={pct(err_A):.3f}%)   "
        f"ωB={res.wB:.6e} (th {wB_th:.6e}, Δ={pct(err_B):.3f}%)   "
        f"ratio ωA/ωB={ratio_meas:.6f} (th {ratio_th:.6f}, Δ={pct(err_ratio):.3f}%)   "
        f"t={res.t_elapsed:.2f}s"
    )

# Backend parity on ratio (most important for parallel correctness)
ratio_serial = res_serial.wA / res_serial.wB
ratio_parallel = res_parallel.wA / res_parallel.wB
gap_ratio = abs(ratio_serial - ratio_parallel) / max(ratio_th, 1e-30)
print(f"\nBackend parity (ratio gap): {pct(gap_ratio):.3f}%")

# ----------------------------
# CSV
# ----------------------------
rows = []
for res in (res_serial, res_parallel):
    rows.append({
        "backend": res.label,
        "omega_A_meas": res.wA,
        "omega_B_meas": res.wB,
        "omega_A_theory": wA_th,
        "omega_B_theory": wB_th,
        "ratio_meas": res.wA / res.wB,
        "ratio_theory": ratio_th,
        "rel_err_A": abs(res.wA - wA_th) / wA_th,
        "rel_err_B": abs(res.wB - wB_th) / wB_th,
        "rel_err_ratio": abs(res.wA/res.wB - ratio_th) / ratio_th,
        "t_elapsed": res.t_elapsed,
        "c": c,
        "k_mag": k_mag,
        "chi_A": chi_A,
        "chi_B": chi_B,
        "N": N, "dt": dt, "dx": dx, "steps": steps,
        "chi0": chi0, "sigma": sigma, "amplitude": amplitude,
    })

out = Path("gravity_analogue_results.csv")
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nWrote {out.name}")
