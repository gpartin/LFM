#!/usr/bin/env python3
"""
lfm_diagnostics.py — unified diagnostic utilities for all LFM tiers.

Outputs per test (under .../<TestID>/diagnostics/):
  - field_spectrum.csv, field_spectrum.png
  - energy_flow.csv,   energy_flow.png
  - phase_corr.csv,    phase_corr.png

Public API (dimension-agnostic, works with CuPy or NumPy):
  energy_total(E, E_prev, dt, dx, c, chi)
  field_spectrum(E, dx, outdir)
  energy_flow(E_series, dt, dx, c, outdir)
  phase_corr(E_series, outdir)
"""

from pathlib import Path
import numpy as np

# Optional CuPy support (module works with pure NumPy as well)
try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = np  # fallback shim: we’ll only use numpy ops in fallback
    _HAS_CUPY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dirs(p): Path(p).mkdir(parents=True, exist_ok=True)

def to_numpy(x):
    """Return a NumPy array for x (CuPy->NumPy if needed)."""
    if _HAS_CUPY and hasattr(x, "get"):
        return x.get()
    return np.asarray(x)

def _grad_sq(E_np, dx):
    """
    |∇E|^2 for 1-D or 2-D NumPy arrays using central differences.
    """
    if E_np.ndim == 1:
        d = (np.roll(E_np, -1) - np.roll(E_np, 1)) / (2.0 * dx)
        return d * d
    elif E_np.ndim == 2:
        d_x = (np.roll(E_np, -1, axis=1) - np.roll(E_np, 1, axis=1)) / (2.0 * dx)
        d_y = (np.roll(E_np, -1, axis=0) - np.roll(E_np, 1, axis=0)) / (2.0 * dx)
        return d_x * d_x + d_y * d_y
    else:
        # Higher-D not used in current plan; degrade gracefully
        raise ValueError(f"Unsupported array ndim={E_np.ndim} (expect 1 or 2)")


# ---------------------------------------------------------------------
# 1) Physics metric: total energy
# ---------------------------------------------------------------------
def energy_total(E, E_prev, dt, dx, c, chi):
    """
    Discrete KG-like energy:
      ∫ 1/2[(E_t)^2 + c^2|∇E|^2 + χ^2 E^2] dV
    Works for 1-D or 2-D fields.
    """
    # Work in CuPy when available for speed, then scalarize
    if _HAS_CUPY and hasattr(E, "__array_priority__"):
        Et = (E - E_prev) / dt
        if E.ndim == 1:
            grad = (cp.roll(E, -1) - cp.roll(E, 1)) / (2.0 * dx)
            dens = 0.5 * (Et * Et + (c * c) * grad * grad + (chi * chi) * E * E)
            return float(cp.sum(dens) * dx)
        elif E.ndim == 2:
            gx = (cp.roll(E, -1, axis=1) - cp.roll(E, 1, axis=1)) / (2.0 * dx)
            gy = (cp.roll(E, -1, axis=0) - cp.roll(E, 1, axis=0)) / (2.0 * dx)
            grad2 = gx * gx + gy * gy
            dens = 0.5 * (Et * Et + (c * c) * grad2 + (chi * chi) * E * E)
            return float(cp.sum(dens) * (dx * dx))
        else:
            raise ValueError(f"Unsupported array ndim={E.ndim}")
    else:
        E_np = to_numpy(E)
        Eprev_np = to_numpy(E_prev)
        Et = (E_np - Eprev_np) / dt
        grad2 = _grad_sq(E_np, dx)
        dens = 0.5 * (Et * Et + (c * c) * grad2 + (chi * chi) * E_np * E_np)
        voxel = (dx if E_np.ndim == 1 else dx * dx)
        return float(np.sum(dens) * voxel)


# ---------------------------------------------------------------------
# 2) Field spectrum (FFT)
# ---------------------------------------------------------------------
def field_spectrum(E, dx, outdir):
    """
    Save FFT spectrum:
      - 1-D: (k, |F(k)|) CSV + line plot
      - 2-D: radial-avg CSV + image of |F(kx,ky)|
    """
    ensure_dirs(outdir)
    e = to_numpy(E)

    if e.ndim == 1:
        F = np.fft.fftshift(np.fft.fft(e))
        k = np.fft.fftshift(np.fft.fftfreq(e.shape[0], d=dx))
        amp = np.abs(F)

        np.savetxt(Path(outdir) / "field_spectrum.csv",
                   np.column_stack([k, amp]),
                   delimiter=",", header="k,|F|", comments="")

        plt.figure(figsize=(5,3))
        plt.plot(k, amp)
        plt.xlabel("k")
        plt.ylabel("|F(k)|")
        plt.title("Field Spectrum (1D)")
        plt.tight_layout()
        plt.savefig(Path(outdir) / "field_spectrum.png", dpi=130)
        plt.close()

    elif e.ndim == 2:
        F = np.fft.fftshift(np.fft.fft2(e))
        amp = np.abs(F)

        # Radial average for CSV
        ny, nx = e.shape
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
        KX, KY = np.meshgrid(kx, ky)
        kr = np.sqrt(KX**2 + KY**2)
        kr_flat = kr.ravel()
        amp_flat = amp.ravel()

        # Bin by radius
        bins = np.linspace(0, kr_flat.max(), 200)
        idx = np.digitize(kr_flat, bins)
        radial = np.array([amp_flat[idx == i].mean() if np.any(idx == i) else 0.0
                           for i in range(1, len(bins))])
        centers = 0.5 * (bins[1:] + bins[:-1])

        np.savetxt(Path(outdir) / "field_spectrum.csv",
                   np.column_stack([centers, radial]),
                   delimiter=",", header="k_radial,|F|_avg", comments="")

        # Image
        plt.figure(figsize=(5,4))
        plt.imshow(amp, cmap="magma", origin="lower",
                   extent=[kx.min(), kx.max(), ky.min(), ky.max()])
        plt.xlabel("kx"); plt.ylabel("ky")
        plt.title("|F(kx,ky)| (2D)")
        plt.colorbar(label="|F|")
        plt.tight_layout()
        plt.savefig(Path(outdir) / "field_spectrum.png", dpi=130)
        plt.close()

    else:
        raise ValueError(f"Unsupported array ndim={e.ndim} (expect 1 or 2)")


# ---------------------------------------------------------------------
# 3) Energy flow (total energy vs. time)
# ---------------------------------------------------------------------
def energy_flow(E_series, dt, dx, c, outdir):
    """
    Compute Σ E(t)^2 over space across the series (proxy for energy trend).
    Saves CSV and a PNG plot.
    """
    ensure_dirs(outdir)
    if not E_series:
        return

    # spacing of samples: caller usually collects every N steps; we
    # keep the given dt and scale time axis by that collection stride if needed.
    # If a stride attribute is provided by the caller, honor it; else assume 10*dt
    # (older runners sample every 10 steps). Caller can pass real spacing via
    # E_series._dt_sample if desired.
    dt_sample = getattr(E_series, "_dt_sample", 10.0 * dt)

    data = []
    for E in E_series:
        e = to_numpy(E)
        data.append(np.sum(e * e))
    data = np.asarray(data)
    t = np.arange(len(data)) * dt_sample

    np.savetxt(Path(outdir) / "energy_flow.csv",
               np.column_stack([t, data]),
               delimiter=",", header="t,energy_like", comments="")

    plt.figure(figsize=(5,3))
    plt.plot(t, data)
    plt.xlabel("time")
    plt.ylabel("Σ E^2")
    plt.title("Energy Flow vs Time")
    plt.tight_layout()
    plt.savefig(Path(outdir) / "energy_flow.png", dpi=130)
    plt.close()


# ---------------------------------------------------------------------
# 4) Phase autocorrelation (temporal coherence)
# ---------------------------------------------------------------------
def phase_corr(E_series, outdir):
    """
    Phase autocorrelation of the mean field over time.
    Saves CSV + PNG.
    """
    ensure_dirs(outdir)
    if not E_series:
        return

    mean_phase = []
    for E in E_series:
        e = to_numpy(E)
        # Use complex analytic signal surrogate if real; otherwise angle directly
        # For our lattice (real), phase from Hilbert would be ideal but heavyweight;
        # using arctangent of normalized pair is sufficient as proxy.
        m = np.mean(e)
        phase = np.angle(m + 0j)
        mean_phase.append(phase)

    phases = np.unwrap(np.asarray(mean_phase))
    pz = phases - phases.mean()
    ac = np.correlate(pz, pz, mode="full")
    lags = np.arange(-len(pz) + 1, len(pz))

    np.savetxt(Path(outdir) / "phase_corr.csv",
               np.column_stack([lags, ac]),
               delimiter=",", header="lag,phase_autocorr", comments="")

    plt.figure(figsize=(5,3))
    plt.plot(lags, ac)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.title("Phase Autocorrelation")
    plt.tight_layout()
    plt.savefig(Path(outdir) / "phase_corr.png", dpi=130)
    plt.close()
