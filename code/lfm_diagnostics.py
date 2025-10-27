#!/usr/bin/env python3
"""
lfm_diagnostics.py — unified diagnostic utilities for all LFM tiers (v1.9.1)
Fixes:
  • Energy drift: baseline-corrected, warm-up excluded, UTF-8 CSV.
  • Spectrum: DC removal + Hann window for 1D/2D.
  • Phase corr: normalized autocorr in [-1,1].
Outputs per test (.../<TestID>/diagnostics/):
  - field_spectrum.csv, field_spectrum.png
  - energy_flow.csv,   energy_flow.png
  - phase_corr.csv,    phase_corr.png
Public API (unchanged):
  energy_total(E, E_prev, dt, dx, c, chi)
  field_spectrum(E, dx, outdir)
  energy_flow(E_series, dt, dx, c, outdir)
  phase_corr(E_series, outdir)
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional CuPy
try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = np
    _HAS_CUPY = False

def ensure_dirs(p): Path(p).mkdir(parents=True, exist_ok=True)
def to_numpy(x):
    if _HAS_CUPY and hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


# ------------------------- 1) Discrete energy -------------------------
def energy_total(E, E_prev, dt, dx, c, chi):
    """∫ ½[(E_t)^2 + c^2|∇E|^2 + χ^2E^2] dV for 1D/2D arrays."""
    E, E_prev = to_numpy(E), to_numpy(E_prev)
    Et = (E - E_prev) / dt
    if E.ndim == 1:
        gx = (np.roll(E, -1) - np.roll(E, 1)) / (2.0 * dx)
        dens = 0.5 * (Et**2 + (c**2) * gx**2 + (chi**2) * E**2)
        return float(np.sum(dens) * dx)
    elif E.ndim == 2:
        gx = (np.roll(E, -1, 1) - np.roll(E, 1, 1)) / (2.0 * dx)
        gy = (np.roll(E, -1, 0) - np.roll(E, 1, 0)) / (2.0 * dx)
        grad2 = gx**2 + gy**2
        dens = 0.5 * (Et**2 + (c**2) * grad2 + (chi**2) * E**2)
        return float(np.sum(dens) * (dx * dx))
    raise ValueError(f"Unsupported ndim={E.ndim}")


# ------------------------- 2) Field spectrum --------------------------
def field_spectrum(E, dx, outdir):
    """1D: save (k, |F|) with mean removal + Hann. 2D: |F(kx,ky)| image."""
    ensure_dirs(outdir)
    e = to_numpy(E)

    if e.ndim == 1:
        e0 = e - np.mean(e)
        w = np.hanning(len(e0))
        Ew = e0 * w
        F = np.fft.fftshift(np.fft.fft(Ew))
        k = np.fft.fftshift(np.fft.fftfreq(len(e0), d=dx))   # cycles per unit
        amp = np.abs(F) / (np.sum(w) + 1e-30)

        np.savetxt(Path(outdir)/"field_spectrum.csv",
                   np.column_stack([k, amp]),
                   delimiter=",",
                   header="k,|F|",
                   comments="",
                   encoding="utf-8")

        plt.figure(figsize=(5,3))
        plt.plot(k, amp)
        plt.xlabel("k (cycles/unit)"); plt.ylabel("|F(k)|")
        plt.title("Field Spectrum (1D, DC-removed, Hann)")
        plt.tight_layout(); plt.savefig(Path(outdir)/"field_spectrum.png", dpi=130); plt.close()

    elif e.ndim == 2:
        e0 = e - np.mean(e)
        w0, w1 = np.hanning(e.shape[0]), np.hanning(e.shape[1])
        w2 = np.outer(w0, w1)
        Ew = e0 * w2
        F = np.fft.fftshift(np.fft.fft2(Ew))
        amp = np.abs(F) / (np.sum(w2) + 1e-30)
        ny, nx = e.shape
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))

        np.savetxt(Path(outdir)/"field_spectrum.csv",
                   np.column_stack([kx, ky, amp.reshape(-1)]),
                   delimiter=",",
                   header="kx,ky,|F|",
                   comments="",
                   encoding="utf-8")

        plt.figure(figsize=(5,4))
        plt.imshow(amp, cmap="magma", origin="lower",
                   extent=[kx.min(), kx.max(), ky.min(), ky.max()])
        plt.xlabel("kx"); plt.ylabel("ky")
        plt.title("|F(kx,ky)| (2D, DC-removed, Hann)")
        plt.colorbar(label="|F|")
        plt.tight_layout(); plt.savefig(Path(outdir)/"field_spectrum.png", dpi=130); plt.close()
    else:
        raise ValueError(f"Unsupported ndim={e.ndim}")


# ------------------------- 3) Energy flow -----------------------------
def energy_flow(E_series, dt, dx, c, outdir):
    """Write t, E_sum_sq, rel_drift (warm-up excluded from rel_drift)."""
    ensure_dirs(outdir)
    if not E_series:
        return
    dt_sample = getattr(E_series, "_dt_sample", 10.0 * dt)
    vals = np.array([float(np.sum(to_numpy(E)**2)) for E in E_series], dtype=float)
    t = np.arange(len(vals)) * dt_sample

    # Robust baseline from first three post-warmup samples
    if len(vals) > 3:
        baseline = float(np.median(vals[1:4]))
    else:
        baseline = float(vals[0])
    rel = (vals - baseline) / (abs(baseline) + 1e-30)

    # Persist both raw and normalized; downstream may ignore row 0 if desired
    np.savetxt(Path(outdir)/"energy_flow.csv",
               np.column_stack([t, vals, rel]),
               delimiter=",",
               header="t,E_sum_sq,rel_drift",
               comments="",
               encoding="utf-8")

    plt.figure(figsize=(5,3))
    plt.plot(t, rel)
    plt.xlabel("time"); plt.ylabel("ΔE/E0")
    plt.title("Energy Drift (baseline-corrected)")
    plt.tight_layout(); plt.savefig(Path(outdir)/"energy_flow.png", dpi=130); plt.close()


# ------------------------- 4) Phase correlation -----------------------
def phase_corr(E_series, outdir):
    """Normalized phase autocorrelation of mean field ([-1,1])."""
    ensure_dirs(outdir)
    if not E_series:
        return
    mean_phase = []
    for E in E_series:
        m = complex(np.mean(to_numpy(E)), 0.0)
        mean_phase.append(np.angle(m))
    phases = np.unwrap(np.asarray(mean_phase))
    pz = phases - np.mean(phases)

    ac_full = np.correlate(pz, pz, mode="full")
    ac_full /= (np.max(np.abs(ac_full)) + 1e-30)
    lags = np.arange(-len(pz)+1, len(pz))

    np.savetxt(Path(outdir)/"phase_corr.csv",
               np.column_stack([lags, ac_full]),
               delimiter=",",
               header="lag,normalized_autocorr",
               comments="",
               encoding="utf-8")

    plt.figure(figsize=(5,3))
    plt.plot(lags, ac_full)
    plt.xlabel("Lag"); plt.ylabel("Normalized C(lag)")
    plt.title("Phase Autocorrelation")
    plt.tight_layout(); plt.savefig(Path(outdir)/"phase_corr.png", dpi=130); plt.close()
