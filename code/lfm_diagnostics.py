#!/usr/bin/env python3
"""
lfm_diagnostics.py — unified diagnostic utilities for all LFM tiers (v1.10.0-compensated)

Changes from v1.9.1:
  • energy_total() now uses Neumaier compensated summation to remove floating-sum drift.
  • No change to physics or solver interfaces.
  • All other diagnostic functions unchanged.
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
    """
    ∫ ½[(E_t)^2 + c^2|∇E|^2 + χ^2E^2] dV
    Uses compensated (Neumaier) summation to suppress rounding error.
    """
    E, E_prev = to_numpy(E), to_numpy(E_prev)
    Et = (E - E_prev) / dt

    if E.ndim == 1:
        gx = (np.roll(E, -1) - np.roll(E, 1)) / (2.0 * dx)
        dens = 0.5 * (Et**2 + (c**2)*gx**2 + (chi**2)*E**2)
        weight = dx
    elif E.ndim == 2:
        gx = (np.roll(E, -1, 1) - np.roll(E, 1, 1)) / (2.0 * dx)
        gy = (np.roll(E, -1, 0) - np.roll(E, 1, 0)) / (2.0 * dx)
        grad2 = gx**2 + gy**2
        dens = 0.5 * (Et**2 + (c**2)*grad2 + (chi**2)*E**2)
        weight = dx * dx
    else:
        raise ValueError(f"Unsupported ndim={E.ndim}")

    # Neumaier compensated sum
    s = 0.0
    c_err = 0.0
    flat = dens.ravel()
    for x in flat:
        t = s + x
        if abs(s) >= abs(x):
            c_err += (s - t) + x
        else:
            c_err += (x - t) + s
        s = t
    return float((s + c_err) * weight)

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
        k = np.fft.fftshift(np.fft.fftfreq(len(e0), d=dx))
        amp = np.abs(F) / (np.sum(w) + 1e-30)
        np.savetxt(Path(outdir)/"field_spectrum.csv",
                   np.column_stack([k, amp]), delimiter=",",
                   header="k,|F|", comments="", encoding="utf-8")
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
                   delimiter=",", header="kx,ky,|F|",
                   comments="", encoding="utf-8")
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
    baseline = float(np.median(vals[1:4])) if len(vals) > 3 else float(vals[0])
    rel = (vals - baseline) / (abs(baseline) + 1e-30)
    np.savetxt(Path(outdir)/"energy_flow.csv",
               np.column_stack([t, vals, rel]), delimiter=",",
               header="t,E_sum_sq,rel_drift", comments="", encoding="utf-8")
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
               np.column_stack([lags, ac_full]), delimiter=",",
               header="lag,normalized_autocorr", comments="", encoding="utf-8")
    plt.figure(figsize=(5,3))
    plt.plot(lags, ac_full)
    plt.xlabel("Lag"); plt.ylabel("Normalized C(lag)")
    plt.title("Phase Autocorrelation")
    plt.tight_layout(); plt.savefig(Path(outdir)/"phase_corr.png", dpi=130); plt.close()
