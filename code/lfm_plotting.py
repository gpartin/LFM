#!/usr/bin/env python3
"""
lfm_plotting.py — Standard plotting utilities for all LFM tiers.
Generates time-series plots (energy, entropy), 2D field snapshots, and
optional overlays for diagnostics. Compatible with quick/full modes.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def timestamp():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------------------------------------
# Basic plots
# ---------------------------------------------------------------------
def plot_energy(times, energy, outdir, title=None, quick=False):
    ensure_dirs(outdir)
    plt.figure(figsize=(6, 4))
    plt.plot(times, energy, lw=1.5)
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title(title or "Energy vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(outdir) / "energy_vs_time.png", dpi=100 if quick else 150)
    plt.close()

def plot_entropy(times, entropy, outdir, title=None, quick=False):
    ensure_dirs(outdir)
    plt.figure(figsize=(6, 4))
    plt.plot(times, entropy, lw=1.5, color="orange")
    plt.xlabel("Time")
    plt.ylabel("Shannon Entropy")
    plt.title(title or "Entropy vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(outdir) / "entropy_vs_time.png", dpi=100 if quick else 150)
    plt.close()

# ---------------------------------------------------------------------
# Field snapshots
# ---------------------------------------------------------------------
def save_field_snapshot(f, outdir, label="field", quick=False):
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    Path(outdir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 3))
    f_np = np.array(f.get() if hasattr(f, "get") else f)

    if f_np.ndim == 1:
        # Convert 1D field to a pseudo-2D image for imshow
        f_np = np.tile(f_np, (10, 1))  # repeat vertically 10× for visibility
        plt.imshow(f_np.real, cmap="inferno", origin="lower", aspect="auto")
        plt.title(f"{label} (1D Field, stretched)")
    elif f_np.ndim == 2:
        plt.imshow(f_np.real, cmap="inferno", origin="lower", aspect="auto")
        plt.title(f"{label} (2D Field)")
    else:
        plt.text(0.1, 0.5, f"Unsupported shape {f_np.shape}", fontsize=10)
        plt.axis("off")

    plt.colorbar(label="Field Amplitude")
    plt.tight_layout()
    plt.savefig(Path(outdir) / f"{label}.png", dpi=(80 if quick else 150))
    plt.close()
# ---------------------------------------------------------------------
# Diagnostic overlays
# ---------------------------------------------------------------------
def overlay_spectrum(freqs, amplitudes, outdir, label="spectrum", quick=False):
    """Plot FFT magnitude spectrum."""
    ensure_dirs(outdir)
    plt.figure(figsize=(6, 4))
    plt.plot(freqs, amplitudes, lw=1.2)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Field Spectrum")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(outdir) / f"{label}.png", dpi=100 if quick else 150)
    plt.close()

def overlay_energy_flow(times, flow, outdir, label="energy_flow", quick=False):
    """Plot energy transport trace (if available)."""
    ensure_dirs(outdir)
    plt.figure(figsize=(6, 4))
    plt.plot(times, flow, lw=1.5, color="green")
    plt.xlabel("Time")
    plt.ylabel("Energy Flux")
    plt.title("Energy Flow Trace")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(outdir) / f"{label}.png", dpi=100 if quick else 150)
    plt.close()
