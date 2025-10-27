#!/usr/bin/env python3
"""
lfm_visualizer.py — Unified visual generator for all LFM tiers (v1.9 Overwrite-Safe)
Now includes:
- Guaranteed PNG output for 1D data (Tier-1 fix)
- Overwrite-safe saving (removes old PNG/GIF/MP4 before write)
- Restored GIF/MP4 animation generation
- Adaptive frame sampling for quick/full modes
- Tier-aware annotation overlays
- Automatic fallback to static PNG if animation fails
"""

import os
import numpy as np
try:
    import cupy as cp  # type: ignore
except Exception:
    import numpy as cp  # type: ignore
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def to_numpy(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)

def timestamp():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def _to_img(field, tile_y=12):
    """Convert 1D arrays to pseudo-2D for visualization."""
    f = np.array(field)
    if f.ndim == 1:
        return np.tile(f, (tile_y, 1))
    return f

def _annotate(ax, text):
    """Adds semi-transparent overlay annotation."""
    ax.text(0.02, 0.95, text, color="white", fontsize=8,
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.4))

def _safe_savefig(path: Path, fig=None, dpi=150):
    """Remove existing file before saving to ensure overwrite."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    (fig or plt).savefig(path, dpi=dpi)
# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def visualize_concept(E_series, chi_series=None, tier=None, test_id="TEST",
                      outdir="plots", quick=False, animate=False, make_animation_mp4=False):
    """
    Generate static + optional animated visualizations.
    - E_series: list of field arrays (CuPy or NumPy)
    - chi_series: optional curvature field
    - tier: int 1–6
    - animate: generate GIF/MP4 if True
    """
    ensure_dirs(outdir)
    tier = int(tier or 0)
    label = f"concept_{test_id}"
    field = to_numpy(E_series[-1])
    chi = to_numpy(chi_series[-1]) if chi_series is not None else None

    # --- Guarantee at least one PNG for 1D fields (Tier-1, etc.) ---
    if field.ndim == 1:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(field.real, lw=1.2, color="goldenrod")
        ax.set_title(f"{label} — 1D Field Snapshot (Tier {tier})")
        ax.set_xlabel("Position index")
        ax.set_ylabel("Field amplitude")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _safe_savefig(Path(outdir)/f"{label}.png", fig, dpi=(80 if quick else 150))
        plt.close(fig)

    # Tier-based visuals
    tier_funcs = {
        1: _visualize_tier1,
        2: _visualize_tier2,
        3: _visualize_tier3,
        4: _visualize_tier4,
        5: _visualize_tier5,
        6: _visualize_tier6
    }
    func = tier_funcs.get(tier, _visualize_generic)
    func(field, chi if tier in (2, 4, 6) else None, outdir, label, quick)

    # Optional animation
    if animate:
        try:
            _make_animation(E_series, outdir, label, quick, make_animation_mp4)
        except Exception as e:
            print(f"[WARN] Animation failed for {test_id}: {e}")

# ---------------------------------------------------------------------
# Tier Visuals (Overwrite-Safe)
# ---------------------------------------------------------------------
def _visualize_tier1(field, chi, outdir, label, quick):
    f_img = _to_img(np.array(field))
    fig, ax = plt.subplots(figsize=(5,3))
    ax.imshow(f_img.real, cmap="plasma", origin="lower", aspect="auto")
    _annotate(ax, "Tier-1: Lorentz isotropy — symmetric propagation")
    ax.set_title(f"{label} — Relativistic Pulse Symmetry")
    plt.colorbar(ax.images[0], ax=ax, label="Field amplitude")
    plt.tight_layout()
    _safe_savefig(Path(outdir)/f"{label}.png", fig, dpi=(80 if quick else 150))
    plt.close(fig)

def _visualize_tier2(field, chi, outdir, label, quick):
    f_img = _to_img(np.array(field))
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(f_img.real, cmap="inferno", origin="lower", alpha=0.9)
    if chi is not None:
        ax.contour(_to_img(np.array(chi)), levels=10, colors="cyan", linewidths=0.5)
    _annotate(ax, "Tier-2: χ-gradient curvature deflection")
    ax.set_title(f"{label} — Weak-Field Deflection")
    plt.tight_layout()
    _safe_savefig(Path(outdir)/f"{label}.png", fig, dpi=(80 if quick else 150))
    plt.close(fig)

def _visualize_tier3(field, chi, outdir, label, quick):
    f_img = _to_img(np.abs(np.array(field)))
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(f_img, cmap="magma", origin="lower")
    _annotate(ax, "Tier-3: Energy transport & damping")
    ax.set_title(f"{label} — Energy Flow")
    plt.colorbar(im, ax=ax, label="|E| magnitude")
    plt.tight_layout()
    _safe_savefig(Path(outdir)/f"{label}.png", fig, dpi=(80 if quick else 150))
    plt.close(fig)

def _visualize_tier4(field, chi, outdir, label, quick):
    f_img = _to_img(np.array(field))
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(f_img.real, cmap="twilight_shifted", origin="lower")
    if chi is not None:
        ax.contour(_to_img(np.array(chi)), levels=8, colors="white", linewidths=0.4)
    _annotate(ax, "Tier-4: Rotation & horizon analogues")
    ax.set_title(f"{label} — Vorticity Map")
    plt.tight_layout()
    _safe_savefig(Path(outdir)/f"{label}.png", fig, dpi=(80 if quick else 150))
    plt.close(fig)

def _visualize_tier5(field, chi, outdir, label, quick):
    f_img = _to_img(np.angle(np.array(field)))
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(f_img, cmap="hsv", origin="lower")
    _annotate(ax, "Tier-5: Quantized interference modes")
    ax.set_title(f"{label} — Phase–Amplitude Map")
    plt.colorbar(im, ax=ax, label="Phase [radians]")
    plt.tight_layout()
    _safe_savefig(Path(outdir)/f"{label}.png", fig, dpi=(80 if quick else 150))
    plt.close(fig)

def _visualize_tier6(field, chi, outdir, label, quick):
    f_img = _to_img(np.array(field))
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(f_img.real, cmap="cividis", origin="lower", alpha=0.9)
    if chi is not None:
        ax.contour(_to_img(np.array(chi)), levels=6, colors="magenta", linewidths=0.5)
    _annotate(ax, "Tier-6: Cosmological expansion")
    ax.set_title(f"{label} — Expansion & χ Feedback")
    plt.tight_layout()
    _safe_savefig(Path(outdir)/f"{label}.png", fig, dpi=(80 if quick else 150))
    plt.close(fig)

def _visualize_generic(field, chi, outdir, label, quick=False):
    f_img = _to_img(np.abs(np.array(field)))
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(f_img, cmap="viridis", origin="lower")
    _annotate(ax, "Generic lattice output")
    ax.set_title(f"{label} — Generic Visualization")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    _safe_savefig(Path(outdir)/f"{label}.png", fig, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------
# Animation Generator (Overwrite-Safe)
# ---------------------------------------------------------------------
def _make_animation(E_series, outdir, label, quick, make_animation_mp4=False):
    ensure_dirs(outdir)
    frames = [_to_img(to_numpy(E)).real for E in E_series[::max(1, len(E_series)//(20 if quick else 100))]]
    fig, ax = plt.subplots(figsize=(5,4))
    img = ax.imshow(frames[0], cmap="plasma", origin="lower", animated=True)
    _annotate(ax, "Lattice field evolution")
    plt.tight_layout()

    def update(i):
        img.set_array(frames[i])
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    gif_path = Path(outdir)/f"{label}.gif"
    try:
        os.remove(gif_path)
    except FileNotFoundError:
        pass
    ani.save(gif_path, writer="pillow", fps=10 if quick else 20)

    if make_animation_mp4:
        try:
            mp4_path = Path(outdir)/f"{label}.mp4"
            try:
                os.remove(mp4_path)
            except FileNotFoundError:
                pass
            ani.save(mp4_path, writer="ffmpeg", fps=20)
        except Exception as e:
            print(f"[WARN] MP4 export failed: {e}")

    plt.close(fig)
