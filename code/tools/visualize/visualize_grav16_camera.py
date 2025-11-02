#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com

"""
Camera-style visualization for GRAV-16 double-slit experiment.

Renders a 2D imshow-style representation showing:
- Screen plane behind barrier with intensity texture
- Barrier and slit markers overlaid
- Time evolution suitable for MP4 encoding

Usage:
  python visualize_grav16_camera.py \
    --input results/Gravity/GRAV-16/diagnostics/field_snapshots_3d_GRAV-16.h5 \
    --output results/Gravity/GRAV-16/camera_frames \
    --z_frac 0.70 \
    --intensity power

Then create MP4 (Windows example):
  C:\\ffmpeg\\bin\\ffmpeg.exe -y -framerate 30 -i results\\Gravity\\GRAV-16\\camera_frames\\cam_%04d.png \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
    results\\Gravity\\GRAV-16\\doubleslit_camera.mp4
"""

import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path


def load_metadata(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        shape = tuple(hf['shape'][()])
        dx = float(hf['dx'][()])
        dt = float(hf['dt'][()])
        steps_per_snap = int(hf['steps_per_snap'][()])
        barrier_z = int(hf['barrier_z'][()])
        slit_positions = hf['slit_positions'][()]
        snap_grp = hf['snapshots']
        keys = sorted(snap_grp.keys())
    return shape, dx, dt, steps_per_snap, barrier_z, slit_positions, keys


def compute_global_scale(h5_path, keys, z_idx, mode='power', q=0.995):
    """Compute a robust global vmax for coloring from all snapshots at z=z_idx.
    Uses a high quantile to avoid single-frame spikes.
    mode: 'abs' for |E|, 'power' for E^2
    """
    vals = []
    with h5py.File(h5_path, 'r') as hf:
        for k in keys:
            fld = hf['snapshots'][k][()]
            slice_yz = fld[:, :, z_idx]
            if mode == 'power':
                v = np.sqrt(np.mean(slice_yz**2))  # RMS as a proxy
                vals.append(v)
            else:
                v = np.mean(np.abs(slice_yz))
                vals.append(v)
    base = np.quantile(vals, q)
    # Scale up to cover peaks
    return float(max(base * 6.0, 1e-6))


def make_camera_frame(ax, 
                      I_screen, time, 
                      shape, dx, barrier_z, slit_positions, z_screen,
                      vmax,
                      title_extra="",
                      cmap=cm.magma):
    """Render one 2D camera frame showing YX cross-section at screen z."""
    ax.cla()

    Nx, Ny, Nz = shape
    # I_screen shape is (Nx, Ny) representing the screen plane at z=z_screen
    
    # Normalize intensity for colors
    I = np.clip(I_screen, 0.0, None)
    I_norm = np.clip(I / vmax, 0.0, 1.0)

    # Display screen plane as imshow (YX orientation)
    extent = [0, Ny*dx, 0, Nx*dx]  # [left, right, bottom, top]
    im = ax.imshow(I_norm, origin='lower', extent=extent, cmap=cmap, aspect='auto', vmin=0, vmax=1.0)

    # Mark barrier and slits
    barrier_pos_z = barrier_z * dx
    # Barrier is at z=barrier_z, we're viewing at z=z_screen*dx, so show reference line if helpful
    # For simplicity: draw horizontal lines at slit y-positions
    for slit_y in slit_positions:
        y_val = float(slit_y) * dx
        ax.axhline(y_val, color='yellow', linewidth=1.5, linestyle='--', alpha=0.6)

    # Annotations
    ax.set_xlabel('Slit separation (y)', fontsize=10)
    ax.set_ylabel('Height (x)', fontsize=10)
    title_str = f'Screen at z={z_screen*dx:.1f} | Barrier at z={barrier_pos_z:.1f} | t={time:.2f}s'
    if title_extra:
        title_str += f' | {title_extra}'
    ax.set_title(title_str, fontsize=11)
    
    return im


def main():
    ap = argparse.ArgumentParser(description='Camera-view renderer for GRAV-16 double-slit')
    ap.add_argument('--input', type=str, default='results/Gravity/GRAV-16/diagnostics/field_snapshots_3d_GRAV-16.h5')
    ap.add_argument('--output', type=str, default='results/Gravity/GRAV-16/camera_frames')
    ap.add_argument('--z_frac', type=float, default=0.70, help='Screen z as fraction of domain (0-1)')
    ap.add_argument('--intensity', type=str, default='power', choices=['abs','power'], help='Intensity metric: abs(E) or power=E^2')
    ap.add_argument('--ema', type=float, default=0.2, help='Exponential smoothing for intensity (0-1)')
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from {in_path} ...")
    shape, dx, dt, steps_per_snap, barrier_z, slit_positions, keys = load_metadata(in_path)
    Nx, Ny, Nz = shape
    z_idx = int(max(0, min(Nz-1, round(args.z_frac * Nz))))
    z_screen = float(z_idx)
    print(f"Grid {Nx}x{Ny}x{Nz}, dx={dx}, dt={dt}, barrier_z={barrier_z}, screen z_idx={z_idx}")

    print("Computing global color scale (robust)...")
    vmax = compute_global_scale(in_path, keys, z_idx, mode=args.intensity, q=0.995)
    print(f"Using vmax={vmax:.3e} for coloring")

    # Prepare figure/axes
    fig, ax = plt.subplots(figsize=(10, 7))

    # Iterate snapshots and render frames
    ema = None
    with h5py.File(in_path, 'r') as hf:
        grp = hf['snapshots']
        for i, k in enumerate(keys):
            ds = grp[k]
            fld = ds[()]
            time = float(ds.attrs['time'])
            slice_yz = fld[:, :, z_idx]
            if args.intensity == 'power':
                I = slice_yz**2
            else:
                I = np.abs(slice_yz)
            if ema is None:
                ema = I.copy()
            else:
                ema = (1.0 - args.ema) * ema + args.ema * I

            make_camera_frame(ax, ema, time, shape, dx, barrier_z, slit_positions, z_screen, vmax)

            frame_path = out_dir / f"cam_{i:04d}.png"
            fig.tight_layout()
            fig.savefig(frame_path, dpi=args.dpi, facecolor='white')
            if (i+1) % 20 == 0:
                print(f"  Saved {i+1}/{len(keys)} frames...")

    plt.close(fig)
    print(f"\n✅ Camera frames saved to: {out_dir}")
    print("Next, create MP4 (Windows example):")
    print(f"  C:\\ffmpeg\\bin\\ffmpeg.exe -y -framerate 30 -i {out_dir.as_posix()}\\cam_%04d.png -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -c:v libx264 -pix_fmt yuv420p -movflags +faststart {out_dir.parent.as_posix()}\\doubleslit_camera.mp4")


if __name__ == '__main__':
    main()
