#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Generate visualization of 3D double-slit experiment from GRAV-16 test.

Renders cross-sections and/or 3D volumetric views showing interference pattern.

Usage:
    python visualize_grav16_doubleslit.py [--input FILE] [--output DIR] [--mode MODE]
    
Modes:
    xz_slice : XZ cross-section at y=center (side view showing both slits)
    yz_slice : YZ cross-section behind barrier (interference pattern)
    frames   : Individual PNG frames for all slices
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

def load_snapshots(h5_path):
    """Load 3D field snapshots from HDF5 file."""
    with h5py.File(h5_path, 'r') as hf:
        shape = tuple(hf['shape'][()])
        dx = float(hf['dx'][()])
        dt = float(hf['dt'][()])
        steps_per_snap = int(hf['steps_per_snap'][()])
        barrier_z = int(hf['barrier_z'][()])
        slit_positions = hf['slit_positions'][()]
        
        snap_grp = hf['snapshots']
        snapshots = []
        for key in sorted(snap_grp.keys()):
            field = snap_grp[key][()]
            time = snap_grp[key].attrs['time']
            snapshots.append((time, field))
    
    return shape, dx, dt, steps_per_snap, barrier_z, slit_positions, snapshots

def create_xz_frame(ax, field, shape, dx, time, barrier_z, slit_y, vmin, vmax, adaptive=True):
    """Create XZ slice at y=slit_y (side view through one slit)."""
    ax.clear()
    
    Nx, Ny, Nz = shape
    y_idx = min(Ny-1, max(0, int(slit_y)))
    
    # Extract XZ slice at y=y_idx
    slice_xz = field[:, y_idx, :]
    
    # Use adaptive colormap range
    if adaptive:
        slice_max = np.abs(slice_xz).max()
        vmax_use = max(slice_max, 1e-6)
        vmin_use = -vmax_use
    else:
        vmin_use, vmax_use = vmin, vmax
    
    # Plot
    extent = [0, Nz*dx, 0, Nx*dx]
    im = ax.imshow(slice_xz, origin='lower', extent=extent, 
                   cmap='RdBu_r', vmin=vmin_use, vmax=vmax_use, aspect='auto')
    
    # Mark barrier location
    ax.axvline(barrier_z*dx, color='black', linewidth=2, linestyle='--', alpha=0.7, label='Barrier')
    
    ax.set_xlabel('Z (propagation direction)')
    ax.set_ylabel('X')
    ax.set_title(f'XZ Slice (y={y_idx}) — Time: {time:.2f}s')
    ax.legend(loc='upper right')
    
    return im

def create_yz_frame(ax, field, shape, dx, time, barrier_z, z_frac, slit_positions, vmin, vmax, adaptive=True):
    """Create YZ slice at given z_frac (interference pattern behind barrier)."""
    ax.clear()
    
    Nx, Ny, Nz = shape
    z_idx = min(Nz-1, max(0, int(z_frac * Nz)))
    
    # Extract YZ slice at z=z_idx
    slice_yz = field[:, :, z_idx]
    
    # Use adaptive colormap range
    if adaptive:
        slice_max = np.abs(slice_yz).max()
        vmax_use = max(slice_max, 1e-6)
        vmin_use = -vmax_use
    else:
        vmin_use, vmax_use = vmin, vmax
    
    # Plot
    extent = [0, Ny*dx, 0, Nx*dx]
    im = ax.imshow(slice_yz, origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=vmin_use, vmax=vmax_use, aspect='auto')
    
    # Mark slit positions
    for slit_y in slit_positions:
        ax.axhline(slit_y*dx, color='yellow', linewidth=1, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Y (slit separation direction)')
    ax.set_ylabel('X')
    position = 'barrier' if z_idx == barrier_z else f'behind barrier (z={z_idx})'
    ax.set_title(f'YZ Slice at {position} — Time: {time:.2f}s')
    
    return im

def main():
    parser = argparse.ArgumentParser(description="Visualize 3D double-slit experiment from GRAV-16")
    parser.add_argument("--input", type=str, 
                       default="results/Gravity/GRAV-16/diagnostics/field_snapshots_3d_GRAV-16.h5",
                       help="Path to input HDF5 file")
    parser.add_argument("--output", type=str, default="results/Gravity/GRAV-16/frames_doubleslit",
                       help="Output directory for frames")
    parser.add_argument("--mode", type=str, default="yz_slice", 
                       choices=["xz_slice", "yz_slice", "both"],
                       help="Visualization mode")
    parser.add_argument("--z_frac", type=float, default=0.70,
                       help="Z fraction for YZ slice (0.7 = far behind barrier)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for output frames")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    print(f"Loading snapshots from {input_path}...")
    shape, dx, dt, steps_per_snap, barrier_z, slit_positions, snapshots = load_snapshots(input_path)
    Nx, Ny, Nz = shape
    print(f"Loaded {len(snapshots)} snapshots")
    print(f"Grid: {Nx}×{Ny}×{Nz}, dx={dx}, dt={dt}")
    print(f"Barrier at z={barrier_z}, slits at y={slit_positions}")
    
    # Compute global field range
    print("Computing field range...")
    all_abs_max = max(np.abs(field).max() for _, field in snapshots)
    vmin_global = -all_abs_max
    vmax_global = all_abs_max
    print(f"Field range: [{vmin_global:.3e}, {vmax_global:.3e}]")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate frames
    if args.mode in ("xz_slice", "both"):
        print(f"\nGenerating XZ slices (side view through slit)...")
        fig_xz, ax_xz = plt.subplots(figsize=(12, 6))
        slit_y = slit_positions[0]  # View through first slit
        
        for i, (time, field) in enumerate(snapshots):
            create_xz_frame(ax_xz, field, shape, dx, time, barrier_z, slit_y, 
                          vmin_global, vmax_global, adaptive=True)
            frame_path = output_dir / f"xz_frame_{i:04d}.png"
            fig_xz.savefig(frame_path, dpi=args.dpi, bbox_inches='tight')
            
            if (i+1) % 20 == 0:
                print(f"  Saved {i+1}/{len(snapshots)} XZ frames...")
        
        plt.close(fig_xz)
        print(f"✅ XZ frames saved to {output_dir}")
    
    if args.mode in ("yz_slice", "both"):
        print(f"\nGenerating YZ slices (interference pattern at z={args.z_frac:.2f})...")
        fig_yz, ax_yz = plt.subplots(figsize=(10, 8))
        
        for i, (time, field) in enumerate(snapshots):
            create_yz_frame(ax_yz, field, shape, dx, time, barrier_z, args.z_frac,
                          slit_positions, vmin_global, vmax_global, adaptive=True)
            frame_path = output_dir / f"yz_frame_{i:04d}.png"
            fig_yz.savefig(frame_path, dpi=args.dpi, bbox_inches='tight')
            
            if (i+1) % 20 == 0:
                print(f"  Saved {i+1}/{len(snapshots)} YZ frames...")
        
        plt.close(fig_yz)
        print(f"✅ YZ frames saved to {output_dir}")
    
    print(f"\n✅ All frames saved!")
    print(f"\nViewing options:")
    print(f"  1. Browse PNG frames in: {output_dir}")
    print(f"  2. Create GIF animation: python visualize_grav16_doubleslit.py --create-gif")
    print(f"  3. Create MP4 with FFmpeg (if installed):")
    if args.mode in ("xz_slice", "both"):
        print(f"     ffmpeg -framerate 30 -i {output_dir}/xz_frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_dir.parent}/doubleslit_xz.mp4")
    if args.mode in ("yz_slice", "both"):
        print(f"     ffmpeg -framerate 30 -i {output_dir}/yz_frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_dir.parent}/doubleslit_yz.mp4")

if __name__ == "__main__":
    main()
