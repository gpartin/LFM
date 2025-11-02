#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com

"""
Generate MP4 visualization of 3D energy dispersion from GRAV-15 test.

Renders volumetric field snapshots as a rotating transparent cube with
energy density shown as colored isosurfaces or volume rendering.

Usage:
    python visualize_grav15_3d.py [--input field_snapshots_3d_GRAV-15.h5] [--output energy_dispersion_3d.mp4]
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def load_snapshots(h5_path):
    """Load 3D field snapshots from HDF5 file."""
    with h5py.File(h5_path, 'r') as hf:
        N = int(hf['N'][()])
        dx = float(hf['dx'][()])
        dt = float(hf['dt'][()])
        steps_per_snap = int(hf['steps_per_snap'][()])
        
        snap_grp = hf['snapshots']
        snapshots = []
        for key in sorted(snap_grp.keys()):
            field = snap_grp[key][()]
            time = snap_grp[key].attrs['time']
            snapshots.append((time, field))
    
    return N, dx, dt, steps_per_snap, snapshots

def compute_energy_density(field):
    """Compute energy density (field squared)."""
    return field ** 2

def create_frame(ax, field, N, dx, time, angle, vmin, vmax, adaptive_threshold=True):
    """Create a single frame showing 3D energy isosurface."""
    ax.clear()
    
    # Compute energy density
    energy = compute_energy_density(field)
    
    # Create mesh grid
    x = np.arange(N) * dx
    y = np.arange(N) * dx
    z = np.arange(N) * dx
    
    # Threshold for isosurface - use adaptive threshold based on current frame
    if adaptive_threshold:
        frame_max = energy.max()
        threshold = max(0.02 * frame_max, 1e-6)  # 2% of current max, or floor at 1e-6
    else:
        threshold = 0.05 * vmax
    
    # Find voxels above threshold
    mask = energy > threshold
    if mask.sum() == 0:
        # No significant energy, skip
        ax.set_xlim([0, N*dx])
        ax.set_ylim([0, N*dx])
        ax.set_zlim([0, N*dx])
        ax.set_title(f"Time: {time:.2f}s")
        return
    
    # Use voxels or scatter plot for energy
    # For efficiency, downsample to ~30^3 points max
    stride = max(1, N // 30)
    x_grid, y_grid, z_grid = np.meshgrid(x[::stride], y[::stride], z[::stride], indexing='ij')
    energy_sample = energy[::stride, ::stride, ::stride]
    
    # Flatten and filter by threshold
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = z_grid.flatten()
    e_flat = energy_sample.flatten()
    
    keep = e_flat > threshold
    x_keep = x_flat[keep]
    y_keep = y_flat[keep]
    z_keep = z_flat[keep]
    e_keep = e_flat[keep]
    
    # Scatter plot with color mapped to energy
    sc = ax.scatter(x_keep, y_keep, z_keep, c=e_keep, s=50, alpha=0.6, 
                    cmap='hot', vmin=vmin, vmax=vmax, marker='o', edgecolors='none')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    frame_energy = np.sum(e_keep) if len(e_keep) > 0 else 0.0
    ax.set_title(f"3D Energy Dispersion — Time: {time:.2f}s (E_vis={frame_energy:.3e})")
    ax.set_xlim([0, N*dx])
    ax.set_ylim([0, N*dx])
    ax.set_zlim([0, N*dx])
    
    # Set view angle (rotating camera)
    ax.view_init(elev=20, azim=angle)
    
    return sc

def main():
    parser = argparse.ArgumentParser(description="Visualize 3D energy dispersion from GRAV-15")
    parser.add_argument("--input", type=str, default="results/Gravity/GRAV-15/diagnostics/field_snapshots_3d_GRAV-15.h5",
                       help="Path to input HDF5 file")
    parser.add_argument("--output", type=str, default="results/Gravity/GRAV-15/energy_dispersion_3d.mp4",
                       help="Path to output MP4 file (or PNG directory if FFmpeg unavailable)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for output video")
    parser.add_argument("--frames", action="store_true", help="Save individual PNG frames instead of video")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    print(f"Loading snapshots from {input_path}...")
    N, dx, dt, steps_per_snap, snapshots = load_snapshots(input_path)
    print(f"Loaded {len(snapshots)} snapshots (N={N}, dx={dx}, dt={dt})")
    
    # Compute global energy range for consistent colormap
    print("Computing energy range...")
    all_energies = [compute_energy_density(field) for _, field in snapshots]
    vmax = max(e.max() for e in all_energies)
    vmin = 0.0
    print(f"Energy range: [{vmin:.3e}, {vmax:.3e}]")
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if args.frames:
        # Save individual PNG frames
        output_dir = Path(args.output).parent / "frames_3d"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(snapshots)} frames to {output_dir}...")
        
        for i, (time, field) in enumerate(snapshots):
            ax.clear()
            angle = 30 + (i / len(snapshots)) * 360  # One full rotation
            create_frame(ax, field, N, dx, time, angle, vmin, vmax, adaptive_threshold=True)
            
            frame_path = output_dir / f"frame_{i:04d}.png"
            fig.savefig(frame_path, dpi=args.dpi, bbox_inches='tight')
            
            if (i+1) % 20 == 0:
                print(f"  Saved {i+1}/{len(snapshots)} frames...")
        
        print(f"✅ Frames saved to {output_dir}")
        print(f"   Total: {len(snapshots)} PNG files")
        print(f"   To create video with FFmpeg:")
        print(f"   ffmpeg -framerate {args.fps} -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {args.output}")
        return
    
    # Animation function
    def animate(i):
        time, field = snapshots[i]
        angle = 30 + (i / len(snapshots)) * 360  # One full rotation
        sc = create_frame(ax, field, N, dx, time, angle, vmin, vmax, adaptive_threshold=True)
        return sc,
    
    print(f"Creating animation with {len(snapshots)} frames...")
    anim = FuncAnimation(fig, animate, frames=len(snapshots), interval=1000/args.fps, blit=False)
    
    # Save to MP4
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_path}...")
    try:
        writer = FFMpegWriter(fps=args.fps, bitrate=5000)
        anim.save(output_path, writer=writer, dpi=args.dpi)
        
        print(f"✅ Animation saved: {output_path}")
        print(f"   Duration: {len(snapshots)/args.fps:.1f}s at {args.fps} fps")
        print(f"   File size: {output_path.stat().st_size / (1024**2):.1f} MB")
    except FileNotFoundError:
        print(f"⚠️  FFmpeg not found. Falling back to PNG frames...")
        print(f"   Rerun with --frames flag or install FFmpeg for video output")
        # Fall back to saving frames
        args.frames = True
        plt.close(fig)
        main()  # Recursive call with frames mode

if __name__ == "__main__":
    main()
