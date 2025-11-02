#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com

"""
Generate animated GIF visualization of GRAV-12 wave packet propagation
Shows field evolution, envelope, chi field, and detector positions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import csv
from pathlib import Path

# Configuration
results_dir = Path("results/Gravity/GRAV-12/diagnostics")
output_path = Path("results/Gravity/GRAV-12/packet_propagation.gif")

# Read field snapshots
snapshots = {}
chi_field = None
x_positions = None

with open(results_dir / "field_snapshots_GRAV-12.csv", 'r') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    step_cols = [h for h in headers if h.startswith('E_step')]
    
    rows = list(reader)
    N = len(rows)
    x_positions = np.array([float(row['x_position']) for row in rows])
    chi_field = np.array([float(row['chi']) for row in rows])
    
    for col in step_cols:
        step = int(col.replace('E_step', ''))
        snapshots[step] = np.array([float(row[col]) for row in rows])

# Read detector signals for envelope overlay
detector_signals = {'before': [], 'after': [], 'times': []}
with open(results_dir / "detector_signals_GRAV-12.csv", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        detector_signals['times'].append(float(row['time']))
        detector_signals['before'].append(float(row['signal_before']))
        detector_signals['after'].append(float(row['signal_after']))

times = np.array(detector_signals['times'])
sig_before = np.array(detector_signals['before'])
sig_after = np.array(detector_signals['after'])

# Compute envelopes using Hilbert transform
from scipy.signal import hilbert
env_before = np.abs(hilbert(sig_before))
env_after = np.abs(hilbert(sig_after))

# Read config to get detector positions
# Hardcode based on latest run
detector_before_x = 19.2  # 0.15 * 64 * 2.0
detector_after_x = 64.0   # 0.50 * 64 * 2.0
slab_x0 = 32.0  # 0.25 * 64 * 2.0
slab_x1 = 51.2  # 0.40 * 64 * 2.0

# Create figure with multiple subplots
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

ax_field = fig.add_subplot(gs[0, :])  # Top: field snapshot
ax_chi = fig.add_subplot(gs[1, :])    # Chi field
ax_env_before = fig.add_subplot(gs[2, 0])  # Envelope at before detector
ax_env_after = fig.add_subplot(gs[2, 1])   # Envelope at after detector
ax_energy = fig.add_subplot(gs[3, :])      # Total energy vs time

# Sort snapshots by step
sorted_steps = sorted(snapshots.keys())
snapshot_times = np.array(sorted_steps) * 0.1  # dt = 0.1

# Read packet analysis for energy
packet_energy = []
packet_times = []
with open(results_dir / "packet_analysis_GRAV-12.csv", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        packet_times.append(float(row['time']))
        packet_energy.append(float(row['total_energy']))

def init():
    """Initialize animation"""
    ax_field.clear()
    ax_chi.clear()
    ax_env_before.clear()
    ax_env_after.clear()
    ax_energy.clear()
    return []

def animate(frame):
    """Update animation frame"""
    ax_field.clear()
    ax_chi.clear()
    ax_env_before.clear()
    ax_env_after.clear()
    ax_energy.clear()
    
    step = sorted_steps[frame]
    t = snapshot_times[frame]
    field = snapshots[step]
    
    # Top panel: Field snapshot with envelope
    ax_field.plot(x_positions, field, 'b-', linewidth=1.5, label='E(x,t)')
    ax_field.axvline(detector_before_x, color='green', linestyle='--', alpha=0.7, label='Before Det')
    ax_field.axvline(detector_after_x, color='red', linestyle='--', alpha=0.7, label='After Det')
    ax_field.axvspan(slab_x0, slab_x1, alpha=0.2, color='orange', label='χ Slab')
    ax_field.set_xlabel('Position x')
    ax_field.set_ylabel('Field E')
    ax_field.set_title(f'Wave Packet Propagation (t = {t:.1f} s, step {step})')
    ax_field.legend(loc='upper right', fontsize=8)
    ax_field.grid(True, alpha=0.3)
    y_max = max(np.abs(field).max(), 0.02)
    ax_field.set_ylim(-y_max*1.2, y_max*1.2)
    
    # Chi field panel
    ax_chi.plot(x_positions, chi_field, 'orange', linewidth=2)
    ax_chi.axvline(detector_before_x, color='green', linestyle='--', alpha=0.5)
    ax_chi.axvline(detector_after_x, color='red', linestyle='--', alpha=0.5)
    ax_chi.axvspan(slab_x0, slab_x1, alpha=0.1, color='orange')
    ax_chi.set_xlabel('Position x')
    ax_chi.set_ylabel('χ(x)')
    ax_chi.set_title('χ-Field Configuration')
    ax_chi.grid(True, alpha=0.3)
    
    # Envelope at before detector (time domain)
    time_idx = np.searchsorted(times, t)
    ax_env_before.plot(times[:time_idx+1], env_before[:time_idx+1], 'g-', linewidth=1)
    ax_env_before.axvline(t, color='gray', linestyle=':', alpha=0.5)
    ax_env_before.set_xlabel('Time (s)')
    ax_env_before.set_ylabel('Envelope')
    ax_env_before.set_title(f'Before Detector (x={detector_before_x:.1f})')
    ax_env_before.grid(True, alpha=0.3)
    ax_env_before.set_xlim(0, times[-1])
    
    # Envelope at after detector
    ax_env_after.plot(times[:time_idx+1], env_after[:time_idx+1], 'r-', linewidth=1)
    ax_env_after.axvline(t, color='gray', linestyle=':', alpha=0.5)
    ax_env_after.set_xlabel('Time (s)')
    ax_env_after.set_ylabel('Envelope')
    ax_env_after.set_title(f'After Detector (x={detector_after_x:.1f})')
    ax_env_after.grid(True, alpha=0.3)
    ax_env_after.set_xlim(0, times[-1])
    
    # Energy evolution
    time_idx_energy = np.searchsorted(packet_times, t)
    if time_idx_energy > 0:
        ax_energy.plot(packet_times[:time_idx_energy+1], packet_energy[:time_idx_energy+1], 'b-', linewidth=1.5)
    ax_energy.axvline(t, color='gray', linestyle=':', alpha=0.5, label='Current time')
    ax_energy.set_xlabel('Time (s)')
    ax_energy.set_ylabel('Total Energy')
    ax_energy.set_title('Packet Energy Evolution')
    ax_energy.grid(True, alpha=0.3)
    ax_energy.set_xlim(0, max(packet_times))
    
    return []

# Create animation
num_frames = len(sorted_steps)
print(f"Creating animation with {num_frames} frames...")

anim = animation.FuncAnimation(
    fig, 
    animate, 
    init_func=init,
    frames=num_frames,
    interval=200,  # 200ms per frame = 5 fps
    blit=True,
    repeat=True
)

# Save as GIF
print(f"Saving animation to {output_path}...")
writer = animation.PillowWriter(fps=5, bitrate=1800)
anim.save(output_path, writer=writer)

print(f"✅ Animation saved successfully!")
print(f"   Output: {output_path}")
print(f"   Frames: {num_frames}")
print(f"   Duration: ~{num_frames*0.2:.1f}s")
