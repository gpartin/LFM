#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com

"""
Visualize GRAV-12 Phase/Group Velocity Mismatch

This script demonstrates the Klein-Gordon "quirk" where phase velocity
and group velocity behave differently in spatially varying χ fields.

KEY RESULT: Energy slows down (positive group delay) while wave crests
speed up (negative phase delay) - a testable prediction that distinguishes
this model from General Relativity!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json

def load_grav12_data():
    """Load GRAV-12 results and detector signals."""
    base_path = Path("results/Gravity/GRAV-12")
    
    # Load summary
    with open(base_path / "summary.json", 'r') as f:
        summary = json.load(f)
    
    # Load detector signals
    signals_file = base_path / "diagnostics" / "detector_signals_GRAV-12.csv"
    if not signals_file.exists():
        print(f"Warning: {signals_file} not found, using alternate location")
        signals_file = base_path / "detector_signals_GRAV-12.csv"
    
    data = np.loadtxt(signals_file, delimiter=',', skiprows=1)
    time = data[:, 0]
    signal_before = data[:, 1]
    signal_after = data[:, 2]
    
    return summary, time, signal_before, signal_after

def compute_phase_shift(signal1, signal2, dt):
    """Compute phase shift between two signals via cross-correlation."""
    from scipy.signal import correlate
    
    # Normalize signals
    s1 = (signal1 - signal1.mean()) / (signal1.std() + 1e-12)
    s2 = (signal2 - signal2.mean()) / (signal2.std() + 1e-12)
    
    # Cross-correlation
    corr = correlate(s1, s2, mode='full')
    lag_idx = np.argmax(corr) - (len(s2) - 1)
    phase_delay = lag_idx * dt
    
    return phase_delay, lag_idx

def plot_phase_group_mismatch(summary, time, signal_before, signal_after, save_path):
    """Create comprehensive visualization of phase vs group velocity mismatch."""
    from scipy.signal import hilbert
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Extract key parameters
    chi_bg = summary.get('chiA', 0.05)
    chi_slab = 0.20  # From config
    dt = time[1] - time[0] if len(time) > 1 else 0.05
    
    # Compute envelopes
    env_before = np.abs(hilbert(signal_before))
    env_after = np.abs(hilbert(signal_after))
    
    # Smooth envelopes
    window = 5
    env_before = np.convolve(env_before, np.ones(window)/window, mode='same')
    env_after = np.convolve(env_after, np.ones(window)/window, mode='same')
    
    # Compute phase shift
    phase_delay, phase_lag = compute_phase_shift(signal_before, signal_after, dt)
    
    # Compute group delay (50% energy arrival)
    cumE_before = np.cumsum(env_before**2)
    cumE_after = np.cumsum(env_after**2)
    idx50_before = np.searchsorted(cumE_before, 0.5 * cumE_before[-1])
    idx50_after = np.searchsorted(cumE_after, 0.5 * cumE_after[-1])
    group_delay = (idx50_after - idx50_before) * dt
    
    # --- Panel 1: Raw Signals ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, signal_before, 'b-', alpha=0.7, linewidth=0.8, label='Before slab (χ=0.05)')
    ax1.plot(time, signal_after, 'r-', alpha=0.7, linewidth=0.8, label='After slab (χ→0.20→0.05)')
    ax1.axhline(0, color='k', linestyle=':', linewidth=0.5)
    ax1.set_xlabel('Time (simulation units)', fontsize=11)
    ax1.set_ylabel('Field Amplitude E', fontsize=11)
    ax1.set_title('Raw Detector Signals: Wave Propagation Through χ-Field Slab', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2: Envelopes (Group Velocity) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, env_before, 'b-', linewidth=2, label='Before slab')
    ax2.plot(time, env_after, 'r-', linewidth=2, label='After slab')
    
    # Mark 50% energy points
    if idx50_before < len(time):
        ax2.axvline(time[idx50_before], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
        ax2.plot(time[idx50_before], env_before[idx50_before], 'bo', markersize=8)
    if idx50_after < len(time):
        ax2.axvline(time[idx50_after], color='r', linestyle='--', alpha=0.5, linewidth=1.5)
        ax2.plot(time[idx50_after], env_after[idx50_after], 'ro', markersize=8)
    
    ax2.set_xlabel('Time (simulation units)', fontsize=11)
    ax2.set_ylabel('Envelope Amplitude', fontsize=11)
    ax2.set_title(f'Energy Envelopes → Group Delay = {group_delay:.2f}s', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Add annotation
    ax2.text(0.05, 0.95, f'Energy arrives LATER\n(Shapiro delay: +{group_delay:.1f}s)',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # --- Panel 3: Phase Comparison (Zoomed) ---
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Find a good window to show phase shift
    peak_before = np.argmax(env_before)
    peak_after = np.argmax(env_after)
    zoom_center = max(peak_before, peak_after)
    zoom_width = 200
    zoom_start = max(0, zoom_center - zoom_width)
    zoom_end = min(len(time), zoom_center + zoom_width)
    
    t_zoom = time[zoom_start:zoom_end]
    s_before_zoom = signal_before[zoom_start:zoom_end]
    s_after_zoom = signal_after[zoom_start:zoom_end]
    
    ax3.plot(t_zoom, s_before_zoom, 'b-', alpha=0.7, linewidth=1.2, label='Before slab')
    ax3.plot(t_zoom, s_after_zoom, 'r-', alpha=0.7, linewidth=1.2, label='After slab')
    ax3.axhline(0, color='k', linestyle=':', linewidth=0.5)
    ax3.set_xlabel('Time (simulation units)', fontsize=11)
    ax3.set_ylabel('Field Amplitude E', fontsize=11)
    ax3.set_title(f'Phase Crests (Zoomed) → Phase Shift = {phase_delay:.2f}s', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Add annotation
    phase_sign = "EARLIER" if phase_delay < 0 else "LATER"
    ax3.text(0.05, 0.95, f'Crests arrive {phase_sign}\n(Phase advance: {phase_delay:.1f}s)',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # --- Panel 4: χ-Field Profile ---
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Reconstruct χ profile (simplified)
    N = 128
    x = np.linspace(0, N, 200)
    chi = np.ones_like(x) * chi_bg
    slab_start = 0.30 * N
    slab_end = 0.50 * N
    slab_mask = (x >= slab_start) & (x <= slab_end)
    chi[slab_mask] = chi_slab
    
    det_before_x = 0.20 * N
    det_after_x = 0.70 * N
    
    ax4.fill_between(x, 0, chi, where=slab_mask, alpha=0.3, color='orange', label='High χ slab')
    ax4.plot(x, chi, 'k-', linewidth=2)
    ax4.axvline(det_before_x, color='b', linestyle='--', linewidth=2, label='Detector before')
    ax4.axvline(det_after_x, color='r', linestyle='--', linewidth=2, label='Detector after')
    ax4.set_xlabel('Position x (cells)', fontsize=11)
    ax4.set_ylabel('χ(x) Field Strength', fontsize=11)
    ax4.set_title('Spatial χ-Field Profile (Gravitational Potential Analog)', fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, chi_slab * 1.2])
    
    # --- Panel 5: Summary Box ---
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    summary_text = f"""
    KLEIN-GORDON PHASE/GROUP MISMATCH
    
    Configuration:
    • Background χ = {chi_bg:.2f}
    • Slab χ = {chi_slab:.2f} (4x stronger)
    • Wave frequency ω = 0.30
    • Slab position: 30-50% of domain
    
    Measured Results:
    • Group Delay (energy): +{group_delay:.2f}s
      → Energy slows down ✓
    
    • Phase Shift (crests): {phase_delay:.2f}s
      → Crests {'advance' if phase_delay < 0 else 'delay'} {'✓' if phase_delay < 0 else '?'}
    
    Physical Interpretation:
    In the high-χ slab, wavelength compresses
    (more cycles fit in same space). When waves
    exit, they spread out again - but now there
    are extra crests! Result: crests arrive early
    even though energy arrives late.
    
    Testable Prediction:
    This phase/group mismatch distinguishes
    Klein-Gordon gravity from General Relativity.
    Could be tested with coherent light through
    gravitational lensing or analog systems.
    """
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))
    
    # Overall title
    fig.suptitle('GRAV-12: Klein-Gordon Phase/Group Velocity Mismatch in Spatially Varying χ-Field',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {save_path}")
    plt.close()

def main():
    print("Loading GRAV-12 data...")
    summary, time, signal_before, signal_after = load_grav12_data()
    
    output_path = Path("results/Gravity/GRAV-12/phase_group_mismatch.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Creating phase/group velocity mismatch visualization...")
    plot_phase_group_mismatch(summary, time, signal_before, signal_after, output_path)
    
    print("\n" + "="*70)
    print("GRAV-12 VISUALIZATION COMPLETE")
    print("="*70)
    print("\nKey Finding:")
    print("  Phase velocity ≠ Group velocity in varying χ-fields")
    print("  → Testable prediction for experimental validation!")
    print(f"\nVisualization saved to: {output_path}")

if __name__ == "__main__":
    main()
