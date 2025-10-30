#!/usr/bin/env python3
"""
Detailed analysis of GRAV-12 arrival times and measurement issues
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import csv
from pathlib import Path
from scipy.signal import hilbert

# Configuration
results_dir = Path("results/Gravity/GRAV-12/diagnostics")
output_path = Path("results/Gravity/GRAV-12/arrival_analysis.png")

# Read detector signals
times, sig_before, sig_after = [], [], []
with open(results_dir / "detector_signals_GRAV-12.csv", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        times.append(float(row['time']))
        sig_before.append(float(row['signal_before']))
        sig_after.append(float(row['signal_after']))

times = np.array(times)
sig_before = np.array(sig_before)
sig_after = np.array(sig_after)
dt = times[1] - times[0]

# Compute envelopes
env_before = np.abs(hilbert(sig_before))
env_after = np.abs(hilbert(sig_after))

# Apply threshold
threshold_frac = 0.6
thr_before = threshold_frac * env_before.max()
thr_after = threshold_frac * env_after.max()

# Find first crossing
def find_first_crossing(env, thr):
    for i, val in enumerate(env):
        if val >= thr:
            return i
    return None

idx_before = find_first_crossing(env_before, thr_before)
idx_after = find_first_crossing(env_after, thr_after)

t_thr_before = times[idx_before] if idx_before else 0
t_thr_after = times[idx_after] if idx_after else 0

# Compute segment centroid
def segment_centroid(env, thr, times):
    i0 = find_first_crossing(env, thr)
    if i0 is None:
        return 0.0
    i1 = i0
    while i1+1 < len(env) and env[i1+1] >= thr:
        i1 += 1
    seg = slice(i0, i1+1)
    w = env[seg]**2
    return float(np.sum(times[seg] * w) / max(np.sum(w), 1e-30))

t_cent_before = segment_centroid(env_before, thr_before, times)
t_cent_after = segment_centroid(env_after, thr_after, times)

# Compute 50% cumulative energy crossing (most robust method)
def cumulative_energy_50(env, times):
    energy_density = env**2
    cumulative_energy = np.cumsum(energy_density)
    total_energy = cumulative_energy[-1]
    if total_energy <= 0:
        return 0.0
    half_energy = 0.5 * total_energy
    idx_50 = np.searchsorted(cumulative_energy, half_energy)
    idx_50 = min(idx_50, len(env) - 1)
    return times[idx_50]

t_50_before = cumulative_energy_50(env_before, times)
t_50_after = cumulative_energy_50(env_after, times)

# Read initial conditions to show packet shape
x_pos, E_init, Eprev_init, chi = [], [], [], []
with open(results_dir / "initial_conditions_GRAV-12.csv", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_pos.append(float(row['x_position']))
        E_init.append(float(row['E_init']))
        Eprev_init.append(float(row['Eprev_init']))
        chi.append(float(row['chi_field']))

x_pos = np.array(x_pos)
E_init = np.array(E_init)
chi = np.array(chi)

# Geometry
detector_before_x = 19.2
detector_after_x = 64.0
slab_x0 = 32.0
slab_x1 = 51.2

# Create comprehensive diagnostic figure
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

# 1. Initial condition
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(x_pos, E_init, 'b-', linewidth=2, label='E(x, t=0)')
ax1.axvline(detector_before_x, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Before Det (x={detector_before_x:.1f})')
ax1.axvline(detector_after_x, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'After Det (x={detector_after_x:.1f})')
ax1.axvspan(slab_x0, slab_x1, alpha=0.2, color='orange', label='χ Slab')
ax1_chi = ax1.twinx()
ax1_chi.plot(x_pos, chi, 'orange', linewidth=1.5, alpha=0.5, linestyle=':')
ax1_chi.set_ylabel('χ(x)', color='orange')
ax1_chi.tick_params(axis='y', labelcolor='orange')
ax1.set_xlabel('Position x')
ax1.set_ylabel('Field E(x, t=0)')
ax1.set_title('Initial Wave Packet Configuration', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. Before detector: signal and envelope
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(times, sig_before, 'g-', alpha=0.3, linewidth=0.5, label='Signal')
ax2.plot(times, env_before, 'g-', linewidth=2, label='Envelope')
ax2.axhline(thr_before, color='darkgreen', linestyle=':', linewidth=1.5, label=f'Threshold ({threshold_frac*100:.0f}%)')
ax2.axvline(t_thr_before, color='darkgreen', linestyle='--', linewidth=2, alpha=0.8, label=f'Threshold: {t_thr_before:.1f}s')
ax2.axvline(t_50_before, color='blue', linestyle='--', linewidth=2.5, alpha=0.9, label=f'50% energy: {t_50_before:.1f}s')
ax2.axvline(t_cent_before, color='lime', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Centroid: {t_cent_before:.1f}s')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Signal / Envelope')
ax2.set_title(f'Before Detector (x={detector_before_x:.1f})', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, min(100, times[-1]))

# 3. After detector: signal and envelope
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(times, sig_after, 'r-', alpha=0.3, linewidth=0.5, label='Signal')
ax3.plot(times, env_after, 'r-', linewidth=2, label='Envelope')
ax3.axhline(thr_after, color='darkred', linestyle=':', linewidth=1.5, label=f'Threshold ({threshold_frac*100:.0f}%)')
ax3.axvline(t_thr_after, color='darkred', linestyle='--', linewidth=2, alpha=0.8, label=f'Threshold: {t_thr_after:.1f}s')
ax3.axvline(t_50_after, color='blue', linestyle='--', linewidth=2.5, alpha=0.9, label=f'50% energy: {t_50_after:.1f}s')
ax3.axvline(t_cent_after, color='salmon', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Centroid: {t_cent_after:.1f}s')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Signal / Envelope')
ax3.set_title(f'After Detector (x={detector_after_x:.1f})', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, min(100, times[-1]))

# 4. Zoomed envelope comparison
ax4 = fig.add_subplot(gs[2, :])
# Normalize envelopes for comparison
env_before_norm = env_before / env_before.max()
env_after_norm = env_after / env_after.max()
ax4.plot(times, env_before_norm, 'g-', linewidth=2, label=f'Before (x={detector_before_x:.1f})', alpha=0.7)
ax4.plot(times, env_after_norm, 'r-', linewidth=2, label=f'After (x={detector_after_x:.1f})', alpha=0.7)
ax4.axhline(threshold_frac, color='black', linestyle=':', linewidth=1, label=f'Threshold ({threshold_frac*100:.0f}%)')
ax4.axvline(t_50_before, color='blue', linestyle='--', linewidth=2, alpha=0.8, label='50% energy')
ax4.axvline(t_50_after, color='blue', linestyle='--', linewidth=2, alpha=0.8)
# Shade propagation window
ax4.axvspan(t_50_before, t_50_after, alpha=0.15, color='blue', label=f'Δt (50%) = {t_50_after-t_50_before:.1f}s')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Normalized Envelope')
ax4.set_title('Envelope Arrival Comparison (Normalized)', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, min(120, times[-1]))
ax4.set_ylim(0, 1.1)

# 5. Measurement summary with geometry diagram
ax5 = fig.add_subplot(gs[3, :])
ax5.axis('off')

# Theory calculation
c = 1.0  # sqrt(alpha/beta)
omega = 0.15
chi_bg = 0.01
chi_slab = 0.10
k_bg = np.sqrt(max(omega**2 - chi_bg**2, 1e-16)) / c
k_slab = np.sqrt(max(omega**2 - chi_slab**2, 1e-16)) / c
vg_bg = (c**2 * k_bg) / omega
vg_slab = (c**2 * k_slab) / omega

L_before = detector_before_x - slab_x0
L_slab = slab_x1 - slab_x0
L_after = detector_after_x - slab_x1
L_total = detector_after_x - detector_before_x

time_with_slab = (L_before / vg_bg) + (L_slab / vg_slab) + (L_after / vg_bg)
time_all_bg = L_total / vg_bg
delay_theory = time_with_slab - time_all_bg

delay_measured = (t_50_after - t_50_before) - time_all_bg
error_pct = abs(delay_measured - delay_theory) / delay_theory * 100

# Draw geometry diagram
y_base = 0.7
ax5.plot([0, 128], [y_base, y_base], 'k-', linewidth=1)
# Slab
ax5.add_patch(plt.Rectangle((slab_x0, y_base-0.05), L_slab, 0.1, 
                             facecolor='orange', alpha=0.3, edgecolor='orange', linewidth=2))
# Detectors
ax5.plot([detector_before_x], [y_base], 'go', markersize=15, label='Before')
ax5.plot([detector_after_x], [y_base], 'ro', markersize=15, label='After')
# Annotations
ax5.annotate('Before Det', xy=(detector_before_x, y_base), xytext=(detector_before_x, y_base+0.15),
            ha='center', fontsize=10, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
ax5.annotate('After Det', xy=(detector_after_x, y_base), xytext=(detector_after_x, y_base+0.15),
            ha='center', fontsize=10, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
ax5.annotate('χ Slab', xy=((slab_x0+slab_x1)/2, y_base), xytext=((slab_x0+slab_x1)/2, y_base-0.2),
            ha='center', fontsize=10, color='orange', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))

# Summary text
summary_text = f"""
MEASUREMENT SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Geometry:
  Before detector: x = {detector_before_x:.1f}  (upstream of slab)
  Slab region:     x = {slab_x0:.1f} → {slab_x1:.1f}  (L = {L_slab:.1f})
  After detector:  x = {detector_after_x:.1f}  (downstream of slab)
  Total separation: {L_total:.1f}

Group Velocities:
  Background:  v_g = {vg_bg:.4f}  (χ = {chi_bg:.3f})
  Slab:        v_g = {vg_slab:.4f}  (χ = {chi_slab:.3f})
  Slowdown:    {vg_bg/vg_slab:.2f}×

Arrival Times (50% Cumulative Energy - Most Robust):
  Before detector: t = {t_50_before:.2f} s
  After detector:  t = {t_50_after:.2f} s
  Propagation Δt:  {t_50_after - t_50_before:.2f} s
  
Comparison - Threshold crossing: Δt = {t_thr_after - t_thr_before:.2f} s
Comparison - Segment centroid:   Δt = {t_cent_after - t_cent_before:.2f} s

Theory vs Measurement:
  Expected Δt (all background):  {time_all_bg:.2f} s
  Expected Δt (with slab):       {time_with_slab:.2f} s
  Expected delay (Shapiro):      {delay_theory:.2f} s
  
  Measured delay:                {delay_measured:.2f} s
  Error:                         {error_pct:.1f}%

STATUS: Using 50% cumulative energy method (most robust to dispersion)
        Threshold method: Δt = {t_thr_after - t_thr_before:.2f}s
        50% energy method: Δt = {t_50_after - t_50_before:.2f}s
"""

ax5.text(0.05, 0.35, summary_text, transform=ax5.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

ax5.set_xlim(0, 128)
ax5.set_ylim(0, 1)

plt.suptitle('GRAV-12: Phase Delay Measurement Diagnostic', 
            fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Diagnostic plot saved to {output_path}")
print(f"\nKey finding:")
print(f"  50% energy crossing: before={t_50_before:.1f}s, after={t_50_after:.1f}s")
print(f"  Measured delay: {delay_measured:.2f}s vs theory {delay_theory:.2f}s ({error_pct:.1f}% error)")
print(f"  Method comparison:")
print(f"    Threshold:  Δt={(t_thr_after - t_thr_before):.2f}s, delay={(t_thr_after - t_thr_before) - time_all_bg:.2f}s")
print(f"    50% energy: Δt={(t_50_after - t_50_before):.2f}s, delay={delay_measured:.2f}s")
print(f"    Centroid:   Δt={(t_cent_after - t_cent_before):.2f}s, delay={(t_cent_after - t_cent_before) - time_all_bg:.2f}s")