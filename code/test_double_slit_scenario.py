# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Test script for double slit scenario - validates before deploying to interactive app
"""
import pytest
pytest.skip("Moved to tests/test_double_slit_scenario.py; module skipped", allow_module_level=True)

import numpy as np
from lfm_equation import lattice_step, energy_total
import matplotlib.pyplot as plt

# Grid setup
GRID_SIZE = 128
DX = 0.5
DT = 0.1

# Barrier geometry
barrier_x = GRID_SIZE // 3
slit_width = 4
slit_separation = 24
center_y = GRID_SIZE // 2
slit1_y = center_y - slit_separation // 2
slit2_y = center_y + slit_separation // 2

print("="*60)
print("DOUBLE SLIT SCENARIO TEST")
print("="*60)
print(f"Grid: {GRID_SIZE}x{GRID_SIZE}")
print(f"Barrier at x={barrier_x}")
print(f"Slits at y={slit1_y} and y={slit2_y}")
print("="*60)

# Initialize fields
E = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
E_prev = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)

# Create initial wave with rightward momentum
wave_start_x = 15
wave_width_x = 8
wave_width_y = 40
wave_amplitude = 0.5  # Reduced amplitude
wave_center_y = GRID_SIZE // 2

y_grid, x_grid = np.ogrid[0:GRID_SIZE, 0:GRID_SIZE]

# Wave packet at t=0
wave_packet = wave_amplitude * np.exp(
    -(x_grid - wave_start_x)**2 / (2 * wave_width_x**2) -
    (y_grid - wave_center_y)**2 / (2 * wave_width_y**2)
)

# Wave packet at t=-1 (offset left to create rightward motion)
E = wave_packet.copy()
E_prev = wave_amplitude * np.exp(
    -(x_grid - (wave_start_x - 2))**2 / (2 * wave_width_x**2) -
    (y_grid - wave_center_y)**2 / (2 * wave_width_y**2)
)

# Physics params
params = {
    'dt': DT,
    'dx': DX,
    'alpha': 1.0,
    'beta': 1.0,
    'chi': 0.0,
    'gamma_damp': 0.05,  # Light damping
    'boundary': 'periodic',
    'stencil_order': 2,
}

# Run simulation with diagnostics
max_steps = 600
check_every = 50

print("\nRunning simulation...")
for i in range(max_steps):
    # Add continuous gentle wave source
    if i < 200:
        source = 0.05 * wave_amplitude * np.exp(
            -(x_grid - 10)**2 / (2 * 4**2) -
            (y_grid - wave_center_y)**2 / (2 * wave_width_y**2)
        ) * np.sin(i * 0.15)
        E += source
    
    # Apply barrier - zero field at barrier except slits
    if i > 15:  # Start barrier after wave forms
        for y in range(GRID_SIZE):
            is_slit = (abs(y - slit1_y) < slit_width) or (abs(y - slit2_y) < slit_width)
            if not is_slit:
                E[y, barrier_x] = 0.0
                E_prev[y, barrier_x] = 0.0
    
    # Step physics
    E_next = lattice_step(E, E_prev, params)
    E_prev = E
    E = E_next
    
    # Diagnostics
    if i % check_every == 0 or i < 20:
        max_field = np.max(np.abs(E))
        field_at_barrier = np.max(np.abs(E[:, barrier_x]))
        field_behind_barrier = np.max(np.abs(E[:, barrier_x+10:barrier_x+20]))
        field_at_screen = np.max(np.abs(E[:, int(GRID_SIZE * 0.75)]))
        
        c = 1.0
        energy = energy_total(E, E_prev, DT, DX, c, 0.0)
        
        print(f"Step {i:4d}: max={max_field:8.3f}, barrier={field_at_barrier:8.3f}, "
              f"behind={field_behind_barrier:8.3f}, screen={field_at_screen:8.3f}, "
              f"energy={energy:.3e}")
        
        # Check for instability
        if max_field > 100 or np.isnan(max_field):
            print("\n❌ FAILED: Numerical instability detected!")
            break
            
        # Check if wave reached screen
        if i > 200 and field_at_screen > 0.01:
            print(f"\n✓ Wave reached detection screen at step {i}!")

print("\n" + "="*60)

# Analyze final state
print("\nFINAL ANALYSIS:")
print(f"  Max field: {np.max(np.abs(E)):.3f}")
print(f"  Field at barrier: {np.max(np.abs(E[:, barrier_x])):.3f}")
print(f"  Field behind barrier: {np.max(np.abs(E[:, barrier_x+10:barrier_x+20])):.3f}")
print(f"  Field at screen: {np.max(np.abs(E[:, int(GRID_SIZE * 0.75)])):.3f}")

# Check for interference pattern
screen_x = int(GRID_SIZE * 0.75)
screen_profile = E[:, screen_x]
peaks = []
for y in range(5, GRID_SIZE-5):
    if screen_profile[y] > screen_profile[y-1] and screen_profile[y] > screen_profile[y+1]:
        if abs(screen_profile[y]) > 0.001:
            peaks.append(y)

print(f"\n  Peaks on screen: {len(peaks)}")
if len(peaks) >= 3:
    print("  ✓ Interference pattern detected!")
else:
    print("  ⚠ No clear interference pattern yet")

# Plot final state
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Field visualization
ax = axes[0]
im = ax.imshow(E.T, cmap='RdBu', origin='lower', aspect='auto', vmin=-0.5, vmax=0.5)
ax.axvline(barrier_x, color='yellow', linewidth=2, label='Barrier')
ax.axvline(screen_x, color='lightgray', linewidth=2, linestyle='--', label='Screen')
ax.axhline(slit1_y, color='yellow', linewidth=1, linestyle=':')
ax.axhline(slit2_y, color='yellow', linewidth=1, linestyle=':')
ax.set_title('Final Field State')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.colorbar(im, ax=ax)

# Cross-section at barrier
ax = axes[1]
ax.plot(E[:, barrier_x], range(GRID_SIZE), 'b-', label='At barrier')
ax.plot(E[:, barrier_x + 10], range(GRID_SIZE), 'g-', label='Behind barrier')
ax.axhline(slit1_y, color='r', linestyle='--', alpha=0.5)
ax.axhline(slit2_y, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Field amplitude')
ax.set_ylabel('Y position')
ax.set_title('Field vs Y position')
ax.legend()
ax.grid(True, alpha=0.3)

# Screen profile
ax = axes[2]
ax.plot(screen_profile, range(GRID_SIZE), 'k-', linewidth=2)
ax.axhline(slit1_y, color='r', linestyle='--', alpha=0.5, label='Slit positions')
ax.axhline(slit2_y, color='r', linestyle='--', alpha=0.5)
for peak_y in peaks:
    ax.plot(screen_profile[peak_y], peak_y, 'ro', markersize=8)
ax.set_xlabel('Field amplitude')
ax.set_ylabel('Y position')
ax.set_title(f'Detection Screen Profile ({len(peaks)} peaks)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('double_slit_test_result.png', dpi=150)
print("\n✓ Saved diagnostic plot to double_slit_test_result.png")

print("\n" + "="*60)
if np.max(np.abs(E)) < 100 and field_at_screen > 0.01 and len(peaks) >= 2:
    print("✅ DOUBLE SLIT TEST PASSED!")
    print("   Ready to deploy to interactive app")
else:
    print("❌ DOUBLE SLIT TEST NEEDS WORK")
    print("   Issues to fix:")
    if np.max(np.abs(E)) >= 100:
        print("   - Numerical instability")
    if field_at_screen < 0.01:
        print("   - Wave not reaching screen")
    if len(peaks) < 2:
        print("   - No interference pattern forming")
print("="*60)
