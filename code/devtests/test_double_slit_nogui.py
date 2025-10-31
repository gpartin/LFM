"""
Headless test of double slit scenario physics
No pygame, no matplotlib GUI - just numerical validation
"""

import numpy as np
import sys
sys.path.insert(0, 'c:\\LFM\\code')
from lfm_equation import lattice_step

def test_double_slit():
    """Test double slit wave propagation without GUI"""
    
    # Grid setup
    GRID_SIZE = 128
    dx = 0.5
    dt = 0.1
    chi = 0.0
    gamma_damp = 0.01  # Much less damping to preserve wave
    
    # Barrier geometry
    barrier_x = GRID_SIZE // 3  # x=42
    slit_y1, slit_y2 = 52, 76
    slit_width = 4
    screen_x = int(0.75 * GRID_SIZE)  # x=96
    
    print("\n" + "="*60)
    print("DOUBLE SLIT PHYSICS TEST (HEADLESS)")
    print("="*60)
    print(f"Grid: {GRID_SIZE}×{GRID_SIZE}, dx={dx}, dt={dt}")
    print(f"Barrier at x={barrier_x}, slits at y={slit_y1} and y={slit_y2}")
    print(f"Detection screen at x={screen_x}")
    print(f"Wave damping: gamma={gamma_damp}")
    print("="*60)
    
    # Initialize fields
    E = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    E_prev = np.zeros_like(E)
    
    # Create plane wave (uniform across y, propagating in +x)
    amplitude = 1.0  # Strong amplitude
    wavelength = 10.0  # Spatial wavelength in grid units
    k_x = 2.0 * np.pi / wavelength  # wave number
    
    y_grid, x_grid = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    
    # Plane wave: E = A * sin(k*x)
    # For traveling wave, E_prev should be phase-shifted
    E[:, :10] = amplitude * np.sin(k_x * x_grid[:, :10])
    E_prev[:, :10] = amplitude * np.sin(k_x * (x_grid[:, :10] - dx))  # Shifted back for rightward motion
    
    # Simulation parameters
    params = {
        'dt': dt,
        'dx': dx,
        'chi': chi,
        'alpha': 1.0,
        'beta': 1.0,  # Must be > 0 for wave equation
        'gamma_damp': gamma_damp,
        'boundary': 'periodic',
        'stencil_order': 2,
        'precision': 'float64'
    }
    
    # Run simulation
    num_steps = 800  # More steps to see pattern develop
    diagnostics_every = 50
    source_duration = 300  # Longer source
    barrier_start_step = 15
    
    print("\nRunning simulation...")
    print(f"  Continuous source for first {source_duration} steps")
    print(f"  Barrier enabled after step {barrier_start_step}")
    print(f"  Total steps: {num_steps}")
    print()
    
    wave_reached_screen = False
    reached_at_step = -1
    max_screen_field = 0.0
    
    for i in range(num_steps):
        # Add continuous plane wave source (left side)
        if i < source_duration:
            # Plane wave oscillating in time
            omega = k_x * np.sqrt(params['alpha'] / params['beta'])  # Angular frequency from dispersion
            source_wave = amplitude * np.sin(omega * i * dt)
            E[:, 8] = source_wave  # Set entire column to plane wave value
        
        # Advance physics
        E_next = lattice_step(E, E_prev, params)
        E_prev[:, :] = E
        E[:, :] = E_next
        
        # Apply barrier (zero field except at slits)
        if i > barrier_start_step:
            for y in range(GRID_SIZE):
                # Check if this row is in a slit
                in_slit = False
                if (slit_y1 <= y < slit_y1 + slit_width) or (slit_y2 <= y < slit_y2 + slit_width):
                    in_slit = True
                
                if not in_slit:
                    E[y, barrier_x] = 0.0
                    E_prev[y, barrier_x] = 0.0
        
        # Diagnostics
        if i % diagnostics_every == 0:
            max_field = np.max(np.abs(E))
            field_at_barrier = np.mean(np.abs(E[:, barrier_x]))
            field_behind_barrier = np.mean(np.abs(E[:, barrier_x+10:barrier_x+20]))
            field_at_screen = np.mean(np.abs(E[:, screen_x]))
            energy = 0.5 * np.sum(E**2 + ((E - E_prev) / dt)**2) * dx * dx
            
            print(f"  Step {i:4d}: max={max_field:.3f}, barrier={field_at_barrier:.3f}, "
                  f"behind={field_behind_barrier:.3f}, screen={field_at_screen:.3f}, E={energy:.2f}")
            
            # Check for numerical instability
            if max_field > 100 or np.isnan(max_field):
                print(f"\n❌ NUMERICAL INSTABILITY at step {i}")
                print(f"   max_field = {max_field}")
                return False
            
            # Check if wave reached screen
            if field_at_screen > 0.01 and not wave_reached_screen:
                wave_reached_screen = True
                reached_at_step = i
                print(f"\n✓ Wave reached screen at step {i}")
            
            max_screen_field = max(max_screen_field, field_at_screen)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    
    # Final analysis
    screen_profile = E[:, screen_x]
    max_screen_amplitude = np.max(np.abs(screen_profile))
    
    print(f"Wave reached screen: {wave_reached_screen}")
    if wave_reached_screen:
        print(f"  Reached at step: {reached_at_step}")
    print(f"Max screen field: {max_screen_field:.4f}")
    print(f"Final screen amplitude: {max_screen_amplitude:.4f}")
    
    # Count peaks for interference pattern
    threshold = 0.1 * max_screen_amplitude
    peaks = []
    for y in range(1, GRID_SIZE - 1):
        val = abs(screen_profile[y])
        if val > threshold and val > abs(screen_profile[y-1]) and val > abs(screen_profile[y+1]):
            peaks.append(y)
    
    print(f"Peaks detected on screen: {len(peaks)}")
    if len(peaks) > 0:
        print(f"  Peak positions: {peaks}")
    
    # Pass/fail criteria
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    passed = True
    issues = []
    
    # 1. Numerical stability
    final_max = np.max(np.abs(E))
    if final_max > 100 or np.isnan(final_max):
        passed = False
        issues.append("Numerical instability")
    else:
        print("✓ Numerically stable")
    
    # 2. Wave reaches screen
    if not wave_reached_screen:
        passed = False
        issues.append("Wave did not reach screen")
    else:
        print(f"✓ Wave reached screen at step {reached_at_step}")
    
    # 3. Interference pattern
    if len(peaks) < 2:
        passed = False
        issues.append(f"No interference pattern (only {len(peaks)} peaks)")
    else:
        print(f"✓ Interference pattern detected ({len(peaks)} peaks)")
    
    if passed:
        print("\n" + "="*60)
        print("✅ TEST PASSED")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    return passed

if __name__ == "__main__":
    success = test_double_slit()
    sys.exit(0 if success else 1)
