#!/usr/bin/env python3
"""
LFM Emergence Test - Can χ-field structure emerge spontaneously?

This test addresses the core question: Does the LFM lattice genuinely generate
gravitational-like effects from energy dynamics, or are they pre-programmed?

Test Design:
1. Start with UNIFORM χ-field (no gravitational structure)
2. Place localized energy packet
3. Allow χ to evolve self-consistently with energy density
4. Measure: Does χ-well spontaneously form around energy?

If successful, this proves genuine emergence rather than circular validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import LFM modules
from lfm_equation import advance, energy_total
from chi_field_equation import evolve_coupled_fields, compute_chi_from_energy_poisson
from lfm_backend import to_numpy

def test_spontaneous_chi_generation():
    """
    Test if χ-field structure can emerge from uniform initial conditions
    """
    print("=== LFM Emergence Test: Spontaneous χ-Field Generation ===")
    
    # Grid setup
    N = 128
    dx = 0.5
    dt = 0.01
    steps = 1000
    
    # Physics parameters
    c = 1.0  # wave speed
    chi_bg = 0.1  # uniform background
    G_coupling = 0.05  # strength of energy → χ coupling
    
    print(f"Grid: {N} points, dx={dx}, dt={dt}")
    print(f"Background χ = {chi_bg}, G_coupling = {G_coupling}")
    
    # Initial conditions: uniform χ + localized energy packet
    x = np.arange(N) * dx
    x_center = N * dx / 2
    sigma = 5.0  # packet width
    
    # Start with COMPLETELY UNIFORM χ-field
    chi_init = np.full(N, chi_bg, dtype=np.float64)
    
    # Localized Gaussian energy packet
    E_init = 0.5 * np.exp(-((x - x_center)**2) / (2 * sigma**2))
    
    print(f"Initial energy packet: center={x_center:.1f}, width={sigma:.1f}")
    print(f"Initial χ variation: {np.std(chi_init):.6f} (should be 0.0)")
    
    # Evolve coupled E-χ system
    print("\nEvolving coupled fields...")
    start_time = time.time()
    
    E_final, chi_final, history = evolve_coupled_fields(
        E_init=E_init,
        chi_init=chi_init,
        dt=dt,
        dx=dx,
        steps=steps,
        G_coupling=G_coupling,
        c=c,
        chi_update_every=1,  # Update χ every step
        verbose=True
    )
    
    elapsed = time.time() - start_time
    print(f"Evolution completed in {elapsed:.2f}s")
    
    # Analysis: Did χ-structure emerge?
    chi_variation_final = np.std(chi_final)
    chi_min, chi_max = np.min(chi_final), np.max(chi_final)
    
    # Find χ-well depth at energy packet location
    packet_center_idx = np.argmax(np.abs(E_final))
    chi_at_packet = chi_final[packet_center_idx]
    chi_enhancement = (chi_at_packet - chi_bg) / chi_bg
    
    print(f"\n=== RESULTS ===")
    print(f"Final χ variation: {chi_variation_final:.6f}")
    print(f"χ range: [{chi_min:.4f}, {chi_max:.4f}]")
    print(f"χ at packet center: {chi_at_packet:.4f} (bg: {chi_bg:.4f})")
    print(f"χ enhancement: {chi_enhancement*100:.1f}%")
    
    # Test for genuine emergence
    emergence_threshold = 0.01  # 1% enhancement required
    emerged = abs(chi_enhancement) > emergence_threshold
    
    print(f"\n=== EMERGENCE TEST ===")
    print(f"Threshold for emergence: {emergence_threshold*100:.1f}%")
    print(f"Observed enhancement: {chi_enhancement*100:.1f}%")
    print(f"EMERGENCE DETECTED: {'YES' if emerged else 'NO'}")
    
    # Energy analysis
    E_final_energy = np.sum(E_final**2) * dx
    E_init_energy = np.sum(E_init**2) * dx
    energy_ratio = E_final_energy / E_init_energy
    
    print(f"\nEnergy conservation: {energy_ratio:.6f} (should be ~1.0)")
    
    # Create plots
    plot_results(x, E_init, E_final, chi_init, chi_final, history)
    
    return {
        'emerged': emerged,
        'chi_enhancement': chi_enhancement,
        'chi_variation': chi_variation_final,
        'energy_ratio': energy_ratio,
        'steps': steps
    }

def plot_results(x, E_init, E_final, chi_init, chi_final, history):
    """
    Visualize the emergence test results
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Initial vs Final E-field
    axes[0,0].plot(x, E_init, 'b-', label='Initial E', alpha=0.7)
    axes[0,0].plot(x, E_final, 'r-', label='Final E', linewidth=2)
    axes[0,0].set_xlabel('Position')
    axes[0,0].set_ylabel('E-field')
    axes[0,0].set_title('Energy Field Evolution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Initial vs Final χ-field  
    axes[0,1].plot(x, chi_init, 'g--', label='Initial χ (uniform)', alpha=0.7)
    axes[0,1].plot(x, chi_final, 'orange', label='Final χ', linewidth=2)
    axes[0,1].set_xlabel('Position')
    axes[0,1].set_ylabel('χ-field')
    axes[0,1].set_title('χ-Field Structure Formation')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Time evolution of χ variation
    if history:
        times = [h[0] for h in history]
        chi_vars = [np.std(h[2]) for h in history]
        axes[1,0].plot(times, chi_vars, 'purple', linewidth=2)
        axes[1,0].set_xlabel('Time Step')
        axes[1,0].set_ylabel('χ Variation (std)')
        axes[1,0].set_title('χ-Structure Growth')
        axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Energy-χ correlation
    energy_density = E_final**2
    axes[1,1].scatter(energy_density, chi_final, alpha=0.6, s=20)
    axes[1,1].set_xlabel('Energy Density |E|²')
    axes[1,1].set_ylabel('χ-field')
    axes[1,1].set_title('Energy-χ Coupling')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('emergence_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as: emergence_test_results.png")

if __name__ == "__main__":
    results = test_spontaneous_chi_generation()
    
    print(f"\n=== FINAL VERDICT ===")
    if results['emerged']:
        print("✅ EMERGENCE CONFIRMED: χ-field structure formed spontaneously!")
        print("   This suggests genuine physics emergence rather than circular validation.")
    else:
        print("❌ NO EMERGENCE: χ-field remained uniform despite energy dynamics.")
        print("   This suggests the system may require pre-programmed χ structure.")
    
    print(f"\nTo clean up: delete emergence_test_results.png")