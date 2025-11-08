# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
EM-21 Domain Size Convergence Study
Test boundary independence by running with increasingly large domains.
"""
import json
import sys
from pathlib import Path
# Bootstrap to import path_utils from the workspace root
_WS_DIR = Path(__file__).resolve().parents[1]
if str(_WS_DIR) not in sys.path:
    sys.path.insert(0, str(_WS_DIR))
from path_utils import add_workspace_to_sys_path, get_workspace_dir
import numpy as np
import matplotlib.pyplot as plt

WS = add_workspace_to_sys_path(__file__)
from run_tier5_electromagnetic import Tier5ElectromagneticHarness

config_path = WS / "config" / "config_tier5_electromagnetic.json"
output_dir = WS / "results" / "Electromagnetic" / "EM-21"
output_dir.mkdir(parents=True, exist_ok=True)

# Domain size multipliers to test
domain_multipliers = [1.0, 1.5, 2.0, 3.0, 4.0]

# We'll scale N while keeping dx constant, so physical domain size increases
# Keep gate region fixed in physical space

results_mur = []
results_pml = []

with open(config_path, 'r', encoding='utf-8') as f:
    base_config = json.load(f)

base_N = int(base_config["parameters"]["N"])
base_dx = float(base_config["parameters"]["dx"])

for mult in domain_multipliers:
    N_scaled = int(base_N * mult)
    
    print(f"\n{'='*70}")
    print(f"Domain Size: {mult}x (N={N_scaled}, L={N_scaled * base_dx:.2f})")
    print(f"{'='*70}")
    
    for boundary_type in ["mur", "pml"]:
        print(f"\n--- Running with {boundary_type.upper()} boundary ---")
        
        # Load fresh config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Find EM-21 test
        em21_test = None
        for t in config.get("tests", []):
            if t.get("id") == "EM-21":
                em21_test = t
                break
        
        if not em21_test:
            print("EM-21 not found")
            continue
        
        # Scale domain size
        config["parameters"]["N"] = N_scaled
        
        # Update boundary config
        em21_test["config"]["boundary"] = boundary_type
        if boundary_type == "pml":
            # Scale PML cells proportionally but cap at reasonable value
            pml_cells = min(32, int(16 * mult))
            em21_test["config"]["pml_cells"] = pml_cells
            em21_test["config"]["pml_sigma_max"] = 1.0
        
        # Gate region stays at same physical location
        # gate_center_frac and gate_width_frac are fractions, so they auto-scale
        
        # Run test
        harness = Tier5ElectromagneticHarness(str(config_path))
        harness.config = config
        res = harness.run_test(em21_test)
        
        # Extract metrics
        delta_t = res.metrics.get("delta_t_measured", float('nan'))
        expected = res.metrics.get("expected_delay", float('nan'))
        rel_err = res.metrics.get("relative_error", float('nan'))
        energy_resid = res.metrics.get("energy_residuals", {})
        L_inf = energy_resid.get("mod_L_inf", float('nan'))
        
        result = {
            'multiplier': mult,
            'N': N_scaled,
            'domain_length': N_scaled * base_dx,
            'delta_t_measured': delta_t,
            'expected_delay': expected,
            'relative_error': rel_err,
            'energy_L_inf': L_inf,
            'passed': res.passed,
            'boundary': boundary_type
        }
        
        if boundary_type == "mur":
            results_mur.append(result)
        else:
            results_pml.append(result)
        
        print(f"  Measured Δt: {delta_t:.4e} s")
        print(f"  Expected:    {expected:.4e} s")
        print(f"  Rel Error:   {rel_err:.3%}")
        print(f"  Energy L∞:   {L_inf:.3e}")
        print(f"  Passed:      {res.passed}")

# Analysis and plotting
print(f"\n{'='*70}")
print("CONVERGENCE ANALYSIS")
print(f"{'='*70}")

# Check convergence using Richardson extrapolation
def richardson_extrapolate(sizes, values, order=2):
    """Extrapolate to infinite domain using Richardson extrapolation"""
    if len(sizes) < 2:
        return float('nan'), float('nan')
    
    # Use last two points for extrapolation
    L1, L2 = sizes[-2], sizes[-1]
    v1, v2 = values[-2], values[-1]
    
    if L2 <= L1:
        return float('nan'), float('nan')
    
    r = L2 / L1  # refinement ratio
    # v_inf = (r^p * v2 - v1) / (r^p - 1)
    denom = r**order - 1
    if abs(denom) < 1e-12:
        return float('nan'), float('nan')
    
    v_inf = (r**order * v2 - v1) / denom
    error_est = abs(v2 - v1) / denom
    
    return v_inf, error_est

mur_deltas = [r['delta_t_measured'] for r in results_mur]
pml_deltas = [r['delta_t_measured'] for r in results_pml]
mur_sizes = [r['domain_length'] for r in results_mur]
pml_sizes = [r['domain_length'] for r in results_pml]

mur_inf, mur_err = richardson_extrapolate(mur_sizes, mur_deltas)
pml_inf, pml_err = richardson_extrapolate(pml_sizes, pml_deltas)

print(f"\nMur boundary:")
print(f"  Largest domain: Δt = {mur_deltas[-1]:.4e} s")
print(f"  Extrapolated:   Δt = {mur_inf:.4e} ± {mur_err:.4e} s")
print(f"  Convergence:    {abs(mur_deltas[-1] - mur_inf)/abs(mur_inf):.2%} from infinite")

print(f"\nPML boundary:")
print(f"  Largest domain: Δt = {pml_deltas[-1]:.4e} s")
print(f"  Extrapolated:   Δt = {pml_inf:.4e} ± {pml_err:.4e} s")
print(f"  Convergence:    {abs(pml_deltas[-1] - pml_inf)/abs(pml_inf):.2%} from infinite")

print(f"\nBoundary agreement:")
if np.isfinite(mur_inf) and np.isfinite(pml_inf):
    boundary_agreement = abs(mur_inf - pml_inf) / abs(mur_inf) * 100
    print(f"  Extrapolated Mur vs PML: {boundary_agreement:.2f}% difference")
    if boundary_agreement < 2.0:
        print(f"  ✓ PASS: Boundary independence certified (<2%)")
    elif boundary_agreement < 5.0:
        print(f"  ⚠ MARGINAL: Boundary dependence {boundary_agreement:.1f}% (target <2%)")
    else:
        print(f"  ✗ FAIL: Significant boundary dependence (target <2%)")
else:
    print(f"  Cannot compute - insufficient data")

# Save results
all_results = results_mur + results_pml
results_file = output_dir / "domain_convergence_results.json"
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump({
        'domain_multipliers': domain_multipliers,
        'base_N': base_N,
        'base_dx': base_dx,
        'results_mur': results_mur,
        'results_pml': results_pml,
        'extrapolated_mur': float(mur_inf) if np.isfinite(mur_inf) else None,
        'extrapolated_pml': float(pml_inf) if np.isfinite(pml_inf) else None,
        'extrapolation_error_mur': float(mur_err) if np.isfinite(mur_err) else None,
        'extrapolation_error_pml': float(pml_err) if np.isfinite(pml_err) else None
    }, f, indent=2)
print(f"\n✓ Saved results to {results_file}")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Measured delay vs domain size
ax = axes[0, 0]
ax.plot(mur_sizes, mur_deltas, 'o-', label='Mur', linewidth=2, markersize=8)
ax.plot(pml_sizes, pml_deltas, 's-', label='PML', linewidth=2, markersize=8)
if np.isfinite(mur_inf):
    ax.axhline(mur_inf, color='blue', linestyle='--', alpha=0.5, label=f'Mur extrapolated')
if np.isfinite(pml_inf):
    ax.axhline(pml_inf, color='orange', linestyle='--', alpha=0.5, label=f'PML extrapolated')
ax.set_xlabel('Domain Length')
ax.set_ylabel('Measured Δt (s)')
ax.set_title('Domain Size Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Relative error vs domain size
ax = axes[0, 1]
mur_rel_errs = [r['relative_error'] * 100 for r in results_mur]
pml_rel_errs = [r['relative_error'] * 100 for r in results_pml]
ax.plot(mur_sizes, mur_rel_errs, 'o-', label='Mur', linewidth=2, markersize=8)
ax.plot(pml_sizes, pml_rel_errs, 's-', label='PML', linewidth=2, markersize=8)
ax.set_xlabel('Domain Length')
ax.set_ylabel('Relative Error (%)')
ax.set_title('Measurement Error vs Domain Size')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Energy residual vs domain size
ax = axes[1, 0]
mur_energy = [r['energy_L_inf'] for r in results_mur]
pml_energy = [r['energy_L_inf'] for r in results_pml]
ax.plot(mur_sizes, mur_energy, 'o-', label='Mur', linewidth=2, markersize=8)
ax.plot(pml_sizes, pml_energy, 's-', label='PML', linewidth=2, markersize=8)
ax.set_xlabel('Domain Length')
ax.set_ylabel('Energy Residual L∞')
ax.set_title('Energy Conservation vs Domain Size')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Convergence ratio
ax = axes[1, 1]
if len(mur_deltas) >= 2:
    mur_conv = [abs(mur_deltas[i] - mur_deltas[i-1]) / abs(mur_deltas[i-1]) for i in range(1, len(mur_deltas))]
    ax.semilogy(mur_sizes[1:], mur_conv, 'o-', label='Mur', linewidth=2, markersize=8)
if len(pml_deltas) >= 2:
    pml_conv = [abs(pml_deltas[i] - pml_deltas[i-1]) / abs(pml_deltas[i-1]) for i in range(1, len(pml_deltas))]
    ax.semilogy(pml_sizes[1:], pml_conv, 's-', label='PML', linewidth=2, markersize=8)
ax.axhline(0.02, color='g', linestyle='--', alpha=0.5, label='2% target')
ax.set_xlabel('Domain Length')
ax.set_ylabel('Relative Change')
ax.set_title('Convergence Rate (successive differences)')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plot_file = output_dir / "domain_convergence_plots.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved plots to {plot_file}")
plt.close()

print(f"\n{'='*70}")
print("DOMAIN CONVERGENCE STUDY COMPLETE")
print(f"{'='*70}")
