# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
EM-21 PML Parameter Sweep
Sweep over PML parameters and compare boundary sensitivity.
"""
import json
import sys
from pathlib import Path
# Bootstrap to import path_utils from the workspace root
_WS_DIR = Path(__file__).resolve().parents[1]
if str(_WS_DIR) not in sys.path:
    sys.path.insert(0, str(_WS_DIR))
from path_utils import add_workspace_to_sys_path, get_workspace_dir
import pandas as pd
import matplotlib.pyplot as plt

WS = add_workspace_to_sys_path(__file__)
from run_tier5_electromagnetic import Tier5ElectromagneticHarness

config_path = WS / "config" / "config_tier5_electromagnetic.json"
output_dir = WS / "results" / "Electromagnetic" / "EM-21"
output_dir.mkdir(parents=True, exist_ok=True)

# Sweep parameters
pml_cells_values = [8, 16, 32]
pml_sigma_max_values = [0.5, 1.0, 2.0]

results = []

for pml_cells in pml_cells_values:
    for pml_sigma_max in pml_sigma_max_values:
        print(f"\n=== Running PML: cells={pml_cells}, sigma_max={pml_sigma_max} ===")
        
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
            print("EM-21 not found in config")
            continue
        
        # Update PML params and enable boundary comparison
        em21_test["config"]["boundary"] = "pml"
        em21_test["config"]["pml_cells"] = pml_cells
        em21_test["config"]["pml_sigma_max"] = pml_sigma_max
        em21_test["config"]["boundary_ab_compare"] = True
        
        # Run test
        harness = Tier5ElectromagneticHarness(str(config_path))
        harness.config = config  # Use modified config
        res = harness.run_test(em21_test)
        
        # Extract metrics
        delta_t_primary = res.metrics.get("delta_t_measured", float('nan'))
        boundary_compare = res.metrics.get("boundary_ab_compare", {})
        delta_t_alt = boundary_compare.get("delta_t_alt", float('nan'))
        abs_delta = boundary_compare.get("abs_delta", float('nan'))
        rel_delta = boundary_compare.get("rel_delta", float('nan'))
        energy_resid = res.metrics.get("energy_residuals", {})
        mod_L_inf = energy_resid.get("mod_L_inf", float('nan'))
        mod_L2 = energy_resid.get("mod_L2", float('nan'))
        
        results.append({
            'pml_cells': pml_cells,
            'pml_sigma_max': pml_sigma_max,
            'delta_t_primary_PML': delta_t_primary,
            'delta_t_alt_Mur': delta_t_alt,
            'abs_delta': abs_delta,
            'rel_delta': rel_delta,
            'mod_L_inf': mod_L_inf,
            'mod_L2': mod_L2,
            'passed': res.passed
        })
        
        print(f"  Primary (PML): {delta_t_primary:.4e}")
        print(f"  Alt (Mur): {delta_t_alt:.4e}")
        print(f"  Abs delta: {abs_delta:.4e}")
        print(f"  Rel delta: {rel_delta:.3%}")
        print(f"  Energy L_inf: {mod_L_inf:.3e}")

# Save results
df = pd.DataFrame(results)
csv_path = output_dir / "pml_sweep_results.csv"
df.to_csv(csv_path, index=False, encoding='utf-8')
print(f"\n✓ Saved results to {csv_path}")
print(df.to_string(index=False))

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for sigma_max in pml_sigma_max_values:
    subset = df[df['pml_sigma_max'] == sigma_max]
    axes[0,0].plot(subset['pml_cells'], subset['delta_t_primary_PML'], 'o-', label=f'σ_max={sigma_max}')
    axes[0,1].plot(subset['pml_cells'], subset['abs_delta'], 'o-', label=f'σ_max={sigma_max}')
    axes[1,0].plot(subset['pml_cells'], subset['rel_delta'], 'o-', label=f'σ_max={sigma_max}')
    axes[1,1].plot(subset['pml_cells'], subset['mod_L_inf'], 'o-', label=f'σ_max={sigma_max}')

axes[0,0].set_xlabel('PML cells')
axes[0,0].set_ylabel('Δt (PML) [s]')
axes[0,0].set_title('Primary measurement (PML boundary)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].set_xlabel('PML cells')
axes[0,1].set_ylabel('|Δt_PML - Δt_Mur| [s]')
axes[0,1].set_title('Boundary sensitivity (absolute)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

axes[1,0].set_xlabel('PML cells')
axes[1,0].set_ylabel('Relative delta')
axes[1,0].set_title('Boundary sensitivity (relative)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

axes[1,1].set_xlabel('PML cells')
axes[1,1].set_ylabel('Energy residual L∞')
axes[1,1].set_title('Energy conservation (PML)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = output_dir / "pml_sweep_plots.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved plots to {plot_path}")
plt.close()

print("\n=== Summary ===")
print(f"Tested {len(results)} PML configurations")
print(f"Mean abs boundary delta: {df['abs_delta'].mean():.3e} s")
print(f"Mean rel boundary delta: {df['rel_delta'].mean():.3%}")
print(f"All tests passed: {df['passed'].all()}")
