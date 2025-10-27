#!/usr/bin/env python3
"""
Quick self-test for lfm_equation.py (v1.3)
Runs a small 1D lattice pulse with diagnostics enabled.
"""

import numpy as np
from lfm_equation import advance

# --- Parameters --------------------------------------------------------
params = {
    "dt": 0.01,
    "dx": 0.1,
    "alpha": 1.0,
    "beta": 1.0,
    "chi": 0.0,
    "gamma_damp": 0.0,
    "boundary": "periodic",
    "stencil_order": 2,
    "energy_monitor_every": 5,
    "debug": {
        "enable_diagnostics": True,
        "energy_tol": 1e-4,
        "profile_steps": 50
    }
}

# --- Initial field: a small Gaussian pulse -----------------------------
nx = 200
x = np.linspace(-10, 10, nx)
E0 = np.exp(-x**2)
steps = 200

# --- Run ---------------------------------------------------------------
print("Running core LFM solver test...")
E_final = advance(E0, params, steps=steps, save_every=0)

# --- Optional: visualize field evolution -----------------------------
import matplotlib.pyplot as plt

# Reload initial field and final field for comparison
x = np.linspace(-10, 10, len(E0))

plt.figure(figsize=(8, 4))
plt.plot(x, E0, label="Initial field", lw=1.5)
plt.plot(x, np.array(E_final), label="Final field", lw=1.2, color="orange")
plt.xlabel("x")
plt.ylabel("E amplitude")
plt.title("LFM Core Equation — Field Propagation (1D Test)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nRun complete.")
print("Last diagnostics:")
print(params.get("_last_diagnostics", {}))
print("\nDiagnostics saved to:", params["debug"].get("diagnostics_path", "diagnostics_core.csv"))