#!/usr/bin/env python3
"""
Unified Validation Harness for LFM Core (v1.4) + Parallel (v1.3.1)

Runs:
  1️⃣  2-D Serial (no parallel)
  2️⃣  2-D Threaded (parallel)
  3️⃣  3-D Serial
  4️⃣  3-D Threaded

Outputs:
  • diagnostics_2d_serial.csv
  • diagnostics_2d_parallel.csv
  • diagnostics_3d_serial.csv
  • diagnostics_3d_parallel.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import time, json, os
from lfm_equation import advance
from lfm_parallel import run_lattice


# ---------------------------------------------------------------------
# Simple inline logger
# ---------------------------------------------------------------------
class SimpleLogger:
    def __init__(self):
        self.logs = []
    def log(self, msg): print(msg)
    def log_json(self, obj): print(json.dumps(obj))


# ---------------------------------------------------------------------
# Utility: initialize Gaussian field
# ---------------------------------------------------------------------
def make_field(shape, sigma=0.5):
    ndim = len(shape)
    grids = np.meshgrid(*[np.linspace(-1, 1, n, endpoint=False) for n in shape], indexing="ij")
    r2 = sum(g**2 for g in grids)
    return np.exp(-r2 / (2 * sigma**2))


# ---------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------
def run_case(name, E0, params, parallel=False, tiles=(1, 1, 1)):
    logger = SimpleLogger()
    t0 = time.time()
    if parallel:
        result = run_lattice(E0, params, steps=params["steps"], backend="thread",
                             tiles=tiles, save_every=0, logger=logger)
    else:
        result = advance(E0, params, steps=params["steps"], save_every=0)
    dt = time.time() - t0
    print(f"\n✅ {name} complete in {dt:.3f}s\n")
    return result


# ---------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------
base_params = {
    "alpha": 1.0,
    "beta": 1.0,
    "dx": 0.05,
    "dt": 0.002,
    "gamma_damp": 0.0,
    "stencil_order": 2,
    "boundary": "periodic",
    "chi": 0.0,
    "steps": 100,
    "energy_monitor_every": 10,
    "debug": {
        "enable_diagnostics": True,
        "energy_tol": 1e-4,
        "verbose": True,
        "profile_steps": True
    }
}


# ---------------------------------------------------------------------
# 1️⃣  2-D Serial
# ---------------------------------------------------------------------
E0_2d = make_field((64, 64))
params_2d_serial = dict(base_params)
params_2d_serial["diagnostics_path"] = "diagnostics_2d_serial.csv"
E_final_2d_serial = run_case("2-D SERIAL", E0_2d, params_2d_serial, parallel=False)


# ---------------------------------------------------------------------
# 2️⃣  2-D Parallel
# ---------------------------------------------------------------------
E0_2d = make_field((64, 64))
params_2d_par = dict(base_params)
params_2d_par["diagnostics_path"] = "diagnostics_2d_parallel.csv"
params_2d_par["threads"] = 4
params_2d_par["debug"]["enable_halo_diag"] = True
E_final_2d_par = run_case("2-D PARALLEL", E0_2d, params_2d_par, parallel=True, tiles=(2, 2, 1))


# ---------------------------------------------------------------------
# 3️⃣  3-D Serial
# ---------------------------------------------------------------------
E0_3d = make_field((32, 32, 32))
params_3d_serial = dict(base_params)
params_3d_serial["diagnostics_path"] = "diagnostics_3d_serial.csv"
E_final_3d_serial = run_case("3-D SERIAL", E0_3d, params_3d_serial, parallel=False)


# ---------------------------------------------------------------------
# 4️⃣  3-D Parallel
# ---------------------------------------------------------------------
E0_3d = make_field((32, 32, 32))
params_3d_par = dict(base_params)
params_3d_par["diagnostics_path"] = "diagnostics_3d_parallel.csv"
params_3d_par["threads"] = 4
params_3d_par["debug"]["enable_halo_diag"] = True
E_final_3d_par = run_case("3-D PARALLEL", E0_3d, params_3d_par, parallel=True, tiles=(2, 2, 2))


# ---------------------------------------------------------------------
# Visualization (2-D runs)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(E0_2d, cmap="inferno", origin="lower")
ax[0].set_title("Initial Field (2-D)")
ax[1].imshow(E_final_2d_serial, cmap="inferno", origin="lower")
ax[1].set_title("Final Field (2-D Serial)")
plt.tight_layout()
plt.show()

print("\nAll four test cases executed. Check diagnostics_*.csv for drift/CFL logs.\n")
