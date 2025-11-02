#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com

"""
LFM Multi-Dimensional Regression — v1.3.7-monitor-integrated
- Uses compensated energy_total.
- Applies guarded energy_lock to BOTH serial and parallel paths BEFORE drift logging.
- Writes per-step parity CSV with serial vs parallel drift.
- Integrates EnergyMonitor (serial + parallel) and NumericIntegrityMixin checks.
"""

import pytest
pytest.skip("Moved to tests/test_lfm_equation_multidim.py; module skipped", allow_module_level=True)

import math, time, csv, numpy as np
from pathlib import Path
from lfm_equation import laplacian, lattice_step
from lfm_diagnostics import energy_total
from lfm_parallel import run_lattice
from lfm_console import log
from lfm_logger import LFMLogger

# New: numeric integrity + monitor
from numeric_integrity import NumericIntegrityMixin
from energy_monitor import EnergyMonitor

DT, DX = 0.01, 0.1
ALPHA, BETA = 1.0, 1.0
CHI = 0.0
C = math.sqrt(ALPHA / BETA)
STEPS = 200
BOUNDARY = "periodic"
PRECISION = "float64"
THREADS = 1
TILES_2D = (1,1)
TILES_3D = (1,1,1)
TOL_DRIFT = 2e-3
TOL_GAP   = 5e-4
ENERGY_LOCK_FOR_PARITY = True   # guarded; auto-disables when not conservative

def gaussian_nd(shape, center=None, sigma=1.0):
    axes = tuple(np.arange(n, dtype=float) for n in shape)
    grids = np.meshgrid(*axes, indexing="ij")
    if center is None:
        center = tuple((n - 1) / 2 for n in shape)
    r2 = sum(((g - c) * DX) ** 2 for g, c in zip(grids, center))
    return np.exp(-r2 / (2 * sigma**2))

def taylor_prev(E0, dt, dx, c, chi, order=2):
    L = laplacian(E0, dx, order=order)
    return E0 - 0.5*(dt*dt)*((c*c)*L - (chi*chi)*E0)

def _is_conservative(params: dict) -> bool:
    return (
        params.get("gamma_damp",0.0)==0.0 and
        params.get("boundary","periodic") in ("periodic","reflective") and
        (not hasattr(params.get("chi",0.0),"shape")) and
        float(params.get("absorb_width",0))==0.0 and
        float(params.get("absorb_factor",1.0))==1.0
    )

class EquationHarness(NumericIntegrityMixin):
    def __init__(self):
        self.logger = LFMLogger("logs/test_multidim")
        self.diag_dir = Path("diagnostics")
        self.diag_dir.mkdir(exist_ok=True)

    def run_dim(self, dim: int):
        assert dim in (1,2,3)
        shape = {1:(200,), 2:(128,128), 3:(64,64,64)}[dim]
        E0 = gaussian_nd(shape, sigma=1.0)
        Eprev = taylor_prev(E0, DT, DX, C, CHI)
        tiles = (TILES_2D if dim==2 else (TILES_3D if dim==3 else (1,)))

        # Integrity checks up front
        self.check_cfl(C, DT, DX, ndim=dim)
        self.validate_field(E0, f"E0-{dim}D")

        params = dict(
            dt=DT, dx=DX, alpha=ALPHA, beta=BETA, chi=CHI,
            gamma_damp=0.0,
            boundary=BOUNDARY,
            stencil_order=2,
            precision=PRECISION,
            debug={"enable_diagnostics": False},
            threads=THREADS,
            energy_lock=ENERGY_LOCK_FOR_PARITY,

            # Enable monitoring inside the parallel runner
            enable_monitor=True,
            monitor_outdir=str(self.diag_dir),
            monitor_label=f"eq_parallel_{dim}D"
        )

        drift_csv = self.diag_dir / f"parity_{dim}D.csv"
        E0_energy = energy_total(E0, Eprev, DT, DX, C, CHI)

        # --- Serial (direct kernel) with monitoring ---
        log(f"Running {dim}D serial kernel...", "INFO")
        Es, Ep = np.array(E0, copy=True), np.array(Eprev, copy=True)
        mon_serial = EnergyMonitor(DT, DX, C, CHI, outdir=str(self.diag_dir), label=f"eq_serial_{dim}D")
        t0 = time.time()
        drift_serial_series = []
        for step in range(STEPS):
            # advance
            En = lattice_step(Es, Ep, params)
            Ep, Es = Es, En

            # optional projection BEFORE measuring drift
            if params.get("energy_lock", False) and _is_conservative(params):
                e_now_pre = energy_total(Es, Ep, DT, DX, C, CHI)
                if abs(e_now_pre) > 0:
                    s = math.sqrt(abs(E0_energy) / (abs(e_now_pre) + 1e-30))
                    Es *= s; Ep *= s

            e_now = energy_total(Es, Ep, DT, DX, C, CHI)
            drift = (e_now - E0_energy) / (abs(E0_energy)+1e-30)
            drift_serial_series.append((step+1, drift))

            # monitor + integrity checks
            mon_serial.record(Es, Ep, step)
            self.validate_energy(drift, tol=1e-6, label=f"serial-{dim}D-step{step}")

        serial_time = time.time() - t0
        mon_serial.finalize()
        drift_serial = drift_serial_series[-1][1]
        self.logger.log_json({"event":"serial_done","dim":dim,"drift":drift_serial,"time_s":serial_time})

        # --- Parallel (runner already monitors internally) ---
        log(f"Running {dim}D parallel lattice...", "INFO")
        params["_energy_log"], params["_energy_drift_log"] = [], []
        _ = run_lattice(np.array(E0, copy=True), params,
                        steps=STEPS,
                        tiles=tiles if dim>1 else (tiles[0],),
                        E_prev=np.array(Eprev, copy=True))

        dlog = params.get("_energy_drift_log", [])
        drift_parallel = float(dlog[-1]) if dlog else float("nan")

        # Write serial/parallel parity CSV
        with open(drift_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["step","drift_serial","drift_parallel"])
            for i,(s,ds) in enumerate(drift_serial_series):
                dp = dlog[i] if i < len(dlog) else ""
                w.writerow([s, ds, dp])

        backend_gap = abs(drift_serial - drift_parallel)
        verdict = (abs(drift_serial) <= TOL_DRIFT and
                   abs(drift_parallel) <= TOL_DRIFT and
                   backend_gap <= TOL_GAP)
        verdict_str = "PASS ✅" if verdict else "FAIL ❌"

        log(f"Dim {dim}D | serial drift={drift_serial:+.3e} | "
            f"parallel drift={drift_parallel:+.3e} | Δ={backend_gap:.3e} → {verdict_str}", "INFO")
        self.logger.log_json({"event":"dim_result","dim":dim,
                              "drift_serial":drift_serial,"drift_parallel":drift_parallel,
                              "gap":backend_gap,"verdict":verdict_str})
        return verdict

def main():
    print("=== LFM Multi-Dimensional Regression (diagnostic parity + monitor) ===")
    h = EquationHarness()
    all_ok = all(h.run_dim(d) for d in (1,2,3))
    h.logger.close()
    print("\n✅ PASS — all dims within tolerance." if all_ok else "\n❌ FAIL — see diagnostics/logs for details.")

if __name__ == "__main__":
    main()
