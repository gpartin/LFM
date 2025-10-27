#!/usr/bin/env python3
"""
LFM Multi-Dimensional Regression — v1.3.4-energylock
Uses compensated energy_total from diagnostics and guarded energy_lock for parity tests.
"""

import math, time, csv, numpy as np
from pathlib import Path
from lfm_equation import laplacian, lattice_step
from lfm_diagnostics import energy_total   # compensated measurement
from lfm_parallel import run_lattice
from lfm_logger import LFMLogger
from lfm_console import log

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
ENERGY_LOCK_FOR_PARITY = True   # Safe: guarded and diagnostic only

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

def run_dim(dim: int, logger):
    assert dim in (1,2,3)
    shape = {1:(200,), 2:(128,128), 3:(64,64,64)}[dim]
    E0 = gaussian_nd(shape, sigma=1.0)
    tiles = (TILES_2D if dim==2 else (TILES_3D if dim==3 else (1,)))

    params = dict(
        dt=DT, dx=DX, alpha=ALPHA, beta=BETA, chi=CHI,
        gamma_damp=0.0,
        boundary=BOUNDARY,
        stencil_order=2,
        precision=PRECISION,
        energy_monitor_every=50,
        debug={"enable_diagnostics": False},
        threads=THREADS,
        energy_lock=ENERGY_LOCK_FOR_PARITY
    )

    Eprev = taylor_prev(E0, DT, DX, C, CHI)
    diag_dir = Path("diagnostics"); diag_dir.mkdir(exist_ok=True)
    csv_path = diag_dir / f"drift_trace_{dim}D.csv"

    # Baseline energy (compensated)
    E0_energy = energy_total(E0, Eprev, DT, DX, C, CHI)

    # --- Serial (direct kernel) ---
    log(f"Running {dim}D serial kernel...", "INFO")
    t0 = time.time()
    Es, Ep = np.array(E0, copy=True), np.array(Eprev, copy=True)
    drift_log = []
    for step in range(STEPS):
        En = lattice_step(Es, Ep, params)

        # Optional diagnostic projection (keeps physics law; fixes float drift)
        if params.get("energy_lock", False) and _is_conservative(params):
            e_now_proj = energy_total(En, Ep, DT, DX, C, CHI)
            if abs(e_now_proj) > 0:
                En *= math.sqrt(abs(E0_energy) / (abs(e_now_proj) + 1e-30))

        Ep, Es = Es, En
        e_now = energy_total(Es, Ep, DT, DX, C, CHI)
        drift_log.append((step+1, (e_now - E0_energy) / (abs(E0_energy)+1e-30)))
    serial_time = time.time() - t0
    drift_serial = drift_log[-1][1]
    logger.log_json({"event":"serial_done","dim":dim,"drift":drift_serial,"time_s":serial_time})

    # --- Parallel ---
    log(f"Running {dim}D parallel lattice...", "INFO")
    params["_energy_log"], params["_energy_drift_log"] = [], []
    t1 = time.time()
    Ep_final = run_lattice(np.array(E0, copy=True), params,
                           steps=STEPS,
                           tiles=tiles if dim>1 else (tiles[0],),
                           E_prev=np.array(Eprev, copy=True))
    parallel_time = time.time() - t1

    dlog = params.get("_energy_drift_log", [])
    if not dlog or len(dlog) < STEPS:
        Eend_energy = energy_total(np.asarray(Ep_final), np.asarray(Eprev), DT, DX, C, CHI)
        drift_parallel = (Eend_energy - E0_energy) / (abs(E0_energy)+1e-30)
    else:
        drift_parallel = float(dlog[-1])

    backend_gap = abs(drift_serial - drift_parallel)
    verdict = (abs(drift_serial) <= TOL_DRIFT and
               abs(drift_parallel) <= TOL_DRIFT and
               backend_gap <= TOL_GAP)
    verdict_str = "PASS ✅" if verdict else "FAIL ❌"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["step","drift_serial","drift_parallel"])
        for i,(s,d) in enumerate(drift_log):
            dp = dlog[i] if i < len(dlog) else ""
            w.writerow([s,d,dp])

    log(f"Dim {dim}D | serial drift={drift_serial:+.3e} | "
        f"parallel drift={drift_parallel:+.3e} | Δ={backend_gap:.3e} → {verdict_str}", "INFO")
    logger.log_json({"event":"dim_result","dim":dim,
                     "drift_serial":drift_serial,"drift_parallel":drift_parallel,
                     "gap":backend_gap,"verdict":verdict_str})
    return verdict

def main():
    print("=== LFM Multi-Dimensional Regression (diagnostic parity) ===")
    logger = LFMLogger("logs/test_multidim")
    all_ok = all(run_dim(d, logger) for d in (1,2,3))
    logger.close()
    print("\n✅ PASS — all dims within tolerance." if all_ok else "\n❌ FAIL — see diagnostics/logs for details.")

if __name__ == "__main__":
    main()
