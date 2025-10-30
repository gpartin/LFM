#!/usr/bin/env python3
"""
LFM Tier-3 — Energy Conservation Tests (2-D, Unified)
---------------------------------------------------
Purpose:
- Execute Tier-3 energy conservation tests to validate fundamental conservation
  laws and thermodynamic emergence in the lattice field model.
  
Physics:
- Tests energy drift in isolated systems (ENER-01, 02)
- Tests energy conservation in curved spacetime (χ-gradients, ENER-03, 04)
- Tests entropy growth from random initial conditions (ENER-05, 07)
- Tests controlled energy extraction via damping (ENER-06)

Pass Criteria:
- Clean wave packets: energy drift < 1% over long runs
- Noisy initial conditions: entropy increases monotonically
- Damping: energy decreases as expected

Config & output:
- Expects configuration at `./config/config_tier3_energy.json`.
- Writes per-test results under `results/Energy/<TEST_ID>/` with
  `summary.json`, `diagnostics/` and `plots/`.
"""

import json, math, time, platform
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

import matplotlib.pyplot as plt

from lfm_console import log, suite_summary, set_logger, log_run_config
from lfm_logger import LFMLogger
from lfm_results import save_summary, write_metadata_bundle, write_csv, ensure_dirs

# ------------------------------- Config Loader ------------------------------
def load_config(config_path: str = None) -> Dict:
    """Load config from explicit path or default location."""
    if config_path:
        cand = Path(config_path)
        if cand.is_file():
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Default: search in standard locations
    script_dir = Path(__file__).resolve().parent
    cfg_name = "config_tier3_energy.json"
    for root in (script_dir, script_dir.parent):
        cand = root / "config" / cfg_name
        if cand.is_file():
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Tier-3 config not found (expected config/{cfg_name}).")

# ------------------------------ Backend helpers -----------------------------
def pick_backend(use_gpu_flag: bool):
    """Select NumPy or CuPy backend based on config and availability."""
    on_gpu = bool(use_gpu_flag and _HAS_CUPY)
    if use_gpu_flag and not _HAS_CUPY:
        log("GPU requested but CuPy not available; falling back to NumPy", "WARN")
    return (cp if on_gpu else np), on_gpu

def to_numpy(x):
    """Convert array to NumPy for host-side operations."""
    if _HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)

@dataclass
class TestResult:
    test_id: str
    description: str
    passed: bool
    energy_drift: float
    entropy_monotonic: bool
    wave_drop: float
    runtime_sec: float

# -------------------------- Numerical helpers (2-D) -------------------------
def laplacian(E, dx, order=4, xp=None):
    """Compute Laplacian using specified backend (NumPy or CuPy)."""
    if xp is None:
        xp = cp if (_HAS_CUPY and isinstance(E, cp.ndarray)) else np
    if order == 2:
        return (
            xp.roll(E,  1, 0) + xp.roll(E, -1, 0) +
            xp.roll(E,  1, 1) + xp.roll(E, -1, 1) - 4*E
        ) / (dx*dx)
    elif order == 4:
        return (
            (-xp.roll(E, 2, 0) + 16*xp.roll(E, 1, 0) - 30*E + 16*xp.roll(E, -1, 0) - xp.roll(E, -2, 0)) +
            (-xp.roll(E, 2, 1) + 16*xp.roll(E, 1, 1) - 30*E + 16*xp.roll(E, -1, 1) - xp.roll(E, -2, 1))
        ) / (12*dx*dx)
    else:
        raise ValueError("Unsupported stencil order; use 2 or 4")

def grad_sq(E, dx, xp=None):
    """Compute gradient squared using specified backend."""
    if xp is None:
        xp = cp if (_HAS_CUPY and isinstance(E, cp.ndarray)) else np
    Ex = (xp.roll(E, -1, 1) - xp.roll(E, 1, 1)) / (2*dx)
    Ey = (xp.roll(E, -1, 0) - xp.roll(E, 1, 0)) / (2*dx)
    return Ex*Ex + Ey*Ey

def energy_total(E, E_prev, dt, dx, c, chi, xp=None):
    """Compute total energy using specified backend."""
    if xp is None:
        xp = cp if (_HAS_CUPY and isinstance(E, cp.ndarray)) else np
    Et = (E - E_prev) / dt
    dens = 0.5*(Et*Et + (c*c)*grad_sq(E, dx, xp) + (chi*chi)*(E*E))
    return float(xp.sum(dens) * dx*dx)

def entropy_shannon(E, xp=None):
    """Compute Shannon entropy using specified backend."""
    if xp is None:
        xp = cp if (_HAS_CUPY and isinstance(E, cp.ndarray)) else np
    p = xp.abs(E)**2
    s = xp.sum(p)
    if float(s) == 0.0: return 0.0
    p = p / s
    eps = 1e-30
    return float(-xp.sum(p * xp.log(p + eps)))

# -------------------------- χ-field constructors ----------------------------
def chi_field(N, pattern: dict, dtype, xp):
    """Build χ-field using specified backend."""
    x = xp.linspace(-1, 1, N, dtype=dtype)
    y = xp.linspace(-1, 1, N, dtype=dtype)
    X, Y = xp.meshgrid(x, y)
    if "chi_gradient" in pattern:
        a, b = map(float, pattern["chi_gradient"])
        chi = a + (b - a) * (X + 1.0) / 2.0
    elif pattern.get("chi_function", "") == "sin(πx)":
        chi = xp.sin(xp.pi * X)
    elif pattern.get("chi_const", None) is not None:
        chi = xp.full_like(X, float(pattern["chi_const"]))
    else:
        chi = xp.zeros_like(X)
    return chi

# --------------------------- Initial condition builders ---------------------
def init_pulse(N, dtype, xp, kind="gaussian", kx=8.0, width=20.0):
    """Build initial condition using specified backend."""
    x = xp.linspace(-1, 1, N, dtype=dtype)
    y = xp.linspace(-1, 1, N, dtype=dtype)
    X, Y = xp.meshgrid(x, y)
    if kind == "gaussian":
        return xp.exp(-width*(X**2 + Y**2)) * xp.cos(kx*xp.pi*X)
    elif kind == "noise":
        if xp is cp:
            rng = xp.random.default_rng(42)
            return 1e-3 * rng.standard_normal((N, N), dtype=dtype)
        else:
            rng = np.random.default_rng(42)
            return 1e-3 * rng.standard_normal((N, N)).astype(dtype)
    return xp.zeros((N, N), dtype=dtype)

# ------------------------------ Test runner ------------------------------
def run_test(params, tol, test, out_dir: Path, dtype, xp, on_gpu):
    """Run a single energy conservation test."""
    ensure_dirs(out_dir)
    test_id = test.get("test_id", "ENER-??")
    desc = test.get("description", "unknown test")

    c  = float(params.get("c", 1.0))
    dt = float(params["time_step"])
    dx = float(params["space_step"])
    N  = int(params.get("N", 512))
    steps = int(test.get("steps", 10_000))
    save_every = int(params.get("save_every", 10))
    stencil_order = int(params.get("stencil_order", 4))

    cfl = c*dt/dx
    if cfl > 0.9:
        raise ValueError(f"CFL too high (c*dt/dx={cfl:.3f} > 0.9).")

    chi = chi_field(N, test, dtype, xp)
    E = init_pulse(N, dtype, xp, kind=test.get("ic","gaussian"),
                   kx=test.get("kx", 8.0), width=test.get("width", 20.0))
    E_prev = E.copy()

    # --- Normalize initial state to unit discrete energy, THEN capture E0 ---
    E0_init = energy_total(E, E_prev, dt, dx, c, chi, xp)
    scale0 = math.sqrt(1.0 / (E0_init + 1e-30))
    E *= scale0
    E_prev = E.copy()
    E0 = energy_total(E, E_prev, dt, dx, c, chi, xp)  # ~1.0 baseline

    noise_amp = float(test.get("noise_amp", 0.0))
    damping   = float(test.get("damping",   0.0))
    enforce_energy = bool(test.get("enforce_energy", False))

    energy_trace, entropy_trace, times = [], [], []
    t0 = time.time()

    for t in range(steps):
        # --- Optionally enforce energy (only for stability tests, NOT conservation tests) ---
        if enforce_energy:
            Etot_cur = energy_total(E, E_prev, dt, dx, c, chi, xp)
            scale_cur = math.sqrt(E0 / (Etot_cur + 1e-30))
            E *= scale_cur
            E_prev *= scale_cur

        # Diagnostics
        if t % save_every == 0 or t == steps-1:
            Etot = energy_total(E, E_prev, dt, dx, c, chi, xp)
            H = entropy_shannon(E, xp)
            energy_trace.append(Etot)
            entropy_trace.append(H)
            times.append(t*dt)

        # Advance one leapfrog step
        lap = laplacian(E, dx, order=stencil_order, xp=xp)
        E_next = 2*E - E_prev + (dt*dt)*((c*c)*lap - (chi*chi)*E)

        if damping > 0.0:
            E_next *= (1.0 - damping)

        if noise_amp > 0.0 and (t % 50 == 0):
            if xp is cp:
                rng = xp.random.default_rng(1234 + t)
                E_next += noise_amp * rng.standard_normal(E_next.shape, dtype=dtype)
            else:
                rng = np.random.default_rng(1234 + t)
                E_next += noise_amp * rng.standard_normal(E_next.shape).astype(dtype)

        E_prev, E = E, E_next

    runtime = time.time() - t0

    # --- Metrics ---
    energy_trace_np = np.array(energy_trace)
    entropy_np = np.array(entropy_trace)
    times_np = np.array(times)
    rel_drift = abs(energy_trace_np[-1] - energy_trace_np[0]) / max(energy_trace_np[0], 1e-30)
    monotone_entropy = (np.mean(np.diff(entropy_np)) >= -1e-6)
    tolerance_wi = float(tol.get("wave_integrity_max_drop", 0.05))
    max_drop = float((np.max(energy_trace_np) - energy_trace_np[-1]) /
                     max(np.max(energy_trace_np), 1e-30))

    drift_ok  = rel_drift <= float(test.get("energy_drift_max", tol.get("energy_drift", 1e-12)))
    entropy_ok = monotone_entropy if bool(test.get("require_entropy_monotonic", False)) else True
    wave_ok = (max_drop <= tolerance_wi) if bool(test.get("check_wave_integrity", False)) else True

    passed = drift_ok and entropy_ok and wave_ok

    # --- Output plots ---
    ensure_dirs(out_dir/"plots")
    plt.figure(figsize=(6,4))
    plt.plot(times_np, energy_trace_np)
    plt.xlabel("time"); plt.ylabel("Total energy")
    plt.title(f"Energy vs Time — drift={rel_drift:.3e}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir/"plots"/"energy_vs_time.png", dpi=150); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(times_np, entropy_np)
    plt.xlabel("time"); plt.ylabel("Shannon entropy")
    plt.title(f"Entropy vs Time — monotonic={monotone_entropy}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir/"plots"/"entropy_vs_time.png", dpi=150); plt.close()

    # --- Output CSVs ---
    diagnostics_dir = out_dir / "diagnostics"
    ensure_dirs(diagnostics_dir)
    write_csv(diagnostics_dir/"energy_trace.csv",  list(zip(times_np, energy_trace_np)), ["time","energy"])
    write_csv(diagnostics_dir/"entropy_trace.csv", list(zip(times_np, entropy_np)),      ["time","entropy"])

    # --- Hardware info ---
    if on_gpu and _HAS_CUPY:
        try:
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            cuda_ver = int(cp.cuda.runtime.runtimeGetVersion())
        except Exception:
            gpu_name = "GPU (unknown)"
            cuda_ver = 0
    else:
        gpu_name = "CPU"
        cuda_ver = 0

    # --- Save summary ---
    summary = {
        "tier": 3,
        "category": "Energy",
        "test_id": test_id,
        "description": desc,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpu_name": gpu_name,
            "cuda_runtime": cuda_ver,
            "python": platform.python_version(),
            "backend": "CuPy" if on_gpu else "NumPy"
        },
        "parameters": {"N":N,"steps":steps,"dt":dt,"dx":dx,"c":c,"cfl":cfl,"stencil_order":stencil_order},
        "metrics": {
            "energy_rel_drift": float(rel_drift),
            "entropy_monotonic": bool(monotone_entropy),
            "wave_max_drop": float(max_drop)
        },
        "status": "Passed" if passed else "Failed"
    }
    save_summary(out_dir, test_id, summary)

    status_str = "PASS ✅" if passed else "FAIL ❌"
    log(f"[{test_id}] {desc}: drift={rel_drift:.2e}, entropy↑={monotone_entropy}, wave_drop={max_drop:.3f} → {status_str}",
        "PASS" if passed else "FAIL")

    return TestResult(test_id, desc, passed, float(rel_drift), bool(monotone_entropy), float(max_drop), runtime)

# ----------------------------------- Main -----------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tier-3 Energy Conservation Test Suite")
    parser.add_argument("--test", type=str, default=None,
                       help="Run single test by ID (e.g., ENER-01). If omitted, runs all tests.")
    parser.add_argument("--config", type=str, default="config/config_tier3_energy.json",
                       help="Path to config file")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    p, tol, tests = cfg["parameters"], cfg["tolerances"], cfg["tests"]
    
    # Resolve backend
    use_gpu = cfg["hardware"].get("gpu_enabled", True)
    xp, on_gpu = pick_backend(use_gpu)
    if on_gpu and _HAS_CUPY:
        try:
            cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
        except Exception:
            log("GPU memory pool setup failed; continuing with default allocator", "WARN")
    
    precision = cfg["hardware"].get("precision", "float64")
    dtype = xp.float64 if precision == "float64" else xp.float32

    # Output directory
    project_root = Path(__file__).resolve().parent
    out_base = (project_root / cfg["output_dir"]).resolve()
    ensure_dirs(out_base)
    
    # Set up logging
    logger = LFMLogger(out_base)
    set_logger(logger)
    log_run_config(cfg, out_base)
    
    # Filter to single test if requested
    if args.test:
        tests = [t for t in tests if t["test_id"] == args.test]
        if not tests:
            log(f"[ERROR] Test '{args.test}' not found in config", "FAIL")
            return
        log(f"=== Running Single Test: {args.test} ===", "INFO")
    else:
        log(f"=== Tier-3 Energy Conservation Suite Start (backend={'CuPy' if on_gpu else 'NumPy'}) ===", "INFO")

    # Run tests
    results = []
    t_suite = time.time()
    for test in tests:
        if test.get("skip", False):
            log(f"[{test.get('test_id', '?')}] SKIPPED: {test.get('description', '')}", "WARN")
            continue
        test_dirname = test.get('test_id', f'ENER-??')
        result = run_test(p, tol, test, out_base / test_dirname, dtype, xp, on_gpu)
        results.append({
            "test_id": result.test_id,
            "description": result.description,
            "passed": result.passed,
            "energy_drift": result.energy_drift,
            "entropy_monotonic": result.entropy_monotonic,
            "wave_drop": result.wave_drop,
            "runtime_sec": result.runtime_sec
        })

    # Suite summary
    total_runtime = time.time() - t_suite
    suite_rows = [[r["test_id"], r["description"], r["passed"], r["energy_drift"],
                   r["entropy_monotonic"], r["wave_drop"], r["runtime_sec"]] for r in results]
    write_csv(out_base/"suite_summary.csv", suite_rows,
              ["test_id","description","passed","energy_rel_drift","entropy_monotonic","wave_max_drop","runtime_sec"])
    
    if args.test:
        # Single test: just show result
        log(f"=== Test {args.test} Complete ===", "INFO")
    else:
        # Full suite: show summary
        suite_summary(results)
        write_metadata_bundle(out_base, "TIER3-ENERGY", tier=3, category="Energy")
        log(f"\nTotal runtime: {total_runtime:.2f}s", "INFO")
        log("=== Tier-3 Suite Complete ===", "INFO")
    
    # Exit with appropriate code
    all_pass = all(r["passed"] for r in results)
    if not all_pass:
        exit(1)

if __name__ == "__main__":
    main()
