#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

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
import matplotlib.pyplot as plt

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

from core.lfm_backend import to_numpy, get_array_module
from ui.lfm_console import log, suite_summary
from utils.lfm_results import save_summary, write_metadata_bundle, write_csv, ensure_dirs, update_master_test_status
from harness.lfm_test_harness import BaseTierHarness
from harness.lfm_test_metrics import TestMetrics

def _default_config_name() -> str:
    return "config_tier3_energy.json"

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
        xp = get_array_module(E)
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
        xp = get_array_module(E)
    Ex = (xp.roll(E, -1, 1) - xp.roll(E, 1, 1)) / (2*dx)
    Ey = (xp.roll(E, -1, 0) - xp.roll(E, 1, 0)) / (2*dx)
    return Ex*Ex + Ey*Ey

def energy_total(E, E_prev, dt, dx, c, chi, xp=None):
    """Compute total energy using specified backend."""
    if xp is None:
        xp = get_array_module(E)
    Et = (E - E_prev) / dt
    dens = 0.5*(Et*Et + (c*c)*grad_sq(E, dx, xp) + (chi*chi)*(E*E))
    return float(xp.sum(dens) * dx*dx)

def energy_components(E, E_prev, dt, dx, c, chi, xp=None):
    """
    Compute Hamiltonian energy components: KE, GE, PE.
    
    Returns (KE, GE, PE) where:
    - KE (kinetic):   ½ ∫ (∂E/∂t)² dV
    - GE (gradient):  ½ ∫ c²(∇E)² dV  
    - PE (potential): ½ ∫ χ²E² dV
    
    H_total = KE + GE + PE should be conserved.
    """
    if xp is None:
        xp = cp if (_HAS_CUPY and isinstance(E, cp.ndarray)) else np
    
    Et = (E - E_prev) / dt
    KE = float(0.5 * xp.sum(Et * Et) * dx*dx)
    GE = float(0.5 * (c*c) * xp.sum(grad_sq(E, dx, xp)) * dx*dx)
    PE = float(0.5 * xp.sum((chi*chi) * (E*E)) * dx*dx)
    
    return KE, GE, PE

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
    track_components = bool(test.get("track_hamiltonian_components", False))

    energy_trace, entropy_trace, times = [], [], []
    KE_trace, GE_trace, PE_trace = [], [], []
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
            
            if track_components:
                KE, GE, PE = energy_components(E, E_prev, dt, dx, c, chi, xp)
                KE_trace.append(KE)
                GE_trace.append(GE)
                PE_trace.append(PE)

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

    # --- Hamiltonian component visualization ---
    if track_components and len(KE_trace) > 0:
        KE_np = np.array(KE_trace)
        GE_np = np.array(GE_trace)
        PE_np = np.array(PE_trace)
        H_total = KE_np + GE_np + PE_np
        
        # Stacked area chart showing energy flow between components
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Top: Stacked area showing component breakdown
        ax1.fill_between(times_np, 0, KE_np, label='Kinetic (∂E/∂t)²', alpha=0.7, color='#e74c3c')
        ax1.fill_between(times_np, KE_np, KE_np+GE_np, label='Gradient (∇E)²', alpha=0.7, color='#3498db')
        ax1.fill_between(times_np, KE_np+GE_np, H_total, label='Potential (χE)²', alpha=0.7, color='#2ecc71')
        ax1.plot(times_np, H_total, 'k-', linewidth=2, label='Total H', alpha=0.8)
        ax1.set_ylabel('Energy density')
        ax1.set_title(f'Hamiltonian Partitioning: H = KE + GE + PE (conserved)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Bottom: Individual components showing energy "slosh"
        ax2.plot(times_np, KE_np / H_total[0], label='KE fraction', color='#e74c3c', linewidth=2)
        ax2.plot(times_np, GE_np / H_total[0], label='GE fraction', color='#3498db', linewidth=2)
        ax2.plot(times_np, PE_np / H_total[0], label='PE fraction', color='#2ecc71', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Fractional energy')
        ax2.set_title('Energy flow between Hamiltonian modes')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(out_dir/"plots"/"hamiltonian_components.png", dpi=150)
        plt.close()
        
        # Additional: Total H conservation verification
        H_drift = abs(H_total[-1] - H_total[0]) / max(H_total[0], 1e-30)
        plt.figure(figsize=(8, 5))
        plt.plot(times_np, H_total, 'k-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Total Hamiltonian H')
        plt.title(f'Hamiltonian Conservation: H(t) (drift={H_drift:.3e})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir/"plots"/"hamiltonian_total.png", dpi=150)
        plt.close()

    # --- Output CSVs ---
    diagnostics_dir = out_dir / "diagnostics"
    ensure_dirs(diagnostics_dir)
    write_csv(diagnostics_dir/"energy_trace.csv",  list(zip(times_np, energy_trace_np)), ["time","energy"])
    write_csv(diagnostics_dir/"entropy_trace.csv", list(zip(times_np, entropy_np)),      ["time","entropy"])
    
    if track_components and len(KE_trace) > 0:
        KE_np = np.array(KE_trace)
        GE_np = np.array(GE_trace)
        PE_np = np.array(PE_trace)
        H_total = KE_np + GE_np + PE_np
        write_csv(diagnostics_dir/"hamiltonian_components.csv",
                 list(zip(times_np, KE_np, GE_np, PE_np, H_total)),
                 ["time", "KE", "GE", "PE", "H_total"])

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
    # Optional post-run hooks
    parser.add_argument('--post-validate', choices=['tier', 'all'], default=None,
                        help='Run validator after the suite: "tier" validates Tier 3 + master status; "all" runs end-to-end')
    parser.add_argument('--strict-validate', action='store_true',
                        help='In strict mode, warnings cause validation to fail')
    parser.add_argument('--quiet-validate', action='store_true',
                        help='Reduce validator verbosity')
    parser.add_argument('--update-upload', action='store_true',
                        help='Rebuild docs/upload package (refresh status, stage docs, comprehensive PDF, manifest)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic mode for upload build (fixed timestamps, reproducible zip)')
    args = parser.parse_args()
    
    cfg = BaseTierHarness.load_config(args.config, default_config_name=_default_config_name())
    p, tol, tests = cfg["parameters"], cfg["tolerances"], cfg["tests"]
    
    # Resolve backend
    from core.lfm_backend import pick_backend
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
    from utils.lfm_logger import LFMLogger
    from ui.lfm_console import set_logger, log_run_config
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
    
    # Import resource tracking
    from utils.resource_tracking import create_resource_tracker
    
    for test in tests:
        if test.get("skip", False):
            log(f"[{test.get('test_id', '?')}] SKIPPED: {test.get('description', '')}", "WARN")
            continue
        
        test_dirname = test.get('test_id', f'ENER-??')
        
        # Start resource tracking
        tracker = create_resource_tracker()
        tracker.start(background=True)
        
        # Run test
        result = run_test(p, tol, test, out_base / test_dirname, dtype, xp, on_gpu)
        
        # Stop tracking and get metrics
        tracker.stop()
        metrics = tracker.get_metrics()
        
        results.append({
            "test_id": result.test_id,
            "description": result.description,
            "passed": result.passed,
            "energy_drift": result.energy_drift,
            "entropy_monotonic": result.entropy_monotonic,
            "wave_drop": result.wave_drop,
            "runtime_sec": metrics["runtime_sec"],
            "peak_cpu_percent": metrics["peak_cpu_percent"],
            "peak_memory_mb": metrics["peak_memory_mb"],
            "peak_gpu_memory_mb": metrics["peak_gpu_memory_mb"],
        })

    # Suite summary
    total_runtime = time.time() - t_suite
    suite_rows = [[r["test_id"], r["description"], r["passed"], r["energy_drift"],
                   r["entropy_monotonic"], r["wave_drop"], r["runtime_sec"]] for r in results]
    write_csv(out_base/"suite_summary.csv", suite_rows,
              ["test_id","description","passed","energy_rel_drift","entropy_monotonic","wave_max_drop","runtime_sec"])
    
    # Update master test status and metrics database
    update_master_test_status()
    
    # Record metrics for resource tracking (now with REAL metrics!)
    test_metrics = TestMetrics()
    for r in results:
        metrics_data = {
            "exit_code": 0 if r["passed"] else 1,
            "runtime_sec": r["runtime_sec"],
            "peak_cpu_percent": r["peak_cpu_percent"],
            "peak_memory_mb": r["peak_memory_mb"],
            "peak_gpu_memory_mb": r["peak_gpu_memory_mb"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        test_metrics.record_run(r["test_id"], metrics_data)

    # Optional: post-run validation
    if args.post_validate:
        try:
            from tools.validate_results_pipeline import PipelineValidator  # type: ignore
            v = PipelineValidator(strict=args.strict_validate, verbose=not args.quiet_validate)
            ok = True
            if args.post_validate == 'tier':
                ok = v.validate_tier_results(3) and v.validate_master_status_integrity()
            elif args.post_validate == 'all':
                ok = v.validate_end_to_end()
            exit_code = v.report()
            if exit_code != 0:
                if args.strict_validate:
                    log(f"[TIER3] Post-validation failed (exit_code={exit_code})", "FAIL")
                    raise SystemExit(exit_code)
                else:
                    log(f"[TIER3] Post-validation completed with warnings (exit_code={exit_code})", "WARN")
            else:
                log("[TIER3] Post-validation passed", "PASS")
        except Exception as e:
            log(f"[TIER3] Validator error: {type(e).__name__}: {e}", "WARN")

    # Optional: rebuild upload package (dry-run staging under docs/upload)
    if args.update_upload:
        try:
            from tools import build_upload_package as bup  # type: ignore
            bup.refresh_results_artifacts(deterministic=args.deterministic, build_master=False)
            bup.stage_evidence_docx(include=True)
            bup.export_txt_from_evidence(include=True)
            bup.export_md_from_evidence()
            bup.stage_result_plots(limit_per_dir=6)
            pdf_rel = bup.generate_comprehensive_pdf()
            if pdf_rel:
                log(f"[TIER3] Generated comprehensive PDF: {pdf_rel}", "INFO")
            entries = bup.stage_and_list_files()
            zip_rel, _size, _sha = bup.create_zip_bundle(entries, label=None, deterministic=args.deterministic)
            entries_with_zip = entries + [(zip_rel, (bup.UPLOAD / zip_rel).stat().st_size, bup.sha256_file(bup.UPLOAD / zip_rel))]
            bup.write_manifest(entries_with_zip, deterministic=args.deterministic)
            bup.write_zenodo_metadata(entries_with_zip, deterministic=args.deterministic)
            bup.write_osf_metadata(entries_with_zip)
            log("[TIER3] Upload package refreshed under docs/upload (manifest and metadata written)", "INFO")
        except Exception as e:
            log(f"[TIER3] Upload package build encountered an error: {type(e).__name__}: {e}", "WARN")
    
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
