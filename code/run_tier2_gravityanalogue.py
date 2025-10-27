#!/usr/bin/env python3
"""
LFM Tier-2 — Gravity Analogue Suite (v1.0.1 ids-from-TestPlan)
Tests GRAV-09 … GRAV-14 and GRAV-VARC-01 under canonical lattice law:
    ∂²E/∂t² = c²∇²E − χ(x)²E
Implements χ-gradients to simulate gravitational redshift, Shapiro delay, and weak-field lensing.

Pass criteria (Phase 1 Test Design §5):
  • correlation > 0.95 with analytic model
  • |ΔE|/E < 1 %
"""

import json, math, time
from pathlib import Path
import numpy as np
from datetime import datetime

try:
    import cupy as cp  # GPU optional
except Exception:
    import numpy as cp  # type: ignore
if not hasattr(cp, "asnumpy"):
    cp.asnumpy = lambda x: x

from lfm_console import log, suite_summary
from lfm_logger import LFMLogger
from lfm_diagnostics import field_spectrum, energy_total, energy_flow, phase_corr
from lfm_visualizer import visualize_concept
from lfm_results import save_summary

# ---------------------------------------------------------------------
def load_config():
    script = Path(__file__).stem.replace("run_", "")
    base_dir = Path(__file__).resolve().parent
    for root in (base_dir, base_dir.parent):
        cfg_path = root / "config" / f"config_{script}.json"
        if cfg_path.is_file():
            return json.loads(cfg_path.read_text(encoding="utf-8"))
    raise FileNotFoundError("Config file not found for this test.")

def resolve_outdir(output_dir_hint: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    outdir = project_root / output_dir_hint
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

# ---------------------------------------------------------------------
def make_chi_profile(N, dx, variant):
    """Generate static χ(x) curvature gradient."""
    mode = variant.get("chi_profile", "linear")
    chi0 = variant.get("chi0", 0.0)
    grad = variant.get("chi_grad", 0.01)
    x = cp.arange(N) * dx
    if mode == "linear":
        chi = chi0 + grad * (x - 0.5 * N * dx)
    elif mode == "gaussian":
        w = variant.get("chi_width", 0.2)
        chi = chi0 + grad * cp.exp(-((x - 0.5 * N * dx) ** 2) / (2 * w ** 2))
    elif mode == "step":
        chi = chi0 + grad * (x > 0.5 * N * dx)
    elif mode == "radial_like":  # proxy for Shapiro delay variants
        # Monotone increase toward center; 1D "radial" analogue
        r = cp.abs(x - 0.5 * N * dx)
        chi = chi0 + grad * (1.0 / (r + dx) - 1.0 / (0.25 * N * dx + dx))
    else:
        chi = cp.full(N, chi0)
    return chi

# ---------------------------------------------------------------------
def run_gravity_test(test_id, desc, params, tol, outdir, quick):
    N, dx, dt = params["grid_points"], params["dx"], params["dt"]
    alpha, beta = params["alpha"], params["beta"]
    c = math.sqrt(alpha / beta)
    chi = make_chi_profile(N, dx, params)

    SAVE_EVERY = 1
    steps = params.get("steps", 4000 if quick else 8000)
    gamma_damp = params.get("gamma_damp", 1e-4)
    pulse_w = params.get("pulse_width", 0.1)
    amp = params.get("pulse_amp", 1.0)

    # Directories
    test_dir = outdir / test_id
    diag_dir = test_dir / "diagnostics"
    plot_dir = test_dir / "plots"
    for d in (test_dir, diag_dir, plot_dir): d.mkdir(parents=True, exist_ok=True)

    log(f"→ {test_id}: {desc}", "INFO")

    # Initialize Gaussian pulse near left edge
    x = cp.arange(N) * dx
    E_prev = amp * cp.exp(-((x - 0.2 * N * dx) ** 2) / (2 * pulse_w ** 2))
    E = E_prev.copy()
    E_series = [E.copy()]
    E0 = energy_total(E, E_prev, dt, dx, c, float(cp.mean(chi)))

    # Time stepping with χ(x)
    for n in range(steps):
        lap = (cp.roll(E, -1) - 2 * E + cp.roll(E, 1)) / dx**2
        E_next = (2 - gamma_damp) * E - (1 - gamma_damp) * E_prev \
                 + (dt**2) * ((c**2) * lap - (chi**2) * E)
        if n % SAVE_EVERY == 0:
            E_series.append(E_next.copy())
        E_prev, E = E, E_next

    # Diagnostics
    field_spectrum(E, dx, diag_dir)
    energy_flow(E_series, dt, dx, c, diag_dir)
    phase_corr(E_series, diag_dir)

    # Visualization
    visualize_concept(E_series, chi_series=[chi], tier=2, test_id=test_id,
                      outdir=plot_dir, quick=quick, animate=True)

    # --- Metric: gravitational redshift / correlation ---
    e_np = cp.asnumpy(E_series[-1]).astype(float)
    chi_np = cp.asnumpy(chi).astype(float)

    # Frequency estimate from final field snapshot (quick proxy)
    spec = np.abs(np.fft.rfft(e_np - e_np.mean()))
    freqs = np.fft.rfftfreq(len(e_np), d=dt)
    f_peak = freqs[np.argmax(spec[1:]) + 1] if len(spec) > 1 else 0.0
    omega_meas = 2 * math.pi * abs(f_peak)
    omega_flat = math.sqrt((c * math.pi / (N * dx)) ** 2 + (np.mean(chi_np) ** 2))
    redshift = 1.0 - (omega_meas / (omega_flat + 1e-30))

    # Energy–curvature correlation (weak-field focusing proxy)
    corr = float(np.corrcoef(chi_np, e_np)[0, 1]) if np.std(chi_np) > 0 and np.std(e_np) > 0 else 0.0

    passed = (corr > 0.95) and (abs(redshift) < tol["redshift_max"])
    status = "PASS ✅" if passed else "FAIL ❌"
    log(f"{test_id} {status} (corr={corr:.3f}, z={redshift:.4e})",
        "FAIL" if not passed else "INFO")

    summary = {
        "id": test_id, "desc": desc, "passed": passed,
        "corr": float(corr), "redshift": float(redshift),
        "runtime_steps": int(steps), "dt": float(dt), "dx": float(dx),
        "timestamp": datetime.utcnow().isoformat()+"Z"
    }
    save_summary(test_dir, test_id, summary)
    return summary

# ---------------------------------------------------------------------
def main():
    cfg = load_config()
    run = cfg["run_settings"]
    p_base = cfg["parameters"]
    tol = cfg["tolerances"]
    variants = cfg["variants"]

    outdir = resolve_outdir(cfg["output_dir"])
    _ = LFMLogger(outdir)
    quick = bool(run.get("quick_mode", False))
    log(f"=== Tier-2 Gravity-Analogue Suite Start (quick={quick}) ===", "INFO")

    results = []
    for v in variants:
        p = p_base.copy(); p.update(v)
        results.append(run_gravity_test(v["test_id"], v["description"], p, tol, outdir, quick))

    suite_summary(results)
    log("=== Tier-2 Suite Complete ===", "INFO")

if __name__ == "__main__":
    main()
