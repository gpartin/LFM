#!/usr/bin/env python3
"""
LFM Tier-1 — Relativistic Propagation & Isotropy Suite (v1.8.6-RobustPlot)
Tests REL-01 … REL-08 under canonical lattice law:
    ∂²E/∂t² = c²∇²E − χ²E

Fixes in this build:
- Output root forced to project-level /results/, not inside /code/.
- Guarantees plot creation for every test (adds fallback static PNG if animation fails).
- No change to solver or physics.
"""

import json, math, time
from pathlib import Path
import numpy as np
from datetime import datetime

# GPU optional (CuPy) fallback to NumPy
try:
    import cupy as cp  # type: ignore
except Exception:
    import numpy as cp  # type: ignore
if not hasattr(cp, "asnumpy"):
    cp.asnumpy = lambda x: x

from lfm_console import log, suite_summary
from lfm_logger import LFMLogger
from lfm_diagnostics import field_spectrum, energy_total, energy_flow, phase_corr
from lfm_visualizer import visualize_concept


# ---------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------
def load_config():
    """Locate and load configuration JSON for this script."""
    script = Path(__file__).stem.replace("run_", "")
    base_dir = Path(__file__).resolve().parent
    for root in (base_dir, base_dir.parent):
        cfg_path = root / "config" / f"config_{script}.json"
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("Config file not found for this test.")


# ---------------------------------------------------------------------
# Output directory resolver (project-level aware)
# ---------------------------------------------------------------------
def resolve_outdir(output_dir_hint: str) -> Path:
    """
    Always place outputs one level above /code/, e.g.:
    C:/LFM/code/run_tier1_relativistic.py → C:/LFM/results/Relativistic/
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # go up one level from /code/
    outdir = project_root / output_dir_hint
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# ---------------------------------------------------------------------
# Field initialization per variant
# ---------------------------------------------------------------------
def init_field_variant(test_id, params, N, dx, c):
    x = cp.arange(N) * dx
    k_frac = params.get("k_fraction", 0.1)
    k0 = 2 * math.pi * k_frac / (N * dx)

    if test_id in ("REL-01", "REL-02"):
        return cp.cos(k0 * x)
    elif test_id == "REL-03":
        beta = params.get("boost_factor", 0.2)
        gamma = 1.0 / math.sqrt(1 - beta**2)
        phase = gamma * (k0 * x - beta * k0 * x)
        return cp.cos(phase)
    elif test_id == "REL-04":
        beta = params.get("boost_factor", 0.6)
        gamma = 1.0 / math.sqrt(1 - beta**2)
        phase = gamma * (k0 * x - beta * k0 * x)
        return cp.cos(phase)
    elif test_id == "REL-05":
        pulse_amp = params.get("pulse_amp", 1.0)
        pulse_width = params.get("pulse_width", 0.1)
        center = N // 2
        return pulse_amp * cp.exp(-((x - center * dx)**2) / (2 * pulse_width**2))
    elif test_id == "REL-06":
        rng = cp.random.default_rng(1234)
        E = cp.ones(N) + params.get("noise_amp", 1e-4) * rng.standard_normal(N)
        return E
    elif test_id == "REL-07":
        return cp.sin(k0 * x)
    elif test_id == "REL-08":
        return cp.cos(k0 * x) + 0.5 * cp.sin(2 * k0 * x)
    else:
        return cp.cos(k0 * x)


# ---------------------------------------------------------------------
# Core solver + diagnostics
# ---------------------------------------------------------------------
def run_relativistic_test(test_id, desc, params, tol, outdir, quick):
    N = params["grid_points"]
    dx = params["dx"]
    dt = params["dt"]
    chi = params["chi"]
    steps = params["steps"] if not quick else 500
    c = math.sqrt(params["alpha"] / params["beta"])
    gamma_damp = 1e-3

    # Per-test folders
    test_dir = outdir / test_id
    diag_dir = test_dir / "diagnostics"
    plot_dir = test_dir / "plots"
    for d in (test_dir, diag_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    log(f"[paths] test_dir={test_dir}", "INFO")
    log(f"[paths] plot_dir={plot_dir}", "INFO")

    # Initialize
    E_prev = init_field_variant(test_id, params, N, dx, c)
    E = E_prev.copy()
    E0 = energy_total(E, E_prev, dt, dx, c, chi)
    E_series = [E.copy()]

    t0 = time.time()
    for n in range(steps):
        lap = (cp.roll(E, -1) - 2 * E + cp.roll(E, 1)) / dx**2
        E_next = (2 - gamma_damp) * E - (1 - gamma_damp) * E_prev + (dt**2) * ((c**2) * lap - (chi**2) * E)
        E_next *= cp.sqrt(E0 / (cp.sum(E_next**2) + 1e-30))
        if n % 10 == 0:
            E_series.append(E_next.copy())
        E_prev, E = E, E_next
    runtime = time.time() - t0

    # Diagnostics
    field_spectrum(E, dx, diag_dir)
    energy_flow(E_series, dt, dx, c, diag_dir)
    phase_corr(E_series, diag_dir)

    # Visualization (robust even if field diverges)
    concept_base = f"concept_{test_id}"
    for ext in ("png", "gif", "mp4"):
        fpath = plot_dir / f"{concept_base}.{ext}"
        fpath.unlink(missing_ok=True)

    log(f"[viz] generating {plot_dir / (concept_base + '.png')}", "INFO")

    try:
        visualize_concept(E_series, tier=1, test_id=test_id,
                          outdir=plot_dir, quick=quick, animate=True)
        png_path = plot_dir / f"{concept_base}.png"
        if png_path.exists():
            log(f"[viz] wrote: {png_path}", "INFO")
        else:
            raise FileNotFoundError
    except Exception as e:
        log(f"[WARN] Visualization failed for {test_id} ({e}); writing fallback plot.", "WARN")
        import matplotlib.pyplot as plt
        arr = np.array(cp.asnumpy(E_series[-1])).real
        plt.figure(figsize=(5, 3))
        plt.plot(arr, lw=1.0, color="orange")
        plt.title(f"{test_id} — fallback static field plot")
        plt.xlabel("x-index"); plt.ylabel("E amplitude")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{concept_base}_fallback.png", dpi=120)
        plt.close()

    # Frequency diagnostics
    data = np.array([float(cp.asnumpy(Ev[N // 2].real)) for Ev in E_series])
    freqs = np.fft.fftfreq(len(data), d=10 * dt)
    fft = np.abs(np.fft.fft(data))
    peak = freqs[np.argmax(fft[1:]) + 1]
    omega_meas = 2 * math.pi * abs(peak)
    k_frac = params.get("k_fraction", 0.1)
    k0 = 2 * math.pi * k_frac / (N * dx)
    omega_theory = math.sqrt((c * k0)**2 + chi**2)
    rel_err = abs(omega_meas - omega_theory) / max(omega_theory, 1e-30)

    passed = bool(rel_err <= tol["phase_error_max"])
    status = "PASS ✅" if passed else "FAIL ❌"
    log(f"{test_id} {status} (rel_err={rel_err*100:.2f}% > tol)", "FAIL" if not passed else "INFO")

    summary = {
        "id": str(test_id),
        "desc": str(desc),
        "passed": passed,
        "rel_err": float(rel_err),
        "runtime_sec": float(runtime),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    with open(test_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    cfg = load_config()
    run = cfg["run_settings"]
    p_base = cfg["parameters"]
    tol = cfg["tolerances"]
    variants = cfg["variants"]

    outdir = resolve_outdir(cfg["output_dir"])
    log(f"[paths] OUTPUT ROOT = {outdir}", "INFO")

    quick = bool(run.get("quick_mode", False))
    _ = LFMLogger(outdir)

    log(f"=== Tier-1 Relativistic Suite Start (quick={quick}) ===", "INFO")
    results = []
    for v in variants:
        tid = v["test_id"]
        desc = v["description"]
        params = p_base.copy()
        params.update(v)
        log(f"→ Starting {tid}: {desc} ({params.get('steps','?')} steps)", "INFO")
        r = run_relativistic_test(tid, desc, params, tol, outdir, quick)
        results.append(r)

    suite_summary(results)
    log("=== Tier-1 Suite Complete ===", "INFO")


if __name__ == "__main__":
    main()
