#!/usr/bin/env python3
"""
LFM Tier‑4 — Resolution Scaling Test
Runs the same LFM update at multiple resolutions and checks that the
numerical error (energy drift) decreases at the expected rate.
Outputs per‐variant go in results/Tier4/Resolution/<variant_id>/.
"""

import json, csv, math, platform
from pathlib import Path
from datetime import datetime

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# --- Config loader: merges master + inherited configs
# ---------------------------------------------------------------------------
def load_config(cfg_name: str):
    root = Path(__file__).resolve().parents[1] / "config"
    cfg_path = root / cfg_name
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config {cfg_path} not found")
    cfg = json.loads(cfg_path.read_text())

    merged = {}
    if "inherits" in cfg:
        for inc in cfg["inherits"]:
            inc_path = root / inc
            if inc_path.exists():
                merged.update(json.loads(inc_path.read_text()))
    merged.update(cfg)
    return merged, root

# ---------------------------------------------------------------------------
# --- Load configuration and setup GPU
# ---------------------------------------------------------------------------
cfg, ROOT = load_config("config_tier4_resolution_scaling.json")
params, tol = cfg["parameters"], cfg["tolerances"]

OUT_BASE = Path(cfg["base_paths"]["results"]) / "Tier4" / "Resolution"
OUT_BASE.mkdir(parents=True, exist_ok=True)

if cfg["hardware"].get("gpu_enabled", True):
    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
precision = cfg["hardware"].get("precision", "float64")
dtype = cp.float64 if precision == "float64" else cp.float32
cp.set_printoptions(precision=6, suppress=True)

def ensure_dirs(p):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# --- Build initial field for different test types
# ---------------------------------------------------------------------------
def initial_field(test_type, X, Y, sigma=0.05):
    if test_type == "isotropy":
        return cp.exp(-100.0 * (X ** 2 + Y ** 2))
    elif test_type == "pulse":
        return cp.exp(-((X - 0.0)**2 + (Y - 0.0)**2)/(2 * sigma**2))
    else:
        # default to isotropic Gaussian
        return cp.exp(-100.0 * (X ** 2 + Y ** 2))

# ---------------------------------------------------------------------------
# --- Variant execution
# ---------------------------------------------------------------------------
def run_variant(params, tol, variant, out_dir):
    ensure_dirs(out_dir)

    test_type = variant.get("test", "isotropy")
    resolutions = variant.get("resolutions", [64, 128, 256])
    base_N   = int(params.get("base_N", resolutions[0]))
    dt_base  = float(params["time_step"])
    dx_base  = float(params["space_step"])
    c        = float(params.get("c", 1.0))
    steps    = int(variant.get("steps", 2000))
    expected = float(variant.get("expected_exponent",
                                 tol.get("expected_exponent", 1.8)))

    drift_errors = []
    dx_values = []

    for N in resolutions:
        # scale dt and dx so that c*dt/dx is constant across resolutions
        dt = dt_base * base_N / N
        dx = dx_base * base_N / N
        r = (c * dt / dx) ** 2

        # grid setup
        x = cp.linspace(-1, 1, N, dtype=dtype)
        y = cp.linspace(-1, 1, N, dtype=dtype)
        X, Y = cp.meshgrid(x, y)
        E = initial_field(test_type, X, Y, sigma=variant.get("pulse_sigma", 0.05))
        E_prev = E.copy()
        E0 = float(cp.sum(E ** 2))

        # run update
        for t in range(steps):
            lap = (cp.roll(E, 1, 0) + cp.roll(E, -1, 0)
                 + cp.roll(E, 1, 1) + cp.roll(E, -1, 1) - 4 * E)
            E_next = 2 * E - E_prev + r * lap
            # normalize to control energy growth
            E_next *= cp.sqrt(E0 / (cp.sum(E_next ** 2) + 1e-30))
            E_prev, E = E, E_next

        # compute energy drift ratio for this resolution
        final_energy = float(cp.sum(E ** 2))
        drift = abs(final_energy - E0) / E0
        drift_errors.append(drift)
        dx_values.append(dx)

    # compute empirical scaling exponent using last two resolutions
    # slope = Δlog(error) / Δlog(dx)
    if len(dx_values) >= 2:
        log_err = np.log(drift_errors)
        log_dx  = np.log(dx_values)
        slope = (log_err[-2] - log_err[-1]) / (log_dx[-2] - log_dx[-1])
    else:
        slope = 0.0
    pass_scaling = slope >= expected

    # write CSV of results
    csv_path = out_dir / "resolution_scaling.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "dx", "drift_error"])
        for N, dx_val, err in zip(resolutions, dx_values, drift_errors):
            w.writerow([N, dx_val, err])

    # log–log plot
    ensure_dirs(out_dir / "plots")
    plt.figure(figsize=(6, 4))
    plt.loglog(dx_values, drift_errors, marker="o")
    plt.xlabel("grid spacing dx")
    plt.ylabel("energy drift error")
    plt.title(f"Scaling exponent = {slope:.2f}, expected ≥ {expected}")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "resolution_scaling.png", dpi=200)
    plt.close()

    # HTML summary
    html_status = "PASS ✅" if pass_scaling else "FAIL ❌"
    html = f"""
    <html><body><h2>Tier‑4 Resolution Scaling — {variant.get('description','variant')}</h2>
    <ul>
      <li>Measured scaling exponent: {slope:.3f}</li>
      <li>Expected exponent (target): {expected:.3f}</li>
      <li>Status: {html_status}</li>
    </ul>
    <img src='plots/resolution_scaling.png' width='550'/>
    </body></html>
    """
    (out_dir / "summary_dashboard.html").write_text(html, encoding="utf-8")

    # JSON summary
    summary = {
        "test_id": "T4_ResScaling",
        "variant": variant.get("description", "variant"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hardware": {
            "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "compute_capability": float(f"{cp.cuda.runtime.getDeviceProperties(0)['major']}.{cp.cuda.runtime.getDeviceProperties(0)['minor']}"),
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "python": platform.python_version()
        },
        "parameters": {
            "resolutions": resolutions,
            "base_N": base_N,
            "dt_base": dt_base,
            "dx_base": dx_base,
            "steps": steps,
            "c": c
        },
        "metrics": {
            "dx_values": dx_values,
            "drift_errors": drift_errors,
            "scaling_exponent": float(slope),
            "pass_scaling": pass_scaling,
            "expected_exponent": expected
        },
        "artifacts": {
            "csv": str(csv_path),
            "plot_png": str(out_dir / "plots" / "resolution_scaling.png"),
            "html_report": str(out_dir / "summary_dashboard.html")
        }
    }

    def sanitize(obj):
        import numpy as np
        if isinstance(obj, (np.bool_,)):  # convert NumPy bools
            return bool(obj)
        if isinstance(obj, (np.integer,)):  # convert NumPy ints
            return int(obj)
        if isinstance(obj, (np.floating,)):  # convert NumPy floats
            return float(obj)
        return str(obj)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=sanitize), encoding="utf-8")

    print(f"[ResScale] {variant.get('description','variant')}: exponent={slope:.2f}, expected≥{expected:.2f} -> {'PASS' if pass_scaling else 'FAIL'}")

# ---------------------------------------------------------------------------
# --- Main
# ---------------------------------------------------------------------------
def main():
    variants = cfg["variants"]
    for i, v in enumerate(variants, 1):
        out_dir = OUT_BASE / f"{i:02d}_{v.get('description','variant').replace(' ','_')}"
        run_variant(params, tol, v, out_dir)

if __name__ == "__main__":
    main()
