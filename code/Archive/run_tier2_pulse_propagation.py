# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
LFM Tier-2 — Pulse Propagation (Flat-χ)
Hierarchical Config Version (uses master_config + tier2_flat_pulse.json)
Covers REL-03..REL-06: group velocity ≈ c, boosted frame sanity, mode freq variants
Outputs per-variant: CSVs, plot, HTML, JSON summary in results/Tier2/Pulse/<variant_id>/
"""

import json, math, os, csv, platform, cupy as cp, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# --- Config loader (master + inherited)
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
# --- Paths & setup
# ---------------------------------------------------------------------------
cfg, ROOT = load_config("config_tier2_pulse_propagation.json")
params, tol = cfg["parameters"], cfg["tolerances"]

OUT_BASE = Path(cfg["base_paths"]["results"]) / "Tier2" / "Pulse"
OUT_BASE.mkdir(parents=True, exist_ok=True)

# GPU setup
if cfg["hardware"].get("gpu_enabled", True):
    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
precision = cfg["hardware"].get("precision", "float64")
dtype = cp.float64 if precision == "float64" else cp.float32
cp.set_printoptions(precision=6, suppress=True)

def ensure_dirs(p): p.mkdir(parents=True, exist_ok=True)
def write_csv(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def summary_json(dst, payload):
    (dst/"summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

def build_pulse(N, sigma, center=(0.0,0.0)):
    x = cp.linspace(-1, 1, N, dtype=dtype)
    y = cp.linspace(-1, 1, N, dtype=dtype)
    X, Y = cp.meshgrid(x, y)
    return cp.exp(-((X-center[0])**2 + (Y-center[1])**2)/(2*sigma**2))

def step_wave(E, E_prev, r):
    lap = (cp.roll(E,1,0)+cp.roll(E,-1,0)+cp.roll(E,1,1)+cp.roll(E,-1,1)-4*E)
    return 2*E - E_prev + r*lap

def centroid(field_sq):
    N = field_sq.shape[0]
    coords = cp.linspace(-1, 1, N, dtype=dtype)
    X, Y = cp.meshgrid(coords, coords)
    s = cp.sum(field_sq) + 1e-30
    cx = float(cp.sum(X*field_sq)/s); cy = float(cp.sum(Y*field_sq)/s)
    return cx, cy

# ---------------------------------------------------------------------------
# --- Core variant runner
# ---------------------------------------------------------------------------
def run_variant(params, tol, variant, out_dir):
    ensure_dirs(out_dir)

    # parameters
    c = float(params.get("c",1.0))
    dt = float(params["time_step"]); dx = float(params["space_step"])
    N  = int(params.get("N",512))
    steps = int(variant.get("steps",4000))
    sigma = float(variant.get("pulse_sigma", params.get("pulse_sigma",0.05)))
    center = tuple(variant.get("pulse_center", params.get("pulse_center",[0.0,0.0])))
    boost_v = float(variant.get("boost_velocity", 0.0))
    r = (c*dt/dx)**2

    # initial field
    E = build_pulse(N, sigma, center)
    E_prev = E.copy()
    E0 = float(cp.sum(E**2))

    times, v_est, centroids = [], [], []

    for t in range(steps):
        E_next = step_wave(E, E_prev, r)
        scale = cp.sqrt(E0 / (cp.sum(E_next**2) + 1e-30))
        E_next *= scale

        # crude "boost" (diagnostic translation)
        if abs(boost_v) > 0:
            pix = int(math.copysign(max(1, round(abs(boost_v)*10)), boost_v))
            E_next = cp.roll(E_next, pix, axis=1)

        E_prev, E = E, E_next

        if t % 10 == 0:
            e2 = E**2
            cx, cy = centroid(e2)
            centroids.append((t, cx, cy))
            if len(centroids) >= 2:
                (t0,x0,y0), (t1,x1,y1) = centroids[-2], centroids[-1]
                dist = math.hypot(x1-x0, y1-y0)
                dt_steps = (t1 - t0)
                v = dist / (dt_steps*dt/dx)
                v_est.append((t1, v))
            times.append(t)

    v_np = np.array(v_est) if v_est else np.zeros((0,2))
    v_mean = float(np.mean(v_np[:,1])) if v_np.size else 0.0
    v_err = abs(v_mean - 1.0)
    pass_velocity = v_err <= float(tol.get("velocity_error", 0.01))

    # save outputs
    write_csv(out_dir/"centroid.csv", centroids, ["step","cx","cy"])
    write_csv(out_dir/"velocity.csv", v_est, ["step","v_est"])

    plt.figure(figsize=(6,4))
    if v_np.size:
        plt.plot(v_np[:,0], v_np[:,1], label="estimated v")
    plt.axhline(1.0, linestyle="--", label="target v=c")
    plt.xlabel("step"); plt.ylabel("velocity (≈c units)")
    plt.title(f"Mean |v−c| = {v_err:.2e}")
    plt.grid(True); plt.legend(); plt.tight_layout()
    ensure_dirs(out_dir/"plots")
    plt.savefig(out_dir/"plots"/"velocity.png", dpi=200)
    plt.close()

    html = f"""
    <html><body>
    <h2>Tier-2 Pulse Propagation — {variant.get('description','variant')}</h2>
    <ul><li>Mean |v−c|: {v_err:.3e} (tol {tol.get('velocity_error',0.01):.2e})</li>
    <li>Status: {'PASS ✅' if pass_velocity else 'FAIL ❌'}</li></ul>
    <img src="plots/velocity.png" width="600"/>
    </body></html>
    """
    (out_dir/"summary_dashboard.html").write_text(html, encoding="utf-8")

    summary_json(out_dir, {
        "test_id": "T2_Pulse",
        "variant": variant.get("description","variant"),
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "hardware": {
            "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "python": platform.python_version()
        },
        "parameters": {"N": N, "steps": steps, "dt": dt, "dx": dx, "c": c,
                       "r": (c*dt/dx)**2, "boost_v": boost_v, "sigma": sigma},
        "metrics": {"mean_abs_v_err": v_err, "pass_velocity": pass_velocity},
        "artifacts": {"centroid_csv": str(out_dir/"centroid.csv"),
                      "velocity_csv": str(out_dir/"velocity.csv"),
                      "plot_png": str(out_dir/"plots"/"velocity.png"),
                      "html_report": str(out_dir/"summary_dashboard.html")}
    })
    print(f"[Pulse] {variant.get('description','variant')}: mean|v−c|={v_err:.3e} -> {'PASS' if pass_velocity else 'FAIL'}")

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
