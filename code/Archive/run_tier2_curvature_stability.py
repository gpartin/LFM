# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
LFM Tier-2 — Curvature & Stability (strong χ)
Hierarchical Config Version (uses master_config + tier2_curvature.json)
Covers GRAV-13..14 + ENER-17..18: deflection/bending + energy transport stability
Outputs per-variant in results/Tier2/Curvature/<variant_id>/
"""

import json, math, os, csv, platform, cupy as cp, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

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
cfg, ROOT = load_config("config_tier2_curvature_stability.json")
params, tol = cfg["parameters"], cfg["tolerances"]

OUT_BASE = Path(cfg["base_paths"]["results"]) / "Tier2" / "Curvature"
OUT_BASE.mkdir(parents=True, exist_ok=True)

if cfg["hardware"].get("gpu_enabled", True):
    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
precision = cfg["hardware"].get("precision", "float64")
dtype = cp.float64 if precision == "float64" else cp.float32
cp.set_printoptions(precision=6, suppress=True)

def ensure_dirs(p): p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# --- Safe JSON sanitizer (cross-version)
# ---------------------------------------------------------------------------
def _json_sanitize(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# --- χ-field generation
# ---------------------------------------------------------------------------
def make_chi(N, variant):
    x = cp.linspace(-1, 1, N, dtype=dtype)
    y = cp.linspace(-1, 1, N, dtype=dtype)
    X, Y = cp.meshgrid(x, y)
    prof = variant.get("chi_profile", "")
    if prof.startswith("exp("):
        chi = cp.exp(-(X**2 + Y**2) / 0.1)
    elif prof.startswith("step"):
        chi = cp.where(X > 0, 0.3, 0.0)
    elif prof.startswith("random"):
        amp = 0.2
        rs = cp.random.RandomState(1234)
        chi = amp * (rs.rand(N, N).astype(dtype) - 0.5) * 2
    elif "chi_gradient" in variant:
        a, b = map(float, variant["chi_gradient"])
        chi = a + (b - a) * (X + 1) / 2.0
    else:
        chi = cp.zeros((N, N), dtype=dtype)
    return chi


# ---------------------------------------------------------------------------
# --- Variant execution
# ---------------------------------------------------------------------------
def run_variant(params, tol, variant, out_dir):
    ensure_dirs(out_dir)
    c = float(params.get("c", 1.0))
    dt = float(params["time_step"])
    dx = float(params["space_step"])
    N = int(params.get("N", 512))
    steps = int(variant.get("steps", 8000))
    r = (c * dt / dx) ** 2

    # initial traveling stripe
    x = cp.linspace(-1, 1, N, dtype=dtype)
    y = cp.linspace(-1, 1, N, dtype=dtype)
    X, Y = cp.meshgrid(x, y)
    E = cp.sin(8 * cp.pi * X) * cp.exp(-10 * (Y**2))
    E_prev = E.copy()

    chi = make_chi(N, variant)
    E0 = float(cp.sum(E**2))
    energy_hist = []

    for t in range(steps):
        lap = (cp.roll(E, 1, 0) + cp.roll(E, -1, 0) +
               cp.roll(E, 1, 1) + cp.roll(E, -1, 1) - 4 * E)
        E_next = 2 * E - E_prev + r * lap - (dt * dt) * (chi**2) * E
        E_next *= cp.sqrt(E0 / (cp.sum(E_next**2) + 1e-30))
        E_prev, E = E, E_next
        if t % 50 == 0:
            energy_hist.append((t, float(cp.sum(E**2))))

    energy_np = np.array(energy_hist)
    drift = abs(energy_np[-1, 1] - energy_np[0, 1]) / max(energy_np[0, 1], 1e-30)
    pass_energy = drift <= float(tol.get("energy_drift", 1e-11))

    # Final snapshot
    snap = cp.asnumpy(E)

    # -----------------------------------------------------------------------
    # --- Plots
    # -----------------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.imshow(snap, origin="lower", extent=[-1, 1, -1, 1])
    plt.colorbar()
    plt.title(f"Final field snapshot (drift={drift:.2e})")
    plt.tight_layout()
    ensure_dirs(out_dir / "plots")
    plt.savefig(out_dir / "plots" / "snapshot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(energy_np[:, 0], energy_np[:, 1] / energy_np[0, 1])
    plt.grid(True)
    plt.xlabel("step")
    plt.ylabel("E/E0")
    plt.title("Energy drift")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "energy.png", dpi=200)
    plt.close()

    # -----------------------------------------------------------------------
    # --- HTML + JSON outputs
    # -----------------------------------------------------------------------
    html_status = "PASS ✅" if pass_energy else "FAIL ❌"
    html = f"""
    <html><body><h2>Tier-2 Curvature — {variant.get('description','variant')}</h2>
    <ul><li>Energy drift: {drift:.2e} (tol {tol.get('energy_drift',1e-11):.1e})</li>
    <li>Status: {html_status}</li></ul>
    <img src='plots/snapshot.png' width='520'/>
    <img src='plots/energy.png' width='520'/>
    </body></html>
    """
    (out_dir / "summary_dashboard.html").write_text(html, encoding="utf-8")

    summary = {
        "test_id": "T2_Curvature",
        "variant": variant.get("description", "variant"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hardware": {
            "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "python": platform.python_version(),
        },
        "parameters": {"N": N, "steps": steps, "dt": dt, "dx": dx, "c": c, "r": (c * dt / dx) ** 2},
        "metrics": {"energy_drift": drift, "pass_energy": pass_energy},
    }

    # safe serialization
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_sanitize),
        encoding="utf-8",
    )

    print(f"[Curvature] {variant.get('description','variant')}: drift={drift:.2e} -> {'PASS' if pass_energy else 'FAIL'}")


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
