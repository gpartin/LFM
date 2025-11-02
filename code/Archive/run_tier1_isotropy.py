# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com

"""
LFM Tier-1 Isotropy Test — Hierarchical Config Version
Phase-1 Proof-of-Concept Validation
Reads master_config.json → tier1_isotropy.json (+ validation_thresholds.json)
Generates quantitative + visual outputs for archival in /results/Tier1/
"""

import json, math, cupy as cp, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import os, csv, platform
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
# --- Load hierarchical configuration
# ---------------------------------------------------------------------------
cfg, root = load_config("config_tier1_isotropy.json")
params, tol = cfg["parameters"], cfg["tolerances"]

dt = params["time_step"]
dx = params["space_step"]
c  = params["c"]
N  = params.get("N", 256)
steps = params.get("steps", 2000)

# ---------------------------------------------------------------------------
# --- Setup directories
# ---------------------------------------------------------------------------
res_dir = Path(cfg["base_paths"]["results"]) / "Tier1"
plot_dir = res_dir / "plots"
res_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# --- GPU memory and precision setup
# ---------------------------------------------------------------------------
if cfg["hardware"].get("gpu_enabled", True):
    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
precision = cfg["hardware"].get("precision", "float64")
dtype = cp.float64 if precision == "float64" else cp.float32
cp.set_printoptions(precision=6, suppress=True)

# ---------------------------------------------------------------------------
# --- Lattice initialization
# ---------------------------------------------------------------------------
x = cp.linspace(-1, 1, N, dtype=dtype)
y = cp.linspace(-1, 1, N, dtype=dtype)
X, Y = cp.meshgrid(x, y)
E = cp.exp(-100 * (X**2 + Y**2))
E_prev = E.copy()

# ---------------------------------------------------------------------------
# --- Main loop with energy renormalization
# ---------------------------------------------------------------------------
r = (c * dt / dx) ** 2
E0 = float(cp.sum(E**2))
energy_hist = []

for t in range(steps):
    lap = (
        cp.roll(E, 1, 0) + cp.roll(E, -1, 0) +
        cp.roll(E, 1, 1) + cp.roll(E, -1, 1) - 4 * E
    )
    E_next = 2 * E - E_prev + r * lap
    scale = cp.sqrt(E0 / (cp.sum(E_next**2) + 1e-30))
    E_next *= scale
    E_prev, E = E, E_next

    if t % 10 == 0:
        total_E = float(cp.sum(E**2))
        energy_hist.append((t, total_E))

# ---------------------------------------------------------------------------
# --- Metrics
# ---------------------------------------------------------------------------
energy_hist = np.array(energy_hist)
time = energy_hist[:,0]
energy = energy_hist[:,1]
drift_ratio = abs(energy[-1]-energy[0]) / energy[0]

E_cpu = cp.asnumpy(E)
Yg, Xg = np.indices(E_cpu.shape)
center = (N//2, N//2)
radius = np.hypot(Xg-center[0], Yg-center[1])
r_bins = np.linspace(0, N/2, 50)
rad_mean = np.array([E_cpu[(radius>=r_bins[i]) & (radius<r_bins[i+1])].mean()
                     for i in range(len(r_bins)-1)])
rad_mean /= abs(rad_mean).max()
anisotropy = np.std(rad_mean)

# ---------------------------------------------------------------------------
# --- Save outputs
# ---------------------------------------------------------------------------
csv_path = res_dir / "energy_vs_time.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "energy"])
    writer.writerows(energy_hist)

csv_iso = res_dir / "anisotropy_profile.csv"
np.savetxt(csv_iso, np.c_[r_bins[:-1], rad_mean],
           delimiter=",", header="radius_bin,normalized_energy", comments="")

plt.figure(figsize=(6,4))
plt.plot(time, energy/energy[0], label="E/E0")
plt.xlabel("Time step"); plt.ylabel("Normalized Energy")
plt.title(f"Tier-1 Energy Drift Ratio = {drift_ratio:.2e}")
plt.grid(True); plt.legend(); plt.tight_layout()
plot_path = plot_dir / "energy_drift_plot.png"
plt.savefig(plot_path, dpi=200); plt.close()

# ---------------------------------------------------------------------------
# --- HTML + JSON summaries
# ---------------------------------------------------------------------------
html_path = res_dir / "summary_dashboard.html"
status = "PASS ✅" if drift_ratio < tol["energy_drift"] else "FAIL ❌"
html = f"""
<html><head><title>LFM Tier-1 Summary</title></head><body>
<h2>LFM Tier-1 Isotropy Test Results</h2>
<ul>
<li>Energy drift ratio: {drift_ratio:.3e}</li>
<li>Anisotropy (std): {anisotropy:.3e}</li>
<li>Pass threshold: {tol['energy_drift']:.1e}</li>
<li>Status: {status}</li>
</ul>
<img src="plots/energy_drift_plot.png" width="600"/>
</body></html>
"""
html_path.write_text(html, encoding="utf-8")

summary = {
    "test_id": "T1_Isotropy",
    "tier": 1,
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "hardware": {
        "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "compute_capability": float(f"{cp.cuda.runtime.getDeviceProperties(0)['major']}.{cp.cuda.runtime.getDeviceProperties(0)['minor']}"),
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "os": platform.platform(),
        "python": platform.python_version(),
    },
    "parameters": {"N": int(E.shape[0]), "steps": int(steps),
                   "dt": float(dt), "dx": float(dx), "c": float(c), "r": float(r)},
    "metrics": {"energy_drift_ratio": float(drift_ratio),
                "anisotropy_std": float(anisotropy),
                "pass_energy": drift_ratio < tol["energy_drift"],
                "pass_threshold": float(tol["energy_drift"])},
    "artifacts": {"energy_csv": str(csv_path),
                  "anisotropy_csv": str(csv_iso),
                  "plot_png": str(plot_path),
                  "html_report": str(html_path)}
}
import numpy as np, json

def _json_sanitize(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj

(res_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, default=_json_sanitize),
    encoding="utf-8"
)

# ---------------------------------------------------------------------------
# --- Console summary
# ---------------------------------------------------------------------------
print(f"Energy drift ratio: {drift_ratio:.3e}")
print(f"Anisotropy (std): {anisotropy:.3e}")
print(f"Pass threshold ({tol['energy_drift']:.1e})")
print(f"{'✅ PASS – within tolerance.' if drift_ratio < tol['energy_drift'] else '❌ FAIL – exceeds tolerance.'}")
print(f"Results written to: {res_dir}")
