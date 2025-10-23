"""
LFM Tier-3 — Energy/Entropy Diagnostics
Hierarchical Config Version (uses master_config + tier3_entropy.json)
Covers ENER-19..21: long-run equilibrium, diffusion, entropy trends
Outputs per-variant in results/Tier3/Entropy/<variant_id>/
"""

import json, os, csv, platform, cupy as cp, numpy as np, matplotlib.pyplot as plt
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
# --- Load hierarchical configuration
# ---------------------------------------------------------------------------
cfg, ROOT = load_config("tier3_entropy.json")
params = cfg["parameters"]
tol = cfg.get("tolerances", {})

OUT_BASE = Path(cfg["base_paths"]["results"]) / "Tier3" / "Entropy"
OUT_BASE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# --- GPU and precision setup
# ---------------------------------------------------------------------------
if cfg["hardware"].get("gpu_enabled", True):
    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
precision = cfg["hardware"].get("precision", "float64")
dtype = cp.float64 if precision == "float64" else cp.float32
cp.set_printoptions(precision=6, suppress=True)

# ---------------------------------------------------------------------------
# --- Utilities
# ---------------------------------------------------------------------------
def ensure_dirs(p): p.mkdir(parents=True, exist_ok=True)
def write_csv(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def entropy_shannon(arr, bins=128):
    a = np.asarray(arr).ravel()
    hist, edges = np.histogram(a, bins=bins, density=True)
    p = hist / (np.sum(hist)+1e-30)
    p = p[p>0]
    return float(-np.sum(p*np.log(p)))

# ---------------------------------------------------------------------------
# --- Variant executor
# ---------------------------------------------------------------------------
def run_variant(params, variant, out_dir):
    ensure_dirs(out_dir)
    c = float(params.get("c",1.0))
    dt = float(params["time_step"]); dx = float(params["space_step"])
    N  = int(params.get("N",256))
    steps = int(variant.get("steps",20000))
    noise = float(variant.get("noise_level",0.0))
    r = (c*dt/dx)**2

    rs = cp.random.RandomState(1234)
    E = 0.1*(rs.rand(N,N).astype(dtype)-0.5)
    E_prev = E.copy()
    E0 = float(cp.sum(E**2))

    energy_hist, entropy_hist = [], []

    for t in range(steps):
        lap = (cp.roll(E,1,0)+cp.roll(E,-1,0)+cp.roll(E,1,1)+cp.roll(E,-1,1)-4*E)
        E_next = 2*E - E_prev + r*lap
        if noise>0:
            E_next += noise*(rs.rand(N,N).astype(dtype)-0.5)
        E_next *= cp.sqrt(E0/(cp.sum(E_next**2)+1e-30))
        E_prev, E = E, E_next

        if t % 200 == 0:
            e = float(cp.sum(E**2))
            energy_hist.append((t, e))
            entropy_hist.append((t, entropy_shannon(cp.asnumpy(E), bins=128)))

    energy_np = np.array(energy_hist)
    entropy_np = np.array(entropy_hist)

    # -----------------------------------------------------------------------
    # --- Plots
    # -----------------------------------------------------------------------
    ensure_dirs(out_dir/"plots")

    plt.figure(figsize=(6,4))
    plt.plot(energy_np[:,0], energy_np[:,1]/energy_np[0,1]); plt.grid(True)
    plt.xlabel("step"); plt.ylabel("E/E0"); plt.title("Energy drift (normalized)")
    plt.tight_layout(); plt.savefig(out_dir/"plots"/"energy.png", dpi=200); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(entropy_np[:,0], entropy_np[:,1]); plt.grid(True)
    plt.xlabel("step"); plt.ylabel("Shannon entropy")
    plt.title(f"Entropy (noise={noise})")
    plt.tight_layout(); plt.savefig(out_dir/"plots"/"entropy.png", dpi=200); plt.close()

    # -----------------------------------------------------------------------
    # --- HTML + JSON outputs
    # -----------------------------------------------------------------------
    html = f"""
    <html><body><h2>Tier-3 Entropy — {variant.get('description','variant')}</h2>
    <ul><li>Final entropy: {entropy_np[-1,1]:.3f}</li></ul>
    <img src='plots/energy.png' width='520'/>
    <img src='plots/entropy.png' width='520'/>
    </body></html>
    """
    (out_dir/"summary_dashboard.html").write_text(html, encoding="utf-8")

    summary = {
        "test_id": "T3_Entropy",
        "variant": variant.get("description","variant"),
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "hardware": {
            "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "python": platform.python_version()
        },
        "parameters": {
            "N": N, "steps": steps, "dt": dt, "dx": dx, "c": c,
            "r": (c*dt/dx)**2, "noise": noise
        },
        "metrics": {
            "final_entropy": float(entropy_np[-1,1])
        }
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[Entropy] {variant.get('description','variant')}: final entropy={entropy_np[-1,1]:.3f}")

# ---------------------------------------------------------------------------
# --- Main
# ---------------------------------------------------------------------------
def main():
    variants = cfg["variants"]
    for i, v in enumerate(variants, 1):
        out_dir = OUT_BASE / f"{i:02d}_{v.get('description','variant').replace(' ','_')}"
        run_variant(params, v, out_dir)

if __name__ == "__main__":
    main()
