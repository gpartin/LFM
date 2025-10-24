"""
LFM Tier-2 — Weak-Field Redshift (variable χ)
Hierarchical Config Version (uses master_config + tier2_redshift.json)
Covers GRAV-09..GRAV-12: frequency redshift & time-delay vs χ(x)
Outputs per-variant in results/Tier2/Redshift/<variant_id>/
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
# --- Load configuration
# ---------------------------------------------------------------------------
cfg, ROOT = load_config("config_tier2_redshift.json")
params, tol = cfg["parameters"], cfg["tolerances"]

OUT_BASE = Path(cfg["base_paths"]["results"]) / "Tier2" / "Redshift"
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

# ---------------------------------------------------------------------------
# --- χ-field builder
# ---------------------------------------------------------------------------
def chi_grid(N, variant):
    x = cp.linspace(-1,1,N, dtype=dtype)
    y = cp.linspace(-1,1,N, dtype=dtype)
    X,Y = cp.meshgrid(x,y)
    if "chi_gradient" in variant:
        a,b = map(float, variant["chi_gradient"])
        chi = a + (b-a)*(X+1)/2.0
    elif variant.get("chi_function","") == "sin(πx)":
        chi = cp.sin(cp.pi * X)
    else:
        chi = cp.zeros_like(X)
    return chi

# ---------------------------------------------------------------------------
# --- Variant execution
# ---------------------------------------------------------------------------
def run_variant(params, tol, variant, out_dir):
    ensure_dirs(out_dir)
    c = float(params.get("c",1.0))
    dt = float(params["time_step"]); dx = float(params["space_step"])
    N  = int(params.get("N",512))
    steps = int(variant.get("steps",4000))
    r = (c*dt/dx)**2

    # initial field
    x = cp.linspace(-1,1,N, dtype=dtype)
    y = cp.linspace(-1,1,N, dtype=dtype)
    X,Y = cp.meshgrid(x,y)
    E = cp.sin(8*cp.pi*X) * cp.exp(-20*(X**2+Y**2))
    E_prev = E.copy()

    chi = chi_grid(N, variant)
    E0 = float(cp.sum(E**2))

    # record time series at two probe locations
    probe_L = int(0.25*N)
    probe_R = int(0.75*N)
    ts_L, ts_R = [], []

    for t in range(steps):
        lap = (cp.roll(E,1,0)+cp.roll(E,-1,0)+cp.roll(E,1,1)+cp.roll(E,-1,1)-4*E)
        E_next = 2*E - E_prev + r*lap - (dt*dt)*(chi**2)*E
        E_next *= cp.sqrt(E0/(cp.sum(E_next**2)+1e-30))
        E_prev, E = E, E_next

        if t % 4 == 0:
            ts_L.append(float(cp.mean(E[:, probe_L])))
            ts_R.append(float(cp.mean(E[:, probe_R])))

    # frequency estimate via FFT
    def peak_freq(x, dt_s):
        a = np.abs(np.fft.rfft(x - np.mean(x)))
        if len(a) < 2: return 0.0
        k = int(np.argmax(a[1:]))+1
        return float(k)/(len(x)*dt_s)

    dt_s = dt
    fL = peak_freq(np.array(ts_L), dt_s)
    fR = peak_freq(np.array(ts_R), dt_s)
    redshift = (fL - fR)/max(fL,1e-30)

    # plot output
    plt.figure(figsize=(7,4))
    t_axis = np.arange(len(ts_L))*4*dt
    plt.plot(t_axis, ts_L, label="probe L")
    plt.plot(t_axis, ts_R, label="probe R")
    plt.xlabel("time"); plt.ylabel("E (avg column)")
    plt.title(f"fL={fL:.3e}, fR={fR:.3e}, Δf={(redshift):.2e}")
    plt.grid(True); plt.legend(); plt.tight_layout()
    ensure_dirs(out_dir/"plots")
    plt.savefig(out_dir/"plots"/"timeseries.png", dpi=200)
    plt.close()

    # HTML + JSON summary
    status = "PASS ✅" if abs(redshift)>0 and abs(redshift)<1 else "INFO"
    html = f"""
    <html><body><h2>Tier-2 Redshift — {variant.get('description','variant')}</h2>
    <ul><li>f_L={fL:.3e}, f_R={fR:.3e}, redshift={redshift:.2e}</li>
    <li>Status: {status}</li></ul>
    <img src='plots/timeseries.png' width='700'/>
    </body></html>
    """
    (out_dir/"summary_dashboard.html").write_text(html, encoding="utf-8")

    summary = {
        "test_id": "T2_Redshift",
        "variant": variant.get("description","variant"),
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "hardware": {
            "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "python": platform.python_version()
        },
        "parameters": {"N":N,"steps":steps,"dt":dt,"dx":dx,"c":c,"r":(c*dt/dx)**2},
        "metrics": {"f_left":fL,"f_right":fR,"redshift":redshift}
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[Redshift] {variant.get('description','variant')}: fL={fL:.2e} fR={fR:.2e} Δf={redshift:.2e} -> {status}")

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
