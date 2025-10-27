#!/usr/bin/env python3
"""
LFM Tier-1 — Lorentz Isotropy & Dispersion
3-D leapfrog KG (χ=0) plane-wave runs across multiple directions.
Measures phase velocity, anisotropy CoV, and energy-drift.
Outputs per-variant go to: results/Tier1/Dispersion/<variant_id>/
"""

import json, csv, math, platform, time
from pathlib import Path
from datetime import datetime, timezone

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config loader (same pattern as Tier-4)
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

CFG, ROOT = load_config("config_tier1_lorentz_dispersion.json")

BASE_RESULTS = Path(CFG["base_paths"]["results"]) / "Tier1" / "Dispersion"
BASE_RESULTS.mkdir(parents=True, exist_ok=True)

# Hardware setup
if CFG["hardware"].get("gpu_enabled", True):
    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

precision = CFG["hardware"].get("precision", "float64")
DTYPE = cp.float64 if precision == "float64" else cp.float32
cp.set_printoptions(precision=6, suppress=True)

PARAM = CFG["parameters"]
TOL   = CFG["tolerances"]

# Normalize tolerance keys for portability (Windows JSON + Greek delta)
if "Δv_over_c_max" in TOL and "dv_over_c_max" not in TOL:
    TOL["dv_over_c_max"] = TOL.pop("Δv_over_c_max")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def centered_coords(N, dx):
    # Periodic grid in [-L/2, L/2], spacing dx
    L = N * dx
    arr = cp.arange(N, dtype=DTYPE) * dx
    arr = arr - (L/2)
    return arr

def discrete_laplacian_3d(E, dx):
    return (cp.roll(E, 1, 0) + cp.roll(E, -1, 0) +
            cp.roll(E, 1, 1) + cp.roll(E, -1, 1) +
            cp.roll(E, 1, 2) + cp.roll(E, -1, 2) - 6*E) / (dx*dx)

def grad_sq_3d(E, dx):
    # centered differences
    gx = (cp.roll(E,-1,0) - cp.roll(E,1,0)) / (2*dx)
    gy = (cp.roll(E,-1,1) - cp.roll(E,1,1)) / (2*dx)
    gz = (cp.roll(E,-1,2) - cp.roll(E,1,2)) / (2*dx)
    return gx*gx + gy*gy + gz*gz

def total_energy(En, Em, dt, c, dx):
    # Discrete energy density: ½[(Δ_t E)^2 + c^2 |∇E|^2] (χ=0)
    dEdt = (En - Em) / dt
    g2   = grad_sq_3d(En, dx)
    eps  = 0.5*(dEdt*dEdt + (c*c)*g2)
    return float(cp.sum(eps))

def make_unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n

def make_plane_wave(N, dx, direction, cycles):
    """
    Build E(x)=sin(k·x) with a given integer number of cycles across the box.
    k = 2π * cycles / L ;  L = N*dx
    """
    ux = make_unit(direction)
    L  = N*dx
    k  = 2.0*np.pi*cycles / L
    # Build coordinate grid aligned to axes, then phase = k*(ux·r)
    x = centered_coords(N, dx); y = centered_coords(N, dx); z = centered_coords(N, dx)
    X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")
    phase = k*(ux[0]*X + ux[1]*Y + ux[2]*Z)
    E0 = cp.sin(phase, dtype=DTYPE)
    return E0, k

def estimate_frequency_tfft(trace, dt):
    """
    Estimate dominant ω via temporal FFT at a probe.
    """
    tr = np.asarray(trace, dtype=np.float64)
    tr = tr - tr.mean()
    spec = np.fft.rfft(tr, n=len(tr))
    freqs = np.fft.rfftfreq(len(tr), d=dt)
    idx = np.argmax(np.abs(spec))
    f = freqs[idx]
    return 2.0*np.pi*f  # ω

# ---------------------------------------------------------------------------
# Per-variant execution
# ---------------------------------------------------------------------------
def run_variant(variant, out_dir: Path):
    ensure_dirs(out_dir)
    # Parameters (kept consistent with Tier-4 style)
    N   = int(variant.get("N", PARAM.get("N", 128)))
    dx  = float(PARAM["space_step"])
    dt  = float(PARAM["time_step"])
    c   = float(PARAM.get("c", math.sqrt(PARAM["alpha"]/PARAM["beta"])))
    steps = int(variant.get("steps", PARAM.get("steps", 2000)))
    cycles = int(variant.get("cycles", PARAM.get("cycles", 4)))  # cycles across domain
    # Stability (CFL): c*dt/dx ≤ 1/√3
    cfl = c*dt/dx
    print(f"CFL ratio = {cfl:.4f} (must be <= 0.577)")
    if cfl > 1/math.sqrt(3):
        raise RuntimeError(f"CFL violation: c*dt/dx={cfl:.3f} > 1/sqrt(3) ≈ {1/math.sqrt(3):.3f}")

    directions = variant.get("directions", PARAM["directions"])

    # Result containers
    rows = []
    vph_list = []
    drifts   = []

    # Central probe index
    mid = (N//2, N//2, N//2)

    for d in directions:
        # --- Proper traveling-wave initialization (exactly aligned with make_plane_wave) ---
        E1, k = make_plane_wave(N, dx, d, cycles)
        omega_guess = c * k                      # dispersion for χ = 0
        phase_shift = -omega_guess * dt

        ux = make_unit(d)
        x = centered_coords(N, dx); y = centered_coords(N, dx); z = centered_coords(N, dx)
        X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")
        phase = k * (ux[0]*X + ux[1]*Y + ux[2]*Z)

        # Backward phase shift → consistent E(t-Δt) for a propagating wave
        E0 = cp.sin(phase + phase_shift, dtype=DTYPE)
        # --- end initialization ---

        # Energy baseline
        E_init = total_energy(E1, E0, dt, c, dx)
        probe = []

        # Leapfrog update (χ=0): E^{n+1} = 2E^n - E^{n-1} + dt^2 * c^2 ∇^2 E^n
        for n in range(steps):
            lap = discrete_laplacian_3d(E1, dx)
            Enext = 2*E1 - E0 + (dt*dt)*(c*c)*lap

            # Collect probe every step (for clean FFT)
            probe.append(float(Enext[mid]))

            E0, E1 = E1, Enext

        # Metrics
        E_final = total_energy(E1, E0, dt, c, dx)
        drift = abs(E_final - E_init)/max(1e-30, E_init)

        omega = estimate_frequency_tfft(probe, dt)
        v_phase = float(omega / k) if k > 0 else float("nan")

        vph_list.append(v_phase)
        drifts.append(drift)
        rows.append({
            "direction": d, "k": k, "omega": omega,
            "v_phase": v_phase, "energy_drift": drift
        })

    # Aggregate metrics
    v_arr = np.array(vph_list, dtype=float)
    cov = float(v_arr.std() / v_arr.mean())
    mean_v = float(v_arr.mean())
    dv_over_c_max = float(np.max(np.abs(v_arr - c))/c)

    max_drift = float(np.max(drifts))
    pass_iso  = cov <= TOL["CoV_max"]
    pass_v    = dv_over_c_max <= TOL["dv_over_c_max"]
    pass_E    = max_drift <= TOL["energy_drift_max"]
    passed = bool(pass_iso and pass_v and pass_E)

    # CSV
    csv_path = out_dir / "per_direction_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dir_x","dir_y","dir_z","k","omega","v_phase","energy_drift"])
        for r in rows:
            dx_, dy_, dz_ = r["direction"]
            w.writerow([dx_, dy_, dz_, r["k"], r["omega"], r["v_phase"], r["energy_drift"]])

    # Plot: phase velocity by direction index
    ensure_dirs(out_dir / "plots")
    plt.figure(figsize=(6,4))
    plt.plot(vph_list, marker="o")
    plt.axhline(c, linestyle="--")
    plt.xlabel("direction index")
    plt.ylabel("phase velocity")
    plt.title("Phase velocity vs. direction")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "v_phase_by_direction.png", dpi=200)
    plt.close()

    # HTML summary
    html = f"""
    <html><body><h2>Tier-1 Lorentz Isotropy & Dispersion — {variant.get('description','variant')}</h2>
    <ul>
      <li>Mean phase velocity: {mean_v:.6f} (target ≈ c={c:.6f})</li>
      <li>Max |v−c|/c: {dv_over_c_max:.6e} (≤ {TOL['dv_over_c_max']})</li>
      <li>Anisotropy CoV: {cov:.6e} (≤ {TOL['CoV_max']})</li>
      <li>Max energy drift: {max_drift:.3e} (≤ {TOL['energy_drift_max']})</li>
      <li>Status: {"PASS ✅" if passed else "FAIL ❌"}</li>
    </ul>
    <img src='plots/v_phase_by_direction.png' width='550'/>
    </body></html>
    """
    (out_dir / "summary_dashboard.html").write_text(html, encoding="utf-8")

    # JSON summary
    props = cp.cuda.runtime.getDeviceProperties(0)
    hw = {
        "gpu_name": props["name"].decode(),
        "compute_capability": float(f"{props['major']}.{props['minor']}"),
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "python": platform.python_version()
    }
    summary = {
        "test_id": "T1_LorentzDispersion",
        "variant": variant.get("description", "variant"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hw,
        "parameters": {
            "N": N, "dx": dx, "dt": dt, "steps": steps, "c": c,
            "cycles": cycles, "directions": directions
        },
        "metrics": {
            "v_phase_mean": mean_v,
            "dv_over_c_max": dv_over_c_max,
            "anisotropy_CoV": cov,
            "energy_drift_max": max_drift,
            "pass_isotropy": pass_iso,
            "pass_velocity": pass_v,
            "pass_energy": pass_E,
            "pass_overall": passed
        },
        "artifacts": {
            "csv": str(csv_path),
            "plot_png": str(out_dir / "plots" / "v_phase_by_direction.png"),
            "html_report": str(out_dir / "summary_dashboard.html")
        }
    }

    def sanitize(obj):
        import numpy as np
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=sanitize), encoding="utf-8")
    print(f"[Tier1] {variant.get('description','variant')}: "
          f"CoV={cov:.3e}, |v−c|/c_max={dv_over_c_max:.3e}, drift={max_drift:.3e} -> {'PASS' if passed else 'FAIL'}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    variants = CFG["variants"]
    for i, v in enumerate(variants, 1):
        out_dir = BASE_RESULTS / f"{i:02d}_{v.get('description','variant').replace(' ','_')}"
        run_variant(v, out_dir)

if __name__ == "__main__":
    main()
