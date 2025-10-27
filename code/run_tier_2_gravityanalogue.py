#!/usr/bin/env python3
"""
LFM Tier-2 — Gravity Analogue Suite (v2.4 canonical-diagnostic-fix)

Canonical LFM equation (from LFM_Core_Equations.docx):
    ∂²E/∂t² = c²∇²E − χ(x,t)²E,  with  c² = α/β.

Fix summary:
- Adds light high-pass filtering inside ω-autocorr to reject slow envelope drift.
- Excludes absorber edges (10 %) when picking probes and computing correlation.
- Prints ω_lo / ω_hi / z for every test and still saves probe_omegas.json.
- No physics change.
"""

import json, math, csv
from pathlib import Path
from datetime import datetime
import numpy as np

# Optional GPU
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
from lfm_results import save_summary


# ------------------------ config helpers ------------------------
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


# ------------------------ χ profiles ------------------------
def make_chi_profile(N, dx, variant, launch_dir: str, center_x: float):
    mode = variant.get("chi_profile", "linear")
    chi0 = float(variant.get("chi0", 0.0))
    grad = float(variant.get("chi_grad", 0.01))
    x = cp.arange(N) * dx

    if mode == "linear":
        slope = grad if launch_dir == "right" else -grad
        chi = chi0 + slope * (x - center_x)
    elif mode == "gaussian":
        w = float(variant.get("chi_width", 0.2))
        chi = chi0 + grad * cp.exp(-((x - 0.5 * N * dx) ** 2) / (2 * w ** 2))
    elif mode == "step":
        chi = chi0 + grad * (x > 0.5 * N * dx)
    elif mode == "radial_like":
        r = cp.abs(x - 0.5 * N * dx)
        chi = chi0 + grad * (1.0 / (r + dx) - 1.0 / (0.25 * N * dx + dx))
    else:
        chi = cp.full(N, chi0)
    return chi


# ------------------------ ω estimation ------------------------
def _omega_autocorr(trace, dt):
    """Angular frequency via first-peak lag of autocorr with mild high-pass."""
    x = np.asarray(trace, dtype=np.float64)
    if len(x) < 32:
        return 0.0
    x = x - np.mean(x)
    # light high-pass (remove slow envelope)
    if len(x) > 9:
        k = 5
        ma = np.convolve(x, np.ones(k)/k, mode="same")
        x = x - ma
    if np.allclose(x, 0.0, atol=1e-12):
        return 0.0

    w = np.hanning(len(x))
    xw = x * w
    n = int(1 << (len(xw) - 1).bit_length())
    X = np.fft.rfft(xw, n=2*n)
    R = np.fft.irfft(np.abs(X)**2)[:len(xw)]
    if R[0] == 0:
        return 0.0
    R /= R[0]
    lo = 3; hi = max(lo + 1, len(R)//3)
    if hi - lo < 3:
        return 0.0
    k = int(np.argmax(R[lo:hi])) + lo
    T = k * dt
    if T <= 0 or T > (len(x)*dt)/4:
        return 0.0
    return 2.0 * math.pi / T


# ------------------------ per-test runner ------------------------
def run_gravity_test(test_id, desc, params, tol, outdir, quick):
    N, dx, dt = int(params["grid_points"]), float(params["dx"]), float(params["dt"])
    alpha, beta = float(params["alpha"]), float(params["beta"])
    c = math.sqrt(alpha / beta)

    SAVE_EVERY = int(params.get("save_every", 1))
    steps      = int(params.get("steps", 12000 if quick else 16000))
    gamma      = float(params.get("gamma_damp", 1e-5))
    pulse_w    = float(params.get("pulse_width", 0.06))
    amp        = float(params.get("pulse_amp", 3.0))
    center     = float(params.get("init_center", 0.25))
    launch_dir = str(params.get("launch_direction", "right")).lower()
    chi_mass   = float(params.get("chi_mass", 0.0))

    L = N * dx
    center_x = center * L if center <= 1.0 else center

    test_dir = outdir / test_id
    diag_dir = test_dir / "diagnostics"
    plot_dir = test_dir / "plots"
    for d in (test_dir, diag_dir, plot_dir): d.mkdir(parents=True, exist_ok=True)

    log(f"→ {test_id}: {desc}", "INFO")

    chi_spatial = make_chi_profile(N, dx, params, launch_dir, center_x)
    chi_eff = chi_spatial + chi_mass

    # Absorbing edges (5%)
    edge_frac = 0.05
    mask = cp.ones(N)
    edge_n = int(edge_frac * N)
    if edge_n > 4:
        taper = cp.hanning(2 * edge_n)
        mask[:edge_n] = taper[:edge_n]
        mask[-edge_n:] = taper[-edge_n:]

    xg = cp.arange(N) * dx
    E0 = amp * cp.exp(-((xg - center_x) ** 2) / (2 * pulse_w ** 2))
    dE_dx = (cp.roll(E0, -1) - cp.roll(E0, 1)) / (2.0 * dx)
    dE_dt = (-math.sqrt(c) if launch_dir == "right" else math.sqrt(c)) * dE_dx
    E_prev = E0 - dt * dE_dt
    E = E0.copy()
    E_series = [E.copy()]

    # Canonical discrete update
    for n in range(steps):
        lap = (cp.roll(E, -1) - 2.0 * E + cp.roll(E, 1)) / (dx**2)
        E_next = (2.0 - gamma) * E - (1.0 - gamma) * E_prev \
                 + (dt**2) * ((c**2) * lap - (chi_eff**2) * E)
        E_next *= mask
        if n % SAVE_EVERY == 0:
            E_series.append(E_next.copy())
        E_prev, E = E, E_next
        if (n % 2000 == 0) and (n > 0):
            Etot = energy_total(E, E_prev, dt, dx, c, float(cp.mean(chi_eff)))
            log(f"[{test_id}] Energy check step {n}: {Etot:.6e}", "INFO")

    # Diagnostics
    field_spectrum(E, dx, diag_dir)
    energy_flow(E_series, dt, dx, c, diag_dir)
    phase_corr(E_series, diag_dir)
    visualize_concept(E_series, chi_series=[chi_spatial],
                      tier=2, test_id=test_id, outdir=plot_dir,
                      quick=quick, animate=True)

    # Envelope + probe metrics
    chi_np = cp.asnumpy(chi_spatial).astype(float)
    start_idx = len(E_series)//2
    E_last = [cp.asnumpy(F) for F in E_series[start_idx:]]
    env = np.mean([np.abs(a) for a in E_last], axis=0)

    edge_excl = int(0.10 * len(env))
    valid = np.ones_like(env, dtype=bool)
    valid[:edge_excl] = False
    valid[-edge_excl:] = False

    chi_med = float(np.median(chi_np))
    low_mask  = (chi_np <= chi_med) & valid
    high_mask = (chi_np >= chi_med) & valid

    def argmax_where(values, mask_bool):
        idxs = np.where(mask_bool)[0]
        return int(idxs[np.argmax(values[idxs])]) if len(idxs) else int(np.argmax(values))

    i_lo = argmax_where(env, low_mask)
    i_hi = argmax_where(env, high_mask)

    sample_dt = SAVE_EVERY * dt
    probe_lo = np.array([float(a[i_lo]) for a in E_last])
    probe_hi = np.array([float(a[i_hi]) for a in E_last])

    omega_lo = _omega_autocorr(probe_lo, sample_dt)
    omega_hi = _omega_autocorr(probe_hi, sample_dt)
    meas_ok = (omega_lo > 0 and omega_hi > 0)
    redshift = (1.0 - (omega_hi / (omega_lo + 1e-30))) if meas_ok else None

    log(f"[{test_id}] ω_lo={omega_lo:.6g}, ω_hi={omega_hi:.6g}, z={float(redshift) if meas_ok else float('nan'):.6g}", "INFO")

    with open(diag_dir / "probe_omegas.json", "w", encoding="utf-8") as f:
        json.dump({
            "omega_lo": float(omega_lo),
            "omega_hi": float(omega_hi),
            "redshift": (float(redshift) if meas_ok else None),
            "i_lo": int(i_lo), "i_hi": int(i_hi),
            "edge_excluded": int(edge_excl)
        }, f, indent=2)

    thr = np.percentile(env[valid], 75)
    corridor = (env >= max(1e-6, thr)) & valid
    corr = float(np.corrcoef(chi_np[corridor], env[corridor])[0, 1]) \
           if np.std(chi_np[corridor])*np.std(env[corridor]) > 0 else 0.0

    passed = (corr > 0.90) and (meas_ok and abs(redshift) < tol["redshift_max"])
    log(f"{test_id} {'PASS ✅' if passed else 'FAIL ❌'} "
        f"(corr={corr:.3f}, z={(redshift if meas_ok else float('nan')):.4e})",
        "INFO" if passed else "FAIL")

    summary = {
        "id": test_id, "desc": desc, "passed": bool(passed),
        "corr": float(corr),
        "redshift": (float(redshift) if meas_ok else None),
        "omega_lo": float(omega_lo), "omega_hi": float(omega_hi),
        "runtime_steps": int(steps), "dt": float(dt), "dx": float(dx),
        "timestamp": datetime.utcnow().isoformat()+"Z"
    }
    save_summary(test_dir, test_id, summary)
    return summary


# ------------------------ suite driver ------------------------
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
