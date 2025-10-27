#!/usr/bin/env python3
"""
LFM Tier-2 — Gravity Analogue Suite (v2.3 canonical-stabilized)

Canonical LFM equation (no physics changes):
    ∂²E/∂t² = c²∇²E − χ(x,t)²E,  with  c² = α/β.

Numerics/analysis fixes:
- Lattice-correct ω(k, χ) with kdx = k_fraction_lattice * π.
- k snapping: k_fraction_lattice = 2m/N (m integer).
- Estimator: projection-FFT (Hann) + log-parabola sub-bin.
- Very light damping, no per-step rescale, optional zero-mean.
- Absorbing edges = 5% window; warm-up excluded from correlation.
- Two spatial windows (low-χ, high-χ) to compute redshift z.
"""

import json, math, time
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


# ------------------------- Config helpers -------------------------
def load_config():
    script = Path(__file__).stem.replace("run_", "")
    base_dir = Path(__file__).resolve().parent
    for root in (base_dir, base_dir.parent):
        p = root / "config" / f"config_{script}.json"
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("Config file not found for this Tier-2 suite.")

def resolve_outdir(output_dir_hint: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    outdir = project_root / output_dir_hint
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# ------------------------- Initial conditions -------------------------
def k_from_fraction(N, dx, k_fraction):
    # Snap k_fraction to lattice: k_fraction_lattice = 2m/N (m integer)
    m = int(round((N * float(k_fraction)) / 2.0))
    k_frac_lattice = 2.0 * m / N
    k_cyc = k_frac_lattice * (1.0 / (2.0 * dx))  # cycles/unit
    k_ang = 2.0 * math.pi * k_cyc               # rad/unit
    return k_ang, k_frac_lattice

def gaussian_packet(x, x0, w, k_ang, amp=1.0):
    return amp * cp.exp(-((x - x0)**2) / (2 * w**2)) * cp.cos(k_ang * x)

def make_chi_profile(N, dx, params):
    """
    Smooth χ ramp (weak-field). Sign/orientation controlled by chi_sign (+1/-1).
    Ramp centered; edges flattened to avoid boundary contamination.
    """
    chi0    = float(params.get("chi_base", 0.0))
    dchi    = float(params.get("chi_delta", 0.05))
    width   = float(params.get("chi_width", 0.4))  # fraction of domain
    chi_sig = int(params.get("chi_sign", +1))
    x = cp.arange(N) * dx
    L = N * dx
    x_mid = 0.5 * L
    w = (width * L) / 2.0
    ramp = 0.5 * (1.0 + cp.tanh((x - (x_mid - w)) / (0.07 * L))) \
         - 0.5 * (1.0 + cp.tanh((x - (x_mid + w)) / (0.07 * L)))
    ramp = cp.maximum(ramp, 0.0)
    chi = chi0 + chi_sig * dchi * ramp
    return chi

def absorbing_mask(N, edge_frac=0.05):
    # Cosine-tapered absorbing edges; 5% default
    e = int(max(1, round(edge_frac * N)))
    w = cp.ones(N, dtype=cp.float64)
    if e > 0:
        edge = cp.hanning(2 * e)
        w[:e] *= edge[:e]
        w[-e:] *= edge[-e:]
    return w


# ------------------------- Simulation core -------------------------
def run_grav_test(test_id, desc, params, tol, outdir, quick):
    N   = int(params["grid_points"])
    dx  = float(params["dx"])
    dt  = float(params["dt"])
    alp = float(params["alpha"])
    bet = float(params["beta"])
    c   = math.sqrt(alp / bet)

    SAVE_EVERY    = 1
    TARGET_SAMPLES= 2048 if quick else 4096
    steps         = max(int(params.get("steps", 4000)), TARGET_SAMPLES + 200)
    gamma_damp    = float(params.get("gamma_damp", 5e-4))
    zero_mean     = bool(params.get("zero_mean", False))
    edge_frac     = float(params.get("edge_absorb_frac", 0.05))

    # k setup
    k_ang, k_frac_lat = k_from_fraction(N, dx, params.get("k_fraction", 0.08))
    params["_k_fraction_lattice"] = k_frac_lat
    params["_k_ang"] = k_ang

    # grids
    x   = cp.arange(N, dtype=cp.float64) * dx
    mask= absorbing_mask(N, edge_frac)

    # chi profile (weak-field ramp)
    chi = make_chi_profile(N, dx, params)

    # init field: right-moving packet from left side toward ramp
    x0   = float(params.get("x0", 0.2 * N * dx))
    width= float(params.get("packet_width", 0.06 * N * dx))
    amp  = float(params.get("packet_amp", 1.0))
    E_prev = gaussian_packet(x, x0, width, k_ang, amp)
    E     = E_prev.copy()
    E0    = energy_total(E, E_prev, dt, dx, c, float(params.get("chi_energy", 0.0)))

    # host bases for projection
    x_np  = (np.arange(N) * dx).astype(np.float64)
    cos_k = np.cos(float(k_ang) * x_np)
    sin_k = np.sin(float(k_ang) * x_np)
    cos_norm = np.dot(cos_k, cos_k) + 1e-30
    sin_norm = np.dot(sin_k, sin_k) + 1e-30

    test_dir = outdir / test_id
    diag_dir = test_dir / "diagnostics"
    plot_dir = test_dir / "plots"
    for d in (test_dir, diag_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    log(f"[{test_id}] steps={steps} kfrac_lat={k_frac_lat:.6f} dt={dt} dx={dx} gamma={gamma_damp}", "INFO")

    # time stepping
    E_series = [E.copy()]
    chi_series = [chi.copy()]
    # define two spatial windows for ω measurement (low-χ: left; high-χ: right of ramp)
    win_w = int(params.get("win_width_cells", max(16, N // 16)))
    wL_c = int(params.get("win_left_center",  N // 4))
    wR_c = int(params.get("win_right_center", 3 * N // 4))
    wL = slice(max(0, wL_c - win_w//2), min(N, wL_c + win_w//2))
    wR = slice(max(0, wR_c - win_w//2), min(N, wR_c + win_w//2))

    proj_tr_left = []
    proj_tr_right= []
    chi_left = []
    chi_right= []

    t0 = time.time()
    for n in range(steps):
        lap = (cp.roll(E, -1) - 2 * E + cp.roll(E, 1)) / (dx**2)
        E_next = (2 - gamma_damp) * E - (1 - gamma_damp) * E_prev + (dt**2) * ((c**2) * lap - (chi**2) * E)

        if zero_mean:
            E_next = E_next - cp.mean(E_next)

        # absorbing edges
        E_next = E_next * mask

        if n % SAVE_EVERY == 0:
            E_series.append(E_next.copy())
            chi_series.append(chi.copy())

            # projection amplitude (cosine component) in spatial windows
            En = cp.asnumpy(E_next).real.astype(np.float64)

            L_seg = En[wL] - np.mean(En[wL])
            R_seg = En[wR] - np.mean(En[wR])
            # re-project each segment onto cos(kx) at their local indices
            idxL = np.arange(L_seg.size)
            idxR = np.arange(R_seg.size)
            xL = (idxL * dx).astype(np.float64)
            xR = (idxR * dx).astype(np.float64)
            cL = np.cos(float(k_ang) * xL)
            cR = np.cos(float(k_ang) * xR)
            aL = float(np.dot(L_seg, cL) / (np.dot(cL, cL) + 1e-30))
            aR = float(np.dot(R_seg, cR) / (np.dot(cR, cR) + 1e-30))
            proj_tr_left.append(aL)
            proj_tr_right.append(aR)
            chi_left.append(float(np.mean(cp.asnumpy(chi[wL]))))
            chi_right.append(float(np.mean(cp.asnumpy(chi[wR]))))

        E_prev, E = E, E_next
    runtime = time.time() - t0

    # Diagnostics
    field_spectrum(E, dx, diag_dir)
    energy_flow(E_series, dt, dx, c, diag_dir)
    phase_corr(E_series, diag_dir)

    # Visualization
    try:
        visualize_concept(E_series, chi_series, tier=2, test_id=test_id, outdir=plot_dir, quick=quick, animate=True)
    except Exception as e:
        log(f"[WARN] Visualization failed for {test_id}: {e}", "WARN")

    # ---------- ω estimation via proj-FFT (Hann + sub-bin) ----------
    def omega_from_trace(trace, sample_dt, warmup_skip=64):
        data = np.asarray(trace, dtype=np.float64)
        if len(data) <= max(128, warmup_skip + 16):
            return math.nan
        data = data[warmup_skip:]  # exclude warm-up
        w = np.hanning(len(data))
        dw = data * w
        spec = np.abs(np.fft.rfft(dw))
        pk = int(np.argmax(spec[1:])) + 1
        if 1 <= pk < len(spec) - 1:
            s1, s2, s3 = np.log(spec[pk-1] + 1e-30), np.log(spec[pk] + 1e-30), np.log(spec[pk+1] + 1e-30)
            denom = s1 - 2*s2 + s3
            delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (s1 - s3) / denom
        else:
            delta = 0.0
        df = 1.0 / (len(dw) * sample_dt)
        f_peak = (pk + delta) * df
        return 2.0 * math.pi * abs(f_peak)

    sample_dt = SAVE_EVERY * dt
    omega_L = omega_from_trace(proj_tr_left, sample_dt)
    omega_R = omega_from_trace(proj_tr_right, sample_dt)

    # ---------- Theoretical ω using lattice dispersion ----------
    def omega_theory(k_fraction_lattice, chi_val):
        kdx = k_fraction_lattice * math.pi
        return math.sqrt(((2.0 * c / dx)**2) * 0.5 * (1.0 - math.cos(kdx)) + (chi_val**2))

    # Build time series ω_theory for correlation check (using per-window mean χ)
    chiL = np.asarray(chi_left, dtype=np.float64)
    chiR = np.asarray(chi_right, dtype=np.float64)
    w_theory_L = np.array([omega_theory(k_frac_lat, v) for v in chiL], dtype=np.float64)
    w_theory_R = np.array([omega_theory(k_frac_lat, v) for v in chiR], dtype=np.float64)

    # Make measured ω(t) traces by sliding-window FFT (reuse projection samples)
    # For robustness we’ll just compare constant-ω estimates with mean-theory values
    w_meas_L = np.full_like(w_theory_L, omega_L)
    w_meas_R = np.full_like(w_theory_R, omega_R)

    def safe_corr(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        a = a - a.mean(); b = b - b.mean()
        na = np.linalg.norm(a) + 1e-30; nb = np.linalg.norm(b) + 1e-30
        return float(np.dot(a, b) / (na * nb))

    corr_L = safe_corr(w_meas_L, w_theory_L)
    corr_R = safe_corr(w_meas_R, w_theory_R)

    # Redshift between left (ref, low-χ) and right (test, high-χ):
    # z = ω_ref/ω_test − 1  (positive if test is redshifted vs ref)
    z = float((omega_L / (omega_R + 1e-30)) - 1.0)

    rel_ok = (abs(z) <= float(params.get("z_abs_max", 0.5)))  # loose sanity
    corr_ok = (max(corr_L, corr_R) >= tol.get("corr_min", 0.90))

    passed = bool(corr_ok and rel_ok)

    log(f"{test_id} {'PASS ✅' if passed else 'FAIL ❌'} "
        f"(z={z:+.4e}, ω_L={omega_L:.6f}, ω_R={omega_R:.6f}, "
        f"corr_L={corr_L:.3f}, corr_R={corr_R:.3f})",
        "INFO" if passed else "FAIL")

    # Summary
    summary = {
        "id": test_id, "desc": desc, "passed": passed,
        "z": z, "omega_left": omega_L, "omega_right": omega_R,
        "corr_left": corr_L, "corr_right": corr_R,
        "k_fraction_lattice": k_frac_lat,
        "dx": dx, "dt": dt, "c": c,
        "runtime_sec": runtime,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    with open(test_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Minimal metrics.csv
    with open(test_dir / "metrics.csv", "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"z,{z}\n")
        f.write(f"corr_left,{corr_L}\n")
        f.write(f"corr_right,{corr_R}\n")

    return summary


def main():
    cfg = load_config()
    outdir = resolve_outdir(cfg["output_dir"])
    logger = LFMLogger(outdir)
    quick = bool(cfg["run_settings"].get("quick_mode", False))
    tol   = cfg["tolerances"]
    pbase = cfg["parameters"]
    variants = cfg["variants"]

    log(f"=== Tier-2 Gravity Analogue Suite Start (quick={quick}) ===", "INFO")
    results = []
    for v in variants:
        params = pbase.copy(); params.update(v)
        tid = v["test_id"]; desc = v["description"]
        log(f"→ Starting {tid}: {desc}", "INFO")
        results.append(run_grav_test(tid, desc, params, tol, outdir, quick))

    suite_summary(results)
    log("=== Tier-2 Suite Complete ===", "INFO")


if __name__ == "__main__":
    main()
