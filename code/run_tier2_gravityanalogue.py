#!/usr/bin/env python3
"""
LFM Tier-2 — Gravity Analogue Suite (v2_13 canonical_localk_packdiag)

Canonical LFM equation (from LFM_Core_Equations.docx):
    ∂²E/∂t² = c²∇²E − χ(x,t)²E,  with  c² = α/β.

What’s new (diagnostics + measurement robustness only; NO physics change):
- ω measurement uses a *local spatial k* per probe (estimated on the probe’s spatial window),
  instead of a single global k. This avoids mismatched carrier vs. window content.
- Each probe reports: phase-fit ω, FFT ω, phase R², residual RMS, mean|z|, and acceptance reason.
- Full candidate tables for low-|χ| and high-|χ| are saved in results/.../diagnostics/pack/.
- If no probes pass strict thresholds, we relax (documented in meta.json) and always pick the
  best available pair so the suite produces actionable evidence rather than failing silently.
- Extra timing + backend prints so we can correlate runtime with CuPy/NumPy usage.

Pass rule is unchanged in spirit: small redshift (per config) + strong χ–envelope correlation.
"""

import json, math, csv, time
from pathlib import Path
from datetime import datetime
import numpy as np

# Optional GPU
try:
    import cupy as cp  # type: ignore
    GPU_BACKEND = f"CuPy backend: {cp.cuda.runtime.getDeviceProperties(0)['name']!r}"
except Exception:
    import numpy as cp  # type: ignore
    GPU_BACKEND = "CuPy unavailable — using NumPy (CPU)"
if not hasattr(cp, "asnumpy"):
    cp.asnumpy = lambda x: x

from lfm_console import log, suite_summary
from lfm_logger import LFMLogger
from lfm_diagnostics import field_spectrum, energy_total, energy_flow, phase_corr
from lfm_visualizer import visualize_concept
from lfm_results import save_summary


# ------------------------ helpers ------------------------
def load_config():
    script = Path(__file__).stem.replace("run_", "")
    base_dir = Path(__file__).resolve().parent
    for root in (base_dir, base_dir.parent):
        cfg_path = root / "config" / f"config_{script}.json"
        if cfg_path.is_file():
            return json.loads(cfg_path.read_text(encoding="utf-8"))
    raise FileNotFoundError("Config file not found.")


def resolve_outdir(output_dir_hint: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    outdir = project_root / output_dir_hint
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# ------------------------ χ profile ------------------------
def make_chi_profile(N, dx, variant, launch_dir, center_x):
    """
    Default orientation: χ DECREASES along launch direction ("ahead_lower").
    Override: set variant["chi_orient"] = "ahead_higher".
    """
    mode = variant.get("chi_profile", "linear")
    chi0 = float(variant.get("chi0", 0.0))
    grad_raw = float(variant.get("chi_grad", 0.01))
    orient = str(variant.get("chi_orient", "ahead_lower")).lower()  # "ahead_lower" | "ahead_higher"

    if orient == "ahead_lower":
        slope = -abs(grad_raw) if launch_dir == "right" else +abs(grad_raw)
    else:
        slope = +abs(grad_raw) if launch_dir == "right" else -abs(grad_raw)

    x = cp.arange(N) * dx
    if mode == "linear":
        chi = chi0 + slope * (x - center_x)
    elif mode == "gaussian":
        w = float(variant.get("chi_width", 0.2))
        shift = (-1 if launch_dir == "right" else +1) * 0.15 * (N * dx)
        if orient == "ahead_higher":
            shift = -shift
        chi = chi0 + grad_raw * cp.exp(-((x - (0.5 * N * dx + shift)) ** 2) / (2 * w ** 2))
    elif mode == "step":
        mid = 0.5 * N * dx
        if (launch_dir == "right" and orient == "ahead_lower") or (launch_dir == "left" and orient == "ahead_higher"):
            chi = chi0 + grad_raw * (x < mid)
        else:
            chi = chi0 + grad_raw * (x > mid)
    else:
        chi = cp.full(N, chi0)
    return chi, float(slope)


# ------------------------ spectral utils ------------------------
def _estimate_k_ang_from_snapshot(E_np, dx):
    """Dominant spatial wavenumber (angular) from whole-domain snapshot (fallback/global)."""
    e = E_np - np.mean(E_np)
    if np.allclose(e, 0.0):
        return 0.0
    w = np.hanning(len(e))
    F = np.fft.rfft(e * w)
    amp = np.abs(F)
    pk = int(np.argmax(amp[1:])) + 1 if len(amp) > 1 else 0
    if pk <= 0:
        return 0.0
    if 1 <= pk < len(amp) - 1:
        s1, s2, s3 = np.log(amp[pk-1] + 1e-30), np.log(amp[pk] + 1e-30), np.log(amp[pk+1] + 1e-30)
        denom = s1 - 2*s2 + s3
        delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (s1 - s3) / denom
    else:
        delta = 0.0
    df = 1.0 / (len(e) * dx)
    f_peak = (pk + delta) * df
    return 2.0 * math.pi * abs(f_peak)


def _estimate_local_k_ang(snapshot_np, i_center, half_w, dx):
    """Local k (angular) estimated on the same spatial window used for projection."""
    L = len(snapshot_np)
    i0, i1 = max(0, i_center - half_w), min(L, i_center + half_w + 1)
    seg = snapshot_np[i0:i1].astype(np.float64)
    seg = seg - seg.mean()
    if len(seg) < 8 or np.allclose(seg, 0.0):
        return 0.0
    w = np.hanning(len(seg))
    F = np.fft.rfft(seg * w)
    amp = np.abs(F)
    pk = int(np.argmax(amp[1:])) + 1 if len(amp) > 1 else 0
    if pk <= 0:
        return 0.0
    if 1 <= pk < len(amp) - 1:
        s1, s2, s3 = np.log(amp[pk-1] + 1e-30), np.log(amp[pk] + 1e-30), np.log(amp[pk+1] + 1e-30)
        denom = s1 - 2*s2 + s3
        delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (s1 - s3) / denom
    else:
        delta = 0.0
    df = 1.0 / (len(seg) * dx)
    f_peak = (pk + delta) * df
    return 2.0 * math.pi * abs(f_peak)


def _proj_series_window_k(E_series_np, i_center, half_w, k_ang, dx):
    """Complex projection time series using windowed inner product vs exp(i k x) with supplied k."""
    n_frames = len(E_series_np)
    L = E_series_np[0].shape[0]
    i0, i1 = max(0, i_center - half_w), min(L, i_center + half_w + 1)
    if i1 - i0 < 5 or k_ang == 0.0:
        return np.zeros(n_frames, dtype=np.complex128)
    x = (np.arange(i0, i1) * dx).astype(np.float64)
    cosk, sink = np.cos(k_ang * x), np.sin(k_ang * x)
    win = np.hanning(i1 - i0).astype(np.float64)
    z = np.zeros(n_frames, dtype=np.complex128)
    denom = (np.dot(cosk * win, cosk) + np.dot(sink * win, sink)) + 1e-30
    for t in range(n_frames):
        seg = (E_series_np[t][i0:i1] - np.mean(E_series_np[t][i0:i1])).astype(np.float64)
        a, b = float(np.dot(seg * win, cosk)), float(np.dot(seg * win, sink))
        z[t] = complex(a, b) / denom
    return z


def _omega_from_phase_fit(z, dt):
    """Linear fit on unwrapped phase; return ω, R², RMS residual, and mean amplitude."""
    if len(z) < 16:
        return 0.0, 0.0, float("inf"), 0.0
    phi = np.unwrap(np.angle(z)).astype(np.float64)
    t = np.arange(len(phi)) * dt
    w = np.hanning(len(phi)).astype(np.float64)
    A = np.vstack([t, np.ones_like(t)]).T
    Aw = A * w[:, None]
    yw = phi * w
    slope, intercept = np.linalg.lstsq(Aw, yw, rcond=None)[0]
    fit = slope * t + intercept
    resid = phi - fit
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((phi - np.mean(phi)) ** 2) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    amp = float(np.mean(np.abs(z)))
    return float(abs(slope)), float(max(min(r2, 1.0), -1.0)), float(math.sqrt(ss_res / len(phi))), amp


def _omega_from_fft(z, dt):
    """FFT peak frequency of complex z(t)."""
    if len(z) < 16:
        return 0.0
    s = z - np.mean(z)
    s = s * np.hanning(len(s))
    Z = np.fft.rfft(s)
    f = np.fft.rfftfreq(len(s), d=dt)
    k = int(np.argmax(np.abs(Z[1:])) + 1) if len(Z) > 1 else 0
    if k <= 0:
        return 0.0
    # parabolic refine in magnitude
    if 1 <= k < len(Z) - 1:
        m1, m2, m3 = np.log(np.abs(Z[k-1]) + 1e-30), np.log(np.abs(Z[k]) + 1e-30), np.log(np.abs(Z[k+1]) + 1e-30)
        denom = m1 - 2*m2 + m3
        delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (m1 - m3) / denom
    else:
        delta = 0.0
    fpk = (k + delta) * f[1]  # rfftfreq spacing is uniform
    return float(2.0 * math.pi * abs(fpk))


# ------------------------ per-test ------------------------
def run_gravity_test(test_id, desc, params, tol, outdir, quick):
    t0 = time.time()
    N, dx, dt = int(params["grid_points"]), float(params["dx"]), float(params["dt"])
    alpha, beta = float(params["alpha"]), float(params["beta"])
    c = math.sqrt(alpha / beta)

    SAVE_EVERY = int(params.get("save_every", 1))
    steps = int(params.get("steps", 12000 if quick else 16000))
    gamma = float(params.get("gamma_damp", 5e-6))
    pulse_w, amp = float(params.get("pulse_width", 0.06)), float(params.get("pulse_amp", 3.0))
    center = float(params.get("init_center", 0.25))
    launch_dir = str(params.get("launch_direction", "right")).lower()
    chi_mass = float(params.get("chi_mass", 0.0))

    # Probe selection thresholds (logged to meta.json)
    q_min_strict = float(params.get("probe_q_min", 0.25))
    env_min_frac = float(params.get("probe_env_min_frac", 0.15))
    pair_tol = float(params.get("probe_pair_tolerance", 1.7))  # |ω_phase - ω_fft| <= pair_tol * median(|ω|)
    # Adaptive relax values if nothing passes:
    q_min_relax = float(params.get("probe_q_min_relax", 0.05))
    env_min_relax = float(params.get("probe_env_min_relax", 0.05))

    L = N * dx
    center_x = center * L if center <= 1.0 else center

    test_dir = outdir / test_id
    diag_dir, plot_dir = test_dir / "diagnostics", test_dir / "plots"
    pack_dir = diag_dir / "pack"
    for d in (test_dir, diag_dir, plot_dir, pack_dir): d.mkdir(parents=True, exist_ok=True)

    log(f"→ {test_id}: {desc}", "INFO")
    print(f"[{test_id}] backend={GPU_BACKEND}")

    chi_spatial, chi_slope = make_chi_profile(N, dx, params, launch_dir, center_x)
    chi_eff = chi_spatial + chi_mass

    edge_frac = 0.02
    mask = cp.ones(N)
    edge_n = int(edge_frac * N)
    if edge_n > 4:
        taper = cp.hanning(2 * edge_n)
        mask[:edge_n] = taper[:edge_n]
        mask[-edge_n:] = taper[-edge_n:]

    xg = cp.arange(N) * dx
    E0 = amp * cp.exp(-((xg - center_x) ** 2) / (2 * pulse_w ** 2))
    dE_dx = (cp.roll(E0, -1) - cp.roll(E0, 1)) / (2 * dx)
    dE_dt = (-math.sqrt(c) if launch_dir == "right" else math.sqrt(c)) * dE_dx
    E_prev = E0 - dt * dE_dt
    E = E0.copy()
    E_series = [E.copy()]

    for n in range(steps):
        lap = (cp.roll(E, -1) - 2 * E + cp.roll(E, 1)) / (dx ** 2)
        E_next = (2 - gamma) * E - (1 - gamma) * E_prev + (dt ** 2) * ((c ** 2) * lap - (chi_eff ** 2) * E)
        E_next *= mask
        if n % SAVE_EVERY == 0:
            E_series.append(E_next.copy())
        E_prev, E = E, E_next
        if (n % 2000 == 0) and n > 0:
            Etot = energy_total(E, E_prev, dt, dx, c, float(cp.mean(chi_eff)))
            log(f"[{test_id}] Energy step {n}: {Etot:.6e}", "INFO")

    # Standard diagnostics
    field_spectrum(E, dx, diag_dir)
    energy_flow(E_series, dt, dx, c, diag_dir)
    phase_corr(E_series, diag_dir)
    visualize_concept(E_series, [chi_spatial], tier=2, test_id=test_id, outdir=plot_dir, quick=quick, animate=True)

    # ---------- Robust metrics ----------
    E_series_np = [cp.asnumpy(F).astype(np.float64) for F in E_series]
    start_idx = len(E_series_np) // 2
    E_tail = E_series_np[start_idx:]
    env = np.mean([np.abs(a) for a in E_tail], axis=0)

    interior_excl = int(0.10 * len(env))
    valid = np.ones_like(env, bool); valid[:interior_excl] = False; valid[-interior_excl:] = False
    chi_np = cp.asnumpy(chi_spatial).astype(float)
    abs_chi = np.abs(chi_np)

    # Candidate pools (interior only)
    idxs = np.where(valid)[0]
    abs_chi_valid = abs_chi[idxs]
    env_valid = env[idxs]
    # thresholds for low/high |χ| candidates (10/90 percentiles for breadth)
    lo_thr, hi_thr = np.percentile(abs_chi_valid, [10, 90])
    cand_low = idxs[abs_chi_valid <= lo_thr]
    cand_high = idxs[abs_chi_valid >= hi_thr]

    # Prepare evaluation over candidates
    snapshot = E_tail[0] if len(E_tail) else E_series_np[-1]
    half_w = max(6, int(0.02 * len(env)))  # ~2% domain

    def eval_candidate(i_center):
        k_loc = _estimate_local_k_ang(snapshot, int(i_center), half_w, dx)
        z = _proj_series_window_k(E_tail, int(i_center), half_w, k_loc, dx)
        dt_s = SAVE_EVERY * dt
        w_phase, r2, rms, amp_mean = _omega_from_phase_fit(z, dt_s)
        w_fft = _omega_from_fft(z, dt_s)
        # vote on acceptance
        env_here = float(env[int(i_center)])
        env_thr = env_min_frac * float(np.max(env_valid) + 1e-30)
        pair_med = np.median([w for w in (w_phase, w_fft) if w > 0]) if (w_phase > 0 or w_fft > 0) else 0.0
        pair_ok = (pair_med > 0) and (abs(w_phase - w_fft) <= pair_tol * max(pair_med, 1e-30))
        q_ok = (r2 >= q_min_strict)
        env_ok = (env_here >= env_thr)
        accepted = int(q_ok and env_ok and pair_ok and (w_phase > 0))
        reason = []
        if not q_ok: reason.append("q")
        if not env_ok: reason.append("env")
        if not pair_ok: reason.append("pair")
        if w_phase <= 0: reason.append("omega")
        return {
            "idx": int(i_center),
            "abs_chi": float(abs_chi[int(i_center)]),
            "env": env_here,
            "q": float(r2),
            "omega_phase": float(w_phase),
            "omega_fft": float(w_fft),
            "omega_pred": float(k_loc),   # local dispersion expects ω≈sqrt(c^2 k^2 + χ^2); here we log k_loc
            "accepted": accepted,
            "reason": ";".join(reason) if reason else "",
            "thr_lo": float(lo_thr),
            "thr_hi": float(hi_thr),
        }

    # Evaluate all candidates
    rows_low = [eval_candidate(i) for i in cand_low]
    rows_high = [eval_candidate(i) for i in cand_high]

    # Save candidate tables
    import pandas as pd
    pd.DataFrame(rows_low).to_csv(pack_dir / "candidates_low.csv", index=False)
    pd.DataFrame(rows_high).to_csv(pack_dir / "candidates_high.csv", index=False)

    # Pick best accepted by highest q, then env; if none accepted, relax thresholds & pick best overall
    def pick(rows, which):
        # strict pass
        acc = [r for r in rows if r["accepted"] == 1]
        used_relax = False
        if not acc:
            # relax: q_min -> q_min_relax; env_min -> env_min_relax; recompute acceptance flags
            def relax_accept(r):
                env_thr = env_min_relax * float(np.max(env_valid) + 1e-30)
                pair_med = np.median([w for w in (r["omega_phase"], r["omega_fft"]) if w > 0]) if (r["omega_phase"] > 0 or r["omega_fft"] > 0) else 0.0
                pair_ok = (pair_med > 0) and (abs(r["omega_phase"] - r["omega_fft"]) <= pair_tol * max(pair_med, 1e-30))
                return (r["q"] >= q_min_relax) and (r["env"] >= env_thr) and pair_ok and (r["omega_phase"] > 0)
            acc = [r for r in rows if relax_accept(r)]
            used_relax = True
        # if still none, just take the single highest-q row so run remains informative
        if not acc:
            used_relax = True
            acc = [max(rows, key=lambda r: (r["q"], r["env"]))]
        # choose best
        acc.sort(key=lambda r: (r["q"], r["env"]), reverse=True)
        pick = acc[0]
        return pick, used_relax

    pick_low, relaxed_low = pick(rows_low, "low")
    pick_high, relaxed_high = pick(rows_high, "high")

    # Build complex series & residual dumps for the chosen pair (for forensic plots)
    def dump_probe_pack(tag, pick_row):
        i_center = pick_row["idx"]
        k_loc = _estimate_local_k_ang(snapshot, i_center, half_w, dx)
        z = _proj_series_window_k(E_tail, i_center, half_w, k_loc, dx)
        dt_s = SAVE_EVERY * dt
        w_phase, r2, rms, amp_mean = _omega_from_phase_fit(z, dt_s)
        # save time-phi-resid
        phi = np.unwrap(np.angle(z)).astype(np.float64)
        tvec = np.arange(len(phi)) * dt_s
        fit = (w_phase * tvec) + (phi[0] - w_phase * tvec[0])
        with open(pack_dir / f"z_{tag}_phi_resid.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["t","phase","phase_fit","resid"])
            for t, ph, fh in zip(tvec, phi, fit):
                w.writerow([t, ph, ph - fh])
        # save the spatial window content for visibility
        i0, i1 = max(0, i_center - half_w), min(len(env), i_center + half_w + 1)
        with open(pack_dir / f"window_{tag}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["i","x","E_last"])
            xs = np.arange(i0, i1) * dx
            for i, x in zip(range(i0, i1), xs):
                w.writerow([i, x, E_tail[-1][i]])

    dump_probe_pack("low", pick_low)
    dump_probe_pack("high", pick_high)

    # Compute final ω using the chosen rows (phase-fit ω)
    omega_low = float(pick_low["omega_phase"])
    omega_high = float(pick_high["omega_phase"])
    redshift = None
    if omega_low > 0 and omega_high > 0:
        redshift = 1.0 - (omega_low / (omega_high + 1e-30))

    # Save minimal quick CSVs for triage
    with open(pack_dir / "env.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["i","env"]); [w.writerow([i, float(env[i])]) for i in range(len(env))]
    with open(pack_dir / "chi_profile.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["i","chi","abs_chi"]); 
        for i in range(len(chi_np)):
            w.writerow([i, float(chi_np[i]), float(abs_chi[i])])

    # meta.json — everything we need to debug selection behavior
    meta = {
        "id": test_id,
        "dt": dt, "dx": dx,
        "backend": GPU_BACKEND,
        "k_ang_global": float(_estimate_k_ang_from_snapshot(E_tail[0], dx)),
        "half_window": int(half_w),
        "chi_slope": float(chi_slope),
        "q_min_strict": float(q_min_strict),
        "q_min_relax": float(q_min_relax),
        "env_min_frac": float(env_min_frac),
        "env_min_relax": float(env_min_relax),
        "pair_tolerance": float(pair_tol),
        "lo_thr_abs_chi": float(lo_thr),
        "hi_thr_abs_chi": float(hi_thr),
        "i_low": int(pick_low["idx"]),
        "i_high": int(pick_high["idx"]),
        "omega_low": float(omega_low),
        "omega_high": float(omega_high),
        "q_low": float(pick_low["q"]),
        "q_high": float(pick_high["q"]),
        "env_low": float(pick_low["env"]),
        "env_high": float(pick_high["env"]),
        "omega_low_fft": float(pick_low["omega_fft"]),
        "omega_high_fft": float(pick_high["omega_fft"]),
        "relaxed_low_used": bool(relaxed_low),
        "relaxed_high_used": bool(relaxed_high),
        "redshift": (float(redshift) if redshift is not None else None)
    }
    (pack_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # χ–envelope correlation (interior, top quartile)
    thr = np.percentile(env[np.where(valid)[0]], 75)
    corridor = (env >= max(1e-9, thr)) & valid
    if np.sum(corridor) >= 5 and np.std(chi_np[corridor]) > 0 and np.std(env[corridor]) > 0:
        corr = float(np.corrcoef(chi_np[corridor], env[corridor])[0, 1])
    else:
        corr = 0.0
    corr_abs = abs(corr)

    # Pass rule
    z_tol = float(tol.get("redshift_max", 0.05))
    z_ok = (redshift is not None) and (-0.01 <= redshift <= z_tol)
    passed = (corr_abs > 0.90) and z_ok

    # Console summary
    print(f"[{test_id}] k_global={meta['k_ang_global']:.4g} half_w={half_w} "
          f"low(i={meta['i_low']} q={meta['q_low']:.3f} env={meta['env_low']:.3f} ω={omega_low:.4g}) "
          f"high(i={meta['i_high']} q={meta['q_high']:.3f} env={meta['env_high']:.3f} ω={omega_high:.4g}) "
          f"z={(redshift if redshift is not None else float('nan')):.3g} | corr={corr:.3f}")

    # Final log + summary
    log(f"{test_id} {'PASS ✅' if passed else 'FAIL ❌'} "
        f"(corr={corr:.3f} |corr|={corr_abs:.3f}, z={(redshift if redshift is not None else float('nan')):.4e})",
        "INFO" if passed else "FAIL")

    summary = {
        "id": test_id,
        "desc": desc,
        "passed": bool(passed),
        "corr": float(corr),
        "corr_abs": float(corr_abs),
        "redshift": (float(redshift) if redshift is not None else None),
        "omega_low": float(omega_low),
        "omega_high": float(omega_high),
        "runtime_steps": int(steps),
        "dt": float(dt),
        "dx": float(dx),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "runtime_sec": round(time.time() - t0, 3),
        "backend": GPU_BACKEND,
    }
    save_summary(test_dir, test_id, summary)
    return summary


def main():
    cfg = load_config()
    run = cfg["run_settings"]; p_base = cfg["parameters"]
    tol = cfg["tolerances"]; variants = cfg["variants"]

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
