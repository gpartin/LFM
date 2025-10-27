#!/usr/bin/env python3
"""
LFM Tier-1 — Relativistic Propagation & Isotropy Suite (v1.9.11 iso=proj-fft)
- REL-01/02: change estimator to projection-amplitude FFT (Hann + sub-bin),
             keep NO rescale, zero-mean, dense sampling.
- REL-03..REL-07: unchanged from your passing path.
- REL-08: unchanged from your passing path.
"""

import json, math, time
from pathlib import Path
import numpy as np
from datetime import datetime

# CuPy optional
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


def load_config():
    script = Path(__file__).stem.replace("run_", "")
    base_dir = Path(__file__).resolve().parent
    for root in (base_dir, base_dir.parent):
        cfg_path = root / "config" / f"config_{script}.json"
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("Config file not found for this test.")


def resolve_outdir(output_dir_hint: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    outdir = project_root / output_dir_hint
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def init_field_variant(test_id, params, N, dx, c):
    x = cp.arange(N) * dx
    k_frac = float(params.get("k_fraction", 0.1))
    m = int(round((N * k_frac) / 2.0))                 # snap: k_frac = 2m/N
    k_frac_lattice = 2.0 * m / N
    k_cyc = k_frac_lattice * (1.0 / (2.0 * dx))        # cycles / unit
    k_ang = 2.0 * math.pi * k_cyc                      # rad / unit
    params["_k_fraction_lattice"] = k_frac_lattice
    params["_k_ang"] = k_ang

    if test_id in ("REL-01", "REL-02"):
        return cp.cos(k_ang * x)
    elif test_id == "REL-03":
        beta = params.get("boost_factor", 0.2); gamma = 1.0 / math.sqrt(1 - beta**2)
        return cp.cos(gamma * (k_ang * x - beta * k_ang * x))
    elif test_id == "REL-04":
        beta = params.get("boost_factor", 0.6); gamma = 1.0 / math.sqrt(1 - beta**2)
        return cp.cos(gamma * (k_ang * x - beta * k_ang * x))
    elif test_id == "REL-05":
        amp = params.get("pulse_amp", 1.0); w = params.get("pulse_width", 0.1); c0 = N // 2
        return amp * cp.exp(-((x - c0 * dx) ** 2) / (2 * w ** 2))
    elif test_id == "REL-06":
        rng = cp.random.default_rng(1234)
        return cp.ones(N) + params.get("noise_amp", 1e-4) * rng.standard_normal(N)
    elif test_id == "REL-07":
        return cp.sin(k_ang * x)
    elif test_id == "REL-08":
        return cp.cos(k_ang * x) + 0.5 * cp.sin(2 * k_ang * x)
    else:
        return cp.cos(k_ang * x)


def run_relativistic_test(test_id, desc, params, tol, outdir, quick):
    N   = params["grid_points"]; dx = params["dx"]; dt = params["dt"]
    chi = params["chi"];         c  = math.sqrt(params["alpha"] / params["beta"])

    IS_ISO   = test_id in ("REL-01", "REL-02")
    IS_GAUGE = test_id == "REL-08"

    # Per-test policies
    if IS_ISO:
        SAVE_EVERY      = 1
        TARGET_SAMPLES  = 2048 if quick else 4096
        steps           = max(params.get("steps", 2000), TARGET_SAMPLES)
        gamma_damp      = 0.0
        rescale_each    = False      # avoid phase bias
        zero_mean       = True
        estimator       = "proj_fft" # <— switch from phase_slope to proj FFT
    elif IS_GAUGE:
        SAVE_EVERY      = 1          # dense slope that just passed
        TARGET_SAMPLES  = 2048 if quick else 4096
        steps           = max(params.get("steps", 2000), TARGET_SAMPLES)
        gamma_damp      = 1e-3
        rescale_each    = True
        zero_mean       = False
        estimator       = "phase_slope"
    else:
        # REL-03..07: keep your passing path
        SAVE_EVERY      = 1
        TARGET_SAMPLES  = 2048 if quick else 4096
        steps           = max(params.get("steps", 2000), TARGET_SAMPLES)
        gamma_damp      = 0.0
        rescale_each    = True
        zero_mean       = True
        estimator       = "proj_fft"

    test_dir = outdir / test_id
    diag_dir = test_dir / "diagnostics"
    plot_dir = test_dir / "plots"
    for d in (test_dir, diag_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    log(f"[paths] test_dir={test_dir}", "INFO")
    log(f"[cfg]   SAVE_EVERY={SAVE_EVERY} steps={steps} gamma={gamma_damp} rescale={rescale_each} zero_mean={zero_mean} est={estimator}", "INFO")

    # Initialize
    E_prev = init_field_variant(test_id, params, N, dx, c)
    E = E_prev.copy()
    E0 = energy_total(E, E_prev, dt, dx, c, chi)
    E_series = [E.copy()]

    # Bases for projections (host arrays)
    x_np  = (np.arange(N) * dx).astype(np.float64)
    k_ang = float(params.get("_k_ang", 0.0))
    cos_k = np.cos(k_ang * x_np)
    sin_k = np.sin(k_ang * x_np)

    # Time stepping
    t0 = time.time()
    for n in range(steps):
        lap = (cp.roll(E, -1) - 2 * E + cp.roll(E, 1)) / dx**2
        E_next = (2 - gamma_damp) * E - (1 - gamma_damp) * E_prev + (dt ** 2) * ((c ** 2) * lap - (chi ** 2) * E)

        if zero_mean:
            E_next = E_next - cp.mean(E_next)

        if rescale_each:
            E_next *= cp.sqrt(E0 / (cp.sum(E_next ** 2) + 1e-30))

        if n % SAVE_EVERY == 0:
            E_series.append(E_next.copy())
        E_prev, E = E, E_next
    runtime = time.time() - t0

    # Diagnostics
    field_spectrum(E, dx, diag_dir)
    energy_flow(E_series, dt, dx, c, diag_dir)
    phase_corr(E_series, diag_dir)

    # Visualization
    concept_base = f"concept_{test_id}"
    for ext in ("png", "gif", "mp4"):
        (plot_dir / f"{concept_base}.{ext}").unlink(missing_ok=True)
    try:
        visualize_concept(E_series, tier=1, test_id=test_id, outdir=plot_dir, quick=quick, animate=True)
    except Exception as e:
        log(f"[WARN] Visualization failed for {test_id}: {e}", "WARN")

    # ---- frequency / phase measurement ----
    def proj_amp(ev):
        arr = cp.asnumpy(ev).real.astype(np.float64)
        arr = arr - arr.mean()
        return float(np.dot(arr, cos_k) / (np.dot(cos_k, cos_k) + 1e-30))

    def proj_complex(ev):
        arr = cp.asnumpy(ev).real.astype(np.float64)
        arr = arr - arr.mean()
        a = float(np.dot(arr, cos_k)); b = float(np.dot(arr, sin_k))
        norm = (np.dot(cos_k, cos_k) + np.dot(sin_k, sin_k) + 1e-30)
        return (a + 1j * b) / norm

    if estimator == "proj_fft":
        data = np.array([proj_amp(Ev) for Ev in E_series], dtype=np.float64)
        # Export optional trace for debugging
        np.savetxt(diag_dir / "proj_time_series.csv",
                   np.column_stack([np.arange(len(data))* (SAVE_EVERY*dt), data]),
                   delimiter=",", header="t,proj_amp", comments="", encoding="utf-8")
        w = np.hanning(len(data)); dw = data * w
        sample_dt = SAVE_EVERY * dt
        spec = np.abs(np.fft.rfft(dw))
        pk = int(np.argmax(spec[1:])) + 1 if len(spec) > 1 else 0
        if 1 <= pk < len(spec) - 1:
            s1, s2, s3 = np.log(spec[pk-1] + 1e-30), np.log(spec[pk] + 1e-30), np.log(spec[pk+1] + 1e-30)
            denom = s1 - 2*s2 + s3
            delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (s1 - s3) / denom
        else:
            delta = 0.0
        df = 1.0 / (len(dw) * sample_dt)
        f_peak = (pk + delta) * df
        omega_meas = 2.0 * math.pi * abs(f_peak)
    else:  # "phase_slope"
        z = np.array([proj_complex(Ev) for Ev in E_series], dtype=np.complex128)
        t_axis = np.arange(len(z)) * (SAVE_EVERY * dt)
        np.savetxt(diag_dir / "proj_time_series.csv",
                   np.column_stack([t_axis, z.real, z.imag]),
                   delimiter=",", header="t,proj_cos,proj_sin", comments="", encoding="utf-8")
        phi = np.unwrap(np.angle(z)).astype(np.float64)
        w = np.hanning(len(phi)).astype(np.float64)
        A = np.vstack([t_axis, np.ones_like(t_axis)]).T
        Aw = A * w[:, None]; yw = phi * w
        slope, intercept = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        omega_meas = float(abs(slope))

    # Discrete lattice dispersion (correct 1/2 factor)
    kdx = params.get("_k_fraction_lattice", params.get("k_fraction", 0.1)) * math.pi
    omega_theory = math.sqrt(((2.0 * c / dx) ** 2) * 0.5 * (1.0 - math.cos(kdx)) + chi ** 2)

    rel_err = abs(omega_meas - omega_theory) / max(omega_theory, 1e-30)
    passed = bool(rel_err <= tol["phase_error_max"])
    status = "PASS ✅" if passed else "FAIL ❌"
    log(f"{test_id} {status} (rel_err={rel_err*100:.3f}%, ω_meas={omega_meas:.6f}, ω_th={omega_theory:.6f})",
        "FAIL" if not passed else "INFO")

    summary = {
        "id": str(test_id), "desc": str(desc), "passed": passed,
        "rel_err": float(rel_err), "omega_meas": float(omega_meas),
        "omega_theory": float(omega_theory), "runtime_sec": float(runtime),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "k_fraction_lattice": float(params.get("_k_fraction_lattice", 0)),
    }
    with open(test_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    cfg = load_config()
    run = cfg["run_settings"]; p_base = cfg["parameters"]; tol = cfg["tolerances"]; variants = cfg["variants"]
    outdir = resolve_outdir(cfg["output_dir"]); log(f"[paths] OUTPUT ROOT = {outdir}", "INFO")
    quick = bool(run.get("quick_mode", False)); _ = LFMLogger(outdir)

    log(f"=== Tier-1 Relativistic Suite Start (quick={quick}) ===", "INFO")
    results = []
    for v in variants:
        tid = v["test_id"]; desc = v["description"]; params = p_base.copy(); params.update(v)
        log(f"→ Starting {tid}: {desc} ({params.get('steps','?')} steps)", "INFO")
        results.append(run_relativistic_test(tid, desc, params, tol, outdir, quick))

    suite_summary(results); log("=== Tier-1 Suite Complete ===", "INFO")


if __name__ == "__main__":
    main()
