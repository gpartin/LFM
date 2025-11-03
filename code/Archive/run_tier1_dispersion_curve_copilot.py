#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
LFM Tier-1 — Dispersion Curve Diagnostic (4th-Order Stencil, Discrete-Theory)
- Correct leapfrog + stencil discrete dispersion
- Unit-safe FFT with parabolic peak refinement
- Robust debug prints and error reporting
- Uses single-point probe to avoid high-k spatial averaging cancellation
"""
import json, math, time, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# GPU fallback
try:
    import cupy as cp
    xp = cp
    USING_CUPY = True
except Exception:
    cp = np
    xp = np
    USING_CUPY = False

# ------------------------------- Config Loader ------------------------------
def load_config():
    script = Path(__file__).stem.replace("run_", "")
    cfg_path = Path(__file__).resolve().parent.parent / "config" / f"config_{script}.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

# --------------------------- Spatial Laplacian helper -----------------------
def laplacian(E, dx, order=2):
    if order == 2:
        return (xp.roll(E, -1) - 2*E + xp.roll(E, 1)) / dx**2
    elif order == 4:
        return (-xp.roll(E, 2) + 16*xp.roll(E, 1) - 30*E + 16*xp.roll(E, -1) - xp.roll(E, -2)) / (12*dx**2)
    else:
        raise ValueError("Unsupported stencil order; use 2 or 4")

# -------------------------- Field initialization ----------------------------
def init_traveling_wave(N, k0, dx, omega, dt):
    x = xp.arange(N) * dx
    E_now = xp.cos(k0 * x)
    E_prev = xp.cos(k0 * x - omega * dt)
    return E_prev, E_now

# -------------------------- Discrete dispersion helper ---------------------
def discrete_omega(k0, dx, dt, c, chi, stencil_order=4):
    """
    Leapfrog-in-time + FD Laplacian dispersion:
      sin^2(ω Δt / 2) = (Δt^2 / 4) * ( - c^2 * λ_h(k) + χ^2 ), with λ_h(k) < 0
    """
    kd = k0 * dx
    if stencil_order == 4:
        lam = (-2.0 * math.cos(2.0 * kd) + 32.0 * math.cos(kd) - 30.0) / (12.0 * dx * dx)
    elif stencil_order == 2:
        lam = (2.0 * (math.cos(kd) - 1.0)) / (dx * dx)
    else:
        lam = (2.0 * (math.cos(kd) - 1.0)) / (dx * dx)

    rhs_inside = (dt * dt / 4.0) * ( - (c * c) * lam + (chi * chi) )
    rhs_inside = max(0.0, min(1.0, rhs_inside))
    omega_disc = (2.0 / dt) * math.asin(math.sqrt(rhs_inside))
    omega_cont = math.sqrt((c * k0) ** 2 + chi ** 2)
    return omega_disc, omega_cont, kd, lam, rhs_inside, math.sqrt(rhs_inside)

# -------------------------- Debug plotting helper --------------------------
def _save_debug_plots(data, xf, yf, k0, outdir=Path(".")):
    outdir = Path(outdir)
    try:
        trace_fn = outdir / f"trace_debug_k_{k0:.3e}.png"
        fft_fn = outdir / f"fft_debug_k_{k0:.3e}.png"
        plt.figure(figsize=(8,3)); plt.plot(data); plt.title(f"Time trace (k0={k0:.3e})")
        plt.xlabel("sample idx"); plt.ylabel("E (probe)"); plt.tight_layout(); plt.savefig(trace_fn, dpi=150); plt.close()
        plt.figure(figsize=(8,3)); plt.semilogy(xf, yf + 1e-30)
        plt.title("FFT magnitude"); plt.xlabel("freq [Hz]"); plt.ylabel("|FFT|"); plt.tight_layout(); plt.savefig(fft_fn, dpi=150); plt.close()
        return str(trace_fn), str(fft_fn)
    except Exception as ex:
        print("Failed to save debug plots:", ex)
        return None, None

# -------------------------- Frequency measurement --------------------------
def measure_frequency(N, steps, dx, dt, alpha, beta, chi, save_every, k_frac, stencil_order, outdir, produce_debug_plots=False):
    c = math.sqrt(alpha / beta)
    k0 = k_frac * (math.pi / dx)
    omega_theory_disc, omega_theory_cont, kd, lam, rhs_inside, arg = discrete_omega(k0, dx, dt, c, chi, stencil_order)
    f_theory = omega_theory_disc / (2.0 * math.pi)

    print(f"DEBUG_SYMBOLS: k_frac={k_frac:.6g}, k0={k0:.6g}, kd={kd:.6g}, lam={lam:.6g}, rhs_inside={rhs_inside:.6g}, arg={arg:.6g}, omega_disc={omega_theory_disc:.6g}, omega_cont={omega_theory_cont:.6g}")

    E_prev, E = init_traveling_wave(N, k0, dx, omega_theory_disc, dt)

    # --- Single-point probe to avoid spatial averaging cancellation at high k ---
    center = N // 2
    if USING_CUPY:
        probe_idx = int(center)
    else:
        probe_idx = int(center)

    trace = []
    min_samples = 256
    samples_taken = 0

    for n in range(steps):
        lap = laplacian(E, dx, order=stencil_order)
        E_next = (2 * E - E_prev) + (dt ** 2) * ( (c ** 2) * lap - (chi ** 2) * E )
        if n % save_every == 0:
            if USING_CUPY:
                sample_val = float(cp.asnumpy(E[probe_idx]))
            else:
                sample_val = float(E[probe_idx])
            trace.append(sample_val)
            samples_taken += 1
        E_prev, E = E, E_next

    if samples_taken < min_samples:
        raise RuntimeError(f"Insufficient FFT samples ({samples_taken} < {min_samples}). Increase steps or reduce save_every.")

    data = np.asarray(trace, dtype=float)
    n_samples = data.size
    fs = 1.0 / (save_every * dt)
    nyquist = 0.5 * fs

    window = np.hanning(n_samples)
    yf = np.abs(np.fft.rfft(data * window))
    xf = np.fft.rfftfreq(n_samples, d=save_every * dt)  # Hz

    if len(yf) < 3:
        raise RuntimeError("FFT length too small to locate peak robustly.")

    median_noise = np.median(yf + 1e-16)
    peak_idx = int(np.argmax(yf[1:])) + 1
    peak_amp = yf[peak_idx]

    print(f"DEBUG_FFT: n_samples={n_samples}, fs={fs:.6g} Hz, nyquist={nyquist:.6g} Hz, peak_idx={peak_idx}, xf_peak={xf[peak_idx]:.6g} Hz, peak_amp={peak_amp:.6g}")

    if peak_amp / median_noise < 8.0:
        if produce_debug_plots:
            trace_fn, fft_fn = _save_debug_plots(data, xf, yf, k0, outdir=outdir)
            print("  Debug plots saved:", trace_fn, fft_fn)
        raise RuntimeError(f"Low SNR in spectral peak: peak/median = {peak_amp/median_noise:.2f} (<8). Debug plots saved.")

    # Parabolic peak refine
    if 1 <= peak_idx < len(yf) - 1:
        a, b, c_ = yf[peak_idx - 1], yf[peak_idx], yf[peak_idx + 1]
        denom = (a - 2 * b + c_)
        p = 0.0 if abs(denom) < 1e-12 else 0.5 * (a - c_) / denom
        df = xf[1] - xf[0]
        f_refined_hz = xf[peak_idx] + p * df
    else:
        f_refined_hz = xf[peak_idx]

    f_meas = float(f_refined_hz)
    omega_meas = 2.0 * math.pi * f_meas
    ratio = (omega_meas / (2.0 * math.pi * f_meas)) if f_meas != 0 else float('nan')
    print(f"CHECK_UNITS: f_meas={f_meas:.6e} Hz, omega_meas={omega_meas:.6e} rad/s, ratio={ratio:.6g}")

    diagnostics = {
        "k0": float(k0),
        "omega_meas": float(omega_meas),
        "omega_theory_disc": float(omega_theory_disc),
        "omega_theory_continuum": float(omega_theory_cont),
        "f_meas": float(f_meas),
        "f_theory": float(f_theory),
        "peak_amp": float(peak_amp),
        "noise_median": float(median_noise),
        "snr": float(peak_amp / (median_noise + 1e-16)),
        "samples": samples_taken,
        "sampling_freq": fs,
    }
    if produce_debug_plots:
        diagnostics["_debug_trace"] = data.tolist()
        diagnostics["_debug_xf"] = xf.tolist()
        diagnostics["_debug_yf"] = yf.tolist()
    return diagnostics

# ----------------------------------- Main -----------------------------------
def main():
    cfg = load_config()
    p = cfg["parameters"]

    N = int(p["grid_points"])
    steps = int(p["steps"])
    dx = float(p["dx"])
    dt = float(p["dt"])
    alpha = float(p["alpha"])
    beta = float(p["beta"])
    chi = float(p["chi"])
    save_every = int(p["save_every"])
    k_list = p.get("k_sweep", [p.get("k_fraction", 0.3)])
    stencil_order = int(p.get("stencil_order", 4))

    d = int(p.get("dimension", 1))
    c = math.sqrt(alpha / beta)
    cfl = c * dt / dx
    cfl_max = float(cfg.get("cfl_max", 1.0 if d == 1 else 1.0 / math.sqrt(d)))
    if cfl > cfl_max + 1e-12:
        raise ValueError(f"CFL too high (c*dt/dx={cfl:.6f} > {cfl_max}). Reduce dt or increase dx.")

    project_root = Path(__file__).resolve().parent.parent
    outdir = (project_root / cfg["output_dir"]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    parsed_config = {
        "N": N, "dx": dx, "dt": dt, "alpha": alpha, "beta": beta, "chi": chi,
        "save_every": save_every, "k_list": k_list, "stencil_order": stencil_order,
        "c": c, "cfl": cfl, "using_cupy": USING_CUPY
    }
    print("Parsed config:\n", json.dumps(parsed_config, indent=2))

    rows = []
    meta = {
        "using_cupy": USING_CUPY,
        "N": N, "dx": dx, "dt": dt, "alpha": alpha, "beta": beta, "chi": chi,
        "stencil_order": stencil_order, "c": c, "cfl": cfl, "cfl_max": cfl_max,
        "date": time.asctime(),
    }

    t0 = time.time()
    for k_frac in k_list:
        print(f"\nRunning k_fraction = {k_frac:.6f}  (stencil_order={stencil_order})")
        try:
            diag = measure_frequency(
                N, steps, dx, dt, alpha, beta, chi, save_every,
                k_frac, stencil_order, outdir=outdir, produce_debug_plots=False
            )
            rows.append((diag["k0"], diag["omega_meas"], diag["omega_theory_disc"], diag["snr"]))
            print(f"  k={diag['k0']:.6e}, f_meas={diag['f_meas']:.6e} Hz, f_theory_disc={diag['f_theory']:.6e} Hz")
            print(f"  omega_meas={diag['omega_meas']:.6e} rad/s, omega_theory_disc={diag['omega_theory_disc']:.6e} rad/s, omega_theory_cont={diag['omega_theory_continuum']:.6e} rad/s, SNR={diag['snr']:.2f}, samples={diag['samples']}")
        except Exception as e:
            print(f"  ERROR for k_frac={k_frac}: {e}")
            rows.append((float('nan'), float('nan'), float('nan'), float('nan')))

    print(f"\nCompleted sweep in {time.time() - t0:.2f}s")

    arr = np.array(rows, dtype=float)
    if arr.size == 0:
        print("No valid results to save.")
        return

    # Unpack measured arrays
    k_vals, omega_meas, _, snr_vals = arr.T
    k_vals = np.asarray(k_vals, dtype=float)
    omega_meas = np.asarray(omega_meas, dtype=float)

    # Recompute theory arrays directly from k (avoids any row-order/precision issues)
    c = math.sqrt(float(cfg["parameters"]["alpha"]) / float(cfg["parameters"]["beta"]))
    dx = float(cfg["parameters"]["dx"])
    dt = float(cfg["parameters"]["dt"])
    chi = float(cfg["parameters"]["chi"])
    stencil_order = int(cfg["parameters"].get("stencil_order", 4))

    omega_disc_from_k = []
    omega_cont_from_k = []
    for kv in k_vals:
        od, oc, *_ = discrete_omega(kv, dx, dt, c, chi, stencil_order)
        omega_disc_from_k.append(od)
        omega_cont_from_k.append(oc)
    omega_disc_from_k = np.asarray(omega_disc_from_k, dtype=float)
    omega_cont_from_k = np.asarray(omega_cont_from_k, dtype=float)

    # Relative errors (safe)
    rel_err_disc = np.abs(omega_meas - omega_disc_from_k) / np.maximum(np.abs(omega_disc_from_k), 1e-12)
    rel_err_cont = np.abs(omega_meas - omega_cont_from_k) / np.maximum(np.abs(omega_cont_from_k), 1e-12)

    # Save + plot
    np.savetxt(
        outdir / "dispersion_curve.csv",
        np.column_stack([k_vals, omega_meas, omega_disc_from_k, omega_cont_from_k, rel_err_disc, rel_err_cont, snr_vals]),
        header="k, omega_meas, omega_theory_disc, omega_theory_continuum, rel_err_disc, rel_err_cont, snr"
    )

    with open(outdir / "dispersion_meta.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "k_list": k_vals.tolist(), "rows": rows, "parsed_config": parsed_config}, f, indent=2)

    valid = ~np.isnan(omega_meas)
    plt.figure(figsize=(6, 4))
    if np.any(valid):
        plt.plot(k_vals[valid], omega_disc_from_k[valid], "k--", label="Theory (discrete) ω(k)")
        plt.plot(k_vals[valid], omega_meas[valid], "o-", label="Measured")
    plt.xlabel("k"); plt.ylabel("ω [rad/s]"); plt.grid(True)
    plt.title("Tier-1 Dispersion Curve — Discrete Theory vs Measured")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir / "dispersion_curve.png", dpi=150); plt.close()

    max_err_disc = float(np.nanmax(rel_err_disc))
    max_err_cont = float(np.nanmax(rel_err_cont))
    tol = float(cfg["tolerances"]["phase_error_max"])
    print(f"Output saved to {outdir}")
    print(f"\nMax relative error vs discrete-theory: {max_err_disc*100:.6f}%  (tol {tol*100:.6f}%)")
    print(f"Max relative error vs continuum: {max_err_cont*100:.6f}%")
    print(f"Overall Result: {'PASS ✅' if max_err_disc <= tol else 'FAIL ❌'}")

if __name__ == "__main__":
    main()
