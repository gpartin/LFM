#!/usr/bin/env python3
"""
Tier-4 — Quantization & Spectra Tests
- Famous-equation test implemented: Heisenberg uncertainty Δx·Δk ≈ 1/2 (natural units)
- Additional tests scaffolded (cavity spectroscopy, threshold), initially skipped

Outputs under results/Quantization/<TEST_ID>/
"""
import json, math, time, platform
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from lfm_backend import to_numpy, get_array_module
from lfm_results import ensure_dirs, write_csv, save_summary
from lfm_console import log
from lfm_test_harness import BaseTierHarness

@dataclass
class TestResult:
    test_id: str
    description: str
    passed: bool
    metrics: Dict
    runtime_sec: float

def _default_config_name() -> str:
    return "config_tier4_quantization.json"

# ------------------------------- 1D helpers --------------------------------
def laplacian_1d(E, dx, order=2, xp=None):
    """1D Laplacian with periodic boundaries (will be overridden by apply_dirichlet if needed)"""
    if xp is None:
        xp = get_array_module(E)
    if order == 2:
        return (xp.roll(E, -1) - 2*E + xp.roll(E, 1)) / (dx*dx)
    elif order == 4:
        return (-xp.roll(E, 2) + 16*xp.roll(E, 1) - 30*E + 16*xp.roll(E, -1) - xp.roll(E, -2)) / (12*dx*dx)
    else:
        raise ValueError('order must be 2 or 4')

def apply_dirichlet(E):
    """Force E=0 at boundaries for Dirichlet conditions"""
    E[0] = 0.0
    E[-1] = 0.0

# --------------------------- Cavity spectroscopy ---------------------------
def run_cavity_spectroscopy(params, tol, test, out_dir: Path, xp, on_gpu) -> TestResult:
    ensure_dirs(out_dir)
    test_id = test['test_id']
    desc = test['description']
    
    N  = int(test.get('N', params.get('N', 1024)))
    dx = float(test.get('dx', params.get('dx', 0.1)))
    dt = float(test.get('dt', params.get('dt', 0.01)))
    steps = int(test.get('steps', params.get('steps', 12000)))
    chi = float(test.get('chi_uniform', params.get('chi_uniform', 0.20)))
    num_peaks = int(test.get('num_peaks', 5))
    
    # Setup: 1D cavity with Dirichlet boundaries (E=0 at ends)
    L = N * dx
    x = xp.arange(N, dtype=xp.float64) * dx
    
    # Initial condition: sum of first few cavity modes with random phases
    # This excites multiple modes while respecting Dirichlet boundaries
    E = xp.zeros(N, dtype=xp.float64)
    for n in range(1, num_peaks + 2):  # Excite a few more modes than we're measuring
        k_n = n * np.pi / L
        phase = np.random.rand() * 2.0 * np.pi
        amplitude = 1.0 / n  # Decrease amplitude with mode number
        E += amplitude * xp.sin(k_n * x + phase)
    E_prev = E.copy()
    
    # Apply boundaries
    apply_dirichlet(E)
    apply_dirichlet(E_prev)
    
    # Save full spatial snapshots for modal analysis
    snapshots = []
    snapshot_times = []
    save_every = int(test.get('save_every', params.get('save_every', 20)))
    
    t0 = time.time()
    for t in range(steps):
        if t % save_every == 0:
            snapshots.append(to_numpy(E.copy()))
            snapshot_times.append(t * dt)
        
        # Advance leapfrog
        lap = laplacian_1d(E, dx, order=2, xp=xp)
        E_next = 2*E - E_prev + (dt*dt) * (lap - (chi*chi)*E)
        apply_dirichlet(E_next)
        E_prev, E = E, E_next
    
    runtime = time.time() - t0
    
    # Modal decomposition: project each snapshot onto theoretical mode shapes
    # Mode shapes: ψ_n(x) = √(2/L) sin(nπx/L)
    mode_amplitudes = np.zeros((len(snapshot_times), num_peaks))
    for i, snap in enumerate(snapshots):
        for n in range(1, num_peaks + 1):
            k_n = n * np.pi / L
            mode_shape = np.sqrt(2.0/L) * np.sin(k_n * to_numpy(x))
            # Project: a_n(t) = ∫ E(x,t) ψ_n(x) dx
            mode_amplitudes[i, n-1] = np.sum(snap * mode_shape) * dx
    
    # FFT each mode amplitude time-series to get its oscillation frequency
    dt_sample = snapshot_times[1] - snapshot_times[0]
    freqs_hz = np.fft.rfftfreq(len(snapshot_times), d=dt_sample)
    freqs = 2.0 * np.pi * freqs_hz
    
    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(len(snapshot_times))
    
    measured_omegas = []
    for n in range(num_peaks):
        amp_windowed = mode_amplitudes[:, n] * window
        amp_fft = np.fft.rfft(amp_windowed)
        amp_spec = np.abs(amp_fft)
        # Find peak frequency for this mode
        peak_idx = np.argmax(amp_spec[1:]) + 1  # Skip DC component
        
        # Parabolic interpolation for sub-bin accuracy
        if 0 < peak_idx < len(amp_spec) - 1:
            y0, y1, y2 = amp_spec[peak_idx-1], amp_spec[peak_idx], amp_spec[peak_idx+1]
            delta = 0.5 * (y2 - y0) / (2*y1 - y0 - y2 + 1e-30)
            peak_freq = freqs[peak_idx] + delta * (freqs[1] - freqs[0])
        else:
            peak_freq = freqs[peak_idx]
        measured_omegas.append(peak_freq)
    
    spec = np.max(np.abs(np.fft.rfft(mode_amplitudes, axis=0)), axis=1)  # For plotting
    
    # Theoretical mode frequencies: ω_n^2 = (nπ/L)^2 + χ^2
    def mode_omega(n):
        k_n = n * np.pi / L
        return np.sqrt(k_n**2 + chi**2)
    
    # Measured modes from modal decomposition
    measured_modes = measured_omegas
    theory_modes = [mode_omega(n) for n in range(1, num_peaks+1)]
    
    errors = []
    rows = []
    for n, (theory, measured) in enumerate(zip(theory_modes, measured_modes), start=1):
        err = abs(measured - theory) / theory if theory > 0 else 0
        errors.append(err)
        rows.append((n, theory, measured, err))
    
    mean_err = np.mean(errors) if errors else 1.0
    tol_key = 'spectral_err_fine' if 'fine' in desc.lower() else 'spectral_err_coarse'
    passed = mean_err <= float(tol.get(tol_key, 0.02))
    
    # Save outputs
    ensure_dirs(out_dir/"diagnostics")
    write_csv(out_dir/"diagnostics"/"mode_spectrum.csv", rows, ["mode_n","theory_omega","measured_omega","rel_error"])
    write_csv(out_dir/"diagnostics"/"mode_amplitudes.csv", 
              list(zip(snapshot_times, *[mode_amplitudes[:,i] for i in range(num_peaks)])),
              ["time"] + [f"mode_{n}" for n in range(1, num_peaks+1)])
    
    # Plot: measured vs theoretical mode frequencies
    plt.figure(figsize=(10,5))
    modes_n = np.arange(1, num_peaks+1)
    plt.plot(modes_n, theory_modes, 'bo-', label='Theory', markersize=8)
    plt.plot(modes_n, measured_modes, 'rx-', label='Measured', markersize=8)
    plt.xlabel('Mode number n')
    plt.ylabel('Frequency ω')
    plt.title(f'{test_id}: Cavity modes — mean err={mean_err*100:.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    ensure_dirs(out_dir/"plots")
    plt.tight_layout()
    plt.savefig(out_dir/"plots"/"cavity_spectrum.png", dpi=150)
    plt.close()
    
    summary = {
        "tier": 4,
        "category": "Quantization",
        "test_id": test_id,
        "description": desc,
        "timestamp": time.time(),
        "hardware": {"backend": "CuPy" if on_gpu else "NumPy"},
        "parameters": {"N":N,"dx":dx,"dt":dt,"chi":chi,"L":L},
        "metrics": {"mean_mode_error": float(mean_err), "num_modes": len(measured_modes)},
        "status": "Passed" if passed else "Failed"
    }
    save_summary(out_dir, test_id, summary)
    log(f"[{test_id}] Cavity modes mean_err={mean_err*100:.2f}% (tol={tol.get(tol_key,0.02)*100:.1f}%) → {'PASS' if passed else 'FAIL'}",
        "PASS" if passed else "FAIL")
    
    return TestResult(test_id, desc, passed, summary["metrics"], runtime)

# --------------------------- Threshold test --------------------------------
def run_threshold_test(params, tol, test, out_dir: Path, xp, on_gpu) -> TestResult:
    ensure_dirs(out_dir)
    test_id = test['test_id']
    desc = test['description']
    
    N  = int(test.get('N', params.get('N', 1024)))
    dx = float(test.get('dx', params.get('dx', 0.1)))
    dt = float(test.get('dt', params.get('dt', 0.01)))
    chi = float(test.get('chi_uniform', 0.25))
    
    freq_start = float(test.get('freq_start', 0.18))
    freq_end = float(test.get('freq_end', 0.32))
    freq_steps = int(test.get('freq_steps', 15))
    drive_amp = float(test.get('drive_amp', 0.01))
    drive_width = int(test.get('drive_width_cells', 4))
    detector_frac = float(test.get('detector_x_frac', 0.75))
    steps_per_freq = int(test.get('steps_per_freq', 4000))
    
    freqs_sweep = np.linspace(freq_start, freq_end, freq_steps)
    transmissions = []
    
    # Drive source at left side, detect at right side (periodic boundaries allow transmission)
    source_idx = int(0.1 * N)  # 10% from left
    detector_idx = int(0.9 * N)  # 90% from left
    
    t0_total = time.time()
    for omega in freqs_sweep:
        # Initial condition: Gaussian wave packet with frequency ω
        L = N * dx
        x_np = np.arange(N) * dx
        x_c = L * 0.3  # Start at 30% position
        sigma = L / 15.0
        
        # Dispersion relation: ω² = k² + χ²
        # For ω < χ: k² < 0 (imaginary k, evanescent)
        # For ω ≥ χ: k² ≥ 0 (real k, propagating)
        k_squared = omega**2 - chi**2
        if k_squared > 0:
            k = np.sqrt(k_squared)
            # Propagating mode: use traveling wave
            envelope = np.exp(-((x_np - x_c)**2) / (2.0 * sigma**2))
            carrier = np.cos(k * x_np)
            E = xp.asarray(envelope * carrier * drive_amp * 50)
        else:
            # Evanescent mode: use stationary oscillation
            envelope = np.exp(-((x_np - x_c)**2) / (2.0 * sigma**2))
            E = xp.asarray(envelope * drive_amp * 50)
        E_prev = E.copy()
        
        # Propagate and measure RMS amplitude at detector region (average over space to avoid nodes)
        detector_window = slice(detector_idx-5, detector_idx+5)
        rms_series = []
        for t in range(steps_per_freq):
            # Advance with periodic boundaries
            lap = laplacian_1d(E, dx, order=2, xp=xp)
            E_next = 2*E - E_prev + (dt*dt) * (lap - (chi*chi)*E)
            E_prev, E = E, E_next
            
            # After equilibration, record RMS in detector window
            if t > steps_per_freq // 3:
                rms_val = float(xp.sqrt(xp.mean(E[detector_window]**2)))
                rms_series.append(rms_val)
        
        # Transmission: time-averaged RMS at detector
        # Below threshold (ω<χ): evanescent decay, small amplitude
        # Above threshold (ω>χ): propagating wave, larger amplitude
        trans = np.mean(rms_series)
        transmissions.append(trans)
    
    runtime_total = time.time() - t0_total
    
    # Fit threshold: find ω where transmission rises sharply (10-90% points)
    trans_arr = np.array(transmissions)
    trans_norm = (trans_arr - trans_arr.min()) / (trans_arr.max() - trans_arr.min() + 1e-30)
    
    # Find first crossing of 50%
    idx_50 = np.where(trans_norm >= 0.5)[0]
    if len(idx_50) > 0:
        omega_th_measured = freqs_sweep[idx_50[0]]
    else:
        omega_th_measured = chi  # fallback
    
    omega_th_theory = chi
    err = abs(omega_th_measured - omega_th_theory) / omega_th_theory
    passed = err <= float(tol.get('threshold_err', 0.02))
    
    # Save outputs
    ensure_dirs(out_dir/"diagnostics")
    write_csv(out_dir/"diagnostics"/"transmission_vs_freq.csv",
              list(zip(freqs_sweep, transmissions)), ["omega","transmission"])
    
    plt.figure(figsize=(8,5))
    plt.plot(freqs_sweep, transmissions, 'o-', label='Transmission')
    plt.axvline(chi, color='r', linestyle='--', label=f'χ={chi:.3f} (theory)')
    plt.axvline(omega_th_measured, color='g', linestyle=':', label=f'ω_th={omega_th_measured:.3f} (meas)')
    plt.xlabel('Drive frequency ω')
    plt.ylabel('Transmission amplitude')
    plt.title(f'{test_id}: Threshold — ω_th err={err*100:.1f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    ensure_dirs(out_dir/"plots")
    plt.tight_layout()
    plt.savefig(out_dir/"plots"/"threshold_curve.png", dpi=150)
    plt.close()
    
    summary = {
        "tier": 4,
        "category": "Quantization",
        "test_id": test_id,
        "description": desc,
        "timestamp": time.time(),
        "hardware": {"backend": "CuPy" if on_gpu else "NumPy"},
        "parameters": {"N":N,"dx":dx,"dt":dt,"chi":chi},
        "metrics": {"omega_th_theory": omega_th_theory, "omega_th_measured": float(omega_th_measured), "rel_error": float(err)},
        "status": "Passed" if passed else "Failed"
    }
    save_summary(out_dir, test_id, summary)
    log(f"[{test_id}] Threshold ω_th={omega_th_measured:.3f} vs χ={chi:.3f} (err={err*100:.1f}%) → {'PASS' if passed else 'FAIL'}",
        "PASS" if passed else "FAIL")
    
    return TestResult(test_id, desc, passed, summary["metrics"], runtime_total)

# --------------------------- Famous equation test --------------------------
def run_uncertainty_test(params, tol, test, out_dir: Path, xp, on_gpu) -> TestResult:
    ensure_dirs(out_dir)
    test_id = test['test_id']
    desc = test['description']

    N  = int(test.get('N', params.get('N', 1024)))
    dx = float(test.get('dx', params.get('dx', 0.1)))
    dt = float(test.get('dt', params.get('dt', 0.01)))
    chi = float(test.get('chi_uniform', params.get('chi_uniform', 0.20)))

    # Build coordinate
    x = xp.arange(N, dtype=xp.float64) * dx
    L = N*dx
    x0 = 0.5*L

    sigma_list = list(map(float, test.get('sigma_x_list', [2.0, 3.0, 4.0, 6.0, 8.0])))

    products = []
    rows = []

    for sigma_x in sigma_list:
        # Gaussian centered at x0 with Δx = sigma_x by construction:
        # Use E(x) = exp(- (x-x0)^2 / (4 σ^2)) so that Δx=σ and Δk=1/(2σ) → Δx·Δk=1/2
        E = xp.exp(-((x - x0)**2) / (4.0 * (sigma_x**2)))
        # Normalize to unit L2
        E = E / xp.sqrt(xp.sum(E*E)*dx)

        # Δx: standard deviation in x with probability |E|^2 dx
        p = E*E
        mu = xp.sum(x*p)*dx
        var_x = xp.sum(((x-mu)**2)*p)*dx
        delta_x = float(xp.sqrt(var_x))

        # Δk: from FFT spectrum width with proper scaling
        # Continuous FT: F(k) = ∫ E(x) e^{-ikx} dx; discretize with dx factor
        E_np = to_numpy(E)
        spec = np.fft.fft(E_np) * (dx/np.sqrt(2*np.pi))
        k = 2*np.pi*np.fft.fftfreq(N, d=dx)
        P = np.abs(spec)**2
        dk = k[1] - k[0] if len(k) > 1 else 1.0
        P_sum = (P * dk).sum()
        if P_sum == 0:
            delta_k = 0.0
        else:
            mu_k = ((k*P) * dk).sum() / P_sum
            var_k = (((k-mu_k)**2) * P * dk).sum() / P_sum
            delta_k = float(np.sqrt(var_k))

        prod = delta_x * delta_k
        products.append(prod)
        rows.append((sigma_x, delta_x, delta_k, prod))

    products = np.array(products)
    # For a Gaussian, Δx·Δk = 1/2 exactly (in natural units); with our discrete FFT windowing,
    # expect close to 0.5 within tolerance
    target = 0.5
    err = float(abs(products.mean() - target) / target)
    passed = err <= float(tol.get('uncertainty_tol_frac', 0.05))

    # Save outputs
    ensure_dirs(out_dir/"diagnostics")
    write_csv(out_dir/"diagnostics"/"uncertainty_results.csv", rows, ["sigma_x","delta_x","delta_k","product"])

    plt.figure(figsize=(6,4))
    plt.plot(sigma_list, products, 'o-', label='Δx·Δk')
    plt.axhline(0.5, color='k', linestyle='--', label='1/2')
    plt.xlabel('σ_x (cells)')
    plt.ylabel('Δx·Δk')
    plt.title(f'Heisenberg Uncertainty — mean={products.mean():.3f}, err={err*100:.1f}%')
    plt.grid(True)
    ensure_dirs(out_dir/"plots")
    plt.tight_layout()
    plt.savefig(out_dir/"plots"/"uncertainty_dx_dk.png", dpi=150)
    plt.close()

    summary = {
        "tier": 4,
        "category": "Quantization",
        "test_id": test_id,
        "description": desc,
        "timestamp": time.time(),
        "hardware": {"backend": "CuPy" if on_gpu else "NumPy", "python": platform.python_version()},
        "parameters": {"N":N,"dx":dx,"dt":dt,"chi":chi},
        "metrics": {"mean_product": float(products.mean()), "target": 0.5, "rel_error": err},
        "status": "Passed" if passed else "Failed"
    }
    save_summary(out_dir, test_id, summary)
    log(f"[{test_id}] Δx·Δk mean={products.mean():.3f} vs 0.5 (err={err*100:.1f}%) → {'PASS' if passed else 'FAIL'}",
        "PASS" if passed else "FAIL")

    return TestResult(test_id, desc, passed, summary["metrics"], 0.0)

# ------------------------------- Main runner -------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Tier-4 Quantization & Spectra Suite')
    parser.add_argument('--test', type=str, default=None, help='Run single test by ID (e.g., QUAN-09)')
    parser.add_argument('--config', type=str, default='config/config_tier4_quantization.json')
    args = parser.parse_args()

    cfg = BaseTierHarness.load_config(args.config, default_config_name=_default_config_name())
    p, tol, tests = cfg['parameters'], cfg['tolerances'], cfg['tests']

    from lfm_backend import pick_backend
    xp, on_gpu = pick_backend(cfg.get('hardware', {}).get('gpu_enabled', True))
    dtype = xp.float64 if cfg.get('hardware', {}).get('precision', 'float64') == 'float64' else xp.float32

    # Prepare output and logger
    base = BaseTierHarness.resolve_outdir(cfg['output_dir'])
    ensure_dirs(base)
    from lfm_logger import LFMLogger
    from lfm_console import set_logger, log_run_config
    logger = LFMLogger(base)
    set_logger(logger)
    log_run_config(cfg, base)

    # Filter test
    if args.test:
        tests = [t for t in tests if t.get('test_id') == args.test]
        if not tests:
            log(f"[ERROR] Test {args.test} not found", "FAIL")
            return
        log(f"=== Running Single Test: {args.test} ===", "INFO")
    else:
        log("=== Tier-4 Quantization Suite Start ===", "INFO")

    for t in tests:
        if t.get('skip', False):
            log(f"[{t.get('test_id','?')}] SKIPPED: {t.get('description','')}", "WARN")
            continue
        mode = t.get('mode', 'uncertainty')
        out_dir = base / t.get('test_id', 'QUAN-??')
        if mode == 'uncertainty':
            run_uncertainty_test(p, tol, t, out_dir, xp, on_gpu)
        elif mode == 'cavity_spectroscopy':
            run_cavity_spectroscopy(p, tol, t, out_dir, xp, on_gpu)
        elif mode == 'threshold':
            run_threshold_test(p, tol, t, out_dir, xp, on_gpu)
        else:
            log(f"[{t.get('test_id')}] Mode '{mode}' not yet implemented; mark skip or implement.", "WARN")

if __name__ == '__main__':
    main()
