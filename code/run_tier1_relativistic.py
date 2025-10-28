#!/usr/bin/env python3
"""
LFM Tier-1 — Relativistic Propagation & Isotropy Suite
----------------------------------------------------
Purpose:
- Execute Tier-1 relativistic propagation and isotropy tests across CPU/GPU
    backends, collect diagnostics, and produce standardized summaries.

Highlights:
- Dual-backend support (NumPy/CuPy) selected by `run_settings.use_gpu` and
    availability of CuPy.
- Keeps arrays on-device during the stepping loop where possible to minimize
    host↔device transfers and avoid mixed-type serialization bugs.
- Converts to NumPy only for host-side diagnostics, plotting, and monitoring.

Config & output:
- Expects configuration at `./config/config_tier1_relativistic.json`.
- Writes per-test results under `<output_dir>/<TEST_ID>/` with
    `summary.json`, `metrics.csv`, `diagnostics/` and `plots/`.
"""

import json, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False
from lfm_console import log, suite_summary, set_logger, log_run_config, test_start, report_progress
from lfm_logger import LFMLogger
from lfm_results import save_summary, write_metadata_bundle
from lfm_diagnostics import field_spectrum, energy_total, energy_flow, phase_corr
from lfm_visualizer import visualize_concept
from energy_monitor import EnergyMonitor
from numeric_integrity import NumericIntegrityMixin


 
def _default_config_name() -> str:
    return "config_tier1_relativistic.json"

def load_config() -> Dict:
    script_dir = Path(__file__).resolve().parent
    for root in (script_dir, script_dir.parent):
        cand = root / "config" / _default_config_name()
        if cand.is_file():
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("Tier-1 config not found (expected config/config_tier1_relativistic.json).")


def resolve_outdir(output_dir_hint: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    outdir = script_dir / output_dir_hint
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


 
def to_numpy(x):
    """Return a NumPy ndarray for host-side routines."""
    if _HAS_CUPY and isinstance(x, cp.ndarray):
        return x.get()
    return np.asarray(x)

def hann_vec(x_len: int) -> np.ndarray:
    return np.hanning(x_len) if x_len else np.array([], dtype=np.float64)


@dataclass
class TestSummary:
    id: str
    description: str
    passed: bool
    rel_err: float
    omega_meas: float
    omega_theory: float
    runtime_sec: float
    k_fraction_lattice: float


 
class Tier1Harness(NumericIntegrityMixin):
    def __init__(self, cfg: Dict, out_root: Path):
        self.cfg = cfg
        self.run_settings = cfg["run_settings"]
        self.base = cfg["parameters"]
        self.tol  = cfg["tolerances"]
        self.variants = cfg["variants"]
        self.quick = bool(self.run_settings.get("quick_mode", False))
        self.use_gpu = bool(self.run_settings.get("use_gpu", False)) and _HAS_CUPY
        self.xp = cp if self.use_gpu else np

        self.out_root = out_root
        self.logger = LFMLogger(self.out_root)
        self.logger.record_env()
        
        try:
            set_logger(self.logger)
        except Exception:
            pass
        
        try:
            log_run_config(self.cfg, self.out_root)
        except Exception:
            pass
        
        self.show_progress = bool(self.run_settings.get("show_progress", True))
        self.progress_percent_stride = int(self.run_settings.get("progress_percent_stride", 5))
        if self.use_gpu:
            log("[accel] Using GPU (CuPy backend).", "INFO")
        else:
            log("[accel] Using CPU (NumPy backend).", "INFO")

    
    def init_field_variant(self, test_id: str, params: Dict, N: int, dx: float, c: float):
        xp = self.xp
        x = xp.arange(N, dtype=xp.float64) * dx

        k_frac = float(params.get("k_fraction", 0.1))
        m = int(round((N * k_frac) / 2.0))
        k_frac_lattice = 2.0 * m / N
        k_cyc = k_frac_lattice * (1.0 / (2.0 * dx))
        k_ang = 2.0 * math.pi * k_cyc
        params["_k_fraction_lattice"] = k_frac_lattice
        params["_k_ang"] = k_ang

        if test_id in ("REL-01", "REL-02"):
            return xp.cos(k_ang * x, dtype=xp.float64)
        elif test_id == "REL-03":
            beta = params.get("boost_factor", 0.2); gamma = 1.0 / math.sqrt(1 - beta**2)
            return xp.cos(gamma * (k_ang * x - beta * k_ang * x), dtype=xp.float64)
        elif test_id == "REL-04":
            beta = params.get("boost_factor", 0.6); gamma = 1.0 / math.sqrt(1 - beta**2)
            return xp.cos(gamma * (k_ang * x - beta * k_ang * x), dtype=xp.float64)
        elif test_id == "REL-05":
            amp = params.get("pulse_amp", 1.0); w = params.get("pulse_width", 0.1); c0 = N // 2
            return amp * xp.exp(-((x - c0 * dx) ** 2) / (2 * w ** 2), dtype=xp.float64)
        elif test_id == "REL-06":
            if xp is np:
                rng = np.random.default_rng(1234)
                noise = rng.standard_normal(N).astype(np.float64)
                return np.ones(N, dtype=np.float64) + params.get("noise_amp", 1e-4) * noise
            else:
                rng = cp.random.default_rng(1234)
                noise = rng.standard_normal(N, dtype=cp.float64)
                return cp.ones(N, dtype=cp.float64) + params.get("noise_amp", 1e-4) * noise
        elif test_id == "REL-07":
            return xp.sin(k_ang * x, dtype=xp.float64)
        elif test_id == "REL-08":
            return xp.cos(k_ang * x, dtype=xp.float64) + 0.5 * xp.sin(2 * k_ang * x, dtype=xp.float64)
        else:
            return xp.cos(k_ang * x, dtype=xp.float64)

    
    def estimate_omega_proj_fft(self, series: np.ndarray, dt: float) -> float:
        data = np.asarray(series, dtype=np.float64)
        data = data - data.mean()
        w = hann_vec(len(data))
        dw = data * w
        spec = np.abs(np.fft.rfft(dw))
        pk = int(np.argmax(spec[1:])) + 1 if len(spec) > 1 else 0
        if 1 <= pk < len(spec) - 1:
            s1, s2, s3 = np.log(spec[pk-1] + 1e-30), np.log(spec[pk] + 1e-30), np.log(spec[pk+1] + 1e-30)
            denom = s1 - 2*s2 + s3
            delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (s1 - s3) / denom
        else:
            delta = 0.0
        df = 1.0 / (len(dw) * dt)
        f_peak = (pk + delta) * df
        return 2.0 * math.pi * abs(f_peak)

    def estimate_omega_phase_slope(self, z_complex: np.ndarray, t_axis: np.ndarray) -> float:
        phi = np.unwrap(np.angle(z_complex)).astype(np.float64)
        w = hann_vec(len(phi))
        A = np.vstack([t_axis, np.ones_like(t_axis)]).T
        Aw = A * w[:, None]; yw = phi * w
        slope, _ = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        return float(abs(slope))

    
    def run_variant(self, v: Dict) -> TestSummary:
        xp = self.xp

        tid  = v["test_id"]
        desc = v.get("description", tid)
        params = self.base.copy(); params.update(v)

        IS_ISO   = tid in ("REL-01", "REL-02")
        IS_GAUGE = tid == "REL-08"
        SAVE_EVERY      = 1
        TARGET_SAMPLES  = 2048 if self.quick else 4096
        steps           = max(params.get("steps", 2000), TARGET_SAMPLES)
        if IS_ISO:
            gamma_damp = 0.0; rescale_each=False; zero_mean=True;  estimator="proj_fft"
        elif IS_GAUGE:
            gamma_damp = 1e-3; rescale_each=True;  zero_mean=False; estimator="phase_slope"
        else:
            gamma_damp = 0.0;  rescale_each=True;  zero_mean=True;  estimator="proj_fft"

        N   = params["grid_points"]; dx = params["dx"]; dt = params["dt"]
        chi = params["chi"];         c  = math.sqrt(params["alpha"] / params["beta"])

        test_dir = self.out_root / tid
        diag_dir = test_dir / "diagnostics"
        plot_dir = test_dir / "plots"
        for d in (test_dir, diag_dir, plot_dir):
            d.mkdir(parents=True, exist_ok=True)

        test_start(tid, desc, steps)
        log(f"Params: steps={steps}, quick={self.quick}, backend={'GPU' if self.use_gpu else 'CPU'}", "INFO")
        log(f"[cfg] gamma={gamma_damp} rescale={rescale_each} zero_mean={zero_mean} est={estimator}", "INFO")

        
        E_prev = self.init_field_variant(tid, params, N, dx, c)
        E = xp.array(E_prev, copy=True)
        params["_k_ang"] = float(params.get("_k_ang", 0.0))
        params["gamma_damp"] = gamma_damp

        
        self.check_cfl(c, dt, dx, ndim=1)
        self.validate_field(to_numpy(E), f"{tid}-E0")

        
        E0 = energy_total(to_numpy(E), to_numpy(E_prev), dt, dx, c, chi)
        mon = EnergyMonitor(dt, dx, c, chi, outdir=str(diag_dir), label=f"{tid}")

        
        x_np  = (np.arange(N) * dx).astype(np.float64)
        k_ang = float(params.get("_k_ang", 0.0))
        cos_k = np.cos(k_ang * x_np)
        sin_k = np.sin(k_ang * x_np)
        cos_norm = float(np.dot(cos_k, cos_k) + 1e-30)
        sin_norm = float(np.dot(sin_k, sin_k) + 1e-30)

        
        # Host-side series + projection buffers
        E_series_host = [to_numpy(E)]
        proj_series: List[float] = []
        z_series: List[complex] = []
        t0 = time.time()

        # Precompute scalars to avoid repeated power ops inside the loop
        dt2 = float(dt) ** 2
        c2 = float(c) ** 2
        chi2 = float(chi) ** 2
        dx2 = float(dx) ** 2
        steps_pct_check = max(1, steps // 100)

        for n in range(steps):
            # Compute laplacian and step (keep all heavy work on xp: NumPy or CuPy)
            Em1 = xp.roll(E, -1); Ep1 = xp.roll(E, 1)
            lap = (Em1 - 2 * E + Ep1) / dx2
            E_next = (2 - gamma_damp) * E - (1 - gamma_damp) * E_prev + dt2 * (c2 * lap - chi2 * E)

            if zero_mean:
                # subtract mean in-place equivalent (creates a temporary on xp)
                E_next = E_next - xp.mean(E_next)

            if rescale_each:
                # compute norm and scale on device, convert only scalar to Python
                denom = float(xp.sum(E_next * E_next)) + 1e-30
                scale = math.sqrt(float(E0) / denom)
                E_next = E_next * scale

            # advance
            E_prev, E = E, E_next

            # Convert to host once per step and reuse for monitoring, appends and diagnostics
            host_E = to_numpy(E)
            host_E_prev = to_numpy(E_prev)

            drift_now = mon.record(host_E, host_E_prev, n)

            # progress reporting throttled to precomputed stride
            if self.show_progress and (n % steps_pct_check == 0):
                pct = int((n + 1) * 100 / max(1, steps))
                if pct % max(1, self.progress_percent_stride) == 0:
                    report_progress(tid, pct)

            # store host-side series at configured cadence
            if n % SAVE_EVERY == 0:
                E_series_host.append(host_E)

            # compute projections on host arrays (single conversion used)
            arr = host_E.astype(np.float64)
            if zero_mean:
                arr = arr - arr.mean()
            proj_series.append(float(np.dot(arr, cos_k) / cos_norm))
            z_series.append(complex(np.dot(arr, cos_k), np.dot(arr, sin_k)) / (cos_norm + sin_norm))

            self.validate_energy(drift_now, tol=1e-6, label=f"{tid}-step{n}")

        runtime = time.time() - t0
        mon.finalize()

        
        try:
            field_spectrum(E_series_host[-1], dx, diag_dir)
            energy_flow(E_series_host, dt, dx, c, diag_dir)
            phase_corr(E_series_host, diag_dir)
        except Exception as e:
            log(f"[WARN] Diagnostics failed for {tid}: {e}", "WARN")

        
        try:
            visualize_concept(E_series_host, tier=1, test_id=tid, outdir=plot_dir, quick=self.quick, animate=not self.quick)
        except Exception as e:
            log(f"[WARN] Visualization failed for {tid}: {e}", "WARN")

        
        if estimator == "proj_fft":
            omega_meas = self.estimate_omega_proj_fft(np.array(proj_series, dtype=np.float64), dt)
        else:
            t_axis = np.arange(len(z_series)) * dt
            omega_meas = self.estimate_omega_phase_slope(np.array(z_series, dtype=np.complex128), t_axis)

        kdx = params.get("_k_fraction_lattice", params.get("k_fraction", 0.1)) * math.pi
        omega_theory = math.sqrt(((2.0 * c / dx) ** 2) * 0.5 * (1.0 - math.cos(kdx)) + chi ** 2)

        rel_err = abs(omega_meas - omega_theory) / max(omega_theory, 1e-30)
        passed = bool(rel_err <= float(self.tol.get("phase_error_max", 0.02)))
        status = "PASS ✅" if passed else "FAIL ❌"
        level  = "INFO" if passed else "FAIL"
        log(f"{tid} {status} (rel_err={rel_err*100:.3f}%, ω_meas={omega_meas:.6f}, ω_th={omega_theory:.6f})", level)

        summary = {
            "id": tid, "description": desc, "passed": passed,
            "rel_err": float(rel_err), "omega_meas": float(omega_meas),
            "omega_theory": float(omega_theory), "runtime_sec": float(runtime),
            "k_fraction_lattice": float(params.get("_k_fraction_lattice", 0)),
            "quick_mode": self.quick,
            "backend": "GPU" if self.use_gpu else "CPU",
            "params": {
                "N": N, "dx": dx, "dt": dt, "alpha": params["alpha"], "beta": params["beta"],
                "chi": chi, "gamma_damp": gamma_damp, "rescale_each": rescale_each,
                "zero_mean": zero_mean, "estimator": estimator, "steps": steps
            }
        }
        metrics = [
            ("rel_err", rel_err),
            ("omega_meas", omega_meas),
            ("omega_theory", omega_theory),
            ("runtime_sec", runtime),
        ]
        save_summary(test_dir, tid, summary, metrics=metrics)

        return TestSummary(
            id=tid, description=desc, passed=passed, rel_err=rel_err,
            omega_meas=omega_meas, omega_theory=omega_theory,
            runtime_sec=runtime, k_fraction_lattice=float(params.get("_k_fraction_lattice", 0.0))
        )

    # ------------------------------ Run --------------------------------
    def run(self) -> List[Dict]:
        results = []
        for v in self.variants:
            res = self.run_variant(v)
            results.append({
                "test_id": res.id,
                "description": res.description,
                "passed": res.passed,
                "rel_err": res.rel_err,
                "omega_meas": res.omega_meas,
                "omega_theory": res.omega_theory,
                "runtime_sec": res.runtime_sec,
                "k_fraction_lattice": res.k_fraction_lattice,
            })
        return results


# --------------------------------- Main --------------------------------
def main():
    cfg = load_config()
    outdir = resolve_outdir(cfg.get("output_dir", "results/Tier1"))
    harness = Tier1Harness(cfg, outdir)

    log(f"[paths] OUTPUT ROOT = {outdir}", "INFO")
    log(f"=== Tier-1 Relativistic Suite Start (quick={harness.quick}) ===", "INFO")

    results = harness.run()
    suite_summary(results)
    write_metadata_bundle(outdir, test_id="TIER1-SUITE", tier=1, category="relativistic")

    log("=== Tier-1 Suite Complete ===", "INFO")


if __name__ == "__main__":
    main()
