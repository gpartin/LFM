#!/usr/bin/env python3
"""
LFM Tier-1 — Relativistic Propagation & Isotropy Suite
v2.1.0 — dual backend (CPU/GPU) with safe host↔device boundaries

- Selects CuPy backend only if available AND config["run_settings"]["use_gpu"] is true.
- Keeps arrays on-device during the step loop to avoid dtype/class bugs.
- Converts to NumPy only when needed (estimators, diagnostics, plotting, monitor).
- No class-based casting; uses xp.asarray / cp.asarray explicitly.

This script expects a config at ./config/config_tier1_relativistic.json
"""

import json, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

# Optional CuPy for GPU acceleration
try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

# Project utilities
from lfm_console import log, suite_summary
from lfm_logger import LFMLogger
from lfm_results import save_summary, write_metadata_bundle
from lfm_diagnostics import field_spectrum, energy_total, energy_flow, phase_corr
from lfm_visualizer import visualize_concept
from energy_monitor import EnergyMonitor
from numeric_integrity import NumericIntegrityMixin


# ------------------------------- Config --------------------------------
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
    project_root = script_dir.parent
    outdir = project_root / output_dir_hint
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# ----------------------------- Utilities -------------------------------
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


# ------------------------------ Harness --------------------------------
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
        if self.use_gpu:
            log("[accel] Using GPU (CuPy backend).", "INFO")
        else:
            log("[accel] Using CPU (NumPy backend).", "INFO")

    # -------------------------- Field init --------------------------
    def init_field_variant(self, test_id: str, params: Dict, N: int, dx: float, c: float):
        xp = self.xp
        x = xp.arange(N, dtype=xp.float64) * dx

        k_frac = float(params.get("k_fraction", 0.1))
        m = int(round((N * k_frac) / 2.0))  # snap to 2m/N lattice fraction
        k_frac_lattice = 2.0 * m / N
        k_cyc = k_frac_lattice * (1.0 / (2.0 * dx))     # cycles / unit
        k_ang = 2.0 * math.pi * k_cyc                   # rad / unit
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
                # CuPy's Generator
                rng = cp.random.default_rng(1234)
                noise = rng.standard_normal(N, dtype=cp.float64)
                return cp.ones(N, dtype=cp.float64) + params.get("noise_amp", 1e-4) * noise
        elif test_id == "REL-07":
            return xp.sin(k_ang * x, dtype=xp.float64)
        elif test_id == "REL-08":
            return xp.cos(k_ang * x, dtype=xp.float64) + 0.5 * xp.sin(2 * k_ang * x, dtype=xp.float64)
        else:
            return xp.cos(k_ang * x, dtype=xp.float64)

    # -------------------------- Estimators --------------------------
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

    # -------------------------- Orchestration -----------------------
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

        log(f"→ Starting {tid}: {desc} (steps={steps}, quick={self.quick}, backend={'GPU' if self.use_gpu else 'CPU'})", "INFO")
        log(f"[cfg] gamma={gamma_damp} rescale={rescale_each} zero_mean={zero_mean} est={estimator}", "INFO")

        # Initialize field on selected backend
        E_prev = self.init_field_variant(tid, params, N, dx, c)
        E = xp.array(E_prev, copy=True)
        params["_k_ang"] = float(params.get("_k_ang", 0.0))
        params["gamma_damp"] = gamma_damp

        # Integrity checks (CFL & finite field) — do on host for safety
        self.check_cfl(c, dt, dx, ndim=1)
        self.validate_field(to_numpy(E), f"{tid}-E0")

        # Baseline energy (compensated, host) + monitor
        E0 = energy_total(to_numpy(E), to_numpy(E_prev), dt, dx, c, chi)
        mon = EnergyMonitor(dt, dx, c, chi, outdir=str(diag_dir), label=f"{tid}")

        # Precompute host projection bases
        x_np  = (np.arange(N) * dx).astype(np.float64)
        k_ang = float(params.get("_k_ang", 0.0))
        cos_k = np.cos(k_ang * x_np)
        sin_k = np.sin(k_ang * x_np)
        cos_norm = float(np.dot(cos_k, cos_k) + 1e-30)
        sin_norm = float(np.dot(sin_k, sin_k) + 1e-30)

        # Time stepping (on selected backend)
        E_series_host = [to_numpy(E)]
        proj_series: List[float] = []
        z_series: List[complex] = []
        t0 = time.time()

        for n in range(steps):
            # Laplacian (periodic) on backend
            Em1 = xp.roll(E, -1); Ep1 = xp.roll(E, 1)
            lap = (Em1 - 2 * E + Ep1) / (dx**2)
            E_next = (2 - gamma_damp) * E - (1 - gamma_damp) * E_prev + (dt ** 2) * ((c ** 2) * lap - (chi ** 2) * E)

            if zero_mean:
                E_next = E_next - xp.mean(E_next)

            if rescale_each:
                # Compute norm on backend, rescale to initial energy proxy
                denom = xp.sum(E_next * E_next) + 1e-30
                scale = math.sqrt(float(E0) / float(denom))
                E_next = E_next * scale

            # Roll state
            E_prev, E = E, E_next

            # Monitor + integrity (host-side)
            drift_now = mon.record(to_numpy(E), to_numpy(E_prev), n)

            # Save sparsely for visuals/diagnostics (host snapshots)
            if n % SAVE_EVERY == 0:
                E_series_host.append(to_numpy(E))

            # Estimator streams (host)
            arr = to_numpy(E).astype(np.float64)
            arr = arr - (arr.mean() if zero_mean else 0.0)
            proj_series.append(float(np.dot(arr, cos_k) / cos_norm))
            z_series.append(complex(np.dot(arr, cos_k), np.dot(arr, sin_k)) / (cos_norm + sin_norm))

            self.validate_energy(drift_now, tol=1e-6, label=f"{tid}-step{n}")

        runtime = time.time() - t0
        mon.finalize()

        # Diagnostics (host)
        try:
            field_spectrum(E_series_host[-1], dx, diag_dir)
            energy_flow(E_series_host, dt, dx, c, diag_dir)
            phase_corr(E_series_host, diag_dir)
        except Exception as e:
            log(f"[WARN] Diagnostics failed for {tid}: {e}", "WARN")

        # Visualization (host)
        try:
            visualize_concept(E_series_host, tier=1, test_id=tid, outdir=plot_dir, quick=self.quick, animate=not self.quick)
        except Exception as e:
            log(f"[WARN] Visualization failed for {tid}: {e}", "WARN")

        # Frequency estimation (host)
        if estimator == "proj_fft":
            omega_meas = self.estimate_omega_proj_fft(np.array(proj_series, dtype=np.float64), dt)
        else:
            t_axis = np.arange(len(z_series)) * dt
            omega_meas = self.estimate_omega_phase_slope(np.array(z_series, dtype=np.complex128), t_axis)

        # Lattice dispersion (discrete) with 1/2 factor
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
