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

def load_config(config_path: str = None) -> Dict:
    if config_path:
        # Use explicit path if provided
        cand = Path(config_path)
        if cand.is_file():
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Default: search in standard locations
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

    
    def init_field_variant(self, test_id: str, params: Dict, N: int, dx: float, c: float, direction: str = "right"):
        """
        Initialize field for test variant.
        
        Args:
            direction: "right" for rightward propagation, "left" for leftward (isotropy tests only)
        """
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
            # Isotropy test: create standing wave that will be given directional momentum
            return xp.cos(k_ang * x, dtype=xp.float64)
        elif test_id == "REL-03":
            beta = params.get("boost_factor", 0.2); gamma = 1.0 / math.sqrt(1 - beta**2)
            return xp.cos(gamma * (k_ang * x - beta * k_ang * x), dtype=xp.float64)
        elif test_id == "REL-04":
            beta = params.get("boost_factor", 0.6); gamma = 1.0 / math.sqrt(1 - beta**2)
            return xp.cos(gamma * (k_ang * x - beta * k_ang * x), dtype=xp.float64)
        elif test_id == "REL-05":
            # Causality test: off-center Gaussian pulse that will propagate
            amp = params.get("pulse_amp", 1.0); w = params.get("pulse_width", 0.1)
            c0 = N // 4  # Start at 1/4 position, not center, so it propagates
            return amp * xp.exp(-((x - c0 * dx) ** 2) / (2 * w ** 2), dtype=xp.float64)
        elif test_id == "REL-06":
            # Causality test: localized noise burst (not uniform background)
            if xp is np:
                rng = np.random.default_rng(1234)
                noise = rng.standard_normal(N).astype(np.float64)
            else:
                rng = cp.random.default_rng(1234)
                noise = rng.standard_normal(N, dtype=cp.float64)
            # Localize noise to central region using window
            c0 = N // 2; w = N // 8
            window = xp.exp(-((xp.arange(N) - c0) ** 2) / (2 * w ** 2), dtype=xp.float64)
            return params.get("noise_amp", 0.1) * noise * window
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

    def measure_isotropy(self, E_right: List[np.ndarray], E_left: List[np.ndarray], 
                        dt: float, dx: float, k_ang: float) -> Dict:
        """
        Test isotropy by comparing dispersion for left and right propagating waves.
        
        In 1D, isotropy means the wave equation has no preferred direction.
        We verify this by checking that ω_right(k) = ω_left(-k).
        
        Args:
            E_right: Time series for right-propagating wave
            E_left: Time series for left-propagating wave
            dt: Time step
            dx: Spatial step
            k_ang: Angular wavenumber
            
        Returns:
            Dict with omega_right, omega_left, anisotropy, passed
        """
        N = len(E_right[0])
        x_positions = np.arange(N) * dx
        
        # Project onto traveling wave modes
        cos_k = np.cos(k_ang * x_positions)
        sin_k = np.sin(k_ang * x_positions)
        cos_norm = float(np.dot(cos_k, cos_k) + 1e-30)
        sin_norm = float(np.dot(sin_k, sin_k) + 1e-30)
        
        # Extract time series for right-propagating wave
        proj_right = []
        z_right = []
        for E in E_right:
            E_np = to_numpy(E).astype(np.float64)
            E_np = E_np - E_np.mean()  # Zero mean
            proj_right.append(float(np.dot(E_np, cos_k) / cos_norm))
            z_right.append(complex(np.dot(E_np, cos_k), np.dot(E_np, sin_k)) / (cos_norm + sin_norm))
        
        # Extract time series for left-propagating wave
        proj_left = []
        z_left = []
        for E in E_left:
            E_np = to_numpy(E).astype(np.float64)
            E_np = E_np - E_np.mean()  # Zero mean
            proj_left.append(float(np.dot(E_np, cos_k) / cos_norm))
            z_left.append(complex(np.dot(E_np, cos_k), np.dot(E_np, sin_k)) / (cos_norm + sin_norm))
        
        # Measure frequency for both
        omega_right = self.estimate_omega_proj_fft(np.array(proj_right, dtype=np.float64), dt)
        omega_left = self.estimate_omega_proj_fft(np.array(proj_left, dtype=np.float64), dt)
        
        # Isotropy: both should give same ω
        anisotropy = abs(omega_right - omega_left) / max(omega_right, omega_left, 1e-30)
        
        # Pass if anisotropy is small
        passed = anisotropy <= 0.01  # 1% tolerance
        
        return {
            "omega_right": omega_right,
            "omega_left": omega_left,
            "anisotropy": anisotropy,
            "passed": passed,
            "message": f"ω_R={omega_right:.6f}, ω_L={omega_left:.6f}, anisotropy={anisotropy*100:.3f}%"
        }

    
    def measure_causality(self, E_series: List[np.ndarray], dx: float, dt: float, c: float, 
                          test_id: str, initial_center: int) -> Dict:
        """
        Measure propagation speed and verify causality (v ≤ c).
        
        For pulse (REL-05): track energy centroid movement over time
        For noise (REL-06): track maximum extent of perturbation from initial distribution
        
        Returns dict with v_measured, max_violation, passed
        """
        N = len(E_series[0])
        x_positions = np.arange(N) * dx
        
        # Determine initial center
        if initial_center is not None:
            x_center_initial = initial_center * dx
        else:
            # For noise field, compute center of mass of initial |E|²
            E0 = E_series[0]
            E0_squared = E0 ** 2
            total_energy = np.sum(E0_squared)
            if total_energy > 0:
                x_center_initial = np.sum(x_positions * E0_squared) / total_energy
            else:
                x_center_initial = 0.5 * N * dx  # Default to domain center
        
        centroid_positions = []
        times = []
        max_violations = []
        
        for step_idx, E in enumerate(E_series):
            t = step_idx * dt
            if t == 0:
                continue
                
            # Compute energy centroid: <x> = Σ(x * |E|²) / Σ|E|²
            E_squared = E ** 2
            total_energy = np.sum(E_squared)
            
            if total_energy > 1e-30:
                x_centroid = np.sum(x_positions * E_squared) / total_energy
            else:
                x_centroid = x_center_initial
            
            # Distance of centroid from initial position
            displacement = abs(x_centroid - x_center_initial)
            
            # Light cone limit: centroid displacement should not exceed c*t
            light_cone_limit = c * t
            violation = displacement - light_cone_limit
            
            max_violations.append(violation)
            centroid_positions.append(displacement)
            times.append(t)
        
        if len(times) < 2:
            return {
                "v_measured": 0.0,
                "v_theory": c,
                "max_violation": 0.0,
                "rel_error": 0.0,
                "passed": False,
                "message": "Insufficient data points"
            }
        
        # Estimate propagation speed via linear fit of centroid displacement vs time
        times_arr = np.array(times)
        centroid_arr = np.array(centroid_positions)
        
        # Linear fit: displacement = v*t + offset
        A = np.vstack([times_arr, np.ones(len(times_arr))]).T
        v_measured, offset = np.linalg.lstsq(A, centroid_arr, rcond=None)[0]
        v_measured = abs(float(v_measured))
        
        # Maximum causality violation across all timesteps
        max_violation = float(np.max(max_violations)) if max_violations else 0.0
        
        # Relative error in speed (should be ≤ c)
        rel_error = abs(v_measured - c) / c if c > 0 else 0.0
        
        # Pass criteria: 
        # 1. Measured speed should not significantly exceed c (allow small numerical error)
        # 2. No timestep should violate light cone by more than tolerance
        # For noise (REL-06), allow larger tolerance due to broader spectrum and dispersion
        tolerance_factor = 10.0 if "REL-06" in test_id else 2.0
        
        speed_ok = v_measured <= c * 1.05  # Allow 5% numerical overshoot
        violation_ok = max_violation <= tolerance_factor * dx  # ~2-10 grid points tolerance
        
        passed = speed_ok and violation_ok
        
        message = f"v={v_measured:.6f} (theory={c:.6f}), max_violation={max_violation:.6e}"
        if not speed_ok:
            message += f" [SPEED VIOLATION: v/c={v_measured/c:.3f}]"
        if not violation_ok:
            message += f" [LIGHT CONE VIOLATION: {max_violation/dx:.1f} grid points]"
        
        return {
            "v_measured": v_measured,
            "v_theory": c,
            "max_violation": max_violation,
            "rel_error": rel_error,
            "passed": passed,
            "message": message,
            "centroid_positions": centroid_positions,
            "times": times
        }

    def run_variant(self, v: Dict) -> TestSummary:
        """Run a single test variant. For isotropy tests, delegates to run_isotropy_variant."""
        tid = v["test_id"]
        
        # Isotropy tests require special handling (run twice with different directions)
        if tid in ("REL-01", "REL-02"):
            return self.run_isotropy_variant(v)
        
        # All other tests use standard single-run logic
        return self.run_standard_variant(v)
    
    def run_isotropy_variant(self, v: Dict) -> TestSummary:
        """Run isotropy test by comparing left and right propagating waves."""
        xp = self.xp
        tid = v["test_id"]
        desc = v.get("description", tid)
        params = self.base.copy(); params.update(v)
        
        N = params["grid_points"]; dx = params["dx"]; dt = params["dt"]
        chi = params["chi"]; c = math.sqrt(params["alpha"] / params["beta"])
        
        test_dir = self.out_root / tid
        diag_dir = test_dir / "diagnostics"
        plot_dir = test_dir / "plots"
        for d in (test_dir, diag_dir, plot_dir):
            d.mkdir(parents=True, exist_ok=True)
        
        test_start(tid, desc, params.get("steps", 2000))
        log(f"Isotropy test: running RIGHT and LEFT propagating waves", "INFO")
        
        # Run simulation for both directions
        E_series_right = self._run_directional_wave(params, N, dx, dt, c, chi, tid, "right")
        E_series_left = self._run_directional_wave(params, N, dx, dt, c, chi, tid, "left")
        
        # Measure isotropy
        k_ang = float(params.get("_k_ang", 0.0))
        iso_result = self.measure_isotropy(E_series_right, E_series_left, dt, dx, k_ang)
        
        anisotropy = iso_result["anisotropy"]
        passed = iso_result["passed"]
        
        status = "PASS ✅" if passed else "FAIL ❌"
        level = "INFO" if passed else "FAIL"
        log(f"{tid} {status} {iso_result['message']}", level)
        
        summary = {
            "id": tid, "description": desc, "passed": passed,
            "anisotropy": float(anisotropy),
            "omega_right": float(iso_result["omega_right"]),
            "omega_left": float(iso_result["omega_left"]),
            "backend": "GPU" if self.use_gpu else "CPU",
        }
        metrics = [
            ("anisotropy", anisotropy),
            ("omega_right", iso_result["omega_right"]),
            ("omega_left", iso_result["omega_left"]),
        ]
        save_summary(test_dir, tid, summary, metrics=metrics)
        
        return TestSummary(
            id=tid, description=desc, passed=passed, rel_err=anisotropy,
            omega_meas=iso_result["omega_right"], omega_theory=iso_result["omega_left"],
            runtime_sec=0.0, k_fraction_lattice=float(params.get("_k_fraction_lattice", 0.0))
        )
    
    def _run_directional_wave(self, params: Dict, N: int, dx: float, dt: float, c: float, chi: float, tid: str, direction: str) -> List[np.ndarray]:
        """Helper to run simulation with directional initial momentum."""
        xp = self.xp
        steps = max(params.get("steps", 2000), 2048 if self.quick else 4096)
        
        # Initialize field
        E_prev = self.init_field_variant(tid, params, N, dx, c, direction)
        E = xp.array(E_prev, copy=True)
        
        # Add directional momentum: E_t = ±c * dE/dx
        dE_dx = (xp.roll(E, -1) - xp.roll(E, 1)) / (2 * dx)
        sign = -1 if direction == "right" else +1  # Right: -c*dE/dx, Left: +c*dE/dx
        E_t_initial = sign * c * dE_dx
        E_prev = E - dt * E_t_initial
        
        E_series = [to_numpy(E)]
        
        # Time integration (simplified, no monitoring for sub-runs)
        dt2 = dt ** 2; c2 = c ** 2; chi2 = chi ** 2; dx2 = dx ** 2
        
        for n in range(steps):
            Em1 = xp.roll(E, -1); Ep1 = xp.roll(E, 1)
            lap = (Em1 - 2 * E + Ep1) / dx2
            E_next = 2 * E - E_prev + dt2 * (c2 * lap - chi2 * E)
            
            # Zero mean
            E_next = E_next - xp.mean(E_next)
            
            E_prev, E = E, E_next
            
            if n % 1 == 0:  # Save every step
                E_series.append(to_numpy(E))
        
        return E_series
    
    def run_standard_variant(self, v: Dict) -> TestSummary:
        """Standard single-run test variant (non-isotropy tests)."""
        xp = self.xp

        tid  = v["test_id"]
        desc = v.get("description", tid)
        params = self.base.copy(); params.update(v)

        IS_ISO   = tid in ("REL-01", "REL-02")
        IS_GAUGE = tid == "REL-08"
        IS_CAUSALITY = tid in ("REL-05", "REL-06")
        SAVE_EVERY      = 1
        TARGET_SAMPLES  = 2048 if self.quick else 4096
        steps           = max(params.get("steps", 2000), TARGET_SAMPLES)
        if IS_ISO:
            gamma_damp = 0.0; rescale_each=False; zero_mean=True;  estimator="proj_fft"
        elif IS_GAUGE:
            gamma_damp = 1e-3; rescale_each=True;  zero_mean=False; estimator="phase_slope"
        elif IS_CAUSALITY:
            gamma_damp = 0.0; rescale_each=False; zero_mean=False; estimator="causality"
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
        
        # For causality tests, add initial rightward velocity to create propagating wave
        if IS_CAUSALITY:
            # E_t ≈ (E - E_prev)/dt, so E_prev = E - dt*E_t
            # For rightward propagation: E_t ≈ -c * dE/dx
            # Approximate with finite difference: dE/dx ≈ (E[i+1] - E[i-1])/(2*dx)
            dE_dx = (xp.roll(E, -1) - xp.roll(E, 1)) / (2 * dx)
            E_t_initial = -c * dE_dx  # Rightward propagation
            E_prev = E - dt * E_t_initial
        
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

        
        # Choose validation method based on test type
        if IS_CAUSALITY:
            # Causality validation: measure signal propagation speed
            initial_center = N // 4 if tid == "REL-05" else N // 2  # REL-05 starts at 1/4, REL-06 at center
            causality_result = self.measure_causality(E_series_host, dx, dt, c, tid, initial_center)
            
            rel_err = causality_result["rel_error"]
            passed = causality_result["passed"]
            omega_meas = causality_result["v_measured"]  # Store as omega_meas for consistency
            omega_theory = causality_result["v_theory"]
            causality_msg = causality_result["message"]
            
            status = "PASS ✅" if passed else "FAIL ❌"
            level  = "INFO" if passed else "FAIL"
            log(f"{tid} {status} {causality_msg}", level)
            
            summary = {
                "id": tid, "description": desc, "passed": passed,
                "rel_err": float(rel_err), 
                "v_measured": float(causality_result["v_measured"]),
                "v_theory": float(causality_result["v_theory"]),
                "max_violation": float(causality_result["max_violation"]),
                "runtime_sec": float(runtime),
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
                ("v_measured", causality_result["v_measured"]),
                ("v_theory", causality_result["v_theory"]),
                ("max_violation", causality_result["max_violation"]),
                ("runtime_sec", runtime),
            ]
        else:
            # Frequency validation: measure dispersion relation
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
    import argparse
    parser = argparse.ArgumentParser(description="Tier-1 Relativistic Test Suite")
    parser.add_argument("--test", type=str, default=None, 
                       help="Run single test by ID (e.g., REL-05). If omitted, runs all tests.")
    parser.add_argument("--config", type=str, default="config/config_tier1_relativistic.json",
                       help="Path to config file")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    outdir = resolve_outdir(cfg.get("output_dir", "results/Tier1"))
    harness = Tier1Harness(cfg, outdir)

    log(f"[paths] OUTPUT ROOT = {outdir}", "INFO")
    
    # Filter to single test if requested
    if args.test:
        harness.variants = [v for v in harness.variants if v["test_id"] == args.test]
        if not harness.variants:
            log(f"[ERROR] Test '{args.test}' not found in config", "FAIL")
            return
        log(f"=== Running Single Test: {args.test} ===", "INFO")
    else:
        log(f"=== Tier-1 Relativistic Suite Start (quick={harness.quick}) ===", "INFO")

    results = harness.run()
    
    if args.test:
        # Single test: just show result
        log(f"=== Test {args.test} Complete ===", "INFO")
    else:
        # Full suite: show summary
        suite_summary(results)
        write_metadata_bundle(outdir, test_id="TIER1-SUITE", tier=1, category="relativistic")
        log("=== Tier-1 Suite Complete ===", "INFO")


if __name__ == "__main__":
    main()
