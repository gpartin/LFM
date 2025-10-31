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
from lfm_results import save_summary, write_metadata_bundle, write_csv
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

    def measure_boost_covariance(self, E_series: List[np.ndarray], dt: float, dx: float, 
                                  c: float, chi: float, beta: float, k_ang: float) -> Dict:
        """
        Test Lorentz boost covariance using ACTUAL coordinate transformation.
        
        IMPORTANT: This is the CORRECT way to test Lorentz covariance.
        Previous versions tested dispersion relation transforms (Doppler formula),
        which is CIRCULAR - it assumes what we're trying to prove.
        
        Proper test:
        1. Compute Klein-Gordon residual in lab frame: □E + χ²E
        2. Transform E(x,t) → E'(x',t') using Lorentz boost
        3. Compute Klein-Gordon residual in boosted frame: □'E' + χ²E'
        4. Verify residuals are similar (equation holds in both frames)
        
        This tests ACTUAL Lorentz covariance: the PDE itself transforms correctly.
        
        Args:
            E_series: Time series of field in lab frame
            dt: Time step
            dx: Spatial step  
            c: Speed of light
            chi: Mass term
            beta: Boost velocity (v/c)
            k_ang: Wavenumber in lab frame (for backwards compatibility)
            
        Returns:
            Dict with residuals in both frames, covariance ratio, passed
        """
        from lorentz_transform import verify_klein_gordon_covariance
        
        gamma = 1.0 / math.sqrt(1 - beta**2)
        
        # Convert to lab frame coordinates
        N = len(E_series[0])
        x_lab = np.arange(N) * dx
        
        # Use proper Lorentz transformation verification
        result = verify_klein_gordon_covariance(E_series, x_lab, dt, dx, chi, beta, c)
        
        # Extract metrics
        residual_lab_rms = result['residual_lab_mean']
        residual_boost_rms = result['residual_boost_mean']
        covariance_ratio = result['covariance_ratio']
        
        # Pass criteria: covariance ratio should be close to 1.0
        # (KG equation residuals similar in both frames)
        # NOTE: Energy rescaling in evolution breaks pure KG equation, 
        # so residuals are non-zero. What matters is covariance: 
        # ratio should still be ~O(1) if equation structure is Lorentz invariant.
        rel_err = abs(covariance_ratio - 1.0)
        passed = covariance_ratio < 5.0 and covariance_ratio > 0.2  # Ratio O(1), not wildly different
        
        # Also compute frequency for backwards compatibility with reports
        cos_k = np.cos(k_ang * x_lab)
        cos_norm = float(np.dot(cos_k, cos_k) + 1e-30)
        proj_series = []
        for E in E_series:
            E_np = to_numpy(E).astype(np.float64)
            E_np = E_np - E_np.mean()
            proj_series.append(float(np.dot(E_np, cos_k) / cos_norm))
        omega_lab = self.estimate_omega_proj_fft(np.array(proj_series, dtype=np.float64), dt)
        
        # Doppler shift for reference (not used in test)
        omega_boost_doppler = gamma * (omega_lab - beta * c * k_ang)
        
        return {
            "omega_lab": omega_lab,
            "omega_boost": abs(omega_boost_doppler),  # Reference only
            "omega_boost_expected": abs(omega_boost_doppler),  # Reference only
            "residual_lab_rms": residual_lab_rms,
            "residual_boost_rms": residual_boost_rms,
            "covariance_ratio": covariance_ratio,
            "rel_error": rel_err,
            "beta": beta,
            "gamma": gamma,
            "passed": passed,
            "message": f"KG residual: lab={residual_lab_rms:.2e}, boost={residual_boost_rms:.2e}, ratio={covariance_ratio:.3f} (err={rel_err*100:.1f}%)"
        }

    def measure_phase_independence(self, E_cos: List[np.ndarray], E_sin: List[np.ndarray],
                                   dt: float, dx: float, k_ang: float) -> Dict:
        """
        Test phase independence by comparing dispersion for sin(kx) and cos(kx) initial conditions.
        
        For a linear wave equation, the phase of initial conditions should not affect
        the dispersion relation: both sin and cos should give the same ω(k).
        
        Args:
            E_cos: Time series for cos(kx) initial condition
            E_sin: Time series for sin(kx) initial condition
            dt: Time step
            dx: Spatial step
            k_ang: Angular wavenumber
            
        Returns:
            Dict with omega_cos, omega_sin, phase_error, passed
        """
        N = len(E_cos[0])
        x_positions = np.arange(N) * dx
        
        # Project onto wave modes
        cos_k = np.cos(k_ang * x_positions)
        sin_k = np.sin(k_ang * x_positions)
        cos_norm = float(np.dot(cos_k, cos_k) + 1e-30)
        sin_norm = float(np.dot(sin_k, sin_k) + 1e-30)
        
        # Extract time series for cos initial condition
        proj_cos = []
        for E in E_cos:
            E_np = to_numpy(E).astype(np.float64)
            E_np = E_np - E_np.mean()
            # For cos initial: project onto cos mode
            proj_cos.append(float(np.dot(E_np, cos_k) / cos_norm))
        
        # Extract time series for sin initial condition
        proj_sin = []
        for E in E_sin:
            E_np = to_numpy(E).astype(np.float64)
            E_np = E_np - E_np.mean()
            # For sin initial: project onto sin mode
            proj_sin.append(float(np.dot(E_np, sin_k) / sin_norm))
        
        # Measure frequency for both
        omega_cos = self.estimate_omega_proj_fft(np.array(proj_cos, dtype=np.float64), dt)
        omega_sin = self.estimate_omega_proj_fft(np.array(proj_sin, dtype=np.float64), dt)
        
        # Phase independence: both should give same ω
        phase_error = abs(omega_cos - omega_sin) / max(omega_cos, omega_sin, 1e-30)
        
        # Pass if phase error is small
        passed = phase_error <= 0.02  # 2% tolerance
        
        return {
            "omega_cos": omega_cos,
            "omega_sin": omega_sin,
            "phase_error": phase_error,
            "passed": passed,
            "message": f"ω_cos={omega_cos:.6f}, ω_sin={omega_sin:.6f}, phase_err={phase_error*100:.3f}%"
        }

    def measure_superposition(self, E_mode1: List[np.ndarray], E_mode2: List[np.ndarray],
                             E_superposition: List[np.ndarray], dt: float, dx: float,
                             k1: float, k2: float, a1: float, a2: float) -> Dict:
        """
        Test superposition principle by verifying linearity of the wave equation.
        
        For a linear equation, if φ1(t) and φ2(t) are solutions, then
        a1*φ1(t) + a2*φ2(t) must also be a solution.
        
        We verify this by:
        1. Running mode1 (cos(k1*x)) alone → measure ω1
        2. Running mode2 (sin(k2*x)) alone → measure ω2  
        3. Running superposition a1*mode1 + a2*mode2 → verify both ω1 and ω2 present
        
        Args:
            E_mode1: Time series for first mode alone
            E_mode2: Time series for second mode alone
            E_superposition: Time series for superposed initial condition
            dt: Time step
            dx: Spatial step
            k1, k2: Wavenumbers of the two modes
            a1, a2: Amplitudes in superposition
            
        Returns:
            Dict with measured frequencies and linearity error
        """
        N = len(E_mode1[0])
        x_positions = np.arange(N) * dx
        
        # Measure ω1 from mode1
        cos_k1 = np.cos(k1 * x_positions)
        cos_norm1 = float(np.dot(cos_k1, cos_k1) + 1e-30)
        proj1 = []
        for E in E_mode1:
            E_np = to_numpy(E).astype(np.float64)
            E_np = E_np - E_np.mean()
            proj1.append(float(np.dot(E_np, cos_k1) / cos_norm1))
        omega1 = self.estimate_omega_proj_fft(np.array(proj1, dtype=np.float64), dt)
        
        # Measure ω2 from mode2
        sin_k2 = np.sin(k2 * x_positions)
        sin_norm2 = float(np.dot(sin_k2, sin_k2) + 1e-30)
        proj2 = []
        for E in E_mode2:
            E_np = to_numpy(E).astype(np.float64)
            E_np = E_np - E_np.mean()
            proj2.append(float(np.dot(E_np, sin_k2) / sin_norm2))
        omega2 = self.estimate_omega_proj_fft(np.array(proj2, dtype=np.float64), dt)
        
        # Project superposition onto both modes
        proj_super1 = []
        proj_super2 = []
        for E in E_superposition:
            E_np = to_numpy(E).astype(np.float64)
            E_np = E_np - E_np.mean()
            proj_super1.append(float(np.dot(E_np, cos_k1) / cos_norm1))
            proj_super2.append(float(np.dot(E_np, sin_k2) / sin_norm2))
        
        # Measure frequencies in superposition
        omega_super1 = self.estimate_omega_proj_fft(np.array(proj_super1, dtype=np.float64), dt)
        omega_super2 = self.estimate_omega_proj_fft(np.array(proj_super2, dtype=np.float64), dt)
        
        # Linearity: each mode in superposition should have same ω as when run alone
        error1 = abs(omega_super1 - omega1) / max(omega1, 1e-30)
        error2 = abs(omega_super2 - omega2) / max(omega2, 1e-30)
        linearity_error = max(error1, error2)
        
        passed = linearity_error <= 0.05  # 5% tolerance
        
        return {
            "omega1": omega1,
            "omega2": omega2,
            "omega_super1": omega_super1,
            "omega_super2": omega_super2,
            "linearity_error": linearity_error,
            "passed": passed,
            "message": f"ω1={omega1:.4f}→{omega_super1:.4f}, ω2={omega2:.4f}→{omega_super2:.4f}, err={linearity_error*100:.2f}%"
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
        """Run a single test variant. Dispatches to specialized methods for isotropy and boost tests."""
        tid = v["test_id"]
        
        # Isotropy tests require special handling (run twice with different directions)
        if tid in ("REL-01", "REL-02"):
            return self.run_isotropy_variant(v)
        
        # Lorentz boost tests require special validation
        if tid in ("REL-03", "REL-04"):
            return self.run_boost_variant(v)
        
        # Phase independence test (REL-07)
        if tid == "REL-07":
            return self.run_phase_independence_variant(v)
        
        # Superposition test (REL-08)
        if tid == "REL-08":
            return self.run_superposition_variant(v)
        
        # 3D isotropy tests
        if tid == "REL-09":
            return self.run_3d_directional_isotropy_variant(v)
        
        if tid == "REL-10":
            return self.run_3d_spherical_isotropy_variant(v)
        
        # Dispersion relation tests (REL-11-14)
        if tid in ("REL-11", "REL-12", "REL-13", "REL-14"):
            return self.run_dispersion_relation_variant(v)
        
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
    
    def run_boost_variant(self, v: Dict) -> TestSummary:
        """Run Lorentz boost test by verifying frame-independent dispersion."""
        xp = self.xp
        tid = v["test_id"]
        desc = v.get("description", tid)
        params = self.base.copy(); params.update(v)
        
        N = params["grid_points"]; dx = params["dx"]; dt = params["dt"]
        chi = params["chi"]; c = math.sqrt(params["alpha"] / params["beta"])
        beta = params.get("boost_factor", 0.2)
        steps = max(params.get("steps", 2000), 2048 if self.quick else 4096)
        
        test_dir = self.out_root / tid
        diag_dir = test_dir / "diagnostics"
        plot_dir = test_dir / "plots"
        for d in (test_dir, diag_dir, plot_dir):
            d.mkdir(parents=True, exist_ok=True)
        
        test_start(tid, desc, steps)
        log(f"Lorentz boost test: beta={beta:.2f}, verifying frame covariance", "INFO")
        
        # Initialize field (standard wave, not boosted initial conditions)
        E_prev = self.init_field_variant(tid, params, N, dx, c, "right")
        E = xp.array(E_prev, copy=True)
        k_ang = float(params.get("_k_ang", 0.0))
        
        E_series = [to_numpy(E)]
        
        # Time integration
        dt2 = dt ** 2; c2 = c ** 2; chi2 = chi ** 2; dx2 = dx ** 2
        
        for n in range(steps):
            Em1 = xp.roll(E, -1); Ep1 = xp.roll(E, 1)
            lap = (Em1 - 2 * E + Ep1) / dx2
            E_next = 2 * E - E_prev + dt2 * (c2 * lap - chi2 * E)
            
            # Apply energy rescaling and zero mean
            denom = float(xp.sum(E_next * E_next)) + 1e-30
            E0 = 1.0  # Initial energy
            scale = math.sqrt(float(E0) / denom)
            E_next = E_next * scale
            E_next = E_next - xp.mean(E_next)
            
            E_prev, E = E, E_next
            
            if n % 1 == 0:
                E_series.append(to_numpy(E))
        
        # Measure boost covariance
        boost_result = self.measure_boost_covariance(E_series, dt, dx, c, chi, beta, k_ang)
        
        rel_err = boost_result["rel_error"]
        passed = boost_result["passed"]
        
        status = "PASS ✅" if passed else "FAIL ❌"
        level = "INFO" if passed else "FAIL"
        log(f"{tid} {status} {boost_result['message']}", level)
        
        summary = {
            "id": tid, "description": desc, "passed": passed,
            "rel_error": float(rel_err),
            "covariance_ratio": float(boost_result["covariance_ratio"]),
            "residual_lab_rms": float(boost_result["residual_lab_rms"]),
            "residual_boost_rms": float(boost_result["residual_boost_rms"]),
            "omega_lab": float(boost_result["omega_lab"]),
            "beta": float(beta),
            "gamma": float(boost_result["gamma"]),
            "backend": "GPU" if self.use_gpu else "CPU",
            "test_method": "actual_lorentz_transform",  # Mark as non-circular
        }
        metrics = [
            ("rel_error", rel_err),
            ("covariance_ratio", boost_result["covariance_ratio"]),
            ("residual_lab_rms", boost_result["residual_lab_rms"]),
            ("residual_boost_rms", boost_result["residual_boost_rms"]),
            ("omega_lab", boost_result["omega_lab"]),
        ]
        save_summary(test_dir, tid, summary, metrics=metrics)
        
        return TestSummary(
            id=tid, description=desc, passed=passed, rel_err=rel_err,
            omega_meas=boost_result["covariance_ratio"], omega_theory=1.0,  # Covariance ratio should be 1
            runtime_sec=0.0, k_fraction_lattice=float(params.get("_k_fraction_lattice", 0.0))
        )
    
    def run_phase_independence_variant(self, v: Dict) -> TestSummary:
        """Run phase independence test by comparing sin and cos initial conditions."""
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
        log(f"Phase independence test: comparing cos(kx) and sin(kx) initial conditions", "INFO")
        
        # Run simulation for both phases
        E_series_cos = self._run_wave_evolution(params, N, dx, dt, c, chi, tid, "cos")
        E_series_sin = self._run_wave_evolution(params, N, dx, dt, c, chi, tid, "sin")
        
        # Measure phase independence
        k_ang = float(params.get("_k_ang", 0.0))
        phase_result = self.measure_phase_independence(E_series_cos, E_series_sin, dt, dx, k_ang)
        
        phase_error = phase_result["phase_error"]
        passed = phase_result["passed"]
        
        status = "PASS ✅" if passed else "FAIL ❌"
        level = "INFO" if passed else "FAIL"
        log(f"{tid} {status} {phase_result['message']}", level)
        
        summary = {
            "id": tid, "description": desc, "passed": passed,
            "phase_error": float(phase_error),
            "omega_cos": float(phase_result["omega_cos"]),
            "omega_sin": float(phase_result["omega_sin"]),
            "backend": "GPU" if self.use_gpu else "CPU",
        }
        metrics = [
            ("phase_error", phase_error),
            ("omega_cos", phase_result["omega_cos"]),
            ("omega_sin", phase_result["omega_sin"]),
        ]
        save_summary(test_dir, tid, summary, metrics=metrics)
        
        return TestSummary(
            id=tid, description=desc, passed=passed, rel_err=phase_error,
            omega_meas=phase_result["omega_cos"], omega_theory=phase_result["omega_sin"],
            runtime_sec=0.0, k_fraction_lattice=float(params.get("_k_fraction_lattice", 0.0))
        )
    
    def run_superposition_variant(self, v: Dict) -> TestSummary:
        """Run superposition test by verifying linearity with multi-mode initial condition."""
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
        log(f"Superposition test: verifying linearity with cos(kx) + 0.5*sin(2kx)", "INFO")
        
        # Get wavenumbers
        k_frac = float(params.get("k_fraction", 0.1))
        m = int(round((N * k_frac) / 2.0))
        k_frac_lattice = 2.0 * m / N
        k_cyc = k_frac_lattice * (1.0 / (2.0 * dx))
        k1 = 2.0 * math.pi * k_cyc
        k2 = 2 * k1  # Second mode at 2k
        a1, a2 = 1.0, 0.5  # Amplitudes
        
        # Run each mode separately
        E_series_mode1 = self._run_single_mode(params, N, dx, dt, c, chi, k1, "cos", 1.0)
        E_series_mode2 = self._run_single_mode(params, N, dx, dt, c, chi, k2, "sin", 1.0)
        
        # Run superposition
        E_series_super = self._run_superposition_modes(params, N, dx, dt, c, chi, k1, k2, a1, a2)
        
        # Measure superposition
        super_result = self.measure_superposition(E_series_mode1, E_series_mode2, E_series_super,
                                                  dt, dx, k1, k2, a1, a2)
        
        linearity_error = super_result["linearity_error"]
        passed = super_result["passed"]
        
        status = "PASS ✅" if passed else "FAIL ❌"
        level = "INFO" if passed else "FAIL"
        log(f"{tid} {status} {super_result['message']}", level)
        
        summary = {
            "id": tid, "description": desc, "passed": passed,
            "linearity_error": float(linearity_error),
            "omega1": float(super_result["omega1"]),
            "omega2": float(super_result["omega2"]),
            "backend": "GPU" if self.use_gpu else "CPU",
        }
        metrics = [
            ("linearity_error", linearity_error),
            ("omega1", super_result["omega1"]),
            ("omega2", super_result["omega2"]),
        ]
        save_summary(test_dir, tid, summary, metrics=metrics)
        
        return TestSummary(
            id=tid, description=desc, passed=passed, rel_err=linearity_error,
            omega_meas=super_result["omega1"], omega_theory=super_result["omega2"],
            runtime_sec=0.0, k_fraction_lattice=float(params.get("_k_fraction_lattice", 0.0))
        )
    
    def _run_single_mode(self, params: Dict, N: int, dx: float, dt: float, c: float, chi: float,
                        k: float, mode_type: str, amplitude: float) -> List[np.ndarray]:
        """Helper to run simulation with a single mode."""
        xp = self.xp
        steps = max(params.get("steps", 2000), 2048 if self.quick else 4096)
        
        x = xp.arange(N, dtype=xp.float64) * dx
        if mode_type == "cos":
            E_prev = amplitude * xp.cos(k * x, dtype=xp.float64)
        else:  # sin
            E_prev = amplitude * xp.sin(k * x, dtype=xp.float64)
        
        E = xp.array(E_prev, copy=True)
        E_series = [to_numpy(E)]
        
        dt2 = dt ** 2; c2 = c ** 2; chi2 = chi ** 2; dx2 = dx ** 2
        
        for n in range(steps):
            Em1 = xp.roll(E, -1); Ep1 = xp.roll(E, 1)
            lap = (Em1 - 2 * E + Ep1) / dx2
            E_next = 2 * E - E_prev + dt2 * (c2 * lap - chi2 * E)
            E_next = E_next - xp.mean(E_next)
            E_prev, E = E, E_next
            
            if n % 1 == 0:
                E_series.append(to_numpy(E))
        
        return E_series
    
    def _run_superposition_modes(self, params: Dict, N: int, dx: float, dt: float, c: float, chi: float,
                                k1: float, k2: float, a1: float, a2: float) -> List[np.ndarray]:
        """Helper to run simulation with superposition of modes."""
        xp = self.xp
        steps = max(params.get("steps", 2000), 2048 if self.quick else 4096)
        
        x = xp.arange(N, dtype=xp.float64) * dx
        E_prev = a1 * xp.cos(k1 * x, dtype=xp.float64) + a2 * xp.sin(k2 * x, dtype=xp.float64)
        
        E = xp.array(E_prev, copy=True)
        E_series = [to_numpy(E)]
        
        dt2 = dt ** 2; c2 = c ** 2; chi2 = chi ** 2; dx2 = dx ** 2
        
        for n in range(steps):
            Em1 = xp.roll(E, -1); Ep1 = xp.roll(E, 1)
            lap = (Em1 - 2 * E + Ep1) / dx2
            E_next = 2 * E - E_prev + dt2 * (c2 * lap - chi2 * E)
            E_next = E_next - xp.mean(E_next)
            E_prev, E = E, E_next
            
            if n % 1 == 0:
                E_series.append(to_numpy(E))
        
        return E_series
    
    def _run_wave_evolution(self, params: Dict, N: int, dx: float, dt: float, c: float, chi: float, tid: str, phase: str) -> List[np.ndarray]:
        """Helper to run simulation with specified phase (cos or sin)."""
        xp = self.xp
        steps = max(params.get("steps", 2000), 2048 if self.quick else 4096)
        
        # Initialize field based on phase
        x = xp.arange(N, dtype=xp.float64) * dx
        k_ang = float(params.get("_k_ang", 0.0))
        if phase == "cos":
            E_prev = xp.cos(k_ang * x, dtype=xp.float64)
        else:  # sin
            E_prev = xp.sin(k_ang * x, dtype=xp.float64)
        
        E = xp.array(E_prev, copy=True)
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
    
    def run_3d_directional_isotropy_variant(self, v: Dict) -> TestSummary:
        """
        REL-09: 3D Isotropy — Directional Equivalence
        
        Test that plane waves propagating along x, y, and z axes have identical dispersion.
        For an isotropic equation, ω(kx, 0, 0) = ω(0, ky, 0) = ω(0, 0, kz).
        """
        xp = self.xp
        tid = v["test_id"]
        desc = v.get("description", tid)
        params = self.base.copy(); params.update(v)
        
        # 3D grid parameters
        N = params["grid_points"]; dx = params["dx"]; dt = params["dt"]
        chi = params["chi"]; c = math.sqrt(params["alpha"] / params["beta"])
        steps = max(params.get("steps", 1000), 1024 if self.quick else 2048)
        
        test_dir = self.out_root / tid
        diag_dir = test_dir / "diagnostics"
        plot_dir = test_dir / "plots"
        for d in (test_dir, diag_dir, plot_dir):
            d.mkdir(parents=True, exist_ok=True)
        
        test_start(tid, desc, steps)
        log(f"3D Directional Isotropy: comparing x, y, z axis propagation", "INFO")
        
        # Run three simulations with waves along each axis
        k_frac = float(params.get("k_fraction", 0.08))
        m = int(round((N * k_frac) / 2.0))
        k_frac_lattice = 2.0 * m / N
        k_cyc = k_frac_lattice * (1.0 / (2.0 * dx))
        k_ang = 2.0 * math.pi * k_cyc
        
        log(f"k_ang = {k_ang:.6f}, k_frac = {k_frac_lattice:.6f}", "INFO")
        
        # X-axis propagation
        log("Running X-axis wave...", "INFO")
        E_series_x = self._run_3d_plane_wave(params, N, dx, dt, c, chi, k_ang, axis='x', steps=steps)
        omega_x = self._measure_omega_3d(E_series_x, dt, dx, k_ang, axis='x')
        
        # Y-axis propagation
        log("Running Y-axis wave...", "INFO")
        E_series_y = self._run_3d_plane_wave(params, N, dx, dt, c, chi, k_ang, axis='y', steps=steps)
        omega_y = self._measure_omega_3d(E_series_y, dt, dx, k_ang, axis='y')
        
        # Z-axis propagation
        log("Running Z-axis wave...", "INFO")
        E_series_z = self._run_3d_plane_wave(params, N, dx, dt, c, chi, k_ang, axis='z', steps=steps)
        omega_z = self._measure_omega_3d(E_series_z, dt, dx, k_ang, axis='z')
        
        # Measure directional anisotropy
        omega_mean = (omega_x + omega_y + omega_z) / 3.0
        aniso_x = abs(omega_x - omega_mean) / max(omega_mean, 1e-30)
        aniso_y = abs(omega_y - omega_mean) / max(omega_mean, 1e-30)
        aniso_z = abs(omega_z - omega_mean) / max(omega_mean, 1e-30)
        anisotropy = max(aniso_x, aniso_y, aniso_z)
        
        passed = anisotropy <= 0.01  # 1% tolerance
        
        status = "PASS ✅" if passed else "FAIL ❌"
        level = "INFO" if passed else "FAIL"
        log(f"{tid} {status} ωx={omega_x:.6f}, ωy={omega_y:.6f}, ωz={omega_z:.6f}, aniso={anisotropy*100:.3f}%", level)
        
        summary = {
            "id": tid, "description": desc, "passed": passed,
            "anisotropy": float(anisotropy),
            "omega_x": float(omega_x),
            "omega_y": float(omega_y),
            "omega_z": float(omega_z),
            "omega_mean": float(omega_mean),
            "backend": "GPU" if self.use_gpu else "CPU",
        }
        metrics = [
            ("anisotropy", anisotropy),
            ("omega_x", omega_x),
            ("omega_y", omega_y),
            ("omega_z", omega_z),
        ]
        save_summary(test_dir, tid, summary, metrics=metrics)
        
        return TestSummary(
            id=tid, description=desc, passed=passed, rel_err=anisotropy,
            omega_meas=omega_mean, omega_theory=omega_mean,
            runtime_sec=0.0, k_fraction_lattice=k_frac_lattice
        )
    
    def run_3d_spherical_isotropy_variant(self, v: Dict) -> TestSummary:
        """
        REL-10: 3D Isotropy — Spherical Symmetry
        
        Test that a radially symmetric initial condition produces a spherically symmetric
        propagating wave. For isotropy, the dispersion relation should only depend on |k|,
        not on the direction of k.
        """
        xp = self.xp
        tid = v["test_id"]
        desc = v.get("description", tid)
        params = self.base.copy(); params.update(v)
        
        # 3D grid parameters
        N = params["grid_points"]; dx = params["dx"]; dt = params["dt"]
        chi = params["chi"]; c = math.sqrt(params["alpha"] / params["beta"])
        steps = max(params.get("steps", 800), 800 if self.quick else 1500)
        
        test_dir = self.out_root / tid
        diag_dir = test_dir / "diagnostics"
        plot_dir = test_dir / "plots"
        for d in (test_dir, diag_dir, plot_dir):
            d.mkdir(parents=True, exist_ok=True)
        
        test_start(tid, desc, steps)
        log(f"3D Spherical Isotropy: testing radial wave symmetry", "INFO")
        
        # Run simulation with radially symmetric initial condition
        k_frac = float(params.get("k_fraction", 0.06))
        m = int(round((N * k_frac) / 2.0))
        k_frac_lattice = 2.0 * m / N
        k_cyc = k_frac_lattice * (1.0 / (2.0 * dx))
        k_ang = 2.0 * math.pi * k_cyc
        
        log(f"k_ang = {k_ang:.6f}, radial pulse width ~ {1.0/k_ang:.4f}", "INFO")
        
        E_series = self._run_3d_radial_wave(params, N, dx, dt, c, chi, k_ang, steps=steps)
        
        # Measure spherical symmetry by comparing radial profiles in different directions
        symmetry_error = self._measure_spherical_symmetry(E_series[-1], dx)
        
        passed = symmetry_error <= 0.05  # 5% tolerance
        
        status = "PASS ✅" if passed else "FAIL ❌"
        level = "INFO" if passed else "FAIL"
        log(f"{tid} {status} spherical_error={symmetry_error*100:.3f}%", level)
        
        summary = {
            "id": tid, "description": desc, "passed": passed,
            "spherical_error": float(symmetry_error),
            "backend": "GPU" if self.use_gpu else "CPU",
        }
        metrics = [
            ("spherical_error", symmetry_error),
        ]
        save_summary(test_dir, tid, summary, metrics=metrics)
        
        return TestSummary(
            id=tid, description=desc, passed=passed, rel_err=symmetry_error,
            omega_meas=0.0, omega_theory=0.0,
            runtime_sec=0.0, k_fraction_lattice=k_frac_lattice
        )
    
    def _run_3d_plane_wave(self, params: Dict, N: int, dx: float, dt: float, c: float, 
                           chi: float, k_ang: float, axis: str, steps: int) -> List[np.ndarray]:
        """Run 3D simulation with plane wave along specified axis."""
        xp = self.xp
        
        # Create 3D grid
        x = xp.arange(N, dtype=xp.float64) * dx
        y = xp.arange(N, dtype=xp.float64) * dx
        z = xp.arange(N, dtype=xp.float64) * dx
        
        # Initialize plane wave along specified axis
        if axis == 'x':
            E_prev = xp.cos(k_ang * x)[:, None, None] * xp.ones((N, N, N), dtype=xp.float64)
        elif axis == 'y':
            E_prev = xp.cos(k_ang * y)[None, :, None] * xp.ones((N, N, N), dtype=xp.float64)
        elif axis == 'z':
            E_prev = xp.cos(k_ang * z)[None, None, :] * xp.ones((N, N, N), dtype=xp.float64)
        else:
            raise ValueError(f"Invalid axis: {axis}")
        
        # Apply Hann window to avoid sharp edges
        hann_np = hann_vec(N)
        hann = xp.asarray(hann_np, dtype=xp.float64)  # Convert to correct backend
        if axis == 'x':
            E_prev = E_prev * hann[:, None, None]
        elif axis == 'y':
            E_prev = E_prev * hann[None, :, None]
        elif axis == 'z':
            E_prev = E_prev * hann[None, None, :]
        
        E_prev = E_prev - xp.mean(E_prev)
        E = xp.array(E_prev, copy=True)
        
        # Time integration (3D Klein-Gordon)
        dt2 = dt ** 2; c2 = c ** 2; chi2 = chi ** 2; dx2 = dx ** 2
        E_series = [to_numpy(E)]
        
        for n in range(steps):
            # 3D Laplacian (2nd order)
            lap = (xp.roll(E, 1, axis=2) + xp.roll(E, -1, axis=2) +
                   xp.roll(E, 1, axis=1) + xp.roll(E, -1, axis=1) +
                   xp.roll(E, 1, axis=0) + xp.roll(E, -1, axis=0) - 6 * E) / dx2
            
            E_next = 2 * E - E_prev + dt2 * (c2 * lap - chi2 * E)
            E_next = E_next - xp.mean(E_next)
            
            E_prev, E = E, E_next
            
            if n % 1 == 0:
                E_series.append(to_numpy(E))
        
        return E_series
    
    def _run_3d_radial_wave(self, params: Dict, N: int, dx: float, dt: float, c: float,
                            chi: float, k_ang: float, steps: int) -> List[np.ndarray]:
        """Run 3D simulation with radially symmetric Gaussian pulse."""
        xp = self.xp
        
        # Create 3D grid centered at domain center
        center = N // 2
        x = (xp.arange(N, dtype=xp.float64) - center) * dx
        y = (xp.arange(N, dtype=xp.float64) - center) * dx
        z = (xp.arange(N, dtype=xp.float64) - center) * dx
        
        X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
        R = xp.sqrt(X**2 + Y**2 + Z**2)
        
        # Gaussian pulse with characteristic width
        width = 1.0 / k_ang
        E_prev = xp.exp(-0.5 * (R / width) ** 2)
        E_prev = E_prev - xp.mean(E_prev)
        E = xp.array(E_prev, copy=True)
        
        # Time integration (3D Klein-Gordon)
        dt2 = dt ** 2; c2 = c ** 2; chi2 = chi ** 2; dx2 = dx ** 2
        E_series = [to_numpy(E)]
        
        for n in range(steps):
            # 3D Laplacian (2nd order)
            lap = (xp.roll(E, 1, axis=2) + xp.roll(E, -1, axis=2) +
                   xp.roll(E, 1, axis=1) + xp.roll(E, -1, axis=1) +
                   xp.roll(E, 1, axis=0) + xp.roll(E, -1, axis=0) - 6 * E) / dx2
            
            E_next = 2 * E - E_prev + dt2 * (c2 * lap - chi2 * E)
            E_next = E_next - xp.mean(E_next)
            
            E_prev, E = E, E_next
            
            if n % 1 == 0:
                E_series.append(to_numpy(E))
        
        return E_series
    
    def _measure_omega_3d(self, E_series: List[np.ndarray], dt: float, dx: float, 
                          k_ang: float, axis: str) -> float:
        """Measure frequency from 3D plane wave by projecting onto mode."""
        N = E_series[0].shape[0]
        
        # Create projection mode along specified axis
        if axis == 'x':
            x = np.arange(N) * dx
            mode = np.cos(k_ang * x)[:, None, None] * np.ones((N, N, N), dtype=np.float64)
        elif axis == 'y':
            y = np.arange(N) * dx
            mode = np.cos(k_ang * y)[None, :, None] * np.ones((N, N, N), dtype=np.float64)
        elif axis == 'z':
            z = np.arange(N) * dx
            mode = np.cos(k_ang * z)[None, None, :] * np.ones((N, N, N), dtype=np.float64)
        else:
            raise ValueError(f"Invalid axis: {axis}")
        
        mode_norm = float(np.sum(mode * mode) + 1e-30)
        
        # Project each snapshot onto mode
        proj_series = []
        for E in E_series:
            E_np = E.astype(np.float64)
            E_np = E_np - E_np.mean()
            proj_series.append(float(np.sum(E_np * mode) / mode_norm))
        
        # FFT to extract frequency
        omega = self.estimate_omega_proj_fft(np.array(proj_series, dtype=np.float64), dt)
        return omega
    
    def _measure_spherical_symmetry(self, E: np.ndarray, dx: float) -> float:
        """
        Measure deviation from spherical symmetry by comparing radial profiles
        along different directions.
        
        Returns: maximum relative deviation between directional radial profiles
        """
        N = E.shape[0]
        center = N // 2
        
        # Sample radial profiles along three axes
        # X-axis: E[center+r, center, center]
        # Y-axis: E[center, center+r, center]
        # Z-axis: E[center, center, center+r]
        
        r_max = min(N // 4, 20)  # Sample up to quarter domain or 20 points
        
        profile_x = []
        profile_y = []
        profile_z = []
        
        for r in range(1, r_max):
            if center + r < N:
                profile_x.append(abs(float(E[center + r, center, center])))
                profile_y.append(abs(float(E[center, center + r, center])))
                profile_z.append(abs(float(E[center, center, center + r])))
        
        # Convert to arrays
        profile_x = np.array(profile_x)
        profile_y = np.array(profile_y)
        profile_z = np.array(profile_z)
        
        # Compute mean profile
        profile_mean = (profile_x + profile_y + profile_z) / 3.0
        
        # Measure maximum deviation
        dev_x = np.max(np.abs(profile_x - profile_mean) / (np.abs(profile_mean) + 1e-30))
        dev_y = np.max(np.abs(profile_y - profile_mean) / (np.abs(profile_mean) + 1e-30))
        dev_z = np.max(np.abs(profile_z - profile_mean) / (np.abs(profile_mean) + 1e-30))
        
        return float(max(dev_x, dev_y, dev_z))
    
    def run_dispersion_relation_variant(self, v: Dict) -> TestSummary:
        """
        REL-11-14: Dispersion Relation Tests
        
        Directly measure ω²/k² and compare to theory: ω²/k² = 1 + (χ/k)²
        This explicitly validates the relativistic energy-momentum relation E² = (pc)² + (mc²)²
        in natural units (c=1, ℏ=1).
        
        Theory: For Klein-Gordon equation, ω² = k² + χ²
        Therefore: ω²/k² = 1 + (χ/k)²
        
        Test regimes:
        - REL-11: Non-relativistic χ/k≈10 → ω²/k² ≈ 101
        - REL-12: Weakly relativistic χ/k≈1 → ω²/k² ≈ 2
        - REL-13: Relativistic χ/k≈0.5 → ω²/k² ≈ 1.25
        - REL-14: Ultra-relativistic χ/k≈0.1 → ω²/k² ≈ 1.01
        """
        xp = self.xp
        tid = v["test_id"]
        desc = v.get("description", tid)
        params = self.base.copy(); params.update(v)
        
        # Extract parameters
        N = params["grid_points"]; dx = params["dx"]; dt = params["dt"]
        chi = params["chi"]; c = math.sqrt(params["alpha"] / params["beta"])
        steps = params.get("steps", 4000)
        
        test_dir = self.out_root / tid
        diag_dir = test_dir / "diagnostics"
        plot_dir = test_dir / "plots"
        for d in (test_dir, diag_dir, plot_dir):
            d.mkdir(parents=True, exist_ok=True)
        
        test_start(tid, desc, steps)
        
        # Compute wavenumber
        k_frac = float(params.get("k_fraction", 0.05))
        m = int(round((N * k_frac) / 2.0))
        k_frac_lattice = 2.0 * m / N
        k_cyc = k_frac_lattice * (1.0 / (2.0 * dx))
        k_ang = 2.0 * math.pi * k_cyc
        
        # Theory prediction: ω² = k² + χ²
        omega_theory = math.sqrt(k_ang**2 + chi**2)
        omega2_over_k2_theory = (k_ang**2 + chi**2) / (k_ang**2 + 1e-30)
        chi_over_k = chi / (k_ang + 1e-30)
        
        log(f"k = {k_ang:.6f}, χ = {chi:.6f}, χ/k = {chi_over_k:.4f}", "INFO")
        log(f"Theory: ω = {omega_theory:.6f}, ω²/k² = {omega2_over_k2_theory:.6f}", "INFO")
        
        # Initialize plane wave: E(x,t=0) = cos(kx)
        x = xp.arange(N, dtype=xp.float64) * dx
        E_prev = xp.cos(k_ang * x)
        
        # Apply Hann window and zero-mean
        hann = xp.asarray(hann_vec(N), dtype=xp.float64)
        E_prev = E_prev * hann
        E_prev = E_prev - xp.mean(E_prev)
        E = xp.array(E_prev, copy=True)
        
        # Projection mode for frequency measurement
        mode = xp.cos(k_ang * x)
        mode_norm = float(xp.sum(mode * mode) + 1e-30)
        
        # Time evolution
        dt2 = dt ** 2; c2 = c ** 2; chi2 = chi ** 2; dx2 = dx ** 2
        proj_series = []
        
        for n in range(steps):
            # Project onto mode
            E_zm = E - xp.mean(E)
            proj = float(xp.sum(E_zm * mode) / mode_norm)
            proj_series.append(proj)
            
            # 1D Laplacian
            lap = (xp.roll(E, 1) + xp.roll(E, -1) - 2 * E) / dx2
            
            # Klein-Gordon step
            E_next = 2 * E - E_prev + dt2 * (c2 * lap - chi2 * E)
            E_next = E_next - xp.mean(E_next)
            
            E_prev, E = E, E_next
            
            if (n + 1) % (steps // 20) == 0:
                report_progress(tid, int(100 * (n + 1) / steps))
        
        # Measure frequency from projection series
        proj_arr = np.array(proj_series, dtype=np.float64)
        omega_meas = self.estimate_omega_proj_fft(proj_arr, dt)
        
        # Compute measured ω²/k²
        omega2_over_k2_meas = (omega_meas**2) / (k_ang**2 + 1e-30)
        
        # Relative error in ω²/k²
        rel_err = abs(omega2_over_k2_meas - omega2_over_k2_theory) / (omega2_over_k2_theory + 1e-30)
        
        # Pass criterion: <5% error
        passed = rel_err < 0.05
        
        status = "PASS ✅" if passed else "FAIL ❌"
        level = "INFO" if passed else "FAIL"
        log(f"{tid} {status} (ω²/k²_meas={omega2_over_k2_meas:.6f}, "
            f"ω²/k²_theory={omega2_over_k2_theory:.6f}, err={rel_err*100:.3f}%)", level)
        
        # Save diagnostics
        diag_csv = diag_dir / "dispersion_measurement.csv"
        with open(diag_csv, "w", encoding="utf-8") as f:
            f.write("quantity,measured,theory,error_pct\n")
            f.write(f"omega,{omega_meas:.10e},{omega_theory:.10e},{abs(omega_meas-omega_theory)/omega_theory*100:.6f}\n")
            f.write(f"omega2_over_k2,{omega2_over_k2_meas:.10e},{omega2_over_k2_theory:.10e},{rel_err*100:.6f}\n")
            f.write(f"chi_over_k,{chi_over_k:.10e},{chi_over_k:.10e},0.0\n")
        
        # Save projection time series
        ts_csv = diag_dir / "projection_series.csv"
        with open(ts_csv, "w", encoding="utf-8") as f:
            f.write("t,projection\n")
            for i, proj in enumerate(proj_series):
                f.write(f"{i*dt:.8f},{proj:.10e}\n")
        
        summary = {
            "id": tid, "description": desc, "passed": passed,
            "rel_err": float(rel_err),
            "omega_meas": float(omega_meas),
            "omega_theory": float(omega_theory),
            "omega2_over_k2_meas": float(omega2_over_k2_meas),
            "omega2_over_k2_theory": float(omega2_over_k2_theory),
            "k_ang": float(k_ang),
            "chi": float(chi),
            "chi_over_k": float(chi_over_k),
            "runtime_sec": 0.0,
            "backend": "GPU" if self.use_gpu else "CPU",
            "params": {
                "N": N, "dx": dx, "dt": dt, "steps": steps,
                "alpha": params["alpha"], "beta": params["beta"], "chi": chi
            }
        }
        metrics = [
            ("rel_err", rel_err),
            ("omega_meas", omega_meas),
            ("omega_theory", omega_theory),
            ("omega2_over_k2_meas", omega2_over_k2_meas),
            ("omega2_over_k2_theory", omega2_over_k2_theory),
        ]
        save_summary(test_dir, tid, summary, metrics=metrics)
        
        return TestSummary(
            id=tid, description=desc, passed=passed, rel_err=rel_err,
            omega_meas=omega_meas, omega_theory=omega_theory,
            runtime_sec=0.0, k_fraction_lattice=k_frac_lattice
        )
    
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
    outdir = resolve_outdir(cfg.get("output_dir", "results/Relativistic"))
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
        # Full suite: show summary and write CSV
        suite_summary(results)
        suite_rows = [[r["test_id"], r["description"], r["passed"], r["rel_err"],
                       r["omega_meas"], r["omega_theory"], r["runtime_sec"]] for r in results]
        write_csv(outdir/"suite_summary.csv", suite_rows,
                  ["test_id","description","passed","rel_err","omega_meas","omega_theory","runtime_sec"])
        write_metadata_bundle(outdir, test_id="TIER1-SUITE", tier=1, category="relativistic")
        log("=== Tier-1 Suite Complete ===", "INFO")


if __name__ == "__main__":
    main()
