#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
lfm_test_harness.py — Base harness class for LFM tier test runners
-------------------------------------------------------------------
Purpose:
    Eliminate duplicate code across tier runners by providing shared:
    - Config loading with standard search paths
    - Logger/output directory setup
    - Backend selection and initialization
    - Common frequency measurement methods (FFT-based)
    - Hanning window utilities
    - Output directory resolution

All tier-specific harnesses (Tier1Harness, Tier2Harness, etc.) should
inherit from BaseTierHarness to reduce code duplication.

Benefits:
    - Single source of truth for config loading (~50 lines per file)
    - Consistent FFT-based frequency estimation (~30 lines per file)
    - Standardized logger setup (~20 lines per file)
    - Reduces copy-paste errors and improves maintainability
"""

import json
import os
import math
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np

from core.lfm_backend import pick_backend
from ui.lfm_console import log, set_logger, log_run_config
from utils.lfm_logger import LFMLogger
from utils.numeric_integrity import NumericIntegrityMixin
from utils.resource_tracking import create_resource_tracker
# Support both package-relative and absolute imports for robustness
try:
    from ..utils.cache_manager_runtime import TestCacheManager  # type: ignore
except Exception:
    from utils.cache_manager_runtime import TestCacheManager  # type: ignore


class BaseTierHarness(NumericIntegrityMixin):
    """
    Base class for all tier test harnesses.
    
    Provides common functionality:
    - Config loading from standard locations
    - Backend (NumPy/CuPy) selection
    - Logger initialization
    - FFT-based frequency estimation
    - Output directory management
    
    Subclasses should:
    1. Call super().__init__(cfg, out_root, config_name) in __init__
    2. Override test-specific methods (run_variant, init_field_variant, etc.)
    3. Use self.xp for backend-agnostic array operations
    4. Use self.estimate_omega_fft() for frequency measurement
    """
    
    def __init__(
        self,
        cfg: Dict,
        out_root: Path,
        config_name: str = "config.json",
        backend: str = "baseline",
        tier_number: Optional[int] = None,
    ):
        """
        Initialize base harness with config and output directory.
        
        Args:
            cfg: Configuration dictionary loaded from JSON
            out_root: Root output directory for test results
            config_name: Name of config file (for error messages)
            backend: Physics backend ('baseline' or 'fused'); tier runners should
                     pass this from CLI args to enable GPU acceleration
        """
        self.cfg = cfg
        self.config_name = config_name
        self.run_settings = cfg.get("run_settings", {})
        self.base = cfg.get("parameters", {})
        self.tol = cfg.get("tolerances", {})
        self.quick = bool(self.run_settings.get("quick_mode", False))
        
        # Backend selection - ALWAYS use GPU (NVIDIA GeForce RTX 4060 Laptop with CuPy)
        # Check both use_gpu and gpu_enabled for compatibility with different config schemas
        use_gpu = bool(
            self.run_settings.get("use_gpu", True) or 
            cfg.get("hardware", {}).get("gpu_enabled", True)
        )
        self.xp, self.use_gpu = pick_backend(use_gpu)
        
        # Physics backend (baseline vs fused kernel)
        self.backend = backend
        
        # Output directory
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        
        # Logger setup
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
        
        # Progress reporting
        self.show_progress = bool(self.run_settings.get("show_progress", True))
        self.progress_percent_stride = int(
            self.run_settings.get("progress_percent_stride", 5)
        )
        
        # Resource tracking
        self.enable_resource_tracking = bool(
            self.run_settings.get("enable_resource_tracking", True)
        )
        self._current_tracker = None  # Active tracker for current test
        
        # Log backend info
        backend_name = "GPU (CuPy)" if self.use_gpu else "CPU (NumPy)"
        log(f"[accel] Using {backend_name} backend.", "INFO")
        log(f"[physics] Using '{self.backend}' physics backend.", "INFO")
        
        # ---------------- Tier validation metadata auto-load ----------------
        self.tier_number = tier_number
        self.tier_meta: Dict = {}
        # Maintain backward compatibility with existing code expecting _tier_meta
        self._tier_meta: Dict = {}
        if tier_number is not None:
            try:
                from harness.validation import load_tier_metadata  # lazy import to avoid circulars
                self.tier_meta = load_tier_metadata(int(tier_number))
                self._tier_meta = self.tier_meta
                log(f"[integrity] Tier{tier_number} metadata loaded for validation checks", "INFO")
            except FileNotFoundError as e:
                log(f"[integrity] WARNING: Tier{tier_number} metadata file not found: {e}", "WARN")
            except Exception as e:
                log(f"[integrity] WARNING: Could not load Tier{tier_number} metadata: {type(e).__name__}: {e}", "WARN")
        # ---------------- Diagnostics (env + override file) ----------------
        # Global env var: LFM_DIAGNOSTICS = off|basic|full
        self.global_diagnostics_mode = os.environ.get("LFM_DIAGNOSTICS", "off").strip().lower()
        # Per-test overrides: workspace/config/diagnostics_overrides.json
        self._diagnostics_overrides = self._load_diagnostics_overrides()

        # ---------------- Caching setup ----------------
        # Defaults: caching enabled unless explicitly disabled
        self.use_cache = bool(self.run_settings.get("use_cache", True))
        self.force_rerun = bool(self.run_settings.get("force_rerun", False))

        # Determine workspace root (…/workspace) and cache root (…/build/cache/test_results)
        try:
            # This file is …/workspace/src/harness/lfm_test_harness.py → workspace = parents[2]
            workspace_root = Path(__file__).resolve().parents[2]
        except Exception:
            workspace_root = Path.cwd()

        build_root = workspace_root.parent / "build"
        cache_root = build_root / "cache" / "test_results"

        self.cache_manager: Optional[TestCacheManager]
        if self.use_cache:
            try:
                self.cache_manager = TestCacheManager(cache_root=cache_root, workspace_root=workspace_root)
                log(f"[cache] Test caching enabled (root: {cache_root})", "INFO")
            except Exception as e:
                self.cache_manager = None
                log(f"[cache] Failed to initialize cache manager: {e}", "WARN")
        else:
            self.cache_manager = None
            log("[cache] Test caching disabled by configuration.", "INFO")
    
    @staticmethod
    def load_config(
        config_path: Optional[str] = None,
        default_config_name: str = "config.json"
    ) -> Dict:
        """
        Load configuration from JSON file.
        
        Search strategy:
        1. If config_path provided and is absolute, use it directly
        2. If config_path provided and is relative, search for it:
           - Relative to script directory
           - Relative to script parent directory
           - Relative to current working directory
        3. If no config_path, search for default_config_name:
           - {script_dir}/config/{default_config_name}
           - {script_dir}/../config/{default_config_name}
        
        Args:
            config_path: Explicit path to config file (optional, can be relative)
            default_config_name: Default config filename to search for
            
        Returns:
            Dictionary with configuration
            
        Raises:
            FileNotFoundError: If config file not found
            
        Example:
            >>> cfg = BaseTierHarness.load_config(
            ...     default_config_name="config_tier1_relativistic.json"
            ... )
        """
        import inspect
        caller_frame = inspect.stack()[1]
        caller_file = Path(caller_frame.filename).resolve()
        script_dir = caller_file.parent
        
        if config_path:
            cand = Path(config_path)
            
            # If absolute path, try it directly
            if cand.is_absolute():
                if cand.is_file():
                    with open(cand, "r", encoding="utf-8") as f:
                        return json.load(f)
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            # If relative path, try multiple locations
            search_roots = [script_dir, script_dir.parent, Path.cwd()]
            for root in search_roots:
                full_path = root / config_path
                if full_path.is_file():
                    with open(full_path, "r", encoding="utf-8") as f:
                        return json.load(f)
            
            # Not found in any location
            raise FileNotFoundError(
                f"Config file not found: {config_path} "
                f"(searched relative to {script_dir}, {script_dir.parent}, and {Path.cwd()})"
            )
        
        # No config_path provided, use default_config_name
        for root in (script_dir, script_dir.parent):
            cand = root / "config" / default_config_name
            if cand.is_file():
                with open(cand, "r", encoding="utf-8") as f:
                    return json.load(f)
        
        raise FileNotFoundError(
            f"Config not found: {default_config_name} "
            f"(searched in {script_dir}/config and {script_dir.parent}/config)"
        )
    
    @staticmethod
    def resolve_outdir(output_dir_hint: str) -> Path:
        """
        Resolve output directory to workspace/results/{category}.
        
        This method ensures all test results are written to the standard location:
        workspace/results/{category} regardless of where the script is executed from.
        
        Args:
            output_dir_hint: Directory path hint (e.g., "results/Energy", "Energy", "../results/Energy")
                            The category name will be extracted automatically.
            
        Returns:
            Resolved Path object (workspace/results/{category}), directory created if needed
            
        Example:
            >>> outdir = BaseTierHarness.resolve_outdir("results/Energy")
            >>> # Returns: workspace/results/Energy (regardless of cwd)
        """
        import inspect
        caller_frame = inspect.stack()[1]
        caller_file = Path(caller_frame.filename).resolve()
        script_dir = caller_file.parent

        # Find workspace root by walking up the directory tree
        workspace_root = None
        for p in [script_dir] + list(script_dir.parents):
            if p.name.lower() == "workspace":
                workspace_root = p
                break

        # If we can't find workspace root, use script_dir.parent as fallback
        if workspace_root is None:
            # Assume script is in workspace/src, so parent is workspace
            workspace_root = script_dir.parent

        # Extract category name from hint
        # Handle formats like: "results/Energy", "Energy", "../results/Energy", "results\\Energy"
        hint_path = Path(output_dir_hint)
        category = hint_path.name  # Get the last component (e.g., "Energy" from "results/Energy")
        
        # Always resolve to workspace/results/{category}
        outdir = workspace_root / "results" / category
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    # ---------------- Diagnostics helpers ----------------
    def _workspace_root(self) -> Path:
        try:
            return Path(__file__).resolve().parents[2]
        except Exception:
            return Path.cwd()

    def _load_diagnostics_overrides(self) -> dict:
        cfg_path = self._workspace_root() / "config" / "diagnostics_overrides.json"
        try:
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
        except Exception as e:
            log(f"[diag] Failed to read diagnostics_overrides.json: {e}", "WARN")
        return {}

    def get_diagnostics_mode(self, test_id: str) -> str:
        # Priority: explicit test override > env var > file default
        try:
            # File override by test ID
            tests = self._diagnostics_overrides.get("tests", {}) if isinstance(self._diagnostics_overrides, dict) else {}
            if test_id in tests:
                return str(tests[test_id]).strip().lower()
        except Exception:
            pass
        # Env var
        if self.global_diagnostics_mode in {"off", "basic", "full"}:
            return self.global_diagnostics_mode
        # File default
        try:
            return str(self._diagnostics_overrides.get("default_mode", "off")).strip().lower()
        except Exception:
            return "off"
    
    @staticmethod
    def hann_window(length: int) -> np.ndarray:
        """
        Create Hanning (Hann) window for FFT windowing.
        
        Args:
            length: Length of window
            
        Returns:
            NumPy array with Hann window values
            
        Example:
            >>> w = BaseTierHarness.hann_window(1024)
            >>> windowed_data = data * w
        """
        return np.hanning(length) if length > 0 else np.array([], dtype=np.float64)
    
    def estimate_omega_fft(
        self,
        series: np.ndarray,
        dt: float,
        method: str = "parabolic"
    ) -> float:
        """
        Estimate angular frequency ω from time series using FFT.
        
        Uses Hanning window and parabolic interpolation for sub-bin accuracy.
        
        Algorithm:
        1. Remove DC component (mean)
        2. Apply Hanning window to reduce spectral leakage
        3. Compute FFT and find peak in magnitude spectrum
        4. Use parabolic interpolation around peak for sub-bin accuracy
        5. Convert frequency to angular frequency: ω = 2πf
        
        Args:
            series: Time series data (1D NumPy array)
            dt: Time step between samples
            method: "parabolic" for sub-bin interpolation, "simple" for peak-only
            
        Returns:
            Estimated angular frequency ω (rad/time)
            
        Example:
            >>> probe_series = np.array([...])  # Time series of field at probe
            >>> omega = harness.estimate_omega_fft(probe_series, dt=0.01)
        """
        data = np.asarray(series, dtype=np.float64)
        data = data - data.mean()  # Remove DC component
        
        if len(data) < 16:
            return 0.0
        
        # Apply Hanning window
        w = self.hann_window(len(data))
        windowed = data * w
        
        # Compute FFT
        spec = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), dt)
        
        # Find peak (skip DC component at index 0)
        if len(spec) < 2:
            return 0.0
        peak_idx = int(np.argmax(spec[1:])) + 1
        
        if method == "parabolic" and 1 <= peak_idx < len(spec) - 1:
            # Parabolic interpolation for sub-bin accuracy
            # Fit parabola to log-magnitude around peak
            y1 = np.log(spec[peak_idx - 1] + 1e-30)
            y2 = np.log(spec[peak_idx] + 1e-30)
            y3 = np.log(spec[peak_idx + 1] + 1e-30)
            
            denom = y1 - 2*y2 + y3
            if abs(denom) > 1e-12:
                delta = 0.5 * (y1 - y3) / denom
                delta = np.clip(delta, -0.5, 0.5)  # Limit to half-bin
            else:
                delta = 0.0
            
            # Interpolate frequency
            refined_idx = peak_idx + delta
            f_peak = np.interp(refined_idx, np.arange(len(freqs)), freqs)
        else:
            # Simple peak finding (no interpolation)
            f_peak = freqs[peak_idx]
        
        # Convert to angular frequency
        return 2.0 * math.pi * abs(f_peak)
    
    def estimate_omega_phase_slope(
        self,
        z_complex: np.ndarray,
        t_axis: np.ndarray
    ) -> float:
        """
        Estimate angular frequency from phase unwrapping and linear fit.
        
        Useful when signal is nearly monochromatic. Measures instantaneous
        frequency from rate of phase change: ω = dφ/dt.
        
        Args:
            z_complex: Complex time series (amplitude × exp(iωt))
            t_axis: Time values corresponding to each sample
            
        Returns:
            Estimated angular frequency ω (rad/time)
            
        Example:
            >>> z = probe_cos + 1j * probe_sin  # Complex projection
            >>> omega = harness.estimate_omega_phase_slope(z, t_values)
        """
        # Unwrap phase to remove 2π discontinuities
        phi = np.unwrap(np.angle(z_complex)).astype(np.float64)
        
        # Weighted least-squares fit with Hanning window
        w = self.hann_window(len(phi))
        A = np.vstack([t_axis, np.ones_like(t_axis)]).T
        Aw = A * w[:, None]
        yw = phi * w
        
        slope, _ = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        return float(abs(slope))
    
    def compute_field_energy(
        self,
        E,
        E_prev,
        dt: float,
        dx: float,
        c: float,
        chi,
        dims: str = '3d'
    ) -> float:
        """
        Compute total Klein-Gordon field energy (universal across all tiers).
        
        Energy functional:
            E_total = ∫ [½(∂E/∂t)² + ½c²|∇E|² + ½χ²E²] dV
        
        Components:
            - Kinetic:   ½∫(∂E/∂t)² dV
            - Gradient:  ½∫c²|∇E|² dV
            - Potential: ½∫χ²E² dV
        
        This is the SINGLE SOURCE OF TRUTH for energy calculation.
        All tiers should use this method instead of implementing their own.
        
        Args:
            E: Current field state (1D, 2D, or 3D array)
            E_prev: Previous field state (for time derivative)
            dt: Time step
            dx: Grid spacing (assumed uniform)
            c: Wave speed (natural units, typically 1.0)
            chi: Mass field parameter (scalar or array matching E shape)
            dims: Dimensionality - '1d', '2d', or '3d'
            
        Returns:
            Total energy (scalar float)
            
        Example:
            >>> energy = harness.compute_field_energy(E, E_prev, dt, dx, c, chi, dims='3d')
            >>> energy_drift = abs(energy - energy_initial) / energy_initial
        
        Notes:
            - Uses 2nd-order central differences for spatial derivatives
            - Assumes periodic boundary conditions (via roll)
            - Backend-agnostic (works with NumPy or CuPy arrays)
        """
        # Get backend module (numpy or cupy)
        try:
            from core.lfm_backend import get_array_module
            xp = get_array_module(E)
        except Exception:
            xp = self.xp
        
        # Time derivative: ∂E/∂t ≈ (E - E_prev) / dt
        Et = (E - E_prev) / dt
        
        # Spatial derivatives with periodic boundaries
        if dims == '1d':
            # 1D: ∂E/∂x using central difference
            Ex = (xp.roll(E, -1) - xp.roll(E, 1)) / (2 * dx)
            grad_sq = Ex * Ex
            dV = dx
        elif dims == '2d':
            # 2D: ∇E = (∂E/∂x, ∂E/∂y)
            Ex = (xp.roll(E, -1, axis=1) - xp.roll(E, 1, axis=1)) / (2 * dx)
            Ey = (xp.roll(E, -1, axis=0) - xp.roll(E, 1, axis=0)) / (2 * dx)
            grad_sq = Ex * Ex + Ey * Ey
            dV = dx * dx
        else:  # 3d
            # 3D: ∇E = (∂E/∂x, ∂E/∂y, ∂E/∂z)
            Ex = (xp.roll(E, -1, axis=2) - xp.roll(E, 1, axis=2)) / (2 * dx)
            Ey = (xp.roll(E, -1, axis=1) - xp.roll(E, 1, axis=1)) / (2 * dx)
            Ez = (xp.roll(E, -1, axis=0) - xp.roll(E, 1, axis=0)) / (2 * dx)
            grad_sq = Ex * Ex + Ey * Ey + Ez * Ez
            dV = dx * dx * dx
        
        # Energy density: ε(x) = ½[(∂E/∂t)² + c²|∇E|² + χ²E²]
        chi_sq = chi * chi if not hasattr(chi, 'shape') else chi * chi
        energy_density = 0.5 * (Et * Et + c * c * grad_sq + chi_sq * E * E)
        
        # Total energy: E_total = ∫ ε(x) dV ≈ Σ ε(xᵢ) · dV
        total_energy = xp.sum(energy_density) * dV
        
        # Convert to Python float (works with both numpy and cupy)
        try:
            return float(total_energy)
        except Exception:
            # Fallback for cupy arrays
            return float(xp.asnumpy(total_energy))
    
    def make_lattice_params(
        self,
        dt: float,
        dx: float,
        c: float,
        chi,
        **kwargs
    ) -> dict:
        """
        Create standardized parameter dict for lattice_step() calls.
        
        Eliminates boilerplate and ensures consistent parameter naming
        across all tier tests. All tier tests should use this helper
        instead of manually constructing parameter dicts.
        
        Standard keys:
            - dt: Time step
            - dx: Spatial step (uniform grid assumed)
            - alpha: c² (correct form for Klein-Gordon equation)
            - beta: Velocity scaling (default 1.0)
            - chi: Mass field parameter
            - backend: Physics backend ('baseline' or 'fused')
        
        Args:
            dt: Time step
            dx: Spatial grid spacing
            c: Wave speed (natural units)
            chi: Mass field parameter (scalar or array)
            **kwargs: Additional tier-specific parameters
                     (e.g., B_field for EM tests, source terms, etc.)
        
        Returns:
            Dict with standardized keys ready for lattice_step()
            
        Example:
            >>> params = harness.make_lattice_params(dt, dx, c, chi)
            >>> E_next = lattice_step(E, E_prev, params)
            
            >>> # With tier-specific additions:
            >>> params = harness.make_lattice_params(
            ...     dt, dx, c, chi,
            ...     B_field=B, source=source_term
            ... )
        
        Notes:
            - Always uses alpha=c² (not 'c' key) for correct equation form
            - Automatically includes backend from harness state
            - Additional kwargs merged without overwriting standard keys
        """
        params = {
            'dt': dt,
            'dx': dx,
            'alpha': c * c,  # α = c² for Klein-Gordon equation
            'beta': 1.0,     # Velocity scaling factor
            'chi': chi,
            'backend': self.backend
        }
        
        # Merge tier-specific parameters (without overwriting standards)
        for key, value in kwargs.items():
            if key not in params:  # Don't allow override of standard keys
                params[key] = value
        
        return params
    
    def start_test_tracking(self, background: bool = False):
        """
        Start resource tracking for current test.
        
        Args:
            background: If True, use background thread for continuous monitoring
        
        Example:
            >>> harness.start_test_tracking(background=True)
            >>> # ... run test ...
            >>> metrics = harness.stop_test_tracking()
        """
        if not self.enable_resource_tracking:
            return
        
        self._current_tracker = create_resource_tracker(sample_interval=0.5)
        self._current_tracker.start(background=background)
    
    def sample_test_resources(self):
        """
        Manually sample current resource usage.
        
        Call this periodically during test execution if not using background mode.
        
        Example:
            >>> harness.start_test_tracking(background=False)
            >>> for step in range(steps):
            >>>     # ... compute ...
            >>>     if step % 100 == 0:
            >>>         harness.sample_test_resources()
        """
        if self._current_tracker:
            self._current_tracker.sample()
    
    def stop_test_tracking(self) -> Dict:
        """
        Stop resource tracking and return metrics.
        
        Returns:
            Dict with resource metrics (cpu, memory, gpu, runtime)
            Returns zeros if tracking disabled
        
        Example:
            >>> metrics = harness.stop_test_tracking()
            >>> print(f"Peak CPU: {metrics['peak_cpu_percent']:.1f}%")
        """
        if not self._current_tracker:
            # Return empty metrics if tracking not started
            return {
                "peak_cpu_percent": 0.0,
                "peak_memory_mb": 0.0,
                "peak_gpu_memory_mb": 0.0,
                "runtime_sec": 0.0
            }
        
        self._current_tracker.stop()
        metrics = self._current_tracker.get_metrics()
        self._current_tracker = None
        return metrics
    
    def log_test_start(self, test_id: str, description: str, steps: int):
        """
        Log start of a test variant.
        
        Args:
            test_id: Test identifier (e.g., "REL-01")
            description: Human-readable test description
            steps: Number of time steps for this test
        """
        log(f"-> Starting {test_id}: {description} ({steps} steps)", "INFO")
    
    def log_test_result(
        self,
        test_id: str,
        passed: bool,
        message: str
    ):
        """
        Log result of a test variant.
        
        Args:
            test_id: Test identifier
            passed: True if test passed criteria
            message: Result message with metrics
        """
        status = "PASS ✅" if passed else "FAIL ❌"
        log(f"{test_id} {status} {message}", "INFO")

    # ---------------- Cache wrapper ----------------
    def run_test_with_cache(
        self,
        test_id: str,
        test_func,
        config: Dict,
        test_config: Dict,
        output_dir: Path,
    ):
        """Run a test function with transparent cache check/store.

        Args:
            test_id: Unique test identifier (e.g., "EM-01")
            test_func: Callable(config, test_config, output_dir) -> result
            config: Tier configuration dictionary
            test_config: Test-specific configuration dict
            output_dir: Destination directory for this test's results

        Returns:
            The result object returned by test_func, or a lightweight object
            reconstructed from cached summary when served from cache.
        """
        # Resolve config file path if available for hashing
        config_file_path = None
        try:
            cfg_path = Path(self.config_name)
            if cfg_path.exists():
                config_file_path = cfg_path
        except Exception:
            config_file_path = None

        # Use cache if enabled and not forcing re-run
        if getattr(self, "cache_manager", None) and self.use_cache and not self.force_rerun:
            try:
                if self.cache_manager.is_cache_valid(test_id, config_file_path):
                    cached_dir = self.cache_manager.get_cached_results(test_id)
                    if cached_dir and cached_dir.exists():
                        # Replace output_dir with cached contents
                        if output_dir.exists():
                            shutil.rmtree(output_dir)
                        shutil.copytree(cached_dir, output_dir)
                        # Attempt to reconstruct a minimal result from summary.json
                        summary_path = output_dir / "summary.json"
                        if summary_path.exists():
                            data = json.loads(summary_path.read_text(encoding="utf-8"))
                            class _CachedResult:
                                def __init__(self, d):
                                    self.test_id = d.get("test_id", test_id)
                                    self.description = d.get("description", "")
                                    self.passed = bool(d.get("passed", False))
                                    self.metrics = d.get("metrics", {})
                                    self.runtime_sec = float(d.get("runtime_sec", 0.0))
                            log(f"[cache] Using cached results for {test_id}", "INFO")
                            return _CachedResult(data)
                        else:
                            log(f"[cache] Cached results found for {test_id} but summary.json missing; re-running.", "WARN")
                    else:
                        log(f"[cache] Cache index valid but data missing for {test_id}; re-running.", "WARN")
                else:
                    log(f"[cache] No valid cache for {test_id}; running test.", "INFO")
            except Exception as e:
                log(f"[cache] Error during cache lookup for {test_id}: {e}", "WARN")

        # Inject diagnostics mode into test_config for fine-grained control
        try:
            mode = self.get_diagnostics_mode(test_id)
        except Exception:
            mode = "off"
        if not isinstance(test_config, dict):
            test_config = {}
        test_config = {**test_config, "diagnostics": {"mode": mode}}
        if mode != "off":
            log(f"[diag] Diagnostics mode for {test_id}: {mode}", "INFO")
        # Run the real test
        result = test_func(config, test_config, output_dir)

        # Store results if possible
        try:
            if getattr(self, "cache_manager", None) and self.use_cache:
                meta = {
                    "passed": bool(getattr(result, "passed", False)),
                    "runtime_sec": float(getattr(result, "runtime_sec", 0.0)),
                }
                self.cache_manager.store_test_results(test_id, output_dir, config_file_path, meta)
                log(f"[cache] Cached results stored for {test_id}", "INFO")
        except Exception as e:
            log(f"[cache] Failed to store cache for {test_id}: {e}", "WARN")

        # NEW: Record metrics automatically after test completes (cached or fresh)
        try:
            from harness.lfm_test_metrics import TestMetrics
            metrics_tracker = TestMetrics()
            
            # Extract metrics from result object or summary.json
            metrics_data = self._extract_metrics_for_tracking(
                test_id, result, output_dir
            )
            
            if metrics_data:
                metrics_tracker.record_run(test_id, metrics_data)
                log(f"[metrics] Recorded run metrics for {test_id}", "INFO")
        except Exception as e:
            log(f"[metrics] Failed to record metrics for {test_id}: {e}", "WARN")
            # Don't fail test on metrics errors

        return result

    def _extract_metrics_for_tracking(
        self, 
        test_id: str, 
        result: Any, 
        output_dir: Path
    ) -> Optional[Dict]:
        """
        Extract metrics from test result for TestMetrics tracking.
        
        Handles both:
        - Fresh test runs (result object with attributes)
        - Cached results (reconstructed from summary.json)
        
        Args:
            test_id: Test identifier
            result: Test result object (may be fresh or cached)
            output_dir: Directory containing test results
        
        Returns:
            Dict with keys: exit_code, runtime_sec, peak_cpu_percent,
            peak_memory_mb, peak_gpu_memory_mb, timestamp
            Or None if metrics cannot be extracted
        """
        import time
        
        metrics = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        # Try extracting from result object
        if hasattr(result, 'passed'):
            metrics["exit_code"] = 0 if result.passed else 1
        if hasattr(result, 'runtime_sec'):
            metrics["runtime_sec"] = result.runtime_sec
        if hasattr(result, 'peak_cpu_percent'):
            metrics["peak_cpu_percent"] = result.peak_cpu_percent
        if hasattr(result, 'peak_memory_mb'):
            metrics["peak_memory_mb"] = result.peak_memory_mb
        if hasattr(result, 'peak_gpu_memory_mb'):
            metrics["peak_gpu_memory_mb"] = result.peak_gpu_memory_mb
        
        # If result object incomplete, try summary.json
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text(encoding='utf-8'))
                
                # Override with summary.json data (more reliable for cached results)
                if "passed" in data:
                    metrics["exit_code"] = 0 if data["passed"] else 1
                if "runtime_sec" in data:
                    metrics["runtime_sec"] = data["runtime_sec"]
                if "peak_cpu_percent" in data:
                    metrics["peak_cpu_percent"] = data["peak_cpu_percent"]
                if "peak_memory_mb" in data:
                    metrics["peak_memory_mb"] = data["peak_memory_mb"]
                if "peak_gpu_memory_mb" in data:
                    metrics["peak_gpu_memory_mb"] = data["peak_gpu_memory_mb"]
            except Exception as e:
                log(f"[metrics] Error reading summary.json: {e}", "WARN")
        
        # Ensure required fields present
        if "exit_code" not in metrics or "runtime_sec" not in metrics:
            return None
        
        # Set defaults for optional fields
        metrics.setdefault("peak_cpu_percent", 100.0)
        metrics.setdefault("peak_memory_mb", 500.0)
        metrics.setdefault("peak_gpu_memory_mb", 0.0)
        
        return metrics
