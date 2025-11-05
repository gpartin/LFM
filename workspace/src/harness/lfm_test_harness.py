#!/usr/bin/env python3
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
import math
import shutil
from pathlib import Path
from typing import Dict, Optional
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
        config_name: str = "config.json"
    ):
        """
        Initialize base harness with config and output directory.
        
        Args:
            cfg: Configuration dictionary loaded from JSON
            out_root: Root output directory for test results
            config_name: Name of config file (for error messages)
        """
        self.cfg = cfg
        self.config_name = config_name
        self.run_settings = cfg.get("run_settings", {})
        self.base = cfg.get("parameters", {})
        self.tol = cfg.get("tolerances", {})
        self.quick = bool(self.run_settings.get("quick_mode", False))
        
        # Backend selection
        use_gpu = bool(self.run_settings.get("use_gpu", False))
        self.xp, self.use_gpu = pick_backend(use_gpu)
        
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
        1. If config_path provided, use it directly
        2. Otherwise search in:
           - {script_dir}/config/{default_config_name}
           - {script_dir}/../config/{default_config_name}
        
        Args:
            config_path: Explicit path to config file (optional)
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
        if config_path:
            # Use explicit path if provided
            cand = Path(config_path)
            if cand.is_file():
                with open(cand, "r", encoding="utf-8") as f:
                    return json.load(f)
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Default: search in standard locations
        import inspect
        caller_frame = inspect.stack()[1]
        caller_file = Path(caller_frame.filename).resolve()
        script_dir = caller_file.parent
        
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
        Resolve output directory relative to script location.
        
        Args:
            output_dir_hint: Directory path (e.g., "results/Tier1")
            
        Returns:
            Resolved Path object, directory created if needed
            
        Example:
            >>> outdir = BaseTierHarness.resolve_outdir("results/MyTest")
        """
        import inspect
        caller_frame = inspect.stack()[1]
        caller_file = Path(caller_frame.filename).resolve()
        script_dir = caller_file.parent
        outdir = script_dir / output_dir_hint
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir
    
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

        return result
