#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Test Metrics Database - Resource usage tracking and estimation
============================================================
Persistent storage of test execution metrics to enable dynamic scheduling.
Tracks runtime, CPU, RAM, and GPU usage for each test run.
"""

import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class TestMetrics:
    """
    Manages test execution metrics database for resource-aware scheduling.
    
    Design philosophy:
    - Uses LAST successful run only for estimates (not averages across runs)
    - System state varies too much between runs for historical data to be useful
    - DB file is committed to repo as "seed data" for new users
    - Each user's system will adapt estimates to their hardware on first run
    """
    
    def __init__(self, db_path: Path = None):
        """
        Initialize metrics database.
        
        Args:
            db_path: Path to JSON database file. Defaults to results/test_metrics_history.json
        """
        if db_path is None:
            db_path = Path(__file__).parent / "results" / "test_metrics_history.json"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize database
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {}
    
    def save(self):
        """Persist database to disk."""
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def record_run(self, test_id: str, metrics: Dict):
        """
        Record test execution metrics.
        
        Args:
            test_id: Test identifier (e.g., "REL-01")
            metrics: Dict with keys: runtime_sec, peak_cpu_percent, peak_memory_mb,
                    peak_gpu_memory_mb, exit_code, timestamp
        """
        if test_id not in self.data:
            self.data[test_id] = {"runs": [], "estimated_resources": None, "priority": 50}
        
        # Add run record
        self.data[test_id]["runs"].append(metrics)
        
        # Keep only last 10 runs
        if len(self.data[test_id]["runs"]) > 10:
            self.data[test_id]["runs"] = self.data[test_id]["runs"][-10:]
        
        # Update estimated resources
        self.data[test_id]["estimated_resources"] = self._compute_estimate(test_id)
        self.data[test_id]["last_run"] = metrics.get("timestamp")
        
        # Update priority based on success
        if metrics.get("exit_code", 0) != 0:
            self.data[test_id]["priority"] = 90  # Failed test = high priority
        
        self.save()
    
    def _compute_estimate(self, test_id: str) -> Dict:
        """
        Compute resource estimate from the LAST successful run only.
        System state varies between runs, so historical averages are unreliable.
        """
        runs = self.data[test_id]["runs"]
        if not runs:
            return None
        
        # Find the most recent successful run
        last_success = None
        for r in reversed(runs):
            if r.get("exit_code", 0) == 0:
                last_success = r
                break
        
        if not last_success:
            # No successful runs - use the last run regardless
            last_success = runs[-1]
        
        # Extract raw metrics from last run
        runtime = last_success.get("runtime_sec", 1.0)
        cpu_percent = last_success.get("peak_cpu_percent", 100.0)
        memory_mb = last_success.get("peak_memory_mb", 500.0)
        gpu_memory_mb = last_success.get("peak_gpu_memory_mb", 0.0)
        
        # Convert process CPU percent to core-equivalent usage
        # psutil returns percent where 100 ~= 1 full core for the process
        cpu_cores_needed = max(0.75, cpu_percent / 100.0)
        
        # Determine if test uses GPU (threshold: 50MB to catch CuPy overhead)
        uses_gpu = gpu_memory_mb > 50
        
        # Apply modest safety buffers (10% for runtime/CPU, 20% for memory)
        runtime_buffered = runtime * 1.1
        cpu_buffered = cpu_cores_needed * 1.1
        memory_buffered = memory_mb * 1.2
        gpu_buffered = gpu_memory_mb * 1.1 if uses_gpu else 0.0
        
        # Adaptive timeout multiplier: check if last few runs timed out
        timeout_count = sum(1 for r in runs[-3:] if r.get("exit_code") == -2)
        timeout_multiplier = 3.0 + (timeout_count * 0.5)  # 3x base, +0.5x per recent timeout
        
        return {
            "runtime_sec": max(1.0, runtime_buffered),
            "cpu_cores_needed": max(0.75, cpu_buffered),
            "memory_mb": max(100, memory_buffered),
            "gpu_memory_mb": gpu_buffered,
            "uses_gpu": uses_gpu,
            "confidence": "last_run",
            "sample_size": len(runs),
            "timeout_multiplier": timeout_multiplier
        }
    
    def get_estimate(self, test_id: str, test_config: Dict = None) -> Dict:
        """
        Get resource estimate for a test.
        
        Args:
            test_id: Test identifier
            test_config: Config dict for estimation if no history exists
        
        Returns:
            Dict with estimated resource requirements
        """
        # If we have history, use it
        if test_id in self.data and self.data[test_id].get("estimated_resources"):
            return self.data[test_id]["estimated_resources"]
        
        # Otherwise, estimate from config
        if test_config:
            return self._estimate_from_config(test_id, test_config)
        
        # Fallback: assume expensive test
        return {
            "runtime_sec": 300.0,
            "cpu_cores_needed": 4.0,
            "memory_mb": 2000,
            "gpu_memory_mb": 3000,
            "uses_gpu": True,
            "confidence": "default_conservative",
            "sample_size": 0
        }
    
    def _estimate_from_config(self, test_id: str, config: Dict) -> Dict:
        """
        Estimate resource requirements from test configuration.
        Conservative estimates for never-run tests.
        """
        # Parse config parameters
        dimensions = config.get("dimensions", 1)
        grid_points = config.get("grid_points", config.get("N", 512))
        steps = config.get("steps", 6000)
        uses_gpu = config.get("use_gpu", config.get("gpu_enabled", True))
        
        # Memory estimation: total grid cells × 8 bytes × 5 fields
        # grid_points may be:
        #  - int (e.g., 512) with dimensions specifying power
        #  - list/tuple (e.g., [128,128,256])
        #  - dict with Nx/Ny/Nz keys
        if isinstance(grid_points, (list, tuple)):
            grid_size = 1
            for g in grid_points:
                try:
                    grid_size *= int(g)
                except Exception:
                    grid_size *= 1
        elif isinstance(grid_points, dict):
            # Common keys Nx, Ny, Nz
            keys = ["Nx", "Ny", "Nz"]
            grid_size = 1
            for k in keys:
                if k in grid_points:
                    try:
                        grid_size *= int(grid_points[k])
                    except Exception:
                        grid_size *= 1
            # Fallback: product of all numeric values
            if grid_size == 1:
                vals = [v for v in grid_points.values() if isinstance(v, (int, float))]
                grid_size = int(math.prod(vals)) if vals else 512
        else:
            # Assume scalar; raise to the number of dimensions
            try:
                grid_size = int(grid_points) ** int(dimensions)
            except Exception:
                grid_size = 512 ** int(dimensions)
        memory_per_field_mb = grid_size * 8 / (1024**2)
        memory_mb = memory_per_field_mb * 5 * 1.5  # 5 fields + 50% overhead
        
        # GPU memory same as RAM for GPU tests
        gpu_memory_mb = memory_mb if uses_gpu else 0
        
        # Runtime estimation (heuristic)
        # Baseline: 1M grid points, 1k steps = ~10 seconds on GPU, ~50 on CPU
        complexity_factor = (grid_size / 1e6) * (steps / 1000)
        runtime_sec = complexity_factor * (10 if uses_gpu else 50)
        
        # CPU cores (rough estimate)
        cpu_cores = 4 if dimensions == 3 else 2
        
        return {
            "runtime_sec": max(5.0, runtime_sec),
            "cpu_cores_needed": float(cpu_cores),
            "memory_mb": max(500, memory_mb),
            "gpu_memory_mb": gpu_memory_mb,
            "uses_gpu": uses_gpu,
            "confidence": "estimated_from_config",
            "sample_size": 0
        }
    
    def get_priority(self, test_id: str) -> int:
        """
        Get scheduling priority for test.
        
        Returns:
            Priority score (higher = run sooner)
            100 = never run, 90 = failed last time, 50-70 = normal
        """
        if test_id not in self.data:
            return 100  # Never run = highest priority
        
        if not self.data[test_id]["runs"]:
            return 100
        
        # Check last run
        last_run = self.data[test_id]["runs"][-1]
        if last_run.get("exit_code", 0) != 0:
            return 90  # Failed = high priority
        
        # Priority by tier (from test_id prefix)
        tier_priorities = {
            "REL": 70,   # Tier 1
            "GRAV": 60,  # Tier 2
            "ENER": 50,  # Tier 3
            "QUAN": 50,  # Tier 4
            "UNIF": 80   # Unification tests
        }
        
        prefix = test_id.split("-")[0]
        return tier_priorities.get(prefix, 50)
    
    def get_all_test_ids(self) -> List[str]:
        """Get list of all test IDs in database."""
        return list(self.data.keys())

    def get_all_runs(self, test_id: str) -> List[Dict]:
        """Return list of historical runs for a test id (may be empty)."""
        entry = self.data.get(test_id)
        if not entry:
            return []
        return list(entry.get("runs", []))
    
    def get_summary(self) -> Dict:
        """Get summary statistics of metrics database."""
        total_tests = len(self.data)
        with_history = sum(1 for t in self.data.values() if t["runs"])
        no_history = total_tests - with_history
        
        avg_runtime = 0
        if with_history > 0:
            runtimes = []
            for test_data in self.data.values():
                if test_data["runs"]:
                    runtimes.append(test_data["runs"][-1]["runtime_sec"])
            avg_runtime = sum(runtimes) / len(runtimes) if runtimes else 0
        
        return {
            "total_tests": total_tests,
            "with_history": with_history,
            "no_history": no_history,
            "avg_runtime_sec": avg_runtime
        }


def load_test_configs(tier: int) -> List[Tuple[str, Dict]]:
    """
    Load test configurations from tier config file.
    
    Args:
        tier: Tier number (1-4)
    
    Returns:
        List of (test_id, config_dict) tuples
    """
    tier_files = {
        1: "config/config_tier1_relativistic.json",
        2: "config/config_tier2_gravityanalogue.json",
        3: "config/config_tier3_energy.json",
        4: "config/config_tier4_quantization.json"
    }
    
    config_path = Path(__file__).parent.parent / tier_files.get(tier)
    if not config_path.exists():
        return []
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # Extract tests/variants
    tests = []
    
    if tier == 1:
        variants = cfg.get("variants", [])
        params = cfg.get("parameters", {})
        for v in variants:
            test_id = v["test_id"]
            # Merge params with variant-specific overrides
            test_cfg = {**params, **v}
            test_cfg["use_gpu"] = cfg.get("run_settings", {}).get("use_gpu", True)
            tests.append((test_id, test_cfg))
    
    elif tier == 2:
        variants = cfg.get("variants", [])  # Tier 2 uses "variants" like Tier 1
        params = cfg.get("parameters", {})
        for v in variants:
            test_id = v["test_id"]
            test_cfg = {**params, **v}
            test_cfg["use_gpu"] = cfg.get("run_settings", {}).get("use_gpu", True)
            tests.append((test_id, test_cfg))
    
    elif tier in (3, 4):
        test_list = cfg.get("tests", [])
        params = cfg.get("parameters", {})
        for t in test_list:
            test_id = t["test_id"]
            test_cfg = {**params, **t}
            test_cfg["gpu_enabled"] = cfg.get("hardware", {}).get("gpu_enabled", True)
            tests.append((test_id, test_cfg))
    
    return tests


if __name__ == "__main__":
    # Demo/test
    metrics = TestMetrics()
    
    # Record a sample run
    metrics.record_run("REL-01", {
        "runtime_sec": 2.1,
        "peak_cpu_percent": 12.5,
        "peak_memory_mb": 385,
        "peak_gpu_memory_mb": 0,
        "exit_code": 0,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })
    
    print("Metrics database demo:")
    print(f"Estimate for REL-01: {metrics.get_estimate('REL-01')}")
    print(f"Priority for REL-01: {metrics.get_priority('REL-01')}")
    print(f"Summary: {metrics.get_summary()}")
