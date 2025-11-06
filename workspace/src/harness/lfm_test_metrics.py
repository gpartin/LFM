#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
LFM Test Metrics Database - Resource usage tracking and estimation
============================================================
Persistent storage of test execution metrics to enable dynamic scheduling.
Tracks runtime, CPU, RAM, and GPU usage for each test run.
"""

import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def compute_relative_error(expected: float, actual: float, *, characteristic: Optional[float] = None,
						   eps: float = 1e-12) -> float:
	"""Compute a robust relative error.

	Rules:
	- If |expected| is appreciable (> eps), use |actual-expected|/|expected|.
	- Else, if a characteristic scale is provided and appreciable, use that as denominator.
	- Else, fall back to absolute error scaled by max(|actual|, |expected|, eps) to avoid blowups.

	This avoids false 100% errors when both expected and actual are near zero.
	"""
	denom = None
	if abs(expected) > eps:
		denom = abs(expected)
	elif characteristic is not None and abs(characteristic) > eps:
		denom = abs(characteristic)
	else:
		denom = max(abs(actual), abs(expected), eps)
	return abs(actual - expected) / denom

class TestMetrics:
	"""
	Manages test execution metrics database for resource-aware scheduling.
	"""
	def __init__(self, db_path: Path = None):
		if db_path is None:
			db_path = Path(__file__).parent / "results" / "test_metrics_history.json"
		self.db_path = Path(db_path)
		self.db_path.parent.mkdir(parents=True, exist_ok=True)
		if self.db_path.exists():
			with open(self.db_path, 'r', encoding='utf-8') as f:
				self.data = json.load(f)
		else:
			self.data = {}
	def save(self):
		with open(self.db_path, 'w', encoding='utf-8') as f:
			json.dump(self.data, f, indent=2)
	def record_run(self, test_id: str, metrics: Dict):
		if test_id not in self.data:
			self.data[test_id] = {"runs": [], "estimated_resources": None, "priority": 50}
		self.data[test_id]["runs"].append(metrics)
		if len(self.data[test_id]["runs"]) > 10:
			self.data[test_id]["runs"] = self.data[test_id]["runs"][-10:]
		self.data[test_id]["estimated_resources"] = self._compute_estimate(test_id)
		self.data[test_id]["last_run"] = metrics.get("timestamp")
		if metrics.get("exit_code", 0) != 0:
			self.data[test_id]["priority"] = 90
		self.save()
	def _compute_estimate(self, test_id: str) -> Dict:
		runs = self.data[test_id]["runs"]
		if not runs:
			return None
		last_success = None
		for r in reversed(runs):
			if r.get("exit_code", 0) == 0:
				last_success = r
				break
		if not last_success:
			last_success = runs[-1]
		runtime = last_success.get("runtime_sec", 1.0)
		cpu_percent = last_success.get("peak_cpu_percent", 100.0)
		memory_mb = last_success.get("peak_memory_mb", 500.0)
		gpu_memory_mb = last_success.get("peak_gpu_memory_mb", 0.0)
		cpu_cores_needed = max(0.75, cpu_percent / 100.0)
		uses_gpu = gpu_memory_mb > 50
		runtime_buffered = runtime * 1.1
		cpu_buffered = cpu_cores_needed * 1.1
		memory_buffered = memory_mb * 1.2
		gpu_buffered = gpu_memory_mb * 1.1 if uses_gpu else 0.0
		timeout_count = sum(1 for r in runs[-3:] if r.get("exit_code") == -2)
		timeout_multiplier = 3.0 + (timeout_count * 0.5)
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
		if test_id in self.data and self.data[test_id].get("estimated_resources"):
			return self.data[test_id]["estimated_resources"]
		if test_config:
			return self._estimate_from_config(test_id, test_config)
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
		dimensions = config.get("dimensions", 1)
		grid_points = config.get("grid_points", config.get("N", 512))
		steps = config.get("steps", 6000)
		uses_gpu = config.get("use_gpu", config.get("gpu_enabled", True))
		if isinstance(grid_points, (list, tuple)):
			grid_size = 1
			for g in grid_points:
				try:
					grid_size *= int(g)
				except Exception:
					grid_size *= 1
		elif isinstance(grid_points, dict):
			keys = ["Nx", "Ny", "Nz"]
			grid_size = 1
			for k in keys:
				if k in grid_points:
					try:
						grid_size *= int(grid_points[k])
					except Exception:
						grid_size *= 1
			if grid_size == 1:
				vals = [v for v in grid_points.values() if isinstance(v, (int, float))]
				grid_size = int(math.prod(vals)) if vals else 512
		else:
			try:
				grid_size = int(grid_points) ** int(dimensions)
			except Exception:
				grid_size = 512 ** int(dimensions)
		memory_per_field_mb = grid_size * 8 / (1024**2)
		memory_mb = memory_per_field_mb * 5 * 1.5
		gpu_memory_mb = memory_mb if uses_gpu else 0
		complexity_factor = (grid_size / 1e6) * (steps / 1000)
		runtime_sec = complexity_factor * (10 if uses_gpu else 50)
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
		if test_id not in self.data:
			return 100
		if not self.data[test_id]["runs"]:
			return 100
		last_run = self.data[test_id]["runs"][-1]
		if last_run.get("exit_code", 0) != 0:
			return 90
		tier_priorities = {
			"REL": 70,
			"GRAV": 60,
			"ENER": 50,
			"QUAN": 50,
			"UNIF": 80
		}
		prefix = test_id.split("-")[0]
		return tier_priorities.get(prefix, 50)
	def get_all_test_ids(self) -> List[str]:
		return list(self.data.keys())
	def get_all_runs(self, test_id: str) -> List[Dict]:
		entry = self.data.get(test_id)
		if not entry:
			return []
		return list(entry.get("runs", []))
	def get_summary(self) -> Dict:
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
	"""Load test configurations for a tier using the central registry.

	Supports schema types:
	- "variants": list of variants + parameters (tiers 1–2 style)
	- "tests": list of tests + parameters (tiers 3–4 style)
	"""
	try:
		from harness.lfm_tiers import get_tier_by_number
	except Exception:
		get_tier_by_number = None

	# Fallback legacy mapping if registry isn't available
	legacy_map = {
		1: "config/config_tier1_relativistic.json",
		2: "config/config_tier2_gravityanalogue.json",
		3: "config/config_tier3_energy.json",
		4: "config/config_tier4_quantization.json",
	}
	if get_tier_by_number:
		tdef = get_tier_by_number(int(tier))
		if not tdef:
			return []
		config_rel = tdef.get("config")
		schema = tdef.get("config_schema", "variants")
		config_path = Path(__file__).parent / config_rel if config_rel else None
	else:
		config_path = Path(__file__).parent / legacy_map.get(tier)
		schema = "variants" if tier in (1, 2) else "tests"

	if not config_path or not config_path.exists():
		return []
	with open(config_path, 'r', encoding='utf-8') as f:
		cfg = json.load(f)

	tests: List[Tuple[str, Dict]] = []
	params = cfg.get("parameters", {})
	if schema == "variants":
		variants = cfg.get("variants", [])
		for v in variants:
			test_id = v.get("test_id")
			if not test_id:
				continue
			test_cfg = {**params, **v}
			# Harmonize GPU flag
			test_cfg["use_gpu"] = cfg.get("run_settings", {}).get("use_gpu", cfg.get("hardware", {}).get("gpu_enabled", True))
			tests.append((test_id, test_cfg))
	else:  # schema == "tests"
		test_list = cfg.get("tests", [])
		for t in test_list:
			test_id = t.get("test_id") or t.get("id")  # Support both formats
			if not test_id:
				continue
			test_cfg = {**params, **t}
			# Harmonize GPU flag
			test_cfg["gpu_enabled"] = cfg.get("hardware", {}).get("gpu_enabled", cfg.get("run_settings", {}).get("use_gpu", True))
			tests.append((test_id, test_cfg))
	return tests
