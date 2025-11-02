#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com

"""
Resource Monitor - Real-time system resource tracking
====================================================
Monitors available CPU cores, RAM, and GPU memory to enable safe parallel execution.
"""

import psutil
import subprocess
from typing import Dict, List, Optional


class ResourceMonitor:
    """Monitors system resources for dynamic test scheduling."""
    
    def __init__(self, reserve_cpu_cores: int = 2, reserve_memory_mb: int = 2048, 
                 reserve_gpu_memory_mb: int = 500):
        """
        Initialize resource monitor.
        
        Args:
            reserve_cpu_cores: CPU cores to reserve for OS
            reserve_memory_mb: RAM to reserve for OS (MB)
            reserve_gpu_memory_mb: GPU memory to reserve for driver (MB)
        """
        self.total_cpu_cores = psutil.cpu_count(logical=True)
        self.total_memory_mb = psutil.virtual_memory().total / (1024**2)
        self.total_gpu_memory_mb = self._query_total_gpu_memory()
        
        self.reserve_cpu_cores = reserve_cpu_cores
        self.reserve_memory_mb = reserve_memory_mb
        self.reserve_gpu_memory_mb = reserve_gpu_memory_mb
        
        # GPU availability
        self.has_gpu = self.total_gpu_memory_mb > 0
    
    def _query_total_gpu_memory(self) -> float:
        """Query total GPU memory using nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse output (may have multiple GPUs, use first)
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    return float(lines[0])
        except Exception:
            pass
        return 0.0
    
    def _query_gpu_free_memory(self) -> float:
        """Query available GPU memory using nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    return float(lines[0])
        except Exception:
            pass
        return 0.0
    
    def available_resources(self) -> Dict[str, float]:
        """
        Get currently available resources.
        
        Returns:
            Dict with keys: cpu_cores, memory_mb, gpu_memory_mb
        """
        # CPU availability (approximate from current usage)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_used_cores = (cpu_percent / 100.0) * self.total_cpu_cores
        cpu_available = max(0, self.total_cpu_cores - cpu_used_cores - self.reserve_cpu_cores)
        
        # Memory availability
        mem_available_mb = psutil.virtual_memory().available / (1024**2)
        mem_available = max(0, mem_available_mb - self.reserve_memory_mb)
        
        # GPU availability
        if self.has_gpu:
            gpu_free = self._query_gpu_free_memory()
            gpu_available = max(0, gpu_free - self.reserve_gpu_memory_mb)
        else:
            gpu_available = 0
        
        return {
            "cpu_cores": cpu_available,
            "memory_mb": mem_available,
            "gpu_memory_mb": gpu_available
        }
    
    def can_fit(self, test_estimate: Dict, running_tests: List[Dict]) -> tuple[bool, str]:
        """
        Check if test can start given current resource usage.
        
        Args:
            test_estimate: Resource estimate dict from TestMetrics
            running_tests: List of currently running test estimates
        
        Returns:
            (can_fit: bool, reason: str)
        """
        available = self.available_resources()
        
        # Calculate total resource needs if we add this test
        total_cpu = sum(t.get("cpu_cores_needed", 2) for t in running_tests) + test_estimate.get("cpu_cores_needed", 2)
        total_memory = sum(t.get("memory_mb", 500) for t in running_tests) + test_estimate.get("memory_mb", 500)
        total_gpu = sum(t.get("gpu_memory_mb", 0) for t in running_tests) + test_estimate.get("gpu_memory_mb", 0)
        
        # Check each constraint
        if total_memory > available["memory_mb"]:
            needed = test_estimate.get("memory_mb", 500)
            avail = available["memory_mb"] - sum(t.get("memory_mb", 500) for t in running_tests)
            return False, f"Insufficient RAM (need {needed:.0f}MB, have {avail:.0f}MB free)"
        
        if test_estimate.get("uses_gpu", False) and total_gpu > available["gpu_memory_mb"]:
            needed = test_estimate.get("gpu_memory_mb", 0)
            avail = available["gpu_memory_mb"] - sum(t.get("gpu_memory_mb", 0) for t in running_tests)
            return False, f"Insufficient VRAM (need {needed:.0f}MB, have {avail:.0f}MB free)"
        
        if total_cpu > available["cpu_cores"]:
            needed = test_estimate.get("cpu_cores_needed", 2)
            avail = available["cpu_cores"] - sum(t.get("cpu_cores_needed", 2) for t in running_tests)
            return False, f"Insufficient CPU cores (need {needed:.1f}, have {avail:.1f} free)"
        
        return True, "OK"
    
    def get_system_info(self) -> Dict:
        """Get system resource summary."""
        return {
            "cpu_cores": self.total_cpu_cores,
            "memory_gb": self.total_memory_mb / 1024,
            "gpu_memory_gb": self.total_gpu_memory_mb / 1024 if self.has_gpu else 0,
            "has_gpu": self.has_gpu,
            "reserves": {
                "cpu_cores": self.reserve_cpu_cores,
                "memory_mb": self.reserve_memory_mb,
                "gpu_memory_mb": self.reserve_gpu_memory_mb
            }
        }
    
    def check_gpu_exclusive_mode(self, test_estimate: Dict, running_tests: List[Dict]) -> bool:
        """
        Check if GPU should be in exclusive mode.
        Phase 1: Allow multiple lightweight GPU tests to share GPU.
        
        Args:
            test_estimate: Estimate for test we want to start
            running_tests: Currently running test estimates
        
        Returns:
            True if GPU is available for this test
        """
        if not test_estimate.get("uses_gpu", False):
            return True  # CPU test, no GPU constraint
        
        # Calculate total GPU memory that would be used
        current_gpu_usage = sum(t.get("gpu_memory_mb", 0) for t in running_tests if t.get("uses_gpu", False))
        new_total = current_gpu_usage + test_estimate.get("gpu_memory_mb", 0)
        
        # Allow if total GPU usage stays under 6GB (leaving 2GB+ headroom on 8GB card)
        # This enables multiple lightweight tests to share GPU
        available_gpu = self.total_gpu_memory_mb - self.reserve_gpu_memory_mb
        return new_total <= (available_gpu * 0.75)  # Use max 75% of available GPU memory


if __name__ == "__main__":
    # Demo
    monitor = ResourceMonitor()
    
    print("=== System Resources ===")
    info = monitor.get_system_info()
    print(f"CPU: {info['cpu_cores']} cores")
    print(f"RAM: {info['memory_gb']:.1f} GB")
    print(f"GPU: {info['gpu_memory_gb']:.1f} GB" if info['has_gpu'] else "GPU: Not available")
    
    print("\n=== Available Now ===")
    avail = monitor.available_resources()
    print(f"CPU cores: {avail['cpu_cores']:.1f}")
    print(f"RAM: {avail['memory_mb']:.0f} MB")
    print(f"VRAM: {avail['gpu_memory_mb']:.0f} MB")
    
    print("\n=== Can Fit Test? ===")
    test_estimate = {
        "cpu_cores_needed": 2.0,
        "memory_mb": 500,
        "gpu_memory_mb": 1200,
        "uses_gpu": True
    }
    can_fit, reason = monitor.can_fit(test_estimate, [])
    print(f"Test needs: {test_estimate}")
    print(f"Can fit: {can_fit} ({reason})")
