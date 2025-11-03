#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
resource_tracking.py — Resource Monitoring for LFM Test Runners
---------------------------------------------------------------
Purpose:
    Track CPU, RAM, and GPU usage during test execution to collect
    accurate resource metrics for performance analysis and budgeting.

Features:
    - Per-process CPU and memory tracking via psutil
    - GPU memory tracking via nvidia-smi (if available)
    - Lightweight sampling with minimal overhead
    - Thread-safe background monitoring for long-running tests

Usage:
    >>> tracker = ResourceTracker()
    >>> tracker.start()
    >>> # ... run test ...
    >>> tracker.sample()  # or use background sampling
    >>> metrics = tracker.get_metrics()
    >>> print(f"Peak CPU: {metrics['peak_cpu_percent']:.1f}%")
    >>> print(f"Peak RAM: {metrics['peak_memory_mb']:.1f} MB")
"""

import subprocess
import time
import threading
from typing import Dict, Optional
from pathlib import Path

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    psutil = None


class ResourceTracker:
    """
    Track CPU, RAM, and GPU usage during test execution.
    
    Monitors the current process and records peak usage for:
    - CPU percentage (multi-core aware)
    - Memory (RSS in MB)
    - GPU memory (if nvidia-smi available)
    
    Supports both manual sampling and background monitoring.
    """
    
    def __init__(self, sample_interval: float = 0.5):
        """
        Initialize resource tracker.
        
        Args:
            sample_interval: Time between samples in background mode (seconds)
        """
        self.sample_interval = sample_interval
        
        # Metrics
        self.peak_cpu = 0.0
        self.peak_memory_mb = 0.0
        self.peak_gpu_mb = 0.0
        self.start_time = None
        self.stop_time = None
        
        # Process handle
        self._process = None
        self._gpu_available = None  # Cache GPU availability check
        self._gpu_baseline_mb = 0.0  # Baseline GPU usage before test
        
        # Background monitoring
        self._monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
    def start(self, background: bool = False):
        """
        Begin tracking resources.
        
        Args:
            background: If True, start background thread for continuous sampling
        """
        self.start_time = time.time()
        
        # Initialize process handle
        if _HAS_PSUTIL:
            try:
                self._process = psutil.Process()
                # Establish CPU baseline (first call with interval=0 returns 0.0)
                self._process.cpu_percent(interval=0)
            except Exception:
                self._process = None
        else:
            self._process = None
        
        # Check GPU availability and get baseline
        if self._gpu_available is None:
            self._gpu_available = self._check_gpu_available()
        
        if self._gpu_available:
            self._gpu_baseline_mb = self._query_gpu_total_mb()
        
        # Start background monitoring if requested
        if background:
            self.start_background_monitoring()
    
    def stop(self):
        """Stop tracking and finalize metrics."""
        self.stop_time = time.time()
        if self._monitoring:
            self.stop_background_monitoring()
    
    def sample(self):
        """
        Sample current resource usage and update peaks.
        
        Call this periodically during test execution (e.g., every N steps)
        or use background monitoring for automatic sampling.
        """
        # CPU and memory via psutil
        if self._process and _HAS_PSUTIL:
            try:
                # CPU percentage (can be > 100% on multi-core)  
                # Use small interval to get accurate reading
                cpu = self._process.cpu_percent(interval=0.01)
                self.peak_cpu = max(self.peak_cpu, cpu)
                
                # Memory in MB
                mem_info = self._process.memory_info()
                mem_mb = mem_info.rss / (1024**2)
                self.peak_memory_mb = max(self.peak_memory_mb, mem_mb)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # GPU via nvidia-smi
        if self._gpu_available:
            gpu_mb = self._query_gpu_total_mb()
            # Use delta from baseline (accounts for system overhead)
            gpu_used = max(0.0, gpu_mb - self._gpu_baseline_mb)
            self.peak_gpu_mb = max(self.peak_gpu_mb, gpu_used)
    
    def start_background_monitoring(self):
        """Start background thread for continuous sampling."""
        if self._monitoring:
            return  # Already monitoring
        
        self._stop_event.clear()
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_background_monitoring(self):
        """Stop background monitoring thread."""
        if not self._monitoring:
            return
        
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        self._monitoring = False
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            self.sample()
            self._stop_event.wait(self.sample_interval)
    
    def get_metrics(self) -> Dict:
        """
        Return collected metrics.
        
        Returns:
            Dict with keys:
                - peak_cpu_percent: Maximum CPU usage (%)
                - peak_memory_mb: Maximum memory usage (MB)
                - peak_gpu_memory_mb: Maximum GPU memory usage (MB)
                - runtime_sec: Total runtime (seconds)
        """
        runtime = 0.0
        if self.start_time:
            end = self.stop_time if self.stop_time else time.time()
            runtime = end - self.start_time
        
        return {
            "peak_cpu_percent": float(self.peak_cpu),
            "peak_memory_mb": float(self.peak_memory_mb),
            "peak_gpu_memory_mb": float(self.peak_gpu_mb),
            "runtime_sec": float(runtime)
        }
    
    def _check_gpu_available(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False
    
    def _query_gpu_total_mb(self) -> float:
        """Query total GPU memory used (system-wide) via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                # Sum all GPUs (first value from each line)
                total = 0.0
                for line in result.stdout.strip().split('\n'):
                    try:
                        total += float(line.split()[0])
                    except (ValueError, IndexError):
                        continue
                return total
        except Exception:
            pass
        return 0.0


class DummyResourceTracker:
    """
    Dummy tracker that returns zeros when psutil is not available.
    Allows tests to run without psutil but with degraded metrics.
    """
    
    def __init__(self, sample_interval: float = 0.5):
        self.start_time = None
        self.stop_time = None
    
    def start(self, background: bool = False):
        self.start_time = time.time()
    
    def stop(self):
        self.stop_time = time.time()
    
    def sample(self):
        pass
    
    def start_background_monitoring(self):
        pass
    
    def stop_background_monitoring(self):
        pass
    
    def get_metrics(self) -> Dict:
        runtime = 0.0
        if self.start_time:
            end = self.stop_time if self.stop_time else time.time()
            runtime = end - self.start_time
        
        return {
            "peak_cpu_percent": 0.0,
            "peak_memory_mb": 0.0,
            "peak_gpu_memory_mb": 0.0,
            "runtime_sec": float(runtime)
        }


def create_resource_tracker(sample_interval: float = 0.5) -> ResourceTracker:
    """
    Factory function to create appropriate resource tracker.
    
    Args:
        sample_interval: Sampling interval for background monitoring
        
    Returns:
        ResourceTracker if psutil available, DummyResourceTracker otherwise
    """
    if _HAS_PSUTIL:
        return ResourceTracker(sample_interval)
    else:
        return DummyResourceTracker(sample_interval)
