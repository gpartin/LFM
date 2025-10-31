#!/usr/bin/env python3
"""
Adaptive Test Scheduler - Dynamic parallel test execution
========================================================
Schedules and runs tests in parallel based on resource availability and priorities.
"""

import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from test_metrics import TestMetrics, load_test_configs
from resource_monitor import ResourceMonitor
from monitored_test_runner import MonitoredTestRunner


@dataclass
class RunningTest:
    """Represents a currently executing test."""
    test_id: str
    tier: int
    estimate: Dict
    thread: threading.Thread
    start_time: float
    metrics: Optional[Dict] = None  # Filled when test completes


class AdaptiveScheduler:
    """Schedules tests dynamically based on resource availability."""
    
    def __init__(self, metrics_db_path: Path = None, verbose: bool = True):
        """
        Initialize scheduler.
        
        Args:
            metrics_db_path: Path to metrics database JSON file
            verbose: If True, print progress messages
        """
        self.metrics = TestMetrics(metrics_db_path)
        self.monitor = ResourceMonitor()
        self.runner = MonitoredTestRunner()
        self.verbose = verbose
        
        self.running_tests: List[RunningTest] = []
        self.completed_tests: List[Dict] = []
        self.failed_tests: List[Dict] = []
        
        # Thread lock for updating shared state
        self.lock = threading.Lock()
    
    def _log(self, message: str, level: str = "INFO"):
        """Print log message if verbose."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}", flush=True)
    
    def _test_runner_thread(self, test_id: str, tier: int, force_cpu: bool):
        """Thread function to run a test with monitoring."""
        try:
            metrics = self.runner.run_test_monitored(
                test_id, tier, force_cpu=force_cpu, 
                verbose=self.verbose, 
                progress_callback=lambda pct: self._log(f"  [{test_id}] {pct:.0f}% complete", "PROGRESS")
            )
            
            # Update metrics database
            self.metrics.record_run(test_id, metrics)
            
            # Mark test as complete
            with self.lock:
                # Find this test in running list
                for rt in self.running_tests:
                    if rt.test_id == test_id and rt.metrics is None:
                        rt.metrics = metrics
                        break
        
        except Exception as e:
            self._log(f"Test {test_id} crashed: {e}", "ERROR")
            # Record failed run
            failed_metrics = {
                "exit_code": -1,
                "runtime_sec": 0.0,
                "peak_cpu_percent": 0.0,
                "peak_memory_mb": 0.0,
                "peak_gpu_memory_mb": 0.0,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "error": str(e)
            }
            self.metrics.record_run(test_id, failed_metrics)
            
            with self.lock:
                for rt in self.running_tests:
                    if rt.test_id == test_id and rt.metrics is None:
                        rt.metrics = failed_metrics
                        break
    
    def _try_start_test(self, test_id: str, tier: int, test_config: Dict) -> bool:
        """
        Try to start a test if resources are available.
        
        Returns:
            True if test was started
        """
        # Get resource estimate
        estimate = self.metrics.get_estimate(test_id, test_config)
        
        # Check if it fits
        running_estimates = [rt.estimate for rt in self.running_tests]
        can_fit, reason = self.monitor.can_fit(estimate, running_estimates)
        
        if not can_fit:
            return False
        
        # Check GPU exclusive mode (Phase 1: only 1 GPU test at a time)
        if estimate.get("uses_gpu", False):
            if not self.monitor.check_gpu_exclusive_mode(estimate, running_estimates):
                return False
        
        # Start the test!
        force_cpu = False
        if estimate.get("uses_gpu", False):
            # For GPU tests, check if we should force CPU instead
            # (if GPU not actually needed or too crowded)
            pass  # For now, respect estimate
        
        # Create thread
        thread = threading.Thread(
            target=self._test_runner_thread,
            args=(test_id, tier, force_cpu),
            daemon=True
        )
        
        running_test = RunningTest(
            test_id=test_id,
            tier=tier,
            estimate=estimate,
            thread=thread,
            start_time=time.time()
        )
        
        with self.lock:
            self.running_tests.append(running_test)
        
        thread.start()
        
        # Log
        runtime_est = estimate.get("runtime_sec", 0)
        mem_est = estimate.get("memory_mb", 0)
        gpu_est = estimate.get("gpu_memory_mb", 0)
        confidence = estimate.get("confidence", "unknown")
        
        self._log(
            f"✓ Started {test_id} (est: {runtime_est:.1f}s, "
            f"RAM: {mem_est:.0f}MB, GPU: {gpu_est:.0f}MB, {confidence})"
        )
        
        return True
    
    def _poll_completed_tests(self) -> List[RunningTest]:
        """Check for completed tests and return them."""
        completed = []
        
        with self.lock:
            still_running = []
            for rt in self.running_tests:
                if rt.metrics is not None:
                    # Test completed
                    completed.append(rt)
                    
                    # Log completion
                    runtime = rt.metrics["runtime_sec"]
                    status = "PASS" if rt.metrics["exit_code"] == 0 else "FAIL"
                    peak_ram = rt.metrics["peak_memory_mb"]
                    peak_gpu = rt.metrics["peak_gpu_memory_mb"]
                    
                    self._log(
                        f"✓ Completed {rt.test_id} ({runtime:.1f}s, {status}, "
                        f"RAM: {peak_ram:.0f}MB, GPU: {peak_gpu:.0f}MB)"
                    )
                    
                    # Track results
                    if rt.metrics["exit_code"] == 0:
                        self.completed_tests.append({
                            "test_id": rt.test_id,
                            "tier": rt.tier,
                            "metrics": rt.metrics
                        })
                    else:
                        self.failed_tests.append({
                            "test_id": rt.test_id,
                            "tier": rt.tier,
                            "metrics": rt.metrics
                        })
                else:
                    still_running.append(rt)
            
            self.running_tests = still_running
        
        return completed
    
    def schedule_tests(self, test_list: List[Tuple[str, int, Dict]], 
                      max_concurrent: int = 4) -> Dict:
        """
        Main scheduling loop.
        
        Args:
            test_list: List of (test_id, tier, config) tuples
            max_concurrent: Maximum tests to run concurrently
        
        Returns:
            Summary dict with statistics
        """
        start_time = time.time()
        
        # Priority sort
        def priority_key(item):
            test_id, tier, config = item
            return -self.metrics.get_priority(test_id)  # Negative for descending
        
        pending = sorted(test_list, key=priority_key)
        
        self._log(f"=== Scheduling {len(pending)} tests ===")
        self._log(f"Max concurrent: {max_concurrent}")
        
        sys_info = self.monitor.get_system_info()
        self._log(
            f"System: {sys_info['cpu_cores']} cores, "
            f"{sys_info['memory_gb']:.1f}GB RAM, "
            f"{sys_info['gpu_memory_gb']:.1f}GB GPU"
        )
        
        # Main loop
        while pending or self.running_tests:
            # Try to start new tests
            started_any = False
            while pending and len(self.running_tests) < max_concurrent:
                test_id, tier, config = pending[0]
                
                # Skip tests marked as skip in config
                if config.get("skip", False):
                    self._log(f"⊘ Skipping {test_id} (skip=true in config)", "WARN")
                    pending.pop(0)
                    continue
                
                if self._try_start_test(test_id, tier, config):
                    pending.pop(0)
                    started_any = True
                else:
                    # Can't start this test, wait for resources
                    break
            
            # Wait a bit before polling again
            time.sleep(1.0)
            
            # Check for completed tests
            completed = self._poll_completed_tests()
            
            # Show progress
            if completed or started_any:
                total = len(self.completed_tests) + len(self.failed_tests) + len(self.running_tests) + len(pending)
                done = len(self.completed_tests) + len(self.failed_tests)
                running_names = ", ".join([rt.test_id for rt in self.running_tests])
                self._log(f"Progress: {done}/{total} complete, {len(self.running_tests)} running ({running_names}), {len(pending)} pending")
        
        # All tests finished
        total_runtime = time.time() - start_time
        
        self._log("=== All Tests Complete ===")
        self._log(f"Total runtime: {total_runtime:.1f}s ({total_runtime/60:.1f} min)")
        self._log(f"Passed: {len(self.completed_tests)}")
        self._log(f"Failed: {len(self.failed_tests)}")
        
        if self.failed_tests:
            self._log("Failed tests:")
            for ft in self.failed_tests:
                self._log(f"  - {ft['test_id']}", "FAIL")
        
        return {
            "total_runtime_sec": total_runtime,
            "completed": len(self.completed_tests),
            "failed": len(self.failed_tests),
            "completed_tests": self.completed_tests,
            "failed_tests": self.failed_tests
        }


if __name__ == "__main__":
    # Demo with a few fast tests
    scheduler = AdaptiveScheduler(verbose=True)
    
    # Test with 4 fast tests
    test_list = [
        ("REL-01", 1, {"dimensions": 1, "grid_points": 512, "steps": 6000, "use_gpu": True}),
        ("REL-02", 1, {"dimensions": 1, "grid_points": 512, "steps": 6000, "use_gpu": True}),
        ("GRAV-12", 2, {"dimensions": 1, "grid_points": 64, "steps": 4800, "use_gpu": True}),
        ("GRAV-23", 2, {"dimensions": 1, "grid_points": 64, "steps": 600, "use_gpu": True}),
    ]
    
    print("=== ADAPTIVE SCHEDULER DEMO ===\n")
    results = scheduler.schedule_tests(test_list, max_concurrent=3)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total time: {results['total_runtime_sec']:.1f}s")
    print(f"Completed: {results['completed']}")
    print(f"Failed: {results['failed']}")
