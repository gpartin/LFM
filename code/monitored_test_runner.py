#!/usr/bin/env python3
"""
Monitored Test Runner - Execute tests with resource tracking
===========================================================
Launches tests via subprocess and monitors CPU, RAM, and GPU usage.
"""

import os
import psutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class MonitoredTestRunner:
    """Runs tests with real-time resource monitoring."""
    
    def __init__(self, code_dir: Path = None):
        """
        Initialize test runner.
        
        Args:
            code_dir: Directory containing test scripts (defaults to current dir)
        """
        if code_dir is None:
            code_dir = Path(__file__).parent
        self.code_dir = Path(code_dir)
        
        # Tier runner scripts
        self.tier_runners = {
            1: "run_tier1_relativistic.py",
            2: "run_tier2_gravityanalogue.py",
            3: "run_tier3_energy.py",
            4: "run_tier4_quantization.py"
        }
    
    def _query_gpu_usage(self) -> float:
        """Query current GPU memory usage in MB."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    return float(lines[0])
        except Exception:
            pass
        return 0.0
    
    def run_test_monitored(self, test_id: str, tier: int, force_cpu: bool = False,
                          verbose: bool = False, progress_callback=None) -> Dict:
        """
        Run a test with resource monitoring.
        
        Args:
            test_id: Test identifier (e.g., "REL-01")
            tier: Tier number (1-4)
            force_cpu: If True, hide GPU from test (CUDA_VISIBLE_DEVICES="")
            verbose: If True, print stdout/stderr
        
        Returns:
            Dict with metrics: runtime_sec, peak_cpu_percent, peak_memory_mb,
                             peak_gpu_memory_mb, exit_code, timestamp, stdout, stderr
        """
        # Build command
        runner_script = self.tier_runners.get(tier)
        if not runner_script:
            raise ValueError(f"Unknown tier: {tier}")
        
        cmd = ["python", str(self.code_dir / runner_script), "--test", test_id]
        
        # Setup environment (GPU control)
        env = os.environ.copy()
        if force_cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPU
        
        # Start process
        if verbose:
            print(f"[{test_id}] Starting (force_cpu={force_cpu})...", flush=True)
        
        # Add -u flag to Python to disable output buffering
        cmd_unbuffered = ["python", "-u"] + cmd[1:]
        
        # Always use PIPE and stream output in real-time
        process = subprocess.Popen(
            cmd_unbuffered,
            cwd=self.code_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Get psutil process handle for monitoring
        try:
            ps_process = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            # Process already finished
            stdout, stderr = process.communicate()
            return {
                "exit_code": process.returncode,
                "runtime_sec": 0.0,
                "peak_cpu_percent": 0.0,
                "peak_memory_mb": 0.0,
                "peak_gpu_memory_mb": 0.0,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "stdout": stdout,
                "stderr": stderr
            }
        
        # Track metrics
        metrics = {
            "peak_cpu_percent": 0.0,
            "peak_memory_mb": 0.0,
            "peak_gpu_memory_mb": 0.0,
            "samples": []
        }
        
        start_time = time.time()
        sample_interval = 0.5  # Poll every 0.5 seconds
        last_progress_report = 0
        
        # Thread to stream output in real-time
        output_lines = []
        def stream_output():
            for line in process.stdout:
                if verbose:
                    print(f"[{test_id}] {line.rstrip()}", flush=True)
                output_lines.append(line)
        
        import threading
        output_thread = threading.Thread(target=stream_output, daemon=True)
        output_thread.start()
        
        # Monitor loop
        while process.poll() is None:
            try:
                # CPU and RAM (this process only, not children for now)
                cpu_percent = ps_process.cpu_percent(interval=0.1)
                memory_info = ps_process.memory_info()
                memory_mb = memory_info.rss / (1024**2)
                
                # GPU usage (system-wide, best we can do without per-process GPU tracking)
                gpu_mb = self._query_gpu_usage() if not force_cpu else 0.0
                
                # Update peaks
                metrics["peak_cpu_percent"] = max(metrics["peak_cpu_percent"], cpu_percent)
                metrics["peak_memory_mb"] = max(metrics["peak_memory_mb"], memory_mb)
                metrics["peak_gpu_memory_mb"] = max(metrics["peak_gpu_memory_mb"], gpu_mb)
                
                # Store sample
                metrics["samples"].append({
                    "time": time.time() - start_time,
                    "cpu": cpu_percent,
                    "memory": memory_mb,
                    "gpu": gpu_mb
                })
                
                # Report progress every 5 seconds via callback
                elapsed = time.time() - start_time
                if progress_callback and (elapsed - last_progress_report) >= 5.0:
                    progress_pct = (elapsed / 10.0) * 100  # Rough estimate
                    progress_callback(min(95, progress_pct))
                    last_progress_report = elapsed
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process ended or can't access
                break
            
            time.sleep(sample_interval)
        
        # Wait for process and output thread to finish
        process.wait()
        output_thread.join(timeout=2.0)
        
        runtime = time.time() - start_time
        
        # Combine captured output
        stdout = "".join(output_lines)
        stderr = ""
        
        if verbose:
            print(f"\n[{test_id}] Completed in {runtime:.1f}s (exit_code={process.returncode})", flush=True)
            if process.returncode != 0:
                print(f"[{test_id}] Test FAILED", flush=True)
        
        return {
            "exit_code": process.returncode,
            "runtime_sec": runtime,
            "peak_cpu_percent": metrics["peak_cpu_percent"],
            "peak_memory_mb": metrics["peak_memory_mb"],
            "peak_gpu_memory_mb": metrics["peak_gpu_memory_mb"],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stdout": stdout if not verbose else None,
            "stderr": stderr if not verbose else "",
            "sample_count": len(metrics["samples"])
        }
    
    def run_test_simple(self, test_id: str, tier: int, force_cpu: bool = False) -> bool:
        """
        Run test without detailed monitoring (faster for quick tests).
        
        Returns:
            True if test passed (exit_code == 0)
        """
        runner_script = self.tier_runners.get(tier)
        if not runner_script:
            raise ValueError(f"Unknown tier: {tier}")
        
        cmd = ["python", str(self.code_dir / runner_script), "--test", test_id]
        
        env = os.environ.copy()
        if force_cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        
        result = subprocess.run(
            cmd,
            cwd=self.code_dir,
            env=env,
            capture_output=True,
            text=True
        )
        
        return result.returncode == 0


if __name__ == "__main__":
    # Demo
    import sys
    
    runner = MonitoredTestRunner()
    
    print("=== Testing MonitoredTestRunner ===")
    print("Running REL-01 with monitoring...")
    
    metrics = runner.run_test_monitored("REL-01", tier=1, verbose=True)
    
    print(f"\nResults:")
    print(f"  Runtime: {metrics['runtime_sec']:.2f}s")
    print(f"  Peak CPU: {metrics['peak_cpu_percent']:.1f}%")
    print(f"  Peak RAM: {metrics['peak_memory_mb']:.0f} MB")
    print(f"  Peak GPU: {metrics['peak_gpu_memory_mb']:.0f} MB")
    print(f"  Exit code: {metrics['exit_code']}")
    print(f"  Samples: {metrics['sample_count']}")
    
    if metrics['exit_code'] == 0:
        print("\n✓ Test PASSED")
    else:
        print("\n✗ Test FAILED")
        print(f"STDERR: {metrics['stderr'][:500]}")
