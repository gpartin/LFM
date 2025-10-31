#!/usr/bin/env python3
"""
Parallel test runner using multiprocessing for maximum speed.
Output is buffered per-test but progress is reported when each test completes.

Resource estimation and scheduling:
- Uses last successful run metrics for each test (not historical averages)
- Hybrid scheduling: priority tiers + longest-job-first within tier
- Dynamic budgets based on available CPU/RAM/GPU at start time
- Starvation guard ensures progress even when tests exceed budgets
- Metrics DB (test_metrics_history.json) is committed as seed data for new users
"""

import subprocess
import sys
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
from datetime import datetime

import psutil  # system memory info

from test_metrics import TestMetrics, load_test_configs


def run_single_test(test_id: str, tier: int, timeout_sec: int) -> Tuple[str, int, Dict]:
    """
    Run a single test in a subprocess with real-time resource monitoring.
    Returns (test_id, tier, result_dict).
    """
    # Map tier to runner script
    runners = {
        1: "run_tier1_relativistic.py",
        2: "run_tier2_gravityanalogue.py",
        3: "run_tier3_energy.py",
        4: "run_tier4_quantization.py"
    }
    
    runner = runners.get(tier)
    if not runner:
        return (test_id, tier, {
            "exit_code": -1,
            "runtime_sec": 0.0,
            "error": f"Unknown tier {tier}"
        })
    
    # Run with -u for unbuffered output
    cmd = ["python", "-u", runner, "--test", test_id]
    
    start_time = time.time()
    
    # Helper: query total GPU memory used (system-wide)
    def _query_gpu_total_mb() -> float:
        try:
            gpu_query = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if gpu_query.returncode == 0 and gpu_query.stdout.strip():
                return float(gpu_query.stdout.strip().split()[0])
        except Exception:
            pass
        return 0.0

    # Helper: query per-PID GPU memory used; returns None if unsupported/unavailable
    def _query_pid_gpu_mb(pid: int) -> Optional[float]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if proc.returncode != 0:
                return None
            out = proc.stdout.strip()
            if not out:
                return 0.0
            total = 0.0
            pid_str = str(pid)
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0] == pid_str:
                    try:
                        total += float(parts[1])
                    except Exception:
                        continue
            return total
        except Exception:
            return None
    
    # Run and monitor resources
    try:
        # Measure GPU baseline before starting test (for delta fallback)
        gpu_baseline_mb = _query_gpu_total_mb()
        
        # Start subprocess
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Monitor resources while running
        peak_cpu = 0.0
        peak_memory_mb = 0.0
        peak_gpu_used_mb = 0.0  # Track per-PID used if available, else total delta
        
        try:
            ps_process = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            ps_process = None
        
        output_lines = []
        poll_interval = 0.5  # seconds
        last_poll = start_time
        
        while True:
            # Check if process finished
            retcode = process.poll()
            if retcode is not None:
                # Collect remaining output
                remaining = process.stdout.read()
                if remaining:
                    output_lines.append(remaining)
                break
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                process.kill()
                process.wait()
                runtime = time.time() - start_time
                return (test_id, tier, {
                    "exit_code": -2,
                    "runtime_sec": runtime,
                    "error": f"Test timed out after {timeout_sec} seconds",
                    "peak_cpu_percent": peak_cpu,
                    "peak_memory_mb": peak_memory_mb,
                    "peak_gpu_memory_mb": peak_gpu_used_mb,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                })
            
            # Poll resources periodically
            now = time.time()
            if now - last_poll >= poll_interval:
                if ps_process:
                    try:
                        cpu_pct = ps_process.cpu_percent(interval=0.1)
                        mem_info = ps_process.memory_info()
                        mem_mb = mem_info.rss / (1024**2)
                        peak_cpu = max(peak_cpu, cpu_pct)
                        peak_memory_mb = max(peak_memory_mb, mem_mb)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Query per-PID GPU usage if supported; fallback to system delta
                pid_used = _query_pid_gpu_mb(process.pid)
                if pid_used is None:
                    gpu_current_mb = _query_gpu_total_mb()
                    gpu_used = max(0.0, gpu_current_mb - gpu_baseline_mb)
                else:
                    gpu_used = max(0.0, pid_used)
                peak_gpu_used_mb = max(peak_gpu_used_mb, gpu_used)
                
                last_poll = now
            
            # Read output in small chunks (non-blocking)
            time.sleep(0.1)
        
        runtime = time.time() - start_time
        stdout = "".join(output_lines)
        
        return (test_id, tier, {
            "exit_code": retcode,
            "runtime_sec": runtime,
            "stdout": stdout,
            "stderr": "",
            "peak_cpu_percent": peak_cpu,
            "peak_memory_mb": peak_memory_mb,
            "peak_gpu_memory_mb": peak_gpu_used_mb,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })
    
    except Exception as e:
        runtime = time.time() - start_time
        return (test_id, tier, {
            "exit_code": -3,
            "runtime_sec": runtime,
            "error": str(e),
            "peak_cpu_percent": 0.0,
            "peak_memory_mb": 0.0,
            "peak_gpu_memory_mb": 0.0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })


def progress_callback(result: Tuple[str, int, Dict]):
    """Called when a test completes - print progress immediately."""
    test_id, tier, metrics = result
    status = "✓ PASS" if metrics["exit_code"] == 0 else "✗ FAIL"
    runtime = metrics.get("runtime_sec", 0.0)
    print(f"  {status} {test_id} ({runtime:.1f}s)", flush=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests in parallel for maximum speed")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fast", action="store_true",
                      help="Run 4 fast tests for quick validation")
    group.add_argument("--tiers", type=str,
                      help="Run tests from specific tiers (comma-separated, e.g. '1,2')")
    group.add_argument("--tests", type=str,
                      help="Run specific tests (comma-separated, e.g. 'REL-01,GRAV-12')")
    
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count - 2)")
    parser.add_argument("--timeout", type=int, default=7200,
                       help="Minimum per-test timeout in seconds (default: 7200 = 120 min). Adaptive estimates may increase this for long tests.")
    
    return parser.parse_args()


def get_fast_tests() -> List[Tuple[str, int]]:
    """Get fast test list."""
    return [
        ("REL-01", 1),
        ("REL-02", 1),
        ("GRAV-12", 2),
        ("GRAV-23", 2),
    ]


def get_tests_from_tiers(tiers: List[int]) -> List[Tuple[str, int]]:
    """Load all tests from specified tiers."""
    tests = []
    
    for tier in tiers:
        tier_configs = load_test_configs(tier)
        for test_id, config in tier_configs:
            tests.append((test_id, tier))
    
    return tests


def get_specific_tests(test_ids: List[str]) -> List[Tuple[str, int]]:
    """Get specific tests by ID."""
    # Map test prefixes to tiers
    tier_map = {
        "REL": 1,
        "GRAV": 2,
        "ENRG": 3,
        "QUANT": 4
    }
    
    tests = []
    for test_id in test_ids:
        prefix = test_id.split("-")[0]
        tier = tier_map.get(prefix)
        if tier:
            tests.append((test_id, tier))
        else:
            print(f"Warning: Unknown test ID format: {test_id}")
    
    return tests


def update_master_test_status():
    """
    Scan results directory and update MASTER_TEST_STATUS.csv with current test results.
    """
    results_dir = Path("results")
    
    # Test categories and expected counts
    categories = {
        1: {"name": "Relativistic", "prefix": "REL", "expected": 15},
        2: {"name": "Gravity Analogue", "prefix": "GRAV", "expected": 25},
        3: {"name": "Energy Conservation", "prefix": "ENRG", "expected": 11},
        4: {"name": "Quantization", "prefix": "QUANT", "expected": 9},
    }
    
    # Scan all summary.json files
    all_tests = {}
    for tier, cat_info in categories.items():
        cat_dir = results_dir / cat_info["name"].replace(" ", "")
        if not cat_dir.exists():
            continue
            
        for test_dir in cat_dir.iterdir():
            if not test_dir.is_dir():
                continue
                
            summary_file = test_dir / "summary.json"
            if summary_file.exists():
                try:
                    # Read JSON summaries as UTF-8 to support non-ASCII characters
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    test_id = summary.get("test_id", test_dir.name)
                    status = summary.get("status", "UNKNOWN")
                    description = summary.get("description", "")
                    notes = summary.get("notes", "")
                    
                    all_tests[test_id] = {
                        "tier": tier,
                        "description": description,
                        "status": status,
                        "notes": notes
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not read {summary_file}: {e}")
    
    # Generate CSV content
    lines = []
    lines.append("MASTER TEST STATUS REPORT - LFM Lattice Field Model")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Validation Rule: Suite marked NOT RUN if any test missing from CSV")
    lines.append("")
    lines.append("CATEGORY SUMMARY")
    lines.append("Tier,Category,Expected_Tests,Tests_In_CSV,Status,Pass_Rate")
    
    # Category summaries
    for tier in sorted(categories.keys()):
        cat_info = categories[tier]
        prefix = cat_info["prefix"]
        expected = cat_info["expected"]
        
        # Find all tests for this category
        tier_tests = {tid: info for tid, info in all_tests.items() if info["tier"] == tier}
        tests_in_csv = len(tier_tests)
        
        # Count statuses
        passed = sum(1 for t in tier_tests.values() if t["status"] == "PASS")
        failed = sum(1 for t in tier_tests.values() if t["status"] == "FAIL")
        skipped = sum(1 for t in tier_tests.values() if t["status"] in ["SKIP", "SKIPPED"])
        unknown = sum(1 for t in tier_tests.values() if t["status"] not in ["PASS", "FAIL", "SKIP", "SKIPPED"])
        missing = expected - tests_in_csv
        
        # Determine overall status
        if tests_in_csv == 0 or missing > expected // 2:
            status = "NOT RUN"
        elif passed == tests_in_csv:
            status = "PASS"
        elif passed > 0:
            status = "PARTIAL"
        else:
            status = "FAIL"
        
        # Build pass rate string
        parts = []
        if passed > 0:
            parts.append(f"{passed}/{tests_in_csv} passed")
        if skipped > 0:
            parts.append(f"{skipped} skipped")
        if missing > 0:
            parts.append(f"{missing} missing")
        if unknown > 0:
            parts.append(f"{unknown} unknown")
        
        pass_rate = " - ".join(parts) if parts else f"{passed}/{tests_in_csv} (100%)"
        
        lines.append(f"Tier {tier},{cat_info['name']},{expected},{tests_in_csv},{status},{pass_rate}")
    
    lines.append("")
    lines.append("DETAILED TEST RESULTS")
    lines.append("")
    
    # Detailed results by tier
    for tier in sorted(categories.keys()):
        cat_info = categories[tier]
        tier_tests = {tid: info for tid, info in all_tests.items() if info["tier"] == tier}
        
        if not tier_tests:
            continue
            
        lines.append(f"TIER {tier} - {cat_info['name'].upper()} ({len(tier_tests)}/{cat_info['expected']} tests)")
        lines.append("Test_ID,Description,Status,Notes")
        
        # Sort by test ID
        for test_id in sorted(tier_tests.keys()):
            info = tier_tests[test_id]
            desc = info["description"].replace(",", ";")  # Escape commas
            notes = info["notes"].replace(",", ";")  # Escape commas
            lines.append(f"{test_id},{desc},{info['status']},{notes}")
        
        lines.append("")
    
    # Write to file
    output_file = results_dir / "MASTER_TEST_STATUS.csv"
    # Write CSV with UTF-8 BOM so it's Excel-friendly on Windows and avoids cp1252 encode errors
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        f.write('\n'.join(lines))
    
    print(f"\n✓ Updated {output_file}")


def main():
    """Run tests in parallel."""
    args = parse_args()
    
    # Determine test list
    if args.fast:
        tests = get_fast_tests()
        print("=== FAST TEST MODE ===")
    elif args.tiers:
        tiers = [int(t.strip()) for t in args.tiers.split(",")]
        tests = get_tests_from_tiers(tiers)
        print(f"=== TIER MODE: {tiers} ===")
    elif args.tests:
        test_ids = [t.strip() for t in args.tests.split(",")]
        tests = get_specific_tests(test_ids)
        print(f"=== SPECIFIC TESTS MODE ===")
    else:
        print("Error: Must specify --fast, --tiers, or --tests")
        sys.exit(1)
    
    if not tests:
        print("Error: No tests to run")
        sys.exit(1)
    
    # Data-driven resource budgets and scheduling (no hardcoded test groups)
    # Build map of test_id -> config so we can estimate resources for never-run tests
    test_configs: Dict[str, Dict] = {}
    tier_set = sorted(list({tier for _, tier in tests}))
    for tier in tier_set:
        for tid, cfg in load_test_configs(tier):
            test_configs[tid] = cfg

    # Helper: GPU total/free memory (MB)
    def _gpu_total_free_mb() -> Tuple[int, int]:
        try:
            out = subprocess.run([
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                first = out.stdout.splitlines()[0].split(",")
                total = int(first[0].strip())
                free = int(first[1].strip())
                return total, free
        except Exception:
            pass
        return 0, 0  # No GPU or not available

    # Budgets (CPU cores, RAM MB, GPU MB)
    cpu_cores_total = mp.cpu_count()
    cpu_cores_budget = max(1.0, float(cpu_cores_total - 2))  # keep 2 for OS/IDE
    mem_avail_mb = psutil.virtual_memory().available / (1024**2)
    mem_budget_mb = max(1024.0, mem_avail_mb * 0.7)  # use 70% of available
    gpu_total_mb, gpu_free_mb = _gpu_total_free_mb()
    gpu_budget_mb = max(0.0, gpu_free_mb * 0.75)  # keep headroom on GPU

    # Worker pool size: allow many slots, scheduling will enforce budgets
    if args.workers:
        num_workers = min(args.workers, len(tests))
    else:
        num_workers = min(cpu_cores_total, len(tests))

    print("="*70)
    print("PARALLEL TEST RUNNER")
    print("="*70)
    print(f"Running {len(tests)} tests with pool size {num_workers}")
    print(f"Budgets -> CPU cores: {cpu_cores_budget:.1f}, RAM: {mem_budget_mb:.0f} MB, GPU: {gpu_budget_mb:.0f} MB")
    print(f"Output buffered per-test, progress shown on completion\n")
    
    # Load test metrics
    test_metrics = TestMetrics()
    
    # Prepare pending queue with estimates and hybrid scheduling score
    # Sort strategy: Longest Job First within priority tiers to minimize makespan and avoid stragglers
    enqueue_time = time.time()
    pending: List[Tuple[float, str, int, Dict, int]] = []  # (sort_key, test_id, tier, estimate, timeout_sec)
    for test_id, tier in tests:
        cfg = test_configs.get(test_id)
        est = test_metrics.get_estimate(test_id, cfg)
        # Timeout policy: NEVER less than --timeout (default 7200s)
        # If an estimate exists, allow larger than default; but do not go below default
        est_rt = float(est.get("runtime_sec", 0.0))
        timeout_mult = float(est.get("timeout_multiplier", 3.0))
        estimated_timeout = int(est_rt * timeout_mult) if est_rt > 0 else int(args.timeout)
        timeout_sec = max(int(args.timeout), estimated_timeout)
        prio = test_metrics.get_priority(test_id)
        
        # Add metadata for metrics tracking
        est["enqueued_at"] = enqueue_time
        est["fit_denied_cpu"] = 0
        est["fit_denied_mem"] = 0
        est["fit_denied_gpu"] = 0
        
        # Compute hybrid scheduling score:
        # 1) Group by priority tier (failed/never-run=0, high-prio=1, normal=2)
        if prio >= 90:
            priority_tier = 0  # Failed or never run - absolute priority
        elif prio >= 70:
            priority_tier = 1  # High priority (REL, UNIF)
        else:
            priority_tier = 2  # Normal priority
        
        # 2) Within tier: schedule longer tests first to avoid end-of-run stragglers
        # Negative runtime so longer jobs sort first
        runtime_key = -est_rt
        
        # 3) Tie-breaker: resource footprint (higher resource tests first for better packing)
        cpu_need = float(est.get("cpu_cores_needed", 1.0))
        mem_need = float(est.get("memory_mb", 500.0))
        gpu_need = float(est.get("gpu_memory_mb", 0.0))
        resource_footprint = -(cpu_need + mem_need/1000.0 + gpu_need/1000.0)
        
        sort_key = (priority_tier, runtime_key, resource_footprint, test_id)  # test_id for determinism
        pending.append((sort_key, test_id, tier, est, timeout_sec))
    
    pending.sort(key=lambda x: x[0])  # Sort by composite key

    # Track running usage and results
    running_cpu = 0.0
    running_mem = 0.0
    running_gpu = 0.0
    running: List[Dict] = []  # dict with keys: id,tier,est,async
    all_results: List[Tuple[str, int, Dict]] = []

    start_time = time.time()
    last_progress_time = start_time
    completed_count = 0
    total_count = len(tests)
    
    # Precompute total estimated work at start (sum of all test's last runtime)
    # This is our initial ETA baseline - we'll subtract completed work as tests finish
    initial_work_sec = sum(float(est.get("runtime_sec", 300.0)) for _, _, _, est, _ in pending)
    remaining_work_sec = initial_work_sec  # Decreases as tests complete
    
    stall_since: Optional[float] = None
    with mp.Pool(processes=num_workers) as pool:
        while pending or running:
            # Show live progress every 10 seconds
            now = time.time()
            if now - last_progress_time >= 10.0 and (running or completed_count > 0):
                # Simple ETA: sum of estimated runtimes for pending + running tests
                # This naturally decreases as tests complete (we subtract their estimate when done)
                pending_est_sec = sum(float(est.get("runtime_sec", 300.0)) for _, _, _, est, _ in pending)
                running_est_sec = sum(float(item["est"].get("runtime_sec", 300.0)) for item in running)
                eta_sec = pending_est_sec + running_est_sec
                eta_min = max(0.0, eta_sec / 60.0)
                
                # Show active tests
                active_tests = [item["id"] for item in running[:3]]  # Show first 3
                active_str = ", ".join(active_tests)
                if len(running) > 3:
                    active_str += f" +{len(running)-3} more"
                
                print(f"[Progress] {completed_count}/{total_count} complete | Active: {active_str} | Pending: {len(pending)} | ETA: {eta_min:.1f} min", flush=True)
                last_progress_time = now
            
            # Try to start as many as fit within budgets
            started_any = False
            i = 0
            while i < len(pending):
                sort_key, tid, tier, est, timeout_sec = pending[i]
                need_cpu_raw = float(est.get("cpu_cores_needed", 1.0))
                # Temporary sanity cap to avoid pathological CPU estimates; record both for diagnostics
                cpu_cap = 2.0
                need_cpu = min(need_cpu_raw, cpu_cap)
                need_mem = float(est.get("memory_mb", 500.0))
                need_gpu = float(est.get("gpu_memory_mb", 0.0))

                fits_cpu = (running_cpu + need_cpu) <= cpu_cores_budget
                fits_mem = (running_mem + need_mem) <= mem_budget_mb
                fits_gpu = True if gpu_budget_mb <= 0.0 else (running_gpu + need_gpu) <= gpu_budget_mb

                # Also ensure we don't exceed pool parallelism with too many scheduled
                if fits_cpu and fits_mem and fits_gpu and len(running) < num_workers:
                    ar = pool.apply_async(run_single_test, args=(tid, tier, timeout_sec))
                    running.append({
                        "id": tid,
                        "tier": tier,
                        "est": est,
                        "timeout": timeout_sec,
                        "async": ar,
                        "need_cpu": need_cpu,
                        "need_cpu_raw": need_cpu_raw,
                        "cpu_cap": cpu_cap,
                        "need_mem": need_mem,
                        "need_gpu": need_gpu,
                        "started_at": time.time(),
                        "enqueued_at": est.get("enqueued_at", time.time()),
                        "concurrency_at_start": len(running),
                        "was_forced": False,
                        "fit_denied_cpu": est.get("fit_denied_cpu", 0),
                        "fit_denied_mem": est.get("fit_denied_mem", 0),
                        "fit_denied_gpu": est.get("fit_denied_gpu", 0)
                    })
                    running_cpu += need_cpu
                    running_mem += need_mem
                    running_gpu += need_gpu
                    # Remove from pending without skipping next item (since we removed current index)
                    pending.pop(i)
                    started_any = True
                else:
                    # Track why this test didn't fit (for metrics)
                    if not fits_cpu:
                        est["fit_denied_cpu"] = est.get("fit_denied_cpu", 0) + 1
                    if not fits_mem:
                        est["fit_denied_mem"] = est.get("fit_denied_mem", 0) + 1
                    if not fits_gpu:
                        est["fit_denied_gpu"] = est.get("fit_denied_gpu", 0) + 1
                    i += 1

            # If nothing could be started, wait for any completion or force-start if stalled
            if not started_any:
                # If nothing is running but we still have pending tests, we may be stuck due to over-estimated resources.
                if not running and pending:
                    if stall_since is None:
                        stall_since = time.time()
                        # Log why the next test can't fit to aid debugging
                        sort_key, tid, tier, est, timeout_sec = pending[0]
                        need_cpu_check = min(float(est.get("cpu_cores_needed", 1.0)), 2.0)
                        need_mem_check = float(est.get("memory_mb", 500.0))
                        need_gpu_check = float(est.get("gpu_memory_mb", 0.0))
                        fits_cpu_check = need_cpu_check <= cpu_cores_budget
                        fits_mem_check = need_mem_check <= mem_budget_mb
                        fits_gpu_check = True if gpu_budget_mb <= 0.0 else need_gpu_check <= gpu_budget_mb
                        reasons = []
                        if not fits_cpu_check:
                            reasons.append(f"CPU:{need_cpu_check:.1f}>{cpu_cores_budget:.1f}")
                        if not fits_mem_check:
                            reasons.append(f"RAM:{need_mem_check:.0f}MB>{mem_budget_mb:.0f}MB")
                        if not fits_gpu_check:
                            reasons.append(f"GPU:{need_gpu_check:.0f}MB>{gpu_budget_mb:.0f}MB")
                        if reasons:
                            print(f"[Scheduler] {tid} waiting (exceeds budget: {', '.join(reasons)})", flush=True)
                    # If stalled for > 10 seconds, force start the next pending test ignoring budget constraints
                    if time.time() - stall_since > 10.0:
                        sort_key, tid, tier, est, timeout_sec = pending.pop(0)
                        # Mirror normal scheduling caps/needs for consistency
                        need_cpu_raw = float(est.get("cpu_cores_needed", 1.0))
                        cpu_cap = 2.0
                        need_cpu = min(need_cpu_raw, cpu_cap)
                        need_mem = float(est.get("memory_mb", 500.0))
                        need_gpu = float(est.get("gpu_memory_mb", 0.0))
                        print(
                            f"[Scheduler] Starvation guard: no tasks running for >10s. Forcing start of {tid} "
                            f"(cpu={need_cpu:.1f}, mem={need_mem:.0f}MB, gpu={need_gpu:.0f}MB)",
                            flush=True
                        )
                        ar = pool.apply_async(run_single_test, args=(tid, tier, timeout_sec))
                        running.append({
                            "id": tid,
                            "tier": tier,
                            "est": est,
                            "timeout": timeout_sec,
                            "async": ar,
                            "need_cpu": need_cpu,
                            "need_cpu_raw": need_cpu_raw,
                            "cpu_cap": cpu_cap,
                            "need_mem": need_mem,
                            "need_gpu": need_gpu,
                            "started_at": time.time(),
                            "enqueued_at": est.get("enqueued_at", time.time()),
                            "concurrency_at_start": 0,
                            "was_forced": True,
                            "fit_denied_cpu": est.get("fit_denied_cpu", 0),
                            "fit_denied_mem": est.get("fit_denied_mem", 0),
                            "fit_denied_gpu": est.get("fit_denied_gpu", 0)
                        })
                        running_cpu += need_cpu
                        running_mem += need_mem
                        running_gpu += need_gpu
                        stall_since = None
                        continue
                else:
                    stall_since = None

                # Poll for completions
                completed_idx: Optional[int] = None
                for idx, item in enumerate(running):
                    if item["async"].ready():
                        completed_idx = idx
                        break
                if completed_idx is None:
                    time.sleep(0.2)
                    continue

                # Collect result and free resources
                item = running.pop(completed_idx)
                res: Tuple[str, int, Dict] = item["async"].get()
                test_id_done, tier_done, metrics = res
                running_cpu -= item["need_cpu"]
                running_mem -= item["need_mem"]
                running_gpu -= item["need_gpu"]
                
                # Retry logic for timeouts and exceptions
                exit_code = metrics.get("exit_code", 1)
                should_retry = False
                
                # Retry if timeout (-2) or exception (-3), but only if not already a retry
                if exit_code in [-2, -3] and not item["est"].get("is_retry", False):
                    # Check if this test has a recent history of consistent failures
                    history = test_metrics.get_all_runs(test_id_done)
                    if history and len(history) >= 3:
                        # If last 3 runs all failed with same error, don't retry (consistent failure)
                        recent_exits = [h.get("exit_code", 0) for h in history[-3:]]
                        if all(code in [-2, -3] for code in recent_exits):
                            should_retry = False  # Consistent failure pattern
                        else:
                            should_retry = True
                    else:
                        # No consistent failure pattern, allow retry
                        should_retry = True
                
                if should_retry:
                    # Re-add to pending with retry marker and highest priority (priority=0)
                    print(f"  ⟳ RETRY {test_id_done} (exit_code={exit_code})", flush=True)
                    retry_est = item["est"].copy()
                    retry_est["is_retry"] = True
                    # Pending tuple shape: (sort_key, test_id, tier, estimate, timeout_sec)
                    # Retry gets absolute priority: sort_key tuple with priority_tier=0, max runtime, max footprint
                    retry_sort_key = (-1, -9999999.0, -9999999.0, test_id_done)  # Sorts before everything
                    pending.insert(0, (retry_sort_key, test_id_done, tier_done, retry_est, item["timeout"]))  # Insert at front
                else:
                    # No retry, record result with scheduler metrics
                    # Add scheduler metrics to the result
                    metrics["queue_wait_sec"] = item["started_at"] - item["enqueued_at"]
                    metrics["concurrency_at_start"] = item["concurrency_at_start"]
                    metrics["was_forced_by_starvation"] = item["was_forced"]
                    metrics["fit_denied_cpu"] = item["fit_denied_cpu"]
                    metrics["fit_denied_mem"] = item["fit_denied_mem"]
                    metrics["fit_denied_gpu"] = item["fit_denied_gpu"]
                    metrics["cpu_cores_needed_raw"] = item.get("need_cpu_raw", metrics.get("cpu_cores_needed_raw"))
                    metrics["cpu_cores_used_for_budget"] = item.get("need_cpu", metrics.get("cpu_cores_used_for_budget"))
                    metrics["cpu_cap_applied"] = (item.get("need_cpu_raw") or 0) > (item.get("need_cpu") or 0)
                    # Calculate estimate error ratio
                    est_runtime = float(item["est"].get("runtime_sec", 1.0))
                    actual_runtime = metrics.get("runtime_sec", 0.0)
                    if actual_runtime > 0:
                        metrics["estimate_error_ratio"] = est_runtime / actual_runtime
                    else:
                        metrics["estimate_error_ratio"] = 1.0
                    
                    all_results.append(res)
                    completed_count += 1
                    status = "✓ PASS" if exit_code == 0 else "✗ FAIL"
                    runtime = metrics.get("runtime_sec", 0.0)
                    print(f"  {status} {test_id_done} ({runtime:.1f}s)", flush=True)
    
    total_time = time.time() - start_time
    
    # Record metrics and count pass/fail
    passed = 0
    failed = 0
    failed_tests = []
    
    for test_id, tier, metrics in all_results:
        # Always record to database (even timeouts/errors provide timing data)
        test_metrics.record_run(test_id, metrics)
        
        # Count results
        if metrics["exit_code"] == 0:
            passed += 1
        else:
            failed += 1
            failed_tests.append((test_id, metrics))
    
    # Update MASTER_TEST_STATUS.csv
    update_master_test_status()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    # Show failed test details
    if failed > 0:
        print("\nFailed tests:")
        for test_id, metrics in failed_tests:
            runtime = metrics.get("runtime_sec", 0.0)
            error = metrics.get("error", "Unknown error")
            print(f"  ✗ {test_id} ({runtime:.1f}s) - {error}")
        
        # Offer to show output
        print("\nTo see full output of a failed test, check the logs or run individually:")
        for test_id, _ in failed_tests:
            print(f"  python run_tier*_*.py --test {test_id}")
        
        sys.exit(1)
    else:
        speedup = sum(m["runtime_sec"] for _, _, m in all_results if "runtime_sec" in m) / total_time
        print(f"\n✓ All tests passed!")
        print(f"Speedup: {speedup:.1f}x vs sequential")
        sys.exit(0)


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()
