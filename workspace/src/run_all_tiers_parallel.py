#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
Run all tiers with GPU, with the option to parallelize INDIVIDUAL TESTS.

This script supports two modes:
- tiers  (default prior behavior): run Tier 1-5 as separate processes in parallel
- tests  (new default): enumerate every test across all tiers and run them in parallel with per-test timeouts

Both modes:
- Force UTF-8-safe output capture (no Windows cp1252 crashes on Unicode like ω, χ)
- Prefer GPU execution (configs already enable it)
- Summarize results and failures succinctly
"""

import subprocess
import time
import sys
import json
import os
import threading
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime

WORKSPACE = Path(__file__).parent.parent  # Points to c:\LFM\workspace

TIER_CONFIGS = {
    1: {
        "script": "src/run_tier1_relativistic.py",
        "name": "Relativistic",
        "config": "config/config_tier1_relativistic.json"
    },
    2: {
        "script": "src/run_tier2_gravityanalogue.py",
        "name": "Gravity Analogue",
        "config": "config/config_tier2_gravityanalogue.json"
    },
    3: {
        "script": "src/run_tier3_energy.py",
        "name": "Energy Conservation",
        "config": "config/config_tier3_energy.json"
    },
    4: {
        "script": "src/run_tier4_quantization.py",
        "name": "Quantization",
        "config": "config/config_tier4_quantization.json"
    },
    5: {
        "script": "src/run_tier5_electromagnetic.py",
        "name": "Electromagnetic",
        "config": "config/config_tier5_electromagnetic.json"
    }
}


def run_tier(tier_num: int) -> dict:
    """Run a single tier test suite."""
    config = TIER_CONFIGS[tier_num]
    
    print(f"[Tier {tier_num}] Starting: {config['name']}")
    start_time = time.time()
    
    cmd = [
        sys.executable,
        str(WORKSPACE / config["script"]),
        "--config", str(WORKSPACE / config["config"])
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600  # 1 hour timeout per tier
        )
        
        runtime = time.time() - start_time
        success = result.returncode == 0
        
        return {
            "tier": tier_num,
            "name": config["name"],
            "success": success,
            "runtime": runtime,
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
            "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
        }
    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        return {
            "tier": tier_num,
            "name": config["name"],
            "success": False,
            "runtime": runtime,
            "returncode": -1,
            "stdout": "",
            "stderr": "Test suite timed out after 1 hour"
        }
    except Exception as e:
        runtime = time.time() - start_time
        return {
            "tier": tier_num,
            "name": config["name"],
            "success": False,
            "runtime": runtime,
            "returncode": -999,
            "stdout": "",
            "stderr": f"Exception: {type(e).__name__}: {e}"
        }


def print_banner(mode: str, max_workers: int, timeout_s: int):
    """Print startup banner."""
    print("=" * 80)
    print("LFM PARALLEL TEST RUNNER")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  [*] GPU Acceleration: Enabled (use_gpu: true)")
    print("  [*] Fused CUDA Kernels: Enabled (use_fused_cuda: true)")
    print("  [*] GPU Diagnostics: Enabled (automatic)")
    print("  [*] Test Caching: Enabled (automatic)")
    if mode == "tiers":
        print("  [*] Parallel Execution Mode: Tiers (5 processes)")
    else:
        print(f"  [*] Parallel Execution Mode: Tests ({max_workers} workers, {timeout_s}s/test timeout)")
    print()
    print("Expected Performance:")
    print("  - Fused CUDA kernels: 2-3x faster lattice evolution")
    print("  - GPU diagnostics: 10-50x faster energy calculations")
    print("  - Test caching: Skip unchanged tests (instant)")
    print()
    print("Targets:")
    if mode == "tiers":
        for tier_num, config in TIER_CONFIGS.items():
            print(f"  Tier {tier_num}: {config['name']}")
    else:
        print("  All tests across Tiers 1-5 (enumerated from configs)")
    print()
    print("=" * 80)
    print()


def print_progress(completed: int, total: int, result: dict):
    """Print progress update."""
    status = "[PASS]" if result["success"] else "[FAIL]"
    runtime_str = f"{result['runtime']:.1f}s"
    print(f"[{completed}/{total}] Tier {result['tier']} ({result['name']}): {status} in {runtime_str}")


def print_summary(results: list):
    """Print final summary."""
    print()
    print("=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print()
    
    # Sort by tier number
    results = sorted(results, key=lambda r: r["tier"])
    
    total_runtime = sum(r["runtime"] for r in results)
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed
    
    print(f"Total Runtime: {total_runtime:.1f}s ({total_runtime/60:.1f} minutes)")
    print(f"Tests: {passed} passed, {failed} failed out of {len(results)} tiers")
    print()
    
    print("Per-Tier Results:")
    print("-" * 80)
    for r in results:
        status = "[PASS]" if r["success"] else "[FAIL]"
        print(f"  Tier {r['tier']} ({r['name']:20s}): {status:8s}  {r['runtime']:6.1f}s  (exit: {r['returncode']})")
    print()
    
    # Show failures in detail
    failures = [r for r in results if not r["success"]]
    if failures:
        print("FAILED TIERS:")
        print("-" * 80)
        for r in failures:
            print(f"\nTier {r['tier']} ({r['name']}):")
            if r["stderr"]:
                print(f"  Error: {r['stderr'][:500]}")
            if r["returncode"] == -1:
                print("  Reason: Timeout (exceeded 1 hour)")
            elif r["returncode"] == -999:
                print("  Reason: Execution error")
        print()
    
    print("=" * 80)
    
    # Return exit code
    return 0 if failed == 0 else 1


# ----------------------------- Test-level mode -----------------------------

def _read_tests_from_config(config_path: Path, tier_num: int) -> list[dict]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    tier_name = TIER_CONFIGS[tier_num]["name"]
    # Resolve output_dir from run_settings first (canonical), then fallback
    run_settings = cfg.get("run_settings", {}) if isinstance(cfg, dict) else {}
    output_dir = (
        run_settings.get("output_dir")
        or cfg.get("output_dir")
        or TIER_CONFIGS[tier_num]["name"].replace(" ", "")
    )

    items = []
    # Prefer 'tests', fallback to 'variants'
    test_list = cfg.get("tests") or cfg.get("variants") or []
    params = cfg.get("parameters", {})

    def _estimate_from_entry(entry: dict) -> float:
        # Simple heuristic: steps * (N or grid_points)
        steps = entry.get("steps") or entry.get("steps_quick") or params.get("steps") or params.get("steps_quick") or 1000
        N = entry.get("N") or entry.get("grid_points") or params.get("N") or params.get("grid_points") or 256
        try:
            if isinstance(N, (list, tuple)):
                # 3D grid estimate
                vol = 1
                for d in N:
                    vol *= int(d)
                scale = vol ** (1/3)
            else:
                scale = int(N)
        except Exception:
            scale = 256
        return float(steps) * float(max(scale, 1))

    for t in test_list:
        # Normalize id field
        test_id = t.get("test_id") or t.get("id") or "?"
        if t.get("skip", False) is True:
            continue

        # Determine last runtime if available
        out_dir = WORKSPACE / output_dir / test_id
        last_runtime = None
        try:
            s = out_dir / "summary.json"
            if s.exists():
                with s.open("r", encoding="utf-8") as sf:
                    sj = json.load(sf)
                    last_runtime = float(sj.get("runtime_sec") or sj.get("runtime") or 0.0)
                    if last_runtime <= 0:
                        last_runtime = None
        except Exception:
            last_runtime = None
        est = last_runtime if last_runtime is not None else _estimate_from_entry(t)

        items.append({
            "tier": tier_num,
            "tier_name": tier_name,
            "test_id": test_id,
            "config_path": str(config_path),
            "script": str(WORKSPACE / TIER_CONFIGS[tier_num]["script"]),
            "output_dir": str(WORKSPACE / output_dir / test_id),
            "estimate": est
        })
    return items


def enumerate_all_tests() -> list[dict]:
    all_items: list[dict] = []
    for tier_num, meta in TIER_CONFIGS.items():
        cfg_path = WORKSPACE / meta["config"]
        try:
            all_items.extend(_read_tests_from_config(cfg_path, tier_num))
        except Exception as e:
            print(f"[WARN] Failed to read tests for Tier {tier_num}: {type(e).__name__}: {e}")
    return all_items


def run_single_test(item: dict, timeout_s: int) -> dict:
    """Run a single test via the tier runner's --test, with timeout and UTF-8 capture."""
    test_id = item["test_id"]
    tier = item["tier"]
    tier_name = item["tier_name"]

    # Announce start so users see activity immediately (include rough estimate)
    start_stamp = time.strftime("%H:%M:%S")
    est = item.get("estimate")
    est_note = f" (est~{int(est)} units)" if isinstance(est, (int, float)) and est > 0 else ""
    print(f">> [{start_stamp}] START {tier_name}::{test_id}{est_note}")

    cmd = [
        sys.executable,
        item["script"],
        "--config", item["config_path"],
        "--test", test_id,
    ]

    env = os.environ.copy()
    # Force UTF-8 in child so Unicode from tests doesn't crash decoding
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # Ensure output directory exists and set log file path to avoid pipe deadlocks
    out_dir = Path(item.get("output_dir", WORKSPACE / "results" / tier_name / test_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "runner.log"

    start = time.time()
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as logfh:
            result = subprocess.run(
                cmd,
                cwd=str(WORKSPACE),
                stdout=logfh,
                stderr=logfh,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                env=env,
            )
        runtime = time.time() - start
        tail = ""
        if result.returncode != 0:
            try:
                # Read last lines on failure to include snippet
                with log_path.open("r", encoding="utf-8", errors="replace") as rf:
                    lines = rf.readlines()
                    tail = "".join(lines[-40:])
            except Exception:
                tail = ""
        return {
            "tier": tier,
            "tier_name": tier_name,
            "test_id": test_id,
            "success": result.returncode == 0,
            "runtime": runtime,
            "returncode": result.returncode,
            "stdout": "",
            "stderr": tail,
        }
    except subprocess.TimeoutExpired:
        runtime = time.time() - start
        return {
            "tier": tier,
            "tier_name": tier_name,
            "test_id": test_id,
            "success": False,
            "runtime": runtime,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Test timed out after {timeout_s}s",
        }
    except Exception as e:
        runtime = time.time() - start
        return {
            "tier": tier,
            "tier_name": tier_name,
            "test_id": test_id,
            "success": False,
            "runtime": runtime,
            "returncode": -999,
            "stdout": "",
            "stderr": f"Exception: {type(e).__name__}: {e}",
        }


def print_test_progress(done: int, total: int, r: dict):
    status = "[PASS]" if r["success"] else "[FAIL]"
    print(f"[{done}/{total}] {r['tier_name']}::{r['test_id']}: {status} in {r['runtime']:.1f}s")


def print_test_summary(results: list) -> int:
    print()
    print("=" * 80)
    print("EXECUTION SUMMARY (per-test mode)")
    print("=" * 80)
    print()

    total_runtime = sum(r["runtime"] for r in results)
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed

    print(f"Total Runtime: {total_runtime:.1f}s ({total_runtime/60:.1f} minutes)")
    print(f"Tests: {passed} passed, {failed} failed out of {len(results)} tests")
    print()

    print("Failures:")
    print("-" * 80)
    for r in results:
        if not r["success"]:
            print(f"  {r['tier_name']}::{r['test_id']}  (exit {r['returncode']})  {r['runtime']:.1f}s")
            if r["stderr"]:
                print(f"    {r['stderr'][:300]}")
    print()
    print("=" * 80)
    return 0 if failed == 0 else 1


def write_parallel_summary(results: list, mode: str, dest: Path) -> None:
    """Write a machine-readable summary json for the parallel run."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "mode": mode,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if mode == "tests":
        total = len(results)
        passed = sum(1 for r in results if r.get("success"))
        by_tier: dict = {}
        for r in results:
            tn = r.get("tier_name", "?")
            d = by_tier.setdefault(tn, {"total": 0, "passed": 0, "failed": 0, "tests": []})
            d["total"] += 1
            if r.get("success"):
                d["passed"] += 1
            else:
                d["failed"] += 1
            d["tests"].append({
                "test_id": r.get("test_id"),
                "success": r.get("success"),
                "runtime_sec": r.get("runtime"),
                "exit_code": r.get("returncode"),
            })
        payload.update({
            "totals": {"total": total, "passed": passed, "failed": total - passed},
            "by_tier": by_tier,
        })
    else:
        payload.update({
            "tiers": [
                {
                    "tier": r.get("tier"),
                    "name": r.get("name"),
                    "success": r.get("success"),
                    "runtime_sec": r.get("runtime"),
                    "exit_code": r.get("returncode"),
                } for r in results
            ]
        })
    with dest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def heartbeat(futures: list, total: int, stop_event: threading.Event, interval: int = 15):
    """Periodic heartbeat showing accurate Active/Queued/Done counts."""
    tick = 0
    while not stop_event.wait(interval):
        tick += 1
        done = 0
        active = 0
        queued = 0
        for f in futures:
            if f.done():
                done += 1
            elif getattr(f, "running", None) and f.running():
                active += 1
            else:
                queued += 1
        ts = time.strftime("%H:%M:%S")
        print(f"[heartbeat {tick}] {ts}  done: {done}/{total} | active: {active} | queued: {queued}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LFM Parallel Test Runner")
    parser.add_argument("--mode", choices=["tiers", "tests"], default="tests",
                        help="Run full tiers in parallel (tiers) or individual tests in parallel (tests)")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Maximum parallel workers in test mode")
    parser.add_argument("--test-timeout", type=int, default=900,
                        help="Per-test timeout in seconds in test mode")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N tests discovered (tests mode only), for smoke runs")
    parser.add_argument("--only-missing", action="store_true",
                        help="In tests mode, run only tests that have no prior summary.json (not yet executed)")
    parser.add_argument("--only-failed", action="store_true",
                        help="In tests mode, run only tests that previously produced summary.json with a FAIL status")
    args = parser.parse_args()

    # Friendly banner
    print_banner(args.mode, args.max_workers, args.test_timeout)

    start_time = time.time()

    if args.mode == "tiers":
        # Run all tiers in parallel (previous behavior)
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(run_tier, tier_num): tier_num for tier_num in TIER_CONFIGS.keys()}
            results = []
            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                tier_num = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    print_progress(completed, total, result)
                except Exception as e:
                    print(f"[ERROR] Tier {tier_num} raised unexpected exception: {e}")
                    results.append({
                        "tier": tier_num,
                        "name": TIER_CONFIGS[tier_num]["name"],
                        "success": False,
                        "runtime": 0,
                        "returncode": -999,
                        "stdout": "",
                        "stderr": f"Unexpected exception: {type(e).__name__}: {e}"
                    })
                    completed += 1
        exit_code = print_summary(results)
    else:
        # Enumerate and run tests in parallel with per-test timeout
        items = enumerate_all_tests()
        if args.only_missing and args.only_failed:
            print("[WARN] Both --only-missing and --only-failed specified; prioritizing --only-failed.")
        if args.only_failed:
            # Keep only tests that have a prior summary.json with FAILED status
            filtered = []
            for it in items:
                try:
                    out_dir = Path(it.get("output_dir")) if it.get("output_dir") else None
                    if out_dir is None:
                        continue
                    summary = out_dir / "summary.json"
                    if not summary.exists():
                        continue
                    with summary.open("r", encoding="utf-8", errors="replace") as sf:
                        sj = json.load(sf)
                    # Determine status (ALIGN with utils.lfm_results.update_master_test_status)
                    status = None
                    if sj.get("skipped") is True:
                        status = "SKIP"
                    elif "status" in sj:
                        status = str(sj["status"]) if sj["status"] is not None else "UNKNOWN"
                    elif "passed" in sj:
                        pv = sj["passed"]
                        if isinstance(pv, bool):
                            status = "PASS" if pv else "FAIL"
                        else:
                            status = "UNKNOWN"
                    else:
                        status = "UNKNOWN"
                    su = str(status).upper()
                    if su in ["FAILED", "FAIL", "FALSE"]:
                        filtered.append(it)
                except Exception:
                    # On parse errors, skip (treat as not failed for this filter)
                    continue
            items = filtered
        elif args.only_missing:
            # Keep only tests that do not have a prior summary.json in their output directory
            filtered = []
            for it in items:
                try:
                    out_dir = Path(it.get("output_dir")) if it.get("output_dir") else None
                    if out_dir is None:
                        filtered.append(it)
                        continue
                    summary = out_dir / "summary.json"
                    if not summary.exists():
                        filtered.append(it)
                except Exception:
                    # On any error determining status, keep the test to be safe
                    filtered.append(it)
            items = filtered
        # Schedule longest-first using past runtimes when available
        items.sort(key=lambda it: it.get("estimate", 0.0), reverse=True)
        if args.limit is not None:
            try:
                limit = max(0, int(args.limit))
                items = items[:limit]
            except Exception:
                pass
        if args.only_failed:
            print(f"Discovered {len(items)} failed tests across tiers 1-5")
        elif args.only_missing:
            print(f"Discovered {len(items)} missing tests across tiers 1-5")
        else:
            print(f"Discovered {len(items)} tests across tiers 1-5")
        results = []
        done = 0
        total = len(items)

        # Use threads to manage subprocess.run calls efficiently
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(run_single_test, it, args.test_timeout) for it in items]

            # Start heartbeat thread
            stop_event = threading.Event()
            hb_thread = threading.Thread(target=heartbeat, args=(futures, total, stop_event, 15), daemon=True)
            hb_thread.start()

            try:
                for future in as_completed(futures):
                    r = future.result()
                    results.append(r)
                    done += 1
                    print_test_progress(done, total, r)
            finally:
                stop_event.set()
                hb_thread.join(timeout=2)
        exit_code = print_test_summary(results)
        # Final banner + summary file
        print("\n=== ALL TESTS COMPLETE ===", flush=True)
        try:
            write_parallel_summary(results, args.mode, WORKSPACE / "results" / "parallel_run_summary.json")
            print("Summary written to results/parallel_run_summary.json", flush=True)
        except Exception as e:
            print(f"[WARN] Could not write summary file: {type(e).__name__}: {e}", flush=True)

    total_time = time.time() - start_time
    print(f"Parallel execution completed in {total_time:.1f}s ({total_time/60:.1f} minutes)", flush=True)
    print()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
