# -*- coding: utf-8 -*-
"""
Adaptive Scheduler (minimal implementation)
------------------------------------------
Runs tier tests in parallel with a simple FIFO scheduler and a cap on
concurrency. Intended as a lightweight in-repo replacement for the
external AdaptiveScheduler module referenced by run_parallel_suite.py.

Contract:
- class AdaptiveScheduler(verbose: bool = True)
- schedule_tests(test_list, max_concurrent=4) -> dict
  * test_list: List of (test_id: str, tier: int, config: Dict)
  * returns dict with keys:
      - total_runtime_sec: float
      - completed: int
      - failed: int
      - failed_tests: List[Dict{test_id, tier, exit_code}]

Notes:
- Spawns child processes that run the appropriate tier runner:
  Tier 1 → run_tier1_relativistic.py
  Tier 2 → run_tier2_gravityanalogue.py
  Tier 3 → run_tier3_energy.py
  Tier 4 → run_tier4_quantization.py
  Tier 5 → run_tier5_electromagnetic.py
- Propagates current environment (including LFM_PHYSICS_BACKEND) so
  --backend selection flows to child processes via env or CLI.
"""

from __future__ import annotations

import os
import time
import queue
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


_TIER_TO_RUNNER: Dict[int, str] = {
    1: "run_tier1_relativistic.py",
    2: "run_tier2_gravityanalogue.py",
    3: "run_tier3_energy.py",
    4: "run_tier4_quantization.py",
    5: "run_tier5_electromagnetic.py",
}


@dataclass
class _Task:
    test_id: str
    tier: int
    config: Dict


class AdaptiveScheduler:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def _build_command(self, task: _Task) -> List[str]:
        """
        Build the subprocess command to run a single test.
        Prefers invoking the runner script explicitly with --test.
        """
        src_dir = Path(__file__).resolve().parent
        runner_name = _TIER_TO_RUNNER.get(task.tier)
        if not runner_name:
            raise ValueError(f"Unsupported tier {task.tier} for test {task.test_id}")
        runner_path = src_dir / runner_name
        if not runner_path.exists():
            raise FileNotFoundError(f"Runner not found: {runner_path}")

        # Use current Python executable and run the tier runner with --test
        cmd = [
            os.environ.get("PYTHON_EXECUTABLE", os.sys.executable),
            str(runner_path),
            "--test", task.test_id,
        ]
        # If parallel suite was given a config path in task.config, we could pass it here.
        # The current load_test_configs API generally handles configs internally.
        return cmd

    def schedule_tests(self, test_list: List[Tuple[str, int, Dict]], max_concurrent: int = 4) -> Dict:
        t_start = time.perf_counter()

        # Build task queue
        q: "queue.Queue[_Task]" = queue.Queue()
        for test_id, tier, cfg in test_list:
            q.put(_Task(test_id=test_id, tier=tier, config=cfg))

        running: List[Tuple[_Task, subprocess.Popen]] = []
        failed_tests: List[Dict] = []
        completed = 0

        env = os.environ.copy()  # propagate LFM_PHYSICS_BACKEND, etc.

        def _start_next() -> bool:
            if q.empty():
                return False
            task = q.get_nowait()
            cmd = self._build_command(task)
            if self.verbose:
                print(f"→ START {task.test_id} (Tier {task.tier}) | cmd: {' '.join(cmd)}")
            # Use cwd=workspace root so default config paths like 'config/...' resolve
            # Script dir (src) is still added to sys.path automatically by Python
            src_dir = Path(__file__).resolve().parent
            workspace_root = src_dir.parent
            # Windows: creationflags to hide new console windows not necessary here
            proc = subprocess.Popen(cmd, cwd=str(workspace_root), env=env)
            running.append((task, proc))
            return True

        # Prime the pool
        while len(running) < max_concurrent and _start_next():
            pass

        # Poll until done
        while running:
            still_running: List[Tuple[_Task, subprocess.Popen]] = []
            for task, proc in running:
                code = proc.poll()
                if code is None:
                    still_running.append((task, proc))
                    continue
                # Completed
                completed += 1
                if code != 0:
                    failed_tests.append({
                        "test_id": task.test_id,
                        "tier": task.tier,
                        "exit_code": code,
                    })
                    if self.verbose:
                        print(f"✗ FAIL {task.test_id} (Tier {task.tier}) exit={code}")
                else:
                    if self.verbose:
                        print(f"✓ PASS {task.test_id} (Tier {task.tier})")
            running = still_running

            # Fill available slots
            while len(running) < max_concurrent and _start_next():
                pass

            # Avoid tight loop
            time.sleep(0.05)

        total_runtime = time.perf_counter() - t_start
        return {
            "total_runtime_sec": total_runtime,
            "completed": completed,
            "failed": len(failed_tests),
            "failed_tests": failed_tests,
        }
