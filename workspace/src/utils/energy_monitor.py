#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
energy_monitor.py — deterministic energy tracking for all LFM tiers
v1.1.0 (fast-buffered)
- Buffered CSV writes (flush every N records, default 10)
- CuPy→NumPy safe handling for chi when computing energy_total()
- Drop-in compatible with previous EnergyMonitor API
"""

import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

from utils.lfm_diagnostics import energy_total, to_numpy


class EnergyMonitor:
    """
    Deterministic energy drift tracker.

    Args:
        dt, dx, c, chi: physical parameters
        outdir (str): directory to place CSV/JSON outputs
        label (str): filename stem, e.g. "REL-01_serial"
        threshold (float): per-step drift warn threshold (not enforced here)
        flush_interval (int): how many records to buffer before writing

    Usage:
        mon = EnergyMonitor(dt, dx, c, chi, outdir="diagnostics", label="REL-01")
        mon.record(E, E_prev, step)
        mon.finalize()
    """
    def __init__(
        self,
        dt: float,
        dx: float,
        c: float,
        chi,
        outdir: str = "diagnostics",
        label: str = "unknown",
        threshold: float = 1e-6,
        flush_interval: int = 10,
        quiet_warnings: bool = False,
    ) -> None:
        self.dt, self.dx, self.c, self.chi = float(dt), float(dx), float(c), chi
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.label = label
        self.threshold = threshold
        self.quiet_warnings = quiet_warnings
        self.baseline: Optional[float] = None
        self.series: List[Tuple[int, float, float]] = []  # (step, energy, drift)
        self.buffer: List[List[str]] = []
        self.flush_interval = max(1, int(flush_interval))
        # Averaging window for baseline determination (can be overridden by caller)
        self._avg_window = 1
        self._pending_baseline: List[float] = []

        self.csv_path = self.outdir / f"energy_trace_{label}.csv"
        # Write header once
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "energy", "drift", "timestamp"])

    def _chi_as_numpy(self):
        """Return chi as a NumPy scalar/array (works for floats, NumPy, or CuPy)."""
        chi_safe = self.chi
        try:
            # lazy import so this file doesn't require CuPy when not installed
            import cupy as cp  # type: ignore
            if isinstance(chi_safe, cp.ndarray):
                chi_safe = chi_safe.get()
        except Exception:
            pass
        return chi_safe

    def _flush(self) -> None:
        if not self.buffer:
            return
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(self.buffer)
        self.buffer.clear()

    def record(self, E, E_prev, step: int) -> float:
        """Record energy and drift for this step; returns the drift value.

        Baseline averaging logic:
        - For very small baseline energies the relative drift metric can be artificially inflated by
          tiny absolute fluctuations. To mitigate this for broad-spectrum low-amplitude tests (e.g. REL-06),
          we allow an averaging window (self._avg_window) during which we accumulate energies and set the
          baseline to their mean once the window is filled.
        - If _avg_window == 1 (default) behavior is unchanged.
        """
        chi_np = self._chi_as_numpy()
        e_now = energy_total(to_numpy(E), to_numpy(E_prev), self.dt, self.dx, self.c, chi_np)

        # Initialize averaging buffer lazily
        if not hasattr(self, "_avg_window"):
            self._avg_window = 1
            self._pending_baseline = []

        if self.baseline is None:
            # Collect samples until we have _avg_window energies then set baseline to their mean
            self._pending_baseline.append(e_now)
            if len(self._pending_baseline) >= self._avg_window:
                self.baseline = float(np.mean(self._pending_baseline))
        else:
            # Baseline already established (averaged or single sample)
            pass

        # If baseline still None (window not filled), provisional baseline = current energy
        effective_baseline = self.baseline if self.baseline is not None else e_now

        denom = abs(effective_baseline) + 1e-30
        drift = (e_now - effective_baseline) / denom

        self.series.append((int(step), float(e_now), float(drift)))
        # buffered write
        self.buffer.append([int(step), f"{e_now:.10e}", f"{drift:.6e}", datetime.utcnow().isoformat() + "Z"])
        if len(self.buffer) >= self.flush_interval:
            self._flush()

        return drift

    def summary(self):
        if not self.series:
            return {"max_drift": None}
        drifts = [abs(d) for (_, _, d) in self.series]
        return {
            "max_drift": max(drifts),
            "mean_drift": float(np.mean(drifts)),
            "steps": len(drifts),
        }

    def finalize(self) -> None:
        """Flush pending rows and write summary JSON alongside CSV."""
        # ensure all records are written
        self._flush()
        # Use central write_json which handles NumPy scalars/arrays safely
        from utils.lfm_results import write_json
        summary_path = self.csv_path.with_suffix(".json")
        write_json(summary_path, self.summary())
