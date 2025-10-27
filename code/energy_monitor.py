#!/usr/bin/env python3
"""
energy_monitor.py — deterministic energy tracking for all LFM tiers
v1.0.0
Records compensated energy, drift, and alerts when thresholds are exceeded.
"""

import math, csv
import numpy as np
from pathlib import Path
from datetime import datetime
from lfm_diagnostics import energy_total, to_numpy

class EnergyMonitor:
    """
    Deterministic energy drift tracker.
    Usage:
        mon = EnergyMonitor(dt, dx, c, chi, outdir="diagnostics", label="REL-01")
        mon.record(E, E_prev, step)
    """

    def __init__(self, dt, dx, c, chi, outdir="diagnostics", label="unknown", threshold=1e-6):
        self.dt, self.dx, self.c, self.chi = dt, dx, c, chi
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.label = label
        self.threshold = threshold
        self.baseline = None
        self.series = []
        self.csv_path = self.outdir / f"energy_trace_{label}.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step","energy","drift","timestamp"])

    def record(self, E, E_prev, step):
        e_now = energy_total(to_numpy(E), to_numpy(E_prev), self.dt, self.dx, self.c, self.chi)
        if self.baseline is None:
            self.baseline = e_now
        drift = (e_now - self.baseline) / (abs(self.baseline) + 1e-30)
        self.series.append((step, e_now, drift))
        if abs(drift) > self.threshold:
            print(f"[EnergyMonitor:{self.label}] step {step:4d} ΔE/E₀={drift:+.3e}")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([step, f"{e_now:.10e}", f"{drift:.6e}", datetime.utcnow().isoformat() + "Z"])
        return drift

    def summary(self):
        if not self.series:
            return {"max_drift": None}
        drifts = [abs(d) for (_,_,d) in self.series]
        return {
            "max_drift": max(drifts),
            "mean_drift": float(np.mean(drifts)),
            "steps": len(drifts)
        }

    def finalize(self):
        """Write summary JSON alongside CSV."""
        import json
        summary_path = self.csv_path.with_suffix(".json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
