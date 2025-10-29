#!/usr/bin/env python3
"""
Regression validation for lfm_parallel.py
Compares serial vs threaded parallel diagnostics (2D & 3D)
to ensure numerical equivalence after backend changes.
"""

import pandas as pd
from pathlib import Path

# Results directory
RESULTS_DIR = Path(__file__).parent / "results" / "Tests" / "diagnostics"

def compare_pair(serial_file, parallel_file, label, tol=1e-3):
    if not Path(serial_file).exists() or not Path(parallel_file).exists():
        print(f"⚠️ Missing diagnostics for {label} ({serial_file}, {parallel_file})")
        return None

    s = pd.read_csv(serial_file)
    p = pd.read_csv(parallel_file)

    # Align by step count
    n = min(len(s), len(p))
    s, p = s.iloc[:n], p.iloc[:n]

    rel_energy = ((p["energy"] - s["energy"]).abs() / (s["energy"].abs() + 1e-30))
    rel_drift  = (p["drift"] - s["drift"]).abs()

    result = {
        "label": label,
        "max_rel_energy": rel_energy.max(),
        "mean_rel_energy": rel_energy.mean(),
        "max_drift_diff": rel_drift.max(),
        "mean_drift_diff": rel_drift.mean(),
        "pass": rel_energy.max() < tol and rel_drift.max() < tol
    }
    return result


def main():
    pairs = [
        (RESULTS_DIR / "diagnostics_2d_serial.csv", RESULTS_DIR / "diagnostics_2d_parallel.csv", "2D"),
        (RESULTS_DIR / "diagnostics_3d_serial.csv", RESULTS_DIR / "diagnostics_3d_parallel.csv", "3D"),
    ]
    results = []
    for s, p, lbl in pairs:
        r = compare_pair(str(s), str(p), lbl)
        if r:
            results.append(r)

    if not results:
        print(f"No diagnostics found in {RESULTS_DIR}. Run test_lfm_equation_parallel_all.py first.")
        return

    print("\n=== Parallel Regression Summary ===")
    print(f"{'Label':<6} {'MaxRelE':>10} {'MeanRelE':>10} {'MaxDriftΔ':>10} {'MeanDriftΔ':>10} {'PASS':>6}")
    for r in results:
        print(f"{r['label']:<6} {r['max_rel_energy']:10.3e} {r['mean_rel_energy']:10.3e} "
              f"{r['max_drift_diff']:10.3e} {r['mean_drift_diff']:10.3e} {str(r['pass']):>6}")
    print("\nTolerance: 1e-3 ⇒ PASS means serial ≈ parallel within 0.1%.")

if __name__ == "__main__":
    main()
