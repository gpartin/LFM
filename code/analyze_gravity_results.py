#!/usr/bin/env python3
"""
Quick diagnostic analyzer for Tier-2 Gravity Analogue runs.
Reads summary.json files and gives quick interpretation of failures.
Run this from inside /code (next to run_tier2_gravityanalogue.py).
"""

import json, math
from pathlib import Path

# Locate results folder relative to /code
ROOT = Path(__file__).resolve().parent.parent / "results" / "GravityAnalogue"
if not ROOT.exists():
    print(f"❌ No GravityAnalogue results found at {ROOT}")
    exit(1)

def safe_load_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def assess(corr, z):
    msg = []
    if corr is None or math.isnan(corr) or abs(corr) < 0.2:
        msg.append("χ–E correlation too low → field not deflecting (gradient too small or wrong sign).")
    elif corr < 0.9:
        msg.append("weak χ coupling → increase chi_grad or pulse_amp.")
    if z is None or math.isnan(z):
        msg.append("redshift NaN → instability or zero FFT signal.")
    elif abs(z) > 0.1:
        if z > 0:
            msg.append("large redshift (wave slowed too much) → maybe dt too high or χ² sign error.")
        else:
            msg.append("blueshift (χ acting inverted) → check sign of χ² term.")
    return "; ".join(msg) if msg else "✅ metrics within expected range."

rows = []
for summary_path in ROOT.glob("*/summary.json"):
    s = safe_load_json(summary_path)
    tid = s.get("id", summary_path.parent.name)
    corr = s.get("corr")
    z = s.get("redshift")
    passed = s.get("passed", False)
    rows.append((tid, corr, z, passed))

if not rows:
    print("❌ No summary.json files found under", ROOT)
    exit(1)

print("\n=== Tier-2 Gravity Analogue Diagnostics ===")
for tid, corr, z, passed in sorted(rows):
    status = "PASS ✅" if passed else "FAIL ❌"
    issue = assess(corr, z)
    print(f"{tid:>12} | corr={corr:+.3f} | z={z:+.3e} | {status} | {issue}")
