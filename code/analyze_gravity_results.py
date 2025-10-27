#!/usr/bin/env python3
"""
lfm_scan_results.py — Tier-2 diagnostics auditor (v1.0)

Scans C:/LFM/results/Gravity for all tests, reads:
 • debug_*.txt → extracts ω_lowChi, ω_highChi, z, env amplitudes
 • summary.json (if present)
Flags:
 - missing or unreadable files
 - huge |z| > 1, or ω ratio > 2.0
 - env_lowChi / env_highChi < 0.3 (energy imbalance)
"""

import re, json
from pathlib import Path

root = Path("C:/LFM/results/Gravity")
if not root.exists():
    print("❌ Directory not found:", root)
    raise SystemExit

print("=== LFM Gravity Diagnostics Scan ===")
for test_dir in root.glob("GRAV*"):
    dbg_files = list(test_dir.rglob("debug_*.txt"))
    if not dbg_files:
        print(f"⚠️  {test_dir.name}: no debug file found")
        continue

    dbg = dbg_files[-1]
    text = dbg.read_text(encoding="utf-8", errors="ignore")

    def grab(label):
        m = re.search(fr"{label}=([-\d\.eE\+]+)", text)
        return float(m.group(1)) if m else None

    w_lo = grab("omega_lowChi")
    w_hi = grab("omega_highChi")
    z = grab("z")
    env_lo = grab("env_at_lowChi")
    env_hi = grab("env_at_highChi")

    ratio = w_hi / (w_lo + 1e-30) if w_lo and w_hi else None
    energy_ratio = env_lo / (env_hi + 1e-30) if env_lo and env_hi else None

    flags = []
    if not all([w_lo, w_hi]): flags.append("ω missing")
    if z and abs(z) > 1: flags.append(f"|z|={z:.2f}")
    if ratio and ratio > 2.0: flags.append(f"ω_hi/lo={ratio:.2f}")
    if energy_ratio and energy_ratio < 0.3: flags.append(f"weak low probe ({energy_ratio:.2f})")

    if flags:
        print(f"❌ {test_dir.name}: {'; '.join(flags)}")
    else:
        print(f"✅ {test_dir.name}: ok  (z={z:.3g}, ω_hi/lo={ratio:.2f})")

print("=== Scan complete ===")
