# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]

TIER_CONFIGS = {
    1: {"name": "Relativistic", "config": WORKSPACE / "config" / "config_tier1_relativistic.json"},
    2: {"name": "Gravity Analogue", "config": WORKSPACE / "config" / "config_tier2_gravityanalogue.json"},
    3: {"name": "Energy Conservation", "config": WORKSPACE / "config" / "config_tier3_energy.json"},
    4: {"name": "Quantization", "config": WORKSPACE / "config" / "config_tier4_quantization.json"},
    5: {"name": "Electromagnetic", "config": WORKSPACE / "config" / "config_tier5_electromagnetic.json"},
}

def read_cfg(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def enumerate_tests():
    items = []
    for tier, meta in TIER_CONFIGS.items():
        try:
            cfg = read_cfg(meta['config'])
        except Exception as e:
            print(f"[WARN] Tier {tier} read failed: {e}")
            continue
        run_settings = cfg.get('run_settings', {}) if isinstance(cfg, dict) else {}
        out_root = run_settings.get('output_dir') or cfg.get('output_dir') or meta['name'].replace(' ', '')
        test_list = cfg.get('tests') or cfg.get('variants') or []
        for t in test_list:
            if t.get('skip', False):
                continue
            test_id = t.get('test_id') or t.get('id') or '?'
            out_dir = WORKSPACE / out_root / test_id
            items.append({
                'tier': tier,
                'tier_name': meta['name'],
                'test_id': test_id,
                'out_dir': str(out_dir),
                'summary': str(out_dir / 'summary.json')
            })
    return items

if __name__ == "__main__":
    items = enumerate_tests()
    per_tier = {}
    missing = []
    for it in items:
        summary = Path(it['summary'])
        is_missing = not summary.exists()
        if is_missing:
            missing.append(it)
        d = per_tier.setdefault(it['tier_name'], {'total': 0, 'missing': 0})
        d['total'] += 1
        if is_missing:
            d['missing'] += 1

    print("Totals:")
    print(f"  total tests: {len(items)}")
    print(f"  missing: {len(missing)}")
    print("By tier:")
    for tier_name, d in per_tier.items():
        print(f"  {tier_name:20s}  total={d['total']:3d}  missing={d['missing']:3d}")

    # Show a short sample of missing tests
    sample = ", ".join(f"{m['tier_name']}::{m['test_id']}" for m in missing[:15])
    print("Sample missing:", sample)
