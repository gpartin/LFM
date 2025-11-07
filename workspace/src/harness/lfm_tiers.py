#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Central Tier Registry for LFM test suites.

Provides a single source of truth for:
- Tier number
- Display/category names
- Results directory name
- Test ID prefix and canonical ID pattern
- Runner script path
- Config file path and schema type
- Expected test count (for reporting)

Reads from config/tiers_registry.json if present; otherwise falls back to the
current default tiers (1-4). This allows adding Tier 5 by editing the JSON only.
"""
from __future__ import annotations
from pathlib import Path
import json
import re
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
DEFAULTS: List[Dict] = [
    {
        "tier": 1,
        "name": "Relativistic",
        "category_name": "Relativistic",
        "dir": "Relativistic",
        "prefix": "REL",
        "id_pattern": r"^REL-\\d+$",
        "runner": "run_tier1_relativistic.py",
        "config": "config/config_tier1_relativistic.json",
        "config_schema": "variants",
        "expected": 16,
    },
    {
        "tier": 2,
        "name": "Gravity",
        "category_name": "Gravity Analogue",
        "dir": "Gravity",
        "prefix": "GRAV",
        "id_pattern": r"^GRAV-\\d+$",
        "runner": "run_tier2_gravityanalogue.py",
        "config": "config/config_tier2_gravityanalogue.json",
        "config_schema": "variants",
        "expected": 26,
    },
    {
        "tier": 3,
        "name": "Energy",
        "category_name": "Energy Conservation",
        "dir": "Energy",
        "prefix": "ENER",
        "id_pattern": r"^ENER-\\d+$",
        "runner": "run_tier3_energy.py",
        "config": "config/config_tier3_energy.json",
        "config_schema": "tests",
        "expected": 11,
    },
    {
        "tier": 4,
        "name": "Quantization",
        "category_name": "Quantization",
        "dir": "Quantization",
        "prefix": "QUAN",
        "id_pattern": r"^QUAN-\\d+$",
        "runner": "run_tier4_quantization.py",
        "config": "config/config_tier4_quantization.json",
        "config_schema": "tests",
        "expected": 9,
    },
    {
        "tier": 5,
        "name": "Electromagnetic",
        "category_name": "Electromagnetic",
        "dir": "Electromagnetic",
        "prefix": "EM",
        "id_pattern": r"^EM-\\d+$",
        "runner": "run_tier5_electromagnetic.py",
        "config": "config/config_tier5_electromagnetic.json",
        "config_schema": "tests",
        "expected": 21,
    },
    {
        "tier": 6,
        "name": "Coupling",
        "category_name": "Multi-Domain Coupling",
        "dir": "Coupling",
        "prefix": "COUP",
        "id_pattern": r"^COUP-\\d+$",
        "runner": "run_tier6_coupling.py",
        "config": "config/config_tier6_coupling.json",
        "config_schema": "tests",
        "expected": 12,
    },
    {
        "tier": 7,
        "name": "Thermodynamics",
        "category_name": "Thermodynamics & Statistical Mechanics",
        "dir": "Thermodynamics",
        "prefix": "THERM",
        "id_pattern": r"^THERM-\\d+$",
        "runner": "run_tier7_thermodynamics.py",
        "config": "config/config_tier7_thermodynamics.json",
        "config_schema": "tests",
        "expected": 5,
    },
]

_REGISTRY_CACHE: Optional[List[Dict]] = None


def get_tiers() -> List[Dict]:
    """Return the list of tier definitions from JSON or defaults.

    The array is sorted by tier number.
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    registry_path = ROOT / "config" / "tiers_registry.json"
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Basic validation and compile patterns if needed
            tiers: List[Dict] = []
            for t in data:
                entry = dict(t)
                if "tier" not in entry or "dir" not in entry or "prefix" not in entry:
                    continue
                # default/normalize optional fields
                entry.setdefault("name", entry.get("dir"))
                entry.setdefault("category_name", entry.get("name"))
                entry.setdefault("expected", 0)
                entry.setdefault("config_schema", "variants")
                entry.setdefault("runner", None)
                entry.setdefault("config", None)
                entry.setdefault("id_pattern", rf"^{entry['prefix']}-\\d+$")
                tiers.append(entry)
            _REGISTRY_CACHE = sorted(tiers, key=lambda x: x["tier"]) or DEFAULTS
        except Exception:
            _REGISTRY_CACHE = sorted(DEFAULTS, key=lambda x: x["tier"])  # fallback
    else:
        _REGISTRY_CACHE = sorted(DEFAULTS, key=lambda x: x["tier"])  # fallback
    return _REGISTRY_CACHE


def get_tier_by_number(tier: int) -> Optional[Dict]:
    for t in get_tiers():
        if int(t["tier"]) == int(tier):
            return t
    return None


def get_by_prefix(prefix: str) -> Optional[Dict]:
    p = prefix.upper().strip()
    for t in get_tiers():
        if t["prefix"].upper() == p:
            return t
    return None


def canonical_id_regex_for(tier: int) -> re.Pattern:
    t = get_tier_by_number(tier)
    import re as _re
    if not t:
        return _re.compile(r"^[A-Z]+-\\d+$")
    return _re.compile(t.get("id_pattern", rf"^{t['prefix']}-\\d+$"))
