# -*- coding: utf-8 -*-
"""Implications Registry Builder

Loads `docs/implications/implications_registry.json`, validates against schema,
optionally auto-extends entries by mining canonical test registry and discoveries.

Future extensions (TODO):
- Integrate with metadata_driven_builder.py to inject implications section into uploads
- Auto-generate website page data for implications
- Derive new chained implications based on dependency graph traversal

Run:
    cd c:\LFM\workspace\tools
    python build_implications_registry.py --validate

"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

SCHEMA_PATH = Path(__file__).parent.parent / "docs" / "implications" / "implications_schema.json"
REGISTRY_PATH = Path(__file__).parent.parent / "docs" / "implications" / "implications_registry.json"

REQUIRED_CORE_FIELDS = [
    "id", "title", "domain", "tier_relevance", "summary", "derivation",
    "evidence_tests", "chain_of_reasoning", "potential_claim_types", "ip_strategy",
    "prior_art_delta", "required_followup_work", "maturity_level", "risk_assessment",
    "created", "updated"
]


def load_json(p: Path) -> Any:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def basic_validate(entry: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for field in REQUIRED_CORE_FIELDS:
        if field not in entry:
            errors.append(f"Missing required field: {field}")
    if 'id' in entry and not entry['id'].startswith('IMP-'):
        errors.append("id must start with IMP-")
    if 'risk_assessment' in entry:
        ra = entry['risk_assessment']
        if not isinstance(ra, dict) or 'likelihood' not in ra or 'impact' not in ra:
            errors.append("risk_assessment must include likelihood and impact")
    return errors


def validate_registry(registry: List[Dict[str, Any]]) -> None:
    all_errors: Dict[str, List[str]] = {}
    ids = set()
    for e in registry:
        errs = basic_validate(e)
        if e.get('id') in ids:
            errs.append("Duplicate id")
        ids.add(e.get('id'))
        if errs:
            all_errors[e.get('id', '<missing>')] = errs
    if all_errors:
        msg_lines = ["Implications registry validation failed:"]
        for k, v in all_errors.items():
            msg_lines.append(f"  {k}:")
            for err in v:
                msg_lines.append(f"    - {err}\n")
        raise SystemExit('\n'.join(msg_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or extend the implications registry.")
    parser.add_argument('--validate', action='store_true', help='Validate existing registry and exit')
    args = parser.parse_args()

    registry = load_json(REGISTRY_PATH)
    if not isinstance(registry, list):
        raise SystemExit("Registry root must be a list")

    if args.validate:
        validate_registry(registry)
        print(f"Validation OK: {len(registry)} entries")
        return

    # Placeholder for future auto-extension logic
    print("No extension operations defined yet (TODO)")


if __name__ == '__main__':
    main()
