# -*- coding: utf-8 -*-
"""
Unit tests for the unified validation core.
Validates aggregate_validation() and validation_block() behavior, including:
- Energy threshold lookup and pass/fail
- Primary metric name resolution and threshold comparison
- Stable output structure of validation_block()
"""
from typing import Dict

from harness.validation import aggregate_validation, validation_block

# Simple test runner compatibility: expose pytest-style functions and a main.
__all__ = [
    'test_aggregate_validation_pass',
    'test_aggregate_validation_primary_fail',
    'test_validation_block_structure'
]



def _meta(primary_metric: str = "local_frequency_ratio_error", primary_threshold: float = 0.02,
          energy_threshold: float = 1e-3) -> Dict:
    return {
        "tests": {
            "GRAV-TEST": {
                "validation_criteria": {
                    "energy_conservation": {"threshold": energy_threshold},
                    "primary": {"metric": primary_metric, "threshold": primary_threshold}
                }
            }
        }
    }


def test_aggregate_validation_pass():
    meta = _meta(primary_metric="local_frequency_ratio_error", primary_threshold=0.02, energy_threshold=1e-3)
    # Mapping in validation resolves 'local_frequency_ratio_error' -> 'rel_err_ratio', so provide that key
    metrics = {"rel_err_ratio": 0.01}
    agg = aggregate_validation(meta, "GRAV-TEST", energy_drift=5e-4, metrics=metrics)

    assert agg.energy_ok is True
    assert abs(agg.energy_threshold - 1e-3) < 1e-12
    assert agg.primary_ok is True
    # primary_metric is the resolved key used for comparison
    assert agg.primary_metric in ("rel_err_ratio", "local_frequency_ratio_error")
    assert agg.primary_value == 0.01
    assert abs((agg.primary_threshold or 0.0) - 0.02) < 1e-12


def test_aggregate_validation_primary_fail():
    meta = _meta(primary_metric="redshift_ratio_error", primary_threshold=0.01, energy_threshold=1e-3)
    # Mapping resolves 'redshift_ratio_error' -> 'ratio_match_error', so provide that key
    metrics = {"ratio_match_error": 0.025}
    agg = aggregate_validation(meta, "GRAV-TEST", energy_drift=5e-4, metrics=metrics)

    assert agg.energy_ok is True
    assert agg.primary_ok is False  # exceeds threshold
    assert agg.primary_metric in ("ratio_match_error", "redshift_ratio_error")
    assert agg.primary_value == 0.025
    assert abs((agg.primary_threshold or 0.0) - 0.01) < 1e-12


def test_validation_block_structure():
    meta = _meta(primary_metric="time_delay_error", primary_threshold=0.05, energy_threshold=2e-3)
    # Mapping resolves 'time_delay_error' -> 'delay_error'
    metrics = {"delay_error": 0.01, "extra_metric": 42.0}
    agg = aggregate_validation(meta, "GRAV-TEST", energy_drift=1e-3, metrics=metrics)
    block = validation_block(agg)

    # Top-level keys
    assert set(block.keys()) == {"test_id", "energy", "primary", "metrics", "timestamp"}

    # Energy section
    assert isinstance(block["energy"], dict)
    assert block["energy"]["ok"] is True
    assert abs(block["energy"]["drift"] - 1e-3) < 1e-12
    assert abs(block["energy"]["threshold"] - 2e-3) < 1e-12

    # Primary section
    assert isinstance(block["primary"], dict)
    assert block["primary"]["ok"] is True
    assert block["primary"]["metric"] in ("delay_error", "time_delay_error")
    assert abs(block["primary"]["value"] - 0.01) < 1e-12
    assert abs((block["primary"]["threshold"] or 0.0) - 0.05) < 1e-12

    # Metrics passthrough should include both provided metrics
    assert block["metrics"]["delay_error"] == 0.01
    assert block["metrics"]["extra_metric"] == 42.0


if __name__ == "__main__":  # Allow ad-hoc execution via python file
    # Execute tests manually if run as a script.
    failures = 0
    try:
        test_aggregate_validation_pass()
    except AssertionError as e:
        print("test_aggregate_validation_pass FAILED", e)
        failures += 1
    try:
        test_aggregate_validation_primary_fail()
    except AssertionError as e:
        print("test_aggregate_validation_primary_fail FAILED", e)
        failures += 1
    try:
        test_validation_block_structure()
    except AssertionError as e:
        print("test_validation_block_structure FAILED", e)
        failures += 1
    if failures:
        raise SystemExit(1)
    print("All validation core tests passed.")
