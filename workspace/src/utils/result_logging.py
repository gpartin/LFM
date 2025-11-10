# -*- coding: utf-8 -*-
"""
Unified result logging for tier validation tests.

Provides consistent PASS/FAIL output formatting across all tiers.
"""
from typing import Dict, Optional, Any
from ui.lfm_console import log


def log_test_result(
    test_id: str,
    description: str,
    validation: Any,  # ValidationResult from aggregate_validation()
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Log test result with consistent formatting.
    
    Args:
        test_id: Test identifier (e.g., "REL-01", "ENER-03")
        description: Brief test description
        validation: ValidationResult from aggregate_validation()
        metrics: Optional dict of metric_name -> value for display
    
    Format:
        [TEST-ID] Description — primary_metric=value, energy_drift=value → PASS|FAIL
    
    Example:
        [ENER-11] Momentum conservation — momentum_drift=2.1e-06, energy_drift=4.5e-08 → PASS
    """
    metrics = metrics or {}
    
    # Extract primary metric and energy_drift for display
    primary_name = validation.primary_metric
    primary_value = metrics.get(primary_name, validation.primary_value)
    energy_drift = validation.energy_drift
    
    # Build metric display string
    metric_parts = [f"{primary_name}={primary_value:.3e}"]
    
    # Always show energy_drift if available
    if energy_drift is not None:
        metric_parts.append(f"energy_drift={energy_drift:.3e}")
    
    # Optionally show additional metrics (avoid duplicating primary or energy)
    for key, value in sorted(metrics.items()):
        if key not in {primary_name, "energy_drift"}:
            metric_parts.append(f"{key}={value:.3e}")
    
    metrics_str = ", ".join(metric_parts)
    
    # Compute overall pass from energy_ok and primary_ok
    passed = validation.energy_ok and validation.primary_ok
    result_str = "PASS" if passed else "FAIL"
    
    log(
        f"[{test_id}] {description} — {metrics_str} → {result_str}",
        level="PASS" if passed else "FAIL"
    )
