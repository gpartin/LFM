#!/usr/bin/env python3
"""Recompute all test estimates with updated algorithm."""

from test_metrics import TestMetrics

tm = TestMetrics()

print("Recomputing estimates for all tests...")
for test_id in tm.get_all_test_ids():
    if tm.data[test_id]["runs"]:
        old_est = tm.data[test_id].get("estimated_resources", {})
        new_est = tm._compute_estimate(test_id)
        tm.data[test_id]["estimated_resources"] = new_est
        
        old_mem = old_est.get("memory_mb", 0)
        new_mem = new_est.get("memory_mb", 0)
        
        if abs(new_mem - old_mem) > 100:  # Show significant changes
            print(f"  {test_id}: {old_mem:.0f} MB -> {new_mem:.0f} MB")

tm.save()
print("\nEstimates recomputed and saved!")
