#!/usr/bin/env python3
"""Show sample estimates to verify last-run-only calculation."""

from test_metrics import TestMetrics

tm = TestMetrics()
tests = ['REL-01', 'REL-09', 'GRAV-12', 'GRAV-23']

print('Test estimates (from last run only):')
print('-' * 80)
for t in tests:
    if t in tm.data and tm.data[t].get("estimated_resources"):
        est = tm.data[t]["estimated_resources"]
        print(f'{t:8s}: runtime={est["runtime_sec"]:5.1f}s, cpu={est["cpu_cores_needed"]:4.2f}, '
              f'mem={est["memory_mb"]:6.0f}MB, gpu={est["gpu_memory_mb"]:5.0f}MB')
