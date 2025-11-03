# Tier Test Runner Analysis — Standardization Report

**Date:** October 31, 2025  
**Purpose:** Identify and abstract differences in how Tier 1-4 test runners handle diagnostics, configuration, reporting, parallel execution, and test lifecycle management.

---

## Executive Summary

All 4 tier runners (Tier1-Relativistic, Tier2-GravityAnalogue, Tier3-Energy, Tier4-Quantization) share similar architectures but have accumulated **inconsistencies** in:

1. **Resource tracking**: Tier 1,3 record placeholder metrics (0.0 CPU/RAM/GPU), Tier 2,4 don't record at all
2. **Diagnostic depth**: Tier 1 has most comprehensive diagnostics, others minimal
3. **Parallel compatibility**: Test runners assume direct execution, parallel runner wraps externally
4. **Config structure**: All use similar JSON but keys vary (`variants` vs `tests`, `hardware` vs `run_settings`)
5. **Reporting format**: Suite summaries have different column sets per tier
6. **Error handling**: Inconsistent handling of skipped tests and failures

**Recommendation:** Create unified test execution framework with standardized diagnostics, metrics recording, and configuration schema.

---

## 1. Architecture Comparison

### 1.1 Inheritance & Backend Selection

| Tier | Harness Class | Base Class | Backend Selection |
|------|---------------|------------|-------------------|
| **Tier 1** | `Tier1Harness` | `BaseTierHarness` | Via `BaseTierHarness.__init__` (standard) |
| **Tier 2** | `None` | `BaseTierHarness` (used directly) | Via `pick_backend()` in main (inconsistent) |
| **Tier 3** | `None` | `BaseTierHarness` (imported but not used) | Via `pick_backend()` in main (inconsistent) |
| **Tier 4** | `None` | `BaseTierHarness` (imported but not used) | Via `pick_backend()` in main (inconsistent) |

**Issue:** Only Tier 1 properly uses `BaseTierHarness`. Tiers 2-4 import it but don't inherit, duplicating backend selection logic.

**Fix:** Standardize all tiers to inherit from `BaseTierHarness` and use `self.xp`, `self.use_gpu` from base class.

---

### 1.2 Configuration Schema

#### Common Structure
```json
{
  "run_settings": {
    "use_gpu": bool,
    "quick_mode": bool,
    "verbose": bool,
    "show_progress": bool
  },
  "parameters": { /* shared defaults */ },
  "tolerances": { /* pass/fail criteria */ },
  "tests" or "variants": [ /* test array */ ]
}
```

#### Key Differences

| Tier | Test Array Key | Hardware Section | Notes |
|------|----------------|------------------|-------|
| **Tier 1** | `variants` | Inside `run_settings` | Uses `variants` terminology |
| **Tier 2** | `tests` | Separate `hardware` section | Mixed naming |
| **Tier 3** | `tests` | Separate `hardware` section | Uses `tests` |
| **Tier 4** | `tests` | N/A (uses `run_settings` only) | Most minimal |

**Issue:** Inconsistent naming (`variants` vs `tests`) and hardware config location.

**Fix:** Standardize on single schema:
```json
{
  "run_settings": {
    "use_gpu": bool,
    "quick_mode": bool,
    "verbose": bool
  },
  "parameters": { /* defaults */ },
  "tolerances": { /* thresholds */ },
  "tests": [ /* array of test specs */ ]
}
```

---

## 2. Diagnostic & Monitoring Differences

### 2.1 Energy Monitoring

| Tier | Energy Monitor Usage | Drift Tracking | Energy Lock | Per-Step Logging |
|------|---------------------|----------------|-------------|------------------|
| **Tier 1** | ✅ Used in some tests (REL-05, 06) | ✅ Yes | ❌ No | ✅ Yes (CSV export) |
| **Tier 2** | ✅ Used in time-series tests | ✅ Yes | ❌ No | ✅ Yes (diagnostics/) |
| **Tier 3** | ✅ Core focus (ENER-01–07) | ✅ Yes | ❌ No | ✅ Yes (CSV export) |
| **Tier 4** | ✅ Used in QUAN-01,02,11 | ✅ Yes | ❌ No | ⚠️ Partial (some tests) |

**Observation:** All tiers use `EnergyMonitor` but with varying depth. Tier 3 most comprehensive (it's the energy suite), others selective.

**Recommendation:** Make energy monitoring **optional but standardized**:
- All tests should support `enable_energy_monitor` flag in config
- Energy CSVs should follow same format: `[step, time, KE, GE, PE, total, drift_pct]`
- Use `energy_monitor.py` consistently (already in place)

---

### 2.2 Diagnostics Output Structure

#### Tier 1 (Most Comprehensive)
```
results/Relativistic/
  REL-01/
    summary.json
    metrics.csv
    diagnostics/
      spectrum.csv
      energy_flow.csv
      phase_corr.csv
    plots/
      dispersion.png
      isotropy.png
```

#### Tier 2 (Moderate)
```
results/Gravity/
  GRAV-01/
    summary.json
    diagnostics/
      local_frequency.csv
      packet_tracking.csv  (if enabled)
      time_series.csv      (if enabled)
    plots/
      chi_field.png
      frequency_map.png
```

#### Tier 3 (Focused on Energy)
```
results/Energy/
  ENER-01/
    summary.json
    energy_log.csv
    entropy_log.csv
    plots/
      energy_drift.png
      entropy_growth.png
```

#### Tier 4 (Variable)
```
results/Quantization/
  QUAN-01/
    summary.json
    energy_log.csv       (some tests)
    harmonics.csv        (QUAN-05, 06)
    planck_occupation.csv (QUAN-14)
    plots/
      <test-specific>.png
```

**Issue:** No standardized diagnostic structure. Hard to compare tests across tiers.

**Fix:** Unified structure:
```
results/<Category>/
  <TEST-ID>/
    summary.json         (required - standard schema)
    metrics.csv          (optional - test-specific metrics)
    diagnostics/         (optional - detailed data)
      energy.csv
      spectrum.csv
      <custom>.csv
    plots/               (optional - visualizations)
      <test-specific>.png
```

---

## 3. Metrics Recording & Resource Tracking

### 3.1 Current Implementation

#### Tier 1 (run_tier1_relativistic.py:2051-2060)
```python
test_metrics = TestMetrics()
for r in results:
    metrics_data = {
        "exit_code": 0 if r["passed"] else 1,
        "runtime_sec": r["runtime_sec"],
        "peak_cpu_percent": 0.0,  # ❌ NOT TRACKED
        "peak_memory_mb": 0.0,     # ❌ NOT TRACKED
        "peak_gpu_memory_mb": 0.0, # ❌ NOT TRACKED
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    test_metrics.record_run(r["test_id"], metrics_data)
```

#### Tier 2 (run_tier2_gravityanalogue.py)
```python
# ❌ NO METRICS RECORDING AT ALL
# After results loop:
update_master_test_status()  # Only updates status CSV
```

#### Tier 3 (run_tier3_energy.py:460-470)
```python
test_metrics = TestMetrics()
for r in results:
    metrics_data = {
        "exit_code": 0 if r["passed"] else 1,
        "runtime_sec": r["runtime_sec"],
        "peak_cpu_percent": 0.0,  # ❌ NOT TRACKED
        "peak_memory_mb": 0.0,     # ❌ NOT TRACKED
        "peak_gpu_memory_mb": 0.0, # ❌ NOT TRACKED
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    test_metrics.record_run(r["test_id"], metrics_data)
```

#### Tier 4 (run_tier4_quantization.py)
```python
# ❌ NO METRICS RECORDING AT ALL
# After results loop:
update_master_test_status()  # Only updates status CSV
```

#### Parallel Runner (run_parallel_tests.py:85-195)
```python
# ✅ ACTUALLY TRACKS RESOURCES
peak_cpu = 0.0
peak_memory_mb = 0.0
peak_gpu_used_mb = 0.0

# Polls psutil during test execution
ps_process = psutil.Process(process.pid)
cpu_pct = ps_process.cpu_percent(interval=0.1)
mem_info = ps_process.memory_info()
mem_mb = mem_info.rss / (1024**2)
peak_cpu = max(peak_cpu, cpu_pct)
peak_memory_mb = max(peak_memory_mb, mem_mb)

# Returns actual metrics
return (test_id, tier, {
    "exit_code": retcode,
    "runtime_sec": runtime,
    "peak_cpu_percent": peak_cpu,
    "peak_memory_mb": peak_memory_mb,
    "peak_gpu_memory_mb": peak_gpu_used_mb,
    ...
})
```

**Problem:** Tier runners record **placeholder zeros** for CPU/RAM/GPU, only parallel runner tracks real metrics.

**Impact:**
- Metrics database (`test_metrics_history.json`) has accurate data only from parallel runs
- Direct tier runner invocations pollute DB with useless zero entries
- Resource budgeting in parallel runner relies on incomplete historical data

---

### 3.2 Proposed Solution: Resource Tracking Mixin

Create `ResourceTrackingMixin` to add resource monitoring to tier runners:

```python
# resource_tracking.py
import psutil
import time
from typing import Dict, Optional

class ResourceTracker:
    """Track CPU, RAM, GPU usage during test execution."""
    
    def __init__(self):
        self.peak_cpu = 0.0
        self.peak_memory_mb = 0.0
        self.peak_gpu_mb = 0.0
        self.start_time = None
        self._process = None
        
    def start(self):
        """Begin tracking resources."""
        self.start_time = time.time()
        try:
            self._process = psutil.Process()
        except Exception:
            self._process = None
            
    def sample(self):
        """Sample current resource usage and update peaks."""
        if not self._process:
            return
            
        try:
            cpu = self._process.cpu_percent(interval=0)
            mem = self._process.memory_info().rss / (1024**2)
            self.peak_cpu = max(self.peak_cpu, cpu)
            self.peak_memory_mb = max(self.peak_memory_mb, mem)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
        # GPU tracking (if available)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                gpu_mb = float(result.stdout.strip().split()[0])
                self.peak_gpu_mb = max(self.peak_gpu_mb, gpu_mb)
        except Exception:
            pass
            
    def get_metrics(self) -> Dict:
        """Return collected metrics."""
        return {
            "peak_cpu_percent": self.peak_cpu,
            "peak_memory_mb": self.peak_memory_mb,
            "peak_gpu_memory_mb": self.peak_gpu_mb,
            "runtime_sec": time.time() - self.start_time if self.start_time else 0.0
        }
```

**Integration into BaseTierHarness:**
```python
class BaseTierHarness(NumericIntegrityMixin):
    def __init__(self, cfg, out_root, config_name):
        # ... existing init ...
        self.resource_tracker = ResourceTracker()
        self._enable_resource_tracking = cfg.get("run_settings", {}).get(
            "enable_resource_tracking", True
        )
        
    def run_test_with_tracking(self, test_func, *args, **kwargs):
        """Wrapper to track resources during test execution."""
        if self._enable_resource_tracking:
            self.resource_tracker.start()
            
        # Run test
        result = test_func(*args, **kwargs)
        
        # Sample periodically during long tests (background thread)
        # Or sample at end for quick tests
        if self._enable_resource_tracking:
            self.resource_tracker.sample()
            result.metrics.update(self.resource_tracker.get_metrics())
            
        return result
```

---

## 4. Parallel Execution Compatibility

### 4.1 Current Parallel Runner Architecture

**run_parallel_tests.py** spawns subprocesses:
```python
def run_single_test(test_id, tier, timeout_sec):
    runners = {
        1: "run_tier1_relativistic.py",
        2: "run_tier2_gravityanalogue.py",
        3: "run_tier3_energy.py",
        4: "run_tier4_quantization.py"
    }
    cmd = ["python", "-u", runners[tier], "--test", test_id]
    # ... subprocess execution with resource monitoring ...
```

**Pros:**
- ✅ Complete isolation (separate Python processes)
- ✅ Accurate resource tracking (can monitor child process)
- ✅ Timeout enforcement
- ✅ No interference between tests

**Cons:**
- ❌ Tier runners unaware they're running in parallel
- ❌ Can't optimize for parallel context (e.g., different logging verbosity)
- ❌ Duplicates subprocess overhead for each test

---

### 4.2 Parallel-Aware Test Execution

**Proposal:** Add parallel mode detection and optimization

```python
# In BaseTierHarness
def __init__(self, cfg, out_root, config_name):
    # ... existing init ...
    
    # Detect parallel execution context
    self.parallel_mode = os.environ.get("LFM_PARALLEL_MODE") == "1"
    
    if self.parallel_mode:
        # Reduce console verbosity (parent handles output)
        self.show_progress = False
        # Disable interactive prompts
        # Use buffered logging
        
# In run_parallel_tests.py
def run_single_test(test_id, tier, timeout_sec):
    # Set environment variable
    env = os.environ.copy()
    env["LFM_PARALLEL_MODE"] = "1"
    
    cmd = ["python", "-u", runners[tier], "--test", test_id]
    process = subprocess.Popen(cmd, env=env, ...)
```

**Benefits:**
- Tests can optimize for parallel context (less verbose logging)
- Cleaner output in parallel mode
- Tests aware of resource constraints

---

## 5. Summary JSON Schema Standardization

### 5.1 Current Schemas (Inconsistent)

#### Tier 1 Summary
```json
{
  "test_id": "REL-01",
  "description": "...",
  "category": "relativistic",
  "tier": 1,
  "passed": true,
  "metrics": {
    "rel_err": 0.0012,
    "omega_meas": 3.1415,
    "omega_theory": 3.1416,
    "k_fraction_lattice": 0.1
  },
  "runtime_sec": 12.3,
  "timestamp": 1234567890
}
```

#### Tier 2 Summary
```json
{
  "test_id": "GRAV-01",
  "description": "...",
  "category": "gravity",
  "tier": 2,
  "status": "Passed",  // ❌ Inconsistent with "passed": bool
  "metrics": {
    "ratio_meas_serial": 0.85,
    "ratio_meas_parallel": 0.85,
    "ratio_theory": 0.857,
    "rel_err_ratio": 0.008
  },
  "runtime_sec": 45.6
}
```

#### Tier 3 Summary
```json
{
  "test_id": "ENER-01",
  "description": "...",
  "tier": 3,
  "category": "energy",
  "passed": true,
  "metrics": {
    "energy_drift": 0.0023,
    "entropy_monotonic": true,
    "wave_drop": 0.15
  },
  "runtime_sec": 89.1
}
```

#### Tier 4 Summary
```json
{
  "test_id": "QUAN-09",
  "description": "...",
  "tier": 4,
  "category": "Quantization",  // ❌ Capitalized vs lowercase
  "status": "Passed",  // ❌ Inconsistent
  "metrics": {
    "delta_x": 0.5,
    "delta_k": 1.0,
    "product": 0.5,
    "uncertainty_principle_satisfied": true
  },
  "runtime_sec": 3.2
}
```

### 5.2 Standardized Schema

**Proposed universal summary.json:**
```json
{
  "test_id": "TEST-ID",
  "description": "Human-readable description",
  "tier": 1,
  "category": "relativistic",  // lowercase, standardized categories
  "status": "passed" | "failed" | "skipped",  // lowercase enum
  "timestamp": 1234567890.123,
  "hardware": {
    "backend": "CuPy" | "NumPy",
    "gpu_enabled": true,
    "python_version": "3.13.9"
  },
  "parameters": {
    // Test-specific input parameters
    "N": 1024,
    "dx": 0.1,
    "dt": 0.005,
    ...
  },
  "metrics": {
    // Test-specific output metrics (flexible schema)
    "rel_err": 0.0012,
    ...
  },
  "resources": {
    "runtime_sec": 12.3,
    "peak_cpu_percent": 45.2,
    "peak_memory_mb": 512.0,
    "peak_gpu_memory_mb": 2048.0
  }
}
```

**Changes:**
1. `"passed": bool` → `"status": "passed"|"failed"|"skipped"`
2. Add `"hardware"` section (backend, GPU, Python version)
3. Add `"parameters"` section (test inputs)
4. Move resource metrics to `"resources"` section
5. Standardize category names (lowercase)
6. Add `timestamp` as float (epoch seconds with subsecond precision)

---

## 6. Error Handling & Skipped Tests

### 6.1 Current Handling

| Tier | Skip Mechanism | Error Handling | Status Reporting |
|------|----------------|----------------|------------------|
| **Tier 1** | ❌ No skip support | ✅ Try/except in test loop | ✅ Logs pass/fail |
| **Tier 2** | ✅ `"skip": true` in config | ✅ Try/except in test loop | ✅ Logs pass/fail/skip |
| **Tier 3** | ✅ `"skip": true` in config | ⚠️ Minimal error handling | ✅ Logs pass/fail/skip |
| **Tier 4** | ✅ `"skip": true` in config | ✅ Try/except in test loop | ✅ Logs pass/fail/skip |

**Tier 1** example (no skip):
```python
for variant in harness.variants:
    try:
        result = harness.run_variant(variant)
        results.append(result)
    except Exception as e:
        log(f"Error in {variant['test_id']}: {e}", "FAIL")
        # Test not added to results → missing from summary
```

**Tier 2** example (skip support):
```python
for test in harness.variants:
    if test.get("skip", False):
        log(f"[{test['test_id']}] SKIPPED", "WARN")
        continue  # ❌ Skipped tests not in results summary
    try:
        result = run_test(test)
        results.append(result)
    except Exception as e:
        log(f"Error: {e}", "FAIL")
```

**Issue:** Skipped tests don't appear in final results, making it hard to track test coverage.

**Fix:** Always record skipped tests in results:
```python
for test in tests:
    if test.get("skip", False):
        log(f"[{test['test_id']}] SKIPPED: {test.get('skip_reason', 'No reason')}", "WARN")
        results.append({
            "test_id": test["test_id"],
            "status": "skipped",
            "skip_reason": test.get("skip_reason", "No reason provided"),
            "runtime_sec": 0.0
        })
        continue
    
    try:
        result = run_test(test)
        results.append(result)
    except Exception as e:
        log(f"[{test['test_id']}] ERROR: {e}", "FAIL")
        results.append({
            "test_id": test["test_id"],
            "status": "error",
            "error_message": str(e),
            "runtime_sec": 0.0
        })
```

---

## 7. Recommended Refactoring Strategy

### Phase 1: Immediate Fixes (Low Risk)
1. **Standardize config schema**: All tiers use `tests` (not `variants`), consistent `run_settings` structure
2. **Fix metrics recording**: Remove placeholder zeros, either track real resources or don't record at all
3. **Standardize summary.json**: Use unified schema across all tiers
4. **Add skip tracking**: Record skipped tests in results

### Phase 2: Structural Improvements (Medium Risk)
1. **Create ResourceTracker mixin**: Add real resource monitoring to tier runners
2. **Standardize diagnostics structure**: Unified output directory layout
3. **Add parallel mode detection**: Optimize logging/output for parallel execution
4. **Unified error handling**: Common exception handling and error reporting

### Phase 3: Major Refactor (High Risk, High Value)
1. **Unified test executor**: Single `run_test()` function shared by all tiers
2. **Plugin architecture**: Tier-specific physics in plugins, common infrastructure shared
3. **Streaming diagnostics**: Emit structured events (JSON-L) for real-time monitoring
4. **Configuration validation**: Schema validation for all config files

---

## 8. Proposed Common Test Executor

**New file:** `lfm_test_executor.py`

```python
from dataclasses import dataclass
from typing import Dict, Callable, Any
from pathlib import Path
import time

from resource_tracking import ResourceTracker
from lfm_results import save_summary, ensure_dirs
from lfm_console import log

@dataclass
class TestExecutionContext:
    """Context passed to test functions."""
    test_id: str
    config: Dict
    out_dir: Path
    xp: Any  # NumPy or CuPy
    on_gpu: bool
    parameters: Dict
    tolerances: Dict
    
@dataclass
class TestResult:
    """Standardized test result."""
    test_id: str
    description: str
    status: str  # "passed", "failed", "skipped", "error"
    metrics: Dict
    parameters: Dict
    resources: Dict
    error_message: str = None
    skip_reason: str = None
    
class TestExecutor:
    """
    Unified test executor for all tiers.
    
    Handles:
    - Resource tracking
    - Error handling
    - Skip detection
    - Metrics recording
    - Summary generation
    """
    
    def __init__(self, enable_resource_tracking=True):
        self.enable_tracking = enable_resource_tracking
        
    def execute_test(
        self,
        test_func: Callable,
        context: TestExecutionContext
    ) -> TestResult:
        """
        Execute a single test with full monitoring.
        
        Args:
            test_func: Test function to execute
            context: Test execution context
            
        Returns:
            TestResult with status, metrics, resources
        """
        # Check skip
        if context.config.get("skip", False):
            return TestResult(
                test_id=context.test_id,
                description=context.config.get("description", ""),
                status="skipped",
                metrics={},
                parameters=context.parameters,
                resources={"runtime_sec": 0.0},
                skip_reason=context.config.get("skip_reason", "No reason provided")
            )
        
        # Setup
        ensure_dirs(context.out_dir)
        tracker = ResourceTracker() if self.enable_tracking else None
        
        if tracker:
            tracker.start()
        
        # Execute
        t0 = time.time()
        try:
            result = test_func(context)
            status = "passed" if result.passed else "failed"
            metrics = result.metrics
            error_msg = None
            
        except Exception as e:
            log(f"[{context.test_id}] ERROR: {e}", "FAIL")
            status = "error"
            metrics = {}
            error_msg = str(e)
        
        runtime = time.time() - t0
        
        # Collect resources
        if tracker:
            tracker.sample()
            resources = tracker.get_metrics()
            resources["runtime_sec"] = runtime
        else:
            resources = {"runtime_sec": runtime}
        
        # Build result
        result = TestResult(
            test_id=context.test_id,
            description=context.config.get("description", ""),
            status=status,
            metrics=metrics,
            parameters=context.parameters,
            resources=resources,
            error_message=error_msg
        )
        
        # Save summary
        self._save_result(result, context)
        
        return result
    
    def _save_result(self, result: TestResult, context: TestExecutionContext):
        """Save standardized summary.json."""
        summary = {
            "test_id": result.test_id,
            "description": result.description,
            "tier": context.config.get("tier"),
            "category": context.config.get("category"),
            "status": result.status,
            "timestamp": time.time(),
            "hardware": {
                "backend": "CuPy" if context.on_gpu else "NumPy",
                "gpu_enabled": context.on_gpu
            },
            "parameters": result.parameters,
            "metrics": result.metrics,
            "resources": result.resources
        }
        
        if result.error_message:
            summary["error_message"] = result.error_message
        if result.skip_reason:
            summary["skip_reason"] = result.skip_reason
        
        save_summary(context.out_dir, result.test_id, summary)
```

**Usage in tier runners:**
```python
# In run_tierX.py
from lfm_test_executor import TestExecutor, TestExecutionContext

executor = TestExecutor(enable_resource_tracking=True)

for test in tests:
    context = TestExecutionContext(
        test_id=test["test_id"],
        config=test,
        out_dir=out_root / test["test_id"],
        xp=xp,
        on_gpu=on_gpu,
        parameters=params,
        tolerances=tol
    )
    
    result = executor.execute_test(
        test_func=lambda ctx: run_my_test(ctx),
        context=context
    )
    
    results.append(result)
```

---

## 9. Action Items

### Immediate (This Session)
- [ ] Standardize `summary.json` schema across all 4 tiers
- [ ] Remove placeholder resource metrics (0.0 CPU/RAM/GPU) from Tier 1, 3
- [ ] Add metrics recording to Tier 2, 4
- [ ] Document current differences in this file

### Short Term (Next Week)
- [ ] Implement `ResourceTracker` class
- [ ] Add resource tracking to all tier runners
- [ ] Standardize config schema (all use `tests`, not `variants`)
- [ ] Unify error handling and skip tracking

### Medium Term (Next Month)
- [ ] Create `TestExecutor` unified execution framework
- [ ] Migrate Tier 1 to use `TestExecutor`
- [ ] Migrate Tiers 2-4 to use `TestExecutor`
- [ ] Add configuration schema validation

### Long Term (Future)
- [ ] Plugin architecture for tier-specific physics
- [ ] Streaming diagnostics (JSON-L event log)
- [ ] Real-time monitoring dashboard
- [ ] Automated performance regression detection

---

## 10. Conclusion

All 4 tier runners work correctly but have accumulated **technical debt** from independent development. Main issues:

1. **Inconsistent resource tracking**: Only parallel runner tracks real metrics
2. **Config schema drift**: Similar but not identical structures
3. **Diagnostics fragmentation**: No standardized output format
4. **Duplicate code**: Backend selection, metrics recording repeated 4 times

**Recommended approach:**
1. Start with **low-risk standardization** (schemas, remove placeholders)
2. Add **ResourceTracker** for real metrics in tier runners
3. Gradually migrate to **unified TestExecutor** framework
4. Maintain backward compatibility throughout

**Expected benefits:**
- Consistent test results across direct and parallel execution
- Accurate resource metrics for all tests
- Easier maintenance (single codebase for common functionality)
- Better test coverage tracking (skipped tests recorded)
- Cleaner parallel execution (tests aware of context)

---

**End of Analysis**
