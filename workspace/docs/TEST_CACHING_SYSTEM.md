# Test Result Caching System

## Overview

The LFM test suite now includes an industry-standard caching system that automatically avoids re-running tests when their dependencies haven't changed. This significantly speeds up development workflows by reusing valid test results.

## Architecture

### Cache Storage
- **Location**: `c:\LFM\build\cache\test_results/`
- **Structure**: 
  ```
  build/cache/test_results/
  ├── cache_index.json           # Master index of all cached tests
  └── <TEST-ID>/                 # Per-test cache directories
      ├── manifest.json          # Dependency hashes and metadata
      └── results/               # Copied test output files
          ├── summary.json
          ├── plots/
          └── ...
  ```

### Dependency Tracking

The cache system uses SHA256 content hashing to track test dependencies:

1. **Config File**: Tier configuration file (e.g., `config_tier5_electromagnetic.json`)
2. **Source Code**: All Python files in `workspace/src/`
3. **Test Files**: Test implementation files
4. **Settings**: Global settings from `config/settings.yaml`

When any of these change, the cache is automatically invalidated and the test re-runs.

## Integration

### Transparent Caching

Caching is integrated directly into `BaseTierHarness`, making it transparent to all test runners:

```python
class BaseTierHarness:
    def __init__(self, ...):
        # Cache automatically initialized if available
        if self.use_cache and _CACHE_AVAILABLE:
            self.cache_manager = TestCacheManager(cache_root, workspace_root)
```

All tier harnesses inherit this functionality:
- `Tier5ElectromagneticHarness`
- `Tier1Harness`
- `Tier2Harness`
- etc.

### Automatic Cache Checking

The `run_test_with_cache()` method wraps test execution:

```python
def run_test(self, test_config: Dict) -> TestResult:
    # ... test setup ...
    
    # Use caching wrapper if available
    if self.cache_manager and not self.force_rerun:
        result = self.run_test_with_cache(
            test_id, func, self.config, test_config, output_dir
        )
    else:
        result = func(self.config, test_config, output_dir)
```

## Usage

### Default Behavior

Caching is **enabled by default**. Simply run tests normally:

```bash
python src/run_tier5_electromagnetic.py --test EM-01
```

**First run:**
```
[INFO] [cache] Cache invalid for EM-01, running test
[INFO] [cache] Cached results for EM-01 (0.38s)
[INFO]   EM-01: PASSED (0.38s)
```

**Second run (dependencies unchanged):**
```
[INFO] [cache] Using cached results for EM-01
[INFO]   EM-01: PASSED (0.38s)
```

### Force Re-run

To bypass cache and force re-execution:

```bash
python src/run_tier5_electromagnetic.py --test EM-01 --no-cache
```

Output:
```
[INFO] Cache disabled for this run
[INFO]   EM-01: PASSED (0.40s)
```

### Clear Cache

To clear all cached results before running:

```bash
python src/run_tier5_electromagnetic.py --test EM-01 --clear-cache
```

Output:
```
[INFO] Clearing test result cache...
[INFO] Cache cleared
[INFO] [cache] Cache invalid for EM-01, running test
```

## Cache Management

### View Statistics

```bash
python ..\build\scripts\test_cache_manager.py stats
```

Output:
```json
{
  "total_cached_tests": 1,
  "cache_size_mb": 0.25,
  "cached_tests": [
    "EM-01"
  ]
}
```

### Check Specific Test

```bash
python ..\build\scripts\test_cache_manager.py check EM-01
```

### Invalidate Specific Test

```bash
python ..\build\scripts\test_cache_manager.py invalidate EM-01
```

### Clear All Cache

```bash
python ..\build\scripts\test_cache_manager.py clear
```

## Cache Invalidation

Cache is **automatically invalidated** when:

1. **Config file changes**: Any modification to tier configuration
2. **Source code changes**: Any Python file in `src/` modified
3. **Test file changes**: Test implementation modified
4. **Settings changes**: Global settings modified

Example:
```bash
# First run - creates cache
python src/run_tier5_electromagnetic.py --test EM-01

# Modify config file
notepad config\config_tier5_electromagnetic.json

# Second run - cache invalid, test re-runs
python src/run_tier5_electromagnetic.py --test EM-01
# Output: [INFO] [cache] Cache invalid for EM-01, running test
```

## Benefits

### Development Workflow
- **Faster iteration**: Skip unchanged tests during development
- **Smart rebuilds**: Only re-run tests affected by changes
- **CI/CD optimization**: Reuse test results across pipeline stages

### Performance
- **Typical test**: 0.4s execution → <0.1s cache retrieval
- **Complex tests**: Several seconds → instant cache hit
- **Full suite**: Minutes → seconds (when most tests cached)

### Reliability
- **SHA256 hashing**: Industry-standard cryptographic integrity
- **Automatic invalidation**: Never uses stale results
- **Graceful degradation**: Falls back to normal execution if cache unavailable

## Implementation Details

### Cache Manager Class

Located in `build/scripts/test_cache_manager.py`:

```python
class TestCacheManager:
    def __init__(self, cache_root: Path, workspace_root: Path):
        """Initialize cache manager with storage location"""
        
    def is_cache_valid(self, test_id: str, config_file: Path) -> bool:
        """Check if cached results are still valid"""
        
    def get_cached_results(self, test_id: str) -> Optional[Path]:
        """Retrieve cached results directory"""
        
    def store_test_results(self, test_id: str, results_dir: Path, 
                          config_file: Path, metadata: Dict = None):
        """Store test results in cache"""
        
    def clear_cache(self):
        """Clear entire cache"""
```

### Harness Integration

Located in `workspace/src/harness/lfm_test_harness.py`:

```python
class BaseTierHarness:
    def run_test_with_cache(self, test_id: str, test_runner_func, 
                           *args, **kwargs):
        """Execute test with automatic caching"""
        
        # Check cache validity
        if self.cache_manager.is_cache_valid(test_id, config_file):
            # Return cached results
            return self._load_cached_result(test_id)
        
        # Run test
        result = test_runner_func(*args, **kwargs)
        
        # Store in cache
        self.cache_manager.store_test_results(test_id, output_dir, config_file)
        
        return result
```

## Migration Guide

### For Test Developers

No code changes required! Caching is automatic for all tests that:
1. Inherit from `BaseTierHarness`
2. Follow standard test signature: `func(config, test_config, output_dir)`
3. Save results to the provided `output_dir`

### For New Test Suites

To add caching to a new tier:

1. Inherit from `BaseTierHarness`:
```python
class TierXHarness(BaseTierHarness):
    def __init__(self, config_path=None):
        super().__init__(config, out_root, config_path)
        self.config_file_path = str(config_file.resolve())
```

2. Use caching wrapper in `run_test()`:
```python
if self.cache_manager and not self.force_rerun:
    result = self.run_test_with_cache(
        test_id, test_function, config, test_config, output_dir
    )
else:
    result = test_function(config, test_config, output_dir)
```

3. Add command-line arguments:
```python
parser.add_argument("--no-cache", action="store_true",
                   help="Disable caching, force re-run")
parser.add_argument("--clear-cache", action="store_true",
                   help="Clear cache before running")
```

## Troubleshooting

### Cache Not Working

Check that cache manager is available:
```python
# Should see this in logs:
[INFO] [cache] Test caching enabled (cache root: C:\LFM\build\cache\test_results)
```

If not available:
```python
# Check import path
sys.path.insert(0, str(_BUILD_DIR / 'scripts'))
from test_cache_manager import TestCacheManager
```

### Stale Cache

Force invalidation:
```bash
python src/run_tier5_electromagnetic.py --test EM-01 --clear-cache
```

### Cache Size Growing

Monitor cache size:
```bash
python ..\build\scripts\test_cache_manager.py stats
```

Clear old results:
```bash
python ..\build\scripts\test_cache_manager.py clear
```

## Future Enhancements

Potential improvements:
- **Parallel caching**: Cache multiple tests concurrently
- **Distributed cache**: Share cache across machines
- **Smart expiration**: Auto-remove old cached results
- **Cache warming**: Pre-populate cache for common test runs
- **Dependency analysis**: More granular invalidation (per-file tracking)

## References

- **Cache Manager**: `build/scripts/test_cache_manager.py`
- **Base Harness**: `workspace/src/harness/lfm_test_harness.py`
- **Tier 5 Example**: `workspace/src/run_tier5_electromagnetic.py`
- **Cache CLI**: `python ..\build\scripts\test_cache_manager.py --help`
