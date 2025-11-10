# LFM Coding Standards

## File Encoding Rules

### **MANDATORY: Always specify UTF-8 encoding**

**Problem:** Windows defaults to `cp1252` encoding, which fails on Unicode characters (checkmarks ✓, arrows →, mathematical symbols ∇²).

**Solution:** ALWAYS specify `encoding='utf-8'` when opening files.

### File I/O Rules

```python
# ✓ CORRECT - Always specify encoding
with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

with open(filename, 'w', encoding='utf-8') as f:
    f.write(content)

with open(filename, 'a', encoding='utf-8') as f:
    f.write(content)

# ✗ WRONG - Platform-dependent encoding
with open(filename, 'r') as f:  # Fails on Windows with Unicode!
    content = f.read()
```

### JSON Files

```python
import json

# ✓ CORRECT
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# ✗ WRONG
with open('data.json', 'w') as f:
    json.dump(data, f)
```

### CSV Files

```python
import csv

# ✓ CORRECT
with open('data.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

with open('data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)

# ✗ WRONG
with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
```

### Pandas DataFrames

```python
import pandas as pd

# ✓ CORRECT
df.to_csv('data.csv', encoding='utf-8', index=False)
df = pd.read_csv('data.csv', encoding='utf-8')

# ✗ WRONG
df.to_csv('data.csv', index=False)
```

### Path.write_text() / Path.read_text()

```python
from pathlib import Path

# ✓ CORRECT
Path('file.txt').write_text(content, encoding='utf-8')
content = Path('file.txt').read_text(encoding='utf-8')

# ✗ WRONG
Path('file.txt').write_text(content)  # Platform-dependent!
```

---

## Python Source File Encoding

### Always include UTF-8 declaration at top of file

```python
# -*- coding: utf-8 -*-
"""
Module docstring here.
"""

import numpy as np
```

**Why:** Ensures Python parser interprets source code as UTF-8, even on Windows.

**When needed:** 
- If file contains Unicode characters in comments/docstrings
- If file writes Unicode to stdout/stderr
- **Best practice:** Include in ALL Python files for consistency

---

## Print Statements and Logging

### Unicode-safe printing

```python
# ✓ CORRECT - Works on all platforms
print("Energy conservation: PASS")  # Use ASCII when possible
print("Algorithm 1: PASS")

# If you MUST use Unicode:
import sys
if sys.platform == 'win32':
    # Reconfigure stdout for UTF-8 on Windows
    sys.stdout.reconfigure(encoding='utf-8')

print("Energy conservation: ✓")  # Now safe

# ⚠️ CAUTION - May fail on Windows without reconfiguration
print("✓ Test passed")  # Can cause UnicodeEncodeError
```

### Logging

```python
import logging

# ✓ CORRECT - Configure logger for UTF-8
logging.basicConfig(
    filename='app.log',
    encoding='utf-8',  # Python 3.9+
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# For Python < 3.9:
handler = logging.FileHandler('app.log', encoding='utf-8')
logger.addHandler(handler)
```

---

## Common Unicode Characters - Use Carefully

### Safe to use in files (with UTF-8 encoding):
- Mathematical: `∇ ∂ ∫ ∑ π α β γ Δ`
- Arrows: `→ ← ↔ ⇒`
- Symbols: `≈ ≠ ≤ ≥ ±`
- Checkmarks: `✓ ✗`
- Bullets: `• ◦ ▪`

### Avoid in terminal output (unless stdout reconfigured):
- Use ASCII alternatives: `PASS`, `FAIL`, `->`, `!=`, `<=`, `>=`

---

## Git Configuration

Ensure git handles UTF-8 correctly:

```bash
git config --global core.quotepath false
git config --global i18n.commitEncoding utf-8
git config --global i18n.logOutputEncoding utf-8
```

---

## IDE/Editor Settings

### VS Code (`.vscode/settings.json`)

```json
{
  "files.encoding": "utf8",
  "files.autoGuessEncoding": false,
  "python.analysis.extraPaths": ["./src"],
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
  }
}
```

---

## Testing

Always test on Windows to catch encoding issues:

```python
def test_unicode_file_io():
    """Test file I/O with Unicode characters."""
    from pathlib import Path
    
    test_content = "Energy: ✓ PASS\nGradient: ∇²E\nArrow: →"
    test_file = Path("test_unicode.txt")
    
    # Write with UTF-8
    test_file.write_text(test_content, encoding='utf-8')
    
    # Read with UTF-8
    read_content = test_file.read_text(encoding='utf-8')
    
    assert read_content == test_content
    test_file.unlink()  # Clean up
```

---

## Checklist for New Files

Before committing any Python file that does file I/O:

- [ ] All `open()` calls include `encoding='utf-8'`
- [ ] All `Path.write_text()` / `Path.read_text()` include `encoding='utf-8'`
- [ ] All `json.dump()` calls include `ensure_ascii=False` (optional but recommended)
- [ ] All `pd.read_csv()` / `pd.to_csv()` include `encoding='utf-8'`
- [ ] Source file includes `# -*- coding: utf-8 -*-` if using Unicode in source
- [ ] Tested on Windows (if applicable)

---

## Quick Reference Card

```python
# File I/O
open(path, 'r', encoding='utf-8')
open(path, 'w', encoding='utf-8')
Path(path).read_text(encoding='utf-8')
Path(path).write_text(text, encoding='utf-8')

# JSON
json.dump(data, f, ensure_ascii=False, indent=2)  # encoding='utf-8' on file handle

# CSV
csv.writer(f)  # encoding='utf-8' on file handle
pd.read_csv(path, encoding='utf-8')
pd.to_csv(path, encoding='utf-8')

# Logging
logging.basicConfig(filename='app.log', encoding='utf-8')

# Stdout (Windows fix)
sys.stdout.reconfigure(encoding='utf-8')
```

---

## Why This Matters

**Historical Context:**
- Python 3 uses Unicode (UTF-8) internally
- Windows still defaults to legacy encodings (cp1252, cp437)
- Mac/Linux default to UTF-8
- Result: Code that works on Mac/Linux fails on Windows

**LFM Project:**
- Scientific notation requires Unicode: `∇²E`, `∂²E/∂t²`
- Terminal output uses checkmarks: `✓`, `✗`
- Cross-platform compatibility is critical
- **One missed `encoding='utf-8'` breaks Windows builds**

**Solution:** Make UTF-8 encoding MANDATORY in all file operations.

---

## Enforcement

This is a **CRITICAL** requirement. All pull requests will be reviewed for encoding compliance.

**Automated check coming soon:** Pre-commit hook to detect missing `encoding=` parameters.

---

## Metrics Recording Pattern

### **Automatic Metrics Recording in BaseTierHarness**

**Overview:** As of 2025-11-10, all test executions automatically record metrics to `test_metrics_history.json` via the centralized system in `BaseTierHarness`.

**How It Works:**
1. `BaseTierHarness.run_with_standard_wrapper()` automatically calls `TestMetrics.record_run()` after every test
2. Works for ALL execution paths:
   - Direct tier runner invocation (`python run_tier5_electromagnetic.py --test EM-01`)
   - Parallel suite (`python run_parallel_suite.py --tiers "5,6,7"`)
   - Cached results (metrics updated even on cache hits)
3. Metrics extracted from:
   - Result object attributes (`result.passed`, `result.runtime_sec`, etc.)
   - Cached `summary.json` files (for cache hits)
   - Default values for missing fields

**Tier Runner Pattern** (NO manual metrics code needed):

```python
# ✓ CORRECT - Metrics recorded automatically by BaseTierHarness
class TierNHarness(BaseTierHarness):
    def run_variant(self, v: Dict) -> TestSummary:
        # Test implementation
        result = TestSummary(
            test_id=v["test_id"],
            description=v["description"],
            passed=passed,
            runtime_sec=runtime,
            # ... other fields
        )
        return result  # Metrics automatically recorded here

def main():
    harness = TierNHarness(cfg, out_root, config_name)
    results = harness.run()
    # NO need to call TestMetrics.record_run() manually
    update_master_test_status()  # Update master status as usual
```

```python
# ✗ WRONG - Redundant manual metrics recording
def main():
    harness = TierNHarness(cfg, out_root, config_name)
    results = harness.run()
    
    # DON'T DO THIS - metrics already recorded!
    test_metrics = TestMetrics()
    for r in results:
        test_metrics.record_run(r["test_id"], {...})  # Redundant!
```

**Benefits:**
- **DRY principle**: Single source of truth for metrics recording
- **Zero maintenance**: New tiers inherit automatically
- **Cache-aware**: Metrics updated even when test doesn't re-run
- **Fail-safe**: Metrics errors don't break tests
- **Thread-safe**: Works with parallel execution

**Metrics Schema:**
```python
{
    "exit_code": 0,              # 0=pass, 1=fail
    "runtime_sec": 2.45,         # Wall-clock time
    "peak_cpu_percent": 85.0,    # Peak CPU usage
    "peak_memory_mb": 450.0,     # Peak RAM usage
    "peak_gpu_memory_mb": 2048.0,# Peak GPU memory
    "timestamp": "2025-11-10T21:26:00Z"  # ISO-8601 UTC
}
```

**Testing:**
- Unit tests: `workspace/tests/test_metrics_recording.py`
- Integration coverage: Metrics automatically recorded in all tier test runs
- Validation: Run `analysis/analyze_cache_metrics_consistency.py` to check metrics coverage

**Migration:**
- Tiers 1-4: Removed redundant manual `record_run()` calls (2025-11-10)
- Tiers 5-7: Never had manual recording, now automatically covered
- **NO action needed for new tiers** - metrics just work!

---

## Implementation Notes

**Location:** `workspace/src/harness/lfm_test_harness.py`

**Key Methods:**
- `run_with_standard_wrapper()`: Entry point for all test executions, calls metrics recording
- `_extract_metrics_for_tracking()`: Extracts metrics from result objects or summary.json

**Safety Net:** `run_parallel_suite.py` includes backup metrics recording for edge cases.

**Last Updated:** 2025-11-10
