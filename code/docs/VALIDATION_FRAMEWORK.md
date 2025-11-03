# LFM Validation Framework

## Overview

The LFM validation framework ensures data integrity across the entire results generation and upload pipeline:

```
Test Execution → Results Artifacts → MASTER_TEST_STATUS.csv → Upload Package → Manifest
```

## Quick Start

```powershell
# Validate entire pipeline (recommended)
python tools/validate_results_pipeline.py --all

# Validate specific components
python tools/validate_results_pipeline.py --test REL-01      # Single test
python tools/validate_results_pipeline.py --tier 1           # Tier 1 tests
python tools/validate_results_pipeline.py --master-status    # CSV integrity
python tools/validate_results_pipeline.py --upload-package   # Upload readiness

# Strict mode (fail on warnings)
python tools/validate_results_pipeline.py --all --strict

# Quiet mode (summary only)
python tools/validate_results_pipeline.py --all --quiet
```

## Validation Levels

### 1. Test Result Validation (`--test`, `--tier`)

Validates individual test result directories:

**Checked:**
- ✅ `summary.json` exists and is valid JSON
- ✅ Required fields: `test_id` or `id`, `description`
- ✅ Status field: `status` OR `passed` must be present
- ✅ Status values properly normalized (case-insensitive)
- ✅ `readme.txt` exists (auto-generated)
- ℹ️ Optional artifacts: `metrics.csv`, `plots/`, `diagnostics/`

**Example:**
```bash
python tools/validate_results_pipeline.py --test REL-01
python tools/validate_results_pipeline.py --tier 3
```

### 2. Master Status Validation (`--master-status`)

Validates `MASTER_TEST_STATUS.csv` integrity:

**Checked:**
- ✅ CSV file exists and is parsable
- ✅ All tests in `results/` are reflected in CSV
- ⚠️ No orphaned tests (in CSV but not in results/)
- ✅ Status values properly normalized (`PASS`, `FAIL`, `SKIP`, `UNKNOWN`)
- ✅ Test counts match actual results
- ✅ Category summaries accurate

**Example:**
```bash
python tools/validate_results_pipeline.py --master-status
```

### 3. Upload Package Validation (`--upload-package`)

Validates upload package completeness:

**Checked:**
- ✅ Required core files present (`LICENSE`, `NOTICE`, `README.md`, etc.)
- ✅ Evidence DOCX files (4 required)
- ✅ Comprehensive PDF exists and is recent (<24 hours)
- ✅ MANIFEST.md exists with valid checksums
- ✅ Metadata files are valid JSON (`zenodo_metadata.json`, `osf_metadata.json`)
- ✅ `results_MASTER_TEST_STATUS.csv` present
- ℹ️ Plots, MD files, TXT files properly staged

**Example:**
```bash
python tools/validate_results_pipeline.py --upload-package
```

### 4. End-to-End Validation (`--all`)

Comprehensive validation across all components:

**Workflow:**
1. Validates all tier results (summary.json, readme.txt, etc.)
2. Validates MASTER_TEST_STATUS.csv integrity
3. Validates upload package completeness
4. Cross-checks consistency (results/ CSV vs. upload/ CSV)

**Example:**
```bash
python tools/validate_results_pipeline.py --all
```

## Integration with Workflow

### After Running Tests

Always validate after running tests to ensure results are properly captured:

```powershell
# Run a test
python run_tier1_relativistic.py --test REL-01

# Validate results
python tools/validate_results_pipeline.py --test REL-01

# Rebuild upload package
python tools/build_upload_package.py

# Validate upload integrity
python tools/validate_results_pipeline.py --upload-package
```

### Before Uploading to OSF/Zenodo

Run full validation before uploading:

```powershell
# Full validation
python tools/validate_results_pipeline.py --all --strict

# If passed, rebuild with ZIP
python tools/build_upload_package.py

# Upload docs/upload/ contents to OSF/Zenodo
```

### Automated CI Integration

Add to CI pipeline (GitHub Actions, etc.):

```yaml
- name: Validate Results Pipeline
  run: python tools/validate_results_pipeline.py --all --strict
```

## Exit Codes

- `0` - All checks passed
- `1` - Errors found (validation failed)
- `2` - Warnings found (strict mode only)

## Common Issues and Fixes

### Issue: Missing `test_id` field

**Error:** `summary.json missing 'test_id' or 'id' field`

**Fix:** Ensure test harness writes `test_id` or `id` to summary.json:
```python
summary = {
    "test_id": "REL-01",  # or "id": "REL-01"
    "description": "Test description",
    "status": "PASS"  # or "passed": True
}
```

### Issue: Status not normalized

**Error:** `Test has non-normalized status 'Passed'`

**Fix:** The validator automatically normalizes status values, but for consistency use uppercase in CSV:
- `"status": "PASS"` (preferred)
- `"status": "Passed"` (accepted, normalized to PASS)
- `"passed": True` (accepted, converted to PASS)

### Issue: MASTER_TEST_STATUS.csv out of sync

**Error:** `Test exists in results/ but not in MASTER_TEST_STATUS.csv`

**Fix:** Regenerate master status:
```python
from lfm_results import update_master_test_status
update_master_test_status()
```

### Issue: Upload CSV mismatch

**Error:** `results/MASTER_TEST_STATUS.csv does not match upload/results_MASTER_TEST_STATUS.csv`

**Fix:** Rebuild upload package:
```powershell
python tools/build_upload_package.py
```

### Issue: Comprehensive PDF stale

**Warning:** `Comprehensive PDF is XX hours old (may be stale)`

**Fix:** Regenerate PDF:
```powershell
python tools/build_comprehensive_pdf.py
```

## Validator Architecture

### Key Components

1. **`PipelineValidator`** - Main validation engine
   - Tracks errors, warnings, and info messages
   - Provides detailed logging with emoji indicators
   - Supports strict mode (warnings become errors)

2. **Validation Methods**
   - `validate_test_result()` - Single test validation
   - `validate_tier_results()` - Tier-level validation
   - `validate_master_status_integrity()` - CSV validation
   - `validate_upload_package()` - Upload readiness
   - `validate_end_to_end()` - Full pipeline validation

3. **Reporting**
   - Colored output (✅ ❌ ⚠️ ℹ️)
   - Summary report with counts
   - Exit codes for CI integration

### Extensibility

Add custom validation checks:

```python
def validate_custom_check(self) -> bool:
    """Add custom validation logic."""
    if condition_fails:
        self.log("Custom check failed", 'error')
        return False
    return True
```

## Best Practices

1. **Run validation after every test execution**
2. **Use `--strict` mode in CI pipelines**
3. **Validate before uploading to repositories**
4. **Keep comprehensive PDF updated (<24 hours)**
5. **Regenerate master status after bulk test runs**

## Status Normalization

The framework accepts various status formats and normalizes them:

| Input Format | Normalized | Notes |
|--------------|------------|-------|
| `"status": "PASS"` | `PASS` | Preferred format |
| `"status": "Passed"` | `PASS` | Accepted |
| `"status": "Pass"` | `PASS` | Accepted |
| `"passed": true` | `PASS` | Boolean format |
| `"status": "FAIL"` | `FAIL` | Preferred format |
| `"status": "Failed"` | `FAIL` | Accepted |
| `"passed": false` | `FAIL` | Boolean format |
| `"status": "SKIP"` | `SKIP` | Skipped tests |
| `"status": "UNKNOWN"` | `UNKNOWN` | Indeterminate |

## Future Enhancements

- [ ] Pre-commit hook integration
- [ ] GitHub Actions workflow
- [ ] Automatic fix mode (repair common issues)
- [ ] JSON/XML report output for CI
- [ ] Performance regression detection
- [ ] Checksum verification for all files in MANIFEST.md
- [ ] Validation of plot file contents (not just existence)

## Support

For issues or questions about the validation framework:
1. Check error messages carefully (they include specific file/field names)
2. Review this documentation
3. Run with `--quiet` removed for verbose output
4. Check `summary.json` field names match validator expectations
