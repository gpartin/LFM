# Deterministic Upload Package Build System — Design Document

**Status**: PROPOSED  
**Created**: 2025-11-06  
**Author**: GitHub Copilot (AI Agent)  
**Purpose**: Make OSF/Zenodo upload package generation fully deterministic and extensible

---

## Problem Statement

Current `workspace/tools/build_upload_package.py` has these issues:

1. **Non-deterministic**: Manually edited docs (EXECUTIVE_SUMMARY.md, RESULTS_COMPREHENSIVE.md) become stale
2. **Hardcoded tier configs**: Adding Tier 6 requires changing code in multiple places
3. **No single source of truth**: Tier metadata duplicated across runners, configs, docs
4. **Manual plot selection**: Representative plots copied manually, prone to errors
5. **Fragile**: Changes to test structure break upload generation

**Goal**: Running `python tools/build_upload_package.py` twice with same inputs produces **byte-for-byte identical** outputs.

---

## Architecture Overview

```
INPUT SOURCES (Single Source of Truth)
├── workspace/config/tier_metadata_comprehensive.json   # Tier definitions
├── workspace/results/{category}/{test-id}/summary.json # Test results
├── workspace/docs/text/*.txt                            # Static prose
├── workspace/docs/discoveries/discoveries.json          # Discovery log
└── workspace/VERSION                                    # Version info

    ↓ (Transform)

BUILD PROCESS (workspace/tools/build_upload_package.py)
├── Load tier_metadata_comprehensive.json
├── Scan results/ for actual test outcomes
├── Generate documents from templates + data
├── Copy representative plots (deterministic selection)
└── Write manifests with SHA256 checksums

    ↓ (Deterministic Output)

UPLOAD PACKAGES (Reproducible)
├── workspace/uploads/osf/
│   ├── TIER_1_ACHIEVEMENTS.md          # Generated from tier_metadata + results
│   ├── TIER_2_ACHIEVEMENTS.md
│   ├── TIER_3_ACHIEVEMENTS.md
│   ├── TIER_4_ACHIEVEMENTS.md
│   ├── TIER_5_ACHIEVEMENTS.md
│   ├── RESULTS_COMPREHENSIVE.md        # Generated from tier_metadata + MASTER_TEST_STATUS
│   ├── EXECUTIVE_SUMMARY.md            # Generated from templates + tier pass rates
│   ├── plot_tier1_*.png                # Copied from tier_metadata.representative_plot
│   ├── plot_tier2_*.png
│   ├── plot_tier3_*.png
│   ├── plot_tier4_*.png
│   ├── plot_tier5_*.png
│   ├── results/                        # Complete test results tree
│   └── MANIFEST.md                     # SHA256 checksums for reproducibility
└── workspace/uploads/zenodo/
    └── (same structure as osf/)
```

---

## Key Design Principles

### 1. **Single Source of Truth**

**Tier Metadata**: `workspace/config/tier_metadata_comprehensive.json`

```json
{
  "tiers": [
    {
      "tier_number": 1,
      "tier_name": "Tier 1 — Relativistic",
      "category_dir": "Relativistic",
      "test_prefix": "REL",
      "test_count": 15,
      "description": "...",
      "significance": "...",
      "key_validations": ["..."],
      "representative_plot": {
        "test_id": "REL-13",
        "source_path": "results/Relativistic/REL-13/plots/dispersion_REL-13.png",
        "dest_filename": "plot_tier1_relativistic_dispersion.png"
      }
    },
    ...
  ]
}
```

**Adding Tier 6**: Just add entry to JSON, no code changes needed.

### 2. **Template-Based Document Generation**

Instead of manually editing `EXECUTIVE_SUMMARY.md`:

```python
def generate_executive_summary(dest_dir, tier_metadata, test_results, deterministic=False):
    """Generate EXECUTIVE_SUMMARY.md from template + data."""
    template = load_template("executive_summary.md.j2")
    
    # Calculate aggregate stats
    total_tests = sum(t['test_count'] for t in tier_metadata['tiers'])
    total_passed = sum(count_passed_tests(t['category_dir'], test_results) 
                       for t in tier_metadata['tiers'])
    
    # Render with Jinja2 or simple string replacement
    content = template.render(
        header=tier_metadata['document_templates']['executive_summary_header'],
        overview=tier_metadata['document_templates']['executive_summary_overview'],
        tiers=tier_metadata['tiers'],
        total_tests=total_tests,
        total_passed=total_passed,
        pass_percentage=100.0 * total_passed / total_tests,
        generated_date=deterministic_timestamp() if deterministic else datetime.now()
    )
    
    dest_path = dest_dir / "EXECUTIVE_SUMMARY.md"
    dest_path.write_text(content, encoding='utf-8')
    return dest_path
```

**Benefits**:
- Update pass rates automatically when tests change
- No manual editing = no staleness
- Version-controlled templates separate from dynamic data

### 3. **Deterministic Timestamps**

For reproducibility (e.g., CI/CD verification):

```python
def deterministic_timestamp():
    """Return fixed timestamp from SOURCE_DATE_EPOCH or config."""
    return os.environ.get('SOURCE_DATE_EPOCH', '1970-01-01 00:00:00')
```

**Production use**: Set `SOURCE_DATE_EPOCH` to actual publication date:
```bash
export SOURCE_DATE_EPOCH=$(date +%s)
python tools/build_upload_package.py --deterministic
```

### 4. **Automatic Plot Selection**

From `tier_metadata_comprehensive.json`:

```python
def copy_representative_plots(dest_dir, tier_metadata, results_dir):
    """Copy tier plots specified in metadata."""
    for tier in tier_metadata['tiers']:
        plot_config = tier['representative_plot']
        source = results_dir / plot_config['source_path']
        dest = dest_dir / plot_config['dest_filename']
        
        if source.exists():
            shutil.copy2(source, dest)
            print(f"Copied {tier['tier_name']} plot: {plot_config['dest_filename']}")
        else:
            print(f"[WARN] Missing representative plot for {tier['tier_name']}: {source}")
```

**Adding Tier 6 plot**: Just update JSON, script automatically copies it.

### 5. **Extensibility Without Code Changes**

**Current** (code change required):
```python
# In build_upload_package.py line 1480
tier_configs = [
    {"tier_name": "Tier 1 — Relativistic", ...},
    {"tier_name": "Tier 2 — Gravity Analogue", ...},
    # Add Tier 6 here <-- CODE CHANGE
]
```

**Proposed** (config change only):
```python
# In build_upload_package.py
tier_configs = load_tier_metadata()['tiers']  # Reads JSON
# No code change when adding Tier 6!
```

---

## Implementation Plan

### Phase 1: Extract Hardcoded Metadata to JSON ✓

- [x] Create `workspace/config/tier_metadata_comprehensive.json`
- [ ] Validate JSON schema and field completeness
- [ ] Add unit tests for JSON loading

### Phase 2: Refactor build_upload_package.py

- [ ] Add `load_tier_metadata()` function
- [ ] Replace hardcoded `tier_configs` list with JSON load
- [ ] Update `generate_tier_achievements()` to use JSON
- [ ] Update `copy_representative_plots()` to use JSON

### Phase 3: Template-Based Document Generation

- [ ] Create `workspace/tools/templates/` directory
- [ ] Create `executive_summary.md.j2` template
- [ ] Create `results_comprehensive.md.j2` template
- [ ] Implement `generate_from_template()` function
- [ ] Replace manual document editing with template rendering

### Phase 4: Validation and Testing

- [ ] Run `build_upload_package.py` twice, verify byte-identical output
- [ ] Add `--verify-deterministic` flag to automate this check
- [ ] Create test: add dummy Tier 6, verify it appears in all docs
- [ ] Document how to add new tiers (just edit JSON)

### Phase 5: Pipeline Audit

- [ ] Trace where tier runners write summary.json
- [ ] Verify all runners use same schema
- [ ] Ensure plot filenames match tier_metadata expectations
- [ ] Document test result schema in tier_metadata.json

---

## Migration Strategy

1. **Parallel Implementation**: Build new system alongside existing
2. **Validation**: Generate uploads with both systems, diff outputs
3. **Switchover**: Add `--use-metadata-json` flag, test thoroughly
4. **Deprecation**: Remove old hardcoded approach after validation
5. **Documentation**: Update README with new workflow

---

## Benefits

✅ **Deterministic**: Same input → same output (byte-for-byte)  
✅ **Extensible**: Add Tier 6 by editing JSON, no code changes  
✅ **Auditable**: Single source of truth for all tier metadata  
✅ **Maintainable**: Templates separate structure from content  
✅ **Testable**: Easy to verify reproducibility in CI/CD  
✅ **Professional**: Publication-quality systematic approach  

---

## Open Questions for User

1. **Template Engine**: Use Jinja2, Mako, or simple string replacement?
2. **Validation**: Add JSON schema validation with `jsonschema` package?
3. **Backwards Compatibility**: Keep old hardcoded path as fallback during migration?
4. **Test Runner Integration**: Should runners read tier_metadata.json too?
5. **Plot Selection Logic**: If plot missing, fail or fallback to another test's plot?

---

## Next Steps

**Immediate** (this session):
1. ✅ Remove empty `plots/` directories
2. ✅ Create comprehensive tier metadata JSON design
3. ⏳ Get user approval on design
4. ⏳ Begin Phase 2 refactoring

**Short-term** (next session):
1. Implement template-based document generation
2. Add deterministic build validation
3. Test adding dummy Tier 6

**Long-term** (future work):
1. Integrate tier_metadata.json into test runners
2. Auto-generate tier runner scripts from metadata
3. Build schema validator for test results

---

## File Changes Summary

**New Files**:
- `workspace/config/tier_metadata_comprehensive.json` (master tier config)
- `workspace/tools/templates/executive_summary.md.j2` (Jinja2 template)
- `workspace/tools/templates/results_comprehensive.md.j2` (Jinja2 template)
- `workspace/tools/DETERMINISTIC_BUILD_DESIGN.md` (this document)

**Modified Files**:
- `workspace/tools/build_upload_package.py` (load JSON, generate from templates)

**Deleted**:
- `workspace/uploads/osf/plots/` (empty directory)
- `workspace/uploads/zenodo/plots/` (empty directory)

---

## References

- **Reproducible Builds**: https://reproducible-builds.org/
- **Jinja2 Templates**: https://jinja.palletsprojects.com/
- **JSON Schema**: https://json-schema.org/
- **SOURCE_DATE_EPOCH**: https://reproducible-builds.org/docs/source-date-epoch/
