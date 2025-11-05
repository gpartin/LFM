# Upload Directory Restructuring - Quick Reference

## Summary
**Changed upload structure from common+ZIP bundles to flat platform folders with CITATION.cff included.**

## New Structure
```
upload/
├── zenodo/  [24 files - ALL flat, no subdirs]
│   ├── [23 shared content files]
│   └── zenodo_metadata.json
└── osf/     [24 files - ALL flat, no subdirs]
    ├── [23 shared content files]
    └── osf_project.json
```

## Files in Each Platform Folder

### Markdown Documentation (8 files)
- README.md
- EXECUTIVE_SUMMARY.md
- MASTER_DOCUMENT.md
- CORE_EQUATIONS.md
- TEST_DESIGN.md
- RESULTS_COMPREHENSIVE.md
- ELECTROMAGNETIC_ACHIEVEMENTS.md
- EVIDENCE_REVIEW.md

### PDF Documents (5 files)
- Executive_Summary.pdf
- LFM_Master.pdf
- LFM_Core_Equations.pdf
- LFM_Phase1_Test_Design.pdf
- LFM_Comprehensive_Report.pdf

### Legal & Citation (4 files)
- LICENSE
- NOTICE
- **CITATION.cff** ← NOW INCLUDED!
- THIRD_PARTY_LICENSES.md

### Visualizations (3 files, flattened naming)
- plot_relativistic_dispersion.png
- plot_quantum_interference.png
- plot_quantum_bound_states.png

### Overview Documents (3 files)
- PLOTS_OVERVIEW.md
- DISCOVERIES_OVERVIEW.md
- MANIFEST.md

### Platform Metadata (+1 unique per folder)
- zenodo/zenodo_metadata.json (Zenodo only)
- osf/osf_project.json (OSF only)

## Key Changes
1. ✅ No files in upload/ root
2. ✅ All files flat (no subdirectories)
3. ✅ CITATION.cff included in both platforms
4. ✅ No ZIP bundles needed
5. ✅ Ready for direct drag-and-drop upload

## How to Rebuild
```bash
cd C:\LFM\workspace_test
python tools/metadata_driven_builder.py
```

## Verification Commands
```bash
# Count files in zenodo/
Get-ChildItem c:\LFM\workspace_test\docs\upload\zenodo -File | Measure-Object

# Count files in osf/
Get-ChildItem c:\LFM\workspace_test\docs\upload\osf -File | Measure-Object

# Check for CITATION.cff
Test-Path c:\LFM\workspace_test\docs\upload\zenodo\CITATION.cff
Test-Path c:\LFM\workspace_test\docs\upload\osf\CITATION.cff
```

## Why This Change?
**Original Problem:** CITATION.cff was missing from OSF and Zenodo upload packages because:
1. It wasn't in the upload metadata schema
2. ZIP bundles were excluding it inadvertently

**Solution:** 
- Added CITATION.cff to schema as required file
- Changed structure to flat folders (eliminates ZIP bundle confusion)
- Meets Zenodo's "no subdirectories" requirement
- Simplifies upload process (no unzipping needed)

---
**Updated:** 2025-11-04
**Schema Version:** 1.0.0
**Status:** ✅ Ready for implementation
