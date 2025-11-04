# Upload Directory Restructuring Summary

## Changes Made

### 1. Directory Structure Change
**OLD Structure:**
```
upload/
├── [COMMON FILES] - Shared files in root
├── zenodo/
│   ├── zenodo_metadata.json
│   └── ZENODO_UPLOAD.zip  (bundle of common files)
└── osf/
    ├── osf_project.json
    └── OSF_UPLOAD.zip  (bundle of common files)
```

**NEW Structure:**
```
upload/
├── zenodo/  (FLAT - no subdirectories)
│   ├── README.md
│   ├── EXECUTIVE_SUMMARY.md
│   ├── MASTER_DOCUMENT.md
│   ├── CORE_EQUATIONS.md
│   ├── TEST_DESIGN.md
│   ├── RESULTS_COMPREHENSIVE.md
│   ├── ELECTROMAGNETIC_ACHIEVEMENTS.md
│   ├── EVIDENCE_REVIEW.md
│   ├── Executive_Summary.pdf
│   ├── LFM_Master.pdf
│   ├── LFM_Core_Equations.pdf
│   ├── LFM_Phase1_Test_Design.pdf
│   ├── LFM_Comprehensive_Report.pdf
│   ├── LICENSE
│   ├── NOTICE
│   ├── CITATION.cff  (NOW INCLUDED!)
│   ├── THIRD_PARTY_LICENSES.md
│   ├── plot_relativistic_dispersion.png  (flattened naming)
│   ├── plot_quantum_interference.png
│   ├── plot_quantum_bound_states.png
│   ├── PLOTS_OVERVIEW.md
│   ├── DISCOVERIES_OVERVIEW.md
│   ├── MANIFEST.md
│   └── zenodo_metadata.json
└── osf/  (FLAT - no subdirectories)
    ├── [Same 23 files as zenodo/]
    └── osf_project.json
```

### 2. Key Architectural Changes

#### A. Destination Types
- **Removed:** `destination='common'`
- **Added:** `destination='both'`
- **Kept:** `destination='zenodo'`, `destination='osf'`

Files marked as `destination='both'` are **duplicated** into both `zenodo/` and `osf/` directories.

#### B. Flat File Structure
- **No subdirectories allowed** in zenodo/ or osf/ (Zenodo requirement)
- Plot files renamed from `plots/filename.png` → `plot_filename.png`
- All files go directly into platform folders

#### C. ZIP Bundles Removed
- Removed `ZENODO_UPLOAD.zip` and `OSF_UPLOAD.zip`
- Each platform folder now contains **complete, ready-to-upload** file set
- No need to unzip before uploading

#### D. CITATION.cff Now Included
- Added `CITATION.cff` to the schema as a required file
- Will be duplicated into both zenodo/ and osf/ directories
- Ensures proper citation metadata in upload packages

### 3. Modified Files

#### `workspace/tools/upload_metadata_schema.py`
- Updated directory structure documentation
- Changed `destination` type from `Literal['common', 'zenodo', 'osf']` to `Literal['zenodo', 'osf', 'both']`
- Replaced all `destination='common'` with `destination='both'`
- Added `CITATION.cff` as required file with `destination='both'`
- Flattened plot filenames (removed `plots/` prefix)
- Removed ZIP bundle file entries
- Updated helper methods:
  - Removed: `get_common_files()`
  - Added: `get_both_platform_files()`, `get_all_zenodo_files()`, `get_all_osf_files()`

#### `workspace/tools/metadata_driven_builder.py`
- Updated `_build_common_files()` to `_build_common_files()` (kept name but changed logic)
  - Now builds 'both' files into **both** zenodo/ and osf/ directories
- Updated `_build_single_file()` signature:
  - Added `dest_filename` parameter (full path like 'zenodo/README.md')
  - Added `source_filename` parameter (display name like 'README.md')
- Deprecated ZIP bundle creation methods:
  - `_create_zenodo_bundle()` - now returns True with warning
  - `_create_osf_bundle()` - now returns True with warning
  - Removed `_create_platform_bundle()` entirely

### 4. Benefits

1. **Zenodo Compliance:** Flat structure meets Zenodo's "no subdirectories" requirement
2. **Upload Simplicity:** Each folder is drag-and-drop ready for upload
3. **No Unzipping:** Users/uploaders don't need to unzip bundles
4. **Citation Included:** CITATION.cff now properly included in both upload packages
5. **Consistency:** Both platforms use identical flat structure
6. **Clarity:** No confusion about what files go where

### 5. File Counts

- **Shared files (both platforms):** 23 files
- **Zenodo-only:** 1 file (zenodo_metadata.json)
- **OSF-only:** 1 file (osf_project.json)
- **Total in zenodo/:** 24 files
- **Total in osf/:** 24 files
- **Total unique source files:** 25 files

### 6. Migration Impact

Existing upload builds will need to be regenerated with the new structure:
```bash
cd c:\LFM\workspace_test
python tools/metadata_driven_builder.py
```

This will create the new flat structure with CITATION.cff included.

### 7. Validation

Schema validation successful:
```
✓ 23 shared files defined
✓ 1 Zenodo-only file (metadata)
✓ 1 OSF-only file (metadata)
✓ All files have proper source mappings
✓ CITATION.cff included as required file
✓ No subdirectories in file paths
```

---

**Status:** ✅ COMPLETE
**Date:** 2025-11-04
**Impact:** Major restructuring - requires upload rebuild
