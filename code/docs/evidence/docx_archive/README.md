# LFM Master Documents Archive

This directory contains archived .docx files that are no longer actively maintained.

## Archive Contents

**File:** `LFM_Master_Documents_Archive_20251103.zip`
**Date:** November 3, 2025
**Contents:**
- Executive_Summary.docx
- LFM_Core_Equations.docx  
- LFM_Master.docx
- LFM_Phase1_Test_Design.docx

## Why Archived?

The .docx files were archived because:
1. **Maintenance burden** - Binary files are hard to version control
2. **Sync issues** - Keeping .docx and .txt versions in sync was error-prone
3. **Citation updates** - Academic attribution changes were difficult to maintain
4. **Development friction** - Slowed down documentation workflow

## Current Workflow

**Source of Truth:** Text files in `../docx_text/` directory
- `LFM_Master.txt`
- `LFM_Core_Equations.txt` 
- `Executive_Summary.txt`
- `LFM_Phase1_Test_Design.txt`

**For Final Documents:** Use pandoc or build tools to generate .docx/.pdf from markdown

## Academic Updates Applied

The archived .docx files DO NOT include the Klein-Gordon citations that were added on November 3, 2025. The updated text files in `../docx_text/` include proper attribution to:
- Klein, O. (1926). Quantentheorie und fünfdimensionale Relativitätstheorie. Zeitschrift für Physik, 37(12), 895-906.
- Gordon, W. (1926). Der Comptoneffekt nach der Schrödingerschen Theorie. Zeitschrift für Physik, 40(1-2), 117-133.

## Future Document Generation

To generate updated .docx files with current content:
```bash
# Use pandoc or the build_master_docs.py tool
cd ../../..
python tools/build_master_docs.py
```

---
*Archive created: November 3, 2025*  
*Reason: Transition to markdown-first documentation workflow*