# Upload Evidence Directory

**Note:** .docx files removed on November 3, 2025

## Why Removed?
The .docx files in this directory were outdated (from November 2nd) and did not include the Klein-Gordon citations that were added to the source text files on November 3rd.

## Current Workflow
1. **Source Files:** Text files in `../evidence/docx_text/` 
2. **Generation:** Use build tools to create fresh .docx/.pdf when needed
3. **Upload:** Generate clean, up-to-date documents from text sources

## To Regenerate Documents
```bash
cd ../../..
python tools/build_master_docs.py
```

This will create fresh .docx and .pdf files with current Klein-Gordon citations.

---
*Cleaned: November 3, 2025 - Transition to markdown-first workflow*