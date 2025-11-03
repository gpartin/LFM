# Reviewing .docx files in this repo

This project includes a helper to extract text from .docx files for quick review.

## Where to put your .docx
- Place Word documents under `documents/` at the repo root (create it if missing)
- You can nest subfolders; the extractor preserves structure

## Run the extractor (no extra dependencies)
From the repo root:

```powershell
python tools/docx_text_extract.py documents docs/evidence/docx_text
```

- Outputs `.txt` files under `docs/evidence/docx_text/` mirroring your folder structure
- If you want to scan a different folder, replace `documents` with its path

## What we review for
- Third-party text or images that may require permission/attribution
- Confidential or sensitive information (PII, internal URLs, access tokens)
- License mismatches (e.g., statements that conflict with the repo LICENSE)
- Claims that could be construed as warranties or guarantees
- Patent-sensitive disclosures (novel method claims) — ensure they’re already disclosed as prior art via OSF/Zenodo

## After extraction
- Let me know and I’ll automatically scan the generated `.txt` files and summarize:
  - Potential legal/IP risks, suggested edits, and exact passages
  - A redline-friendly checklist to apply back to the original .docx

Tip: Keep canonical content in Markdown in this repository when possible; it’s easier to version and review.
