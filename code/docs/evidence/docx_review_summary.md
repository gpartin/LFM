# .docx review summary (automated scan)

Date: 2025-11-01
Scope: docs/evidence/docx_text/ (12 extracted files)

This summarizes potential legal/IP signals detected in the extracted text. Human review recommended for the flagged lines.

## Key findings

1) License inconsistencies across documents
- Some documents declare: “Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (CC BY-NC-ND 4.0)”.
  - Examples: `LFM_Master.txt` (lines ~4, 101–103), `LFM_Core_Equations.txt` (lines ~4, 104–106), `LFM_Phase1_Test_Design.txt` (lines ~4, 85–87)
- Others or the repository license indicate: CC BY-NC-ND 4.0 (no ND)
  - Examples: `LFM_Phase1_Test_Design.txt` (line ~70), OSF Main Paper shows CC BY-NC-ND 4.0
- Several documents include “All rights reserved.” lines (e.g., `LFM_Master.txt` ~99, `LFM_Phase1_Test_Design.txt` ~83, `LFM_Core_Equations.txt` ~103), which can conflict with Creative Commons terms.

Action: Choose a single policy for prose (CC BY-NC vs CC BY-NC-ND) and harmonize headers across all documents; remove or reword “All rights reserved.” to avoid conflict. See `docs/LEGAL_SNIPPETS.md` for ready-to-paste headers.

2) Defensive publication statements present (good)
- Executive_Summary, LFM_Phase1_Test_Design, LFM_Master, LFM_Core_Equations include a “Defensive Publication Statement” and define an effective date (e.g., Oct 29, 2025).

Action: Ensure the date matches your OSF/Zenodo first-publication timestamps; cross-link URLs/DOIs in final versions.

3) PII – public contact email present
- Email found: latticefieldmediumresearch@gmail.com (multiple documents and metadata).

Action: If you prefer a project email/alias, update to that. Otherwise acceptable.

4) Patent/novelty language
- Several passages discuss “novelty” and unifying claims. No problematic promises detected; statements appear descriptive.

Action: Ensure claims remain factual and qualified; consider adding a universal “No warranties / As-is” disclaimer to all documents.

5) Confidentiality / NDA language
- Reference to sending a "peer brief under NDA" (historical plan). No document marked Confidential.

Action: None required unless an actual confidential draft exists; keep public releases free of confidentiality marks.

6) Warranty/liability
- No explicit warranties or liability statements detected; no "as-is" either.

Action: Add a standard disclaimer paragraph (see LEGAL_SNIPPETS.md) to prevent implied warranties.

## File inventory analyzed
- Executive_Summary.txt
- LFM_Core_Equations.txt
- LFM_Master.txt
- LFM_Phase1_Test_Design.txt
- OSF/contents/LFM_Main_Paper.txt
- OSF/contents/LFM_OSF_Metadata.txt
- OSF/contents/LFM_OSF_Abstract.txt
- OSF/contents/LFM_OSF_Methods.txt
- Version1/* duplicates (prior draft set)

## Recommended next steps
- Decide on CC BY-NC vs CC BY-NC-ND for prose and standardize all document headers.
- Insert the "No Commercial Use" and "As-is, No Warranty" snippets where appropriate.
- Replace personal email with project alias if desired.
- Add cross-links (GitHub public URL, OSF/Zenodo DOIs) before finalizing.

If you want, I can generate redline-ready text blocks per document with exact line replacements.
