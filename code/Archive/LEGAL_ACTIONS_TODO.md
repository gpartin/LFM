# Legal actions checklist (working)

This checklist tracks legal/IP follow-ups. The Git timeline export is deferred until after the repo is public.

## When repo is made public (tomorrow)
- [ ] Export commit timeline to evidence
  - Target outputs:
    - docs/evidence/commit_timeline.csv
    - docs/evidence/commit_timeline.pdf (optional print-friendly)
  - Then upload both files to your OSF and Zenodo records as supplemental prior-art evidence.

Suggested commands (run from the repo root) — adjust as needed:
- CSV: `git log --date=iso8601 --pretty=format:"%h,%ad,%an,%ae,%s" > docs/evidence/commit_timeline.csv`
- Pretty text (for PDF conversion): `git log --stat --date=iso8601 > docs/evidence/commit_timeline.txt` then convert to PDF using your preferred tool.

## Next actions you can do now
- [ ] Register U.S. copyright for the codebase
  - Prepare deposit: current LICENSE, NOTICE, RED_TEAM_LEGAL_REVIEW.md, and a ZIP snapshot of the repo (source only)
  - File online via copyright.gov (TX form)
  - Use OSF/Zenodo DOIs as prior-art references; note first publication date
- [ ] Set up Google Alerts (weekly digest)
  - See: docs/ALERTS_SETUP.md for one-click links and a helper script (tools/open_alert_links.ps1)
  - Terms: "LFM lattice field model", "Lorentzian Field Model", repo owner name + keywords, your DOIs, and key file names (e.g., lfm_equation.py)
- [ ] Confirm service-of-process address
  - Ensure NOTICE lists a current postal address or a registered agent; update if missing
- [ ] Commercial licensing intake (template)
  - Create a short intake (email template or form) for commercial inquiries and link it from README and NOTICE
- [ ] Optional: trademark screening
  - Do a quick USPTO/TESS and web search on the project name; save notes to docs/evidence/trademark_screening.md

## Evidence folder
Place artifacts (exports, PDFs, receipts) under `docs/evidence/` for easy reference and future filings.
