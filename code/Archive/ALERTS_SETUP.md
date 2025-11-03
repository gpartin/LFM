# Google Alerts and Scholar Alerts setup

This guide helps you set up monitoring for mentions of the project, DOIs, and key files.

## Before you start
- Sign in to your Google account in your default browser
- Open: https://www.google.com/alerts
- For each link below, click it, review the preview, then click "Create alert"
- After creating each alert, edit the settings:
  - Frequency: At most once a week
  - Sources: Automatic
  - Language: Any (or your preference)
  - Region: Any region
  - How many: All results
  - Deliver to: your project email

## One-click alert links (pre-filled queries)
Click each link; it opens Google Alerts with the query pre-filled.

Project names and core phrases:
- LFM lattice field model: https://www.google.com/alerts?hl=en&q=%22LFM%20lattice%20field%20model%22
- Lorentzian Field Model: https://www.google.com/alerts?hl=en&q=%22Lorentzian%20Field%20Model%22
- Lattice Field Model Lorentzian: https://www.google.com/alerts?hl=en&q=%22Lattice%20Field%20Model%22%20%22Lorentzian%22

Key file names (copies and forks often preserve filenames):
- lfm_equation.py: https://www.google.com/alerts?hl=en&q=%22lfm_equation.py%22
- lfm_parallel.py: https://www.google.com/alerts?hl=en&q=%22lfm_parallel.py%22
- lfm_simulator.py: https://www.google.com/alerts?hl=en&q=%22lfm_simulator.py%22

Repository identifiers:
- gpartin LFM (GitHub owner + repo): https://www.google.com/alerts?hl=en&q=%22gpartin%22%20%22LFM%22
- gpartin Lattice Field Model: https://www.google.com/alerts?hl=en&q=%22gpartin%22%20%22Lattice%20Field%20Model%22

Prior-art records (OSF/Zenodo):
- OSF record id (6agn8): https://www.google.com/alerts?hl=en&q=%226agn8%22%20osf.io
- Zenodo record id (17478758): https://www.google.com/alerts?hl=en&q=%2217478758%22%20zenodo.org
- OSF URL: https://www.google.com/alerts?hl=en&q=%22osf.io%2F6agn8%22
- Zenodo URL: https://www.google.com/alerts?hl=en&q=%22zenodo.org%2Frecords%2F17478758%22

Optional focused queries (reduce noise from your own pages):
- Lorentzian Field Model (exclude your GitHub/OSF/Zenodo):
  https://www.google.com/alerts?hl=en&q=%22Lorentzian%20Field%20Model%22%20-site%3Agithub.com%20-site%3Aosf.io%20-site%3Azenodo.org

Tip: After creating alerts, you can edit each to set "Deliver to" as a weekly digest.

## Google Scholar alerts (citations and references)
For DOIs and papers/records, set Scholar alerts:
1. Go to https://scholar.google.com and sign in
2. Search for your Zenodo DOI or title; open the record page
3. Click the envelope icon ("Create alert") to get notified on new citations or related results
4. Repeat for your OSF record title

If you don’t have the DOI yet, create an alert on the Zenodo record title and your author name(s).

## Optional: site-specific alerts
- GitHub code search RSS (3rd-party or GitHub Actions) can be used, but Google Alerts usually catches public mirrors and posts.
- You can also add queries for “commercial license”, “for sale”, or “SaaS” combined with your project name if you want early signals of misuse.

## Verification checklist
- [ ] Alerts created for the 3 project/name phrases
- [ ] Alerts created for 2–3 key filenames
- [ ] Alerts created for OSF/Zenodo identifiers
- [ ] Frequency set to weekly; delivery to project email
- [ ] Scholar alerts created for DOI/title
