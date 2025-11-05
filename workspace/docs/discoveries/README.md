# Discoveries (workspace/docs/discoveries)

Purpose
- Curate and track emergent phenomena discovered during experiments.
- Provide a single place to aggregate claims, timestamps, and promotion status.

How this folder feeds other documentation and processes
- Promotion gates: Gate 2 (Validated) records discoveries and timestamps referenced from `experiments/README.md`.
- Upload packages: Summaries of discoveries are included in `uploads/osf/` and/or `uploads/zenodo/` packages.
- Cross-link to evidence: Each discovery should reference corroborating artifacts in `docs/evidence/`.

Recommended files
- `discoveries.json` — machine-readable index with entries like:
  {
    "id": "emergent_gravity_orbiting_mass",
    "title": "Stable orbit emergence in LFM",
    "date": "2025-11-05",
    "gate": 2,
    "evidence": ["../evidence/orbit_period_plot.png"],
    "notes": "Summary of validation criteria and thresholds"
  }
- `notes.md` — human-readable narrative and rationale.

Conventions
- Keep entries concise and reference reproducible tests in `tests/`.
- Use ISO-8601 dates and stable IDs.

Related docs
- Experiments: `experiments/README.md` (gates & promotion)
- Evidence: `docs/evidence/`
- Uploads: `uploads/` (discoveries summarized in public packages)
