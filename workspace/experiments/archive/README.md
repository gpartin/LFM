# Experiments Archive (workspace/experiments/archive)

Purpose
- Preserve completed, superseded, or abandoned investigations with full context.
- Maintain provenance so results and decisions remain audit-able.

Relationship to the gates
- Gate 0 (Exploratory) work that is discontinued should be moved here with notes.
- Gate 1 (Candidate) work that fails reproducibility or is superseded should be archived here.

What to include when archiving an experiment
- `notes.md` summarizing the final state and rationale for archival.
- `manifest.json` with inputs, parameters, results summary, and commit hash.
- Pointers to any evidence promoted to `docs/evidence/`.

How this folder feeds other documentation and processes
- Discoveries: even archived experiments may contribute observations recorded in `docs/discoveries/`.
- Uploads: archival material may be referenced in historical sections of OSF/Zenodo packages.

Related docs
- Experiments overview: `experiments/README.md`
- Discoveries: `docs/discoveries/`
- Evidence: `docs/evidence/`
