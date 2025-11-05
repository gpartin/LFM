# Evidence (workspace/docs/evidence)

Purpose
- Central repository for figures, tables, datasets, logs, and other artifacts that support claims.
- Decouples raw experiment outputs from curated, publication-ready evidence.

How this folder feeds other documentation and processes
- Upload packages: Selected artifacts are included in `uploads/osf/` and `uploads/zenodo/` builds.
- Discoveries: Entries in `docs/discoveries/` should link to specific evidence files here.
- Reports: Technical docs in `docs/` and root-level reports reference evidence paths.

Organization
- `figures/` — finalized images (PNG/SVG/PDF) with clear captions in metadata or adjacent `.md` files.
- `tables/`  — CSV/JSON tables with schema notes; include data dictionary.
- `datasets/` — cleaned data ready for analysis; raw data should remain with experiments.
- `logs/` — run logs, environment manifests, hardware info.

Provenance requirements
- Record origin: experiment name, commit hash, parameters, date.
- Keep transformations reproducible: include the script/notebook that produced each artifact.
- UTF-8 encoding for all text-based files (see `CODING_STANDARDS.md`).

Related docs
- Discoveries: `docs/discoveries/`
- Experiments: `experiments/`
- Uploads: `uploads/`
