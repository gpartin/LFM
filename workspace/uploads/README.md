# Uploads (workspace/uploads)

Purpose
- Staging area for public-release packages and deposit artifacts (OSF, Zenodo).
- Ensures each release is reproducible with clearly defined provenance.

Structure
- `uploads/osf/` — Open Science Framework package structure and README.
- `uploads/zenodo/` — Zenodo deposit structure and README.

How this folder feeds other documentation and processes
- Pulls source texts and headers/footers from `docs/templates/` and `docs/text/` when building narratives.
- Includes figures and tables assembled from `docs/evidence/` and selected experiment outputs.
- References discovery timelines recorded via promotion gates (see `experiments/README.md`) and any entries maintained in `docs/discoveries/`.
- Built via tooling noted in workspace READMEs (e.g., `tools/build_upload_package.py`).

Minimum contents for a release
- A top-level README with license notice and how to reproduce.
- A manifest of inputs, configuration, and commit hash.
- Links to associated tests in `tests/` and any benchmark notes if performance is reported.

Standards
- License: CC BY-NC-ND 4.0 unless explicitly overridden.
- File encodings are UTF-8 throughout (see `CODING_STANDARDS.md`).

Related docs
- Root: `DIRECTORY_STRUCTURE.md`
- Docs source: `docs/` (templates, text, evidence)
- Upload targets: `uploads/osf/README.md`, `uploads/zenodo/README.md`
