# Archive (workspace/archive)

Purpose
- Long-term storage for deprecated, superseded, or paused work that we still want to preserve for provenance.
- Keeps the active workspace focused while maintaining a complete historical record.

What belongs here
- Old analysis notebooks/scripts no longer maintained but referenced by docs.
- Prior app prototypes, visualization drafts, and temporary tools.
- Frozen results supporting claims in docs/evidence or docs/discoveries.

How this folder feeds other documentation
- Evidence linkage: archived results may be cited by files in `docs/evidence/`.
- Discovery record: when experiments are archived, their notes often inform entries in `docs/discoveries/`.
- Upload packaging: if an archived artifact is part of a release, it gets referenced by upload manifests under `uploads/`.

Governance
- Do not delete; move here instead to preserve reproducibility.
- Add a short COMMENT.md alongside large artifacts that explains why it was archived and the replacement location.
- If an archived item supports a paper/report, include a minimal manifest (inputs, parameters, outputs, commit hash).

Related docs
- Root: `DIRECTORY_STRUCTURE.md`
- Experiments: `experiments/README.md` (archive lane for retired investigations)
- Uploads: `uploads/` (packages may cite archived data)
