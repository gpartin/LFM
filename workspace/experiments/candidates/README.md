# Candidates (workspace/experiments/candidates)

Purpose
- Home for experiments that passed Gate 0 and are under Gate 1 reproducibility review.
- Stabilize code, control randomness, and document dependencies before promotion to tests.

Promotion criteria (Gate 1 → Gate 2)
- Script is non-interactive and reproducible across machines.
- Variability characterized (seeds fixed or variance reported).
- Dependencies pinned and documented.
- Output files generated consistently; compute time documented.

Feeds into other processes
- On promotion, a pytest skeleton is created in `tests/tier{N}/{domain}/`.
- Discovery timeline updated in `docs/discoveries/` (timestamp + summary).
- Any finalized figures/tables move to `docs/evidence/` for publication use.

Tools
- `tools/promote_experiment.py {name}` — move from Gate 0 to here with checklist.
- `tools/promote_candidate.py {name}` — generate test scaffolding for Gate 2.
- `tools/check_promotion_readiness.py {name}` — verify criteria automatically.

Related docs
- Experiments overview: `experiments/README.md`
- Tests: `tests/`
- Discoveries: `docs/discoveries/`
- Evidence: `docs/evidence/`
