# Workspace Results — Experiment Suite Outputs

**Role in hierarchy**: `workspace/` → `experiments/` → `lattice_optimization_suite/` → `{analysis|results}/`

**Purpose**  
Store raw outputs and analysis artifacts from lattice optimization experiments.

**Structure**  
- `analysis/` — post-processing scripts, comparison plots, summary tables
- `results/` — per-algorithm run outputs (timing, energy, validation)

**Lifecycle**  
These support active experiments. If promoted to candidates or tests, curate key artifacts to `workspace/docs/evidence/`.

**Related folders**  
- `workspace/experiments/lattice_optimization_suite/` (parent experiment)
- `workspace/performance/` (performance work may reference these results)
- `DIRECTORY_STRUCTURE.md` (root map)
