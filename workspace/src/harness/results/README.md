# Harness Results

**Role in hierarchy**: `workspace/` → `src/` → `harness/` → `results/`

**Purpose**  
Internal test harness working directory for temporary outputs during validation suite execution.

**Lifecycle**  
Ephemeral; safe to clean. Final results copied to `workspace/results/{category}/`.

**Related folders**  
- `workspace/src/harness/` (test harness code)
- `workspace/results/` (curated outputs)
- `DIRECTORY_STRUCTURE.md` (root map)
