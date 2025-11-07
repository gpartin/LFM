# Workspace Results Category Folders

**Role in hierarchy**: `workspace/` → `results/` → `{category}/`

**Purpose**  
Organize per-run outputs by physics domain (Electromagnetic, Gravity, Energy, etc.). Each category contains test-specific subfolders with diagnostics, plots, and raw data.

**Structure**  
```
results/
  Electromagnetic/
    EM-01/ (test outputs)
    EM-02/
    ...
  Gravity/
    GRAV-01/
    PlanetFlyby3D/ (app-specific)
    ...
  Energy/
  Quantization/
  Relativistic/
  ...
```

**Lifecycle**  
These are working outputs; curate and promote significant artifacts to `workspace/docs/evidence/` for publication.

**Related folders**  
- `workspace/results/` (parent)
- `workspace/docs/evidence/` (publication-ready artifacts)
- `workspace/tests/` (tests that generate these outputs)
- `DIRECTORY_STRUCTURE.md` (root map)
