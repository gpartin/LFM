# Workspace Results Category Folders

**Role in hierarchy**: `workspace/` → `results/` → `{category}/`

**Purpose**  
Organize per-run outputs by physics domain (Electromagnetic, Gravity, Energy, etc.). Each category contains test-specific subfolders with diagnostics, plots, and raw data.

**Structure**  
```
results/
  Gravity/
    GRAV-01/ (test outputs)
    GRAV-02/
    PlanetFlyby3D/ (app-specific)
    ...
```

**Lifecycle**  
These are working outputs; curate and promote significant artifacts to `workspace/docs/evidence/` for publication.

**Related folders**  
- `workspace/results/` (parent)
- `workspace/docs/evidence/` (publication-ready artifacts)
- `workspace/apps/` (apps like lfm_gravity_playground output here)
- `DIRECTORY_STRUCTURE.md` (root map)
