# FD Covariance Synthetic Validation (REL-03 Support)

Purpose: Provide controlled analytic benchmarks for Lorentz covariance residual measurement used in REL-03.

## Background
Legacy REL-03 method used Doppler-transformed (ω,k) dispersion comparison. That approach masks transform discretization errors and produces a stable covariance_ratio ~1.225 under certain numerical artifacts.

Finite-difference (FD) attempt computing boosted-frame PDE residuals directly inflated the boosted residual by ~1e11× (ratio ~1.9e11) due to per-point time mapping and mixed interpolation.

## Objectives
1. Spatial-only covariance prototype: Compare spatial+mass operator consistency between frames ignoring temporal second derivative to decouple time mapping issues.
2. Plane wave synthetic tests across β values to confirm spatial covariance ratio ~1.
3. Convergence studies varying (dt, dx, N) to establish error scaling.
4. Error budget documentation: quantify contributions from spatial interpolation, Lorentz coordinate mapping, temporal aliasing.

## Files
- `fd_covariance_synthetic.py` (in `src/tools/`): Generates analytic plane wave and runs current FD method (for historical baseline).
- (planned) `spatial_covariance_sweep.py`: Multi-β spatial-only residual comparison.
- (planned) `convergence_rel03.py`: Convergence study script.
- (planned) `error_budget_rel03.md`: Human-readable summary of identified error sources.

## Stages
| Stage | Artifact | Goal |
|-------|----------|------|
| S1 | spatial-only method | Diagnostic only; not Lorentz-invariant (see Results) |
| S2 | β sweep | Show stability across β (0.1–0.6) |
| S3 | Convergence | Demonstrate residual ∝ dt² + dx^p (p=2 or 4) |
| S4 | Integration | Replace harness method with validated FD approach |

## Acceptance Criteria for Method Promotion
- Full FD method (uniform t' spacetime remeshing, with time derivative) yields ratio within 5% on analytic wave at two resolutions.
- REL-03 using FD method passes covariance threshold (≤5% rel_error) without relaxing tolerance.

## Next Actions
1. Implement `verify_klein_gordon_covariance_spatial` in `physics/lorentz_transform.py`. ✅ 2025-11-09 added.
2. Add spatial covariance synthetic script. ✅ 2025-11-09 added (`tools/spatial_covariance_synthetic.py`).
3. Run β sweep; record results to JSON under `results/Relativistic/REL-03_SYNTH/`. ✅ 2025-11-09.
4. Pivot to uniform t' spacetime remeshing (x', t' grid), then compute full FD residual in boosted frame.
5. Document error budget (PENDING).

## Notes
Preserve legacy method as fallback until FD method validated. Do not adjust REL-03 thresholds during development; prioritize scientific accuracy over convenience.

Spatial-only operator RMS was tested and found not to be a covariance metric due to relativity of simultaneity: a constant t_lab slice does not correspond to a constant t' slice, so comparing L_x+χ²E across frames is not meaningful. Results (β∈{0.1,0.2,0.3,0.4}) showed large spatial_ratio deviations (≈6–40). The correct path forward is to construct uniform t' slices via spacetime remeshing and compute the full KG residual (including ∂²/∂t'²) in the boosted frame.

Artifacts:
- `results/Relativistic/REL-03_SYNTH/spatial_covariance_synthetic.json` — recorded spatial-only sweep for diagnostics.
