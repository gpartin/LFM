# Cross-Tier Refactor Plan (2025-11-09)

Goal: Apply Tier 1 modular validation + evaluator pattern uniformly across Tiers 2–7 to eliminate duplicated metric logic and hardcoded thresholds.

## Principles
1. Physics metric computation separate from threshold evaluation.
2. Single source of truth for thresholds: tierN_validation_metadata.json.
3. Runners emit raw metrics only; pass/fail derived exclusively via `harness.validation` helpers.
4. Energy conservation check standardized (`energy_conservation_check`).
5. No silent divergence: every metric added to summary.json.

## Inventory of Duplication (Initial)
| Tier | Common Metrics Observed | Current Duplication | Target Helper |
|------|-------------------------|---------------------|---------------|
| 2 (Gravity) | frequency_shift, time_delay, group_delay, redshift_ratio | Calculated ad hoc per test | `evaluate_redshift`, `evaluate_time_delay` |
| 3 (Energy) | drift_percent, partition_flow (KE↔GE↔PE) | Inlined drift comparison | `evaluate_energy_drift` (already), `evaluate_partition_balance` |
| 4 (Quantization) | eigenvalue_error, tunneling_probability | Manual error calc | `evaluate_eigenvalue_error`, `evaluate_tunneling_probability` |
| 5 (EM) | maxwell_residual, divergence_error | Repeated residual thresholds | `evaluate_maxwell_residual` |
| 6 (Coupling) | coupling_strength_error, energy_transfer_efficiency | Repeated ratio checks | `evaluate_coupling_error` |
| 7 (Thermo) | entropy_change, dissipation_rate | Manual drift checks | `evaluate_entropy_change`, `evaluate_dissipation_rate` |

## Phased Migration
1. Tier 2 (Gravity) — introduce redshift/time-delay helpers; patch runner imports; add metadata file scaffold.
2. Tier 3 (Energy) — move partition flow logic into helper; unify drift naming (`energy_drift`).
3. Tier 4 (Quantization) — add eigen/tunneling helpers; metadata file creation with thresholds.
4. Tier 5 (EM) — consolidate Maxwell residual evaluation; remove per-test threshold hardcoding.
5. Tier 6 (Coupling) — implement coupling efficiency helper; centralize transfer metric.
6. Tier 7 (Thermo) — entropy & dissipation helpers; ensure metadata gating for irreversible processes.

## Metadata File Creation
For tiers lacking validation metadata (2–7):
1. Create `workspace/config/tier{N}_validation_metadata.json` with schema:
```json
{
  "tier": N,
  "tests": {
    "GRAV-01": {
      "validation_criteria": {
        "primary": {"metric": "redshift_ratio_error", "threshold": 0.05},
        "energy_conservation": {"threshold": 0.01}
      }
    }
  }
}
```
2. Include rationale block (`"rationale": "Why threshold is chosen (numerical order, dispersion limits)."`).
3. Lock Tier 2 metadata after initial population (prevent drift similar to Tier 1).

## New Helper Stubs (Planned Additions)
```python
def evaluate_redshift(meta, test_id, redshift_error: float): ...
def evaluate_time_delay(meta, test_id, delay_error: float): ...
def evaluate_eigenvalue_error(meta, test_id, eig_err: float): ...
def evaluate_tunneling_probability(meta, test_id, prob_err: float): ...
def evaluate_maxwell_residual(meta, test_id, residual: float): ...
def evaluate_coupling_error(meta, test_id, coupling_err: float): ...
def evaluate_entropy_change(meta, test_id, entropy_err: float): ...
def evaluate_dissipation_rate(meta, test_id, dissipation_err: float): ...
```

## Acceptance Criteria Per Tier
- All runners use evaluator helpers exclusively for primary metrics.
- No hardcoded numeric thresholds remain in runners (except temporary diagnostic warnings).
- Metadata files supply all gating thresholds.
- Unit tests exist for each new evaluator (synthetic data, pass/fail boundary cases).

## Risk & Mitigation
| Risk | Mitigation |
|------|-----------|
| Threshold mismatch introduces false failures | Start with generous thresholds, tighten after baseline metrics collected. |
| Performance overhead from extra metadata loads | Cache per-tier metadata in runner init (already pattern from Tier 1). |
| Helper proliferation becomes unwieldy | Group related helpers (e.g., redshift/time-delay) or use generic `evaluate_primary(meta, id, {metric: value})`. |
| Missing metadata causes silent pass | Fallback currently defers to summary.pass; add warning log when metadata absent. |

## Immediate Next Actions
1. Implement Tier 2 metadata file scaffold.
2. Add redshift & time-delay helpers to `validation.py`.
3. Patch Tier 2 runner to import and use helpers.
4. Add unit tests for new helpers.

## Status
Created: 2025-11-09
Tier 1 complete; Tier 2 migration pending.
