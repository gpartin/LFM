---
title: "Tier 2 — Gravity Analogue Validation Achievements"
author: "Greg D. Partin"
institution: "LFM Research, Los Angeles CA USA"
license: "CC BY-NC-ND 4.0"
contact: "latticefieldmediumresearch@gmail.com"
orcid: "https://orcid.org/0009-0004-0327-6528"
doi: "https://zenodo.org/records/17618474"
generated: "2025-11-15 09:12:55"
---

# Tier 2 — Gravity Analogue Validation Achievements

Generated: 2025-11-15 09:12:55

## Overview

This tier validates χ-field gradient effects including gravitational redshift, time dilation, and light bending analogues.

## Key Achievements

- **Total Tier 2 — Gravity Analogue Tests**: 25
- **Tests Passed**: 25 (100.0%)

## Test Results Summary

| Test ID | Status | Description |
|---------|--------|-------------|
| GRAV-01 | PASS | Local frequency — linear χ-gradient (weak) |
| GRAV-02 | PASS | Local frequency — Gaussian well (strong curvature) |
| GRAV-03 | PASS | Local frequency — Gaussian well (broader potential) |
| GRAV-04 | PASS | Local frequency — Gaussian well (shallow potential) |
| GRAV-05 | PASS | Local frequency — linear χ-gradient (moderate) |
| GRAV-06 | PASS | Local frequency — Gaussian well (stable reference) |
| GRAV-07 | PASS | Time dilation — bound states in double-well potential (KNOWN: Packet becomes trapped, demonstrates bound state physics) [ISOLATED] |
| GRAV-08 | PASS | Time dilation — uniform χ diagnostic (isolate grid dispersion) |
| GRAV-09 | SKIP | Time dilation — 2x refined grid (N=128, dx=0.5) [OPTIMIZED: matched baseline duration for fair convergence comparison] |
| GRAV-10 | PASS | Gravitational redshift — measure frequency shift in 1D potential well |
| GRAV-11 | PASS | Time delay — packet through χ slab (Shapiro-like) |
| GRAV-12 | PASS | Phase delay — continuous wave through χ slab (DEMONSTRATES: Klein-Gordon phase/group velocity mismatch - testable prediction!) |
| GRAV-13 | PASS | Local frequency — double well (ω∝χ verification) |
| GRAV-14 | PASS | Group delay — differential timing with vs without slab |
| GRAV-15 | PASS | 3D radial energy dispersion visualizer — central excitation, volumetric snapshots for MP4 |
| GRAV-16 | PASS | 3D double-slit interference — quantum wave through slits showing χ-field localization |
| GRAV-17 | PASS | Gravitational redshift — frequency shift climbing out of χ-well |
| GRAV-18 | PASS | Gravitational redshift — linear gradient (Pound-Rebka analogue) |
| GRAV-19 | PASS | Gravitational redshift — radial χ-profile (Schwarzschild analogue) |
| GRAV-20 | PASS | Self-consistent chi from E-energy (Poisson) - verify omega~=chi at center (1D) |
| GRAV-21 | PASS | GR calibration - redshift to G_eff mapping (weak-field limit) |
| GRAV-22 | PASS | GR calibration - Shapiro delay correspondence (group velocity through slab) |
| GRAV-23 | PASS | Dynamic χ-field evolution — full wave equation □χ=-4πGρ with causal propagation (gravitational wave analogue) |
| GRAV-24 | PASS | Gravitational wave propagation — oscillating source radiates χ-waves, validate 1/r decay and propagation speed |
| GRAV-25 | PASS | Light bending — ray tracing through χ-gradient, measure deflection angle |
| GRAV-26 | PASS | Weak Equivalence Principle — mass-independent gravitational acceleration |

## Significance

These validations demonstrate that gravity-like phenomena (redshift, lensing, time delay) emerge from χ-field gradients, suggesting LFM may provide a unified framework for gravity and quantum mechanics.

---
License: CC BY-NC-ND 4.0


## Skip Disclosure

# Skip Disclosure Policy and Rationale


We explicitly disclose any tests that are marked as SKIPPED and do not count them against pass rates.

Policy:

1. Pass rates are computed over executed tests only (i.e., total − skipped).
2. Skipped tests are listed with a clear technical rationale.
3. Skips reflect out‑of‑scope or method‑incompatible designs, not physics failures.
4. When the incompatibility is resolved or the design is re‑scoped, the test may be re‑enabled.

Deterministic accounting:

- The master status file (MASTER_TEST_STATUS.csv) includes columns: Total_Tests, Executed_Tests, Passed, Failed, Skipped, Pass_Rate_Executed.
- Upload documents copy this file verbatim and summarize executed pass rates for tiers.

Current exemplar (Tier 2):

- GRAV-09 — Time dilation — 2x refined grid (N=128, dx=0.5)

Reason: Test design incompatible with discrete Klein–Gordon dispersion on a finite grid. Continuous theory allows bound states with ω≈χ (k→0 limit), but the discrete grid requires representing fields as Fourier sums with k_min=2π/L. Any localized initial condition couples to grid modes with k≈2.26 (≈36% of k_max=π/dx≈6.28), giving ω²≈k²+χ²≈5.1 where k-content dominates χ² by ~100×, making a pure χ-oscillation measurement infeasible. This is a test‑design limitation, not a physics failure.

Implication:

- Excluding GRAV-09 from pass rate computation is scientifically justified and preserves the integrity of the validation metrics.


---

License: CC BY-NC-ND 4.0


| Test ID | Description | Reason |
|---------|-------------|--------|
| GRAV-09 | Time dilation — 2x refined grid (N=128, dx=0.5) [OPTIMIZED: matched baseline duration for fair convergence comparison] | Test design incompatible with discrete Klein-Gordon dispersion. Continuous theory allows bound states with ω≈χ (k→0 limit); but discrete grid with dx=0.5 requires representing all fields as Fourier sum with k_min=2π/L. Any localized initial |
