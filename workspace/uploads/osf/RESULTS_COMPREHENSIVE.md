# Comprehensive Test Results

## Overview

This report is generated directly from the results/* directories. As tests are added or updated, this section automatically reflects the current state.

## Test Summary

| Tier | Category | Tests | Pass Rate | Notes |
|------|----------|--------|-----------|-------|
| Tier 1 | Relativistic | 15 | 15/15 (100%) | Key achievements available in tier descriptions |
| Tier 2 | Gravity | 25 | 24/25 (96%) | Key achievements available in tier descriptions |
| Tier 3 | Energy | 10 | 0/10 (0%) | Key achievements available in tier descriptions |
| Tier 4 | Quantization | 14 | 2/14 (14%) | Key achievements available in tier descriptions |
| Tier 5 | Electromagnetic | 21 | 21/21 (100%) | Key achievements available in tier descriptions |
| chi_emergence | chi_emergence | 0 | 0/0 (0%) | Key achievements available in tier descriptions |
| Demo | Demo | 0 | 0/0 (0%) | Key achievements available in tier descriptions |

---

## Tier 1 — Relativistic (Lorentz invariance, isotropy, causality)

- REL-01: PASS — Isotropy — Coarse Grid
- REL-02: PASS — Isotropy — Fine Grid
- REL-03: PASS — Lorentz Boost — Low Velocity
- REL-04: PASS — Lorentz Boost — High Velocity
- REL-05: PASS — Causality — Pulse Propagation
- REL-06: PASS — Causality — Noise Perturbation
- REL-07: PASS — Phase Independence Test
- REL-08: PASS — Superposition Principle Test
- REL-09: PASS — 3D Isotropy — Directional Equivalence
- REL-10: PASS — 3D Isotropy — Spherical Symmetry
- REL-11: PASS — Dispersion Relation — Non-relativistic (χ/k≈10)
- REL-12: PASS — Dispersion Relation — Weakly Relativistic (χ/k≈1)
- REL-13: PASS — Dispersion Relation — Relativistic (χ/k≈0.5)
- REL-14: PASS — Dispersion Relation — Ultra-relativistic (χ/k≈0.1)
- REL-15: PASS — Causality — Space-like correlation test (light cone violation check)

## Tier 2 — Gravity Analogue (χ-field gradients, redshift, lensing)

- GRAV-01: PASS — Local frequency — linear χ-gradient (weak)
- GRAV-02: PASS — Local frequency — Gaussian well (strong curvature)
- GRAV-03: PASS — Local frequency — Gaussian well (broader potential)
- GRAV-04: PASS — Local frequency — Gaussian well (shallow potential)
- GRAV-05: PASS — Local frequency — linear χ-gradient (moderate)
- GRAV-06: PASS — Local frequency — Gaussian well (stable reference)
- GRAV-07: FAIL — Time dilation — bound states in double-well potential (KNOWN: Packet becomes trapped, demonstrates bound state physics)
- GRAV-08: PASS — Time dilation — uniform χ diagnostic (isolate grid dispersion)
- GRAV-09: PASS — Time dilation — 2x refined grid (N=128, dx=0.5)
- GRAV-10: PASS — Gravitational redshift — measure frequency shift in 1D potential well
- GRAV-11: PASS — Time delay — packet through χ slab (Shapiro-like)
- GRAV-12: PASS — Phase delay — continuous wave through χ slab (DEMONSTRATES: Klein-Gordon phase/group velocity mismatch - testable prediction!)
- GRAV-13: PASS — Local frequency — double well (ω∝χ verification)
- GRAV-14: PASS — Group delay — differential timing with vs without slab
- GRAV-15: PASS — 3D radial energy dispersion visualizer — central excitation, volumetric snapshots for MP4
- GRAV-16: PASS — 3D double-slit interference — quantum wave through slits showing χ-field localization
- GRAV-17: PASS — Gravitational redshift — frequency shift climbing out of χ-well
- GRAV-18: PASS — Gravitational redshift — linear gradient (Pound-Rebka analogue)
- GRAV-19: PASS — Gravitational redshift — radial χ-profile (Schwarzschild analogue)
- GRAV-20: PASS — Self-consistent chi from E-energy (Poisson) - verify omega~=chi at center (1D)
- GRAV-21: PASS — GR calibration - redshift to G_eff mapping (weak-field limit)
- GRAV-22: PASS — GR calibration - Shapiro delay correspondence (group velocity through slab)
- GRAV-23: PASS — Dynamic χ-field evolution — full wave equation □χ=-4πGρ with causal propagation (gravitational wave analogue)
- GRAV-24: PASS — Gravitational wave propagation — oscillating source radiates χ-waves, validate 1/r decay and propagation speed
- GRAV-25: PASS — Light bending — ray tracing through χ-gradient, measure deflection angle

## Tier 3 — Energy Conservation (Hamiltonian partitioning, dissipation)

- ENER-01: UNKNOWN — Global conservation — short
- ENER-02: UNKNOWN — Global conservation — long
- ENER-03: UNKNOWN — Wave integrity — mild curvature
- ENER-04: UNKNOWN — Wave integrity — steep curvature
- ENER-05: UNKNOWN — Hamiltonian partitioning — uniform χ (KE ↔ GE flow)
- ENER-06: UNKNOWN — Hamiltonian partitioning — with mass term (KE ↔ GE ↔ PE flow)
- ENER-07: UNKNOWN — Hamiltonian partitioning — χ-gradient field (energy flow in curved spacetime)
- ENER-08: UNKNOWN — Dissipation — weak damping (exponential decay, γ=1e-3 per unit time)
- ENER-09: UNKNOWN — Dissipation — strong damping (exponential decay, γ=1e-2 per unit time)
- ENER-10: UNKNOWN — Thermalization — noise + damping reaches steady state

## Tier 4 — Quantization (Discrete exchange, spectral linearity, uncertainty)

- QUAN-01: UNKNOWN — ΔE Transfer — Low Energy
- QUAN-02: UNKNOWN — ΔE Transfer — High Energy
- QUAN-03: UNKNOWN — Spectral Linearity — Coarse Steps
- QUAN-04: UNKNOWN — Spectral Linearity — Fine Steps
- QUAN-05: UNKNOWN — Phase-Amplitude Coupling — Low Noise
- QUAN-06: UNKNOWN — Phase-Amplitude Coupling — High Noise
- QUAN-07: UNKNOWN — Nonlinear Wavefront Stability
- QUAN-08: UNKNOWN — High-Energy Lattice Blowout Test
- QUAN-09: UNKNOWN — Heisenberg uncertainty — Δx·Δk ≈ 1/2
- QUAN-10: PASS — Bound state quantization — discrete energy eigenvalues E_n emerge from boundary conditions
- QUAN-11: UNKNOWN — Zero-point energy — ground state E₀ = ½ℏω ≠ 0 (vacuum fluctuations)
- QUAN-12: PASS — Quantum tunneling — barrier penetration when E < V (classically forbidden)
- QUAN-13: UNKNOWN — Wave-particle duality — which-way information destroys interference
- QUAN-14: UNKNOWN — Non-thermalization — validates Klein-Gordon conserves energy (doesn't approach Planck)

## Tier 5 — Electromagnetic (Maxwell equations, Coulomb, Lorentz force, EM waves, lensing)

- EM-01: PASS — Gauss's Law Verification: ∇·E = ρ/ε₀
- EM-02: PASS — Magnetic Field Generation: ∇×B = μ₀J
- EM-03: PASS — Faraday's Law Implementation: ∇×E = -∂B/∂t
- EM-04: PASS — Ampère's Law with Displacement Current: ∇×B = μ₀(J + ε₀∂E/∂t)
- EM-05: PASS — Electromagnetic Wave Propagation (FDTD)
- EM-06: PASS — Poynting Vector Conservation: ∇·S + ∂u/∂t = 0
- EM-07: PASS — χ-Field Electromagnetic Coupling: LFM mediates EM wave propagation
- EM-08: PASS — Mass-Energy Equivalence: E = mc²
- EM-09: PASS — Photon-Matter Interaction
- EM-10: PASS — Larmor Radiation from Accelerated Charges
- EM-11: PASS — Electromagnetic Rainbow Lensing & Dispersion
- EM-12: PASS — Time-Varying χ-Field EM Response
- EM-13: PASS — Electromagnetic Standing Waves in Cavity
- EM-14: PASS — Doppler Effect and Relativistic Corrections
- EM-15: PASS — Electromagnetic Scattering from χ-Inhomogeneities
- EM-16: PASS — Synchrotron Radiation from Accelerated Charges
- EM-17: PASS — Electromagnetic Pulse Propagation (FDTD)
- EM-18: PASS — Multi-Scale EM-χ Coupling
- EM-19: PASS — Gauge Invariance Verification: Physical fields unchanged under gauge transformations
- EM-20: PASS — Charge Conservation: ∂ρ/∂t + ∇·J = 0
- EM-21: PASS — Time-Varying χ-Field Pulse Response (FDTD)

## chi_emergence


## Demo



Generated: 2025-11-04 12:11:59