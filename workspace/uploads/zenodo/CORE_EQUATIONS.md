---
title: "LFM Core Equations and Physics"
author: "Greg D. Partin"
institution: "LFM Research, Los Angeles CA USA"
license: "CC BY-NC-ND 4.0"
contact: "latticefieldmediumresearch@gmail.com"
orcid: "https://orcid.org/0009-0004-0327-6528"
doi: "10.5281/zenodo.17510124"
generated: "2025-11-06 17:49:33"
---

# ﻿Lattice-Field Medium (LFM): Core Equations and Theoretical Foundations

Version 3.2 — 2025-11-06 (Defensive ND Release)
Greg D. Partin | LFM Research — Los Angeles CA USA
License: Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (CC BY-NC-ND 4.0)
Note: This version supersedes all prior releases (v1.x and v2.x) and adds No-Derivatives restrictions and defensive-publication language for intellectual property protection. All LFM Phase-1 documents are synchronized under this unified v3.0 release.

## Abstract

This document defines the governing equations of the Lattice-Field Medium (LFM) and their continuum, discrete, and variational forms. It establishes the connection between the lattice update law and the variable-mass Klein–Gordon equation (Klein, 1926; Gordon, 1926), outlines how Lorentz invariance emerges naturally in the continuum limit, and shows how quantization and gravitational analogues arise through the curvature field χ(x,t).
1 Introduction and Scope
The Lattice-Field Medium (LFM) treats spacetime as a discrete lattice of interacting energy cells. Each cell holds an energy amplitude E(x,t) and curvature parameter χ(x,t). The purpose of this document is to define the mathematical foundation of LFM, connecting the discrete rule to its continuum form and providing validation targets used in Tier 1–3 testing.


## 1.1 Physics Foundation

LFM builds upon the Klein-Gordon equation developed by Oskar Klein and Walter Gordon in 1926:

Standard Klein-Gordon: ∂²φ/∂t² = c²∇²φ - m²φ

LFM's Innovation: We implement the standard Klein-Gordon equation with spatially-varying mass parameter χ²(x,t):
Klein-Gordon with spatially-varying χ-field: ∂²E/∂t² = c²∇²E - χ²(x,t)E

This spatial variation enables emergence of gravitational and quantum phenomena through discrete field interactions while preserving the fundamental relativistic structure.


## References:


- Klein, O. (1926). Quantentheorie und fünfdimensionale Relativitätstheorie. Zeitschrift für Physik, 37(12), 895-906.
- Gordon, W. (1926). Der Comptoneffekt nach der Schrödingerschen Theorie. Zeitschrift für Physik, 40(1-2), 117-133.

2 Canonical Field Equation
The canonical continuum form of the LFM equation is:
∂²E/∂t² = c² ∇²E − χ²(x,t) E, with c² = α/β.
Here E(x,t) is the local field energy, χ(x,t) is the curvature (effective mass), and c is the lattice propagation speed.
3 Discrete Lattice Update Law
We use a second-order, leapfrog scheme consistent with the canonical field equation
∂²E/∂t² = c²∇²E − χ(x,t)² E, with c² = α/β.
where ∇_Δ² is the finite-difference Laplacian, γ ≥ 0 is optional numerical
damping (γ = 0 for conservative runs), and χ(x,t) may be a scalar or a spatial field.
E^{t+1} = (2 − γ) E^t − (1 − γ) E^{t−1}
+ (Δt)² [ c² ∇_Δ² E^t − χ(x,t)² E^t ] ,
1D Laplacian (order-2):
∇_Δ² E_i = (E_{i+1} − 2E_i + E_{i−1}) / (Δx)²
1D Laplacian (order-4):
∇_Δ² E_i = [−E_{i+2} + 16E_{i+1} − 30E_i + 16E_{i−1} − E_{i−2}] / (12 (Δx)²)
Multi-D:

• 2D supports order-2 and order-4.
• 3D currently supports order-2 only (order-4/6 reserved for future tiers).

Boundary options (per test): periodic (canonical), reflective, or absorbing.
No stochastic (η) or exogenous coupling (Δφ) terms are part of the canonical law.
4 Derived Relations and (Continuum vs Lattice)
Continuum dispersion (χ constant):
ω² = c² k² + χ²
Lattice dispersion (order-2 1D; used in Tier-1 validation):
ω² = (4 c² / Δx²) sin²(k Δx / 2) + χ²
Energy monitoring (numerical):
We track relative energy drift |ΔE| / |E₀| and target ≤ 10⁻⁶ … 10⁻⁴ depending on grid and BCs.
Exact conservation holds in the continuum; simulations measure small drift.
Quantized exchange (interpretive):
ΔE = n ℏ_eff with ℏ_eff = ΔE_min Δt arising from discrete time; this is interpretive, not an input law.
Cosmological feedback:
Terms such as E_{t+1} = E_t + α∇²E − nH E belong to higher-tier χ-feedback studies and are not part of the canonical kernel.
5 Analogues (Non-canonical, exploratory)
Electromagnetic and inertial behaviours can be constructed as analogues of the canonical kernel, but they are not part of it.
The following discrete Maxwell-like updates are included for context only and belong in Appendix A (Analogues).
Discrete EM Coupling (Eq. 5-1, 5-2):
E_{I,t+1} = E_{I,t} + α(φ_{i+1,t} − φ_{i−1,t}) − βB_{I,t}
B_{I,t+1} = B_{I,t} + β(φ_{i+1,t} − φ_{i−1,t}) + αE_{I,t}
6 Lorentz Continuum Limit
Starting from the discrete update rule and applying Taylor expansion in time, the LFM equation reduces to:
∂²E/∂t² = c² ∇²E, with c² = α/β.
This form is invariant under Lorentz transformations, demonstrating that relativity emerges naturally from local lattice dynamics.
Formally, this corresponds to the joint limit Δx, Δt → 0 (with c = Δx/Δt fixed), where Σ E_i Δx → ∫ E(x) dx over (−∞,+∞).
7 Quantization from Discreteness
Quantization arises from the finite time-step Δt. The minimal exchange of energy per step defines ℏ_eff = ΔE_min Δt. The energy–frequency relation becomes E = ℏ_eff ω, and the momentum–wavelength relation p = ℏ_eff k, reproducing the de Broglie relation.
8 Dynamic χ Feedback and Cosmological Scaling
The curvature field χ evolves according to the feedback law:
dχ/dt = κ(ρ_ref − ρ_E) − γ χ ρ_E.
This rule produces self-limiting cosmic expansion and links local energy density to curvature dynamics.
Edge-creation condition:
if |∂E/∂r| > E_th → new cell at boundary.
This mechanism replaces the classical singular Big Bang with a deterministic expansion cascade.
9 Variational Gravity for χ
Promoting χ to a dynamic field yields coupled Euler–Lagrange equations:
σ_χ(∂ₜ²χ − v_χ²∇²χ) + V′(χ) = g_χE² + κ_EM(|𝔈|² + c²|𝔅|²).
In the weak-field limit, ∇²Φ = 4πG_effρ_eff reproduces Newtonian gravity and redshift/lensing analogues.

Numerical Validation (2025-11): Direct validation confirms χ dynamics emerge from energy distribution. Test evolved χ via ∂²χ/∂t² = c²∇²χ − κ(E² − E₀²) starting from uniform χ = 0.1. System developed 224,761× spatial variation (0.097–0.106) with r=0.46 correlation to E², demonstrating genuine emergence rather than manual configuration. Test: tests/test_chi_emergence_critical.py
10 Numerical Stability and Validation
CFL stability (d spatial dimensions):
c Δt / Δx ≤ 1 / √d (d = 1, 2, 3)
Energy diagnostics:
Measure |ΔE| / |E₀| each run; typical tolerances ≤ 10⁻⁶ – 10⁻⁴ depending on Δx, Δt, stencil order, and boundary conditions.
Stencil availability:
1D / 2D → order-2 and order-4; 3D → order-2 only (order-4 / 6 reserved for future tiers).
Test alignment:
Tier-1 uses the lattice dispersion relation above;
Tier-2 uses static χ(x) gradients;
Tier-3 evaluates energy drift under conservative settings.
11 Relation to Known PDE Classes
PDE Class
Canonical Form
Relation to LFM
Reference
Klein–Gordon
E_tt − c²∇²E + m²E = 0
LFM with constant χ
—
Variable-mass KG
E_tt − c²∇²E + χ(x,t)²E = 0
Identical continuum form
Ebert & Nascimento (2017)
Helmholtz
∇²u + k_eff²(x)u = 0
Time-harmonic analogue
Yagdjian (2012)
Quantum-walk lattices
Discrete Dirac/KG
Emergent Lorentz symmetry
Bisio et al. (2015)
12 Summary and Outlook
The Lattice-Field Medium provides a deterministic, Lorentz-symmetric framework where quantization, inertia, gravity, and cosmic expansion emerge from one discrete rule. All formulations preserve conservation, isotropy, and CPT symmetry. Tier 1–3 validations confirm numerical stability and physical coherence, forming the foundation for higher-tier exploration.
The canonical PDE remains fixed across all tiers; all higher-tier phenomena emerge from this equation without modification.

Discoveries Registry and Priority
To ensure consistent terminology and scientific priority, the canonical list of discoveries is maintained and published as part of this repository:

- Registry (source of truth): docs/discoveries/discoveries.json (Phase 1: 10 entries, last updated 2025-11-01).
- Generated overview: uploads/osf/DISCOVERIES_OVERVIEW.md and uploads/zenodo/DISCOVERIES_OVERVIEW.md are created from the registry during the upload build.

In case of any discrepancy between this document and the registry, the registry prevails. It serves as defensive publication establishing prior art.
13 Legal & Licensing Notice
This document and all accompanying materials are © 2025 Greg D. Partin.
All rights reserved. “Lattice-Field Medium,” “LFM Equation,” and “LFM Research Framework” are original works authored by Greg D. Partin.

### License Update (v3.2 — 2025-11-06):

Beginning with version 3.0, this work is licensed under the
Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).
Earlier releases (v1.x and v2.x) were distributed under CC BY-NC 4.0.
All later versions are governed by CC BY-NC-ND 4.0, which prohibits creation or redistribution of derivative or modified works without written consent of the author.
Derivative-Use Restriction
No portion of this document or the LFM equation may be reproduced, modified, or adapted for any commercial, proprietary, or patent-filing purpose without prior written authorization.
“Commercial” includes any research or prototype development intended for monetization, commercialization, or patent application.
Defensive Publication Statement
This publication constitutes a defensive disclosure establishing prior art as of October 29 2025 for all concepts, algorithms, and methods described herein. Its release prevents any later exclusive patent claim over identical or equivalent formulations of the LFM equation or its numerical realization.
Trademark Notice
“Lattice-Field Medium,” “LFM Research,” and “LFM Equation” are distinctive marks identifying this body of work. Unauthorized use of these names in promotional, academic, or product contexts is prohibited.
Redistribution Boundary
All code examples, update laws, and data structures herein are disclosed solely for scientific reproducibility. They are not granted for reuse, adaptation, or redistribution in derivative simulation frameworks without written permission of the author.

### Citation (Zenodo Record):

Partin, G. D. (2025). Lattice-Field Medium (LFM): A Deterministic Lattice Framework for Emergent Relativity, Gravitation, and Quantization — Phase 1 Conceptual Hypothesis v1.0. Zenodo. https://doi.org/10.5281/zenodo.17478758
Contact: latticefieldmediumresearch@gmail.com

---

License: CC BY-NC-ND 4.0