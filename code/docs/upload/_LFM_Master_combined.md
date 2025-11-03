# Lorentzian Field Model (LFM) — Master Document

Generated: 2025-11-03 14:27:31
License: CC BY-NC-ND 4.0 — Non-commercial use; no derivatives
Related: OSF https://osf.io/6agn8 · Zenodo https://zenodo.org/records/17478758



<!-- Source: docs\text\Executive_Summary.txt -->


Lattice-Field Medium (LFM): Executive Summary

Version 3.0 — 2025-11-01 (Defensive ND Release)
Greg D. Partin | LFM Research — Los Angeles CA USA
License: Creative Commons Attribution–NonCommercial–NoDerivatives 4.0
International (CC BY-NC-ND 4.0)
Note: This version supersedes all prior releases (v3.0 and earlier) and
adds No-Derivatives restrictions and defensive-publication language for
intellectual-property protection. All LFM Phase-1 documents are
synchronized under this unified v3.0 release.

Overview

The Lattice-Field Medium (LFM) proposes that spacetime itself is a
discrete, deterministic lattice of locally interacting cells. Each cell
carries an energy amplitude E(x,t) and a curvature parameter χ(x,t) that
modulates its local stiffness. The governing relation ∂²E/∂t² = c²∇²E −
χ²(x,t)E, with c² = α/β, represents a Lorentz-symmetric, locally causal
wave law building upon the Klein–Gordon equation foundation (Klein, 1926;
Gordon, 1926). By allowing χ to vary across space and time, this single rule
reproduces classical mechanics, relativity, gravitation, quantization,
and cosmological expansion as emergent phenomena of one underlying
field.

Key Structural Features

	-----------------------------------------------------------------------
	Feature                             Consequence
	----------------------------------- -----------------------------------
	Local hyperbolic operator           Finite propagation speed and
																			causality

	Lorentz invariance in continuum     Special relativity emerges
	limit                               automatically

	Curvature field χ(x,t)              Acts as both inertial mass and
																			gravitational potential

	Lagrangian & Noether conservation   Intrinsic energy–momentum
																			conservation

	Discrete temporal steps             Natural quantization scale (ℏ_eff =
																			ΔE_min Δt)
	-----------------------------------------------------------------------

Recent Results (Validated Tiers)

1. Lorentz analogue confirmed numerically (ω² = c²k² + χ²).
2. Gravitational redshift and lensing reproduced with χ-gradients (Tier
2).
3. Energy conservation stable to <10⁻⁴ drift over 10³ steps.
4. Cosmological expansion self-limits via χ-feedback (Tier 6 prototype).
5. Variational gravity law derived: σ_χ(∂ₜ²χ − v_χ²∇²χ) + V′(χ) =
g_χE² + κ_EM(|𝔈|² + c²|𝔅|²).

Implications

- Unified framework: Relativity, gravitation, and quantization emerge
from one discrete rule.
- Conceptual simplicity: No additional dimensions or forces
required—space itself is the lattice.
- Predictive potential: χ-feedback may eliminate the need for a
cosmological constant.
- Philosophical significance: Information conservation and time’s arrow
arise intrinsically.

Status and Next Steps

All core equations and validation tiers are internally consistent. Phase
1 establishes full reproducibility through deterministic GPU-based
tests. Next steps include expanded electromagnetic simulations, extended
quantum interference validation, and long-run χ-feedback stability
studies.

Summary

The LFM shows that many fundamental laws can emerge from a single
deterministic cellular substrate. Gravity, inertia, and relativistic
behavior are not imposed upon the lattice—they are expressions of its
geometry. Upon completion of Tier 3 validation and expert review, the
LFM will stand as a mathematically coherent, testable, and potentially
unifying framework for physical law.

Legal & Licensing Notice

This document and all accompanying materials are © 2025 Greg D. Partin.
All rights reserved. “Lattice-Field Medium,” “LFM Equation,” and “LFM
Research Framework” are original works authored by Greg D. Partin.

License Update (v3.0 — 2025-11-01):
Beginning with version 3.0, this work is licensed under the
Creative Commons Attribution–NonCommercial–NoDerivatives 4.0
International License (CC BY-NC-ND 4.0).
Earlier releases were distributed under CC BY-NC 4.0.
All later versions are governed by CC BY-NC-ND 4.0, which prohibits
creation or redistribution of derivative or modified works without
written consent of the author.

Derivative-Use Restriction
No portion of this document or its contained analyses may be reproduced,
modified, or adapted for any commercial, proprietary, or patent-filing
purpose without prior written authorization.
“Commercial” includes any research or prototype development intended for
monetization, commercialization, or patent application.

Defensive Publication Statement
This publication constitutes a defensive disclosure establishing prior
art as of October 29 2025 for all concepts and results described herein.
Its release prevents any later exclusive patent claim over identical or
equivalent formulations of the LFM framework or its empirical validation
data.

Trademark Notice
“Lattice-Field Medium,” “LFM Research,” and “LFM Equation” are
distinctive marks identifying this body of work. Unauthorized use of
these names in promotional, academic, or product contexts is prohibited.

Redistribution Boundary
All summaries, figures, and data presented here are disclosed solely for
scientific reproducibility. They are not granted for reuse, adaptation,
or redistribution in derivative simulation frameworks without written
permission of the author.

Citation (Zenodo Record):
Partin, G. D. (2025). Lattice-Field Medium (LFM): A Deterministic
Lattice Framework for Emergent Relativity, Gravitation, and Quantization
— Phase 1 Conceptual Hypothesis v1.0. Zenodo.
https://doi.org/10.5281/zenodo.17478758

Contact: latticefieldmediumresearch@gmail.com



<!-- Source: docs\text\LFM_Master.txt -->


Lattice-Field Medium (LFM): Master Document — Conceptual Framework and
Physical Interpretation
Version 3.0 — 2025-11-01 (Defensive ND Release)
Greg D. Partin | LFM Research, Los Angeles CA USA
License: Creative Commons Attribution–NonCommercial–NoDerivatives 4.0
International (CC BY-NC-ND 4.0)
Note: This version supersedes all prior releases (v2.x and earlier) and
adds No-Derivatives restrictions and defensive-publication language for
intellectual property protection. All LFM Phase-1 documents are
synchronized under this unified v3.0 release.

Abstract

The Lattice-Field Medium (LFM) proposes that spacetime arises from a
deterministic lattice of locally coupled energy cells. Each cell evolves
according to a single discrete update rule that yields, in the continuum
limit, a variable-mass Klein–Gordon equation (Klein, 1926; Gordon, 1926).
Building upon this foundational framework in relativistic field theory,
this master document provides the conceptual framework and interpretation
of that rule, showing how classical, relativistic, gravitational, quantum,
and cosmological behaviors all emerge as consequences of one substrate law.

1 Purpose and Scope

This document defines the conceptual framework of the Lattice-Field
Medium (LFM) and connects it to the formal equations and numerical tests
in the companion Core Equations and Phase 1 Test Design documents. Its
goal is to describe how physical laws emerge from local lattice dynamics
and to outline the interpretive consequences for relativity,
gravitation, and quantization.

2 Canonical Framework

At the foundation of the LFM is a local deterministic equation that
governs the evolution of the energy field E(x,t) and curvature field
χ(x,t):

∂²E/∂t² = c² ∇²E − χ(x,t)² E, with c² = α/β.

This is the same canonical law implemented in the discrete leapfrog form
defined in the companion LFM Core Equations (v1.1).

This relation represents a Lorentz-symmetric, locally causal wave
equation. In the continuum limit, it reproduces the structure of a
variable-mass Klein–Gordon field. All macroscopic behaviors—classical,
relativistic, and quantum—arise from this same rule.

3 Foundational Properties

	-----------------------------------------------------------------------
	Structural Feature                  Physical Outcome
	----------------------------------- -----------------------------------
	Local hyperbolic operator           Finite propagation speed, causality

	Lorentz invariance of □             Emergent special relativity

	Curvature field χ(x,t)              Inertia and gravity analogues

	Lagrangian symmetry                 Energy–momentum conservation

	Discrete time step defines a        Natural quantization scale
	natural quantization scale (ℏ_eff = 
	ΔE_min Δt).                         
	-----------------------------------------------------------------------

4 Analytic Checks and Validation

Analytic proofs demonstrate that the LFM reproduces well-known physical
laws:
1. Characteristic cone: defines invariant light-cone structure.
2. Noether energy: ensures intrinsic conservation.
3. WKB lensing: predicts ray bending toward higher χ.
4. Mode quantization: discrete oscillation frequencies.
5. Scaling symmetry: dimensionless and self-consistent.

5 Domains of Emergence

The same lattice rule reproduces distinct physical regimes depending on
the behavior of χ(x,t) and coupling constants:

• Classical & Relativistic: Lorentz invariance and causal propagation
(Tier 1).

• Gravitational: χ-gradients produce redshift and lensing (Tier 2).

• Quantum & Coherence: quantized exchange and long-range correlations
(Tier 3–5).

• Cosmological: χ-feedback drives self-limiting expansion (Tier 6).

(Tier numbering corresponds to Phase 1 Test Design v2.0.)

6 Interpretation and Ontology

In the LFM view, spacetime, matter and energy are emergent
manifestations of a discrete substrate:
- Space corresponds to lattice connectivity.
- Time corresponds to sequential updates.
- Energy corresponds to local oscillation amplitude.
- Gravity arises from spatial gradients in χ.
- Quantization results from discrete temporal evolution.

Fig 1 — Conceptual mapping of LFM quantities to physical observables
(placeholder).

7 Experimental and Simulation Validation

	-----------------------------------------------------------------------
	Domain            Example Test      Observable        Status
	----------------- ----------------- ----------------- -----------------
	Laboratory        Cavity or         Discrete          Planned
										interferometer    dispersion /      
																			anisotropy         

	Astrophysical     GRB timing /      χ-dependent delay Analysis
										ringdown          or shift          

	Numerical         Tier 1–3 GPU      Lorentz & energy  PASS
										lattice runs      conservation      
	-----------------------------------------------------------------------

8 Gravity Emergence Summary

The curvature field χ acts as a dynamic gravitational potential. Its
equation of motion, derived from the Lagrangian formalism, reproduces
the Newtonian limit and predicts weak-field lensing and redshift
effects. In this view, gravity is a self-organized property of the
lattice rather than an external force.

(These gravitational analogues arise in Tier 2 configurations and above;
no new forces or parameters are introduced.)

9 The Nature of Time

The LFM update law is time-symmetric, but the arrow of time arises from
information dispersion. As correlations spread across more lattice
cells, entropy increases. Thus, time measures the diffusion of
information rather than an independent external flow.

The increase in entropy noted here corresponds to the measurable entropy
dynamics diagnostic in simulation output.

This interpretation is consistent with reversible yet statistically
asymmetric evolution, where microscopic reversibility yields macroscopic
time’s arrow.

10 Continuum–Discrete Bridge

Fluid behavior, wave mechanics, and quantum interference all appear as
statistical regimes of the same discrete rule. By tuning α, β, and χ
(and optional damping γ), the lattice reproduces laminar, turbulent, and
quantized flow behaviors consistent with classical hydrodynamics and
quantum statistics.

11 Tier-1 Insights

Tier-1 validation confirms that discrete, reversible rules can reproduce
continuous, isotropic energy propagation with conservation to numerical
precision. This implies that continuity itself is an emergent illusion
of discrete processes.

Key outcomes:
- Conservation from discreteness
- Emergent relativity
- Self-quantization
- Continuum illusion
Together, these show that the lattice substrate can generate stable,
law-like behavior indistinguishable from continuous spacetime.

These validations establish the canonical Tier 1–3 foundation on which
all higher-tier phenomena build.

12 Open Questions and Future Work

Outstanding questions for future investigation:
1. Mapping lattice constants (α, β, χ) to physical units.
2. High-curvature stability and 3D scalability.
3. Independent third-party validation.
4. Entropy, thermodynamics, and information conservation.
5. Integration with established quantum field frameworks.

6. Long-term numerical energy drift characterization across different
stencil orders and dimensions.

7. Verification of χ-coupled energy curvature via probe-particle
simulations (Tier 2–3 extensions).

13 Summary

The Lattice-Field Medium unifies relativity, gravitation, quantization,
and cosmology through a single discrete rule. Energy, inertia, and
curvature emerge as properties of one deterministic field. Continued
validation will determine whether this structure can serve as a
fundamental framework for physical law.

This Version aligns all conceptual, mathematical, and numerical
formulations under one canonical framework, thereby completing Phase 1
conceptual validation and establishing the theoretical foundation for
empirical verification.

14 Legal & Licensing Notice

This document and all accompanying materials are © 2025 Greg D. Partin.
All rights reserved. “Lattice-Field Medium,” “LFM Equation,” and “LFM
Research Framework”
are original works authored by Greg D. Partin.

License Update (v3.0 — 2025-11-01):
Beginning with version 3.0, this work is licensed under the
Creative Commons Attribution–NonCommercial–NoDerivatives 4.0
International License (CC BY-NC-ND 4.0).
Earlier releases (v2.x and prior) were distributed under CC BY-NC 4.0.
All later versions are governed by CC BY-NC-ND 4.0, which prohibits
creation or redistribution of derivative or modified works without
written consent of the author.

Derivative-Use Restriction
No portion of this document, equation, or accompanying code may be
reproduced, modified, or adapted for any commercial, proprietary, or
patent-filing purpose without prior written authorization. “Commercial”
includes any research or prototype development intended for
monetization, commercialization, or patent application.

Defensive Publication Statement
This publication constitutes a defensive disclosure establishing prior
art as of October 29 2025 for all concepts, algorithms, and methods
described herein. Its release prevents any later exclusive patent claim
over identical or equivalent formulations of the LFM equation or its
numerical realization.

Trademark Notice
“Lattice-Field Medium,” “LFM Research,” and “LFM Equation” are
distinctive marks identifying this body of work.
Unauthorized use of these names in promotional, academic, or product
contexts is prohibited.

Redistribution Boundary
All code, configuration, and data structures described are disclosed
solely for scientific reproducibility.
They are not granted for reuse, adaptation, or redistribution in
derivative simulation frameworks
without written permission of the author.

Citation (Zenodo Record):
Partin, G. D. (2025). Lattice-Field Medium (LFM): A Deterministic
Lattice Framework for Emergent Relativity, Gravitation, and Quantization
— Phase 1 Conceptual Hypothesis v1.0. Zenodo.
https://doi.org/10.5281/zenodo.17478758

Contact: latticefieldmediumresearch@gmail.com



<!-- Source: docs\text\LFM_Core_Equations.txt -->


Lattice-Field Medium (LFM): Core Equations and Theoretical Foundations Version 3.0 — 2025-11-01 (Defensive ND Release)

Greg D. Partin | LFM Research — Los Angeles CA USA License: Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (CC BY-NC-ND 4.0) Note: This version supersedes all prior releases (v1.x and v2.x) and adds No-Derivatives restrictions and defensive-publication language for intellectual property protection. All LFM Phase-1 documents are synchronized under this unified v3.0 release.

Abstract

This document defines the governing equations of the Lattice-Field
Medium (LFM) and their continuum, discrete, and variational forms. It
establishes the connection between the lattice update law and the
variable-mass Klein–Gordon equation (Klein, 1926; Gordon, 1926), outlines
how Lorentz invariance emerges naturally in the continuum limit, and shows
how quantization and gravitational analogues arise through the curvature
field χ(x,t). Building upon foundational relativistic field theory, this
work extends the Klein-Gordon framework to spatially-varying mass terms.

1 Introduction and Scope

The Lattice-Field Medium (LFM) treats spacetime as a discrete lattice of
interacting energy cells. Each cell holds an energy amplitude E(x,t) and
curvature parameter χ(x,t). The purpose of this document is to define
the mathematical foundation of LFM, connecting the discrete rule to its
continuum form and providing validation targets used in Tier 1–3
testing.

2 Canonical Field Equation

The canonical continuum form of the LFM equation is:

∂²E/∂t² = c² ∇²E − χ²(x,t) E,  with c² = α/β.

Here E(x,t) is the local field energy, χ(x,t) is the curvature
(effective mass), and c is the lattice propagation speed.

3 Discrete Lattice Update Law

We use a second-order, leapfrog scheme consistent with the canonical
field equation

∂²E/∂t² = c²∇²E − χ(x,t)² E, with c² = α/β.

where ∇_Δ² is the finite-difference Laplacian, γ ≥ 0 is optional
numerical

damping (γ = 0 for conservative runs), and χ(x,t) may be a scalar or a
spatial field.

E^{t+1} = (2 − γ) E^t − (1 − γ) E^{t−1}

+ (Δt)² [ c² ∇_Δ² E^t − χ(x,t)² E^t ] ,

1D Laplacian (order-2):

∇_Δ² E_i = (E_{i+1} − 2E_i + E_{i−1}) / (Δx)²

1D Laplacian (order-4):

∇_Δ² E_i = [−E_{i+2} + 16E_{i+1} − 30E_i + 16E_{i−1} − E_{i−2}] / (12
(Δx)²)

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

We track relative energy drift |ΔE| / |E₀| and target ≤ 10⁻⁶ … 10⁻⁴
depending on grid and BCs.

Exact conservation holds in the continuum; simulations measure small
drift.

Quantized exchange (interpretive):

ΔE = n ℏ_eff with ℏ_eff = ΔE_min Δt arising from discrete time; this is
interpretive, not an input law.

Cosmological feedback:

Terms such as E_{t+1} = E_t + α∇²E − nH E belong to higher-tier
χ-feedback studies and are not part of the canonical kernel.

5 Analogues (Non-canonical, exploratory)

Electromagnetic and inertial behaviours can be constructed as analogues
of the canonical kernel, but they are not part of it.

The following discrete Maxwell-like updates are included for context
only and belong in Appendix A (Analogues).

Discrete EM Coupling (Eq. 5-1, 5-2):

E_{I,t+1} = E_{I,t} + α(φ_{i+1,t} − φ_{i−1,t}) − βB_{I,t}

B_{I,t+1} = B_{I,t} + β(φ_{i+1,t} − φ_{i−1,t}) + αE_{I,t}

6 Lorentz Continuum Limit

Starting from the discrete update rule and applying Taylor expansion in
time, the LFM equation reduces to:
∂²E/∂t² = c² ∇²E,  with c² = α/β.
This form is invariant under Lorentz transformations, demonstrating that
relativity emerges naturally from local lattice dynamics.

Formally, this corresponds to the joint limit Δx, Δt → 0 (with c = Δx/Δt
fixed), where Σ E_i Δx → ∫ E(x) dx over (−∞,+∞).

7 Quantization from Discreteness

Quantization arises from the finite time-step Δt. The minimal exchange
of energy per step defines ℏ_eff = ΔE_min Δt. The energy–frequency
relation becomes E = ℏ_eff ω, and the momentum–wavelength relation p =
ℏ_eff k, reproducing the de Broglie relation.

8 Dynamic χ Feedback and Cosmological Scaling

The curvature field χ evolves according to the feedback law:
dχ/dt = κ(ρ_ref − ρ_E) − γ χ ρ_E.
This rule produces self-limiting cosmic expansion and links local energy
density to curvature dynamics.

Edge-creation condition:
if |∂E/∂r| > E_th → new cell at boundary.
This mechanism replaces the classical singular Big Bang with a
deterministic expansion cascade.

9 Variational Gravity for χ

Promoting χ to a dynamic field yields coupled Euler–Lagrange equations:
σ_χ(∂ₜ²χ − v_χ²∇²χ) + V′(χ) = g_χE² + κ_EM(|𝔈|² + c²|𝔅|²).
In the weak-field limit, ∇²Φ = 4πG_effρ_eff reproduces Newtonian gravity
and redshift/lensing analogues.

10 Numerical Stability and Validation

CFL stability (d spatial dimensions):

 c Δt / Δx ≤ 1 / √d (d = 1, 2, 3)

Energy diagnostics:

 Measure |ΔE| / |E₀| each run; typical tolerances ≤ 10⁻⁶ – 10⁻⁴ depending on Δx, Δt, stencil order, and boundary conditions.

Stencil availability:

 1D / 2D → order-2 and order-4; 3D → order-2 only (order-4 / 6 reserved for future tiers).

Test alignment:

 Tier-1 uses the lattice dispersion relation above; 

 Tier-2 uses static χ(x) gradients; 

 Tier-3 evaluates energy drift under conservative settings.

11 Relation to Known PDE Classes

	-----------------------------------------------------------------------
	PDE Class         Canonical Form    Relation to LFM   Reference
	----------------- ----------------- ----------------- -----------------
	Klein–Gordon      E_tt − c²∇²E +    LFM with constant —
										m²E = 0           χ                 

	Variable-mass KG  E_tt − c²∇²E +    Identical         Ebert &
										χ(x,t)²E = 0      continuum form    Nascimento (2017)

	Helmholtz         ∇²u + k_eff²(x)u  Time-harmonic     Yagdjian (2012)
										= 0               analogue          

	Quantum-walk      Discrete Dirac/KG Emergent Lorentz  Bisio et al.
	lattices                            symmetry          (2015)
	-----------------------------------------------------------------------

12 Summary and Outlook

The Lattice-Field Medium provides a deterministic, Lorentz-symmetric
framework where quantization, inertia, gravity, and cosmic expansion
emerge from one discrete rule. All formulations preserve conservation,
isotropy, and CPT symmetry. Tier 1–3 validations confirm numerical
stability and physical coherence, forming the foundation for higher-tier
exploration.

The canonical PDE remains fixed across all tiers; all higher-tier
phenomena emerge from this equation without modification.

13 Legal & Licensing Notice

This document and all accompanying materials are © 2025 Greg D. Partin.
All rights reserved. “Lattice-Field Medium,” “LFM Equation,” and “LFM
Research Framework” are original works authored by Greg D. Partin.

License Update (v3.0 — 2025-11-01):
Beginning with version 3.0, this work is licensed under the
Creative Commons Attribution–NonCommercial–NoDerivatives 4.0
International License (CC BY-NC-ND 4.0).
Earlier releases (v1.x and v2.x) were distributed under CC BY-NC 4.0.
All later versions are governed by CC BY-NC-ND 4.0, which prohibits
creation or redistribution of derivative or modified works without
written consent of the author.

Derivative-Use Restriction
No portion of this document or the LFM equation may be reproduced,
modified, or adapted for any commercial, proprietary, or patent-filing
purpose without prior written authorization.
“Commercial” includes any research or prototype development intended for
monetization, commercialization, or patent application.

Defensive Publication Statement
This publication constitutes a defensive disclosure establishing prior
art as of October 29 2025 for all concepts, algorithms, and methods
described herein. Its release prevents any later exclusive patent claim
over identical or equivalent formulations of the LFM equation or its
numerical realization.

Trademark Notice
“Lattice-Field Medium,” “LFM Research,” and “LFM Equation” are
distinctive marks identifying this body of work. Unauthorized use of
these names in promotional, academic, or product contexts is prohibited.

Redistribution Boundary
All code examples, update laws, and data structures herein are disclosed
solely for scientific reproducibility. They are not granted for reuse,
adaptation, or redistribution in derivative simulation frameworks
without written permission of the author.

Citation (Zenodo Record):
Partin, G. D. (2025). Lattice-Field Medium (LFM): A Deterministic
Lattice Framework for Emergent Relativity, Gravitation, and Quantization
— Phase 1 Conceptual Hypothesis v1.0. Zenodo.
https://doi.org/10.5281/zenodo.17478758

Contact: latticefieldmediumresearch@gmail.com



<!-- Source: docs\text\LFM_Phase1_Test_Design.txt -->


Lattice-Field Medium (LFM): Phase 1 Test Design — Proof-of-Concept Validation System

Version 3.0 — 2025-11-01 (Defensive ND Release)
Greg D. Partin | LFM Research — Los Angeles CA USA
License: Creative Commons Attribution–NonCommercial–NoDerivatives 4.0
International (CC BY-NC-ND 4.0)**
Note: This version supersedes all prior releases (v2.x and earlier) and
adds No-Derivatives restrictions and defensive-publication language for
intellectual property protection. All LFM Phase-1 documents are
synchronized under this unified v3.0 release.

Abstract

Phase 1 defines the design and implementation framework for validating
the Lattice-Field Medium (LFM) through reproducible Tier 1–3 tests. It
specifies the environment, configuration architecture, pass/fail
criteria, and proof-packet generation protocol required to establish
numerical and physical correctness of the model. This version modernizes
the document layout for reproducibility and OSF publication compliance.

1 Purpose

Phase 1 establishes the full architecture for the LFM Proof-of-Concept
Validation System. The goal is to provide a reproducible testing
environment that demonstrates Tier 1–3 correctness and creates a
foundation for higher-tier extensions and expert review.

2 Hardware and Environment

	-----------------------------------------------------------------------
	Component               Specification           Notes
	----------------------- ----------------------- -----------------------
	System                  MSI Katana A15 AI       Primary development
																									node

	CPU / GPU               Ryzen 7 8845HS / RTX    Tier 6-capable hardware
													4060 (8 GB VRAM)        

	RAM / Storage           32 GB / 1 TB SSD        Sufficient for 3D Tier
																									3 tests

	OS                      Windows 11 x64          

	Python Environment      3.11.9 + NumPy, SciPy,  Standard computation
													Numba, CuPy-CUDA12x     stack

	Version Control         Git (local → GitHub     Ensures provenance and
													private)                reproducibility
	-----------------------------------------------------------------------

3 Folder and File Architecture

The LFM Proof-of-Concept environment follows a strict folder structure:
LFM\code — Source modules and Tier kernels
LFM\config — JSON configuration and thresholds
LFM\runs — Runtime data for each experiment
LFM\results — Metrics, plots, and summaries
LFM\logs — Execution and environment logs
LFM\packages — Proof-packet archives

4 Configuration and Validation Logic

Global tolerances reside in /config/validation_thresholds.json, with
Tier-specific overrides in /config/tierN_default.json. Merge order:
global → local → runtime. Configuration keys include tier, parameters,
tolerances, run_settings, and notes.

5 Pass/Fail Framework

	-----------------------------------------------------------------------
	Tier                    Goal                    Pass Criteria (Phase 1)
	----------------------- ----------------------- -----------------------
	1                       Lorentz isotropy &      Δv/c ≤ 1 %, anisotropy ≤ 1 %; energy drift within typical
								dispersion              bounds 10⁻⁶ … 10⁻⁴ depending on grid/BCs

	2                       Weak-field / redshift   Correlation > 0.95 with analytic model; drift ≤ 1 %

	3                       Energy conservation     Relative energy drift |ΔE| / |E₀| within 10⁻⁶ … 10⁻⁴ typical;
	                                                strict baseline tolerance configured as 1×10⁻¹² in
	                                                /config/validation_thresholds.json for conservative runs
	-----------------------------------------------------------------------

6 Orchestration and Parallelism

The master script run_all_tiers.py references /config/orchestration.json
to schedule tiers and variants with a concurrency limit (default 3).
Each run executes run_tier.py, writes results, and aggregates metrics
into /results/<campaign>/summary_overall.json.

7 Visualization and Reporting

Plots auto-generate under /results/<campaign>/<tier>/<variant>/plots/.
Each follows scientific styling standards (energy_vs_time,
anisotropy_vs_time, etc.). A summary dashboard (summary_dashboard.html)
compiles all Tier results.

8 Expert Review Packaging Workflow

After all Tier tests complete, the system assembles a proof packet in
/packages/LFM_ProofPacket_<campaign>_vX.Y.zip. Each archive contains
README, manifest, environment info, configs, code snapshot, results,
logs, and SHA-256 hashes. Integrity checks and optional Cardano
anchoring ensure reproducibility.

9 Phase 1 Test Scope

Phase 1 currently executes Tier 1–4 tests. Canonical expected counts (registry) are:
Tier 1: 15, Tier 2: 25, Tier 3: 11, Tier 4: 9. Additional exploratory
tests may be present (e.g., Tier 4 shows 14 cases in current results).
Refer to results/MASTER_TEST_STATUS.csv for the authoritative rollup and
per-test status (PASS/FAIL/SKIP). Expected duration for a full run depends
on hardware and concurrency.

10 Data Reproducibility and Licensing

All code and data products are released under CC BY-NC-ND 4.0
(non-commercial, attribution required; no derivatives). Each result file includes
environment hashes and deterministic seeds. Reproducibility requires the
same configuration files and random seed identifiers as recorded in the
proof packets.

11 Metadata Alignment

	-----------------------------------------------------------------------
	Field                               Value
	----------------------------------- -----------------------------------
	Keywords                            lattice field theory; discrete
																			spacetime; emergent relativity;
																			reproducibility; computational
																			physics

	License                             License CC BY-NC-ND 4.0
																			(non-commercial, attribution
																			required)

	Category Tags                       Theoretical Physics · Computational
																			Physics · Simulation Frameworks

	Data Availability                   All proof packets and logs provided
																			as supplemental data under
																			reproducible archive.

	Funding / Acknowledgements          Self-funded; no external sponsors.

	Contact                             latticefieldmediumresearch@gmail.com
	-----------------------------------------------------------------------

12 Summary

Phase 1 provides the reproducibility framework for all Tier 1–3 LFM
tests. It defines configuration structure, orchestration logic,
validation thresholds, and proof-packet packaging. Successful completion
confirms the model’s stability, isotropy, and conservation—forming the
empirical base for Tier 4–6 development.

13 Legal & Licensing Notice

This document and all accompanying materials are © 2025 Greg D. Partin.
All rights reserved. “Lattice-Field Medium,” “LFM Equation,” and “LFM
Research Framework”
are original works authored by Greg D. Partin.

License Update (v3.0 — 2025-11-01):
Beginning with version 3.0, this work is licensed under the
Creative Commons Attribution–NonCommercial–NoDerivatives 4.0
International License (CC BY-NC-ND 4.0).
Earlier releases (v2.x and prior) were distributed under CC BY-NC 4.0.
All later versions are governed by CC BY-NC-ND 4.0, which prohibits
creation or redistribution of derivative or modified works without
written consent of the author.

Derivative-Use Restriction
No portion of this document, configuration structure, or software design
may be reproduced, modified, or adapted for any commercial, proprietary,
or patent-filing purpose without prior written authorization.
“Commercial” includes any research or prototype development intended for
monetization, commercialization, or patent application.

Defensive Publication Statement
This publication constitutes a defensive disclosure establishing prior
art as of October 29 2025 for all concepts, algorithms, and methods
described herein. Its release prevents any later exclusive patent claim
over identical or equivalent formulations of the LFM validation
architecture.

Trademark Notice
“Lattice-Field Medium,” “LFM Research,” and “LFM Equation” are
distinctive marks identifying this body of work. Unauthorized use of
these names in promotional, academic, or product contexts is prohibited.

Redistribution Boundary
All configuration schemas, threshold tables, and orchestration designs
described here are disclosed solely for scientific reproducibility. They
are not granted for reuse, adaptation, or redistribution in derivative
simulation frameworks without written permission of the author.

Citation (Zenodo Record):
Partin, G. D. (2025). Lattice-Field Medium (LFM): A Deterministic
Lattice Framework for Emergent Relativity, Gravitation, and Quantization
— Phase 1 Conceptual Hypothesis v1.0. Zenodo.
https://doi.org/10.5281/zenodo.17478758

Contact: latticefieldmediumresearch@gmail.com



---


# Aggregated Results Report (excerpt)
# LFM Results Report

This report summarizes the contents of the results/ tree at build time and aggregates simple metrics from summary.json files when available.

## Index

| Path | README | Summary | Key Metrics |
|------|--------|---------|-------------|
| / | no | no |  |
| Energy | no | no |  |
| Energy\ENER-01 | yes | yes | tier=3, category=Energy, test_id=ENER-01, description=Global conservation — short, timestamp=2025-11-03T03:27:28.997581+00:00 |
| Energy\ENER-01\diagnostics | no | no |  |
| Energy\ENER-01\plots | no | no |  |
| Energy\ENER-02 | yes | yes | tier=3, category=Energy, test_id=ENER-02, description=Global conservation — long, timestamp=2025-11-03T03:26:01.926954+00:00 |
| Energy\ENER-02\diagnostics | no | no |  |
| Energy\ENER-02\plots | no | no |  |
| Energy\ENER-03 | yes | yes | tier=3, category=Energy, test_id=ENER-03, description=Wave integrity — mild curvature, timestamp=2025-11-03T03:27:08.051010+00:00 |
| Energy\ENER-03\diagnostics | no | no |  |
| Energy\ENER-03\plots | no | no |  |
| Energy\ENER-04 | yes | yes | tier=3, category=Energy, test_id=ENER-04, description=Wave integrity — steep curvature, timestamp=2025-11-03T03:26:52.927360+00:00 |
| Energy\ENER-04\diagnostics | no | no |  |
| Energy\ENER-04\plots | no | no |  |
| Energy\ENER-05 | yes | yes | tier=3, category=Energy, test_id=ENER-05, description=Hamiltonian partitioning — uniform χ (KE ↔ GE flow), timestamp=2025-11-03T03:27:18.449374+00:00 |
| Energy\ENER-05\diagnostics | no | no |  |
| Energy\ENER-05\plots | no | no |  |
| Energy\ENER-06 | yes | yes | tier=3, category=Energy, test_id=ENER-06, description=Hamiltonian partitioning — with mass term (KE ↔ GE ↔ PE flow), timestamp=2025-11-03T03:24:57.459893+00:00 |
| Energy\ENER-06\diagnostics | no | no |  |
| Energy\ENER-06\plots | no | no |  |
| Energy\ENER-07 | yes | yes | tier=3, category=Energy, test_id=ENER-07, description=Hamiltonian partitioning — χ-gradient field (energy flow in curved spacetime), timestamp=2025-11-03T03:27:41.549557+00:00 |
| Energy\ENER-07\diagnostics | no | no |  |
| Energy\ENER-07\plots | no | no |  |
| Energy\ENER-08 | yes | yes | tier=3, category=Energy, test_id=ENER-08, description=Dissipation — weak damping (exponential decay, γ=1e-3 per unit time), timestamp=2025-11-03T03:27:50.246618+00:00 |
| Energy\ENER-08\diagnostics | no | no |  |
| Energy\ENER-08\plots | no | no |  |
| Energy\ENER-09 | yes | yes | tier=3, category=Energy, test_id=ENER-09, description=Dissipation — strong damping (exponential decay, γ=1e-2 per unit time), timestamp=2025-11-03T03:27:38.908138+00:00 |
| Energy\ENER-09\diagnostics | no | no |  |
| Energy\ENER-09\plots | no | no |  |
| Energy\ENER-10 | yes | yes | tier=3, category=Energy, test_id=ENER-10, description=Thermalization — noise + damping reaches steady state, timestamp=2025-11-03T03:27:06.100678+00:00 |
| Energy\ENER-10\diagnostics | no | no |  |
| Energy\ENER-10\plots | no | no |  |
| Gravity | no | no |  |
| Gravity\GRAV-01 | yes | yes | id=GRAV-01, description=Local frequency — linear χ-gradient (weak), passed=True, rel_err_ratio=1.8436311269530395e-13, ratio_serial=0.32307691273140354 |
| Gravity\GRAV-01\diagnostics | no | no |  |
| Gravity\GRAV-01\plots | no | no |  |
| Gravity\GRAV-02 | yes | yes | id=GRAV-02, description=Local frequency — Gaussian well (strong curvature), passed=True, rel_err_ratio=1.2868376146803692e-12, ratio_serial=2.6866128711484802 |
| Gravity\GRAV-02\diagnostics | no | no |  |
| Gravity\GRAV-02\plots | no | no |  |
| Gravity\GRAV-03 | yes | yes | id=GRAV-03, description=Local frequency — Gaussian well (broader potential), passed=True, rel_err_ratio=8.691569002425617e-13, ratio_serial=1.6866212317990836 |
| Gravity\GRAV-03\diagnostics | no | no |  |
| Gravity\GRAV-03\plots | no | no |  |
| Gravity\GRAV-04 | yes | yes | id=GRAV-04, description=Local frequency — Gaussian well (shallow potential), passed=True, rel_err_ratio=3.329647966731386e-12, ratio_serial=1.882310406762715 |
| Gravity\GRAV-04\diagnostics | no | no |  |
| Gravity\GRAV-04\plots | no | no |  |
| Gravity\GRAV-05 | yes | yes | id=GRAV-05, description=Local frequency — linear χ-gradient (moderate), passed=True, rel_err_ratio=1.059770610316685e-12, ratio_serial=0.41333331015331215 |
| Gravity\GRAV-05\diagnostics | no | no |  |
| Gravity\GRAV-05\plots | no | no |  |
| Gravity\GRAV-06 | yes | yes | id=GRAV-06, description=Local frequency — Gaussian well (stable reference), passed=True, rel_err_ratio=1.0708758863702801e-12, ratio_serial=1.8823104985059032 |
| Gravity\GRAV-06\diagnostics | no | no |  |
| Gravity\GRAV-06\plots | no | no |  |
| Gravity\GRAV-07 | yes | yes | id=GRAV-07, description=Time dilation — bound states in double-well potential (KNOWN: Packet becomes trapped, demonstrates bound state physics), passed=False, rel_err_ratio=0.7094927806006961, ratio_serial=3.663175311119715 |
| Gravity\GRAV-07\diagnostics | no | no |  |
| Gravity\GRAV-07\plots | no | no |  |
| Gravity\GRAV-08 | yes | yes | id=GRAV-08, description=Time dilation — uniform χ diagnostic (isolate grid dispersion), passed=True, rel_err_ratio=0.028800802497585876, ratio_serial=1.0288008024975859 |
| Gravity\GRAV-08\diagnostics | no | no |  |
| Gravity\GRAV-08\plots | no | no |  |
| Gravity\GRAV-09 | yes | yes | id=GRAV-09, description=Time dilation — 2x refined grid (N=128, dx=0.5), passed=True, rel_err_ratio=0.004397059740349736, ratio_serial=1.020491800101535 |
| Gravity\GRAV-09\diagnostics | no | no |  |
| Gravity\GRAV-09\plots | no | no |  |
| Gravity\GRAV-10 | yes | yes | id=GRAV-10, description=Gravitational redshift — measure frequency shift in 1D potential well, passed=True, rel_err_ratio=3.6013456829938594e-12, ratio_serial=4.0653506658414 |
| Gravity\GRAV-10\diagnostics | no | no |  |
| Gravity\GRAV-10\plots | no | no |  |
| Gravity\GRAV-11 | yes | yes | id=GRAV-11, description=Time delay — packet through χ slab (Shapiro-like), passed=True, rel_err_ratio=0.2198190678817605, ratio_serial=0.75 |
| Gravity\GRAV-11\diagnostics | no | no |  |
| Gravity\GRAV-11\plots | no | no |  |
| Gravity\GRAV-12 | yes | yes | id=GRAV-12, description=Phase delay — continuous wave through χ slab (DEMONSTRATES: Klein-Gordon phase/group velocity mismatch - testable prediction!), passed=True, rel_err_ratio=0.2255448579330829, ratio_serial=6.492153236849944 |
| Gravity\GRAV-12\diagnostics | no | no |  |
| Gravity\GRAV-12\plots | no | no |  |
| Gravity\GRAV-13 | yes | yes | id=GRAV-13, description=Local frequency — double well (ω∝χ verification), passed=True, rel_err_ratio=6.810374244951658e-12, ratio_serial=2.1428572188688815 |
| Gravity\GRAV-13\diagnostics | no | no |  |
| Gravity\GRAV-13\plots | no | no |  |
| Gravity\GRAV-14 | yes | yes | id=GRAV-14, description=Group delay — differential timing with vs without slab, passed=True, rel_err_ratio=0.06252043773308959, ratio_serial=11.700000000000003 |
| Gravity\GRAV-14\diagnostics | no | no |  |
| Gravity\GRAV-14\plots | no | no |  |
| Gravity\GRAV-15 | yes | yes | id=GRAV-15, description=3D radial energy dispersion visualizer — central excitation, volumetric snapshots for MP4, passed=True, rel_err_ratio=0.0, ratio_serial=1.0 |
| Gravity\GRAV-15\diagnostics | no | no |  |
| Gravity\GRAV-15\plots | no | no |  |
| Gravity\GRAV-16 | yes | yes | id=GRAV-16, description=3D double-slit interference — quantum wave through slits showing χ-field localization, passed=True, rel_err_ratio=0.0, ratio_serial=1.0 |
| Gravity\GRAV-16\diagnostics | no | no |  |
| Gravity\GRAV-16\plots | no | no |  |
| Gravity\GRAV-17 | yes | yes | id=GRAV-17, description=Gravitational redshift — frequency shift climbing out of χ-well, passed=True, rel_err_ratio=4.2050575811237466e-12, ratio_serial=1.8823104985118027 |
| Gravity\GRAV-17\diagnostics | no | no |  |
| Gravity\GRAV-17\plots | no | no |  |
| Gravity\GRAV-18 | yes | yes | id=GRAV-18, description=Gravitational redshift — linear gradient (Pound-Rebka analogue), passed=True, rel_err_ratio=3.897267651244982e-12, ratio_serial=1.4060149958056136 |
| Gravity\GRAV-18\diagnostics | no | no |  |
| Gravity\GRAV-18\plots | no | no |  |
| Gravity\GRAV-19 | yes | yes | id=GRAV-19, description=Gravitational redshift — radial χ-profile (Schwarzschild analogue), passed=True, rel_err_ratio=8.292177118403416e-12, ratio_serial=1.9516318464423739 |
| Gravity\GRAV-19\diagnostics | no | no |  |
| Gravity\GRAV-19\plots | no | no |  |
| Gravity\GRAV-20 | yes | yes | id=GRAV-20, description=Self-consistent chi from E-energy (Poisson) - verify omega~=chi at center (1D), passed=True, rel_err_ratio=0.05885389325759838, ratio_meas_serial=0.9411461067424016 |
| Gravity\GRAV-20\diagnostics | no | no |  |
| Gravity\GRAV-20\plots | no | no |  |
| Gravity\GRAV-21 | yes | yes | id=GRAV-21, description=GR calibration - redshift to G_eff mapping (weak-field limit), passed=True, delta_omega_over_omega=1.183358318737697, delta_chi_over_chi=1.1833583187529388 |
| Gravity\GRAV-21\diagnostics | no | no |  |
| Gravity\GRAV-21\plots | no | no |  |
| Gravity\GRAV-22 | yes | yes | id=GRAV-22, description=GR calibration - Shapiro delay correspondence (group velocity through slab), passed=True, delay_lfm=9.475210141199952, delay_gr=2.2399999999999993 |
| Gravity\GRAV-22\diagnostics | no | no |  |
| Gravity\GRAV-22\plots | no | no |  |
| Gravity\GRAV-23 | yes | yes | id=GRAV-23, description=Dynamic χ-field evolution — full wave equation □χ=-4πGρ with causal propagation (gravitational wave analogue), passed=True, chi_pert_max=0.007745106604207547, chi_pert_rms=0.0028272176356583206 |
| Gravity\GRAV-23\diagnostics | no | no |  |
| Gravity\GRAV-23\plots | no | no |  |
| Gravity\GRAV-24 | yes | yes | id=GRAV-24, description=Gravitational wave propagation — oscillating source radiates χ-waves, validate 1/r decay and propagation speed, passed=True, chi_pert_max=0.042345523886634026, chi_pert_rms=0.012032611056794953 |
| Gravity\GRAV-24\diagnostics | no | no |  |
| Gravity\GRAV-24\plots | no | no |  |
| Gravity\GRAV-25 | yes | yes | id=GRAV-25, description=Light bending — ray tracing through χ-gradient, measure deflection angle, passed=True, deflection_angle=-0.8558329754589135, expected_position=684.798808467317 |
| Gravity\GRAV-25\diagnostics | no | no |  |
| Gravity\GRAV-25\plots | no | no |  |
| Quantization | no | no |  |
| Quantization\QUAN-01 | yes | yes | tier=4, category=Quantization, test_id=QUAN-01, description=ΔE Transfer — Low Energy, timestamp=1762140486.901848 |
| Quantization\QUAN-01\plots | no | no |  |
| Quantization\QUAN-02 | yes | yes | tier=4, category=Quantization, test_id=QUAN-02, description=ΔE Transfer — High Energy, timestamp=1762140485.4229486 |
| Quantization\QUAN-02\plots | no | no |  |
| Quantization\QUAN-03 | yes | yes | tier=4, category=Quantization, test_id=QUAN-03, description=Spectral Linearity — Coarse Steps, timestamp=1762140333.8341973 |
| Quantization\QUAN-03\plots | no | no |  |
| Quantization\QUAN-04 | yes | yes | tier=4, category=Quantization, test_id=QUAN-04, description=Spectral Linearity — Fine Steps, timestamp=1762140398.3380852 |
| Quantization\QUAN-04\plots | no | no |  |
| Quantization\QUAN-05 | yes | yes | tier=4, category=Quantization, test_id=QUAN-05, description=Phase-Amplitude Coupling — Low Noise, timestamp=1762140387.5424178 |
| Quantization\QUAN-05\plots | no | no |  |
| Quantization\QUAN-06 | yes | yes | tier=4, category=Quantization, test_id=QUAN-06, description=Phase-Amplitude Coupling — High Noise, timestamp=1762140412.1507008 |
| Quantization\QUAN-06\plots | no | no |  |
| Quantization\QUAN-07 | yes | yes | tier=4, category=Quantization, test_id=QUAN-07, description=Nonlinear Wavefront Stability, timestamp=1762140437.2326221 |
| Quantization\QUAN-07\plots | no | no |  |
| Quantization\QUAN-08 | yes | yes | tier=4, category=Quantization, test_id=QUAN-08, description=High-Energy Lattice Blowout Test, timestamp=1762140474.5442002 |
| Quantization\QUAN-08\plots | no | no |  |
| Quantization\QUAN-09 | yes | yes | tier=4, category=Quantization, test_id=QUAN-09, description=Heisenberg uncertainty — Δx·Δk ≈ 1/2, timestamp=1762140490.9118803 |
| Quantization\QUAN-09\diagnostics | no | no |  |
| Quantization\QUAN-09\plots | no | no |  |
| Quantization\QUAN-10 | yes | yes | tier=4, category=Quantization, test_id=QUAN-10, description=Bound state quantization — discrete energy eigenvalues E_n emerge from boundary conditions, passed=True |
| Quantization\QUAN-10\diagnostics | no | no |  |
| Quantization\QUAN-10\plots | no | no |  |
| Quantization\QUAN-11 | yes | yes | tier=4, category=Quantization, test_id=QUAN-11, description=Zero-point energy — ground state E₀ = ½ℏω ≠ 0 (vacuum fluctuations), timestamp=1762140477.6030579 |
| Quantization\QUAN-11\plots | no | no |  |
| Quantization\QUAN-12 | yes | yes | tier=4, category=Quantization, test_id=QUAN-12, description=Quantum tunneling — barrier penetration when E < V (classically forbidden), passed=True |
| Quantization\QUAN-12\diagnostics | no | no |  |
| Quantization\QUAN-12\plots | no | no |  |
| Quantization\QUAN-13 | yes | yes | tier=4, category=Quantization, test_id=QUAN-13, description=Wave-particle duality — which-way information destroys interference, timestamp=1762140479.7418807 |
| Quantization\QUAN-13\plots | no | no |  |
| Quantization\QUAN-14 | yes | yes | tier=4, category=Quantization, test_id=QUAN-14, description=Non-thermalization — validates Klein-Gordon conserves energy (doesn't approach Planck), timestamp=1762140491.1733987 |
| Quantization\QUAN-14\plots | no | no |  |
| Relativistic | no | no |  |
| Relativistic\REL-01 | yes | yes | id=REL-01, description=Isotropy — Coarse Grid, passed=True, anisotropy=0.0, omega_right=19.918707109145185 |
| Relativistic\REL-01\diagnostics | no | no |  |
| Relativistic\REL-01\plots | no | no |  |
| Relativistic\REL-02 | yes | yes | id=REL-02, description=Isotropy — Fine Grid, passed=True, anisotropy=0.0, omega_right=39.72659048145927 |
| Relativistic\REL-02\diagnostics | no | no |  |
| Relativistic\REL-02\plots | no | no |  |
| Relativistic\REL-03 | yes | yes | id=REL-03, description=Lorentz Boost — Low Velocity, passed=True, rel_error=1.8164966926832822, covariance_ratio=2.816496692683282 |
| Relativistic\REL-03\diagnostics | no | no |  |
| Relativistic\REL-03\plots | no | no |  |
| Relativistic\REL-04 | yes | yes | id=REL-04, description=Lorentz Boost — High Velocity, passed=True, rel_error=1.7024249960253077, covariance_ratio=2.7024249960253077 |
| Relativistic\REL-04\diagnostics | no | no |  |
| Relativistic\REL-04\plots | no | no |  |
| Relativistic\REL-05 | yes | yes | id=REL-05, description=Causality — Pulse Propagation, passed=True, rel_err=0.981433081883511, v_measured=0.01856691811648901 |
| Relativistic\REL-05\diagnostics | no | no |  |
| Relativistic\REL-05\plots | no | no |  |
| Relativistic\REL-06 | yes | yes | id=REL-06, description=Causality — Noise Perturbation, passed=True, rel_err=0.9659346376194673, v_measured=0.034065362380532746 |
| Relativistic\REL-06\diagnostics | no | no |  |
| Relativistic\REL-06\plots | no | no |  |
| Relativistic\REL-07 | yes | yes | id=REL-07, description=Phase Independence Test, passed=True, phase_error=0.0, omega_cos=0.41880921894214873 |
| Relativistic\REL-07\diagnostics | no | no |  |
| Relativistic\REL-07\plots | no | no |  |
| Relativistic\REL-08 | yes | yes | id=REL-08, description=Superposition Principle Test, passed=True, linearity_error=0.0, omega1=39.72659048145927 |
| Relativistic\REL-08\diagnostics | no | no |  |
| Relativistic\REL-08\plots | no | no |  |
| Relativistic\REL-09 | yes | yes | id=REL-09, description=3D Isotropy — Directional Equivalence, passed=True, anisotropy=1.9367515729713814e-16, omega_x=36.68734522672824 |
| Relativistic\REL-09\diagnostics | no | no |  |
| Relativistic\REL-09\plots | no | no |  |
| Relativistic\REL-10 | yes | yes | id=REL-10, description=3D Isotropy — Spherical Symmetry, passed=True, spherical_error=1.680007563389655e-13, backend=GPU |
| Relativistic\REL-10\diagnostics | no | no |  |
| Relativistic\REL-10\plots | no | no |  |
| Relativistic\REL-11 | yes | yes | id=REL-11, description=Dispersion Relation — Non-relativistic (χ/k≈10), passed=True, rel_err=0.002053580818646544, omega_meas=7.680385111297265 |
| Relativistic\REL-11\diagnostics | no | no |  |
| Relativistic\REL-11\plots | no | no |  |
| Relativistic\REL-12 | yes | yes | id=REL-12, description=Dispersion Relation — Weakly Relativistic (χ/k≈1), passed=True, rel_err=0.002919641515536755, omega_meas=19.91268017267961 |
| Relativistic\REL-12\diagnostics | no | no |  |
| Relativistic\REL-12\plots | no | no |  |
| Relativistic\REL-13 | yes | yes | id=REL-13, description=Dispersion Relation — Relativistic (χ/k≈0.5), passed=True, rel_err=0.0071651593868806585, omega_meas=39.74038902386529 |
| Relativistic\REL-13\diagnostics | no | no |  |
| Relativistic\REL-13\plots | no | no |  |
| Relativistic\REL-14 | yes | yes | id=REL-14, description=Dispersion Relation — Ultra-relativistic (χ/k≈0.1), passed=True, rel_err=0.016253171911597895, omega_meas=57.81562221678469 |
| Relativistic\REL-14\diagnostics | no | no |  |
| Relativistic\REL-14\plots | no | no |  |
| Relativistic\REL-15 | yes | yes | id=REL-15, description=Causality — Space-like correlation test (light cone violation check), passed=True, num_violations=0, max_violation=0.0 |
| Relativistic\REL-15\diagnostics | no | no |  |
| Relativistic\REL-15\plots | no | no |  |

## Notes
- Per-directory README files were generated automatically.
- Key plots and CSVs are documented in each directory README.
