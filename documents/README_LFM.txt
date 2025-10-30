Lattice-Field Medium (LFM) — Submission Package v2.0 (2025-10-29)
Author: Greg D. Partin | LFM Research — Los Angeles CA USA
CONTACT: gpartin@gmail.com
License: This work is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License (CC BY-NC 4.0).

Note: “LFM Research” refers to an independent personal research project by Greg D. Partin and is not an incorporated entity.

FILES INCLUDED

Primary Research Documents currently included

LFM_Master.docx — Conceptual framework and physical interpretation of the lattice equation.
LFM_Core_Equations.docx — Canonical field equation, discrete update rule, and derivations.
LFM_Phase1_Test_Design.docx — Proof-of-Concept validation design, hardware specs, tolerances, and proof-packet workflow.
LFM_Test_Plan.xlsx — Tier-level pass/fail criteria, tolerances, and seed data.
Executive_Summary.docx — Overview of results, implications, and next-phase objectives.

COMING IN GITHUB PROJECT SOON:

CORE SIMULATION & TEST SUITES
-----------------------------
 run_tier1_relativistic.py        — Tier-1 Relativistic Propagation & Isotropy Suite.
 run_tier2_gravityanalogue.py     — Tier-2 Gravity-Analogue Tests (χ-dependent frequency shift).
 run_unif00_core_principle.py     — UNIF-00 Core Unification Principle Proof Test.
 test_lfm_equation_quick.py       — 1-D Core Equation Regression (Taylor-corrected initialization).
 test_lfm_equation_multidim.py    — Multi-Dimensional Serial vs Parallel Parity Regression.
 test_lfm_equation_parallel_all.py— Unified Serial/Parallel Equation Validation (2-D/3-D).
 test_lfm_dispersion_3d.py        — 3-D Dispersion and Isotropy Verification.

ANALYSIS & VALIDATION UTILITIES
-------------------------------
 analyze.py                       — Generic Regression Analyzer for Diagnostic CSVs.
 analyze_gravity_results.py       — Tier-2 Gravity Result Processor and Comparison Tool.
 benchmark_optimizations.py       — Performance Benchmark (energy summation & runner).
 test_lfm_logger.py               — Concurrent Logging and JSONL Integrity Unit Test.

CORE PHYSICS ENGINE MODULES
---------------------------
 lfm_equation.py                  — Canonical LFM Lattice Solver (∂²E/∂t² = c²∇²E − χ²E).
 lfm_parallel.py                  — Parallel/Threaded Lattice Runner with Monitoring.
 lfm_diagnostics.py               — Energy, Spectrum, and Phase Correlation Diagnostics.
 lfm_visualizer.py                — Visualization Engine (PNG/GIF/MP4 Generation).
 lfm_plotting.py                  — Standard Plotting Utilities (Energy, Entropy, Spectrum).
 lfm_results.py                   — Unified Result Writer (JSON/CSV Proof Bundles).
 lfm_logger.py                    — Dual-Format Session Logger (Text + JSONL).
 lfm_console.py                   — Console Utilities for Tier Execution and Progress.

NUMERIC INTEGRITY & ENERGY TRACKING
-----------------------------------
 numeric_integrity.py             — CFL Stability, Field Sanity, and Drift Warning Mixin.
 energy_monitor.py                — Deterministic Energy Drift Tracker (Buffered CSV Output).

--------------------------------------------------------------------
Purpose:
  Provides a full reference of all executable and analytical modules
  comprising the LFM proof-of-concept framework.  Used for automated
  discovery, tier execution, and reproducibility documentation.

Last Updated: 2025-10-29

DIRECTORY STRUCTURE
/Code — simulation scripts  
Code/Config — JSON configuration  
Code/Results — metrics, plots, and diagnostics  
  
All directories and data are released under CC BY-NC 4.0 for non-commercial reproducible research.

Revision Notes (v2.0 — 2025-10-29)

This version supersedes LFM Submission Package v1.0 (2025-10-27) and incorporates the following updates and clarifications:

Document Revisions

LFM_Master.docx (v2.0): Expanded interpretation of curvature field χ as emergent gravitational potential; clarified Tier-mapping and entropy-time relationship.

LFM_Core_Equations.docx (v1.1): Added formal continuum-limit definition (Σ → ∫ E(x) dx) 

LFM_Phase1_Test_Design.docx (v2.1): Corrected licensing text.

Executive_Summary.docx (v3.0): Added recent Tier-1–3 validation metrics, χ-feedback summary, and revised implications section.

LFM_Test_Plan.xlsx: Confirmed all Tier 1–3 parameters and tolerances; unified seed index naming.

System & Repository Updates

Introduced standardized folder structure (LFM/code, LFM/config, LFM/results, etc.) for reproducibility.

Added complete Python module inventory for the upcoming public GitHub project.

Incorporated numeric integrity and deterministic energy-tracking utilities for audit reproducibility.

Harmonized version metadata, license text, and environment specifications across all files.

Summary
Version 2.0 finalizes the Phase 1 conceptual and numerical baseline for the Lattice-Field Medium project, ensuring full internal consistency between conceptual, mathematical, and computational components while preserving the canonical field law unchanged.