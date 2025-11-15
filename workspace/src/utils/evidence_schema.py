#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evidence Schema - World-Class Test Artifacts Definition

Defines required physics-driven artifacts per test domain.
Each test produces domain-appropriate diagnostics and plots that show:
  1. What happened (time series, field evolution)
  2. How physics emerged (dispersion, convergence)
  3. Quantitative validation (theory comparison)
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from pathlib import Path


@dataclass
class ArtifactSpec:
    """Specification for a required artifact."""
    name: str
    path: str  # Relative to test results dir
    artifact_type: str  # 'csv', 'png', 'gif'
    description: str
    required: bool = True
    
    def full_path(self, test_dir: Path) -> Path:
        return test_dir / self.path


@dataclass
class DomainSchema:
    """Evidence requirements for a test domain."""
    domain: str
    tier: int
    required_diagnostics: List[ArtifactSpec]
    required_plots: List[ArtifactSpec]
    optional_artifacts: List[ArtifactSpec]
    
    def all_required(self) -> List[ArtifactSpec]:
        return [a for a in (self.required_diagnostics + self.required_plots) if a.required]
    
    def validate_test_output(self, test_dir: Path) -> Dict[str, bool]:
        """Check which required artifacts exist."""
        results = {}
        for artifact in self.all_required():
            full = artifact.full_path(test_dir)
            results[artifact.name] = full.exists()
        return results


# ===================== TIER 1: RELATIVISTIC =====================

TIER1_ISOTROPY = DomainSchema(
    domain="isotropy",
    tier=1,
    required_diagnostics=[
        ArtifactSpec(
            "frequency_measurement",
            "diagnostics/frequency_measurement.csv",
            "csv",
            "Measured frequencies: direction, omega_measured, anisotropy, energy_drift"
        ),
    ],
    required_plots=[
        ArtifactSpec(
            "isotropy_comparison",
            "plots/isotropy_comparison.png",
            "png",
            "Frequency comparison (left/right) with error metrics"
        ),
    ],
    optional_artifacts=[
        ArtifactSpec(
            "energy_evolution",
            "diagnostics/energy_evolution.csv",
            "csv",
            "Energy conservation time series (optional)",
            required=False
        ),
    ]
)

TIER1_BOOST = DomainSchema(
    domain="lorentz_boost",
    tier=1,
    required_diagnostics=[
        ArtifactSpec(
            "boost_analysis",
            "diagnostics/boost_analysis.csv",
            "csv",
            "Frame comparison: frame, omega, residual_rms, gamma"
        ),
        ArtifactSpec(
            "energy_evolution",
            "diagnostics/energy_evolution.csv",
            "csv",
            "Energy time series in lab frame"
        ),
    ],
    required_plots=[
        ArtifactSpec(
            "covariance_test",
            "plots/covariance_test.png",
            "png",
            "Residual comparison: lab vs boosted frame"
        ),
        ArtifactSpec(
            "energy_evolution",
            "plots/energy_evolution.png",
            "png",
            "Energy conservation during evolution"
        ),
    ],
    optional_artifacts=[
        ArtifactSpec(
            "field_snapshots",
            "plots/field_snapshots.gif",
            "gif",
            "Animated field evolution",
            required=False
        ),
    ]
)

TIER1_DISPERSION = DomainSchema(
    domain="dispersion_relation",
    tier=1,
    required_diagnostics=[
        ArtifactSpec(
            "dispersion_measurement",
            "diagnostics/dispersion_measurement.csv",
            "csv",
            "Measured dispersion: quantity, measured, theory, error_pct"
        ),
        ArtifactSpec(
            "projection_series",
            "diagnostics/projection_series.csv",
            "csv",
            "Mode projection time series: time, projection"
        ),
        ArtifactSpec(
            "energy_evolution",
            "diagnostics/energy_evolution.csv",
            "csv",
            "Energy conservation: time, E_rms"
        ),
    ],
    required_plots=[
        ArtifactSpec(
            "dispersion_spectrum",
            "plots/dispersion_spectrum.png",
            "png",
            "FFT spectrum with measured/theory omega peaks"
        ),
        ArtifactSpec(
            "omega_squared_ratio",
            "plots/omega_squared_ratio.png",
            "png",
            "ω²/k² measured vs theory (validates E²=p²c²+m²c⁴)"
        ),
        ArtifactSpec(
            "projection_series",
            "plots/projection_series.png",
            "png",
            "Mode amplitude oscillation over time"
        ),
    ],
    optional_artifacts=[]
)

# ===================== TIER 2: GRAVITY ANALOGUE =====================

TIER2_REDSHIFT = DomainSchema(
    domain="gravitational_redshift",
    tier=2,
    required_diagnostics=[
        ArtifactSpec(
            "frequency_shifts",
            "diagnostics/frequency_shifts.csv",
            "csv",
            "Position-dependent redshift: position, chi, omega_measured, omega_theory, z"
        ),
        ArtifactSpec(
            "chi_profile",
            "diagnostics/chi_profile.csv",
            "csv",
            "Chi field spatial distribution"
        ),
    ],
    required_plots=[
        ArtifactSpec(
            "redshift_curve",
            "plots/redshift_curve.png",
            "png",
            "Redshift z vs chi-field strength"
        ),
        ArtifactSpec(
            "frequency_vs_position",
            "plots/frequency_vs_position.png",
            "png",
            "Omega(x) showing gravitational blueshift/redshift"
        ),
    ],
    optional_artifacts=[]
)

TIER2_DEFLECTION = DomainSchema(
    domain="gravitational_deflection",
    tier=2,
    required_diagnostics=[
        ArtifactSpec(
            "trajectory_data",
            "diagnostics/trajectory_data.csv",
            "csv",
            "Ray path: time, x, y, deflection_angle"
        ),
        ArtifactSpec(
            "chi_field_map",
            "diagnostics/chi_field_map.csv",
            "csv",
            "2D chi-field distribution"
        ),
    ],
    required_plots=[
        ArtifactSpec(
            "ray_trajectory",
            "plots/ray_trajectory.png",
            "png",
            "Wave packet path with chi-field contours"
        ),
        ArtifactSpec(
            "deflection_angle",
            "plots/deflection_angle.png",
            "png",
            "Measured vs predicted deflection"
        ),
    ],
    optional_artifacts=[
        ArtifactSpec(
            "trajectory_animation",
            "plots/trajectory_evolution.gif",
            "gif",
            "Animated ray bending",
            required=False
        ),
    ]
)

# ===================== TIER 3: ENERGY CONSERVATION =====================

TIER3_CONSERVATION = DomainSchema(
    domain="energy_conservation",
    tier=3,
    required_diagnostics=[
        ArtifactSpec(
            "energy_components",
            "diagnostics/energy_components.csv",
            "csv",
            "Energy breakdown: time, E_kinetic, E_potential, E_gradient, E_total"
        ),
        ArtifactSpec(
            "drift_analysis",
            "diagnostics/drift_analysis.csv",
            "csv",
            "Conservation metrics: step, drift, rate_of_change"
        ),
    ],
    required_plots=[
        ArtifactSpec(
            "energy_time_series",
            "plots/energy_time_series.png",
            "png",
            "All energy components vs time"
        ),
        ArtifactSpec(
            "drift_evolution",
            "plots/drift_evolution.png",
            "png",
            "Relative drift over simulation"
        ),
        ArtifactSpec(
            "energy_partition",
            "plots/energy_partition.png",
            "png",
            "Stacked area: kinetic/potential/gradient"
        ),
    ],
    optional_artifacts=[]
)

# ===================== TIER 4: QUANTIZATION =====================

TIER4_BOUND_STATES = DomainSchema(
    domain="bound_states",
    tier=4,
    required_diagnostics=[
        ArtifactSpec(
            "eigenvalues",
            "diagnostics/eigenvalues.csv",
            "csv",
            "Energy levels: mode_n, theory_energy, measured_energy, rel_error"
        ),
        ArtifactSpec(
            "mode_shapes",
            "diagnostics/mode_shapes.csv",
            "csv",
            "Spatial wavefunctions: x, psi_1, psi_2, ..., psi_n"
        ),
    ],
    required_plots=[
        ArtifactSpec(
            "energy_spectrum",
            "plots/quantized_energies.png",
            "png",
            "Discrete energy levels E_n vs quantum number n"
        ),
        ArtifactSpec(
            "mode_shapes",
            "plots/bound_state_modes.png",
            "png",
            "Spatial wavefunctions ψ_n(x) for lowest modes"
        ),
    ],
    optional_artifacts=[
        ArtifactSpec(
            "wavefunction_evolution",
            "plots/wavefunction_evolution.gif",
            "gif",
            "Time evolution of |ψ|²",
            required=False
        ),
    ]
)

TIER4_TUNNELING = DomainSchema(
    domain="tunneling",
    tier=4,
    required_diagnostics=[
        ArtifactSpec(
            "transmission_data",
            "diagnostics/transmission_data.csv",
            "csv",
            "Barrier penetration: energy, transmission_coeff, theory"
        ),
        ArtifactSpec(
            "barrier_profile",
            "diagnostics/barrier_profile.csv",
            "csv",
            "Potential barrier: x, V(x)"
        ),
    ],
    required_plots=[
        ArtifactSpec(
            "transmission_curve",
            "plots/transmission_curve.png",
            "png",
            "T(E) vs barrier height with exponential fit"
        ),
        ArtifactSpec(
            "barrier_schematic",
            "plots/barrier_schematic.png",
            "png",
            "Potential + incident/transmitted wavepacket"
        ),
    ],
    optional_artifacts=[]
)

# ===================== TIER 5: ELECTROMAGNETIC =====================

TIER5_MAXWELL = DomainSchema(
    domain="maxwell_equations",
    tier=5,
    required_diagnostics=[
        ArtifactSpec(
            "field_components",
            "diagnostics/field_components.csv",
            "csv",
            "EM field component summary (initial implementation; may be single snapshot)"
        ),
        ArtifactSpec(
            "poynting_flux",
            "diagnostics/poynting_flux.csv",
            "csv",
            "Poynting flux estimation (time, flux)"
        ),
    ],
    required_plots=[
        ArtifactSpec(
            "poynting_vector",
            "plots/poynting_vector.png",
            "png",
            "Energy flux direction and magnitude"
        ),
    ],
    optional_artifacts=[
        ArtifactSpec(
            "field_evolution",
            "plots/field_evolution.png",
            "png",
            "E and B field magnitudes vs time (optional; requires time series)",
            required=False
        ),
        ArtifactSpec(
            "field_animation",
            "plots/field_animation.gif",
            "gif",
            "Animated E,B field propagation",
            required=False
        ),
    ]
)

# ===================== TIER 2 (ADDITIVE): NEW GRAVITY DOMAINS =====================

TIER2_PHASE_DELAY = DomainSchema(
    domain="phase_delay",
    tier=2,
    required_diagnostics=[
        ArtifactSpec(
            "delay_measurement",
            "diagnostics/delay_measurement.csv",
            "csv",
            "Phase / time delay metrics: ratio_theory, ratio_measured, ratio_error, chi_A, chi_B"
        ),
    ],
    required_plots=[
        ArtifactSpec(
            "arrival_curve",
            "plots/arrival_curve.png",
            "png",
            "Envelope amplitude before vs after slab with delay annotation"
        ),
    ],
    optional_artifacts=[
        ArtifactSpec(
            "envelope_measurement_raw",
            "diagnostics/envelope_measurement_GRAV-12.csv",
            "csv",
            "Raw envelope sampling (time, env_before, env_after)",
            required=False
        ),
    ]
)

TIER2_DYNAMIC_CHI = DomainSchema(
    domain="dynamic_chi",
    tier=2,
    required_diagnostics=[
        ArtifactSpec(
            "dynamic_chi_metrics",
            "diagnostics/dynamic_chi_metrics.csv",
            "csv",
            "Dynamic chi perturbation metrics: chi_pert_max, chi_pert_rms, chi_edge_value, energy_drift_frac"
        ),
    ],
    required_plots=[
        ArtifactSpec(
            "chi_wave_profile",
            "plots/chi_wave_profile.png",
            "png",
            "Annotated summary of χ wave evolution statistics"
        ),
    ],
    optional_artifacts=[
        ArtifactSpec(
            "chi_wave_evolution",
            "plots/chi_wave_evolution_GRAV-23.png",
            "png",
            "Original evolution plot (if produced by test harness)",
            required=False
        ),
        ArtifactSpec(
            "chi_wave_spectrum",
            "plots/chi_wave_spectrum.png",
            "png",
            "Frequency-domain spectrum of χ perturbation amplitude (FFT)",
            required=False
        ),
        ArtifactSpec(
            "chi_wave_spectrum_data",
            "diagnostics/chi_wave_spectrum.csv",
            "csv",
            "Spectrum data: frequency, amplitude",
            required=False
        ),
    ]
)

# ===================== TIER 6: COUPLING =====================

TIER6_COUPLING = DomainSchema(
    domain="coupling",
    tier=6,
    required_diagnostics=[
        ArtifactSpec(
            "coupling_analysis",
            "diagnostics/coupling_analysis.csv",
            "csv",
            "Coupling metrics: coupling_strength_error, convergence validation"
        ),
        ArtifactSpec(
            "wave_speed_profile",
            "diagnostics/wave_speed_profile.csv",
            "csv",
            "Wave propagation speeds: position, measured speed, theory speed"
        ),
    ],
    required_plots=[
        ArtifactSpec(
            "coupling_strength_plot",
            "plots/coupling_strength.png",
            "png",
            "Measured vs theoretical coupling strength visualization"
        ),
    ],
    optional_artifacts=[
        ArtifactSpec(
            "wave_speed_comparison",
            "plots/wave_speed_comparison.png",
            "png",
            "Wave speed measured vs c=1 comparison (if applicable)",
            required=False
        ),
    ]
)

# ===================== TIER 7: THERMODYNAMICS =====================

TIER7_THERMODYNAMICS = DomainSchema(
    domain="thermodynamics",
    tier=7,
    required_diagnostics=[
        ArtifactSpec(
            "entropy_evolution",
            "diagnostics/entropy_evolution.csv",
            "csv",
            "Time series: time, entropy (synthetic approach to equilibrium)"
        ),
        ArtifactSpec(
            "thermalization_metrics",
            "diagnostics/thermalization_metrics.csv",
            "csv",
            "Metrics: entropy_initial/final/increase, tau, energy_drift"
        ),
    ],
    required_plots=[
        ArtifactSpec(
            "entropy_increase_plot",
            "plots/entropy_increase.png",
            "png",
            "Initial vs final entropy bar chart (second law validation)"
        ),
        ArtifactSpec(
            "second_law_validation",
            "plots/second_law_validation.png",
            "png",
            "Entropy increase with tolerance threshold visualization"
        ),
    ],
    optional_artifacts=[
        ArtifactSpec(
            "entropy_time_series",
            "plots/entropy_time_series.png",
            "png",
            "Entropy evolution over time (exponential approach if tau known)",
            required=False
        ),
    ]
)

# ===================== DOMAIN REGISTRY =====================

DOMAIN_SCHEMAS: Dict[str, DomainSchema] = {
    # Tier 1
    "isotropy": TIER1_ISOTROPY,
    "lorentz_boost": TIER1_BOOST,
    "dispersion_relation": TIER1_DISPERSION,
    
    # Tier 2
    "gravitational_redshift": TIER2_REDSHIFT,
    "gravitational_deflection": TIER2_DEFLECTION,
    "phase_delay": TIER2_PHASE_DELAY,
    "dynamic_chi": TIER2_DYNAMIC_CHI,
    
    # Tier 3
    "energy_conservation": TIER3_CONSERVATION,
    
    # Tier 4
    "bound_states": TIER4_BOUND_STATES,
    "tunneling": TIER4_TUNNELING,
    
    # Tier 5
    "maxwell_equations": TIER5_MAXWELL,
    
    # Tier 6
    "coupling": TIER6_COUPLING,
    
    # Tier 7
    "thermodynamics": TIER7_THERMODYNAMICS,
}


def get_schema_for_test(test_id: str) -> Optional[DomainSchema]:
    """
    Infer domain schema from test ID.
    
    Returns None if no specific schema defined (test will pass validation
    as long as summary.json exists).
    """
    test_id = test_id.upper()
    
    # Tier 1 mappings (well-defined)
    if test_id in ("REL-01", "REL-02", "REL-09", "REL-10"):
        return DOMAIN_SCHEMAS["isotropy"]
    elif test_id in ("REL-03", "REL-04", "REL-17"):
        return DOMAIN_SCHEMAS["lorentz_boost"]
    elif test_id in ("REL-11", "REL-12", "REL-13", "REL-14"):
        return DOMAIN_SCHEMAS["dispersion_relation"]
    
    # Tier 2 gravity analogue mappings (incremental: only GRAV-12 & GRAV-23 for now)
    elif test_id == "GRAV-12":
        return DOMAIN_SCHEMAS.get("phase_delay")
    elif test_id == "GRAV-23":
        return DOMAIN_SCHEMAS.get("dynamic_chi")
    elif test_id == "GRAV-13":  # Local frequency / redshift verification
        return DOMAIN_SCHEMAS.get("gravitational_redshift")
    elif test_id == "GRAV-25":  # Light bending / deflection
        return DOMAIN_SCHEMAS.get("gravitational_deflection")
    elif test_id.startswith("GRAV-"):
        # Other gravity tests remain minimal until their schemas are defined
        return None
    
    # Tier 3 mappings
    elif test_id.startswith("ENER-"):
        return DOMAIN_SCHEMAS.get("energy_conservation")
    
    # Tier 4 mappings
    elif test_id in ("QUAN-10",):  # Bound state tests
        return DOMAIN_SCHEMAS.get("bound_states")
    elif test_id in ("QUAN-12",):  # Tunneling
        return DOMAIN_SCHEMAS.get("tunneling")
    
    # Tier 5 mappings
    elif test_id.startswith("EM-"):
        return DOMAIN_SCHEMAS.get("maxwell_equations")
    
    # Tier 6 mappings
    elif test_id.startswith("COUP-"):
        return DOMAIN_SCHEMAS.get("coupling")
    
    # Tier 7 mappings
    elif test_id.startswith("THERM-"):
        return DOMAIN_SCHEMAS.get("thermodynamics")
    
    return None


def validate_test_evidence(test_dir: Path, test_id: str) -> Dict[str, any]:
    """
    Validate that a test produced all required evidence artifacts.
    
    If no schema is defined, validates that summary.json exists (minimal).
    
    Returns:
        {
            "test_id": str,
            "domain": str,
            "complete": bool,
            "missing": List[str],
            "artifacts": Dict[str, bool],
            "schema_status": str  # "validated", "minimal", "no_schema"
        }
    """
    schema = get_schema_for_test(test_id)
    
    # No schema defined - check minimal evidence (summary.json)
    if schema is None:
        summary_exists = (test_dir / "summary.json").exists()
        return {
            "test_id": test_id,
            "domain": "generic",
            "complete": summary_exists,
            "missing": [] if summary_exists else ["summary.json"],
            "artifacts": {"summary.json": summary_exists},
            "schema_status": "minimal"
        }
    
    # Schema defined - validate against it
    validation = schema.validate_test_output(test_dir)
    missing = [name for name, exists in validation.items() if not exists]
    
    return {
        "test_id": test_id,
        "domain": schema.domain,
        "complete": len(missing) == 0,
        "missing": missing,
        "artifacts": validation,
        "schema_status": "validated"
    }
