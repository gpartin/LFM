#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared validation utilities for tier test harnesses.

Goals:
- Centralize pass/fail decisions and threshold lookup from tier metadata
- Reduce duplication in tier runners (energy checks, metric comparisons)
- Provide tolerant metric key mapping between metadata and summary.json

Usage patterns:
- energy_passed = energy_conservation_check(meta, test_id, energy_drift)
- primary_ok, details = check_primary_metric(meta, test_id, summary_dict)

Notes:
- This module does NOT compute physics metrics; it only evaluates them.
- Keep it dependency-light (stdlib + typing + pathlib + json + math).
"""

from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# Map metadata "primary.metric" names → keys expected in summary.json
PRIMARY_METRIC_KEY_MAP: Dict[str, str] = {
    # Tier 1 (Relativistic)
    "anisotropy": "anisotropy",
    "directional_anisotropy": "anisotropy",
    "dispersion_error": "rel_err",  # dispersion variants store rel_err
    "phase_error": "phase_error",
    "linearity_error": "linearity_error",
    "symmetry_error": "spherical_error",
    "spacelike_correlation": "max_violation",  # conservative (treat any violation amplitude)
    "momentum_drift": "momentum_drift",
    "invariant_mass_error": "invariant_mass_error",
    # Covariance residual terminology varies; use rel_error field in summaries
    "covariance_residual": "rel_error",
    # Causality speed caps
    "pulse_speed": "v_measured",
    "max_signal_speed": "v_measured",
    
    # Tier 2 (Gravity Analogue)
    "frequency_shift": "frequency_shift",
    "redshift": "frequency_shift",
    # Local frequency ratio tests
    "local_frequency_ratio_error": "rel_err_ratio",  # prefer central or band median ratio
    # Redshift ratio tests
    "redshift_ratio_error": "ratio_match_error",
    # Allow direct mapping if summary provides same name
    "ratio_match_error": "ratio_match_error",
    # Time dilation tests
    "time_dilation_ratio_error": "time_dilation_ratio_error",
    "time_dilation_uniformity": "time_dilation_uniformity",
    # Time delay / phase delay
    "time_delay_error": "delay_error",
    "delay_error": "delay_error",
    "time_delay": "delay_error",
    "phase_shift_error": "phase_shift_error",
    "phase_delay": "phase_shift_error",
    "phase_delay_consistency": "phase_delay_consistency",
    # Trajectory / geodesic
    "path_deviation": "path_deviation",
    "geodesic_error": "path_deviation",
    "geodesic_deviation": "geodesic_deviation",
    # Self-consistency / calibration
    "consistency_error": "consistency_error",
    "self_consistency": "consistency_error",
    "calibration_deviation": "calibration_deviation",
    # Dynamic chi / amplitude
    "coupling_strength": "coupling_strength",
    "amplitude_error": "amplitude_error",
    # Bending
    "bending_error": "bending_error",
    
    # Tier 3 (Energy Conservation)
    "partition_imbalance": "partition_imbalance",
    "entropy_change": "entropy_change",
    "dissipation_rate": "dissipation_rate",
    "thermalization_error": "thermalization_error",
    
    # Tier 4 (Quantization)
    "eigenvalue_error": "eigenvalue_error",
    "transmission_coefficient": "transmission_coefficient",
    "coherence_time": "coherence_time",
    "uncertainty_product": "uncertainty_product",
    "mode_overlap": "mode_overlap",
    
    # Tier 5 (Electromagnetic)
    "maxwell_residual": "maxwell_residual",
    "gauss_error": "gauss_error",
    "faraday_error": "faraday_error",
    "ampere_error": "ampere_error",
    "poynting_error": "poynting_error",
    "charge_drift": "charge_drift",
    "larmor_power_error": "larmor_power_error",
    "doppler_shift": "doppler_shift",
    "scattering_error": "scattering_error",
    "spectrum_error": "spectrum_error",
    "gauge_error": "gauge_error",
    
    # Tier 6 (Coupling)
    "transfer_efficiency": "transfer_efficiency",
    "amplification": "amplification",
    "resonance": "amplification",
    "coupling_strength_error": "coupling_strength_error",
    "energy_transfer_efficiency": "energy_transfer_efficiency",
    "transmission_coefficient_error": "transmission_coefficient_error",
    "localization_length_error": "localization_length_error",
    "resonance_amplitude_ratio": "resonance_amplitude_ratio",
    "dynamic_response_error": "dynamic_response_error",
    "asymmetry_ratio": "asymmetry_ratio",
    "nonlinearity_strength": "nonlinearity_strength",
    "fringe_visibility": "fringe_visibility",
    "saturation_threshold_error": "saturation_threshold_error",
    
    # Tier 7 (Thermodynamics)
    "entropy_violations": "entropy_violations",
    "equipartition_error": "equipartition_error",
    "temperature_fit_r2": "temperature_fit_r2",
    "irreversibility": "irreversibility",
    "thermalization_time": "thermalization_time",
}


@dataclass
class ValidationResult:
    """Structured validation outcome combining energy conservation and primary metric.

    Fields:
        test_id: Test identifier (e.g., 'REL-01')
        energy_ok: Whether energy drift is within threshold
        energy_drift: Relative energy drift value
        energy_threshold: Threshold used for energy drift comparison
        primary_ok: Whether primary metric meets its threshold
        primary_metric: The metric key used for the primary comparison
        primary_value: The measured value for the primary metric
        primary_threshold: The threshold used for primary metric (None if not applicable)
        metrics: Arbitrary additional metrics included in evaluation
        timestamp: ISO-like timestamp for when the evaluation occurred
    """
    test_id: str
    energy_ok: bool
    energy_drift: float
    energy_threshold: float
    primary_ok: bool
    primary_metric: str
    primary_value: float
    primary_threshold: Optional[float]
    metrics: Dict[str, float]
    timestamp: str


def aggregate_validation(
    meta: Dict,
    test_id: str,
    energy_drift: float,
    metrics: Dict[str, float],
    *,
    energy_default: float = 0.01,
) -> ValidationResult:
    """
    Aggregate energy conservation and primary metric evaluation into a single result.

    Args:
        meta: Loaded tier metadata dictionary
        test_id: Test identifier (e.g., 'REL-01')
        energy_drift: Relative energy drift value
        metrics: Dict of candidate primary metrics (e.g., {"rel_err": 0.012})
        energy_default: Default energy threshold if not in metadata

    Returns:
        ValidationResult with combined outcome.
    """
    energy_ok, energy_thr, _ = energy_conservation_check(meta, test_id, float(energy_drift))
    p_ok, p_key, p_val, p_thr = check_primary_metric(meta, test_id, dict(metrics))
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    return ValidationResult(
        test_id=str(test_id),
        energy_ok=bool(energy_ok),
        energy_drift=float(energy_drift),
        energy_threshold=float(energy_thr if energy_thr is not None else energy_default),
        primary_ok=bool(p_ok),
        primary_metric=str(p_key) if p_key else "",
        primary_value=float(p_val) if p_val is not None else 0.0,
        primary_threshold=float(p_thr) if p_thr is not None else None,
        metrics={k: float(v) for k, v in metrics.items()},
        timestamp=ts,
    )


def validation_block(result: ValidationResult) -> Dict:
    """
    Convert ValidationResult to a summary-embeddable validation block.

    Returns a plain dict suitable for inclusion under summary['validation'].
    """
    d = asdict(result)
    # Keep structure stable and concise in summaries
    return {
        "test_id": d["test_id"],
        "energy": {
            "ok": d["energy_ok"],
            "drift": d["energy_drift"],
            "threshold": d["energy_threshold"],
        },
        "primary": {
            "ok": d["primary_ok"],
            "metric": d["primary_metric"],
            "value": d["primary_value"],
            "threshold": d["primary_threshold"],
        },
        "metrics": d["metrics"],
        "timestamp": d["timestamp"],
    }


def _workspace_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd()


def load_tier_metadata(tier: int) -> Dict:
    """Load tierN_validation_metadata.json for given tier.

    Raises FileNotFoundError if missing.
    """
    cfg = _workspace_root() / "config" / f"tier{tier}_validation_metadata.json"
    with open(cfg, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_test_meta(meta: Dict, test_id: str) -> Optional[Dict]:
    return (meta.get("tests") or {}).get(test_id)


def get_energy_threshold(meta: Dict, test_id: str, default: float = 0.01) -> float:
    t = _get_test_meta(meta, test_id)
    try:
        return float(t["validation_criteria"]["energy_conservation"]["threshold"]) if t else float(default)
    except Exception:
        return float(default)


def energy_conservation_check(meta: Dict, test_id: str, energy_drift: float) -> Tuple[bool, float, str]:
    """Evaluate energy conservation against metadata threshold.

    Returns: (passed, threshold, message)
    """
    # Respect metadata role 'diagnostic' (do not enforce threshold)
    try:
        t = _get_test_meta(meta, test_id) or {}
        crit = (t.get("validation_criteria") or {}).get("energy_conservation", {})
        role = str(crit.get("role", "")).strip().lower()
        if role == "diagnostic":
            # Energy conservation is informational only for this test
            return True, float("inf"), f"diagnostic: drift={energy_drift:.2e}"
    except Exception:
        pass

    thr = get_energy_threshold(meta, test_id, default=0.01)
    ok = (energy_drift < thr)
    msg = (f"drift={energy_drift:.2e}" if ok else f"drift={energy_drift:.2e} (EXCEEDS {thr:.2e})")
    return ok, thr, msg


def _resolve_primary_key(meta_metric_name: str) -> str:
    key = PRIMARY_METRIC_KEY_MAP.get(meta_metric_name.strip().lower())
    return key or meta_metric_name


def check_primary_metric(meta: Dict, test_id: str, summary: Dict) -> Tuple[bool, str, float, Optional[float]]:
    """Check the primary metric declared in metadata against summary.json.

    Returns: (passed, metric_key_used, value, threshold_or_None)
    Falls back to 'passed' flag in summary if metric missing.
    """
    t = _get_test_meta(meta, test_id)
    if not t:
        # If metadata missing, defer to summary 'passed'
        return bool(summary.get("passed", False)), "passed", float(summary.get("rel_err", 0.0)), None

    primary = (t.get("validation_criteria") or {}).get("primary", {})
    name = str(primary.get("metric", "")).strip()
    metric_key = _resolve_primary_key(name) if name else None

    # Threshold may be 'threshold', 'threshold_max', or 'threshold_min'
    thr = primary.get("threshold")
    thr_max = primary.get("threshold_max")
    thr_min = primary.get("threshold_min")

    # Extract value
    val = None
    if metric_key and metric_key in summary:
        try:
            val = float(summary[metric_key])
        except Exception:
            val = None

    if val is None:
        # Try a few known alternates for Tier 1
        alternates = [
            "rel_err", "anisotropy", "phase_error", "linearity_error",
            "omega2_over_k2_error", "max_violation", "momentum_drift",
            "invariant_mass_error", "v_measured"
        ]
        for k in alternates:
            if k in summary:
                try:
                    val = float(summary[k])
                    metric_key = k
                    break
                except Exception:
                    pass

    # If still missing, fall back to boolean passed
    if val is None or (thr is None and thr_max is None and thr_min is None):
        return bool(summary.get("passed", False)), metric_key or "passed", float(summary.get("rel_err", 0.0)), None

    # Compare against threshold (standard upper bound: value < threshold)
    if thr is not None:
        try:
            thr_f = float(thr)
        except Exception:
            thr_f = None
        ok = (val < thr_f) if thr_f is not None else bool(summary.get("passed", False))
        return ok, metric_key or name, val, thr_f

    # threshold_max: value <= threshold_max
    if thr_max is not None:
        try:
            thr_m = float(thr_max)
        except Exception:
            thr_m = None
        ok = (val <= thr_m) if thr_m is not None else bool(summary.get("passed", False))
        return ok, metric_key or name, val, thr_m

    # threshold_min: value >= threshold_min (MINIMUM required value)
    if thr_min is not None:
        try:
            thr_min_f = float(thr_min)
        except Exception:
            thr_min_f = None
        ok = (val >= thr_min_f) if thr_min_f is not None else bool(summary.get("passed", False))
        return ok, metric_key or name, val, thr_min_f

    # Default: pass-through
    return bool(summary.get("passed", False)), metric_key or name, float(val), None


# ---- Helper evaluators (thin wrappers around check_primary_metric) ----

def evaluate_primary(meta: Dict, test_id: str, metrics: Dict[str, float]) -> Tuple[bool, str, float, Optional[float]]:
    """
    Generic primary-metric evaluator using metadata.

    Args:
        meta: Loaded tier metadata dictionary
        test_id: Test identifier (e.g., 'REL-01')
        metrics: Dict of raw metric values keyed by their semantic names

    Returns:
        (ok, metric_key_used, value, threshold)
    """
    return check_primary_metric(meta, test_id, metrics)


def evaluate_isotropy(meta: Dict, test_id: str, anisotropy: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate isotropy using 'anisotropy' as primary metric."""
    return evaluate_primary(meta, test_id, {"anisotropy": float(anisotropy)})


def evaluate_directional_isotropy(meta: Dict, test_id: str, anisotropy: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate 3D directional isotropy; maps to the same 'anisotropy' key."""
    return evaluate_primary(meta, test_id, {"anisotropy": float(anisotropy)})


def evaluate_spherical_symmetry(meta: Dict, test_id: str, spherical_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate spherical isotropy using 'spherical_error' as primary metric."""
    return evaluate_primary(meta, test_id, {"spherical_error": float(spherical_error)})


def evaluate_dispersion(meta: Dict, test_id: str, rel_err: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate dispersion relation accuracy using relative error value (rel_err)."""
    return evaluate_primary(meta, test_id, {"rel_err": float(rel_err)})


def evaluate_spacelike_correlation(meta: Dict, test_id: str, max_violation: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate space-like correlation test using maximum violation amplitude."""
    return evaluate_primary(meta, test_id, {"max_violation": float(max_violation)})


def evaluate_momentum_drift(meta: Dict, test_id: str, momentum_drift: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate momentum conservation using relative momentum change as 'momentum_drift'."""
    return evaluate_primary(meta, test_id, {"momentum_drift": float(momentum_drift)})


def evaluate_invariant_mass(meta: Dict, test_id: str, invariant_mass_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate Lorentz invariance via invariant mass error."""
    return evaluate_primary(meta, test_id, {"invariant_mass_error": float(invariant_mass_error)})


# ===== Tier 2 Evaluators (Gravity Analogue) =====

def evaluate_redshift(meta: Dict, test_id: str, frequency_shift: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate gravitational redshift/blueshift."""
    return evaluate_primary(meta, test_id, {"frequency_shift": float(frequency_shift)})


def evaluate_time_delay(meta: Dict, test_id: str, delay_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate time delay error (measured vs expected)."""
    return evaluate_primary(meta, test_id, {"delay_error": float(delay_error)})


def evaluate_phase_delay(meta: Dict, test_id: str, phase_shift_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate phase delay accuracy."""
    return evaluate_primary(meta, test_id, {"phase_shift_error": float(phase_shift_error)})


def evaluate_geodesic_error(meta: Dict, test_id: str, path_deviation: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate geodesic path deviation from analytical solution."""
    return evaluate_primary(meta, test_id, {"path_deviation": float(path_deviation)})


def evaluate_self_consistency(meta: Dict, test_id: str, consistency_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate self-consistency check (e.g., round-trip accuracy)."""
    return evaluate_primary(meta, test_id, {"consistency_error": float(consistency_error)})


def evaluate_dynamic_chi_coupling(meta: Dict, test_id: str, coupling_strength: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate dynamic χ-field coupling strength."""
    return evaluate_primary(meta, test_id, {"coupling_strength": float(coupling_strength)})


def evaluate_gravitational_wave_amplitude(meta: Dict, test_id: str, amplitude_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate gravitational wave amplitude accuracy."""
    return evaluate_primary(meta, test_id, {"amplitude_error": float(amplitude_error)})


def evaluate_light_bending_angle(meta: Dict, test_id: str, bending_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate light bending angle deviation."""
    return evaluate_primary(meta, test_id, {"bending_error": float(bending_error)})


# ===== Tier 3 Evaluators (Energy Conservation) =====

def evaluate_partition_balance(meta: Dict, test_id: str, kinetic_fraction: float, gradient_fraction: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate energy partition balance (KE vs GE)."""
    # Check if partition is within expected range (metadata should specify)
    imbalance = abs(kinetic_fraction + gradient_fraction - 1.0)
    return evaluate_primary(meta, test_id, {"partition_imbalance": float(imbalance)})


def evaluate_entropy_change(meta: Dict, test_id: str, entropy_initial: float, entropy_final: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate entropy monotonicity (ΔS ≥ 0)."""
    delta_s = entropy_final - entropy_initial
    return evaluate_primary(meta, test_id, {"entropy_change": float(delta_s)})


def evaluate_dissipation_rate(meta: Dict, test_id: str, dissipation: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate dissipation rate (should be near zero for conservative systems)."""
    return evaluate_primary(meta, test_id, {"dissipation_rate": float(dissipation)})


def evaluate_thermalization_time(meta: Dict, test_id: str, time_measured: float, time_expected: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate thermalization timescale accuracy."""
    error = abs(time_measured - time_expected) / (time_expected + 1e-12)
    return evaluate_primary(meta, test_id, {"thermalization_error": float(error)})


# ===== Tier 4 Evaluators (Quantization) =====

def evaluate_eigenvalue_error(meta: Dict, test_id: str, eigenvalue_measured: float, eigenvalue_expected: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate bound state eigenvalue accuracy."""
    error = abs(eigenvalue_measured - eigenvalue_expected) / (abs(eigenvalue_expected) + 1e-12)
    return evaluate_primary(meta, test_id, {"eigenvalue_error": float(error)})


def evaluate_tunneling_probability(meta: Dict, test_id: str, transmission_coefficient: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate quantum tunneling transmission coefficient."""
    return evaluate_primary(meta, test_id, {"transmission_coefficient": float(transmission_coefficient)})


def evaluate_coherence_time(meta: Dict, test_id: str, coherence_duration: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate quantum coherence duration."""
    return evaluate_primary(meta, test_id, {"coherence_time": float(coherence_duration)})


def evaluate_uncertainty_product(meta: Dict, test_id: str, delta_x_delta_k: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate Heisenberg uncertainty relation (Δx·Δk ≥ 1/2)."""
    return evaluate_primary(meta, test_id, {"uncertainty_product": float(delta_x_delta_k)})


def evaluate_spectral_linearity(meta: Dict, test_id: str, linearity_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate spectral response linearity."""
    return evaluate_primary(meta, test_id, {"linearity_error": float(linearity_error)})


def evaluate_mode_orthogonality(meta: Dict, test_id: str, overlap_integral: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate mode orthogonality (overlap should be near zero)."""
    return evaluate_primary(meta, test_id, {"mode_overlap": float(overlap_integral)})


# ===== Tier 5 Evaluators (Electromagnetic) =====

def evaluate_maxwell_residual(meta: Dict, test_id: str, gauss_error: float, faraday_error: float, ampere_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate Maxwell equation residuals (Gauss, Faraday, Ampère)."""
    max_error = max(gauss_error, faraday_error, ampere_error)
    return evaluate_primary(meta, test_id, {"maxwell_residual": float(max_error)})


def evaluate_poynting_conservation(meta: Dict, test_id: str, energy_flux_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate Poynting conservation: ∇·S + ∂u/∂t = 0."""
    return evaluate_primary(meta, test_id, {"poynting_error": float(energy_flux_error)})


def evaluate_charge_conservation(meta: Dict, test_id: str, charge_drift: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate charge conservation: ∂ρ/∂t + ∇·J = 0."""
    return evaluate_primary(meta, test_id, {"charge_drift": float(charge_drift)})


def evaluate_larmor_power(meta: Dict, test_id: str, radiated_power: float, expected_power: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate Larmor radiation power accuracy."""
    error = abs(radiated_power - expected_power) / (expected_power + 1e-12)
    return evaluate_primary(meta, test_id, {"larmor_power_error": float(error)})


def evaluate_doppler_shift(meta: Dict, test_id: str, frequency_shift: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate Doppler effect frequency shift."""
    return evaluate_primary(meta, test_id, {"doppler_shift": float(frequency_shift)})


def evaluate_scattering_cross_section(meta: Dict, test_id: str, cross_section_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate EM scattering cross-section accuracy."""
    return evaluate_primary(meta, test_id, {"scattering_error": float(cross_section_error)})


def evaluate_synchrotron_spectrum(meta: Dict, test_id: str, spectrum_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate synchrotron radiation spectrum shape."""
    return evaluate_primary(meta, test_id, {"spectrum_error": float(spectrum_error)})


def evaluate_gauge_invariance(meta: Dict, test_id: str, gauge_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate gauge invariance (observable quantities should be gauge-independent)."""
    return evaluate_primary(meta, test_id, {"gauge_error": float(gauge_error)})


# ===== Tier 6 Evaluators (Coupling) =====

def evaluate_coupling_strength(meta: Dict, test_id: str, coupling_parameter: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate E-χ coupling strength."""
    return evaluate_primary(meta, test_id, {"coupling_strength": float(coupling_parameter)})


def evaluate_energy_transfer_efficiency(meta: Dict, test_id: str, transfer_rate: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate energy transfer efficiency between E and χ fields."""
    return evaluate_primary(meta, test_id, {"transfer_efficiency": float(transfer_rate)})


def evaluate_transmission_coefficient(meta: Dict, test_id: str, barrier_transparency: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate barrier transmission coefficient."""
    return evaluate_primary(meta, test_id, {"transmission_coefficient": float(barrier_transparency)})


def evaluate_resonance_amplification(meta: Dict, test_id: str, amplification_factor: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate resonance amplification factor."""
    return evaluate_primary(meta, test_id, {"amplification": float(amplification_factor)})


# ===== Tier 7 Evaluators (Thermodynamics) =====

def evaluate_entropy_monotonicity(meta: Dict, test_id: str, entropy_violations: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate entropy monotonicity violations (fraction of timesteps with ΔS < 0)."""
    return evaluate_primary(meta, test_id, {"entropy_violations": float(entropy_violations)})


def evaluate_equipartition(meta: Dict, test_id: str, mode_energy_variance: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate equipartition of energy across modes."""
    return evaluate_primary(meta, test_id, {"equipartition_error": float(mode_energy_variance)})


def evaluate_temperature_emergence(meta: Dict, test_id: str, boltzmann_fit_r2: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate temperature emergence via Boltzmann distribution fit quality."""
    return evaluate_primary(meta, test_id, {"temperature_fit_r2": float(boltzmann_fit_r2)})


def evaluate_irreversibility(meta: Dict, test_id: str, time_reversal_error: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate irreversibility via time-reversal asymmetry."""
    return evaluate_primary(meta, test_id, {"irreversibility": float(time_reversal_error)})


def evaluate_thermalization_rate(meta: Dict, test_id: str, relaxation_time: float) -> Tuple[bool, str, float, Optional[float]]:
    """Evaluate thermalization rate (relaxation timescale)."""
    return evaluate_primary(meta, test_id, {"thermalization_time": float(relaxation_time)})


@dataclass
class ValidationResult:
    """
    Structured validation outcome combining energy conservation and primary metric.

    Fields:
        test_id: Test identifier (e.g., 'REL-01')
        energy_ok: Whether energy drift is within threshold
        energy_drift: Relative energy drift value
        energy_threshold: Threshold used for energy drift comparison
        primary_ok: Whether primary metric meets its threshold
        primary_metric: The metric key used for the primary comparison
        primary_value: The measured value for the primary metric
        primary_threshold: The threshold used for primary metric (None if not applicable)
        metrics: Arbitrary additional metrics included in evaluation
        timestamp: ISO-like timestamp for when the evaluation occurred
    """
    test_id: str
    energy_ok: bool
    energy_drift: float
    energy_threshold: float
    primary_ok: bool
    primary_metric: str
    primary_value: float
    primary_threshold: Optional[float]
    metrics: Dict[str, float]
    timestamp: str


def aggregate_validation(
    meta: Dict,
    test_id: str,
    energy_drift: float,
    metrics: Dict[str, float],
    *,
    energy_default: float = 0.01,
) -> ValidationResult:
    """
    Aggregate energy conservation and primary metric evaluation into a single result.

    Args:
        meta: Loaded tier metadata dictionary
        test_id: Test identifier (e.g., 'REL-01')
        energy_drift: Relative energy drift value
        metrics: Dict of candidate primary metrics (e.g., {"rel_err": 0.012})
        energy_default: Default energy threshold if not in metadata

    Returns:
        ValidationResult with combined outcome.
    """
    energy_ok, energy_thr, _ = energy_conservation_check(meta, test_id, float(energy_drift))
    p_ok, p_key, p_val, p_thr = check_primary_metric(meta, test_id, dict(metrics))
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    return ValidationResult(
        test_id=str(test_id),
        energy_ok=bool(energy_ok),
        energy_drift=float(energy_drift),
        energy_threshold=float(energy_thr if energy_thr is not None else energy_default),
        primary_ok=bool(p_ok),
        primary_metric=str(p_key) if p_key else "",
        primary_value=float(p_val) if p_val is not None else 0.0,
        primary_threshold=float(p_thr) if p_thr is not None else None,
        metrics={k: float(v) for k, v in metrics.items()},
        timestamp=ts,
    )


def validation_block(result: ValidationResult) -> Dict:
    """
    Convert ValidationResult to a summary-embeddable validation block.

    Returns a plain dict suitable for inclusion under summary['validation'].
    """
    d = asdict(result)
    # Keep structure stable and concise in summaries
    return {
        "test_id": d["test_id"],
        "energy": {
            "ok": d["energy_ok"],
            "drift": d["energy_drift"],
            "threshold": d["energy_threshold"],
        },
        "primary": {
            "ok": d["primary_ok"],
            "metric": d["primary_metric"],
            "value": d["primary_value"],
            "threshold": d["primary_threshold"],
        },
        "metrics": d["metrics"],
        "timestamp": d["timestamp"],
    }
