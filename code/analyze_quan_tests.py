#!/usr/bin/env python3
"""
Comprehensive analysis of all 14 QUAN tests.
Validates that each test is testing what it claims with proper thresholds.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

def load_test_summary(test_id: str) -> Dict[str, Any]:
    """Load summary.json for a test."""
    path = Path(f"results/Quantization/{test_id}/summary.json")
    if not path.exists():
        return {"error": "No summary file"}
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_config() -> Dict[str, Any]:
    """Load test configuration."""
    with open("config/config_tier4_quantization.json", encoding='utf-8') as f:
        return json.load(f)

def analyze_test(test_id: str, summary: Dict, config: Dict) -> Dict[str, Any]:
    """Analyze a single test."""
    # Get test config
    test_cfg = next((t for t in config["tests"] if t["test_id"] == test_id), {})
    tolerances = config["tolerances"]
    
    analysis = {
        "test_id": test_id,
        "description": summary.get("description", test_cfg.get("description", "?")),
        "mode": test_cfg.get("mode", "?"),
        "status_in_json": summary.get("status", summary.get("passed", "?")),
        "metrics": summary.get("metrics", {}),
        "issues": [],
        "recommendations": []
    }
    
    # Analyze by mode
    mode = test_cfg.get("mode")
    metrics = summary.get("metrics", {})
    
    if mode == "energy_transfer":
        # Should test energy conservation
        max_drift = metrics.get("max_energy_drift", 999)
        tolerance = tolerances.get("energy_transfer_conservation", 0.01)
        analysis["expected_metric"] = "max_energy_drift"
        analysis["expected_value"] = f"< {tolerance*100}%"
        analysis["actual_value"] = f"{max_drift*100:.3f}%"
        analysis["testing_correctly"] = max_drift < tolerance
        
        if max_drift >= tolerance:
            analysis["issues"].append(f"Energy drift {max_drift*100:.3f}% exceeds {tolerance*100}% tolerance")
    
    elif mode == "spectral_linearity":
        # Should test E ∝ A²
        mean_err = metrics.get("mean_linearity_error", 999)
        tolerance = tolerances.get("spectral_linearity_error", 0.05)
        analysis["expected_metric"] = "mean_linearity_error"
        analysis["expected_value"] = f"< {tolerance*100}%"
        analysis["actual_value"] = f"{mean_err*100:.2f}%"
        analysis["testing_correctly"] = mean_err < tolerance
        
        if mean_err >= tolerance:
            analysis["issues"].append(f"Linearity error {mean_err*100:.2f}% exceeds {tolerance*100}% tolerance")
    
    elif mode == "phase_amplitude_coupling":
        # Should test AM→PM coupling = 0 for linear system
        coupling = metrics.get("coupling_ratio", 999)
        tolerance = tolerances.get("phase_amplitude_coupling", 0.1)
        analysis["expected_metric"] = "coupling_ratio"
        analysis["expected_value"] = f"< {tolerance}"
        analysis["actual_value"] = f"{coupling:.3f}"
        analysis["testing_correctly"] = coupling < tolerance
        
        am_depth = metrics.get("am_depth", 0)
        pm_depth = metrics.get("pm_depth", 0)
        
        if coupling >= tolerance:
            analysis["issues"].append(f"AM→PM coupling {coupling:.3f} exceeds tolerance {tolerance}")
            analysis["issues"].append(f"Detected PM depth {pm_depth:.3f} with AM depth {am_depth:.3f}")
            analysis["recommendations"].append(
                "ISSUE: Test detects dispersion artifacts (different ω travel at different speeds). "
                "Klein-Gordon dispersion ω²=k²+χ² creates group/phase velocity mismatch. "
                "This is NOT nonlinear coupling (system IS linear). Test methodology needs refinement."
            )
    
    elif mode == "wavefront_stability":
        # Should test no nonlinear steepening
        growth = metrics.get("gradient_growth", 999)
        # No explicit tolerance in config, but should be < 10x
        analysis["expected_metric"] = "gradient_growth"
        analysis["expected_value"] = "< 10x (dispersion allowed)"
        analysis["actual_value"] = f"{growth:.2f}x"
        analysis["testing_correctly"] = growth < 10
        
        if growth >= 10:
            analysis["issues"].append(f"Gradient grew {growth:.2f}x - possible instability")
    
    elif mode == "lattice_blowout":
        # Should test numerical stability
        blew_up = metrics.get("blew_up", True)
        max_energy = metrics.get("max_energy", 999)
        tolerance = tolerances.get("blowout_energy_limit", 100.0)
        analysis["expected_metric"] = "no_blowup"
        analysis["expected_value"] = f"max_energy < {tolerance}"
        analysis["actual_value"] = f"blew_up={blew_up}, max_energy={max_energy:.2e}"
        analysis["testing_correctly"] = not blew_up and max_energy < tolerance
        
        if blew_up:
            analysis["issues"].append("Numerical blowup detected")
        elif max_energy >= tolerance:
            analysis["issues"].append(f"Max energy {max_energy:.2e} exceeds limit {tolerance}")
    
    elif mode == "uncertainty":
        # Should test Δx·Δk ≥ 0.5
        mean_prod = metrics.get("mean_product", 0)
        target = 0.5
        rel_err = metrics.get("rel_error", 999)
        tolerance = tolerances.get("uncertainty_tol_frac", 0.05)
        analysis["expected_metric"] = "mean_product"
        analysis["expected_value"] = f"≈ {target} (within {tolerance*100}%)"
        analysis["actual_value"] = f"{mean_prod:.3f} (error {rel_err*100:.1f}%)"
        analysis["testing_correctly"] = rel_err < tolerance
        
        if rel_err >= tolerance:
            analysis["issues"].append(f"Uncertainty product error {rel_err*100:.1f}% exceeds {tolerance*100}%")
    
    elif mode == "bound_state_quantization":
        # Should test discrete energy levels
        mean_err = metrics.get("mean_error", 999)
        tolerance = tolerances.get("bound_state_error", 0.02)
        analysis["expected_metric"] = "mean_error"
        analysis["expected_value"] = f"< {tolerance*100}%"
        analysis["actual_value"] = f"{mean_err*100:.2f}%"
        analysis["testing_correctly"] = mean_err < tolerance
        
        if mean_err >= tolerance:
            analysis["issues"].append(f"Energy level error {mean_err*100:.2f}% exceeds {tolerance*100}%")
    
    elif mode == "tunneling":
        # Should test T > 0 through classically forbidden barrier
        trans_coeff = metrics.get("transmission_coefficient", 0)
        is_forbidden = summary.get("parameters", {}).get("is_classically_forbidden", False)
        analysis["expected_metric"] = "transmission_coefficient"
        analysis["expected_value"] = "> 0 when E < V"
        analysis["actual_value"] = f"{trans_coeff*100:.2f}%"
        analysis["testing_correctly"] = trans_coeff > 0 and is_forbidden
        
        if trans_coeff == 0:
            analysis["issues"].append("No tunneling detected (T=0)")
        if not is_forbidden:
            analysis["issues"].append("Barrier is NOT classically forbidden (ω > χ)")
    
    elif mode == "zero_point_energy":
        # Should test E_0 ∝ ω (not E_0 = 0)
        mean_err = metrics.get("mean_zpe_error", metrics.get("mean_error", 999))
        tolerance = tolerances.get("zero_point_error", 0.15)
        analysis["expected_metric"] = "mean_zpe_error"
        analysis["expected_value"] = f"< {tolerance*100}%"
        analysis["actual_value"] = f"{mean_err*100:.2f}%"
        analysis["testing_correctly"] = mean_err < tolerance
        
        if mean_err >= tolerance:
            analysis["issues"].append(f"Zero-point energy error {mean_err*100:.2f}% exceeds {tolerance*100}%")
    
    elif mode == "wave_particle_duality":
        # Should test visibility drop when which-way info known
        vis_drop = metrics.get("visibility_drop", 0)
        tolerance = tolerances.get("duality_visibility_drop", 0.2)
        analysis["expected_metric"] = "visibility_drop"
        analysis["expected_value"] = f"> {tolerance}"
        analysis["actual_value"] = f"{vis_drop:.3f}"
        analysis["testing_correctly"] = vis_drop > tolerance
        
        if vis_drop <= tolerance:
            analysis["issues"].append(f"Visibility drop {vis_drop:.3f} below threshold {tolerance}")
    
    elif mode == "planck_distribution":
        # Known limitation - cannot thermalize
        analysis["expected_metric"] = "planck_fit_error"
        analysis["expected_value"] = "< 50% (known limitation)"
        analysis["actual_value"] = f"{metrics.get('planck_fit_error', '?')}"
        analysis["testing_correctly"] = False  # Fundamentally broken
        analysis["issues"].append("KNOWN LIMITATION: Conservative Klein-Gordon cannot thermalize")
        analysis["recommendations"].append(
            "This test artificially initializes Planck distribution and measures what was input. "
            "Cannot test thermalization without damping/interaction mechanism."
        )
    
    return analysis

def main():
    """Main analysis."""
    config = load_config()
    
    print("="*80)
    print("TIER 4 QUANTIZATION TESTS - COMPREHENSIVE ANALYSIS")
    print("="*80)
    print()
    
    results = []
    
    for i in range(1, 15):
        test_id = f"QUAN-{i:02d}"
        summary = load_test_summary(test_id)
        
        if "error" in summary:
            print(f"⚠️  {test_id}: {summary['error']}")
            continue
        
        analysis = analyze_test(test_id, summary, config)
        results.append(analysis)
        
        # Print analysis
        status_icon = "✅" if analysis.get("testing_correctly") else "❌"
        json_status = analysis["status_in_json"]
        
        print(f"{status_icon} {test_id}: {analysis['description']}")
        print(f"   Mode: {analysis['mode']}")
        print(f"   JSON Status: {json_status}")
        print(f"   Testing: {analysis.get('expected_metric', '?')} {analysis.get('expected_value', '?')}")
        print(f"   Actual: {analysis.get('actual_value', '?')}")
        
        if analysis["issues"]:
            print(f"   ⚠️  Issues:")
            for issue in analysis["issues"]:
                print(f"      - {issue}")
        
        if analysis["recommendations"]:
            print(f"   💡 Recommendations:")
            for rec in analysis["recommendations"]:
                print(f"      - {rec}")
        
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    total = len(results)
    testing_correctly = sum(1 for r in results if r.get("testing_correctly", False))
    
    print(f"Total tests analyzed: {total}")
    print(f"Testing correctly: {testing_correctly}/{total} ({testing_correctly/total*100:.1f}%)")
    print()
    
    # List problematic tests
    problematic = [r for r in results if not r.get("testing_correctly", False)]
    if problematic:
        print(f"⚠️  Tests with issues ({len(problematic)}):")
        for r in problematic:
            print(f"   - {r['test_id']}: {r['description']}")
            if r['issues']:
                print(f"     Issues: {'; '.join(r['issues'][:2])}")
        print()
    
    # Reporting bug
    print("="*80)
    print("PARALLEL TEST RUNNER BUG IDENTIFIED")
    print("="*80)
    print("The parallel test runner reports '✓ PASS' based on exit_code == 0,")
    print("NOT on actual test pass/fail status in summary.json.")
    print()
    print("Example:")
    print("  - QUAN-05 shows '✓ PASS' in runner output (exit_code=0)")
    print("  - But summary.json has 'status': 'Failed'")
    print()
    print("This is misleading. The runner should check the JSON status field.")
    print()

if __name__ == "__main__":
    main()
