#!/usr/bin/env python3
"""
Run All Lattice Optimization Experiments
=========================================

Executes baseline + 3 algorithms sequentially, then generates comparison report.

Order:
1. Baseline (full lattice update)
2. Algorithm 1: Active Region Mask
3. Algorithm 2: Gradient-Adaptive Culling
4. Algorithm 3: Wavefront Prediction

Outputs:
- Individual results in results/
- Comparison report with metrics table
- Trajectory plots comparing all methods
"""

import sys
import subprocess
from pathlib import Path
import json
import time

def run_experiment(script_name):
    """Run single experiment script and capture output"""
    script_path = Path(__file__).parent / script_name
    
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print(f"{'='*70}\n")
    
    t_start = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        elapsed = time.time() - t_start
        
        if result.returncode != 0:
            print(f"\n❌ {script_name} FAILED (exit code {result.returncode})")
            return False
        else:
            print(f"\n✓ {script_name} completed in {elapsed:.1f}s")
            return True
    
    except Exception as e:
        print(f"\n❌ {script_name} FAILED with exception: {e}")
        return False


def load_summary(filename):
    """Load summary JSON file"""
    results_dir = Path(__file__).parent / "results"
    filepath = results_dir / filename
    
    if not filepath.exists():
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_comparison_report():
    """Generate comprehensive comparison report"""
    print("\n" + "="*70)
    print("GENERATING COMPARISON REPORT")
    print("="*70 + "\n")
    
    # Load all summaries
    baseline = load_summary("baseline_summary.json")
    alg1 = load_summary("algorithm1_summary.json")
    alg2 = load_summary("algorithm2_summary.json")
    alg3 = load_summary("algorithm3_summary.json")
    
    if not all([baseline, alg1, alg2, alg3]):
        print("❌ Not all experiments completed successfully")
        return
    
    # Extract baseline metrics
    base_time = baseline["baseline_summary"]["mean_time_s"]
    base_drift = baseline["baseline_summary"]["mean_drift"]
    
    # Build comparison table
    report_lines = []
    report_lines.append("="*90)
    report_lines.append("LATTICE OPTIMIZATION EXPERIMENT RESULTS")
    report_lines.append("="*90)
    report_lines.append("")
    report_lines.append("Test Configuration:")
    report_lines.append("  Grid: 128×128×128 = 2,097,152 cells")
    report_lines.append("  Steps: 300 timesteps")
    report_lines.append("  Test case: Earth-Moon circular orbit (kinematic gravity)")
    report_lines.append("  Baseline: Full lattice update every timestep")
    report_lines.append("")
    report_lines.append("="*90)
    report_lines.append(f"{'Algorithm':<35} {'Time (s)':<12} {'Speedup':<10} {'Energy Drift':<15} {'Pass?':<6}")
    report_lines.append("-"*90)
    
    # Baseline row
    report_lines.append(
        f"{'Baseline (Full Update)':<35} "
        f"{base_time:>10.3f}  "
        f"{'1.00x':<10} "
        f"{base_drift:<14.2e} "
        f"{'✓' if base_drift < 1e-4 else '✗':<6}"
    )
    
    # Algorithm rows
    algorithms = [
        ("Algorithm 1: Active Mask", alg1, "algorithm1_summary"),
        ("Algorithm 2: Gradient-Adaptive", alg2, "algorithm2_summary"),
        ("Algorithm 3: Wavefront Predict", alg3, "algorithm3_summary")
    ]
    
    for name, data, key in algorithms:
        summary = data[key]
        alg_time = summary["mean_time_s"]
        alg_drift = summary["mean_drift"]
        speedup = base_time / alg_time
        passed = summary["all_pass"]
        
        report_lines.append(
            f"{name:<35} "
            f"{alg_time:>10.3f}  "
            f"{speedup:<9.2f}x "
            f"{alg_drift:<14.2e} "
            f"{'✓' if passed else '✗':<6}"
        )
    
    report_lines.append("="*90)
    report_lines.append("")
    
    # Detailed breakdown
    report_lines.append("DETAILED ANALYSIS:")
    report_lines.append("")
    
    # Algorithm 1
    alg1_summary = alg1["algorithm1_summary"]
    alg1_trial = alg1["trials"][0]  # First trial has detailed params
    report_lines.append(f"Algorithm 1: Active Region Mask (Conservative)")
    report_lines.append(f"  Approach: Fixed radius around particles + buffer zone")
    report_lines.append(f"  Active radius: {alg1_trial['params']['active_radius']} cells")
    report_lines.append(f"  Buffer: {alg1_trial['params']['buffer_cells']} cells")
    report_lines.append(f"  Mean active cells: {alg1_trial['mean_active_pct']:.1f}%")
    report_lines.append(f"  Speedup: {base_time / alg1_summary['mean_time_s']:.2f}x")
    report_lines.append(f"  Energy error: {alg1_summary['mean_drift']:.2e} (target < 1e-4)")
    report_lines.append(f"  Risk level: LOW - Very conservative, buffer ensures safety")
    report_lines.append("")
    
    # Algorithm 2
    alg2_summary = alg2["algorithm2_summary"]
    alg2_trial = alg2["trials"][0]
    report_lines.append(f"Algorithm 2: Gradient-Adaptive Culling (Moderate)")
    report_lines.append(f"  Approach: Dynamic masking based on |E| and |∇E|")
    report_lines.append(f"  Field threshold: {alg2_trial['params']['field_threshold']:.1e}")
    report_lines.append(f"  Gradient threshold: {alg2_trial['params']['grad_threshold']:.1e}")
    report_lines.append(f"  Halo size: {alg2_trial['params']['halo_size']} cells")
    report_lines.append(f"  Mean active cells: {alg2_trial['mean_active_pct']:.1f}%")
    report_lines.append(f"  Speedup: {base_time / alg2_summary['mean_time_s']:.2f}x")
    report_lines.append(f"  Energy error: {alg2_summary['mean_drift']:.2e} (target < 1e-4)")
    report_lines.append(f"  Risk level: MEDIUM - Threshold-dependent, more adaptive")
    report_lines.append("")
    
    # Algorithm 3
    alg3_summary = alg3["algorithm3_summary"]
    alg3_trial = alg3["trials"][0]
    report_lines.append(f"Algorithm 3: Wavefront Prediction (Physics-Informed)")
    report_lines.append(f"  Approach: Predict wave propagation using c·dt")
    report_lines.append(f"  Field threshold: {alg3_trial['params']['field_threshold']:.1e}")
    report_lines.append(f"  Rate threshold (∂E/∂t): {alg3_trial['params']['rate_threshold']:.1e}")
    report_lines.append(f"  Wave speed: c = {alg3_trial['params']['c']:.2f}")
    report_lines.append(f"  Propagation distance/step: {alg3_trial['params']['c'] * alg3_trial['params']['dt']:.3f} cells")
    report_lines.append(f"  Mean active cells: {alg3_trial['mean_active_pct']:.1f}%")
    report_lines.append(f"  Speedup: {base_time / alg3_summary['mean_time_s']:.2f}x")
    report_lines.append(f"  Energy error: {alg3_summary['mean_drift']:.2e} (target < 1e-4)")
    report_lines.append(f"  Risk level: MEDIUM-HIGH - Most sophisticated, prediction-based")
    report_lines.append("")
    
    report_lines.append("="*90)
    report_lines.append("RECOMMENDATIONS:")
    report_lines.append("")
    
    # Determine best algorithm
    speedups = [
        (base_time / alg1_summary['mean_time_s'], "Algorithm 1", alg1_summary['all_pass']),
        (base_time / alg2_summary['mean_time_s'], "Algorithm 2", alg2_summary['all_pass']),
        (base_time / alg3_summary['mean_time_s'], "Algorithm 3", alg3_summary['all_pass'])
    ]
    
    # Filter to passing algorithms only
    passing = [(s, n) for s, n, p in speedups if p]
    
    if passing:
        best_speedup, best_name = max(passing, key=lambda x: x[0])
        report_lines.append(f"✓ BEST PERFORMER: {best_name} ({best_speedup:.2f}x speedup)")
        report_lines.append(f"  All physics constraints satisfied (energy drift < 1e-4)")
        report_lines.append("")
        
        if best_name == "Algorithm 1":
            report_lines.append("  Algorithm 1 is the most conservative and robust.")
            report_lines.append("  RECOMMENDED for production use in physics simulations.")
        elif best_name == "Algorithm 2":
            report_lines.append("  Algorithm 2 adapts to field dynamics.")
            report_lines.append("  RECOMMENDED if field localization is predictable.")
        else:
            report_lines.append("  Algorithm 3 uses physics-informed prediction.")
            report_lines.append("  RECOMMENDED for highly localized wave phenomena.")
    else:
        report_lines.append("⚠️  WARNING: No algorithms passed energy conservation test!")
        report_lines.append("  Consider:")
        report_lines.append("    - Relaxing thresholds (more conservative)")
        report_lines.append("    - Increasing buffer zones")
        report_lines.append("    - Using baseline for physics validation")
    
    report_lines.append("")
    report_lines.append("NEXT STEPS:")
    report_lines.append("")
    report_lines.append("1. Review trajectory plots in results/ directory")
    report_lines.append("2. Compare trajectories between baseline and optimized versions")
    report_lines.append("3. If passing: Promote winning algorithm to experiments/candidates/")
    report_lines.append("4. Test on additional scenarios (wave packets, multiple particles)")
    report_lines.append("5. Consider for formal validation (Gate 2)")
    report_lines.append("")
    report_lines.append("="*90)
    
    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save report
    results_dir = Path(__file__).parent / "results"
    report_file = results_dir / "comparison_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nReport saved to: {report_file}")
    
    # Also save as JSON for programmatic access
    comparison_data = {
        "baseline": {
            "name": "Baseline (Full Update)",
            "time_s": base_time,
            "speedup": 1.0,
            "energy_drift": base_drift,
            "passed": base_drift < 1e-4
        },
        "algorithm1": {
            "name": "Algorithm 1: Active Mask",
            "time_s": alg1_summary["mean_time_s"],
            "speedup": base_time / alg1_summary["mean_time_s"],
            "energy_drift": alg1_summary["mean_drift"],
            "passed": alg1_summary["all_pass"]
        },
        "algorithm2": {
            "name": "Algorithm 2: Gradient-Adaptive",
            "time_s": alg2_summary["mean_time_s"],
            "speedup": base_time / alg2_summary["mean_time_s"],
            "energy_drift": alg2_summary["mean_drift"],
            "passed": alg2_summary["all_pass"]
        },
        "algorithm3": {
            "name": "Algorithm 3: Wavefront Prediction",
            "time_s": alg3_summary["mean_time_s"],
            "speedup": base_time / alg3_summary["mean_time_s"],
            "energy_drift": alg3_summary["mean_drift"],
            "passed": alg3_summary["all_pass"]
        }
    }
    
    comparison_file = results_dir / "comparison_data.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"Data saved to: {comparison_file}")


def main():
    print("\n" + "#"*70)
    print("# LATTICE OPTIMIZATION SUITE - FULL EXPERIMENT RUN")
    print("#"*70)
    print("\nThis will run 4 experiments × 3 trials each = 12 total runs")
    print("Estimated time: 10-15 minutes on GPU, 30-60 minutes on CPU")
    print("\nExperiments:")
    print("  1. Baseline (full lattice update)")
    print("  2. Algorithm 1: Active Region Mask")
    print("  3. Algorithm 2: Gradient-Adaptive Culling")
    print("  4. Algorithm 3: Wavefront Prediction")
    
    input("\nPress Enter to begin, or Ctrl+C to cancel...")
    
    experiments = [
        "baseline_benchmark.py",
        "algorithm1_active_mask.py",
        "algorithm2_gradient_adaptive.py",
        "algorithm3_wavefront_prediction.py"
    ]
    
    results = []
    for exp in experiments:
        success = run_experiment(exp)
        results.append((exp, success))
        
        if not success:
            print(f"\n⚠️  {exp} failed. Continuing to next experiment...")
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT RUN SUMMARY")
    print("="*70)
    for exp, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}  {exp}")
    
    all_passed = all(s for _, s in results)
    
    if all_passed:
        print("\n✓ All experiments completed successfully!")
        generate_comparison_report()
    else:
        print("\n⚠️  Some experiments failed. Check logs above.")
        print("Generating partial report from completed experiments...")
        generate_comparison_report()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment run cancelled by user.")
        sys.exit(1)
