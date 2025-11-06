#!/usr/bin/env python3
"""
Test only EM-01 to debug the Gauss's Law implementation
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from run_tier5_electromagnetic import test_gauss_law_fixed

def main():
    # Load config
    config_path = Path("config/config_tier5_electromagnetic.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Test config for EM-01
    test_config = {
        "charge_density": 0.1,
        "charge_radius": 0.5
    }
    
    # Output directory
    output_dir = Path("results/Electromagnetic/EM-01-debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Testing EM-01 Gauss's Law with fixed analytical implementation...")
    
    # Run the test
    result = test_gauss_law_fixed(config, test_config, output_dir)
    
    print(f"Test ID: {result.test_id}")
    print(f"Description: {result.description}")
    print(f"Passed: {result.passed}")
    print(f"Runtime: {result.runtime_sec:.3f}s")
    print(f"Metrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    
    # Save results
    summary = {
        "test_id": result.test_id,
        "description": result.description,
        "passed": result.passed,
        "metrics": result.metrics,
        "runtime_sec": result.runtime_sec
    }
    
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()