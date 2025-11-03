#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Hamiltonian Component Visualization Tool
=========================================

Creates publication-quality visualizations of energy conservation tests
showing how energy "sloshes" between kinetic, gradient, and potential
modes while total Hamiltonian H = KE + GE + PE remains constant.

Physics:
- Validates Noether's theorem: time symmetry → energy conservation
- Shows Hamiltonian partitioning: H = ½∫[(∂E/∂t)² + (∇E)² + (χE)²]dV
- Demonstrates energy flow between modes WITHOUT external forcing

Usage:
    python visualize_hamiltonian.py --tests ENER-05 ENER-06 ENER-07
    python visualize_hamiltonian.py --test ENER-05 --output custom.png
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_hamiltonian_data(test_id: str, results_dir: Path = None):
    """Load Hamiltonian component data from test results."""
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "results" / "Energy"
    
    csv_path = results_dir / test_id / "diagnostics" / "hamiltonian_components.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Hamiltonian data not found: {csv_path}")
    
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    return {
        'time': data[:, 0],
        'KE': data[:, 1],
        'GE': data[:, 2],
        'PE': data[:, 3],
        'H_total': data[:, 4]
    }

def get_test_description(test_id: str):
    """Get human-readable description for each test."""
    descriptions = {
        'ENER-05': 'Uniform χ=0 (massless wave)',
        'ENER-06': 'Uniform χ=0.15 (massive wave)',
        'ENER-07': 'χ-gradient (curved spacetime)'
    }
    return descriptions.get(test_id, test_id)

def create_combined_visualization(test_ids: list, output_path: Path = None):
    """Create multi-panel visualization showing Hamiltonian conservation."""
    n_tests = len(test_ids)
    
    # Load all data
    data_dict = {}
    for tid in test_ids:
        try:
            data_dict[tid] = load_hamiltonian_data(tid)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
    
    if not data_dict:
        print("Error: No valid test data found.")
        return
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, len(data_dict), hspace=0.3, wspace=0.3)
    
    colors = {'KE': '#e74c3c', 'GE': '#3498db', 'PE': '#2ecc71'}
    
    # Row 1: Stacked area charts
    for i, (tid, data) in enumerate(data_dict.items()):
        ax = fig.add_subplot(gs[0, i])
        
        KE = data['KE']
        GE = data['GE']
        PE = data['PE']
        H = data['H_total']
        times = data['time']
        
        # Stacked area
        ax.fill_between(times, 0, KE, label='KE', alpha=0.7, color=colors['KE'])
        ax.fill_between(times, KE, KE+GE, label='GE', alpha=0.7, color=colors['GE'])
        ax.fill_between(times, KE+GE, H, label='PE', alpha=0.7, color=colors['PE'])
        ax.plot(times, H, 'k-', linewidth=2, alpha=0.8, label='H total')
        
        drift = abs(H[-1] - H[0]) / max(H[0], 1e-30) * 100
        ax.set_title(f'{tid}: {get_test_description(tid)}\n(drift={drift:.2f}%)', fontsize=10)
        ax.set_ylabel('Energy')
        ax.grid(True, alpha=0.3)
        if i == len(data_dict) - 1:
            ax.legend(loc='upper right', fontsize=8)
    
    # Row 2: Fractional components (energy flow)
    for i, (tid, data) in enumerate(data_dict.items()):
        ax = fig.add_subplot(gs[1, i])
        
        KE = data['KE']
        GE = data['GE']
        PE = data['PE']
        H = data['H_total']
        times = data['time']
        
        # Normalize to show fractional flow
        H0 = H[0]
        ax.plot(times, KE/H0, label='KE', color=colors['KE'], linewidth=2)
        ax.plot(times, GE/H0, label='GE', color=colors['GE'], linewidth=2)
        ax.plot(times, PE/H0, label='PE', color=colors['PE'], linewidth=2)
        
        ax.set_ylabel('Fractional energy')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Row 3: Total Hamiltonian conservation
    for i, (tid, data) in enumerate(data_dict.items()):
        ax = fig.add_subplot(gs[2, i])
        
        H = data['H_total']
        times = data['time']
        
        # Plot relative deviation from initial value
        H0 = H[0]
        rel_dev = (H - H0) / H0 * 100  # Convert to percentage
        
        ax.plot(times, rel_dev, 'k-', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('H deviation (%)')
        ax.grid(True, alpha=0.3)
        
        # Add tolerance band (1%)
        ax.fill_between(times, -1, 1, alpha=0.1, color='green', label='±1% tolerance')
        if i == len(data_dict) - 1:
            ax.legend(loc='upper right', fontsize=8)
    
    # Overall title
    fig.suptitle('Hamiltonian Conservation: H = KE + GE + PE\n' +
                 'Energy flow between modes while H remains constant (validates Noether\'s theorem)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        default_path = Path(__file__).parent.parent.parent / "results" / "Energy" / "hamiltonian_conservation_summary.png"
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {default_path}")
    
    plt.close()

def create_single_test_detail(test_id: str, output_path: Path = None):
    """Create detailed single-test visualization."""
    data = load_hamiltonian_data(test_id)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    KE = data['KE']
    GE = data['GE']
    PE = data['PE']
    H = data['H_total']
    times = data['time']
    
    colors = {'KE': '#e74c3c', 'GE': '#3498db', 'PE': '#2ecc71'}
    
    # Panel 1: Stacked area
    axes[0].fill_between(times, 0, KE, label='Kinetic (∂E/∂t)²', alpha=0.7, color=colors['KE'])
    axes[0].fill_between(times, KE, KE+GE, label='Gradient (∇E)²', alpha=0.7, color=colors['GE'])
    axes[0].fill_between(times, KE+GE, H, label='Potential (χE)²', alpha=0.7, color=colors['PE'])
    axes[0].plot(times, H, 'k-', linewidth=2.5, alpha=0.9, label='Total H')
    axes[0].set_ylabel('Energy', fontsize=12)
    axes[0].set_title(f'{test_id}: {get_test_description(test_id)}\nHamiltonian Partitioning', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Individual components (absolute)
    axes[1].plot(times, KE, label='KE', color=colors['KE'], linewidth=2)
    axes[1].plot(times, GE, label='GE', color=colors['GE'], linewidth=2)
    axes[1].plot(times, PE, label='PE', color=colors['PE'], linewidth=2)
    axes[1].set_ylabel('Component energy', fontsize=12)
    axes[1].set_title('Energy flow between Hamiltonian modes', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Conservation verification
    H0 = H[0]
    rel_dev = (H - H0) / H0 * 100
    drift = abs(H[-1] - H0) / H0 * 100
    
    axes[2].plot(times, rel_dev, 'k-', linewidth=2)
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].fill_between(times, -1, 1, alpha=0.1, color='green', label='±1% tolerance')
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].set_ylabel('H deviation (%)', fontsize=12)
    axes[2].set_title(f'Conservation verification (final drift = {drift:.3f}%)', fontsize=12)
    axes[2].legend(loc='upper right', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        default_path = Path(__file__).parent.parent.parent / "results" / "Energy" / f"{test_id}_hamiltonian_detail.png"
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {default_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Visualize Hamiltonian component conservation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--test', type=str, help='Single test ID for detailed visualization')
    parser.add_argument('--tests', type=str, nargs='+', 
                       default=['ENER-05', 'ENER-06', 'ENER-07'],
                       help='Multiple test IDs for combined visualization')
    parser.add_argument('--output', type=str, help='Output PNG path')
    parser.add_argument('--results-dir', type=str, 
                       help='Custom results directory (default: results/Energy)')
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    
    if args.test:
        # Single test detailed view
        print(f"Creating detailed visualization for {args.test}...")
        create_single_test_detail(args.test, output_path)
    else:
        # Combined multi-test view
        print(f"Creating combined visualization for {len(args.tests)} tests...")
        create_combined_visualization(args.tests, output_path)
    
    print("Done!")

if __name__ == '__main__':
    main()
