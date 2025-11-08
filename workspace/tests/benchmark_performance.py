#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
benchmark_performance.py — Performance benchmarking suite
v1.0.0

Measures speedups from optimization phases and validates numerical accuracy.

Tests:
- Phase 1: Optimized Laplacian (2-3x expected)
- Phase 1: GPU diagnostics (10-50x expected)
- Phase 2: Fused CUDA kernel (5-20x expected)
- Phase 3: Multi-scale AMR (100-1000x expected)

Usage:
    python workspace/tests/benchmark_performance.py
    python workspace/tests/benchmark_performance.py --grid-size 256 --gpu
"""

import time
import argparse
import sys
from pathlib import Path
import numpy as np

# Add workspace/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.lfm_equation import laplacian as laplacian_classic
from core.lfm_equation import lattice_step as lattice_step_classic
from utils.lfm_diagnostics import energy_total

# Optional optimizations
try:
    from core.lfm_equation_optimized import OptimizedLatticeKernel, laplacian_optimized
    HAS_OPTIMIZED = True
except ImportError:
    HAS_OPTIMIZED = False

try:
    from utils.lfm_diagnostics_gpu import energy_total_gpu
    HAS_GPU_DIAGNOSTICS = True
except ImportError:
    HAS_GPU_DIAGNOSTICS = False

try:
    from core.lfm_cuda_kernels import FusedVerletKernel, is_fused_kernel_available
    HAS_FUSED_CUDA = is_fused_kernel_available()
except ImportError:
    HAS_FUSED_CUDA = False

try:
    from physics.multi_scale_lattice import MultiScaleLattice, create_solar_system_lattice
    HAS_MULTISCALE = True
except ImportError:
    HAS_MULTISCALE = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


# ============================================================================
# Benchmark Utilities
# ============================================================================

def timer_context(label: str, repeats: int = 1):
    """Context manager for timing code blocks."""
    class TimerContext:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
            self.per_call = self.elapsed / repeats
            print(f"{label}: {self.elapsed:.4f}s total, {self.per_call:.6f}s per call")
    
    return TimerContext()


def validate_arrays_close(a, b, rtol=1e-10, atol=1e-12, label="Arrays"):
    """
    Validate two arrays are numerically equivalent.
    
    Args:
        a, b: Arrays to compare
        rtol: Relative tolerance
        atol: Absolute tolerance
        label: Description for error messages
    
    Returns:
        True if close, False otherwise
    """
    # Convert to NumPy if needed
    if hasattr(a, 'get'):
        a = a.get()
    if hasattr(b, 'get'):
        b = b.get()
    
    a = np.asarray(a)
    b = np.asarray(b)
    
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(a - b))
        rel_diff = np.max(np.abs(a - b) / (np.abs(a) + atol))
        print(f"❌ {label} differ: max_diff={max_diff:.3e}, max_rel_diff={rel_diff:.3e}")
        return False
    
    print(f"✅ {label} match (within tol={rtol:.1e})")
    return True


# ============================================================================
# Phase 1.1: Optimized Laplacian Benchmark
# ============================================================================

def benchmark_laplacian(grid_size: int = 128, use_gpu: bool = False):
    """
    Benchmark Phase 1.1: Optimized Laplacian vs. classic.
    
    Expected: 2-3x speedup for large grids.
    """
    print("\n" + "="*70)
    print(f"Phase 1.1: Optimized Laplacian Benchmark (N={grid_size}³)")
    print("="*70)
    
    if not HAS_OPTIMIZED:
        print("❌ Optimized kernel not available (import failed)")
        return
    
    xp = cp if (use_gpu and HAS_CUPY) else np
    dtype = xp.float64
    
    # Create test field
    E = xp.random.randn(grid_size, grid_size, grid_size).astype(dtype)
    dx = 1.0
    repeats = 100
    
    # Warm-up
    _ = laplacian_classic(E, dx, order=2)
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
    
    # Classic implementation
    with timer_context(f"Classic Laplacian ({repeats} calls)", repeats) as t_classic:
        for _ in range(repeats):
            L_classic = laplacian_classic(E, dx, order=2)
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
    
    # Optimized implementation (persistent kernel)
    kernel = OptimizedLatticeKernel(E.shape, dx, dtype, xp, order=2)
    with timer_context(f"Optimized Laplacian ({repeats} calls)", repeats) as t_opt:
        for _ in range(repeats):
            L_opt = kernel.compute_laplacian_inplace(E, boundary='periodic')
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
    
    # Validate numerical equivalence
    L_classic_copy = xp.array(L_classic, copy=True)
    L_opt_copy = xp.array(L_opt, copy=True)
    validate_arrays_close(L_classic_copy, L_opt_copy, label="Laplacian results")
    
    # Report speedup
    speedup = t_classic.per_call / t_opt.per_call
    print(f"\n🚀 Speedup: {speedup:.2f}x")
    
    if speedup < 1.5:
        print(f"⚠️  Warning: Speedup below expected 2-3x (got {speedup:.2f}x)")
    elif speedup >= 2.0:
        print(f"✅ Excellent speedup achieved!")
    
    return speedup


# ============================================================================
# Phase 1.2: GPU Diagnostics Benchmark
# ============================================================================

def benchmark_gpu_diagnostics(grid_size: int = 256):
    """
    Benchmark Phase 1.2: GPU-accelerated diagnostics.
    
    Expected: 10-50x speedup (eliminates CPU↔GPU transfer).
    """
    print("\n" + "="*70)
    print(f"Phase 1.2: GPU Diagnostics Benchmark (N={grid_size}³)")
    print("="*70)
    
    if not (HAS_CUPY and HAS_GPU_DIAGNOSTICS):
        print("❌ GPU diagnostics not available (CuPy or module not found)")
        return
    
    # Create GPU arrays
    E = cp.random.randn(grid_size, grid_size, grid_size)
    E_prev = cp.random.randn(grid_size, grid_size, grid_size)
    dt, dx, c, chi = 0.01, 1.0, 1.0, 0.1
    repeats = 100
    
    # Classic (CPU transfer)
    with timer_context(f"Classic energy_total (CPU, {repeats} calls)", repeats) as t_classic:
        for _ in range(repeats):
            E_cpu = E.get()
            E_prev_cpu = E_prev.get()
            energy_classic = energy_total(E_cpu, E_prev_cpu, dt, dx, c, chi)
    
    # GPU-accelerated
    with timer_context(f"GPU energy_total_gpu ({repeats} calls)", repeats) as t_gpu:
        for _ in range(repeats):
            energy_gpu = energy_total_gpu(E, E_prev, dt, dx, c, chi, xp=cp)
        cp.cuda.Stream.null.synchronize()
    
    # Validate numerical equivalence
    rel_diff = abs(energy_classic - energy_gpu) / (abs(energy_classic) + 1e-30)
    if rel_diff < 1e-6:
        print(f"✅ Energy values match: {energy_classic:.6e} vs {energy_gpu:.6e}")
    else:
        print(f"❌ Energy values differ: {energy_classic:.6e} vs {energy_gpu:.6e} (rel_diff={rel_diff:.3e})")
    
    # Report speedup
    speedup = t_classic.per_call / t_gpu.per_call
    print(f"\n🚀 Speedup: {speedup:.2f}x")
    
    if speedup < 5.0:
        print(f"⚠️  Warning: Speedup below expected 10-50x (got {speedup:.2f}x)")
    elif speedup >= 10.0:
        print(f"✅ Excellent speedup achieved!")
    
    return speedup


# ============================================================================
# Phase 2.1: Fused CUDA Kernel Benchmark
# ============================================================================

def benchmark_fused_cuda(grid_size: int = 128):
    """
    Benchmark Phase 2.1: Fused CUDA kernel vs. multi-kernel.
    
    Expected: 5-20x speedup (kernel launch overhead eliminated).
    """
    print("\n" + "="*70)
    print(f"Phase 2.1: Fused CUDA Kernel Benchmark (N={grid_size}³)")
    print("="*70)
    
    if not HAS_FUSED_CUDA:
        print("❌ Fused CUDA kernel not available")
        return
    
    # Create GPU arrays
    E = cp.random.randn(grid_size, grid_size, grid_size)
    E_prev = cp.random.randn(grid_size, grid_size, grid_size)
    chi = 0.1
    c, dt, dx, gamma = 1.0, 0.01, 1.0, 0.0
    repeats = 100
    
    # Multi-kernel approach (classic)
    params = {'dt': dt, 'dx': dx, 'alpha': c*c, 'beta': 1.0, 'chi': chi, 'gamma_damp': gamma}
    with timer_context(f"Multi-kernel lattice_step ({repeats} calls)", repeats) as t_multi:
        E_test = E.copy()
        E_prev_test = E_prev.copy()
        for _ in range(repeats):
            E_next_multi = lattice_step_classic(E_test, E_prev_test, params)
            E_prev_test, E_test = E_test, E_next_multi
        cp.cuda.Stream.null.synchronize()
    
    # Fused kernel
    kernel = FusedVerletKernel(dtype='float64')
    with timer_context(f"Fused kernel step_3d ({repeats} calls)", repeats) as t_fused:
        E_test = E.copy()
        E_prev_test = E_prev.copy()
        for _ in range(repeats):
            E_next_fused = kernel.step_3d(E_test, E_prev_test, chi, grid_size, grid_size, grid_size, dx, dt, c, gamma)
            E_prev_test, E_test = E_test, E_next_fused
        cp.cuda.Stream.null.synchronize()
    
    # Validate numerical equivalence
    validate_arrays_close(E_next_multi, E_next_fused, rtol=1e-8, label="Fused kernel results")
    
    # Report speedup
    speedup = t_multi.per_call / t_fused.per_call
    print(f"\n🚀 Speedup: {speedup:.2f}x")
    
    if speedup < 2.0:
        print(f"⚠️  Warning: Speedup below expected 5-20x (got {speedup:.2f}x)")
    elif speedup >= 5.0:
        print(f"✅ Excellent speedup achieved!")
    
    return speedup


# ============================================================================
# Phase 3.1: Multi-Scale AMR Benchmark
# ============================================================================

def benchmark_multiscale():
    """
    Benchmark Phase 3.1: Multi-scale AMR vs. uniform grid.
    
    Expected: 100-1000x speedup for cosmology scales.
    """
    print("\n" + "="*70)
    print("Phase 3.1: Multi-Scale AMR Benchmark")
    print("="*70)
    
    if not HAS_MULTISCALE:
        print("❌ Multi-scale AMR not available")
        return
    
    print("Creating 3-level AMR lattice (solar system scale)...")
    lattice = create_solar_system_lattice(use_gpu=False)
    
    params = {'alpha': 1.0, 'beta': 1.0, 'dt': 100.0}
    steps = 10
    
    with timer_context(f"AMR lattice ({steps} steps)", steps) as t_amr:
        for i in range(steps):
            lattice.step(dt=params['dt'], params=params)
            if i % 2 == 0:
                print(f"  Step {i+1}/{steps} complete")
    
    # Compute equivalent uniform grid size
    finest_dx = lattice.levels[-1].dx
    coarsest_extent = lattice.levels[0].physical_extent()[1][0] - lattice.levels[0].physical_extent()[0][0]
    equivalent_N = int(coarsest_extent / finest_dx)
    
    print(f"\n📊 AMR Statistics:")
    print(f"  Levels: {len(lattice.levels)}")
    print(f"  Total cells (all levels): {sum(np.prod(l.shape) for l in lattice.levels):,}")
    print(f"  Equivalent uniform grid: {equivalent_N}³ = {equivalent_N**3:,} cells")
    print(f"  Memory savings: {equivalent_N**3 / sum(np.prod(l.shape) for l in lattice.levels):.1f}x")
    
    print(f"\n✅ AMR framework functional (performance comparison requires full simulation)")
    
    return t_amr.per_call


# ============================================================================
# Main Benchmark Suite
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LFM Performance Benchmark Suite")
    parser.add_argument('--grid-size', type=int, default=128, 
                       help="Grid size for benchmarks (default: 128)")
    parser.add_argument('--gpu', action='store_true',
                       help="Run GPU benchmarks")
    parser.add_argument('--phase', choices=['1', '2', '3', 'all'], default='all',
                       help="Which phase to benchmark (default: all)")
    args = parser.parse_args()
    
    print("="*70)
    print("LFM Performance Benchmark Suite v1.0.0")
    print("="*70)
    print(f"Grid size: {args.grid_size}³")
    print(f"GPU mode: {'Enabled' if args.gpu else 'Disabled'}")
    print(f"NumPy available: ✅")
    print(f"CuPy available: {'✅' if HAS_CUPY else '❌'}")
    print(f"Optimized kernels: {'✅' if HAS_OPTIMIZED else '❌'}")
    print(f"GPU diagnostics: {'✅' if HAS_GPU_DIAGNOSTICS else '❌'}")
    print(f"Fused CUDA: {'✅' if HAS_FUSED_CUDA else '❌'}")
    print(f"Multi-scale AMR: {'✅' if HAS_MULTISCALE else '❌'}")
    
    results = {}
    
    # Phase 1.1: Optimized Laplacian
    if args.phase in ['1', 'all']:
        try:
            speedup = benchmark_laplacian(args.grid_size, use_gpu=args.gpu)
            if speedup:
                results['Phase 1.1'] = speedup
        except Exception as e:
            print(f"❌ Phase 1.1 benchmark failed: {e}")
    
    # Phase 1.2: GPU Diagnostics
    if args.phase in ['1', 'all'] and args.gpu:
        try:
            speedup = benchmark_gpu_diagnostics(args.grid_size)
            if speedup:
                results['Phase 1.2'] = speedup
        except Exception as e:
            print(f"❌ Phase 1.2 benchmark failed: {e}")
    
    # Phase 2.1: Fused CUDA
    if args.phase in ['2', 'all'] and args.gpu:
        try:
            speedup = benchmark_fused_cuda(args.grid_size)
            if speedup:
                results['Phase 2.1'] = speedup
        except Exception as e:
            print(f"❌ Phase 2.1 benchmark failed: {e}")
    
    # Phase 3.1: Multi-scale AMR
    if args.phase in ['3', 'all']:
        try:
            time_per_step = benchmark_multiscale()
            if time_per_step:
                results['Phase 3.1'] = f"{time_per_step:.3f}s/step"
        except Exception as e:
            print(f"❌ Phase 3.1 benchmark failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("Benchmark Summary")
    print("="*70)
    for phase, result in results.items():
        if isinstance(result, float):
            print(f"{phase}: {result:.2f}x speedup")
        else:
            print(f"{phase}: {result}")
    
    print("\n✅ Benchmark suite complete!")


if __name__ == '__main__':
    main()
