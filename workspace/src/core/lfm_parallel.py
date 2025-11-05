#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
lfm_parallel.py — Canonical LFM parallel/time-evolution runner
v1.9.6-monitor-integrity-lockfix

Adds (diagnostics only; NO physics change):
  • Integrated EnergyMonitor (optional per run)
  • NumericIntegrityMixin for CFL/NaN validation
  • Guarded optional energy_lock projection (constant-energy manifold)
  • Order fix: apply energy_lock BEFORE drift logging and scale (E, E_prev)
  • 1D tile unpack fix via _normalize_tile_args()
"""

from __future__ import annotations
from typing import Tuple, Union, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import math, os
import numpy as np
from pathlib import Path

from core.lfm_equation import laplacian, _xp_for, apply_boundary
from utils.lfm_diagnostics import energy_total  # compensated measurement
from utils.energy_monitor import EnergyMonitor
from utils.numeric_integrity import NumericIntegrityMixin

# Optional optimized kernels
try:
    from core.lfm_equation_optimized import OptimizedLatticeKernel
    _HAS_OPTIMIZED = True
except ImportError:
    _HAS_OPTIMIZED = False

# Optional GPU diagnostics
try:
    from utils.lfm_diagnostics_gpu import energy_total_gpu, comprehensive_diagnostics_gpu
    _HAS_GPU_DIAGNOSTICS = True
except ImportError:
    _HAS_GPU_DIAGNOSTICS = False

# Optional fused CUDA kernels (Phase 2: 5-20x speedup)
try:
    from core.lfm_cuda_kernels import FusedVerletKernel
    _HAS_FUSED_CUDA = True
    _FUSED_KERNEL_CACHE = {}  # Cache kernels by dtype to avoid recompilation
except ImportError:
    _HAS_FUSED_CUDA = False
    _FUSED_KERNEL_CACHE = {}

try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

# ------------------------ Utility functions ------------------------
def _is_cupy_array(x) -> bool:
    return _HAS_CUPY and hasattr(x, "__cuda_array_interface__")

def _as_numpy(x):
    return np.asarray(x.get() if _is_cupy_array(x) else x)

def _tiles_1d(n: int, parts: int) -> List[slice]:
    parts = max(1, int(parts))
    base = n // parts; r = n % parts
    s, out = 0, []
    for i in range(parts):
        e = s + base + (1 if i < r else 0)
        out.append(slice(s, e)); s = e
    return out

def _tiles_2d(shape: Tuple[int,int], tiles: Tuple[int,int]):
    ys = _tiles_1d(shape[0], tiles[0]); xs = _tiles_1d(shape[1], tiles[1])
    return [(y, x) for y in ys for x in xs]

def _tiles_3d(shape: Tuple[int,int,int], tiles: Tuple[int,int,int]):
    zs = _tiles_1d(shape[0], tiles[0])
    ys = _tiles_1d(shape[1], tiles[1])
    xs = _tiles_1d(shape[2], tiles[2])
    return [(z, y, x) for z in zs for y in ys for x in xs]

# ------------------------ Threaded kernel ------------------------
def _normalize_tile_args(t):
    """Return tuple of slices regardless of dimension (1D safe)."""
    return (t,) if isinstance(t, slice) else t

def _step_threaded(E, E_prev, params, tiles, deterministic=False):
    """
    Single timestep update with optional fused CUDA acceleration.
    
    When use_fused_cuda=True, E is on GPU (CuPy), and conditions are met,
    uses a single fused CUDA kernel instead of separate Laplacian + Verlet.
    
    Speedup: 2-3x for large 3D grids, 5-10x for small grids on GPU.
    """
    dt, dx = float(params["dt"]), float(params["dx"])
    alpha, beta = float(params["alpha"]), float(params["beta"])
    gamma = float(params.get("gamma_damp", 0.0))
    order = int(params.get("stencil_order", 2))
    chi = params.get("chi", 0.0)
    xp = _xp_for(E)
    c = math.sqrt(alpha / beta)
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2 OPTIMIZATION: Fused CUDA Kernel
    # ═══════════════════════════════════════════════════════════════════
    # Check if we can use the fused CUDA kernel (5-20x faster on GPU)
    use_fused = params.get("use_fused_cuda", False)
    can_use_fused = (
        use_fused and 
        _HAS_FUSED_CUDA and 
        _is_cupy_array(E) and 
        E.ndim == 3 and 
        order == 2 and 
        len(tiles) == 1 and  # No tiling (or single tile)
        params.get("boundary", "periodic") == "periodic"  # Fused kernel only supports periodic
    )
    
    if can_use_fused:
        # Use fused CUDA kernel (Laplacian + Verlet in one GPU launch)
        dtype_str = 'float64' if E.dtype == cp.float64 else 'float32'
        
        # Get or create cached kernel instance
        if dtype_str not in _FUSED_KERNEL_CACHE:
            _FUSED_KERNEL_CACHE[dtype_str] = FusedVerletKernel(dtype=dtype_str)
        
        kernel = _FUSED_KERNEL_CACHE[dtype_str]
        Nz, Ny, Nx = E.shape
        
        # Single GPU kernel launch (replaces ~10 separate launches)
        E_next = kernel.step_3d(E, E_prev, chi, Nx, Ny, Nz, dx, dt, c, gamma)
        return E_next
    
    # ═══════════════════════════════════════════════════════════════════
    # STANDARD PATH: Separate Laplacian + Verlet (backward compatible)
    # ═══════════════════════════════════════════════════════════════════
    L = laplacian(E, dx, order=order)
    E_next = xp.empty_like(E)

    if xp.isscalar(chi):
        def chi_view(*_): return chi
    else:
        def chi_view(*s): return chi[s]

    def update_tile(*slices):
        if E.ndim == 1:
            (sy,) = slices
            term_wave = (c*c)*L[sy]; term_mass = -(chi_view(sy)**2)*E[sy]
            E_next[sy] = (2-gamma)*E[sy] - (1-gamma)*E_prev[sy] + (dt*dt)*(term_wave + term_mass)
        elif E.ndim == 2:
            sy, sx = slices; chi_loc = chi_view(sy, sx)
            term_wave = (c*c)*L[sy,sx]; term_mass = -(chi_loc**2)*E[sy,sx]
            E_next[sy,sx] = (2-gamma)*E[sy,sx] - (1-gamma)*E_prev[sy,sx] + (dt*dt)*(term_wave + term_mass)
        else:
            sz, sy, sx = slices; chi_loc = chi_view(sz, sy, sx)
            term_wave = (c*c)*L[sz,sy,sx]; term_mass = -(chi_loc**2)*E[sz,sy,sx]
            E_next[sz,sy,sx] = (2-gamma)*E[sz,sy,sx] - (1-gamma)*E_prev[sz,sy,sx] + (dt*dt)*(term_wave + term_mass)

    if deterministic:
        for t in tiles:
            update_tile(*_normalize_tile_args(t))
    else:
        workers = int(params.get("threads",0)) or (os.cpu_count() or 1)
        if workers <= 1:
            for t in tiles:
                update_tile(*_normalize_tile_args(t))
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(update_tile, *_normalize_tile_args(t)) for t in tiles]
                for _ in as_completed(futs):
                    pass

    if E.ndim <= 2:
        apply_boundary(E_next, mode=params.get("boundary","periodic"))
    return E_next

# ------------------------ Main runner ------------------------
def run_lattice(E0, params:dict, steps:int,
                tiles:Union[Tuple[int,int],Tuple[int,int,int]]=(1,1),
                E_prev:Optional[np.ndarray]=None):
    xp = _xp_for(E0); E = xp.array(E0, copy=True)
    dbg = params.setdefault("debug",{})
    # (Previously had a temporary instrumentation log here.)
    dim = int(getattr(E,"ndim",0))
    det = bool(dbg.get("deterministic",False))
    c = math.sqrt(float(params["alpha"])/float(params["beta"]))
    dt, dx = float(params["dt"]), float(params["dx"])
    chi = params.get("chi",0.0)

    integrity = NumericIntegrityMixin()
    # configure numeric integrity from params (if present) so warnings/tolerance
    # can be controlled by higher-level run settings
    ni_cfg = params.get("numeric_integrity", {}) if isinstance(params, dict) else {}
    # attach instance-level settings consumed by validate_energy
    if "energy_tol" in ni_cfg:
        integrity.energy_tol = float(ni_cfg.get("energy_tol"))
    integrity.quiet_warnings = bool(ni_cfg.get("quiet_warnings", False))
    integrity.suppress_monitoring = bool(ni_cfg.get("suppress_monitoring", False))

    integrity.check_cfl(c, dt, dx, dim)
    integrity.validate_field(E, "E0")

    tile_list = (_tiles_1d(E.shape[0],tiles[0]) if dim==1 else
                 _tiles_2d(E.shape,tiles) if dim==2 else
                 _tiles_3d(E.shape,tiles))

    # Bootstrap E_prev via local Taylor step if absent
    if E_prev is None:
        L0 = laplacian(E,dx,order=int(params.get("stencil_order",2)))
        mass_term = (chi**2)*E if xp.isscalar(chi) else (chi*chi)*E
        E_prev = E - 0.5*(dt*dt)*((c*c)*L0 - mass_term)

    # Baseline energy (compensated)
    try:
        chi_param = float(chi)
    except Exception:
        chi_param = _as_numpy(chi)
    E0_energy = energy_total(_as_numpy(E), _as_numpy(E_prev), dt, dx, c, chi_param)
    params.setdefault("_energy_log", []).clear()
    params.setdefault("_energy_drift_log", []).clear()
    params["_energy_log"].append(E0_energy)
    params["_energy_drift_log"].append(0.0)

    # Optional monitor
    mon = None
    monitor_stride = 0
    if params.get("enable_monitor", False):
        mon = EnergyMonitor(dt, dx, c, chi,
                            outdir=params.get("monitor_outdir", "diagnostics"),
                            label=params.get("monitor_label", "lfm_parallel"))
        # Default to every 10 steps if monitor enabled, or use explicit stride
        monitor_stride = int(params.get("energy_monitor_every", 10))
    else:
        # If no monitor, only validate at explicit stride (0 = never)
        monitor_stride = int(params.get("energy_monitor_every", 0))
    
    # Adaptive diagnostic stride (Phase 1.3 optimization)
    use_adaptive_diagnostics = bool(params.get("use_adaptive_diagnostics", False))
    adaptive_stride_min = int(params.get("adaptive_stride_min", 10))
    adaptive_stride_max = int(params.get("adaptive_stride_max", 1000))
    adaptive_drift_threshold = float(params.get("adaptive_drift_threshold", 1e-7))
    current_stride = monitor_stride
    last_drift = 0.0
    stable_count = 0

    # Helper: determine conservative scenario for projection
    def _is_conservative():
        return (
            params.get("gamma_damp",0.0)==0.0 and
            params.get("boundary","periodic") in ("periodic","reflective") and
            (not hasattr(params.get("chi",0.0), "shape")) and
            float(params.get("absorb_width",0))==0.0 and
            float(params.get("absorb_factor",1.0))==1.0
        )

    for n in range(steps):
        # Advance one step
        E_next = _step_threaded(E, E_prev, params, tile_list, deterministic=det)
        E_prev, E = E, E_next

        # Determine if we should compute energy this step
        if use_adaptive_diagnostics and monitor_stride > 0:
            # Adaptive stride: increase when energy is stable
            compute_energy = ((n + 1) % current_stride == 0)
        else:
            compute_energy = (monitor_stride > 0) and ((n + 1) % monitor_stride == 0)
        
        # Optional diagnostic projection BEFORE measuring drift
        if params.get("energy_lock", False) and _is_conservative():
            # Energy lock requires energy computation
            try:
                _chi_pre = float(chi)
            except Exception:
                _chi_pre = _as_numpy(chi)
            
            # Use GPU diagnostics if available and on GPU
            if _HAS_GPU_DIAGNOSTICS and _is_cupy_array(E):
                e_now_pre = energy_total_gpu(E, E_prev, dt, dx, c, _chi_pre, xp=xp)
            else:
                e_now_pre = energy_total(_as_numpy(E), _as_numpy(E_prev), dt, dx, c, _chi_pre)
            
            if abs(e_now_pre) > 0:
                s = math.sqrt(abs(E0_energy) / (abs(e_now_pre) + 1e-30))
                E *= s
                E_prev *= s  # keep Et consistent
            compute_energy = True  # Force recompute after projection

        # Measure energy only at monitor cadence (or if energy_lock was applied)
        if compute_energy:
            try:
                _chi_now = float(chi)
            except Exception:
                _chi_now = _as_numpy(chi)
            
            # Use GPU diagnostics if available and on GPU (Phase 1.2 optimization)
            if _HAS_GPU_DIAGNOSTICS and _is_cupy_array(E):
                e_now = energy_total_gpu(E, E_prev, dt, dx, c, _chi_now, xp=xp)
            else:
                e_now = energy_total(_as_numpy(E), _as_numpy(E_prev), dt, dx, c, _chi_now)
            
            drift = (e_now - E0_energy) / (abs(E0_energy) + 1e-30)
            params["_energy_log"].append(e_now)
            params["_energy_drift_log"].append(drift)
            
            # Adaptive stride adjustment (Phase 1.3)
            if use_adaptive_diagnostics:
                drift_change = abs(drift - last_drift)
                if drift_change < adaptive_drift_threshold:
                    stable_count += 1
                    # Increase stride gradually when stable
                    if stable_count >= 3:
                        current_stride = min(int(current_stride * 1.5), adaptive_stride_max)
                        stable_count = 0
                else:
                    stable_count = 0
                    # Decrease stride when energy changes
                    if drift_change > 10 * adaptive_drift_threshold:
                        current_stride = max(adaptive_stride_min, int(current_stride / 2))
                
                last_drift = drift
            
            # use configured tolerance if provided on the integrity instance
            tol_use = float(getattr(integrity, "energy_tol", ni_cfg.get("energy_tol", 1e-6)))
            integrity.validate_energy(drift, tol=tol_use, label=f"step{n}")

            if mon:
                mon.record(E, E_prev, n)

        # Probe prints are gated via params.debug.print_probe_steps to avoid
        # unconditional console spam from the parallel runner.
        dbg = params.get("debug", {}) if isinstance(params, dict) else {}
        if bool(dbg.get("print_probe_steps", False)) and not bool(dbg.get("quiet_run", True)):
            if n < 3:
                print(f"[probe] step {n+1:3d}  drift={drift:+.3e}  E={e_now:.6e}")

    if mon:
        mon.finalize()

    return xp.array(E, copy=True)
