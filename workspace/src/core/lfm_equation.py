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
# Commercial licensing: https://github.com/gpartin/LFM/blob/main/workspace/COMMERCIAL_LICENSE_REQUEST.md
# Contact: latticefieldmediumresearch@gmail.com | licensing@emergentphysicslab.com

"""
lfm_equation.py — Canonical LFM lattice update (v1.5 — 3D Extended, χ-field safe)

═══════════════════════════════════════════════════════════════════════════════
                    ⚠️  SINGLE SOURCE OF TRUTH FOR LFM PHYSICS  ⚠️
═══════════════════════════════════════════════════════════════════════════════

This file contains the CANONICAL implementation of the LFM equation:

    ∂²E/∂t² = c²∇²E − χ²(x,t)E,   with   c² = α/β

This is the standard Klein-Gordon equation (Klein, 1926; Gordon, 1926) with
spatially-varying mass parameter χ(x,t). The Laplacian (∇²) is the fundamental
spatial operator that makes this a wave equation with local causality.

ARCHITECTURAL PRINCIPLE:
- The physics equation lives in ONE place only: THIS FILE
- The Laplacian implementation is HERE (function: laplacian())
- All other modules must import and delegate to these functions
- DO NOT duplicate the Laplacian or equation logic elsewhere

See CORE_EQUATIONS.md for the mathematical derivation and claim validation.

Changes in v1.5:
  • Fix: energy_total() and advance() now accept full χ(x) fields, not only scalars.
  • Fix: core_metrics() no longer forces χ to float — supports spatial χ arrays.
  • No physics or numerical updates — equation and solver are unchanged.
"""

from __future__ import annotations
from typing import Dict
import math, time, warnings
import numpy as _np
try:
    import cupy as _cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _xp_for(arr):
    if _HAS_CUPY and hasattr(arr, "__cuda_array_interface__"):
        return _cp
    return _np


def _asarray(x, xp, dtype=_np.float64):
    if xp is _cp and _HAS_CUPY:
        return _cp.asarray(x, dtype=dtype)
    return _np.asarray(x, dtype=dtype)


# ═════════════════════════════════════════════════════════════════════
# LAPLACIAN — THE FUNDAMENTAL SPATIAL OPERATOR
# ═════════════════════════════════════════════════════════════════════
# This is the CANONICAL implementation of the discrete Laplacian ∇²E.
# 
# CRITICAL: This is the spatial operator that defines the LFM wave equation.
# Our claim: "Klein-Gordon applied via Laplacian to a lattice is what 
# reality looks like and causes physics to emerge."
#
# DO NOT duplicate this implementation elsewhere. All optimization wrappers
# must delegate to this function to maintain single source of truth.
#
# Stencil formulas match CORE_EQUATIONS.md exactly:
#   1D order-2: (E[i+1] - 2E[i] + E[i-1]) / dx²
#   1D order-4: [-E[i+2] + 16E[i+1] - 30E[i] + 16E[i-1] - E[i-2]] / (12dx²)
#   2D/3D: Multi-axis extensions of order-2/4 stencils
# ═════════════════════════════════════════════════════════════════════
def laplacian(E, dx, order=2):
    """
    Compute discrete Laplacian ∇²E using finite-difference stencils.
    
    This is the CANONICAL spatial operator for the LFM equation. The choice
    of stencil order affects numerical dispersion but not the fundamental
    physics: ∂²E/∂t² = c²∇²E − χ²E
    
    Args:
        E: Field array (1D, 2D, or 3D)
        dx: Grid spacing (uniform in all directions)
        order: Stencil order (2 or 4; order-4 not supported for 3D)
    
    Returns:
        ∇²E with same shape as E
    
    Implementation note: Uses np.roll() for periodic boundaries (canonical).
    For optimization wrappers that eliminate roll() overhead, see
    lfm_equation_optimized.py (which delegates back to this function).
    """
    xp = _xp_for(E)
    if E.ndim == 1:
        if order == 2:
            return (xp.roll(E, -1) - 2 * E + xp.roll(E, 1)) / (dx * dx)
        elif order == 4:
            return (
                -1/12 * xp.roll(E, 2)
                + 4/3  * xp.roll(E, 1)
                - 5/2  * E
                + 4/3  * xp.roll(E, -1)
                - 1/12 * xp.roll(E, -2)
            ) / (dx * dx)

    elif E.ndim == 2:
        if order == 2:
            Exp1 = xp.roll(E, 1, 1); Exn1 = xp.roll(E, -1, 1)
            Eyp1 = xp.roll(E, 1, 0); Eyn1 = xp.roll(E, -1, 0)
            return (Exp1 + Exn1 + Eyp1 + Eyn1 - 4 * E) / (dx * dx)
        elif order == 4:
            c0, c1, c2 = -10/3, 2/3, 1/6
            Ex1p = xp.roll(E, 1, 1); Ex1n = xp.roll(E, -1, 1)
            Ey1p = xp.roll(E, 1, 0); Ey1n = xp.roll(E, -1, 0)
            Ex2p = xp.roll(E, 2, 1); Ex2n = xp.roll(E, -2, 1)
            Ey2p = xp.roll(E, 2, 0); Ey2n = xp.roll(E, -2, 0)
            return (
                c0 * E
                + c1 * (Ex1p + Ex1n + Ey1p + Ey1n)
                + c2 * (Ex2p + Ex2n + Ey2p + Ey2n)
            ) / (dx * dx)

    elif E.ndim == 3:
        if order == 2:
            Exn1 = xp.roll(E, -1, 2); Exp1 = xp.roll(E, 1, 2)
            Eyn1 = xp.roll(E, -1, 1); Eyp1 = xp.roll(E, 1, 1)
            Ezn1 = xp.roll(E, -1, 0); Ezp1 = xp.roll(E, 1, 0)
            return (Exp1 + Exn1 + Eyp1 + Eyn1 + Ezp1 + Ezn1 - 6 * E) / (dx * dx)
        else:
            raise ValueError("4th-order Laplacian not implemented for 3D")
    raise ValueError(f"Unsupported ndim={E.ndim}")


# ---------------------------------------------------------------------
# Boundaries (1D–3D)
# ---------------------------------------------------------------------
def apply_boundary(E, mode="periodic", absorb_width=0, absorb_factor=1.0):
    xp = _xp_for(E)
    if mode == "periodic":
        return
    if E.ndim == 1:
        if mode == "reflective":
            E[0] = E[1]; E[-1] = E[-2]
        elif mode == "absorbing":
            w = max(1, int(absorb_width or 1))
            if absorb_factor >= 1.0:
                E[:w] = 0; E[-w:] = 0
            else:
                E[:w] *= absorb_factor; E[-w:] *= absorb_factor
    elif E.ndim == 2:
        if mode == "reflective":
            E[0, :] = E[1, :]; E[-1, :] = E[-2, :]
            E[:, 0] = E[:, 1]; E[:, -1] = E[:, -2]
        elif mode == "absorbing":
            w = max(1, int(absorb_width or 1))
            if absorb_factor >= 1.0:
                E[:w, :] = 0; E[-w:, :] = 0; E[:, :w] = 0; E[:, -w:] = 0
            else:
                E[:w, :] *= absorb_factor; E[-w:, :] *= absorb_factor
                E[:, :w] *= absorb_factor; E[:, -w:] *= absorb_factor
    elif E.ndim == 3:
        if mode == "reflective":
            E[0, :, :] = E[1, :, :]; E[-1, :, :] = E[-2, :, :]
            E[:, 0, :] = E[:, 1, :]; E[:, -1, :] = E[:, -2, :]
            E[:, :, 0] = E[:, :, 1]; E[:, :, -1] = E[:, :, -2]
        elif mode == "absorbing":
            w = max(1, int(absorb_width or 1))
            slices = [slice(None)] * 3
            for axis in range(3):
                leading = [slice(None)] * 3
                trailing = [slice(None)] * 3
                leading[axis] = slice(0, w)
                trailing[axis] = slice(-w, None)
                if absorb_factor >= 1.0:
                    E[tuple(leading)] = 0; E[tuple(trailing)] = 0
                else:
                    E[tuple(leading)] *= absorb_factor
                    E[tuple(trailing)] *= absorb_factor


# ---------------------------------------------------------------------
# Energy (1D–3D)
# ---------------------------------------------------------------------
def energy_total(E, E_prev, dt, dx, c, chi):
    E_np, E_prev_np = _asarray(E, _np), _asarray(E_prev, _np)
    Et = (E_np - E_prev_np) / dt
    if E_np.ndim == 1:
        gx = (_np.roll(E_np, -1) - _np.roll(E_np, 1)) / (2 * dx)
        dens = 0.5 * (Et**2 + (c**2)*gx**2 + (chi**2)*E_np**2)
        return float(_np.sum(dens) * dx)
    elif E_np.ndim == 2:
        gx = (_np.roll(E_np, -1, 1) - _np.roll(E_np, 1, 1)) / (2 * dx)
        gy = (_np.roll(E_np, -1, 0) - _np.roll(E_np, 1, 0)) / (2 * dx)
        dens = 0.5 * (Et**2 + (c**2)*(gx**2 + gy**2) + (chi**2)*E_np**2)
        return float(_np.sum(dens) * (dx * dx))
    elif E_np.ndim == 3:
        gx = (_np.roll(E_np, -1, 2) - _np.roll(E_np, 1, 2)) / (2 * dx)
        gy = (_np.roll(E_np, -1, 1) - _np.roll(E_np, 1, 1)) / (2 * dx)
        gz = (_np.roll(E_np, -1, 0) - _np.roll(E_np, 1, 0)) / (2 * dx)
        dens = 0.5 * (Et**2 + (c**2)*(gx**2 + gy**2 + gz**2) + (chi**2)*E_np**2)
        return float(_np.sum(dens) * (dx**3))


# ═════════════════════════════════════════════════════════════════════
# LATTICE_STEP — THE FUNDAMENTAL TIME-STEPPING OPERATOR
# ═════════════════════════════════════════════════════════════════════
# This implements the Verlet integration of the LFM equation:
#
#   E^{t+1} = (2−γ)E^t − (1−γ)E^{t−1} + Δt² [c²∇²E^t − χ²(x,t)E^t]
#
# This is a leapfrog scheme for the Klein-Gordon equation with variable χ(x,t).
# The Laplacian (∇²) is computed by calling laplacian() above.
# ═════════════════════════════════════════════════════════════════════
def lattice_step(E, E_prev, params):
    """
    Advance the lattice field by one timestep using the canonical LFM equation.
    
    Implements: ∂²E/∂t² = c²∇²E − χ²(x,t)E
    
    Using Verlet (leapfrog) integration:
        E_next = (2−γ)E − (1−γ)E_prev + dt²(c²∇²E − χ²E)
    
    This is the CANONICAL time-stepping function. All simulation runs
    must use this or delegate to it to ensure consistency with published claims.
    
    Backend Selection:
        params['backend'] = 'baseline' (default) - Canonical implementation
        params['backend'] = 'fused' - GPU-accelerated kernel (~1.7-3.5× speedup)
        
        The fused backend computes identical physics but launches a single
        optimized CUDA kernel. Falls back to baseline if CuPy unavailable.
    
    Args:
        E: Current field E^t
        E_prev: Previous field E^{t-1}
        params: Dict with dt, dx, alpha, beta, chi, gamma_damp, boundary, backend, etc.
    
    Returns:
        E_next: Field at next timestep E^{t+1}
    """
    backend = params.get('backend', 'baseline')
    
    # Fused GPU backend (optional accelerator)
    if backend == 'fused':
        xp = _xp_for(E)
        try:
            import cupy as cp
            if xp is cp and E.ndim == 3:  # Fused requires GPU + 3D
                from core.lfm_equation_fused import fused_verlet_step
                
                # Extract params for fused kernel
                dt = float(params["dt"])
                dx = float(params["dx"])
                alpha = float(params["alpha"])
                beta = float(params["beta"])
                chi = params.get("chi", 0.0)
                gamma = float(params.get("gamma_damp", 0.0))
                c = math.sqrt(alpha / beta)
                
                # Call fused kernel (returns E_next without modifying inputs)
                E_next = fused_verlet_step(E, E_prev, chi, dt, dx, c, gamma)
                return E_next
        except (ImportError, Exception):
            # Fall back to baseline if fused unavailable
            pass
    
    # Baseline implementation (canonical)
    return _baseline_lattice_step(E, E_prev, params)


def _baseline_lattice_step(E, E_prev, params):
    """
    Canonical baseline implementation — single source of truth for physics.
    
    This function contains the reference implementation that all backends
    must match. Used for validation and as fallback when accelerators unavailable.
    """
    xp = _xp_for(E)
    dt, dx = float(params["dt"]), float(params["dx"])
    alpha, beta = float(params["alpha"]), float(params["beta"])
    chi = params.get("chi", 0.0)
    gamma = float(params.get("gamma_damp", 0.0))
    boundary = params.get("boundary", "periodic")
    order = int(params.get("stencil_order", 2))
    absorb_w = int(params.get("absorb_width", 1))
    absorb_f = float(params.get("absorb_factor", 1.0))
    precision = params.get("precision", "float64")
    dtype = _np.float32 if precision == "float32" else _np.float64

    E = xp.asarray(E, dtype=dtype)
    E_prev = xp.asarray(E_prev, dtype=dtype)

    c = math.sqrt(alpha / beta)
    cfl_limit = 1.0 / math.sqrt(E.ndim if E.ndim > 0 else 1)
    cfl_ratio = c * dt / dx
    params["_cfl_ratio"] = cfl_ratio
    if cfl_ratio > cfl_limit:
        warnings.warn(f"[CFL] Stability risk: c·dt/dx = {cfl_ratio:.3f} > {cfl_limit:.3f}")

    chi_field = chi if xp.isscalar(chi) else _asarray(chi, xp, dtype)
    lap = laplacian(E, dx, order)

    term_wave = (c * c) * lap
    term_mass = - (chi_field * chi_field) * E
    E_next = (2 - gamma) * E - (1 - gamma) * E_prev + (dt * dt) * (term_wave + term_mass)

    apply_boundary(E_next, mode=boundary, absorb_width=absorb_w, absorb_factor=absorb_f)
    return E_next


# ---------------------------------------------------------------------
# Diagnostics utilities
# ---------------------------------------------------------------------
def _get_debug(params: Dict):
    dbg = params.get("debug", {}) or {}
    return {
        "enable": bool(dbg.get("enable_diagnostics", False)),
        "energy_tol": float(dbg.get("energy_tol", 1e-4)),
        "check_nan": bool(dbg.get("check_nan", True)),
        "profile_steps": int(dbg.get("profile_steps", 0)),
        "edge_band": int(dbg.get("edge_band", 0)),
        "checksum_stride": int(dbg.get("checksum_stride", 4096)),
        "diagnostics_path": params.get("diagnostics_path", "diagnostics_core.csv"),
        "print_probe_steps": bool(dbg.get("print_probe_steps", False)),  # control probe output
        "quiet_run": bool(dbg.get("quiet_run", True)),  # suppress most runtime prints
    }


def _checksum32_sampled(arr, stride=4096):
    a = _np.asarray(arr, dtype=_np.float64).ravel()
    if a.size == 0:
        return 0
    s = max(1, int(stride))
    view = a[::min(s, max(1, a.size // s))]
    s1 = float(view.sum()); s2 = float((view * view).sum()); s3 = float(_np.abs(view).sum())
    h = _np.uint64(1469598103934665603)
    for v in (s1, s2, s3):
        x = _np.frombuffer(_np.float64(v).tobytes(), dtype=_np.uint64)[0]
        h ^= x; h *= _np.uint64(1099511628211)
    return _np.uint32(h & _np.uint64(0xFFFFFFFF)).item()


def _rms(x):
    x = _np.asarray(x, dtype=_np.float64)
    return float(_np.sqrt(_np.mean(x * x))) if x.size else 0.0


def core_metrics(E, E_prev, params, E0, dbg):
    xp = _xp_for(E)
    E_np = _np.asarray(E.get() if _HAS_CUPY and hasattr(E, "get") else E, dtype=_np.float64)
    E_prev_np = _np.asarray(E_prev.get() if _HAS_CUPY and hasattr(E_prev, "get") else E_prev, dtype=_np.float64)
    dt, dx = float(params["dt"]), float(params["dx"])
    c = math.sqrt(float(params["alpha"]) / float(params["beta"]))

    chi_param = params.get("chi", 0.0)
    try:
        import cupy as _cp
        if hasattr(chi_param, "__cuda_array_interface__"):
            chi_param = chi_param.get()
    except Exception:
        pass
    en = energy_total(E_np, E_prev_np, dt, dx, c, chi_param)

    drift = ((en - E0) / (abs(E0) + 1e-30)) if E0 is not None else 0.0
    cfl_ratio = c * dt / dx
    cfl_limit = 1.0 / math.sqrt(max(1, E_np.ndim))
    max_abs = float(_np.max(_np.abs(E_np))) if E_np.size else 0.0
    has_bad = not _np.all(_np.isfinite(E_np)) if dbg["check_nan"] else False

    band = dbg["edge_band"]
    if band <= 0:
        band = max(2, int(0.02 * min(E_np.shape)))
    edge_mask = _np.zeros_like(E_np, dtype=bool)
    for axis in range(E_np.ndim):
        slices_low = [slice(None)] * E_np.ndim
        slices_high = [slice(None)] * E_np.ndim
        slices_low[axis] = slice(0, band)
        slices_high[axis] = slice(-band, None)
        edge_mask[tuple(slices_low)] = True
        edge_mask[tuple(slices_high)] = True
    edge = E_np[edge_mask]
    center = E_np[~edge_mask] if (~edge_mask).any() else E_np
    rms_edge, rms_center = _rms(edge), _rms(center)
    edge_ratio = (rms_edge / (rms_center + 1e-30)) if rms_center > 0 else 0.0

    grad_ratio = 1.0
    if E_np.ndim >= 2:
        gx = (_np.roll(E_np, -1, -1) - _np.roll(E_np, 1, -1)) / (2 * dx)
        gy = (_np.roll(E_np, -1, -2) - _np.roll(E_np, 1, -2)) / (2 * dx)
        components = [gx, gy]
        if E_np.ndim == 3:
            gz = (_np.roll(E_np, -1, 0) - _np.roll(E_np, 1, 0)) / (2 * dx)
            components.append(gz)
        rms_vals = [_rms(g) for g in components]
        gmin, gmax = (min(rms_vals), max(rms_vals)) if all(r > 0 for r in rms_vals) else (1.0, 1.0)
        grad_ratio = gmin / (gmax + 1e-30)

    checksum = _checksum32_sampled(E_np, dbg["checksum_stride"]) if dbg["enable"] else 0
    return {
        "energy": float(en),
        "drift": float(drift),
        "cfl_ratio": float(cfl_ratio),
        "cfl_limit": float(cfl_limit),
        "max_abs": float(max_abs),
        "edge_ratio": float(edge_ratio),
        "grad_ratio": float(grad_ratio),
        "has_bad": bool(has_bad),
        "checksum": int(checksum),
    }


# ---------------------------------------------------------------------
# Advance
# ---------------------------------------------------------------------
def advance(E0, params, steps, save_every=0):
    xp = _xp_for(E0)
    E_prev = xp.array(E0, copy=True)
    E = xp.array(E0, copy=True)

    dbg = _get_debug(params)
    quiet_run = dbg["quiet_run"]
    diagnostics_enabled = dbg["enable"] or bool(params.get("energy_monitor_every", 0))

    c = math.sqrt(float(params["alpha"]) / float(params["beta"]))

    # --- FIXED χ HANDLING ---
    chi_param = params.get("chi", 0.0)
    try:
        import cupy as _cp
        if hasattr(chi_param, "__cuda_array_interface__"):
            chi_param = chi_param.get()
    except Exception:
        pass
    E0_val = energy_total(
        _np.asarray(E.get() if _HAS_CUPY and hasattr(E, "get") else E),
        _np.asarray(E_prev.get() if _HAS_CUPY and hasattr(E_prev, "get") else E_prev),
        float(params["dt"]), float(params["dx"]), c, chi_param
    )
    # -------------------------

    if diagnostics_enabled:
        with open(dbg["diagnostics_path"], "w", encoding="utf-8") as f:
            f.write("step,energy,drift,cfl_ratio,cfl_limit,max_abs,edge_ratio,grad_ratio,has_bad,checksum\\n")

    series = [xp.array(E0, copy=True)] if save_every > 0 else None
    monitor_every = int(params.get("energy_monitor_every", 0)) or (100 if dbg["enable"] else 0)
    profile_every = dbg.get("profile_steps", 0)
    t0 = time.time()

    for n in range(steps):
        E_next = lattice_step(E, E_prev, params)
        E_prev, E = E, E_next

        do_save = save_every > 0 and ((n + 1) % save_every == 0)
        do_diag = diagnostics_enabled and (monitor_every > 0) and ((n + 1) % monitor_every == 0)

        if do_diag:
            met = core_metrics(E, E_prev, params, E0_val, dbg)
            with open(dbg["diagnostics_path"], "a", encoding="utf-8") as f:
                f.write(f"{n+1},{met['energy']:.10e},{met['drift']:.6e},{met['cfl_ratio']:.6f},{met['cfl_limit']:.6f},"
                        f"{met['max_abs']:.6e},{met['edge_ratio']:.6f},{met['grad_ratio']:.6f},{int(met['has_bad'])},{met['checksum']}\\n")
            if not quiet_run:
                if abs(met["drift"]) > dbg["energy_tol"]:
                    warnings.warn(f"[ENERGY] |ΔE/E0|={abs(met['drift']):.3e} > tol={dbg['energy_tol']:.1e}")
                if met["has_bad"]:
                    warnings.warn("[NUMERIC] Non-finite values detected in field")
                if dbg["print_probe_steps"]:
                    print(f"[probe] step {n+1} drift={met['drift']:.6e} E={met['energy']:.6e}")
            params["_last_diagnostics"] = met

        if do_save:
            series.append(xp.array(E, copy=True))

        if not quiet_run and profile_every and (n + 1) % profile_every == 0:
            dt_run = time.time() - t0
            print(f"[PROFILE] step {n+1}/{steps} elapsed={dt_run:.3f}s avg/step={dt_run/(n+1):.6f}s")

    if series is not None:
        return series
    return E
