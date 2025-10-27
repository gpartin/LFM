#!/usr/bin/env python3
"""
lfm_equation.py — Canonical LFM lattice update (v1.1)

Implements the canonical continuum equation:
    ∂²E/∂t² = c² ∇²E − χ(x,t)² E,   with   c² = α/β

This version adds stability and reproducibility safeguards:
  • CFL stability check (warns if c·dt/dx > 1/√dim)
  • Precision enforcement (float64 default)
  • Optional energy monitor hook (records total energy every N steps)

Compatible with NumPy and CuPy.
"""
from __future__ import annotations
from typing import Dict, Literal, Optional
import math, warnings
import numpy as _np
try:
    import cupy as _cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False

ArrayLike = _np.ndarray

def _xp_for(arr: ArrayLike):
    if _HAS_CUPY and hasattr(arr, "__cuda_array_interface__"):
        return _cp
    return _np

def _asarray(x, xp, dtype=_np.float64):
    if xp is _cp and _HAS_CUPY:
        return _cp.asarray(x, dtype=dtype)
    return _np.asarray(x, dtype=dtype)

# --------------------- Laplacian ----------------------
def laplacian(E, dx, order=2):
    xp = _xp_for(E)
    if E.ndim == 1:
        if order == 2:
            return (xp.roll(E, -1) - 2 * E + xp.roll(E, 1)) / (dx * dx)
        elif order == 4:
            return (-1/12 * xp.roll(E, 2) + 4/3 * xp.roll(E, 1) - 5/2 * E + 4/3 * xp.roll(E, -1) - 1/12 * xp.roll(E, -2)) / (dx * dx)
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
            return (c0 * E + c1 * (Ex1p + Ex1n + Ey1p + Ey1n) + c2 * (Ex2p + Ex2n + Ey2p + Ey2n)) / (dx * dx)
    raise ValueError(f"Unsupported ndim={E.ndim}")

# -------------------- Boundaries ----------------------
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
            E[0, :] = E[1, :]; E[-1, :] = E[-2, :]; E[:, 0] = E[:, 1]; E[:, -1] = E[:, -2]
        elif mode == "absorbing":
            w = max(1, int(absorb_width or 1))
            if absorb_factor >= 1.0:
                E[:w, :] = 0; E[-w:, :] = 0; E[:, :w] = 0; E[:, -w:] = 0
            else:
                E[:w, :] *= absorb_factor; E[-w:, :] *= absorb_factor; E[:, :w] *= absorb_factor; E[:, -w:] *= absorb_factor

# -------------------- Energy --------------------------
def energy_total(E, E_prev, dt, dx, c, chi):
    xp = _xp_for(E)
    E_np, E_prev_np = _asarray(E, _np), _asarray(E_prev, _np)
    Et = (E_np - E_prev_np) / dt
    if E_np.ndim == 1:
        gx = ( _np.roll(E_np, -1) - _np.roll(E_np, 1) ) / (2 * dx)
        dens = 0.5 * (Et**2 + (c**2)*gx**2 + (chi**2)*E_np**2)
        return float(_np.sum(dens)*dx)
    elif E_np.ndim == 2:
        gx = (_np.roll(E_np, -1, 1) - _np.roll(E_np, 1, 1)) / (2 * dx)
        gy = (_np.roll(E_np, -1, 0) - _np.roll(E_np, 1, 0)) / (2 * dx)
        dens = 0.5 * (Et**2 + (c**2)*(gx**2+gy**2) + (chi**2)*E_np**2)
        return float(_np.sum(dens)*(dx*dx))

# -------------------- Step ----------------------------
def lattice_step(E, E_prev, params):
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

    # Enforce precision
    E = xp.asarray(E, dtype=dtype)
    E_prev = xp.asarray(E_prev, dtype=dtype)

    # Propagation speed and CFL check
    c = math.sqrt(alpha / beta)
    cfl_limit = 1.0 / math.sqrt(E.ndim if E.ndim>0 else 1)
    if c * dt / dx > cfl_limit:
        warnings.warn(f"[CFL] Stability risk: c·dt/dx = {c*dt/dx:.3f} > {cfl_limit:.3f}")

    chi_field = chi if xp.isscalar(chi) else _asarray(chi, xp, dtype)
    lap = laplacian(E, dx, order)

    term_wave = (c*c)*lap
    term_mass = - (chi_field*chi_field)*E
    E_next = (2-gamma)*E - (1-gamma)*E_prev + (dt*dt)*(term_wave + term_mass)

    apply_boundary(E_next, mode=boundary, absorb_width=absorb_w, absorb_factor=absorb_f)
    return E_next

# -------------------- Advance -------------------------
def advance(E0, params, steps, save_every=0):
    xp = _xp_for(E0)
    E_prev = xp.array(E0, copy=True)
    E = xp.array(E0, copy=True)

    if save_every>0:
        series = [xp.array(E0, copy=True)]
        monitor_every = int(params.get("energy_monitor_every", 0))
        energy_log = []
        for n in range(steps):
            E_next = lattice_step(E, E_prev, params)
            E_prev, E = E, E_next
            if monitor_every and (n % monitor_every == 0):
                c = math.sqrt(params["alpha"] / params["beta"])
                en = energy_total(E, E_prev, params["dt"], params["dx"], c, params.get("chi", 0.0))
                energy_log.append((n, en))
            if (n+1)%save_every==0:
                series.append(xp.array(E, copy=True))
        if energy_log:
            _np.savetxt("energy_monitor.csv", _np.array(energy_log), delimiter=",", header="step,energy", comments="")
        return series
    else:
        for _ in range(steps):
            E_next = lattice_step(E, E_prev, params)
            E_prev, E = E, E_next
        return E
