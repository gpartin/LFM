#!/usr/bin/env python3
"""
lfm_equation.py — Canonical LFM lattice update (v1.2)

Implements the canonical continuum equation:
    ∂²E/∂t² = c² ∇²E − χ(x,t)² E,   with   c² = α/β

This version adds stability and reproducibility safeguards:
  • CFL stability check (warns if c·dt/dx > 1/√dim)
  • Precision enforcement (float64 default)
  • Optional energy monitor hook (records total energy every N steps)
  • Lightweight lattice-level diagnostics (no per-cell logging)

Diagnostics (toggle via params["debug"]):
  • energy_total + drift rate (ΔE/E0 per tick)
  • CFL ratio persisted (c·dt/dx and limit)
  • edge vs center RMS ratio (boundary reflection sentinel)
  • 2D gradient isotropy (rms(∂x) vs rms(∂y))
  • NaN/Inf detector + max|E|
  • 32-bit checksum (sampled) for determinism checks
Writes CSV to params["diagnostics_path"] or "diagnostics_core.csv" when enabled.
"""

from __future__ import annotations
from typing import Dict, Literal
import math, warnings
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


# ---------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------
def laplacian(E, dx, order=2):
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
    raise ValueError(f"Unsupported ndim={E.ndim}")


# ---------------------------------------------------------------------
# Boundaries
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


# ---------------------------------------------------------------------
# Energy
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


# ---------------------------------------------------------------------
# One-step update
# ---------------------------------------------------------------------
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

    E = xp.asarray(E, dtype=dtype)
    E_prev = xp.asarray(E_prev, dtype=dtype)

    c = math.sqrt(alpha / beta)
    cfl_limit = 1.0 / math.sqrt(E.ndim if E.ndim > 0 else 1)
    if c * dt / dx > cfl_limit:
        warnings.warn(f"[CFL] Stability risk: c·dt/dx = {c*dt/dx:.3f} > {cfl_limit:.3f}")

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
    }


def _checksum32_sampled(arr, stride=4096):
    a = _np.asarray(arr, dtype=_np.float64).ravel()
    if a.size == 0:
        return 0
    s = max(1, int(stride))
    view = a[::min(s, max(1, a.size // s))]
    s1 = float(view.sum())
    s2 = float((view * view).sum())
    s3 = float(_np.abs(view).sum())
    h = _np.uint64(1469598103934665603)
    for v in (s1, s2, s3):
        x = _np.frombuffer(_np.float64(v).tobytes(), dtype=_np.uint64)[0]
        h ^= x
        h *= _np.uint64(1099511628211)
    return _np.uint32(h & _np.uint64(0xFFFFFFFF)).item()


def _rms(x):
    x = _np.asarray(x, dtype=_np.float64)
    return float(_np.sqrt(_np.mean(x * x))) if x.size else 0.0


def _core_metrics(E, E_prev, params, E0, dbg):
    xp = _xp_for(E)
    E_np = _np.asarray(E.get() if _HAS_CUPY and hasattr(E, "get") else E, dtype=_np.float64)
    E_prev_np = _np.asarray(E_prev.get() if _HAS_CUPY and hasattr(E_prev, "get") else E_prev, dtype=_np.float64)
    dt, dx = float(params["dt"]), float(params["dx"])
    c = math.sqrt(float(params["alpha"]) / float(params["beta"]))

    en = energy_total(E_np, E_prev_np, dt, dx, c, float(params.get("chi", 0.0)))
    drift = ((en - E0) / (abs(E0) + 1e-30)) if E0 is not None else 0.0
    cfl_ratio = c * dt / dx
    cfl_limit = 1.0 / math.sqrt(max(1, E_np.ndim))
    max_abs = float(_np.max(_np.abs(E_np))) if E_np.size else 0.0
    has_bad = not _np.all(_np.isfinite(E_np)) if dbg["check_nan"] else False

    band = dbg["edge_band"]
    if band <= 0:
        band = max(2, int(0.02 * (E_np.shape[0] if E_np.ndim == 1 else min(E_np.shape))))
    if E_np.ndim == 1:
        edge = _np.concatenate([E_np[:band], E_np[-band:]]) if E_np.size >= 2 * band else E_np
        center = E_np[band:-band] if E_np.size > 2 * band else E_np
    else:
        edge_mask = _np.zeros_like(E_np, dtype=bool)
        edge_mask[:band, :] = True; edge_mask[-band:, :] = True
        edge_mask[:, :band] = True; edge_mask[:, -band:] = True
        edge = E_np[edge_mask]
        center = E_np[~edge_mask] if (~edge_mask).any() else E_np
    rms_edge = _rms(edge)
    rms_center = _rms(center)
    edge_ratio = (rms_edge / (rms_center + 1e-30)) if rms_center > 0 else 0.0

    grad_ratio = 1.0
    if E_np.ndim == 2:
        gx = (_np.roll(E_np, -1, 1) - _np.roll(E_np, 1, 1)) / (2 * dx)
        gy = (_np.roll(E_np, -1, 0) - _np.roll(E_np, 1, 0)) / (2 * dx)
        rx, ry = _rms(gx), _rms(gy)
        gmin, gmax = (min(rx, ry), max(rx, ry)) if (rx > 0 and ry > 0) else (1.0, 1.0)
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
    diagnostics_enabled = dbg["enable"] or bool(params.get("energy_monitor_every", 0))

    c = math.sqrt(float(params["alpha"]) / float(params["beta"]))
    E0_val = energy_total(
        _np.asarray(E.get() if _HAS_CUPY and hasattr(E, "get") else E),
        _np.asarray(E_prev.get() if _HAS_CUPY and hasattr(E_prev, "get") else E_prev),
        float(params["dt"]), float(params["dx"]), c, float(params.get("chi", 0.0))
    )

    if diagnostics_enabled:
        with open(dbg["diagnostics_path"], "w", encoding="utf-8") as f:
            f.write("step,energy,drift,cfl_ratio,cfl_limit,max_abs,edge_ratio,grad_ratio,has_bad,checksum\\n")

    series = [xp.array(E0, copy=True)] if save_every > 0 else None
    monitor_every = int(params.get("energy_monitor_every", 0)) or (100 if dbg["enable"] else 0)

    for n in range(steps):
        E_next = lattice_step(E, E_prev, params)
        E_prev, E = E, E_next

        do_save = save_every > 0 and ((n + 1) % save_every == 0)
        do_diag = diagnostics_enabled and (monitor_every > 0) and ((n + 1) % monitor_every == 0)

        if do_diag:
            met = _core_metrics(E, E_prev, params, E0_val, dbg)
            with open(dbg["diagnostics_path"], "a", encoding="utf-8") as f:
                f.write(f"{n+1},{met['energy']:.10e},{met['drift']:.6e},{met['cfl_ratio']:.6f},{met['cfl_limit']:.6f},"
                        f"{met['max_abs']:.6e},{met['edge_ratio']:.6f},{met['grad_ratio']:.6f},{int(met['has_bad'])},{met['checksum']}\\n")
            if abs(met["drift"]) > dbg["energy_tol"]:
                warnings.warn(f"[ENERGY] |ΔE/E0|={abs(met['drift']):.3e} > tol={dbg['energy_tol']:.1e}")
            if met["has_bad"]:
                warnings.warn("[NUMERIC] Non-finite values detected in field")
            params["_last_diagnostics"] = met

        if do_save:
            series.append(xp.array(E, copy=True))

    if series is not None:
        return series
    return E


# ---------------------------------------------------------------------
# End of file — canonical LFM equation core (v1.2)
# ---------------------------------------------------------------------
