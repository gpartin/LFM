#!/usr/bin/env python3
"""
lfm_parallel.py — Canonical LFM parallel/time-evolution runner
v1.9.3-energyguard-tilefix

Includes:
  • Compensated energy_total (from lfm_diagnostics)
  • Guarded optional energy_lock (diagnostic-only)
  • FIX: 1D tile unpacking bug in _step_threaded() (slices normalized)
  • No change to solver physics.
"""

from __future__ import annotations
from typing import Tuple, Union, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import math, time, os, csv
import numpy as np
from lfm_equation import laplacian, _xp_for, apply_boundary
from lfm_diagnostics import energy_total  # compensated measurement

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
    dt, dx = float(params["dt"]), float(params["dx"])
    alpha, beta = float(params["alpha"]), float(params["beta"])
    gamma = float(params.get("gamma_damp", 0.0))
    order = int(params.get("stencil_order", 2))
    chi = params.get("chi", 0.0)
    xp = _xp_for(E)
    c = math.sqrt(alpha / beta)
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
    dim = int(getattr(E,"ndim",0))
    det = bool(dbg.get("deterministic",False))
    c = math.sqrt(float(params["alpha"])/float(params["beta"]))
    dt, dx = float(params["dt"]), float(params["dx"])
    chi = params.get("chi",0.0)

    # Tile layout
    tile_list = (_tiles_1d(E.shape[0],tiles[0]) if dim==1 else
                 _tiles_2d(E.shape,tiles) if dim==2 else
                 _tiles_3d(E.shape,tiles))

    # Sync E_prev
    if E_prev is None:
        L0 = laplacian(E,dx,order=int(params.get("stencil_order",2)))
        mass_term = (chi**2)*E if xp.isscalar(chi) else (chi*chi)*E
        E_prev = E - 0.5*(dt*dt)*((c*c)*L0 - mass_term)

    E0_energy = energy_total(_as_numpy(E), _as_numpy(E_prev), dt, dx, c, float(chi))
    if "_energy_log" not in params: params["_energy_log"]=[]
    if "_energy_drift_log" not in params: params["_energy_drift_log"]=[]
    params["_energy_log"].clear(); params["_energy_drift_log"].clear()
    params["_energy_log"].append(E0_energy); params["_energy_drift_log"].append(0.0)

    for n in range(steps):
        E_next = _step_threaded(E,E_prev,params,tile_list,deterministic=det)
        E_prev,E = E,E_next

        # Diagnostic energy and drift
        e_now = energy_total(_as_numpy(E),_as_numpy(E_prev),dt,dx,c,float(chi))
        drift = (e_now - E0_energy)/(abs(E0_energy)+1e-30)
        params["_energy_log"].append(e_now); params["_energy_drift_log"].append(drift)

        # --- Optional diagnostic energy lock (guarded)
        conservative = (
            params.get("gamma_damp",0.0)==0.0 and
            params.get("boundary","periodic") in ("periodic","reflective") and
            (not hasattr(params.get("chi",0.0),"shape")) and
            float(params.get("absorb_width",0))==0.0 and
            float(params.get("absorb_factor",1.0))==1.0
        )
        if params.get("energy_lock",False) and conservative:
            if abs(e_now)>0:
                scale = math.sqrt(abs(E0_energy)/(abs(e_now)+1e-30))
                E *= scale

        if n<3:
            print(f"[probe] step {n+1:3d} drift={drift:+.3e} E={e_now:.6e}")

    return xp.array(E, copy=True)
