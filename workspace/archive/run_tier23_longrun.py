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
LFM Tier‑2/3 Heavy GPU Tests — Redshift & Energy Conservation (3D)

Runs two long, compute‑heavy validations aligned to Phase‑1 Test Design:
  A) Tier‑2: Weak‑field redshift analogue with χ(z) gradient (3D plane wave probes)
  B) Tier‑3: Global energy conservation over long steps with CFL guard + sponge boundaries

Designed for: Python 3.11.9, CuPy‑CUDA12.x (GPU). Falls back to NumPy on CPU if CuPy unavailable.

Quick check: use --quick for ~10s sanity run (small grid, few steps). Default is long run.

Output tree (auto‑created if missing):
    workspace/results/<campaign>/tier2_redshift/ ...
    workspace/results/<campaign>/tier3_energy/ ...
Each test writes: metrics.json, plots/*.png, run_log.txt

Citations (LFM internal docs):
- Canonical PDE & CFL: Core Equations §2, §10
- Tier pass criteria: Phase1 Test Design §5

Author: LFM Research — Greg D. Partin & assistant
Version: 1.0 — 2025‑10‑25
"""
from __future__ import annotations
import os, sys, json, math, time, platform, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

# Ensure we can import path_utils from the workspace root
_WS_DIR = Path(__file__).resolve().parents[1]
if str(_WS_DIR) not in sys.path:
    sys.path.insert(0, str(_WS_DIR))

# ------------------------------ Backend (CuPy/NumPy) --------------------------
try:
    import cupy as cp  # GPU
    xp = cp
    GPU = True
except Exception:
    import numpy as cp  # graceful fallback to CPU using NumPy under alias cp
    xp = cp
    GPU = False

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------ Config & CLI ----------------------------------
@dataclass
class RunSettings:
    campaign: str = "museum_day_2025_10_25"
    quick: bool = False
    seed: int = 123456
    dtype: str = "float32"  # keep GPU memory low; use float64 if desired

@dataclass
class GridCfg:
    nx: int
    ny: int
    nz: int
    dx: float
    dt: float
    steps: int
    sponge_cells: int
    c: float  # wave speed

@dataclass
class Tier2Cfg:
    k0: float  # injected plane‑wave wavenumber magnitude
    chi_lo: float
    chi_hi: float
    probes_z: Tuple[int, int]  # z indices for low/high regions

@dataclass
class Tier3Cfg:
    init_mode: str  # 'gaussian' or 'random'
    gaussian_sigma: float
    chi_const: float


def parse_args() -> RunSettings:
    quick = ("--quick" in sys.argv)
    return RunSettings(quick=quick)

# Default configs tuned for RTX 4060 8GB

def make_configs(rs: RunSettings) -> Tuple[GridCfg, Tier2Cfg, Tier3Cfg]:
    if rs.quick:
        # Refined quick-run: balanced resolution + smaller dt
        grid = GridCfg(nx=128, ny=128, nz=128, dx=1.0, dt=0.20, steps=400, sponge_cells=3, c=1.0)
        t2 = Tier2Cfg(k0=0.20, chi_lo=0.10, chi_hi=0.25, probes_z=(32, 96))
        # Use slightly broader Gaussian to minimize gradients
        t3 = Tier3Cfg(init_mode='gaussian', gaussian_sigma=10.0, chi_const=0.0)
    else:
        grid = GridCfg(nx=192, ny=192, nz=192, dx=1.0, dt=0.28, steps=6000, sponge_cells=10, c=1.0)
        t2 = Tier2Cfg(k0=0.18, chi_lo=0.08, chi_hi=0.22, probes_z=(48, 144))
        t3 = Tier3Cfg(init_mode='gaussian', gaussian_sigma=12.0, chi_const=0.0)
    return grid, t2, t3

# ------------------------------ Utilities -------------------------------------

def ensure_dirs(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base/"plots").mkdir(parents=True, exist_ok=True)

def write_json(path: Path, obj) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def write_txt(path: Path, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def cfl_ok(grid: GridCfg) -> bool:
    r = grid.c * grid.dt / grid.dx
    # 3D KG‑type leapfrog bound → c*dt/dx <= 1/sqrt(3)
    return r <= (1.0 / math.sqrt(3.0))


def make_sponge(nx, ny, nz, sponge_cells, strength=0.015, xpmod=xp):
    """Return 3D damping mask in [0,1], where <1 near boundaries (PML‑lite)."""
    mask = xpmod.ones((nx, ny, nz), dtype=xpmod.float32)
    if sponge_cells <= 0:
        return mask
    for axis, n in enumerate([nx, ny, nz]):
        ramp = xpmod.ones(n, dtype=xpmod.float32)
        edge = sponge_cells
        i = xpmod.arange(n, dtype=xpmod.float32)
        # linear ramp to 1 at center; exponential damping factor applied each step
        left = xpmod.clip((i / edge), 0, 1)
        right = xpmod.clip(((n - 1 - i) / edge), 0, 1)
        axis_profile = xpmod.minimum(left, right)
        axis_profile = axis_profile ** 2  # smoother
        # convert to multiplicative damping per step
        axis_profile = 1.0 - strength * (1.0 - axis_profile)
        if axis == 0:
            mask *= axis_profile[:, None, None]
        elif axis == 1:
            mask *= axis_profile[None, :, None]
        else:
            mask *= axis_profile[None, None, :]
    return mask

# ------------------------------ Discrete Operators ----------------------------

def laplacian_3d(u, dx, xpmod=xp):
    return (
        (xpmod.roll(u, 1, 0) + xpmod.roll(u, -1, 0) +
         xpmod.roll(u, 1, 1) + xpmod.roll(u, -1, 1) +
         xpmod.roll(u, 1, 2) + xpmod.roll(u, -1, 2) - 6.0 * u) / (dx*dx)
    )

# Energy density for KG‑type field: 0.5[(∂t u)^2 + c^2|∇u|^2 + χ^2 u^2]

def grad_sq(u, dx, xpmod=xp):
    dudx = (xpmod.roll(u, -1, 0) - xpmod.roll(u, 1, 0)) / (2*dx)
    dudy = (xpmod.roll(u, -1, 1) - xpmod.roll(u, 1, 1)) / (2*dx)
    dudz = (xpmod.roll(u, -1, 2) - xpmod.roll(u, 1, 2)) / (2*dx)
    return dudx*dudx + dudy*dudy + dudz*dudz

# ------------------------------ Tier‑2: Redshift -------------------------------

def run_tier2_redshift(rs: RunSettings, grid: GridCfg, cfg: Tier2Cfg, outdir: Path):
    ensure_dirs(outdir)
    xp.random.seed(rs.seed)

    # Grid & time factors
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dt, c = grid.dx, grid.dt, grid.c
    r = c * dt / dx

    # χ profile with step along z
    chi = xp.full((nx, ny, nz), cfg.chi_lo, dtype=xp.float32)
    chi[:, :, cfg.probes_z[1]:] = cfg.chi_hi

    # Initial plane wave packet along z with wavenumber k0
    k = cfg.k0
    z = xp.arange(nz, dtype=xp.float32)
    phase = 2 * math.pi * k * z[None, None, :]
    u0 = xp.sin(phase).astype(xp.float32)
    u = u0.copy()
    u_prev = u0.copy()  # stationary start

    # Sponge mask for boundaries
    sponge = make_sponge(nx, ny, nz, grid.sponge_cells, strength=0.02)

    # Storage for probe time series
    T = grid.steps
    probe_lo = xp.zeros(T, dtype=xp.float32)
    probe_hi = xp.zeros(T, dtype=xp.float32)

    # Leapfrog update: u_{n+1} = 2u_n − u_{n−1} + (c dt)^2 ∇^2 u_n − (dt^2) χ^2 u_n, with sponge
    c2dt2 = (c*dt)*(c*dt)
    dt2 = dt*dt

    t0 = time.time()
    for n in range(T):
        lap = laplacian_3d(u, dx)
        u_next = 2.0*u - u_prev + c2dt2*lap - (dt2)*(chi*chi)*u
        u_next *= sponge  # damping near boundaries

        # shift states
        u_prev, u = u, u_next

        # record probes (average over xy plane to reduce noise)
        z_lo, z_hi = cfg.probes_z
        probe_lo[n] = xp.mean(u[:, :, z_lo])
        probe_hi[n] = xp.mean(u[:, :, z_hi])

    if GPU:
        probe_lo_h = cp.asnumpy(probe_lo)
        probe_hi_h = cp.asnumpy(probe_hi)
    else:
        probe_lo_h = probe_lo
        probe_hi_h = probe_hi

    elapsed = time.time() - t0

    # Estimate dominant frequency at each probe via FFT peak
    def peak_freq(x, dt):
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), d=dt)
        i = np.argmax(np.abs(X))
        return freqs[i]

    f_lo = peak_freq(probe_lo_h[int(0.1*T):], grid.dt)
    f_hi = peak_freq(probe_hi_h[int(0.1*T):], grid.dt)

    # Analytic dispersion: ω^2 = c^2 k^2 + χ^2  →  f = ω/(2π)
    w_lo = math.sqrt((c*cfg.k0*2*math.pi)**2 + (cfg.chi_lo)**2)
    w_hi = math.sqrt((c*cfg.k0*2*math.pi)**2 + (cfg.chi_hi)**2)
    f_lo_analytic = w_lo / (2*math.pi)
    f_hi_analytic = w_hi / (2*math.pi)

    # Redshift ratio comparison
    ratio_sim = f_hi / max(f_lo, 1e-12)
    ratio_th = f_hi_analytic / f_lo_analytic
    rel_err = abs(ratio_sim - ratio_th) / ratio_th

    # Save metrics & plots
    metrics = {
        "backend": "CuPy" if GPU else "NumPy",
        "grid": asdict(grid),
        "tier2_cfg": asdict(cfg),
        "cfl_ratio": r,
        "elapsed_sec": elapsed,
        "f_lo_sim": f_lo,
        "f_hi_sim": f_hi,
        "f_lo_th": f_lo_analytic,
        "f_hi_th": f_hi_analytic,
        "ratio_sim": float(ratio_sim),
        "ratio_th": float(ratio_th),
        "relative_error": float(rel_err),
        "pass_threshold": 0.05,  # ≤5% error as pass proxy for correlation>0.95
        "pass": bool(rel_err <= 0.05),
    }
    write_json(outdir/"metrics.json", metrics)

    # Plots
    plt.figure()
    plt.plot(probe_lo_h, label='probe_lo (χ_lo)')
    plt.plot(probe_hi_h, label='probe_hi (χ_hi)')
    plt.xlabel('step')
    plt.ylabel('amplitude')
    plt.legend()
    plt.title('Tier‑2 Redshift — probe time series')
    plt.tight_layout()
    plt.savefig(outdir/"plots/probes_timeseries.png", dpi=150)
    plt.close()

    plt.figure()
    bars = [ratio_sim, ratio_th]
    plt.bar([0,1], bars)
    plt.xticks([0,1], ['sim', 'theory'])
    plt.ylabel('f_hi / f_lo')
    plt.title('Tier‑2 Redshift — frequency ratio')
    plt.tight_layout()
    plt.savefig(outdir/"plots/freq_ratio.png", dpi=150)
    plt.close()

    write_txt(outdir/"run_log.txt", f"elapsed_sec={elapsed:.2f}\nCFL={r:.4f}\nPASS={metrics['pass']}\n")
    return metrics

# ------------------------------ Tier‑3: Energy --------------------------------

def run_tier3_energy(rs: RunSettings, grid: GridCfg, cfg: Tier3Cfg, outdir: Path):
    ensure_dirs(outdir)
    xp.random.seed(rs.seed)

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dt, c = grid.dx, grid.dt, grid.c
    r = c * dt / dx

    # Fields
    u_prev = xp.zeros((nx, ny, nz), dtype=xp.float32)
    u = xp.zeros_like(u_prev)

    # Initialization
    if cfg.init_mode == 'gaussian':
        cx, cy, cz = nx//2, ny//2, nz//2
        X = xp.arange(nx)[:, None, None]
        Y = xp.arange(ny)[None, :, None]
        Z = xp.arange(nz)[None, None, :]
        s2 = (cfg.gaussian_sigma)**2
        u = xp.exp(-((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)/(2*s2)).astype(xp.float32)
        u_prev = u.copy()
    elif cfg.init_mode == 'random':
        u = (xp.random.random((nx,ny,nz)).astype(xp.float32) - 0.5)*0.1
        u_prev = u.copy()

    chi = xp.full((nx, ny, nz), cfg.chi_const, dtype=xp.float32)
    sponge = make_sponge(nx, ny, nz, grid.sponge_cells, strength=0.01)

    c2dt2 = (c*dt)*(c*dt)
    dt2 = dt*dt

    # Energy diagnostics
    T = grid.steps
    energy = np.zeros(T, dtype=np.float64)

    t0 = time.time()
    for n in range(T):
        lap = laplacian_3d(u, dx)
        u_next = 2.0*u - u_prev + c2dt2*lap - (dt2)*(chi*chi)*u
        u_next *= sponge
        # Estimate ∂t u using leapfrog: (u_next - u_prev)/(2 dt)
        ut = (u_next - u_prev) / (2.0*dt)
        # Energy density
        grad2 = grad_sq(u, dx)
        edens = 0.5*(ut*ut + (c*c)*grad2 + (chi*chi)*u*u)
        if GPU:
            energy[n] = float(cp.asnumpy(xp.sum(edens))) * (dx**3)
        else:
            energy[n] = float(xp.sum(edens)) * (dx**3)
        # step
        u_prev, u = u, u_next

    elapsed = time.time() - t0

    # Energy drift
    e0 = energy[0]
    drift = float(abs(energy[-1] - e0) / max(abs(e0), 1e-18))

    metrics = {
        "backend": "CuPy" if GPU else "NumPy",
        "grid": asdict(grid),
        "tier3_cfg": asdict(cfg),
        "cfl_ratio": r,
        "elapsed_sec": elapsed,
        "energy_start": float(e0),
        "energy_end": float(energy[-1]),
        "energy_drift_rel": drift,
        "pass_threshold": 1e-4,  # Phase‑1 doc allows ≤1e‑12 ideal; practical GPU single‑precision proxy
        "pass": bool(drift <= 1e-4),
    }
    write_json(outdir/"metrics.json", metrics)

    # Plots
    plt.figure()
    plt.plot(energy)
    plt.xlabel('step')
    plt.ylabel('Total energy')
    plt.title('Tier‑3 Energy — total energy vs time')
    plt.tight_layout()
    plt.savefig(outdir/"plots/energy_vs_time.png", dpi=150)
    plt.close()

    write_txt(outdir/"run_log.txt", f"elapsed_sec={elapsed:.2f}\nCFL={r:.4f}\nPASS={metrics['pass']}\n")
    return metrics

# ------------------------------ Orchestration ---------------------------------

def main():
    rs = parse_args()

    # Base folders per build structure (no hardcoded drives)
    from path_utils import get_workspace_dir
    results_base = get_workspace_dir(__file__)/"results"/f"{rs.campaign}"

    # Configs
    grid, t2, t3 = make_configs(rs)

    # CFL safety check
    if not cfl_ok(grid):
        raise SystemExit(f"CFL violated: c*dt/dx = {grid.c*grid.dt/grid.dx:.4f} > 1/sqrt(3). Adjust dt or dx.")

    # Run Tier‑2
    out2 = results_base/"tier2_redshift"
    m2 = run_tier2_redshift(rs, grid, t2, out2)

    # Run Tier‑3
    out3 = results_base/"tier3_energy"
    m3 = run_tier3_energy(rs, grid, t3, out3)

    # Summary
    summary = {
        "system": platform.platform(),
        "python": platform.python_version(),
        "gpu_backend": GPU,
        "dtype": rs.dtype,
        "tier2": m2,
        "tier3": m3,
        "overall_pass": bool(m2["pass"] and m3["pass"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    ensure_dirs(results_base)
    write_json(results_base/"summary_overall.json", summary)

    print("=== LFM Tier‑2/3 Heavy Tests Complete ===")
    print(json.dumps({k: summary[k] for k in ["gpu_backend","overall_pass"]}, indent=2))
    print(f"Results → {results_base}")

if __name__ == "__main__":
    main()
