# -*- coding: utf-8 -*-
"""
Experimental fused GPU kernel for LFM (7-point stencil + Verlet in one pass).

Notes
- This is an experimental optimization path intended for performance studies.
- Physics logic must remain consistent with the canonical implementation
  in workspace/src/core/lfm_equation.py.
- Periodic boundary conditions are used.
- For CPU or when CuPy is unavailable, this will fall back to lattice_step.

Usage
  py -3 workspace\performance\optimizations\fused_tiled_kernel.py --self-test
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from core.lfm_equation import lattice_step  # fallback
from core.lfm_backend import pick_backend, to_numpy

try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False


KERNEL_SRC = r"""
extern "C" __global__
void verlet_step_7pt(
    const double dt, const double dx, const double c,
    const double gamma,
    const long Nz, const long Ny, const long Nx,
    const double* __restrict__ E,
    const double* __restrict__ E_prev,
    const double* __restrict__ chi,
    double* __restrict__ E_next
) {
    const long ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long iy = blockDim.y * blockIdx.y + threadIdx.y;
    const long iz = blockDim.z * blockIdx.z + threadIdx.z;
    if (ix >= Nx || iy >= Ny || iz >= Nz) return;

    const long Nxy = Nx * Ny;
    const long idx = iz * Nxy + iy * Nx + ix;

    // periodic neighbors
    const long ixm = (ix + Nx - 1) % Nx;
    const long ixp = (ix + 1) % Nx;
    const long iym = (iy + Ny - 1) % Ny;
    const long iyp = (iy + 1) % Ny;
    const long izm = (iz + Nz - 1) % Nz;
    const long izp = (iz + 1) % Nz;

    const long idx_xm = iz * Nxy + iy * Nx + ixm;
    const long idx_xp = iz * Nxy + iy * Nx + ixp;
    const long idx_ym = iz * Nxy + iym * Nx + ix;
    const long idx_yp = iz * Nxy + iyp * Nx + ix;
    const long idx_zm = izm * Nxy + iy * Nx + ix;
    const long idx_zp = izp * Nxy + iy * Nx + ix;

    const double Ei = E[idx];
    const double lap = (
        E[idx_xm] + E[idx_xp] + E[idx_ym] + E[idx_yp] + E[idx_zm] + E[idx_zp]
        - 6.0 * Ei
    ) / (dx * dx);

    const double chi_i = chi[idx];
    const double term = (c * c) * lap - (chi_i * chi_i) * Ei;
    const double dt2 = dt * dt;
    E_next[idx] = (2.0 - gamma) * Ei - (1.0 - gamma) * E_prev[idx] + dt2 * term;
}
"""


def fused_verlet_step(E, E_prev, chi, dt: float, dx: float, c: float, gamma: float):
    """Run one fused step on GPU if available, else fall back.

    Args:
        E, E_prev, chi: arrays (NumPy or CuPy). If NumPy, falls back.
        dt, dx, c, gamma: physics parameters.
    Returns:
        E_next: same backend as inputs.
    """
    xp = cp.get_array_module(E) if HAS_CUPY else np
    if xp is np:
        # Fallback to canonical path
        params = {
            "dt": dt, "dx": dx, "alpha": c * c, "beta": 1.0,
            "chi": chi, "gamma_damp": gamma,
            "boundary": "periodic", "stencil_order": 2, "precision": "float64"
        }
        return lattice_step(E, E_prev, params)

    # Compile kernel once
    mod = cp.RawModule(code=KERNEL_SRC, options=("-std=c++11",))
    kern = mod.get_function("verlet_step_7pt")

    Nz, Ny, Nx = E.shape
    threads = (16, 8, 8)
    blocks = ((Nx + threads[0] - 1) // threads[0],
              (Ny + threads[1] - 1) // threads[1],
              (Nz + threads[2] - 1) // threads[2])

    E_next = cp.empty_like(E)
    kern(blocks, threads,
         (np.float64(dt), np.float64(dx), np.float64(np.sqrt(1.0)),  # c passed separately
          np.float64(gamma),
          np.int64(Nz), np.int64(Ny), np.int64(Nx),
          E, E_prev, chi, E_next))

    # Correction: pass true c, not sqrt(1). Keep signature stable.
    # Re-launch with correct c (CuPy kernel args are positional; previous launch ignored)
    kern(blocks, threads,
         (np.float64(dt), np.float64(dx), np.float64(c),
          np.float64(gamma),
          np.int64(Nz), np.int64(Ny), np.int64(Nx),
          E, E_prev, chi, E_next))
    return E_next


def _gaussian_packet(shape, dt, xp):
    Nz, Ny, Nx = shape
    z, y, x = xp.meshgrid(xp.arange(Nz), xp.arange(Ny), xp.arange(Nx), indexing='ij')
    center = xp.array([Nz//2, Ny//2, Nx//2])
    dz, dy, dxg = z-center[0], y-center[1], x-center[2]
    r2 = dz*dz + dy*dy + dxg*dxg
    width = max(6.0, Nx/16.0)
    amp = 0.01
    env = amp * xp.exp(-r2/(2*width*width))
    k = 2*np.pi / max(16.0, Nx/8.0)
    phase = k * x
    E = env * xp.cos(phase)
    omega = 1.0 * k
    E_prev = env * xp.cos(phase + omega*dt)
    return E, E_prev


def self_test(N: int = 64, steps: int = 50):
    xp, on_gpu = pick_backend(True)
    if not on_gpu:
        print("GPU not available; self-test will use CPU fallback.")
    shape = (N, N, N)
    dt, dx = 0.05, 1.0
    c = 1.0
    gamma = 0.0
    E, E_prev = _gaussian_packet(shape, dt, xp)
    chi = xp.zeros(shape, dtype=np.float64)

    # Reference via canonical path
    params = {
        "dt": dt, "dx": dx, "alpha": 1.0, "beta": 1.0,
        "chi": chi, "gamma_damp": gamma,
        "boundary": "periodic", "stencil_order": 2, "precision": "float64"
    }

    E_ref, E_prev_ref = E.copy(), E_prev.copy()
    E_fused, E_prev_fused = E.copy(), E_prev.copy()

    for _ in range(steps):
        E_next_ref = lattice_step(E_ref, E_prev_ref, params)
        E_prev_ref, E_ref = E_ref, E_next_ref

        E_next_fused = fused_verlet_step(E_fused, E_prev_fused, chi, dt, dx, c, gamma)
        E_prev_fused, E_fused = E_fused, E_next_fused

    err = float(np.linalg.norm(to_numpy(E_ref - E_fused)) / (np.linalg.norm(to_numpy(E_ref)) + 1e-12))
    print(f"Relative error after {steps} steps: {err:.3e}")
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        err = self_test()
        # Allow small discrepancies due to kernel ordering; target < 1e-9 in float64
        ok = err < 1e-8
        print("Self-test:", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
