# -*- coding: utf-8 -*-
"""
Synthetic validation for verify_klein_gordon_covariance_fd_scalar.

Generates an analytic Klein-Gordon plane wave and checks the scalar residual
invariance approach for Lorentz covariance across a sweep of beta values.

Run from: c:\LFM\workspace\src
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import json
import numpy as np

# Enforce correct working directory
if Path.cwd().name != 'src':
    print("ERROR: Must run from workspace/src directory")
    print("Fix: cd c:\\LFM\\workspace\\src")
    raise SystemExit(1)

# Ensure src directory (package root) is on sys.path
src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from physics.lorentz_transform import verify_klein_gordon_covariance_fd_scalar  # type: ignore


def generate_plane_wave_series(N: int, steps: int, dx: float, dt: float, k: float, chi: float, c: float = 1.0):
    x = (np.arange(N) - N//2) * dx
    omega = math.sqrt((c*c) * (k*k) + chi*chi)
    series = []
    for n in range(steps):
        t = n * dt
        E = np.cos(k * x - omega * t).astype(np.float64)
        series.append(E)
    return series, x, omega


def main():
    # Parameters
    c = 1.0
    chi = 0.05
    # Keep domain moderate; scalar method less sensitive to domain/time skew
    N = 2048
    L = 50.0
    dx = L / N
    k = 2.0 * math.pi * 8.0 / L
    dt = 0.0006
    steps = 5000

    betas = [0.1, 0.2, 0.3, 0.4]

    E_series, x_coords, omega = generate_plane_wave_series(N, steps, dx, dt, k, chi, c=c)

    results = {
        "params": {"N": N, "dx": dx, "dt": dt, "k": k, "omega": omega, "chi": chi, "c": c, "steps": steps},
        "betas": {},
    }

    for beta in betas:
        r = verify_klein_gordon_covariance_fd_scalar(E_series, x_coords, dt, dx, chi, beta, c=c, order=4, max_time_slices=128)
        results["betas"][str(beta)] = r

    out_path = Path("..") / "results" / "Relativistic" / "REL-03_SYNTH" / "fd_covariance_scalar_synthetic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Scalar-residual covariance check:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
