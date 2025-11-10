# -*- coding: utf-8 -*-
"""
Synthetic validation for verify_klein_gordon_covariance_spatial.

Generates an analytic Klein-Gordon plane wave:
    E(x,t) = cos(k*x - omega*t), omega = sqrt(c^2 k^2 + chi^2)
Builds a time series on a uniform grid (periodic), then verifies that the
spatial-only covariance checker produces spatial_boost/spatial_lab ≈ 1 across betas.

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

from physics.lorentz_transform import verify_klein_gordon_covariance_spatial  # type: ignore


def generate_plane_wave_series(N: int, steps: int, dx: float, dt: float, k: float, chi: float, c: float = 1.0):
    """Generate E(x,t) = cos(k x - omega t) time series with periodic grid."""
    x = (np.arange(N) - N//2) * dx  # centered grid
    omega = math.sqrt((c*c) * (k*k) + chi*chi)
    series = []
    for n in range(steps):
        t = n * dt
        E = np.cos(k * x - omega * t).astype(np.float64)
        series.append(E)
    return series, x


def main():
    # Parameters
    c = 1.0
    chi = 0.05
    N = 1024
    L = 100.0
    dx = L / N
    k = 2.0 * math.pi * 6.0 / L  # 6 cycles across domain
    omega = math.sqrt((c*c) * (k*k) + chi*chi)

    dt = 0.0008
    steps = 4000

    betas = [0.1, 0.2, 0.3, 0.4]

    E_series, x_coords = generate_plane_wave_series(N, steps, dx, dt, k, chi, c=c)

    results = {
        "params": {"N": N, "dx": dx, "dt": dt, "k": k, "omega": omega, "chi": chi, "c": c, "steps": steps},
        "betas": {},
    }

    for beta in betas:
        r = verify_klein_gordon_covariance_spatial(E_series, x_coords, dt, dx, chi, beta, c=c, order=4)
        results["betas"][str(beta)] = r

    out_path = Path("..") / "results" / "Relativistic" / "REL-03_SYNTH" / "spatial_covariance_synthetic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Spatial-only covariance check:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
