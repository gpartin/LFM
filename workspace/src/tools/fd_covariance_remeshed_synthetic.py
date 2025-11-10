# -*- coding: utf-8 -*-
"""Synthetic test for remeshed FD covariance method.
Run from workspace/src.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import json, numpy as np

if Path.cwd().name != 'src':
    print('ERROR: Must run from workspace/src directory')
    raise SystemExit(1)

src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from physics.lorentz_transform import verify_klein_gordon_covariance_fd_remeshed

def generate_wave(N, steps, dx, dt, k, chi, c=1.0):
    x = (np.arange(N) - N//2) * dx
    omega = math.sqrt((c*c)*(k*k) + chi*chi)
    series = []
    for n in range(steps):
        t = n*dt
        series.append(np.cos(k*x - omega*t).astype(np.float64))
    return series, x, omega


def main():
    c = 1.0; chi = 0.05
    N = 4096; L = 80.0; dx = L/N
    k = 2.0*math.pi*10.0/L
    dt = 0.0005; steps = 8000
    betas = [0.2, 0.3, 0.4]
    series, x_lab, omega = generate_wave(N, steps, dx, dt, k, chi, c)
    out = { 'params': { 'N': N, 'dx': dx, 'dt': dt, 'k': k, 'omega': omega, 'chi': chi, 'c': c, 'steps': steps}, 'betas': {} }
    for beta in betas:
        r = verify_klein_gordon_covariance_fd_remeshed(series, x_lab, dt, dx, chi, beta, c, order=4, max_time_slices=96)
        out['betas'][str(beta)] = r
    path = Path('..')/'results'/'Relativistic'/'REL-03_SYNTH'/'fd_covariance_remeshed_synthetic.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
