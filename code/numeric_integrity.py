#!/usr/bin/env python3
"""
numeric_integrity.py — universal field and CFL validation for LFM
v1.0.0
"""

import numpy as np, math

class NumericIntegrityMixin:
    """Attach to any test harness or solver for auto-checks."""
    def validate_field(self, E, label="field"):
        if not np.all(np.isfinite(E)):
            raise ValueError(f"[{label}] NaN or Inf detected.")
        if np.max(np.abs(E)) > 1e6:
            raise ValueError(f"[{label}] amplitude blow-up > 1e6.")
        return True

    def check_cfl(self, c, dt, dx, ndim):
        limit = 1.0 / math.sqrt(ndim)
        if c * dt / dx > limit:
            raise ValueError(f"CFL violation: c*dt/dx={c*dt/dx:.3f} > {limit:.3f}.")
        return True

    def validate_energy(self, drift, tol=1e-6, label="energy"):
        if abs(drift) > tol:
            print(f"[{label}] warning: drift {drift:+.3e} exceeds tolerance {tol:.1e}")
        return True
