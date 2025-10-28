#!/usr/bin/env python3
"""
numeric_integrity.py — universal field and CFL validation for LFM
v1.0.0
"""

import numpy as np, math
from typing import Dict

from lfm_console import log


class NumericIntegrityMixin:
    """Attach to any test harness or solver for auto-checks.

    This mixin avoids spamming the console with repeated identical
    energy warnings by caching whether a given `label` has already
    produced a warning. When the drift falls back below the tolerance
    the cached state is cleared so future violations will be reported.
    """
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

    def validate_energy(self, drift: float, tol: float = 1e-6, label: str = "energy") -> bool:
        """Warn once per sustained violation for a given `label`.

        Rules:
        - If |drift| > tol and we haven't warned for `label`, emit a single
          warning and mark `label` as warned.
        - If |drift| <= tol and label was previously warned, clear the
          warned state so future violations will warn again.
        """
        # lazy init of the per-instance cache
        if not hasattr(self, "_energy_warn_cache"):
            # mapping: label -> bool (True == currently warned)
            self._energy_warn_cache: Dict[str, bool] = {}

        exceeded = abs(drift) > float(tol)
        was_warned = bool(self._energy_warn_cache.get(label, False))

        if exceeded and not was_warned:
            # log once (WARN level) and remember that we've warned
            log(f"[{label}] warning: drift {drift:+.3e} exceeds tolerance {tol:.1e}", "WARN")
            self._energy_warn_cache[label] = True
        elif not exceeded and was_warned:
            # clear the warned state so a future exceedance will warn again
            self._energy_warn_cache[label] = False

        return True
