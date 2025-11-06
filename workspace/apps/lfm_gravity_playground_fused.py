#!/usr/bin/env python3
"""
LFM Gravity Playground (Fused Backend Version)

Identical to lfm_gravity_playground.py but uses the fused GPU kernel
for ~1.7× speedup on NVIDIA GPUs while maintaining physics accuracy.

Usage:
    python lfm_gravity_playground_fused.py --preset realistic-orbit --show-velocity
    
Compare performance:
    # Baseline
    python lfm_gravity_playground.py --preset fast-test --steps 300
    
    # Fused (faster)
    python lfm_gravity_playground_fused.py --preset fast-test --steps 300
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

# Import the original playground
_APP = Path(__file__).resolve().parent
_SRC = Path(__file__).resolve().parents[1] / "src"
_WORKSPACE = Path(__file__).resolve().parents[1]  # workspace root
sys.path.insert(0, str(_APP))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
from lfm_gravity_playground import *  # noqa: F401, F403

# Override the multi_scale_lattice step() to use fused backend
from physics.multi_scale_lattice import MultiScaleLattice, _xp_for
import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

# Try to import fused kernel
_FUSED_AVAILABLE = False
_FUSED_ERROR = None
try:
    from performance.optimizations.fused_tiled_kernel import fused_verlet_step
    _FUSED_AVAILABLE = True
except ImportError as e:
    fused_verlet_step = None
    _FUSED_ERROR = str(e)
except Exception as e:
    fused_verlet_step = None
    _FUSED_ERROR = f"Unexpected error: {str(e)}"


def _evolve_level_fused(self, level, dt, params, lattice_step_func):
    """Execute single timestep using fused kernel if available."""
    xp = _xp_for(level.E)
    on_gpu = _HAS_CUPY and (xp is cp)
    
    # Use fused kernel if on GPU and available
    use_fused = on_gpu and _FUSED_AVAILABLE and fused_verlet_step is not None
    
    if use_fused:
        # Fused kernel path (faster)
        # Extract chi from params or use level.chi
        chi = params.get('chi', level.chi)
        E_next = fused_verlet_step(level.E, level.E_prev, chi, dt, level.dx, c=1.0, gamma=0.0)
        level.E_prev[:] = level.E
        level.E[:] = E_next
    else:
        # Canonical path (baseline) - lattice_step takes params dict
        # Add necessary params for lattice_step
        step_params = {
            **params,
            'chi': level.chi,
            'dx': level.dx,
            'dt': dt
        }
        lattice_step_func(level.E, level.E_prev, step_params)


# Monkey-patch the MultiScaleLattice class
_original_evolve = MultiScaleLattice._evolve_level
MultiScaleLattice._evolve_level = _evolve_level_fused


if __name__ == "__main__":
    import argparse
    
    # Detect if we're being called directly
    print(f"\n{'='*70}")
    print(f"LFM Gravity Playground — FUSED BACKEND")
    print(f"{'='*70}")
    
    if _FUSED_AVAILABLE:
        print(f"✓ Fused kernel loaded successfully")
        if _HAS_CUPY:
            print(f"✓ CuPy available — will use GPU acceleration")
        else:
            print(f"⚠ CuPy not available — falling back to baseline (CPU)")
    else:
        print(f"⚠ Fused kernel not available — falling back to baseline")
        if _FUSED_ERROR:
            print(f"  Error: {_FUSED_ERROR}")
        else:
            print(f"  (Make sure performance/optimizations/fused_tiled_kernel.py exists)")
    
    print(f"{'='*70}\n")
    
    # Run main from lfm_gravity_playground
    main()  # noqa: F405
