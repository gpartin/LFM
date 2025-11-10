# -*- coding: utf-8 -*-
"""
Momentum calculation utilities for physics validation tests.

Provides reusable functions for computing and analyzing momentum
in field simulations.
"""
from typing import Union
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def compute_1d_momentum(
    E_t: Union[np.ndarray, 'cp.ndarray'],
    E_x: Union[np.ndarray, 'cp.ndarray']
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Compute 1D momentum density: p = E_t * E_x
    
    For Klein-Gordon field E(x,t), momentum density is the product
    of temporal and spatial derivatives.
    
    Args:
        E_t: Temporal derivative (∂E/∂t)
        E_x: Spatial derivative (∂E/∂x)
    
    Returns:
        Momentum density array (same shape as inputs)
    """
    return E_t * E_x


def total_momentum_1d(
    E_t: Union[np.ndarray, 'cp.ndarray'],
    E_x: Union[np.ndarray, 'cp.ndarray'],
    dx: float
) -> float:
    """
    Compute total momentum by integrating momentum density.
    
    Args:
        E_t: Temporal derivative
        E_x: Spatial derivative
        dx: Grid spacing
    
    Returns:
        Total momentum (scalar)
    """
    p_density = compute_1d_momentum(E_t, E_x)
    
    # Use appropriate backend for sum
    if CUPY_AVAILABLE and isinstance(p_density, cp.ndarray):
        return float(cp.sum(p_density)) * dx
    else:
        return float(np.sum(p_density)) * dx


def relative_momentum_change(
    p_series: Union[np.ndarray, list]
) -> float:
    """
    Compute relative change in momentum over time series.
    
    Args:
        p_series: Array or list of momentum values over time
    
    Returns:
        Relative change: |p_final - p_initial| / max(|p_initial|, 1e-30)
    """
    if len(p_series) < 2:
        return 0.0
    
    p_initial = float(p_series[0])
    p_final = float(p_series[-1])
    
    return abs(p_final - p_initial) / max(abs(p_initial), 1e-30)
