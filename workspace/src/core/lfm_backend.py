#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
LFM Backend Selection and Array Conversion Utilities
====================================================
Centralized backend management for NumPy/CuPy interoperability.

Usage:
    from core.lfm_backend import pick_backend, to_numpy, HAS_CUPY
    
    xp, on_gpu = pick_backend(use_gpu=True)
    result_np = to_numpy(gpu_array)
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def pick_backend(use_gpu: bool):
    """Select NumPy or CuPy backend based on availability and request.
    
    Args:
        use_gpu: Whether to use GPU if available
    
    Returns:
        Tuple of (xp, on_gpu) where:
        - xp: Module reference (np or cp)
        - on_gpu: Boolean indicating if GPU is actually being used
    
    Example:
        >>> xp, on_gpu = pick_backend(use_gpu=True)
        >>> array = xp.zeros((100, 100))  # Creates on appropriate device
    """
    on_gpu = bool(use_gpu and HAS_CUPY)
    
    if use_gpu and not HAS_CUPY:
        # Lazy import to avoid circular dependency
        try:
            from ui.lfm_console import log
            log("GPU requested but CuPy not available; using NumPy", "WARN")
        except ImportError:
            import warnings
            warnings.warn("GPU requested but CuPy not available; using NumPy")
    
    return (cp if on_gpu else np), on_gpu


def to_numpy(x):
    """Convert array to NumPy, handling CuPy arrays transparently.
    
    Args:
        x: Array (NumPy, CuPy, or array-like)
    
    Returns:
        NumPy array
    
    Example:
        >>> gpu_array = cp.array([1, 2, 3])
        >>> cpu_array = to_numpy(gpu_array)
        >>> type(cpu_array)
        <class 'numpy.ndarray'>
    """
    if HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def ensure_device(x, xp):
    """Ensure array is on the correct device (CPU or GPU).
    
    Args:
        x: Input array
        xp: Target backend module (np or cp)
    
    Returns:
        Array on the correct device
    
    Example:
        >>> cpu_array = np.array([1, 2, 3])
        >>> gpu_array = ensure_device(cpu_array, cp)
    """
    if xp is cp and not isinstance(x, cp.ndarray):
        return cp.asarray(x)
    if xp is np and HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x


def get_array_module(x):
    """Get the appropriate array module (np or cp) for an array.
    
    Args:
        x: Array object
    
    Returns:
        Module (np or cp) corresponding to array type
    
    Example:
        >>> x = cp.array([1, 2, 3])
        >>> xp = get_array_module(x)
        >>> xp.zeros_like(x)  # Creates CuPy array
    """
    if HAS_CUPY:
        return cp.get_array_module(x)
    return np
