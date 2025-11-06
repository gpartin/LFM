# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
lfm_config.py — Typed configuration for LFM simulations

Provides type-safe, validated configuration objects to replace sprawling params dicts.
Includes computed properties for derived quantities (c, CFL ratio, etc.).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Union, Literal, Dict, Any
import numpy as np
import json


@dataclass
class LFMConfig:
    """
    Type-safe configuration for LFM lattice simulations.
    
    Validates parameters at construction and provides computed properties
    for derived quantities like speed of light and CFL limits.
    
    Example:
        >>> config = LFMConfig(dt=0.1, dx=0.5, alpha=1.0, beta=1.0)
        >>> print(config.c)  # Speed of light
        1.0
        >>> print(config.cfl_ratio(ndim=2))  # CFL ratio for 2D
        0.2
    """
    
    # Core simulation parameters (required)
    dt: float
    dx: float
    
    # Physics parameters
    alpha: float = 1.0          # Wave equation coefficient (c² = α/β)
    beta: float = 1.0           # Normalization coefficient
    chi: Union[float, np.ndarray] = 0.0  # Mass parameter (scalar or field)
    gamma_damp: float = 0.0     # Damping coefficient [0, 1)
    
    # Numerical methods
    boundary: Literal["periodic", "absorbing", "dirichlet"] = "periodic"
    stencil_order: Literal[2, 4] = 2
    absorb_width: int = 1       # Width of absorbing boundary layer
    absorb_factor: float = 1.0  # Strength of absorption
    precision: Literal["float32", "float64"] = "float64"
    
    # Runtime options
    use_gpu: bool = False
    threads: int = 1
    tiles: tuple = field(default_factory=lambda: (1,))
    
    # Diagnostics & monitoring
    enable_diagnostics: bool = False
    energy_lock: bool = False
    monitor_every: int = 0  # 0 = disabled
    
    # Computed properties (private, cached)
    _c: float = field(init=False, repr=False, default=None)
    
    def __post_init__(self):
        """Validate configuration and compute derived quantities."""
        # Validate beta != 0
        if self.beta == 0:
            raise ValueError(
                "beta cannot be zero (would cause division by zero in c = sqrt(alpha/beta))"
            )
            
        # Validate dt, dx > 0
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.dx <= 0:
            raise ValueError(f"dx must be positive, got {self.dx}")
            
        # Validate gamma_damp in [0, 1)
        if not (0 <= self.gamma_damp < 1):
            raise ValueError(f"gamma_damp must be in [0, 1), got {self.gamma_damp}")
            
        # Compute speed of light
        self._c = (self.alpha / self.beta) ** 0.5
        
    @property
    def c(self) -> float:
        """Speed of light: c = sqrt(alpha/beta)"""
        return self._c
        
    def cfl_ratio(self, ndim: int) -> float:
        """
        CFL ratio for given dimensionality: (c * dt) / dx
        
        Args:
            ndim: Number of spatial dimensions (1, 2, or 3)
            
        Returns:
            CFL ratio (should be < cfl_limit for stability)
        """
        return self.c * self.dt / self.dx
        
    def cfl_limit(self, ndim: int) -> float:
        """
        CFL stability limit: 1 / sqrt(ndim)
        
        Args:
            ndim: Number of spatial dimensions
            
        Returns:
            Maximum stable CFL ratio
        """
        return 1.0 / (ndim ** 0.5)
        
    def is_stable(self, ndim: int) -> bool:
        """
        Check if CFL condition is satisfied for stability.
        
        Args:
            ndim: Number of spatial dimensions
            
        Returns:
            True if configuration satisfies CFL condition
        """
        return self.cfl_ratio(ndim) <= self.cfl_limit(ndim)
        
    def validate_cfl(self, ndim: int, warn: bool = True) -> bool:
        """
        Validate CFL condition and optionally issue warning.
        
        Args:
            ndim: Number of spatial dimensions
            warn: If True, print warning if CFL violated
            
        Returns:
            True if CFL satisfied, False otherwise
        """
        ratio = self.cfl_ratio(ndim)
        limit = self.cfl_limit(ndim)
        
        if ratio > limit:
            if warn:
                print(f"[CFL WARNING] Ratio {ratio:.3f} exceeds limit {limit:.3f} - may be unstable")
            return False
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary (for backward compatibility).
        
        Returns:
            Dictionary with all configuration parameters
        """
        d = asdict(self)
        # Remove private fields
        d = {k: v for k, v in d.items() if not k.startswith('_')}
        # Add computed values
        d['c'] = self.c
        return d
        
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LFMConfig:
        """
        Create config from dictionary (for backward compatibility).
        
        Args:
            d: Dictionary with configuration parameters
            
        Returns:
            New LFMConfig instance
            
        Example:
            >>> params = {'dt': 0.1, 'dx': 0.5, 'alpha': 1.0, 'beta': 1.0}
            >>> config = LFMConfig.from_dict(params)
        """
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values() if f.init}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
        
    @classmethod
    def from_json_file(cls, filepath: str) -> LFMConfig:
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to JSON config file
            
        Returns:
            New LFMConfig instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
        
    def to_json_file(self, filepath: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def copy(self, **changes) -> LFMConfig:
        """
        Create a copy of this config with specified changes.
        
        Args:
            **changes: Fields to modify
            
        Returns:
            New LFMConfig instance
            
        Example:
            >>> config2 = config.copy(chi=2.0, dt=0.05)
        """
        d = self.to_dict()
        d.update(changes)
        return self.from_dict(d)
        
    def __repr__(self) -> str:
        """Concise representation showing key parameters."""
        return (
            f"LFMConfig(dt={self.dt}, dx={self.dx}, "
            f"c={self.c:.2f}, chi={self.chi}, "
            f"boundary={self.boundary})"
        )


# ---------------------------------------------------------------------
# Convenience constructors for common configurations
# ---------------------------------------------------------------------

def make_default_config(dt: float = 0.1, dx: float = 0.5, **kwargs) -> LFMConfig:
    """
    Create a default configuration with sensible physics parameters.
    
    Args:
        dt: Time step
        dx: Spatial step
        **kwargs: Additional parameters to override
        
    Returns:
        LFMConfig with defaults
    """
    defaults = {
        'alpha': 1.0,
        'beta': 1.0,
        'chi': 0.0,
        'gamma_damp': 0.0,
        'boundary': 'periodic',
        'stencil_order': 2,
    }
    defaults.update(kwargs)
    return LFMConfig(dt=dt, dx=dx, **defaults)


def make_test_config(grid_size: int = 64, cfl_factor: float = 0.5, **kwargs) -> LFMConfig:
    """
    Create a configuration for testing with automatic CFL-safe timestep.
    
    Args:
        grid_size: Number of grid points per dimension
        cfl_factor: Fraction of CFL limit to use (default 0.5 = conservative)
        **kwargs: Additional parameters
        
    Returns:
        LFMConfig with CFL-safe timestep
        
    Example:
        >>> config = make_test_config(grid_size=128, cfl_factor=0.8, chi=1.0)
    """
    dx = 1.0  # Normalized
    ndim = kwargs.get('ndim', 2)
    alpha = kwargs.get('alpha', 1.0)
    beta = kwargs.get('beta', 1.0)
    c = (alpha / beta) ** 0.5
    
    # CFL-safe timestep: c * dt / dx < 1/sqrt(ndim)
    cfl_limit = 1.0 / (ndim ** 0.5)
    dt = (cfl_factor * cfl_limit * dx) / c
    
    return LFMConfig(dt=dt, dx=dx, **kwargs)


# ---------------------------------------------------------------------
# Migration helper: convert old params dict to config
# ---------------------------------------------------------------------

def params_to_config(params: Dict[str, Any]) -> LFMConfig:
    """
    Convert old-style params dict to new LFMConfig.
    
    This is a migration helper for backward compatibility.
    Issues deprecation warning.
    
    Args:
        params: Old-style parameters dictionary
        
    Returns:
        New LFMConfig instance
    """
    import warnings
    warnings.warn(
        "Using dict-based params is deprecated. "
        "Please migrate to LFMConfig for type safety and validation.",
        DeprecationWarning,
        stacklevel=2
    )
    return LFMConfig.from_dict(params)
