# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com

"""
lfm_simulator.py — High-level simulation engine with state management

Provides LFMSimulator class that encapsulates lattice state and evolution,
offering a clean API for interactive tools, test harnesses, and analysis.
"""

from __future__ import annotations
from typing import Callable, Optional, Dict, Any, Tuple
import numpy as np
import time

from lfm_config import LFMConfig
from lfm_equation import lattice_step, energy_total


class LFMSimulator:
    """
    High-level lattice field simulation engine.
    
    Manages simulation state (E, E_prev) and provides clean methods for
    time evolution, diagnostics, and interactive manipulation.
    
    Example:
        >>> config = LFMConfig(dt=0.1, dx=0.5, chi=1.0)
        >>> E0 = np.zeros((128, 128))
        >>> sim = LFMSimulator(E0, config)
        >>> sim.add_gaussian_pulse((64, 64), amplitude=1.0)
        >>> sim.run(100)
        >>> print(f"Energy: {sim.energy:.3e}")
    """
    
    def __init__(self, initial_field: np.ndarray, config: LFMConfig):
        """
        Initialize simulator with initial field and configuration.
        
        Args:
            initial_field: Initial field values E(t=0)
            config: Simulation configuration
        """
        self.config = config
        self.ndim = initial_field.ndim
        
        # State variables
        self.E = initial_field.copy()
        self.E_prev = initial_field.copy()
        self.t = 0.0
        self.step_count = 0
        
        # History tracking
        self._energy_history = []
        self._time_history = []
        
        # Performance tracking
        self._total_step_time = 0.0
        self._last_step_time = 0.0
        
        # Validate CFL condition
        if not config.is_stable(self.ndim):
            import warnings
            warnings.warn(
                f"CFL ratio {config.cfl_ratio(self.ndim):.3f} exceeds "
                f"stability limit {config.cfl_limit(self.ndim):.3f}",
                UserWarning
            )
    
    # ----------------------------------------------------------------
    # Time Evolution
    # ----------------------------------------------------------------
    
    def step(self) -> None:
        """
        Advance simulation by one timestep: t → t + dt.
        
        Updates E_prev and E using Verlet integration via lattice_step().
        """
        t0 = time.perf_counter()
        
        # Call core physics kernel
        E_next = lattice_step(self.E, self.E_prev, self.config.to_dict())
        
        # Update state
        self.E_prev = self.E
        self.E = E_next
        self.t += self.config.dt
        self.step_count += 1
        
        # Track performance
        self._last_step_time = time.perf_counter() - t0
        self._total_step_time += self._last_step_time
        
    def run(self, n_steps: int, callback: Optional[Callable[[LFMSimulator], None]] = None) -> None:
        """
        Run simulation for n timesteps with optional per-step callback.
        
        Args:
            n_steps: Number of timesteps to execute
            callback: Optional function called after each step with simulator as argument
            
        Example:
            >>> def monitor(sim):
            ...     if sim.step_count % 10 == 0:
            ...         print(f"Step {sim.step_count}, E={sim.energy:.3e}")
            >>> sim.run(100, callback=monitor)
        """
        for _ in range(n_steps):
            self.step()
            if callback is not None:
                callback(self)
                
    def run_until(self, t_final: float, callback: Optional[Callable] = None) -> None:
        """
        Run simulation until time t >= t_final.
        
        Args:
            t_final: Target simulation time
            callback: Optional per-step callback
        """
        while self.t < t_final:
            self.step()
            if callback is not None:
                callback(self)
    
    # ----------------------------------------------------------------
    # State Manipulation
    # ----------------------------------------------------------------
    
    def reset(self, field: Optional[np.ndarray] = None) -> None:
        """
        Reset simulation to initial state.
        
        Args:
            field: New initial field (if None, resets to zeros)
        """
        if field is None:
            self.E.fill(0.0)
            self.E_prev.fill(0.0)
        else:
            if field.shape != self.E.shape:
                raise ValueError(f"Field shape {field.shape} does not match {self.E.shape}")
            self.E = field.copy()
            self.E_prev = field.copy()
            
        self.t = 0.0
        self.step_count = 0
        self._energy_history.clear()
        self._time_history.clear()
        self._total_step_time = 0.0
        
    def add_gaussian_pulse(self, center: Tuple[int, ...], amplitude: float = 1.0, width: float = 3.0) -> None:
        """
        Add Gaussian pulse to current field.
        
        Args:
            center: Grid coordinates of pulse center (x,) or (x, y) or (x, y, z)
            amplitude: Pulse amplitude
            width: Pulse width (standard deviation in grid units)
            
        Example:
            >>> sim.add_gaussian_pulse((64, 64), amplitude=2.0, width=5.0)
        """
        if len(center) != self.ndim:
            raise ValueError(f"Center dimension {len(center)} does not match field {self.ndim}D")
            
        if self.ndim == 1:
            cx, = center
            x = np.arange(self.E.shape[0])
            r_sq = (x - cx) ** 2
            self.E += amplitude * np.exp(-r_sq / (2 * width**2))
            
        elif self.ndim == 2:
            cx, cy = center
            y, x = np.ogrid[0:self.E.shape[0], 0:self.E.shape[1]]
            r_sq = (x - cx)**2 + (y - cy)**2
            self.E += amplitude * np.exp(-r_sq / (2 * width**2))
            
        elif self.ndim == 3:
            cx, cy, cz = center
            z, y, x = np.ogrid[0:self.E.shape[0], 0:self.E.shape[1], 0:self.E.shape[2]]
            r_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
            self.E += amplitude * np.exp(-r_sq / (2 * width**2))
            
    def add_plane_wave(self, axis: int, position: int, amplitude: float = 1.0, width: float = 3.0) -> None:
        """
        Add plane wave pulse perpendicular to specified axis.
        
        Args:
            axis: Axis perpendicular to wave (0=x, 1=y, 2=z)
            position: Position along axis
            amplitude: Pulse amplitude
            width: Pulse width
            
        Example:
            >>> sim.add_plane_wave(axis=0, position=64, amplitude=1.0)  # vertical line in 2D
        """
        if axis >= self.ndim:
            raise ValueError(f"Axis {axis} invalid for {self.ndim}D field")
            
        coords = np.arange(self.E.shape[axis])
        pulse_1d = amplitude * np.exp(-(coords - position)**2 / (2 * width**2))
        
        # Broadcast along axis
        shape = [1] * self.ndim
        shape[axis] = -1
        pulse_1d = pulse_1d.reshape(shape)
        self.E += pulse_1d
    
    # ----------------------------------------------------------------
    # Diagnostics & Properties
    # ----------------------------------------------------------------
    
    @property
    def energy(self) -> float:
        """
        Current total energy (kinetic + potential + mass).
        
        Returns:
            Total energy as float
        """
        return energy_total(self.E, self.E_prev, 
                           self.config.dt, self.config.dx,
                           self.config.c, self.config.chi)
        
    @property
    def cfl_ratio(self) -> float:
        """Current CFL ratio: (c * dt) / dx"""
        return self.config.cfl_ratio(self.ndim)
        
    @property
    def cfl_limit(self) -> float:
        """CFL stability limit for current dimensionality."""
        return self.config.cfl_limit(self.ndim)
        
    @property
    def is_stable(self) -> bool:
        """Check if current configuration satisfies CFL condition."""
        return self.config.is_stable(self.ndim)
        
    def field_stats(self) -> Dict[str, float]:
        """
        Get statistical summary of current field.
        
        Returns:
            Dictionary with min, max, mean, std, rms
        """
        return {
            'min': float(self.E.min()),
            'max': float(self.E.max()),
            'mean': float(self.E.mean()),
            'std': float(self.E.std()),
            'rms': float(np.sqrt(np.mean(self.E**2))),
        }
        
    def performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with timing information
        """
        avg_step_time = self._total_step_time / max(self.step_count, 1)
        return {
            'total_time': self._total_step_time,
            'avg_step_time': avg_step_time,
            'last_step_time': self._last_step_time,
            'steps_per_sec': 1.0 / avg_step_time if avg_step_time > 0 else 0.0,
        }
    
    # ----------------------------------------------------------------
    # Save/Load State
    # ----------------------------------------------------------------
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get complete simulation state for saving/checkpointing.
        
        Returns:
            Dictionary containing all state variables
        """
        return {
            'E': self.E.copy(),
            'E_prev': self.E_prev.copy(),
            't': self.t,
            'step_count': self.step_count,
            'config': self.config.to_dict(),
            'energy_history': self._energy_history.copy(),
            'time_history': self._time_history.copy(),
        }
        
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore simulation from saved state.
        
        Args:
            state: State dictionary from get_state()
        """
        self.E = state['E'].copy()
        self.E_prev = state['E_prev'].copy()
        self.t = state['t']
        self.step_count = state['step_count']
        
        if 'config' in state:
            self.config = LFMConfig.from_dict(state['config'])
            
        if 'energy_history' in state:
            self._energy_history = state['energy_history'].copy()
        if 'time_history' in state:
            self._time_history = state['time_history'].copy()
            
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save simulation state to file.
        
        Args:
            filepath: Path to save checkpoint (.npz file)
        """
        state = self.get_state()
        np.savez_compressed(filepath, **state)
        
    @classmethod
    def load_checkpoint(cls, filepath: str) -> LFMSimulator:
        """
        Load simulation from checkpoint file.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            New LFMSimulator instance with restored state
        """
        data = np.load(filepath, allow_pickle=True)
        config = LFMConfig.from_dict(data['config'].item())
        sim = cls(data['E'], config)
        sim.set_state({k: data[k] for k in data.files})
        return sim
    
    # ----------------------------------------------------------------
    # String Representation
    # ----------------------------------------------------------------
    
    def __repr__(self) -> str:
        """Concise representation of simulator state."""
        return (
            f"LFMSimulator(shape={self.E.shape}, t={self.t:.2f}, "
            f"steps={self.step_count}, E={self.energy:.3e})"
        )
        
    def summary(self) -> str:
        """
        Detailed summary of simulation state.
        
        Returns:
            Multi-line string with full state information
        """
        stats = self.field_stats()
        perf = self.performance_stats()
        
        lines = [
            f"LFM Simulator Summary",
            f"=" * 50,
            f"Configuration:",
            f"  Grid: {self.E.shape} ({self.ndim}D)",
            f"  dt={self.config.dt}, dx={self.config.dx}",
            f"  c={self.config.c:.3f}, chi={self.config.chi}",
            f"  Boundary: {self.config.boundary}",
            f"",
            f"State:",
            f"  Time: {self.t:.3f}",
            f"  Steps: {self.step_count}",
            f"  Energy: {self.energy:.6e}",
            f"",
            f"Field Statistics:",
            f"  Min: {stats['min']:.6e}",
            f"  Max: {stats['max']:.6e}",
            f"  Mean: {stats['mean']:.6e}",
            f"  RMS: {stats['rms']:.6e}",
            f"",
            f"Performance:",
            f"  Steps/sec: {perf['steps_per_sec']:.1f}",
            f"  Avg step time: {perf['avg_step_time']*1000:.2f} ms",
            f"",
            f"Stability:",
            f"  CFL ratio: {self.cfl_ratio:.3f} / {self.cfl_limit:.3f}",
            f"  Stable: {'✓' if self.is_stable else '✗'}",
        ]
        
        return "\n".join(lines)
