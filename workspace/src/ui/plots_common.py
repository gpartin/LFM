# -*- coding: utf-8 -*-
"""
Common plotting utilities for tier validation tests.

Provides reusable plot functions to eliminate duplication and prevent
import shadowing bugs. Import matplotlib at module level ONCE.
"""
from pathlib import Path
from typing import Optional, Union
import numpy as np

# Import matplotlib ONCE at module level (prevents shadowing bugs)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_energy_over_time(
    times: np.ndarray,
    energy: np.ndarray,
    title: str,
    out_path: Union[str, Path],
    ylabel: str = "Energy",
    show_initial: bool = True
) -> None:
    """
    Plot energy (or any single quantity) vs time.
    
    Args:
        times: Time array
        energy: Energy values (same length as times)
        title: Plot title
        out_path: Output file path
        ylabel: Y-axis label (default: "Energy")
        show_initial: If True, add horizontal line at initial value
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, energy, 'b-', linewidth=1.5, label=ylabel)
    
    if show_initial and len(energy) > 0:
        ax.axhline(energy[0], color='gray', linestyle='--', 
                   linewidth=1, alpha=0.7, label=f"Initial ({energy[0]:.4f})")
    
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_partition_fractions(
    times: np.ndarray,
    kinetic: np.ndarray,
    potential: np.ndarray,
    total: np.ndarray,
    title: str,
    out_path: Union[str, Path]
) -> None:
    """
    Plot energy partition (kinetic, potential, total) vs time.
    
    Args:
        times: Time array
        kinetic: Kinetic energy values
        potential: Potential energy values
        total: Total energy values
        title: Plot title
        out_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, kinetic, 'r-', linewidth=1.5, label='Kinetic', alpha=0.8)
    ax.plot(times, potential, 'b-', linewidth=1.5, label='Potential', alpha=0.8)
    ax.plot(times, total, 'k--', linewidth=2, label='Total', alpha=0.9)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_momentum(
    times: np.ndarray,
    momentum: np.ndarray,
    title: str,
    out_path: Union[str, Path],
    ylabel: str = "Momentum",
    show_zero: bool = True
) -> None:
    """
    Plot momentum vs time.
    
    Args:
        times: Time array
        momentum: Momentum values
        title: Plot title
        out_path: Output file path
        ylabel: Y-axis label (default: "Momentum")
        show_zero: If True, add horizontal line at zero
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, momentum, 'g-', linewidth=1.5, label=ylabel)
    
    if show_zero:
        ax.axhline(0, color='gray', linestyle='--', 
                   linewidth=1, alpha=0.7, label="Zero")
    
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_dual_series(
    times: np.ndarray,
    series1: np.ndarray,
    series2: np.ndarray,
    label1: str,
    label2: str,
    title: str,
    out_path: Union[str, Path],
    ylabel: str = "Value",
    xlabel: str = "Time"
) -> None:
    """
    Plot two series on the same axes.
    
    Args:
        times: Time array
        series1: First data series
        series2: Second data series
        label1: Label for first series
        label2: Label for second series
        title: Plot title
        out_path: Output file path
        ylabel: Y-axis label
        xlabel: X-axis label (default: "Time")
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, series1, 'b-', linewidth=1.5, label=label1, alpha=0.8)
    ax.plot(times, series2, 'r-', linewidth=1.5, label=label2, alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_theory_vs_measured(
    x_values: np.ndarray,
    theory_values: np.ndarray,
    measured_values: np.ndarray,
    title: str,
    out_path: Union[str, Path],
    xlabel: str = "Mode number n",
    ylabel: str = "Value"
) -> None:
    """
    Plot theoretical vs measured values (for quantization tests).
    
    Args:
        x_values: X-axis values (e.g., mode numbers)
        theory_values: Theoretical predictions
        measured_values: Measured/simulated values
        title: Plot title
        out_path: Output file path
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, theory_values, 'bo-', label='Theory', 
            markersize=10, linewidth=2)
    ax.plot(x_values, measured_values, 'rx--', label='Measured', 
            markersize=10, linewidth=2)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_scatter_with_fit(
    x_data: np.ndarray,
    y_data: np.ndarray,
    y_fit: np.ndarray,
    title: str,
    out_path: Union[str, Path],
    xlabel: str = "X",
    ylabel: str = "Y",
    fit_label: Optional[str] = None
) -> None:
    """
    Plot scatter data with fitted line.
    
    Args:
        x_data: X values
        y_data: Measured Y values (scatter points)
        y_fit: Fitted Y values (line)
        title: Plot title
        out_path: Output file path
        xlabel: X-axis label
        ylabel: Y-axis label
        fit_label: Label for fit line (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_data, y_data, 'bo', markersize=10, label='Measured')
    ax.plot(x_data, y_fit, 'r--', linewidth=2, 
            label=fit_label or 'Fit')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_multi_series(
    x_data: np.ndarray,
    y_data_list: list,
    labels: list,
    title: str,
    out_path: Union[str, Path],
    xlabel: str = "X",
    ylabel: str = "Y"
) -> None:
    """
    Plot multiple series on same axes.
    
    Args:
        x_data: Shared X values
        y_data_list: List of Y arrays (one per series)
        labels: List of labels (one per series)
        title: Plot title
        out_path: Output file path
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for y_data, label in zip(y_data_list, labels):
        ax.plot(x_data, y_data, label=label, alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
