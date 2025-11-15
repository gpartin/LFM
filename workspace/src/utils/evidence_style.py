#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standardized plotting style utilities for evidence generation.

Provides a unified visual identity across all test artifacts:
- Consistent color palette
- Grid styling
- Font size adjustments
- Lightweight helper to apply titles/labels

Usage:
    from utils.evidence_style import apply_standard_style, COLORS
    fig, ax = plt.subplots(figsize=STD_FIGSIZE, dpi=STD_DPI)
    ax.plot(x, y, color=COLORS['primary'])
    apply_standard_style(ax, title='Energy vs Time', xlabel='t', ylabel='E')

Design principles:
- Keep dependencies minimal (Matplotlib only)
- Do not mutate global rcParams aggressively; apply per-axis for isolation
- Graceful no-op if Matplotlib unavailable
"""
from __future__ import annotations
from typing import Dict

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

STD_DPI = 120
STD_FIGSIZE = (6.4, 3.6)
COLORS: Dict[str, str] = {
    'primary': '#1f77b4',
    'secondary': '#d62728',
    'accent': '#2ca02c',
    'warning': '#ff7f0e',
    'muted': '#7f7f7f',
}

_DEF_GRID_ALPHA = 0.3
_DEF_LABEL_FONTSIZE = 10
_DEF_TITLE_FONTSIZE = 11
_DEF_TICK_FONTSIZE = 9


def apply_standard_style(ax, title: str = '', xlabel: str = '', ylabel: str = '') -> None:
    """Apply consistent styling to an axis.

    Args:
        ax: Matplotlib axis
        title: Title text
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    if plt is None:  # pragma: no cover
        return
    ax.set_title(title, fontsize=_DEF_TITLE_FONTSIZE, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=_DEF_LABEL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=_DEF_LABEL_FONTSIZE)
    ax.grid(True, alpha=_DEF_GRID_ALPHA)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(_DEF_TICK_FONTSIZE)
    # Remove top/right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

__all__ = [
    'apply_standard_style', 'COLORS', 'STD_DPI', 'STD_FIGSIZE'
]
