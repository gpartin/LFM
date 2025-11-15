#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gravity Analogue: Light Bending / Deflection Evidence Generator (GRAV-25).

Produces:
- diagnostics/trajectory_data.csv: x,chi,E_final sampled from light_bending_GRAV-25.csv
- diagnostics/chi_field_map.csv: synthetic or copied χ profile (x,chi)
- plots/ray_trajectory.png: χ(x) with expected vs actual packet positions annotated
- plots/deflection_angle.png: visualization of deflection angle and time delay

If raw light_bending_{tid}.csv missing, writes placeholder explaining absence.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable
import json
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.evidence_style import apply_standard_style, STD_FIGSIZE, STD_DPI, COLORS
except Exception:  # pragma: no cover
    plt = None
    apply_standard_style = None
    STD_FIGSIZE = (6.4, 3.6); STD_DPI = 120
    COLORS = {'primary': '#1f77b4', 'secondary': '#d62728', 'accent': '#2ca02c', 'warning': '#ff7f0e'}


def _log(log_fn: Optional[Callable[[str,str], None]], msg: str, level: str = 'INFO') -> None:
    if callable(log_fn):
        try:
            log_fn(msg, level)
        except Exception:
            pass


def generate_deflection_evidence(results_dir: Path, log_fn: Optional[Callable[[str,str], None]] = None) -> bool:
    summary_path = results_dir / 'summary.json'
    if not summary_path.exists():
        return False
    data = json.loads(summary_path.read_text(encoding='utf-8'))
    tid = (data.get('id') or data.get('test_id') or '').upper()
    if tid != 'GRAV-25':
        return False

    diag_dir = results_dir / 'diagnostics'
    plot_dir = results_dir / 'plots'
    diag_dir.mkdir(exist_ok=True); plot_dir.mkdir(exist_ok=True)

    defl = data.get('deflection_angle')
    expected_pos = data.get('expected_position')
    actual_pos = data.get('actual_position')
    chi_grad = data.get('chi_gradient')
    time_delay = data.get('time_delay')

    # Light bending raw CSV
    raw_csv = diag_dir / f'light_bending_{tid}.csv'
    traj_csv = diag_dir / 'trajectory_data.csv'
    chi_map_csv = diag_dir / 'chi_field_map.csv'

    if raw_csv.exists():
        content = raw_csv.read_text(encoding='utf-8').strip().splitlines()
        # Copy trajectory_data.csv with standardized header
        with traj_csv.open('w', encoding='utf-8') as f:
            f.write('x,chi,E_final\n')
            for ln in content[1:]:
                f.write(ln + '\n')
        # Generate chi_field_map.csv (subset sampling for map)
        xs = []
        chis = []
        for ln in content[1:]:
            parts = ln.split(',')
            if len(parts) >= 2:
                try:
                    xs.append(float(parts[0])); chis.append(float(parts[1]))
                except Exception:
                    continue
        with chi_map_csv.open('w', encoding='utf-8') as f:
            f.write('x,chi\n')
            for x_val, chi_val in zip(xs, chis):
                f.write(f'{x_val:.6e},{chi_val:.6e}\n')
    else:
        # Placeholder minimal data
        traj_csv.write_text('x,chi,E_final\n0.0,0.0,0.0\n', encoding='utf-8')
        chi_map_csv.write_text('x,chi\n0.0,0.0\n', encoding='utf-8')

    # Plots
    if plt is not None:
        # Ray trajectory plot
        fig, ax = plt.subplots(figsize=STD_FIGSIZE, dpi=STD_DPI)
        if chi_map_csv.exists():
            rows = chi_map_csv.read_text(encoding='utf-8').strip().splitlines()[1:]
            xs = [float(r.split(',')[0]) for r in rows]
            chis = [float(r.split(',')[1]) for r in rows]
            ax.plot(xs, chis, color=COLORS.get('primary','#1f77b4'), linewidth=1.2, label='χ(x) gradient')
        ax.axvline(expected_pos, color=COLORS.get('secondary','#d62728'), linestyle='--', label='Expected (no bend)')
        ax.axvline(actual_pos, color=COLORS.get('accent','#2ca02c'), linestyle='--', label='Actual (bent)')
        if apply_standard_style:
            apply_standard_style(ax, title=f'Ray Trajectory (Deflection={defl:.3f} rad)', xlabel='x', ylabel='χ(x)')
        else:
            ax.set_title(f'Ray Trajectory (Deflection={defl:.3f} rad)'); ax.set_xlabel('x'); ax.set_ylabel('χ(x)'); ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(plot_dir / 'ray_trajectory.png', bbox_inches='tight'); plt.close(fig)

        # Deflection angle visualization
        fig, ax = plt.subplots(figsize=STD_FIGSIZE, dpi=STD_DPI)
        ax.bar([0],[defl], color=COLORS.get('warning','#ff7f0e'))
        ax.set_xticks([0]); ax.set_xticklabels(['Deflection'])
        note = f'χ-gradient={chi_grad}, time_delay={time_delay:.2f}s'
        if apply_standard_style:
            apply_standard_style(ax, title='Deflection Angle', xlabel='Metric', ylabel='Angle (rad)')
        else:
            ax.set_title('Deflection Angle'); ax.set_xlabel('Metric'); ax.set_ylabel('Angle (rad)'); ax.grid(alpha=0.3)
        ax.text(0.0, defl, note, ha='left', va='bottom', fontsize=8)
        fig.tight_layout(); fig.savefig(plot_dir / 'deflection_angle.png', bbox_inches='tight'); plt.close(fig)

    _log(log_fn, '[gravity] deflection evidence created')
    return True

__all__ = ['generate_deflection_evidence']
