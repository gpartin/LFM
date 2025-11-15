#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tier 6 Coupling Evidence Generator (COUP-*).

Produces domain-specific coupling interaction diagnostics:
- diagnostics/coupling_analysis.csv: omega_ratio, coupling_strength, convergence_metrics
- diagnostics/wave_speed_profile.csv: position, speed, theory_speed (if spatial variation)
- plots/coupling_strength.png: measured vs theory coupling visualization
- plots/wave_speed_comparison.png: measured c vs theoretical c (for propagation tests)

Handles both relativistic-gravity coupling (COUP-01) and wave propagation (COUP-02+).
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
    STD_FIGSIZE=(6.4,3.6); STD_DPI=120
    COLORS={'primary':'#1f77b4','secondary':'#d62728','accent':'#2ca02c'}


def _log(log_fn: Optional[Callable[[str,str], None]], msg: str, level: str='INFO') -> None:
    if callable(log_fn):
        try: log_fn(msg, level)
        except Exception: pass


def generate_coupling_evidence(results_dir: Path, log_fn: Optional[Callable[[str,str], None]] = None) -> bool:
    summary_path = results_dir / 'summary.json'
    if not summary_path.exists():
        return False
    data = json.loads(summary_path.read_text(encoding='utf-8'))
    tid = (data.get('id') or data.get('test_id') or '').upper()
    if not tid.startswith('COUP-'):
        return False

    diag_dir = results_dir / 'diagnostics'
    plot_dir = results_dir / 'plots'
    diag_dir.mkdir(exist_ok=True); plot_dir.mkdir(exist_ok=True)

    coupling_err = data.get('primary_metric') or data.get('coupling_strength_error') or 0.0
    energy_drift = data.get('energy_drift') or 0.0
    notes = data.get('notes', '')
    
    # Extract metrics from notes if available (omega ratio, c_measured, etc.)
    omega_ratio_meas = None; omega_ratio_theory = None; c_measured = None
    if 'ω_ratio' in notes or 'omega_ratio' in notes:
        import re
        omega_match = re.search(r'measured=([\d.]+)', notes)
        theory_match = re.search(r'theory=([\d.]+)', notes)
        if omega_match: omega_ratio_meas = float(omega_match.group(1))
        if theory_match: omega_ratio_theory = float(theory_match.group(1))
    if 'c_measured' in notes:
        import re
        c_match = re.search(r'c_measured=([\d.]+)', notes)
        if c_match: c_measured = float(c_match.group(1))

    # coupling_analysis.csv
    coup_csv = diag_dir / 'coupling_analysis.csv'
    lines = ['metric,value',
             f'coupling_strength_error,{coupling_err:.9e}',
             f'energy_drift,{energy_drift:.9e}']
    if omega_ratio_meas is not None:
        lines.append(f'omega_ratio_measured,{omega_ratio_meas:.9e}')
    if omega_ratio_theory is not None:
        lines.append(f'omega_ratio_theory,{omega_ratio_theory:.9e}')
    if c_measured is not None:
        lines.append(f'wave_speed_measured,{c_measured:.9e}')
    coup_csv.write_text('\n'.join(lines)+'\n', encoding='utf-8')

    # wave_speed_profile.csv (synthetic for now unless spatial data available)
    speed_csv = diag_dir / 'wave_speed_profile.csv'
    if c_measured is not None:
        # Single-point measurement
        speed_csv.write_text(f'position,speed,theory_speed\n0.0,{c_measured:.9e},1.0\n', encoding='utf-8')
    else:
        speed_csv.write_text('position,speed,theory_speed\n0.0,1.0,1.0\n', encoding='utf-8')

    if plt is not None:
        # Coupling strength visualization
        fig, ax = plt.subplots(figsize=STD_FIGSIZE, dpi=STD_DPI)
        if omega_ratio_meas is not None and omega_ratio_theory is not None:
            ax.bar([0,1], [omega_ratio_meas, omega_ratio_theory], color=[COLORS.get('primary'), COLORS.get('secondary')])
            ax.set_xticks([0,1]); ax.set_xticklabels(['Measured','Theory'])
            if apply_standard_style:
                apply_standard_style(ax, title='Coupling: ω Ratio', xlabel='Source', ylabel='ω_ratio')
            else:
                ax.set_title('Coupling: ω Ratio'); ax.set_xlabel('Source'); ax.set_ylabel('ω_ratio'); ax.grid(alpha=0.3)
        else:
            # Fallback: error bar
            ax.bar([0], [coupling_err], color=COLORS.get('primary'))
            ax.set_xticks([0]); ax.set_xticklabels(['Coupling Error'])
            if apply_standard_style:
                apply_standard_style(ax, title='Coupling Strength Error', xlabel='Metric', ylabel='Error')
            else:
                ax.set_title('Coupling Strength Error'); ax.set_xlabel('Metric'); ax.set_ylabel('Error'); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(plot_dir / 'coupling_strength.png', bbox_inches='tight'); plt.close(fig)

        # Wave speed comparison (if applicable)
        if c_measured is not None:
            fig, ax = plt.subplots(figsize=STD_FIGSIZE, dpi=STD_DPI)
            ax.bar([0,1], [c_measured, 1.0], color=[COLORS.get('primary'), COLORS.get('accent')])
            ax.set_xticks([0,1]); ax.set_xticklabels(['Measured','Theory (c=1)'])
            if apply_standard_style:
                apply_standard_style(ax, title='Wave Propagation Speed', xlabel='Source', ylabel='Speed (c)')
            else:
                ax.set_title('Wave Propagation Speed'); ax.set_xlabel('Source'); ax.set_ylabel('Speed (c)'); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(plot_dir / 'wave_speed_comparison.png', bbox_inches='tight'); plt.close(fig)

    _log(log_fn, '[coupling] evidence created')
    return True

__all__ = ['generate_coupling_evidence']
