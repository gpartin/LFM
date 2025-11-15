#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gravity Analogue: Gravitational Redshift Evidence Generator (GRAV-13).

Produces:
- diagnostics/frequency_shifts.csv: chi, omega_measured, omega_theory, shift_pct, rel_error
- diagnostics/chi_profile.csv: position index (placeholder), chi value (two-point or multi-point if data available)
- plots/redshift_curve.png: omega vs chi with theoretical scaling overlay
- plots/frequency_vs_position.png: Placeholder position plot (if spatial profile absent)

If only summary.json exists (no spatial χ profile), creates synthetic two-point dataset from A/B probe values.
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
    COLORS = {'primary': '#1f77b4', 'secondary': '#d62728', 'accent': '#2ca02c'}


def _log(log_fn: Optional[Callable[[str,str], None]], msg: str, level: str = 'INFO') -> None:
    if callable(log_fn):
        try:
            log_fn(msg, level)
        except Exception:
            pass


def generate_redshift_evidence(results_dir: Path, log_fn: Optional[Callable[[str,str], None]] = None) -> bool:
    summary_path = results_dir / 'summary.json'
    if not summary_path.exists():
        return False
    data = json.loads(summary_path.read_text(encoding='utf-8'))
    tid = (data.get('id') or data.get('test_id') or '').upper()
    if tid != 'GRAV-13':
        return False

    diag_dir = results_dir / 'diagnostics'
    plot_dir = results_dir / 'plots'
    diag_dir.mkdir(exist_ok=True); plot_dir.mkdir(exist_ok=True)

    chiA = data.get('chiA'); chiB = data.get('chiB')
    omegaA_meas = data.get('omegaA_serial') or data.get('omegaA_measured') or data.get('omegaA_parallel')
    omegaB_meas = data.get('omegaB_serial') or data.get('omegaB_measured') or data.get('omegaB_parallel')
    omegaA_theory = data.get('omegaA_theory'); omegaB_theory = data.get('omegaB_theory')

    # Construct synthetic profile (two points)
    chi_vals = np.array([chiA, chiB], dtype=float)
    omega_meas = np.array([omegaA_meas, omegaB_meas], dtype=float)
    omega_th = np.array([omegaA_theory, omegaB_theory], dtype=float)

    # Frequency shift percent vs theory
    shift_pct = (omega_meas - omega_th) / np.where(omega_th != 0, omega_th, 1.0) * 100.0
    rel_error = np.abs(omega_meas - omega_th) / np.where(omega_th != 0, omega_th, 1.0)

    # Save chi_profile.csv (synthetic index positions 0,1)
    chi_profile_csv = diag_dir / 'chi_profile.csv'
    chi_lines = ['index,chi'] + [f'{i},{val:.9e}' for i, val in enumerate(chi_vals)]
    chi_profile_csv.write_text('\n'.join(chi_lines)+'\n', encoding='utf-8')

    # Save frequency_shifts.csv
    freq_csv = diag_dir / 'frequency_shifts.csv'
    freq_lines = ['chi,omega_measured,omega_theory,shift_pct,rel_error']
    for c, om_m, om_t, sp, re in zip(chi_vals, omega_meas, omega_th, shift_pct, rel_error):
        freq_lines.append(f'{c:.9e},{om_m:.9e},{om_t:.9e},{sp:.6f},{re:.6e}')
    freq_csv.write_text('\n'.join(freq_lines)+'\n', encoding='utf-8')

    if plt is not None:
        # Redshift curve omega vs chi
        fig, ax = plt.subplots(figsize=STD_FIGSIZE, dpi=STD_DPI)
        ax.scatter(chi_vals, omega_meas, color=COLORS.get('primary','#1f77b4'), label='Measured ω')
        ax.plot(chi_vals, omega_th, color=COLORS.get('secondary','#d62728'), label='Theory ω')
        # Fit simple proportional scaling (ω ∝ χ) for visual aid
        if np.all(chi_vals > 0):
            k_fit = np.sum(chi_vals * omega_meas) / np.sum(chi_vals * chi_vals)
            chi_grid = np.linspace(min(chi_vals), max(chi_vals), 50)
            ax.plot(chi_grid, k_fit * chi_grid, color=COLORS.get('accent','#2ca02c'), linestyle='--', label=f'Fit ω≈{k_fit:.3f}χ')
        if apply_standard_style:
            apply_standard_style(ax, title='Gravitational Redshift: ω vs χ', xlabel='χ', ylabel='ω')
        else:
            ax.set_title('Gravitational Redshift: ω vs χ'); ax.set_xlabel('χ'); ax.set_ylabel('ω'); ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_dir / 'redshift_curve.png', bbox_inches='tight')
        plt.close(fig)

        # Frequency vs position placeholder (indices)
        fig, ax = plt.subplots(figsize=STD_FIGSIZE, dpi=STD_DPI)
        ax.bar([0,1], omega_meas, color=COLORS.get('primary','#1f77b4'))
        ax.set_xticks([0,1]); ax.set_xticklabels(['Probe A','Probe B'])
        if apply_standard_style:
            apply_standard_style(ax, title='Local Frequencies (Probes)', xlabel='Probe', ylabel='ω_measured')
        else:
            ax.set_title('Local Frequencies (Probes)'); ax.set_xlabel('Probe'); ax.set_ylabel('ω_measured'); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / 'frequency_vs_position.png', bbox_inches='tight')
        plt.close(fig)

    _log(log_fn, '[gravity] redshift evidence created')
    return True

__all__ = ['generate_redshift_evidence']
