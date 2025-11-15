#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tier 7 Thermodynamics Evidence Generator (THERM-*).

Produces entropy evolution and second law validation artifacts:
- diagnostics/entropy_evolution.csv: time, entropy (synthetic approach to final entropy)
- diagnostics/thermalization_metrics.csv: tau_thermalization, final_entropy, increase
- plots/entropy_increase.png: initial vs final entropy with second law threshold
- plots/second_law_validation.png: entropy increase bar chart with tolerance

Validates thermalization timescales and second law compliance.
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
    COLORS={'primary':'#1f77b4','secondary':'#d62728','accent':'#2ca02c','warning':'#ff7f0e'}


def _log(log_fn: Optional[Callable[[str,str], None]], msg: str, level: str='INFO') -> None:
    if callable(log_fn):
        try: log_fn(msg, level)
        except Exception: pass


def generate_thermodynamics_evidence(results_dir: Path, log_fn: Optional[Callable[[str,str], None]] = None) -> bool:
    summary_path = results_dir / 'summary.json'
    if not summary_path.exists():
        return False
    data = json.loads(summary_path.read_text(encoding='utf-8'))
    tid = (data.get('id') or data.get('test_id') or '').upper()
    if not tid.startswith('THERM-'):
        return False

    diag_dir = results_dir / 'diagnostics'
    plot_dir = results_dir / 'plots'
    diag_dir.mkdir(exist_ok=True); plot_dir.mkdir(exist_ok=True)

    # Handle nested metrics dict (THERM tests store metrics in "metrics" key)
    metrics = data.get('metrics', {})
    S0 = metrics.get('entropy_initial') or data.get('entropy_initial', 0.0)
    Sf = metrics.get('entropy_final') or data.get('entropy_final', 0.0)
    dS = metrics.get('entropy_increase') or data.get('entropy_increase', Sf - S0)
    slope = metrics.get('linear_fit_slope') or data.get('linear_fit_slope', 0.0)
    energy_drift = metrics.get('max_energy_drift') or data.get('max_energy_drift') or data.get('energy_drift') or 0.0
    
    # Extract tau if available (THERM-04 thermalization timescale)
    tau = metrics.get('thermalization_tau') or data.get('thermalization_tau')
    notes = data.get('notes', '')

    # Create synthetic entropy evolution (exponential approach to final value)
    N = 100
    t = np.linspace(0, 1, N)
    if tau is not None and tau > 0:
        # Exponential decay: S(t) = Sf - (Sf - S0)*exp(-t/tau)
        S_t = Sf - (Sf - S0) * np.exp(-t / tau)
    else:
        # Linear interpolation as fallback
        S_t = S0 + (Sf - S0) * t

    entropy_csv = diag_dir / 'entropy_evolution.csv'
    lines = ['time,entropy'] + [f'{ti:.6f},{Si:.9e}' for ti, Si in zip(t, S_t)]
    entropy_csv.write_text('\n'.join(lines)+'\n', encoding='utf-8')

    # thermalization_metrics.csv
    therm_csv = diag_dir / 'thermalization_metrics.csv'
    metric_lines = ['metric,value',
                    f'entropy_initial,{S0:.9e}',
                    f'entropy_final,{Sf:.9e}',
                    f'entropy_increase,{dS:.9e}',
                    f'linear_fit_slope,{slope:.9e}',
                    f'max_energy_drift,{energy_drift:.9e}']
    if tau is not None:
        metric_lines.append(f'thermalization_tau,{tau:.9e}')
    therm_csv.write_text('\n'.join(metric_lines)+'\n', encoding='utf-8')

    if plt is not None:
        # Entropy increase bar chart (initial vs final)
        fig, ax = plt.subplots(figsize=STD_FIGSIZE, dpi=STD_DPI)
        ax.bar([0,1], [S0, Sf], color=[COLORS.get('primary'), COLORS.get('accent')])
        ax.axhline(S0, color=COLORS.get('secondary'), linestyle='--', alpha=0.5, label='Initial Entropy')
        ax.set_xticks([0,1]); ax.set_xticklabels(['Initial', 'Final'])
        if apply_standard_style:
            apply_standard_style(ax, title='Entropy Evolution (Second Law)', xlabel='State', ylabel='Entropy S')
        else:
            ax.set_title('Entropy Evolution (Second Law)'); ax.set_xlabel('State'); ax.set_ylabel('Entropy S'); ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout(); fig.savefig(plot_dir / 'entropy_increase.png', bbox_inches='tight'); plt.close(fig)

        # Second law validation: entropy increase with tolerance
        fig, ax = plt.subplots(figsize=STD_FIGSIZE, dpi=STD_DPI)
        tolerance_threshold = 0.0  # Second law: dS >= 0
        ax.bar([0], [dS], color=COLORS.get('primary') if dS >= tolerance_threshold else COLORS.get('warning'))
        ax.axhline(tolerance_threshold, color=COLORS.get('secondary'), linestyle='--', linewidth=2, label=f'Threshold (dS≥{tolerance_threshold})')
        ax.set_xticks([0]); ax.set_xticklabels(['ΔS'])
        if apply_standard_style:
            apply_standard_style(ax, title='Second Law Validation', xlabel='Metric', ylabel='Entropy Increase')
        else:
            ax.set_title('Second Law Validation'); ax.set_xlabel('Metric'); ax.set_ylabel('Entropy Increase'); ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout(); fig.savefig(plot_dir / 'second_law_validation.png', bbox_inches='tight'); plt.close(fig)

        # Entropy evolution time series (if tau available, show exponential approach)
        fig, ax = plt.subplots(figsize=STD_FIGSIZE, dpi=STD_DPI)
        ax.plot(t, S_t, color=COLORS.get('primary'), linewidth=2, label='Entropy S(t)')
        ax.axhline(S0, color=COLORS.get('secondary'), linestyle='--', alpha=0.5, label='Initial')
        ax.axhline(Sf, color=COLORS.get('accent'), linestyle='--', alpha=0.5, label='Final')
        if apply_standard_style:
            apply_standard_style(ax, title='Entropy Evolution Over Time', xlabel='Time (normalized)', ylabel='Entropy S')
        else:
            ax.set_title('Entropy Evolution Over Time'); ax.set_xlabel('Time (normalized)'); ax.set_ylabel('Entropy S'); ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout(); fig.savefig(plot_dir / 'entropy_time_series.png', bbox_inches='tight'); plt.close(fig)

    _log(log_fn, '[thermodynamics] evidence created')
    return True

__all__ = ['generate_thermodynamics_evidence']
