#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tier 4 Quantization Evidence Generators (tunneling, bound states).

Implemented: QUAN-12 (tunneling)
Produces:
- diagnostics/transmission_data.csv (energy, transmission_coeff, theory, rel_error)
- diagnostics/barrier_profile.csv (x,V(x)) synthetic rectangular barrier
- plots/transmission_curve.png (measured vs theory, log-scale)
- plots/barrier_schematic.png (potential + incident/transmitted annotation)
Optional future: fit_residuals.csv & transmission_fit_residual.png
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


def generate_tunneling_evidence(results_dir: Path, log_fn: Optional[Callable[[str,str], None]] = None) -> bool:
    summary_path = results_dir / 'summary.json'
    if not summary_path.exists():
        return False
    data = json.loads(summary_path.read_text(encoding='utf-8'))
    tid = (data.get('id') or data.get('test_id') or '').upper()
    if tid != 'QUAN-12':
        return False

    diag_dir = results_dir / 'diagnostics'
    plot_dir = results_dir / 'plots'
    diag_dir.mkdir(exist_ok=True); plot_dir.mkdir(exist_ok=True)

    metrics = data.get('metrics', {})
    transmission_coeff = metrics.get('transmission_coefficient')
    theory_transmission = metrics.get('theory_transmission')
    rel_error = metrics.get('relative_error')
    kappa = metrics.get('kappa')
    params = data.get('parameters', {})
    barrier_width = params.get('barrier_width')
    chi_barrier = params.get('chi_barrier')
    packet_omega = params.get('packet_omega')

    # transmission_data.csv
    trans_csv = diag_dir / 'transmission_data.csv'
    trans_csv.write_text('energy,transmission_coeff,theory,rel_error\n'
                         f'{packet_omega:.9e},{transmission_coeff:.9e},{theory_transmission:.9e},{rel_error:.9e}\n',
                         encoding='utf-8')

    # barrier_profile.csv (simple rectangular barrier)
    barrier_csv = diag_dir / 'barrier_profile.csv'
    x = np.linspace(0, barrier_width, 50)
    V = np.full_like(x, chi_barrier)
    barrier_lines = ['x,V'] + [f'{xi:.9e},{vi:.9e}' for xi,vi in zip(x,V)]
    barrier_csv.write_text('\n'.join(barrier_lines)+'\n', encoding='utf-8')

    if plt is not None:
        # Transmission curve (single point + theory curve placeholder exponential exp(-2κL))
        fig, ax = plt.subplots(figsize=STD_FIGSIZE, dpi=STD_DPI)
        ax.scatter([packet_omega],[transmission_coeff], color=COLORS.get('primary'))
        # Theory exponential scaling curve
        omega_grid = np.linspace(packet_omega*0.5, packet_omega*1.5, 100)
        theory_curve = np.exp(-2.0 * kappa * barrier_width) * np.ones_like(omega_grid)
        ax.plot(omega_grid, theory_curve, color=COLORS.get('secondary'), label='Theory exp(-2κL)')
        ax.set_yscale('log')
        if apply_standard_style:
            apply_standard_style(ax, title='Tunneling Transmission', xlabel='Energy (ω)', ylabel='Transmission (log)')
        else:
            ax.set_title('Tunneling Transmission'); ax.set_xlabel('Energy (ω)'); ax.set_ylabel('Transmission'); ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(plot_dir / 'transmission_curve.png', bbox_inches='tight'); plt.close(fig)

        # Barrier schematic
        fig, ax = plt.subplots(figsize=STD_FIGSIZE, dpi=STD_DPI)
        ax.plot(x, V, color=COLORS.get('warning'), linewidth=2, label='Barrier V(x)')
        ax.axhline(packet_omega, color=COLORS.get('primary'), linestyle='--', label='Packet Energy ω')
        if apply_standard_style:
            apply_standard_style(ax, title='Barrier Schematic', xlabel='x', ylabel='Potential / Energy')
        else:
            ax.set_title('Barrier Schematic'); ax.set_xlabel('x'); ax.set_ylabel('Potential / Energy'); ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(plot_dir / 'barrier_schematic.png', bbox_inches='tight'); plt.close(fig)

    _log(log_fn, '[quantization] tunneling evidence created')
    return True

__all__ = ['generate_tunneling_evidence']
