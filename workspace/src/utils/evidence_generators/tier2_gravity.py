#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tier 2 Gravity Analogue evidence generators.

Artifacts produced:
- delay_measurement.csv (phase/time delay metrics for slab propagation GRAV-12)
- arrival_curve.png (envelope before/after vs time)
- dynamic_chi_metrics.csv (χ-wave evolution summary for GRAV-23)
- chi_wave_profile.png (χ perturbation amplitude vs time)

Graceful degradation: if required diagnostics absent, writes placeholder rows and
annotated plot explaining missing raw data.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable, List
import json
import numpy as np

try:
    from utils.evidence_style import apply_standard_style, STD_FIGSIZE, STD_DPI, COLORS
except Exception:  # pragma: no cover
    apply_standard_style = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _log(log_fn: Optional[Callable[[str,str], None]], msg: str, level: str = "INFO") -> None:
    if callable(log_fn):
        try:
            log_fn(msg, level)
        except Exception:
            pass


def generate_phase_delay_evidence(results_dir: Path, log_fn: Optional[Callable[[str,str], None]] = None) -> bool:
    """Generate phase/time delay evidence for GRAV-12 (phase_delay).

    Enhancements:
    - Cross-correlation based delay estimate (correlation_delay_seconds)
    - Unified styling for arrival_curve plot
    """
    summary_path = results_dir / 'summary.json'
    if not summary_path.exists():
        return False
    data = json.loads(summary_path.read_text(encoding='utf-8'))
    test_id = (data.get('id') or data.get('test_id') or '').upper()
    if test_id != 'GRAV-12':
        return False

    diag_dir = results_dir / 'diagnostics'
    plot_dir = results_dir / 'plots'
    diag_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # Extract delay-related metrics from summary if present
    ratio_theory = data.get('ratio_theory')
    ratio_serial = data.get('ratio_serial')
    ratio_match_error = data.get('ratio_match_error')
    chiA = data.get('chiA'); chiB = data.get('chiB')

    # We'll append correlation-based delay after processing envelope file
    delay_lines: List[str] = ['metric,value',
                              f'ratio_theory,{ratio_theory}',
                              f'ratio_measured,{ratio_serial}',
                              f'ratio_error,{ratio_match_error}',
                              f'chi_A,{chiA}',
                              f'chi_B,{chiB}']

    # Envelope measurement file
    env_csv = diag_dir / 'envelope_measurement_GRAV-12.csv'
    made_plot = False
    if env_csv.exists() and plt is not None:
        try:
            raw = env_csv.read_text(encoding='utf-8').strip().splitlines()
            header = raw[0].split(',')
            col_map = {name: idx for idx, name in enumerate(header)}
            time_vals = []
            before_env = []
            after_env = []
            for ln in raw[1:]:
                parts = ln.split(',')
                if len(parts) < len(header):
                    continue
                time_vals.append(float(parts[col_map['time']]))
                before_env.append(float(parts[col_map['env_before']]))
                after_env.append(float(parts[col_map['env_after']]))
            time_arr = np.array(time_vals); b_arr = np.array(before_env); a_arr = np.array(after_env)
            if len(time_arr) > 5:
                fig, ax = plt.subplots(figsize=STD_FIGSIZE if apply_standard_style else (6.4,3.6),
                                       dpi=STD_DPI if apply_standard_style else 120)
                ax.plot(time_arr, b_arr, label='Before (env)', color=COLORS.get('primary','#1f77b4'))
                ax.plot(time_arr, a_arr, label='After (env)', color=COLORS.get('secondary','#d62728'))
                if apply_standard_style:
                    apply_standard_style(ax, title='Envelope Amplitude vs Time — Phase Delay', xlabel='time (s)', ylabel='envelope amplitude')
                else:
                    ax.set_title('Envelope Amplitude vs Time — Phase Delay')
                    ax.set_xlabel('time (s)'); ax.set_ylabel('envelope amplitude'); ax.grid(alpha=0.3)
                ax.legend(frameon=False)
                # Simple delay estimate: difference in time when after_env drops below half its initial value
                half_initial = a_arr[0] * 0.5
                crossing_idx = np.argmax(a_arr < half_initial)
                if crossing_idx > 0:
                    delay_est = time_arr[crossing_idx]
                    ax.axvline(delay_est, color='k', ls='--', lw=1)
                    ax.text(delay_est, max(a_arr)*0.9, f'delay≈{delay_est:.2f}s', rotation=90,
                            ha='right', va='top', fontsize=8)
                # Cross-correlation delay estimate (lag of peak correlation)
                try:
                    if len(b_arr) == len(a_arr) and len(b_arr) > 10:
                        b_norm = b_arr - b_arr.mean()
                        a_norm = a_arr - a_arr.mean()
                        corr = np.correlate(b_norm, a_norm, mode='full')
                        lag = np.argmax(corr) - (len(b_arr) - 1)
                        dt = float(np.median(np.diff(time_arr))) if len(time_arr) > 1 else 0.0
                        corr_delay = lag * dt
                        ax.text(time_arr[0], max(a_arr)*0.75, f'corr_delay≈{corr_delay:.2f}s', fontsize=8,
                                ha='left', va='center', color=COLORS.get('accent','#2ca02c'))
                        delay_lines.append(f'correlation_delay_seconds,{corr_delay}')
                except Exception as ce:  # pragma: no cover
                    _log(log_fn, f'[gravity] correlation delay failed: {ce}', 'WARN')
                fig.tight_layout()
                fig.savefig(plot_dir / 'arrival_curve.png', bbox_inches='tight')
                plt.close(fig)
                made_plot = True
        except Exception as e:  # pragma: no cover
            _log(log_fn, f'[gravity] arrival_curve generation failed: {e}', 'WARN')

    if not made_plot and plt is not None:
        # Placeholder plot
        fig, ax = plt.subplots(figsize=(5,3), dpi=120)
        ax.text(0.5,0.5,'envelope_measurement missing', ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title('Arrival Curve (placeholder)')
        fig.savefig(plot_dir / 'arrival_curve.png', bbox_inches='tight')
        plt.close(fig)

    # Write delay measurement CSV (now includes correlation delay if computed)
    delay_csv = diag_dir / 'delay_measurement.csv'
    delay_csv.write_text('\n'.join(delay_lines) + '\n', encoding='utf-8')

    _log(log_fn, '[gravity] phase delay evidence created')
    return True


def generate_dynamic_chi_evidence(results_dir: Path, log_fn: Optional[Callable[[str,str], None]] = None) -> bool:
    """Generate dynamic χ-field evolution evidence for GRAV-23.

    Enhancements:
    - Unified style usage
    - Frequency spectrum (FFT) of χ-wave perturbation if history file present
    """
    summary_path = results_dir / 'summary.json'
    if not summary_path.exists():
        return False
    data = json.loads(summary_path.read_text(encoding='utf-8'))
    test_id = (data.get('id') or data.get('test_id') or '').upper()
    if test_id != 'GRAV-23':
        return False

    diag_dir = results_dir / 'diagnostics'
    plot_dir = results_dir / 'plots'
    diag_dir.mkdir(exist_ok=True); plot_dir.mkdir(exist_ok=True)

    chi_max = data.get('chi_pert_max'); chi_rms = data.get('chi_pert_rms'); chi_edge = data.get('chi_pert_edge')
    drift_frac = data.get('energy_drift_frac')

    dyn_csv = diag_dir / 'dynamic_chi_metrics.csv'
    dyn_lines = ['metric,value',
                 f'chi_pert_max,{chi_max}',
                 f'chi_pert_rms,{chi_rms}',
                 f'chi_edge_value,{chi_edge}',
                 f'energy_drift_frac,{drift_frac}']
    dyn_csv.write_text('\n'.join(dyn_lines) + '\n', encoding='utf-8')

    # Existing plot (chi_wave_evolution_GRAV-23.png) may already exist; create a supplemental magnitude profile
    existing = plot_dir / 'chi_wave_evolution_GRAV-23.png'
    if existing.exists() and plt is not None:
        # Create small annotation figure summarizing stats
        fig, ax = plt.subplots(figsize=STD_FIGSIZE if apply_standard_style else (5,3),
                               dpi=STD_DPI if apply_standard_style else 120)
        ax.axis('off')
        if apply_standard_style:
            ax.set_title('Dynamic χ Summary')
        else:
            ax.set_title('Dynamic χ Summary')
        txt = (f"max: {chi_max:.4e}\n"
               f"rms: {chi_rms:.4e}\n"
               f"edge: {chi_edge:.2e}\n"
               f"drift: {drift_frac:.2f}")
        ax.text(0.05,0.85,txt, ha='left', va='top', fontsize=9)
        fig.savefig(plot_dir / 'chi_wave_profile.png', bbox_inches='tight')
        plt.close(fig)
    elif plt is not None:
        fig, ax = plt.subplots(figsize=(5,3), dpi=120)
        ax.text(0.5,0.5,'chi_wave_evolution plot missing', ha='center', va='center')
        ax.axis('off'); fig.savefig(plot_dir / 'chi_wave_profile.png', bbox_inches='tight'); plt.close(fig)

    # Spectrum generation (optional) from chi_wave_history_GRAV-23.csv
    if plt is not None:
        history_csv = diag_dir / 'chi_wave_history_GRAV-23.csv'
        if history_csv.exists():
            try:
                lines = history_csv.read_text(encoding='utf-8').strip().splitlines()
                header = [h.strip() for h in lines[0].split(',')]
                # Attempt to identify time column
                time_idx = None
                amp_idx = None
                for i, h in enumerate(header):
                    hl = h.lower()
                    if time_idx is None and hl in ('time','t','step_time'):
                        time_idx = i
                    if amp_idx is None and any(token in hl for token in ('amp','amplitude','chi_pert','chi_val')):
                        amp_idx = i
                if time_idx is None:
                    time_idx = 0  # fallback first numeric
                if amp_idx is None:
                    amp_idx = 1 if len(header) > 1 else 0
                t_vals = []
                a_vals = []
                for ln in lines[1:]:
                    parts = [p.strip() for p in ln.split(',')]
                    if len(parts) <= max(time_idx, amp_idx):
                        continue
                    try:
                        t_vals.append(float(parts[time_idx]))
                        a_vals.append(float(parts[amp_idx]))
                    except Exception:
                        continue
                t_arr = np.array(t_vals, dtype=float)
                a_arr = np.array(a_vals, dtype=float)
                if len(t_arr) > 16:
                    # Detrend
                    a_arr = a_arr - a_arr.mean()
                    dt = float(np.median(np.diff(t_arr))) if len(t_arr) > 1 else 1.0
                    freqs = np.fft.rfftfreq(len(a_arr), dt)
                    spec = np.abs(np.fft.rfft(a_arr))
                    # Normalize spectrum
                    if spec.max() > 0:
                        spec /= spec.max()
                    # Save CSV
                    spectrum_csv = diag_dir / 'chi_wave_spectrum.csv'
                    spectrum_lines = ['frequency,amplitude_norm'] + [f'{f:.6e},{s:.6e}' for f,s in zip(freqs, spec)]
                    spectrum_csv.write_text('\n'.join(spectrum_lines)+ '\n', encoding='utf-8')
                    # Plot
                    fig, ax = plt.subplots(figsize=STD_FIGSIZE if apply_standard_style else (6,3.6),
                                           dpi=STD_DPI if apply_standard_style else 120)
                    ax.plot(freqs, spec, color=COLORS.get('accent','#2ca02c'))
                    if apply_standard_style:
                        apply_standard_style(ax, title='χ-Wave Spectrum', xlabel='frequency (arb)', ylabel='normalized amplitude')
                    else:
                        ax.set_title('χ-Wave Spectrum'); ax.set_xlabel('frequency (arb)'); ax.set_ylabel('normalized amplitude'); ax.grid(alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(plot_dir / 'chi_wave_spectrum.png', bbox_inches='tight')
                    plt.close(fig)
            except Exception as se:  # pragma: no cover
                _log(log_fn, f'[gravity] chi spectrum failed: {se}', 'WARN')

    _log(log_fn, '[gravity] dynamic chi evidence created')
    return True

__all__ = ['generate_phase_delay_evidence','generate_dynamic_chi_evidence']
