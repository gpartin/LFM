#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evidence artifact utilities for LFM tests.

Generates lightweight, standardized per-test artifacts that improve
reviewability without changing physics code paths.

Artifacts (all UTF-8 paths; images are PNG):
- diagnostics/summary_metrics.csv
- plots/summary_card.png (key metrics rendered as an at-a-glance card)
- Optional: plots from common time-series CSVs if present
    * diagnostics/projection_series.csv -> plots/projection_series.png
    * diagnostics/energy_evolution.csv  -> plots/energy_evolution.png

Design goals:
- Zero GPU deps; operate on host (NumPy / Matplotlib)
- Best-effort: never throw; log via provided logger callable
- Small and fast: render minimal figures at ~100–140 dpi
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Callable
import json
import hashlib
import time
from utils.evidence_schema import get_schema_for_test

import numpy as np

# Matplotlib is only used for PNG generation; import lazily to avoid overhead
try:
    import matplotlib
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None


def _safe_log(log_fn: Optional[Callable[[str,str], None]], msg: str, level: str = "INFO") -> None:
    try:
        if callable(log_fn):
            log_fn(msg, level)
    except Exception:
        pass


def emit_summary_artifacts(results_dir: Path, log_fn: Optional[Callable[[str,str], None]] = None) -> bool:
    """Create world-class domain-specific evidence artifacts.

    Generates:
      1. Universal: diagnostics/summary_metrics.csv
      2. Domain-specific physics plots (isotropy, dispersion, etc.)
      3. Optional: time series from diagnostics CSVs

    Returns True if any artifact was created.
    """
    created_any = False
    try:
        summary_path = results_dir / 'summary.json'
        if not summary_path.exists():
            return False
        data = json.loads(summary_path.read_text(encoding='utf-8'))
        
        # Extract key fields defensively
        test_id = str(data.get('id') or data.get('test_id') or '')
        metric_name = str(data.get('metric_name') or data.get('validation', {}).get('primary', {}).get('name') or '')
        primary_metric = data.get('primary_metric')
        if primary_metric is None:
            # Try common names
            for k in ('rel_err', 'anisotropy', 'phase_error', 'spherical_error', 'linearity_error'):
                if k in data:
                    primary_metric = data[k]
                    metric_name = metric_name or k
                    break
        energy_drift = data.get('energy_drift')
        runtime_sec = data.get('runtime_sec')
        passed = bool(data.get('passed', False))

        # Ensure directories
        diag_dir = results_dir / 'diagnostics'
        plot_dir = results_dir / 'plots'
        diag_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Write summary CSV
        csv_lines = [
            'metric,value',
            f"test_id,{test_id}",
            f"passed,{passed}",
        ]
        if metric_name and (primary_metric is not None):
            csv_lines.append(f"{metric_name},{primary_metric:.12e}")
        if energy_drift is not None:
            csv_lines.append(f"energy_drift,{float(energy_drift):.12e}")
        if runtime_sec is not None:
            csv_lines.append(f"runtime_sec,{float(runtime_sec):.6f}")
        (diag_dir / 'summary_metrics.csv').write_text('\n'.join(csv_lines) + '\n', encoding='utf-8')
        created_any = True

        # Generate domain-specific evidence (world-class physics plots)
        try:
            from utils.evidence_generators.tier1_relativistic import (
                generate_isotropy_evidence,
                generate_dispersion_evidence,
                generate_boost_evidence
            )
            # Electromagnetic (Tier 5)
            try:
                from utils.evidence_generators.tier5_electromagnetic import (
                    generate_electromagnetic_evidence
                )
            except Exception as em_err:  # pragma: no cover
                generate_electromagnetic_evidence = None  # type: ignore
                _safe_log(log_fn, f"[evidence] EM generator import failed: {em_err}", 'WARN')
            # Gravity analogue (Tier 2 incremental)
            try:
                from utils.evidence_generators.tier2_gravity import (
                    generate_phase_delay_evidence,
                    generate_dynamic_chi_evidence
                )
            except Exception as grav_err:  # pragma: no cover
                generate_phase_delay_evidence = None  # type: ignore
                generate_dynamic_chi_evidence = None  # type: ignore
                _safe_log(log_fn, f"[evidence] gravity generator import failed: {grav_err}", 'WARN')
            # Additional gravity analogue generators (redshift, deflection)
            try:
                from utils.evidence_generators.tier2_gravity_redshift import generate_redshift_evidence
            except Exception as grav_r_err:  # pragma: no cover
                generate_redshift_evidence = None  # type: ignore
                _safe_log(log_fn, f"[evidence] redshift generator import failed: {grav_r_err}", 'WARN')
            try:
                from utils.evidence_generators.tier2_gravity_deflection import generate_deflection_evidence
            except Exception as grav_d_err:  # pragma: no cover
                generate_deflection_evidence = None  # type: ignore
                _safe_log(log_fn, f"[evidence] deflection generator import failed: {grav_d_err}", 'WARN')
            # Energy (Tier 3)
            try:
                from utils.evidence_generators.tier3_energy import generate_energy_evidence
            except Exception as energy_err:  # pragma: no cover
                generate_energy_evidence = None  # type: ignore
                _safe_log(log_fn, f"[evidence] energy generator import failed: {energy_err}", 'WARN')
            # Quantization (tunneling)
            try:
                from utils.evidence_generators.tier4_quantization import generate_tunneling_evidence
            except Exception as quan_err:  # pragma: no cover
                generate_tunneling_evidence = None  # type: ignore
                _safe_log(log_fn, f"[evidence] tunneling generator import failed: {quan_err}", 'WARN')
            # Quantization (bound states QUAN-10)
            try:
                from utils.evidence_generators.tier4_bound_states import generate_bound_states_evidence
            except Exception as bs_err:  # pragma: no cover
                generate_bound_states_evidence = None  # type: ignore
                _safe_log(log_fn, f"[evidence] bound states generator import failed: {bs_err}", 'WARN')
            # Coupling (Tier 6)
            try:
                from utils.evidence_generators.tier6_coupling import generate_coupling_evidence
            except Exception as coup_err:  # pragma: no cover
                generate_coupling_evidence = None  # type: ignore
                _safe_log(log_fn, f"[evidence] coupling generator import failed: {coup_err}", 'WARN')
            # Thermodynamics (Tier 7)
            try:
                from utils.evidence_generators.tier7_thermodynamics import generate_thermodynamics_evidence
            except Exception as therm_err:  # pragma: no cover
                generate_thermodynamics_evidence = None  # type: ignore
                _safe_log(log_fn, f"[evidence] thermodynamics generator import failed: {therm_err}", 'WARN')
            
            # Dispatch based on test type
            if test_id.upper() in ("REL-01", "REL-02", "REL-09", "REL-10"):
                created_any = generate_isotropy_evidence(results_dir, log_fn) or created_any
            elif test_id.upper() in ("REL-11", "REL-12", "REL-13", "REL-14"):
                created_any = generate_dispersion_evidence(results_dir, log_fn) or created_any
            elif test_id.upper() in ("REL-03", "REL-04", "REL-17"):
                created_any = generate_boost_evidence(results_dir, log_fn) or created_any
            elif test_id.upper().startswith("EM-") and generate_electromagnetic_evidence:
                created_any = generate_electromagnetic_evidence(results_dir, log_fn) or created_any
            elif test_id.upper() == "GRAV-12" and generate_phase_delay_evidence:
                created_any = generate_phase_delay_evidence(results_dir, log_fn) or created_any
            elif test_id.upper() == "GRAV-23" and generate_dynamic_chi_evidence:
                created_any = generate_dynamic_chi_evidence(results_dir, log_fn) or created_any
            elif test_id.upper() == "GRAV-13" and generate_redshift_evidence:
                created_any = generate_redshift_evidence(results_dir, log_fn) or created_any
            elif test_id.upper() == "GRAV-25" and generate_deflection_evidence:
                created_any = generate_deflection_evidence(results_dir, log_fn) or created_any
            elif test_id.upper().startswith("ENER-") and generate_energy_evidence:
                created_any = generate_energy_evidence(results_dir, log_fn) or created_any
            elif test_id.upper() == "QUAN-12" and generate_tunneling_evidence:
                created_any = generate_tunneling_evidence(results_dir, log_fn) or created_any
            elif test_id.upper() == "QUAN-10" and generate_bound_states_evidence:
                created_any = generate_bound_states_evidence(results_dir, log_fn) or created_any
            elif test_id.upper().startswith("COUP-") and generate_coupling_evidence:
                created_any = generate_coupling_evidence(results_dir, log_fn) or created_any
            elif test_id.upper().startswith("THERM-") and generate_thermodynamics_evidence:
                created_any = generate_thermodynamics_evidence(results_dir, log_fn) or created_any
        except Exception as e:
            _safe_log(log_fn, f"[evidence] Domain-specific generation skipped: {e}", 'WARN')

        # Fallback: generic time series plots from CSVs
        created_any = _maybe_plot_time_series(diag_dir, plot_dir, log_fn) or created_any

        # Write artifact manifest (world-class reproducibility)
        try:
            _write_artifact_manifest(results_dir, test_id, log_fn)
        except Exception as e:
            _safe_log(log_fn, f"[evidence] manifest write failed: {type(e).__name__}: {e}", 'WARN')

        # Sanity report: basic checks on diagnostics CSVs (non-empty, variability)
        try:
            _write_sanity_report(results_dir, test_id, log_fn)
        except Exception as e:
            _safe_log(log_fn, f"[evidence] sanity report failed: {type(e).__name__}: {e}", 'WARN')

    except Exception as e:  # pragma: no cover
        _safe_log(log_fn, f"[evidence] emit_summary_artifacts failed: {type(e).__name__}: {e}", 'WARN')
    return created_any


def _maybe_plot_time_series(diag_dir: Path, plot_dir: Path, log_fn: Optional[Callable[[str,str], None]]) -> bool:
    """Render small time-series plots from common CSVs.

    Primary sources live under diagnostics/. For backward compatibility,
    also look at the test root for legacy files (e.g., Tier 4 energy CSV).
    """
    made = False

    # Prefer diagnostics/, fall back to test root for legacy writers
    test_root = diag_dir.parent

    # projection_series.csv -> projection_series.png
    proj_csv = diag_dir / 'projection_series.csv'
    if not proj_csv.exists():
        legacy_proj = test_root / 'projection_series.csv'
        if legacy_proj.exists():
            proj_csv = legacy_proj
    if proj_csv.exists():
        try:
            t_vals, y_vals = _read_two_column_csv(proj_csv)
            _plot_xy(t_vals, y_vals, plot_dir / 'projection_series.png', 't', 'projection', 'Projection vs Time')
            made = True
        except Exception as e:
            _safe_log(log_fn, f"[evidence] projection_series plot skipped: {type(e).__name__}: {e}", 'WARN')

    # energy_evolution.csv -> energy_evolution.png (2nd column is plotted)
    energy_csv = diag_dir / 'energy_evolution.csv'
    if not energy_csv.exists():
        legacy_energy = test_root / 'energy_evolution.csv'
        if legacy_energy.exists():
            energy_csv = legacy_energy
    if energy_csv.exists():
        try:
            t_vals, y_vals = _read_two_column_csv(energy_csv)
            _plot_xy(t_vals, y_vals, plot_dir / 'energy_evolution.png', 't', 'E', 'Energy Evolution')
            made = True
        except Exception as e:
            _safe_log(log_fn, f"[evidence] energy_evolution plot skipped: {type(e).__name__}: {e}", 'WARN')
    return made


def _read_two_column_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Expect header then two columns
    lines = path.read_text(encoding='utf-8').strip().splitlines()
    rows = [ln for ln in lines if ln and not ln.lower().startswith('time,') and not ln.lower().startswith('t,')]
    t_vals = []
    y_vals = []
    for ln in rows:
        parts = [p.strip() for p in ln.split(',')]
        if len(parts) >= 2:
            try:
                t_vals.append(float(parts[0]))
                y_vals.append(float(parts[1]))
            except Exception:
                continue
    return np.array(t_vals, dtype=np.float64), np.array(y_vals, dtype=np.float64)


def _plot_xy(x: np.ndarray, y: np.ndarray, out_path: Path, xlabel: str, ylabel: str, title: str) -> None:
    if plt is None:  # pragma: no cover
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=120)
    ax.plot(x, y, 'b-')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _write_artifact_manifest(results_dir: Path, test_id: str, log_fn: Optional[Callable[[str,str], None]]) -> None:
    """Generate a machine-readable manifest of all artifacts for a test.

    The manifest improves reproducibility, downstream packaging, and integrity
    verification. Each entry includes size and SHA256 for tamper detection.

    Format (artifacts_manifest.json):
    {
      "test_id": "REL-01",
      "timestamp_utc": "2025-11-14T21:42:03Z",
      "artifacts": [
         {"name": "summary.json", "rel_path": "summary.json", "bytes": 1234, "sha256": "...", "category": "metadata"},
         {"name": "summary_metrics.csv", "rel_path": "diagnostics/summary_metrics.csv", "bytes": 123, "sha256": "...", "category": "diagnostic"},
         {"name": "isotropy_comparison.png", "rel_path": "plots/isotropy_comparison.png", "bytes": 45678, "sha256": "...", "category": "plot"}
      ]
    }
    """
    manifest_path = results_dir / 'artifacts_manifest.json'
    # Collect diagnostics and plots plus core summary
    categories = [
        ('metadata', [results_dir / 'summary.json']),
        ('diagnostic', list((results_dir / 'diagnostics').glob('*')) if (results_dir / 'diagnostics').exists() else []),
        ('plot', list((results_dir / 'plots').glob('*')) if (results_dir / 'plots').exists() else []),
    ]
    schema = get_schema_for_test(test_id)
    required_names = set()
    optional_names = set()
    if schema is not None:
        required_names.update(a.name for a in schema.required_diagnostics + schema.required_plots if a.required)
        optional_names.update(a.name for a in schema.optional_artifacts if not a.required)
    artifacts = []
    for category, paths in categories:
        for p in paths:
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
                sha = _sha256_file(p)
                # Derive logical artifact name key for role classification
                logical_name = p.name.replace('.png','').replace('.csv','').replace('.gif','')
                role = 'supplemental'
                if logical_name in required_names:
                    role = 'required'
                elif logical_name in optional_names:
                    role = 'optional'
                artifacts.append({
                    'name': p.name,
                    'rel_path': str(p.relative_to(results_dir)),
                    'bytes': size,
                    'sha256': sha,
                    'category': category,
                    'role': role
                })
            except Exception as e:  # pragma: no cover
                _safe_log(log_fn, f"[evidence] manifest skip {p.name}: {type(e).__name__}: {e}", 'WARN')
                continue

    manifest = {
        'test_id': test_id,
        'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'artifact_count': len(artifacts),
        'artifacts': artifacts,
        'version': '1.0'
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    _safe_log(log_fn, f"[evidence] manifest written: {manifest_path.name} ({len(artifacts)} artifacts)")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:  # binary read OK
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _write_sanity_report(results_dir: Path, test_id: str, log_fn: Optional[Callable[[str,str], None]]) -> None:
    """Generate diagnostics/sanity_report.csv validating basic numeric health.

    Checks:
      - File exists & non-zero size
      - For numeric CSVs: more than one unique value
      - No NaN in parsed numeric columns (first two columns sampled)
    """
    diag_dir = results_dir / 'diagnostics'
    if not diag_dir.exists():
        return
    rows = ['file,status,detail']
    for path in sorted(diag_dir.glob('*.csv')):
        try:
            size = path.stat().st_size
            if size == 0:
                rows.append(f'{path.name},FAIL,empty file')
                continue
            content = path.read_text(encoding='utf-8').strip().splitlines()
            if len(content) <= 1:
                rows.append(f'{path.name},WARN,header only')
                continue
            # Parse first two numeric columns if possible
            header = [h.strip() for h in content[0].split(',')]
            vals1 = []
            vals2 = []
            for ln in content[1:]:
                parts = [p.strip() for p in ln.split(',')]
                if len(parts) < 2:
                    continue
                try:
                    v1 = float(parts[0])
                    v2 = float(parts[1])
                except Exception:
                    continue
                vals1.append(v1); vals2.append(v2)
            if not vals1:
                rows.append(f'{path.name},WARN,no numeric rows')
                continue
            import math
            if any(math.isnan(v) for v in vals1+vals2):
                rows.append(f'{path.name},FAIL,NaN detected')
                continue
            if len(set(vals2)) <= 1:
                rows.append(f'{path.name},WARN,constant second column')
            else:
                rows.append(f'{path.name},PASS,ok')
        except Exception as e:  # pragma: no cover
            rows.append(f'{path.name},ERROR,{type(e).__name__}: {e}')
    (diag_dir / 'sanity_report.csv').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    _safe_log(log_fn, f"[evidence] sanity report written ({len(rows)-1} files)")
