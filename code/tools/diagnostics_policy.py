#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Diagnostics policy enforcement for LFM test configurations.

This enforces minimal diagnostics required for troubleshooting per the repo policy:
- Time dilation: ensure diagnostics.save_time_series = true
- Time delay: ensure diagnostics.track_packet = true, diagnostics.log_packet_stride set
              enable energy monitoring and debug diagnostics at reasonable cadence

Behavior: Non-destructive and conservative. It updates the in-memory cfg dict (no file I/O).
It returns the modified cfg and a list of human-readable notes describing changes.
"""
from __future__ import annotations
from typing import Dict, List, Tuple


def enforce_for_cfg(cfg: Dict) -> Tuple[Dict, List[str]]:
    notes: List[str] = []
    if not isinstance(cfg, dict):
        return cfg, notes

    tests = cfg.get('tests') or cfg.get('variants') or []
    diags = cfg.setdefault('diagnostics', {}) or {}
    debug = cfg.setdefault('debug', {}) or cfg.get('run_settings', {}).get('debug', {}) or {}

    # Identify if any test requires specific diagnostics
    has_time_delay = any(str(t.get('mode', '')) == 'time_delay' for t in tests if isinstance(t, dict))
    has_time_dilation = any(str(t.get('mode', '')) == 'time_dilation' for t in tests if isinstance(t, dict))

    # Time delay requirements
    if has_time_delay:
        if not bool(diags.get('track_packet', False)):
            diags['track_packet'] = True
            notes.append('Enabled diagnostics.track_packet for time_delay tests')
        # default stride if missing
        if int(diags.get('log_packet_stride', 0) or 0) <= 0:
            diags['log_packet_stride'] = 100
            notes.append('Set diagnostics.log_packet_stride=100')
        # energy monitor cadence
        if int(diags.get('energy_monitor_every', 0) or 0) <= 0:
            diags['energy_monitor_every'] = 100
            notes.append('Set diagnostics.energy_monitor_every=100')
        # enable debug diagnostics
        if not bool(debug.get('enable_diagnostics', False)):
            debug['enable_diagnostics'] = True
            notes.append('Enabled debug.enable_diagnostics for propagation diagnostics')

    # Time dilation requirements
    if has_time_dilation:
        if not bool(diags.get('save_time_series', False)):
            diags['save_time_series'] = True
            notes.append('Enabled diagnostics.save_time_series for time_dilation tests')
        if not bool(debug.get('enable_diagnostics', False)):
            debug['enable_diagnostics'] = True
            notes.append('Enabled debug.enable_diagnostics for FFT validation')

    # Prefer verbose logging when enforcing diagnostics for visibility
    run_settings = cfg.setdefault('run_settings', {})
    if (has_time_delay or has_time_dilation) and not bool(run_settings.get('verbose', False)):
        run_settings['verbose'] = True
        notes.append('Enabled run_settings.verbose for detailed per-step logging')
    # Avoid quiet runs when diagnosing
    dbg = run_settings.setdefault('debug', {})
    if (has_time_delay or has_time_dilation) and bool(dbg.get('quiet_run', False)):
        dbg['quiet_run'] = False
        notes.append('Set debug.quiet_run=False for better visibility')

    # Reassign updated sections
    cfg['diagnostics'] = diags
    cfg['run_settings'] = run_settings
    # Prefer keeping debug under run_settings.debug if it existed
    run_settings['debug'] = dbg if dbg else run_settings.get('debug', {})

    return cfg, notes
