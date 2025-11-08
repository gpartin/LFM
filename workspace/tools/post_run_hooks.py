#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Shared post-run hooks for validators and upload package staging.

Use these helpers from tier runners and the parallel orchestrator to avoid duplication.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional


def run_validation(scope: str, *, tiers: Optional[Iterable[int]] = None, strict: bool = False, quiet: bool = False) -> int:
    """Run validation based on scope.

    scope: 'all' | 'tiers' | 'tier'
    tiers: iterable of tier numbers to validate when scope is 'tiers' or 'tier'
    returns exit code from validator.report()
    """
    try:
        from tools.validate_results_pipeline import PipelineValidator  # type: ignore
    except Exception:
        return 1

    v = PipelineValidator(strict=strict, verbose=not quiet)
    ok = True
    if scope == 'all':
        ok = v.validate_end_to_end()
    elif scope in ('tiers', 'tier'):
        tiers_list = list(tiers or [])
        for t in tiers_list:
            ok = v.validate_tier_results(t) and ok
        ok = v.validate_master_status_integrity() and ok
    else:
        # unknown scope
        ok = False
    # Always include resource metrics check if we can
    try:
        ok = v.validate_resource_metrics() and ok
    except Exception:
        pass
    code = v.report()
    # Normalize return based on ok if report returned 0 but ok is False
    return code if code != 0 else (0 if ok else 1)


def rebuild_upload(*, deterministic: bool = False) -> bool:
    """Rebuild docs/upload package with deterministic options as desired.

    Returns True on (best-effort) success.
    """
    try:
        from tools import build_upload_package as bup  # type: ignore
    except Exception:
        return False

    try:
        bup.refresh_results_artifacts(deterministic=deterministic, build_master=False)
        bup.stage_evidence_docx(include=True)
        bup.export_txt_from_evidence(include=True)
        bup.export_md_from_evidence()
        bup.stage_result_plots(limit_per_dir=6)
        bup.generate_comprehensive_pdf()
        entries = bup.stage_and_list_files()
        zip_rel, _size, _sha = bup.create_zip_bundle(entries, label=None, deterministic=deterministic)
        entries_with_zip = entries + [(zip_rel, (bup.UPLOAD / zip_rel).stat().st_size, bup.sha256_file(bup.UPLOAD / zip_rel))]
        bup.write_manifest(entries_with_zip, deterministic=deterministic)
        bup.write_zenodo_metadata(entries_with_zip, deterministic=deterministic)
        bup.write_osf_metadata(entries_with_zip)
        return True
    except Exception:
        return False
