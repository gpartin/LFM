#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
Path utilities for determining repository and workspace roots without hardcoded drives.

Usage:
    from utils.path_utils import get_repo_root, get_workspace_dir, add_workspace_to_sys_path

Contract:
- get_repo_root(start) ascends from 'start' (file path or __file__) until it finds
  a directory that contains both 'build/metadata/structure.yaml' and a 'workspace' folder.
- get_workspace_dir(start) returns repo_root / 'workspace'.
- add_workspace_to_sys_path(start) prepends the workspace path to sys.path if missing.

Edge cases handled:
- Called from anywhere inside the repo tree.
- If markers not found (e.g., running from an extracted subfolder), falls back to
  walking up to the drive root and returns the nearest ancestor that contains 'workspace'.
"""
from __future__ import annotations
from pathlib import Path
import sys
from typing import Optional

MARKER_REL = Path('build/metadata/structure.yaml')
WORKSPACE_NAME = 'workspace'

def _has_markers(dirpath: Path) -> bool:
    return (dirpath / MARKER_REL).exists() and (dirpath / WORKSPACE_NAME).exists()

def get_repo_root(start: Optional[str|Path] = None) -> Path:
    p = Path(start if start is not None else __file__).resolve()
    # If 'start' is a file, use its parent
    if p.is_file():
        p = p.parent
    # Walk up to root
    for parent in [p] + list(p.parents):
        try:
            if _has_markers(parent):
                return parent
        except Exception:
            continue
    # Fallback: nearest ancestor containing 'workspace'
    for parent in [p] + list(p.parents):
        if (parent / WORKSPACE_NAME).exists():
            return parent
    # Last resort: two levels up
    return Path(start if start is not None else __file__).resolve().parents[2]

def get_workspace_dir(start: Optional[str|Path] = None) -> Path:
    return get_repo_root(start) / WORKSPACE_NAME

def add_workspace_to_sys_path(start: Optional[str|Path] = None) -> Path:
    ws = get_workspace_dir(start)
    ws_str = str(ws)
    if ws_str not in sys.path:
        sys.path.insert(0, ws_str)
    return ws
