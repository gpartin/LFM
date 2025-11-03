#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Backward-compatible shim for visualize_grav15_3d.

This script is preserved for convenience. The implementation now lives under:
  tools/visualize/visualize_grav15_3d.py
"""

import sys
import runpy
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve()
    target = here.parent / "tools" / "visualize" / here.name
    if not target.exists():
        sys.stderr.write(f"Error: relocated visualizer not found at {target}\n")
        sys.exit(1)
    runpy.run_path(str(target), run_name="__main__")
