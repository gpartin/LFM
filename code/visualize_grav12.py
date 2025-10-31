#!/usr/bin/env python3
"""
Backward-compatible shim for visualize_grav12.

This script is preserved for convenience. The implementation now lives under:
  tools/visualize/visualize_grav12.py
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
