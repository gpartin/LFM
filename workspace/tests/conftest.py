"""
Pytest configuration for test discovery and path setup.

This file ensures that:
1. The src/ directory is added to sys.path for imports
2. Tests can import modules like: from lfm_simulator import LFMSimulator
3. Path resolution works regardless of where pytest is invoked
"""
import sys
from pathlib import Path

# Add src to path for imports
TEST_ROOT = Path(__file__).parent
PROJECT_ROOT = TEST_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"

if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
