# -*- coding: utf-8 -*-
"""
Unit tests for build_upload_package: ensure stale parallel_* files are removed
and not included in uploads when absent from source results.
"""
from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
import shutil

# Ensure repository workspace root is on sys.path so we can import tools.build_upload_package
import sys
_THIS_FILE = Path(__file__).resolve()
_WORKSPACE_ROOT = _THIS_FILE.parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

# Module under test (fully-qualified import now resolves)
from tools.build_upload_package import _copy_results_summaries_from_to


class TestUploadResultsCleanup(unittest.TestCase):
    def setUp(self) -> None:
        # Create temporary source and destination roots
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.src_root = Path(self.tmpdir.name) / "results_src"
        self.dst_root = Path(self.tmpdir.name) / "dest"
        self.src_root.mkdir(parents=True, exist_ok=True)
        self.dst_root.mkdir(parents=True, exist_ok=True)

        # Populate source with minimal canonical files (no parallel_* files)
        (self.src_root / "MASTER_TEST_STATUS.csv").write_text(
            "Test_ID,Description,Status\nREL-01,Test,PASS\n",
            encoding="utf-8",
        )
        (self.src_root / "README.md").write_text("Results readme", encoding="utf-8")

        # Create one category/test with summary
        cat = self.src_root / "Relativistic"
        test = cat / "REL-01"
        (test).mkdir(parents=True, exist_ok=True)
        (cat / "README.md").write_text("Category readme", encoding="utf-8")
        (test / "summary.json").write_text("{}", encoding="utf-8")
        (test / "readme.txt").write_text("ok", encoding="utf-8")

        # Pre-seed destination with stale files that must be removed
        dst_results = self.dst_root / "results"
        dst_results.mkdir(parents=True, exist_ok=True)
        (dst_results / "parallel_run_summary.json").write_text("{}", encoding="utf-8")
        (dst_results / "parallel_test_results.json").write_text("{}", encoding="utf-8")

    def test_stale_parallel_files_are_removed(self):
        copied = _copy_results_summaries_from_to(self.src_root, self.dst_root, include_full=False)

        # Ensure top-level allowlist files copied
        self.assertIn("results/MASTER_TEST_STATUS.csv", copied)
        self.assertIn("results/README.md", copied)

        # Ensure stale parallel files removed and not reintroduced
        dst_results = self.dst_root / "results"
        self.assertFalse((dst_results / "parallel_run_summary.json").exists())
        self.assertFalse((dst_results / "parallel_test_results.json").exists())

        # Ensure per-test essential files copied
        self.assertTrue((dst_results / "Relativistic" / "REL-01" / "summary.json").exists())
        self.assertTrue((dst_results / "Relativistic" / "REL-01" / "readme.txt").exists())


if __name__ == "__main__":
    unittest.main()
