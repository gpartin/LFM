# -*- coding: utf-8 -*-
"""
Tests for results copying completeness into upload payloads.
Verifies that per-test JSON/CSV/TXT, plots (png/jpg/svg/gif/mp4), and diagnostics are copied.
"""
from pathlib import Path
import json

from tools.build_upload_package import _copy_results_summaries_from_to  # type: ignore


def _make_file(p: Path, content: str = "x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding='utf-8')


def test_copy_results_includes_metadata_plots_and_diagnostics(tmp_path: Path):
    # Create a minimal fake results tree
    src_root = tmp_path / "results_src"
    category = src_root / "Relativistic"
    testdir = category / "REL-01"

    # Essentials
    _make_file(src_root / "MASTER_TEST_STATUS.csv", "Test_ID,Status\nREL-01,PASS\n")
    _make_file(category / "README.md", "# Relativistic Results\n")
    _make_file(testdir / "summary.json", json.dumps({"description": "Test", "status": "PASS"}))
    _make_file(testdir / "readme.txt", "Readme")

    # Root-level metadata files
    _make_file(testdir / "metrics.json", json.dumps({"energy_drift": 1e-6}))
    _make_file(testdir / "energy.csv", "t,E\n0,1.0\n")
    _make_file(testdir / "notes.txt", "Some notes")

    # Plots with multiple formats
    _make_file(testdir / "plots" / "plot.png", "png")
    _make_file(testdir / "plots" / "plot.svg", "svg")
    _make_file(testdir / "plots" / "anim.mp4", "mp4")

    # Diagnostics recursive
    _make_file(testdir / "diagnostics" / "log.csv", "a,b\n1,2\n")
    _make_file(testdir / "diagnostics" / "diag.txt", "diag")
    _make_file(testdir / "diagnostics" / "sub" / "deep.csv", "c,d\n3,4\n")

    dest_root = tmp_path / "uploads_osf"

    copied = _copy_results_summaries_from_to(src_root, dest_root, include_full=True)

    # Assert essentials present
    assert (dest_root / "results" / "MASTER_TEST_STATUS.csv").exists()
    assert (dest_root / "results" / "Relativistic" / "README.md").exists()
    assert (dest_root / "results" / "Relativistic" / "REL-01" / "summary.json").exists()
    assert (dest_root / "results" / "Relativistic" / "REL-01" / "readme.txt").exists()

    # Root-level metadata files copied
    assert (dest_root / "results" / "Relativistic" / "REL-01" / "metrics.json").exists()
    assert (dest_root / "results" / "Relativistic" / "REL-01" / "energy.csv").exists()
    assert (dest_root / "results" / "Relativistic" / "REL-01" / "notes.txt").exists()

    # Plots copied (png, svg, mp4)
    assert (dest_root / "results" / "Relativistic" / "REL-01" / "plots" / "plot.png").exists()
    assert (dest_root / "results" / "Relativistic" / "REL-01" / "plots" / "plot.svg").exists()
    assert (dest_root / "results" / "Relativistic" / "REL-01" / "plots" / "anim.mp4").exists()

    # Diagnostics copied recursively
    assert (dest_root / "results" / "Relativistic" / "REL-01" / "diagnostics" / "log.csv").exists()
    assert (dest_root / "results" / "Relativistic" / "REL-01" / "diagnostics" / "diag.txt").exists()
    assert (dest_root / "results" / "Relativistic" / "REL-01" / "diagnostics" / "sub" / "deep.csv").exists()

    # Sanity: copied list contains some of these entries
    assert any("plots/plot.svg" in s for s in copied)
    assert any("diagnostics/sub/deep.csv" in s for s in copied)
