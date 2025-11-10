# -*- coding: utf-8 -*-
"""Core harness unit tests (lightweight, fast).

Focus:
- Config loader search paths
- Frequency estimator accuracy on synthetic signal
- Hann window length correctness
- Backend selection (GPU flag)
"""
from pathlib import Path
import numpy as np

from harness.lfm_test_harness import BaseTierHarness
from harness.validation import (
    evaluate_isotropy,
    evaluate_directional_isotropy,
    evaluate_dispersion,
    evaluate_momentum_drift,
    evaluate_invariant_mass,
)


def _make_sine(dt: float, omega: float, n: int) -> np.ndarray:
    t = np.arange(n) * dt
    return np.sin(omega * t)


def test_hann_window_length():
    from harness.lfm_test_harness import BaseTierHarness
    w = BaseTierHarness.hann_window(128)
    assert len(w) == 128
    assert np.isclose(w[0], 0.0)
    assert np.isclose(w[-1], 0.0)


def test_frequency_estimator_parabolic():
    from harness.lfm_test_harness import BaseTierHarness
    dt = 0.001
    omega_true = 2.0 * np.pi * 50.0  # 50 Hz in angular units
    data = _make_sine(dt, omega_true, 4096)
    cfg = {"run_settings": {"use_gpu": False}, "parameters": {}, "tolerances": {}}
    est = BaseTierHarness(cfg, Path("."), "config.json").estimate_omega_fft(data, dt, method="parabolic")
    rel_err = abs(est - omega_true) / omega_true
    assert rel_err < 0.01, f"Frequency estimator error too large: {rel_err:.4f}"


def test_backend_selection_gpu_flag(monkeypatch):
    from harness.lfm_test_harness import BaseTierHarness
    # Force use_gpu True in cfg
    cfg = {"run_settings": {"use_gpu": True}, "parameters": {}, "tolerances": {}}
    h = BaseTierHarness(cfg, Path("./tmp_out"), "config.json")
    # xp should be cupy when available; accept numpy fallback but assert flag presence
    assert hasattr(h, 'xp')
    assert isinstance(h.use_gpu, bool)


def test_config_loader_search(tmp_path):
    # Create dummy config and load via explicit path
    cfg_file = tmp_path / "dummy.json"
    cfg_file.write_text("{\n\"run_settings\": {\"quick_mode\": true}\n}", encoding="utf-8")
    import harness.lfm_test_harness as th
    cfg = th.BaseTierHarness.load_config(str(cfg_file), default_config_name="dummy.json")
    assert cfg.get("run_settings", {}).get("quick_mode") is True


def test_evaluate_isotropy_helper():
    # Fake metadata with threshold 0.01
    meta = {"tests": {"REL-01": {"validation_criteria": {"primary": {"metric": "anisotropy", "threshold": 0.01}}}}}
    ok, key, val, thr = evaluate_isotropy(meta, "REL-01", 0.005)
    assert ok and key == "anisotropy" and thr == 0.01
    ok2, _, val2, _ = evaluate_isotropy(meta, "REL-01", 0.02)
    assert not ok2 and val2 == 0.02


def test_evaluate_dispersion_helper():
    meta = {"tests": {"REL-11": {"validation_criteria": {"primary": {"metric": "dispersion_error", "threshold": 0.05}}}}}
    ok, key, val, thr = evaluate_dispersion(meta, "REL-11", 0.049)
    assert ok and key in ("rel_err", "dispersion_error")
    ok2, _, val2, _ = evaluate_dispersion(meta, "REL-11", 0.051)
    assert not ok2 and val2 == 0.051


def test_evaluate_momentum_drift_helper():
    meta = {"tests": {"REL-16": {"validation_criteria": {"primary": {"metric": "momentum_drift", "threshold": 0.01}}}}}
    ok, key, val, thr = evaluate_momentum_drift(meta, "REL-16", 0.005)
    assert ok and key == "momentum_drift" and thr == 0.01
    ok2, _, val2, _ = evaluate_momentum_drift(meta, "REL-16", 0.02)
    assert not ok2 and val2 == 0.02


def test_evaluate_invariant_mass_helper():
    meta = {"tests": {"REL-17": {"validation_criteria": {"primary": {"metric": "invariant_mass_error", "threshold": 0.01}}}}}
    ok, key, val, thr = evaluate_invariant_mass(meta, "REL-17", 0.003)
    assert ok and key == "invariant_mass_error"
    ok2, _, val2, _ = evaluate_invariant_mass(meta, "REL-17", 0.02)
    assert not ok2 and val2 == 0.02
