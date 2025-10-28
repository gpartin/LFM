#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LFM Tier-2 — Gravity Analogue Suite (PASS-READY Build)
------------------------------------------------------
- FFT estimator upgraded (parabolic interpolation + DC skip)
- Phase bias in gaussian_packet to break symmetry
- CuPy/NumPy safe
- JSON-safe summaries
- Sparse diagnostics & buffered energy monitor
"""

import json, math, time
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import numpy as np

# Optional CuPy
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

from lfm_console import log, suite_summary
from lfm_logger import LFMLogger
from lfm_results import save_summary, write_metadata_bundle
from lfm_diagnostics import energy_total
from lfm_visualizer import visualize_concept
from numeric_integrity import NumericIntegrityMixin
from energy_monitor import EnergyMonitor
from lfm_equation import advance
from lfm_parallel import run_lattice

# ----------------------------- Config -----------------------------
def _default_config_name() -> str:
    return "config_tier2_gravityanalogue.json"

def load_config() -> Dict:
    script_dir = Path(__file__).resolve().parent
    for root in (script_dir, script_dir.parent):
        cand = root / "config" / _default_config_name()
        if cand.is_file():
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("Tier-2 config not found (expected config/config_tier2_gravityanalogue.json).")

def resolve_outdir(output_dir_hint: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    outdir = project_root / output_dir_hint
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

# --------------------------- Utilities ---------------------------
def pick_backend(use_gpu_flag: bool):
    on_gpu = bool(use_gpu_flag and _HAS_CUPY)
    return (cp if on_gpu else np), on_gpu

def to_numpy(x):
    if _HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)

def scalar_fast(v):
    try:
        return float(v.item())
    except Exception:
        return float(v)

def hann_fft_freq(series: List[float], dt: float) -> float:
    """Hann-windowed FFT with DC skip and sub-bin interpolation."""
    x = np.asarray(series, float)
    x = x - x.mean()
    if len(x) < 8:
        return 0.0
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), dt)
    mag = np.abs(X)
    k = int(np.argmax(mag[1:])) + 1
    if k <= 0 or k >= len(mag) - 1:
        return 2 * math.pi * f[k]
    y1, y2, y3 = np.log(mag[k-1] + 1e-30), np.log(mag[k] + 1e-30), np.log(mag[k+1] + 1e-30)
    denom = y1 - 2*y2 + y3
    delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (y1 - y3) / denom
    k_refined = k + np.clip(delta, -0.5, 0.5)
    return 2 * math.pi * np.interp(k_refined, np.arange(len(f)), f)

@dataclass
class VariantResult:
    test_id: str
    description: str
    passed: bool
    rel_err_ratio: float
    ratio_meas_serial: float
    ratio_meas_parallel: float
    ratio_theory: float
    runtime_sec: float
    on_gpu: bool

# ----------------------------- Builders -----------------------------
def build_chi_field(kind: str, N: int, dx: float, params: Dict, xp):
    if kind == "linear":
        g = float(params.get("chi_grad", 0.0))
        chi0 = float(params.get("chi_base", 0.0))
        ax = xp.arange(N, dtype=xp.float64)
        xmid = (N - 1) / 2.0
        chi_1d = chi0 + g * (ax - xmid) * dx
        return xp.broadcast_to(chi_1d, (N, N, N))
    chi0 = float(params.get("chi0", params.get("chi_delta", 0.25)))
    sigma = float(params.get("sigma", params.get("chi_width", 18.0)))
    c = (N - 1) / 2.0
    ax = xp.arange(N, dtype=xp.float64)
    g1 = xp.exp(-((ax - c) ** 2) / (2.0 * sigma * sigma))
    g2 = g1[:, xp.newaxis] * g1[xp.newaxis, :]
    g3 = g2[:, :, xp.newaxis] * g1[xp.newaxis, xp.newaxis, :]
    return chi0 * g3

def gaussian_packet(N: int, kvec, amplitude: float, env_sigma_cells: float, xp):
    c = (N - 1) / 2.0
    ax = xp.arange(N, dtype=xp.float64)
    g1 = xp.exp(-((ax - c) ** 2) / (2.0 * env_sigma_cells * env_sigma_cells))
    env3d = (g1[:, xp.newaxis] * g1[xp.newaxis, :])[:, :, xp.newaxis] * g1[xp.newaxis, xp.newaxis, :]
    # Add phase bias to avoid symmetry collapse
    sin_phase = xp.sin(kvec[0] * ax + 0.5)[xp.newaxis, xp.newaxis, :]
    return amplitude * env3d * sin_phase

def local_omega_theory(c: float, k_mag: float, chi_val: float) -> float:
    return math.sqrt((c * c) * (k_mag * k_mag) + chi_val * chi_val)

# ----------------------------- Harness -----------------------------
class Tier2Harness(NumericIntegrityMixin):
    def __init__(self, cfg: Dict, out_root: Path):
        self.cfg = cfg
        self.run_settings = cfg["run_settings"]
        self.base = cfg["parameters"]
        self.tol = cfg["tolerances"]
        self.variants = cfg["variants"]
        self.quick = bool(self.run_settings.get("quick_mode", False))
        self.out_root = out_root
        self.logger = LFMLogger(self.out_root)
        self.logger.record_env()
        self.xp, self.on_gpu = pick_backend(self.run_settings.get("use_gpu", False))
        self.dtype = self.xp.float32 if self.quick else self.xp.float64

    def run_variant(self, v: Dict) -> VariantResult:
        xp = self.xp
        tid, desc = v["test_id"], v.get("description", v["test_id"])
        p = {**self.base, **v}
        N = int(p.get("grid_points", 64))
        dx, dt = float(p["dx"]), float(p["dt"])
        alpha, beta = float(p["alpha"]), float(p["beta"])
        steps = int(p.get("steps_quick" if self.quick else "steps", 600))
        amplitude = float(p.get("packet_amp", 1e-2))
        env_sigma = float(p.get("packet_width_cells", 18))
        k_fraction = float(p.get("k_fraction", 2.0 / N))
        use_linear = (p.get("chi_profile", "gaussian") == "linear")
        tiles3 = tuple(p.get("tiles3", (2, 2, 2)))
        if tiles3 == (2, 2, 2) and N >= 64:
            tiles3 = (4, 4, 2)

        c = math.sqrt(alpha / beta)
        kvec = (k_fraction * math.pi / dx) * np.array([1.0, 0.0, 0.0], float)
        k_mag = float(np.linalg.norm(kvec))

        test_dir = self.out_root / tid
        diag_dir, plot_dir = test_dir / "diagnostics", test_dir / "plots"
        for d in (test_dir, diag_dir, plot_dir):
            d.mkdir(parents=True, exist_ok=True)

        log(f"→ Starting {tid}: {desc} (N={N}³, steps={steps}, quick={self.quick})", "INFO")

        chi_field = build_chi_field("linear" if use_linear else "gaussian", N, dx, p, xp).astype(self.dtype, copy=False)
        E0 = gaussian_packet(N, kvec, amplitude, env_sigma, xp).astype(self.dtype, copy=False)
        center = (N // 2, N // 2, N // 2)
        chi_center = float(to_numpy(chi_field[center]))
        w_init = local_omega_theory(c, k_mag, chi_center)
        Eprev0 = (E0 * math.cos(dt * w_init)).astype(self.dtype, copy=False)

        self.check_cfl(c, dt, dx, ndim=3)
        params = dict(dt=dt, dx=dx, alpha=alpha, beta=beta, boundary="periodic",
                      chi=to_numpy(chi_field) if xp is np else chi_field)

        PROBE_A, PROBE_B = center, (N // 2, N // 2, int(0.70 * N))
        E0_energy = energy_total(to_numpy(E0), to_numpy(Eprev0), dt, dx, c, to_numpy(chi_field))

        # --- Serial ---
        series_A, series_B = [], []
        mon = EnergyMonitor(dt, dx, c, 0.0, outdir=str(diag_dir), label=f"{tid}_serial")
        E, Ep = E0.copy(), Eprev0.copy()
        t0 = time.time()
        for n in range(steps):
            E_next = advance(E, params, 1)
            Ep, E = E, E_next
            series_A.append(scalar_fast(E[PROBE_A]))
            series_B.append(scalar_fast(E[PROBE_B]))
            if (not self.quick) or (n % 25 == 0):
                mon.record(to_numpy(E), to_numpy(Ep), n)
        mon.finalize()
        t_serial = time.time() - t0

        # --- Parallel ---
        series_Ap, series_Bp = [], []
        E, Ep = E0.copy(), Eprev0.copy()
        t0 = time.time()
        for n in range(steps):
            E_next = run_lattice(E, params, 1, tiles=tiles3)
            Ep, E = E, E_next
            series_Ap.append(scalar_fast(E[PROBE_A]))
            series_Bp.append(scalar_fast(E[PROBE_B]))
        t_parallel = time.time() - t0

        # --- Analysis ---
        wA_s, wB_s = hann_fft_freq(series_A, dt), hann_fft_freq(series_B, dt)
        wA_p, wB_p = hann_fft_freq(series_Ap, dt), hann_fft_freq(series_Bp, dt)
        chiA, chiB = float(to_numpy(chi_field[PROBE_A])), float(to_numpy(chi_field[PROBE_B]))
        wA_th, wB_th = local_omega_theory(c, k_mag, chiA), local_omega_theory(c, k_mag, chiB)
        ratio_th = wA_th / max(wB_th, 1e-30)
        ratio_s = wA_s / max(wB_s, 1e-30)
        ratio_p = wA_p / max(wB_p, 1e-30)
        err = max(abs(ratio_s - ratio_th) / ratio_th, abs(ratio_p - ratio_th) / ratio_th)
        passed = err <= float(self.tol.get("ratio_error_max", 0.05))

        if not self.quick:
            try:
                visualize_concept([to_numpy(E0), to_numpy(E)],
                                  chi_series=[to_numpy(chi_field)],
                                  tier=2, test_id=tid, outdir=plot_dir,
                                  quick=self.quick, animate=False)
            except Exception as e:
                log(f"[WARN] Visualization failed for {tid}: {e}", "WARN")

        save_summary(test_dir, tid, {
            "id": str(tid), "description": str(desc), "passed": bool(passed),
            "rel_err_ratio": float(err),
            "omegaA_serial": float(wA_s), "omegaB_serial": float(wB_s),
            "omegaA_parallel": float(wA_p), "omegaB_parallel": float(wB_p),
            "omegaA_theory": float(wA_th), "omegaB_theory": float(wB_th),
            "ratio_serial": float(ratio_s), "ratio_parallel": float(ratio_p),
            "ratio_theory": float(ratio_th),
            "N": int(N), "dx": float(dx), "dt": float(dt),
            "alpha": float(alpha), "beta": float(beta), "c": float(c),
            "k_fraction": float(k_fraction), "on_gpu": bool(self.on_gpu),
            "quick_mode": bool(self.quick), "steps": int(steps)
        })

        log(f"{tid} {'PASS ✅' if passed else 'FAIL ❌'} "
            f"(ratio_err={err*100:.2f}% serial={ratio_s:.3f} parallel={ratio_p:.3f} th={ratio_th:.3f})",
            "INFO" if passed else "FAIL")

        return VariantResult(tid, desc, passed, err, ratio_s, ratio_p, ratio_th,
                             t_serial + t_parallel, self.on_gpu)

    def run(self) -> List[Dict]:
        return [self.run_variant(v).__dict__ for v in self.variants]

# ----------------------------- Main -----------------------------
def main():
    cfg = load_config()
    outdir = resolve_outdir(cfg["run_settings"].get("output_dir", "results/Gravity"))
    harness = Tier2Harness(cfg, outdir)
    log(f"[paths] OUTPUT ROOT = {outdir}", "INFO")
    log(f"=== Tier-2 Gravity Analogue Suite Start (quick={harness.quick}) ===", "INFO")
    results = harness.run()
    suite_summary(results)
    write_metadata_bundle(outdir, test_id="TIER2-GRAVITY", tier=2, category="Gravity")
    log("=== Tier-2 Suite Complete ===", "INFO")

if __name__ == "__main__":
    main()
