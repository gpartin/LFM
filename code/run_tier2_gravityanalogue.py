#!/usr/bin/env python3
"""
LFM Tier-2 — Gravity Analogue Suite
-----------------------------------
Purpose:
- Execute Tier-2 gravity-analogue tests (3D lattice runs) to validate
    dispersion and gravity-analogue behaviour, capture diagnostics, and produce
    standardized summaries.

Highlights:
- Optimized to minimize unnecessary CuPy→NumPy transfers and reduce log spam
    via throttling and structured events.
- Parallel and serial execution modes supported; configurable tiling for
    threaded parallel runs.

Config & output:
- Expects configuration at `./config/config_tier2_gravityanalogue.json`.
- Writes per-test results under `results/Gravity/<TEST_ID>/` with
    `summary.json`, `metrics.csv`, `diagnostics/` and `plots/`.
"""

import json, math, time, sys
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

from lfm_console import log, suite_summary, set_logger, log_run_config, report_progress
from lfm_logger import LFMLogger
from lfm_results import save_summary, write_metadata_bundle
from lfm_diagnostics import energy_total
from lfm_visualizer import visualize_concept
from numeric_integrity import NumericIntegrityMixin
from energy_monitor import EnergyMonitor
from lfm_equation import advance
from lfm_parallel import run_lattice

 
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
    x = np.asarray(series, float)
    x = x - x.mean()
    if len(x) < 16:
        return 0.0
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), dt)
    mag = np.abs(X)
    k = int(np.argmax(mag[1:])) + 1
    if k <= 0 or k >= len(mag) - 1:
        return 2 * math.pi * f[k]
    y1, y2, y3 = np.log(mag[k-1]+1e-30), np.log(mag[k]+1e-30), np.log(mag[k+1]+1e-30)
    denom = y1 - 2*y2 + y3
    delta = 0.0 if abs(denom) < 1e-12 else 0.5*(y1 - y3)/denom
    k_refined = k + np.clip(delta, -0.5, 0.5)
    return 2*math.pi*np.interp(k_refined, np.arange(len(f)), f)

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
    g1 = xp.exp(-((ax - c)**2)/(2.0*sigma*sigma))
    g2 = g1[:, xp.newaxis] * g1[xp.newaxis, :]
    g3 = g2[:, :, xp.newaxis] * g1[xp.newaxis, xp.newaxis, :]
    return chi0 * g3

def gaussian_packet(N, kvec, amplitude, width, xp):
    c = (N - 1) / 2.0
    ax = xp.arange(N, dtype=xp.float64)
    g1 = xp.exp(-((ax - c)**2)/(2.0*width*width))
    env3 = (g1[:, xp.newaxis]*g1[xp.newaxis,:])[:, :, xp.newaxis]*g1[xp.newaxis,xp.newaxis,:]
    sin_phase = xp.sin(kvec[0]*ax + 0.5)[xp.newaxis, xp.newaxis, :]
    return amplitude * env3 * sin_phase

def local_omega_theory(c, k_mag, chi):
    return math.sqrt((c*c)*(k_mag*k_mag) + chi*chi)


 
def _default_config_name() -> str:
    return "config_tier2_gravityanalogue.json"

def load_config(config_path: str = None) -> Dict:
    """Search for the Tier-2 config in script `config/` (current or parent).

    This mirrors the Tier-1 loader behavior so running from different CWDs
    works the same across harnesses.
    """
    if config_path:
        # Use explicit path if provided
        cand = Path(config_path)
        if cand.is_file():
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Default: search in standard locations
    script_dir = Path(__file__).resolve().parent
    for root in (script_dir, script_dir.parent):
        cand = root / "config" / _default_config_name()
        if cand.is_file():
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Tier-2 config not found (expected config/{_default_config_name()}).")

 
class Tier2Harness(NumericIntegrityMixin):
    def __init__(self, cfg: Dict, out_root: Path):
        self.cfg = cfg
        self.run_settings = cfg["run_settings"]
        self.base = cfg["parameters"]
        self.tol = cfg["tolerances"]
        self.variants = cfg["variants"]
        self.quick = bool(self.run_settings.get("quick_mode", False))
        self.verbose = bool(self.run_settings.get("verbose", False))
        self.monitor_stride = int(self.run_settings.get("monitor_stride_quick", 25))
        
        if "numeric_integrity" in self.run_settings:
            ni_cfg = self.run_settings["numeric_integrity"]
            self.energy_tol = float(ni_cfg.get("energy_tol", 1e-6))
            self.quiet_warnings = bool(ni_cfg.get("quiet_warnings", False))
            if "debug" not in self.run_settings:
                self.run_settings["debug"] = {}
            self.run_settings["debug"].update({
                "quiet_run": True,
                "print_probe_steps": False,
                "energy_tol": self.energy_tol
            })
            self.verbose_stride = int(self.run_settings.get("verbose_stride", 200))
            self.monitor_flush_interval = int(self.run_settings.get("monitor_flush_interval", 50))
            self.show_progress = bool(self.run_settings.get("show_progress", True))
            self.progress_percent_stride = int(self.run_settings.get("progress_percent_stride", 5))
            self.xp, self.on_gpu = pick_backend(self.run_settings.get("use_gpu", False))
            self.dtype = self.xp.float32 if self.quick else self.xp.float64
            self.out_root = out_root
            self.logger = LFMLogger(out_root)
            self.logger.record_env()
            try:
                set_logger(self.logger)
            except Exception:
                pass
            try:
                log_run_config(self.cfg, self.out_root)
            except Exception:
                pass
            try:
                from lfm_console import set_diagnostics_enabled
                dbg_cfg = self.run_settings.get("debug", {}) or {}
                set_diagnostics_enabled(bool(dbg_cfg.get("enable_diagnostics", False)))
            except Exception:
                pass
            log(f"[backend] on_gpu={self.on_gpu} (CuPy available={_HAS_CUPY})", "INFO")

    def run_variant(self, v: Dict) -> VariantResult:
        xp = self.xp
        tid = v["test_id"]
        desc = v.get("description", tid)
        p = {**self.base, **v}
        N = int(p.get("grid_points", 64))
        dx, dt = float(p["dx"]), float(p["dt"])
        alpha, beta = float(p["alpha"]), float(p["beta"])
        steps = int(p.get("steps_quick" if self.quick else "steps", 600))
        amplitude = float(p.get("packet_amp", 1e-2))
        width = float(p.get("packet_width_cells", 18))
        k_fraction = float(p.get("k_fraction", 2.0/N))
        tiles3 = tuple(p.get("tiles3", (2,2,2)))
        if tiles3 == (2,2,2) and N >= 64:
            tiles3 = (4,4,2)

        c = math.sqrt(alpha/beta)
        kvec = (k_fraction*math.pi/dx)*np.array([1.0,0.0,0.0],float)
        k_mag = float(np.linalg.norm(kvec))
        center = (N//2, N//2, N//2)

        test_dir = self.out_root / tid
        diag_dir, plot_dir = test_dir / "diagnostics", test_dir / "plots"
        for d in (test_dir, diag_dir, plot_dir):
            d.mkdir(parents=True, exist_ok=True)
        from lfm_console import test_start
        test_start(tid, desc, steps)
        log(f"Params: N={N}³, steps={steps}, quick={self.quick}", "INFO")

        chi_field = build_chi_field(p.get("chi_profile","linear"), N, dx, p, xp).astype(self.dtype)
        E0 = gaussian_packet(N, kvec, amplitude, width, xp).astype(self.dtype)
        chi_center = float(to_numpy(chi_field[center]))
        w_init = local_omega_theory(c, k_mag, chi_center)
        Eprev0 = (E0*math.cos(dt*w_init)).astype(self.dtype)

        self.check_cfl(c, dt, dx, ndim=3)
        params = dict(dt=dt, dx=dx, alpha=alpha, beta=beta, boundary="periodic",
                      chi=to_numpy(chi_field) if xp is np else chi_field)
        if "debug" in self.run_settings:
            params.setdefault("debug", {})
            params["debug"].update(self.run_settings.get("debug", {}))
        if "numeric_integrity" in self.run_settings:
            params.setdefault("numeric_integrity", {})
            params["numeric_integrity"].update(self.run_settings.get("numeric_integrity", {}))

        PROBE_A, PROBE_B = center, (N//2, N//2, int(0.7 * N))

        series_A, series_B = [], []
        # Use consistent energy tolerance across all components
        energy_tol = float(self.run_settings.get("numeric_integrity", {}).get("energy_tol", 1e-3))
        mon = EnergyMonitor(dt, dx, c, 0.0,
                           outdir=str(diag_dir),
                           label=f"{tid}_serial",
                           threshold=energy_tol,
                           flush_interval=self.monitor_flush_interval)
        E, Ep = E0.copy(), Eprev0.copy()
        advance_local = advance
        mon_record = mon.record
        to_numpy_local = to_numpy
        scalar_fast_local = scalar_fast
        verbose_stride = self.verbose_stride
        monitor_stride_local = int(self.monitor_stride)
        t0 = time.time()
        # progress threshold (next percent to print)
        next_pct = self.progress_percent_stride if self.progress_percent_stride > 0 else 100
        # only compute pct at this cadence to avoid per-iteration division
        steps_pct_check = max(1, steps // 100)
        for n in range(steps):
            E_next = advance_local(E, params, 1)
            Ep, E = E, E_next
            # Throttle expensive host<->device transfers: convert once on monitor steps
            if (n % monitor_stride_local) == 0:
                host_E = to_numpy_local(E)
                host_Ep = to_numpy_local(Ep)
                mon_record(host_E, host_Ep, n)
                # extract probes from host copy (avoids additional device->host transfers)
                series_A.append(float(host_E[PROBE_A]))
                series_B.append(float(host_E[PROBE_B]))
            else:
                series_A.append(scalar_fast_local(E[PROBE_A]))
                series_B.append(scalar_fast_local(E[PROBE_B]))
            if self.verbose and (n % verbose_stride == 0):
                log(f"[{tid}/serial] step {n}/{steps}", "INFO")
            # percentage progress reporting (controlled by run_settings)
            if self.show_progress and (n % steps_pct_check == 0):
                pct = int((n + 1) * 100 / max(1, steps))
                if pct >= next_pct:
                    report_progress(tid, pct, phase="serial")
                    next_pct += self.progress_percent_stride
        mon.finalize()
        t_serial = time.time() - t0

        # Parallel
        series_Ap, series_Bp = [], []
        E, Ep = E0.copy(), Eprev0.copy()
        t0 = time.time()
        next_pct = self.progress_percent_stride if self.progress_percent_stride > 0 else 100
        run_lattice_local = run_lattice
        scalar_fast_local = scalar_fast
        # reuse the same steps_pct_check for parallel progress throttling
        for n in range(steps):
            E_next = run_lattice_local(E, params, 1, tiles=tiles3)
            Ep, E = E, E_next
            # When appropriate, convert device->host once and reuse for probes to
            # avoid two separate small transfers; otherwise use fast scalar path.
            if (n % monitor_stride_local) == 0:
                host_E = to_numpy_local(E)
                series_Ap.append(float(host_E[PROBE_A]))
                series_Bp.append(float(host_E[PROBE_B]))
            else:
                series_Ap.append(scalar_fast_local(E[PROBE_A]))
                series_Bp.append(scalar_fast_local(E[PROBE_B]))
            if self.verbose and (n % verbose_stride == 0):
                log(f"[{tid}/par] step {n}/{steps}", "INFO")
            # percentage progress reporting (controlled by run_settings)
            if self.show_progress and (n % steps_pct_check == 0):
                pct = int((n + 1) * 100 / max(1, steps))
                if pct >= next_pct:
                    report_progress(tid, pct, phase="parallel")
                    next_pct += self.progress_percent_stride
        t_parallel = time.time() - t0

        wA_s,wB_s = hann_fft_freq(series_A,dt),hann_fft_freq(series_B,dt)
        wA_p,wB_p = hann_fft_freq(series_Ap,dt),hann_fft_freq(series_Bp,dt)
        chiA,chiB = chi_center,float(to_numpy(chi_field[PROBE_B]))
        wA_th,wB_th = local_omega_theory(c,k_mag,chiA),local_omega_theory(c,k_mag,chiB)
        ratio_th = wA_th/max(wB_th,1e-30)
        ratio_s,ratio_p = wA_s/max(wB_s,1e-30),wA_p/max(wB_p,1e-30)
        err = max(abs(ratio_s-ratio_th)/ratio_th,abs(ratio_p-ratio_th)/ratio_th)
        passed = err <= float(self.tol.get("ratio_error_max",0.05))

        save_summary(test_dir,tid,{"id":tid,"description":desc,"passed":passed,
            "rel_err_ratio":float(err),"ratio_serial":float(ratio_s),
            "ratio_parallel":float(ratio_p),"ratio_theory":float(ratio_th),
            "omegaA_serial":float(wA_s),"omegaB_serial":float(wB_s),
            "omegaA_parallel":float(wA_p),"omegaB_parallel":float(wB_p),
            "omegaA_theory":float(wA_th),"omegaB_theory":float(wB_th),
            "N":int(N),"dx":float(dx),"dt":float(dt),"steps":int(steps)})

        log(f"{tid} {'PASS ✅' if passed else 'FAIL ❌'} "
            f"(ratio_err={err*100:.2f}%)","INFO" if passed else "FAIL")
        return VariantResult(tid,desc,passed,err,ratio_s,ratio_p,ratio_th,
                             t_serial+t_parallel,self.on_gpu)

    def run(self)->List[Dict]:
        return [self.run_variant(v).__dict__ for v in self.variants]

# --------------------------- Main ---------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tier-2 Gravity Analogue Test Suite")
    parser.add_argument("--test", type=str, default=None,
                       help="Run single test by ID (e.g., GRAV-01). If omitted, runs all tests.")
    parser.add_argument("--config", type=str, default="config/config_tier2_gravityanalogue.json",
                       help="Path to config file")
    args = parser.parse_args()
    
    # Load config relative to this script file so running from any CWD works
    cfg = load_config(args.config)
    # Resolve output directory similar to Tier-1 harness conventions
    script_dir = Path(__file__).resolve().parent
    outdir = script_dir / cfg.get("run_settings", {}).get("output_dir", "results/Gravity")
    outdir.mkdir(parents=True, exist_ok=True)
    
    harness = Tier2Harness(cfg, outdir)
    
    # Filter to single test if requested
    if args.test:
        harness.variants = [v for v in harness.variants if v["test_id"] == args.test]
        if not harness.variants:
            log(f"[ERROR] Test '{args.test}' not found in config", "FAIL")
            return
        log(f"=== Running Single Test: {args.test} ===", "INFO")
    else:
        log(f"=== Tier-2 Gravity Analogue Suite Start ===", "INFO")
    
    results = harness.run()
    
    if args.test:
        # Single test: just show result
        log(f"=== Test {args.test} Complete ===", "INFO")
    else:
        # Full suite: show summary
        suite_summary(results)
        write_metadata_bundle(outdir, "TIER2-GRAVITY", tier=2, category="Gravity")
        log("=== Tier-2 Suite Complete ===", "INFO")

if __name__=="__main__":
    main()
