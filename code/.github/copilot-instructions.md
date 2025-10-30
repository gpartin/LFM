## LFM — Copilot / AI Coding Instructions

This file captures the immediate, actionable knowledge an AI assistant needs to be productive in this repository.

### Big picture (quick)
- Core solver: `lfm_equation.py` — single-step kernel (`lattice_step`) and serial integrator (`advance`). Energy/diagnostics helpers live here (`energy_total`, `core_metrics`).
- Parallel runner: `lfm_parallel.py` — threaded tile-based runner (`run_lattice`) using ThreadPoolExecutor, tile helpers (`_tiles_*`) and deterministic vs threaded flows.
- Monitoring & integrity: `energy_monitor.py` and `numeric_integrity.py` — used to validate CFL, finite values, and energy drift.
- I/O & logging: `lfm_results.py` (structured outputs), `lfm_logger.py` (text + JSONL), `lfm_console.py` (human-friendly prints).
- Tier harnesses/scripts: `run_tier1_relativistic.py`, `run_tier2_*` in `Archive/` — these orchestrate configs, backends (CPU/GPU), and outputs under `results/`.

### Key conventions & patterns (do not break these)
- Backend-agnostic arrays: code uses a NumPy/CuPy switch. Look for `_xp_for(...)`, `xp = cp if use_gpu else np`, and helpers like `to_numpy()` and `.get()` for CuPy arrays. Keep device/host conversions explicit and minimal.
- Parameters by dict: runtime options are passed via `params` dicts. Important keys: `dt, dx, alpha, beta, chi, gamma_damp, boundary, stencil_order, precision, threads, tiles, energy_lock, enable_monitor, monitor_outdir, monitor_label` and a nested `debug` dict (`enable_diagnostics`, `energy_tol`, `check_nan`, `profile_steps`, `edge_band`, `checksum_stride`, `diagnostics_path`).
- Energy handling: many modules compute compensated energy and keep an `_energy_log` / `_energy_drift_log` in `params`. When `energy_lock` is enabled and the run is conservative, code applies a global rescale BEFORE measuring/reporting drift.
- Parallel tiling: `lfm_parallel.py` divides work by slices; for edits touch `_tiles_*` and `_normalize_tile_args()` carefully.

### Typical developer workflows
- Run a Tier-1 suite (expects `config/config_tier1_relativistic.json`):
```powershell
python run_tier1_relativistic.py
```
Outputs are written under the project `results/` tree (e.g. `results/Tier1/<TEST_ID>/diagnostics`, `plots`).
- Tests: repository uses pytest-style test files (`test_*.py`). Run full tests with:
```powershell
python -m pytest -q
```

### Diagnostic verification before running tests (MANDATORY)
**Before executing any test harness or individual test, ALWAYS verify the configuration file has appropriate diagnostics enabled for troubleshooting.**

Critical diagnostic settings to check in config files (e.g., `config/config_tier2_gravityanalogue.json`):
- `diagnostics.track_packet` — Should be `true` for time_delay tests (tracks packet position for propagation debugging)
- `diagnostics.save_time_series` — Should be `true` for time_dilation tests (saves probe data for FFT analysis)
- `diagnostics.log_packet_stride` — Controls how often packet position is logged (default 100 steps)
- `diagnostics.energy_monitor_every` — Set to >0 (e.g., 25-100) to track per-step energy drift; 0 disables
- `debug.enable_diagnostics` — Set to `true` to enable CFL checks, NaN detection, edge effects monitoring
- `debug.quiet_run` — Set to `false` for verbose debugging output
- `run_settings.verbose` — Set to `true` for detailed per-step logging

**Standard diagnostic configurations by test mode:**
1. **Local frequency tests** (GRAV-01-06): Minimal diagnostics needed (single-step measurement)
   - `enable_diagnostics: false` (optional)
   - `energy_monitor_every: 0` (not critical)
   
2. **Time dilation tests** (GRAV-07-10): Time series required
   - `save_time_series: true` ✅ REQUIRED
   - `enable_diagnostics: true` (recommended for FFT validation)
   - `energy_monitor_every: 100` (recommended)

3. **Time delay tests** (GRAV-11-12): Packet tracking required
   - `track_packet: true` ✅ REQUIRED
   - `log_packet_stride: 100` (or smaller for high-resolution tracking)
   - `enable_diagnostics: true` (recommended to catch NaN/Inf during propagation)
   - `energy_monitor_every: 100` (recommended for conservation checks)

**Action required before every test run:**
1. Read the relevant config file section for the test being run
2. Identify the test mode (local_frequency, time_dilation, or time_delay)
3. Verify the diagnostics section has appropriate settings for that mode
4. If diagnostics are insufficient for troubleshooting, inform the user and suggest enabling:
   - For propagation issues: `track_packet`, `enable_diagnostics`, `energy_monitor_every`
   - For frequency analysis issues: `save_time_series`, `print_probe_steps`
   - For numerical stability issues: `enable_diagnostics`, `energy_monitor_every`, `check_nan`
5. Document what diagnostics will be available in test outputs (e.g., "Will generate packet_tracking CSVs for serial and parallel runs")

**If running tests in debug mode after failures:**
- Enable full diagnostics: `enable_diagnostics: true`, `energy_monitor_every: 25`
- Reduce `quiet_run: false` and `verbose: true` for maximum visibility
- Consider reducing step count (`steps_quick`) for faster iteration

### Integration points & extension notes
- GPU acceleration: optional CuPy (`cupy`) — code checks availability and chooses backend only when `run_settings.use_gpu` (or harness `use_gpu`) is set. Keep device allocations and `.get()` usage consistent.
- Energy/diagnostics hooks: `EnergyMonitor` is frequently used to record per-step energy and to create diagnostics CSVs. If adding diagnostics, integrate with `params["debug"]` and `params["_energy_log"]` so tooling downstream can consume them.
- Logging: use `LFMLogger` to write both human logs and JSONL structured events. Use `log_json()` for structured metadata (tests report metadata here).

### Files to inspect before changing core behavior
- `lfm_equation.py`, `lfm_parallel.py`, `numeric_integrity.py`, `energy_monitor.py`, `lfm_diagnostics.py`, `lfm_logger.py`, `lfm_results.py`, `lfm_plotting.py`, `lfm_visualizer.py`, and the test files `test_lfm_equation_multidim.py`, `test_lfm_dispersion_3d.py`.

### Small checklist for numeric changes (must be followed)
1. Preserve numerical shape/backends: ensure operations work for 1D/2D/3D and for NumPy/CuPy arrays.
2. Run affected unit tests and a small harness run (Tier-1 quick/quick_mode) to validate no regression in drift/energy.
3. If changing energy-related code, verify `_energy_log` and `energy_lock` behavior and update any monitoring hooks.

### Examples (patterns you can safely reuse)
- One-step update (use patterns already in repo):
  - call `laplacian(E, dx, order)` then compute mass term `-(chi**2)*E` and combine into E_next exactly as in `lattice_step`.
- Tile updates: use `_normalize_tile_args()` and update per-slice in `_step_threaded` to match current threading behavior.

### Physics preservation rule (MANDATORY)
All contributors and automated agents must preserve the physical behavior implemented in `lfm_equation.py`.
Do NOT change the physics (wave operator, mass term, CFL semantics, energy definition) without an explicit, reviewed justification.

Required checklist for any edit that touches `lfm_equation.py`:
- Write a 2–4 bullet "contract" describing the expected inputs/outputs and numerical invariants (e.g., shapes, dtype, meaning of `chi`, energy conserved when gamma_damp==0 and absorbing off).
- Run the full unit tests plus a short Tier-1 quick run (set `run_settings.quick_mode=true` or reduce steps) and confirm no significant energy drift or changed dispersion.
- Produce a small numeric regression report: before/after baseline energy and 1–2 probe frequencies (e.g. run `advance()` or `lattice_step()` on a compact initial condition). Include tolerances and a human-readable summary for reviewers.
- If the change alters mathematical terms (Laplacian stencil, sign or placement of mass term, time-centering), mark the PR with `BREAKS-PHYSICS` and add a clear rationale in the PR body. Such changes require an explicit review by a domain maintainer before merge.

Guidance for automated agents:
- Prefer non-physics edits (refactor, perf, logging) that keep all numerical expressions identical. When in doubt, add a short regression test and run it.
- If you cannot run the numeric regression locally (no GPU or limited resources), still open a draft PR with the change and include a suggested numeric test and the expected tolerance so maintainers can run CI-level checks.

If any of these areas are unclear or you need an expanded section (testing commands, CI, or config locations), tell me which part to expand and I will iterate.
