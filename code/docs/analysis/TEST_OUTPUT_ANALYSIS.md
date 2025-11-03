# LFM Test Suite - Scientific Output Requirements Analysis

**Status:** Comprehensive analysis of 65 tests across 4 tiers to ensure proper scientific validation artifacts.  
**Date:** Generated from comprehensive code inspection  
**Purpose:** Ensure every test produces appropriate plots, CSVs, and images that physicists need to verify correctness.

---

## Executive Summary

### Current State
- **All tests (65/65)** produce: `summary.json`, `diagnostics/`, `plots/` directories
- **Resource tracking:** All tests now collect CPU/RAM/GPU metrics (validated 65/65)
- **Test-specific outputs:** Many tests already produce specialized outputs (dispersion CSVs, packet tracking, energy time series)

### Critical Gaps Identified
1. **REL-11-14 (Dispersion):** Missing dispersion curve plots (ω vs k visualization)
2. **REL-01-02 (Isotropy):** Missing directional frequency comparison plots
3. **GRAV-16 (Double-slit):** Missing interference pattern image (2D heatmap)
4. **QUAN-10 (Bound states):** Missing individual wavefunction plots for each mode
5. **QUAN-12 (Tunneling):** Has basic plots but missing wavefunction detail in barrier region
6. **QUAN-09 (Uncertainty):** Missing scatter plot of (Δx, Δk) with theoretical bound overlay

### Recommendation Priority
**HIGH:** GRAV-16 (interference pattern), REL-11-14 (dispersion curves), QUAN-10 (wavefunction plots)  
**MEDIUM:** Isotropy comparison plots, uncertainty scatter plots  
**LOW:** Enhancement of existing plots (annotations, overlays, better legends)

---

## Part 1: Standard Outputs (ALL 65 Tests)

These outputs are produced by every test across all 4 tiers:

### 1.1 Summary Metadata
- **File:** `summary.json`
- **Content:** Test ID, description, pass/fail status, metrics, parameters, timestamps
- **Purpose:** Machine-readable test results for analysis and CI/CD integration
- **Current status:** ✅ Implemented in all tests

### 1.2 Diagnostic Data Directory
- **Directory:** `diagnostics/`
- **Content:** Test-specific CSV files, time series data, measurement logs
- **Purpose:** Raw numerical data for post-processing, reanalysis, reproducibility
- **Current status:** ✅ Implemented in all tests

### 1.3 Visualization Directory
- **Directory:** `plots/`
- **Content:** PNG images of key results (energy plots, field visualizations, etc.)
- **Purpose:** Quick visual verification of test behavior
- **Current status:** ✅ Implemented in all tests (though content varies)

### 1.4 Resource Metrics (NEW)
- **Content:** CPU usage (%), RAM usage (GB), GPU usage (%) if applicable
- **Storage:** In `summary.json` under `"resource_metrics"` key
- **Purpose:** Performance monitoring, regression detection, hardware utilization tracking
- **Current status:** ✅ Recently implemented (65/65 validated)

### 1.5 Master Status Tracking
- **File:** `results/MASTER_TEST_STATUS.csv`
- **Content:** Latest status of all 65 tests (pass/fail, last run time, key metrics)
- **Purpose:** High-level dashboard of entire test suite health
- **Current status:** ✅ Implemented and updated by all tier runners

---

## Part 2: Test-Specific Outputs by Physics Category

### 2.1 TIER 1: Relativistic Tests (15 tests)

#### 2.1.1 Isotropy Tests (REL-01, REL-02, REL-09, REL-10)
**Physics:** Wave propagation should be identical in all spatial directions (rotational symmetry).

**Current outputs:**
- `summary.json` with anisotropy metric

**Missing outputs (CRITICAL):**
- ❌ **`isotropy_comparison.png`** - Bar chart comparing ω_left, ω_right, ω_up, ω_down
  - X-axis: Direction (left/right/up/down)
  - Y-axis: Measured frequency ω
  - Expected: All bars equal height (within tolerance)
  - Tolerance band: Horizontal dashed lines at ω_avg ± 1%
  
- ❌ **`frequency_data.csv`** - Numerical comparison table
  - Columns: direction, measured_omega, expected_omega, abs_error, rel_error_pct
  - Purpose: Quantitative verification of isotropy claim

**Implementation priority:** MEDIUM

---

#### 2.1.2 Lorentz Boost Tests (REL-03, REL-04)
**Physics:** Wave packet behavior under coordinate transformation (special relativity).

**Current outputs:**
- `summary.json` with boost validation metrics
- Energy/momentum conservation data in diagnostics/

**Current status:** ✅ ADEQUATE (quantitative metrics sufficient for this test type)

---

#### 2.1.3 Causality Tests (REL-05, REL-06, REL-15)
**Physics:** Information propagates at/below speed of light (no superluminal signals).

**Current outputs:**
- **REL-15:** ✅ `correlation_vs_distance_{tid}.png` (semilogy plot showing correlation decay outside light cone)
- **REL-15:** ✅ `correlation_data.csv` (time, position, correlation, lightcone_radius)
- **REL-15:** ✅ `violations.csv` (if space-like correlations detected)
- **REL-05/06:** Summary metrics only

**Current status:** ✅ GOOD for REL-15 (has visualization); ADEQUATE for REL-05/06 (pulse/noise tests don't need plots)

---

#### 2.1.4 Dispersion Relation Tests (REL-11, REL-12, REL-13, REL-14)
**Physics:** Klein-Gordon dispersion ω² = k² + χ² (validates relativistic energy-momentum relation E² = (pc)² + (mc²)²).

**Current outputs:**
- ✅ `dispersion_measurement.csv` - Quantitative comparison of ω²/k² measured vs theory
- ✅ `projection_series.csv` - Time series of mode projection (for FFT analysis)

**Missing outputs (CRITICAL):**
- ❌ **`dispersion_curve_{regime}.png`** - The single most important plot for these tests!
  - X-axis: Wavenumber k (or χ/k ratio)
  - Y-axis: Measured ω vs theoretical ω = √(k² + χ²)
  - Plot both measured points (red circles) and theoretical curve (blue line)
  - Annotate regime: "Non-relativistic χ/k=10", "Weakly relativistic χ/k=1", "Relativistic χ/k=0.5", "Ultra-relativistic χ/k=0.1"
  - Add error percentage labels on data points
  - Purpose: **THIS IS THE DISPERSION CURVE** - fundamental physics validation
  
- ❌ **`velocity_comparison.png`** (optional enhancement)
  - Subplot 1: Group velocity v_g = dω/dk vs k (should approach c as χ/k → 0)
  - Subplot 2: Phase velocity v_p = ω/k vs k (superluminal for massive particles)
  - Purpose: Show relativistic vs non-relativistic regimes

**Implementation priority:** HIGH (dispersion curve is fundamental to these tests)

**Note:** Currently only CSV data exists. Physicists expect to SEE the dispersion curve, not just read numbers.

---

#### 2.1.5 Phase Independence & Superposition (REL-07, REL-08)
**Physics:** Linearity of Klein-Gordon equation.

**Current outputs:**
- `summary.json` with linearity metrics

**Current status:** ✅ ADEQUATE (quantitative validation sufficient)

---

#### 2.1.6 Momentum Conservation (REL-16, if enabled)
**Physics:** Field momentum conservation in collisions.

**Current outputs:**
- ✅ `momentum_conservation_{tid}.png` - Two subplots showing momentum history
- ✅ `momentum_history.csv` - Time series data
- ✅ `momentum_density.csv` - Spatial distribution

**Current status:** ✅ EXCELLENT (comprehensive output)

---

### 2.2 TIER 2: Gravity Analogue Tests (25 tests)

#### 2.2.1 Local Frequency Tests (GRAV-01 through GRAV-06)
**Physics:** Gravitational redshift analog - frequency shifts in varying χ-field (ω_local/ω_∞ = √(1+χ)).

**Current outputs:**
- ✅ `self_consistency_profile_overlay_{tid}.png` - ω(x) vs χ(x) spatial profiles
- ✅ `self_consistency_ratio_{tid}.png` - ω/√(1+χ) ratio verification
- ✅ `self_consistency_profile_{tid}.csv` - Position, omega, chi, ratio data
- ✅ `local_freq_profile_{tid}.csv` - Additional frequency profile data

**Current status:** ✅ EXCELLENT (comprehensive spatial frequency analysis with plots)

---

#### 2.2.2 Time Dilation Tests (GRAV-07 through GRAV-10)
**Physics:** Gravitational time dilation analog - phase accumulation differs in curved spacetime.

**Current outputs:**
- ✅ `detector_signals_{tid}.csv` - Time series from probes at different positions
- ✅ `initial_conditions_{tid}.csv` - Initial field configuration
- ✅ `field_snapshots_{tid}.csv` - Field evolution snapshots

**Missing outputs (MEDIUM priority):**
- ⚠️ **`time_dilation_comparison.png`** - Would enhance understanding but CSVs may suffice
  - Subplot 1: Phase accumulation Φ(t) at probe 1 vs probe 2
  - Subplot 2: Frequency shift Δω/ω₀ vs position
  - Purpose: Visualize gravitational time dilation effect

**Current status:** ✅ GOOD (CSV data sufficient for FFT analysis, plots would be enhancement)

---

#### 2.2.3 Time Delay Tests / Shapiro Delay (GRAV-11, GRAV-12)
**Physics:** Wave packet delay when traversing curved spacetime region (gravitational time delay analog).

**Current outputs:**
- ✅ `packet_tracking_{tid}_serial.csv` - Packet centroid position vs time (serial run)
- ✅ `packet_tracking_{tid}_parallel.csv` - Packet centroid position vs time (parallel run)
- ✅ `detector_signal_{tid}_slab.csv` - Signal through χ-slab
- ✅ `detector_signal_{tid}_control.csv` - Signal in flat spacetime (control)
- ✅ `envelope_measurement_{tid}.csv` - Packet envelope analysis
- ✅ `packet_analysis_{tid}.csv` - Detailed packet metrics

**Missing outputs (LOW priority - data is very complete):**
- ⚠️ **`shapiro_delay_trajectory.png`** (optional enhancement)
  - Plot packet position x(t) for both slab and control runs
  - Show time delay Δt graphically with annotations
  - Purpose: Visual demonstration of Shapiro delay

**Current status:** ✅ EXCELLENT (comprehensive CSV data; plot would be nice-to-have)

---

#### 2.2.4 Dynamic χ-field Tests (GRAV-13, GRAV-14)
**Physics:** Field evolution in time-varying background curvature.

**Current outputs:**
- ✅ `chi_wave_history_{tid}.csv` - Time series of χ-field evolution
- ✅ `chi_wave_evolution_{tid}.png` - Multi-panel visualization of χ(x,t)
- ✅ `chi_evolution_{tid}.csv` - Detailed χ-field data
- ✅ `chi_coupling_{tid}.png` - Field-curvature coupling visualization

**Current status:** ✅ EXCELLENT (comprehensive output)

---

#### 2.2.5 Double-Slit Interference (GRAV-16)
**Physics:** Wave interference pattern - quintessential wave phenomenon demonstrating coherent superposition.

**Current outputs:**
- ✅ `snapshot_{GRAV-16}.h5` - 3D field data in HDF5 format (multiple time snapshots)
- Summary JSON with runtime metrics

**Missing outputs (CRITICAL - HIGHEST PRIORITY):**
- ❌ **`interference_pattern.png`** - THE DEFINING OUTPUT FOR THIS TEST
  - 2D heatmap of |E(x,y)|² at screen behind slits
  - Color scale: intensity from 0 (dark) to max (bright)
  - X-axis: Transverse position (across screen)
  - Y-axis: Vertical position
  - Clearly visible interference fringes (alternating bright/dark bands)
  - Slit positions marked with annotations
  - Colorbar showing intensity scale
  - Title: "Double-Slit Interference Pattern - Fringe Spacing = {spacing:.3f}"
  - Purpose: **THIS IS THE DOUBLE-SLIT EXPERIMENT** - must show fringes!

- ❌ **`intensity_profile.csv`**
  - Columns: x_transverse, intensity, fringe_number
  - Purpose: Quantitative analysis of fringe positions and spacing

- ❌ **`fringe_analysis.txt`** or in summary JSON
  - Measured fringe spacing: d_measured
  - Theoretical fringe spacing: d_theory = λD/d (wavelength × distance / slit separation)
  - Fringe visibility: V = (I_max - I_min)/(I_max + I_min)
  - Purpose: Quantitative validation of interference theory

**Implementation priority:** HIGHEST (this is a landmark physics demonstration test)

**Note:** HDF5 data exists but is not analyzed/visualized. Must extract screen slice and create interference pattern image.

---

#### 2.2.6 3D Wave Propagation (GRAV-15, GRAV-17-25)
**Physics:** Full 3D wave dynamics, spherical wave fronts, 3D geometric effects.

**Current outputs:**
- ✅ HDF5 snapshot files for 3D tests
- ✅ Various CSV files depending on specific test mode

**Missing outputs (MEDIUM priority):**
- ⚠️ **3D visualizations** - Currently data exists but no pre-rendered images
  - Could add: Cross-section slices, isosurface renders, or volumetric plots
  - However, 3D viz is complex and HDF5 data allows post-processing
  - Recommendation: Add at least one representative slice image per test

**Current status:** ✅ GOOD (3D data saved; visualization is post-processing step)

---

### 2.3 TIER 3: Energy Conservation Tests (11 tests)

#### 2.3.1 Global Energy Conservation (ENER-01, ENER-02)
**Physics:** Total energy E = KE + PE conserved over time in Hamiltonian dynamics.

**Current outputs:**
- ✅ `energy_vs_time.png` - Energy trace showing drift
- ✅ `energy_trace.csv` - Time series: time, energy
- ✅ `entropy_vs_time.png` - Entropy evolution
- ✅ `entropy_trace.csv` - Time series: time, entropy

**Current status:** ✅ EXCELLENT (comprehensive energy tracking)

---

#### 2.3.2 Wave Integrity Tests (ENER-03, ENER-04)
**Physics:** Wave packet maintains coherence in curved spacetime (χ-gradient).

**Current outputs:**
- Same as global conservation tests (energy and entropy plots/CSVs)

**Current status:** ✅ ADEQUATE (energy tracking sufficient for integrity validation)

---

#### 2.3.3 Hamiltonian Partitioning (ENER-05, ENER-06, ENER-07)
**Physics:** Energy flow between kinetic (KE), gradient (GE), and potential (PE) components.

**Current outputs:**
- ✅ `hamiltonian_components.png` - Stacked area plot showing KE/GE/PE evolution
- ✅ `hamiltonian_total.png` - Total Hamiltonian conservation plot
- ✅ `hamiltonian_components.csv` - Time series: time, KE, GE, PE, total
- ✅ Energy and entropy plots (as above)

**Current status:** ✅ EXCELLENT (shows energy flow between components - critical for curved spacetime tests)

---

#### 2.3.4 Dissipation Tests (ENER-08, ENER-09)
**Physics:** Energy decay under damping γ·∂E/∂t (exponential decay E ~ exp(-γt)).

**Current outputs:**
- ✅ Energy and entropy plots showing exponential decay

**Current status:** ✅ ADEQUATE (decay is visible in plots; could add exponential fit overlay)

---

#### 2.3.5 Thermalization Test (ENER-10)
**Physics:** System with noise + damping approaches thermal equilibrium (entropy increases, energy stabilizes).

**Current outputs:**
- ✅ Energy and entropy plots showing relaxation to steady state

**Current status:** ✅ ADEQUATE (steady-state behavior is visible)

---

#### 2.3.6 Momentum Conservation (ENER-11, currently SKIPPED)
**Physics:** Momentum conservation in packet collisions.

**Current outputs:**
- N/A (test skipped)

**Current status:** ⏸️ SKIPPED (not currently active)

---

### 2.4 TIER 4: Quantization Tests (14 tests)

#### 2.4.1 Energy Transfer (QUAN-01, QUAN-02)
**Physics:** Energy exchange between modes with discrete quantum transitions ΔE = ℏω.

**Current outputs:**
- ✅ `energy_transfer.png` - Two subplots: total energy + mode1/mode2 energy evolution
- ✅ `energy_evolution.csv` - Time series: time, total_energy, mode1_energy, mode2_energy, transfer

**Current status:** ✅ EXCELLENT (shows mode coupling dynamics)

---

#### 2.4.2 Spectral Linearity (QUAN-03, QUAN-04)
**Physics:** Energy scales linearly with amplitude squared: E ∝ A² (classical EM field analog).

**Current outputs:**
- ✅ `spectral_linearity.png` - Scatter plot of E vs A² with linear fit
- ✅ `linearity_test.csv` - Data: amplitude, amplitude_squared, energy, predicted_energy

**Current status:** ✅ EXCELLENT (demonstrates E ∝ A² scaling)

---

#### 2.4.3 Phase-Amplitude Coupling (QUAN-05, QUAN-06)
**Physics:** Verify linear superposition without spurious AM→PM conversion.

**Current outputs:**
- ✅ `linearity_superposition.png` - 4-panel plot showing individual components and superposition
- ✅ `linearity.csv` - Error metrics for different test cases

**Current status:** ✅ EXCELLENT (comprehensive linearity validation)

---

#### 2.4.4 Wavefront Stability (QUAN-07)
**Physics:** Large-amplitude wave maintains shape (non-dispersive in uniform region).

**Current outputs:**
- ✅ `wavefront_stability.png` - Two subplots: shape comparison (t=0 vs t=final) + energy conservation
- ✅ `wavefront_evolution.csv` - Time series: time, peak_position, peak_amplitude, width, shape_error

**Current status:** ✅ EXCELLENT (shows shape preservation)

---

#### 2.4.5 Lattice Blowout (QUAN-08)
**Physics:** Numerical stability test at high energy (CFL condition, no NaN/Inf).

**Current outputs:**
- ✅ `stability_test.png` - Energy and field magnitude vs time
- ✅ `stability_test.csv` - Time series: time, energy, max_field

**Current status:** ✅ ADEQUATE (stability metrics tracked)

---

#### 2.4.6 Uncertainty Principle (QUAN-09)
**Physics:** Heisenberg uncertainty Δx·Δk ≥ 1/2 (Fourier transform limit).

**Current outputs:**
- ✅ `uncertainty_dx_dk.png` - Plot showing Δx·Δk product for various σ values
- ✅ `uncertainty_results.csv` - Data: sigma_x, delta_x, delta_k, product

**Missing outputs (MEDIUM priority):**
- ⚠️ **Enhance `uncertainty_dx_dk.png`** to be a scatter plot:
  - X-axis: Δx
  - Y-axis: Δk
  - Plot measured points (Δx, Δk) from different σ values
  - Add hyperbola Δx·Δk = 0.5 (theoretical bound)
  - Shade region Δx·Δk < 0.5 as "forbidden" (red)
  - All points should lie above/on the curve
  - Purpose: Visual demonstration of uncertainty bound

**Current status:** ✅ GOOD (has plot and CSV; scatter format would be clearer)

---

#### 2.4.7 Bound State Quantization (QUAN-10)
**Physics:** Discrete energy eigenvalues E_n in box/cavity (quantum number n).

**Current outputs:**
- ✅ `quantized_energies.png` - E_n vs n plot showing measured vs theory
- ✅ `mode_evolution.png` - Mode amplitude oscillations vs time
- ✅ `eigenvalues.csv` - Data: mode_n, theory_energy, measured_energy, rel_error

**Missing outputs (CRITICAL):**
- ❌ **`wavefunction_mode_{n}.png`** - Individual wavefunction plots for each mode
  - Plot ψ_n(x) = √(2/L) sin(nπx/L) for n=1,2,3,...
  - X-axis: Position x
  - Y-axis: Wavefunction amplitude
  - Show measured (from field snapshot) vs analytical
  - Title: "Mode n={n}, E_n = {energy:.4f}"
  - Purpose: Visualize spatial structure of quantum eigenstates
  - **This is a fundamental quantum output** - physicists expect to see ψ_n(x)!

- ❌ **`wavefunction_overlay.png`** (optional)
  - Plot all measured wavefunctions ψ_1, ψ_2, ψ_3 on single plot
  - Different colors/linestyles for each mode
  - Purpose: Show orthogonality and spatial structure hierarchy

**Implementation priority:** HIGH (wavefunctions are as important as energy levels)

**Note:** Code extracts mode amplitudes from snapshots but doesn't plot the spatial wavefunctions.

---

#### 2.4.8 Zero-Point Energy (QUAN-11)
**Physics:** Ground state energy E₀ = ½ℏω (quantum vacuum fluctuations).

**Current outputs:**
- ✅ `zero_point_energy.png` - Histogram and time series showing ground state energy
- ✅ `zero_point_energy.csv` - Data: time, ground_state_energy, fluctuation

**Current status:** ✅ EXCELLENT (demonstrates quantum ground state)

---

#### 2.4.9 Quantum Tunneling (QUAN-12)
**Physics:** Wave penetrates classically forbidden barrier (E < V).

**Current outputs:**
- ✅ `tunneling_transmission.png` - Energy distribution in regions + transmission bar chart
- ✅ `tunneling_snapshot.png` - Final wavefunction E(x) with barrier potential overlay
- ✅ `energy_regions.csv` - Time series: energy_left, energy_barrier, energy_right

**Missing outputs (LOW priority - current plots are good):**
- ⚠️ **Enhance `tunneling_snapshot.png`** (optional):
  - Add log-scale subplot showing exponential decay in barrier: log|E| vs x
  - Show κ = √(χ² - ω²) decay rate annotation
  - Purpose: Explicitly show exp(-κx) tunneling behavior

**Current status:** ✅ GOOD (basic plots adequate; enhancement would show exponential decay more clearly)

---

#### 2.4.10 Wave-Particle Duality (QUAN-13)
**Physics:** Double-slit interference destroyed by "which-way" information (complementarity principle).

**Current outputs:**
- ✅ `wave_particle_duality.png` - Comparison of interference with/without which-way info
- ✅ `interference_patterns.csv` - Data: x_position, intensity_coherent, intensity_which_way, visibility

**Current status:** ✅ EXCELLENT (demonstrates complementarity)

---

#### 2.4.11 Non-Thermalization (QUAN-14)
**Physics:** Klein-Gordon conserves energy (doesn't approach Planck distribution in isolated system).

**Current outputs:**
- ✅ `non_thermalization.png` - Energy and mode energy evolution (conservation check)
- ✅ `non_thermalization.csv` - Time series: time, total_energy, E_avg, mode_n_energy

**Current status:** ✅ ADEQUATE (shows energy conservation; primary physics is conserved dynamics)

---

#### 2.4.12 Additional Quantum Tests (If Enabled)

**Cavity Spectrum Test:**
- ✅ `cavity_spectrum.png` - Mode frequencies vs theory
- ✅ `mode_spectrum.csv` - mode_n, theory_omega, measured_omega
- ✅ `mode_amplitudes.csv` - Mode amplitude data

**Threshold Test:**
- ✅ `threshold_curve.png` - Transmission vs frequency
- ✅ `transmission_vs_freq.csv` - Frequency sweep data

**Current status:** ✅ EXCELLENT (these are well-instrumented tests)

---

## Part 3: Summary Tables

### 3.1 Output Inventory by Test Type

| Test Category | Current PNG Outputs | Current CSV Outputs | Missing Critical Outputs |
|---------------|---------------------|---------------------|--------------------------|
| **Isotropy** | None | None | ❌ Isotropy comparison bar chart |
| **Lorentz Boost** | None | Energy/momentum CSVs | ✅ None (metrics sufficient) |
| **Causality** | Correlation plot (REL-15) | Correlation CSV, violations CSV | ✅ None |
| **Dispersion** | None | Measurement CSV, time series CSV | ❌ **Dispersion curve ω(k) plot** |
| **Local Frequency** | Profile overlay, ratio plots | Profile CSVs | ✅ None |
| **Time Dilation** | None | Detector signals CSVs | ⚠️ Phase comparison plot (optional) |
| **Shapiro Delay** | None | Packet tracking CSVs | ⚠️ Trajectory plot (optional) |
| **Double-Slit 3D** | None | HDF5 snapshots | ❌ **Interference pattern image** |
| **Energy Conservation** | Energy/entropy vs time | Energy/entropy CSVs | ✅ None |
| **Hamiltonian** | Component stacked plot, total plot | Components CSV | ✅ None |
| **Energy Transfer** | Transfer plot (modes) | Evolution CSV | ✅ None |
| **Spectral Linearity** | E vs A² scatter + fit | Linearity CSV | ✅ None |
| **Wavefront Stability** | Shape comparison, energy | Evolution CSV | ✅ None |
| **Uncertainty** | Δx·Δk product plot | Results CSV | ⚠️ Scatter plot format (optional) |
| **Bound States** | Energy level plot, mode evolution | Eigenvalues CSV | ❌ **Individual ψ_n(x) plots** |
| **Zero-Point Energy** | Histogram + time series | Energy CSV | ✅ None |
| **Tunneling** | Transmission bar, snapshot | Energy regions CSV | ⚠️ Log-scale decay (optional) |
| **Wave-Particle** | Interference comparison | Patterns CSV | ✅ None |

### 3.2 Priority Matrix

| Priority | Test IDs | Missing Output | Estimated Implementation Effort |
|----------|----------|----------------|--------------------------------|
| **CRITICAL** | GRAV-16 | Interference pattern 2D heatmap | **HIGH** (needs HDF5 parsing, screen slice extraction, matplotlib imshow) |
| **CRITICAL** | REL-11-14 | Dispersion curve ω vs k plot | **MEDIUM** (data exists in CSV, just need to plot) |
| **HIGH** | QUAN-10 | Individual wavefunction plots ψ_n(x) | **MEDIUM** (need to extract spatial modes from snapshots) |
| **MEDIUM** | REL-01-02, REL-09-10 | Isotropy comparison bar chart | **LOW** (data already in summary, just plot) |
| **MEDIUM** | QUAN-09 | Uncertainty scatter plot (Δx vs Δk) | **LOW** (data exists, reformat plot) |
| **LOW** | GRAV-11-12 | Shapiro delay trajectory plot | **LOW** (CSV data exists, simple line plot) |
| **LOW** | QUAN-12 | Tunneling log-scale decay subplot | **LOW** (enhancement to existing plot) |

---

## Part 4: Implementation Recommendations

### 4.1 Immediate Actions (Required for Completeness)

#### Action 1: Add Dispersion Curve Plot (REL-11-14)
**File:** `run_tier1_relativistic.py`  
**Function:** `run_dispersion_relation_variant()`  
**Location:** After CSV save, before return

```python
# After saving CSVs (around line 1365):

# Generate dispersion curve plot
import matplotlib.pyplot as plt

# Since we only measure one (k, ω) point per test, we need to collect across tests
# For now, create a simple measured vs theory comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: ω measured vs theory
ax1.scatter([k_ang], [omega_meas], s=100, c='red', marker='o', label='Measured', zorder=3)
ax1.plot([k_ang], [omega_theory], 'bx', markersize=15, markeredgewidth=3, label='Theory', zorder=3)
k_range = np.linspace(0, k_ang * 1.5, 100)
omega_curve = np.sqrt(k_range**2 + chi**2)
ax1.plot(k_range, omega_curve, 'b-', alpha=0.3, label=r'$\omega = \sqrt{k^2 + \chi^2}$')
ax1.set_xlabel('Wavenumber k', fontsize=12)
ax1.set_ylabel('Frequency ω', fontsize=12)
ax1.set_title(f'{tid}: Dispersion Relation\nχ/k = {chi_over_k:.3f}', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: ω²/k² measured vs theory (the actual test metric)
ax2.scatter([chi_over_k], [omega2_over_k2_meas], s=100, c='red', marker='o', label='Measured')
ax2.scatter([chi_over_k], [omega2_over_k2_theory], s=100, c='blue', marker='x', label='Theory')
ax2.set_xlabel(r'Mass ratio $\chi/k$', fontsize=12)
ax2.set_ylabel(r'$\omega^2/k^2$', fontsize=12)
ax2.set_title(f'Relativistic Energy-Momentum\nError: {rel_err*100:.2f}%', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_yscale('log')

# Annotate regime
regime_text = {
    "REL-11": "Non-relativistic\n(χ >> k)",
    "REL-12": "Weakly relativistic\n(χ ≈ k)",
    "REL-13": "Relativistic\n(χ < k)",
    "REL-14": "Ultra-relativistic\n(χ << k)"
}.get(tid, "")
if regime_text:
    ax2.text(0.05, 0.95, regime_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(plot_dir / f"dispersion_curve_{tid}.png", dpi=150)
plt.close()
```

**Rationale:** Dispersion relation is THE fundamental test of Klein-Gordon physics. Must have visual confirmation.

---

#### Action 2: Add Interference Pattern Image (GRAV-16)
**File:** `run_tier2_gravityanalogue.py`  
**Function:** `run_variant()` in double_slit_3d mode  
**Location:** After HDF5 save (around line 1618)

```python
# After saving HDF5 snapshots:

# Extract interference pattern at screen
if len(snapshots_3d) > 0:
    # Use final snapshot
    final_field = snapshots_3d[-1]  # Shape: (Nz, Ny, Nx)
    
    # Define screen position (behind slits, e.g., 80% through domain)
    screen_z_frac = 0.8
    screen_z_idx = int(screen_z_frac * final_field.shape[0])
    
    # Extract screen slice: intensity = |E|²
    screen_slice = final_field[screen_z_idx, :, :]  # Shape: (Ny, Nx)
    intensity = np.abs(screen_slice)**2
    
    # Plot interference pattern
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 2D heatmap
    im = ax1.imshow(intensity, cmap='hot', origin='lower', aspect='auto',
                    extent=[0, intensity.shape[1]*dx, 0, intensity.shape[0]*dx])
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title(f'{tid}: Interference Pattern at Screen', fontsize=14)
    plt.colorbar(im, ax=ax1, label='Intensity |E|²')
    
    # Mark slit positions (if available from config)
    if 'slit1_y' in p and 'slit2_y' in p:
        slit1_y = p['slit1_y']
        slit2_y = p['slit2_y']
        ax1.axhline(slit1_y, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label='Slit 1')
        ax1.axhline(slit2_y, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label='Slit 2')
        ax1.legend()
    
    # 1D intensity profile (slice through center)
    center_x_idx = intensity.shape[1] // 2
    intensity_profile = intensity[:, center_x_idx]
    y_coords = np.arange(len(intensity_profile)) * dx
    
    ax2.plot(y_coords, intensity_profile, 'b-', linewidth=2)
    ax2.set_xlabel('Y Position (transverse)', fontsize=12)
    ax2.set_ylabel('Intensity |E|²', fontsize=12)
    ax2.set_title('Intensity Profile Showing Fringes', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Measure fringe visibility
    I_max = np.max(intensity_profile)
    I_min = np.min(intensity_profile)
    visibility = (I_max - I_min) / (I_max + I_min + 1e-30)
    ax2.text(0.05, 0.95, f'Fringe Visibility: {visibility:.3f}', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(plot_dir / f"interference_pattern_{tid}.png", dpi=150)
    plt.close()
    
    log(f"Generated interference pattern plot: visibility={visibility:.3f}", "INFO")
    
    # Save intensity profile CSV
    profile_csv = diag_dir / f"intensity_profile_{tid}.csv"
    with open(profile_csv, 'w') as f:
        f.write("y_position,intensity\n")
        for y, I in zip(y_coords, intensity_profile):
            f.write(f"{y:.6f},{I:.6e}\n")
```

**Rationale:** Double-slit experiment is THE defining wave interference test. Must show fringes visually.

---

#### Action 3: Add Wavefunction Plots (QUAN-10)
**File:** `run_tier4_quantization.py`  
**Function:** `run_bound_state_quantization()`  
**Location:** After mode_evolution plot save (around line 1920)

```python
# After saving mode_evolution.png:

# Plot individual wavefunctions ψ_n(x) for each mode
x_np = to_numpy(x)

# Create multi-panel figure for all modes
num_modes_plot = min(num_modes, 6)  # Plot first 6 modes
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for n in range(1, num_modes_plot + 1):
    ax = axes[n - 1]
    
    # Analytical wavefunction
    k_n = n * np.pi / L
    psi_analytical = np.sqrt(2.0 / L) * np.sin(k_n * x_np)
    
    # Measured wavefunction (reconstruct from last snapshot if available)
    # For now, use the initial mode shape as proxy (since we don't save field snapshots in current code)
    # TO DO: Extract from actual simulation snapshot
    psi_measured = psi_analytical  # Placeholder - need actual field data
    
    # Plot
    ax.plot(x_np, psi_analytical, 'b-', linewidth=2, label='Analytical', alpha=0.7)
    ax.plot(x_np, psi_measured, 'r--', linewidth=2, label='Measured', alpha=0.7)
    ax.set_xlabel('Position x', fontsize=10)
    ax.set_ylabel(f'ψ_{n}(x)', fontsize=10)
    ax.set_title(f'Mode n={n}, E_n={theory_energies[n-1]:.4f}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(out_dir / "plots" / "wavefunctions_bound_states.png", dpi=150)
plt.close()

# Also save individual high-res plots for key modes (n=1, 2, 3)
for n in [1, 2, 3]:
    if n <= num_modes:
        plt.figure(figsize=(10, 6))
        k_n = n * np.pi / L
        psi_analytical = np.sqrt(2.0 / L) * np.sin(k_n * x_np)
        
        plt.plot(x_np, psi_analytical, 'b-', linewidth=3, label='Analytical ψ_n(x)')
        plt.xlabel('Position x', fontsize=14)
        plt.ylabel(f'ψ_{n}(x)', fontsize=14)
        plt.title(f'Bound State Wavefunction: n={n}, E_n={theory_energies[n-1]:.4f}', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.axhline(0, color='black', linewidth=1)
        
        # Add node annotations
        num_nodes = n - 1
        plt.text(0.05, 0.95, f'Nodes: {num_nodes}', transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / f"wavefunction_mode_{n}.png", dpi=150)
        plt.close()

log(f"Generated wavefunction plots for {num_modes_plot} modes", "INFO")
```

**Rationale:** Wavefunctions ψ_n(x) are as important as energy levels E_n for bound state validation. Physicists need to see spatial structure.

**Note:** Current code doesn't save field snapshots during bound state test. Will need to either:
1. Add snapshot saving during evolution loop, OR
2. Run a quick post-simulation with analytical initial conditions to extract steady-state modes

---

### 4.2 Medium Priority Enhancements

#### Action 4: Isotropy Comparison Bar Chart (REL-01-02, REL-09-10)
**Implementation:** Add bar chart after frequency measurements are completed.

#### Action 5: Uncertainty Scatter Plot (QUAN-09)
**Implementation:** Reformat existing plot to show (Δx, Δk) points with ΔxΔk=0.5 hyperbola.

---

### 4.3 Low Priority / Optional Enhancements

These are nice-to-have improvements but not critical for test validity:

1. **Shapiro delay trajectory plot** (GRAV-11-12) - CSV data is comprehensive
2. **Time dilation phase comparison plot** (GRAV-07-10) - FFT analysis from CSV is standard workflow
3. **Tunneling log-scale decay** (QUAN-12) - Current plots adequate
4. **3D visualization slices** (various 3D tests) - HDF5 post-processing is acceptable

---

## Part 5: Testing & Validation Plan

### 5.1 Implementation Sequence

1. **Week 1:** Implement critical outputs (GRAV-16, REL-11-14, QUAN-10)
2. **Week 2:** Test all 4 tier runners with new outputs
3. **Week 3:** Implement medium-priority enhancements (isotropy, uncertainty)
4. **Week 4:** Documentation and final validation run (65/65 tests)

### 5.2 Validation Checklist

For each implemented output, verify:
- [ ] File is created in correct directory (plots/ or diagnostics/)
- [ ] File naming convention matches: `{output_type}_{test_id}.png`
- [ ] Plot has proper labels (title, axis labels, legend, units)
- [ ] Plot shows expected physics (fringes for interference, dispersion curve for REL-11-14, etc.)
- [ ] CSV has proper header row with column names
- [ ] CSV data is formatted correctly (no NaNs, proper precision)
- [ ] Output is referenced in summary.json metadata (if applicable)
- [ ] Test passes/fails as expected when physics is correct/incorrect

### 5.3 Regression Testing

After implementing new outputs:
- Run full test suite: `python run_tier1_relativistic.py`, etc.
- Verify all 65 tests still pass (no broken physics logic)
- Check resource metrics still collected (CPU/RAM/GPU)
- Validate MASTER_TEST_STATUS.csv updates correctly

---

## Part 6: Documentation Requirements

### 6.1 Update Test README Files

For each tier, update README/docstring to list new outputs:

**Example for Tier 1:**
```markdown
### REL-11-14: Dispersion Relation Tests

**Outputs:**
- `summary.json` - Test metadata and metrics
- `diagnostics/dispersion_measurement.csv` - Measured vs theory ω²/k²
- `diagnostics/projection_series.csv` - Time series for FFT
- `plots/dispersion_curve_{test_id}.png` - **NEW**: ω(k) dispersion curve with regime annotation
- `plots/dispersion_curve_{test_id}.png` - **NEW**: ω²/k² vs χ/k on log-log scale
```

### 6.2 Update test_output_requirements.py

Add new output requirements to validation framework:

```python
# In get_test_output_requirements():

if test_id.startswith("REL-") and int(test_id.split("-")[1]) >= 11:
    # Dispersion tests
    requirements["required_plots"].append("dispersion_curve_{test_id}.png")
    requirements["notes"] = "Dispersion curve ω(k) is fundamental validation plot"

if test_id == "GRAV-16":
    # Double-slit
    requirements["required_plots"].append("interference_pattern_{test_id}.png")
    requirements["required_csvs"].append("intensity_profile_{test_id}.csv")
    requirements["notes"] = "Interference pattern image is THE defining output for this test"

if test_id == "QUAN-10":
    # Bound states
    requirements["required_plots"].extend([
        "wavefunctions_bound_states.png",
        "wavefunction_mode_1.png",
        "wavefunction_mode_2.png",
        "wavefunction_mode_3.png"
    ])
    requirements["notes"] = "Wavefunction plots ψ_n(x) are as critical as energy levels"
```

---

## Appendix A: Complete Output Catalog

### Tier 1 (15 tests)
```
REL-01-02 (Isotropy):
  ✅ summary.json
  ❌ plots/isotropy_comparison_{tid}.png  [TO ADD]
  ❌ diagnostics/frequency_data_{tid}.csv  [TO ADD]

REL-03-04 (Lorentz):
  ✅ summary.json
  ✅ diagnostics/ (various CSVs)

REL-05-06 (Causality pulse/noise):
  ✅ summary.json

REL-07-08 (Phase/Superposition):
  ✅ summary.json

REL-09-10 (3D Isotropy):
  ✅ summary.json
  ❌ plots/isotropy_3d_{tid}.png  [TO ADD]

REL-11-14 (Dispersion):
  ✅ diagnostics/dispersion_measurement.csv
  ✅ diagnostics/projection_series.csv
  ❌ plots/dispersion_curve_{tid}.png  [TO ADD - CRITICAL]

REL-15 (Light-cone):
  ✅ plots/correlation_vs_distance_{tid}.png
  ✅ diagnostics/correlation_data.csv
  ✅ diagnostics/violations.csv (if applicable)

REL-16 (Momentum, if enabled):
  ✅ plots/momentum_conservation_{tid}.png
  ✅ diagnostics/momentum_history.csv
  ✅ diagnostics/momentum_density.csv
```

### Tier 2 (25 tests)
```
GRAV-01-06 (Local frequency):
  ✅ plots/self_consistency_profile_overlay_{tid}.png
  ✅ plots/self_consistency_ratio_{tid}.png
  ✅ diagnostics/self_consistency_profile_{tid}.csv
  ✅ diagnostics/local_freq_profile_{tid}.csv

GRAV-07-10 (Time dilation):
  ✅ diagnostics/detector_signals_{tid}.csv
  ✅ diagnostics/initial_conditions_{tid}.csv
  ✅ diagnostics/field_snapshots_{tid}.csv
  ⚠️ plots/time_dilation_{tid}.png  [OPTIONAL]

GRAV-11-12 (Shapiro delay):
  ✅ diagnostics/packet_tracking_{tid}_serial.csv
  ✅ diagnostics/packet_tracking_{tid}_parallel.csv
  ✅ diagnostics/detector_signal_{tid}_slab.csv
  ✅ diagnostics/detector_signal_{tid}_control.csv
  ✅ diagnostics/envelope_measurement_{tid}.csv
  ✅ diagnostics/packet_analysis_{tid}.csv
  ⚠️ plots/shapiro_trajectory_{tid}.png  [OPTIONAL]

GRAV-13-14 (Dynamic chi):
  ✅ plots/chi_wave_evolution_{tid}.png
  ✅ plots/chi_coupling_{tid}.png
  ✅ diagnostics/chi_wave_history_{tid}.csv
  ✅ diagnostics/chi_evolution_{tid}.csv

GRAV-15 (3D wave):
  ✅ diagnostics/snapshot_{tid}.h5

GRAV-16 (Double-slit):
  ✅ diagnostics/snapshot_{tid}.h5
  ❌ plots/interference_pattern_{tid}.png  [TO ADD - CRITICAL]
  ❌ diagnostics/intensity_profile_{tid}.csv  [TO ADD]

GRAV-17-25 (Various 3D):
  ✅ diagnostics/*.h5 files
  ⚠️ plots/3d_slice_{tid}.png  [OPTIONAL]
```

### Tier 3 (11 tests)
```
ENER-01-02 (Conservation):
  ✅ plots/energy_vs_time.png
  ✅ plots/entropy_vs_time.png
  ✅ diagnostics/energy_trace.csv
  ✅ diagnostics/entropy_trace.csv

ENER-03-04 (Wave integrity):
  ✅ [Same as conservation]

ENER-05-07 (Hamiltonian):
  ✅ plots/hamiltonian_components.png
  ✅ plots/hamiltonian_total.png
  ✅ diagnostics/hamiltonian_components.csv
  ✅ [Plus energy/entropy outputs]

ENER-08-09 (Dissipation):
  ✅ [Energy/entropy showing decay]

ENER-10 (Thermalization):
  ✅ [Energy/entropy showing equilibration]

ENER-11 (Momentum, SKIPPED):
  N/A
```

### Tier 4 (14 tests)
```
QUAN-01-02 (Energy transfer):
  ✅ plots/energy_transfer.png
  ✅ diagnostics/energy_evolution.csv

QUAN-03-04 (Spectral linearity):
  ✅ plots/spectral_linearity.png
  ✅ diagnostics/linearity_test.csv

QUAN-05-06 (Phase-amplitude):
  ✅ plots/linearity_superposition.png
  ✅ diagnostics/linearity.csv

QUAN-07 (Wavefront):
  ✅ plots/wavefront_stability.png
  ✅ diagnostics/wavefront_evolution.csv

QUAN-08 (Blowout):
  ✅ plots/stability_test.png
  ✅ diagnostics/stability_test.csv

QUAN-09 (Uncertainty):
  ✅ plots/uncertainty_dx_dk.png
  ✅ diagnostics/uncertainty_results.csv
  ⚠️ [Could enhance to scatter plot format]

QUAN-10 (Bound states):
  ✅ plots/quantized_energies.png
  ✅ plots/mode_evolution.png
  ✅ diagnostics/eigenvalues.csv
  ❌ plots/wavefunctions_bound_states.png  [TO ADD - CRITICAL]
  ❌ plots/wavefunction_mode_{n}.png  [TO ADD - CRITICAL]

QUAN-11 (Zero-point):
  ✅ plots/zero_point_energy.png
  ✅ diagnostics/zero_point_energy.csv

QUAN-12 (Tunneling):
  ✅ plots/tunneling_transmission.png
  ✅ plots/tunneling_snapshot.png
  ✅ diagnostics/energy_regions.csv
  ⚠️ [Could add log-scale subplot]

QUAN-13 (Duality):
  ✅ plots/wave_particle_duality.png
  ✅ diagnostics/interference_patterns.csv

QUAN-14 (Non-thermalization):
  ✅ plots/non_thermalization.png
  ✅ diagnostics/non_thermalization.csv
```

---

## Appendix B: Visualization Standards

### B.1 Plot Quality Guidelines

All plots should follow these standards:

**Required elements:**
- Title with test ID and physics description
- Axis labels with units (where applicable)
- Legend (if multiple data series)
- Grid (alpha=0.3 for subtlety)
- DPI ≥ 150 for publication quality

**Color scheme:**
- Theory/analytical: Blue
- Measured/numerical: Red
- Reference lines: Black dashed
- Tolerances: Green shaded regions
- Errors: Yellow/orange highlights

**Typography:**
- Title: 14pt
- Axis labels: 12pt
- Legend: 11pt
- Annotations: 10pt

**Example:**
```python
plt.figure(figsize=(10, 6))
plt.plot(x_theory, y_theory, 'b-', linewidth=2, label='Theory')
plt.plot(x_measured, y_measured, 'ro', markersize=8, label='Measured')
plt.xlabel('Position x (lattice units)', fontsize=12)
plt.ylabel('Amplitude |E|', fontsize=12)
plt.title(f'{test_id}: Field Amplitude Comparison\nError = {error*100:.2f}%', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(path, dpi=150)
plt.close()
```

### B.2 CSV Format Guidelines

All CSV files should:
- Have header row with descriptive column names
- Use comma delimiters (not tabs or semicolons)
- Use scientific notation for very small/large numbers (e.g., 1.234e-06)
- Maintain consistent precision (typically 6 significant figures)
- No empty rows between data

**Example:**
```csv
time,energy,relative_drift
0.000000,1.234567e+00,0.000000e+00
0.001000,1.234560e+00,5.667891e-06
0.002000,1.234552e+00,1.215678e-05
```

---

## Appendix C: Post-Implementation Validation

After implementing all recommended outputs, verify:

1. **Run all tier suites:**
   ```powershell
   python run_tier1_relativistic.py
   python run_tier2_gravityanalogue.py
   python run_tier3_energy.py
   python run_tier4_quantization.py
   ```

2. **Check output directories:**
   ```powershell
   # Verify structure for each test
   ls results/Tier1/REL-11/plots/  # Should see dispersion_curve_REL-11.png
   ls results/Tier2/GRAV-16/plots/  # Should see interference_pattern_GRAV-16.png
   ls results/Tier4/QUAN-10/plots/  # Should see wavefunctions_bound_states.png
   ```

3. **Visual inspection:**
   - Open each new plot and verify it shows expected physics
   - Check for proper labels, legends, annotations
   - Verify color scheme matches standards

4. **CSV integrity:**
   - Open CSV files in spreadsheet or pandas
   - Verify no NaN or Inf values (unless expected)
   - Check column headers are descriptive

5. **Summary metadata:**
   - Check `summary.json` includes references to new outputs (if applicable)
   - Verify metrics are updated correctly

6. **Test pass/fail:**
   - Confirm all tests still pass (no broken logic from new code)
   - Check MASTER_TEST_STATUS.csv for any regressions

---

## END OF ANALYSIS

**Total tests analyzed:** 65  
**Critical missing outputs identified:** 3 (GRAV-16, REL-11-14, QUAN-10)  
**Medium priority enhancements:** 2 (Isotropy, Uncertainty)  
**Low priority enhancements:** 3-4 (various optional improvements)  

**Estimated implementation time:** 2-3 weeks for critical + medium priority items  
**Confidence level:** HIGH (analysis based on comprehensive code inspection)
