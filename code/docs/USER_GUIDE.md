# LFM User Guide

**For:** Researchers and users running LFM simulations  
**Prerequisites:** Basic knowledge of wave physics, command-line usage  
**License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
**Author:** Greg D. Partin | LFM Research  
**Contact:** latticefieldmediumresearch@gmail.com  
**DOI:** [10.5281/zenodo.17478758](https://zenodo.org/records/17478758)  
**Repository:** [OSF: osf.io/6agn8](https://osf.io/6agn8)  
**Last Updated:** 2025-11-01

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding Test Tiers](#understanding-test-tiers)
3. [Running Tests](#running-tests)
4. [Interpreting Results](#interpreting-results)
5. [Configuration Guide](#configuration-guide)
6. [Output Files](#output-files)
7. [Visualization](#visualization)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## Quick Start

### Your First Simulation

```bash
# Navigate to code directory
cd c:/LFM/code

# Run a single test (completes in ~5 seconds)
python run_tier1_relativistic.py --test REL-01

# View results
ls results/Relativistic/REL-01/
# → summary.json, run_log.txt, plots/isotropy_comparison.png
```

### Understanding the Output

```bash
# Open summary file
cat results/Relativistic/REL-01/summary.json
```

```json
{
  "test_id": "REL-01",
  "description": "Isotropy — Coarse Grid",
  "status": "PASS",
  "metrics": {
    "isotropy_cov": 0.0021,        // Directional variation (<1% = isotropic)
    "omega_x": 0.3142,              // Frequency in x-direction
    "omega_y": 0.3145,              // Frequency in y-direction  
    "peak_cpu_percent": 45.2,       // Resource usage
    "peak_memory_mb": 234.5
  },
  "timestamp": "2025-11-01T10:30:45"
}
```

**Status meanings:**
- `PASS`: Test succeeded within tolerances
- `FAIL`: Exceeded error threshold (may be physics issue or numerical artifact)
- `ERROR`: Runtime error (check `run_log.txt`)

---

## Understanding Test Tiers

LFM organizes tests into 4 validation tiers:

### Tier 1: Relativistic Propagation (15 tests)
**Purpose:** Validate Special Relativity analogues

**Tests:**
- **Isotropy (REL-01, REL-02):** Wave speed independent of direction
- **Lorentz Boost (REL-03, REL-04):** Field transforms correctly under boosts
- **Causality (REL-05, REL-06, REL-15):** No faster-than-light propagation
- **Phase Independence (REL-07):** Phase doesn't affect propagation
- **Superposition (REL-08):** Linear wave equation verified
- **3D Isotropy (REL-09, REL-10):** Spherical symmetry in 3D
- **Dispersion Relations (REL-11–14):** Klein-Gordon ω²=k²+χ² validated

**Example:**
```bash
# Run full Tier 1 suite (~3 minutes)
python run_tier1_relativistic.py

# Results: results/Relativistic/REL-01/ through REL-15/
```

### Tier 2: Gravity Analogue (25 tests)
**Purpose:** Simulate gravitational effects via χ-field curvature

**Tests:**
- **Local Frequency (GRAV-01–06):** ω ∝ χ in potential wells
- **Time Dilation (GRAV-07–10):** Frequency shift in curved spacetime
- **Time/Phase Delay (GRAV-11–12):** Shapiro-like delays through χ-slabs
- **Redshift (GRAV-10, GRAV-17–19):** Climbing out of potential wells
- **3D Visualization (GRAV-15–16):** Volumetric wave propagation, double-slit interference
- **Self-Consistent χ (GRAV-20):** Energy → curvature feedback
- **GR Calibration (GRAV-21–22):** Map to effective Newtonian G
- **Gravitational Waves (GRAV-23–24):** χ-wave propagation
- **Light Bending (GRAV-25):** Ray deflection through gradients

**Example:**
```bash
# Run time-dilation test
python run_tier2_gravityanalogue.py --test GRAV-07

# View time series data
cat results/Gravity/GRAV-07/diagnostics/probe_serial.csv
```

### Tier 3: Energy Conservation (11 tests)
**Purpose:** Validate Hamiltonian structure and thermodynamics

**Tests:**
- **Global Conservation (ENER-01–02):** Total energy conserved (undamped)
- **Wave Integrity (ENER-03–04):** Energy maintained through curvature
- **Hamiltonian Partitioning (ENER-05–07):** KE ↔ GE ↔ PE exchange
- **Dissipation (ENER-08–09):** Exponential decay with damping
- **Thermalization (ENER-10):** Noise + damping → steady state
- **Momentum Conservation (ENER-11):** Noether's theorem validation

**Example:**
```bash
# Run long energy conservation test
python run_tier3_energy.py --test ENER-02

# Plot energy drift
python -c "
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('results/Energy/ENER-02/diagnostics/energy_log.csv')
plt.plot(df['time'], df['E_total'])
plt.xlabel('Time')
plt.ylabel('Energy')
plt.show()
"
```

### Tier 4: Quantization (14 tests)
**Purpose:** Demonstrate quantum-like behavior in discrete spacetime

**Tests:**
- **ΔE Transfer (QUAN-01–02):** Energy quantization
- **Spectral Linearity (QUAN-03–04):** Linear dispersion preservation
- **Phase-Amplitude Coupling (QUAN-05–06):** Nonlinear stability
- **Heisenberg Uncertainty (QUAN-09):** Δx·Δk ≥ 1/2
- **Bound States (QUAN-10):** Discrete energy eigenvalues
- **Zero-Point Energy (QUAN-11):** Vacuum fluctuations (E₀ ≠ 0)
- **Tunneling (QUAN-12):** Barrier penetration
- **Wave-Particle Duality (QUAN-13):** Which-way information destroys interference
- **Non-Thermalization (QUAN-14):** Klein-Gordon doesn't approach Planck distribution

**Example:**
```bash
# Run bound-state test
python run_tier4_quantization.py --test QUAN-10

# View eigenstate energies
cat results/Quantization/QUAN-10/diagnostics/eigenstate_energies.csv
```

---

## Running Tests

### Command-Line Interface

**Basic usage:**
```bash
# Run full tier suite
python run_tier1_relativistic.py

# Run single test
python run_tier1_relativistic.py --test REL-01

# Quick mode (reduced resolution, faster)
python run_tier1_relativistic.py --quick

# Force GPU acceleration
python run_tier1_relativistic.py --gpu
```

**Combining flags:**
```bash
python run_tier2_gravityanalogue.py --test GRAV-16 --gpu
```

### Quick Mode

Quick mode reduces grid size and step count for rapid verification:

```bash
# Normal mode: N=256, steps=4000 (~2 minutes)
python run_tier1_relativistic.py --test REL-11

# Quick mode: N=128, steps=1000 (~15 seconds)
python run_tier1_relativistic.py --test REL-11 --quick
```

**When to use quick mode:**
- Debugging configuration changes
- Verifying installation
- Rapid iteration during development

**When NOT to use quick mode:**
- Final validation runs
- Publication-quality results
- Benchmarking performance

### GPU Acceleration

**Check GPU availability:**
```bash
python -c "import cupy as cp; print(f'GPU: {cp.cuda.Device().name}')"
```

**Enable GPU in config:**
```json
{
  "run_settings": {
    "use_gpu": true
  }
}
```

**Or use command-line flag:**
```bash
python run_tier1_relativistic.py --gpu
```

**Performance comparison** (typical):
| Test | CPU Time | GPU Time | Speedup |
|------|----------|----------|---------|
| REL-11 (1D, 4000 steps) | 8s | 2s | 4x |
| GRAV-16 (3D, 2000 steps) | 180s | 12s | 15x |
| ENER-02 (2D, 10000 steps) | 45s | 5s | 9x |

---

## Interpreting Results

### Summary Metrics

Every test produces `summary.json` with standard structure:

```json
{
  "tier": 1,
  "category": "Relativistic",
  "test_id": "REL-11",
  "description": "Dispersion Relation — Non-relativistic (χ/k≈10)",
  "timestamp": "2025-11-01T10:30:45",
  "status": "PASS",
  
  "metrics": {
    // Test-specific metrics
    "frequency_error": 0.00205,    // 0.2% error (excellent)
    "omega_measured": 0.20164,
    "omega_theory": 0.20123,
    
    // Resource usage (always present)
    "peak_cpu_percent": 42.1,
    "peak_memory_mb": 456.2,
    "peak_gpu_memory_mb": 0.0,     // 0 if CPU-only
    "runtime_sec": 8.34
  }
}
```

### Understanding Errors

**Frequency Error (Dispersion Tests):**
```
error = |ω_measured - ω_theory| / ω_theory
```
- **<1%:** Excellent agreement
- **1-2%:** Good (typical for discrete systems)
- **2-5%:** Acceptable (numerical dispersion or finite-size effects)
- **>5%:** Investigate (may indicate bug or config issue)

**Energy Drift (Conservation Tests):**
```
drift = |E_final - E_initial| / E_initial
```
- **<0.01%:** Machine precision limit (perfect conservation)
- **0.01-1%:** Excellent for long runs
- **1-5%:** Acceptable (check for leaking boundaries)
- **>5%:** Problem (check CFL, boundary conditions, precision)

**Isotropy Coefficient of Variation:**
```
CoV = σ / μ  (standard deviation / mean of directional frequencies)
```
- **<0.5%:** Highly isotropic
- **0.5-2%:** Acceptable anisotropy (grid effects)
- **>2%:** Investigate (may indicate bug in Laplacian)

### Common Pass/Fail Reasons

**PASS:**
- All metrics within configured tolerances
- No NaN/Inf values
- Energy conserved (if expected)
- Matches theoretical predictions

**FAIL:**
- Frequency error exceeds threshold
- Energy drift too large
- CFL stability violated
- Packet tracking lost
- Signal too weak for measurement

**ERROR:**
- Python exception during execution
- Out of memory (CPU or GPU)
- Invalid configuration
- Missing input files

---

## Configuration Guide

### Editing Configuration Files

Configs are in `config/` directory as JSON files:

```bash
# Edit Tier 1 config
notepad config/config_tier1_relativistic.json

# Or use any text editor
code config/config_tier1_relativistic.json
```

### Key Parameters

**Time/Space Discretization:**
```json
{
  "parameters": {
    "dt": 0.01,    // Time step (smaller = more accurate, slower)
    "dx": 0.1      // Spatial step (smaller = higher resolution, slower)
  }
}
```

**CFL Stability Rule:**  
Must satisfy: `c·dt/dx ≤ 1/√D` where D=dimensions

Examples:
- 1D: `dt/dx ≤ 1.0`
- 2D: `dt/dx ≤ 0.707`
- 3D: `dt/dx ≤ 0.577`

**Wave Speed:**
```json
{
  "parameters": {
    "alpha": 1.0,    // Numerator
    "beta": 1.0,     // Denominator
    "c": 1.0         // Effective c² = alpha/beta
  }
}
```

**Mass Parameter (χ):**
```json
{
  "parameters": {
    "chi": 0.0     // 0 = massless (wave equation)
                   // >0 = massive (Klein-Gordon)
  }
}
```

**Damping:**
```json
{
  "parameters": {
    "gamma_damp": 0.0    // 0 = undamped (conservative)
                         // >0 = exponential decay
  }
}
```

**Boundary Conditions:**
```json
{
  "parameters": {
    "boundary": "periodic"    // "periodic" or "absorbing"
  }
}
```
- `periodic`: Waves wrap around (ideal for plane waves)
- `absorbing`: Waves decay at edges (reduces reflections for packets)

**Numerical Accuracy:**
```json
{
  "parameters": {
    "stencil_order": 2,     // 2, 4, or 6 (higher = more accurate, slower)
    "precision": "float64"  // "float64" or "float32"
  }
}
```

### Modifying Test Variants

To change parameters for a specific test:

```json
{
  "variants": [
    {
      "test_id": "REL-11",
      "description": "Dispersion Relation — Non-relativistic",
      "k_fraction": 0.02,    // Wavenumber as fraction of grid Nyquist
      "chi": 0.2,            // Mass parameter
      "steps": 4000,         // Time steps (override default)
      "N": 256               // Grid points (override default)
    }
  ]
}
```

---

## Output Files

### Directory Structure

```
results/
├── MASTER_TEST_STATUS.csv          # Overall status summary
├── test_metrics_history.json       # Historical run database
└── <Category>/                     # Relativistic, Gravity, Energy, Quantization
    └── <TEST-ID>/
        ├── summary.json            # Test results + metrics
        ├── run_log.txt             # Human-readable log
        ├── run_log.jsonl           # Structured event log
        ├── diagnostics/            # Data files
        │   ├── energy_log.csv
        │   ├── probe_serial.csv
        │   ├── packet_tracking_serial.csv
        │   └── field_snapshots_3d_*.h5
        └── plots/                  # Visualizations
            ├── dispersion_*.png
            ├── interference_pattern_*.png
            └── energy_drift.png
```

### File Formats

**summary.json** (Always present)
```json
{
  "tier": 1,
  "category": "Relativistic",
  "test_id": "REL-11",
  "description": "...",
  "timestamp": "2025-11-01T10:30:45",
  "status": "PASS",
  "metrics": { /* test-specific */ }
}
```

**energy_log.csv** (If energy monitoring enabled)
```csv
step,time,E_total,E_kinetic,E_gradient,E_potential,drift
0,0.0,1.000000e+00,5.0e-01,5.0e-01,0.0,0.0
100,1.0,9.999876e-01,4.99e-01,5.01e-01,0.0,1.24e-05
```

**probe_serial.csv** (Time-dilation tests)
```csv
step,E_real,E_imag,time
0,0.0,0.0,0.0
1,0.0342,0.0113,0.01
```

**field_snapshots_3d_*.h5** (3D visualization tests)
```
HDF5 file with datasets:
/snapshots/0000, /snapshots/0001, ..., /snapshots/0160
Each dataset: (N, N, N) float64 array
```

---

## Visualization

### Built-in Plots

Most tests automatically generate plots in `plots/` directory:

**Dispersion curves** (REL-11–14):
```bash
# View dispersion plot
start results/Relativistic/REL-11/plots/dispersion_REL-11.png
```
Shows measured vs theoretical ω(k) relationship.

**Interference patterns** (GRAV-16):
```bash
start results/Gravity/GRAV-16/plots/interference_pattern_GRAV-16.png
```
Shows double-slit interference intensity.

**Energy drift** (ENER-01–02):
```bash
start results/Energy/ENER-02/plots/energy_drift.png
```
Shows energy conservation over time.

### Custom Visualization Scripts

**3D energy dispersion** (GRAV-15):
```bash
python tools/visualize/visualize_grav15_3d.py
# Output: results/Gravity/GRAV-15/energy_dispersion_3d.mp4
```

**Double-slit 3D** (GRAV-16):
```bash
python tools/visualize/visualize_grav16_doubleslit.py
# Output: results/Gravity/GRAV-16/double_slit_3d.mp4
```

**Hamiltonian partitioning** (ENER-05–07):
```bash
python tools/visualize/visualize_hamiltonian.py --tests ENER-05 ENER-06
# Output: KE/GE/PE evolution plots
```

### Creating Custom Plots

**Example: Plot frequency spectrum**

```python
import numpy as np
import matplotlib.pyplot as plt

# Load probe data
data = np.loadtxt('results/Gravity/GRAV-07/diagnostics/probe_serial.csv', 
                  delimiter=',', skiprows=1)
times = data[:, 3]
E_real = data[:, 1]

# FFT
fft = np.fft.fft(E_real)
freqs = np.fft.fftfreq(len(times), times[1] - times[0])
power = np.abs(fft)**2

# Plot
plt.figure(figsize=(10, 6))
plt.plot(freqs[freqs > 0], power[freqs > 0])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Frequency Spectrum')
plt.yscale('log')
plt.savefig('custom_spectrum.png')
plt.show()
```

---

## Troubleshooting

### Test Hangs or Runs Very Slowly

**Possible causes:**
1. Large grid size (N≥512)
2. Many time steps (>100,000)
3. CPU-only on large 3D problem

**Solutions:**
```bash
# Use GPU
python run_tier2_gravityanalogue.py --test GRAV-16 --gpu

# Use quick mode
python run_tier2_gravityanalogue.py --test GRAV-16 --quick

# Monitor progress (enable in config)
{
  "run_settings": {
    "show_progress": true,
    "progress_percent_stride": 5
  }
}
```

### Out of Memory Errors

**Symptoms:**
- Python crashes with `MemoryError`
- `CuPy.OutOfMemoryError` (GPU)

**Solutions:**
1. Reduce grid size in config:
   ```json
   {"parameters": {"N": 128}}  // Instead of 256
   ```

2. Use CPU instead of GPU:
   ```json
   {"run_settings": {"use_gpu": false}}
   ```

3. Close other applications

4. Skip large 3D tests (GRAV-15, GRAV-16)

### Unexpected Test Failures

**Check logs:**
```bash
cat results/<Category>/<TEST-ID>/run_log.txt
```

**Common issues:**

1. **CFL Violation:**
   ```
   ERROR: CFL violated: c*dt/dx = 1.2 > 0.577
   ```
   Solution: Reduce `dt` or increase `dx` in config

2. **NaN/Inf Values:**
   ```
   ERROR: Field contains NaN values at step 147
   ```
   Causes: CFL violation, numerical instability, extreme parameters
   Solution: Check CFL, reduce `dt`, increase `dx`

3. **Signal Too Weak:**
   ```
   WARNING: Peak amplitude below threshold (SNR < 10)
   ```
   Solution: Increase initial amplitude or reduce damping

### Results Don't Match Expected Values

**Debugging steps:**

1. Check configuration matches documented test design
2. Verify CFL stability: `c·dt/dx ≤ 1/√D`
3. Compare to known-good results (use git history)
4. Enable full diagnostics:
   ```json
   {
     "debug": {
       "enable_diagnostics": true,
       "check_nan": true
     }
   }
   ```
5. Run in isolation: `--test <TEST-ID>`
6. Check for warnings in `run_log.txt`

---

## FAQ

### Q: How long do test suites take to run?

**A:** Typical run times (CPU, standard configs):
- Tier 1: ~3 minutes (15 tests)
- Tier 2: ~10 minutes (25 tests)
- Tier 3: ~5 minutes (11 tests)
- Tier 4: ~8 minutes (14 tests)

GPU acceleration reduces this by 5-15x for 3D tests.

### Q: Can I run multiple tests in parallel?

**A:** Not currently supported. Run tests sequentially:
```bash
python run_tier1_relativistic.py
python run_tier2_gravityanalogue.py
# etc.
```

### Q: How do I cite LFM in my research?

**A:** Use this BibTeX entry:
```bibtex
@software{lfm_simulator,
  author = {Partin, Greg D.},
  title = {LFM: Lattice Field Medium Simulator},
  year = {2025},
  publisher = {LFM Research},
  license = {CC BY-NC-ND 4.0},
  doi = {10.5281/zenodo.17478758},
  url = {https://zenodo.org/records/17478758},
  repository = {https://osf.io/6agn8}
}
```

### Q: Can I use LFM for commercial purposes?

**A:** No. LFM is licensed under CC BY-NC-ND 4.0 (non-commercial only). Contact latticefieldmediumresearch@gmail.com for commercial licensing inquiries.

### Q: What Python version is required?

**A:** Python 3.9 or higher. Tested on 3.9, 3.10, 3.11, 3.12.

### Q: Do I need a GPU?

**A:** No. All tests run on CPU (NumPy). GPU (CuPy) is optional for performance.

### Q: Which tests are known to fail?

**A:** 4 tests have documented issues (not bugs):
- **GRAV-07**: Packet trapped in double-well (physics, not failure)
- **GRAV-09**: Grid dispersion at high resolution
- **GRAV-11**: Packet tracking measurement issues
- **GRAV-14**: Signal too weak for differential timing

See `results/MASTER_TEST_STATUS.csv` for details.

### Q: Can I create custom tests?

**A:** Yes! See `docs/DEVELOPER_GUIDE.md` section "Adding New Tests".

### Q: Where can I get help?

**A:** 
1. Check this User Guide
2. Read `docs/DEVELOPER_GUIDE.md` for technical details
3. Check `docs/INSTALL.md` for setup issues
4. Email latticefieldmediumresearch@gmail.com with questions or bug reports

---

## Next Steps

After mastering the basics:

1. **Explore advanced features:**
   - Custom χ-field profiles
   - Multi-packet scenarios
   - Long-run energy conservation

2. **Read technical documentation:**
   - `docs/DEVELOPER_GUIDE.md` — Architecture and internals
   - `docs/API_REFERENCE.md` — Function documentation
   - Research papers (coming soon)

3. **Contribute:**
   - Report bugs and suggest features
   - Improve documentation
   - Add new tests

---

**Happy Simulating!** 🌊

For questions or feedback, contact: latticefieldmediumresearch@gmail.com

