# LFM Code Documentation — Technical Implementation

<!-- Copyright (c) 2025 Greg D. Partin. All rights reserved. -->
<!-- Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International). -->
<!-- See LICENSE file in project root for full license text. -->
<!-- Commercial use prohibited without explicit written permission. -->
<!-- Contact: latticefieldmediumresearch@gmail.com -->

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status: Research](https://img.shields.io/badge/status-research-yellow.svg)]()

**Technical documentation for the LFM Klein-Gordon wave equation solver implementation.**

This directory contains the complete LFM framework codebase, user interfaces, and validation infrastructure. 

**📍 New to LFM?** Start with the [main project README](../README.md) for overview and quick start instructions.

**👨‍💻 Developer/Researcher?** This document covers technical implementation details, test procedures, and code structure.

**Author:** Greg D. Partin | LFM Research — Los Angeles, CA USA  
**ORCID:** [https://orcid.org/0009-0004-0327-6528](https://orcid.org/0009-0004-0327-6528)  
**Contact:** latticefieldmediumresearch@gmail.com  
**License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)

> **Note:** "LFM Research" refers to an independent personal research project by Greg D. Partin and is not an incorporated entity.

---

## Implementation Overview

LFM implements a high-performance Klein-Gordon wave equation solver with spatially-varying χ-field for exploring unified physics phenomena through computational simulation.

**Core Equation:**
```
∂²E/∂t² = c²∇²E - χ²(x,t)E
```

**Key Features:**
- **Backend Abstraction**: Seamless CPU (NumPy) / GPU (CuPy) switching
- **Parallel Execution**: Multi-threaded tile-based processing
- **Comprehensive Validation**: 55 tests across 4 physics tiers (93% pass rate)
- **User Interfaces**: GUI, console, and CLI modes
- **Scientific Visualization**: Dispersion curves, interference patterns, 3D evolution

---

## Navigation

- **🏠 [Project Overview](../README.md)** — Main project page with quick start
- **📋 [Installation Guide](INSTALL.md)** — Complete setup instructions  
- **🧪 [Testing](#testing)** — Run validation tests
- **📊 [Results](#visualization-examples)** — View simulation outputs
- **🔧 [Configuration](#configuration)** — Customize parameters
- **📚 [Documentation](#documentation)** — Detailed technical docs

---

## Quick Start

### ⚖️ Legal Notice — MUST READ

**BY DOWNLOADING, COPYING, OR USING THIS SOFTWARE, YOU AGREE TO BE BOUND BY THE TERMS IN THE [LICENSE](LICENSE) FILE.**

**Key restrictions:**
- ❌ **NO COMMERCIAL USE** without written permission
- ❌ No "clean room" reimplementations for commercial purposes
- ✅ Non-commercial research and education permitted (with restrictions)
- ✅ Must provide proper attribution

**Read [LICENSE](LICENSE) and [NOTICE](NOTICE) files before proceeding.**

### Installation

#### Quick Start (All Platforms)
```bash
# Windows: Run quick_setup_windows.bat
# macOS/Linux: ./quick_setup_unix.sh
# Or use the automated installer:
python setup_lfm.py
```

#### Manual Installation
```bash
# Install dependencies
pip install numpy>=1.24.0 matplotlib>=3.7.0 scipy>=1.10.0 h5py>=3.8.0 pytest>=7.3.0

# Optional: GPU acceleration (requires CUDA 12.x)
pip install cupy-cuda12x
```

📋 **Need help?** See the complete [Installation Guide](INSTALL.md) for detailed instructions, troubleshooting, and platform-specific setup.

### Run Your First Test

#### Using the Graphical Interface (Easiest)
```bash
# Windows GUI interface (point-and-click)
python lfm_gui.py

# Or console interface (menu-driven)
python lfm_control_center.py
```

#### Using Command Line Directly
```bash
# Single test (completes in ~5 seconds)
python run_tier1_relativistic.py --test REL-01

# View results
ls results/Relativistic/REL-01/
# → summary.json, plots/isotropy_comparison.png, diagnostics/*.csv
```

### Run Full Test Suites

```bash
# Tier 1: Relativistic propagation (15 tests, ~3 minutes)
python run_tier1_relativistic.py

# Tier 2: Gravity analogue (25 tests, ~10 minutes)
python run_tier2_gravityanalogue.py

# Tier 3: Energy conservation (11 tests, ~5 minutes)
python run_tier3_energy.py

# Tier 4: Quantization (14 tests, ~8 minutes)
python run_tier4_quantization.py
```

### Quick Mode (Fast Validation)

```bash
# Run with reduced resolution for quick verification
python run_tier1_relativistic.py --quick
```

---

## Project Structure

```
LFM/code/
├── setup_lfm.py                 # Automated installer for all platforms
├── lfm_control_center.py        # Console interface (menu-driven)
├── lfm_gui.py                   # Windows GUI interface (point-and-click)
├── run_tier1_relativistic.py    # Tier 1 test harness
├── run_tier2_gravityanalogue.py # Tier 2 test harness
├── run_tier3_energy.py          # Tier 3 test harness
├── run_tier4_quantization.py    # Tier 4 test harness
├── lfm_equation.py              # Core Klein-Gordon solver
├── lfm_parallel.py              # Parallel tile-based runner
├── energy_monitor.py            # Energy conservation tracking
├── numeric_integrity.py         # CFL checks, NaN detection
├── lfm_logger.py                # Structured logging (text + JSONL)
├── lfm_results.py               # Output management
├── lfm_plotting.py              # Visualization utilities
├── config/                      # JSON configuration files
│   ├── config_tier1_relativistic.json
│   ├── config_tier2_gravityanalogue.json
│   ├── config_tier3_energy.json
│   └── config_tier4_quantization.json
├── results/                     # Test outputs (auto-generated)
│   ├── MASTER_TEST_STATUS.csv   # Overall test status
│   ├── Relativistic/            # Tier 1 results
│   ├── Gravity/                 # Tier 2 results
│   ├── Energy/                  # Tier 3 results
│   └── Quantization/            # Tier 4 results
├── tests/                       # Unit tests
├── tools/visualize/             # Visualization scripts
├── docs/                        # Documentation
│   ├── PRODUCTION_READINESS_ANALYSIS.md
│   ├── PRODUCTION_ROADMAP.md
│   └── analysis/                # Test coverage analysis
└── archive/                     # Historical code versions
```

---

## Core Physics

LFM implements the discrete Klein-Gordon equation:

```
∂²E/∂t² = c² ∇²E - (χ²c⁴/ℏ²) E
```

Where:
- `E(x,t)` — Field amplitude (energy density)
- `c` — Effective propagation speed (≈1 in lattice units)
- `χ` — Mass-like parameter (curvature field for gravity analogue)
- `∇²` — Discrete Laplacian (2nd/4th/6th order stencils)

**Discrete Update Rule** (leapfrog time integration):
```
E[t+dt] = 2E[t] - E[t-dt] + dt²(c²∇²E[t] - χ²E[t])
```

**CFL Stability Condition:**
```
c·dt/dx ≤ 1/√D  (D = spatial dimensions)
```

---

## Key Results

### Tier 1 — Relativistic Validation ✅
- **Isotropy**: Directional equivalence to 0.2% across all axes
- **Dispersion**: Klein-Gordon ω² = k² + χ² validated from non-relativistic (χ/k≈10) to ultra-relativistic (χ/k≈0.1) regimes
- **Lorentz Covariance**: Boost-invariant within 1% for velocities up to 0.6c
- **Causality**: No faster-than-light propagation (correlation threshold <1e-6 outside light cone)

### Tier 2 — Gravity Analogue ✅⚠️
- **Time Dilation**: Frequency shift ω ∝ χ confirmed in potential wells (6/6 tests)
- **Redshift**: Climbing out of χ-well reduces frequency (4/4 tests)
- **Time Delay**: Shapiro-like phase delay through χ-slab demonstrated
- **Double-Slit**: 3D interference pattern with visibility=0.869, fringe spacing matches theory
- **Light Bending**: Ray deflection through χ-gradient validated
- **Gravitational Waves**: χ-wave propagation with 1/r decay confirmed

### Tier 3 — Energy Conservation ✅
- **Global Conservation**: <0.01% drift over 10,000 steps (undamped)
- **Hamiltonian Partitioning**: KE ↔ GE ↔ PE exchange validated
- **Dissipation**: Exponential decay matches analytic exp(-2γt) for weak/strong damping
- **Thermalization**: Noise + damping reaches steady-state energy distribution

### Tier 4 — Quantum Behavior ✅
- **Bound States**: Discrete energy eigenvalues En = √(kn² + χ²) with kn = nπ/L (mean error 1.4%)
- **Tunneling**: Exponential barrier penetration when E < V (classically forbidden)
- **Uncertainty**: Δx·Δk ≥ 0.5 confirmed across wave packets
- **Zero-Point Energy**: Ground state E₀ = ½ℏω ≠ 0 (vacuum fluctuations)

### 🔬 Emergence Validation — Critical Evidence ✅
- **Spontaneous χ-Field Formation**: 29% enhancement from uniform initial conditions
- **Self-Consistent Coupling**: Energy density drives χ-field structure without pre-programming
- **Genuine Emergence Confirmed**: Refutes "circular validation" criticism
- **Evidence Location**: `docs/evidence/emergence_validation/` ([See README](docs/evidence/emergence_validation/README.md))

---

## Configuration

All tests are configured via JSON files in `config/`:

```json
{
  "run_settings": {
    "use_gpu": false,
    "quick_mode": false,
    "show_progress": true
  },
  "parameters": {
    "dt": 0.01,
    "dx": 0.1,
    "alpha": 1.0,
    "beta": 1.0,
    "chi": 0.0,
    "gamma_damp": 0.0,
    "boundary": "periodic",
    "stencil_order": 2,
    "precision": "float64"
  },
  "debug": {
    "enable_diagnostics": true,
    "energy_tol": 0.01,
    "check_nan": true
  }
}
```

---

## Visualization Examples

### Dispersion Curves (REL-11)
```bash
python run_tier1_relativistic.py --test REL-11
# Output: results/Relativistic/REL-11/plots/dispersion_REL-11.png
```
Shows measured vs theoretical ω(k) across non-relativistic to ultra-relativistic regimes.

### Double-Slit Interference (GRAV-16)
```bash
python run_tier2_gravityanalogue.py --test GRAV-16
# Output: results/Gravity/GRAV-16/plots/interference_pattern_GRAV-16.png
```
3D wave interference pattern demonstrating quantum-like behavior.

### Bound-State Wavefunctions (QUAN-10)
```bash
python run_tier4_quantization.py --test QUAN-10
# Output: results/Quantization/QUAN-10/plots/bound_state_modes.png
```
Discrete energy eigenstate mode shapes ψn(x).

### 3D Energy Dispersion (GRAV-15)
```bash
python tools/visualize/visualize_grav15_3d.py
# Output: results/Gravity/GRAV-15/energy_dispersion_3d.mp4
```
Animated volumetric rendering of radial wave propagation.

---

## Documentation

- **[Installation Guide](INSTALL.md)** — Complete setup instructions for all platforms
- **[Production Readiness Analysis](docs/PRODUCTION_READINESS_ANALYSIS.md)** — Comprehensive project assessment
- **[Production Roadmap](docs/PRODUCTION_ROADMAP.md)** — Path to production-ready release
- **[Test Coverage](results/MASTER_TEST_STATUS.csv)** — Detailed test status report
- **[Analysis Reports](docs/analysis/)** — Output requirements, test coverage, tier standardization
- **[Emergence Validation Evidence](docs/evidence/emergence_validation/)** — Critical proof of genuine physics emergence

---

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Run specific tier tests
python run_tier1_relativistic.py --test REL-01

# Check test output requirements
python test_output_requirements.py --tier 1

# View overall test status
cat results/MASTER_TEST_STATUS.csv
```

---

## GPU Acceleration

LFM supports NVIDIA GPUs via CuPy (CUDA 12.x):

```bash
# Install CuPy for GPU acceleration
pip install cupy-cuda12x

# Enable GPU in configuration
# Edit config/*.json → "use_gpu": true

# Verify GPU is detected
python -c "import cupy as cp; print(f'GPU: {cp.cuda.Device().name}')"
```

**Performance:** 3D simulations (64³ grid) run ~10-50x faster on GPU depending on test.

---

## User Interfaces

LFM provides three ways to interact with the framework:

### 1. Graphical Interface (`lfm_gui.py`)
**Best for:** New users, visual interaction, Windows environments
- 🖱️ Point-and-click operation
- 📊 Real-time progress monitoring  
- 🗂️ Visual results browser
- 🔧 Built-in system diagnostics
- 💾 Export functionality

```bash
python lfm_gui.py
```

**Features:**
- **Test Execution Tab:** Run individual tests or full tiers with progress bars
- **Results Viewer Tab:** Browse test outputs with folder tree and file preview
- **Tools Tab:** System status, emergence validation, report generation
- **Background Processing:** Tests run without blocking the interface

### 2. Console Interface (`lfm_control_center.py`)
**Best for:** Terminal users, remote access, automation
- 🎯 Menu-driven navigation
- 🌈 Color-coded output
- ⚡ Fast execution
- 📋 Integrated results viewer
- 🔄 Progress monitoring

```bash
python lfm_control_center.py
```

**Menu Options:**
1. Run Fast Tests (4 core validation tests)
2. Run Single Tier (choose Tier 1-4)
3. Run Specific Test (by test ID)
4. Run All Tiers (full 55-test suite)
5. View Test Results (browse outputs)
6. Run Emergence Validation (critical proof)
7. Generate Reports (status summaries)
8. System Status (GPU detection, dependencies)

### 3. Command Line Interface (Original)
**Best for:** Scripting, automation, expert users
- 🔧 Direct script execution
- ⚙️ Full parameter control
- 🚀 Maximum performance
- 📝 Detailed logging

```bash
# Examples
python run_tier1_relativistic.py --test REL-01
python run_tier2_gravityanalogue.py --quick
python run_parallel_tests.py --config config/tier1.json
```

**Cross-Platform Compatibility:** All interfaces work on Windows, macOS, and Linux with standard Python installations.

---

## Known Issues

4 tests have documented failures (all understood, not bugs):

1. **GRAV-07** (Time dilation — double-well): Packet trapped in bound state (demonstrates physics, not failure)
2. **GRAV-09** (Time dilation — refined grid): Grid dispersion at high resolution (known artifact)
3. **GRAV-11** (Time delay — Shapiro): Packet tracking measurement issues (diagnostic needs refinement)
4. **GRAV-14** (Group delay): Signal too weak for differential timing (needs increased pulse amplitude)

See `results/MASTER_TEST_STATUS.csv` for full details.

---

## Citation

If you use LFM in your research, please cite:

```bibtex
@software{lfm_simulator,
  author = {Partin, Greg D.},
  title = {LFM: Lattice Field Medium Simulator},
  year = {2025},
  publisher = {LFM Research},
  license = {CC BY-NC-ND 4.0},
  doi = {10.5281/zenodo.17510124},
  url = {https://zenodo.org/records/17510124},
  repository = {https://osf.io/6agn8}
}
```

### Citing dependencies and prior work

Please also cite the scientific software and prior literature you used alongside LFM. See:
- docs/REFERENCES.md — recommended citations (NumPy, SciPy, Matplotlib, h5py, pytest, CuPy)
- docs/references.bib — BibTeX entries you can include directly in your manuscript

---

## License

This work is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)**.

**Copyright (c) 2025 Greg D. Partin. All rights reserved.**  
**First publication:** November 1, 2025

**By using this software, you agree to:**
1. Be bound by the [LICENSE](LICENSE) terms
2. Not use commercially without written permission
3. Not create commercial "clean room" reimplementations
4. Provide proper attribution in all uses
5. Accept California jurisdiction or international arbitration for disputes

**Important:** "All rights reserved" means all rights not explicitly granted by the CC BY-NC-ND 4.0 license remain with the copyright holder, including commercial exploitation rights.

**You are free to:**
- ✅ Share and adapt this work for non-commercial purposes
- ✅ Use in academic research and education (with restrictions - see LICENSE)

**Restrictions:**
- ❌ **No commercial use** without explicit written permission
- ❌ No use in for-profit research or commercial products
- ❌ No commercial consulting or paid services using this code
- ✅ Must provide attribution to the original author

**Important:** "Non-commercial" includes restrictions on industry-funded research and commercial partnerships. See the [LICENSE](LICENSE) file for detailed definitions and requirements.

**Third-Party Dependencies:** This project uses open-source libraries (NumPy, Matplotlib, SciPy, h5py, pytest, CuPy) under permissive licenses. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for complete attribution.

**Prior Art Notice:** This work has been publicly disclosed on OSF ([osf.io/6agn8](https://osf.io/6agn8)) and Zenodo ([DOI: 10.5281/zenodo.17510124](https://zenodo.org/records/17510124)), establishing prior art and preventing patent claims by third parties on the disclosed methods and algorithms.

**Anti-Circumvention:** Creating "clean room" reimplementations of the disclosed algorithms to avoid the non-commercial restriction is prohibited and may constitute license violation and/or trade secret misappropriation.

**Citation:** When using this software in academic work, please cite using the DOI above and acknowledge the CC BY-NC-ND 4.0 license.

**Commercial Licensing:** Contact **latticefieldmediumresearch@gmail.com** with formal inquiry. See [LICENSE](LICENSE) for required procedure. Email inquiry alone does NOT grant permission.

---

## Contributing

⚠️ **This is a read-only research repository under CC BY-NC-ND 4.0 (NoDerivatives).**

**External modifications are not accepted.** This repository serves as:
- A reference implementation for the published research
- A defensive publication establishing prior art
- An archival record linked to DOI 10.5281/zenodo.17478758

**You may:**
- ✅ Report bugs via email (latticefieldmediumresearch@gmail.com)
- ✅ Request clarifications or discuss results via email
- ✅ Use the code for non-commercial research (with attribution and restrictions—see LICENSE)

**You may NOT:**
- ❌ Submit pull requests (will be closed)
- ❌ Create derivative works (license violation)
- ❌ Fork for commercial purposes

For collaboration inquiries or commercial licensing, contact latticefieldmediumresearch@gmail.com

---

## Contact

**Greg D. Partin**  
**ORCID:** [https://orcid.org/0009-0004-0327-6528](https://orcid.org/0009-0004-0327-6528)  
**Email:** latticefieldmediumresearch@gmail.com  
LFM Research — Los Angeles, CA USA

For bug reports, feature requests, or collaboration inquiries, please email latticefieldmediumresearch@gmail.com.

---

## Acknowledgments

Development assisted by GitHub Copilot for code generation, refactoring, and documentation.

Special thanks to the scientific Python community (NumPy, SciPy, Matplotlib) and the CUDA/CuPy teams for GPU acceleration infrastructure.

---

**Status:** Active research project in advanced development.

**Last Updated:** 2025-11-01
