# LFM — Lattice Field Medium Simulator

<!-- Copyright (c) 2025 Greg D. Partin. All rights reserved. -->
<!-- Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International). -->
<!-- See LICENSE file in project root for full license text. -->
<!-- Commercial use prohibited without explicit written permission. -->
<!-- Contact: latticefieldmediumresearch@gmail.com -->

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17510124.svg)](https://doi.org/10.5281/zenodo.17510124)
[![OSF](https://img.shields.io/badge/OSF-10.17605%2FOSF.IO%2F6AGN8-blue)](https://osf.io/6agn8)

**High-performance Klein-Gordon wave equation solver exploring unified physics through emergent phenomena.**

Simulates relativistic wave propagation, gravity analogues, and quantum behavior in discrete spacetime using the lattice field medium hypothesis — that fundamental physics emerges from discrete field interactions.

**Author:** Greg D. Partin | LFM Research — Los Angeles, CA USA  
**Contact:** latticefieldmediumresearch@gmail.com  
**ORCID:** [https://orcid.org/0009-0004-0327-6528](https://orcid.org/0009-0004-0327-6528)

---

## 🚀 Quick Start

### One-Command Installation

#### Windows
```cmd
cd LFM\code
quick_setup_windows.bat
```

#### macOS/Linux
```bash
cd LFM/code
./quick_setup_unix.sh
```

### Launch LFM
```bash
# Graphical interface (recommended for beginners)
python lfm_gui.py

# Console interface (menu-driven)
python lfm_control_center.py

# Run your first test
python run_tier1_relativistic.py --test REL-01
```

**📋 Complete Setup Guide:** See [`code/INSTALL.md`](code/INSTALL.md) for detailed instructions.

---

## 🎯 What LFM Does

LFM validates a **unified physics hypothesis** through computational simulation:

> **The lattice field medium hypothesis:** Fundamental physics (relativity, gravity, quantum mechanics) emerges from discrete field interactions in a structured spacetime lattice.

### Core Physics Simulation
- **Klein-Gordon Equation:** `∂²E/∂t² = c²∇²E - χ²(x,t)E`
- **Spatially-Varying χ-Field:** Enables gravity analogue and quantum behavior
- **Emergence Mechanism:** Energy density drives χ-field structure formation

### Validated Phenomena (55 Tests, 93% Success Rate)

#### ✅ **Tier 1: Relativistic Physics** (15 tests, 100% passing)
- Lorentz invariance and causality
- Dispersion relations: ω² = k² + χ²
- Light-speed propagation limits
- Isotropy across all spatial directions

#### ✅ **Tier 2: Gravity Analogue** (25 tests, 84% passing)  
- Time dilation in χ-potential wells
- Gravitational redshift and frequency shifts
- Light bending through χ-gradients
- Gravitational wave propagation
- 3D double-slit interference patterns

#### ✅ **Tier 3: Energy Conservation** (11 tests, 91% passing)
- Global energy conservation (<0.01% drift)
- Hamiltonian partitioning (KE ↔ PE ↔ GE)
- Thermalization and dissipation dynamics
- Stability under long-time evolution

#### ✅ **Tier 4: Quantum Behavior** (14 tests, 100% passing)
- Discrete bound-state energies: En = √(kn² + χ²)
- Quantum tunneling through barriers
- Heisenberg uncertainty: Δx·Δk ≥ 0.5
- Zero-point energy and vacuum fluctuations

### 🔬 **Critical Evidence: Genuine Emergence**
- **Self-Organization:** 29% χ-field enhancement from uniform initial conditions
- **No Pre-Programming:** Energy-χ coupling drives structure formation
- **Validated:** Refutes "circular validation" criticism ([Evidence](code/docs/evidence/emergence_validation/))

---

## 🖥️ User Interfaces

LFM provides three interaction modes:

### 1. **Graphical Interface** (`lfm_gui.py`)
Perfect for new users and visual interaction:
- 🖱️ Point-and-click test execution
- 📊 Real-time progress monitoring
- 🗂️ Visual results browser with folder trees
- 🔧 Built-in system diagnostics

### 2. **Console Interface** (`lfm_control_center.py`)
Ideal for terminal users and automation:
- 🎯 Menu-driven navigation (1-8 options)
- 🌈 Color-coded output and status
- ⚡ Fast execution with progress bars
- 📋 Integrated results viewer

### 3. **Command Line Interface**
For expert users and scripting:
- 🔧 Direct script execution
- ⚙️ Full parameter control
- 🚀 Maximum performance
- 📝 Comprehensive logging

**Cross-Platform:** All interfaces work on Windows, macOS, and Linux.

---

## 📁 Project Structure

```
LFM/
├── README.md                    # 👈 This file - project overview & quick start
├── code/                        # 🎯 Main codebase - START HERE
│   ├── INSTALL.md              #    Complete installation guide
│   ├── README.md               #    Technical implementation docs
│   ├── setup_lfm.py            #    Automated installer
│   ├── lfm_gui.py              #    Windows GUI interface
│   ├── lfm_control_center.py   #    Console interface
│   ├── lfm_equation.py         #    Core Klein-Gordon solver
│   ├── run_tier*_*.py          #    Test harnesses (Tiers 1-4)
│   ├── config/                 #    JSON configuration files
│   ├── results/                #    Test outputs (auto-generated)
│   ├── docs/                   #    Technical documentation
│   └── tools/                  #    Visualization utilities
├── config/                      # Additional configuration files
└── installer/                   # Installation utilities (auto-generated)
```

**🎯 Start Here:** Navigate to [`code/`](code/) directory for the complete framework.

---

## 🏆 Key Results & Scientific Impact

### Physics Validation
- **Lorentz Covariance:** Validated to 1% accuracy up to 0.6c
- **Gravity Simulation:** Time dilation, redshift, wave propagation confirmed
- **Quantum Phenomena:** Bound states, tunneling, uncertainty principles reproduced
- **Energy Conservation:** <0.01% drift over 10,000 simulation steps

### Computational Performance
- **CPU Mode:** Multi-threaded execution on all available cores
- **GPU Mode:** 10-50x speedup with NVIDIA CUDA acceleration
- **Scalability:** Handles 1D-3D simulations with adaptive mesh refinement
- **Efficiency:** Full validation suite (55 tests) completes in ~30 minutes

### Scientific Contributions
- **Unified Framework:** Single equation generates relativistic, gravitational, and quantum behavior
- **Emergence Validation:** Demonstrates spontaneous structure formation
- **Computational Method:** Novel spatially-varying χ-field approach
- **Open Science:** Full source code and results publicly available

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| **[Installation Guide](code/INSTALL.md)** | Complete setup for all platforms |
| **[Technical README](code/README.md)** | Detailed physics and implementation |
| **[Production Analysis](code/docs/PRODUCTION_READINESS_ANALYSIS.md)** | Project maturity assessment |
| **[Emergence Evidence](code/docs/evidence/emergence_validation/)** | Critical proof of genuine physics emergence |
| **[Test Results](code/results/MASTER_TEST_STATUS.csv)** | Complete validation status |

---

## 🚀 Getting Started

1. **Clone or download** this repository
2. **Navigate** to the `code/` directory
3. **Install** using one of these methods:
   - **Easy:** Run `quick_setup_windows.bat` (Windows) or `./quick_setup_unix.sh` (macOS/Linux)
   - **Advanced:** Run `python setup_lfm.py` for full automation
   - **Manual:** Follow [`code/INSTALL.md`](code/INSTALL.md)
4. **Launch** your preferred interface:
   - GUI: `python lfm_gui.py`
   - Console: `python lfm_control_center.py`
   - CLI: `python run_tier1_relativistic.py --test REL-01`
5. **Explore** the results in `results/` directory

**🎯 First Test Recommendation:** REL-01 (relativistic propagation) — completes in ~5 seconds and validates core physics.

---

## 🔬 Scientific Background

LFM explores the **lattice field medium hypothesis** — a theoretical framework proposing that:

1. **Spacetime is discrete** at the fundamental level
2. **Field interactions** on this lattice generate observed physics
3. **Emergent phenomena** (relativity, gravity, quantum mechanics) arise naturally
4. **Unification** occurs through a single underlying field equation

This computational approach provides:
- **Testable predictions** for fundamental physics
- **Numerical validation** of theoretical concepts  
- **Bridge** between discrete and continuum physics
- **Platform** for exploring modified gravity and quantum theories

**Status:** Active research project with promising preliminary results.

---

## ⚖️ License & Usage

**Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**

**Copyright (c) 2025 Greg D. Partin. All rights reserved.**

### ✅ **You CAN:**
- Use for academic research and education
- Share and discuss results
- Cite in scientific publications
- Study the source code and methods

### ❌ **You CANNOT:**
- Use commercially without written permission
- Create commercial derivatives or "clean room" implementations
- Incorporate into for-profit products or services
- Use in industry-funded research without permission

**📧 Commercial Licensing:** Contact latticefieldmediumresearch@gmail.com

**📋 Full Terms:** See [`code/LICENSE`](code/LICENSE) for complete legal text.

---

## 📞 Contact & Citation

**Greg D. Partin**  
**ORCID:** [https://orcid.org/0009-0004-0327-6528](https://orcid.org/0009-0004-0327-6528)  
**Email:** latticefieldmediumresearch@gmail.com  
LFM Research — Los Angeles, CA USA

### Citation
```bibtex
@software{lfm_simulator,
  author = {Partin, Greg D.},
  title = {LFM: Lattice Field Medium Simulator},
  year = {2025},
  publisher = {LFM Research},
  license = {CC BY-NC-ND 4.0},
  doi = {10.5281/zenodo.17510124},
  url = {https://zenodo.org/records/17510124}
}
```

---

## 🙏 Acknowledgments

Development assisted by GitHub Copilot for code generation and documentation.

Special thanks to the scientific Python community (NumPy, SciPy, Matplotlib) and NVIDIA CUDA/CuPy teams for computational infrastructure.

---

**Ready to explore unified physics? Start in the [`code/`](code/) directory! 🎉**

*Last Updated: November 3, 2025*