# LFM Installation Guide

<!-- Copyright (c) 2025 Greg D. Partin. All rights reserved. -->
<!-- Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International). -->
<!-- See LICENSE file in project root for full license text. -->
<!-- Commercial use prohibited without explicit written permission. -->
<!-- Contact: latticefieldmediumresearch@gmail.com -->

This guide covers multiple installation methods for the Lattice Field Medium (LFM) framework on Windows, macOS, and Linux.

## Quick Installation (All Platforms)

### Method 1: One-Click Setup Scripts

#### Windows
1. Download the LFM code directory
2. Open Command Prompt or PowerShell
3. Navigate to the `code` folder: `cd c:\LFM\code`
4. Run: `quick_setup_windows.bat`
5. Follow the prompts

#### macOS/Linux
1. Download the LFM code directory
2. Open Terminal
3. Navigate to the code folder: `cd /path/to/LFM/code`
4. Run: `chmod +x quick_setup_unix.sh && ./quick_setup_unix.sh`
5. Follow the prompts

### Method 2: Automated Python Installer
```bash
cd /path/to/LFM/code
python setup_lfm.py
```

This comprehensive installer will:
- ✅ Check Python version compatibility (3.9+ required)
- ✅ Install all required packages
- ✅ Detect GPU and offer acceleration
- ✅ Run validation tests
- ✅ Create convenient shortcuts

## Manual Installation

### Prerequisites
- **Python 3.9 or higher** ([download here](https://python.org/downloads/))
- **pip** (included with Python)

### Core Dependencies
```bash
pip install numpy>=1.24.0 matplotlib>=3.7.0 scipy>=1.10.0 h5py>=3.8.0 pytest>=7.3.0
```

### Optional: GPU Acceleration
If you have an NVIDIA GPU with CUDA 12.x:
```bash
pip install cupy-cuda12x
```

### Verification
Test your installation:
```bash
python -c "import numpy, matplotlib, scipy, h5py, pytest; print('✅ Installation successful!')"
```

## Starting LFM

After installation, you have three options:

### 1. Graphical Interface (Recommended for beginners)
```bash
python lfm_gui.py
```
- Point-and-click operation
- Real-time progress monitoring
- Visual results browser

### 2. Console Interface (Terminal users)
```bash
python lfm_control_center.py
```
- Menu-driven navigation
- Color-coded output
- Fast execution

### 3. Command Line (Expert users)
```bash
python run_tier1_relativistic.py --test REL-01
```
- Direct script control
- Full parameter access
- Automation-friendly

## Troubleshooting

### Common Issues

#### "Python not found"
- **Windows:** Install Python from python.org, check "Add to PATH"
- **macOS:** Install via Homebrew: `brew install python3`
- **Linux:** Install via package manager: `sudo apt install python3 python3-pip`

#### "tkinter not found" (GUI won't start)
- **Linux:** `sudo apt-get install python3-tk`
- **macOS:** Tkinter included with Python
- **Windows:** Tkinter included with Python

#### "CUDA not found" (GPU acceleration)
- Install NVIDIA drivers and CUDA toolkit
- Or continue with CPU-only mode (still very fast)

#### Tests fail to run
1. Check Python version: `python --version` (need 3.9+)
2. Verify dependencies: `pip list | grep numpy`
3. Try the validation test: `python -c "from lfm_equation import LFMEquation; print('✅ LFM works!')"`

### Getting Help

If you encounter issues:
1. Check the error message carefully
2. Verify Python version and dependencies
3. Try the manual installation method
4. Contact: latticefieldmediumresearch@gmail.com

## System Requirements

### Minimum
- **OS:** Windows 10+, macOS 10.14+, or Linux (any recent distribution)
- **Python:** 3.9 or higher
- **RAM:** 4 GB
- **Storage:** 1 GB free space

### Recommended
- **OS:** Windows 11, macOS 12+, or Ubuntu 20.04+
- **Python:** 3.11 or higher
- **RAM:** 8 GB or more
- **GPU:** NVIDIA GTX 1060 or better (for acceleration)
- **Storage:** 5 GB free space (for results)

## Performance Notes

- **CPU Mode:** Fast enough for most tests, uses all available cores
- **GPU Mode:** 10-50x speedup for large 3D simulations
- **Quick Mode:** Use `--quick` flag for rapid validation
- **Parallel Mode:** Full test suite completes in ~30 minutes

## Next Steps

After installation:
1. **Run your first test:** REL-01 (relativistic propagation) or EM-01 (Coulomb's law)
2. **View the results:** Check `results/Relativistic/REL-01/` or `results/Electromagnetic/EM-01/`
3. **Try the GUI:** Launch `lfm_gui.py` for visual interface
4. **Explore:** Browse the full 55-test validation suite
5. **Read:** Check `README.md` for complete documentation

Welcome to LFM! 🎉