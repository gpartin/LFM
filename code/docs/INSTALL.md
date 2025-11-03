# LFM Installation Guide

Complete installation instructions for the Lattice Field Medium (LFM) simulator.

**License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
**Author:** Greg D. Partin | LFM Research  
**Contact:** latticefieldmediumresearch@gmail.com  
**DOI:** [10.5281/zenodo.17478758](https://zenodo.org/records/17478758)  
**Repository:** [OSF: osf.io/6agn8](https://osf.io/6agn8)

---

## System Requirements

### Minimum Requirements (CPU Only)
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS 11+
- **Python**: 3.9 or higher
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 2 GB for code + 10-50 GB for test results (depending on tests run)
- **CPU**: Multi-core processor recommended (tests use parallelization)

### Recommended Requirements (GPU Accelerated)
- **GPU**: NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal, Volta, Turing, Ampere, Ada)
- **CUDA**: Version 12.x
- **GPU RAM**: 4 GB minimum, 8 GB+ recommended for large 3D simulations
- **Driver**: NVIDIA driver 525.60.13+ (Linux) or 527.41+ (Windows)

---

## Installation Steps

### 1. Clone the Repository

```bash
# Navigate to your projects directory
cd /path/to/your/projects

# Copy the LFM directory
# (Repository location will be provided separately)
```

### 2. Set Up Python Environment (Recommended)

**Option A: Using venv (built-in)**
```bash
# Create virtual environment
python -m venv lfm_env

# Activate (Windows PowerShell)
.\lfm_env\Scripts\Activate.ps1

# Activate (Linux/macOS)
source lfm_env/bin/activate
```

**Option B: Using conda**
```bash
# Create conda environment
conda create -n lfm python=3.11
conda activate lfm
```

### 3. Install Dependencies

**CPU Only (Basic Installation)**
```bash
pip install -r requirements.txt
```

This installs:
- `numpy` — Core numerical arrays
- `matplotlib` — Plotting
- `scipy` — Signal processing
- `h5py` — HDF5 file I/O
- `pytest` — Testing framework

**GPU Acceleration (Optional)**
```bash
# Requires NVIDIA GPU with CUDA 12.x
pip install cupy-cuda12x

# Verify GPU is detected
python -c "import cupy as cp; print(f'GPU: {cp.cuda.Device().name}')"
```

**Alternative GPU Installation (CUDA 11.x)**
```bash
# For older CUDA 11.x systems
pip install cupy-cuda11x
```

### 4. Verify Installation

```bash
# Run quick test (should complete in ~5 seconds)
python run_tier1_relativistic.py --test REL-01

# Expected output:
# ════════════════════════════════════════════════════════════════
# LFM Tier-1 Relativistic Test Suite
# ════════════════════════════════════════════════════════════════
# Running test: REL-01 (Isotropy — Coarse Grid)
# [PASS] REL-01 completed in X.XX seconds
```

If this succeeds, your installation is working correctly!

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'cupy'`

**Cause**: CuPy not installed or GPU not available.

**Solution**:
1. If you don't have an NVIDIA GPU, ignore this error — tests will run on CPU
2. If you have a GPU:
   ```bash
   pip install cupy-cuda12x
   ```
3. Verify GPU detection:
   ```bash
   python -c "import cupy as cp; cp.show_config()"
   ```

### Issue: `ImportError: DLL load failed` (Windows CuPy)

**Cause**: Missing CUDA runtime libraries.

**Solution**:
1. Download and install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
2. Verify installation:
   ```bash
   nvcc --version
   ```
3. Reinstall CuPy:
   ```bash
   pip uninstall cupy-cuda12x
   pip install cupy-cuda12x
   ```

### Issue: Tests fail with `CFL violated` error

**Cause**: Time step too large for spatial resolution.

**Solution**: This is a physics constraint, not a bug. The error message shows the CFL ratio and required adjustment. Either:
- Use default configurations (already CFL-stable)
- Reduce `dt` in config file
- Increase `dx` in config file

### Issue: `FileNotFoundError: config/config_tier1_relativistic.json`

**Cause**: Running from wrong directory.

**Solution**: Always run tests from the `LFM/code/` directory:
```bash
cd /path/to/LFM/code
python run_tier1_relativistic.py
```

### Issue: Out of memory errors

**Cause**: Test requires more RAM/GPU memory than available.

**Solution**:
1. Use `--quick` mode for faster, lower-resolution tests:
   ```bash
   python run_tier1_relativistic.py --quick
   ```
2. Close other applications
3. For GPU memory issues:
   - Reduce grid size in config file
   - Run on CPU instead (slower but uses system RAM)
   - Skip large 3D tests (GRAV-15, GRAV-16)

### Issue: Tests hang or freeze

**Cause**: Long-running simulation or deadlock in parallel execution.

**Solution**:
1. Be patient — some tests take several minutes (especially 3D with large grids)
2. Check CPU usage — if low, may be deadlocked (Ctrl+C and report issue)
3. Use `--quick` mode for faster iteration
4. Check `results/` for partial outputs to see progress

---

## Configuration

### GPU vs CPU

Edit `config/*.json` files to toggle GPU acceleration:

```json
{
  "run_settings": {
    "use_gpu": true  // Set to false for CPU-only
  }
}
```

**Performance comparison** (typical):
- CPU (Intel i7): 3D tests take 2-5 minutes
- GPU (RTX 3080): 3D tests take 10-30 seconds

### Quick Mode

For rapid verification during development:

```json
{
  "run_settings": {
    "quick_mode": true  // Reduces grid size and step count
  }
}
```

Or use command-line flag:
```bash
python run_tier1_relativistic.py --quick
```

---

## Next Steps

After successful installation:

1. **Run Full Test Suite** (validates installation):
   ```bash
   python run_tier1_relativistic.py
   ```

2. **View Test Results**:
   ```bash
   # Summary of all tests
   cat results/MASTER_TEST_STATUS.csv
   
   # Individual test results
   ls results/Relativistic/REL-01/
   ```

3. **Explore Visualizations**:
   ```bash
   # After running GRAV-15 test
   python tools/visualize/visualize_grav15_3d.py
   ```

4. **Read User Guide** (Coming Soon):
   - Configuration reference
   - Output interpretation
   - Custom test development

---

## Development Setup (Optional)

For contributors and developers:

```bash
# Install development dependencies
pip install pytest-cov black flake8 mypy

# Run tests with coverage
pytest tests/ -v --cov=. --cov-report=html

# Format code
black *.py

# Lint code
flake8 --max-line-length=120 *.py

# Type check
mypy lfm_equation.py lfm_parallel.py
```

---

## Platform-Specific Notes

### Windows

- Use PowerShell (not Command Prompt) for best compatibility
- Virtual environment activation:
  ```powershell
  .\lfm_env\Scripts\Activate.ps1
  ```
- If execution policy blocks activation:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### Linux

- Ensure Python development headers are installed:
  ```bash
  sudo apt-get install python3-dev  # Ubuntu/Debian
  ```
- For GPU support, install NVIDIA drivers and CUDA toolkit:
  ```bash
  sudo apt-get install nvidia-driver-525 nvidia-cuda-toolkit
  ```

### macOS

- CuPy not supported (Apple GPUs use Metal, not CUDA)
- Use CPU-only installation
- Install via Homebrew for easier dependency management:
  ```bash
  brew install python@3.11 hdf5
  ```

---

## Support

If you encounter issues not covered here:

1. Check this installation guide and troubleshooting section
2. Review `docs/USER_GUIDE.md` for usage issues
3. Email latticefieldmediumresearch@gmail.com with:
   - OS and Python version (`python --version`)
   - Full error message
   - Steps to reproduce
   - GPU info (if applicable): `nvidia-smi` output

---

**Last Updated:** 2025-11-01
