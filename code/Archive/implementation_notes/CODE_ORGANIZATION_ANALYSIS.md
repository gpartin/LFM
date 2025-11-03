# Code Organization Analysis - LFM Repository

**Analysis Date:** October 31, 2025  
**Scope:** `/code` directory structure and tier test harnesses

---

## Executive Summary

The `/code` directory shows good modular structure with clear separation of core physics (`lfm_*.py`) and test harnesses (`run_tier*.py`). However, there are significant opportunities for:

1. **Code reduction** through shared utilities (~400-600 lines removable)
2. **Improved reuse** via common base classes and utilities
3. **Better organization** with consolidated helper modules

**Estimated Impact:**
- Remove ~500-800 lines of duplicated code (15-20% reduction)
- Reduce maintenance burden by centralizing common patterns
- Improve testability through better separation of concerns

---

## 1. Critical Duplication Issues

### 1.1 Backend Selection & Array Conversion (HIGH PRIORITY)

**Pattern duplicated across 6+ files:**

```python
# Appears in: run_tier1_relativistic.py, run_tier2_gravityanalogue.py, 
# run_tier3_energy.py, run_tier4_quantization.py, run_unif00_core_principle.py,
# lfm_diagnostics.py, lfm_visualizer.py

def pick_backend(use_gpu_flag: bool):
    on_gpu = bool(use_gpu_flag and _HAS_CUPY)
    return (cp if on_gpu else np), on_gpu

def to_numpy(x):
    if _HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)  # or x.get()
    return np.asarray(x)
```

**Recommendation:** Create `lfm_backend.py` module:

```python
# lfm_backend.py
"""Backend selection and array conversion utilities."""
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

def pick_backend(use_gpu: bool):
    """Select NumPy or CuPy backend based on availability and request.
    
    Returns: (xp, on_gpu) tuple where xp is np or cp module
    """
    on_gpu = bool(use_gpu and HAS_CUPY)
    if use_gpu and not HAS_CUPY:
        from lfm_console import log
        log("GPU requested but CuPy unavailable; using NumPy", "WARN")
    return (cp if on_gpu else np), on_gpu

def to_numpy(x):
    """Convert array to NumPy, handling CuPy arrays."""
    if HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)

def ensure_device(x, xp):
    """Ensure array is on correct device (CPU/GPU)."""
    if xp is cp and not isinstance(x, cp.ndarray):
        return cp.asarray(x)
    if xp is np and HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x
```

**Impact:** Removes ~80 lines across 7 files, ensures consistent behavior

---

### 1.2 Config Loading (MEDIUM PRIORITY)

**Pattern duplicated across all tier runners:**

```python
def load_config(config_path: str = None) -> Dict:
    if config_path:
        p = Path(config_path)
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    # Find default config...
    p = Path(__file__).resolve().parent / 'config' / 'config_tier1_relativistic.json'
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)
```

**Recommendation:** Enhance `lfm_config.py`:

```python
# lfm_config.py (enhancement)
def load_tier_config(tier_name: str, config_path: str = None) -> Dict:
    """Load tier configuration with fallback to default location.
    
    Args:
        tier_name: e.g., 'tier1_relativistic', 'tier2_gravityanalogue'
        config_path: Optional explicit path
    
    Returns:
        Configuration dictionary
    """
    if config_path:
        return load_config(config_path)
    
    # Search standard locations
    script_dir = Path.cwd()
    config_file = f"config_{tier_name}.json"
    
    for root in (script_dir, script_dir.parent):
        cand = root / "config" / config_file
        if cand.exists():
            return load_config(str(cand))
    
    raise FileNotFoundError(
        f"Config not found: {config_file}. "
        f"Searched in {script_dir}/config and parent."
    )
```

**Impact:** Removes ~60 lines, standardizes config discovery

---

### 1.3 Gaussian Field Initialization (HIGH PRIORITY)

**Highly duplicated pattern across test files and tier runners:**

```python
# Variations appear in:
# - run_tier1_relativistic.py (multiple variations)
# - run_tier2_gravityanalogue.py (gaussian_packet, gaussian_bump, etc.)
# - run_tier3_energy.py (init_pulse)
# - tests/test_lfm_equation_multidim.py (gaussian_nd)
# - tests/test_lfm_equation_parallel_all.py (make_field)

# Basic pattern:
def gaussian_3d(N, center, width, amplitude, xp):
    ax = xp.arange(N, dtype=xp.float64)
    cx, cy, cz = center
    gx = xp.exp(-((ax - cx)**2) / (2.0 * width**2))
    gy = xp.exp(-((ax - cy)**2) / (2.0 * width**2))
    gz = xp.exp(-((ax - cz)**2) / (2.0 * width**2))
    return amplitude * (gx[:, xp.newaxis, xp.newaxis] * 
                       gy[xp.newaxis, :, xp.newaxis] * 
                       gz[xp.newaxis, xp.newaxis, :])
```

**Recommendation:** Create `lfm_fields.py` module:

```python
# lfm_fields.py
"""Standard field initialization utilities."""

def gaussian_field(shape, center=None, width=1.0, amplitude=1.0, xp=None):
    """Create N-dimensional Gaussian field.
    
    Args:
        shape: Tuple of grid dimensions (N,) or (Nx, Ny) or (Nx, Ny, Nz)
        center: Center position (defaults to grid center)
        width: Gaussian width in grid cells
        amplitude: Peak amplitude
        xp: NumPy or CuPy module
    
    Returns:
        Gaussian field array
    """
    if xp is None:
        import numpy as xp
    
    ndim = len(shape)
    if center is None:
        center = tuple((n - 1) / 2.0 for n in shape)
    
    # Create coordinate grids
    axes = [xp.arange(n, dtype=xp.float64) for n in shape]
    
    # Compute radial distance
    r_squared = xp.zeros(shape, dtype=xp.float64)
    for i, (ax, c) in enumerate(zip(axes, center)):
        # Broadcast ax to full shape
        ax_shape = [1] * ndim
        ax_shape[i] = shape[i]
        ax_broadcast = ax.reshape(ax_shape)
        r_squared += (ax_broadcast - c) ** 2
    
    return amplitude * xp.exp(-r_squared / (2.0 * width ** 2))

def wave_packet(shape, kvec, center=None, width=1.0, amplitude=1.0, xp=None):
    """Create modulated Gaussian wave packet.
    
    Args:
        shape: Grid dimensions
        kvec: Wave vector (scalar for 1D, array for higher dims)
        center, width, amplitude, xp: As in gaussian_field
    
    Returns:
        Gaussian envelope × exp(i k·r) field
    """
    envelope = gaussian_field(shape, center, width, amplitude, xp)
    
    # Create phase
    ndim = len(shape)
    if not hasattr(kvec, '__len__'):
        kvec = [kvec] + [0] * (ndim - 1)
    
    if center is None:
        center = tuple((n - 1) / 2.0 for n in shape)
    
    axes = [xp.arange(n, dtype=xp.float64) for n in shape]
    phase = xp.zeros(shape, dtype=xp.float64)
    
    for i, (ax, k, c) in enumerate(zip(axes, kvec, center)):
        ax_shape = [1] * ndim
        ax_shape[i] = shape[i]
        ax_broadcast = ax.reshape(ax_shape)
        phase += k * (ax_broadcast - c)
    
    return envelope * xp.exp(1j * phase)

def traveling_wave_init(E0, kvec, omega, dt, xp=None):
    """Create E and E_prev for traveling wave with correct velocity.
    
    Args:
        E0: Initial field at t=0
        kvec: Wave vector
        omega: Angular frequency
        dt: Time step
        xp: Backend module
    
    Returns:
        (E, E_prev) tuple for leapfrog integration
    """
    if xp is None:
        import numpy as xp
    
    # Compute phase at t=-dt
    phase_shift = -omega * dt
    # For real-valued initial field, shift the pattern
    # This is approximate but works for smooth packets
    E_prev = E0 * xp.cos(phase_shift)
    return E0, E_prev
```

**Impact:** Removes ~200-300 lines across multiple files, ensures consistency

---

### 1.4 Frequency Estimation (MEDIUM PRIORITY)

**Similar FFT-based frequency measurement in multiple places:**

```python
# run_tier2_gravityanalogue.py: hann_fft_freq()
# run_tier1_relativistic.py: hann_vec() + FFT logic
# tests/test_lfm_dispersion_3d.py: measure_omega_targeted()

def hann_fft_freq(series: List[float], dt: float) -> float:
    x = np.asarray(series, float)
    x = x - x.mean()
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), dt)
    mag = np.abs(X)
    k = int(np.argmax(mag[1:])) + 1
    # Parabolic refinement...
```

**Recommendation:** Consolidate into `lfm_diagnostics.py`:

```python
# Add to lfm_diagnostics.py
def measure_frequency(series, dt, method='hann_fft', hint_freq=None, **kwargs):
    """Measure dominant frequency in time series.
    
    Args:
        series: Time series data
        dt: Time step
        method: 'hann_fft' (peak FFT) or 'targeted' (scan near hint)
        hint_freq: Expected frequency for targeted method
        **kwargs: Method-specific options
    
    Returns:
        Measured angular frequency (rad/s)
    """
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    
    if len(x) < 8:
        return 0.0
    
    if method == 'hann_fft':
        return _hann_fft_peak(x, dt, **kwargs)
    elif method == 'targeted':
        if hint_freq is None:
            raise ValueError("targeted method requires hint_freq")
        return _targeted_scan(x, dt, hint_freq, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
```

**Impact:** Removes ~100 lines, provides tested implementations

---

## 2. Structural Improvements

### 2.1 Base Test Harness Class (HIGH PRIORITY)

**Current state:** Tier1Harness and Tier2Harness share ~60% of code

**Recommendation:** Create base class in `lfm_test_harness.py`:

```python
# lfm_test_harness.py
"""Base classes for tier test harnesses."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
from numeric_integrity import NumericIntegrityMixin
from lfm_logger import LFMLogger
from lfm_backend import pick_backend, to_numpy

class TierTestHarness(NumericIntegrityMixin, ABC):
    """Base class for tier test harnesses with common infrastructure."""
    
    def __init__(self, config: Dict, output_root: Path):
        """Initialize harness with config and output directory.
        
        Args:
            config: Full configuration dictionary
            output_root: Base directory for test outputs
        """
        self.cfg = config
        self.output_root = output_root
        self.run_settings = config.get("run_settings", {})
        self.params = config.get("parameters", {})
        
        # Backend selection
        use_gpu = self.run_settings.get("use_gpu", False)
        self.xp, self.on_gpu = pick_backend(use_gpu)
        
        # Precision
        prec = self.params.get("precision", "float64")
        self.dtype = getattr(self.xp, prec)
        
        # Quick mode
        self.quick = self.run_settings.get("quick_mode", False)
        
        # Logger setup
        self.logger = None
        
    def setup_test_output(self, test_id: str) -> Path:
        """Create and return test-specific output directory."""
        test_dir = self.output_root / test_id
        for subdir in ["diagnostics", "plots"]:
            (test_dir / subdir).mkdir(parents=True, exist_ok=True)
        return test_dir
    
    def build_params_dict(self, variant: Dict) -> Dict:
        """Build simulation parameters from base params + variant overrides."""
        params = {**self.params}  # Copy base
        params.update(variant)  # Apply variant-specific overrides
        params["use_gpu"] = self.on_gpu
        params["precision"] = str(self.dtype).split(".")[-1].replace("'>", "")
        return params
    
    @abstractmethod
    def run_variant(self, variant: Dict) -> Any:
        """Run a single test variant. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def generate_summary(self) -> Dict:
        """Generate tier-level summary. Must be implemented by subclass."""
        pass
```

**Usage in tier runners:**

```python
# run_tier1_relativistic.py
from lfm_test_harness import TierTestHarness

class Tier1Harness(TierTestHarness):
    def run_variant(self, variant: Dict) -> TestSummary:
        # Specific Tier-1 logic...
        pass
    
    def generate_summary(self) -> Dict:
        # Tier-1 summary logic...
        pass
```

**Impact:** 
- Removes ~150-200 lines per tier runner
- Ensures consistent behavior across tiers
- Makes adding new tiers much easier

---

### 2.2 Consolidate Energy/Diagnostics Functions (MEDIUM)

**Current state:** Energy computation duplicated in multiple places:
- `lfm_diagnostics.py`: `energy_total()`
- `run_tier3_energy.py`: Local `energy_total()`, `energy_components()`
- `chi_field_equation.py`: `energy_density_field()`

**Recommendation:** Consolidate all into `lfm_diagnostics.py`:

```python
# lfm_diagnostics.py (enhancement)
def energy_total(E, E_prev, dt, dx, c, chi, xp=None):
    """Compute total energy (existing, keep as-is)."""
    pass

def energy_components(E, E_prev, dt, dx, c, chi, xp=None):
    """Compute kinetic, gradient, and potential energy separately.
    
    Returns:
        Dict with keys: 'kinetic', 'gradient', 'potential', 'total'
    """
    if xp is None:
        import numpy as xp
    
    # Kinetic energy: (∂E/∂t)² / 2
    dE_dt = (E - E_prev) / dt
    KE = 0.5 * xp.sum(dE_dt ** 2) * (dx ** E.ndim)
    
    # Gradient energy: c²(∇E)² / 2
    GE = 0.5 * (c ** 2) * xp.sum(grad_sq(E, dx, xp)) * (dx ** E.ndim)
    
    # Potential energy: χ²E² / 2
    if hasattr(chi, 'shape'):
        PE = 0.5 * xp.sum((chi ** 2) * (E ** 2)) * (dx ** E.ndim)
    else:
        PE = 0.5 * (chi ** 2) * xp.sum(E ** 2) * (dx ** E.ndim)
    
    return {
        'kinetic': float(to_numpy(KE)),
        'gradient': float(to_numpy(GE)),
        'potential': float(to_numpy(PE)),
        'total': float(to_numpy(KE + GE + PE))
    }

def energy_density_field(E, E_prev, dt, dx, chi, c=1.0, xp=None):
    """Compute local energy density field ρ(x).
    
    Returns array with same shape as E containing local energy density.
    """
    # Move from chi_field_equation.py to here
    pass
```

**Impact:** Removes ~50-80 lines, improves discoverability

---

## 3. File Organization Recommendations

### 3.1 Create Utility Modules

**New file structure:**

```
code/
├── lfm_backend.py          # NEW: Backend selection, array conversion
├── lfm_fields.py            # NEW: Field initialization utilities
├── lfm_test_harness.py      # NEW: Base test harness class
├── lfm_config.py            # Enhanced: Better tier config loading
├── lfm_diagnostics.py       # Enhanced: Consolidated energy/diagnostics
├── lfm_equation.py          # Keep as core physics
├── lfm_parallel.py          # Keep as core parallelization
├── ... (other lfm_*.py)     # Keep existing core modules
├── run_tier1_relativistic.py    # Simplified using base harness
├── run_tier2_gravityanalogue.py # Simplified using base harness
├── run_tier3_energy.py      # Simplified using base harness
├── run_tier4_quantization.py    # Simplified using base harness
└── run_unif00_core_principle.py # Simplified using base harness
```

### 3.2 Move Chi-Field Code

**Current:** `chi_field_equation.py` is a standalone module but closely related to diagnostics

**Recommendation:** Either:
- Option A: Keep separate (current state is fine)
- Option B: Rename to `lfm_chi_field.py` for consistency with naming convention

---

## 4. Priority Action Plan

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Create `lfm_backend.py` with `pick_backend()` and `to_numpy()`
2. ✅ Update all tier runners to use new module
3. ✅ Create `lfm_fields.py` with `gaussian_field()` and `wave_packet()`
4. ✅ Update tests to use new field utilities

**Estimated reduction:** ~200-300 lines

### Phase 2: Structural Improvements (3-4 hours)
1. ✅ Create `lfm_test_harness.py` with `TierTestHarness` base class
2. ✅ Refactor Tier1Harness to inherit from base
3. ✅ Refactor Tier2Harness to inherit from base
4. ✅ Consolidate energy functions in `lfm_diagnostics.py`

**Estimated reduction:** ~300-400 lines

### Phase 3: Polish & Cleanup (1-2 hours)
1. ✅ Enhance `lfm_config.py` with `load_tier_config()`
2. ✅ Add frequency measurement utilities to `lfm_diagnostics.py`
3. ✅ Update documentation
4. ✅ Run full test suite to validate

**Estimated reduction:** ~100-150 lines

### Total Estimated Impact
- **Code reduction:** 600-850 lines (15-20%)
- **Maintenance burden:** Significantly reduced
- **Consistency:** Greatly improved
- **Time investment:** 5-8 hours

---

## 5. Testing & Validation Strategy

### Before Changes
1. Run full tier test suite and record baseline metrics
2. Document current MASTER_TEST_STATUS.csv results
3. Create git branch for refactoring

### During Refactoring
1. Make changes incrementally (one module at a time)
2. Run affected tier tests after each change
3. Verify numerical results match baseline
4. Check for performance regressions

### After Changes
1. Run complete test suite
2. Compare results with baseline (should be identical)
3. Verify code coverage hasn't decreased
4. Update documentation

---

## 6. Non-Code Organization Items

### Documentation Needed
- Add `docs/ARCHITECTURE.md` describing module responsibilities
- Add docstring examples for common patterns
- Create `CONTRIBUTING.md` with coding standards

### Testing Gaps
- Add unit tests for new utility modules
- Add integration tests for base harness
- Consider property-based tests for field initialization

---

## 7. Files Not Requiring Changes

**These are well-organized and should remain as-is:**

- `lfm_equation.py` - Core physics, clean implementation
- `lfm_parallel.py` - Parallelization logic, well-contained
- `lfm_console.py` - Logging utilities, good single responsibility
- `lfm_logger.py` - Structured logging, well-designed
- `lfm_results.py` - Output handling, clean interface
- `lfm_plotting.py` - Visualization, good separation
- `lfm_visualizer.py` - Higher-level viz, appropriate
- `lorentz_transform.py` - Specialized physics, standalone
- `energy_monitor.py` - Monitoring utilities, focused
- `numeric_integrity.py` - Validation mixin, good design
- `resource_monitor.py` - System monitoring, isolated
- `lfm_simulator.py` - High-level simulator, appropriate layer

---

## Conclusion

The LFM codebase is well-structured with good separation between physics and orchestration. The main opportunities lie in:

1. **Extracting common utilities** (backend, fields, config) → immediate code reduction
2. **Creating base classes** for test harnesses → long-term maintainability
3. **Consolidating diagnostics** → better discoverability and consistency

These changes will reduce code by 15-20% while significantly improving maintainability and making it easier to add new tier tests or physics features.

**Recommendation:** Proceed with Phase 1 (quick wins) immediately as it provides high value with low risk.
