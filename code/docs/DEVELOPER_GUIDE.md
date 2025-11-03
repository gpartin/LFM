# LFM Developer Guide

**Target Audience:** Human developers and AI coding assistants  
**Purpose:** Complete reference for understanding, modifying, and extending the LFM codebase  
**License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
**Author:** Greg D. Partin | LFM Research  
**Contact:** latticefieldmediumresearch@gmail.com  
**DOI:** [10.5281/zenodo.17478758](https://zenodo.org/records/17478758)  
**Repository:** [OSF: osf.io/6agn8](https://osf.io/6agn8)  
**Last Updated:** 2025-11-01

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Module Reference](#core-module-reference)
3. [Physics Implementation](#physics-implementation)
4. [Configuration System](#configuration-system)
5. [Test Harness Pattern](#test-harness-pattern)
6. [Adding New Tests](#adding-new-tests)
7. [Backend Abstraction (CPU/GPU)](#backend-abstraction)
8. [Common Patterns](#common-patterns)
9. [Debugging Guide](#debugging-guide)
10. [AI Assistant Quick Reference](#ai-assistant-quick-reference)

---

## Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    TIER HARNESSES                           │
│  run_tier1_relativistic.py  run_tier2_gravityanalogue.py   │
│  run_tier3_energy.py        run_tier4_quantization.py      │
│                                                             │
│  Responsibilities:                                          │
│  - Load test configurations from JSON                       │
│  - Execute test variants                                    │
│  - Collect metrics and validate against tolerances          │
│  - Save outputs (summary.json, plots, diagnostics)          │
└──────────────────┬──────────────────────────────────────────┘
                   │ inherits from
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              lfm_test_harness.py                            │
│              BaseTierHarness                                │
│                                                             │
│  Shared functionality:                                      │
│  - Config loading with inheritance                          │
│  - Backend selection (NumPy/CuPy)                          │
│  - Logger initialization                                    │
│  - FFT-based frequency measurement                          │
│  - Numeric integrity validation                             │
└──────────────────┬──────────────────────────────────────────┘
                   │ uses
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                 CORE PHYSICS LAYER                          │
│  lfm_equation.py      - Single source of truth for physics  │
│  lfm_parallel.py      - Threaded tile-based execution       │
│  energy_monitor.py    - Energy conservation tracking        │
│  numeric_integrity.py - CFL checks, NaN detection           │
└──────────────────┬──────────────────────────────────────────┘
                   │ uses
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                 INFRASTRUCTURE LAYER                        │
│  lfm_backend.py       - NumPy/CuPy abstraction             │
│  lfm_config.py        - Config loading utilities            │
│  lfm_logger.py        - Structured logging (text + JSONL)   │
│  lfm_results.py       - Output management                   │
│  lfm_plotting.py      - Visualization utilities             │
│  lfm_console.py       - Human-readable progress output      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
config/*.json 
    ↓
Tier Harness loads config
    ↓
For each test variant:
    ↓
Initialize fields (E, E_prev)
    ↓
Time loop: lfm_equation.advance() or lfm_parallel.run_lattice()
    ├─→ energy_monitor.py records energy per step
    ├─→ numeric_integrity.py validates CFL, NaN, edges
    └─→ lfm_diagnostics.py saves snapshots (optional)
    ↓
Post-processing:
    ├─→ Measure frequencies (FFT)
    ├─→ Calculate metrics (errors, drift, visibility)
    └─→ Generate plots (matplotlib)
    ↓
Save outputs:
    ├─→ summary.json (lfm_results.py)
    ├─→ plots/*.png (lfm_plotting.py)
    └─→ diagnostics/*.csv (CSV writers)
    ↓
Update test_metrics_history.json
```

---

## Core Module Reference

### lfm_equation.py — **PHYSICS KERNEL (CRITICAL)**

**Purpose:** Single source of truth for Klein-Gordon wave equation physics.

**Key Functions:**

```python
def lattice_step(E, E_prev, params):
    """
    Single Klein-Gordon time step using leapfrog integration.
    
    Physics: E[t+dt] = 2E[t] - E[t-dt] + dt²(c²∇²E[t] - χ²E[t] - γ∂E/∂t)
    
    Args:
        E: Current field (N,) or (N,N) or (N,N,N)
        E_prev: Previous field (same shape)
        params: Dict with keys:
            - dt: Time step
            - dx: Spatial step
            - alpha, beta: Wave speed params (c²=alpha/beta)
            - chi: Mass-like parameter
            - gamma_damp: Damping coefficient
            - boundary: 'periodic' or 'absorbing'
            - stencil_order: 2, 4, or 6
            - use_gpu: bool
    
    Returns:
        E_next: Field at t+dt (same shape as E)
    
    CRITICAL: Do not modify physics without explicit review.
    See .github/copilot-instructions.md for physics preservation rules.
    """

def advance(E, E_prev, steps, params, callback=None):
    """
    Serial time integration for multiple steps.
    
    Args:
        E: Initial field
        E_prev: Field at t=-dt
        steps: Number of time steps
        params: Physics parameters
        callback: Optional function(step, E, E_prev, params) called each step
    
    Returns:
        E_final, E_prev_final: Fields at end of integration
    """

def laplacian(E, dx, order=2, boundary='periodic'):
    """
    Discrete Laplacian with selectable stencil order.
    
    Stencil orders:
    - order=2: Standard 3-point (1D), 5-point (2D), 7-point (3D)
    - order=4: 5-point (1D), 9-point (2D), etc. (higher accuracy)
    - order=6: 7-point (1D), etc. (research-grade accuracy)
    
    Boundary conditions:
    - 'periodic': Wrap around (default for wave tests)
    - 'absorbing': Exponential decay at edges (reduces reflections)
    """

def energy_total(E, E_prev, dt, dx, params):
    """
    Compute total Hamiltonian energy.
    
    H = KE + GE + PE
    KE = (1/2) ∫ (∂E/∂t)² dV  (kinetic)
    GE = (1/2) ∫ (∇E)² dV     (gradient)
    PE = (1/2) ∫ χ²E² dV      (potential/mass)
    
    Returns:
        Dict with keys: 'total', 'kinetic', 'gradient', 'potential'
    """
```

**Physics Preservation Contract:**
- **DO NOT** modify wave operator, mass term, or time integration without explicit review
- **DO** add diagnostics, refactor for performance, improve comments
- **MUST** run numeric regression tests before any physics changes (see below)

**Numeric Regression Test Pattern:**
```python
# Before changing physics
E_before, _ = advance(E_init, E_prev, steps=100, params=params)
E_before_baseline = energy_total(E_before, E_prev, dt, dx, params)

# After making change
E_after, _ = advance(E_init, E_prev, steps=100, params=params)
E_after_test = energy_total(E_after, E_prev, dt, dx, params)

# Validate
drift = abs(E_after_test['total'] - E_before_baseline['total'])
assert drift / E_before_baseline['total'] < 1e-10, "Physics changed!"
```

---

### lfm_parallel.py — **PARALLEL EXECUTION**

**Purpose:** Threaded tile-based runner for multi-core performance.

**Key Functions:**

```python
def run_lattice(E, E_prev, steps, params, callback=None):
    """
    Parallel time integration using tile decomposition.
    
    Strategy:
    - Divides spatial domain into overlapping tiles
    - Uses ThreadPoolExecutor for concurrent tile updates
    - Deterministic: same results as serial advance() (bit-for-bit)
    
    Args:
        Same as lfm_equation.advance()
        params must include:
            - threads: Number of worker threads
            - tiles: Number of spatial tiles
    
    Returns:
        E_final, E_prev_final: Same as advance()
    
    Note: For small grids (N<128), serial is faster due to overhead.
    Use parallel for N≥128 or 3D simulations.
    """
```

**When to use:**
- 3D simulations (N³ grids)
- Long runs (>10,000 steps)
- Large 2D grids (N≥256)

**When NOT to use:**
- Quick tests (overhead dominates)
- GPU runs (CuPy already parallelized)
- Small 1D problems (serial is faster)

---

### energy_monitor.py — **CONSERVATION TRACKING**

**Purpose:** Per-step energy logging and integrity validation.

**Key Class:**

```python
class EnergyMonitor:
    def __init__(self, outdir, label="", energy_every=10):
        """
        Initialize energy monitoring.
        
        Args:
            outdir: Directory for diagnostics CSV
            label: Test identifier
            energy_every: Log energy every N steps (0 = disable)
        """
    
    def record(self, step, E, E_prev, dt, dx, params):
        """
        Record energy at current step.
        
        Computes total, kinetic, gradient, potential energy.
        Stores in internal log for later analysis.
        """
    
    def save(self, filepath):
        """
        Write energy log to CSV.
        
        Columns: step, time, E_total, E_kinetic, E_gradient, E_potential, drift
        """
    
    def get_drift(self):
        """
        Calculate energy drift since start.
        
        Returns:
            (absolute_drift, relative_drift_percent)
        """
```

**Usage Pattern:**

```python
monitor = EnergyMonitor(diag_dir, label="TEST-01", energy_every=25)

# In time loop
for step in range(steps):
    E, E_prev = lattice_step(E, E_prev, params)
    monitor.record(step, E, E_prev, dt, dx, params)

# After loop
monitor.save(diag_dir / "energy_log.csv")
abs_drift, rel_drift = monitor.get_drift()
```

---

### numeric_integrity.py — **VALIDATION MIXIN**

**Purpose:** CFL checks, NaN detection, edge effects monitoring.

**Key Methods (inherited by BaseTierHarness):**

```python
class NumericIntegrityMixin:
    def check_cfl_stability(self, dt, dx, c, dimensions):
        """
        Validate CFL condition: c·dt/dx ≤ 1/√D
        
        Raises:
            ValueError if CFL violated (prevents unstable runs)
        """
    
    def check_field_validity(self, E, label="E"):
        """
        Check for NaN, Inf, or excessive values.
        
        Raises:
            ValueError if field contains invalid values
        """
    
    def check_energy_conservation(self, energy_log, tolerance=0.01):
        """
        Validate energy drift is within tolerance.
        
        Args:
            energy_log: List of (step, E_total) tuples
            tolerance: Max relative drift (default 1%)
        
        Returns:
            bool: True if conserved within tolerance
        """
```

**Automatic Integration:**
All tier harnesses inherit these checks. They're called automatically before expensive operations.

---

### lfm_logger.py — **STRUCTURED LOGGING**

**Purpose:** Dual-mode logging (human text + machine JSONL).

**Key Class:**

```python
class LFMLogger:
    def __init__(self, output_dir):
        """
        Create logger writing to:
        - output_dir/run_log.txt (human-readable)
        - output_dir/run_log.jsonl (structured events)
        """
    
    def info(self, message):
        """Log informational message (both formats)."""
    
    def warning(self, message):
        """Log warning (both formats)."""
    
    def error(self, message):
        """Log error (both formats)."""
    
    def log_json(self, event_type, data):
        """
        Log structured event (JSONL only).
        
        Example:
            logger.log_json("test_start", {
                "test_id": "REL-01",
                "timestamp": time.time(),
                "params": {"N": 128, "steps": 1000}
            })
        """
    
    def record_env(self):
        """Log environment info (Python version, GPU, CPU, memory)."""
```

**Usage:**

```python
logger = LFMLogger(results_dir)
logger.record_env()  # Once at start
logger.info(f"Starting test {test_id}")
logger.log_json("test_start", {"test_id": test_id, "config": params})
# ... run test ...
logger.info(f"Test {test_id} completed: {status}")
logger.log_json("test_end", {"test_id": test_id, "status": status, "metrics": metrics})
```

**Why JSONL?**
- Machine-parseable for automated analysis
- One event per line (append-safe, grep-friendly)
- Preserves full metadata for post-processing

---

### lfm_results.py — **OUTPUT MANAGEMENT**

**Purpose:** Standardized output structure and saving.

**Key Functions:**

```python
def save_summary(result_dir, test_id, description, status, metrics, 
                 tier="Unknown", category="Unknown", extra_fields=None):
    """
    Save summary.json with standardized structure.
    
    Args:
        result_dir: Output directory (e.g., results/Relativistic/REL-01/)
        test_id: Test identifier
        description: Human-readable test description
        status: "PASS", "FAIL", "ERROR"
        metrics: Dict of numeric results
        tier: Tier number (1-4)
        category: "Relativistic", "Gravity", "Energy", "Quantization"
        extra_fields: Optional dict merged into summary (e.g., interference metrics)
    
    Output:
        result_dir/summary.json with keys:
        - tier, category, test_id, description
        - timestamp, status
        - metrics (with resource usage: CPU, memory, GPU)
        - [extra_fields if provided]
    """

def ensure_output_structure(result_dir):
    """
    Create standard output directories:
    - result_dir/
    - result_dir/diagnostics/
    - result_dir/plots/
    """
```

**Standard Output Structure:**

```
results/
├── MASTER_TEST_STATUS.csv
├── test_metrics_history.json
└── <Category>/
    └── <TEST-ID>/
        ├── summary.json          # Core results
        ├── run_log.txt           # Human logs
        ├── run_log.jsonl         # Structured events
        ├── diagnostics/          # CSV data
        │   ├── energy_log.csv
        │   ├── probe_serial.csv
        │   └── packet_tracking.csv
        └── plots/                # Visualizations
            ├── dispersion_*.png
            └── interference_*.png
```

---

## Physics Implementation

### Klein-Gordon Equation

**Continuous Form:**

$$\frac{\partial^2 E}{\partial t^2} = c^2 \nabla^2 E - \chi^2 c^4 E - \gamma \frac{\partial E}{\partial t}$$

**Discrete Update Rule (Leapfrog):**

$$E^{n+1} = 2E^n - E^{n-1} + \Delta t^2 \left[ c^2 \nabla^2_h E^n - \chi^2 E^n - \gamma \frac{E^n - E^{n-1}}{\Delta t} \right]$$

Where:
- $E^n$ is field at time step $n$
- $\Delta t$ is time step
- $\nabla^2_h$ is discrete Laplacian
- $c^2 = \alpha/\beta$ is wave speed squared
- $\chi$ is mass-like parameter
- $\gamma$ is damping coefficient

**Stability Condition (CFL):**

$$c \frac{\Delta t}{\Delta x} \leq \frac{1}{\sqrt{D}}$$

Where $D$ is spatial dimensions (1, 2, or 3).

### Energy Conservation

**Total Hamiltonian:**

$$H = \int dV \left[ \frac{1}{2}\left(\frac{\partial E}{\partial t}\right)^2 + \frac{1}{2}c^2(\nabla E)^2 + \frac{1}{2}\chi^2 c^4 E^2 \right]$$

Components:
- **Kinetic Energy (KE):** $\frac{1}{2}\int (\partial E/\partial t)^2 dV$
- **Gradient Energy (GE):** $\frac{1}{2}\int c^2 (\nabla E)^2 dV$
- **Potential Energy (PE):** $\frac{1}{2}\int \chi^2 c^4 E^2 dV$

**Discrete Approximation:**

```python
E_dot = (E - E_prev) / dt  # Temporal derivative
KE = 0.5 * dx**D * np.sum(E_dot**2)

grad_E = np.gradient(E, dx)  # Spatial gradient
GE = 0.5 * c**2 * dx**D * np.sum(grad_E**2)

PE = 0.5 * chi**2 * c**4 * dx**D * np.sum(E**2)

H_total = KE + GE + PE
```

**Expected Behavior:**
- **Undamped (γ=0, no boundaries):** $H$ conserved to machine precision
- **With damping (γ>0):** $H(t) = H_0 \exp(-2\gamma t)$ (exponential decay)
- **With absorbing boundaries:** $H$ decreases as waves exit domain

---

## Configuration System

### Hierarchical Config Loading

Config files support inheritance for DRY principle:

**masterconfig.json** (base parameters):
```json
{
  "parameters": {
    "dt": 0.01,
    "dx": 0.1,
    "alpha": 1.0,
    "beta": 1.0,
    "c": 1.0
  }
}
```

**config_tier1_relativistic.json** (inherits + overrides):
```json
{
  "inherits": ["masterconfig.json"],
  "run_settings": {
    "use_gpu": false,
    "quick_mode": false
  },
  "parameters": {
    "chi": 0.0,
    "gamma_damp": 0.0
  },
  "variants": [
    {
      "test_id": "REL-01",
      "description": "Isotropy — Coarse Grid",
      "k_fraction": 0.05
    }
  ]
}
```

**Loading Pattern:**

```python
def load_config(cfg_name: str):
    """
    Load config with inheritance.
    
    1. Read target config file
    2. If 'inherits' key present, load parent configs recursively
    3. Merge: parent → child (child overrides parent)
    4. Return merged config
    """
    cfg_path = Path("config") / cfg_name
    cfg = json.loads(cfg_path.read_text())
    
    merged = {}
    if "inherits" in cfg:
        for parent_name in cfg["inherits"]:
            parent = load_config(parent_name)
            merged.update(parent)
    
    merged.update(cfg)
    return merged
```

### Key Configuration Sections

```json
{
  "run_settings": {
    "use_gpu": false,              // CuPy vs NumPy backend
    "quick_mode": false,            // Reduced resolution for fast tests
    "show_progress": true,          // Print per-step progress
    "progress_percent_stride": 5    // Progress reporting interval (%)
  },
  
  "parameters": {
    // Discretization
    "dt": 0.01,                     // Time step
    "dx": 0.1,                      // Spatial step
    
    // Physics
    "alpha": 1.0,                   // Wave speed numerator
    "beta": 1.0,                    // Wave speed denominator (c²=α/β)
    "chi": 0.0,                     // Mass-like parameter
    "gamma_damp": 0.0,              // Damping coefficient
    
    // Numerics
    "boundary": "periodic",         // 'periodic' or 'absorbing'
    "stencil_order": 2,            // 2, 4, or 6 (Laplacian accuracy)
    "precision": "float64",         // 'float64' or 'float32'
    
    // Parallel
    "threads": 4,                   // Worker threads (lfm_parallel.py)
    "tiles": 4,                     // Spatial tiles (lfm_parallel.py)
    
    // Energy tracking
    "energy_lock": false,           // Rescale to conserve energy exactly
    "enable_monitor": true,         // Log energy per step
    "monitor_outdir": "diagnostics",
    "monitor_label": ""
  },
  
  "debug": {
    "enable_diagnostics": true,     // CFL checks, NaN detection
    "energy_tol": 0.01,            // Max relative energy drift (1%)
    "check_nan": true,             // Halt on NaN/Inf
    "profile_steps": false,        // Time each step (slow)
    "edge_band": 5,                // Edge effect monitoring width
    "checksum_stride": 100,        // Field checksum interval
    "diagnostics_path": "diagnostics"
  },
  
  "diagnostics": {
    "save_snapshots": false,        // HDF5 volumetric snapshots
    "snapshot_stride": 100,         // Snapshot interval
    "save_time_series": false,      // Probe time series
    "probe_locations": [[64, 64]],  // Probe positions
    "track_packet": false,          // Track wave packet centroid
    "log_packet_stride": 100        // Packet logging interval
  },
  
  "tolerances": {
    "energy_drift_max": 0.01,       // 1% max drift for PASS
    "frequency_error_max": 0.02,    // 2% max frequency error
    "phase_error_max": 0.05         // 5% max phase error
  }
}
```

---

## Test Harness Pattern

### BaseTierHarness Usage

**Every tier runner follows this pattern:**

```python
from lfm_test_harness import BaseTierHarness

class Tier1Harness(BaseTierHarness):
    """
    Tier 1: Relativistic propagation tests.
    
    Inherits from BaseTierHarness:
    - self.cfg: Loaded configuration
    - self.xp: Backend (np or cp)
    - self.logger: LFMLogger instance
    - self.estimate_omega_fft(): Frequency measurement
    - self.check_cfl_stability(): CFL validation
    """
    
    def __init__(self):
        cfg = self.load_config("config_tier1_relativistic.json")
        out_root = Path("results/Relativistic")
        super().__init__(cfg, out_root, "config_tier1_relativistic.json")
    
    def run_variant(self, variant: Dict) -> TestSummary:
        """
        Execute single test variant.
        
        Pattern:
        1. Extract parameters from variant + base config
        2. Initialize fields (E, E_prev)
        3. Run time integration (advance or run_lattice)
        4. Measure observables (frequency, energy, etc.)
        5. Calculate errors vs theory
        6. Generate plots
        7. Save outputs (summary.json, CSVs, plots)
        8. Return TestSummary
        """
        test_id = variant["test_id"]
        result_dir = self.out_root / test_id
        
        # Setup
        from lfm_results import ensure_output_structure
        ensure_output_structure(result_dir)
        
        # Initialize
        E, E_prev = self.init_field_variant(variant)
        
        # Run
        from lfm_equation import advance
        E_final, E_prev_final = advance(E, E_prev, steps, params)
        
        # Analyze
        omega_meas = self.estimate_omega_fft(E_time_series, dt)
        omega_theory = self.theory_dispersion(k, chi, c)
        error = abs(omega_meas - omega_theory) / omega_theory
        
        # Validate
        status = "PASS" if error < tolerance else "FAIL"
        
        # Save
        from lfm_results import save_summary
        save_summary(
            result_dir, test_id, variant["description"],
            status, {"error": error, "omega_meas": omega_meas},
            tier=1, category="Relativistic"
        )
        
        return TestSummary(test_id, status, error)
```

### Standard Test Flow

```python
def main():
    """
    Standard tier runner main function.
    
    1. Parse command-line args (--test, --quick, --gpu)
    2. Load configuration
    3. Initialize harness
    4. If --test specified: run single variant
    5. Else: run all variants
    6. Collect results
    7. Print summary
    8. Update MASTER_TEST_STATUS.csv
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="Run single test (e.g., REL-01)")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--gpu", action="store_true", help="Force GPU")
    args = parser.parse_args()
    
    harness = Tier1Harness()
    
    if args.test:
        # Single test
        variant = harness.find_variant(args.test)
        result = harness.run_variant(variant)
        print(f"[{result.status}] {result.test_id}")
    else:
        # Full suite
        results = []
        for variant in harness.cfg["variants"]:
            result = harness.run_variant(variant)
            results.append(result)
        
        # Summary
        passed = sum(1 for r in results if r.status == "PASS")
        print(f"\nTier 1 Results: {passed}/{len(results)} passed")
```

---

## Adding New Tests

### Step 1: Add Test Variant to Config

```json
{
  "variants": [
    {
      "test_id": "REL-99",
      "description": "My New Test",
      "mode": "custom_mode",
      "N": 256,
      "steps": 2000,
      "custom_param": 0.5
    }
  ]
}
```

### Step 2: Implement Test Logic in Harness

```python
def run_variant(self, variant: Dict) -> TestSummary:
    test_id = variant["test_id"]
    mode = variant.get("mode", "default")
    
    if mode == "custom_mode":
        return self.run_custom_mode_variant(variant)
    # ... other modes ...

def run_custom_mode_variant(self, variant: Dict) -> TestSummary:
    """
    Implement new test mode.
    
    Checklist:
    - [ ] Load parameters from variant
    - [ ] Initialize fields appropriately
    - [ ] Run simulation
    - [ ] Compute observables
    - [ ] Compare to theory
    - [ ] Generate diagnostics plot
    - [ ] Save outputs
    - [ ] Return TestSummary
    """
    test_id = variant["test_id"]
    result_dir = self.out_root / test_id
    ensure_output_structure(result_dir)
    
    # Implementation here
    
    return TestSummary(
        test_id=test_id,
        status=status,
        primary_metric=error
    )
```

### Step 3: Add Expected Outputs to test_output_requirements.py

```python
SPECIAL_TEST_REQUIREMENTS["REL-99"] = {
    "description": "My New Test",
    "additional_outputs": [
        "plots/custom_plot.png",
        "diagnostics/custom_data.csv"
    ],
    "validation": {
        "custom_data.csv": {
            "type": "csv",
            "required_columns": ["x", "value"],
            "min_rows": 100
        }
    }
}
```

### Step 4: Add Documentation

Update this guide with:
- Test purpose and physics
- Expected outputs
- Success criteria
- Known issues (if any)

---

## Backend Abstraction (CPU/GPU)

### Using Backend-Agnostic Code

**Pattern:**

```python
from lfm_backend import pick_backend

# Select backend based on config
use_gpu = params.get("use_gpu", False)
xp, use_gpu = pick_backend(use_gpu)

# Now use 'xp' instead of 'np' or 'cp'
E = xp.zeros((N, N, N), dtype=xp.float64)
E_fft = xp.fft.fftn(E)
result = xp.sum(E**2)

# Convert to NumPy for plotting/CSV (if needed)
if use_gpu:
    E_cpu = E.get()  # CuPy → NumPy
else:
    E_cpu = E  # Already NumPy

# Plot
import matplotlib.pyplot as plt
plt.imshow(E_cpu[N//2, :, :])  # Requires NumPy array
```

### Device Transfer Guidelines

**Minimize transfers:**
```python
# BAD: Repeated GPU → CPU transfers in loop
for step in range(steps):
    E = lattice_step_gpu(E, E_prev, params)
    E_cpu = E.get()  # Transfer every step (slow!)
    log_energy(E_cpu)

# GOOD: Keep on GPU, transfer only at end
for step in range(steps):
    E = lattice_step_gpu(E, E_prev, params)
E_cpu = E.get()  # Single transfer
log_energy(E_cpu)
```

**Helper function:**
```python
def to_numpy(arr):
    """Convert CuPy or NumPy array to NumPy."""
    if hasattr(arr, 'get'):
        return arr.get()  # CuPy
    return arr  # Already NumPy
```

---

## Common Patterns

### 1. FFT-Based Frequency Measurement

```python
def estimate_omega_fft(self, time_series, dt):
    """
    Estimate dominant frequency from time series.
    
    Args:
        time_series: 1D array of field values at fixed spatial point
        dt: Time step
    
    Returns:
        omega: Dominant angular frequency (rad/s)
    """
    # Apply Hann window to reduce spectral leakage
    window = self.xp.hanning(len(time_series))
    windowed = time_series * window
    
    # FFT
    fft = self.xp.fft.fft(windowed)
    power = self.xp.abs(fft)**2
    freqs = self.xp.fft.fftfreq(len(time_series), dt)
    
    # Find peak (positive frequencies only)
    positive_mask = freqs > 0
    peak_idx = self.xp.argmax(power[positive_mask])
    freq_peak = freqs[positive_mask][peak_idx]
    
    omega = 2 * np.pi * float(freq_peak)
    return omega
```

### 2. Wave Packet Initialization

```python
def init_gaussian_packet(N, dx, x0, sigma, k0, xp):
    """
    Create Gaussian wave packet.
    
    ψ(x) = exp(-((x-x0)/σ)²) * exp(i k0 x)
    
    Args:
        N: Grid points
        dx: Spatial step
        x0: Center position
        sigma: Width
        k0: Carrier wavenumber
        xp: Backend (np or cp)
    
    Returns:
        E: Real field (Re[ψ])
        E_prev: Field at t=-dt (for leapfrog)
    """
    x = xp.arange(N) * dx
    envelope = xp.exp(-((x - x0) / sigma)**2)
    carrier = xp.cos(k0 * x)
    E = envelope * carrier
    
    # Initialize E_prev for zero initial velocity
    E_prev = E.copy()
    
    return E, E_prev
```

### 3. Compensated Summation (Kahan)

```python
def kahan_sum(arr):
    """
    Numerically stable summation.
    
    Reduces rounding errors for large arrays.
    Use for energy calculations.
    """
    total = 0.0
    compensation = 0.0
    
    for value in arr.flat:
        y = value - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    
    return total
```

### 4. Progress Reporting

```python
def show_progress(step, total_steps, percent_stride=5):
    """
    Print progress at regular intervals.
    
    Args:
        step: Current step
        total_steps: Total steps
        percent_stride: Report every N percent
    """
    percent = 100 * step / total_steps
    if step == 0 or step == total_steps - 1:
        print(f"Progress: {percent:.1f}%")
    elif int(percent) % percent_stride == 0:
        last_percent = 100 * (step - 1) / total_steps
        if int(last_percent) // percent_stride != int(percent) // percent_stride:
            print(f"Progress: {percent:.1f}%")
```

---

## Debugging Guide

### Common Issues and Solutions

#### Issue 1: CFL Instability

**Symptoms:**
- Field values explode to NaN/Inf within first 100 steps
- Energy grows exponentially

**Diagnosis:**
```python
c = np.sqrt(params["alpha"] / params["beta"])
cfl_ratio = c * dt / dx
D = len(E.shape)  # Dimensions
cfl_limit = 1 / np.sqrt(D)

print(f"CFL ratio: {cfl_ratio:.4f}")
print(f"CFL limit: {cfl_limit:.4f}")
print(f"Stable: {cfl_ratio <= cfl_limit}")
```

**Solution:**
- Reduce `dt` or increase `dx` in config
- Use `check_cfl_stability()` before runs (already in BaseTierHarness)

#### Issue 2: Energy Drift

**Symptoms:**
- Energy increases or decreases monotonically
- Not explained by damping or boundaries

**Diagnosis:**
```python
monitor = EnergyMonitor(...)
# ... run simulation ...
abs_drift, rel_drift = monitor.get_drift()
print(f"Absolute drift: {abs_drift:.6e}")
print(f"Relative drift: {rel_drift:.2f}%")

# Check for systematic drift
energy_log = monitor.get_log()
plt.plot([e["step"] for e in energy_log], [e["total"] for e in energy_log])
plt.show()
```

**Common Causes:**
1. **Leaking boundaries:** Use `boundary="absorbing"` or increase domain size
2. **Numerical dispersion:** Reduce `dt` or increase `stencil_order`
3. **Insufficient precision:** Use `precision="float64"` instead of `float32`
4. **External forcing:** Check for unintended source terms

#### Issue 3: Test Failure (Unexpected)

**Debug Steps:**

1. **Check logs:**
   ```bash
   cat results/<Category>/<TEST-ID>/run_log.txt
   tail -20 results/<Category>/<TEST-ID>/run_log.jsonl
   ```

2. **Inspect outputs:**
   ```bash
   ls results/<Category>/<TEST-ID>/diagnostics/
   cat results/<Category>/<TEST-ID>/summary.json
   ```

3. **Compare to baseline:**
   ```python
   # Run known-good version
   git checkout <last-good-commit>
   python run_tier1_relativistic.py --test REL-01
   
   # Compare outputs
   diff results/Relativistic/REL-01/summary.json <baseline-summary.json>
   ```

4. **Enable full diagnostics:**
   ```json
   {
     "debug": {
       "enable_diagnostics": true,
       "energy_tol": 0.001,
       "check_nan": true,
       "profile_steps": true
     }
   }
   ```

5. **Run in isolation:**
   ```bash
   python run_tier1_relativistic.py --test REL-01 --quick
   ```

#### Issue 4: GPU Out of Memory

**Symptoms:**
- `CuPy.OutOfMemoryError`
- System hangs during GPU allocation

**Solutions:**

1. **Reduce grid size:**
   ```json
   {"parameters": {"N": 128}}  // Instead of 256
   ```

2. **Use CPU:**
   ```json
   {"run_settings": {"use_gpu": false}}
   ```

3. **Clear GPU cache:**
   ```python
   import cupy as cp
   cp.get_default_memory_pool().free_all_blocks()
   ```

4. **Use managed memory:**
   ```python
   import cupy as cp
   cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
   ```

---

## AI Assistant Quick Reference

### When Asked to Modify Physics

**STOP and verify:**

1. Is this a physics change or a diagnostic/performance change?
2. If physics: Has user explicitly approved?
3. Run numeric regression test (see lfm_equation.py section)
4. Document expected behavior change

**Safe modifications:**
- Add logging/diagnostics
- Refactor for performance (preserve numerical results)
- Fix bugs (NaN handling, edge cases)
- Improve comments/documentation

**Requires review:**
- Change Laplacian stencil
- Modify time integration scheme
- Alter energy calculation
- Change boundary conditions

### When Adding New Features

**Checklist:**

- [ ] Add config parameters with defaults
- [ ] Update relevant harness (Tier1/2/3/4)
- [ ] Add expected outputs to `test_output_requirements.py`
- [ ] Write docstrings with Args/Returns/Examples
- [ ] Add test variant to config
- [ ] Run test and verify outputs
- [ ] Update this documentation
- [ ] Commit with descriptive message

### When Debugging Test Failures

**Investigation order:**

1. Read `run_log.txt` for error messages
2. Check `summary.json` for metrics
3. Inspect diagnostic CSVs
4. Look for NaN/Inf in field snapshots
5. Verify CFL condition
6. Compare to last passing run
7. Isolate with `--quick` flag
8. Enable full diagnostics

### When Reviewing Code

**Key questions:**

1. Does it preserve physics? (if touching lfm_equation.py)
2. Is it backend-agnostic? (uses `xp`, not `np` directly)
3. Does it handle GPU memory correctly? (minimize transfers)
4. Are outputs saved to standard locations? (use `lfm_results.py`)
5. Is logging structured? (use `LFMLogger.log_json()`)
6. Are errors handled gracefully? (raise informative exceptions)
7. Is documentation updated? (docstrings + this guide)

### Common Code Smells

**Anti-patterns to avoid:**

```python
# BAD: Hardcoded NumPy (breaks GPU)
import numpy as np
E = np.zeros((128, 128))

# GOOD: Use backend abstraction
from lfm_backend import pick_backend
xp, _ = pick_backend(use_gpu)
E = xp.zeros((128, 128))

# BAD: Silent failures
try:
    result = expensive_computation()
except:
    pass  # Lost error information!

# GOOD: Explicit error handling
try:
    result = expensive_computation()
except ValueError as e:
    logger.error(f"Computation failed: {e}")
    raise

# BAD: Mixing physics and I/O
def lattice_step(E, E_prev, params):
    E_next = physics_update(E, E_prev, params)
    plt.imshow(E_next)  # Side effect in physics function!
    return E_next

# GOOD: Separate concerns
def lattice_step(E, E_prev, params):
    return physics_update(E, E_prev, params)

# In caller:
E_next = lattice_step(E, E_prev, params)
if should_plot:
    plot_field(E_next)
```

### Performance Optimization Priorities

**Optimization order:**

1. **Algorithm first:** Better algorithm > GPU > threading > micro-optimization
2. **Measure before optimizing:** Use profiling, don't guess bottlenecks
3. **Optimize hot paths:** 90% of time is in 10% of code (find that 10%)
4. **Consider trade-offs:** Speed vs accuracy vs readability

**Quick profiling:**

```python
import time

# Profile single operation
start = time.perf_counter()
result = expensive_function()
elapsed = time.perf_counter() - start
print(f"Time: {elapsed:.3f}s")

# Profile with context manager
from contextlib import contextmanager

@contextmanager
def timer(label):
    start = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - start:.3f}s")

with timer("FFT computation"):
    fft = xp.fft.fftn(E)
```

---

## File Organization Reference

```
LFM/code/
├── Core Physics (DO NOT MODIFY WITHOUT REVIEW)
│   ├── lfm_equation.py          # Klein-Gordon solver
│   └── lfm_parallel.py          # Parallel runner
│
├── Infrastructure (Safe to modify)
│   ├── lfm_backend.py           # NumPy/CuPy abstraction
│   ├── lfm_config.py            # Config loading
│   ├── lfm_logger.py            # Logging
│   ├── lfm_results.py           # Output management
│   ├── lfm_plotting.py          # Visualization
│   ├── lfm_console.py           # Console output
│   ├── energy_monitor.py        # Energy tracking
│   ├── numeric_integrity.py     # Validation
│   └── lfm_test_harness.py      # Base harness
│
├── Tier Harnesses (Extend for new tests)
│   ├── run_tier1_relativistic.py
│   ├── run_tier2_gravityanalogue.py
│   ├── run_tier3_energy.py
│   └── run_tier4_quantization.py
│
├── Configuration (Edit for test parameters)
│   └── config/
│       ├── masterconfig.json
│       ├── config_tier1_relativistic.json
│       ├── config_tier2_gravityanalogue.json
│       ├── config_tier3_energy.json
│       └── config_tier4_quantization.json
│
├── Documentation (Update for features)
│   └── docs/
│       ├── DEVELOPER_GUIDE.md   # This file
│       ├── USER_GUIDE.md        # User-facing docs
│       ├── API_REFERENCE.md     # Function reference
│       └── INSTALL.md           # Installation
│
└── Tests (Add for new features)
    ├── tests/                    # Unit tests
    ├── test_output_requirements.py
    └── devtests/                 # Dev sanity checks
```

---

## Summary for AI Assistants

**When working on this codebase:**

1. **Physics in `lfm_equation.py` is sacred** — modify only with explicit approval + regression tests
2. **Use backend abstraction** — `xp` not `np`, minimize GPU ↔ CPU transfers
3. **Follow harness pattern** — inherit from `BaseTierHarness`, use standard flow
4. **Log structured events** — `logger.log_json()` for machine-parseable data
5. **Save outputs correctly** — `lfm_results.save_summary()`, standard directory structure
6. **Document thoroughly** — docstrings + update this guide for new features
7. **Validate before committing** — run affected tests, check for regressions

**Key invariants to preserve:**

- CFL stability condition enforced
- Energy conservation validated (undamped cases)
- Deterministic execution (parallel = serial results)
- Standard output structure (summary.json, plots/, diagnostics/)
- Backend-agnostic code (works on CPU and GPU)

**This guide is your map. Refer to it when:**

- Adding new tests (see "Adding New Tests")
- Debugging failures (see "Debugging Guide")
- Optimizing performance (see "Performance Optimization")
- Reviewing code (see "When Reviewing Code")
- Making physics changes (see "lfm_equation.py" section)

**Last updated:** 2025-11-01  
**Maintainer:** Greg D. Partin (latticefieldmediumresearch@gmail.com)

