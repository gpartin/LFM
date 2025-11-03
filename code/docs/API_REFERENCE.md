# LFM API Reference

Quick reference for all public functions and classes. For detailed implementation notes, see `DEVELOPER_GUIDE.md`.

**License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
**Author:** Greg D. Partin | LFM Research  
**Contact:** latticefieldmediumresearch@gmail.com  
**DOI:** [10.5281/zenodo.17478758](https://zenodo.org/records/17478758)  
**Repository:** [OSF: osf.io/6agn8](https://osf.io/6agn8)  
**Last Updated:** 2025-11-01

---

## Core Physics (lfm_equation.py)

### `lattice_step(E, E_prev, params) -> E_next`

Single Klein-Gordon time step using leapfrog integration.

**Parameters:**
- `E` (ndarray): Current field (N,) or (N,N) or (N,N,N)
- `E_prev` (ndarray): Previous field (same shape as E)
- `params` (dict): Physics parameters
  - `dt` (float): Time step
  - `dx` (float): Spatial step
  - `alpha`, `beta` (float): Wave speed params (c²=α/β)
  - `chi` (float): Mass-like parameter
  - `gamma_damp` (float): Damping coefficient
  - `boundary` (str): 'periodic' or 'absorbing'
  - `stencil_order` (int): 2, 4, or 6
  - `use_gpu` (bool): Use CuPy backend

**Returns:**
- `E_next` (ndarray): Field at t+dt

**Example:**
```python
E_next = lattice_step(E, E_prev, params)
```

---

### `advance(E, E_prev, steps, params, callback=None) -> (E_final, E_prev_final)`

Serial time integration for multiple steps.

**Parameters:**
- `E`, `E_prev` (ndarray): Initial fields
- `steps` (int): Number of time steps
- `params` (dict): Physics parameters (see `lattice_step`)
- `callback` (callable, optional): Function called each step: `callback(step, E, E_prev, params)`

**Returns:**
- `E_final`, `E_prev_final` (ndarray): Fields at end of integration

**Example:**
```python
E_final, E_prev_final = advance(E, E_prev, steps=1000, params=params)
```

---

### `laplacian(E, dx, order=2, boundary='periodic') -> lap_E`

Discrete Laplacian with selectable stencil order.

**Parameters:**
- `E` (ndarray): Input field
- `dx` (float): Spatial step
- `order` (int): Stencil order (2, 4, or 6)
- `boundary` (str): 'periodic' or 'absorbing'

**Returns:**
- `lap_E` (ndarray): ∇²E (same shape as E)

**Stencil Accuracy:**
- `order=2`: O(dx²) error
- `order=4`: O(dx⁴) error
- `order=6`: O(dx⁶) error

---

### `energy_total(E, E_prev, dt, dx, params) -> dict`

Compute total Hamiltonian energy and components.

**Parameters:**
- `E`, `E_prev` (ndarray): Current and previous fields
- `dt`, `dx` (float): Time and spatial steps
- `params` (dict): Must include `chi`, `alpha`, `beta`

**Returns:**
- `dict` with keys:
  - `'total'` (float): H = KE + GE + PE
  - `'kinetic'` (float): KE = ½∫(∂E/∂t)²dV
  - `'gradient'` (float): GE = ½∫c²(∇E)²dV
  - `'potential'` (float): PE = ½∫χ²c⁴E²dV

**Example:**
```python
energy = energy_total(E, E_prev, dt, dx, params)
print(f"Total energy: {energy['total']:.6e}")
print(f"Kinetic: {energy['kinetic']:.6e}")
```

---

### `core_metrics(E, E_prev, dt, dx, params) -> dict`

Compute comprehensive diagnostics (energy + additional metrics).

**Returns:**
- All keys from `energy_total()` plus:
  - `'peak_amplitude'`: max(|E|)
  - `'rms_amplitude'`: √(⟨E²⟩)
  - `'gradient_magnitude'`: ⟨|∇E|⟩

---

## Parallel Execution (lfm_parallel.py)

### `run_lattice(E, E_prev, steps, params, callback=None) -> (E_final, E_prev_final)`

Parallel time integration using tile decomposition.

**Parameters:**
- Same as `advance()`
- `params` must additionally include:
  - `threads` (int): Number of worker threads
  - `tiles` (int): Number of spatial tiles

**Returns:**
- Same as `advance()`

**Note:** Results are deterministic (bit-for-bit identical to serial `advance()`).

**When to use:**
- 3D simulations (N³ grids)
- Large 2D grids (N≥256)
- Long runs (>10,000 steps)

**When NOT to use:**
- Small problems (overhead dominates)
- GPU runs (CuPy already parallelized)
- Quick tests (serial is faster)

---

## Energy Monitoring (energy_monitor.py)

### Class: `EnergyMonitor`

Track energy conservation during simulations.

#### `__init__(outdir, label="", energy_every=10)`

**Parameters:**
- `outdir` (Path): Directory for diagnostics
- `label` (str): Test identifier
- `energy_every` (int): Log interval (0 = disable)

---

#### `record(step, E, E_prev, dt, dx, params)`

Record energy at current step.

**Parameters:**
- `step` (int): Current time step
- `E`, `E_prev` (ndarray): Fields
- `dt`, `dx` (float): Time/spatial steps
- `params` (dict): Physics parameters

---

#### `save(filepath)`

Write energy log to CSV.

**Columns:**
- `step`, `time`, `E_total`, `E_kinetic`, `E_gradient`, `E_potential`, `drift`

---

#### `get_drift() -> (float, float)`

Calculate energy drift since start.

**Returns:**
- `abs_drift` (float): Absolute energy change
- `rel_drift` (float): Relative drift (percentage)

**Example:**
```python
monitor = EnergyMonitor(diag_dir, label="TEST-01", energy_every=25)

for step in range(steps):
    E, E_prev = lattice_step(E, E_prev, params)
    monitor.record(step, E, E_prev, dt, dx, params)

monitor.save(diag_dir / "energy_log.csv")
abs_drift, rel_drift = monitor.get_drift()
print(f"Energy drift: {rel_drift:.3f}%")
```

---

## Logging (lfm_logger.py)

### Class: `LFMLogger`

Dual-mode logging (human text + machine JSONL).

#### `__init__(output_dir)`

Create logger writing to:
- `output_dir/run_log.txt` (human-readable)
- `output_dir/run_log.jsonl` (structured events)

---

#### `info(message)`, `warning(message)`, `error(message)`

Log messages to both formats.

---

#### `log_json(event_type, data)`

Log structured event (JSONL only).

**Parameters:**
- `event_type` (str): Event category (e.g., "test_start", "error")
- `data` (dict): Event payload

**Example:**
```python
logger = LFMLogger(results_dir)
logger.record_env()  # Log environment info
logger.info("Starting test REL-01")
logger.log_json("test_start", {
    "test_id": "REL-01",
    "timestamp": time.time(),
    "params": {"N": 128, "steps": 1000}
})
```

---

#### `record_env()`

Log environment information (Python version, CPU, GPU, memory).

---

## Results Management (lfm_results.py)

### `save_summary(result_dir, test_id, description, status, metrics, tier, category, extra_fields=None)`

Save standardized summary.json.

**Parameters:**
- `result_dir` (Path): Output directory
- `test_id` (str): Test identifier (e.g., "REL-01")
- `description` (str): Human-readable description
- `status` (str): "PASS", "FAIL", or "ERROR"
- `metrics` (dict): Numeric results
- `tier` (int): Tier number (1-4)
- `category` (str): "Relativistic", "Gravity", "Energy", "Quantization"
- `extra_fields` (dict, optional): Additional fields to merge

**Output:** `result_dir/summary.json`

**Example:**
```python
save_summary(
    result_dir, "REL-01", "Isotropy test",
    "PASS", {"error": 0.0021, "omega_x": 0.314},
    tier=1, category="Relativistic"
)
```

---

### `ensure_output_structure(result_dir)`

Create standard output directories:
- `result_dir/`
- `result_dir/diagnostics/`
- `result_dir/plots/`

---

## Backend Abstraction (lfm_backend.py)

### `pick_backend(use_gpu) -> (module, bool)`

Select NumPy or CuPy backend.

**Parameters:**
- `use_gpu` (bool): Request GPU if available

**Returns:**
- `xp` (module): `numpy` or `cupy`
- `gpu_available` (bool): Whether GPU is actually being used

**Example:**
```python
xp, use_gpu = pick_backend(use_gpu=True)
E = xp.zeros((256, 256))  # NumPy or CuPy array
```

---

### `to_numpy(arr) -> np.ndarray`

Convert CuPy or NumPy array to NumPy.

**Example:**
```python
E_cpu = to_numpy(E)  # Safe for both backends
plt.imshow(E_cpu)    # Plotting requires NumPy
```

---

## Configuration (lfm_config.py)

### `load_config_with_inheritance(config_name) -> dict`

Load JSON config with hierarchical inheritance.

**Parameters:**
- `config_name` (str): Config filename (e.g., "config_tier1_relativistic.json")

**Returns:**
- `config` (dict): Merged configuration

**Inheritance:**
If config contains `"inherits": ["parent.json"]`, parent is loaded first and child overrides.

**Example:**
```python
cfg = load_config_with_inheritance("config_tier1_relativistic.json")
dt = cfg["parameters"]["dt"]
```

---

## Test Harness Base (lfm_test_harness.py)

### Class: `BaseTierHarness`

Base class for all tier test harnesses.

#### `__init__(cfg, out_root, config_name)`

**Parameters:**
- `cfg` (dict): Loaded configuration
- `out_root` (Path): Root output directory
- `config_name` (str): Config filename (for error messages)

**Provides:**
- `self.cfg`: Configuration
- `self.xp`: Backend (numpy or cupy)
- `self.logger`: LFMLogger instance
- `self.use_gpu`: Whether GPU is active

---

#### `estimate_omega_fft(time_series, dt) -> float`

Estimate dominant frequency from time series using FFT.

**Parameters:**
- `time_series` (ndarray): Field values at fixed spatial point
- `dt` (float): Time step

**Returns:**
- `omega` (float): Dominant angular frequency (rad/s)

**Example:**
```python
omega = self.estimate_omega_fft(probe_data, dt)
```

---

#### `check_cfl_stability(dt, dx, c, dimensions)`

Validate CFL condition: c·dt/dx ≤ 1/√D.

**Raises:**
- `ValueError` if CFL violated

---

#### `check_field_validity(E, label="E")`

Check for NaN, Inf, or excessive values.

**Raises:**
- `ValueError` if field contains invalid values

---

## Numeric Integrity (numeric_integrity.py)

### Class: `NumericIntegrityMixin`

Validation methods inherited by `BaseTierHarness`.

Methods: See `check_cfl_stability()`, `check_field_validity()` above.

---

## Plotting Utilities (lfm_plotting.py)

### `plot_dispersion_spectrum(k_vals, omega_meas, omega_theory, output_path)`

Generate dual-panel dispersion plot (spectrum + bar chart).

**Parameters:**
- `k_vals` (array): Wavenumbers
- `omega_meas` (array): Measured frequencies
- `omega_theory` (array): Theoretical frequencies
- `output_path` (Path): Output PNG file

---

### `plot_interference_pattern(intensity_2d, y_coords, profile_1d, output_path)`

Generate interference pattern visualization.

**Parameters:**
- `intensity_2d` (ndarray): 2D intensity map
- `y_coords` (array): Y-axis coordinates
- `profile_1d` (array): 1D intensity profile
- `output_path` (Path): Output PNG file

---

### `plot_energy_evolution(energy_log, output_path)`

Plot energy components vs time.

**Parameters:**
- `energy_log` (list): List of dicts with keys: 'step', 'E_total', 'E_kinetic', 'E_gradient', 'E_potential'
- `output_path` (Path): Output PNG file

---

## Diagnostics (lfm_diagnostics.py)

### `save_field_snapshot(E, filepath, metadata=None)`

Save field to HDF5.

**Parameters:**
- `E` (ndarray): Field to save
- `filepath` (Path): Output HDF5 file
- `metadata` (dict, optional): Additional attributes

---

### `load_field_snapshot(filepath) -> ndarray`

Load field from HDF5.

**Returns:**
- `E` (ndarray): Loaded field

---

### `track_packet_centroid(E, dx) -> (float, float, float)`

Compute wave packet center of mass.

**Parameters:**
- `E` (ndarray): Field (1D, 2D, or 3D)
- `dx` (float): Spatial step

**Returns:**
- `x_cm`, `y_cm`, `z_cm` (float): Centroid coordinates (0.0 if dimension absent)

---

## Console Output (lfm_console.py)

### `log(message, level="INFO")`

Print formatted log message.

---

### `test_start(test_id, desc, steps=None)`

Print test start banner.

---

### `test_pass(test_id, metric=None)`

Print PASS message with optional metric.

---

### `test_fail(test_id, reason=None)`

Print FAIL message with optional reason.

---

### `progress_bar(current, total, width=50)`

Display progress bar.

**Example:**
```python
for i in range(total):
    # ... work ...
    progress_bar(i+1, total)
```

---

## Field Initialization (lfm_fields.py)

### `init_gaussian_packet(N, dx, x0, sigma, k0, xp)`

Create Gaussian wave packet.

**Parameters:**
- `N` (int): Grid points
- `dx` (float): Spatial step
- `x0` (float): Center position
- `sigma` (float): Width
- `k0` (float): Carrier wavenumber
- `xp` (module): Backend (numpy or cupy)

**Returns:**
- `E`, `E_prev` (ndarray): Initial fields

---

### `init_plane_wave(N, dx, k, xp)`

Create plane wave exp(i k·x).

**Returns:**
- `E`, `E_prev` (ndarray): Initial fields

---

### `init_standing_wave(N, dx, k, xp)`

Create standing wave sin(k·x).

**Returns:**
- `E`, `E_prev` (ndarray): Initial fields

---

## Lorentz Transforms (lorentz_transform.py)

### `lorentz_boost_field(E, dx, dt, beta, xp)`

Apply Lorentz boost to field.

**Parameters:**
- `E` (ndarray): Field to boost
- `dx`, `dt` (float): Discretization
- `beta` (float): Boost velocity (units of c)
- `xp` (module): Backend

**Returns:**
- `E_boosted` (ndarray): Transformed field

---

## Quick Reference Table

| Task | Function | Module |
|------|----------|--------|
| Run simulation | `advance()` or `run_lattice()` | lfm_equation, lfm_parallel |
| Compute energy | `energy_total()` | lfm_equation |
| Track energy | `EnergyMonitor` | energy_monitor |
| Log events | `LFMLogger` | lfm_logger |
| Save results | `save_summary()` | lfm_results |
| Initialize field | `init_gaussian_packet()` | lfm_fields |
| Check CFL | `check_cfl_stability()` | BaseTierHarness |
| Plot results | `plot_*()` functions | lfm_plotting |
| Select backend | `pick_backend()` | lfm_backend |
| Load config | `load_config_with_inheritance()` | lfm_config |

---

## Type Signatures (Summary)

```python
# Core physics
lattice_step(E: ndarray, E_prev: ndarray, params: dict) -> ndarray
advance(E: ndarray, E_prev: ndarray, steps: int, params: dict, 
        callback: Optional[Callable] = None) -> Tuple[ndarray, ndarray]
energy_total(E: ndarray, E_prev: ndarray, dt: float, dx: float, 
             params: dict) -> dict

# Energy monitoring
class EnergyMonitor:
    def __init__(self, outdir: Path, label: str = "", energy_every: int = 10)
    def record(self, step: int, E: ndarray, E_prev: ndarray, 
               dt: float, dx: float, params: dict)
    def save(self, filepath: Path)
    def get_drift(self) -> Tuple[float, float]

# Logging
class LFMLogger:
    def __init__(self, output_dir: Path)
    def info(self, message: str)
    def log_json(self, event_type: str, data: dict)
    def record_env(self)

# Results
save_summary(result_dir: Path, test_id: str, description: str, 
             status: str, metrics: dict, tier: int, category: str,
             extra_fields: Optional[dict] = None)

# Backend
pick_backend(use_gpu: bool) -> Tuple[ModuleType, bool]
to_numpy(arr: Union[np.ndarray, cp.ndarray]) -> np.ndarray
```

---

For detailed usage examples and implementation notes, see:
- **User Guide:** `docs/USER_GUIDE.md`
- **Developer Guide:** `docs/DEVELOPER_GUIDE.md`

**License:** CC BY-NC-ND 4.0 — Non-commercial use only  
**Contact:** latticefieldmediumresearch@gmail.com

**Last Updated:** 2025-11-01

