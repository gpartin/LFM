# LFM Performance Directory

This directory tracks computational performance optimization efforts for the Lattice Field Model. Performance work follows a **separate pipeline** from physics validation.

## Key Principle: Physics ≠ Performance

**Physics validation** answers: "Is this result scientifically correct?"
**Performance optimization** answers: "Can we compute this result faster/cheaper?"

These are **independent concerns**. A slow simulation can be scientifically valid. A fast simulation can be numerically wrong. Never conflate speed with correctness.

## Directory Structure

```
performance/
├── README.md                    # This file
├── benchmarks/                  # Performance baselines
│   ├── {test_name}_baseline.py # Original implementation
│   ├── {test_name}_profile.py  # Profiling script
│   └── {test_name}_report.md   # Performance analysis
└── optimizations/               # Optimization attempts
    ├── {optimization_name}/     # E.g., sparse_lattice_v1
    │   ├── implementation.py   # Optimized code
    │   ├── validation.py       # Correctness tests
    │   ├── benchmark.py        # Speed comparison
    │   └── report.md           # Results and analysis
    └── active/                 # Currently testing
```

## Performance Workflow

### Step 1: Establish Baseline
**Location**: `performance/benchmarks/`

Before optimizing, measure current performance:

```bash
# Profile a validated test
python tools/create_performance_baseline.py tier2/gravity/test_circular_orbits.py

# This creates:
# - benchmarks/circular_orbits_baseline.py (reference implementation)
# - benchmarks/circular_orbits_profile.py (profiling harness)
# - benchmarks/circular_orbits_report.md (performance metrics)
```

**Baseline Metrics**:
- Wall-clock time (total runtime)
- Memory usage (peak and average)
- GPU utilization
- CPU utilization
- Cache hit rates
- I/O operations

**Tools**: `cProfile`, `line_profiler`, `nvidia-smi`, `memory_profiler`

---

### Step 2: Identify Bottlenecks

Analyze profile data to find:
- Hot loops (>10% of runtime)
- Memory allocations
- Data transfers (CPU↔GPU)
- Redundant computations
- I/O waits

**Document findings** in baseline report with specific line numbers and time percentages.

---

### Step 3: Develop Optimization
**Location**: `performance/optimizations/{name}/`

Create optimization-specific directory:

```bash
mkdir performance/optimizations/sparse_lattice_v1
cd performance/optimizations/sparse_lattice_v1
```

**Required files**:
1. `implementation.py` - Optimized algorithm
2. `validation.py` - Proves correctness vs baseline
3. `benchmark.py` - Measures speedup
4. `report.md` - Documents results

**Golden Rule**: **Validate first, benchmark second**

---

### Step 4: Validate Correctness

The optimization MUST produce identical results (within numerical precision):

```python
# validation.py template
import numpy as np
from benchmarks.circular_orbits_baseline import baseline_run
from optimizations.sparse_lattice_v1.implementation import optimized_run

def test_identical_results():
    baseline_output = baseline_run(seed=42)
    optimized_output = optimized_run(seed=42)
    
    # Check trajectory positions
    np.testing.assert_allclose(
        baseline_output['positions'],
        optimized_output['positions'],
        rtol=1e-5,  # Relative tolerance
        atol=1e-8   # Absolute tolerance
    )
    
    # Check energies
    np.testing.assert_allclose(
        baseline_output['energies'],
        optimized_output['energies'],
        rtol=1e-5,
        atol=1e-8
    )
    
    print("✓ Optimization produces identical results")

if __name__ == '__main__':
    test_identical_results()
```

**If validation fails**: Fix the optimization. Never compromise correctness for speed.

---

### Step 5: Benchmark Performance

Only after validation passes:

```python
# benchmark.py template
import time
import numpy as np
from benchmarks.circular_orbits_baseline import baseline_run
from optimizations.sparse_lattice_v1.implementation import optimized_run

def benchmark(n_runs=10):
    baseline_times = []
    optimized_times = []
    
    for i in range(n_runs):
        # Baseline
        t0 = time.perf_counter()
        baseline_run(seed=42+i)
        baseline_times.append(time.perf_counter() - t0)
        
        # Optimized
        t0 = time.perf_counter()
        optimized_run(seed=42+i)
        optimized_times.append(time.perf_counter() - t0)
    
    baseline_mean = np.mean(baseline_times)
    optimized_mean = np.mean(optimized_times)
    speedup = baseline_mean / optimized_mean
    
    print(f"Baseline:   {baseline_mean:.3f}s ± {np.std(baseline_times):.3f}s")
    print(f"Optimized:  {optimized_mean:.3f}s ± {np.std(optimized_times):.3f}s")
    print(f"Speedup:    {speedup:.2f}x")
    
    return speedup

if __name__ == '__main__':
    speedup = benchmark()
    with open('speedup.txt', 'w') as f:
        f.write(f"{speedup:.2f}x\n")
```

**Benchmark Requirements**:
- Multiple runs (n≥10) for statistical reliability
- Warm-up runs (exclude first run if needed)
- Same hardware for baseline vs optimized
- Document GPU model, driver version, CUDA version
- Report mean, std dev, and confidence intervals

---

### Step 6: Document Results

Create `report.md` with:

```markdown
# Sparse Lattice Optimization V1

## Optimization Strategy
[Describe what you changed and why]

## Validation Results
✓ Passed validation.py
- Position error: < 1e-6 (relative)
- Energy error: < 1e-7 (relative)

## Performance Results
- Baseline: 12.3s ± 0.4s
- Optimized: 3.1s ± 0.2s
- **Speedup: 3.97x**

## Hardware
- GPU: NVIDIA RTX 3090
- CUDA: 11.8
- Driver: 522.06
- Python: 3.10.12
- CuPy: 12.0.0

## Memory Usage
- Baseline: 4.2 GB
- Optimized: 1.8 GB
- **Reduction: 57%**

## Trade-offs
[Any limitations or caveats]

## Next Steps
[Ideas for further optimization]
```

---

## Optimization Categories

### 1. Algorithmic Optimizations
- Sparse storage for low-density fields
- Adaptive mesh refinement (AMR)
- Fast multipole methods
- FFT-based convolutions

### 2. Memory Optimizations
- In-place operations
- Memory pooling
- Reduced precision (FP16 where safe)
- Lazy evaluation

### 3. Parallelization
- Multi-GPU distribution
- CPU-GPU pipeline overlap
- Vectorization (SIMD)
- Thread pool optimization

### 4. I/O Optimizations
- Compressed output formats
- Deferred file writes
- Memory-mapped arrays
- Incremental checkpointing

---

## Best Practices

### Do's ✓
- **Always validate before benchmarking**
- Profile before optimizing (measure, don't guess)
- Optimize hot paths first (Pareto principle)
- Document trade-offs honestly
- Keep baseline implementations for comparison
- Test on representative workloads
- Report negative results (failed optimizations teach us too)

### Don'ts ✗
- Don't optimize without profiling
- Don't sacrifice correctness for speed
- Don't cherry-pick favorable benchmarks
- Don't extrapolate from tiny test cases
- Don't assume GPU is always faster
- Don't conflate speed with scientific validity

---

## Integration with Physics Validation

Once an optimization is **validated and benchmarked**:

1. **Create pytest in `tests/performance/`**:
   ```python
   # tests/performance/test_sparse_lattice_speedup.py
   def test_sparse_lattice_speedup():
       """Ensure sparse lattice optimization maintains >3x speedup"""
       speedup = benchmark_sparse_lattice()
       assert speedup >= 3.0, f"Speedup {speedup:.2f}x below threshold"
   ```

2. **Add to CI/CD** (optional, for critical optimizations)

3. **Update physics tests** to use optimized code (if speedup > 2x and well-validated)

4. **Document in PERFORMANCE_OPTIMIZATIONS_README.md**

---

## Current Optimization Phases

As documented in `PERFORMANCE_OPTIMIZATIONS_README.md`:

- **Phase 1**: Memory layout and GPU memory management (COMPLETED)
- **Phase 2**: Fused CUDA kernels (IN PROGRESS)
- **Phase 3**: Multi-scale AMR (PLANNED)

Each phase follows this workflow:
1. Baseline → 2. Bottleneck analysis → 3. Implementation → 4. Validation → 5. Benchmark → 6. Documentation

---

## Tools Reference

Performance measurement tools:
- `python -m cProfile -o profile.stats script.py` - CPU profiling
- `python -m memory_profiler script.py` - Memory profiling
- `nvidia-smi dmon -s pucvmet -i 0` - GPU monitoring
- `py-spy record -o profile.svg -- python script.py` - Flamegraph profiling

LFM-specific tools:
- `tools/create_performance_baseline.py {test}` - Generate baseline
- `tools/validate_optimization.py {opt_name}` - Run validation suite
- `tools/benchmark_optimization.py {opt_name}` - Run benchmarks
- `tools/profile_test.py {test}` - Profile a specific test

---

## Performance != Validation

**Final reminder**: Performance optimization is engineering, not science. The scientific claims live in `tests/`. The speed improvements live here in `performance/`. Keep them separate in your mind and in your code.

Fast wrong answers are worthless. Slow correct answers are valuable. Fast correct answers are ideal.
