# Lattice Optimization Suite

**Status**: Experiment (Gate 0)  
**Created**: 2025-11-05  
**Objective**: Find optimal algorithm for 3D lattice updates while preserving physics

## Quick Start

```bash
# Run all experiments (baseline + 3 algorithms, 3 trials each)
cd experiments/lattice_optimization_suite
python run_all_experiments.py

# Or run individually:
python baseline_benchmark.py
python algorithm1_active_mask.py
python algorithm2_gradient_adaptive.py
python algorithm3_wavefront_prediction.py
```

### Wave Packet Benchmark (baseline vs fused)

The Gaussian wave packet benchmark focuses on field propagation quality and timing and now supports an optional fused GPU backend.

```powershell
# Small smoke test (GPU, fused)
python .\baseline_wave_packet.py --gpu --backend fused --grid 64 --steps 60 --trials 1

# 256^3 with tighter drift target (GPU, fused)
python .\baseline_wave_packet.py --gpu --backend fused --grid 256 --steps 150 --dt 0.018 --width 36 --trials 1

# Compare canonical baseline at same params
python .\baseline_wave_packet.py --gpu --backend baseline --grid 256 --steps 150 --dt 0.018 --width 36 --trials 1
```

Outputs are written under `experiments/lattice_optimization_suite/results/`:
- `baseline_wave_results.json` or `fused_wave_summary.json`
- `baseline_wave_active_fractions.csv`

## Problem Statement

Current LFM implementation updates ALL cells every timestep, regardless of field activity. For sparse systems (orbiting bodies, localized wave packets), this wastes computation on "empty" space.

**Goal**: Speed up computation WITHOUT breaking physics laws.

## Constraints (Non-Negotiable)

- ✅ Energy conservation: |ΔE/E₀| < 10⁻⁴
- ✅ Wave propagation: Changes propagate at speed c (local causality)
- ✅ Periodic boundaries: No edge effects
- ✅ CFL stability: c·dt/dx < 1/√3 for 3D
- ✅ Laplacian accuracy: Must use canonical implementation

## Test Case

**Circular Orbit (Earth-Moon Analog)**
- Grid: 128³ = 2,097,152 cells
- Duration: 300 timesteps
- Earth chi=50.0 (centered)
- Moon chi=0.617 (orbiting at r=32 cells)
- Physics: Kinematic gravity (a = k_acc × ∇χ)

**Why this test?**
- Realistic: Tests emergent gravity phenomenon
- Sparse: ~10-20% active cells (good for optimization)
- Conservative: Orbit must close (strict energy/momentum requirements)

## Three Algorithms

### 1. Active Region Mask (Conservative)
**File**: `algorithm1_active_mask.py`

**Approach**:
- Update cells within fixed radius R of particles
- Add 2-cell buffer for wave propagation safety
- Freeze all other cells (maintain previous values)

**Expected**:
- Speedup: 2-5x
- Energy error: < 10⁻⁵ (very safe)
- Risk: LOW

**Best for**: Production use, maximum physics safety

---

### 2. Gradient-Adaptive Culling (Moderate)
**File**: `algorithm2_gradient_adaptive.py`

**Approach**:
- Dynamic mask based on |E| and |∇E| thresholds
- Adapts each timestep to field state
- Includes 1-cell halo around active regions

**Expected**:
- Speedup: 3-8x
- Energy error: 10⁻⁴ to 10⁻⁵
- Risk: MEDIUM (threshold-dependent)

**Best for**: Predictable field localization

---

### 3. Wavefront Prediction (Physics-Informed)
**File**: `algorithm3_wavefront_prediction.py`

**Approach**:
- Predict wave propagation using c·dt
- Source cells: |E| > ε OR |∂E/∂t| > ε
- Update only cells within wavefront distance

**Expected**:
- Speedup: 4-10x
- Energy error: < 10⁻⁴
- Risk: MEDIUM-HIGH (most complex)

**Best for**: Highly localized wave phenomena

---

## Metrics Collected

1. **Energy conservation error**: |ΔE/E₀| (PRIMARY - must be < 10⁻⁴)
2. **Wall-clock time**: Total runtime (seconds)
3. **Speedup factor**: Time_baseline / Time_optimized
4. **Memory usage**: Peak allocation (MB)
5. **Active cells**: Percentage updated each step
6. **Trajectory accuracy**: Orbit deviation from baseline

## Results Location

```
results/
├── baseline_summary.json          # Baseline performance (3 trials)
├── baseline_trajectory.csv        # Moon position over time
├── algorithm1_summary.json        # Algorithm 1 results
├── algorithm2_summary.json        # Algorithm 2 results
├── algorithm3_summary.json        # Algorithm 3 results
├── comparison_report.txt          # Human-readable comparison
└── comparison_data.json           # Machine-readable comparison
```

## Success Criteria

**Must Pass** (Gate 1: Reproducibility):
- [x] Energy error < 10⁻⁴ (primary physics constraint)
- [x] Trajectory error < 1% vs. baseline
- [x] No NaN/Inf values
- [x] Speedup > 1.0 (otherwise pointless)
- [x] Runs without manual intervention
- [x] Results reproduce within 5% variance

**Nice to Have**:
- [ ] Speedup > 3x
- [ ] Memory reduction > 50%
- [ ] Generalizes to other scenarios

## Next Steps

### If Successful (Energy < 10⁻⁴, Speedup > 2x):
1. Promote to `experiments/candidates/` using:
   ```bash
   python ../../tools/promote_experiment.py lattice_optimization_suite
   ```
2. Test on additional scenarios:
   - Wave packet propagation
   - Multiple orbiting bodies
   - Different grid sizes (64³, 256³)
3. Run reproducibility tests on different machines/GPUs
4. Consider for formal validation (Gate 2)

### If Failed (Energy > 10⁻⁴):
1. Document failure mode in notes.md
2. Adjust parameters:
   - Increase buffer zones (Algorithm 1)
   - Lower thresholds (Algorithms 2 & 3)
   - Add safety factors
3. Re-run experiments
4. If fundamentally flawed, archive and document lessons learned

## Files

- `notes.md` - Research notes and hypothesis
- `baseline_benchmark.py` - Reference implementation
- `algorithm1_active_mask.py` - Conservative approach
- `algorithm2_gradient_adaptive.py` - Adaptive approach
- `algorithm3_wavefront_prediction.py` - Predictive approach
- `run_all_experiments.py` - Master runner
- `README.md` - This file

## Dependencies

- Python 3.8+
- NumPy
- CuPy (optional, for GPU acceleration)
- LFM core modules (src/core/lfm_equation.py, src/core/lfm_backend.py)

## Estimated Runtime

- **With GPU**: 10-15 minutes total (4 experiments × 3 trials)
- **With CPU**: 30-60 minutes total

## Key Insights

**Physics First**: We're not approximating the Laplacian or the wave equation. We're just avoiding computation where the field is trivial (zero everywhere). The sacred Klein-Gordon equation stays intact.

**Conservative by Design**: All three algorithms prioritize correctness over speed. Better to be slow and correct than fast and wrong.

**Reproducibility**: Each experiment runs 3 trials with different seeds to verify statistical consistency.

---

**Remember**: The goal is to support discovery while maintaining rigor. These experiments are exploratory - failure is a valid scientific outcome!
