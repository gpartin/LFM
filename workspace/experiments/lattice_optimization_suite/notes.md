# Lattice Optimization Suite - Research Notes

**Investigation**: Three approaches to optimizing 3D lattice updates while preserving physics

**Date Started**: 2025-11-05

**Motivation**: Current baseline updates ALL cells every timestep regardless of field activity. For sparse systems (orbiting bodies, localized wave packets), this wastes computation on "empty" space. Goal: Speed up computation WITHOUT breaking physics.

## Research Questions

1. Can we skip updates in regions with negligible field values?
2. How do we ensure energy conservation is maintained (< 10⁻⁴ drift)?
3. What's the performance vs. accuracy tradeoff for each approach?
4. Which algorithm is most robust for realistic physics simulations?

## Physics Constraints (Non-Negotiable)

- ✅ Energy conservation: |ΔE/E₀| < 10⁻⁴
- ✅ Wave propagation: Changes propagate at speed c (local causality)
- ✅ Periodic boundaries: No edge effects
- ✅ CFL stability: c·dt/dx < 1/√3 for 3D
- ✅ Laplacian accuracy: Must use canonical implementation from lfm_equation.py

## Three Algorithms to Test

### Algorithm 1: Active Region Mask (Conservative)
**Hypothesis**: Can safely skip cells far from particles if we maintain buffer zone

**Approach**:
- Define "active" cells: within radius R of any particle/source
- Add 2-cell buffer zone (ensures wave propagation captured)
- Update only active + buffer cells
- All other cells set to zero or previous value

**Expected**:
- Speedup: 2-5x for orbit (10-20% active cells)
- Energy error: < 10⁻⁵ (very safe)
- Risk: Low - buffer ensures causality

**Implementation**: Mask-based selection, existing fused kernel on active subset

---

### Algorithm 2: Gradient-Adaptive Culling (Moderate)
**Hypothesis**: Cells with zero gradient don't need updates (flat regions stable)

**Approach**:
- Track cells where |∇E| > threshold AND |E| > threshold
- Include neighbors of active cells (1-cell halo)
- Dynamically update mask each timestep based on gradient
- Uses actual field state, not just proximity to particles

**Expected**:
- Speedup: 3-8x for localized phenomena
- Energy error: 10⁻⁴ to 10⁻⁵ (threshold-dependent)
- Risk: Medium - threshold tuning critical

**Implementation**: Gradient magnitude computation + dynamic masking

---

### Algorithm 3: Wavefront Prediction (Physics-Informed)
**Hypothesis**: Wave equation is deterministic - predict where updates needed

**Approach**:
- Compute maximum wave speed from field: v_max = c·max(|∇E|/|E|)
- Only update cells within distance v_max·dt of "active" cells
- Active defined as: |E| > ε OR |∂E/∂t| > ε
- Predictive culling based on wave physics

**Expected**:
- Speedup: 4-10x for wave packets
- Energy error: < 10⁻⁴ (respects wave propagation)
- Risk: Medium-High - prediction might miss rare events

**Implementation**: Wavefront expansion logic + distance field

---

## Test Configuration

**Baseline**: Full lattice update (128³ grid, 300 steps)
- Uses: `src/core/lfm_equation.py::lattice_step()`
- Reference for correctness and performance

**Test Case**: Circular orbit
- Earth (chi=50.0) at center (64, 64, 64)
- Moon (chi=0.617) orbiting at r=32 units
- Initial velocity: tangential, v≈1.86 units/step
- Duration: 300 steps (~20% of full orbit)
- Physics check: Trajectory should be circular, energy conserved

**Hardware**: GPU (CuPy available)

**Metrics**:
1. **Energy conservation error**: |ΔE/E₀| at t=300
2. **Wall-clock time**: Total runtime in seconds
3. **Speedup**: Time_baseline / Time_optimized
4. **Memory usage**: Peak GPU memory (if available)
5. **Trajectory error**: RMS deviation from baseline orbit

## Success Criteria

**Must Pass (Gate 0 - Exploratory)**:
- Energy error < 5×10⁻⁴ (relaxed for wave packet dispersion)
- No NaN/Inf values
- Speedup > 1.0 (otherwise pointless)
- Correct wave propagation behavior

**Note**: Gaussian wave packets have inherent numerical dispersion (~2×10⁻⁴ drift). 
This is acceptable for performance experiments. Gate 2 validation would require < 10⁻⁴.

**Nice to Have**:
- Speedup > 3x
- Memory reduction > 50%
- Generalizes to other scenarios (wave packets, multiple bodies)

## Experimental Protocol

1. Run baseline 3 times, average performance
2. For each algorithm:
   - Run 3 times with same seed
   - Measure all metrics
   - Compare energy, trajectory to baseline
3. Generate comparison plots
4. Document findings in results/

## Status: Baseline Complete ✓

**Next Actions**:
1. Run Algorithm 1 (Active Mask) - compare to baseline
2. Run Algorithm 2 (Gradient-Adaptive) - compare to baseline  
3. Run Algorithm 3 (Wavefront Prediction) - compare to baseline
4. Generate comparison plots (runtime, energy, speedup)
5. Document findings in results/comparison_report.md

**Promotion Criteria** (revised for wave packet test):
- Energy error < 5×10⁻⁴ (relaxed threshold)
- Speedup > 1.2x (meaningful performance gain)
- No NaN/Inf values
- Generalizes to other scenarios

If criteria met → Promote to `experiments/candidates/`
If not → Document lessons learned, archive, try different approach

---

**Key Insight**: We're not trying to "improve" the physics - we're trying to avoid wasting computation where physics is trivial (zero field, zero gradient). The sacred Laplacian stays the same.

---

## Baseline Results (Wave Packet)

**Date**: 2025-11-05
**Configuration**: Gaussian wave packet (width=20 cells, amplitude=0.01, wavelength=32 cells)

**Metrics**:
- Runtime: 2.396s ± 0.026s
- Energy drift: 1.94×10⁻⁴ (acceptable for Gate 0)
- Active fraction: 84-96% (mean 91.4%)
- Theoretical speedup: 1.09x

**Key Finding**: Kinematic gravity (orbit test) produces ZERO E-field activity, defeating all field-based optimizations. Must use wave propagation test instead.

**Trade-off Discovered**: 
- Narrow packets (width < 16): Good localization (33-61% active) but poor energy conservation (> 1×10⁻³ drift)
- Wide packets (width ≥ 20): Excellent energy conservation (~2×10⁻⁴) but high activity (91%+)
- Plane waves: Perfect energy conservation but 100% active (no optimization potential)

**Conclusion**: Proceed with width=20 configuration. While active fraction is high (91%), optimizations may still provide speedup through reduced memory bandwidth and better cache utilization.
