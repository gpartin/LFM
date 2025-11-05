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

**Must Pass**:
- Energy error < 10⁻⁴ (primary)
- Trajectory error < 1% (orbit must look identical)
- No NaN/Inf values
- Speedup > 1.0 (otherwise pointless)

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

## Next Steps After Results

If successful (energy error < 10⁻⁴, speedup > 2x):
- Promote to `experiments/candidates/`
- Test on different scenarios (wave packet, multiple particles)
- Consider for formal validation (Gate 2)

If failed (energy error > 10⁻⁴):
- Document failure mode
- Adjust thresholds/parameters
- Archive if fundamentally flawed

---

**Key Insight**: We're not trying to "improve" the physics - we're trying to avoid wasting computation where physics is trivial (zero field, zero gradient). The sacred Laplacian stays the same.
