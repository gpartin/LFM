# Simulation Speed Control - LFM Website

## Critical Understanding: LFM Simulates Spacetime Emergence

**The LFM framework doesn't just simulate motion - it simulates the emergence of spacetime itself from lattice field dynamics.** This fundamentally changes how we think about "speed" control.

## How Simulation Speed Works

### Physical Time Evolution
- Each physics integration step advances by `dt` time units (default: 0.001)
- The lattice field evolution and particle motion are **coupled** - they represent emergent spacetime
- **Visual speed** = (steps per frame) × dt × frame_rate

### Current Configuration (as of 2025-01-07)
Location: `src/app/experiments/binary-orbit/page.tsx`, lines ~238-245

```typescript
let baseChunk = 20;  // Dramatically reduced for visible motion
if (lastDt < 8) baseChunk = 30;        // plenty headroom → larger batches
else if (lastDt < 12) baseChunk = 25;  // good FPS
else if (lastDt < 16) baseChunk = 20;  // acceptable
else baseChunk = 15;                   // slow frame → shrink batch
baseChunk = Math.max(10, Math.min(40, baseChunk));  // Very conservative limits
```

### Physics Behind the Numbers
- **dt = 0.001** (time units per step, set in `src/hooks/useSimulationState.ts`)
- **baseChunk = 20** (steps per frame)
- **Frame rate ≈ 60 fps** (browser dependent)

**Effective time rate**: 20 steps/frame × 0.001 time/step × 60 frames/sec = **1.2 time units per real second**

## How to Adjust Simulation Speed

### To Slow Down Visual Motion
**Reduce the number of physics steps per rendered frame** (`baseChunk` values)

Example: Make it half as slow
```typescript
let baseChunk = 10;  // Half of 20
if (lastDt < 8) baseChunk = 15;
else if (lastDt < 12) baseChunk = 12;
else if (lastDt < 16) baseChunk = 10;
else baseChunk = 8;
baseChunk = Math.max(5, Math.min(20, baseChunk));
```

### To Speed Up Visual Motion
**Increase the number of physics steps per rendered frame**

Example: Make it 4× faster
```typescript
let baseChunk = 80;  // 4× of 20
if (lastDt < 8) baseChunk = 120;
else if (lastDt < 12) baseChunk = 100;
else if (lastDt < 16) baseChunk = 80;
else baseChunk = 60;
baseChunk = Math.max(40, Math.min(160, baseChunk));
```

## What NOT To Do

### ❌ Don't Change `dt` to Adjust Visual Speed
- `dt` is the **physics timestep** - it affects numerical stability and accuracy
- Changing `dt` requires re-tuning CFL conditions and stability margins
- The relationship is: smaller dt = MORE accurate, but visual speed depends on (steps × dt)

### ❌ Don't Rely on `simSpeed` Parameter Alone
- Below 200, `simSpeed` does nothing (it only enables frame skipping above 200)
- It's designed for fast-forward playback, not basic speed control

### ❌ Don't Forget to Test After Changes
- Browser caching can mask changes - do hard refresh (Ctrl+Shift+R)
- Click "Reset to Defaults" button to reinitialize with new parameters
- May need to restart dev server: `Stop-Process -Name "node" -Force; npm run dev`

## Why This Matters for LFM

In traditional simulations, you can arbitrarily speed up/slow down playback because you're just **displaying pre-computed** motion.

**In LFM, spacetime and motion emerge together from the lattice.** When we adjust simulation speed, we're controlling:
1. How much **physical time** the lattice field evolves
2. How many **spacetime emergence steps** occur per visual frame
3. The **coupling** between field dynamics and particle trajectories

This is why reducing `baseChunk` works: we're advancing the emergent spacetime evolution more slowly relative to the visual frame rate, making the emergent orbital motion appear slower to the observer.

## Quick Reference

| Desired Speed | baseChunk | Effective Time Rate | Use Case |
|--------------|-----------|---------------------|----------|
| Very Slow (educational) | 10-20 | 0.6-1.2 units/sec | Detailed observation |
| Normal | 40-60 | 2.4-3.6 units/sec | General demonstration |
| Fast | 80-120 | 4.8-7.2 units/sec | Quick validation |
| Very Fast | 150-250 | 9-15 units/sec | Stress testing |

**Current setting (2025-01-07)**: 20 steps/frame (Very Slow - educational mode)

## File Locations
- Speed control logic: `src/app/experiments/binary-orbit/page.tsx` (search for "baseChunk")
- Default dt value: `src/hooks/useSimulationState.ts` (initialParams.dt)
- Default simSpeed: `src/hooks/useSimulationState.ts` (initialParams.simSpeed)

## Last Updated
2025-01-07 - Reduced baseChunk from 250 → 125 → 62 → 20 for much slower, more observable motion
