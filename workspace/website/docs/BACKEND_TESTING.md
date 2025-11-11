# Testing the Multi-Backend System

## Quick Start

The development server should already be running at http://localhost:3000

### Test the Backend Detector

Visit the new backend test page:

```
http://localhost:3000/backend-test
```

This page will:
1. **Auto-detect** your GPU (or fall back to CPU)
2. Show detailed capabilities of your hardware
3. Let you **manually switch** backends (GPU users can try CPU mode)
4. Run a quick physics test to measure performance

## What You'll See

### If You Have a GPU (Most Likely):

**Auto-Detected Backend:**
- 🚀 GPU (WebGPU) - 64³ lattice max
- Real LFM Physics: ✓ Yes
- Energy Conservation: ✓ Verified
- Estimated FPS: ~60fps

**Backend Selector:**
- Dropdown showing: GPU (WebGPU) [Active] and CPU (JavaScript)
- Click to switch to CPU and see performance comparison
- CPU option will show: 🐌 CPU (JavaScript) - 32³ @ 15fps

**Quick Physics Test:**
- Click "Run Physics Test"
- See actual performance: steps/sec, energy drift, etc.
- Switch backends and run again to compare

### If You Don't Have a GPU (Rare):

**Auto-Detected Backend:**
- 🐌 CPU (JavaScript) - 32³ lattice max
- Real LFM Physics: ✓ Yes (same equation!)
- Energy Conservation: ✓ Verified
- Estimated FPS: ~15fps

**Backend Selector:**
- Only shows CPU option (GPU grayed out as "Hardware Required")
- Cannot upgrade to GPU (hardware limitation)

**Amber Banner:**
- Shows warning that you're on CPU fallback
- Explains physics is authentic, just slower
- Recommends browser upgrades or GPU acceleration

## Testing Experiments

After checking the backend test page, try the actual experiment:

```
http://localhost:3000/experiments/binary-orbit
```

The experiment will automatically use your detected backend, but you'll see:
- Backend indicator badge in the corner
- Smooth 60fps with GPU or acceptable 15fps with CPU
- Same physics regardless of backend

## Browser Compatibility Testing

To test different backends, try:

### Chrome/Edge (Best Support)
- Version 113+: WebGPU ✓
- Version 56+: WebGL2 ✓
- Always: CPU ✓

### Firefox
- Version 125+: WebGPU ✓ (enable in about:config)
- Version 51+: WebGL2 ✓
- Always: CPU ✓

### Safari
- Version 18+: WebGPU ✓
- Version 15+: WebGL2 ✓
- Always: CPU ✓

## Manual Backend Override (Developer Testing)

You can force a specific backend by modifying the test page code:

```typescript
// Force CPU mode (even if GPU available)
const { lattice } = await createLattice('cpu');

// Force GPU mode (will fail gracefully if unavailable)
const { lattice } = await createLattice('webgpu');
```

## Performance Benchmarks

Expected results on modern hardware:

| Backend | Lattice | 100 Steps | Steps/Sec | Energy Drift |
|---------|---------|-----------|-----------|--------------|
| WebGPU  | 64³     | ~2ms      | ~50,000   | <0.001%      |
| WebGL2  | 64³     | ~5ms      | ~20,000   | <0.001%      |
| CPU     | 32³     | ~100ms    | ~1,000    | <0.001%      |

*Note: Actual performance varies by device*

## Troubleshooting

### "WebGPU not available"
- Check: chrome://gpu (Chrome) or about:support (Firefox)
- Enable: Hardware acceleration in browser settings
- Update: Browser to latest version
- Platform: Some Linux distros need flags enabled

### "Backend test runs slow"
- Normal on CPU backend (32³ lattice is intentionally small)
- Check: Are you on a laptop in power-saving mode?
- Try: Plugging in laptop for full GPU performance
- Note: Mobile devices will always use CPU mode

### Physics looks different on CPU?
- **It shouldn't!** Same equation, same energy conservation
- If you see differences, file a bug report
- Check energy drift % - should be <0.01% on all backends

## Questions?

- Backend detection issues? Check browser console for WebGPU errors
- Want to add WebGL2 implementation? See `lattice-cpu.ts` as template
- Performance problems? Check `BackendBanner.tsx` recommendations
