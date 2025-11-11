# Visual Consistency Audit - Showcase Experiments

**Date**: 2025-11-11  
**Status**: AUDIT IN PROGRESS

---

## Inconsistencies Found

### 1. Background Color Inconsistency ✅ FIXED

**Issue**: Three-body experiment had different background color than other experiments.

**Previous State**:
- **binary-orbit, black-hole, stellar-collapse, big-bang**: Use OrbitCanvas with `#0a0e27` background
- **three-body**: Used NBodyCanvas with `#0a0a1a` background (WRONG)

**Files Fixed**:
- ✅ `src/components/visuals/NBodyCanvas.tsx` - Changed from `#0a0a1a` to `[0.039, 0.055, 0.152]` (equals `#0a0e27`)

**Validation**: Automated check now passes ✅
```
🎨 Checking Canvas Background Color Consistency...
  ✅ NBodyCanvas.tsx: Standard background color
  ✅ OrbitCanvas.tsx: Standard background color
```

---

### 2. Visualization Options Not Rendering (three-body) ❌

**Issue**: Three-body experiment passes `visualizationOptions` prop to ExperimentLayout, but user reports options don't render at top of page.

**Root Cause**: UNDER INVESTIGATION - Code structure appears correct.

**Current State**:
- ✅ **binary-orbit**: Visualization options render correctly at top
- ❌ **three-body**: User reports visualization options NOT visible
- ✅ **black-hole**: Visualization options render correctly at top
- ✅ **stellar-collapse**: Visualization options render correctly at top
- ✅ **big-bang**: Visualization options render correctly at top

**Code Review**:
- three-body DOES import StandardVisualizationOptions ✅
- three-body DOES pass visualizationOptions prop to ExperimentLayout ✅
- ExperimentLayout DOES render {visualizationOptions} in correct location ✅
- StandardVisualizationOptions component is structurally correct ✅
- useSimulationState initializes state.ui correctly ✅

**Hypotheses**:
1. React hydration issue (SSR mismatch)?
2. CSS z-index or positioning issue?
3. JavaScript error preventing render?
4. State initialization timing issue?

**Investigation Steps**:
1. ✅ Code review (complete - structure is correct)
2. ⏳ Browser DOM inspection (check if element exists but hidden)
3. ⏳ Browser console errors (check for JS errors)
4. ⏳ React DevTools component tree inspection
5. ⏳ Compare rendered HTML between working (binary-orbit) and broken (three-body)

**Fix Required**: Complete browser investigation to identify root cause.

---

### 3. Canvas Container Classes (PENDING AUDIT)

**Need to verify**: Are canvas container classes consistent?

| Experiment | Canvas Container Class | Expected |
|------------|------------------------|----------|
| binary-orbit | `panel h-[600px] relative overflow-hidden` | ? |
| three-body | `bg-space-panel rounded-lg overflow-hidden border border-space-border h-[600px]` | ? |
| black-hole | ? | ? |
| stellar-collapse | ? | ? |
| big-bang | ? | ? |

**Action**: Audit all 5 experiments for consistent canvas styling.

---

### 4. Button Styles (PENDING AUDIT)

**Need to verify**: Are Play/Pause/Reset button styles consistent?

**Action**: Check button classes across all 5 experiments.

---

### 5. Grid Layout Consistency (PENDING AUDIT)

**Need to verify**: Grid layouts match specification?

| Experiment | Expected Pattern | Actual | Compliant? |
|------------|------------------|--------|------------|
| binary-orbit | Pattern A (3-col) | ? | ? |
| three-body | Pattern B (4-col) | 4-col ✅ | ? |
| black-hole | Pattern A (3-col) | ? | ? |
| stellar-collapse | Pattern B (4-col) | ? | ? |
| big-bang | Pattern B (4-col) | ? | ? |

---

## Standardization Requirements

### A. Background Color Standard

**MANDATORY**: All canvas components MUST use `#0a0e27` background.

```tsx
// CORRECT (OrbitCanvas style)
<color attach="background" args={[0.039, 0.055, 0.152]} /> {/* #0a0e27 */}

// INCORRECT (NBodyCanvas current)
<color attach="background" args={['#0a0a1a']} />
```

**Files to Fix**:
- [ ] `src/components/visuals/NBodyCanvas.tsx`

---

### B. Canvas Container Classes Standard

**MANDATORY**: All canvas containers MUST use consistent classes.

**Proposed Standard** (to be confirmed):
```tsx
<div className="panel h-[600px] relative overflow-hidden">
  <Canvas>...</Canvas>
</div>
```

**OR** (alternative):
```tsx
<div className="bg-space-panel rounded-lg overflow-hidden border border-space-border h-[600px]">
  <Canvas>...</Canvas>
</div>
```

**Action Required**: Choose ONE standard, enforce across all experiments.

---

### C. Button Classes Standard

**MANDATORY**: Play/Pause/Reset buttons MUST use consistent classes.

**Audit Required**: Document current button classes in all 5 experiments, choose standard.

---

### D. Metrics Panel Title Colors

**Current State** (from previous audit):
- binary-orbit: `text-accent-chi` (default)
- three-body: `text-accent-chi` (default)
- black-hole: `text-purple-400`
- stellar-collapse: `text-purple-400`
- big-bang: `text-purple-400`

**Question**: Is this intentional? Should all use `text-accent-chi`?

---

## Validation Script Enhancements

### New Checks to Add

1. **Background Color Consistency**
   - Parse canvas component files
   - Check `<color attach="background"` values
   - Error if not `#0a0e27` or equivalent RGB

2. **Canvas Container Class Consistency**
   - Check all experiments use same canvas container classes
   - Error if mismatched

3. **Button Class Consistency**
   - Check Play/Pause/Reset button classes match
   - Warn if inconsistent

4. **Visual Regression Detection**
   - Capture screenshots of each experiment
   - Compare against baseline
   - Flag visual differences

---

## Browser Testing Checklist

### For Each Experiment (binary-orbit, three-body, black-hole, stellar-collapse, big-bang):

- [ ] Background color is `#0a0e27` (dark blue-grey)
- [ ] Visualization options render at TOP of page (above canvas)
- [ ] Visualization options are horizontally aligned
- [ ] All toggles work (Bodies, Trails, Stars, etc.)
- [ ] Canvas is same height (600px)
- [ ] Play/Pause/Reset buttons look identical
- [ ] Metrics panel title color is consistent (or intentionally different)
- [ ] Grid layout matches specification (3-col or 4-col)
- [ ] Page loads without console errors

---

## Next Steps

1. ✅ Document inconsistencies (this file)
2. ⏳ Fix background color in NBodyCanvas
3. ⏳ Debug three-body visualization options not rendering
4. ⏳ Audit canvas container classes
5. ⏳ Audit button classes
6. ⏳ Standardize all visual elements
7. ⏳ Update validation script with visual checks
8. ⏳ Update EXPERIMENT_PAGE_SPECIFICATION.md with visual standards
9. ⏳ Browser test all 5 experiments
10. ⏳ Document final standards

---

## Enforcement

Once standards are defined:

1. Update `EXPERIMENT_PAGE_SPECIFICATION.md` with visual standards
2. Update `scripts/validate-experiments.ts` with visual checks
3. Require validation to pass before commit
4. Include visual standards in AI agent instructions

---

**Status**: INCOMPLETE - Fixes in progress
**Blocker**: Need to complete audit before establishing final standards
