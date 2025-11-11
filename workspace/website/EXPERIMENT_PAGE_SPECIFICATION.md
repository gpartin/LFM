# Experiment Page Specification (BINDING CONTRACT)

**Purpose**: Enforce 100% consistency across all showcase experiment pages.  
**Status**: MANDATORY for all showcase experiments (binary-orbit, three-body, black-hole, stellar-collapse, big-bang)  
**Last Updated**: 2025-11-11

---

## The Problem We're Solving

AI agents (and humans) create inconsistent experiment pages:
- Different import patterns
- Different component hierarchies  
- Different grid layouts
- Duplicate code instead of shared components
- Missing standardized visualization options
- Inconsistent metrics displays

**This document is the SINGLE SOURCE OF TRUTH.**

---

## Showcase Experiment Anatomy (REQUIRED STRUCTURE)

### 1. Required Imports (EXACT ORDER)

```typescript
'use client';

/* Copyright header */

import { useEffect, useRef, useCallback, useState } from 'react';
import ExperimentLayout from '@/components/experiment/ExperimentLayout';
import StandardVisualizationOptions from '@/components/experiment/StandardVisualizationOptions';
import StandardMetricsPanel from '@/components/experiment/StandardMetricsPanel';
import ParameterSlider from '@/components/ui/ParameterSlider';
import { detectBackend } from '@/physics/core/backend-detector';
import { useSimulationState } from '@/hooks/useSimulationState';
// Simulation-specific imports (OrbitCanvas, NBodyCanvas, etc.)
```

**Validation Rule**: Every showcase experiment MUST import StandardVisualizationOptions and StandardMetricsPanel.

---

### 2. Component Structure (MANDATORY HIERARCHY)

```tsx
export default function ExperimentPage() {
  // State management (useSimulationState hook)
  const [state, dispatch] = useSimulationState();
  
  // Backend detection
  const [backend, setBackend] = useState<'webgpu' | 'cpu'>('webgpu');
  
  // Refs for simulation
  const simRef = useRef<SimulationType | null>(null);
  const rafRef = useRef<number | null>(null);
  const isRunningRef = useRef<boolean>(false);
  
  // Effects: backend detection, simulation init, animation loop
  
  return (
    <ExperimentLayout
      title="..."
      description="..."
      backend={backend}
      experimentId="experiment-id"
      visualizationOptions={
        <StandardVisualizationOptions
          state={state.ui}
          onChange={(key, value) => dispatch({ type: 'UPDATE_UI', payload: { key: key as any, value } })}
          labelOverrides={{ /* optional */ }}
          showAdvancedOptions={true|false}
          additionalControls={/* optional */}
        />
      }
      footerContent={/* optional explanation section */}
    >
      {/* Main experiment grid - see below */}
    </ExperimentLayout>
  );
}
```

**Validation Rules**:
- ✅ MUST use `<ExperimentLayout>` wrapper
- ✅ MUST pass `visualizationOptions` prop with `<StandardVisualizationOptions>`
- ✅ MUST use `useSimulationState` hook
- ❌ NEVER inline visualization checkboxes manually
- ❌ NEVER create duplicate MetricDisplay components

---

### 3. Grid Layout Patterns

#### Pattern A: 2-Column Layout (binary-orbit, black-hole)
```tsx
<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
  {/* Canvas: 2/3 width */}
  <div className="lg:col-span-2">
    <div className="panel h-[600px] relative overflow-hidden">
      <WebGPUErrorBoundary>
        <OrbitCanvas {...props} />
      </WebGPUErrorBoundary>
    </div>
    
    {/* Play/Pause/Reset buttons */}
    <div className="mt-4 flex gap-4">
      <button>Play/Pause</button>
      <button>Reset</button>
    </div>
  </div>

  {/* Controls + Metrics: 1/3 width */}
  <div className="lg:col-span-1 space-y-6">
    {/* Parameters panel */}
    <div className="panel">
      <h3>Parameters</h3>
      {/* Sliders */}
    </div>

    {/* Metrics panel */}
    <StandardMetricsPanel
      coreMetrics={{ energy, drift, angularMomentum }}
      additionalMetrics={[...]} /* optional */
      title="System Metrics"
      titleColorClass="text-accent-chi"
    />
  </div>
</div>
```

#### Pattern B: 4-Column Layout (three-body, stellar-collapse, big-bang)
```tsx
<div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
  {/* Canvas: 3/4 width */}
  <div className="lg:col-span-3">
    <div className="panel h-[600px]">
      <NBodyCanvas {...props} />
    </div>
    
    {/* Play/Pause/Reset buttons */}
  </div>

  {/* Controls + Metrics: 1/4 width */}
  <div className="lg:col-span-1 space-y-6">
    {/* Parameters */}
    {/* Metrics */}
  </div>
</div>
```

**Validation Rule**: Choose pattern based on experiment complexity:
- **Pattern A** (3-col): Complex parameter controls (5+ sliders)
- **Pattern B** (4-col): Simple controls (1-3 sliders) or preset-only

---

### 4. StandardVisualizationOptions Usage

#### Basic Usage (3 options only)
```tsx
<StandardVisualizationOptions
  state={state.ui}
  onChange={(key, value) => dispatch({ type: 'UPDATE_UI', payload: { key: key as any, value } })}
  showAdvancedOptions={false} // Only Bodies, Trails, Stars
/>
```

#### Full Usage (9 options)
```tsx
<StandardVisualizationOptions
  state={state.ui}
  onChange={(key, value) => dispatch({ type: 'UPDATE_UI', payload: { key: key as any, value } })}
  showAdvancedOptions={true} // All visualization options
  labelOverrides={{
    showParticles: 'Earth & Moon',
    showTrails: 'Orbital Paths',
    showIsoShells: 'Event Horizon Shell',
  }}
  additionalControls={
    <div>Custom toggle (e.g., Force CPU)</div>
  }
/>
```

**Validation Rules**:
- ✅ ALWAYS use `state.ui` from `useSimulationState`
- ✅ ALWAYS use dispatch for onChange
- ❌ NEVER hardcode visualization checkboxes
- ❌ NEVER create custom VisualizationOptions component per experiment

---

### 5. StandardMetricsPanel Usage

#### Core Metrics Only
```tsx
<StandardMetricsPanel
  coreMetrics={{
    energy: state.metrics.energy,
    drift: state.metrics.drift,
    angularMomentum: state.metrics.angularMomentum,
  }}
/>
```

#### With Additional Metrics
```tsx
<StandardMetricsPanel
  coreMetrics={{
    energy: state.metrics.energy,
    drift: state.metrics.drift,
    angularMomentum: state.metrics.angularMomentum,
  }}
  additionalMetrics={[
    { label: 'Separation', value: '384,400 km', status: 'neutral' },
    { label: 'Speed / v_circ', value: '1.02', status: 'good' },
    { label: 'FPS', value: '60', status: 'good' },
  ]}
  title="System Stats"
  titleColorClass="text-accent-chi"
/>
```

**Validation Rules**:
- ✅ ALWAYS include core 3 metrics (energy, drift, angularMomentum)
- ✅ Use `additionalMetrics` array for experiment-specific metrics
- ❌ NEVER create inline MetricDisplay components
- ❌ NEVER duplicate the MetricDisplay JSX

---

## Current Experiment Status (Audit as of 2025-11-11)

| Experiment | StandardVizOptions | StandardMetricsPanel | Grid Layout | Status |
|------------|-------------------|---------------------|-------------|---------|
| binary-orbit | ✅ YES | ✅ YES | 3-col (Pattern A) | ✅ COMPLIANT |
| three-body | ⚠️ YES (but wrong layout) | ✅ YES | 4-col (Pattern B) | ⚠️ NEEDS FIX |
| black-hole | ✅ YES | ✅ YES | 3-col (Pattern A) | ✅ COMPLIANT |
| stellar-collapse | ✅ YES | ✅ YES | 4-col (Pattern B) | ✅ COMPLIANT |
| big-bang | ✅ YES | ✅ YES | 4-col (Pattern B) | ✅ COMPLIANT |

**Issue**: three-body has visualization options passed to ExperimentLayout but layout doesn't match other experiments.

---

## Validation Checklist (Run Before Commit)

Before committing changes to ANY showcase experiment:

### Manual Checklist
- [ ] Imports StandardVisualizationOptions from shared component
- [ ] Imports StandardMetricsPanel from shared component
- [ ] Uses ExperimentLayout wrapper
- [ ] Passes visualizationOptions prop to ExperimentLayout
- [ ] Does NOT contain inline VisualizationCheckbox components
- [ ] Does NOT contain duplicate MetricDisplay JSX
- [ ] Uses correct grid layout pattern (3-col or 4-col)
- [ ] Canvas is on left, controls/metrics on right
- [ ] Play/Pause/Reset buttons below canvas
- [ ] Uses useSimulationState hook
- [ ] Backend detection follows standard pattern

### Automated Validation (TODO: Script)
```bash
npm run validate:experiments
```

This should check:
1. Required imports present
2. No duplicate components
3. ExperimentLayout structure correct
4. Grid layout matches specification

---

## Enforcement Strategy

1. **This document is BINDING** - No exceptions without explicit approval
2. **Pre-commit validation** - Must pass before merge
3. **AI agent instructions** - Reference this doc in copilot-instructions.md
4. **Code review** - Check against this spec
5. **Refactoring** - Update this doc first, then code

---

## Evolution Process

To change this specification:
1. Update this document FIRST
2. Get approval for spec change
3. Update all affected experiments
4. Run validation
5. Update copilot-instructions.md if needed

**Never change code structure without updating this spec first.**

---

---

## Visual Consistency Standards (MANDATORY)

### Canvas Background Color

**REQUIRED**: All canvas components MUST use `#0a0e27` background color.

```tsx
// CORRECT - Use RGB values for consistency
<color attach="background" args={[0.039, 0.055, 0.152]} /> {/* #0a0e27 */}
```

**Validation**: Automated check in `scripts/validate-experiments.ts`

**Files**:
- ✅ `src/components/visuals/OrbitCanvas.tsx` - Compliant
- ✅ `src/components/visuals/NBodyCanvas.tsx` - Fixed 2025-11-11

---

### Canvas Container Classes

**ALLOWED PATTERNS** (both are acceptable):

**Pattern A** (with panel class):
```tsx
<div className="panel h-[600px] relative overflow-hidden">
```

**Pattern B** (explicit bg classes):
```tsx
<div className="bg-space-panel rounded-lg overflow-hidden border border-space-border h-[600px]">
```

**Validation**: Warning if neither pattern detected

---

## Next Steps

1. ✅ Create this specification document
2. ✅ Create validation script (validate-experiments.ts)
3. ✅ Add visual consistency checks (background color)
4. ⏳ Fix three-body visualization options rendering issue
5. ⏳ Add pre-commit hook for validation
6. ⏳ Update copilot-instructions.md to reference this doc
