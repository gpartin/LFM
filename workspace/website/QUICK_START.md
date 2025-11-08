# Quick Start: Adding a New Experiment

This guide shows you how to add a new physics experiment to the LFM website in under 2 hours.

---

## Step 1: Plan Your Experiment (15 minutes)

### Define Core Concept
- What physics phenomenon are you demonstrating?
- What parameters should users control?
- What metrics should be displayed?

### Choose Category
- `orbital-mechanics` - Orbits, trajectories, gravitational systems
- `gravity` - Gravitational effects, spacetime curvature
- `electromagnetic` - EM fields, Maxwell equations
- `quantization` - Quantum effects, bound states
- `relativistic` - Lorentz transformations, time dilation
- `energy-conservation` - Energy/momentum conservation
- `thermodynamics` - Heat, entropy, equilibrium
- `coupling` - Field interactions
- `advanced` - Multi-domain or complex experiments

### Difficulty Level
- `beginner` - Easy to understand, minimal prerequisites
- `intermediate` - Some physics background helpful
- `advanced` - Requires deeper physics knowledge
- `research` - Cutting-edge, experimental

---

## Step 2: Create Experiment Implementation (45 minutes)

### File Location
Create `src/experiments/{category}/{experiment-id}.tsx`

Example: `src/experiments/electromagnetic/wave-interference.tsx`

### Template

```typescript
/* -*- coding: utf-8 -*- */
/**
 * {Your Experiment Name}
 * 
 * {Brief description of what this simulates}
 */

import type { 
  Experiment, 
  ExperimentFactory,
  ExperimentMetadata,
  ExperimentConfig,
  ExperimentMetrics,
  ExperimentResults,
  ExperimentParameter,
} from '@/types/experiment';
import { YourPhysicsSimulation } from '@/physics/...';
import dynamic from 'next/dynamic';

// Lazy load visualization (reuse OrbitCanvas or create custom)
const YourCanvas = dynamic(() => import('@/components/visuals/YourCanvas'), { ssr: false });

/**
 * Parameter definitions
 */
const PARAMETERS: ExperimentParameter[] = [
  {
    key: 'parameter1',
    label: 'Parameter 1 Label',
    description: 'What this parameter controls',
    type: 'number',
    defaultValue: 1.0,
    min: 0.1,
    max: 10.0,
    step: 0.1,
    unit: 'units',
    liveUpdate: true,  // Can change while running?
  },
  // Add more parameters...
];

/**
 * Your Experiment Implementation
 */
class YourExperiment implements Experiment {
  private device: GPUDevice;
  private simulation: YourPhysicsSimulation | null = null;
  private isRunning: boolean = false;
  private parameters: Record<string, any>;
  
  metadata: ExperimentMetadata;
  config: ExperimentConfig;
  RenderComponent: React.ComponentType<any>;
  
  constructor(device: GPUDevice, initialParams?: Partial<Record<string, any>>) {
    this.device = device;
    
    // Initialize parameters with defaults
    this.parameters = {};
    PARAMETERS.forEach(param => {
      this.parameters[param.key] = initialParams?.[param.key] ?? param.defaultValue;
    });
    
    // Metadata
    this.metadata = {
      id: 'your-experiment-id',
      title: 'Your Experiment Title',
      shortDescription: 'One-sentence description for cards',
      fullDescription: 'Longer description for experiment page',
      category: 'electromagnetic',  // Choose appropriate category
      tags: ['tag1', 'tag2', 'webgpu'],
      difficulty: 'intermediate',
      version: '1.0.0',
      created: new Date().toISOString(),
      updated: new Date().toISOString(),
      featured: false,  // Set to true for home page
      backend: {
        minBackend: 'webgpu',
        requiredFeatures: ['compute'],
        estimatedVRAM: 256,
      },
      education: {
        whatYouSee: 'What users will observe in this simulation',
        principles: [
          'Physics principle 1',
          'Physics principle 2',
        ],
        realWorld: 'Real-world relevance (optional)',
        references: [
          {
            title: 'Related Research',
            url: '/docs/research.pdf',
            type: 'documentation',
          },
        ],
      },
      thumbnail: '/thumbnails/your-experiment.png',
      estimatedRuntime: 60,
    };
    
    // Configuration
    this.config = {
      parameters: PARAMETERS,
      presets: [
        {
          id: 'default',
          label: 'Default',
          description: 'Standard configuration',
          values: {
            parameter1: 1.0,
            // ... other parameter values
          },
        },
      ],
      defaultViews: {
        showParticles: true,
        showField: true,
        // ... other view options
      },
    };
    
    // Render component
    this.RenderComponent = YourCanvas as any;
  }
  
  async initialize(): Promise<void> {
    // Set up GPU resources, create simulation
    this.simulation = new YourPhysicsSimulation(this.device, this.parameters);
    await this.simulation.initialize();
  }
  
  async cleanup(): Promise<void> {
    this.pause();
    if (this.simulation) {
      this.simulation.destroy();
      this.simulation = null;
    }
  }
  
  async reset(): Promise<void> {
    if (this.simulation) {
      this.simulation.reset();
      await this.simulation.initialize();
    }
  }
  
  start(): void {
    this.isRunning = true;
  }
  
  pause(): void {
    this.isRunning = false;
  }
  
  async step(frames: number = 1): Promise<void> {
    if (this.simulation) {
      await this.simulation.stepBatch(frames);
    }
  }
  
  updateParameters(params: Record<string, any>): void {
    Object.assign(this.parameters, params);
    if (this.simulation) {
      this.simulation.updateParameters(this.parameters);
    }
  }
  
  getMetrics(): ExperimentMetrics[] {
    if (!this.simulation) return [];
    
    const state = this.simulation.getState();
    
    return [
      {
        label: 'Metric 1',
        value: state.value1.toFixed(2),
        unit: 'units',
        status: 'good',
        tooltip: 'What this metric means',
      },
      // Add more metrics...
    ];
  }
  
  getResults(): ExperimentResults {
    return {
      timestamp: new Date().toISOString(),
      parameters: this.parameters,
      metrics: {
        // Final metric values
      },
      notes: 'Experiment results',
    };
  }
  
  async exportResults(format: 'json' | 'csv' = 'json'): Promise<string> {
    const results = this.getResults();
    
    if (format === 'json') {
      return JSON.stringify(results, null, 2);
    } else {
      // CSV format
      const headers = ['parameter', 'value'];
      const rows = Object.entries(results.parameters).map(([k, v]) => `${k},${v}`);
      return [headers.join(','), ...rows].join('\n');
    }
  }
}

/**
 * Factory function
 */
const createYourExperiment: ExperimentFactory = (device, initialConfig) => {
  return new YourExperiment(device, initialConfig);
};

export default createYourExperiment;
```

---

## Step 3: Register in Registry (5 minutes)

Edit `src/lib/experimentRegistry.ts`:

```typescript
export const EXPERIMENTS: ExperimentRegistryEntry[] = [
  // ... existing experiments
  
  // Add your new experiment
  {
    metadata: {
      id: 'your-experiment-id',
      title: 'Your Experiment Title',
      shortDescription: 'Brief description',
      fullDescription: 'Full description',
      category: 'electromagnetic',
      tags: ['tag1', 'tag2', 'webgpu'],
      difficulty: 'intermediate',
      version: '1.0.0',
      created: '2025-11-07T00:00:00Z',
      updated: '2025-11-07T00:00:00Z',
      featured: false,
      backend: {
        minBackend: 'webgpu',
        requiredFeatures: ['compute'],
        estimatedVRAM: 256,
      },
      education: {
        whatYouSee: 'What users see',
        principles: ['Principle 1', 'Principle 2'],
      },
      thumbnail: '/thumbnails/your-experiment.png',
      estimatedRuntime: 60,
    },
    loader: () => import('@/experiments/electromagnetic/your-experiment'),
  },
];
```

---

## Step 4: Create Next.js Page (30 minutes)

Create `src/app/experiments/your-experiment-id/page.tsx`:

You can **copy and adapt** from existing experiments:
- `src/app/experiments/binary-orbit/page.tsx` (orbital systems)
- `src/app/experiments/black-hole/page.tsx` (gravity experiments)

### Key Changes to Make:

1. **Update titles and descriptions**
2. **Adjust parameter sliders** (min, max, step, labels)
3. **Customize metrics** (what to display in side panel)
4. **Update educational content** (explanation panel at bottom)
5. **Change color scheme** (optional, for visual distinction)

### Example Pattern:

```tsx
'use client';

import { useEffect, useRef, useCallback } from 'react';
import { useSimulationState } from '@/hooks/useSimulationState';
// ... other imports

export default function YourExperimentPage() {
  const [state, dispatch] = useSimulationState();
  // ... simulation setup (copy from existing page)
  
  return (
    <div className="min-h-screen flex flex-col bg-space-dark">
      <Header />
      
      <main className="flex-1 pt-20">
        <div className="container mx-auto px-4 py-8">
          {/* Page Header */}
          <h1>Your Experiment Title</h1>
          
          {/* 3D Canvas */}
          <YourCanvas simulation={simRef} {...viewOptions} />
          
          {/* Control Panel */}
          <ParameterSliders />
          
          {/* Metrics */}
          <MetricsPanel />
          
          {/* Explanation */}
          <ExplanationPanel />
        </div>
      </main>
      
      <Footer />
    </div>
  );
}
```

---

## Step 5: Write Tests (20 minutes)

Add test to `src/__tests__/experimentRegistry.test.ts`:

```typescript
describe('Your Experiment', () => {
  it('should be registered', () => {
    const exp = getExperimentById('your-experiment-id');
    expect(exp).toBeDefined();
    expect(exp?.metadata.title).toBe('Your Experiment Title');
  });
  
  it('should be in correct category', () => {
    const exps = getExperimentsByCategory('electromagnetic');
    const yourExp = exps.find(e => e.metadata.id === 'your-experiment-id');
    expect(yourExp).toBeDefined();
  });
  
  it('should be searchable', () => {
    const results = searchExperiments('your experiment');
    expect(results.length).toBeGreaterThan(0);
  });
});
```

Run tests:

```bash
npm test
```

---

## Step 6: Test in Browser (10 minutes)

```bash
# Start dev server
npm run dev

# Navigate to:
# http://localhost:3000/experiments/your-experiment-id

# Test:
# - All parameters work
# - Simulation runs smoothly
# - Metrics update correctly
# - No console errors
```

---

## Step 7: Document (10 minutes)

Update `EXPERIMENT_SYSTEM.md` with your new experiment:

```markdown
### 3. Your Experiment Name

**ID**: `your-experiment-id`  
**Category**: Electromagnetic  
**Difficulty**: Intermediate  
**Features**: Feature 1, Feature 2, Feature 3  
**Status**: ✅ Production Ready
```

---

## Checklist

Before submitting:

- [ ] Experiment implements `Experiment` interface
- [ ] Registered in `experimentRegistry.ts`
- [ ] Next.js page created
- [ ] Tests written and passing
- [ ] Accessible (ARIA labels, keyboard nav)
- [ ] No TypeScript errors
- [ ] No linting errors
- [ ] Tested in browser
- [ ] Documentation updated

---

## Tips & Best Practices

### Performance
- Use GPU for all heavy computation
- Batch physics steps (100+ per chunk)
- Minimize CPU ↔ GPU data transfer
- Profile with DevTools Performance tab

### Accessibility
- Add `aria-label` to all controls
- Use semantic HTML (`<button>`, not `<div onClick>`)
- Test with keyboard only (no mouse)
- Check color contrast (WCAG AA)

### Code Quality
- Use TypeScript strict mode
- Add JSDoc comments to public methods
- Keep functions small and focused
- Follow existing naming conventions

### Physics
- Document your equations (LaTeX in comments)
- Explain physical assumptions
- Include references to papers/docs
- Validate against known results

---

## Getting Help

### Documentation
- `EXPERIMENT_SYSTEM.md` - Complete architecture guide
- `MODERNIZATION_SUMMARY.md` - Project overview
- Type definitions in `src/types/experiment.ts`

### Examples
- `src/experiments/orbital-mechanics/binary-orbit.tsx`
- `src/experiments/gravity/black-hole.tsx`

### Community
- GitHub Issues: Report bugs or ask questions
- Pull Requests: Submit your experiment
- Discussions: Share ideas and get feedback

---

## Example: Wave Interference (5-minute version)

**Fastest path to a working experiment:**

1. **Copy** `src/experiments/gravity/black-hole.tsx` → `src/experiments/electromagnetic/wave-interference.tsx`
2. **Find/Replace**: "Black Hole" → "Wave Interference", "black-hole" → "wave-interference"
3. **Update** parameters (frequency, amplitude, wavelength instead of mass, distance)
4. **Copy** `src/app/experiments/black-hole/page.tsx` → `src/app/experiments/wave-interference/page.tsx`
5. **Find/Replace**: Same as step 2
6. **Register** in `experimentRegistry.ts`
7. **Run**: `npm run dev`

You now have a working experiment scaffold. Refine physics and UI from there.

---

**Estimated Total Time**: ~2 hours (for a complete, tested, documented experiment)

**Fastest Path**: ~30 minutes (copy/paste/adapt existing experiment)
