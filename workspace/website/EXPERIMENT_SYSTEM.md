# LFM Website - Experiment System Architecture

## Overview

The LFM website uses an extensible, type-safe experiment system that makes it easy to add new physics simulations while maintaining consistent UI/UX, accessibility, and performance.

## Key Features

✅ **Extensible** - Add new experiments by creating a single file and registering it  
✅ **Type-Safe** - Full TypeScript interfaces for experiment metadata, config, and behavior  
✅ **Searchable** - Built-in search and filter UI for experiment discovery  
✅ **GPU-Accelerated** - WebGPU-first design for maximum performance  
✅ **Accessible** - ARIA labels, keyboard navigation, screen reader support  
✅ **Tested** - Unit tests for all components and registry functions  

---

## Architecture

### 1. Experiment Type System

**File**: `src/types/experiment.ts`

Defines the standard interface all experiments must implement:

```typescript
interface Experiment {
  metadata: ExperimentMetadata;  // Title, description, tags, etc.
  config: ExperimentConfig;      // Parameters, presets
  initialize: () => Promise<void>;
  cleanup: () => Promise<void>;
  reset: () => Promise<void>;
  start: () => void;
  pause: () => void;
  updateParameters: (params: Record<string, any>) => void;
  getMetrics: () => ExperimentMetrics[];
  getResults: () => ExperimentResults;
  exportResults: (format?: 'json' | 'csv') => Promise<string>;
  RenderComponent: React.ComponentType<...>;
}
```

### 2. Experiment Registry

**File**: `src/lib/experimentRegistry.ts`

Central registry for all experiments. Provides:

- `getAllExperiments()` - Get all registered experiments
- `getExperimentById(id)` - Get specific experiment
- `searchExperiments(query)` - Full-text search
- `filterExperiments(filters)` - Filter by category, difficulty, tags
- `getFeaturedExperiments()` - Get featured experiments

**Lazy Loading**: Experiments are loaded on-demand, not at startup.

### 3. Experiment Implementations

**Location**: `src/experiments/{category}/{experiment-id}.tsx`

Each experiment is a self-contained module that exports a factory function:

```typescript
const createMyExperiment: ExperimentFactory = (device, initialConfig) => {
  return new MyExperiment(device, initialConfig);
};

export default createMyExperiment;
```

### 4. Experiment Pages

**Location**: `src/app/experiments/{experiment-id}/page.tsx`

Next.js pages for each experiment. These pages:

- Detect GPU backend (WebGPU/WebGL/CPU)
- Initialize simulation
- Render UI controls and visualization
- Handle state management

---

## Adding a New Experiment

### Step 1: Create Experiment Implementation

Create `src/experiments/{category}/{experiment-id}.tsx`:

```typescript
import type { Experiment, ExperimentFactory } from '@/types/experiment';

class MyExperiment implements Experiment {
  metadata = { /* ... */ };
  config = { /* ... */ };
  
  async initialize() { /* Setup GPU resources */ }
  async cleanup() { /* Clean up resources */ }
  // ... implement other methods
}

const createMyExperiment: ExperimentFactory = (device, config) => {
  return new MyExperiment(device, config);
};

export default createMyExperiment;
```

### Step 2: Register in Registry

Add to `src/lib/experimentRegistry.ts`:

```typescript
{
  metadata: {
    id: 'my-experiment',
    title: 'My Experiment',
    shortDescription: 'Brief description',
    category: 'gravity',  // or other category
    tags: ['tag1', 'tag2'],
    difficulty: 'intermediate',
    // ... more metadata
  },
  loader: () => import('@/experiments/gravity/my-experiment'),
}
```

### Step 3: Create Next.js Page

Create `src/app/experiments/my-experiment/page.tsx`:

```typescript
'use client';

export default function MyExperimentPage() {
  // Use existing patterns from binary-orbit or black-hole pages
  // Or create a shared experiment page component
}
```

### Step 4: Test

Run tests:

```bash
npm test
```

Your experiment is now discoverable in:
- `/experiments/browse` (search/filter UI)
- Direct link: `/experiments/my-experiment`

---

## Experiment Metadata Schema

### Required Fields

```typescript
{
  id: string;              // Unique, kebab-case (used in URLs)
  title: string;           // Display title
  shortDescription: string; // 1-2 sentences for cards
  fullDescription: string;  // Shown on experiment page
  category: ExperimentCategory;
  tags: string[];          // For search/filtering
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'research';
  version: string;         // Semantic versioning (e.g., "1.0.0")
  created: string;         // ISO 8601 date
  updated: string;         // ISO 8601 date
  backend: {
    minBackend: 'webgpu' | 'webgl' | 'cpu';
    requiredFeatures?: string[];
    estimatedVRAM?: number;  // MB
  };
  education: {
    whatYouSee: string;
    principles: string[];
    realWorld?: string;
    references?: Array<{
      title: string;
      url: string;
      type: 'paper' | 'documentation' | 'evidence' | 'external';
    }>;
  };
}
```

### Optional Fields

- `featured?: boolean` - Show on home page
- `thumbnail?: string` - Preview image path
- `estimatedRuntime?: number` - Seconds
- `authors?: string[]` - Creator names

---

## Parameter System

Experiments define tunable parameters:

```typescript
config: {
  parameters: [
    {
      key: 'gravity',
      label: 'Gravity Strength',
      description: 'How strong the gravitational pull is',
      type: 'number',
      defaultValue: 0.25,
      min: 0.05,
      max: 0.5,
      step: 0.01,
      unit: '',
      liveUpdate: true,  // Can change while running?
    },
    // ... more parameters
  ],
  presets: [
    {
      id: 'earth-moon',
      label: 'Earth-Moon',
      description: 'Default Earth-Moon configuration',
      values: { gravity: 0.25, distance: 3.0 },
    },
  ],
}
```

---

## Performance Optimization

### GPU Acceleration

All experiments use WebGPU for:
- Physics computation (compute shaders)
- Rendering (vertex/fragment shaders)
- Data transfer (minimal CPU ↔ GPU communication)

### Lazy Loading

Experiments are loaded on-demand:

```typescript
loader: () => import('@/experiments/gravity/black-hole')
```

This prevents loading all experiments on startup.

### Code Splitting

Next.js automatically splits experiment code into separate chunks.

---

## Testing

### Unit Tests

**Location**: `src/__tests__/`

Tests for:
- Experiment registry (`experimentRegistry.test.ts`)
- UI components (`components.test.tsx`)
- Physics modules (`physics.test.ts`)

Run tests:

```bash
npm test              # Run once
npm run test:watch    # Watch mode
npm run test:coverage # Coverage report
```

### Test Coverage Goals

- Registry functions: 100%
- UI components: >90%
- Physics modules: >80%

---

## Accessibility

All experiments include:

✅ **ARIA Labels** - `aria-label`, `aria-checked`, `aria-pressed`  
✅ **Keyboard Navigation** - Tab, Enter, Space, Arrow keys  
✅ **Screen Reader Support** - Descriptive labels and roles  
✅ **Color Contrast** - WCAG AA compliant  
✅ **Focus Indicators** - Visible focus states  

Example:

```tsx
<input
  type="checkbox"
  checked={checked}
  onChange={onChange}
  aria-label="Show gravity field"
  aria-checked={checked}
  role="switch"
/>
```

---

## File Structure

```
src/
├── types/
│   └── experiment.ts          # Type definitions
├── lib/
│   └── experimentRegistry.ts  # Central registry
├── experiments/
│   ├── orbital-mechanics/
│   │   └── binary-orbit.tsx   # Binary orbit experiment
│   └── gravity/
│       └── black-hole.tsx     # Black hole experiment
├── app/
│   └── experiments/
│       ├── browse/
│       │   └── page.tsx       # Browse UI
│       ├── binary-orbit/
│       │   └── page.tsx       # Binary orbit page
│       └── black-hole/
│           └── page.tsx       # Black hole page
├── __tests__/
│   ├── experimentRegistry.test.ts
│   ├── components.test.tsx
│   └── physics.test.ts
└── components/
    ├── visuals/
    │   └── OrbitCanvas.tsx    # 3D visualization
    └── ui/
        ├── ParameterSlider.tsx
        ├── MetricDisplay.tsx
        └── ViewToggle.tsx
```

---

## Current Experiments

### 1. Binary Orbit (Earth-Moon)

**ID**: `binary-orbit`  
**Category**: Orbital Mechanics  
**Difficulty**: Beginner  
**Features**:
- Earth and Moon orbiting due to emergent gravity
- Energy conservation visualization
- Adjustable mass ratios and orbital parameters

### 2. Black Hole Orbit

**ID**: `black-hole`  
**Category**: Gravity  
**Difficulty**: Intermediate  
**Features**:
- Extreme gravity simulation
- Event horizon visualization
- Orbital precession and time dilation effects

---

## Future Enhancements

### Planned Features

- [ ] Shared experiment page component (reduce boilerplate)
- [ ] Export results to CSV/JSON
- [ ] Share experiment links with parameters
- [ ] Experiment comparison view
- [ ] Performance profiler overlay
- [ ] Mobile-optimized controls
- [ ] VR/AR visualization support

### Experiment Ideas

- [ ] Wave interference patterns
- [ ] Electromagnetic field interactions
- [ ] Quantum tunneling
- [ ] Thermodynamic equilibrium
- [ ] Relativistic particle collision

---

## Best Practices

### 1. Keep Experiments Self-Contained

Each experiment should be independent - no shared state between experiments.

### 2. Use TypeScript Strictly

Enable strict mode and use explicit types everywhere.

### 3. Optimize for GPU

Minimize CPU ↔ GPU data transfer. Use compute shaders for physics.

### 4. Test Everything

Write tests for new components and registry entries.

### 5. Document Physics

Include educational content explaining what users are seeing.

### 6. Accessibility First

Add ARIA labels, keyboard support, and tooltips.

---

## Contributing

### Adding a New Experiment

1. Fork the repository
2. Create experiment implementation
3. Register in registry
4. Create Next.js page
5. Write tests
6. Submit pull request

### Code Style

- Use UTF-8 encoding (`/* -*- coding: utf-8 -*- */`)
- Follow existing naming conventions
- Add JSDoc comments for public APIs
- Run linter before committing

---

## Resources

- [LFM Project Documentation](../docs/)
- [WebGPU Specification](https://gpuweb.github.io/gpuweb/)
- [Next.js Documentation](https://nextjs.org/docs)
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber/)
- [Accessibility Guidelines (WCAG)](https://www.w3.org/WAI/WCAG21/quickref/)

---

## License

See [LICENSE](../LICENSE) file in repository root.

---

## Questions?

For technical questions or contributions, see:
- GitHub Issues: [github.com/gpartin/LFM/issues]
- Documentation: `workspace/docs/`
- Main README: `workspace/README.md`
