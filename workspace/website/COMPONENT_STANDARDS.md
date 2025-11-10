# Website Component Standards

**Last Updated:** November 8, 2025  
**Purpose:** Maintain consistency across all experiment pages and UI components

---

## Critical Rules - MUST Follow

### 1. Naming Consistency

**Experiment Names:**
- **Earth-Moon Orbit** (NOT "Binary Orbit") - Used everywhere:
  - Page titles
  - Navigation menu
  - Cards on home page
  - Scientific disclosure `experimentName` prop
  - URL remains `/experiments/binary-orbit` for stability

**Other Experiments:**
- **Three-Body Problem** (not "Three Body" or "3-Body")
- **Black Hole** (not "Blackhole" or "Black-Hole")
- **Stellar Collapse** (not "Star Collapse")
- **Big Bang** (not "BigBang" or "The Big Bang")

### 2. Required Components on ALL Experiment Pages

Every experiment page MUST include:

```tsx
import ScientificDisclosure from '@/components/ui/ScientificDisclosure';
import BackendBadge from '@/components/ui/BackendBadge';
import ParameterSlider from '@/components/ui/ParameterSlider';

// In JSX:
<ScientificDisclosure experimentName="Earth-Moon Orbit" />
<BackendBadge backend={backend} />
```

**ScientificDisclosure provides:**
- Yellow "⚠️ Read About This Project" button
- Single disclosure message (not duplicated)
- Link to About page with experiment name

**BackendBadge provides:**
- Shows GPU (WebGPU) or CPU status
- Performance information (lattice size, FPS)
- Consistent across ALL experiments

### 3. Parameter Sliders - Tooltips REQUIRED

**ALWAYS use the shared ParameterSlider component:**

```tsx
<ParameterSlider
  label="Mass Ratio"
  value={state.params.massRatio}
  min={1}
  max={200}
  step={1}
  unit="×"
  onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { massRatio: v } })}
  tooltip="Earth is 81.3× more massive than the Moon. Try different ratios to see Jupiter-moon systems!"
/>
```

**Tooltip Guidelines:**
- ✅ **DO:** Explain what the parameter does physically
- ✅ **DO:** Give context (real-world examples)
- ✅ **DO:** Warn about instability if relevant
- ❌ **DON'T:** Just repeat the label
- ❌ **DON'T:** Use technical jargon without explanation

**Examples of Good Tooltips:**

```tsx
// Good - explains physics + gives context
tooltip="Chi field coupling strength - determines how strongly the field gradient pulls objects together."

// Good - explains behavior + warns
tooltip="Controls how fast emergent spacetime evolves - larger timestep = faster orbit motion. Too large may become unstable."

// Bad - just repeats label
tooltip="The mass ratio parameter"

// Bad - too technical
tooltip="Gaussian sigma parameter for chi field kernel"
```

### 4. Navigation Consistency

**Header About Link:**
- Header automatically passes correct `from` parameter based on current page
- Uses `usePathname()` to detect current location
- No manual configuration needed

**About Page Back Button:**
- Automatically shows "Back to [Experiment Name]" using `from` parameter
- Uses `router.back()` to return to previous page

**Links to About:**
- ALWAYS include `from` parameter: `href="/about?from=Earth-Moon Orbit"`
- Use `encodeURIComponent()` if experiment name has spaces

### 5. Page Structure Template

**Every experiment page must follow this structure:**

```tsx
'use client';

import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import ScientificDisclosure from '@/components/ui/ScientificDisclosure';
import ParameterSlider from '@/components/ui/ParameterSlider';
import BackendBadge from '@/components/ui/BackendBadge';

export default function ExperimentPage() {
  return (
    <div className="min-h-screen bg-space-dark text-text-primary">
      <Header />
      <main className="container mx-auto px-4 pt-24 pb-12">
        <div className="mb-8">
          {/* Page Title */}
          <div className="mb-4">
            <h1 className="text-4xl font-bold text-accent-chi mb-2">
              Experiment Name
            </h1>
            <p className="text-text-secondary">
              Brief description
            </p>
          </div>
          
          {/* REQUIRED: Scientific Disclosure */}
          <ScientificDisclosure experimentName="Experiment Name" />
        </div>

        {/* REQUIRED: Backend Status */}
        <div className="mb-8">
          <BackendBadge backend={backend} />
        </div>

        {/* Visualization + Controls Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Visualization (3 cols) */}
          <div className="lg:col-span-3">
            {/* Canvas/WebGPU component */}
          </div>
          
          {/* Controls Panel (1 col) */}
          <div>
            <div className="panel">
              <h3 className="text-lg font-bold mb-4">Parameters</h3>
              
              {/* ALWAYS use ParameterSlider with tooltips */}
              <ParameterSlider
                label="Parameter Name"
                value={value}
                min={min}
                max={max}
                step={step}
                unit="unit"
                onChange={onChange}
                tooltip="Clear explanation of what this does"
              />
            </div>
          </div>
        </div>

        {/* Optional: Explanation Section */}
        <div className="mt-12 panel">
          <h2 className="text-2xl font-bold mb-4">How It Works</h2>
          {/* Detailed explanation */}
        </div>
      </main>
      <Footer />
    </div>
  );
}
```

---

## Component Library Reference

### UI Components (src/components/ui/)

| Component | Purpose | Required Props | Optional Props | Required On All Pages? |
|-----------|---------|----------------|----------------|----------------------|
| `ScientificDisclosure` | Warning banner + About button (SINGLE message, not duplicated) | `experimentName: string` | - | ✅ YES |
| `BackendBadge` | Shows GPU/CPU status, lattice size, FPS | `backend: 'webgpu' \| 'cpu'` | - | ✅ YES |
| `ParameterSlider` | Slider with tooltip | `label`, `value`, `min`, `max`, `step`, `unit`, `onChange` | `tooltip`, `onDragStart`, `onDragEnd` | Only if page has parameters |
| `VisualizationOptions` | Standardized toggle controls for visibility | `toggles: VisualizationToggle[]`, `onChange: (key, value) => void` | - | ✅ YES (sidebar) |

### Layout Components (src/components/layout/)

| Component | Purpose |
|-----------|---------|
| `Header` | Top navigation with logo, experiments dropdown, research links, About |
| `Footer` | Bottom navigation with license, contact, links |

---

## Testing Checklist

Before committing changes to ANY experiment page, verify:

- [ ] Page title uses standard experiment name (see §1)
- [ ] `<ScientificDisclosure experimentName="..." />` present (SINGLE instance, not duplicated)
- [ ] `<BackendBadge backend={backend} />` present (REQUIRED on all experiment pages)
- [ ] ALL parameter sliders use `<ParameterSlider />` component
- [ ] ALL sliders have meaningful tooltips explaining the parameter
- [ ] About links pass `from` parameter: `/about?from=Experiment Name`
- [ ] Page matches structure template (§5)
- [ ] Tested navigation: Header About → Back button returns to experiment
- [ ] Tested navigation: Yellow About button → Back button returns to experiment
- [ ] No duplicate disclosure messages (ScientificDisclosure shows ONE message)

---

## Why These Standards Matter

**Problem:** Without standards, the site becomes inconsistent:
- Some experiments have tooltips, others don't
- Some have disclosure warnings, others don't
- Navigation breaks (back button goes wrong place)
- Experiment names vary across pages
- Users get confused, site looks unprofessional

**Solution:** Enforce these standards rigorously.

**Enforcement:**
1. Use shared components (`ScientificDisclosure`, `ParameterSlider`, etc.)
2. Follow page structure template
3. Review checklist before committing
4. If you add a feature to one experiment, add it to ALL experiments

---

## Common Mistakes to Avoid

❌ **Don't create inline components** - Extract to `src/components/ui/`  
❌ **Don't duplicate disclosure messages** - ScientificDisclosure shows ONE message  
❌ **Don't hardcode experiment names** - Pass as props  
❌ **Don't forget tooltips** - EVERY slider needs explanation  
❌ **Don't vary styling** - Use shared components for consistency  
❌ **Don't mix naming** - "Binary Orbit" vs "Earth-Moon" is confusing  
❌ **Don't skip BackendBadge** - REQUIRED on ALL experiment pages  

✅ **Do use shared components**  
✅ **Do show BackendBadge on every experiment page**  
✅ **Do use single ScientificDisclosure (not duplicated)**  
✅ **Do add tooltips to everything interactive**  
✅ **Do pass experiment names as props**  
✅ **Do follow the page structure template**  
✅ **Do test navigation flows**  

---

## Future Standards (To Be Added)

- Metrics display component (energy, angular momentum, drift)
- Control button styling (Play/Pause/Reset)
- Preset system (dropdown for predefined configurations)
- Error boundary patterns
- Loading states
- Mobile responsive breakpoints

---

**Remember:** Consistency is not optional. Users notice when things look different across pages. Follow these standards religiously.
