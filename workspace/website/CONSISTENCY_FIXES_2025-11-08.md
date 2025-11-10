# Website Consistency Fixes - November 8, 2025

## Issues Identified

1. **❌ Inconsistent naming:** "Binary Orbit" vs "Earth-Moon Orbit" used interchangeably
2. **❌ Tooltips only on one experiment:** Binary orbit had tooltips, others didn't
3. **❌ Duplicate component code:** ParameterSlider defined inline in multiple places
4. **❌ No standards document:** No reference for developers to follow

## Fixes Implemented

### 1. Standardized Naming (✅ COMPLETE)

**Rule:** Always use "Earth-Moon Orbit" (not "Binary Orbit")

**Changed:**
- `src/components/layout/Header.tsx` - Navigation menu
- `src/app/page.tsx` - Home page card
- URL remains `/experiments/binary-orbit` for backward compatibility

### 2. Created Reusable Components (✅ COMPLETE)

**New Component:** `src/components/ui/ParameterSlider.tsx`
- Includes tooltip support with info icon
- Adaptive decimal formatting
- Consistent styling across all experiments
- Proper accessibility (aria-label, title attributes)

**New Component:** `src/components/ui/ScientificDisclosure.tsx`
- Reusable warning banner
- Yellow "Read About This Project" button
- Automatically passes experiment name to About page

### 3. Updated Existing Pages (✅ COMPLETE)

**Earth-Moon Orbit** (`binary-orbit/page.tsx`):
- ✅ Removed inline ParameterSlider definition
- ✅ Imports shared component
- ✅ Has tooltips on all 5 parameters
- ✅ Uses ScientificDisclosure component

**Black Hole** (`black-hole/page.tsx`):
- ✅ Removed inline ParameterSlider definition
- ✅ Imports shared component
- ✅ Already had tooltips (kept them)
- ✅ Uses ScientificDisclosure component

**Three-Body Problem** (`three-body/page.tsx`):
- ✅ Uses ScientificDisclosure component
- ⚠️ No parameters yet (uses experiment registry)

**Stellar Collapse** (`stellar-collapse/page.tsx`):
- ✅ Uses ScientificDisclosure component
- ⚠️ No parameters yet (uses experiment registry)

**Big Bang** (`big-bang/page.tsx`):
- ✅ Uses ScientificDisclosure component
- ⚠️ No parameters yet (uses experiment registry)

### 4. Created Standards Document (✅ COMPLETE)

**New File:** `COMPONENT_STANDARDS.md`

**Contents:**
- Naming conventions (§1)
- Required components (§2)
- Parameter slider tooltip guidelines (§3)
- Navigation consistency rules (§4)
- Page structure template (§5)
- Component library reference
- Testing checklist
- Common mistakes to avoid

## Current Status

### Fully Compliant Pages
- ✅ **Earth-Moon Orbit** - All standards met
- ✅ **Black Hole** - All standards met

### Partially Compliant Pages
- ⚠️ **Three-Body Problem** - Has disclosure, needs parameters when implemented
- ⚠️ **Stellar Collapse** - Has disclosure, needs parameters when implemented
- ⚠️ **Big Bang** - Has disclosure, needs parameters when implemented

### Future Work

When implementing parameters for Three-Body, Stellar Collapse, or Big Bang:

1. Import `ParameterSlider` from `@/components/ui/ParameterSlider`
2. Add meaningful tooltips for EVERY parameter
3. Follow the page structure template in `COMPONENT_STANDARDS.md`
4. Test navigation flows (About → Back button)

## Tooltip Examples Added

**Good tooltips explain physics + give context:**

```tsx
"Earth is 81.3× more massive than the Moon. Try different ratios to see Jupiter-moon systems!"

"Initial separation between Earth and Moon. Larger distances = slower, wider orbits."

"Chi field coupling strength - determines how strongly the field gradient pulls objects together."

"Gaussian width of chi field - how far the gravity 'reaches'. Larger σ = longer-range force."

"Controls how fast emergent spacetime evolves - larger timestep = faster orbit motion. Too large may become unstable."

"Starting distance from black hole. Closer = stronger gravity, faster orbit."

"Field coupling strength - how strong the gravitational pull is."

"How concentrated the black hole's field is. Smaller = more extreme gradients."
```

## Verification

Run these checks on any experiment page:

```bash
# Check naming consistency
grep -r "Binary Orbit" website/src/app/experiments/
# Should only find it in URLs, not display text

# Check for shared component usage
grep -r "import ParameterSlider from" website/src/app/experiments/
# Should see imports, not inline definitions

# Check for tooltip coverage
grep -r "tooltip=" website/src/app/experiments/
# Every ParameterSlider should have one
```

## Development Process Improvements

**Added to process:**
1. ✅ Component standards document (COMPONENT_STANDARDS.md)
2. ✅ Shared component library (src/components/ui/)
3. ✅ Testing checklist before commits
4. ✅ Clear naming conventions
5. ✅ Page structure templates

**Enforcement:**
- Review `COMPONENT_STANDARDS.md` before adding new experiments
- Use shared components instead of creating inline versions
- Add tooltips to EVERY interactive element
- Test navigation flows (especially About page)

---

**Bottom Line:** The site now has:
- ✅ Consistent naming
- ✅ Shared, reusable components
- ✅ Tooltips on all interactive elements (where implemented)
- ✅ Clear standards for future development
- ✅ Professional, consistent look across all pages
