# Quality Control System for Website Development

**Problem**: AI agents (including me) write inconsistent, buggy code with:
- Duplicate components
- Different import patterns
- Inconsistent layouts
- Missing shared components

**Solution**: Automated validation + binding specification document

---

## System Components

### 1. **EXPERIMENT_PAGE_SPECIFICATION.md** (BINDING CONTRACT)
- **Location**: `workspace/website/EXPERIMENT_PAGE_SPECIFICATION.md`
- **Purpose**: Single source of truth for ALL showcase experiment pages
- **Status**: MANDATORY - no exceptions without approval
- **Contents**:
  - Required imports (exact order)
  - Component hierarchy (mandatory structure)
  - Grid layout patterns (3-col vs 4-col)
  - StandardVisualizationOptions usage
  - StandardMetricsPanel usage
  - Validation checklist
  - Current experiment status audit

**Rule**: Update spec FIRST, then code. Never change code without updating spec.

---

### 2. **Automated Validation Script**
- **Location**: `workspace/website/scripts/validate-experiments.ts`
- **Command**: `npm run validate:experiments`
- **Checks**:
  - ✅ Required imports present (StandardVisualizationOptions, StandardMetricsPanel)
  - ✅ No forbidden patterns (duplicate MetricDisplay, inline VisualizationCheckbox)
  - ✅ Required patterns (ExperimentLayout, visualizationOptions prop)
  - ✅ Proper state management (useSimulationState)
  - ✅ Backend detection
  - ⚠️ Grid layout structure
  - ⚠️ Standard hooks usage

**Output Example**:
```
🔍 Experiment Page Validation Results
============================================================
✅ PASS binary-orbit
✅ PASS three-body
✅ PASS black-hole
✅ PASS stellar-collapse
✅ PASS big-bang
============================================================
📊 Summary: 5 passed, 0 failed
✅ All experiments validated successfully!
```

---

### 3. **Pre-Commit Workflow**

**BEFORE committing ANY showcase experiment changes:**

```bash
# 1. Run validation
npm run validate:experiments

# 2. Check TypeScript
npm run type-check

# 3. Test in browser
npm run dev
# Navigate to http://localhost:3000/experiments/{experiment-name}
# Verify visualization options appear at top
# Verify metrics display correctly
# Test all controls

# 4. If all pass → commit
git add .
git commit -m "feat: update {experiment} experiment"
```

**DO NOT commit if validation fails.**

---

## Current Status (2025-11-11)

### Validated Components
- ✅ **StandardVisualizationOptions.tsx** - Shared visualization controls
- ✅ **StandardMetricsPanel.tsx** - Shared metrics display
- ✅ **ExperimentLayout.tsx** - Shared page layout

### Showcase Experiments Status
| Experiment | Spec Compliant | Validation | TypeScript | Browser Tested |
|------------|---------------|------------|------------|----------------|
| binary-orbit | ✅ YES | ✅ PASS | ✅ PASS | ✅ YES |
| three-body | ⚠️ PARTIAL | ✅ PASS | ✅ PASS | ⏳ NEEDS TEST |
| black-hole | ✅ YES | ✅ PASS | ✅ PASS | ✅ YES |
| stellar-collapse | ✅ YES | ✅ PASS | ✅ PASS | ✅ YES |
| big-bang | ✅ YES | ✅ PASS | ✅ PASS | ✅ YES |

**Issue Identified**: three-body visualization options not rendering correctly due to layout structure mismatch.

**Action Required**: Restructure three-body page to match 4-col pattern (see EXPERIMENT_PAGE_SPECIFICATION.md Pattern B).

---

## AI Agent Instructions

When working on showcase experiments:

### MUST DO:
1. ✅ Read `EXPERIMENT_PAGE_SPECIFICATION.md` BEFORE making changes
2. ✅ Use shared components (StandardVisualizationOptions, StandardMetricsPanel)
3. ✅ Follow exact import order from spec
4. ✅ Use correct grid layout pattern (3-col or 4-col)
5. ✅ Run `npm run validate:experiments` before claiming completion
6. ✅ Test in browser at `localhost:3000/experiments/{name}`
7. ✅ Update spec if proposing structural changes (get approval first)

### MUST NOT DO:
1. ❌ Create duplicate components (MetricDisplay, VisualizationCheckbox)
2. ❌ Hardcode visualization options inline
3. ❌ Invent new layout patterns
4. ❌ Skip validation
5. ❌ Change code without updating spec
6. ❌ Assume it works without browser testing

---

## Enforcement

### Level 1: Automated Validation
- Runs via `npm run validate:experiments`
- Catches structural issues immediately
- Exits with error code 1 if validation fails

### Level 2: Type Checking
- Runs via `npm run type-check`
- Catches type errors, missing props, wrong interfaces

### Level 3: Browser Testing
- Manual verification required
- Check visualization options render at top
- Check metrics display correctly
- Verify all controls work

### Level 4: Code Review
- Human review against specification
- Check for design consistency
- Verify user experience

---

## Evolution Process

**To change the system:**

1. **Propose change** in spec document
2. **Get approval** from project owner
3. **Update EXPERIMENT_PAGE_SPECIFICATION.md**
4. **Update validation script** if needed
5. **Update all affected experiments**
6. **Run validation** to confirm compliance
7. **Document in this file**

**Never bypass this process.**

---

## Benefits

### Before Quality Control System:
- ❌ Inconsistent component usage
- ❌ Duplicate code (150+ lines wasted)
- ❌ Different layouts per experiment
- ❌ Missing shared components
- ❌ No validation before commit
- ❌ Bugs discovered by users

### After Quality Control System:
- ✅ Enforced consistency via validation
- ✅ Shared components (DRY principle)
- ✅ Standardized layouts (user familiarity)
- ✅ Automated validation (catch issues early)
- ✅ Clear specification (single source of truth)
- ✅ Bugs caught before commit

---

## Metrics

### Code Quality
- **Lines eliminated**: ~150+ through component sharing
- **Validation coverage**: 5/5 showcase experiments (100%)
- **TypeScript errors**: 0
- **Specification compliance**: 4.5/5 (90%) - three-body needs layout fix

### Process Quality
- **Pre-commit validation**: Automated via script
- **Specification adherence**: Enforced via binding contract
- **Change management**: Update spec first, then code

---

## Next Steps

1. ⏳ Fix three-body layout to match specification Pattern B
2. ⏳ Add validation to CI/CD pipeline
3. ⏳ Create pre-commit git hook for auto-validation
4. ⏳ Extend validation to research experiment pages
5. ⏳ Add visual regression testing

---

## Contact

For questions about this system:
- **Specification**: See `EXPERIMENT_PAGE_SPECIFICATION.md`
- **Validation**: Run `npm run validate:experiments`
- **Issues**: Check validation output for details
- **Changes**: Propose in spec document first

---

**Remember**: Consistency is king. When in doubt, check the specification.
