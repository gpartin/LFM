#!/usr/bin/env node
/**
 * Experiment Page Validation Script
 * 
 * Enforces EXPERIMENT_PAGE_SPECIFICATION.md rules across all showcase experiments.
 * Run before commit to catch inconsistencies.
 * 
 * Usage:
 *   npm run validate:experiments
 *   node scripts/validate-experiments.ts
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface ValidationResult {
  experiment: string;
  passed: boolean;
  errors: string[];
  warnings: string[];
}

const SHOWCASE_EXPERIMENTS = [
  'binary-orbit',
  'three-body',
  'black-hole',
  'stellar-collapse',
  'big-bang',
  'quantum-tunneling', // Newly added quantum profile experiment
];

// STRICT CONSISTENCY REQUIREMENTS - ALL showcase experiments MUST match
const STRICT_REQUIREMENTS = {
  showAdvancedOptions: true,  // ALL experiments MUST have all 9 visualization options
  metricsTitle: 'System Metrics',  // EXACT title required
  backgroundColorCanvas: '#0a0a1a',  // Standardized background color
  gridLayout: 'grid-cols-4',  // 4-column grid (3 canvas, 1 controls)
  canvasHeight: 'h-[600px]',  // Standard canvas height
  defaultShowBackground: true,  // Stars & Background should default ON
  defaultShowLattice: false,  // Simulation Grid should default OFF
};

const REQUIRED_IMPORTS_BASE = [
  'StandardMetricsPanel',
  'ExperimentLayout',
  'useSimulationState',
];

const FORBIDDEN_PATTERNS = [
  { pattern: /const\s+MetricDisplay\s*=/, message: 'Found duplicate MetricDisplay component (use StandardMetricsPanel)' },
  { pattern: /<VisualizationCheckbox\s+(?!.*StandardVisualizationOptions)/, message: 'Found inline VisualizationCheckbox (use StandardVisualizationOptions)' },
  { pattern: /function\s+MetricDisplay\(/, message: 'Found inline MetricDisplay function (use StandardMetricsPanel)' },
];

const REQUIRED_PATTERNS = [
  { pattern: /<ExperimentLayout/, message: 'Must use ExperimentLayout wrapper' },
  { pattern: /visualizationOptions=\{/, message: 'Must pass visualizationOptions prop to ExperimentLayout' },
  { pattern: /<StandardVisualizationOptions/, message: 'Must use StandardVisualizationOptions component' },
  { pattern: /<StandardMetricsPanel/, message: 'Must use StandardMetricsPanel component' },
  { pattern: /useSimulationState/, message: 'Must use useSimulationState hook' },
];

function validateExperiment(experimentName: string): ValidationResult {
  const result: ValidationResult = {
    experiment: experimentName,
    passed: true,
    errors: [],
    warnings: [],
  };

  const filePath = path.join(
    __dirname,
    '..',
    'src',
    'app',
    'experiments',
    experimentName,
    'page.tsx'
  );

  if (!fs.existsSync(filePath)) {
    result.passed = false;
    result.errors.push(`File not found: ${filePath}`);
    return result;
  }

  const content = fs.readFileSync(filePath, 'utf-8');
  // Detect experiment profile:
  // 1) Prefer explicit experimentMeta.profile if present
  // 2) Fallback: infer quantum if QuantumVisualizationOptions is used
  const metaProfileMatchTop = content.match(/experimentMeta[^]*profile:\s*['\"](quantum|classical)['\"]/);
  const inferredQuantum = /QuantumVisualizationOptions/.test(content);
  const profileTop = metaProfileMatchTop ? metaProfileMatchTop[1] : (inferredQuantum ? 'quantum' : 'classical');
  const isQuantumTop = profileTop === 'quantum';

  // Check required imports (profile-aware)
  const REQUIRED_IMPORTS = isQuantumTop
    ? [...REQUIRED_IMPORTS_BASE, 'QuantumVisualizationOptions']
    : [...REQUIRED_IMPORTS_BASE, 'StandardVisualizationOptions'];
  for (const requiredImport of REQUIRED_IMPORTS) {
    if (!content.includes(requiredImport)) {
      result.passed = false;
      result.errors.push(`Missing required import: ${requiredImport}`);
    }
  }

  // Check forbidden patterns
  for (const { pattern, message } of FORBIDDEN_PATTERNS) {
    if (pattern.test(content)) {
      result.passed = false;
      result.errors.push(message);
    }
  }

  // Check required patterns
  // Profile-aware required patterns
  const REQUIRED_PATTERNS_PROFILE = [
    { pattern: /<ExperimentLayout/, message: 'Must use ExperimentLayout wrapper' },
    { pattern: /visualizationOptions=\{/, message: 'Must pass visualizationOptions prop to ExperimentLayout' },
    { pattern: isQuantumTop ? /<QuantumVisualizationOptions/ : /<StandardVisualizationOptions/, message: isQuantumTop ? 'Quantum profile must use QuantumVisualizationOptions component' : 'Classical profile must use StandardVisualizationOptions component' },
    { pattern: /<StandardMetricsPanel/, message: 'Must use StandardMetricsPanel component' },
    { pattern: /useSimulationState/, message: 'Must use useSimulationState hook' },
  ];
  for (const { pattern, message } of REQUIRED_PATTERNS_PROFILE) {
    if (!pattern.test(content)) {
      result.passed = false;
      result.errors.push(message);
    }
  }

  // Check for proper grid layout
  const hasGridLayout = /grid grid-cols-1 lg:grid-cols-[34]/.test(content);
  if (!hasGridLayout) {
    result.warnings.push('Grid layout pattern not detected (expected 3-col or 4-col)');
  }

  // Check for proper ExperimentLayout structure
  const hasVisualizationOptionsOutsideChildren = content.includes('visualizationOptions={');
  if (!hasVisualizationOptionsOutsideChildren) {
    result.passed = false;
    result.errors.push('visualizationOptions must be passed as prop to ExperimentLayout, not inside children');
  }

  // Check for backend detection
  if (!content.includes('detectBackend')) {
    result.warnings.push('Backend detection pattern not found');
  }

  // Check for profile-based UI/Sim decision usage (advisory)
  if (!/decideSimulationProfile\(|__dim:/.test(content)) {
    result.warnings.push('Advisory: consider using decideSimulationProfile() to choose advanced (3D) vs simple (1D) mode based on backend.');
  }

  // Check for proper state management
  if (!content.includes('const [state, dispatch] = useSimulationState()')) {
    result.warnings.push('useSimulationState hook usage pattern not standard');
  }

  // Visual consistency checks
  checkVisualConsistency(experimentName, content, result);
  
  // Check visualization options configuration
  checkVisualizationConfig(experimentName, content, result);

  return result;
}

function checkVisualizationConfig(
  experimentName: string,
  content: string,
  result: ValidationResult
): void {
  // Detect experiment profile (same logic as validateExperiment)
  const metaProfileMatch = content.match(/experimentMeta[^]*profile:\s*['\"](quantum|classical)['\"]/);
  const inferredQuantum = /QuantumVisualizationOptions/.test(content);
  const profile = metaProfileMatch ? metaProfileMatch[1] : (inferredQuantum ? 'quantum' : 'classical');
  const isQuantum = profile === 'quantum';
  // STRICT: Check showAdvancedOptions is TRUE (required for all experiments)
  const showAdvancedMatch = content.match(/showAdvancedOptions=\{(true|false)\}/);
  if (showAdvancedMatch) {
    const actualValue = showAdvancedMatch[1] === 'true';
    if (actualValue !== STRICT_REQUIREMENTS.showAdvancedOptions) {
      result.errors.push(
        `showAdvancedOptions MUST be ${STRICT_REQUIREMENTS.showAdvancedOptions} for consistency (found ${actualValue})`
      );
      result.passed = false;
    }
  } else {
    result.warnings.push('showAdvancedOptions should be explicitly set to {true}');
  }

  // STRICT: Check metrics title (only match title attribute directly on StandardMetricsPanel)
  const metricsTitleMatch = content.match(/<StandardMetricsPanel[^>]*\btitle="([^"]+)"/);
  if (metricsTitleMatch) {
    const actualTitle = metricsTitleMatch[1];
    if (actualTitle !== STRICT_REQUIREMENTS.metricsTitle) {
      result.errors.push(
        `Metrics title MUST be "${STRICT_REQUIREMENTS.metricsTitle}" (found "${actualTitle}")`
      );
      result.passed = false;
    }
  } else {
    result.errors.push(
      `StandardMetricsPanel must have title="${STRICT_REQUIREMENTS.metricsTitle}" attribute`
    );
    result.passed = false;
  }

  // NEW: Enforce unified parameters panel title
  const hasUnifiedParamsTitle = /<h3[^>]*>\s*Experiment Parameters\s*<\/h3>/.test(content);
  const hasUnifiedParamsDataAttr = /data-panel=\"experiment-parameters\"/.test(content);
  if (!hasUnifiedParamsTitle) {
    result.errors.push('Parameters panel title MUST be "Experiment Parameters"');
    result.passed = false;
  }
  if (!hasUnifiedParamsDataAttr) {
    result.warnings.push('Consider adding data-panel="experiment-parameters" for validator-friendly markup');
  }

  // STRICT: Check grid layout
  if (!content.includes(STRICT_REQUIREMENTS.gridLayout)) {
    result.errors.push(
      `Grid layout MUST use "${STRICT_REQUIREMENTS.gridLayout}" (4-column: 3 canvas, 1 controls)`
    );
    result.passed = false;
  }

  // STRICT: Check canvas height
  if (!content.includes(STRICT_REQUIREMENTS.canvasHeight)) {
    result.errors.push(
      `Canvas height MUST be "${STRICT_REQUIREMENTS.canvasHeight}"`
    );
    result.passed = false;
  }

  // Default UI state validation adjusted for quantum profile
  if (!content.includes('useSimulationState')) {
    result.errors.push(
      'Must use useSimulationState hook (ensures baseline defaults)'
    );
    result.passed = false;
  } else {
    const showBackgroundOverride = content.match(/showBackground:\s*false/);
    const showLatticeOverride = content.match(/showLattice:\s*true/);

    if (showBackgroundOverride) {
      result.errors.push('Stars & Background MUST default to ON (showBackground: true).');
      result.passed = false;
    }

    if (showLatticeOverride) {
      if (isQuantum) {
        // Quantum experiments may start with lattice ON for barrier visualization
        result.warnings.push('Quantum profile: lattice default ON accepted (showLattice: true).');
      } else {
        result.errors.push('Simulation Grid MUST default to OFF (showLattice: false) for classical profile.');
        result.passed = false;
      }
    }
  }

  // Quantum-specific advisory metrics check (non-blocking for now)
  if (isQuantum) {
    const hasTransmission = /Transmission\b|T\b/.test(content);
    const hasReflection = /Reflection\b|R\b/.test(content);
    if (!hasTransmission || !hasReflection) {
      result.warnings.push('Quantum profile: consider adding Transmission (T) and Reflection (R) metrics.');
    }
    // Advisory: Fallback simple UI when GPU not available
    if (!/SimpleCanvas/.test(content)) {
      result.warnings.push('Quantum profile: provide a simplified fallback UI (SimpleCanvas) when GPU is not available.');
    }
  }
}

function checkVisualConsistency(
  experimentName: string,
  content: string,
  result: ValidationResult
): void {
  // Check canvas container classes consistency
  const canvasContainerMatch = content.match(/className="([^"]*h-\[600px\][^"]*)"/);
  if (canvasContainerMatch) {
    const classes = canvasContainerMatch[1];
    // Standard patterns (allow either one)
    const pattern1 = 'panel h-[600px] relative overflow-hidden';
    const pattern2 = 'bg-space-panel rounded-lg overflow-hidden border border-space-border h-[600px]';
    
    if (!classes.includes(pattern1) && !classes.includes(pattern2)) {
      result.warnings.push(`Canvas container classes may be non-standard: "${classes}"`);
    }
  }

  // Check button consistency (Play/Pause/Reset should have consistent classes)
  const buttonPattern = /className=.*?(bg-accent-chi|bg-yellow-600|bg-space-border).*?>/g;
  const buttonMatches = content.match(buttonPattern);
  if (buttonMatches && buttonMatches.length > 0) {
    // Just warn if buttons found - detailed check would require more parsing
    result.warnings.push(`Found ${buttonMatches.length} button elements - verify styling consistency manually`);
  }
}

function checkCanvasBackgroundConsistency(): void {
  // Check NBodyCanvas and OrbitCanvas for consistent background colors
  const canvasFiles = [
    path.join(__dirname, '..', 'src', 'components', 'visuals', 'NBodyCanvas.tsx'),
    path.join(__dirname, '..', 'src', 'components', 'visuals', 'OrbitCanvas.tsx'),
  ];

  console.log('\n🎨 Checking Canvas Background Color Consistency...\n');

  const standardBackground = STRICT_REQUIREMENTS.backgroundColorCanvas;

  for (const file of canvasFiles) {
    if (!fs.existsSync(file)) {
      console.log(`  ⚠️  File not found: ${path.basename(file)}`);
      continue;
    }

    const content = fs.readFileSync(file, 'utf-8');
    const backgroundMatch = content.match(/<color attach="background" args=\{([^\}]+)\}/);
    
    if (backgroundMatch) {
      const bgValue = backgroundMatch[1];
      const fileName = path.basename(file);
      
      // Check if it matches standard
      if (bgValue.includes(standardBackground)) {
        console.log(`  ✅ ${fileName}: Standard background color (${standardBackground})`);
      } else {
        console.log(`  ❌ ${fileName}: NON-STANDARD background color: ${bgValue}`);
        console.log(`     Expected: ${standardBackground}`);
      }
    } else {
      console.log(`  ⚠️  ${path.basename(file)}: No background color found`);
    }
  }
}

function printResults(results: ValidationResult[]): void {
  console.log('\n🔍 Experiment Page Validation Results\n');
  console.log('=' .repeat(60));

  let totalPassed = 0;
  let totalFailed = 0;

  for (const result of results) {
    const status = result.passed ? '✅ PASS' : '❌ FAIL';
    const color = result.passed ? '\x1b[32m' : '\x1b[31m';
    const reset = '\x1b[0m';

    console.log(`\n${color}${status}${reset} ${result.experiment}`);

    if (result.errors.length > 0) {
      console.log(`  Errors:`);
      for (const error of result.errors) {
        console.log(`    ❌ ${error}`);
      }
      totalFailed++;
    } else {
      totalPassed++;
    }

    if (result.warnings.length > 0) {
      console.log(`  Warnings:`);
      for (const warning of result.warnings) {
        console.log(`    ⚠️  ${warning}`);
      }
    }
  }

  console.log('\n' + '='.repeat(60));
  console.log(`\n📊 Summary: ${totalPassed} passed, ${totalFailed} failed\n`);

  if (totalFailed > 0) {
    console.log('❌ Validation FAILED - Fix errors before committing\n');
    console.log('📖 See EXPERIMENT_PAGE_SPECIFICATION.md for details\n');
    process.exit(1);
  } else {
    console.log('✅ All experiments validated successfully!\n');
    process.exit(0);
  }
}

function checkRegistryEntries(): string[] {
  const errors: string[] = [];
  
  try {
    // Import experiments registry
    const registryPath = path.join(__dirname, '../src/data/experiments.ts');
    const registryContent = fs.readFileSync(registryPath, 'utf-8');
    
    // Extract experiment IDs from SHOWCASE_EXPERIMENTS array
    const showcaseMatch = registryContent.match(/const SHOWCASE_EXPERIMENTS[^=]*=\s*\[([\s\S]*?)\];/);
    if (!showcaseMatch) {
      errors.push('Could not parse SHOWCASE_EXPERIMENTS array in experiments.ts');
      return errors;
    }
    
    const showcaseArrayContent = showcaseMatch[1];
    const registeredIds: string[] = [];
    
    // Extract all "id: 'xxx'" entries
    const idMatches = showcaseArrayContent.matchAll(/id:\s*['"]([^'"]+)['"]/g);
    for (const match of idMatches) {
      registeredIds.push(match[1]);
    }
    
    console.log(`\n📋 Checking Experiment Registry Integration...\n`);
    console.log(`  Found ${registeredIds.length} experiments in registry`);
    console.log(`  Validating ${SHOWCASE_EXPERIMENTS.length} experiment pages\n`);
    
    // Check each validated page has registry entry
    for (const expId of SHOWCASE_EXPERIMENTS) {
      if (!registeredIds.includes(expId)) {
        errors.push(`❌ ${expId}: Page validated but NOT in experiments registry`);
        console.log(`  ❌ ${expId}: Missing from src/data/experiments.ts`);
      } else {
        console.log(`  ✅ ${expId}: Found in registry`);
      }
    }
    
    // Check for registry entries without pages (orphaned)
    for (const regId of registeredIds) {
      if (!SHOWCASE_EXPERIMENTS.includes(regId)) {
        console.log(`  ⚠️  ${regId}: In registry but no page validated (might be in development)`);
      }
    }
    
  } catch (error) {
    errors.push(`Failed to check registry: ${error}`);
  }
  
  return errors;
}

function main() {
  const args = process.argv.slice(2);
  const targets = args.length > 0 ? args : SHOWCASE_EXPERIMENTS;

  console.log('🔬 Validating showcase experiment pages...\n');
  console.log(`Checking ${targets.length} experiment(s) against specification\n`);

  // Check canvas background color consistency first
  checkCanvasBackgroundConsistency();

  const results = targets.map(validateExperiment);
  
  // Check registry integration for selected targets
  const registryErrors = checkRegistryEntriesForTargets(targets);
  
  if (registryErrors.length > 0) {
    console.log('\n❌ Registry Integration Errors:\n');
    for (const error of registryErrors) {
      console.log(`  ${error}`);
    }
    console.log('\n📝 Action Required:');
    console.log('  Ensure validated experiments are present in src/data/experiments.ts\n');
  }
  
  printResults(results);
  
  // Exit with error if registry validation failed
  if (registryErrors.length > 0) {
    process.exit(1);
  }
}

function checkRegistryEntriesForTargets(targets: string[]): string[] {
  const errors: string[] = [];
  try {
    const registryPath = path.join(__dirname, '../src/data/experiments.ts');
    const registryContent = fs.readFileSync(registryPath, 'utf-8');
    const showcaseMatch = registryContent.match(/const SHOWCASE_EXPERIMENTS[^=]*=\s*\[([\s\S]*?)\];/);
    if (!showcaseMatch) {
      errors.push('Could not parse SHOWCASE_EXPERIMENTS array in experiments.ts');
      return errors;
    }
    const showcaseArrayContent = showcaseMatch[1];
    const registeredIds: string[] = [];
    const idMatches = showcaseArrayContent.matchAll(/id:\s*['"]([^'"]+)['"]/g);
    for (const match of idMatches) {
      registeredIds.push(match[1]);
    }

    console.log(`\n📋 Checking Experiment Registry Integration for ${targets.length} target(s)...\n`);
    for (const expId of targets) {
      if (!registeredIds.includes(expId)) {
        errors.push(`❌ ${expId}: Page validated but NOT in experiments registry`);
        console.log(`  ❌ ${expId}: Missing from src/data/experiments.ts`);
      } else {
        console.log(`  ✅ ${expId}: Found in registry`);
      }
    }
  } catch (error) {
    errors.push(`Failed to check registry: ${error}`);
  }
  return errors;
}

main();
