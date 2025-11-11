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

const REQUIRED_IMPORTS = [
  'StandardVisualizationOptions',
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

  // Check required imports
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
  for (const { pattern, message } of REQUIRED_PATTERNS) {
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
    // Default is true, but we require it to be explicit
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

  // Check default UI state (must use useSimulationState with correct defaults)
  // Note: All experiments should use the shared hook which has the correct defaults
  if (!content.includes('useSimulationState')) {
    result.errors.push(
      'Must use useSimulationState hook (ensures correct defaults: showBackground=true, showLattice=false)'
    );
    result.passed = false;
  } else {
    // Check for custom overrides that violate defaults
    const showBackgroundOverride = content.match(/showBackground:\s*false/);
    const showLatticeOverride = content.match(/showLattice:\s*true/);
    
    if (showBackgroundOverride) {
      result.errors.push(
        'Stars & Background MUST default to ON (showBackground: true). Remove custom override.'
      );
      result.passed = false;
    }
    
    if (showLatticeOverride) {
      result.errors.push(
        'Simulation Grid MUST default to OFF (showLattice: false). Remove custom override.'
      );
      result.passed = false;
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

function main() {
  console.log('🔬 Validating showcase experiment pages...\n');
  console.log(`Checking ${SHOWCASE_EXPERIMENTS.length} experiments against specification\n`);

  // Check canvas background color consistency first
  checkCanvasBackgroundConsistency();

  const results = SHOWCASE_EXPERIMENTS.map(validateExperiment);
  printResults(results);
}

main();
