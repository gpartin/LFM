#!/usr/bin/env node
/**
 * Script to generate research experiment template code from tier test metadata.
 * Automates boilerplate creation for new experiments added to the registry.
 * 
 * Usage: npx ts-node src/scripts/generate-experiment-template.ts --test-id REL-01
 */

import fs from 'fs';
import path from 'path';

interface GenerateOptions {
  testId: string;
  outputDir?: string;
}

/**
 * Generate a research experiment template for a given test ID.
 */
function generateExperimentTemplate(options: GenerateOptions): void {
  const { testId, outputDir = './src/data/experiments' } = options;
  
  // Extract tier and type from test ID
  const tierPrefix = testId.split('-')[0];
  const tierMap: Record<string, string> = {
    'REL': 'tier1',
    'GRAV': 'tier2',
    'ENER': 'tier3',
    'QUAN': 'tier4',
    'EM': 'tier5',
    'COUP': 'tier6',
    'ADV': 'tier7',
  };
  
  const tier = tierMap[tierPrefix] || 'unknown';
  
  // Template for experiment metadata
  const template = {
    id: testId.toLowerCase(),
    tier,
    title: `${testId} - [EDIT: Add descriptive title]`,
    category: '[EDIT: relativistic | gravity | energy | quantization | em | coupling | advanced]',
    description: '[EDIT: Add physics description]',
    significance: '[EDIT: Explain why this test matters]',
    simulationType: '[EDIT: wave-packet | n-body | field-dynamics]' as const,
    status: 'not-run' as const,
    initialConditions: {
      latticeSize: 128,
      dt: 0.0001,
      dx: 0.05,
      steps: 1000,
      chi: 1.0,
      // [EDIT: Add type-specific params like wavePacket, particles, etc.]
    },
    expectedResults: {
      energyConservation: '< 1e-4',
      primaryMetric: '[EDIT: Add expected metric]',
    },
    validation: {
      criteria: [
        'Energy drift < 1e-4',
        '[EDIT: Add additional criteria]',
      ],
      thresholds: {
        energyDrift: 1e-4,
        // [EDIT: Add other thresholds]
      },
    },
    links: {
      config: `../config/config_${tier}_*.json`,
      results: `../results/${testId}/`,
      discovery: '[EDIT: Link to discovery doc if applicable]',
    },
  };
  
  // Write template to file
  const outputPath = path.join(outputDir, `${testId.toLowerCase()}.json`);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(template, null, 2), 'utf-8');
  
  console.log(`✓ Generated experiment template: ${outputPath}`);
  console.log('  Next steps:');
  console.log('  1. Edit the [EDIT:...] placeholders with actual values');
  console.log('  2. Add type-specific parameters to initialConditions');
  console.log('  3. Import into experiments/index.ts registry');
  console.log('  4. Create or link corresponding canvas component');
}

// CLI parsing
const args = process.argv.slice(2);
const testIdIndex = args.indexOf('--test-id');
if (testIdIndex === -1 || !args[testIdIndex + 1]) {
  console.error('Usage: npx ts-node generate-experiment-template.ts --test-id <TEST-ID>');
  process.exit(1);
}

generateExperimentTemplate({ testId: args[testIdIndex + 1] });
