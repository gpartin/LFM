#!/usr/bin/env node
/**
 * STRICT SHOWCASE CONSISTENCY FIX
 * 
 * Automatically fixes all showcase experiments to match strict requirements:
 * - Grid layout: 4 columns (3 canvas, 1 controls)
 * - Metrics title: "System Metrics" 
 * - Visualization options: showAdvancedOptions={true} (explicit)
 * - Background color: #0a0a1a
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const EXPERIMENTS = ['binary-orbit', 'black-hole', 'stellar-collapse', 'big-bang'];

function fixExperiment(experimentName: string): void {
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
    console.log(`  ⚠️  ${experimentName}: File not found`);
    return;
  }

  let content = fs.readFileSync(filePath, 'utf-8');
  let changed = false;

  // Fix 1: Change grid-cols-3 to grid-cols-4
  if (content.includes('lg:grid-cols-3')) {
    content = content.replace(/lg:grid-cols-3/g, 'lg:grid-cols-4');
    content = content.replace(/lg:col-span-2/g, 'lg:col-span-3');
    changed = true;
    console.log(`  ✅ ${experimentName}: Fixed grid layout (3-col → 4-col)`);
  }

  // Fix 2: Add explicit showAdvancedOptions={true}
  const showAdvancedRegex = /<StandardVisualizationOptions\s+state=\{state\.ui\}\s+onChange=\{[^}]+\}/;
  if (!content.includes('showAdvancedOptions={true}')) {
    content = content.replace(
      showAdvancedRegex,
      (match) => match + '\n          showAdvancedOptions={true}'
    );
    changed = true;
    console.log(`  ✅ ${experimentName}: Added explicit showAdvancedOptions={true}`);
  }

  // Fix 3: Change metrics title to "System Metrics"
  const titleRegex = /title="[^"]*"\s*(?=\/>|$)/;
  const metricsMatch = content.match(/<StandardMetricsPanel[^>]*\/>/s);
  if (metricsMatch && !metricsMatch[0].includes('title="System Metrics"')) {
    content = content.replace(
      /<StandardMetricsPanel([^>]*?)(?:title="[^"]*")?([^>]*?)\/>/s,
      '<StandardMetricsPanel$1 title="System Metrics"$2/>'
    );
    changed = true;
    console.log(`  ✅ ${experimentName}: Fixed metrics title → "System Metrics"`);
  }

  if (changed) {
    fs.writeFileSync(filePath, content, 'utf-8');
    console.log(`  💾 ${experimentName}: Saved changes`);
  } else {
    console.log(`  ℹ️  ${experimentName}: No changes needed`);
  }
}

console.log('\n🔧 Applying strict consistency fixes to showcase experiments...\n');

for (const exp of EXPERIMENTS) {
  console.log(`\n${exp}:`);
  fixExperiment(exp);
}

console.log('\n\n✅ All fixes applied. Run validation to confirm.\n');
