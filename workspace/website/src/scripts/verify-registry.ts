#!/usr/bin/env node
/**
 * Script to verify the integrity of the append-only certification registry.
 * Checks that all entries have valid certification files and matching hashes.
 * 
 * Usage: npx ts-node src/scripts/verify-registry.ts
 */

import { verifyRegistry, readRegistry } from '../lib/certification-advanced';

console.log('🔍 Verifying certification registry...\n');

const { valid, errors } = verifyRegistry();

if (valid) {
  console.log('✅ Registry verification passed!');
  const entries = readRegistry();
  console.log(`   ${entries.length} certification(s) verified.`);
} else {
  console.log('❌ Registry verification FAILED!\n');
  console.log('Errors:');
  errors.forEach((err) => console.log(`  - ${err}`));
  process.exit(1);
}
