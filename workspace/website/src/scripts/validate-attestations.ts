#!/usr/bin/env node
/**
 * Script to validate attestations across all experiment results.
 * Automates checking that certification files exist, hashes match, and timestamps are valid.
 * 
 * Usage: npx ts-node src/scripts/validate-attestations.ts
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

interface AttestationResult {
  experimentId: string;
  status: 'valid' | 'missing' | 'hash-mismatch' | 'invalid-timestamp';
  details?: string;
}

/**
 * Compute deterministic SHA-256 hash of experiment results (same as certification.ts).
 */
function computeHash(data: Record<string, unknown>): string {
  const sortedKeys = Object.keys(data).sort();
  const normalized: Record<string, unknown> = {};
  sortedKeys.forEach((key) => {
    normalized[key] = data[key];
  });
  const content = JSON.stringify(normalized);
  return crypto.createHash('sha256').update(content, 'utf-8').digest('hex');
}

/**
 * Validate attestation for a single experiment.
 */
function validateAttestation(experimentId: string, resultsDir: string): AttestationResult {
  const certPath = path.join(resultsDir, 'certifications', `${experimentId}.json`);
  
  // Check if certification file exists
  if (!fs.existsSync(certPath)) {
    return { experimentId, status: 'missing', details: 'Certification file not found' };
  }
  
  // Load certification
  const certData = JSON.parse(fs.readFileSync(certPath, 'utf-8'));
  const { hash, timestamp, validation } = certData;
  
  // Validate timestamp format (ISO 8601)
  const timestampRegex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;
  if (!timestamp || !timestampRegex.test(timestamp)) {
    return { experimentId, status: 'invalid-timestamp', details: `Invalid timestamp: ${timestamp}` };
  }
  
  // Recompute hash and compare
  const recomputedHash = computeHash(validation || {});
  if (recomputedHash !== hash) {
    return {
      experimentId,
      status: 'hash-mismatch',
      details: `Expected ${hash}, got ${recomputedHash}`,
    };
  }
  
  return { experimentId, status: 'valid' };
}

/**
 * Validate all attestations in the results directory.
 */
function validateAllAttestations(resultsDir: string): void {
  console.log('Validating attestations...\n');
  
  const certDir = path.join(resultsDir, 'certifications');
  if (!fs.existsSync(certDir)) {
    console.error('❌ Certification directory not found:', certDir);
    process.exit(1);
  }
  
  const certFiles = fs.readdirSync(certDir).filter((f) => f.endsWith('.json'));
  const results: AttestationResult[] = [];
  
  for (const file of certFiles) {
    const experimentId = path.basename(file, '.json');
    const result = validateAttestation(experimentId, resultsDir);
    results.push(result);
  }
  
  // Summary
  const valid = results.filter((r) => r.status === 'valid').length;
  const invalid = results.filter((r) => r.status !== 'valid').length;
  
  console.log(`\n📊 Summary:`);
  console.log(`  ✓ Valid: ${valid}`);
  console.log(`  ✗ Invalid: ${invalid}`);
  
  if (invalid > 0) {
    console.log('\n❌ Invalid attestations:');
    results
      .filter((r) => r.status !== 'valid')
      .forEach((r) => console.log(`  ${r.experimentId}: ${r.status} - ${r.details}`));
    process.exit(1);
  } else {
    console.log('\n✅ All attestations valid!');
  }
}

// Run validation
const resultsDir = path.resolve(__dirname, '../../../../workspace/results');
validateAllAttestations(resultsDir);
