/*
 * Certification utilities — deterministic hash + timestamp for validation attestation
 * Writes to workspace/results/certifications/{experimentId}_{uuid}.json
 * Standardized for all tiers
 */

import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

export interface CertificationData {
  certificationId: string;        // NEW: Unique ID per validation
  experimentId: string;
  timestamp: string;
  hash: string;
  validatorFingerprint?: string;  // NEW: Browser/validator fingerprint
  environment?: ValidationEnvironment; // NEW: Environment metadata
  validation: any;
  version: string;                // NEW: Schema version
}

export interface ValidationEnvironment {
  pythonVersion?: string;
  numpyVersion?: string;
  cupyVersion?: string;
  os?: string;
  gpu?: string;
  backend?: string;
  latticeSize?: number;
  dt?: number;
  dx?: number;
}

export function computeCertificationHash(data: any): string {
  // Deterministic hash of validation result JSON
  const json = JSON.stringify(data, Object.keys(data).sort());
  return crypto.createHash('sha256').update(json).digest('hex');
}

export function writeCertification(
  experimentId: string, 
  validation: any,
  validatorFingerprint?: string,
  environment?: ValidationEnvironment
): string {
  // Generate deterministic certification ID from fingerprint + experimentId
  // This prevents duplicate certifications from the same validator for the same experiment
  const certificationIdSource = validatorFingerprint 
    ? `${experimentId}:${validatorFingerprint}`
    : `${experimentId}:${crypto.randomUUID()}`; // Fallback to random if no fingerprint
  
  const certificationId = crypto.createHash('sha256')
    .update(certificationIdSource)
    .digest('hex')
    .substring(0, 16); // Use first 16 chars for readability
  
  const timestamp = new Date().toISOString();
  const hash = computeCertificationHash(validation);
  const attestation: CertificationData = {
    certificationId,
    experimentId,
    timestamp,
    hash,
    validatorFingerprint,
    environment,
    validation,
    version: '2.0',
  };
  const workspaceDir = path.resolve(process.cwd(), '..');
  const certDir = path.join(workspaceDir, 'results', 'certifications');
  if (!fs.existsSync(certDir)) {
    fs.mkdirSync(certDir, { recursive: true });
  }
  
  // Use deterministic certification ID in filename
  // This will overwrite previous certifications from the same validator
  const certPath = path.join(certDir, `${experimentId}_${certificationId}.json`);
  fs.writeFileSync(certPath, JSON.stringify(attestation, null, 2), { encoding: 'utf-8' });
  return certPath;
}

export function extractAttestationFromSummary(summary: any): { hash: string; timestamp: string } | null {
  // Standardized extraction for all tiers
  if (!summary) return null;
  
  // Try multiple field names for compatibility
  const hash = summary.certification_hash || summary.hash || summary.sha256;
  const timestamp = summary.certification_timestamp || summary.timestamp || summary.time;
  
  if (hash && timestamp) {
    return { hash, timestamp };
  }
  
  return null;
}
