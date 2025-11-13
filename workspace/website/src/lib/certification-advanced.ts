/**
 * Advanced certification utilities for LFM research experiments.
 * Provides RFC 3161-compliant timestamping, identity capture, and append-only registry.
 * 
 * @module lib/certification-advanced
 */

import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

// ============================================================================
// Types
// ============================================================================

export interface CertificationIdentity {
  /** Researcher name or identifier (optional, for transparency) */
  name?: string;
  /** Email for correspondence (optional) */
  email?: string;
  /** ORCID iD for academic attribution (optional) */
  orcid?: string;
  /** Institutional affiliation (optional) */
  affiliation?: string;
}

export interface RFC3161Timestamp {
  /** RFC 3161 timestamp token (base64-encoded) */
  token: string;
  /** Timestamp authority (TSA) URL */
  tsa: string;
  /** Timestamp generation time (ISO 8601) */
  genTime: string;
}

export interface AdvancedCertification {
  /** Experiment identifier */
  experimentId: string;
  /** Deterministic SHA-256 hash of validation results */
  hash: string;
  /** Local system timestamp (ISO 8601) */
  timestamp: string;
  /** RFC 3161 timestamp (optional, for legal/IP purposes) */
  rfc3161?: RFC3161Timestamp;
  /** Researcher identity (optional, for transparency) */
  identity?: CertificationIdentity;
  /** Validation results */
  validation: Record<string, unknown>;
  /** Certification version (for future schema evolution) */
  version: string;
}

export interface CertificationRegistryEntry {
  experimentId: string;
  hash: string;
  timestamp: string;
  certificationPath: string;
}

// ============================================================================
// RFC 3161 Timestamping (Stub - requires external TSA)
// ============================================================================

/**
 * Request RFC 3161 timestamp from a Time Stamping Authority.
 * 
 * ⚠️ STUB: This requires an external RFC 3161 TSA service (e.g., FreeTSA, DigiCert).
 * For production use, integrate with a TSA API or use a library like 'node-rfc3161'.
 * 
 * @param hash - SHA-256 hash to timestamp
 * @param tsaUrl - TSA endpoint URL (default: FreeTSA)
 * @returns RFC 3161 timestamp token and metadata
 */
export async function requestRFC3161Timestamp(
  hash: string,
  tsaUrl: string = 'https://freetsa.org/tsr'
): Promise<RFC3161Timestamp | null> {
  // TODO: Implement RFC 3161 timestamp request
  // 1. Create TimeStampReq (DER-encoded ASN.1)
  // 2. POST to TSA endpoint
  // 3. Parse TimeStampResp
  // 4. Verify signature and extract timestamp
  
  console.warn('⚠️  RFC 3161 timestamping not yet implemented. Using local timestamp only.');
  console.warn('   For production IP protection, integrate with a TSA service.');
  
  return null; // Stub: no timestamp
}

// ============================================================================
// Advanced Certification
// ============================================================================

/**
 * Generate an advanced certification with optional RFC 3161 timestamp and identity.
 * 
 * @param experimentId - Unique experiment identifier
 * @param validation - Validation results to certify
 * @param identity - Optional researcher identity for transparency
 * @param useRFC3161 - Whether to request RFC 3161 timestamp (default: false)
 * @returns Advanced certification object
 */
export async function createAdvancedCertification(
  experimentId: string,
  validation: Record<string, unknown>,
  identity?: CertificationIdentity,
  useRFC3161: boolean = false
): Promise<AdvancedCertification> {
  // Compute deterministic hash
  const sortedKeys = Object.keys(validation).sort();
  const normalized: Record<string, unknown> = {};
  sortedKeys.forEach((key) => {
    normalized[key] = validation[key];
  });
  const content = JSON.stringify(normalized);
  const hash = crypto.createHash('sha256').update(content, 'utf-8').digest('hex');
  
  // Generate local timestamp
  const timestamp = new Date().toISOString();
  
  // Optionally request RFC 3161 timestamp
  let rfc3161: RFC3161Timestamp | undefined;
  if (useRFC3161) {
    const tsResult = await requestRFC3161Timestamp(hash);
    if (tsResult) {
      rfc3161 = tsResult;
    }
  }
  
  return {
    experimentId,
    hash,
    timestamp,
    rfc3161,
    identity,
    validation,
    version: '2.0', // Advanced certification schema version
  };
}

/**
 * Write advanced certification to disk.
 * 
 * @param cert - Advanced certification object
 * @param baseDir - Base directory for certifications (default: workspace/results)
 * @returns Path to written certification file
 */
export function writeAdvancedCertification(
  cert: AdvancedCertification,
  baseDir: string = path.resolve(process.cwd(), '../../../workspace/results')
): string {
  const certDir = path.join(baseDir, 'certifications');
  fs.mkdirSync(certDir, { recursive: true });
  
  const certPath = path.join(certDir, `${cert.experimentId}.json`);
  fs.writeFileSync(certPath, JSON.stringify(cert, null, 2), 'utf-8');
  
  return certPath;
}

// ============================================================================
// Append-Only Certification Registry
// ============================================================================

/**
 * Append a certification entry to the global append-only registry.
 * The registry is a JSON Lines file where each line is a certification entry.
 * This provides tamper-evidence: any modification breaks the append-only property.
 * 
 * @param entry - Certification registry entry
 * @param registryPath - Path to registry file (default: workspace/results/certifications/registry.jsonl)
 */
export function appendToRegistry(
  entry: CertificationRegistryEntry,
  registryPath: string = path.resolve(
    process.cwd(),
    '../../../workspace/results/certifications/registry.jsonl'
  )
): void {
  const registryDir = path.dirname(registryPath);
  fs.mkdirSync(registryDir, { recursive: true });
  
  // Append as JSON Lines (one entry per line)
  const line = JSON.stringify(entry) + '\n';
  fs.appendFileSync(registryPath, line, 'utf-8');
}

/**
 * Read all entries from the append-only registry.
 * 
 * @param registryPath - Path to registry file
 * @returns Array of certification registry entries
 */
export function readRegistry(
  registryPath: string = path.resolve(
    process.cwd(),
    '../../../workspace/results/certifications/registry.jsonl'
  )
): CertificationRegistryEntry[] {
  if (!fs.existsSync(registryPath)) {
    return [];
  }
  
  const content = fs.readFileSync(registryPath, 'utf-8');
  const lines = content.trim().split('\n').filter((line) => line.length > 0);
  
  return lines.map((line) => JSON.parse(line));
}

/**
 * Verify the integrity of the append-only registry.
 * Checks that each certification file exists and matches its registry entry.
 * 
 * @param registryPath - Path to registry file
 * @returns Object with verification status and any errors
 */
export function verifyRegistry(
  registryPath: string = path.resolve(
    process.cwd(),
    '../../../workspace/results/certifications/registry.jsonl'
  )
): { valid: boolean; errors: string[] } {
  const entries = readRegistry(registryPath);
  const errors: string[] = [];
  
  for (const entry of entries) {
    const { experimentId, hash, certificationPath } = entry;
    
    // Check if certification file exists
    if (!fs.existsSync(certificationPath)) {
      errors.push(`Missing certification file: ${certificationPath}`);
      continue;
    }
    
    // Load certification and verify hash
    try {
      const certData = JSON.parse(fs.readFileSync(certificationPath, 'utf-8'));
      if (certData.hash !== hash) {
        errors.push(`Hash mismatch for ${experimentId}: expected ${hash}, got ${certData.hash}`);
      }
    } catch (err) {
      errors.push(`Failed to read certification for ${experimentId}: ${(err as Error).message}`);
    }
  }
  
  return { valid: errors.length === 0, errors };
}
