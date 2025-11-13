/*
 * Certifications API — returns validation statistics per experiment
 * Used to show validation count and unique validator count
 */

import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic'; // Always fresh data

interface CertificationStats {
  totalValidations: number;      // Total number of validations
  uniqueValidators: number;      // Number of unique validators (by fingerprint)
}

interface CertificationCounts {
  [experimentId: string]: CertificationStats;
}

export async function GET() {
  try {
    const workspaceDir = path.resolve(process.cwd(), '..');
    const certDir = path.join(workspaceDir, 'results', 'certifications');
    
    const counts: CertificationCounts = {};
    const validatorSets: Record<string, Set<string>> = {};
    
    // Check if certifications directory exists
    if (!fs.existsSync(certDir)) {
      return NextResponse.json({ counts: {}, total: 0 });
    }
    
    // Read all certification files
    // Only accept filenames that match our certification naming pattern:
    //   <EXPERIMENT_ID>_<CERT_ID>.json
    // Where CERT_ID is either a 16-char hex (deterministic) or a UUID v4 (36 chars with hyphens)
    // Example: REL-01_abcdef0123456789.json
    const CERT_FILE_RE = /^(?:[A-Z0-9-]+)_(?:[a-f0-9]{16}|[0-9a-fA-F-]{36})\.json$/;
    const files = fs
      .readdirSync(certDir)
      .filter(
        (f) => f.endsWith('.json') && !f.endsWith('registry.jsonl') && CERT_FILE_RE.test(f)
      );
    
    for (const file of files) {
      try {
        // Parse certification file
        const filePath = path.join(certDir, file);
        const content = fs.readFileSync(filePath, { encoding: 'utf-8' });
        const cert = JSON.parse(content);
        
        const experimentId = cert.experimentId;
        if (!experimentId) continue;
        
        // Initialize stats if not exists
        if (!counts[experimentId]) {
          counts[experimentId] = {
            totalValidations: 0,
            uniqueValidators: 0,
          };
        }
        // init validator set for this experiment
        if (!validatorSets[experimentId]) {
          validatorSets[experimentId] = new Set<string>();
        }

        counts[experimentId].totalValidations++;
        // Track unique validators using Set (deduplicated in memory)
        const fingerprint: string = cert.validatorFingerprint
          ? String(cert.validatorFingerprint)
          : `anonymous_${cert.certificationId || file}`;
        validatorSets[experimentId].add(fingerprint);
      } catch (err) {
        // Ignore malformed or non-cert JSON files silently to avoid noisy logs
        continue;
      }
    }
    
    // Fill unique validator counts per experiment from the sets built above
    for (const experimentId in counts) {
      counts[experimentId].uniqueValidators = validatorSets[experimentId]?.size || 0;
    }
    
    return NextResponse.json({
      counts,
      total: files.length,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('[Certifications API] Error:', error);
    return NextResponse.json(
      { error: 'Failed to read certifications', message: error?.message },
      { status: 500 }
    );
  }
}
