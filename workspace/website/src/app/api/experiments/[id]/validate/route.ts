/*
 * Validation API — runs the Python test harness for a given experiment id.
 * Guarded by env var ALLOW_HARNESS_RUNS=1 to prevent accidental remote execution.
 */

import { NextRequest } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { writeCertification, computeCertificationHash } from '@/lib/certification';
import {
  createAdvancedCertification,
  writeAdvancedCertification,
  appendToRegistry,
  type CertificationIdentity,
} from '@/lib/certification-advanced';

export const runtime = 'nodejs';

function mapPrefixToRunnerAndConfig(id: string): { runner: string; config: string } | null {
  // Prefix-based mapping; ensure gravity analogue uses correct harness filename.
  const prefix = id.split('-')[0].toUpperCase();
  switch (prefix) {
    case 'REL':
      return { runner: 'run_tier1_relativistic.py', config: '../config/config_tier1_relativistic.json' };
    case 'GRAV':
      // Corrected runner name (previously run_tier2_gravity.py which does not exist).
      return { runner: 'run_tier2_gravityanalogue.py', config: '../config/config_tier2_gravityanalogue.json' };
    case 'ENER':
      return { runner: 'run_tier3_energy.py', config: '../config/config_tier3_energy.json' };
    case 'QUAN':
      return { runner: 'run_tier4_quantization.py', config: '../config/config_tier4_quantization.json' };
    case 'EM':
      return { runner: 'run_tier5_electromagnetic.py', config: '../config/config_tier5_electromagnetic.json' };
    case 'COUP':
      return { runner: 'run_tier6_coupling.py', config: '../config/config_tier6_coupling.json' };
    case 'THERM':
      return { runner: 'run_tier7_thermodynamics.py', config: '../config/config_tier7_thermodynamics.json' };
    default:
      return null;
  }
}

function getPaths() {
  // websiteDir = c:\LFM\workspace\website
  const websiteDir = process.cwd();
  const workspaceDir = path.resolve(websiteDir, '..'); // c:\LFM\workspace
  const srcDir = path.join(workspaceDir, 'src'); // c:\LFM\workspace\src
  const repoRoot = path.resolve(workspaceDir, '..'); // c:\LFM
  const pythonExe = path.join(repoRoot, '.venv', 'Scripts', 'python.exe');
  return { websiteDir, workspaceDir, srcDir, repoRoot, pythonExe };
}

export async function POST(req: NextRequest, { params }: { params: { id: string } }) {
  const allow = process.env.ALLOW_HARNESS_RUNS === '1';
  const id = decodeURIComponent(params.id);
  // If harness execution is disabled, attempt to serve a static summary fallback (read-only evidence) if present.
  if (!allow) {
    let staticSummary: any = null;
    try {
      const { workspaceDir } = getPaths();
      const fallbackPaths: string[] = [];
      // Results canonical locations to attempt (ordered)
      fallbackPaths.push(path.join(workspaceDir, 'results', id.toUpperCase(), 'summary.json'));
      const tierFolderMap: Record<string, string> = {
        'REL': 'Relativistic',
        'GRAV': 'Gravity',
        'ENER': 'Energy',
        'QUAN': 'Quantization',
        'EM': 'Electromagnetic',
        'COUP': 'Coupling',
        'THERM': 'Thermodynamics',
      };
      const prefix = id.split('-')[0].toUpperCase();
      const tierFolder = tierFolderMap[prefix];
      if (tierFolder) {
        fallbackPaths.push(path.join(workspaceDir, 'results', tierFolder, id.toUpperCase(), 'summary.json'));
      }
      for (const p of fallbackPaths) {
        if (staticSummary) break;
        if (fs.existsSync(p)) {
          try {
            const raw = fs.readFileSync(p, { encoding: 'utf-8' });
            staticSummary = JSON.parse(raw);
            // Mark as static / non-executed for frontend clarity.
            staticSummary.static_mode = true;
            staticSummary.execution_disabled = true;
          } catch {}
        }
      }
    } catch {}
    return new Response(JSON.stringify({
      ok: false,
      status: 'disabled',
      message: 'Harness execution disabled – set ALLOW_HARNESS_RUNS=1 to enable live validation.',
      details: {
        summary: staticSummary,
        disabledReason: 'env:ALLOW_HARNESS_RUNS!=1',
      },
    }), { status: 503, headers: { 'content-type': 'application/json; charset=utf-8' } });
  }
  // Normal execution path below
  const mapping = mapPrefixToRunnerAndConfig(id);
  if (!mapping) {
    return new Response(JSON.stringify({ ok: false, message: `Unknown experiment id prefix for ${id}` }), { status: 400, headers: { 'content-type': 'application/json; charset=utf-8' } });
  }

  const { srcDir, pythonExe } = getPaths();
  if (!fs.existsSync(pythonExe)) {
    return new Response(JSON.stringify({ ok: false, message: `Python venv not found at ${pythonExe}` }), { status: 500, headers: { 'content-type': 'application/json; charset=utf-8' } });
  }

  // Build arguments — prefer parallel suite even for single test
  // GPU usage is enabled via config (gpu_enabled: true); the suite does not accept a --gpu flag
  const runner = path.join(srcDir, 'run_parallel_suite.py');
  const args = [runner, '--tests', id, '--max-concurrent', '1'];

  // Spawn Python process in srcDir
  const stdoutChunks: string[] = [];
  const stderrChunks: string[] = [];

  const child = spawn(pythonExe, args, { cwd: srcDir, windowsHide: true });

  const exited = new Promise<{ code: number | null }>((resolve) => {
    child.stdout.setEncoding('utf-8');
    child.stderr.setEncoding('utf-8');
    child.stdout.on('data', (d) => stdoutChunks.push(String(d)));
    child.stderr.on('data', (d) => stderrChunks.push(String(d)));
    child.on('close', (code) => resolve({ code }));
  });

  // 5 minute hard timeout
  const timeoutMs = 5 * 60 * 1000;
  const timeout = new Promise<{ code: number | null }>((resolve) => setTimeout(() => {
    try { child.kill('SIGTERM'); } catch {}
    resolve({ code: -1 });
  }, timeoutMs));

  const { code } = await Promise.race([exited, timeout]);

  const stdout = stdoutChunks.join('');
  const stderr = stderrChunks.join('');

  // Try to find summary - check multiple locations
  let summary: any = null;
  try {
    // Strategy 1: Look for SUMMARY_JSON: marker in stdout
    const summaryMatch = stdout.match(/SUMMARY_JSON:\s*(.*\.json)/);
    if (summaryMatch) {
      const summaryPath = summaryMatch[1].trim();
      if (fs.existsSync(summaryPath)) {
        const content = fs.readFileSync(summaryPath, { encoding: 'utf-8' });
        summary = JSON.parse(content);
      }
    }
    
    // Strategy 2: Look in workspace/results/{id}/summary.json
    if (!summary) {
      const { workspaceDir } = getPaths();
      const resultsSummaryPath = path.join(workspaceDir, 'results', id.toUpperCase(), 'summary.json');
      if (fs.existsSync(resultsSummaryPath)) {
        const content = fs.readFileSync(resultsSummaryPath, { encoding: 'utf-8' });
        summary = JSON.parse(content);
      }
    }
    // Strategy 2b: Tier folder fallback e.g., results/Coupling/COUP-01/summary.json
    if (!summary) {
      const { workspaceDir } = getPaths();
      const tierFolderMap: Record<string, string> = {
        'REL': 'Relativistic',
        'GRAV': 'Gravity',
        'ENER': 'Energy',
        'QUAN': 'Quantization',
        'EM': 'Electromagnetic',
        'COUP': 'Coupling',
        'THERM': 'Thermodynamics',
      };
      const prefix = id.split('-')[0].toUpperCase();
      const tierFolder = tierFolderMap[prefix];
      if (tierFolder) {
        const tierResultsSummaryPath = path.join(workspaceDir, 'results', tierFolder, id.toUpperCase(), 'summary.json');
        if (fs.existsSync(tierResultsSummaryPath)) {
          const content = fs.readFileSync(tierResultsSummaryPath, { encoding: 'utf-8' });
          summary = JSON.parse(content);
        }
      }
    }
    
    // Strategy 3: Look in workspace/src/results/{id}/summary.json  
    if (!summary) {
      const { srcDir } = getPaths();
      const srcResultsSummaryPath = path.join(srcDir, 'results', id.toUpperCase(), 'summary.json');
      if (fs.existsSync(srcResultsSummaryPath)) {
        const content = fs.readFileSync(srcResultsSummaryPath, { encoding: 'utf-8' });
        summary = JSON.parse(content);
      }
    }
  } catch (err) {
    console.error('[API] Error parsing summary:', err);
  }

  // Determine pass/fail strictly from summary when available.
  // Harness may exit with code 2 even on success; rely on summary.passed/status fields.
  const summaryPassed = summary && (
    summary.passed === true ||
    (typeof summary.status === 'string' && summary.status.toLowerCase() === 'passed')
  );
  // Fallback: if no summary, treat exit codes 0/2 as pass (legacy behavior) to maintain compatibility.
  const ok = summary ? summaryPassed : (code === 0 || code === 2);

  // Attach explicit exit code + pass flag for frontend diagnostics.
  if (summary) {
    summary.exit_code = code; // non-canonical diagnostic field
    summary.pass_flag = summaryPassed; // normalized boolean
  }

  // Parse request body to check for advanced certification options
  let body: any = null;
  try {
    body = await req.json();
  } catch {
    // No body or invalid JSON - use basic certification
  }

  // Extract validator fingerprint and environment from request
  const validatorFingerprint = body?.validatorFingerprint;
  const environment = body?.environment;

  // Certification: deterministic hash + timestamp, write to results/certifications
  let certificationPath = null;
  if (ok && summary) {
    try {
      // Check for advanced certification request (optional identity in request body)
      const useAdvanced = body?.useAdvancedCertification || false;
      const identity: CertificationIdentity | undefined = body?.identity;
      
      if (useAdvanced) {
        // Generate advanced certification with optional RFC 3161 timestamp
        const useRFC3161 = body?.useRFC3161 || false;
        const advancedCert = await createAdvancedCertification(
          id,
          summary,
          identity,
          useRFC3161
        );
        certificationPath = writeAdvancedCertification(advancedCert);
        
        // Append to tamper-evident registry
        appendToRegistry({
          experimentId: id,
          hash: advancedCert.hash,
          timestamp: advancedCert.timestamp,
          certificationPath,
        });
      } else {
        // Use basic certification (backward compatible)
        certificationPath = writeCertification(id, summary, validatorFingerprint, environment);
      }
      
      // Append to registry for tamper-evident log
      if (certificationPath && !certificationPath.startsWith('Error')) {
        appendToRegistry({
          experimentId: id,
          hash: summary.certification_hash || summary.hash || computeCertificationHash(summary),
          timestamp: new Date().toISOString(),
          certificationPath,
        });
      }
    } catch (err) {
      certificationPath = `Error writing certification: ${err}`;
    }
  }

  return new Response(JSON.stringify({
    ok,
    message: ok ? `Validation completed for ${id}` : `Validation failed (exit code ${code}${summary ? ', summary.pass_flag=' + (summary.pass_flag ? 'true' : 'false') : ''})`,
    details: {
      summary,
      exitCode: code,
      certificationPath: certificationPath ? path.basename(certificationPath) : null, // Only filename, not full path
    },
  }), { status: ok ? 200 : 500, headers: { 'content-type': 'application/json; charset=utf-8' } });
}
 
