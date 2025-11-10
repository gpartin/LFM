/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';

export async function POST(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const experimentId = params.id;
  
  try {
    const body = await request.json();
    const { parameters, metrics: uiMetrics } = body;
    
    // Map experiment ID prefix to test harness tier runner
    // Uses prefix matching for all 105 tests
    const getTierInfo = (testId: string): { runner: string; tier: string; config: string; resultsDir: string } | null => {
      if (testId.startsWith('REL-')) {
        return { 
          runner: 'run_tier1_relativistic.py', 
          tier: '1',
          config: '../config/config_tier1_relativistic.json',
          resultsDir: 'Relativistic'
        };
      } else if (testId.startsWith('GRAV-')) {
        return { 
          runner: 'run_tier2_gravity.py', 
          tier: '2',
          config: '../config/config_tier2_gravityanalogue.json',
          resultsDir: 'Gravity'
        };
      } else if (testId.startsWith('ENER-')) {
        return { 
          runner: 'run_tier3_energy.py', 
          tier: '3',
          config: '../config/config_tier3_energy.json',
          resultsDir: 'Energy'
        };
      } else if (testId.startsWith('QUAN-')) {
        return { 
          runner: 'run_tier4_quantization.py', 
          tier: '4',
          config: '../config/config_tier4_quantization.json',
          resultsDir: 'Quantization'
        };
      } else if (testId.startsWith('EM-')) {
        return { 
          runner: 'run_tier5_electromagnetic.py', 
          tier: '5',
          config: '../config/config_tier5_electromagnetic.json',
          resultsDir: 'Electromagnetic'
        };
      } else if (testId.startsWith('COUP-')) {
        return { 
          runner: 'run_tier6_coupling.py', 
          tier: '6',
          config: '../config/config_tier6_coupling.json',
          resultsDir: 'Coupling'
        };
      } else if (testId.startsWith('THERM-')) {
        return { 
          runner: 'run_tier7_thermodynamics.py', 
          tier: '7',
          config: '../config/config_tier7_thermodynamics.json',
          resultsDir: 'Thermodynamics'
        };
      }
      return null;
    };
    
    const testInfo = getTierInfo(experimentId);
    if (!testInfo) {
      return NextResponse.json(
        { error: `Test ${experimentId} not found in validation mapping` },
        { status: 404 }
      );
    }
    
    // Paths (absolute from repository root)
    const workspaceRoot = path.join(process.cwd(), '..');
    const repoRoot = path.join(workspaceRoot, '..'); // Go up from workspace to repo root
    const srcDir = path.join(workspaceRoot, 'src');
    const resultsDir = path.join(workspaceRoot, 'results', 'website_validation', experimentId);
    const pythonPath = path.join(repoRoot, '.venv', 'Scripts', 'python.exe');
    
    // Ensure results directory exists
    await fs.mkdir(resultsDir, { recursive: true });
    
    const startTime = Date.now();
    
    // Spawn Python test harness
    const result = await new Promise<{
      exitCode: number;
      stdout: string;
      stderr: string;
    }>((resolve, reject) => {
      const child = spawn(
        pythonPath,
        [
          testInfo.runner,
          '--test', experimentId,
          '--config', testInfo.config,
          // GPU usage is controlled by config file (run_settings.use_gpu)
        ],
        {
          cwd: srcDir,
          env: {
            ...process.env,
            PYTHONPATH: srcDir,
          },
        }
      );
      
      let stdout = '';
      let stderr = '';
      
      child.stdout.on('data', (data) => {
        stdout += data.toString();
        console.log(`[${experimentId}] ${data.toString()}`);
      });
      
      child.stderr.on('data', (data) => {
        stderr += data.toString();
        console.error(`[${experimentId}] ${data.toString()}`);
      });
      
      child.on('close', (exitCode) => {
        resolve({ exitCode: exitCode || 0, stdout, stderr });
      });
      
      child.on('error', (error) => {
        reject(error);
      });
      
      // Timeout after 5 minutes
      setTimeout(() => {
        child.kill();
        reject(new Error('Test execution timeout (5 minutes)'));
      }, 5 * 60 * 1000);
    });
    
    const duration = (Date.now() - startTime) / 1000;
    
    // Load validation thresholds
    const thresholdsPath = path.join(workspaceRoot, 'config', 'validation_thresholds.json');
    const thresholdsContent = await fs.readFile(thresholdsPath, 'utf-8');
    const thresholds = JSON.parse(thresholdsContent);
    
    // Read test results from harness output directory
    const testResultsPath = path.join(workspaceRoot, 'results', testInfo.resultsDir, experimentId, 'summary.json');
    let testResults: any = null;
    let metrics: any = {};
    let status = 'FAIL';
    let validationDetails: string[] = [];
    
    try {
      const summaryContent = await fs.readFile(testResultsPath, 'utf-8');
      testResults = JSON.parse(summaryContent);
      
      // Extract key metrics from harness
      if (testResults.anisotropy !== undefined) {
        metrics.anisotropy = testResults.anisotropy;
      }
      if (testResults.energy_drift !== undefined) {
        metrics.energyDrift = testResults.energy_drift;
      }
      if (testResults.omega_right !== undefined) {
        metrics.omega_right = testResults.omega_right;
      }
      if (testResults.omega_left !== undefined) {
        metrics.omega_left = testResults.omega_left;
      }
    } catch (error) {
      console.error(`Failed to read test results for ${experimentId}:`, error);
      metrics = { error: 'Results file not found' };
    }
    
    // Validate: Compare UI metrics against Python harness metrics
    let allChecksPassed = true;
    
    // If no UI metrics provided, validation cannot proceed
    if (!uiMetrics || Object.keys(uiMetrics).length === 0) {
      allChecksPassed = false;
      validationDetails.push('❌ No UI metrics provided - run simulation first before validating');
      status = 'FAIL';
    } else {
      // Check energy drift against threshold
      if (uiMetrics.energyDrift !== undefined) {
        const threshold = thresholds.tolerances.energy_drift;
        const passed = Math.abs(uiMetrics.energyDrift) < threshold;
        if (!passed) {
          allChecksPassed = false;
          validationDetails.push(`❌ Energy drift ${uiMetrics.energyDrift.toExponential(3)} exceeds threshold ${threshold.toExponential(3)}`);
        } else {
          validationDetails.push(`✓ Energy drift ${uiMetrics.energyDrift.toExponential(3)} < ${threshold.toExponential(3)}`);
        }
      }
      
      // Check anisotropy against threshold
      if (uiMetrics.anisotropy !== undefined) {
        const threshold = thresholds.tolerances.anisotropy;
        const passed = Math.abs(uiMetrics.anisotropy) < threshold;
        if (!passed) {
          allChecksPassed = false;
          validationDetails.push(`❌ Anisotropy ${uiMetrics.anisotropy.toExponential(3)} exceeds threshold ${threshold.toExponential(3)}`);
        } else {
          validationDetails.push(`✓ Anisotropy ${uiMetrics.anisotropy.toExponential(3)} < ${threshold.toExponential(3)}`);
        }
      }
      
      // Compare UI vs Python metrics (if available)
      if (metrics.anisotropy !== undefined && uiMetrics.anisotropy !== undefined) {
        const diff = Math.abs(metrics.anisotropy - uiMetrics.anisotropy);
        const maxAllowed = 0.01; // Allow 1% difference
        if (diff > maxAllowed) {
          validationDetails.push(`⚠ UI anisotropy (${uiMetrics.anisotropy.toExponential(3)}) differs from Python (${metrics.anisotropy.toExponential(3)}) by ${diff.toExponential(3)}`);
        } else {
          validationDetails.push(`✓ UI anisotropy matches Python within ${maxAllowed}`);
        }
      }
      
      status = allChecksPassed ? 'PASS' : 'FAIL';
    }
    
    // Write diagnostic JSON for website tracking
    const diagnosticData = {
      testId: experimentId,
      timestamp: new Date().toISOString(),
      status,
      uiMetrics,
      pythonMetrics: metrics,
      validationDetails,
      duration,
      exitCode: result.exitCode,
      platform: process.platform,
      notes: result.exitCode === 0 
        ? 'Test harness completed successfully'
        : `Test harness exited with code ${result.exitCode}`,
      stdout: result.stdout.slice(-1000), // Last 1000 chars
      stderr: result.stderr.slice(-1000),
    };
    
    await fs.writeFile(
      path.join(resultsDir, 'validation.json'),
      JSON.stringify(diagnosticData, null, 2),
      'utf-8'
    );
    
    return NextResponse.json({
      status,
      uiMetrics,
      pythonMetrics: metrics,
      validationDetails,
      duration,
      testId: experimentId,
      timestamp: diagnosticData.timestamp,
      notes: diagnosticData.notes,
    });
    
  } catch (error: any) {
    console.error(`Validation error for ${experimentId}:`, error);
    return NextResponse.json(
      { 
        error: error.message,
        testId: experimentId,
        status: 'ERROR',
      },
      { status: 500 }
    );
  }
}
