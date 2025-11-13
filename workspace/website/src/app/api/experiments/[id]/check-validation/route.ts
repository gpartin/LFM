/*
 * Check Validation API — verifies if current validator already validated this experiment
 * Returns existing certification if found
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface RouteParams {
  params: {
    id: string;
  };
}

export async function POST(req: NextRequest, { params }: RouteParams) {
  try {
    const { id: experimentId } = params;
    const body = await req.json();
    const { validatorFingerprint } = body;

    if (!validatorFingerprint) {
      return NextResponse.json(
        { error: 'Missing validatorFingerprint' },
        { status: 400 }
      );
    }

    // Generate deterministic certification ID (same logic as writeCertification)
    const certificationIdSource = `${experimentId}:${validatorFingerprint}`;
    const certificationId = crypto
      .createHash('sha256')
      .update(certificationIdSource)
      .digest('hex')
      .substring(0, 16);

    // Check if certification file exists
    const workspaceDir = path.resolve(process.cwd(), '..');
    const certDir = path.join(workspaceDir, 'results', 'certifications');
    const certPath = path.join(certDir, `${experimentId}_${certificationId}.json`);

    if (fs.existsSync(certPath)) {
      // Read existing certification
      const content = fs.readFileSync(certPath, { encoding: 'utf-8' });
      const cert = JSON.parse(content);

      return NextResponse.json({
        alreadyValidated: true,
        certification: cert,
        certificationPath: path.basename(certPath),
      });
    }

    return NextResponse.json({
      alreadyValidated: false,
    });
  } catch (error: any) {
    console.error('[Check Validation API] Error:', error);
    return NextResponse.json(
      { error: 'Failed to check validation', message: error?.message },
      { status: 500 }
    );
  }
}
