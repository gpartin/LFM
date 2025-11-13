/**
 * Browser fingerprinting for validator identity
 * Creates a unique, persistent ID for each browser/device
 * Privacy-preserving: hash only, no PII collected
 */

export interface BrowserFingerprint {
  fingerprintHash: string;
  timestamp: string;
}

/**
 * Generate browser fingerprint on client side
 * Uses available browser APIs to create unique identifier
 */
export function generateBrowserFingerprint(): string {
  const components: string[] = [];
  
  // Browser information
  components.push(navigator.userAgent);
  components.push(navigator.language);
  components.push(String(navigator.hardwareConcurrency || 0));
  // deviceMemory is non-standard; feature-detect to satisfy TS DOM typings
  const deviceMemory = (typeof (navigator as any).deviceMemory === 'number') ? (navigator as any).deviceMemory : 0;
  components.push(String(deviceMemory));
  
  // Screen information
  components.push(String(screen.width));
  components.push(String(screen.height));
  components.push(String(screen.colorDepth));
  components.push(String(window.devicePixelRatio || 1));
  
  // Timezone
  components.push(Intl.DateTimeFormat().resolvedOptions().timeZone);
  components.push(String(new Date().getTimezoneOffset()));
  
  // Canvas fingerprint (simplified)
  try {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.textBaseline = 'top';
      ctx.font = '14px Arial';
      ctx.fillText('LFM Validator', 2, 2);
      components.push(canvas.toDataURL());
    }
  } catch (e) {
    // Canvas fingerprinting blocked
  }
  
  // WebGL fingerprint (simplified)
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl') as WebGLRenderingContext;
    if (gl) {
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      if (debugInfo) {
        components.push(gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL));
        components.push(gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL));
      }
    }
  } catch (e) {
    // WebGL fingerprinting blocked
  }
  
  // Hash all components
  const fingerprint = components.join('|||');
  
  // Use SubtleCrypto if available, otherwise simple hash
  if (typeof window !== 'undefined' && window.crypto && window.crypto.subtle) {
    // Return promise-based hash (caller must await)
    return fingerprint; // Will be hashed in getValidatorFingerprint()
  }
  
  return fingerprint;
}

/**
 * Get or create persistent validator fingerprint
 * Stores in localStorage for consistency across sessions
 */
export async function getValidatorFingerprint(): Promise<string> {
  if (typeof window === 'undefined') {
    return 'server-side';
  }
  
  // Check if we already have a fingerprint
  const stored = localStorage.getItem('lfm_validator_fingerprint');
  if (stored) {
    return stored;
  }
  
  // Generate new fingerprint
  const rawFingerprint = generateBrowserFingerprint();
  
  // Hash using SubtleCrypto
  try {
    const encoder = new TextEncoder();
    const data = encoder.encode(rawFingerprint);
    const hashBuffer = await window.crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    const fingerprint = `sha256:${hashHex}`;
    
    // Store for future use
    localStorage.setItem('lfm_validator_fingerprint', fingerprint);
    
    return fingerprint;
  } catch (err) {
    // Fallback to simple hash if SubtleCrypto fails
    const simpleHash = rawFingerprint.split('').reduce((acc, char) => {
      return ((acc << 5) - acc) + char.charCodeAt(0);
    }, 0);
    const fingerprint = `fallback:${Math.abs(simpleHash).toString(16)}`;
    localStorage.setItem('lfm_validator_fingerprint', fingerprint);
    return fingerprint;
  }
}

/**
 * Get environment metadata for validation
 */
export function getEnvironmentMetadata() {
  if (typeof window === 'undefined') {
    return null;
  }
  
  return {
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    language: navigator.language,
    screenResolution: `${screen.width}x${screen.height}`,
    colorDepth: screen.colorDepth,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  };
}
