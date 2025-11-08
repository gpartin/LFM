/**
 * DiagnosticLogger - Framework for tracking simulation metrics over time
 * 
@ * ⚠️ SECURITY NOTE: This class is designed for INTERNAL USE ONLY.
@ * Do NOT expose diagnostic recording or CSV export to public website UI.
@ * Use only in local development scripts (see /analysis folder).
@ * 
 * Provides:
 * - Real-time metric tracking
 * - Configurable sampling rates
 * - CSV export
 * - Time-series data access for plotting
 */

export interface DiagnosticMetric {
  name: string;
  value: number;
  unit?: string;
  timestamp: number;
}

export interface DiagnosticSnapshot {
  time: number;
  stepCount: number;
  metrics: Record<string, number>;
}

export interface DiagnosticConfig {
  maxSamples?: number;        // Ring buffer size (default 10000)
  samplingInterval?: number;  // Min ms between samples (0 = every call)
  autoExport?: boolean;       // Auto-export on buffer full
}

export class DiagnosticLogger {
  private snapshots: DiagnosticSnapshot[] = [];
  private maxSamples: number;
  private samplingInterval: number;
  private lastSampleTime: number = 0;
  private startTime: number;
  private metricNames: Set<string> = new Set();

  constructor(config: DiagnosticConfig = {}) {
    this.maxSamples = config.maxSamples ?? 10000;
    this.samplingInterval = config.samplingInterval ?? 0;
    this.startTime = performance.now();
  }

  /**
   * Record a snapshot of current metrics
   */
  record(time: number, stepCount: number, metrics: Record<string, number>): void {
    const now = performance.now();
    if (this.samplingInterval > 0 && now - this.lastSampleTime < this.samplingInterval) {
      return; // Skip this sample
    }
    this.lastSampleTime = now;

    // Track metric names
    Object.keys(metrics).forEach(name => this.metricNames.add(name));

    // Add snapshot (ring buffer)
    this.snapshots.push({ time, stepCount, metrics });
    if (this.snapshots.length > this.maxSamples) {
      this.snapshots.shift();
    }
  }

  /**
   * Get all snapshots
   */
  getSnapshots(): DiagnosticSnapshot[] {
    return this.snapshots;
  }

  /**
   * Get time series for a specific metric
   */
  getMetricSeries(metricName: string): { time: number; value: number }[] {
    return this.snapshots
      .filter(s => s.metrics[metricName] !== undefined)
      .map(s => ({ time: s.time, value: s.metrics[metricName] }));
  }

  /**
   * Get latest snapshot
   */
  getLatest(): DiagnosticSnapshot | null {
    return this.snapshots.length > 0 ? this.snapshots[this.snapshots.length - 1] : null;
  }

  /**
   * Export to CSV string
   */
  toCSV(): string {
    if (this.snapshots.length === 0) return '';
    
    const metrics = Array.from(this.metricNames).sort();
    const header = ['time', 'stepCount', ...metrics].join(',');
    const rows = this.snapshots.map(s => {
      const values = [
        s.time.toFixed(6),
        s.stepCount.toString(),
        ...metrics.map(m => (s.metrics[m] ?? '').toString())
      ];
      return values.join(',');
    });
    
    return [header, ...rows].join('\n');
  }

  /**
   * Download CSV file (browser only)
   */
  downloadCSV(filename: string = 'diagnostics.csv'): void {
    if (typeof window === 'undefined') return;
    
    const csv = this.toCSV();
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.snapshots = [];
    this.metricNames.clear();
    this.lastSampleTime = 0;
    this.startTime = performance.now();
  }

  /**
   * Get summary statistics for a metric
   */
  getStats(metricName: string): { min: number; max: number; mean: number; latest: number } | null {
    const series = this.getMetricSeries(metricName);
    if (series.length === 0) return null;
    
    const values = series.map(s => s.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const latest = values[values.length - 1];
    
    return { min, max, mean, latest };
  }

  /**
   * Get metric count
   */
  getSampleCount(): number {
    return this.snapshots.length;
  }

  /**
   * Get elapsed time since logger creation
   */
  getElapsedTime(): number {
    return (performance.now() - this.startTime) / 1000;
  }
}
