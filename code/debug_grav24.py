"""Debug GRAV-24 gravitational wave test"""
import numpy as np
import json

# Load summary
with open('results/Gravity/GRAV-24/summary.json') as f:
    summary = json.load(f)

print("GRAV-24 Summary:")
print(f"  Passed: {summary['passed']}")
print(f"  Wave speed measured: {summary['wave_speed_measured']:.3f}")
print(f"  Wave speed expected: {summary['wave_speed_expected']:.3f}")
print(f"  Speed error: {summary['speed_error']*100:.1f}%")
print(f"  χ amplitude: {summary['chi_amplitude']:.6e}")
print(f"  Wave propagated: {summary['wave_propagated']}")

# Load wave propagation data
data = np.loadtxt('results/Gravity/GRAV-24/diagnostics/wave_propagation_GRAV-24.csv',
                  delimiter=',', skiprows=1)

print("\nWave propagation data:")
print("  step, time, peak_position, wave_speed")
for i, row in enumerate(data[:10]):
    print(f"  {int(row[0]):4d}, {row[1]:6.2f}, {row[2]:6.2f}, {row[3]:6.3f}")
print("  ...")
for i, row in enumerate(data[-5:]):
    print(f"  {int(row[0]):4d}, {row[1]:6.2f}, {row[2]:6.2f}, {row[3]:6.3f}")

# Analysis
steps = data[:, 0]
times = data[:, 1]
positions = data[:, 2]
speeds = data[:, 3]

print(f"\nPosition range: {positions.min():.2f} to {positions.max():.2f}")
print(f"Speed range: {speeds.min():.3f} to {speeds.max():.3f}")
print(f"Mean speed (non-zero): {speeds[speeds > 0].mean():.3f}" if np.any(speeds > 0) else "  No non-zero speeds")

# Count how many have zero position
zero_pos = np.sum(positions == 0)
print(f"Zero positions: {zero_pos}/{len(positions)}")
