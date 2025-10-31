"""Debug GRAV-25 light bending test"""
import numpy as np
import matplotlib.pyplot as plt

# Load trajectory data
data = np.loadtxt('results/Gravity/GRAV-25/diagnostics/light_bending_GRAV-25.csv', 
                  delimiter=',', skiprows=1)

x = data[:, 0]
chi = data[:, 1]
E_final = data[:, 2]

# Parameters from config
dx = 0.05
dt = 0.005
steps = 800
k_fraction = 0.10
N = 1024

# Calculate expected parameters
kx = k_fraction * np.pi / dx
print(f"kx = {kx:.4f}")

# Initial position (20% from left)
x0_cell = N * 0.2
x0 = x0_cell * dx
print(f"Initial position: x0 = {x0:.2f} (cell {x0_cell:.0f})")

# Chi at initial position
chi_min = 0.01
chi_max = chi_min + 0.02
chi_at_x0 = chi_min + (chi_max - chi_min) * 0.2
print(f"χ at x0: {chi_at_x0:.6f}")

# Dispersion relation
c = 1.0
omega = np.sqrt(c**2 * kx**2 + chi_at_x0**2)
print(f"ω = {omega:.6f}")

# Group velocity (corrected formula)
vg = c**2 * kx / omega
print(f"Group velocity: vg = {vg:.6f}")

# Expected travel distance
time_total = steps * dt
distance = vg * time_total
print(f"Time: {time_total:.2f}, Distance: {distance:.2f}")

# Expected final position
expected_x = x0 + distance
expected_cell = expected_x / dx
print(f"Expected final position: {expected_x:.2f} (cell {expected_cell:.0f})")

# Actual final position (find peak)
peak_idx = np.argmax(np.abs(E_final))
actual_x = x[peak_idx]
actual_cell = peak_idx
print(f"Actual final position: {actual_x:.2f} (cell {actual_cell})")

# Analysis
print(f"\nΔx = {actual_x - expected_x:.2f}")
print(f"Δ cells = {actual_cell - expected_cell:.0f}")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Chi field
axes[0].plot(x, chi, 'r-', linewidth=2)
axes[0].axvline(x0, color='g', linestyle='--', label='Initial x0')
axes[0].axvline(expected_x, color='b', linestyle='--', label='Expected final')
axes[0].axvline(actual_x, color='orange', linestyle='--', linewidth=2, label='Actual final')
axes[0].set_xlabel('x')
axes[0].set_ylabel('χ(x)')
axes[0].set_title('χ-field gradient')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# E-field (full range)
axes[1].plot(x, E_final, 'b-', linewidth=1)
axes[1].axvline(x0, color='g', linestyle='--', alpha=0.5)
axes[1].axvline(expected_x, color='b', linestyle='--', alpha=0.5)
axes[1].axvline(actual_x, color='orange', linestyle='--', linewidth=2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('E(x)')
axes[1].set_title('Final E-field')
axes[1].grid(True, alpha=0.3)

# E-field (zoomed to packet region)
# Find region with significant E
E_abs = np.abs(E_final)
threshold = 0.01 * np.max(E_abs)
significant = np.where(E_abs > threshold)[0]
if len(significant) > 0:
    x_min = max(0, significant[0] - 20)
    x_max = min(len(x)-1, significant[-1] + 20)
    axes[2].plot(x[x_min:x_max], E_final[x_min:x_max], 'b-', linewidth=1.5)
    axes[2].axvline(x0, color='g', linestyle='--', alpha=0.5, label='Initial')
    axes[2].axvline(expected_x, color='b', linestyle='--', alpha=0.5, label='Expected')
    axes[2].axvline(actual_x, color='orange', linestyle='--', linewidth=2, label='Actual')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('E(x)')
    axes[2].set_title('Final E-field (zoomed to packet)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_grav25.png', dpi=150)
print("\nSaved debug_grav25.png")
plt.close()

# Check if E-field is non-zero anywhere
print(f"\nE-field stats:")
print(f"  Max |E|: {np.max(np.abs(E_final)):.6e}")
print(f"  RMS E: {np.sqrt(np.mean(E_final**2)):.6e}")
print(f"  Non-zero cells: {np.sum(np.abs(E_final) > 1e-100)} / {len(E_final)}")
