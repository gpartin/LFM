import numpy as np

chi_bg = 0.01
chi_slab = 0.10
omega = 0.15

print(f"GRAV-12 Dispersion Analysis")
print(f"=" * 50)
print(f"ω = {omega}")
print(f"χ_bg = {chi_bg}, χ_slab = {chi_slab}")
print()

# Check if wave is evanescent
omega_sq = omega**2
chi_bg_sq = chi_bg**2
chi_slab_sq = chi_slab**2

print(f"ω² = {omega_sq:.6f}")
print(f"χ²_bg = {chi_bg_sq:.6f}, ω²-χ²_bg = {omega_sq - chi_bg_sq:.6f}")
print(f"χ²_slab = {chi_slab_sq:.6f}, ω²-χ²_slab = {omega_sq - chi_slab_sq:.6f}")
print()

if omega_sq < chi_slab_sq:
    print("⚠️  CRITICAL: ω² < χ²_slab → EVANESCENT WAVE IN SLAB")
    print("   Wave cannot propagate through slab (exponential decay)")
else:
    k_bg = np.sqrt(omega_sq - chi_bg_sq)
    k_slab = np.sqrt(omega_sq - chi_slab_sq)
    
    vg_bg = k_bg / omega
    vg_slab = k_slab / omega
    
    lambda_bg = 2 * np.pi / k_bg
    lambda_slab = 2 * np.pi / k_slab
    
    print(f"Background: k={k_bg:.4f}, vg={vg_bg:.4f}, λ={lambda_bg:.2f}")
    print(f"Slab:       k={k_slab:.4f}, vg={vg_slab:.4f}, λ={lambda_slab:.2f}")
    print()
    print(f"Group velocity slowdown: {vg_bg/vg_slab:.2f}x")
    print(f"Wavelength change: {lambda_slab/lambda_bg:.2f}x")

print()
print("Recommended fix:")
print("  - Increase ω to 0.30 or 0.40 (well above χ_slab)")
print("  - This ensures propagating waves in both regions")
print("  - Reduces dispersion and improves packet coherence")
