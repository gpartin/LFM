# -*- coding: utf-8 -*-
"""Fix algorithm1_wave.py"""

FIXED_FUNCTION = '''def lattice_step_masked(E, E_prev, chi, c, dt, dx, gamma, mask, xp):
    """Verlet step with masked updates."""
    E_next = xp.array(E, copy=True)
    
    # Compute Laplacian only on active cells
    lap = laplacian_masked(E, dx, mask, xp)
    
    # Verlet update
    c2 = c * c
    dt2 = dt * dt
    
    # Compute chi squared term
    if xp.isscalar(chi):
        chi2 = chi * chi
    else:
        chi2 = chi * chi
    
    term_mass = -chi2 * E
    term_wave = c2 * lap
    
    # Update only masked cells (ensure proper broadcasting)
    update_value = ((2.0 - gamma) * E - 
                    (1.0 - gamma) * E_prev +
                    dt2 * (term_wave + term_mass))
    
    # Apply mask
    E_next = xp.where(mask, update_value, E)
    
    return E_next


'''

with open('algorithm1_wave.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the function start
for i, line in enumerate(lines):
    if 'def lattice_step_masked' in line:
        start_idx = i
        break

# Find the next function
for i in range(start_idx + 1, len(lines)):
    if lines[i].startswith('def '):
        end_idx = i
        break

# Replace the function
output = lines[:start_idx] + [FIXED_FUNCTION] + lines[end_idx:]

with open('algorithm1_wave.py', 'w', encoding='utf-8') as f:
    f.writelines(output)

print('Fixed algorithm1_wave.py successfully!')
