#!/usr/bin/env python3
"""Check GRAV-15 snapshot energy evolution."""

import h5py
import numpy as np

with h5py.File('results/Gravity/GRAV-15/diagnostics/field_snapshots_3d_GRAV-15.h5', 'r') as f:
    snaps = f['snapshots']
    keys = sorted(snaps.keys())
    
    print(f"Total snapshots: {len(keys)}")
    print(f"N = {f['N'][()]}, dx = {f['dx'][()]}, dt = {f['dt'][()]}")
    print()
    
    print("First 10 snapshots:")
    for k in keys[:10]:
        field = snaps[k][()]
        time = snaps[k].attrs['time']
        E_max = np.abs(field).max()
        E_rms = np.sqrt(np.mean(field**2))
        E_total = np.sum(field**2)
        print(f"  {k}: t={time:6.2f}s, E_max={E_max:.6e}, E_rms={E_rms:.6e}, E_total={E_total:.6e}")
    
    print("\nMiddle 10 snapshots:")
    mid = len(keys) // 2
    for k in keys[mid-5:mid+5]:
        field = snaps[k][()]
        time = snaps[k].attrs['time']
        E_max = np.abs(field).max()
        E_rms = np.sqrt(np.mean(field**2))
        E_total = np.sum(field**2)
        print(f"  {k}: t={time:6.2f}s, E_max={E_max:.6e}, E_rms={E_rms:.6e}, E_total={E_total:.6e}")
    
    print("\nLast 10 snapshots:")
    for k in keys[-10:]:
        field = snaps[k][()]
        time = snaps[k].attrs['time']
        E_max = np.abs(field).max()
        E_rms = np.sqrt(np.mean(field**2))
        E_total = np.sum(field**2)
        print(f"  {k}: t={time:6.2f}s, E_max={E_max:.6e}, E_rms={E_rms:.6e}, E_total={E_total:.6e}")
    
    # Check energy decay
    print("\n=== Energy Evolution ===")
    energies = []
    times = []
    for k in keys:
        field = snaps[k][()]
        time = snaps[k].attrs['time']
        E_total = np.sum(field**2)
        energies.append(E_total)
        times.append(time)
    
    E0 = energies[0]
    print(f"Initial energy: {E0:.6e}")
    print(f"Final energy: {energies[-1]:.6e}")
    print(f"Ratio (final/initial): {energies[-1]/E0:.6f}")
    print(f"\nEnergy drops below 1% of initial at snapshot {np.where(np.array(energies) < 0.01*E0)[0][0] if any(np.array(energies) < 0.01*E0) else 'NEVER'}")
