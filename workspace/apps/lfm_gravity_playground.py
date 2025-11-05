#!/usr/bin/env python3
"""
LFM Gravity Playground — Earth–Moon two-body toy using multi-scale lattice

This app builds on physics.multi_scale_lattice to let you "play" with
mass-scale gravity analogues in LFM. It creates two χ-field wells to
represent Earth and the Moon, then evolves the canonical LFM field E on a
multi-resolution hierarchy. Optionally, it moves the Moon marker according to
an illustrative acceleration computed from the χ-gradient (toy Newtonian).

Notes
- Units are "lattice units"; you can set a physical scale if desired.
- This is a visualization/intuition tool, not a calibrated GR solver.
- Keeps commit-friendly outputs: small PNGs and CSV, no large HDF5 files.
"""
from __future__ import annotations
import os, math, argparse, time
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np

try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

import sys
from pathlib import Path
# Ensure 'workspace/src' is on the import path when running as a script
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from physics.multi_scale_lattice import LatticeLevel, MultiScaleLattice  # type: ignore[import-not-found]
from core.lfm_equation import advance, _xp_for  # type: ignore[import-not-found]


@dataclass
class Body:
    name: str
    radius: float  # in lattice units
    chi_center: float  # peak χ in core
    chi_background: float  # χ far away
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    mobile: bool = False


def _gaussian_well_chi(xp, X, Y, Z, cx, cy, cz, radius, chi_center, chi_bg, width_multiplier=4.0):
    r = xp.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    # Smooth well that asymptotes to chi_bg; width controlled by width_multiplier
    return chi_bg + (chi_center - chi_bg) * xp.exp(-(r / max(1e-12, width_multiplier * radius)) ** 2)


def build_levels(N0=96, dx0=4.0,
                 N1=192, dx1=None,
                 N2=192, dx2=None,
                 use_gpu=False,
                 single_level=False,
                 earth: Body | None = None,
                 moon: Body | None = None,
                 chi_width_multiplier=4.0) -> MultiScaleLattice:
    """Create a 3-level hierarchy with Earth and Moon χ-fields.
    
    If single_level=True, only creates Level 0 for fast iteration.

    Defaults:
      - Level0: coarse, spans ~ N0*dx0 units; intended to fit both bodies
      - Level1: medium, centered near Earth; higher resolution
      - Level2: fine, centered near Earth; finest resolution
    """
    if dx1 is None:
        dx1 = dx0 / 4
    if dx2 is None:
        dx2 = dx1 / 4
    xp = cp if (use_gpu and _HAS_CUPY) else np

    # Default bodies (dimensionless scale)
    if earth is None:
        earth = Body("Earth", radius=16.0, chi_center=0.35, chi_background=0.10,
                     position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), mobile=False)
    if moon is None:
        # radius scaled ~ 0.27 Earth; weaker chi depth ~ mass^1/3 toy
        moon = Body("Moon", radius=4.5, chi_center=0.18, chi_background=0.10,
                    position=(60.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), mobile=True)

    # Helper to make grid
    def mk_grid(N, dx, origin):
        # Use arange so grid spacing is exactly dx and aligns with physical_to_grid
        x = origin[0] + xp.arange(N) * dx
        y = origin[1] + xp.arange(N) * dx
        z = origin[2] + xp.arange(N) * dx
        return xp.meshgrid(x, y, z, indexing='ij')

    # Level 0 (coarse): center so both bodies fit
    origin0 = (-N0/2 * dx0, -N0/2 * dx0, -N0/2 * dx0)
    X0, Y0, Z0 = mk_grid(N0, dx0, origin0)
    chi0_e = _gaussian_well_chi(xp, X0, Y0, Z0, *earth.position, earth.radius, earth.chi_center, earth.chi_background, chi_width_multiplier)
    chi0_m = _gaussian_well_chi(xp, X0, Y0, Z0, *moon.position, moon.radius, moon.chi_center, moon.chi_background, chi_width_multiplier)
    chi0 = chi0_e + chi0_m - earth.chi_background

    L0 = LatticeLevel(
        level=0, dx=dx0, origin=origin0, shape=(N0, N0, N0),
        E=xp.zeros((N0, N0, N0), dtype=xp.float64),
        E_prev=xp.zeros((N0, N0, N0), dtype=xp.float64),
        chi=chi0,
    )
    # attach per-body chi for overlays
    L0.chi_earth = chi0_e
    L0.chi_moon = chi0_m
    
    # Fast mode: single level only
    if single_level:
        return MultiScaleLattice([L0])

    # Level 1 (medium): same center but higher resolution window around Earth–Moon
    origin1 = (-N1/2 * dx1, -N1/2 * dx1, -N1/2 * dx1)
    X1, Y1, Z1 = mk_grid(N1, dx1, origin1)
    chi1_e = _gaussian_well_chi(xp, X1, Y1, Z1, *earth.position, earth.radius, earth.chi_center, earth.chi_background, chi_width_multiplier)
    chi1_m = _gaussian_well_chi(xp, X1, Y1, Z1, *moon.position, moon.radius, moon.chi_center, moon.chi_background, chi_width_multiplier)
    chi1 = chi1_e + chi1_m - earth.chi_background

    L1 = LatticeLevel(
        level=1, dx=dx1, origin=origin1, shape=(N1, N1, N1),
        E=xp.zeros((N1, N1, N1), dtype=xp.float64),
        E_prev=xp.zeros((N1, N1, N1), dtype=xp.float64),
        chi=chi1,
    )
    L1.chi_earth = chi1_e
    L1.chi_moon = chi1_m

    # Level 2 (fine)
    origin2 = (-N2/2 * dx2, -N2/2 * dx2, -N2/2 * dx2)
    X2, Y2, Z2 = mk_grid(N2, dx2, origin2)
    chi2_e = _gaussian_well_chi(xp, X2, Y2, Z2, *earth.position, earth.radius, earth.chi_center, earth.chi_background, chi_width_multiplier)
    chi2_m = _gaussian_well_chi(xp, X2, Y2, Z2, *moon.position, moon.radius, moon.chi_center, moon.chi_background, chi_width_multiplier)
    chi2 = chi2_e + chi2_m - earth.chi_background

    L2 = LatticeLevel(
        level=2, dx=dx2, origin=origin2, shape=(N2, N2, N2),
        E=xp.zeros((N2, N2, N2), dtype=xp.float64),
        E_prev=xp.zeros((N2, N2, N2), dtype=xp.float64),
        chi=chi2,
    )
    L2.chi_earth = chi2_e
    L2.chi_moon = chi2_m

    return MultiScaleLattice([L0, L1, L2])


def update_moon_position(level: LatticeLevel, body: Body, k_acc=0.02, dt=1.0, use_kinematic=True, 
                         earth: Body = None, enable_collision=False):
    """Update moon position using either kinematic ∇χ or emergent ∇E² model.
    
    Args:
        use_kinematic: If True, use ∇χ (correct for LFM theory test).
                      If False, use ∇E² (emergent but causes bouncing).
        earth: Earth body for collision detection (if enabled).
        enable_collision: If True, prevents moon from passing through Earth.
    """
    xp = _xp_for(level.E)
    # Sample gradient at nearest grid index
    i, j, k = level.physical_to_grid(*body.position)
    i = max(1, min(level.shape[0]-2, i))
    j = max(1, min(level.shape[1]-2, j))
    k = max(1, min(level.shape[2]-2, k))
    
    if use_kinematic:
        # KINEMATIC MODEL: Use ∇χ directly (tests that ∇χ → gravity)
        # Use ONLY Earth's chi field (not moon's own field!) to avoid self-force
        chi = level.chi_earth
        dchi_dx = (chi[i+1, j, k] - chi[i-1, j, k]) / (2 * level.dx)
        dchi_dy = (chi[i, j+1, k] - chi[i, j-1, k]) / (2 * level.dx)
        dchi_dz = (chi[i, j, k+1] - chi[i, j, k-1]) / (2 * level.dx)
        # Acceleration toward regions of HIGHER chi (where χ peaks)
        # ∇χ points toward increasing χ (inward for Gaussian), so use +k_acc (no minus!)
        ax, ay, az = (+k_acc * float(dchi_dx), +k_acc * float(dchi_dy), +k_acc * float(dchi_dz))
    else:
        # EMERGENT MODEL: Use ∇E² (couples to wave dynamics, causes bouncing)
        E = level.E
        E2 = E**2
        dE2_dx = (E2[i+1, j, k] - E2[i-1, j, k]) / (2 * level.dx)
        dE2_dy = (E2[i, j+1, k] - E2[i, j-1, k]) / (2 * level.dx)
        dE2_dz = (E2[i, j, k+1] - E2[i, j, k-1]) / (2 * level.dx)
        ax, ay, az = (-k_acc * float(dE2_dx), -k_acc * float(dE2_dy), -k_acc * float(dE2_dz))
    
    vx, vy, vz = body.velocity
    vx += ax * dt; vy += ay * dt; vz += az * dt
    px, py, pz = body.position
    px += vx * dt; py += vy * dt; pz += vz * dt
    
    # COLLISION DETECTION: Check if moon would pass through Earth
    if enable_collision and earth is not None:
        import numpy as np
        # Distance between centers
        dx, dy, dz = px - earth.position[0], py - earth.position[1], pz - earth.position[2]
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        contact_dist = earth.radius + body.radius
        
        if dist < contact_dist:
            # Collision! Place moon at contact surface and reflect velocity
            # Normal vector pointing from Earth to Moon
            if dist > 1e-6:
                nx, ny, nz = dx/dist, dy/dist, dz/dist
            else:
                nx, ny, nz = 1.0, 0.0, 0.0  # Fallback
            
            # Place moon just outside Earth's surface
            px = earth.position[0] + nx * contact_dist
            py = earth.position[1] + ny * contact_dist
            pz = earth.position[2] + nz * contact_dist
            
            # Reflect velocity (bounce with energy loss)
            v_normal = vx*nx + vy*ny + vz*nz
            restitution = 0.7  # Inelastic collision (70% bounce)
            vx = vx - (1 + restitution) * v_normal * nx
            vy = vy - (1 + restitution) * v_normal * ny
            vz = vz - (1 + restitution) * v_normal * nz
    
    # Keep inside lattice bounds
    x0, y0, z0 = level.origin
    L = (level.shape[0]-1) * level.dx
    px = max(x0, min(x0 + L, px))
    py = max(y0, min(y0 + L, py))
    pz = max(z0, min(z0 + L, pz))
    body.velocity = (vx, vy, vz)
    body.position = (px, py, pz)
    return body.position


def refresh_chi_for_bodies(level: LatticeLevel, earth: Body, moon: Body, chi_width_multiplier=4.0):
    xp = _xp_for(level.E)
    N = level.shape[0]
    # Match grid generation to exact dx spacing
    x = level.origin[0] + xp.arange(N) * level.dx
    y = level.origin[1] + xp.arange(N) * level.dx
    z = level.origin[2] + xp.arange(N) * level.dx
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    chi_e = _gaussian_well_chi(xp, X, Y, Z, *earth.position, earth.radius, earth.chi_center, earth.chi_background, chi_width_multiplier)
    chi_m = _gaussian_well_chi(xp, X, Y, Z, *moon.position, moon.radius, moon.chi_center, moon.chi_background, chi_width_multiplier)
    level.chi = chi_e + chi_m - earth.chi_background
    # Keep per-body fields for overlays
    level.chi_earth = chi_e
    level.chi_moon = chi_m


def solve_static_E_field(level: LatticeLevel, alpha=1.0, beta=1.0, max_iterations=1000, tolerance=1e-5):
    """Solve the static LFM equation: ∇·(χ∇E) + (β/α)∇χ·∇E = 0
    
    This gives the equilibrium E field configuration for a given χ distribution,
    analogous to solving Poisson's equation ∇²φ = ρ in Newtonian gravity.
    
    Uses vectorized Jacobi iterative relaxation for GPU efficiency.
    """
    xp = _xp_for(level.E)
    dx = level.dx
    chi = level.chi
    N = level.shape[0]
    
    # Initialize E field to small non-zero values (helps convergence)
    E = xp.ones_like(level.E) * 0.001
    E_old = E.copy()
    
    # Precompute chi gradients (finite differences, centered)
    dchi_dx = xp.zeros_like(chi)
    dchi_dy = xp.zeros_like(chi)
    dchi_dz = xp.zeros_like(chi)
    
    dchi_dx[1:-1, :, :] = (chi[2:, :, :] - chi[:-2, :, :]) / (2 * dx)
    dchi_dy[:, 1:-1, :] = (chi[:, 2:, :] - chi[:, :-2, :]) / (2 * dx)
    dchi_dz[:, :, 1:-1] = (chi[:, :, 2:] - chi[:, :, :-2]) / (2 * dx)
    
    # Relaxation parameter (1.0 = Jacobi, 1.5-1.8 = SOR for faster convergence)
    omega = 1.6
    beta_over_alpha = beta / alpha
    
    print(f"[static-solver] Solving ∇·(χ∇E) + ({beta_over_alpha:.2f})∇χ·∇E = 0")
    print(f"[static-solver] Grid: {N}³, dx={dx}, tolerance={tolerance}")
    
    for iteration in range(max_iterations):
        E_old[:] = E
        
        # Vectorized Jacobi update for interior points [1:-1, 1:-1, 1:-1]
        # Compute chi at cell faces (arithmetic mean)
        chi_xp = 0.5 * (chi[1:-1, 1:-1, 1:-1] + chi[2:, 1:-1, 1:-1])
        chi_xm = 0.5 * (chi[1:-1, 1:-1, 1:-1] + chi[:-2, 1:-1, 1:-1])
        chi_yp = 0.5 * (chi[1:-1, 1:-1, 1:-1] + chi[1:-1, 2:, 1:-1])
        chi_ym = 0.5 * (chi[1:-1, 1:-1, 1:-1] + chi[1:-1, :-2, 1:-1])
        chi_zp = 0.5 * (chi[1:-1, 1:-1, 1:-1] + chi[1:-1, 1:-1, 2:])
        chi_zm = 0.5 * (chi[1:-1, 1:-1, 1:-1] + chi[1:-1, 1:-1, :-2])
        
        # Laplacian part: ∇·(χ∇E)
        laplacian = (chi_xp * E_old[2:, 1:-1, 1:-1] + chi_xm * E_old[:-2, 1:-1, 1:-1] +
                     chi_yp * E_old[1:-1, 2:, 1:-1] + chi_ym * E_old[1:-1, :-2, 1:-1] +
                     chi_zp * E_old[1:-1, 1:-1, 2:] + chi_zm * E_old[1:-1, 1:-1, :-2])
        
        denom = chi_xp + chi_xm + chi_yp + chi_ym + chi_zp + chi_zm
        
        # Gradient coupling term: (β/α)·∇χ·∇E
        dE_dx = (E_old[2:, 1:-1, 1:-1] - E_old[:-2, 1:-1, 1:-1]) / (2 * dx)
        dE_dy = (E_old[1:-1, 2:, 1:-1] - E_old[1:-1, :-2, 1:-1]) / (2 * dx)
        dE_dz = (E_old[1:-1, 1:-1, 2:] - E_old[1:-1, 1:-1, :-2]) / (2 * dx)
        
        grad_coupling = beta_over_alpha * (dchi_dx[1:-1, 1:-1, 1:-1] * dE_dx +
                                           dchi_dy[1:-1, 1:-1, 1:-1] * dE_dy +
                                           dchi_dz[1:-1, 1:-1, 1:-1] * dE_dz)
        
        # New value with SOR
        E_new_interior = laplacian / (denom + 1e-10) - (dx * dx * grad_coupling) / (denom + 1e-10)
        E[1:-1, 1:-1, 1:-1] = omega * E_new_interior + (1 - omega) * E_old[1:-1, 1:-1, 1:-1]
        
        # Boundary conditions: Neumann (zero gradient) at domain edges
        E[0, :, :] = E[1, :, :]
        E[-1, :, :] = E[-2, :, :]
        E[:, 0, :] = E[:, 1, :]
        E[:, -1, :] = E[:, -2, :]
        E[:, :, 0] = E[:, :, 1]
        E[:, :, -1] = E[:, :, -2]
        
        # Check convergence every 10 iterations
        if iteration % 10 == 0:
            residual = float(xp.max(xp.abs(E - E_old)))
            if iteration % 50 == 0:
                print(f"[static-solver] Iteration {iteration}: max residual = {residual:.2e}")
            if residual < tolerance:
                print(f"[static-solver] Converged in {iteration} iterations!")
                break
    
    level.E[:] = E
    level.E_prev[:] = E
    E_min, E_max = float(xp.min(E)), float(xp.max(E))
    E_grad_mag = float(xp.sqrt(xp.mean(dchi_dx**2 + dchi_dy**2 + dchi_dz**2)))
    print(f"[static-solver] Solution complete.")
    print(f"[static-solver]   E range: [{E_min:.4f}, {E_max:.4f}]")
    print(f"[static-solver]   |∇E| typical: {E_grad_mag:.4f}")


# ----------------------------
# Visualization helpers
# ----------------------------
def _to_numpy(arr):
    try:
        # cupy arrays have .get()
        return arr.get()
    except Exception:
        return np.asarray(arr)


def _plot_spheres(ax, bodies: List[Body], colors: Optional[List[str]] = None, n: int = 40, alpha: float = 0.6):
    import numpy as _np
    if colors is None:
        colors = ['royalblue', 'lightgray']
    for idx, b in enumerate(bodies):
        u = _np.linspace(0, 2 * _np.pi, n)
        v = _np.linspace(0, _np.pi, n)
        uu, vv = _np.meshgrid(u, v)
        x = b.radius * _np.cos(uu) * _np.sin(vv) + b.position[0]
        y = b.radius * _np.sin(uu) * _np.sin(vv) + b.position[1]
        z = b.radius * _np.cos(vv) + b.position[2]
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color=colors[idx % len(colors)], alpha=alpha, linewidth=0, antialiased=True)
        # tiny marker at center
        ax.scatter([b.position[0]], [b.position[1]], [b.position[2]], color='k', s=8)


def _render_frame_slice(Lf: LatticeLevel, earth: Body, moon: Body, traj: List[Tuple[float,float,float]], n: int, outdir: str):
    import matplotlib.pyplot as plt
    mid = Lf.shape[2] // 2
    E_np = _to_numpy(Lf.E)
    chi_np = _to_numpy(Lf.chi)
    v = np.percentile(np.abs(E_np[:, :, mid]), 99)
    plt.figure(figsize=(6, 5))
    plt.imshow(E_np[:, :, mid].T, origin='lower', cmap='magma', vmin=-v, vmax=v)
    # overlay chi contours per-body to make structure obvious
    levels = np.linspace(float(chi_np.min()), float(chi_np.max()), 8)
    plt.contour(chi_np[:, :, mid].T, levels=levels, colors='white', linewidths=0.3, alpha=0.3)
    if hasattr(Lf, 'chi_earth'):
        chi_e = _to_numpy(Lf.chi_earth)[:, :, mid]
        le = np.linspace(float(chi_e.min()), float(chi_e.max()), 6)
        plt.contour(chi_e.T, levels=le, colors='cyan', linewidths=0.7, alpha=0.7)
    if hasattr(Lf, 'chi_moon'):
        chi_m = _to_numpy(Lf.chi_moon)[:, :, mid]
        lm = np.linspace(float(chi_m.min()), float(chi_m.max()), 6)
        plt.contour(chi_m.T, levels=lm, colors='lime', linewidths=0.7, alpha=0.7)
    plt.colorbar(shrink=0.75, label='E amplitude')
    ex, ey, ez = earth.position
    mx, my, mz = moon.position
    ei, ej, ek = Lf.physical_to_grid(ex, ey, Lf.origin[2] + mid * Lf.dx)
    mi, mj, mk = Lf.physical_to_grid(mx, my, Lf.origin[2] + mid * Lf.dx)
    if len(traj) > 1:
        xs = [Lf.physical_to_grid(px, py, Lf.origin[2] + mid * Lf.dx)[0] for (px, py, _pz) in traj]
        ys = [Lf.physical_to_grid(px, py, Lf.origin[2] + mid * Lf.dx)[1] for (px, py, _pz) in traj]
        plt.plot(xs, ys, color='cyan', linewidth=1.0, alpha=0.9, label='trajectory')
    plt.scatter([ei], [ej], s=100, c='yellow', marker='*', edgecolors='k', linewidths=0.6, label='Earth')
    plt.scatter([mi], [mj], s=50, c='white', marker='o', edgecolors='k', linewidths=0.6, label='Moon')
    plt.title(f"LFM slice — step {n+1}")
    plt.legend(loc='upper right', framealpha=0.6)
    plt.tight_layout()
    png = os.path.join(outdir, f"frame_{n+1:05d}.png")
    plt.savefig(png, dpi=120)
    plt.close()


def _render_frame_spheres(
    lattice: MultiScaleLattice,
    earth: Body,
    moon: Body,
    traj: List[Tuple[float, float, float]],
    n: int,
    outdir: str,
    elev: float = 25.0,
    azim_step: float = 2.0,
    camera_mode: str = "orbit",
    azim0: float = 0.0,
    overlay_chi: bool = True,
    overlay_stride: int = 2,
    sphere_res: int = 48,
    show_velocity: bool = False,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D
    import numpy as np
    # Use finest level for bounds
    Lf = lattice.levels[-1]
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111, projection='3d')
    _plot_spheres(ax, [earth, moon], colors=['royalblue', 'silver'], n=max(12, int(sphere_res)), alpha=0.7)
    
    # Velocity vector arrow
    if show_velocity and moon.velocity is not None:
        vx, vy, vz = moon.velocity
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        if v_mag > 0.01:  # Only show if moving
            # Scale arrow length
            scale = 10.0  # Make arrow visible
            ax.quiver(moon.position[0], moon.position[1], moon.position[2],
                     vx, vy, vz, color='red', arrow_length_ratio=0.3, linewidth=2,
                     length=scale, normalize=True, alpha=0.8)
    
    # trajectory tail
    if len(traj) > 1:
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        zs = [p[2] for p in traj]
        ax.plot(xs, ys, zs, color='cyan', linewidth=1.4, alpha=0.9)
    # bounds from lattice
    x0, y0, z0 = Lf.origin
    L = (Lf.shape[0]-1) * Lf.dx
    ax.set_xlim([x0, x0 + L])
    ax.set_ylim([y0, y0 + L])
    ax.set_zlim([z0, z0 + L])
    ax.set_box_aspect([1, 1, 1], zoom=0.95)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if camera_mode == 'orbit':
        azim = (azim0 + n * azim_step) % 360
    else:
        azim = azim0
    ax.view_init(elev=elev, azim=azim)
    
    # Overlay chi mid-plane as smooth filled contours
    if overlay_chi and hasattr(Lf, 'chi_earth') and hasattr(Lf, 'chi_moon'):
        import numpy as _np
        from matplotlib import cm
        mid = Lf.shape[2] // 2
        # Convert to numpy - use full resolution for smooth contours
        chi_e_full = _to_numpy(Lf.chi_earth)[:, :, mid]
        chi_m_full = _to_numpy(Lf.chi_moon)[:, :, mid]
        
        # Build coordinate grids for the plane z=const
        xs = _np.arange(Lf.shape[0]) * Lf.dx + Lf.origin[0]
        ys = _np.arange(Lf.shape[1]) * Lf.dx + Lf.origin[1]
        Xp, Yp = _np.meshgrid(xs, ys, indexing='ij')
        z_plane = Lf.origin[2] + mid * Lf.dx
        
        # Use logarithmic spacing to make faint distant chi visible
        # Subtract background to show relative enhancement
        chi_bg = float(chi_e_full.min())
        chi_e_rel = chi_e_full - chi_bg
        chi_m_rel = chi_m_full - chi_bg
        
        # Log scale for levels (makes distant fields visible)
        chi_e_max = float(chi_e_rel.max())
        chi_m_max = float(chi_m_rel.max())
        
        # Draw Earth chi as blue filled contours (reversed colormap = dark near center)
        if chi_e_max > 1e-4:
            lev_e = _np.logspace(_np.log10(max(1e-4, chi_e_max/1000)), _np.log10(chi_e_max), 25)
            ax.contourf(Xp, Yp, chi_e_rel, levels=lev_e, zdir='z', offset=z_plane, 
                       cmap=cm.Blues_r, alpha=0.6, antialiased=True)
        
        # Draw Moon chi as green filled contours (only if moon has chi field)
        if chi_m_max > 1e-4:
            lev_m = _np.logspace(_np.log10(max(1e-4, chi_m_max/1000)), _np.log10(chi_m_max), 25)
            ax.contourf(Xp, Yp, chi_m_rel, levels=lev_m, zdir='z', offset=z_plane, 
                       cmap=cm.Greens_r, alpha=0.6, antialiased=True)
    ax.set_title(f"Gravity flyby — step {n+1}")
    plt.tight_layout()
    png = os.path.join(outdir, f"frame_{n+1:05d}.png")
    plt.savefig(png, dpi=140)
    plt.close()


def _try_make_gif(outdir: str, pattern: str, gif_path: str, fps: int = 15):
    try:
        import imageio.v2 as imageio
        frames = []
        # natural order of generated frames
        files = sorted([f for f in os.listdir(outdir) if f.startswith(pattern) and f.endswith('.png')])
        for f in files:
            frames.append(imageio.imread(os.path.join(outdir, f)))
        if frames:
            imageio.mimsave(gif_path, frames, duration=1.0/max(1,fps))
            print(f"[gif] wrote {gif_path} ({len(frames)} frames @ {fps} fps)")
        else:
            print("[gif] no frames found; skipping gif")
    except Exception as e:
        print(f"[gif] skip ({e}) — install imageio if you want automatic GIFs")


def run(args):
    print("[init] Starting LFM Gravity Playground...", flush=True)
    
    # Apply high-level presets before constructing bodies
    if getattr(args, 'preset', '') == 'emergent-gravity':
        print("[init] Applying 'emergent-gravity' preset (E field coupling, longer evolution)", flush=True)
        # Single-level for speed
        if args.N0 == 96:
            args.N0 = 64
        args.single_level = True
        # Smaller timestep for stability with stronger E fields
        if args.dt == 0.2:
            args.dt = 0.1
        # Smaller bodies for clearer field structure
        if args.radius_scale == 1.0:
            args.radius_scale = 0.25
        # Position moon off-axis for flyby
        if args.moon_start_x == 60.0:
            args.moon_start_x = 90.0
        if args.moon_start_y == 0.0:
            args.moon_start_y = 25.0
        # Initial velocity for tangential pass
        if (args.vx0, args.vy0) in [(0.0, 0.4), (0.0, 0.0)]:
            args.vx0 = -0.8
            args.vy0 = -0.1
        # k_acc for E^2 gradients (different scale than chi)
        if args.k_acc == 0.01:
            args.k_acc = 1.5  # E^2 gradients are much smaller
        # Longer run to allow E field structure to develop
        if args.steps == 200:
            args.steps = 600
        if args.save_every == 5:
            args.save_every = 3
        # Fixed camera for clarity
        if args.azim_step == 2.0:
            args.azim_step = 0.0
    
    elif getattr(args, 'preset', '') == 'static-gravity':
        print("[init] Applying 'static-gravity' preset (REALISTIC: static E field solution)", flush=True)
        # Single-level for speed
        if args.N0 == 96:
            args.N0 = 64
        args.single_level = True
        # Use static solver instead of wave excitations
        args.use_static_solver = True
        # Smaller bodies for clearer field structure
        if args.radius_scale == 1.0:
            args.radius_scale = 0.25
        # Position moon off-axis for flyby
        if args.moon_start_x == 60.0:
            args.moon_start_x = 90.0
        if args.moon_start_y == 0.0:
            args.moon_start_y = 25.0
        # Initial velocity for tangential pass
        if (args.vx0, args.vy0) in [(0.0, 0.4), (0.0, 0.0)]:
            args.vx0 = -0.8
            args.vy0 = -0.1
        # k_acc for E^2 gradients from static field
        if args.k_acc == 0.01:
            args.k_acc = 0.5  # Static fields have different gradient scale
        # Moderate run length (static field doesn't evolve)
        if args.steps == 200:
            args.steps = 400
        if args.save_every == 5:
            args.save_every = 2
        # Fixed camera for clarity
        if args.azim_step == 2.0:
            args.azim_step = 0.0
    
    elif getattr(args, 'preset', '') == 'fast-test':
        print("[init] Applying 'fast-test' preset (single-level, small lattice for quick iteration)", flush=True)
        # Much smaller lattice AND single level
        if args.N0 == 96:
            args.N0 = 64
        args.single_level = True  # KEY: only use one level
        # Smaller bodies
        if args.radius_scale == 1.0:
            args.radius_scale = 0.3
        # Flyby setup
        # Flyby setup: start from side, fast tangential velocity, strong pull
        if args.moon_start_x == 60.0:
            args.moon_start_x = 100.0
        if args.moon_start_y == 0.0:
            args.moon_start_y = 20.0
        if (args.vx0, args.vy0) in [(0.0, 0.4), (0.0, 0.0)]:
            args.vx0 = -1.0
            args.vy0 = -0.15
        if args.k_acc == 0.01:
            args.k_acc = 0.15
        if args.save_every == 5:
            args.save_every = 1
        if args.steps == 200:
            args.steps = 300
        # Slower camera rotation for cinematic effect
        if args.azim_step == 2.0:
            args.azim_step = 1.2
    
    elif getattr(args, 'preset', '') == 'small-flyby':
        print("[init] Applying 'small-flyby' preset", flush=True)
        # Keep lattice size; shrink radii and set an inward velocity to fall toward center
        if args.radius_scale == 1.0:
            args.radius_scale = 0.35
        # Start farther out to make approach visible
        if args.moon_start_x in (60.0, 90.0):
            args.moon_start_x = 50.0
        # Aim toward origin along -x, let gravity curve the path
        if (args.vx0, args.vy0) in [(0.0, 0.0), (0.0, 0.4), (-0.25, 0.0)]:
            args.vx0 = -0.6
            args.vy0 = 0.0
        # Slightly stronger pull for narrower wells
        if args.k_acc in (0.01, 0.012):
            args.k_acc = 0.02
        # More frames and longer run
        if args.save_every == 5:
            args.save_every = 1
        if args.steps == 200:
            args.steps = 400
    
    elif getattr(args, 'preset', '') == 'realistic-orbit':
        print("[init] Applying 'realistic-orbit' preset (large lattice, realistic distance scale)", flush=True)
        # LARGE lattice to fit realistic orbital distances
        if args.N0 == 96:
            args.N0 = 128  # Bigger lattice: ±256 units range
        if args.dx0 == 4.0:
            args.dx0 = 4.0  # Keep spacing same
        args.single_level = True  # Single level for speed
        args.mobile = True  # CRITICAL: Enable moon motion!
        
        # KEY: Wider Gaussian for long-range gravity + STRONGER chi amplitude!
        args.chi_width_multiplier = 12.0  # 12× Earth radius instead of 4×
        
        # CRITICAL: Realistic chi amplitudes with correct Earth/Moon mass ratio
        # Earth/Moon mass ratio = 81:1, so Moon chi = Earth chi / 81
        if args.earth_chi == 0.35:
            args.earth_chi = 50.0  # Strong field for orbital motion
        if args.moon_chi == 0.18:
            args.moon_chi = 50.0 / 81.0  # Realistic mass ratio: 0.617
        if args.background_chi == 0.10:
            args.background_chi = 0.10  # Keep background
        
        # Realistic body sizes
        if args.radius_scale == 1.0:
            args.radius_scale = 0.3  # Earth r=4.8, Moon r=1.3
        
        # Orbital setup: ~20 Earth radii distance (1/3 of real 60 ER)
        # With radius_scale=0.3: Earth r=4.8, so 20 ER = 96 units
        if args.moon_start_x == 60.0:
            args.moon_start_x = 96.0  # 20 Earth radii
        if args.moon_start_y == 0.0:
            args.moon_start_y = 0.0
        
        # Orbital velocity calculated for width=12×radius, chi=50.0, at 96 units
        # Exact circular orbit: v = 1.858562 units/step, period = 324.5 steps
        if args.k_acc == 0.01:
            args.k_acc = 0.20  # Standard acceleration
        if (args.vx0, args.vy0) in [(0.0, 0.0), (0.0, 0.4)]:
            args.vx0 = 0.0
            args.vy0 = 1.858562  # EXACT circular orbit velocity
        
        # Run for full orbit based on measured angular velocity (0.2447°/step)
        if args.steps == 200:
            args.steps = 1472  # Exactly 360° based on measurement
        if args.save_every == 5:
            args.save_every = 5  # Save every 5 steps = ~295 frames for full orbit
        
        # Generate GIF animation
        args.gif = True
        if args.fps == 30:
            args.fps = 20  # Slower playback for orbit viewing
        
        # Fixed camera for clear orbit view
        args.camera = 'fixed'
        args.view = 'top'  # Top-down view to see orbit clearly

    # Optional radius scaling without changing lattice scale
    if getattr(args, 'radius_scale', 1.0) != 1.0:
        args.earth_radius *= args.radius_scale
        args.moon_radius *= args.radius_scale

    earth = Body("Earth", radius=args.earth_radius, chi_center=args.earth_chi, chi_background=args.background_chi,
                 position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), mobile=False)
    moon_v0 = (args.vx0, args.vy0, args.vz0)
    moon = Body("Moon", radius=args.moon_radius, chi_center=args.moon_chi, chi_background=args.background_chi,
                position=(args.moon_start_x, args.moon_start_y, args.moon_start_z), velocity=moon_v0, mobile=args.mobile)

    print(f"[init] Bodies configured: Earth at origin (r={earth.radius:.1f}), Moon at {moon.position} (r={moon.radius:.1f})", flush=True)

    # Select device: auto = use GPU if available
    if args.device == 'gpu' and not _HAS_CUPY:
        print("[warn] --device=gpu requested but CuPy not available; falling back to CPU")
    use_gpu = (_HAS_CUPY if args.device == 'auto' else (args.device == 'gpu' and _HAS_CUPY))
    
    print(f"[init] Building multi-scale lattice (N0={args.N0}, N1={args.N1}, N2={args.N2})...", flush=True)
    t_lattice_start = time.time()
    single_level = getattr(args, 'single_level', False)
    if single_level:
        print(f"[init] Using SINGLE LEVEL mode (fast)", flush=True)
    chi_width_mult = getattr(args, 'chi_width_multiplier', 4.0)  # Default 4×, realistic-orbit uses 12×
    print(f"[init] Chi width multiplier: {chi_width_mult}× body radius", flush=True)
    lattice = build_levels(N0=args.N0, dx0=args.dx0, N1=args.N1, dx1=args.dx1, N2=args.N2,
                           dx2=args.dx2 if args.dx2 > 0 else None, use_gpu=use_gpu,
                           single_level=single_level,
                           earth=earth, moon=moon, chi_width_multiplier=chi_width_mult)
    t_lattice = time.time() - t_lattice_start
    print(f"[init] Lattice built in {t_lattice:.2f}s", flush=True)
    
    # DIAGNOSTIC: Check chi field at moon's starting position
    L0 = lattice.levels[0]
    i_moon, j_moon, k_moon = L0.physical_to_grid(*moon.position)
    i_moon = max(1, min(L0.shape[0]-2, i_moon))
    j_moon = max(1, min(L0.shape[1]-2, j_moon))
    k_moon = max(1, min(L0.shape[2]-2, k_moon))
    chi_at_moon = float(L0.chi[i_moon, j_moon, k_moon])
    
    # Calculate expected chi value at moon position
    import numpy as np
    dx, dy, dz = moon.position[0] - earth.position[0], moon.position[1] - earth.position[1], moon.position[2] - earth.position[2]
    r_moon = np.sqrt(dx**2 + dy**2 + dz**2)
    width_expected = chi_width_mult * earth.radius
    chi_expected = earth.chi_center * np.exp(-(r_moon / width_expected)**2)
    
    print(f"[init] Chi field at moon start (r={r_moon:.1f}):", flush=True)
    print(f"       Actual chi: {chi_at_moon:.6f}", flush=True)
    print(f"       Expected (width={width_expected:.1f}): {chi_expected:.6f}", flush=True)
    print(f"       Ratio: {chi_at_moon/chi_expected:.4f}", flush=True)

    # Output directory per viz mode
    sub = "PlanetFlyby3D" if args.viz == 'sphere3d' else "PlanetInfallSlice"
    outdir = os.path.join("results", "Gravity", sub)
    os.makedirs(outdir, exist_ok=True)

    params = {
        "alpha": args.alpha, "beta": args.beta,
        "dt": args.dt, "dx": lattice.levels[0].dx,  # dt for coarse level
        "chi": lattice.levels[0].chi,  # unused directly; per-level handled in AMR
        "boundary": "absorbing", "absorb_width": 2, "absorb_factor": 1.0,
        "debug": {"enable_diagnostics": True, "diagnostics_path": os.path.join(outdir, "diagnostics_core.csv"),
                   "quiet_run": True, "energy_tol": 1e-3},
    }

    # E FIELD INITIALIZATION
    # Two modes: static solver (physically realistic) or wave excitations (dynamic)
    use_static = getattr(args, 'use_static_solver', False)
    
    if use_static:
        print("[init] Using STATIC E field solver (physically realistic geometry)")
        for L in lattice.levels:
            solve_static_E_field(L, alpha=args.alpha, beta=args.beta, 
                               max_iterations=1000, tolerance=1e-5)
    else:
        print("[init] Using wave excitations (dynamic E field evolution)")
        # Structured E field initialization for emergent gravity:
        # Create dipole-like excitations in each chi well to allow E field structure to develop
        for L in lattice.levels:
            xp = _xp_for(L.E)
            N = L.shape[0]
            x = L.origin[0] + xp.arange(N) * L.dx
            y = L.origin[1] + xp.arange(N) * L.dx
            z = L.origin[2] + xp.arange(N) * L.dx
            X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
            
            # Small random noise as baseline
            L.E[:] = (xp.random.random(L.E.shape) - 0.5) * 1e-4
            
            # Add Gaussian excitations at each body position (scalar field)
            # Earth excitation (stronger)
            dx_e = X - earth.position[0]
            dy_e = Y - earth.position[1]
            dz_e = Z - earth.position[2]
            r_e = xp.sqrt(dx_e**2 + dy_e**2 + dz_e**2 + 1e-6)
            excite_e = 0.05 * xp.exp(-r_e**2 / (2*earth.radius**2))
            L.E[:] = L.E + excite_e
            
            # Moon excitation (weaker)
            dx_m = X - moon.position[0]
            dy_m = Y - moon.position[1]
            dz_m = Z - moon.position[2]
            r_m = xp.sqrt(dx_m**2 + dy_m**2 + dz_m**2 + 1e-6)
            excite_m = 0.02 * xp.exp(-r_m**2 / (2*moon.radius**2))
            L.E[:] = L.E + excite_m
            
            L.E_prev[:] = L.E

    # Track trajectory for overlay
    traj = []

    # Apply camera presets if requested
    if hasattr(args, 'view') and args.view != 'custom':
        if args.view == 'top':
            args.elev, args.azim = 90.0, 0.0
        elif args.view == 'side':
            args.elev, args.azim = 0.0, 0.0
        elif args.view == 'isometric':
            args.elev, args.azim = 35.0, 45.0

    # Optional initial frame 0
    if args.save_every > 0:
        print(f"[viz] Rendering initial frame (step 0)")
        if args.viz == 'sphere3d':
            _render_frame_spheres(lattice, earth, moon, traj, n=0, outdir=outdir, elev=args.elev, azim_step=args.azim_step, camera_mode=getattr(args,'camera','orbit'), azim0=getattr(args,'azim',0.0), overlay_chi=getattr(args,'overlay_chi',True), overlay_stride=getattr(args,'overlay_stride',2), sphere_res=getattr(args,'sphere_res',48), show_velocity=getattr(args,'show_velocity',False))
        else:
            _render_frame_slice(lattice.levels[-1], earth, moon, traj, n=0, outdir=outdir)
        print(f"[viz] Initial frame saved to {outdir}")

    # Run steps with optional moon motion (update χ each coarse step)
    print(f"[run] Starting simulation: {args.steps} steps, saving every {args.save_every} steps")
    print(f"[run] Using device: {'GPU' if use_gpu else 'CPU'}")
    print(f"[run] Lattice levels: {len(lattice.levels)}, shapes: {[L.shape for L in lattice.levels]}", flush=True)
    
    t_start = time.time()
    t_last_report = t_start
    step_times = []
    viz_times = []
    
    for n in range(args.steps):
        if n == 0:
            print(f"[run] Starting first step (this may take a moment for GPU warmup)...", flush=True)
        t_step_start = time.time()
        
        if moon.mobile and (n % args.move_every == 0):
            # Choose model: kinematic ∇χ for fast-test/small-flyby, emergent ∇E² for emergent-gravity
            use_kinematic = args.preset in ['fast-test', 'small-flyby', 'realistic-orbit', '']  # Default to kinematic
            # Use Level 0 gradient to estimate acceleration
            enable_collision = getattr(args, 'enable_collision', True)  # Enable by default
            update_moon_position(lattice.levels[0], moon, k_acc=args.k_acc, dt=args.dt, 
                               use_kinematic=use_kinematic, earth=earth, enable_collision=enable_collision)
            # Refresh χ on all levels to reflect new moon position
            for L in lattice.levels:
                refresh_chi_for_bodies(L, earth, moon, chi_width_mult)
        traj.append(moon.position)
        
        # Debug: Log moon position at key intervals
        if n in [0, 50, 100, 150, 200, 250, 300]:
            moon_dist = math.sqrt(sum((p - 0.0)**2 for p in moon.position))
            print(f"[track] Step {n}: Moon pos=({moon.position[0]:.2f}, {moon.position[1]:.2f}, {moon.position[2]:.2f}), dist={moon_dist:.2f}", flush=True)
        lattice.step(dt=args.dt, params={"alpha": args.alpha, "beta": args.beta})
        
        step_time = time.time() - t_step_start
        step_times.append(step_time)
        
        # Progress indicator every 10% or if > 5 sec since last report
        t_now = time.time()
        report_interval = (n + 1) % max(1, args.steps // 10) == 0
        time_threshold = (t_now - t_last_report) > 5.0
        
        if report_interval or time_threshold or (n + 1) == args.steps:
            pct = 100 * (n + 1) / args.steps
            elapsed = t_now - t_start
            avg_step = np.mean(step_times[-20:]) if step_times else 0
            eta = avg_step * (args.steps - n - 1)
            print(f"[progress] {n+1}/{args.steps} ({pct:.0f}%) | elapsed: {elapsed:.1f}s | avg step: {avg_step*1000:.1f}ms | ETA: {eta:.1f}s", flush=True)
            t_last_report = t_now
        
        if (n + 1) % max(1, args.save_every) == 0:
            t_viz_start = time.time()
            try:
                if args.viz == 'sphere3d':
                    _render_frame_spheres(lattice, earth, moon, traj, n=n+1, outdir=outdir, elev=args.elev, azim_step=args.azim_step, camera_mode=getattr(args,'camera','orbit'), azim0=getattr(args,'azim',0.0), overlay_chi=getattr(args,'overlay_chi',True), overlay_stride=getattr(args,'overlay_stride',2), sphere_res=getattr(args,'sphere_res',48), show_velocity=getattr(args,'show_velocity',False))
                else:
                    _render_frame_slice(lattice.levels[-1], earth, moon, traj, n=n+1, outdir=outdir)
                viz_time = time.time() - t_viz_start
                viz_times.append(viz_time)
                if viz_time > 1.0:
                    print(f"[viz] Frame {n+1} took {viz_time:.2f}s", flush=True)
            except Exception as e:
                print(f"[viz] Error at step {n+1}: {e}")

    # Final report
    t_total = time.time() - t_start
    print(f"\n[done] Simulation complete in {t_total:.1f}s!")
    print(f"  Total steps: {args.steps}")
    print(f"  Average step time: {np.mean(step_times)*1000:.1f}ms")
    if viz_times:
        print(f"  Average viz time: {np.mean(viz_times):.2f}s")
        print(f"  Total viz time: {sum(viz_times):.1f}s ({100*sum(viz_times)/t_total:.0f}% of runtime)")
    print(f"  Frames saved: {len([f for f in os.listdir(outdir) if f.startswith('frame_')])}")
    print(f"  Output directory: {outdir}")
    print(f"  Final moon position: {moon.position}")
    moon_dist = math.sqrt(sum((p - 0.0)**2 for p in moon.position))
    print(f"  Final distance from Earth: {moon_dist:.2f}")
    
    if args.gif:
        print(f"\n[gif] Creating animated GIF...")
        pattern = "frame_"
        gif_path = os.path.join(outdir, args.gif_name if args.gif_name else ("flyby.gif" if args.viz == 'sphere3d' else "slice.gif"))
        _try_make_gif(outdir, pattern=pattern, gif_path=gif_path, fps=args.fps)


def main():
    p = argparse.ArgumentParser(description="LFM Gravity Playground — Earth–Moon two-body toy")
    # Resolution & hierarchy
    p.add_argument('--N0', type=int, default=96)
    p.add_argument('--dx0', type=float, default=4.0, help='Coarse dx (lattice units)')
    p.add_argument('--N1', type=int, default=192)
    p.add_argument('--dx1', type=float, default=1.0)
    p.add_argument('--N2', type=int, default=192)
    p.add_argument('--dx2', type=float, default=-1.0, help='If <=0, set to dx1/4')

    # Physics
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--dt', type=float, default=0.2)

    # Bodies
    p.add_argument('--earth-radius', dest='earth_radius', type=float, default=16.0)
    p.add_argument('--earth-chi', dest='earth_chi', type=float, default=0.35)
    p.add_argument('--moon-radius', dest='moon_radius', type=float, default=4.5)
    p.add_argument('--moon-chi', dest='moon_chi', type=float, default=0.18)
    p.add_argument('--background-chi', dest='background_chi', type=float, default=0.10)
    p.add_argument('--moon-start-x', dest='moon_start_x', type=float, default=60.0, help='Initial x position of moon')
    p.add_argument('--moon-start-y', dest='moon_start_y', type=float, default=0.0, help='Initial y position of moon')
    p.add_argument('--moon-start-z', dest='moon_start_z', type=float, default=0.0, help='Initial z position of moon')
    p.add_argument('--mobile', action='store_true', help='Enable toy motion for moon (update χ center)')
    p.add_argument('--k-acc', dest='k_acc', type=float, default=0.01, help='Toy acceleration scaling')
    p.add_argument('--vx0', type=float, default=0.0, help='Initial vx for moon')
    p.add_argument('--vy0', type=float, default=0.4, help='Initial vy for moon (default yields a pass-by)')
    p.add_argument('--vz0', type=float, default=0.0, help='Initial vz for moon')
    p.add_argument('--enable-collision', dest='enable_collision', action='store_true', default=True, help='Enable collision detection (default: enabled)')
    p.add_argument('--no-collision', dest='enable_collision', action='store_false', help='Disable collision detection (moon passes through Earth)')

    # Visualization
    p.add_argument('--viz', choices=['slice','sphere3d'], default='sphere3d', help='slice: 2D central slice; sphere3d: 3D spheres and trajectory')
    p.add_argument('--elev', type=float, default=25.0, help='3D view elevation (degrees)')
    p.add_argument('--azim-step', dest='azim_step', type=float, default=2.0, help='Azimuthal degrees to rotate camera per saved frame')
    p.add_argument('--azim', type=float, default=0.0, help='Azimuth angle (degrees); used as base or fixed value depending on camera mode')
    p.add_argument('--camera', choices=['orbit','fixed'], default='orbit', help='Camera mode: orbit rotates around z; fixed keeps a constant view')
    p.add_argument('--view', choices=['custom','top','side','isometric'], default='custom', help='Camera presets: set elev/azim for top/side/isometric views')
    p.add_argument('--overlay-chi', dest='overlay_chi', action='store_true', help='Overlay Earth and Moon chi mid-plane in 3D view')
    p.add_argument('--no-overlay-chi', dest='overlay_chi', action='store_false')
    p.set_defaults(overlay_chi=True)
    p.add_argument('--overlay-stride', dest='overlay_stride', type=int, default=2, help='Subsampling stride for chi overlay to speed up plotting (>=1)')
    p.add_argument('--sphere-res', dest='sphere_res', type=int, default=48, help='Sphere mesh resolution (higher is smoother and slower)')
    p.add_argument('--show-velocity', dest='show_velocity', action='store_true', help='Show velocity vector on moon')
    p.add_argument('--gif', action='store_true', help='Stitch saved frames into an animated GIF (requires imageio)')
    p.add_argument('--fps', type=int, default=15, help='Frames per second for GIF')
    p.add_argument('--gif-name', dest='gif_name', type=str, default='', help='Output GIF file name (optional)')
    p.add_argument('--radius-scale', dest='radius_scale', type=float, default=1.0, help='Multiply Earth/Moon radii by this factor (lattice scale unchanged)')
    p.add_argument('--preset', choices=['small-flyby', 'fast-test', 'emergent-gravity', 'static-gravity', 'realistic-orbit'], default='', help='Presets: fast-test (kinematic chi), static-gravity (REALISTIC static E field), emergent-gravity (dynamic E field), small-flyby (full lattice), realistic-orbit (large lattice for orbital motion at ~1/3 real scale)')

    # Runtime
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--move-every', dest='move_every', type=int, default=1)
    p.add_argument('--save-every', dest='save_every', type=int, default=5)
    p.add_argument('--device', choices=['auto','gpu','cpu'], default='auto', help='Device selection: auto uses GPU if available')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
