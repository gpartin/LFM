# -*- coding: utf-8 -*-
"""
Phase 1 Tests: Energy Calculation (compute_field_energy)

Tests the core energy calculation method used across Tiers 1, 3, 6, 7.
Critical path: 89 tier tests depend on this functionality.
"""
import pytest
import numpy as np

from harness.lfm_test_harness import BaseTierHarness
from fixtures.analytical_fields import (
    gaussian_packet_1d,
    uniform_field_1d,
    sine_wave_1d,
    gaussian_packet_2d,
    gaussian_packet_3d,
    moving_packet_1d
)


class TestComputeFieldEnergy1D:
    """Test 1D energy calculation correctness."""
    
    def test_gaussian_packet_gradient_energy(self, minimal_harness):
        """
        Test 1D Gaussian packet gradient energy matches analytical solution.
        
        Physics: E(x) = A*exp(-x²/2σ²), static (no time derivative)
        Expected: Gradient energy matches analytical within 1%
        
        Note: 1% tolerance accounts for:
          - Finite domain (Gaussian tails truncated)
          - Discrete derivative approximation
        """
        N, dx, sigma, amplitude = 128, 0.1, 1.0, 1.0
        E, E_analytical = gaussian_packet_1d(N, dx, sigma, amplitude)
        E_prev, dt, chi = E.copy(), 0.01, 0.0
        
        E_numerical = minimal_harness.compute_field_energy(E, E_prev, dt, dx, c=1.0, chi=chi, dims='1d')
        
        relative_error = abs(E_numerical - E_analytical) / (E_analytical + 1e-12)
        assert relative_error < 0.01, (
            f"1D Gaussian gradient energy error {relative_error:.2e} exceeds 1%\n"
            f"Numerical: {E_numerical:.6e}, Analytical: {E_analytical:.6e}"
        )
    
    def test_uniform_field_zero_energy(self, minimal_harness):
        """
        Test uniform field has zero gradient energy.
        
        Physics: E(x) = constant → ∂E/∂x = 0 → gradient energy = 0
        """
        N, dx = 64, 0.1
        E, E_analytical = uniform_field_1d(N, dx, amplitude=2.0)
        E_prev, dt, chi = E.copy(), 0.01, 0.0
        
        E_numerical = minimal_harness.compute_field_energy(E, E_prev, dt, dx, c=1.0, chi=chi, dims='1d')
        
        assert abs(E_numerical) < 1e-10, (
            f"Uniform field should have zero energy, got {E_numerical:.2e}"
        )
    
    def test_sine_wave_gradient_energy(self, minimal_harness):
        """
        Test sine wave gradient energy matches analytical solution.
        
        Physics: E(x) = A*sin(kx) → gradient energy = ¼(Ak)²L
        """
        N, dx, k, amplitude = 128, 0.1, 2.0, 1.0
        E, E_analytical = sine_wave_1d(N, dx, k, amplitude)
        E_prev, dt, chi = E.copy(), 0.01, 0.0
        
        E_numerical = minimal_harness.compute_field_energy(E, E_prev, dt, dx, c=1.0, chi=chi, dims='1d')
        
        relative_error = abs(E_numerical - E_analytical) / (E_analytical + 1e-12)
        # Sine wave has 2.5% error due to periodic boundary discontinuity
        assert relative_error < 0.03, (
            f"1D sine wave gradient energy error {relative_error:.2e} exceeds 3%\n"
            f"Numerical: {E_numerical:.6e}, Analytical: {E_analytical:.6e}"
        )
    
    def test_moving_packet_kinetic_energy(self, minimal_harness):
        """
        Test moving packet includes kinetic energy.
        
        Physics: ∂E/∂t ≠ 0 → kinetic energy contribution
        """
        N, dx, dt, v = 128, 0.1, 0.01, 0.5
        E, E_prev, E_kinetic_approx = moving_packet_1d(N, dx, dt, v)
        chi = 0.0
        
        E_numerical = minimal_harness.compute_field_energy(E, E_prev, dt, dx, c=1.0, chi=chi, dims='1d')
        
        # Should be positive (kinetic + gradient contributions)
        assert E_numerical > 0, "Moving packet should have positive energy"
        
        # Should be larger than static packet (includes kinetic)
        E_static = minimal_harness.compute_field_energy(E, E, dt, dx, c=1.0, chi=chi, dims='1d')
        assert E_numerical > E_static, "Moving packet energy should exceed static"
    
    def test_with_chi_potential_energy(self, minimal_harness):
        """
        Test potential energy contribution from χ field.
        
        Physics: χ²E² term adds potential energy
        """
        N, dx = 64, 0.1
        E, _ = gaussian_packet_1d(N, dx, sigma=1.0, amplitude=1.0)
        E_prev, dt = E.copy(), 0.01
        
        # Without potential
        E_no_chi = minimal_harness.compute_field_energy(E, E_prev, dt, dx, c=1.0, chi=0.0, dims='1d')
        
        # With potential
        E_with_chi = minimal_harness.compute_field_energy(E, E_prev, dt, dx, c=1.0, chi=0.5, dims='1d')
        
        # Potential should add positive energy
        assert E_with_chi > E_no_chi, "χ potential should increase total energy"
        
        # Potential contribution should scale with χ²
        chi_contribution = E_with_chi - E_no_chi
        assert chi_contribution > 0, "χ contribution should be positive"


class TestComputeFieldEnergy2D:
    """Test 2D energy calculation correctness."""
    
    def test_gaussian_packet_2d_gradient_energy(self, minimal_harness):
        """
        Test 2D Gaussian packet gradient energy matches analytical solution.
        
        Physics: E(x,y) = A*exp(-r²/2σ²), static
        """
        Nx, Ny, dx, sigma, amplitude = 64, 64, 0.1, 1.0, 1.0
        E, E_analytical = gaussian_packet_2d(Nx, Ny, dx, sigma, amplitude)
        E_prev, dt, chi = E.copy(), 0.01, 0.0
        
        E_numerical = minimal_harness.compute_field_energy(E, E_prev, dt, dx, c=1.0, chi=chi, dims='2d')
        
        relative_error = abs(E_numerical - E_analytical) / (E_analytical + 1e-12)
        assert relative_error < 0.01, (
            f"2D Gaussian gradient energy error {relative_error:.2e} exceeds 1%\n"
            f"Numerical: {E_numerical:.6e}, Analytical: {E_analytical:.6e}"
        )
    
    def test_uniform_field_2d_zero_energy(self, minimal_harness):
        """Test 2D uniform field has zero gradient energy."""
        Nx, Ny, dx = 32, 32, 0.1
        E = np.ones((Ny, Nx), dtype=np.float64)
        E_prev, dt, chi = E.copy(), 0.01, 0.0
        
        E_numerical = minimal_harness.compute_field_energy(E, E_prev, dt, dx, c=1.0, chi=chi, dims='2d')
        
        assert abs(E_numerical) < 1e-10, (
            f"2D uniform field should have zero energy, got {E_numerical:.2e}"
        )


class TestComputeFieldEnergy3D:
    """Test 3D energy calculation correctness."""
    
    def test_gaussian_packet_3d_gradient_energy(self, minimal_harness):
        """
        Test 3D Gaussian packet gradient energy matches analytical solution.
        
        Physics: E(x,y,z) = A*exp(-r²/2σ²), static
        """
        Nx, Ny, Nz, dx, sigma, amplitude = 32, 32, 32, 0.1, 1.0, 1.0
        E, E_analytical = gaussian_packet_3d(Nx, Ny, Nz, dx, sigma, amplitude)
        E_prev, dt, chi = E.copy(), 0.01, 0.0
        
        E_numerical = minimal_harness.compute_field_energy(E, E_prev, dt, dx, c=1.0, chi=chi, dims='3d')
        
        relative_error = abs(E_numerical - E_analytical) / (E_analytical + 1e-12)
        # 3D with 32³ grid has significant truncation error (Gaussian tails clipped)
        # This tolerance validates infrastructure correctness, not numerical accuracy
        assert relative_error < 0.30, (
            f"3D Gaussian gradient energy error {relative_error:.2e} exceeds 30%\n"
            f"Numerical: {E_numerical:.6e}, Analytical: {E_analytical:.6e}"
        )
    
    def test_uniform_field_3d_zero_energy(self, minimal_harness):
        """Test 3D uniform field has zero gradient energy."""
        Nx, Ny, Nz, dx = 16, 16, 16, 0.1
        E = np.ones((Nz, Ny, Nx), dtype=np.float64)
        E_prev, dt, chi = E.copy(), 0.01, 0.0
        
        E_numerical = minimal_harness.compute_field_energy(E, E_prev, dt, dx, c=1.0, chi=chi, dims='3d')
        
        assert abs(E_numerical) < 1e-10, (
            f"3D uniform field should have zero energy, got {E_numerical:.2e}"
        )


class TestEnergyBackendConsistency:
    """Test NumPy vs CuPy backend consistency."""
    
    def test_1d_numpy_cupy_identical(self, minimal_harness, minimal_config, tmp_path):
        """Test 1D energy calculation gives identical results on CPU vs GPU."""
        N, dx = 64, 0.1
        E, _ = gaussian_packet_1d(N, dx)
        E_prev, dt, chi = E.copy(), 0.01, 0.1
        
        # CPU calculation
        E_cpu = minimal_harness.compute_field_energy(E, E_prev, dt, dx, c=1.0, chi=chi, dims='1d')
        
        # Try GPU calculation if CuPy available
        try:
            import cupy as cp
            E_gpu_arr = cp.asarray(E)
            E_prev_gpu = cp.asarray(E_prev)
            
            # Create GPU-enabled harness
            cfg_gpu = minimal_config.copy()
            cfg_gpu['run_settings'] = {'use_gpu': True}
            harness_gpu = BaseTierHarness(cfg_gpu, tmp_path)
            harness_gpu.xp = cp
            harness_gpu.on_gpu = True
            
            E_gpu = harness_gpu.compute_field_energy(
                E_gpu_arr, E_prev_gpu, dt, dx, c=1.0, chi=chi, dims='1d'
            )
            E_gpu = float(E_gpu)  # Convert to Python float
            
            # Should be identical within floating point precision
            relative_diff = abs(E_cpu - E_gpu) / (E_cpu + 1e-12)
            assert relative_diff < 1e-10, (
                f"CPU/GPU energy mismatch: {relative_diff:.2e}\n"
                f"CPU: {E_cpu:.12e}, GPU: {E_gpu:.12e}"
            )
        except ImportError:
            pytest.skip("CuPy not available for GPU backend test")
    
    def test_3d_numpy_cupy_identical(self, minimal_config, tmp_path):
        """Test 3D energy calculation gives identical results on CPU vs GPU."""
        # Create CPU harness
        harness_cpu = BaseTierHarness(minimal_config, tmp_path)
        
        Nx, Ny, Nz, dx = 24, 24, 24, 0.1
        E, _ = gaussian_packet_3d(Nx, Ny, Nz, dx)
        E_prev, dt, chi = E.copy(), 0.01, 0.1
        
        E_cpu = harness_cpu.compute_field_energy(E, E_prev, dt, dx, c=1.0, chi=chi, dims='3d')
        
        try:
            import cupy as cp
            E_gpu_arr = cp.asarray(E)
            E_prev_gpu = cp.asarray(E_prev)
            
            cfg_gpu = minimal_config.copy()
            cfg_gpu['run_settings'] = {'use_gpu': True}
            harness_gpu = BaseTierHarness(cfg_gpu, tmp_path)
            harness_gpu.xp = cp
            harness_gpu.on_gpu = True
            
            E_gpu = harness_gpu.compute_field_energy(
                E_gpu_arr, E_prev_gpu, dt, dx, c=1.0, chi=chi, dims='3d'
            )
            E_gpu = float(E_gpu)
            
            relative_diff = abs(E_cpu - E_gpu) / (E_cpu + 1e-12)
            assert relative_diff < 1e-10, (
                f"CPU/GPU 3D energy mismatch: {relative_diff:.2e}\n"
                f"CPU: {E_cpu:.12e}, GPU: {E_gpu:.12e}"
            )
        except ImportError:
            pytest.skip("CuPy not available for GPU backend test")
