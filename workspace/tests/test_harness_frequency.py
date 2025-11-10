# -*- coding: utf-8 -*-
"""
Phase 2 Tests: Frequency Estimation Methods

Tests estimate_omega_fft and frequency measurement utilities.
"""
import pytest
import sys
from pathlib import Path
import numpy as np

# Add workspace/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'workspace' / 'src'))


class TestEstimateOmegaFFT:
    """Test estimate_omega_fft method from BaseTierHarness."""
    
    def test_detects_single_frequency_sine_wave(self, minimal_harness):
        """Should correctly identify frequency of pure sine wave."""
        # Create pure sine wave: sin(ωt)
        t = np.linspace(0, 10 * np.pi, 1000)
        dt = t[1] - t[0]
        omega_expected = 2.0
        signal = np.sin(omega_expected * t)
        
        omega_measured = minimal_harness.estimate_omega_fft(signal, dt)
        
        # Should be within 5% of expected
        relative_error = abs(omega_measured - omega_expected) / omega_expected
        assert relative_error < 0.05, (
            f"FFT frequency error {relative_error:.2%} exceeds 5%\n"
            f"Expected: {omega_expected}, Measured: {omega_measured}"
        )
    
    def test_detects_high_frequency(self, minimal_harness):
        """Should detect higher frequency waves."""
        t = np.linspace(0, 10 * np.pi, 2000)
        dt = t[1] - t[0]
        omega_expected = 10.0
        signal = np.sin(omega_expected * t)
        
        omega_measured = minimal_harness.estimate_omega_fft(signal, dt)
        
        relative_error = abs(omega_measured - omega_expected) / omega_expected
        assert relative_error < 0.05
    
    def test_handles_complex_signals(self, minimal_harness):
        """Should extract dominant frequency from multi-component signal."""
        t = np.linspace(0, 20 * np.pi, 2000)
        dt = t[1] - t[0]
        
        # Dominant frequency at 3.0, weaker at 7.0
        omega_dominant = 3.0
        signal = 2.0 * np.sin(omega_dominant * t) + 0.5 * np.sin(7.0 * t)
        
        omega_measured = minimal_harness.estimate_omega_fft(signal, dt)
        
        # Should identify the dominant frequency
        relative_error = abs(omega_measured - omega_dominant) / omega_dominant
        assert relative_error < 0.10, (
            f"Dominant frequency detection error {relative_error:.2%}\n"
            f"Expected: {omega_dominant}, Measured: {omega_measured}"
        )
    
    def test_returns_positive_frequency(self, minimal_harness):
        """Should always return positive frequency."""
        t = np.linspace(0, 10 * np.pi, 1000)
        dt = t[1] - t[0]
        signal = np.sin(2.0 * t)
        
        omega = minimal_harness.estimate_omega_fft(signal, dt)
        
        assert omega > 0, f"Frequency should be positive, got {omega}"
    
    def test_handles_1d_array(self, minimal_harness):
        """Should work with 1D time series."""
        t = np.linspace(0, 10 * np.pi, 1000)
        dt = t[1] - t[0]
        signal = np.sin(2.0 * t)
        
        omega = minimal_harness.estimate_omega_fft(signal, dt)
        
        assert isinstance(omega, (float, np.floating))
        assert omega > 0
    
    def test_handles_2d_array(self, minimal_harness):
        """Should extract frequency from 2D spatial field."""
        # Create 2D wave with time evolution
        Nx, Ny, Nt = 32, 32, 200
        t = np.linspace(0, 10 * np.pi, Nt)
        dt = t[1] - t[0]
        omega_expected = 2.0
        
        # 2D standing wave oscillating in time
        signal_2d = np.zeros((Ny, Nx, Nt))
        for i_t in range(Nt):
            signal_2d[:, :, i_t] = np.sin(omega_expected * t[i_t])
        
        # Extract central point time series
        signal = signal_2d[Ny//2, Nx//2, :]
        omega_measured = minimal_harness.estimate_omega_fft(signal, dt)
        
        relative_error = abs(omega_measured - omega_expected) / omega_expected
        assert relative_error < 0.05
    
    def test_graceful_with_constant_signal(self, minimal_harness):
        """Should handle constant (zero frequency) signal gracefully."""
        signal = np.ones(1000)
        dt = 0.01
        
        omega = minimal_harness.estimate_omega_fft(signal, dt)
        
        # Should return small frequency or zero
        assert omega >= 0
        assert omega < 1.0  # Should be near zero for constant signal
