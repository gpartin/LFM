# -*- coding: utf-8 -*-
"""
Unit Tests for Automatic Metrics Recording
=========================================
Tests the centralized metrics recording system in BaseTierHarness.

Location: workspace/tests/test_metrics_recording.py
Coverage: BaseTierHarness._extract_metrics_for_tracking(), automatic record_run()
"""

import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
import sys
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from harness.lfm_test_harness import BaseTierHarness


class TestMetricsRecording:
    """Test suite for automatic metrics recording in BaseTierHarness."""
    
    def test_extract_metrics_from_result_object(self):
        """Test metrics extraction from fresh test result object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create mock result object
            result = Mock()
            result.passed = True
            result.runtime_sec = 1.23
            result.peak_cpu_percent = 85.0
            result.peak_memory_mb = 450.0
            result.peak_gpu_memory_mb = 2048.0
            
            # Create harness instance
            harness = BaseTierHarness(
                cfg={},
                out_root=output_dir,
                config_name="dummy_config.json"
            )
            
            # Extract metrics
            metrics = harness._extract_metrics_for_tracking(
                "TEST-01",
                result,
                output_dir
            )
            
            # Assertions
            assert metrics is not None
            assert metrics["exit_code"] == 0
            assert metrics["runtime_sec"] == 1.23
            assert metrics["peak_cpu_percent"] == 85.0
            assert metrics["peak_memory_mb"] == 450.0
            assert metrics["peak_gpu_memory_mb"] == 2048.0
            assert "timestamp" in metrics
    
    def test_extract_metrics_from_summary_json(self):
        """Test metrics extraction from cached result (summary.json)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create summary.json
            summary_data = {
                "test_id": "TEST-02",
                "passed": False,
                "runtime_sec": 2.45,
                "peak_cpu_percent": 92.5,
                "peak_memory_mb": 680.0,
                "peak_gpu_memory_mb": 1536.0
            }
            summary_path = output_dir / "summary.json"
            summary_path.write_text(json.dumps(summary_data), encoding='utf-8')
            
            # Create mock result without metrics (simulating cache hit)
            result = Mock()
            result.passed = False
            result.runtime_sec = 0.0  # Will be overridden by summary.json
            
            # Create harness instance
            harness = BaseTierHarness(
                config_name="dummy_config.json",
                out_root=output_dir
            )
            
            # Extract metrics
            metrics = harness._extract_metrics_for_tracking(
                "TEST-02",
                result,
                output_dir
            )
            
            # Assertions - should use summary.json data
            assert metrics is not None
            assert metrics["exit_code"] == 1  # Failed test
            assert metrics["runtime_sec"] == 2.45
            assert metrics["peak_cpu_percent"] == 92.5
            assert metrics["peak_memory_mb"] == 680.0
            assert metrics["peak_gpu_memory_mb"] == 1536.0
    
    def test_extract_metrics_defaults(self):
        """Test that default values are set for missing fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create minimal result object
            result = Mock()
            result.passed = True
            result.runtime_sec = 1.0
            # No peak_* attributes
            
            # Create harness instance
            harness = BaseTierHarness(
                config_name="dummy_config.json",
                out_root=output_dir
            )
            
            # Extract metrics
            metrics = harness._extract_metrics_for_tracking(
                "TEST-03",
                result,
                output_dir
            )
            
            # Assertions - should have defaults
            assert metrics is not None
            assert metrics["exit_code"] == 0
            assert metrics["runtime_sec"] == 1.0
            assert metrics["peak_cpu_percent"] == 100.0  # Default
            assert metrics["peak_memory_mb"] == 500.0    # Default
            assert metrics["peak_gpu_memory_mb"] == 0.0   # Default
    
    def test_extract_metrics_returns_none_on_missing_required(self):
        """Test that None is returned when required fields are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create result with no useful data
            result = Mock(spec=[])  # Empty spec = no attributes
            
            # Create harness instance
            harness = BaseTierHarness(
                config_name="dummy_config.json",
                out_root=output_dir
            )
            
            # Extract metrics
            metrics = harness._extract_metrics_for_tracking(
                "TEST-04",
                result,
                output_dir
            )
            
            # Should return None when required fields missing
            assert metrics is None
    
    @patch('harness.lfm_test_metrics.TestMetrics')
    def test_run_with_standard_wrapper_records_metrics(self, mock_metrics_class):
        """Test that run_with_standard_wrapper calls record_run()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Setup mock TestMetrics
            mock_metrics_instance = MagicMock()
            mock_metrics_class.return_value = mock_metrics_instance
            
            # Create summary.json for metrics extraction
            summary_data = {
                "test_id": "TEST-05",
                "passed": True,
                "runtime_sec": 3.14,
                "peak_cpu_percent": 75.0,
                "peak_memory_mb": 400.0,
                "peak_gpu_memory_mb": 1024.0
            }
            (output_dir / "summary.json").write_text(
                json.dumps(summary_data), 
                encoding='utf-8'
            )
            
            # Create mock test function
            mock_result = Mock()
            mock_result.passed = True
            mock_result.runtime_sec = 3.14
            test_func = Mock(return_value=mock_result)
            
            # Create harness instance
            harness = BaseTierHarness(
                config_name="dummy_config.json",
                out_root=output_dir
            )
            harness.use_cache = False  # Disable cache to simplify test
            
            # Run test
            result = harness.run_with_standard_wrapper(
                test_id="TEST-05",
                test_func=test_func,
                config={},
                test_config={},
                output_dir=output_dir
            )
            
            # Verify test function was called
            assert test_func.called
            
            # Verify TestMetrics was instantiated
            assert mock_metrics_class.called
            
            # Verify record_run was called
            assert mock_metrics_instance.record_run.called
            call_args = mock_metrics_instance.record_run.call_args
            assert call_args[0][0] == "TEST-05"  # First arg is test_id
            metrics_data = call_args[0][1]  # Second arg is metrics dict
            assert metrics_data["exit_code"] == 0
            assert metrics_data["runtime_sec"] == 3.14
    
    def test_metrics_recording_doesnt_break_test_on_error(self):
        """Test that metrics recording errors don't fail the test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create mock test function that succeeds
            mock_result = Mock()
            mock_result.passed = True
            mock_result.runtime_sec = 1.0
            test_func = Mock(return_value=mock_result)
            
            # Create harness instance
            harness = BaseTierHarness(
                config_name="dummy_config.json",
                out_root=output_dir
            )
            harness.use_cache = False
            
            # Patch TestMetrics to raise error
            with patch('harness.lfm_test_metrics.TestMetrics') as mock_metrics:
                mock_metrics.side_effect = Exception("Metrics database error")
                
                # Run test - should NOT raise exception
                result = harness.run_with_standard_wrapper(
                    test_id="TEST-06",
                    test_func=test_func,
                    config={},
                    test_config={},
                    output_dir=output_dir
                )
                
                # Test should still succeed
                assert result is not None
                assert result.passed is True


class TestMetricsIntegration:
    """Integration tests for end-to-end metrics recording."""
    
    @patch('harness.lfm_test_metrics.TestMetrics')
    def test_cached_test_updates_metrics(self, mock_metrics_class):
        """Test that cached test results also update metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            
            # Setup mock TestMetrics
            mock_metrics_instance = MagicMock()
            mock_metrics_class.return_value = mock_metrics_instance
            
            # Create cached summary.json
            summary_data = {
                "test_id": "TEST-CACHED",
                "passed": True,
                "runtime_sec": 0.5,
                "peak_cpu_percent": 50.0,
                "peak_memory_mb": 200.0,
                "peak_gpu_memory_mb": 512.0
            }
            (output_dir / "summary.json").write_text(
                json.dumps(summary_data),
                encoding='utf-8'
            )
            
            # Create harness with cache manager mock
            harness = BaseTierHarness(
                config_name="dummy_config.json",
                out_root=output_dir
            )
            
            # Mock cache hit
            harness.use_cache = True
            harness.force_rerun = False
            mock_cache = Mock()
            mock_cache.is_cache_valid.return_value = True
            mock_cache.get_cached_results.return_value = output_dir
            harness.cache_manager = mock_cache
            
            # Mock test function (shouldn't be called due to cache)
            test_func = Mock()
            
            # Run test (should use cache)
            result = harness.run_with_standard_wrapper(
                test_id="TEST-CACHED",
                test_func=test_func,
                config={},
                test_config={},
                output_dir=output_dir
            )
            
            # Test function should NOT be called (cache hit)
            assert not test_func.called
            
            # But metrics SHOULD still be recorded
            assert mock_metrics_instance.record_run.called
            call_args = mock_metrics_instance.record_run.call_args
            assert call_args[0][0] == "TEST-CACHED"
            metrics_data = call_args[0][1]
            assert metrics_data["runtime_sec"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
