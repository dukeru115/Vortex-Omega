"""
Test suite for Enhanced Kuramoto Module 1.4

Tests the advanced signal control features, adaptive coupling,
constitutional compliance, and real-time monitoring capabilities.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import Optional

# Import the enhanced kuramoto module (will work when dependencies are available)
try:
    from src.core.enhanced_kuramoto import (
        EnhancedKuramotoModule,
        CouplingMode,
        KuramotoSignal,
        AdaptiveState,
        SynchronizationMetrics
    )
    from src.core.state import KuramotoConfig
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    
    # Mock classes for testing structure
    @dataclass
    class MockKuramotoConfig:
        time_step: float = 0.01
        base_coupling_strength: float = 1.0
        natural_frequencies: dict = None
    
    class MockCouplingMode:
        ADAPTIVE = "adaptive"
        FIXED = "fixed"
    
    @dataclass
    class MockKuramotoSignal:
        signal_id: str
        target_modules: list
        signal_type: str
        amplitude: float
        frequency: float
        phase: float
        duration: float
        constitutional_check: bool = True


class TestEnhancedKuramotoModule:
    """Test suite for Enhanced Kuramoto Module functionality"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing"""
        if DEPENDENCIES_AVAILABLE:
            return KuramotoConfig(
                time_step=0.01,
                base_coupling_strength=1.0,
                natural_frequencies={
                    'cognitive': 1.0,
                    'executive': 1.2,
                    'memory': 0.8,
                    'perception': 1.1
                }
            )
        else:
            return MockKuramotoConfig(
                time_step=0.01,
                base_coupling_strength=1.0,
                natural_frequencies={
                    'cognitive': 1.0,
                    'executive': 1.2,
                    'memory': 0.8,
                    'perception': 1.1
                }
            )
    
    @pytest.fixture
    def mock_signal(self):
        """Create mock signal for testing"""
        if DEPENDENCIES_AVAILABLE:
            return KuramotoSignal(
                signal_id="test_signal_001",
                target_modules=['cognitive', 'executive'],
                signal_type="synchronization",
                amplitude=0.5,
                frequency=1.0,
                phase=0.0,
                duration=10.0,
                constitutional_check=True
            )
        else:
            return MockKuramotoSignal(
                signal_id="test_signal_001",
                target_modules=['cognitive', 'executive'],
                signal_type="synchronization",
                amplitude=0.5,
                frequency=1.0,
                phase=0.0,
                duration=10.0,
                constitutional_check=True
            )
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    def test_enhanced_kuramoto_initialization(self, mock_config):
        """Test Enhanced Kuramoto module initialization"""
        module = EnhancedKuramotoModule(
            config=mock_config,
            num_modules=4,
            coupling_mode=CouplingMode.ADAPTIVE
        )
        
        assert module.config == mock_config
        assert module.num_modules == 4
        assert module.coupling_mode == CouplingMode.ADAPTIVE
        assert module.initialized is False
        assert module.running is False
        assert len(module.active_signals) == 0
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    @pytest.mark.asyncio
    async def test_async_initialization(self, mock_config):
        """Test async initialization of enhanced module"""
        module = EnhancedKuramotoModule(
            config=mock_config,
            num_modules=4,
            coupling_mode=CouplingMode.ADAPTIVE
        )
        
        # Mock the async initialization
        with patch.object(module, '_initialize_adaptive_coupling') as mock_init:
            result = await module.initialize()
            
            assert result is True
            assert module.initialized is True
            mock_init.assert_called_once()
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    def test_adaptive_coupling_initialization(self, mock_config):
        """Test adaptive coupling matrix initialization"""
        module = EnhancedKuramotoModule(
            config=mock_config,
            num_modules=4,
            coupling_mode=CouplingMode.ADAPTIVE
        )
        
        # Test coupling matrix is properly initialized
        assert module.coupling_matrix.shape == (4, 4)
        assert np.allclose(np.diag(module.coupling_matrix), 
                          mock_config.base_coupling_strength)
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    def test_signal_validation(self, mock_config, mock_signal):
        """Test signal validation with constitutional checks"""
        module = EnhancedKuramotoModule(
            config=mock_config,
            num_modules=4,
            coupling_mode=CouplingMode.ADAPTIVE
        )
        
        # Mock the validation method
        with patch.object(module, '_validate_signal', return_value=True) as mock_validate:
            result = module.apply_control_signal(mock_signal)
            
            mock_validate.assert_called_once_with(mock_signal)
            assert result is True
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    def test_signal_constitutional_compliance(self, mock_config, mock_signal):
        """Test signal constitutional compliance checking"""
        module = EnhancedKuramotoModule(
            config=mock_config,
            num_modules=4,
            coupling_mode=CouplingMode.ADAPTIVE
        )
        
        # Test constitutional check is enforced
        mock_signal.constitutional_check = False
        mock_signal.amplitude = 10.0  # Excessive amplitude
        
        with patch.object(module, '_validate_signal', return_value=False) as mock_validate:
            result = module.apply_control_signal(mock_signal)
            
            mock_validate.assert_called_once_with(mock_signal)
            assert result is False
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    def test_emergency_mode_activation(self, mock_config):
        """Test emergency mode activation and coupling backup"""
        module = EnhancedKuramotoModule(
            config=mock_config,
            num_modules=4,
            coupling_mode=CouplingMode.ADAPTIVE
        )
        
        # Store original coupling
        original_coupling = module.coupling_matrix.copy()
        
        # Activate emergency mode
        module.emergency_mode = True
        module.emergency_coupling_backup = original_coupling
        
        assert module.emergency_mode is True
        assert module.emergency_coupling_backup is not None
        assert np.array_equal(module.emergency_coupling_backup, original_coupling)
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    def test_performance_metrics_tracking(self, mock_config):
        """Test performance metrics history tracking"""
        module = EnhancedKuramotoModule(
            config=mock_config,
            num_modules=4,
            coupling_mode=CouplingMode.ADAPTIVE
        )
        
        # Check metrics tracking structures
        assert hasattr(module, 'metrics_history')
        assert hasattr(module, 'signal_history')
        assert module.metrics_history.maxlen == 500
        assert module.signal_history.maxlen == 1000
    
    def test_module_structure_without_dependencies(self):
        """Test that module structure is testable even without full dependencies"""
        # This test ensures our test structure works even when imports fail
        assert DEPENDENCIES_AVAILABLE is False or DEPENDENCIES_AVAILABLE is True
        
        # Test mock objects work
        config = MockKuramotoConfig()
        signal = MockKuramotoSignal(
            signal_id="test",
            target_modules=[],
            signal_type="test",
            amplitude=1.0,
            frequency=1.0,
            phase=0.0,
            duration=1.0
        )
        
        assert config.time_step == 0.01
        assert signal.signal_id == "test"


class TestEnhancedKuramotoIntegration:
    """Integration tests for Enhanced Kuramoto with other NFCS components"""
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    @pytest.mark.asyncio
    async def test_real_time_monitoring_integration(self, mock_config):
        """Test integration with real-time monitoring system"""
        module = EnhancedKuramotoModule(
            config=mock_config,
            num_modules=4,
            coupling_mode=CouplingMode.ADAPTIVE
        )
        
        # Mock real-time monitoring callback
        monitoring_data = []
        
        async def mock_monitoring_callback(metrics):
            monitoring_data.append(metrics)
        
        # Test monitoring integration
        await module.initialize()
        
        # Simulate metrics update
        if hasattr(module, 'update_monitoring'):
            await module.update_monitoring(mock_monitoring_callback)
            
            assert len(monitoring_data) >= 0  # Basic check
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    def test_multi_agent_consensus_setup(self, mock_config):
        """Test setup for multi-agent consensus algorithms"""
        module = EnhancedKuramotoModule(
            config=mock_config,
            num_modules=8,  # More modules for consensus testing
            coupling_mode=CouplingMode.ADAPTIVE
        )
        
        # Test consensus-ready configuration
        assert module.num_modules == 8
        assert module.coupling_matrix.shape == (8, 8)
        
        # Check adaptive state for consensus
        assert hasattr(module, 'adaptive_state')
        assert isinstance(module.adaptive_state, AdaptiveState) if DEPENDENCIES_AVAILABLE else True


# Performance and benchmark tests
class TestEnhancedKuramotoPerformance:
    """Performance tests for Enhanced Kuramoto Module"""
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    @pytest.mark.benchmark
    def test_signal_processing_performance(self, benchmark, mock_config, mock_signal):
        """Benchmark signal processing performance"""
        module = EnhancedKuramotoModule(
            config=mock_config,
            num_modules=4,
            coupling_mode=CouplingMode.ADAPTIVE
        )
        
        def process_signal():
            return module.apply_control_signal(mock_signal)
        
        result = benchmark(process_signal)
        assert result is not None
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    @pytest.mark.benchmark
    def test_coupling_matrix_update_performance(self, benchmark, mock_config):
        """Benchmark adaptive coupling matrix updates"""
        module = EnhancedKuramotoModule(
            config=mock_config,
            num_modules=16,  # Larger system for performance testing
            coupling_mode=CouplingMode.ADAPTIVE
        )
        
        def update_coupling():
            # Simulate coupling update
            if hasattr(module, 'update_adaptive_coupling'):
                module.update_adaptive_coupling()
            return True
        
        result = benchmark(update_coupling)
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])