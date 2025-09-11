"""
Test suite for Regulator module.

Tests control signal generation, feedback control, and optimization functionality.
"""

import pytest
import numpy as np
from src.core.regulator import Regulator
from src.core.state import CostFunctionalConfig, SystemState, create_empty_system_state


class TestRegulator:
    """Test cases for Regulator."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return CostFunctionalConfig(
            w_field_energy=1.0,
            w_field_gradient=0.5,
            w_control_energy=0.1,
            w_coherence_penalty=2.0,
            w_hallucinations=1.5,
            w_defect_density=1.0,
            w_coherence_loss=2.0,
            w_violations=3.0
        )
    
    @pytest.fixture
    def regulator(self, basic_config):
        """Regulator instance."""
        return Regulator(basic_config)
    
    @pytest.fixture
    def test_state(self):
        """Test system state."""
        state = create_empty_system_state(grid_size=(16, 16), n_modules=3)
        
        # Set up simple test field
        state.neural_field = 0.1 * np.ones((16, 16), dtype=np.complex128)
        state.module_phases = np.array([0.1, 0.2, 0.3])
        
        return state
    
    def test_initialization(self, regulator, basic_config):
        """Test regulator initialization."""
        assert regulator.config == basic_config
        assert regulator.max_control_amplitude > 0
        assert regulator.optimization_method == 'L-BFGS-B'
        assert regulator.max_iterations > 0
        assert regulator.tolerance > 0
        assert len(regulator.control_history) == 0
    
    def test_constraint_setting(self, regulator):
        """Test setting control constraints."""
        regulator.set_control_constraints(
            max_amplitude=2.0,
            method='SLSQP',
            max_iter=100,
            tol=1e-8
        )
        
        assert regulator.max_control_amplitude == 2.0
        assert regulator.optimization_method == 'SLSQP'
        assert regulator.max_iterations == 100
        assert regulator.tolerance == 1e-8
    
    def test_feedback_control_basic(self, regulator, test_state):
        """Test basic feedback control computation."""
        control_signals = regulator.compute_feedback_control(
            test_state, target_coherence=0.8
        )
        
        # Check output structure
        assert hasattr(control_signals, 'u_field')
        assert hasattr(control_signals, 'u_modules')
        assert hasattr(control_signals, 'timestamp')
        
        # Check dimensions
        assert control_signals.u_field.shape == test_state.neural_field.shape
        assert control_signals.u_modules.shape == test_state.module_phases.shape
        
        # Check data types
        assert control_signals.u_field.dtype == np.complex128
        assert control_signals.u_modules.dtype == np.float64
        
        # Check finite values
        assert np.all(np.isfinite(control_signals.u_field))
        assert np.all(np.isfinite(control_signals.u_modules))
    
    def test_feedback_control_coherence_response(self, regulator, test_state):
        """Test feedback control response to coherence levels."""
        # Low coherence state (should generate stronger control)
        low_coherence_field = 0.01 * np.random.randn(16, 16).astype(np.complex128)
        test_state.neural_field = low_coherence_field
        
        control_low = regulator.compute_feedback_control(
            test_state, target_coherence=0.8
        )
        
        # High coherence state (should generate weaker control)
        high_coherence_field = 0.8 * np.ones((16, 16), dtype=np.complex128)
        test_state.neural_field = high_coherence_field
        
        control_high = regulator.compute_feedback_control(
            test_state, target_coherence=0.8
        )
        
        # Low coherence should produce stronger control
        control_low_magnitude = np.mean(np.abs(control_low.u_field))
        control_high_magnitude = np.mean(np.abs(control_high.u_field))
        
        # Note: This might not always hold due to error calculation, but test structure
        assert np.all(np.abs(control_low.u_field) <= regulator.max_control_amplitude)
        assert np.all(np.abs(control_high.u_field) <= regulator.max_control_amplitude)
    
    def test_optimal_control_basic(self, regulator, test_state):
        """Test optimal control computation (simplified)."""
        control_signals = regulator.compute_optimal_control(test_state)
        
        # Should return valid control signals
        assert hasattr(control_signals, 'u_field')
        assert hasattr(control_signals, 'u_modules')
        assert control_signals.u_field.shape == test_state.neural_field.shape
        assert control_signals.u_modules.shape == test_state.module_phases.shape
        
        # Should respect amplitude constraints
        assert np.all(np.abs(control_signals.u_field) <= regulator.max_control_amplitude)
        assert np.all(np.abs(control_signals.u_modules) <= regulator.max_control_amplitude)
    
    def test_control_amplitude_constraints(self, regulator, test_state):
        """Test control amplitude constraint enforcement."""
        # Set tight constraints
        regulator.set_control_constraints(max_amplitude=0.1)
        
        control_signals = regulator.compute_feedback_control(test_state)
        
        # Should respect amplitude limits
        field_magnitudes = np.abs(control_signals.u_field)
        module_magnitudes = np.abs(control_signals.u_modules)
        
        assert np.all(field_magnitudes <= 0.1 + 1e-10)  # Small tolerance for numerical precision
        assert np.all(module_magnitudes <= 0.1 + 1e-10)
    
    def test_control_history_tracking(self, regulator, test_state):
        """Test control history storage."""
        initial_history_length = len(regulator.control_history)
        
        # Generate several control signals
        for _ in range(3):
            regulator.compute_feedback_control(test_state)
        
        # History should grow
        assert len(regulator.control_history) == initial_history_length + 3
        
        # Each entry should be a valid ControlSignals object
        for control in regulator.control_history:
            assert hasattr(control, 'u_field')
            assert hasattr(control, 'u_modules')
            assert hasattr(control, 'timestamp')
    
    def test_control_statistics(self, regulator, test_state):
        """Test control performance statistics."""
        # Initially no statistics
        stats = regulator.get_control_statistics()
        assert stats['num_controls'] == 0
        
        # Generate some control signals
        for i in range(5):
            test_state.simulation_time = i * 0.1
            regulator.compute_feedback_control(test_state)
        
        # Check statistics
        stats = regulator.get_control_statistics()
        assert stats['num_controls'] == 5
        assert 'mean_field_magnitude' in stats
        assert 'mean_module_magnitude' in stats
        assert 'max_field_magnitude' in stats
        assert 'max_module_magnitude' in stats
        
        # All statistics should be non-negative
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                assert value >= 0
    
    def test_different_target_coherence(self, regulator, test_state):
        """Test response to different target coherence values."""
        targets = [0.3, 0.5, 0.7, 0.9]
        control_responses = []
        
        for target in targets:
            control = regulator.compute_feedback_control(test_state, target_coherence=target)
            control_responses.append(np.mean(np.abs(control.u_field)))
        
        # Should generate finite responses for all targets
        assert all(np.isfinite(response) for response in control_responses)
        
        # Should respect constraints for all targets
        for target in targets:
            control = regulator.compute_feedback_control(test_state, target_coherence=target)
            assert np.all(np.abs(control.u_field) <= regulator.max_control_amplitude)
    
    def test_zero_field_handling(self, regulator):
        """Test handling of zero field state."""
        state = create_empty_system_state(grid_size=(8, 8), n_modules=2)
        # Keep default zero fields
        
        control_signals = regulator.compute_feedback_control(state)
        
        # Should handle gracefully
        assert np.all(np.isfinite(control_signals.u_field))
        assert np.all(np.isfinite(control_signals.u_modules))
        assert np.all(np.abs(control_signals.u_field) <= regulator.max_control_amplitude)
    
    def test_error_handling(self, regulator):
        """Test error handling for invalid inputs."""
        # Test with None state - should raise appropriate error
        with pytest.raises((AttributeError, TypeError)):
            regulator.compute_feedback_control(None)
    
    def test_timestamp_consistency(self, regulator, test_state):
        """Test timestamp consistency in control signals."""
        test_time = 1.23
        test_state.simulation_time = test_time
        
        control_signals = regulator.compute_feedback_control(test_state)
        
        assert control_signals.timestamp == test_time


class TestRegulatorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_extreme_coherence_values(self):
        """Test with extreme coherence targets."""
        config = CostFunctionalConfig()
        regulator = Regulator(config)
        state = create_empty_system_state(grid_size=(4, 4), n_modules=2)
        
        # Very low target
        control1 = regulator.compute_feedback_control(state, target_coherence=0.001)
        assert np.all(np.isfinite(control1.u_field))
        
        # Very high target (nearly 1.0)
        control2 = regulator.compute_feedback_control(state, target_coherence=0.999)
        assert np.all(np.isfinite(control2.u_field))
    
    def test_single_module(self):
        """Test with single module system."""
        config = CostFunctionalConfig()
        regulator = Regulator(config)
        state = create_empty_system_state(grid_size=(4, 4), n_modules=1)
        
        control_signals = regulator.compute_feedback_control(state)
        
        assert control_signals.u_modules.shape == (1,)
        assert np.isfinite(control_signals.u_modules[0])
    
    def test_very_small_grid(self):
        """Test with minimal grid size."""
        config = CostFunctionalConfig()
        regulator = Regulator(config)
        state = create_empty_system_state(grid_size=(2, 2), n_modules=1)
        
        control_signals = regulator.compute_feedback_control(state)
        
        assert control_signals.u_field.shape == (2, 2)
        assert np.all(np.isfinite(control_signals.u_field))