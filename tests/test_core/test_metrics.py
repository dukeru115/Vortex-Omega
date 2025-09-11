"""
Tests for metrics calculator.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.metrics import MetricsCalculator
from core.state import CostFunctionalConfig, SystemState, create_empty_system_state


class TestMetricsCalculator:
    """Test cases for metrics calculator."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = CostFunctionalConfig(
            w_field_energy=1.0,
            w_field_gradient=0.5,
            w_control_energy=0.1,
            w_coherence_penalty=2.0,
            w_hallucinations=1.5,
            w_defect_density=1.0,
            w_coherence_loss=2.0,
            w_violations=3.0
        )
        self.calculator = MetricsCalculator(self.config)
        self.calculator.set_spatial_resolution(0.1, 0.1)
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        assert self.calculator.config.w_field_energy == 1.0
        assert self.calculator.dx == 0.1
        assert self.calculator.dy == 0.1
    
    def test_defect_density_flat_wave(self):
        """Test defect density for flat wave (should be zero)."""
        # Create uniform field (no topological defects)
        field = np.ones((32, 32), dtype=np.complex128)
        
        rho_def = self.calculator.calculate_defect_density(field)
        
        assert rho_def.shape == (32, 32)
        assert np.all(rho_def >= 0)  # Defect density should be non-negative
        # For uniform field, should be approximately zero
        assert np.mean(rho_def) < 0.01
    
    def test_defect_density_vortex(self):
        """Test defect density for field with vortex."""
        # Create simple vortex pattern
        x = np.linspace(-2, 2, 32)
        y = np.linspace(-2, 2, 32)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Vortex: φ = r * exp(i*θ) where θ = atan2(y, x)
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        field = r * np.exp(1j * theta)
        
        # Avoid singularity at center
        field[r < 0.1] = 0.1 * np.exp(1j * theta[r < 0.1])
        
        rho_def = self.calculator.calculate_defect_density(field)
        
        assert rho_def.shape == (32, 32)
        assert np.all(rho_def >= 0)
        # Should detect defect near center
        assert np.max(rho_def) > 0.1
    
    def test_coherence_calculation(self):
        """Test coherence calculation."""
        # High coherence field
        field_high = np.ones((16, 16), dtype=np.complex128)
        phases_high = np.zeros(4)  # All phases aligned
        
        R_glob_high, R_mod_high = self.calculator.calculate_coherence(field_high, phases_high)
        
        assert 0.0 <= R_glob_high <= 1.0
        assert 0.0 <= R_mod_high <= 1.0
        assert R_glob_high == 1.0  # Perfect global coherence
        assert R_mod_high == 1.0   # Perfect modular coherence
        
        # Low coherence
        np.random.seed(42)
        field_low = np.random.randn(16, 16) + 1j * np.random.randn(16, 16)
        phases_low = 2 * np.pi * np.random.rand(4) - np.pi
        
        R_glob_low, R_mod_low = self.calculator.calculate_coherence(field_low, phases_low)
        
        assert 0.0 <= R_glob_low <= 1.0
        assert 0.0 <= R_mod_low <= 1.0
        assert R_mod_low < 0.5  # Should be low for random phases
    
    def test_field_energy_calculation(self):
        """Test field energy calculation."""
        # Unit amplitude field
        field = np.ones((10, 10), dtype=np.complex128)
        
        energy = self.calculator.calculate_field_energy(field)
        
        # Energy = ∫|φ|² dx = 1² * area = 1 * (1.0 * 1.0) = 1.0
        expected_energy = 1.0 * (10 * 0.1) * (10 * 0.1)  # 100 grid points * dx * dy
        assert np.isclose(energy, expected_energy, rtol=1e-10)
        
        # Zero field
        field_zero = np.zeros((10, 10), dtype=np.complex128)
        energy_zero = self.calculator.calculate_field_energy(field_zero)
        assert energy_zero == 0.0
    
    def test_control_energy_calculation(self):
        """Test control energy calculation."""
        # Constant control field
        control = 2.0 * np.ones((10, 10), dtype=np.complex128)
        
        energy = self.calculator.calculate_control_energy(control)
        
        # Energy = ∫|u|² dx = 4 * area
        expected_energy = 4.0 * (10 * 0.1) * (10 * 0.1)
        assert np.isclose(energy, expected_energy, rtol=1e-10)
    
    def test_hallucination_number_calculation(self):
        """Test hallucination number calculation."""
        # Simple defect field
        rho_def_field = 0.1 * np.ones((10, 10))
        
        H_a = self.calculator.calculate_hallucination_number(
            rho_def_field,
            prediction_error=0.2,
            ontological_drift=0.1,
            weight_prediction=1.0,
            weight_ontology=1.0
        )
        
        # H_a = integral of defects + weighted errors
        # = 0.1 * area + 0.2 + 0.1 = 0.1 * 1.0 + 0.3 = 0.4
        expected = 0.1 * (10 * 0.1) * (10 * 0.1) + 0.2 + 0.1
        assert np.isclose(H_a, expected, rtol=1e-10)
    
    def test_systemic_risk_calculation(self):
        """Test systemic risk calculation."""
        risk = self.calculator.calculate_systemic_risk(
            hallucination_number=0.5,
            rho_def_mean=0.1,
            coherence_global=0.8,
            coherence_modular=0.7,
            violations_count=2
        )
        
        # Risk = w_h*H_a + w_d*rho + w_c*(1-R_g) + w_c*(1-R_m) + w_v*violations
        expected = (1.5*0.5 + 1.0*0.1 + 2.0*(1-0.8) + 2.0*(1-0.7) + 3.0*2)
        assert np.isclose(risk, expected, rtol=1e-10)
    
    def test_system_state_processing(self):
        """Test processing of complete system state."""
        # Create test state
        state = create_empty_system_state(grid_size=(16, 16), n_modules=3)
        
        # Set non-trivial field
        state.neural_field = np.ones((16, 16), dtype=np.complex128) * 0.5
        state.module_phases = np.array([0.1, 0.2, 0.3])
        
        # Add spatial extent info for proper scaling
        state.spatial_extent = (1.6, 1.6)  # 16 * 0.1 = 1.6
        
        # Calculate all metrics
        metrics = self.calculator.calculate_all_metrics(state)
        
        # Check that all metrics are computed
        assert hasattr(metrics, 'rho_def_field')
        assert hasattr(metrics, 'hallucination_number')
        assert hasattr(metrics, 'coherence_global')
        assert hasattr(metrics, 'coherence_modular')
        assert hasattr(metrics, 'systemic_risk')
        
        # Check reasonable values
        assert metrics.rho_def_field.shape == (16, 16)
        assert np.isfinite(metrics.hallucination_number)
        assert 0.0 <= metrics.coherence_global <= 1.0
        assert 0.0 <= metrics.coherence_modular <= 1.0
        assert np.isfinite(metrics.systemic_risk)
    
    def test_cost_functional_calculation(self):
        """Test cost functional calculation."""
        # Create test state
        state = create_empty_system_state(grid_size=(8, 8), n_modules=2)
        
        # Simple field and control
        state.neural_field = 0.5 * np.ones((8, 8), dtype=np.complex128)
        state.last_control_signals.u_field = 0.1 * np.ones((8, 8), dtype=np.complex128)
        state.module_phases = np.array([0.0, 0.1])
        
        # Add spatial extent
        state.spatial_extent = (0.8, 0.8)
        
        J = self.calculator.calculate_cost_functional(state)
        
        assert np.isfinite(J)
        assert J >= 0.0  # Cost should be non-negative
    
    def test_input_validation(self):
        """Test input validation."""
        # Test without setting spatial resolution
        calc_no_res = MetricsCalculator(self.config)
        field = np.ones((10, 10), dtype=np.complex128)
        
        with pytest.raises(ValueError):
            calc_no_res.calculate_defect_density(field)
        
        # Test with wrong field type
        with pytest.raises(ValueError):
            real_field = np.ones((10, 10), dtype=np.float64)
            self.calculator.calculate_defect_density(real_field)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])