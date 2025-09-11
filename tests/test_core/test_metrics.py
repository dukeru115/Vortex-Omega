"""
Test suite for risk metrics and coherence computation.

Tests numerical accuracy of topological defect detection, 
Hallucination Number calculation, and cost functional computation.
"""

import pytest
import numpy as np
from src.core.metrics import MetricsCalculator
from src.core.state import CostFunctionalConfig, SystemState, create_empty_system_state


class TestMetricsCalculator:
    """Test cases for metrics calculator."""
    
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
    def calculator(self, basic_config):
        """Metrics calculator instance."""
        return MetricsCalculator(basic_config)
    
    @pytest.fixture
    def test_state(self):
        """Basic system state for testing."""
        state = create_empty_system_state(grid_size=(32, 32), n_modules=4)
        
        # Set up test field (simple pattern)
        nx, ny = 32, 32
        x = np.linspace(-2, 2, nx)
        y = np.linspace(-2, 2, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Gaussian field
        state.neural_field = 0.5 * np.exp(-(X**2 + Y**2)/2) * np.exp(1j * (X + 0.5*Y))
        
        # Random but bounded phases
        state.module_phases = np.array([0.1, 0.3, -0.2, 0.8])
        
        return state
    
    def test_initialization(self, calculator, basic_config):
        """Test proper calculator initialization."""
        assert calculator.config == basic_config
        assert calculator.dx is None  # Not set initially
        assert calculator.dy is None
    
    def test_spatial_resolution_setting(self, calculator):
        """Test spatial resolution configuration."""
        dx, dy = 0.1, 0.2
        calculator.set_spatial_resolution(dx, dy)
        
        assert calculator.dx == dx
        assert calculator.dy == dy
        
        # Test invalid values
        with pytest.raises(ValueError, match="Spatial resolution must be positive"):
            calculator.set_spatial_resolution(-0.1, 0.2)
    
    def test_defect_density_uniform_field(self, calculator):
        """Test defect density for uniform field (should be zero)."""
        calculator.set_spatial_resolution(0.1, 0.1)
        
        # Uniform field (no phase variation)
        uniform_field = np.ones((32, 32), dtype=np.complex128)
        
        rho_def = calculator.calculate_defect_density(uniform_field)
        
        # Should be approximately zero for uniform field
        assert rho_def.shape == (32, 32)
        assert np.all(rho_def >= 0)  # Density is non-negative
        assert np.mean(rho_def) < 0.1  # Should be small for uniform field
    
    def test_defect_density_plane_wave(self, calculator):
        """Test defect density for plane wave (should be zero)."""
        calculator.set_spatial_resolution(0.1, 0.1)
        
        # Plane wave: φ = exp(i k·r)
        nx, ny = 32, 32
        x = np.linspace(0, 2*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        plane_wave = np.exp(1j * (X + 0.5 * Y))
        
        rho_def = calculator.calculate_defect_density(plane_wave)
        
        # Plane wave should have no topological defects
        assert np.mean(rho_def) < 0.05  # Very small
    
    def test_defect_density_vortex(self, calculator):
        """Test defect density for field with vortex."""
        calculator.set_spatial_resolution(0.1, 0.1)
        
        # Create vortex field: φ = r * exp(i*θ) where θ = atan2(y,x)
        nx, ny = 32, 32
        x = np.linspace(-2, 2, nx)
        y = np.linspace(-2, 2, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        
        # Vortex with topological charge m=1
        vortex_field = (r + 0.1) * np.exp(1j * theta)  # +0.1 to avoid singularity
        
        rho_def = calculator.calculate_defect_density(vortex_field)
        
        # Should detect defect near center
        assert np.max(rho_def) > 0.1  # Significant defect density
        
        # Defect should be localized near center
        center = nx // 2
        central_region = rho_def[center-2:center+3, center-2:center+3]
        assert np.mean(central_region) > np.mean(rho_def)
    
    def test_hallucination_number_calculation(self, calculator):
        """Test Hallucination Number computation."""
        calculator.set_spatial_resolution(0.1, 0.1)
        
        # Test field with some defects
        rho_def_field = np.zeros((32, 32))
        rho_def_field[15:17, 15:17] = 0.2  # Localized defect
        
        # Calculate H_a
        H_a = calculator.calculate_hallucination_number(
            rho_def_field,
            prediction_error=0.1,
            ontological_drift=0.05,
            weight_prediction=1.0,
            weight_ontology=1.0
        )
        
        assert isinstance(H_a, float)
        assert H_a > 0  # Should be positive with defects and errors
        
        # Test with zero inputs
        H_a_zero = calculator.calculate_hallucination_number(
            np.zeros((32, 32)),
            prediction_error=0.0,
            ontological_drift=0.0
        )
        
        assert H_a_zero == 0.0  # Should be zero with no defects or errors
    
    def test_coherence_calculation(self, calculator):
        """Test coherence computation."""
        # Test field coherence
        # High amplitude, uniform field -> high coherence
        high_coherence_field = np.ones((32, 32), dtype=np.complex128) * 0.8
        low_coherence_field = 0.01 * np.random.randn(32, 32).astype(np.complex128)
        
        # Test module coherence
        sync_phases = np.array([0.5, 0.5, 0.5, 0.5])  # Perfect sync
        async_phases = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])  # Anti-sync
        
        # High coherence case
        R_global_high, R_modular_high = calculator.calculate_coherence(
            high_coherence_field, sync_phases
        )
        
        # Low coherence case  
        R_global_low, R_modular_low = calculator.calculate_coherence(
            low_coherence_field, async_phases
        )
        
        # Assertions
        assert 0 <= R_global_high <= 1
        assert 0 <= R_global_low <= 1
        assert 0 <= R_modular_high <= 1
        assert 0 <= R_modular_low <= 1
        
        assert R_global_high > R_global_low
        assert R_modular_high > R_modular_low
        assert R_modular_high > 0.8  # Should be high for synchronized phases
        assert R_modular_low < 0.2   # Should be low for anti-synchronized phases
    
    def test_energy_calculations(self, calculator, test_state):
        """Test energy computation functions."""
        calculator.set_spatial_resolution(0.1, 0.1)
        
        # Field energy
        field_energy = calculator.calculate_field_energy(test_state.neural_field)
        assert isinstance(field_energy, float)
        assert field_energy >= 0  # Energy is non-negative
        
        # Gradient energy
        gradient_energy = calculator.calculate_gradient_energy(test_state.neural_field)
        assert isinstance(gradient_energy, float)
        assert gradient_energy >= 0
        
        # Control energy
        control_energy = calculator.calculate_control_energy(
            test_state.last_control_signals.u_field
        )
        assert isinstance(control_energy, float)
        assert control_energy >= 0
        
        # Larger fields should have larger energies
        large_field = 2.0 * test_state.neural_field
        large_field_energy = calculator.calculate_field_energy(large_field)
        assert large_field_energy > field_energy
    
    def test_systemic_risk_calculation(self, calculator):
        """Test total systemic risk computation."""
        # Test inputs
        H_a = 0.3
        rho_def_mean = 0.1
        coherence_global = 0.7
        coherence_modular = 0.8
        violations = 2
        
        risk = calculator.calculate_systemic_risk(
            hallucination_number=H_a,
            rho_def_mean=rho_def_mean,
            coherence_global=coherence_global,
            coherence_modular=coherence_modular,
            violations_count=violations
        )
        
        assert isinstance(risk, float)
        assert risk > 0  # Should be positive with these inputs
        
        # Test with perfect conditions (should be lower)
        perfect_risk = calculator.calculate_systemic_risk(
            hallucination_number=0.0,
            rho_def_mean=0.0,
            coherence_global=1.0,
            coherence_modular=1.0,
            violations_count=0
        )
        
        assert perfect_risk < risk  # Perfect conditions -> lower risk
    
    def test_cost_functional_calculation(self, calculator, test_state):
        """Test cost functional J computation."""
        # Need to set spatial extent for proper resolution calculation
        test_state.spatial_extent = (4.0, 4.0)
        
        cost = calculator.calculate_cost_functional(test_state)
        
        assert isinstance(cost, float)
        assert np.isfinite(cost)
        
        # Cost should increase with larger control signals
        test_state.last_control_signals.u_field *= 2.0  # Double control
        higher_cost = calculator.calculate_cost_functional(test_state)
        
        assert higher_cost > cost  # Larger control -> higher cost
    
    def test_all_metrics_calculation(self, calculator, test_state):
        """Test comprehensive metrics calculation."""
        # Set spatial extent for proper calculation
        test_state.spatial_extent = (4.0, 4.0)
        
        metrics = calculator.calculate_all_metrics(test_state)
        
        # Check all required fields are present
        assert hasattr(metrics, 'rho_def_field')
        assert hasattr(metrics, 'rho_def_mean')
        assert hasattr(metrics, 'hallucination_number')
        assert hasattr(metrics, 'systemic_risk')
        assert hasattr(metrics, 'coherence_global')
        assert hasattr(metrics, 'coherence_modular')
        assert hasattr(metrics, 'field_energy')
        assert hasattr(metrics, 'control_energy')
        
        # Check data types and ranges
        assert isinstance(metrics.rho_def_field, np.ndarray)
        assert metrics.rho_def_field.shape == test_state.neural_field.shape
        assert np.all(metrics.rho_def_field >= 0)
        
        assert isinstance(metrics.rho_def_mean, float)
        assert metrics.rho_def_mean >= 0
        
        assert isinstance(metrics.hallucination_number, float)
        assert metrics.hallucination_number >= 0
        
        assert isinstance(metrics.systemic_risk, float)
        
        assert 0 <= metrics.coherence_global <= 1
        assert 0 <= metrics.coherence_modular <= 1
        
        assert metrics.field_energy >= 0
        assert metrics.control_energy >= 0
    
    def test_defect_topology_analysis(self, calculator):
        """Test topological defect analysis."""
        calculator.set_spatial_resolution(0.1, 0.1)
        
        # Create field with isolated defects
        rho_def_field = np.zeros((32, 32))
        
        # Add two separated defects
        rho_def_field[10:12, 10:12] = 0.3  # Defect 1
        rho_def_field[20:22, 20:22] = 0.2  # Defect 2
        
        analysis = calculator.analyze_defect_topology(
            rho_def_field, threshold=0.1
        )
        
        # Check analysis results
        assert 'num_defects' in analysis
        assert 'total_defect_area' in analysis
        assert 'mean_defect_size' in analysis
        assert 'max_defect_size' in analysis
        
        assert analysis['num_defects'] >= 1  # Should detect defects
        assert analysis['total_defect_area'] > 0
        assert analysis['mean_defect_size'] > 0
    
    def test_error_handling(self, calculator):
        """Test error handling and validation."""
        # Test without setting spatial resolution
        field = np.ones((32, 32), dtype=np.complex128)
        
        with pytest.raises(ValueError, match="Spatial resolution not set"):
            calculator.calculate_defect_density(field)
        
        # Test with non-complex field
        with pytest.raises(ValueError, match="Field must be complex"):
            calculator.set_spatial_resolution(0.1, 0.1)
            real_field = np.ones((32, 32), dtype=np.float64)
            calculator.calculate_defect_density(real_field)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test negative weights (should raise error)
        with pytest.raises(ValueError, match="non-negative"):
            bad_config = CostFunctionalConfig(
                w_field_energy=-1.0,  # Invalid
                w_field_gradient=0.5,
                w_control_energy=0.1,
                w_coherence_penalty=2.0
            )
            MetricsCalculator(bad_config)


class TestMetricsEdgeCases:
    """Test edge cases and special conditions."""
    
    def test_zero_field(self):
        """Test metrics with zero field."""
        config = CostFunctionalConfig()
        calculator = MetricsCalculator(config)
        calculator.set_spatial_resolution(0.1, 0.1)
        
        zero_field = np.zeros((16, 16), dtype=np.complex128)
        
        # Should handle zero field gracefully
        rho_def = calculator.calculate_defect_density(zero_field)
        assert np.all(rho_def == 0.0)
        
        field_energy = calculator.calculate_field_energy(zero_field)
        assert field_energy == 0.0
    
    def test_single_pixel_field(self):
        """Test with minimal field size."""
        config = CostFunctionalConfig()
        calculator = MetricsCalculator(config)
        calculator.set_spatial_resolution(1.0, 1.0)
        
        tiny_field = np.array([[1.0 + 1.0j]], dtype=np.complex128)
        
        # Should work with minimal field
        rho_def = calculator.calculate_defect_density(tiny_field)
        assert rho_def.shape == (1, 1)
        
        energy = calculator.calculate_field_energy(tiny_field)
        assert energy > 0
    
    def test_very_high_defect_density(self):
        """Test with extremely high defect densities."""
        config = CostFunctionalConfig()
        calculator = MetricsCalculator(config)
        calculator.set_spatial_resolution(0.1, 0.1)
        
        # Field with very high defect density everywhere
        high_defect_field = np.ones((16, 16)) * 10.0
        
        H_a = calculator.calculate_hallucination_number(high_defect_field)
        
        assert np.isfinite(H_a)
        assert H_a > 0