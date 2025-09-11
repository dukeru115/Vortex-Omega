"""
Tests for Kuramoto oscillator solver.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.kuramoto_solver import KuramotoSolver
from core.state import KuramotoConfig


class TestKuramotoSolver:
    """Test cases for Kuramoto solver."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = KuramotoConfig(
            natural_frequencies={
                'module1': 1.0,
                'module2': 2.0,
                'module3': 1.5
            },
            base_coupling_strength=0.5,
            time_step=0.01
        )
        self.module_order = ['module1', 'module2', 'module3']
        self.solver = KuramotoSolver(self.config, self.module_order)
    
    def test_solver_initialization(self):
        """Test solver initialization."""
        assert self.solver.num_modules == 3
        assert np.allclose(self.solver.omega, [1.0, 2.0, 1.5])
        assert self.solver.dt == 0.01
    
    def test_uncoupled_evolution(self):
        """Test evolution without coupling."""
        # Initial phases
        phases = np.array([0.0, 0.5, -0.3])
        
        # Zero coupling and control
        coupling = np.zeros((3, 3))
        control = np.zeros(3)
        
        # Evolve one step
        new_phases = self.solver.step(phases, coupling, control)
        
        # Should evolve as θ_i(t+dt) = θ_i(t) + ω_i * dt
        expected = phases + self.solver.omega * self.solver.dt
        expected = np.mod(expected + np.pi, 2*np.pi) - np.pi  # Wrap to [-π, π]
        
        assert np.allclose(new_phases, expected, atol=1e-10)
    
    def test_synchronization_tendency(self):
        """Test that coupling leads to synchronization."""
        # Start with different phases
        phases = np.array([0.0, np.pi/2, np.pi])
        
        # Strong coupling
        coupling = self.solver.build_coupling_matrix(base_strength=2.0)
        control = np.zeros(3)
        
        # Evolve many steps
        for _ in range(100):
            phases = self.solver.step(phases, coupling, control)
        
        # Calculate order parameter
        R, _ = self.solver.compute_order_parameter(phases)
        
        # Should show some synchronization (R > 0.3 is reasonable for this setup)
        assert R > 0.3
    
    def test_order_parameter_calculation(self):
        """Test order parameter calculation."""
        # Perfect synchronization
        phases = np.array([0.5, 0.5, 0.5])
        R, psi = self.solver.compute_order_parameter(phases)
        
        assert np.isclose(R, 1.0, atol=1e-10)
        assert np.isclose(psi, 0.5, atol=1e-10)
        
        # Random phases (low coherence expected)
        np.random.seed(42)
        phases = 2 * np.pi * np.random.rand(10) - np.pi
        R, _ = self.solver.compute_order_parameter(phases)
        
        assert 0.0 <= R <= 1.0
        assert R < 0.5  # Should be low for random phases
    
    def test_coupling_matrix_building(self):
        """Test coupling matrix construction."""
        # Default coupling
        K = self.solver.build_coupling_matrix()
        
        assert K.shape == (3, 3)
        assert np.allclose(K, K.T)  # Should be symmetric
        assert np.allclose(np.diag(K), 0.0)  # Zero diagonal
        
        # With modulation
        modulation = np.ones((3, 3)) * 0.5
        K_mod = self.solver.build_coupling_matrix(modulation_factors=modulation)
        
        assert np.allclose(K_mod, 0.5 * K)
        
        # With sparsity
        sparsity = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        K_sparse = self.solver.build_coupling_matrix(sparsity_mask=sparsity)
        
        assert K_sparse[0, 2] == 0.0  # Should be zero due to mask
        assert K_sparse[1, 0] != 0.0  # Should be non-zero
    
    def test_pairwise_synchronization(self):
        """Test pairwise synchronization calculation."""
        phases = np.array([0.0, 0.1, np.pi])
        sync_matrix = self.solver.compute_pairwise_synchronization(phases)
        
        assert sync_matrix.shape == (3, 3)
        assert np.allclose(np.diag(sync_matrix), 1.0)  # Self-sync = 1
        assert 0.0 <= sync_matrix.min() <= 1.0
        assert 0.0 <= sync_matrix.max() <= 1.0
        
        # Close phases should have high synchronization
        assert sync_matrix[0, 1] > sync_matrix[0, 2]
    
    def test_cluster_coherence(self):
        """Test cluster coherence calculation."""
        phases = np.array([0.1, 0.2, np.pi-0.1, np.pi+0.1])  # Two clusters
        cluster_labels = np.array([0, 0, 1, 1])
        
        coherence = self.solver.compute_cluster_coherence(phases, cluster_labels)
        
        assert 0 in coherence
        assert 1 in coherence
        assert coherence[0] > 0.8  # First cluster should be coherent
        assert coherence[1] > 0.8  # Second cluster should be coherent
    
    def test_control_signal_effect(self):
        """Test that control signals affect evolution."""
        phases = np.array([0.0, 0.0, 0.0])
        coupling = np.zeros((3, 3))
        
        # Without control
        phases1 = self.solver.step(phases.copy(), coupling, np.zeros(3))
        
        # With control
        control = np.array([1.0, -1.0, 0.5])
        phases2 = self.solver.step(phases.copy(), coupling, control)
        
        # Should be different
        assert not np.allclose(phases1, phases2, atol=1e-10)
    
    def test_phase_wrapping(self):
        """Test that phases are properly wrapped to [-π, π]."""
        # Large initial phases
        phases = np.array([3*np.pi, -4*np.pi, 5*np.pi])
        coupling = np.zeros((3, 3))
        control = np.zeros(3)
        
        new_phases = self.solver.step(phases, coupling, control)
        
        # All phases should be in [-π, π]
        assert np.all(new_phases >= -np.pi)
        assert np.all(new_phases <= np.pi)
    
    def test_input_validation(self):
        """Test input validation."""
        phases = np.array([0.0, 0.0, 0.0])
        coupling = np.zeros((3, 3))
        control = np.zeros(3)
        
        # Wrong phase vector size
        with pytest.raises(ValueError):
            self.solver.step(phases[:2], coupling, control)
        
        # Wrong coupling matrix size
        with pytest.raises(ValueError):
            wrong_coupling = np.zeros((2, 2))
            self.solver.step(phases, wrong_coupling, control)
        
        # Wrong control vector size
        with pytest.raises(ValueError):
            wrong_control = np.zeros(2)
            self.solver.step(phases, coupling, wrong_control)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])