"""
Tests for Complex Ginzburg-Landau solver.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.cgl_solver import CGLSolver
from core.state import CGLConfig


class TestCGLSolver:
    """Test cases for CGL solver."""

    def setup_method(self):
        """Setup test configuration."""
        self.config = CGLConfig(
            c1=0.5, c3=1.0, grid_size=(32, 32), time_step=0.01, spatial_extent=(4.0, 4.0)
        )
        self.solver = CGLSolver(self.config)

    def test_solver_initialization(self):
        """Test solver initialization."""
        assert self.solver.c1 == 0.5
        assert self.solver.c3 == 1.0
        assert self.solver.dt == 0.01
        assert self.solver.grid_size == (32, 32)

        # Check that grids are properly set up
        assert self.solver.X.shape == (32, 32)
        assert self.solver.Y.shape == (32, 32)
        assert hasattr(self.solver, "k_squared")

    def test_create_initial_conditions(self):
        """Test creation of initial conditions."""
        # Random noise
        field = self.solver.create_initial_condition("random_noise", amplitude=0.1)
        assert field.shape == (32, 32)
        assert field.dtype == np.complex128
        assert np.max(np.abs(field)) <= 0.15  # Some margin for random

        # Plane wave
        field = self.solver.create_initial_condition("plane_wave", amplitude=1.0, kx=1.0, ky=0.0)
        assert field.shape == (32, 32)
        assert field.dtype == np.complex128
        assert np.allclose(np.abs(field), 1.0)

        # Gaussian
        field = self.solver.create_initial_condition("gaussian", amplitude=1.0, sigma=0.5)
        assert field.shape == (32, 32)
        assert np.max(np.abs(field)) <= 1.0

    def test_single_step_evolution(self):
        """Test single time step evolution."""
        # Initial field
        field = self.solver.create_initial_condition("gaussian", amplitude=0.5)
        control = np.zeros_like(field)

        # Evolve one step
        field_new = self.solver.step(field, control)

        # Check output properties
        assert field_new.shape == field.shape
        assert field_new.dtype == np.complex128
        assert np.isfinite(field_new).all()

        # Field should change (unless it's a perfect solution)
        assert not np.allclose(field, field_new, atol=1e-10)

    def test_conservation_properties(self):
        """Test approximate energy conservation for small systems."""
        # Small amplitude field (linear regime)
        field = self.solver.create_initial_condition("gaussian", amplitude=0.1, sigma=1.0)
        control = np.zeros_like(field)

        initial_energy = self.solver.compute_energy(field)

        # Evolve several steps
        for _ in range(5):
            field = self.solver.step(field, control)

        final_energy = self.solver.compute_energy(field)

        # Energy should not change dramatically in linear regime
        relative_change = abs(final_energy - initial_energy) / abs(initial_energy)
        assert relative_change < 1.0  # Allow some change due to numerical errors

    def test_control_field_effect(self):
        """Test that control field affects evolution."""
        field = self.solver.create_initial_condition("gaussian", amplitude=0.2)

        # Evolution without control
        field1 = self.solver.step(field.copy(), np.zeros_like(field))

        # Evolution with control
        control = 0.1 * np.ones_like(field)
        field2 = self.solver.step(field.copy(), control)

        # Results should be different
        assert not np.allclose(field1, field2, atol=1e-10)

    def test_grid_info(self):
        """Test grid information method."""
        info = self.solver.get_grid_info()

        assert info["grid_size"] == (32, 32)
        assert info["spatial_extent"] == (4.0, 4.0)
        assert info["time_step"] == 0.01
        assert "dx" in info
        assert "dy" in info
        assert "max_k" in info

    def test_input_validation(self):
        """Test input validation."""
        field = np.ones((32, 32), dtype=np.complex128)

        # Wrong shape control field
        with pytest.raises(ValueError):
            wrong_control = np.zeros((16, 16), dtype=np.complex128)
            self.solver.step(field, wrong_control)

        # Wrong shape input field
        with pytest.raises(ValueError):
            wrong_field = np.zeros((16, 16), dtype=np.complex128)
            control = np.zeros((32, 32), dtype=np.complex128)
            self.solver.step(wrong_field, control)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
