"""
Complex Ginzburg-Landau (CGL) equation solver using split-step Fourier method.

Implements the numerical solution of:
∂φ/∂t = φ + (1 + ic₁)∇²φ - (1 + ic₃)|φ|²φ + u(x,t)

Where:
- φ(x,t) is the complex neural field
- c₁ is the linear dispersion parameter
- c₃ is the nonlinear self-action parameter
- u(x,t) is the external control field

The split-step method separates linear and nonlinear parts for efficient computation.
"""

import numpy as np
from typing import Tuple, Optional
import warnings

from .state import CGLConfig


class CGLSolver:
    """
    Implements numerical solution of Complex Ginzburg-Landau equation
    using split-step Fourier method.

    The split-step method works by:
    1. Linear step (Δt/2): Solve linear part in Fourier space
    2. Nonlinear step (Δt): Solve nonlinear part in real space
    3. Linear step (Δt/2): Second linear step in Fourier space
    """

    def __init__(self, config: CGLConfig):
        """
        Initialize CGL solver with given parameters.

        Args:
            config: Configuration object containing CGL parameters.
        """
        self.config = config
        self.c1 = config.c1
        self.c3 = config.c3
        self.dt = config.time_step
        self.grid_size = config.grid_size
        self.spatial_extent = config.spatial_extent

        # Precompute spatial grids and wave vectors
        self._setup_grids()
        self._setup_fourier_operators()

        # Stability check
        self._check_stability()

    def _setup_grids(self):
        """Setup spatial coordinate grids."""
        nx, ny = self.grid_size
        Lx, Ly = self.spatial_extent

        # Real space coordinates
        self.x = np.linspace(-Lx / 2, Lx / 2, nx)
        self.y = np.linspace(-Ly / 2, Ly / 2, ny)
        self.dx = Lx / nx
        self.dy = Ly / ny

        # Create 2D coordinate meshes
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

    def _setup_fourier_operators(self):
        """Setup Fourier space wave vectors and operators."""
        nx, ny = self.grid_size
        Lx, Ly = self.spatial_extent

        # Wave number grids (for FFT with fftshift)
        kx_1d = 2 * np.pi * np.fft.fftfreq(nx, self.dx)
        ky_1d = 2 * np.pi * np.fft.fftfreq(ny, self.dy)

        # 2D wave vector grids
        self.kx, self.ky = np.meshgrid(kx_1d, ky_1d, indexing="ij")

        # Laplacian operator in Fourier space: ∇² → -k²
        self.k_squared = self.kx**2 + self.ky**2

        # Linear operator: φ + (1 + ic₁)∇²φ → φ - (1 + ic₁)k²φ
        # Time evolution operator: exp(dt/2 * [1 - (1 + ic₁)k²])
        linear_operator = 1.0 - (1.0 + 1j * self.c1) * self.k_squared

        # Exponential operators for half and full time steps
        self.exp_linear_half = np.exp(0.5 * self.dt * linear_operator)
        self.exp_linear_full = np.exp(self.dt * linear_operator)

    def _check_stability(self):
        """Check numerical stability conditions."""
        # CFL condition estimate for the nonlinear term
        max_k = np.sqrt(np.max(self.k_squared))
        cfl_linear = self.dt * max_k**2

        if cfl_linear > 0.5:
            warnings.warn(
                f"Linear CFL condition may be violated: {cfl_linear:.3f} > 0.5. "
                f"Consider reducing time step from {self.dt} to {0.5/max_k**2:.6f}",
                UserWarning,
            )

        # Benjamin-Feir instability check
        if self.c1 * self.c3 > 0:
            warnings.warn(
                f"Benjamin-Feir instability condition: c₁*c₃ = {self.c1*self.c3:.3f} > 0. "
                "System may develop modulated waves and defects.",
                UserWarning,
            )

    def _linear_step(self, field_fft: np.ndarray, half_step: bool = True) -> np.ndarray:
        """
        Perform linear evolution step in Fourier space.

        Args:
            field_fft: Field in Fourier space
            half_step: If True, evolve for dt/2, else for dt

        Returns:
            Evolved field in Fourier space
        """
        if half_step:
            return field_fft * self.exp_linear_half
        else:
            return field_fft * self.exp_linear_full

    def _nonlinear_step(self, field: np.ndarray, control_field: np.ndarray) -> np.ndarray:
        """
        Perform nonlinear evolution step in real space.

        Solves: ∂φ/∂t = -(1 + ic₃)|φ|²φ + u(x,t)

        Args:
            field: Complex field in real space
            control_field: External control field u(x,t)

        Returns:
            Field after nonlinear evolution
        """
        # Nonlinear term: -(1 + ic₃)|φ|²φ
        field_magnitude_squared = np.abs(field) ** 2
        nonlinear_term = -(1.0 + 1j * self.c3) * field_magnitude_squared * field

        # Total RHS: nonlinear + control
        rhs = nonlinear_term + control_field

        # Simple explicit Euler step for nonlinear part
        # For better accuracy, could use RK4 or exponential integrator
        return field + self.dt * rhs

    def step(self, current_field: np.ndarray, control_field: np.ndarray) -> np.ndarray:
        """
        Perform one time step integration using split-step method.

        Args:
            current_field: Current neural field φ(x,t)
            control_field: Control field u(x,t)

        Returns:
            Updated neural field φ(x, t + Δt)
        """
        # Validate input
        if current_field.shape != self.grid_size:
            raise ValueError(
                f"Field shape {current_field.shape} doesn't match " f"grid size {self.grid_size}"
            )

        if control_field.shape != self.grid_size:
            raise ValueError(
                f"Control field shape {control_field.shape} doesn't match "
                f"grid size {self.grid_size}"
            )

        # Ensure complex data type
        field = current_field.astype(np.complex128)
        control = control_field.astype(np.complex128)

        # Step 1: First linear half-step in Fourier space
        field_fft = np.fft.fft2(field)
        field_fft = self._linear_step(field_fft, half_step=True)
        field = np.fft.ifft2(field_fft)

        # Step 2: Nonlinear step in real space
        field = self._nonlinear_step(field, control)

        # Step 3: Second linear half-step in Fourier space
        field_fft = np.fft.fft2(field)
        field_fft = self._linear_step(field_fft, half_step=True)
        field = np.fft.ifft2(field_fft)

        return field

    def create_initial_condition(
        self, pattern: str = "random_noise", amplitude: float = 0.1, **kwargs
    ) -> np.ndarray:
        """
        Create initial conditions for the neural field.

        Args:
            pattern: Type of initial pattern
                - "random_noise": Small amplitude random noise
                - "plane_wave": Plane wave with given k-vector
                - "spiral": Spiral wave pattern
                - "gaussian": Gaussian blob
            amplitude: Amplitude of the pattern
            **kwargs: Additional parameters for specific patterns

        Returns:
            Initial field configuration
        """
        nx, ny = self.grid_size

        if pattern == "random_noise":
            # Small random perturbation
            real_part = amplitude * np.random.randn(nx, ny)
            imag_part = amplitude * np.random.randn(nx, ny)
            return real_part + 1j * imag_part

        elif pattern == "plane_wave":
            # Plane wave: φ = A * exp(i k·r)
            kx = kwargs.get("kx", 1.0)
            ky = kwargs.get("ky", 0.0)
            return amplitude * np.exp(1j * (kx * self.X + ky * self.Y))

        elif pattern == "spiral":
            # Spiral wave pattern
            r = np.sqrt(self.X**2 + self.Y**2)
            theta = np.arctan2(self.Y, self.X)
            m = kwargs.get("m", 1)  # Topological charge
            phase = m * theta + kwargs.get("k_radial", 0.5) * r
            return amplitude * r * np.exp(1j * phase)

        elif pattern == "gaussian":
            # Gaussian blob
            sigma = kwargs.get("sigma", 1.0)
            x0 = kwargs.get("x0", 0.0)
            y0 = kwargs.get("y0", 0.0)
            r_squared = (self.X - x0) ** 2 + (self.Y - y0) ** 2
            return amplitude * np.exp(-r_squared / (2 * sigma**2))

        else:
            raise ValueError(f"Unknown initial pattern: {pattern}")

    def compute_energy(self, field: np.ndarray) -> float:
        """
        Compute total energy of the field.

        E = ∫ [|φ|² - ½|∇φ|² - ½|φ|⁴] dx dy

        Args:
            field: Complex neural field

        Returns:
            Total energy
        """
        # Field magnitude squared
        phi_squared = np.abs(field) ** 2

        # Gradient energy (in Fourier space for accuracy)
        field_fft = np.fft.fft2(field)
        grad_phi_squared = np.real(np.fft.ifft2(self.k_squared * field_fft * np.conj(field_fft)))

        # Quartic term
        phi_fourth = phi_squared**2

        # Energy density
        energy_density = phi_squared - 0.5 * grad_phi_squared - 0.5 * phi_fourth

        # Integrate over space
        return np.sum(energy_density) * self.dx * self.dy

    def get_grid_info(self) -> dict:
        """Return information about the computational grid."""
        return {
            "grid_size": self.grid_size,
            "spatial_extent": self.spatial_extent,
            "dx": self.dx,
            "dy": self.dy,
            "x_range": [self.x[0], self.x[-1]],
            "y_range": [self.y[0], self.y[-1]],
            "max_k": np.sqrt(np.max(self.k_squared)),
            "time_step": self.dt,
        }
