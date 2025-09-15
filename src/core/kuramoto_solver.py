"""
Kuramoto model solver for synchronization of cognitive modules.

Implements numerical solution of:
dθᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ(t) sin(θⱼ - θᵢ - αᵢⱼ) + uᵢ(t)

Where:
- θᵢ(t) are module phases
- ωᵢ are natural frequencies
- Kᵢⱼ(t) is dynamic coupling matrix
- αᵢⱼ are phase lags
- uᵢ(t) are control signals

Uses 4th order Runge-Kutta method for stable integration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

from .state import KuramotoConfig


class KuramotoSolver:
    """
    Implements numerical solution of Kuramoto model for synchronization
    of cognitive modules using 4th order Runge-Kutta method.
    """

    def __init__(self, config: KuramotoConfig, module_order: List[str]):
        """
        Initialize Kuramoto solver.

        Args:
            config: Configuration object with Kuramoto parameters
            module_order: Ordered list of module names for indexing
        """
        self.config = config
        self.module_order = module_order
        self.num_modules = len(module_order)
        self.dt = config.time_step

        # Build frequency vector from configuration
        self.omega = np.array([config.natural_frequencies[name] for name in module_order])

        # Validate configuration
        self._validate_setup()

        # Initialize phase lag matrix (can be updated later)
        self.alpha_matrix = np.zeros((self.num_modules, self.num_modules))

        print(f"Kuramoto solver initialized for {self.num_modules} modules:")
        for i, name in enumerate(module_order):
            print(f"  {name}: ω = {self.omega[i]:.2f} Hz")

    def _validate_setup(self):
        """Validate solver setup and parameters."""
        if self.num_modules == 0:
            raise ValueError("No modules specified")

        if self.dt <= 0:
            raise ValueError("Time step must be positive")

        # Check for missing frequencies
        for name in self.module_order:
            if name not in self.config.natural_frequencies:
                raise ValueError(f"Missing frequency for module: {name}")

        # Stability warning for explicit integration
        max_omega = np.max(np.abs(self.omega))
        if self.dt * max_omega > 0.5:
            warnings.warn(
                f"Large time step {self.dt:.4f} with max frequency {max_omega:.2f}. "
                "Consider reducing time step for stability.",
                UserWarning,
            )

    def set_phase_lags(self, alpha_matrix: np.ndarray):
        """
        Set phase lag matrix αᵢⱼ.

        Args:
            alpha_matrix: Phase lag matrix (num_modules x num_modules)
        """
        if alpha_matrix.shape != (self.num_modules, self.num_modules):
            raise ValueError(
                f"Phase lag matrix shape {alpha_matrix.shape} "
                f"doesn't match number of modules {self.num_modules}"
            )

        self.alpha_matrix = alpha_matrix.copy()

    def build_coupling_matrix(
        self,
        base_strength: Optional[float] = None,
        modulation_factors: Optional[np.ndarray] = None,
        sparsity_mask: Optional[np.ndarray] = None,
        symmetrize: bool = True,
    ) -> np.ndarray:
        """
        Build dynamic coupling matrix Kᵢⱼ(t).

        Following equation: Kᵢⱼ(t) = K_base · C_window · T_relevance · C_permission

        Args:
            base_strength: Base coupling strength (uses config value if None)
            modulation_factors: Matrix of modulation factors (default: all ones)
            sparsity_mask: Binary mask for connection topology (default: all connected)
            symmetrize: Whether to symmetrize the coupling matrix

        Returns:
            Coupling matrix Kᵢⱼ(t)
        """
        if base_strength is None:
            base_strength = self.config.base_coupling_strength

        # Start with base coupling
        K = base_strength * np.ones((self.num_modules, self.num_modules))

        # Apply modulation factors if provided
        if modulation_factors is not None:
            if modulation_factors.shape != (self.num_modules, self.num_modules):
                raise ValueError(
                    f"Modulation factors shape {modulation_factors.shape} "
                    f"doesn't match number of modules {self.num_modules}"
                )
            K *= modulation_factors

        # Apply sparsity mask if provided
        if sparsity_mask is not None:
            if sparsity_mask.shape != (self.num_modules, self.num_modules):
                raise ValueError(
                    f"Sparsity mask shape {sparsity_mask.shape} "
                    f"doesn't match number of modules {self.num_modules}"
                )
            K *= sparsity_mask

        # Zero diagonal (no self-coupling)
        np.fill_diagonal(K, 0.0)

        # Symmetrize if requested
        if symmetrize:
            K = 0.5 * (K + K.T)

        # Clip extreme values for numerical stability
        K = np.clip(K, -10 * base_strength, 10 * base_strength)

        return K

    def _compute_derivatives(
        self, phases: np.ndarray, coupling_matrix: np.ndarray, control_signals: np.ndarray
    ) -> np.ndarray:
        """
        Compute phase derivatives dθ/dt for RK4 integration.

        Args:
            phases: Current phase vector θ(t)
            coupling_matrix: Coupling matrix K(t)
            control_signals: Control vector u(t)

        Returns:
            Derivative vector dθ/dt
        """
        # Compute phase differences: θⱼ - θᵢ - αᵢⱼ
        # Using broadcasting: phases[j] - phases[i] - alpha[i,j]
        phase_diff = phases[np.newaxis, :] - phases[:, np.newaxis] - self.alpha_matrix

        # Interaction term: Σⱼ Kᵢⱼ sin(θⱼ - θᵢ - αᵢⱼ)
        interaction_term = np.sum(coupling_matrix * np.sin(phase_diff), axis=1)

        # Total derivative: ωᵢ + interaction + control
        derivatives = self.omega + interaction_term + control_signals

        return derivatives

    def step(
        self, current_phases: np.ndarray, coupling_matrix: np.ndarray, control_signals: np.ndarray
    ) -> np.ndarray:
        """
        Perform one integration step using 4th order Runge-Kutta method.

        Args:
            current_phases: Current phase vector θ(t)
            coupling_matrix: Coupling matrix K(t)
            control_signals: Control vector u(t)

        Returns:
            Updated phase vector θ(t + Δt)
        """
        # Validate inputs
        if current_phases.shape != (self.num_modules,):
            raise ValueError(
                f"Phase vector shape {current_phases.shape} "
                f"doesn't match number of modules {self.num_modules}"
            )

        if coupling_matrix.shape != (self.num_modules, self.num_modules):
            raise ValueError(
                f"Coupling matrix shape {coupling_matrix.shape} "
                f"doesn't match number of modules {self.num_modules}"
            )

        if control_signals.shape != (self.num_modules,):
            raise ValueError(
                f"Control signals shape {control_signals.shape} "
                f"doesn't match number of modules {self.num_modules}"
            )

        # Ensure proper data types
        phases = current_phases.astype(np.float64)
        K = coupling_matrix.astype(np.float64)
        u = control_signals.astype(np.float64)

        # 4th order Runge-Kutta integration
        k1 = self._compute_derivatives(phases, K, u)
        k2 = self._compute_derivatives(phases + 0.5 * self.dt * k1, K, u)
        k3 = self._compute_derivatives(phases + 0.5 * self.dt * k2, K, u)
        k4 = self._compute_derivatives(phases + self.dt * k3, K, u)

        # Combined update
        next_phases = phases + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Wrap phases to [-π, π] to prevent numerical drift
        next_phases = np.mod(next_phases + np.pi, 2 * np.pi) - np.pi

        return next_phases

    def compute_order_parameter(self, phases: np.ndarray) -> Tuple[float, float]:
        """
        Compute Kuramoto order parameter R and mean phase Ψ.

        R = |1/N Σᵢ exp(iθᵢ)|
        Ψ = arg(1/N Σᵢ exp(iθᵢ))

        Args:
            phases: Phase vector θ(t)

        Returns:
            Tuple of (R, Ψ) where R ∈ [0,1] and Ψ ∈ [-π,π]
        """
        # Complex order parameter
        z = np.mean(np.exp(1j * phases))

        # Magnitude and phase
        R = np.abs(z)
        psi = np.angle(z)

        return R, psi

    def compute_pairwise_synchronization(self, phases: np.ndarray) -> np.ndarray:
        """
        Compute pairwise synchronization matrix.

        Sᵢⱼ = |exp(i(θᵢ - θⱼ))|

        Args:
            phases: Phase vector θ(t)

        Returns:
            Synchronization matrix (num_modules x num_modules)
        """
        # Phase difference matrix
        phase_diff = phases[:, np.newaxis] - phases[np.newaxis, :]

        # Synchronization strength
        sync_matrix = np.abs(np.exp(1j * phase_diff))

        return sync_matrix

    def compute_cluster_coherence(
        self, phases: np.ndarray, cluster_labels: np.ndarray
    ) -> Dict[int, float]:
        """
        Compute coherence within each cluster of modules.

        Args:
            phases: Phase vector θ(t)
            cluster_labels: Cluster assignment for each module

        Returns:
            Dictionary mapping cluster ID to coherence value
        """
        cluster_coherence = {}

        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            # Phases in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_phases = phases[cluster_mask]

            if len(cluster_phases) > 0:
                # Order parameter for this cluster
                R, _ = self.compute_order_parameter(cluster_phases)
                cluster_coherence[cluster_id] = R
            else:
                cluster_coherence[cluster_id] = 0.0

        return cluster_coherence

    def detect_synchronization_regimes(
        self, phases_history: np.ndarray, window_size: int = 100
    ) -> Dict[str, float]:
        """
        Analyze synchronization regimes from phase time series.

        Args:
            phases_history: Array of shape (time_steps, num_modules)
            window_size: Window size for moving average

        Returns:
            Dictionary with synchronization metrics
        """
        if phases_history.shape[1] != self.num_modules:
            raise ValueError(
                f"Phase history has {phases_history.shape[1]} modules, "
                f"expected {self.num_modules}"
            )

        # Compute order parameter over time
        R_history = []
        for t in range(phases_history.shape[0]):
            R, _ = self.compute_order_parameter(phases_history[t])
            R_history.append(R)

        R_history = np.array(R_history)

        # Compute metrics
        mean_R = np.mean(R_history)
        std_R = np.std(R_history)
        max_R = np.max(R_history)
        min_R = np.min(R_history)

        # Moving average for trend analysis
        if len(R_history) >= window_size:
            R_smooth = np.convolve(R_history, np.ones(window_size) / window_size, mode="valid")
            final_trend = R_smooth[-1] - R_smooth[0] if len(R_smooth) > 1 else 0.0
        else:
            final_trend = 0.0

        return {
            "mean_coherence": mean_R,
            "coherence_std": std_R,
            "max_coherence": max_R,
            "min_coherence": min_R,
            "final_trend": final_trend,
            "is_synchronized": mean_R > 0.7,  # Threshold for synchronization
            "is_stable": std_R < 0.1,  # Threshold for stability
        }

    def get_module_info(self) -> Dict[str, Dict]:
        """Return information about the modules and their parameters."""
        info = {}
        for i, name in enumerate(self.module_order):
            info[name] = {
                "index": i,
                "natural_frequency": self.omega[i],
                "coupling_row_sum": 0.0,  # Will be updated when coupling matrix is available
            }
        return info
