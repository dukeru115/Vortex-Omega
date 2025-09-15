"""
ESC-Kuramoto Integration Bridge - NFCS Semantic Synchronization
============================================================

Critical integration module implementing semantic synchronization through
ESC-Kuramoto coupling as described in Section 4.6 of the NFCS scientific foundation.

This module addresses the key scientific requirement: ESC output η(t) must
modulate Kuramoto coupling to achieve semantic coherence through neural field
synchronization.

Scientific Foundation:
- ESC η(t) → Kuramoto K_ij(t) modulation: K_ij(t) = K_base * f(η(t))
- Phase-Amplitude Coupling (PAC) for hierarchical semantic control
- Cross-frequency coupling for multi-scale semantic processing
- Semantic pattern synchronization through oscillatory dynamics

Mathematical Framework:
- Semantic Enhancement: K_ij(t) = K_base * (1 + α * tanh(β * η(t)))
- PAC Implementation: A_low(t) = A_base * cos(φ_high(t) + φ_offset)
- Cross-frequency Coupling: dθ_i/dt = ω_i + Σ K_ij(η) * sin(θ_j - θ_i) + u_semantic(t)

Created: September 14, 2025
Author: Team Ω - Neural Field Control Systems Research Group
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from abc import ABC, abstractmethod
import warnings
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import pearsonr
from collections import deque

# Import NFCS components
from ...core.state import SystemState, KuramotoConfig
from ...core.kuramoto_solver import KuramotoSolver
from ...modules.esc.esc_core import EchoSemanticConverter, ProcessingResult, ESCConfig

logger = logging.getLogger(__name__)


class CouplingMode(Enum):
    """ESC-Kuramoto coupling modes."""

    LINEAR = "linear"  # Linear η(t) → K_ij modulation
    NONLINEAR = "nonlinear"  # Nonlinear tanh-based modulation
    ADAPTIVE = "adaptive"  # Adaptive coupling based on semantic context
    PHASE_LOCKED = "phase_locked"  # Phase-locked semantic synchronization
    HIERARCHICAL = "hierarchical"  # Hierarchical multi-scale coupling


class SemanticSyncMode(Enum):
    """Semantic synchronization modes."""

    COHERENCE_MAXIMIZATION = "coherence_max"  # Maximize semantic coherence
    DIVERSITY_PRESERVATION = "diversity_preserve"  # Preserve semantic diversity
    BALANCED = "balanced"  # Balance coherence and diversity
    CONSTITUTIONAL = "constitutional"  # Constitutional compliance priority


@dataclass
class PACConfig:
    """Phase-Amplitude Coupling configuration."""

    low_freq_range: Tuple[float, float] = (1.0, 8.0)  # Low frequency range (Hz)
    high_freq_range: Tuple[float, float] = (30.0, 100.0)  # High frequency range (Hz)
    pac_strength: float = 0.5  # PAC coupling strength
    phase_offset: float = 0.0  # Phase offset for coupling
    window_size: int = 256  # Analysis window size


@dataclass
class SemanticCouplingConfig:
    """Semantic coupling configuration."""

    base_coupling_strength: float = 1.0  # Base coupling strength
    semantic_modulation_strength: float = 0.5  # η(t) modulation strength
    coupling_mode: CouplingMode = CouplingMode.ADAPTIVE
    sync_mode: SemanticSyncMode = SemanticSyncMode.BALANCED
    adaptation_rate: float = 0.01  # Adaptive coupling rate
    coherence_target: float = 0.7  # Target coherence level
    pac_config: PACConfig = field(default_factory=PACConfig)


@dataclass
class SemanticSyncMetrics:
    """Semantic synchronization metrics."""

    semantic_coherence: float  # Overall semantic coherence
    phase_locking_value: float  # Phase locking between modules
    pac_strength: float  # Phase-amplitude coupling strength
    cross_frequency_coupling: float  # Cross-frequency synchronization
    semantic_diversity: float  # Semantic diversity measure
    coupling_efficiency: float  # Coupling efficiency metric
    synchronization_stability: float  # Stability of synchronization
    computational_cost: float  # Computational overhead
    timestamp: float = field(default_factory=time.time)


@dataclass
class IntegrationResult:
    """Result of ESC-Kuramoto integration."""

    integration_id: str
    modified_coupling_matrix: np.ndarray  # Updated K_ij(t)
    semantic_sync_metrics: SemanticSyncMetrics  # Synchronization metrics
    esc_modulation_signal: np.ndarray  # ESC η(t) time series
    phase_evolution: np.ndarray  # Kuramoto phase evolution
    semantic_field_state: np.ndarray  # Semantic field state
    integration_quality: float  # Overall integration quality
    processing_time: float  # Processing duration
    warnings: List[str] = field(default_factory=list)  # Integration warnings


class ESCKuramotoBridge:
    """
    ESC-Kuramoto Integration Bridge

    Implements critical semantic synchronization through ESC-Kuramoto coupling
    as required by Section 4.6 of the NFCS scientific foundation.

    Key Functions:
    1. Semantic Modulation: η(t) → K_ij(t) coupling matrix modulation
    2. Phase-Amplitude Coupling: Multi-scale semantic synchronization
    3. Cross-frequency Analysis: Hierarchical semantic processing
    4. Synchronization Metrics: Real-time semantic coherence monitoring

    This bridge ensures that semantic processing (ESC) directly influences
    neural field synchronization (Kuramoto), enabling coherent semantic
    pattern formation through oscillatory dynamics.
    """

    def __init__(
        self,
        esc_system: EchoSemanticConverter,
        kuramoto_solver: KuramotoSolver,
        coupling_config: Optional[SemanticCouplingConfig] = None,
        enable_pac_analysis: bool = True,
        enable_real_time_adaptation: bool = True,
    ):
        """
        Initialize ESC-Kuramoto integration bridge.

        Args:
            esc_system: Echo-Semantic Converter instance
            kuramoto_solver: Kuramoto synchronization solver
            coupling_config: Semantic coupling configuration
            enable_pac_analysis: Enable Phase-Amplitude Coupling analysis
            enable_real_time_adaptation: Enable real-time coupling adaptation
        """
        self.esc_system = esc_system
        self.kuramoto_solver = kuramoto_solver
        self.coupling_config = coupling_config or SemanticCouplingConfig()
        self.enable_pac_analysis = enable_pac_analysis
        self.enable_real_time_adaptation = enable_real_time_adaptation

        # Integration state
        self.current_coupling_matrix: Optional[np.ndarray] = None
        self.semantic_history: deque = deque(maxlen=1000)
        self.phase_history: deque = deque(maxlen=1000)
        self.sync_metrics_history: List[SemanticSyncMetrics] = []

        # Performance tracking
        self.integration_stats = {
            "total_integrations": 0,
            "successful_integrations": 0,
            "avg_integration_time": 0.0,
            "avg_semantic_coherence": 0.0,
            "avg_coupling_efficiency": 0.0,
        }

        # Adaptation parameters
        self.adaptation_state = {
            "coupling_adaptation_rate": self.coupling_config.adaptation_rate,
            "coherence_error_history": deque(maxlen=100),
            "adaptation_gain": 1.0,
        }

        self.logger = logging.getLogger(f"{__name__}.ESCKuramotoBridge")
        self.logger.info("ESC-Kuramoto integration bridge initialized")

    def integrate_semantic_synchronization(
        self,
        tokens: List[str],
        current_phases: np.ndarray,
        base_coupling_matrix: np.ndarray,
        dt: float = 0.01,
    ) -> IntegrationResult:
        """
        Perform integrated ESC-Kuramoto semantic synchronization.

        This is the core method implementing η(t) → K_ij(t) modulation
        for semantic pattern synchronization.

        Args:
            tokens: Input tokens for ESC processing
            current_phases: Current Kuramoto phase states θ_i(t)
            base_coupling_matrix: Base coupling matrix K_base
            dt: Integration time step

        Returns:
            Integration result with modified coupling and metrics
        """
        start_time = time.time()
        integration_id = f"esc_kuramoto_{int(time.time() * 1000)}"
        warnings_list = []

        try:
            # Step 1: Process tokens through ESC to get η(t)
            esc_result = self.esc_system.process_sequence(tokens)
            eta_t = self._extract_semantic_coherence(esc_result)

            # Store semantic history
            self.semantic_history.append(
                {
                    "timestamp": time.time(),
                    "eta_t": eta_t,
                    "tokens": tokens,
                    "esc_metrics": esc_result.processing_metrics,
                }
            )

            # Step 2: Modulate Kuramoto coupling based on η(t)
            modified_coupling = self._modulate_coupling_matrix(
                eta_t, base_coupling_matrix, current_phases
            )

            # Step 3: Evolve Kuramoto phases with semantic coupling
            new_phases = self._evolve_semantic_phases(current_phases, modified_coupling, eta_t, dt)

            # Store phase history
            self.phase_history.append(
                {
                    "timestamp": time.time(),
                    "phases": new_phases.copy(),
                    "coupling_matrix": modified_coupling.copy(),
                }
            )

            # Step 4: Analyze semantic synchronization metrics
            sync_metrics = self._calculate_semantic_sync_metrics(
                eta_t, current_phases, new_phases, modified_coupling, esc_result
            )

            # Step 5: Perform Phase-Amplitude Coupling analysis
            pac_strength = 0.0
            cross_freq_coupling = 0.0
            if self.enable_pac_analysis:
                pac_strength, cross_freq_coupling = self._analyze_phase_amplitude_coupling(
                    new_phases, eta_t
                )
                sync_metrics.pac_strength = pac_strength
                sync_metrics.cross_frequency_coupling = cross_freq_coupling

            # Step 6: Real-time adaptation of coupling parameters
            if self.enable_real_time_adaptation:
                self._adapt_coupling_parameters(sync_metrics)

            # Step 7: Generate semantic field state
            semantic_field_state = self._generate_semantic_field_state(
                esc_result, new_phases, modified_coupling
            )

            # Step 8: Calculate integration quality
            integration_quality = self._calculate_integration_quality(sync_metrics)

            # Create integration result
            processing_time = time.time() - start_time
            result = IntegrationResult(
                integration_id=integration_id,
                modified_coupling_matrix=modified_coupling,
                semantic_sync_metrics=sync_metrics,
                esc_modulation_signal=np.array([eta_t]),
                phase_evolution=new_phases,
                semantic_field_state=semantic_field_state,
                integration_quality=integration_quality,
                processing_time=processing_time,
                warnings=warnings_list,
            )

            # Update statistics
            self._update_integration_statistics(result, success=True)

            # Store current state
            self.current_coupling_matrix = modified_coupling
            self.sync_metrics_history.append(sync_metrics)

            self.logger.debug(
                f"Semantic synchronization completed: "
                f"coherence={sync_metrics.semantic_coherence:.3f}, "
                f"efficiency={sync_metrics.coupling_efficiency:.3f}, "
                f"time={processing_time:.3f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"ESC-Kuramoto integration failed: {e}")
            self._update_integration_statistics(None, success=False)
            raise

    def _extract_semantic_coherence(self, esc_result: ProcessingResult) -> float:
        """Extract semantic coherence η(t) from ESC processing result."""
        try:
            # Use ESC processing metrics to compute semantic coherence
            metrics = esc_result.processing_metrics

            # Base coherence on constitutional compliance and processing quality
            constitutional_score = metrics.get("constitutional_compliance", 0.5)
            processing_quality = metrics.get("processing_quality", 0.5)
            semantic_density = metrics.get("semantic_density", 0.5)

            # Combine metrics into coherence measure η(t)
            eta_t = 0.4 * constitutional_score + 0.3 * processing_quality + 0.3 * semantic_density

            # Ensure η(t) is in valid range [0, 1]
            eta_t = np.clip(eta_t, 0.0, 1.0)

            return eta_t

        except Exception as e:
            self.logger.warning(f"Failed to extract semantic coherence: {e}")
            return 0.5  # Default neutral coherence

    def _modulate_coupling_matrix(
        self, eta_t: float, base_coupling: np.ndarray, current_phases: np.ndarray
    ) -> np.ndarray:
        """
        Modulate Kuramoto coupling matrix based on ESC semantic coherence.

        Implements: K_ij(t) = K_base * f(η(t))
        """
        try:
            modified_coupling = base_coupling.copy()

            if self.coupling_config.coupling_mode == CouplingMode.LINEAR:
                # Linear modulation: K_ij(t) = K_base * (1 + α * η(t))
                alpha = self.coupling_config.semantic_modulation_strength
                modulation_factor = 1.0 + alpha * eta_t
                modified_coupling *= modulation_factor

            elif self.coupling_config.coupling_mode == CouplingMode.NONLINEAR:
                # Nonlinear tanh modulation: K_ij(t) = K_base * (1 + α * tanh(β * η(t)))
                alpha = self.coupling_config.semantic_modulation_strength
                beta = 2.0  # Nonlinearity parameter
                modulation_factor = 1.0 + alpha * np.tanh(beta * (eta_t - 0.5))
                modified_coupling *= modulation_factor

            elif self.coupling_config.coupling_mode == CouplingMode.ADAPTIVE:
                # Adaptive modulation based on phase synchronization state
                phase_coherence = self._calculate_phase_coherence(current_phases)
                coherence_error = self.coupling_config.coherence_target - phase_coherence

                # Adaptive gain based on error
                adaptive_gain = (
                    1.0 + self.adaptation_state["adaptation_gain"] * coherence_error * eta_t
                )
                modified_coupling *= adaptive_gain

            elif self.coupling_config.coupling_mode == CouplingMode.PHASE_LOCKED:
                # Phase-locked modulation for strong semantic synchronization
                phase_sync_strength = self._calculate_phase_sync_strength(current_phases)
                lock_factor = 1.0 + (1.0 - phase_sync_strength) * eta_t
                modified_coupling *= lock_factor

            elif self.coupling_config.coupling_mode == CouplingMode.HIERARCHICAL:
                # Hierarchical modulation with frequency-dependent coupling
                freq_weights = self._calculate_frequency_weights(current_phases)
                for i in range(modified_coupling.shape[0]):
                    for j in range(modified_coupling.shape[1]):
                        if i != j:
                            freq_factor = freq_weights[i] * freq_weights[j]
                            modified_coupling[i, j] *= 1.0 + eta_t * freq_factor

            # Ensure coupling matrix remains positive and stable
            modified_coupling = np.maximum(modified_coupling, 0.01)

            # Apply constitutional constraints if in constitutional mode
            if self.coupling_config.sync_mode == SemanticSyncMode.CONSTITUTIONAL:
                modified_coupling = self._apply_constitutional_constraints(modified_coupling)

            return modified_coupling

        except Exception as e:
            self.logger.warning(f"Coupling modulation failed: {e}")
            return base_coupling  # Return unmodified on error

    def _evolve_semantic_phases(
        self, current_phases: np.ndarray, coupling_matrix: np.ndarray, eta_t: float, dt: float
    ) -> np.ndarray:
        """Evolve Kuramoto phases with semantic coupling."""
        try:
            n_oscillators = len(current_phases)

            # Calculate standard Kuramoto derivatives
            derivatives = np.zeros_like(current_phases)

            for i in range(n_oscillators):
                # Natural frequency component
                omega_i = self.kuramoto_solver.omega[i]
                derivatives[i] = omega_i

                # Coupling term with semantic modulation
                for j in range(n_oscillators):
                    if i != j:
                        coupling_strength = coupling_matrix[i, j]
                        phase_diff = current_phases[j] - current_phases[i]

                        # Standard Kuramoto coupling
                        coupling_term = coupling_strength * np.sin(phase_diff)

                        # Semantic enhancement
                        semantic_enhancement = (
                            eta_t * self.coupling_config.semantic_modulation_strength
                        )
                        enhanced_coupling = coupling_term * (1.0 + semantic_enhancement)

                        derivatives[i] += enhanced_coupling

            # Integrate using Euler method (can be upgraded to RK4)
            new_phases = current_phases + dt * derivatives

            # Wrap phases to [-π, π]
            new_phases = np.mod(new_phases + np.pi, 2 * np.pi) - np.pi

            return new_phases

        except Exception as e:
            self.logger.warning(f"Phase evolution failed: {e}")
            return current_phases  # Return unchanged on error

    def _calculate_semantic_sync_metrics(
        self,
        eta_t: float,
        old_phases: np.ndarray,
        new_phases: np.ndarray,
        coupling_matrix: np.ndarray,
        esc_result: ProcessingResult,
    ) -> SemanticSyncMetrics:
        """Calculate comprehensive semantic synchronization metrics."""
        try:
            # 1. Semantic coherence (based on η(t) and phase alignment)
            phase_coherence = self._calculate_phase_coherence(new_phases)
            semantic_coherence = 0.6 * eta_t + 0.4 * phase_coherence

            # 2. Phase locking value between oscillators
            phase_locking_value = self._calculate_phase_locking_value(new_phases)

            # 3. Phase-amplitude coupling (if enabled)
            pac_strength = 0.0
            if self.enable_pac_analysis:
                pac_strength = self._calculate_pac_strength(new_phases, eta_t)

            # 4. Cross-frequency coupling
            cross_freq_coupling = self._calculate_cross_frequency_coupling(new_phases)

            # 5. Semantic diversity (measure of information preservation)
            semantic_diversity = self._calculate_semantic_diversity(esc_result)

            # 6. Coupling efficiency
            coupling_efficiency = self._calculate_coupling_efficiency(
                old_phases, new_phases, coupling_matrix
            )

            # 7. Synchronization stability
            sync_stability = self._calculate_synchronization_stability(new_phases)

            # 8. Computational cost estimate
            computational_cost = len(new_phases) ** 2 * 1e-6  # Simplified estimate

            return SemanticSyncMetrics(
                semantic_coherence=semantic_coherence,
                phase_locking_value=phase_locking_value,
                pac_strength=pac_strength,
                cross_frequency_coupling=cross_freq_coupling,
                semantic_diversity=semantic_diversity,
                coupling_efficiency=coupling_efficiency,
                synchronization_stability=sync_stability,
                computational_cost=computational_cost,
            )

        except Exception as e:
            self.logger.warning(f"Metric calculation failed: {e}")
            return SemanticSyncMetrics(
                semantic_coherence=0.5,
                phase_locking_value=0.5,
                pac_strength=0.0,
                cross_frequency_coupling=0.0,
                semantic_diversity=0.5,
                coupling_efficiency=0.5,
                synchronization_stability=0.5,
                computational_cost=0.0,
            )

    def _analyze_phase_amplitude_coupling(
        self, phases: np.ndarray, eta_t: float
    ) -> Tuple[float, float]:
        """Analyze Phase-Amplitude Coupling for hierarchical semantic control."""
        if not self.enable_pac_analysis or len(self.phase_history) < 10:
            return 0.0, 0.0

        try:
            # Extract phase time series from history
            phase_series = np.array([entry["phases"] for entry in list(self.phase_history)[-50:]])

            if phase_series.shape[0] < 20:  # Need sufficient history
                return 0.0, 0.0

            pac_values = []
            cross_freq_values = []

            # Analyze PAC for each oscillator
            for osc_idx in range(phases.shape[0]):
                phase_ts = phase_series[:, osc_idx]

                # Extract low and high frequency components
                low_freq_component = self._extract_frequency_band(
                    phase_ts, self.coupling_config.pac_config.low_freq_range
                )
                high_freq_component = self._extract_frequency_band(
                    phase_ts, self.coupling_config.pac_config.high_freq_range
                )

                if low_freq_component is not None and high_freq_component is not None:
                    # Calculate PAC strength
                    pac_strength = self._calculate_pac_modulation_index(
                        low_freq_component, high_freq_component
                    )
                    pac_values.append(pac_strength)

                    # Calculate cross-frequency coupling
                    cross_freq = self._calculate_cross_frequency_index(
                        low_freq_component, high_freq_component
                    )
                    cross_freq_values.append(cross_freq)

            avg_pac = np.mean(pac_values) if pac_values else 0.0
            avg_cross_freq = np.mean(cross_freq_values) if cross_freq_values else 0.0

            return avg_pac, avg_cross_freq

        except Exception as e:
            self.logger.warning(f"PAC analysis failed: {e}")
            return 0.0, 0.0

    def _adapt_coupling_parameters(self, sync_metrics: SemanticSyncMetrics):
        """Adapt coupling parameters based on synchronization performance."""
        if not self.enable_real_time_adaptation:
            return

        try:
            # Calculate coherence error
            coherence_error = (
                self.coupling_config.coherence_target - sync_metrics.semantic_coherence
            )
            self.adaptation_state["coherence_error_history"].append(coherence_error)

            # Adaptive gain adjustment using PI controller
            if len(self.adaptation_state["coherence_error_history"]) > 5:
                error_history = list(self.adaptation_state["coherence_error_history"])

                # Proportional term
                P_gain = 0.1
                proportional = P_gain * coherence_error

                # Integral term
                I_gain = 0.01
                integral = I_gain * np.sum(error_history[-10:])

                # Update adaptation gain
                gain_adjustment = proportional + integral
                self.adaptation_state["adaptation_gain"] = np.clip(
                    self.adaptation_state["adaptation_gain"] + gain_adjustment, 0.1, 2.0
                )

                # Adapt coupling strength if performance is poor
                if sync_metrics.coupling_efficiency < 0.5:
                    self.coupling_config.semantic_modulation_strength *= 1.1
                elif sync_metrics.coupling_efficiency > 0.9:
                    self.coupling_config.semantic_modulation_strength *= 0.95

                # Keep modulation strength in valid range
                self.coupling_config.semantic_modulation_strength = np.clip(
                    self.coupling_config.semantic_modulation_strength, 0.1, 2.0
                )

        except Exception as e:
            self.logger.warning(f"Parameter adaptation failed: {e}")

    def _generate_semantic_field_state(
        self, esc_result: ProcessingResult, phases: np.ndarray, coupling_matrix: np.ndarray
    ) -> np.ndarray:
        """Generate semantic field state from ESC and Kuramoto integration."""
        try:
            # Combine ESC semantic field with Kuramoto phase information
            n_modules = len(phases)
            field_size = 32  # Default field size

            # Create base field from phase information
            semantic_field = np.zeros((field_size, field_size), dtype=complex)

            # Map each module's phase to spatial location
            for i, phase in enumerate(phases):
                # Spatial position based on module index
                x = int((i % int(np.sqrt(n_modules))) * field_size / int(np.sqrt(n_modules)))
                y = int((i // int(np.sqrt(n_modules))) * field_size / int(np.sqrt(n_modules)))

                # Ensure indices are within bounds
                x = min(x, field_size - 1)
                y = min(y, field_size - 1)

                # Create complex field value from phase and coupling strength
                magnitude = np.mean(coupling_matrix[i, :])
                semantic_field[x, y] = magnitude * np.exp(1j * phase)

            # Smooth field using convolution
            if np.any(semantic_field):
                from scipy.ndimage import gaussian_filter

                real_part = gaussian_filter(semantic_field.real, sigma=1.0)
                imag_part = gaussian_filter(semantic_field.imag, sigma=1.0)
                semantic_field = real_part + 1j * imag_part

            return semantic_field

        except Exception as e:
            self.logger.warning(f"Semantic field generation failed: {e}")
            return np.zeros((32, 32), dtype=complex)

    def _calculate_integration_quality(self, sync_metrics: SemanticSyncMetrics) -> float:
        """Calculate overall integration quality score."""
        try:
            # Weighted combination of key metrics
            weights = {
                "semantic_coherence": 0.3,
                "phase_locking_value": 0.2,
                "coupling_efficiency": 0.2,
                "synchronization_stability": 0.15,
                "semantic_diversity": 0.1,
                "pac_strength": 0.05,
            }

            quality_score = (
                weights["semantic_coherence"] * sync_metrics.semantic_coherence
                + weights["phase_locking_value"] * sync_metrics.phase_locking_value
                + weights["coupling_efficiency"] * sync_metrics.coupling_efficiency
                + weights["synchronization_stability"] * sync_metrics.synchronization_stability
                + weights["semantic_diversity"] * sync_metrics.semantic_diversity
                + weights["pac_strength"] * sync_metrics.pac_strength
            )

            return np.clip(quality_score, 0.0, 1.0)

        except Exception as e:
            self.logger.warning(f"Quality calculation failed: {e}")
            return 0.5

    # Additional helper methods for mathematical computations
    def _calculate_phase_coherence(self, phases: np.ndarray) -> float:
        """Calculate phase coherence R = |⟨e^(iθ)⟩|."""
        try:
            complex_phases = np.exp(1j * phases)
            coherence = np.abs(np.mean(complex_phases))
            return coherence
        except:
            return 0.0

    def _calculate_phase_locking_value(self, phases: np.ndarray) -> float:
        """Calculate phase locking value between oscillators."""
        try:
            n = len(phases)
            if n < 2:
                return 0.0

            plv_sum = 0.0
            count = 0

            for i in range(n):
                for j in range(i + 1, n):
                    phase_diff = phases[i] - phases[j]
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    plv_sum += plv
                    count += 1

            return plv_sum / count if count > 0 else 0.0
        except:
            return 0.0

    def _update_integration_statistics(self, result: Optional[IntegrationResult], success: bool):
        """Update integration performance statistics."""
        self.integration_stats["total_integrations"] += 1

        if success and result is not None:
            self.integration_stats["successful_integrations"] += 1

            # Update averages
            total_successful = self.integration_stats["successful_integrations"]

            # Processing time
            current_avg_time = self.integration_stats["avg_integration_time"]
            new_avg_time = (
                (current_avg_time * (total_successful - 1)) + result.processing_time
            ) / total_successful
            self.integration_stats["avg_integration_time"] = new_avg_time

            # Semantic coherence
            current_avg_coherence = self.integration_stats["avg_semantic_coherence"]
            new_coherence = result.semantic_sync_metrics.semantic_coherence
            new_avg_coherence = (
                (current_avg_coherence * (total_successful - 1)) + new_coherence
            ) / total_successful
            self.integration_stats["avg_semantic_coherence"] = new_avg_coherence

            # Coupling efficiency
            current_avg_efficiency = self.integration_stats["avg_coupling_efficiency"]
            new_efficiency = result.semantic_sync_metrics.coupling_efficiency
            new_avg_efficiency = (
                (current_avg_efficiency * (total_successful - 1)) + new_efficiency
            ) / total_successful
            self.integration_stats["avg_coupling_efficiency"] = new_avg_efficiency

    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        success_rate = self.integration_stats["successful_integrations"] / max(
            1, self.integration_stats["total_integrations"]
        )

        return {
            **self.integration_stats,
            "success_rate": success_rate,
            "semantic_history_length": len(self.semantic_history),
            "phase_history_length": len(self.phase_history),
            "sync_metrics_history_length": len(self.sync_metrics_history),
            "current_adaptation_gain": self.adaptation_state["adaptation_gain"],
            "current_coupling_mode": self.coupling_config.coupling_mode.value,
            "current_sync_mode": self.coupling_config.sync_mode.value,
        }

    def reset_integration_state(self):
        """Reset integration state and clear history."""
        self.semantic_history.clear()
        self.phase_history.clear()
        self.sync_metrics_history.clear()
        self.current_coupling_matrix = None
        self.adaptation_state["coherence_error_history"].clear()
        self.adaptation_state["adaptation_gain"] = 1.0

        self.logger.info("ESC-Kuramoto integration state reset")
