"""
Enhanced Kuramoto Module 1.4 - Advanced Signal Control
======================================================

Advanced Kuramoto oscillator network with:
- Adaptive coupling strength based on phase coherence
- Dynamic connectivity matrix with learning
- Signal-based control with constitutional compliance
- Multi-layer synchronization (local and global)
- Emergency protocol integration
- Real-time performance optimization
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
from collections import deque, defaultdict

from .kuramoto_solver import KuramotoSolver
from .state import KuramotoConfig, SystemState


class CouplingMode(Enum):
    """Coupling strength adaptation modes"""

    STATIC = "static"
    ADAPTIVE = "adaptive"
    LEARNING = "learning"
    CONSTITUTIONAL = "constitutional"


class SyncLevel(Enum):
    """Synchronization analysis levels"""

    LOCAL = "local"  # Within module clusters
    GLOBAL = "global"  # Across all modules
    CROSS = "cross"  # Cross-layer synchronization


@dataclass
class KuramotoSignal:
    """Control signal for Kuramoto network"""

    target_phases: np.ndarray
    coupling_adjustments: Dict[str, float] = field(default_factory=dict)
    frequency_shifts: Dict[str, float] = field(default_factory=dict)
    connectivity_mask: Optional[np.ndarray] = None
    priority: float = 1.0
    duration: float = 1.0
    constitutional_approved: bool = True


@dataclass
class SynchronizationMetrics:
    """Comprehensive synchronization analysis"""

    global_order_parameter: float
    local_order_parameters: Dict[str, float] = field(default_factory=dict)
    phase_coherence: float = 0.0
    frequency_locking: Dict[str, float] = field(default_factory=dict)
    coupling_effectiveness: float = 0.0
    sync_stability: float = 0.0
    energy_efficiency: float = 0.0


@dataclass
class AdaptiveState:
    """State for adaptive coupling learning"""

    coupling_history: deque = field(default_factory=lambda: deque(maxlen=100))
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    learning_rate: float = 0.01
    adaptation_momentum: float = 0.9
    stability_threshold: float = 0.95


class EnhancedKuramotoModule:
    """
    Enhanced Kuramoto Module 1.4 - Advanced Signal Control

    Features:
    - Adaptive coupling with constitutional compliance
    - Multi-level synchronization analysis
    - Learning-based network optimization
    - Emergency protocol integration
    - Real-time performance monitoring
    """

    def __init__(
        self,
        config: Optional[KuramotoConfig] = None,
        num_modules: int = 8,
        coupling_mode: CouplingMode = CouplingMode.ADAPTIVE,
    ):

        self.logger = logging.getLogger("EnhancedKuramoto")

        # Core configuration
        self.config = config or self._create_default_config(num_modules)
        self.num_modules = num_modules
        self.coupling_mode = coupling_mode

        # Core solver
        module_order = list(self.config.natural_frequencies.keys())
        self.solver = KuramotoSolver(self.config, module_order)

        # Enhanced state tracking
        self.current_phases = np.zeros(num_modules)
        self.current_frequencies = np.array(list(self.config.natural_frequencies.values()))
        self.coupling_matrix = np.eye(num_modules) * self.config.base_coupling_strength

        # Adaptive learning state
        self.adaptive_state = AdaptiveState()

        # Signal control
        self.active_signals: List[KuramotoSignal] = []
        self.signal_history: deque = deque(maxlen=1000)

        # Performance tracking
        self.metrics_history: deque = deque(maxlen=500)
        self.last_metrics: Optional[SynchronizationMetrics] = None

        # Asyncio-safe synchronization for real-time updates
        self.lock = None  # Will be initialized in async context
        self.running = False
        self.initialized = False

        # Emergency integration
        self.emergency_mode = False
        self.emergency_coupling_backup = None

        self.logger.info(
            f"Enhanced Kuramoto Module 1.4 initialized: {num_modules} modules, {coupling_mode.value} mode"
        )

    def _create_default_config(self, num_modules: int) -> KuramotoConfig:
        """Create default configuration for enhanced module"""

        # Natural frequencies with some variation
        base_freq = 1.0
        frequencies = {
            f"module_{i}": base_freq + 0.1 * np.sin(2 * np.pi * i / num_modules)
            for i in range(num_modules)
        }

        return KuramotoConfig(
            natural_frequencies=frequencies, base_coupling_strength=0.5, time_step=0.01
        )

    async def initialize(self) -> bool:
        """Initialize the enhanced Kuramoto module"""

        try:
            # Initialize asyncio lock if needed
            if self.lock is None:
                self.lock = asyncio.Lock()

            async with self.lock:
                # Validate base solver setup (KuramotoSolver doesn't need async initialization)
                self.solver._validate_setup()

                # Setup adaptive coupling matrix
                self._initialize_adaptive_coupling()

                # Initialize performance tracking
                self._reset_performance_tracking()

                self.initialized = True
                self.logger.info("Enhanced Kuramoto Module 1.4 successfully initialized")

                return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced Kuramoto: {e}")
            return False

    def _initialize_adaptive_coupling(self):
        """Initialize adaptive coupling matrix with constitutional constraints"""

        # Start with small-world topology
        for i in range(self.num_modules):
            for j in range(self.num_modules):
                if i != j:
                    # Distance in circular arrangement
                    dist = min(abs(i - j), self.num_modules - abs(i - j))

                    if dist == 1:  # Nearest neighbors
                        self.coupling_matrix[i, j] = self.config.base_coupling_strength
                    elif dist == 2:  # Second neighbors
                        self.coupling_matrix[i, j] = self.config.base_coupling_strength * 0.5
                    elif np.random.random() < 0.1:  # Random long-range connections
                        self.coupling_matrix[i, j] = self.config.base_coupling_strength * 0.2
                    else:
                        self.coupling_matrix[i, j] = 0.0

    def apply_control_signal(self, signal: KuramotoSignal) -> bool:
        """Apply control signal with constitutional validation"""

        try:
            with self.lock:
                # Constitutional compliance check
                if not signal.constitutional_approved:
                    self.logger.warning("Rejecting non-constitutional control signal")
                    return False

                # Validate signal parameters
                if not self._validate_signal(signal):
                    self.logger.warning("Invalid signal parameters")
                    return False

                # Apply signal
                self.active_signals.append(signal)
                self.signal_history.append(
                    {
                        "timestamp": time.time(),
                        "signal_type": "control",
                        "priority": signal.priority,
                        "approved": True,
                    }
                )

                self.logger.debug(f"Applied control signal with priority {signal.priority}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to apply control signal: {e}")
            return False

    def _validate_signal(self, signal: KuramotoSignal) -> bool:
        """Validate signal parameters for safety"""

        # Check target phases bounds
        if np.any(np.abs(signal.target_phases) > 2 * np.pi):
            return False

        # Check coupling adjustments are reasonable
        for module, adjustment in signal.coupling_adjustments.items():
            if abs(adjustment) > 2.0:  # Max 200% adjustment
                return False

        # Check frequency shifts are reasonable
        for module, shift in signal.frequency_shifts.items():
            if abs(shift) > 1.0:  # Max 1 Hz shift
                return False

        return True

    async def step(self, control_signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute one integration step with enhanced control"""

        try:
            with self.lock:
                if not self.initialized:
                    raise RuntimeError("Module not initialized")

                # Process active control signals
                effective_coupling, effective_frequencies = self._process_control_signals()

                # Adaptive coupling adjustment
                if self.coupling_mode in [CouplingMode.ADAPTIVE, CouplingMode.LEARNING]:
                    effective_coupling = self._adapt_coupling(effective_coupling)

                # Update solver state
                self.solver.coupling_matrix = effective_coupling
                self.solver.natural_frequencies = effective_frequencies

                # Execute step
                step_result = await self.solver.step()

                # Update internal state
                if "phases" in step_result:
                    self.current_phases = step_result["phases"].copy()

                # Analyze synchronization
                sync_metrics = self._analyze_synchronization()
                self.last_metrics = sync_metrics
                self.metrics_history.append(sync_metrics)

                # Learning update
                if self.coupling_mode == CouplingMode.LEARNING:
                    self._update_learning()

                # Clean up expired signals
                self._cleanup_expired_signals()

                return {
                    "phases": self.current_phases.copy(),
                    "frequencies": effective_frequencies,
                    "coupling_matrix": effective_coupling.copy(),
                    "sync_metrics": sync_metrics,
                    "order_parameter": sync_metrics.global_order_parameter,
                    "performance_score": self._calculate_performance_score(),
                }

        except Exception as e:
            self.logger.error(f"Step execution failed: {e}")
            return {"phases": self.current_phases.copy(), "error": str(e), "performance_score": 0.0}

    def _process_control_signals(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """Process all active control signals"""

        effective_coupling = self.coupling_matrix.copy()
        effective_frequencies = dict(self.config.natural_frequencies)

        # Sort signals by priority
        sorted_signals = sorted(self.active_signals, key=lambda s: s.priority, reverse=True)

        for signal in sorted_signals:
            # Apply coupling adjustments
            for module, adjustment in signal.coupling_adjustments.items():
                try:
                    module_idx = int(module.split("_")[1])
                    if 0 <= module_idx < self.num_modules:
                        # Apply proportional adjustment
                        effective_coupling[module_idx, :] *= 1.0 + adjustment
                        effective_coupling[:, module_idx] *= 1.0 + adjustment
                except (ValueError, IndexError):
                    continue

            # Apply frequency shifts
            for module, shift in signal.frequency_shifts.items():
                if module in effective_frequencies:
                    effective_frequencies[module] += shift

        return effective_coupling, effective_frequencies

    def _adapt_coupling(self, coupling_matrix: np.ndarray) -> np.ndarray:
        """Adapt coupling based on performance feedback"""

        if not self.metrics_history:
            return coupling_matrix

        # Calculate adaptation based on recent performance
        recent_performance = np.array(
            [m.global_order_parameter * m.sync_stability for m in list(self.metrics_history)[-10:]]
        )

        if len(recent_performance) < 2:
            return coupling_matrix

        # Performance trend
        performance_trend = np.mean(np.diff(recent_performance))

        # Adaptive adjustment
        adaptation_factor = 1.0 + self.adaptive_state.learning_rate * performance_trend
        adaptation_factor = np.clip(adaptation_factor, 0.5, 2.0)  # Safety bounds

        adapted_coupling = coupling_matrix * adaptation_factor

        # Update adaptive state
        self.adaptive_state.coupling_history.append(adaptation_factor)
        self.adaptive_state.performance_history.append(np.mean(recent_performance))

        return adapted_coupling

    def _analyze_synchronization(self) -> SynchronizationMetrics:
        """Comprehensive synchronization analysis"""

        # Global order parameter
        complex_phases = np.exp(1j * self.current_phases)
        global_order = np.abs(np.mean(complex_phases))

        # Local order parameters (within clusters)
        local_orders = {}
        cluster_size = max(2, self.num_modules // 4)

        for i in range(0, self.num_modules, cluster_size):
            end_idx = min(i + cluster_size, self.num_modules)
            cluster_phases = complex_phases[i:end_idx]
            local_orders[f"cluster_{i//cluster_size}"] = float(np.abs(np.mean(cluster_phases)))

        # Phase coherence (standard deviation of phases)
        phase_coherence = 1.0 - (np.std(self.current_phases) / (2 * np.pi))
        phase_coherence = max(0.0, phase_coherence)

        # Frequency locking analysis
        frequency_locking = {}
        if len(self.metrics_history) > 5:
            recent_phases = np.array(
                [m.global_order_parameter for m in list(self.metrics_history)[-5:]]
            )
            frequency_locking["stability"] = float(1.0 - np.std(recent_phases))

        # Coupling effectiveness
        coupling_strength = np.mean(np.abs(self.coupling_matrix[self.coupling_matrix != 0]))
        coupling_effectiveness = min(1.0, global_order / max(0.1, coupling_strength))

        # Synchronization stability
        if len(self.metrics_history) > 10:
            recent_orders = [m.global_order_parameter for m in list(self.metrics_history)[-10:]]
            sync_stability = 1.0 - np.std(recent_orders)
        else:
            sync_stability = 0.5

        # Energy efficiency (sync quality vs coupling cost)
        coupling_cost = np.sum(np.abs(self.coupling_matrix)) / (self.num_modules**2)
        energy_efficiency = global_order / max(0.1, coupling_cost)

        return SynchronizationMetrics(
            global_order_parameter=float(global_order),
            local_order_parameters=local_orders,
            phase_coherence=float(phase_coherence),
            frequency_locking=frequency_locking,
            coupling_effectiveness=float(coupling_effectiveness),
            sync_stability=float(max(0.0, sync_stability)),
            energy_efficiency=float(energy_efficiency),
        )

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score"""

        if not self.last_metrics:
            return 0.0

        # Weighted combination of metrics
        weights = {"order": 0.3, "coherence": 0.25, "stability": 0.25, "efficiency": 0.2}

        score = (
            weights["order"] * self.last_metrics.global_order_parameter
            + weights["coherence"] * self.last_metrics.phase_coherence
            + weights["stability"] * self.last_metrics.sync_stability
            + weights["efficiency"] * min(1.0, self.last_metrics.energy_efficiency / 5.0)
        )

        return float(np.clip(score, 0.0, 1.0))

    def _update_learning(self):
        """Update learning parameters based on performance"""

        if len(self.adaptive_state.performance_history) < 10:
            return

        # Analyze performance trend
        performance_trend = np.mean(np.diff(list(self.adaptive_state.performance_history)[-10:]))

        # Adapt learning rate
        if performance_trend > 0:  # Improving
            self.adaptive_state.learning_rate *= 1.01  # Increase slightly
        else:  # Declining
            self.adaptive_state.learning_rate *= 0.99  # Decrease slightly

        # Keep learning rate in reasonable bounds
        self.adaptive_state.learning_rate = np.clip(self.adaptive_state.learning_rate, 0.001, 0.1)

    def _cleanup_expired_signals(self):
        """Remove expired control signals"""

        current_time = time.time()
        self.active_signals = [
            signal
            for signal in self.active_signals
            if current_time - signal.duration < 10.0  # Max 10 second duration
        ]

    def enter_emergency_mode(self, emergency_context: Dict[str, Any]) -> bool:
        """Enter emergency mode with protective coupling"""

        try:
            with self.lock:
                if not self.emergency_mode:
                    # Backup current coupling
                    self.emergency_coupling_backup = self.coupling_matrix.copy()

                    # Apply emergency coupling (increased stability)
                    emergency_coupling = self.coupling_matrix * 1.5  # Increase coupling
                    emergency_coupling = np.clip(emergency_coupling, 0.0, 2.0)  # Safety bounds

                    self.coupling_matrix = emergency_coupling
                    self.emergency_mode = True

                    self.logger.warning("Enhanced Kuramoto: Entered emergency mode")

                return True

        except Exception as e:
            self.logger.error(f"Failed to enter emergency mode: {e}")
            return False

    def exit_emergency_mode(self) -> bool:
        """Exit emergency mode and restore normal coupling"""

        try:
            with self.lock:
                if self.emergency_mode and self.emergency_coupling_backup is not None:
                    # Gradually restore coupling (smooth transition)
                    target_coupling = self.emergency_coupling_backup
                    current_coupling = self.coupling_matrix

                    # Linear interpolation for smooth transition
                    self.coupling_matrix = 0.7 * current_coupling + 0.3 * target_coupling

                    self.emergency_mode = False
                    self.emergency_coupling_backup = None

                    self.logger.info("Enhanced Kuramoto: Exited emergency mode")

                return True

        except Exception as e:
            self.logger.error(f"Failed to exit emergency mode: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive module status"""

        with self.lock:
            status = {
                "initialized": self.initialized,
                "running": self.running,
                "emergency_mode": self.emergency_mode,
                "coupling_mode": self.coupling_mode.value,
                "num_modules": self.num_modules,
                "active_signals": len(self.active_signals),
                "performance_score": self._calculate_performance_score(),
                "learning_rate": self.adaptive_state.learning_rate,
            }

            if self.last_metrics:
                status.update(
                    {
                        "global_order_parameter": self.last_metrics.global_order_parameter,
                        "phase_coherence": self.last_metrics.phase_coherence,
                        "sync_stability": self.last_metrics.sync_stability,
                        "energy_efficiency": self.last_metrics.energy_efficiency,
                    }
                )

            return status

    def _reset_performance_tracking(self):
        """Reset performance tracking data"""

        self.metrics_history.clear()
        self.signal_history.clear()
        self.adaptive_state = AdaptiveState()
        self.last_metrics = None

    async def shutdown(self):
        """Graceful shutdown of enhanced module"""

        try:
            with self.lock:
                self.running = False

                # Clear active signals
                self.active_signals.clear()

                # Exit emergency mode if active
                if self.emergency_mode:
                    self.exit_emergency_mode()

                # Shutdown base solver
                await self.solver.shutdown()

                self.initialized = False
                self.logger.info("Enhanced Kuramoto Module 1.4 shut down")

        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


def create_enhanced_kuramoto(
    num_modules: int = 8, coupling_mode: CouplingMode = CouplingMode.ADAPTIVE
) -> EnhancedKuramotoModule:
    """Factory function for Enhanced Kuramoto Module"""

    return EnhancedKuramotoModule(num_modules=num_modules, coupling_mode=coupling_mode)


def create_learning_kuramoto(num_modules: int = 8) -> EnhancedKuramotoModule:
    """Factory for learning-enabled Kuramoto module"""

    return EnhancedKuramotoModule(num_modules=num_modules, coupling_mode=CouplingMode.LEARNING)


def create_constitutional_kuramoto(num_modules: int = 8) -> EnhancedKuramotoModule:
    """Factory for constitutionally-compliant Kuramoto module"""

    return EnhancedKuramotoModule(
        num_modules=num_modules, coupling_mode=CouplingMode.CONSTITUTIONAL
    )
