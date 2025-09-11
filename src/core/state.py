"""
Core data structures for the Neural Field Control System (NFCS).

This module defines the central state objects, configuration classes,
and data structures used throughout the NFCS system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod


# --- 1. Configuration Structures ---

@dataclass
class CGLConfig:
    """Configuration for Complex Ginzburg-Landau solver."""
    c1: float  # Linear dispersion parameter
    c3: float  # Nonlinear self-action parameter
    grid_size: Tuple[int, int]  # 2D grid dimensions
    time_step: float  # Temporal step size
    spatial_extent: Tuple[float, float]  # Physical domain size
    boundary_conditions: str = "periodic"


@dataclass
class KuramotoConfig:
    """Configuration for Kuramoto oscillator network."""
    natural_frequencies: Dict[str, float]  # Module name -> frequency
    base_coupling_strength: float
    time_step: float


@dataclass
class CostFunctionalConfig:
    """Configuration for cost functional computation."""
    # Energy term weights
    w_field_energy: float = 1.0
    w_field_gradient: float = 0.5
    w_control_energy: float = 0.1
    w_coherence_penalty: float = 2.0
    
    # Risk metrics weights
    w_hallucinations: float = 1.5
    w_defect_density: float = 1.0
    w_coherence_loss: float = 2.0
    w_violations: float = 3.0


@dataclass
class SystemConfig:
    """Overall system configuration."""
    cgl: CGLConfig
    kuramoto: KuramotoConfig
    cost_functional: CostFunctionalConfig
    max_simulation_steps: int = 10000
    emergency_threshold_ha: float = 0.8
    emergency_threshold_defects: float = 0.1
    coherence_target: float = 0.7


# --- 2. Control and Signal Structures ---

@dataclass
class ControlSignals:
    """Control signals for system components."""
    u_field: np.ndarray  # Spatial control field for CGL
    u_modules: np.ndarray  # Control signals for Kuramoto modules
    timestamp: float = 0.0


@dataclass
class RiskMetrics:
    """Computed risk and coherence metrics."""
    # Core risk metrics
    rho_def_field: np.ndarray  # Topological defect density field
    rho_def_mean: float  # Average defect density
    hallucination_number: float  # H_a(t)
    systemic_risk: float  # Aggregated risk score
    
    # Coherence metrics
    coherence_global: float  # Global field coherence
    coherence_modular: float  # R_modular from Kuramoto phases
    
    # Auxiliary metrics
    field_energy: float = 0.0
    control_energy: float = 0.0
    violations_count: int = 0
    
    timestamp: float = 0.0


@dataclass
class ESCState:
    """Echo-Semantic Converter state."""
    active_oscillators: Dict[str, Any] = field(default_factory=dict)
    eta_output: float = 0.0  # Current order parameter
    resonance_frequencies: List[float] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class MemoryState:
    """Multi-scale memory system state."""
    short_term: np.ndarray = field(default_factory=lambda: np.array([]))
    medium_term: np.ndarray = field(default_factory=lambda: np.array([]))
    long_term: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamp: float = 0.0


# --- 3. Main System State ---

@dataclass
class SystemState:
    """Global system state object."""
    
    # Core mathematical fields
    neural_field: np.ndarray  # φ(x,t) - complex neural field
    module_phases: np.ndarray  # θ_i(t) - Kuramoto module phases
    
    # Dynamic coupling matrix
    kuramoto_coupling_matrix: np.ndarray
    
    # Computed metrics and module states
    risk_metrics: RiskMetrics
    esc_state: ESCState
    memory_state: MemoryState
    
    # Control from previous step
    last_control_signals: ControlSignals
    
    # System operation mode
    system_mode: Literal["NORMAL", "EMERGENCY_MODE"] = "NORMAL"
    
    # Simulation metadata
    current_step: int = 0
    simulation_time: float = 0.0
    
    def __post_init__(self):
        """Validate state consistency after initialization."""
        if self.neural_field.dtype != np.complex128:
            raise ValueError("Neural field must be complex128")
        
        n_modules = len(self.module_phases)
        expected_coupling_shape = (n_modules, n_modules)
        if self.kuramoto_coupling_matrix.shape != expected_coupling_shape:
            raise ValueError(f"Coupling matrix shape {self.kuramoto_coupling_matrix.shape} "
                           f"doesn't match number of modules {n_modules}")


# --- 4. Validation and Adapter Structures ---

@dataclass
class ValidationInput:
    """Input data for neuronal mass model validation."""
    control_signals_history: List[ControlSignals]
    simulation_time_steps: np.ndarray
    system_states_history: List[SystemState]


# --- 5. Base Classes and Interfaces ---

class BaseModule(ABC):
    """Abstract base class for cognitive modules."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_active = True
        
    @abstractmethod
    def process(self, state: SystemState) -> SystemState:
        """Process current system state and return updated state."""
        pass
    
    @abstractmethod
    def get_module_name(self) -> str:
        """Return module identification name."""
        pass


class BaseEvolutionOperator(ABC):
    """Abstract base class for evolution operators."""
    
    @abstractmethod
    def mutate(self, config: SystemConfig) -> SystemConfig:
        """Apply mutation to system configuration."""
        pass
    
    @abstractmethod
    def crossover(self, parent1: SystemConfig, parent2: SystemConfig) -> SystemConfig:
        """Create offspring from two parent configurations."""
        pass


# --- 6. Utility Functions ---

def create_empty_system_state(grid_size: Tuple[int, int], n_modules: int) -> SystemState:
    """Create a properly initialized empty system state."""
    
    # Initialize complex neural field
    neural_field = np.zeros(grid_size, dtype=np.complex128)
    
    # Initialize module phases
    module_phases = np.zeros(n_modules)
    
    # Initialize coupling matrix
    coupling_matrix = np.eye(n_modules)
    
    # Initialize empty risk metrics
    risk_metrics = RiskMetrics(
        rho_def_field=np.zeros(grid_size),
        rho_def_mean=0.0,
        hallucination_number=0.0,
        systemic_risk=0.0,
        coherence_global=0.0,
        coherence_modular=0.0
    )
    
    # Initialize empty control signals
    control_signals = ControlSignals(
        u_field=np.zeros(grid_size, dtype=np.complex128),
        u_modules=np.zeros(n_modules)
    )
    
    return SystemState(
        neural_field=neural_field,
        module_phases=module_phases,
        kuramoto_coupling_matrix=coupling_matrix,
        risk_metrics=risk_metrics,
        esc_state=ESCState(),
        memory_state=MemoryState(),
        last_control_signals=control_signals
    )


def validate_system_state(state: SystemState) -> bool:
    """Validate system state for consistency and safety."""
    try:
        # Check for NaN or inf values
        if not np.isfinite(state.neural_field).all():
            return False
        if not np.isfinite(state.module_phases).all():
            return False
        if not np.isfinite(state.kuramoto_coupling_matrix).all():
            return False
            
        # Check phase bounds (should be in [-π, π])
        if np.any(np.abs(state.module_phases) > np.pi):
            return False
            
        # Check coupling matrix properties (should be real and symmetric)
        if not np.allclose(state.kuramoto_coupling_matrix, 
                          state.kuramoto_coupling_matrix.T, rtol=1e-10):
            return False
            
        return True
        
    except Exception:
        return False