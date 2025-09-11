"""
Regulator module for Neural Field Control System.

Implements the control optimization component that computes optimal
control signals u(t) by minimizing the cost functional J[φ,u].

The Regulator acts as the "executive decision maker" in NFCS,
determining how to steer the system toward desired coherence states.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Callable
from scipy.optimize import minimize
import warnings

from .state import SystemState, ControlSignals, CostFunctionalConfig
from .metrics import MetricsCalculator


class Regulator:
    """
    Computes optimal control signals by minimizing cost functional J[φ,u].
    
    Uses gradient-based optimization to find control inputs that:
    1. Minimize energy costs
    2. Maintain target coherence levels  
    3. Suppress hallucinations and defects
    4. Respect control amplitude constraints
    """
    
    def __init__(self, config: CostFunctionalConfig):
        """
        Initialize regulator with cost functional configuration.
        
        Args:
            config: Configuration object with optimization weights
        """
        self.config = config
        self.metrics_calculator = MetricsCalculator(config)
        
        # Control constraints and parameters
        self.max_control_amplitude = 1.0  # Maximum allowed control magnitude
        self.optimization_method = 'L-BFGS-B'
        self.max_iterations = 50
        self.tolerance = 1e-6
        
        # History for adaptive control
        self.control_history = []
        self.cost_history = []
        
        print("Regulator initialized with cost functional optimization")
    
    def set_control_constraints(self, 
                              max_amplitude: float = 1.0,
                              method: str = 'L-BFGS-B',
                              max_iter: int = 50,
                              tol: float = 1e-6):
        """
        Set optimization parameters and control constraints.
        
        Args:
            max_amplitude: Maximum control signal amplitude
            method: Optimization method ('L-BFGS-B', 'SLSQP', 'trust-constr')
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
        """
        self.max_control_amplitude = max_amplitude
        self.optimization_method = method
        self.max_iterations = max_iter
        self.tolerance = tol
    
    def compute_optimal_control(self, state: SystemState) -> ControlSignals:
        """
        Compute optimal control signals by minimizing cost functional J.
        
        Args:
            state: Current system state
            
        Returns:
            Optimal control signals
        """
        # Simplified implementation for now - use feedback control
        return self.compute_feedback_control(state)
    
    def compute_feedback_control(self, 
                               state: SystemState,
                               target_coherence: float = 0.7) -> ControlSignals:
        """
        Compute simple feedback control based on coherence error.
        
        Args:
            state: Current system state
            target_coherence: Desired coherence level
            
        Returns:
            Feedback control signals
        """
        # Get current coherence
        coherence_global, coherence_modular = self.metrics_calculator.calculate_coherence(
            state.neural_field, state.module_phases
        )
        
        # Coherence errors
        error_global = target_coherence - coherence_global
        error_modular = target_coherence - coherence_modular
        
        # Proportional gains
        kp_field = 0.1
        kp_modules = 0.2
        
        # Field control: proportional to coherence error
        field_size = state.neural_field.shape
        control_field = kp_field * error_global * np.ones(field_size, dtype=np.complex128)
        
        # Module control: proportional to phase errors
        num_modules = len(state.module_phases)
        control_modules = kp_modules * error_modular * np.ones(num_modules)
        
        # Clip to constraints
        control_field = np.clip(
            np.abs(control_field), 0, self.max_control_amplitude
        ) * np.exp(1j * np.angle(control_field))
        
        control_modules = np.clip(
            control_modules, -self.max_control_amplitude, self.max_control_amplitude
        )
        
        return ControlSignals(
            u_field=control_field,
            u_modules=control_modules,
            timestamp=state.simulation_time
        )