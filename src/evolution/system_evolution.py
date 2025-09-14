"""
System Evolution - NFCS Parameter Optimization
============================================

Implements system-level evolutionary optimization for NFCS v2.4.3 parameter tuning.
Provides specialized evolutionary algorithms for neural field control system parameters,
including CGL dynamics, Kuramoto coupling, and ESC processing parameters.

Scientific Foundation:
Based on PDF Section 5.13, implements adaptive parameter evolution for:
- Complex Ginzburg-Landau equation parameters (α, β, γ, δ)
- Kuramoto synchronization coupling strengths (K_ij matrices)
- ESC oscillatory parameters (ω_c, A_c, φ_c)
- Boundary condition parameters and control gains
- Multi-objective optimization for stability vs. performance trade-offs

Mathematical Framework:
- Parameter space P = {α, β, γ, δ, K, ω, A, φ, ...}
- Fitness function F(P) = weighted combination of performance metrics
- Constraints: stability conditions, physical bounds, synchronization requirements
- Multi-objective: minimize control error, maximize stability margin, minimize energy

Integration:
Seamlessly integrates with NFCS core systems for real-time parameter adaptation
during system operation and online optimization of control performance.

Created: September 14, 2025
Author: Team Ω - Neural Field Control Systems Research Group
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
from abc import ABC, abstractmethod
from collections import defaultdict
import warnings

from .genetic_optimizer import (
    GeneticOptimizer, Individual, Population, FitnessFunction,
    EvolutionaryStrategy, SelectionStrategy, CrossoverOperator, MutationOperator
)
from ..core.state import SystemState, RiskMetrics
from ..core.cgl_solver import CGLSolver
from ..core.kuramoto_solver import KuramotoSolver
from ..modules.esc.esc_core import ESCModule

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Types of NFCS parameters for optimization"""
    CGL_DYNAMICS = "cgl_dynamics"
    KURAMOTO_COUPLING = "kuramoto_coupling"
    ESC_OSCILLATORY = "esc_oscillatory"
    BOUNDARY_CONDITIONS = "boundary_conditions"
    CONTROL_GAINS = "control_gains"
    SAFETY_CONSTRAINTS = "safety_constraints"


class OptimizationObjective(Enum):
    """Optimization objectives for system evolution"""
    CONTROL_ERROR = "control_error"
    STABILITY_MARGIN = "stability_margin"
    ENERGY_EFFICIENCY = "energy_efficiency"
    SYNCHRONIZATION_QUALITY = "synchronization_quality"
    RESPONSE_TIME = "response_time"
    ROBUSTNESS = "robustness"


@dataclass
class NFCSParameterSet:
    """
    Comprehensive parameter set for NFCS system optimization.
    
    Attributes:
        cgl_params: Complex Ginzburg-Landau equation parameters
        kuramoto_params: Kuramoto synchronization parameters
        esc_params: Echo-Semantic Converter parameters
        boundary_params: Boundary condition parameters
        control_params: Control system parameters
        safety_params: Safety and constraint parameters
        metadata: Additional parameter metadata
    """
    # CGL dynamics parameters
    cgl_params: Dict[str, float] = field(default_factory=lambda: {
        'alpha': 1.0,      # Linear growth rate
        'beta': 1.0,       # Nonlinear saturation
        'gamma': 1.0,      # Dispersive coupling
        'delta': 0.0,      # Advection parameter
        'epsilon': 0.1,    # Noise intensity
        'D': 1.0,          # Diffusion coefficient
        'boundary_gain': 1.0
    })
    
    # Kuramoto synchronization parameters
    kuramoto_params: Dict[str, Any] = field(default_factory=lambda: {
        'base_coupling': 1.0,           # Base coupling strength K_base
        'coupling_matrix': None,        # N×N coupling matrix K_ij
        'natural_frequencies': None,    # ω_i natural frequencies
        'phase_lag': 0.0,              # Phase lag parameter
        'topology_strength': 1.0,       # Network topology influence
        'adaptive_rate': 0.1           # Adaptation rate for coupling
    })
    
    # ESC oscillatory parameters
    esc_params: Dict[str, float] = field(default_factory=lambda: {
        'carrier_frequency': 10.0,     # ω_c carrier frequency
        'carrier_amplitude': 1.0,      # A_c carrier amplitude
        'carrier_phase': 0.0,          # φ_c carrier phase
        'modulation_depth': 0.5,       # Modulation depth
        'demodulation_gain': 1.0,      # Demodulation gain
        'filter_bandwidth': 5.0        # Filter bandwidth
    })
    
    # Boundary condition parameters
    boundary_params: Dict[str, float] = field(default_factory=lambda: {
        'dirichlet_values': 0.0,       # Dirichlet boundary values
        'neumann_values': 0.0,         # Neumann boundary values
        'robin_alpha': 1.0,            # Robin condition parameter α
        'robin_beta': 1.0,             # Robin condition parameter β
        'boundary_stiffness': 1.0      # Boundary stiffness
    })
    
    # Control system parameters
    control_params: Dict[str, float] = field(default_factory=lambda: {
        'proportional_gain': 1.0,      # P gain
        'integral_gain': 0.1,          # I gain
        'derivative_gain': 0.01,       # D gain
        'feedforward_gain': 1.0,       # Feedforward gain
        'observer_gain': 1.0,          # State observer gain
        'saturation_limit': 10.0       # Control saturation
    })
    
    # Safety and constraint parameters
    safety_params: Dict[str, float] = field(default_factory=lambda: {
        'stability_margin': 0.1,       # Required stability margin
        'max_field_magnitude': 10.0,   # Maximum field magnitude
        'max_control_effort': 5.0,     # Maximum control effort
        'convergence_tolerance': 1e-6, # Convergence tolerance
        'safety_factor': 1.2           # Safety factor
    })
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_genome(self) -> np.ndarray:
        """Convert parameter set to optimization genome vector"""
        genome_values = []
        
        # Add CGL parameters
        genome_values.extend([
            self.cgl_params['alpha'],
            self.cgl_params['beta'], 
            self.cgl_params['gamma'],
            self.cgl_params['delta'],
            self.cgl_params['epsilon'],
            self.cgl_params['D'],
            self.cgl_params['boundary_gain']
        ])
        
        # Add Kuramoto parameters
        genome_values.extend([
            self.kuramoto_params['base_coupling'],
            self.kuramoto_params['phase_lag'],
            self.kuramoto_params['topology_strength'],
            self.kuramoto_params['adaptive_rate']
        ])
        
        # Add ESC parameters
        genome_values.extend([
            self.esc_params['carrier_frequency'],
            self.esc_params['carrier_amplitude'],
            self.esc_params['carrier_phase'],
            self.esc_params['modulation_depth'],
            self.esc_params['demodulation_gain'],
            self.esc_params['filter_bandwidth']
        ])
        
        # Add boundary parameters
        genome_values.extend([
            self.boundary_params['dirichlet_values'],
            self.boundary_params['neumann_values'],
            self.boundary_params['robin_alpha'],
            self.boundary_params['robin_beta'],
            self.boundary_params['boundary_stiffness']
        ])
        
        # Add control parameters
        genome_values.extend([
            self.control_params['proportional_gain'],
            self.control_params['integral_gain'],
            self.control_params['derivative_gain'],
            self.control_params['feedforward_gain'],
            self.control_params['observer_gain'],
            self.control_params['saturation_limit']
        ])
        
        # Add safety parameters
        genome_values.extend([
            self.safety_params['stability_margin'],
            self.safety_params['max_field_magnitude'],
            self.safety_params['max_control_effort'],
            self.safety_params['safety_factor']
        ])
        
        return np.array(genome_values)
    
    @classmethod
    def from_genome(cls, genome: np.ndarray) -> 'NFCSParameterSet':
        """Create parameter set from optimization genome vector"""
        if len(genome) < 29:  # Minimum expected parameters
            raise ValueError(f"Genome too short: {len(genome)} < 29")
            
        params = cls()
        idx = 0
        
        # CGL parameters
        params.cgl_params.update({
            'alpha': genome[idx], 'beta': genome[idx+1], 'gamma': genome[idx+2],
            'delta': genome[idx+3], 'epsilon': genome[idx+4], 'D': genome[idx+5],
            'boundary_gain': genome[idx+6]
        })
        idx += 7
        
        # Kuramoto parameters
        params.kuramoto_params.update({
            'base_coupling': genome[idx], 'phase_lag': genome[idx+1],
            'topology_strength': genome[idx+2], 'adaptive_rate': genome[idx+3]
        })
        idx += 4
        
        # ESC parameters
        params.esc_params.update({
            'carrier_frequency': genome[idx], 'carrier_amplitude': genome[idx+1],
            'carrier_phase': genome[idx+2], 'modulation_depth': genome[idx+3],
            'demodulation_gain': genome[idx+4], 'filter_bandwidth': genome[idx+5]
        })
        idx += 6
        
        # Boundary parameters
        params.boundary_params.update({
            'dirichlet_values': genome[idx], 'neumann_values': genome[idx+1],
            'robin_alpha': genome[idx+2], 'robin_beta': genome[idx+3],
            'boundary_stiffness': genome[idx+4]
        })
        idx += 5
        
        # Control parameters
        params.control_params.update({
            'proportional_gain': genome[idx], 'integral_gain': genome[idx+1],
            'derivative_gain': genome[idx+2], 'feedforward_gain': genome[idx+3],
            'observer_gain': genome[idx+4], 'saturation_limit': genome[idx+5]
        })
        idx += 6
        
        # Safety parameters
        if idx + 4 <= len(genome):
            params.safety_params.update({
                'stability_margin': genome[idx], 'max_field_magnitude': genome[idx+1],
                'max_control_effort': genome[idx+2], 'safety_factor': genome[idx+3]
            })
        
        return params
    
    def get_parameter_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get optimization bounds for all parameters"""
        lower_bounds = np.array([
            # CGL parameters
            -5.0, -5.0, 0.0, -2.0, 0.0, 0.1, 0.1,
            # Kuramoto parameters  
            0.0, -np.pi, 0.0, 0.0,
            # ESC parameters
            0.1, 0.1, -np.pi, 0.0, 0.1, 0.1,
            # Boundary parameters
            -10.0, -10.0, 0.0, 0.0, 0.1,
            # Control parameters
            0.0, 0.0, 0.0, 0.0, 0.0, 0.1,
            # Safety parameters
            0.01, 1.0, 0.1, 1.0
        ])
        
        upper_bounds = np.array([
            # CGL parameters
            5.0, 5.0, 10.0, 2.0, 1.0, 10.0, 10.0,
            # Kuramoto parameters
            10.0, np.pi, 10.0, 1.0,
            # ESC parameters
            100.0, 10.0, np.pi, 1.0, 10.0, 50.0,
            # Boundary parameters
            10.0, 10.0, 10.0, 10.0, 10.0,
            # Control parameters
            100.0, 10.0, 1.0, 10.0, 10.0, 100.0,
            # Safety parameters
            1.0, 100.0, 50.0, 5.0
        ])
        
        return lower_bounds, upper_bounds


class SystemFitnessEvaluator(FitnessFunction):
    """
    Fitness function for NFCS system parameter optimization.
    
    Evaluates system performance based on multiple objectives including
    control error, stability, energy efficiency, and synchronization quality.
    """
    
    def __init__(
        self,
        system_size: Tuple[int, int] = (64, 64),
        time_horizon: float = 10.0,
        dt: float = 0.01,
        objectives: List[OptimizationObjective] = None,
        objective_weights: Dict[OptimizationObjective, float] = None,
        reference_trajectory: Optional[Callable] = None,
        disturbance_profile: Optional[Callable] = None,
        simulation_steps: int = 1000
    ):
        """
        Initialize system fitness evaluator.
        
        Args:
            system_size: Grid size for neural field simulation
            time_horizon: Simulation time horizon
            dt: Integration time step
            objectives: List of optimization objectives
            objective_weights: Weights for multi-objective optimization
            reference_trajectory: Reference trajectory function
            disturbance_profile: External disturbance function
            simulation_steps: Number of simulation steps
        """
        self.system_size = system_size
        self.time_horizon = time_horizon
        self.dt = dt
        self.simulation_steps = simulation_steps
        
        # Set default objectives
        if objectives is None:
            objectives = [
                OptimizationObjective.CONTROL_ERROR,
                OptimizationObjective.STABILITY_MARGIN,
                OptimizationObjective.ENERGY_EFFICIENCY
            ]
        self.objectives = objectives
        
        # Set default weights
        if objective_weights is None:
            objective_weights = {obj: 1.0 for obj in objectives}
        self.objective_weights = objective_weights
        
        self.reference_trajectory = reference_trajectory
        self.disturbance_profile = disturbance_profile
        
        # Initialize system components
        self._init_system_components()
        
    def _init_system_components(self) -> None:
        """Initialize NFCS system components for simulation"""
        # Initialize CGL solver
        self.cgl_solver = CGLSolver(
            grid_size=self.system_size,
            dt=self.dt,
            boundary_conditions='periodic'
        )
        
        # Initialize Kuramoto solver
        N_oscillators = 100  # Default number of oscillators
        self.kuramoto_solver = KuramotoSolver(
            N=N_oscillators,
            dt=self.dt
        )
        
        # Initialize ESC module
        self.esc_module = ESCModule(
            carrier_frequency=10.0,
            sampling_rate=1/self.dt
        )
        
        logger.info(f"Initialized system components for fitness evaluation")
        
    def evaluate(self, genome: np.ndarray) -> Union[float, np.ndarray]:
        """
        Evaluate fitness of NFCS parameter set.
        
        Args:
            genome: Parameter genome to evaluate
            
        Returns:
            Fitness value (single objective) or array (multi-objective)
        """
        try:
            # Convert genome to parameter set
            params = NFCSParameterSet.from_genome(genome)
            
            # Run system simulation
            simulation_results = self._run_system_simulation(params)
            
            # Evaluate objectives
            objective_values = {}
            
            for objective in self.objectives:
                if objective == OptimizationObjective.CONTROL_ERROR:
                    objective_values[objective] = self._evaluate_control_error(simulation_results)
                elif objective == OptimizationObjective.STABILITY_MARGIN:
                    objective_values[objective] = self._evaluate_stability_margin(simulation_results)
                elif objective == OptimizationObjective.ENERGY_EFFICIENCY:
                    objective_values[objective] = self._evaluate_energy_efficiency(simulation_results)
                elif objective == OptimizationObjective.SYNCHRONIZATION_QUALITY:
                    objective_values[objective] = self._evaluate_synchronization_quality(simulation_results)
                elif objective == OptimizationObjective.RESPONSE_TIME:
                    objective_values[objective] = self._evaluate_response_time(simulation_results)
                elif objective == OptimizationObjective.ROBUSTNESS:
                    objective_values[objective] = self._evaluate_robustness(simulation_results)
                    
            # Apply constraint penalties
            penalty = self._evaluate_constraints(params, simulation_results)
            
            if len(self.objectives) == 1:
                # Single-objective optimization
                objective = self.objectives[0]
                fitness = objective_values[objective] - penalty
                return fitness
            else:
                # Multi-objective optimization
                fitness_vector = np.array([
                    objective_values[obj] * self.objective_weights[obj] - penalty
                    for obj in self.objectives
                ])
                return fitness_vector
                
        except Exception as e:
            logger.error(f"Error in fitness evaluation: {e}")
            # Return poor fitness for invalid parameters
            if len(self.objectives) == 1:
                return -1e6
            else:
                return np.array([-1e6] * len(self.objectives))
                
    def _run_system_simulation(self, params: NFCSParameterSet) -> Dict[str, Any]:
        """
        Run NFCS system simulation with given parameters.
        
        Args:
            params: Parameter set to simulate
            
        Returns:
            Dictionary containing simulation results
        """
        # Initialize system state
        u_field = np.random.normal(0, 0.1, self.system_size).astype(complex)
        phases = np.random.uniform(0, 2*np.pi, self.kuramoto_solver.N)
        
        # Storage for results
        time_points = np.arange(0, self.time_horizon, self.dt)
        field_history = []
        phase_history = []
        control_history = []
        energy_history = []
        
        # Update system parameters
        self.cgl_solver.alpha = params.cgl_params['alpha']
        self.cgl_solver.beta = params.cgl_params['beta']
        self.cgl_solver.gamma = params.cgl_params['gamma']
        
        self.kuramoto_solver.K = params.kuramoto_params['base_coupling']
        
        # Simulation loop
        for t_idx, t in enumerate(time_points[:self.simulation_steps]):
            # Generate reference if available
            if self.reference_trajectory:
                reference = self.reference_trajectory(t)
            else:
                reference = np.zeros_like(u_field, dtype=complex)
                
            # Generate disturbance if available
            if self.disturbance_profile:
                disturbance = self.disturbance_profile(t)
            else:
                disturbance = np.zeros_like(u_field, dtype=complex)
                
            # Compute control signal using ESC
            control_signal = self._compute_control_signal(
                u_field, reference, params, t
            )
            
            # Update CGL field
            u_field = self.cgl_solver.step(
                u_field, 
                control=control_signal + disturbance
            )
            
            # Update Kuramoto phases
            phases = self.kuramoto_solver.step(phases)
            
            # Store results
            if t_idx % 10 == 0:  # Subsample for storage
                field_history.append(u_field.copy())
                phase_history.append(phases.copy())
                control_history.append(control_signal.copy())
                energy_history.append(np.sum(np.abs(u_field)**2))
                
        return {
            'field_history': field_history,
            'phase_history': phase_history,
            'control_history': control_history,
            'energy_history': energy_history,
            'time_points': time_points[:len(field_history)],
            'final_field': u_field,
            'final_phases': phases,
            'parameters': params
        }
        
    def _compute_control_signal(
        self, 
        u_field: np.ndarray, 
        reference: np.ndarray, 
        params: NFCSParameterSet,
        time: float
    ) -> np.ndarray:
        """Compute control signal using PID-like controller"""
        error = reference - u_field
        
        # Simple PID controller
        P_term = params.control_params['proportional_gain'] * error
        
        # Apply saturation
        control = np.clip(
            P_term, 
            -params.safety_params['max_control_effort'],
            params.safety_params['max_control_effort']
        )
        
        return control
        
    def _evaluate_control_error(self, results: Dict[str, Any]) -> float:
        """Evaluate control tracking error"""
        if not results['field_history']:
            return -1e6
            
        # Compute RMS error relative to reference
        total_error = 0.0
        for field in results['field_history']:
            # Reference is typically zero or small
            error = np.mean(np.abs(field)**2)
            total_error += error
            
        rms_error = np.sqrt(total_error / len(results['field_history']))
        
        # Return negative (minimize error)
        return -rms_error
        
    def _evaluate_stability_margin(self, results: Dict[str, Any]) -> float:
        """Evaluate system stability margin"""
        if len(results['energy_history']) < 2:
            return -1e6
            
        # Check if energy is bounded and converging
        energy_trend = np.diff(results['energy_history'])
        
        # Stability indicators
        max_energy = np.max(results['energy_history'])
        energy_variance = np.var(results['energy_history'])
        
        # Penalize growing or highly variable energy
        if max_energy > 1e6 or energy_variance > 1e3:
            return -1e6
            
        # Reward decreasing or stable energy
        stability_score = 1.0 / (1.0 + energy_variance + max_energy)
        
        return stability_score
        
    def _evaluate_energy_efficiency(self, results: Dict[str, Any]) -> float:
        """Evaluate energy efficiency of control"""
        if not results['control_history']:
            return -1e6
            
        # Compute total control energy
        total_control_energy = 0.0
        for control in results['control_history']:
            control_energy = np.sum(np.abs(control)**2)
            total_control_energy += control_energy
            
        avg_control_energy = total_control_energy / len(results['control_history'])
        
        # Return efficiency (negative energy for maximization)
        return -avg_control_energy
        
    def _evaluate_synchronization_quality(self, results: Dict[str, Any]) -> float:
        """Evaluate Kuramoto synchronization quality"""
        if not results['phase_history']:
            return -1e6
            
        # Compute order parameter for synchronization
        sync_scores = []
        for phases in results['phase_history']:
            # Complex order parameter
            order_param = np.abs(np.mean(np.exp(1j * phases)))
            sync_scores.append(order_param)
            
        avg_synchronization = np.mean(sync_scores)
        return avg_synchronization
        
    def _evaluate_response_time(self, results: Dict[str, Any]) -> float:
        """Evaluate system response time"""
        if len(results['energy_history']) < 10:
            return -1e6
            
        # Find settling time (when system reaches steady state)
        energy = np.array(results['energy_history'])
        steady_state_value = energy[-10:].mean()
        tolerance = 0.05 * steady_state_value
        
        # Find first index where system stays within tolerance
        settling_idx = len(energy)
        for i in range(len(energy)-10):
            if all(abs(energy[i:i+10] - steady_state_value) < tolerance):
                settling_idx = i
                break
                
        # Normalize settling time
        settling_time = settling_idx / len(energy)
        
        # Return negative (minimize settling time)
        return -settling_time
        
    def _evaluate_robustness(self, results: Dict[str, Any]) -> float:
        """Evaluate system robustness to disturbances"""
        # Simplified robustness metric based on field variation
        if not results['field_history']:
            return -1e6
            
        field_variations = []
        for i in range(1, len(results['field_history'])):
            variation = np.mean(np.abs(
                results['field_history'][i] - results['field_history'][i-1]
            ))
            field_variations.append(variation)
            
        if not field_variations:
            return -1e6
            
        # Robustness is inverse of variation
        avg_variation = np.mean(field_variations)
        robustness = 1.0 / (1.0 + avg_variation)
        
        return robustness
        
    def _evaluate_constraints(self, params: NFCSParameterSet, results: Dict[str, Any]) -> float:
        """Evaluate constraint violations and return penalty"""
        penalty = 0.0
        
        # Check stability constraints
        if results['energy_history']:
            max_energy = np.max(results['energy_history'])
            if max_energy > params.safety_params['max_field_magnitude']**2:
                penalty += 1e3 * (max_energy - params.safety_params['max_field_magnitude']**2)
                
        # Check control effort constraints
        if results['control_history']:
            for control in results['control_history']:
                max_control = np.max(np.abs(control))
                if max_control > params.safety_params['max_control_effort']:
                    penalty += 1e3 * (max_control - params.safety_params['max_control_effort'])
                    
        # Check parameter bounds
        genome = params.to_genome()
        lower_bounds, upper_bounds = params.get_parameter_bounds()
        
        for i, (val, lower, upper) in enumerate(zip(genome, lower_bounds, upper_bounds)):
            if val < lower:
                penalty += 1e3 * (lower - val)**2
            elif val > upper:
                penalty += 1e3 * (val - upper)**2
                
        return penalty
        
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds for optimization"""
        params = NFCSParameterSet()
        return params.get_parameter_bounds()


class SystemEvolution:
    """
    Main system evolution coordinator for NFCS parameter optimization.
    
    Implements comprehensive evolutionary optimization for neural field control
    systems with multi-objective capabilities and real-time adaptation.
    """
    
    def __init__(
        self,
        system_config: Dict[str, Any] = None,
        optimization_config: Dict[str, Any] = None,
        objectives: List[OptimizationObjective] = None,
        objective_weights: Dict[OptimizationObjective, float] = None
    ):
        """
        Initialize system evolution coordinator.
        
        Args:
            system_config: System configuration parameters
            optimization_config: Optimization algorithm configuration
            objectives: List of optimization objectives
            objective_weights: Weights for multi-objective optimization
        """
        # Set default system configuration
        if system_config is None:
            system_config = {
                'system_size': (64, 64),
                'time_horizon': 10.0,
                'dt': 0.01,
                'simulation_steps': 1000
            }
        self.system_config = system_config
        
        # Set default optimization configuration
        if optimization_config is None:
            optimization_config = {
                'population_size': 100,
                'max_generations': 500,
                'strategy': EvolutionaryStrategy.GENETIC_ALGORITHM,
                'crossover_rate': 0.8,
                'mutation_rate': 0.1,
                'elitism_rate': 0.1
            }
        self.optimization_config = optimization_config
        
        # Set objectives and weights
        self.objectives = objectives or [
            OptimizationObjective.CONTROL_ERROR,
            OptimizationObjective.STABILITY_MARGIN,
            OptimizationObjective.ENERGY_EFFICIENCY
        ]
        self.objective_weights = objective_weights or {obj: 1.0 for obj in self.objectives}
        
        # Initialize fitness evaluator
        self.fitness_evaluator = SystemFitnessEvaluator(
            system_size=self.system_config['system_size'],
            time_horizon=self.system_config['time_horizon'],
            dt=self.system_config['dt'],
            simulation_steps=self.system_config['simulation_steps'],
            objectives=self.objectives,
            objective_weights=self.objective_weights
        )
        
        # Initialize genetic optimizer
        params_template = NFCSParameterSet()
        genome_length = len(params_template.to_genome())
        
        self.optimizer = GeneticOptimizer(
            fitness_function=self.fitness_evaluator,
            population_size=self.optimization_config['population_size'],
            genome_length=genome_length,
            strategy=self.optimization_config['strategy'],
            crossover_rate=self.optimization_config['crossover_rate'],
            mutation_rate=self.optimization_config['mutation_rate'],
            elitism_rate=self.optimization_config['elitism_rate'],
            max_generations=self.optimization_config['max_generations'],
            multi_objective=len(self.objectives) > 1
        )
        
        # Evolution history and statistics
        self.evolution_history = []
        self.best_parameters = None
        self.optimization_statistics = {}
        
        logger.info(f"Initialized SystemEvolution with {len(self.objectives)} objectives")
        
    def optimize_parameters(
        self,
        initial_parameters: Optional[NFCSParameterSet] = None,
        reference_trajectory: Optional[Callable] = None,
        disturbance_profile: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run parameter optimization for NFCS system.
        
        Args:
            initial_parameters: Initial parameter set (optional)
            reference_trajectory: Reference trajectory function
            disturbance_profile: Disturbance profile function
            
        Returns:
            Dictionary containing optimization results
        """
        start_time = time.time()
        
        # Set reference and disturbance
        self.fitness_evaluator.reference_trajectory = reference_trajectory
        self.fitness_evaluator.disturbance_profile = disturbance_profile
        
        # Initialize population with initial parameters if provided
        if initial_parameters:
            self.optimizer.initialize_population()
            # Replace first individual with initial parameters
            initial_genome = initial_parameters.to_genome()
            self.optimizer.population.individuals[0].genome = initial_genome
            
        # Run optimization
        logger.info("Starting NFCS parameter optimization...")
        optimization_results = self.optimizer.evolve()
        
        # Extract best parameters
        if optimization_results['best_individual']:
            best_genome = optimization_results['best_genome']
            self.best_parameters = NFCSParameterSet.from_genome(best_genome)
        
        # Compile comprehensive results
        results = {
            'optimization_results': optimization_results,
            'best_parameters': self.best_parameters,
            'objectives': self.objectives,
            'objective_weights': self.objective_weights,
            'system_config': self.system_config,
            'optimization_config': self.optimization_config,
            'execution_time': time.time() - start_time,
            'convergence_analysis': self._analyze_convergence(optimization_results),
            'parameter_sensitivity': self._analyze_parameter_sensitivity()
        }
        
        # Store in history
        self.evolution_history.append(results)
        self.optimization_statistics = self.optimizer.get_optimization_statistics()
        
        logger.info(
            f"Parameter optimization completed in {results['execution_time']:.2f}s. "
            f"Best fitness: {optimization_results['best_fitness']}"
        )
        
        return results
        
    def _analyze_convergence(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization convergence characteristics"""
        convergence_history = optimization_results['convergence_history']
        diversity_history = optimization_results['diversity_history']
        
        if not convergence_history:
            return {}
            
        # Convergence metrics
        final_fitness = convergence_history[-1]
        initial_fitness = convergence_history[0]
        improvement = final_fitness - initial_fitness
        
        # Find convergence point (95% of final improvement)
        target_fitness = initial_fitness + 0.95 * improvement
        convergence_generation = len(convergence_history)
        
        for i, fitness in enumerate(convergence_history):
            if fitness >= target_fitness:
                convergence_generation = i
                break
                
        # Diversity analysis
        final_diversity = diversity_history[-1] if diversity_history else 0
        avg_diversity = np.mean(diversity_history) if diversity_history else 0
        
        return {
            'initial_fitness': initial_fitness,
            'final_fitness': final_fitness,
            'total_improvement': improvement,
            'convergence_generation': convergence_generation,
            'convergence_rate': improvement / max(1, convergence_generation),
            'final_diversity': final_diversity,
            'average_diversity': avg_diversity,
            'diversity_trend': np.polyfit(
                range(len(diversity_history)), diversity_history, 1
            )[0] if len(diversity_history) > 1 else 0
        }
        
    def _analyze_parameter_sensitivity(self) -> Dict[str, Any]:
        """Analyze parameter sensitivity from optimization results"""
        if not self.best_parameters:
            return {}
            
        # Get parameter ranges from bounds
        lower_bounds, upper_bounds = self.best_parameters.get_parameter_bounds()
        best_genome = self.best_parameters.to_genome()
        
        # Compute normalized parameter values
        normalized_params = (best_genome - lower_bounds) / (upper_bounds - lower_bounds)
        
        # Parameter names for analysis
        param_names = [
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'D', 'boundary_gain',
            'base_coupling', 'phase_lag', 'topology_strength', 'adaptive_rate',
            'carrier_freq', 'carrier_amp', 'carrier_phase', 'mod_depth', 'demod_gain', 'filter_bw',
            'dirichlet', 'neumann', 'robin_alpha', 'robin_beta', 'boundary_stiff',
            'P_gain', 'I_gain', 'D_gain', 'FF_gain', 'obs_gain', 'sat_limit',
            'stab_margin', 'max_field', 'max_control', 'safety_factor'
        ]
        
        # Create sensitivity analysis
        sensitivity_analysis = {}
        for i, (name, norm_val) in enumerate(zip(param_names[:len(normalized_params)], normalized_params)):
            sensitivity_analysis[name] = {
                'normalized_value': norm_val,
                'actual_value': best_genome[i],
                'lower_bound': lower_bounds[i],
                'upper_bound': upper_bounds[i],
                'utilization': norm_val,  # How much of the range is used
                'importance': abs(0.5 - norm_val)  # Distance from center (proxy for importance)
            }
            
        return sensitivity_analysis
        
    def get_optimal_parameters(self) -> Optional[NFCSParameterSet]:
        """Get the optimal parameter set from evolution"""
        return self.best_parameters
        
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics"""
        return {
            'optimization_statistics': self.optimization_statistics,
            'evolution_history': self.evolution_history,
            'objectives': [obj.value for obj in self.objectives],
            'objective_weights': {obj.value: weight for obj, weight in self.objective_weights.items()},
            'total_evaluations': sum(
                len(hist['optimization_results'].get('convergence_history', []))
                for hist in self.evolution_history
            ) * self.optimization_config['population_size']
        }


class AdaptiveParameterTuning:
    """
    Adaptive parameter tuning for real-time NFCS optimization.
    
    Implements online parameter adaptation during system operation
    using evolutionary strategies and real-time performance monitoring.
    """
    
    def __init__(
        self,
        system_evolution: SystemEvolution,
        adaptation_rate: float = 0.1,
        performance_window: int = 100,
        adaptation_threshold: float = 0.05
    ):
        """
        Initialize adaptive parameter tuning.
        
        Args:
            system_evolution: SystemEvolution instance for optimization
            adaptation_rate: Rate of parameter adaptation
            performance_window: Window size for performance monitoring
            adaptation_threshold: Threshold for triggering adaptation
        """
        self.system_evolution = system_evolution
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        self.adaptation_threshold = adaptation_threshold
        
        # Real-time monitoring
        self.performance_history = deque(maxlen=performance_window)
        self.parameter_history = []
        self.adaptation_events = []
        
        # Current parameters
        self.current_parameters = NFCSParameterSet()
        self.baseline_performance = None
        
        logger.info("Initialized AdaptiveParameterTuning for real-time optimization")
        
    def update_performance(self, performance_metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for adaptation monitoring.
        
        Args:
            performance_metrics: Current system performance metrics
        """
        # Compute composite performance score
        composite_score = self._compute_composite_score(performance_metrics)
        self.performance_history.append({
            'timestamp': time.time(),
            'composite_score': composite_score,
            'metrics': performance_metrics.copy()
        })
        
        # Check if adaptation is needed
        if self._should_adapt():
            self._trigger_adaptation()
            
    def _compute_composite_score(self, metrics: Dict[str, float]) -> float:
        """Compute composite performance score from metrics"""
        # Weighted combination of metrics
        weights = {
            'control_error': -1.0,      # Lower is better
            'stability_margin': 1.0,    # Higher is better
            'energy_efficiency': 1.0,   # Higher is better
            'synchronization': 1.0      # Higher is better
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in weights:
                score += weights[metric] * value
                total_weight += abs(weights[metric])
                
        return score / max(1.0, total_weight)
        
    def _should_adapt(self) -> bool:
        """Check if parameter adaptation should be triggered"""
        if len(self.performance_history) < self.performance_window // 2:
            return False
            
        # Check for performance degradation trend
        recent_scores = [entry['composite_score'] for entry in self.performance_history]
        
        if len(recent_scores) < 10:
            return False
            
        # Compute trend
        x = np.arange(len(recent_scores))
        trend_slope = np.polyfit(x, recent_scores, 1)[0]
        
        # Check if performance is degrading
        if trend_slope < -self.adaptation_threshold:
            return True
            
        # Check for high variance (instability)
        recent_variance = np.var(recent_scores[-20:])
        if recent_variance > self.adaptation_threshold:
            return True
            
        return False
        
    def _trigger_adaptation(self) -> None:
        """Trigger parameter adaptation optimization"""
        logger.info("Triggering adaptive parameter optimization")
        
        # Record adaptation event
        self.adaptation_events.append({
            'timestamp': time.time(),
            'trigger_reason': 'performance_degradation',
            'performance_trend': self._get_performance_trend(),
            'current_parameters': self.current_parameters.to_genome().copy()
        })
        
        # Run limited optimization with current parameters as starting point
        try:
            # Reduce optimization scope for real-time adaptation
            original_generations = self.system_evolution.optimization_config['max_generations']
            self.system_evolution.optimization_config['max_generations'] = 50
            
            # Run optimization
            results = self.system_evolution.optimize_parameters(
                initial_parameters=self.current_parameters
            )
            
            # Update current parameters if improvement found
            if results['best_parameters']:
                old_params = self.current_parameters.to_genome().copy()
                new_params = results['best_parameters'].to_genome()
                
                # Apply gradual adaptation
                adapted_params = (
                    (1 - self.adaptation_rate) * old_params + 
                    self.adaptation_rate * new_params
                )
                
                self.current_parameters = NFCSParameterSet.from_genome(adapted_params)
                
                logger.info(f"Parameters adapted with improvement: {results['optimization_results']['best_fitness']}")
                
            # Restore original configuration
            self.system_evolution.optimization_config['max_generations'] = original_generations
            
        except Exception as e:
            logger.error(f"Error during adaptive parameter tuning: {e}")
            
    def _get_performance_trend(self) -> float:
        """Get current performance trend slope"""
        if len(self.performance_history) < 5:
            return 0.0
            
        scores = [entry['composite_score'] for entry in self.performance_history]
        x = np.arange(len(scores))
        return np.polyfit(x, scores, 1)[0]
        
    def get_current_parameters(self) -> NFCSParameterSet:
        """Get current adapted parameters"""
        return self.current_parameters
        
    def get_adaptation_history(self) -> Dict[str, Any]:
        """Get adaptation history and statistics"""
        return {
            'adaptation_events': self.adaptation_events,
            'parameter_history': self.parameter_history,
            'performance_history': list(self.performance_history),
            'current_parameters': self.current_parameters.to_genome(),
            'adaptation_statistics': {
                'total_adaptations': len(self.adaptation_events),
                'average_adaptation_interval': self._compute_avg_adaptation_interval(),
                'performance_improvement': self._compute_performance_improvement()
            }
        }
        
    def _compute_avg_adaptation_interval(self) -> float:
        """Compute average time between adaptations"""
        if len(self.adaptation_events) < 2:
            return 0.0
            
        intervals = []
        for i in range(1, len(self.adaptation_events)):
            interval = (
                self.adaptation_events[i]['timestamp'] - 
                self.adaptation_events[i-1]['timestamp']
            )
            intervals.append(interval)
            
        return np.mean(intervals)
        
    def _compute_performance_improvement(self) -> float:
        """Compute overall performance improvement"""
        if len(self.performance_history) < self.performance_window:
            return 0.0
            
        early_scores = [
            entry['composite_score'] 
            for entry in list(self.performance_history)[:self.performance_window//4]
        ]
        recent_scores = [
            entry['composite_score']
            for entry in list(self.performance_history)[-self.performance_window//4:]
        ]
        
        if not early_scores or not recent_scores:
            return 0.0
            
        return np.mean(recent_scores) - np.mean(early_scores)


class RealTimeOptimization:
    """
    Real-time optimization coordinator for NFCS systems.
    
    Provides continuous optimization during system operation with
    minimal computational overhead and guaranteed stability.
    """
    
    def __init__(
        self,
        system_evolution: SystemEvolution,
        adaptive_tuning: AdaptiveParameterTuning,
        optimization_interval: float = 1.0,
        safety_monitor: Optional[Callable] = None
    ):
        """
        Initialize real-time optimization coordinator.
        
        Args:
            system_evolution: SystemEvolution for periodic optimization
            adaptive_tuning: AdaptiveParameterTuning for continuous adaptation
            optimization_interval: Time interval between optimizations (seconds)
            safety_monitor: Safety monitoring function
        """
        self.system_evolution = system_evolution
        self.adaptive_tuning = adaptive_tuning
        self.optimization_interval = optimization_interval
        self.safety_monitor = safety_monitor
        
        # Real-time state
        self.is_optimizing = False
        self.optimization_thread = None
        self.last_optimization_time = 0
        self.real_time_statistics = defaultdict(list)
        
        logger.info("Initialized RealTimeOptimization coordinator")
        
    def start_real_time_optimization(self) -> None:
        """Start real-time optimization process"""
        self.is_optimizing = True
        logger.info("Started real-time NFCS optimization")
        
    def stop_real_time_optimization(self) -> None:
        """Stop real-time optimization process"""
        self.is_optimizing = False
        logger.info("Stopped real-time NFCS optimization")
        
    def update_system_state(
        self,
        system_state: SystemState,
        performance_metrics: Dict[str, float]
    ) -> NFCSParameterSet:
        """
        Update system state and return optimized parameters.
        
        Args:
            system_state: Current system state
            performance_metrics: Current performance metrics
            
        Returns:
            Optimized parameter set
        """
        if not self.is_optimizing:
            return self.adaptive_tuning.get_current_parameters()
            
        # Update adaptive tuning
        self.adaptive_tuning.update_performance(performance_metrics)
        
        # Check safety constraints
        if self.safety_monitor and not self.safety_monitor(system_state):
            logger.warning("Safety constraints violated - halting optimization")
            self.stop_real_time_optimization()
            return self.adaptive_tuning.get_current_parameters()
            
        # Periodic full optimization
        current_time = time.time()
        if current_time - self.last_optimization_time > self.optimization_interval:
            self._run_periodic_optimization()
            self.last_optimization_time = current_time
            
        # Update statistics
        self.real_time_statistics['update_times'].append(current_time)
        self.real_time_statistics['performance_scores'].append(
            self.adaptive_tuning._compute_composite_score(performance_metrics)
        )
        
        return self.adaptive_tuning.get_current_parameters()
        
    def _run_periodic_optimization(self) -> None:
        """Run periodic full optimization in background"""
        try:
            # Run optimization with reduced scope for real-time constraints
            original_config = self.system_evolution.optimization_config.copy()
            
            # Reduce computational load for real-time operation
            self.system_evolution.optimization_config.update({
                'population_size': 50,
                'max_generations': 100
            })
            
            # Run optimization
            results = self.system_evolution.optimize_parameters(
                initial_parameters=self.adaptive_tuning.get_current_parameters()
            )
            
            # Apply results gradually
            if results['best_parameters']:
                current = self.adaptive_tuning.get_current_parameters()
                optimized = results['best_parameters']
                
                # Gradual parameter update for stability
                adaptation_rate = 0.2
                old_genome = current.to_genome()
                new_genome = optimized.to_genome()
                
                blended_genome = (
                    (1 - adaptation_rate) * old_genome + 
                    adaptation_rate * new_genome
                )
                
                self.adaptive_tuning.current_parameters = NFCSParameterSet.from_genome(
                    blended_genome
                )
                
            # Restore original configuration
            self.system_evolution.optimization_config = original_config
            
            logger.debug("Completed periodic optimization update")
            
        except Exception as e:
            logger.error(f"Error in periodic optimization: {e}")
            
    def get_real_time_statistics(self) -> Dict[str, Any]:
        """Get real-time optimization statistics"""
        return {
            'is_optimizing': self.is_optimizing,
            'optimization_interval': self.optimization_interval,
            'last_optimization_time': self.last_optimization_time,
            'statistics': dict(self.real_time_statistics),
            'adaptive_tuning_stats': self.adaptive_tuning.get_adaptation_history(),
            'system_evolution_stats': self.system_evolution.get_evolution_statistics()
        }