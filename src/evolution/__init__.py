"""
NFCS Evolutionary Algorithms Module
=================================

Implements genetic algorithms and evolutionary optimization for NFCS v2.4.3
system parameter optimization, neural field evolution, and adaptive coupling
matrix optimization based on PDF Section 5.13.

Core Components:
- GeneticOptimizer: Main evolutionary algorithm engine
- SystemEvolution: NFCS-specific parameter evolution 
- NeuroEvolution: Neural network topology and weight evolution
- AdaptiveCoupling: Dynamic Kuramoto coupling matrix optimization
- MultiObjectiveEvolution: Pareto-optimal solution discovery

Scientific Foundation:
Implements advanced evolutionary strategies including:
- Differential Evolution (DE)
- Particle Swarm Optimization (PSO) 
- Genetic Algorithm (GA) with adaptive operators
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
- Multi-Objective Evolutionary Algorithms (MOEA)

Integration:
Seamlessly interfaces with NFCS core modules for real-time system optimization
and parameter adaptation during neural field evolution and synchronization.

Created: September 14, 2025
Author: Team Ω - Neural Field Control Systems Research Group
Version: 2.4.3
"""

from .genetic_optimizer import (
    GeneticOptimizer,
    Individual,
    Population,
    FitnessFunction,
    SelectionStrategy,
    CrossoverOperator,
    MutationOperator,
    EvolutionaryStrategy
)

from .system_evolution import (
    SystemEvolution,
    NFCSParameterSet,
    SystemFitnessEvaluator,
    AdaptiveParameterTuning,
    RealTimeOptimization
)

from .neuro_evolution import (
    NeuroEvolution,
    NetworkTopology,
    WeightEvolution,
    StructuralEvolution,
    NeuralArchitectureSearch
)

from .adaptive_coupling import (
    AdaptiveCoupling,
    CouplingMatrixEvolution,
    SynchronizationOptimizer,
    DynamicTopologyEvolution,
    KuramotoEvolution
)

from .multi_objective import (
    MultiObjectiveEvolution,
    ParetoFront,
    DominanceComparator,
    HyperVolumeCalculator,
    NSGA2Algorithm,
    MOEA_D_Algorithm
)

__all__ = [
    # Core genetic algorithm components
    'GeneticOptimizer',
    'Individual', 
    'Population',
    'FitnessFunction',
    'SelectionStrategy',
    'CrossoverOperator', 
    'MutationOperator',
    'EvolutionaryStrategy',
    
    # System evolution components
    'SystemEvolution',
    'NFCSParameterSet',
    'SystemFitnessEvaluator', 
    'AdaptiveParameterTuning',
    'RealTimeOptimization',
    
    # Neural evolution components
    'NeuroEvolution',
    'NetworkTopology',
    'WeightEvolution',
    'StructuralEvolution', 
    'NeuralArchitectureSearch',
    
    # Adaptive coupling components  
    'AdaptiveCoupling',
    'CouplingMatrixEvolution',
    'SynchronizationOptimizer',
    'DynamicTopologyEvolution',
    'KuramotoEvolution',
    
    # Multi-objective components
    'MultiObjectiveEvolution',
    'ParetoFront',
    'DominanceComparator',
    'HyperVolumeCalculator', 
    'NSGA2Algorithm',
    'MOEA_D_Algorithm'
]

__version__ = "2.4.3"
__author__ = "Team Ω - Neural Field Control Systems Research Group"
__date__ = "September 14, 2025"