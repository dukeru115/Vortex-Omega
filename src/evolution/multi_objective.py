"""
Multi-Objective Evolution - Pareto-Optimal NFCS Optimization
==========================================================

Implements multi-objective evolutionary algorithms for NFCS v2.4.3 optimization.
Provides Pareto-front discovery, NSGA-II, MOEA/D algorithms for simultaneous
optimization of conflicting objectives in neural field control systems.

Scientific Foundation:
Based on Pareto optimality theory and multi-objective optimization:
- Pareto dominance relations for solution ranking
- Non-dominated sorting for population classification
- Crowding distance for diversity preservation
- Hypervolume indicator for convergence assessment
- Reference point methods for preference articulation

Mathematical Framework:
- Multi-objective problem: minimize f(x) = [f₁(x), f₂(x), ..., fₘ(x)]
- Pareto dominance: x₁ ≺ x₂ if f_i(x₁) ≤ f_i(x₂) ∀i and ∃j: f_j(x₁) < f_j(x₂)
- Hypervolume: HV(S) = volume of union of hypercubes dominated by solutions in S
- Crowding distance: measure of solution density in objective space

Integration:
Optimizes NFCS systems with multiple conflicting objectives: control performance,
energy efficiency, stability margin, synchronization quality, and safety constraints.

Created: September 14, 2025
Author: Team Ω - Neural Field Control Systems Research Group
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import random
import copy
from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
from scipy import stats
from scipy.spatial.distance import cdist, euclidean
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from .genetic_optimizer import Individual, Population, FitnessFunction, GeneticOptimizer
from .system_evolution import OptimizationObjective, NFCSParameterSet

logger = logging.getLogger(__name__)


class DominanceRelation(Enum):
    """Dominance relations between solutions"""
    DOMINATES = "dominates"
    DOMINATED_BY = "dominated_by"
    NON_DOMINATED = "non_dominated"
    INCOMPARABLE = "incomparable"


class SelectionMethod(Enum):
    """Selection methods for multi-objective optimization"""
    NSGA2 = "nsga2"
    SPEA2 = "spea2"
    MOEAD = "moea_d"
    INDICATOR_BASED = "indicator_based"
    REFERENCE_POINT = "reference_point"


class QualityIndicator(Enum):
    """Quality indicators for multi-objective optimization"""
    HYPERVOLUME = "hypervolume"
    GENERATIONAL_DISTANCE = "generational_distance"
    INVERTED_GENERATIONAL_DISTANCE = "inverted_generational_distance"
    SPACING = "spacing"
    SPREAD = "spread"
    EPSILON_INDICATOR = "epsilon_indicator"


@dataclass
class ParetoFront:
    """
    Represents a Pareto front of non-dominated solutions.
    
    Attributes:
        solutions: List of non-dominated individuals
        objectives: Objective function values
        generation: Generation when front was computed
        quality_metrics: Quality indicators for the front
        reference_point: Reference point for hypervolume calculation
        ideal_point: Ideal point in objective space
        nadir_point: Nadir point in objective space
    """
    solutions: List[Individual] = field(default_factory=list)
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    generation: int = 0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    reference_point: Optional[np.ndarray] = None
    ideal_point: Optional[np.ndarray] = None
    nadir_point: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize derived attributes"""
        if len(self.solutions) > 0 and len(self.objectives) == 0:
            self.objectives = np.array([ind.objectives for ind in self.solutions])
            
        if len(self.objectives) > 0:
            self.ideal_point = np.min(self.objectives, axis=0)
            self.nadir_point = np.max(self.objectives, axis=0)
            
            if self.reference_point is None:
                # Set reference point slightly worse than nadir
                self.reference_point = self.nadir_point + 0.1 * (self.nadir_point - self.ideal_point)
    
    def add_solution(self, individual: Individual) -> None:
        """Add solution to Pareto front"""
        self.solutions.append(individual)
        
        # Update objectives matrix
        if len(self.objectives) == 0:
            self.objectives = np.array([individual.objectives])
        else:
            self.objectives = np.vstack([self.objectives, individual.objectives])
            
        # Update ideal and nadir points
        if len(self.objectives) > 0:
            self.ideal_point = np.min(self.objectives, axis=0)
            self.nadir_point = np.max(self.objectives, axis=0)
    
    def remove_solution(self, index: int) -> None:
        """Remove solution from Pareto front"""
        if 0 <= index < len(self.solutions):
            del self.solutions[index]
            self.objectives = np.delete(self.objectives, index, axis=0)
            
            # Update ideal and nadir points
            if len(self.objectives) > 0:
                self.ideal_point = np.min(self.objectives, axis=0)
                self.nadir_point = np.max(self.objectives, axis=0)
    
    def is_dominated_by(self, objectives: np.ndarray) -> bool:
        """Check if given objectives dominate any solution in the front"""
        if len(self.objectives) == 0:
            return False
            
        # Check if objectives dominate any solution in the front
        for front_obj in self.objectives:
            if np.all(objectives <= front_obj) and np.any(objectives < front_obj):
                return True
        return False
    
    def dominates(self, objectives: np.ndarray) -> bool:
        """Check if any solution in the front dominates given objectives"""
        if len(self.objectives) == 0:
            return False
            
        # Check if any solution in the front dominates objectives
        for front_obj in self.objectives:
            if np.all(front_obj <= objectives) and np.any(front_obj < objectives):
                return True
        return False
    
    def update_quality_metrics(self, quality_indicators: List[QualityIndicator] = None) -> None:
        """Update quality metrics for the Pareto front"""
        if len(self.objectives) == 0:
            return
            
        if quality_indicators is None:
            quality_indicators = [
                QualityIndicator.HYPERVOLUME,
                QualityIndicator.SPACING,
                QualityIndicator.SPREAD
            ]
        
        for indicator in quality_indicators:
            if indicator == QualityIndicator.HYPERVOLUME:
                self.quality_metrics['hypervolume'] = self._calculate_hypervolume()
            elif indicator == QualityIndicator.SPACING:
                self.quality_metrics['spacing'] = self._calculate_spacing()
            elif indicator == QualityIndicator.SPREAD:
                self.quality_metrics['spread'] = self._calculate_spread()
    
    def _calculate_hypervolume(self) -> float:
        """Calculate hypervolume indicator"""
        if len(self.objectives) == 0 or self.reference_point is None:
            return 0.0
            
        # Simple hypervolume calculation for 2D and 3D cases
        if self.objectives.shape[1] == 2:
            return self._hypervolume_2d()
        elif self.objectives.shape[1] == 3:
            return self._hypervolume_3d()
        else:
            # Approximate hypervolume for higher dimensions
            return self._hypervolume_monte_carlo()
    
    def _hypervolume_2d(self) -> float:
        """Calculate 2D hypervolume using sweeping algorithm"""
        if len(self.objectives) == 0:
            return 0.0
            
        # Sort by first objective
        sorted_indices = np.argsort(self.objectives[:, 0])
        sorted_objectives = self.objectives[sorted_indices]
        
        hypervolume = 0.0
        prev_y = self.reference_point[1]
        
        for obj in sorted_objectives:
            if obj[1] < prev_y:
                width = self.reference_point[0] - obj[0]
                height = prev_y - obj[1]
                hypervolume += width * height
                prev_y = obj[1]
        
        return max(0.0, hypervolume)
    
    def _hypervolume_3d(self) -> float:
        """Calculate 3D hypervolume using sweeping algorithm"""
        if len(self.objectives) == 0:
            return 0.0
            
        # Simplified 3D hypervolume calculation
        total_volume = 0.0
        
        for i, obj in enumerate(self.objectives):
            # Calculate volume contribution of this point
            dominated_volume = np.prod(np.maximum(0, self.reference_point - obj))
            
            # Subtract overlapping volumes (approximation)
            for j, other_obj in enumerate(self.objectives):
                if i != j:
                    # Check for overlap
                    overlap_min = np.maximum(obj, other_obj)
                    overlap_volume = np.prod(np.maximum(0, self.reference_point - overlap_min))
                    dominated_volume -= 0.5 * overlap_volume  # Rough approximation
            
            total_volume += max(0.0, dominated_volume)
        
        return total_volume
    
    def _hypervolume_monte_carlo(self, samples: int = 10000) -> float:
        """Calculate hypervolume using Monte Carlo sampling"""
        if len(self.objectives) == 0:
            return 0.0
            
        # Sample random points in the reference volume
        dim = len(self.reference_point)
        dominated_count = 0
        
        for _ in range(samples):
            # Generate random point
            random_point = np.random.uniform(self.ideal_point, self.reference_point)
            
            # Check if dominated by any solution in the front
            dominated = False
            for obj in self.objectives:
                if np.all(obj <= random_point):
                    dominated = True
                    break
            
            if dominated:
                dominated_count += 1
        
        # Calculate hypervolume
        reference_volume = np.prod(self.reference_point - self.ideal_point)
        hypervolume = (dominated_count / samples) * reference_volume
        
        return hypervolume
    
    def _calculate_spacing(self) -> float:
        """Calculate spacing metric (uniformity of distribution)"""
        if len(self.objectives) <= 1:
            return 0.0
            
        # Calculate distances between consecutive solutions
        distances = []
        
        for i in range(len(self.objectives)):
            min_distance = float('inf')
            for j in range(len(self.objectives)):
                if i != j:
                    distance = np.linalg.norm(self.objectives[i] - self.objectives[j])
                    min_distance = min(min_distance, distance)
            distances.append(min_distance)
        
        # Spacing is the standard deviation of distances
        return np.std(distances)
    
    def _calculate_spread(self) -> float:
        """Calculate spread metric (extent of the front)"""
        if len(self.objectives) <= 1:
            return 0.0
            
        # Calculate range in each objective
        ranges = self.nadir_point - self.ideal_point
        
        # Spread is the sum of normalized ranges
        max_range = np.max(ranges)
        if max_range > 0:
            normalized_ranges = ranges / max_range
            return np.sum(normalized_ranges) / len(ranges)
        
        return 0.0
    
    def copy(self) -> 'ParetoFront':
        """Create a deep copy of the Pareto front"""
        return ParetoFront(
            solutions=[sol.copy() for sol in self.solutions],
            objectives=self.objectives.copy() if len(self.objectives) > 0 else np.array([]),
            generation=self.generation,
            quality_metrics=self.quality_metrics.copy(),
            reference_point=self.reference_point.copy() if self.reference_point is not None else None,
            ideal_point=self.ideal_point.copy() if self.ideal_point is not None else None,
            nadir_point=self.nadir_point.copy() if self.nadir_point is not None else None
        )


class DominanceComparator:
    """
    Comparator for Pareto dominance relations between solutions.
    """
    
    @staticmethod
    def dominates(obj1: np.ndarray, obj2: np.ndarray, epsilon: float = 1e-10) -> bool:
        """
        Check if obj1 dominates obj2.
        
        Args:
            obj1: First objective vector
            obj2: Second objective vector
            epsilon: Tolerance for floating-point comparison
            
        Returns:
            True if obj1 dominates obj2
        """
        # obj1 dominates obj2 if:
        # 1. obj1 is at least as good in all objectives
        # 2. obj1 is strictly better in at least one objective
        
        at_least_as_good = np.all(obj1 <= obj2 + epsilon)
        strictly_better = np.any(obj1 < obj2 - epsilon)
        
        return at_least_as_good and strictly_better
    
    @staticmethod
    def compare(obj1: np.ndarray, obj2: np.ndarray, epsilon: float = 1e-10) -> DominanceRelation:
        """
        Compare two objective vectors and return dominance relation.
        
        Args:
            obj1: First objective vector
            obj2: Second objective vector
            epsilon: Tolerance for floating-point comparison
            
        Returns:
            Dominance relation between the vectors
        """
        if DominanceComparator.dominates(obj1, obj2, epsilon):
            return DominanceRelation.DOMINATES
        elif DominanceComparator.dominates(obj2, obj1, epsilon):
            return DominanceRelation.DOMINATED_BY
        else:
            # Check if they are approximately equal
            if np.allclose(obj1, obj2, atol=epsilon):
                return DominanceRelation.NON_DOMINATED
            else:
                return DominanceRelation.INCOMPARABLE
    
    @staticmethod
    def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
        """
        Perform fast non-dominated sorting (NSGA-II algorithm).
        
        Args:
            population: Population to sort
            
        Returns:
            List of fronts, where each front is a list of non-dominated individuals
        """
        if not population:
            return []
        
        # Initialize dominance counts and dominated solutions
        domination_counts = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]
        fronts = [[] for _ in range(len(population))]
        
        # Compare all pairs of solutions
        for i, ind1 in enumerate(population):
            for j, ind2 in enumerate(population):
                if i != j:
                    relation = DominanceComparator.compare(ind1.objectives, ind2.objectives)
                    
                    if relation == DominanceRelation.DOMINATES:
                        dominated_solutions[i].append(j)
                    elif relation == DominanceRelation.DOMINATED_BY:
                        domination_counts[i] += 1
            
            # If not dominated by anyone, belongs to first front
            if domination_counts[i] == 0:
                ind1.rank = 0
                fronts[0].append(ind1)
        
        # Generate subsequent fronts
        front_index = 0
        while fronts[front_index]:
            next_front = []
            
            for ind1 in fronts[front_index]:
                ind1_index = population.index(ind1)
                
                # For each solution dominated by current solution
                for j in dominated_solutions[ind1_index]:
                    domination_counts[j] -= 1
                    
                    # If no longer dominated by anyone
                    if domination_counts[j] == 0:
                        population[j].rank = front_index + 1
                        next_front.append(population[j])
            
            front_index += 1
            if next_front:
                fronts[front_index] = next_front
            else:
                break
        
        # Return only non-empty fronts
        return [front for front in fronts if front]
    
    @staticmethod
    def calculate_crowding_distance(front: List[Individual]) -> None:
        """
        Calculate crowding distance for individuals in a front.
        
        Args:
            front: List of individuals in the same front
        """
        if len(front) <= 2:
            # Boundary solutions get infinite distance
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        # Initialize crowding distances
        for ind in front:
            ind.crowding_distance = 0.0
        
        # Get number of objectives
        num_objectives = len(front[0].objectives)
        
        # Calculate distance for each objective
        for obj_index in range(num_objectives):
            # Sort by current objective
            front.sort(key=lambda x: x.objectives[obj_index])
            
            # Boundary solutions get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range of objective
            obj_range = front[-1].objectives[obj_index] - front[0].objectives[obj_index]
            
            if obj_range > 0:
                # Add distance contribution for intermediate solutions
                for i in range(1, len(front) - 1):
                    distance = (front[i + 1].objectives[obj_index] - 
                              front[i - 1].objectives[obj_index]) / obj_range
                    front[i].crowding_distance += distance


class HyperVolumeCalculator:
    """
    Calculator for hypervolume indicator and related metrics.
    """
    
    def __init__(self, reference_point: np.ndarray):
        """
        Initialize hypervolume calculator.
        
        Args:
            reference_point: Reference point for hypervolume calculation
        """
        self.reference_point = reference_point
        
    def calculate_hypervolume(self, objectives: np.ndarray) -> float:
        """
        Calculate hypervolume indicator.
        
        Args:
            objectives: Matrix of objective vectors (n_points × n_objectives)
            
        Returns:
            Hypervolume value
        """
        if len(objectives) == 0:
            return 0.0
            
        # Filter dominated solutions
        non_dominated_indices = self._get_non_dominated_indices(objectives)
        non_dominated_objectives = objectives[non_dominated_indices]
        
        if len(non_dominated_objectives) == 0:
            return 0.0
        
        # Calculate hypervolume based on dimensionality
        if objectives.shape[1] == 1:
            return self._hypervolume_1d(non_dominated_objectives)
        elif objectives.shape[1] == 2:
            return self._hypervolume_2d(non_dominated_objectives)
        elif objectives.shape[1] == 3:
            return self._hypervolume_3d(non_dominated_objectives)
        else:
            return self._hypervolume_monte_carlo(non_dominated_objectives)
    
    def _get_non_dominated_indices(self, objectives: np.ndarray) -> np.ndarray:
        """Get indices of non-dominated solutions"""
        n_points = len(objectives)
        is_dominated = np.zeros(n_points, dtype=bool)
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j and not is_dominated[i]:
                    if DominanceComparator.dominates(objectives[j], objectives[i]):
                        is_dominated[i] = True
                        break
        
        return np.where(~is_dominated)[0]
    
    def _hypervolume_1d(self, objectives: np.ndarray) -> float:
        """Calculate 1D hypervolume"""
        min_obj = np.min(objectives[:, 0])
        return max(0.0, self.reference_point[0] - min_obj)
    
    def _hypervolume_2d(self, objectives: np.ndarray) -> float:
        """Calculate 2D hypervolume using sweeping algorithm"""
        # Sort by first objective (ascending)
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_objectives = objectives[sorted_indices]
        
        hypervolume = 0.0
        prev_y = self.reference_point[1]
        
        for obj in sorted_objectives:
            if obj[1] < prev_y:
                width = self.reference_point[0] - obj[0]
                height = prev_y - obj[1]
                hypervolume += max(0.0, width * height)
                prev_y = obj[1]
        
        return hypervolume
    
    def _hypervolume_3d(self, objectives: np.ndarray) -> float:
        """Calculate 3D hypervolume using inclusion-exclusion principle"""
        hypervolume = 0.0
        n_points = len(objectives)
        
        # Single points contribution
        for i in range(n_points):
            contribution = np.prod(np.maximum(0, self.reference_point - objectives[i]))
            hypervolume += contribution
        
        # Subtract pairwise intersections
        for i in range(n_points):
            for j in range(i + 1, n_points):
                intersection_point = np.maximum(objectives[i], objectives[j])
                intersection_volume = np.prod(np.maximum(0, self.reference_point - intersection_point))
                hypervolume -= intersection_volume
        
        # Add triple intersections (simplified)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                for k in range(j + 1, n_points):
                    intersection_point = np.maximum.reduce([objectives[i], objectives[j], objectives[k]])
                    intersection_volume = np.prod(np.maximum(0, self.reference_point - intersection_point))
                    hypervolume += intersection_volume
        
        return max(0.0, hypervolume)
    
    def _hypervolume_monte_carlo(self, objectives: np.ndarray, samples: int = 50000) -> float:
        """Calculate hypervolume using Monte Carlo sampling"""
        # Determine sampling bounds
        ideal_point = np.min(objectives, axis=0)
        
        # Sample random points
        dominated_count = 0
        
        for _ in range(samples):
            random_point = np.random.uniform(ideal_point, self.reference_point)
            
            # Check if point is dominated by any solution
            for obj in objectives:
                if np.all(obj <= random_point):
                    dominated_count += 1
                    break
        
        # Calculate hypervolume
        sampling_volume = np.prod(self.reference_point - ideal_point)
        hypervolume = (dominated_count / samples) * sampling_volume
        
        return hypervolume
    
    def calculate_hypervolume_contribution(self, objectives: np.ndarray, point_index: int) -> float:
        """Calculate hypervolume contribution of a specific point"""
        if len(objectives) <= 1:
            return self.calculate_hypervolume(objectives)
        
        # Calculate hypervolume with and without the point
        full_hypervolume = self.calculate_hypervolume(objectives)
        
        reduced_objectives = np.delete(objectives, point_index, axis=0)
        reduced_hypervolume = self.calculate_hypervolume(reduced_objectives)
        
        return full_hypervolume - reduced_hypervolume


class NSGA2Algorithm:
    """
    Implementation of NSGA-II (Non-dominated Sorting Genetic Algorithm II).
    
    NSGA-II is a popular multi-objective evolutionary algorithm that uses
    non-dominated sorting and crowding distance for selection.
    """
    
    def __init__(
        self,
        fitness_function: FitnessFunction,
        population_size: int = 100,
        genome_length: int = 10,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        max_generations: int = 250,
        tournament_size: int = 2
    ):
        """
        Initialize NSGA-II algorithm.
        
        Args:
            fitness_function: Multi-objective fitness function
            population_size: Size of the population
            genome_length: Length of individual genomes
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            max_generations: Maximum number of generations
            tournament_size: Tournament selection size
        """
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.genome_length = genome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        
        # Algorithm state
        self.population = []
        self.generation = 0
        self.pareto_front = ParetoFront()
        self.evolution_history = []
        
        logger.info(f"Initialized NSGA-II with population size {population_size}")
    
    def initialize_population(self) -> None:
        """Initialize population with random individuals"""
        self.population = []
        lower_bounds, upper_bounds = self.fitness_function.get_bounds()
        
        for _ in range(self.population_size):
            genome = np.random.uniform(lower_bounds, upper_bounds, self.genome_length)
            individual = Individual(genome=genome)
            self.population.append(individual)
        
        # Evaluate initial population
        self._evaluate_population()
        
        logger.info(f"Initialized NSGA-II population with {len(self.population)} individuals")
    
    def evolve(self) -> Dict[str, Any]:
        """
        Run NSGA-II evolution.
        
        Returns:
            Evolution results including Pareto front
        """
        start_time = time.time()
        
        if not self.population:
            self.initialize_population()
        
        hypervolume_history = []
        front_size_history = []
        
        for generation in range(self.max_generations):
            self.generation = generation
            
            # Generate offspring population
            offspring = self._generate_offspring()
            
            # Combine parent and offspring populations
            combined_population = self.population + offspring
            
            # Evaluate combined population
            self._evaluate_population(combined_population)
            
            # Non-dominated sorting
            fronts = DominanceComparator.fast_non_dominated_sort(combined_population)
            
            # Select next generation
            self.population = []
            front_index = 0
            
            # Add complete fronts
            while (len(self.population) + len(fronts[front_index]) <= self.population_size and
                   front_index < len(fronts)):
                
                # Calculate crowding distance for current front
                DominanceComparator.calculate_crowding_distance(fronts[front_index])
                
                self.population.extend(fronts[front_index])
                front_index += 1
            
            # Add partial front if needed
            if len(self.population) < self.population_size and front_index < len(fronts):
                remaining_slots = self.population_size - len(self.population)
                
                # Calculate crowding distance for the last front
                DominanceComparator.calculate_crowding_distance(fronts[front_index])
                
                # Sort by crowding distance (descending)
                fronts[front_index].sort(key=lambda x: x.crowding_distance, reverse=True)
                
                # Add individuals with highest crowding distance
                self.population.extend(fronts[front_index][:remaining_slots])
            
            # Update Pareto front
            if fronts:
                self.pareto_front = ParetoFront(
                    solutions=fronts[0].copy(),
                    generation=generation
                )
                self.pareto_front.update_quality_metrics()
                
                # Track convergence metrics
                if 'hypervolume' in self.pareto_front.quality_metrics:
                    hypervolume_history.append(self.pareto_front.quality_metrics['hypervolume'])
                front_size_history.append(len(fronts[0]))
            
            # Log progress
            if generation % 50 == 0:
                avg_front_size = len(fronts[0]) if fronts else 0
                logger.info(f"NSGA-II Generation {generation}: Front size = {avg_front_size}")
        
        execution_time = time.time() - start_time
        
        # Compile results
        results = {
            'algorithm': 'NSGA-II',
            'pareto_front': self.pareto_front,
            'final_population': self.population,
            'generations_run': self.generation + 1,
            'execution_time': execution_time,
            'convergence_metrics': {
                'hypervolume_history': hypervolume_history,
                'front_size_history': front_size_history,
                'final_hypervolume': hypervolume_history[-1] if hypervolume_history else 0.0,
                'final_front_size': len(self.pareto_front.solutions)
            },
            'quality_indicators': self.pareto_front.quality_metrics
        }
        
        logger.info(
            f"NSGA-II completed in {execution_time:.2f}s. "
            f"Final Pareto front: {len(self.pareto_front.solutions)} solutions"
        )
        
        return results
    
    def _evaluate_population(self, population: List[Individual] = None) -> None:
        """Evaluate fitness for population"""
        if population is None:
            population = self.population
        
        for individual in population:
            objectives = self.fitness_function.evaluate(individual.genome)
            if isinstance(objectives, (int, float)):
                individual.objectives = np.array([objectives])
            else:
                individual.objectives = np.array(objectives)
    
    def _generate_offspring(self) -> List[Individual]:
        """Generate offspring through selection, crossover, and mutation"""
        offspring = []
        
        while len(offspring) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if np.random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _tournament_selection(self) -> Individual:
        """Tournament selection considering dominance and crowding distance"""
        tournament = random.sample(self.population, self.tournament_size)
        
        # Select best individual from tournament
        best = tournament[0]
        
        for individual in tournament[1:]:
            if self._compare_individuals(individual, best) > 0:
                best = individual
        
        return best
    
    def _compare_individuals(self, ind1: Individual, ind2: Individual) -> int:
        """
        Compare individuals based on dominance and crowding distance.
        
        Returns:
            1 if ind1 is better, -1 if ind2 is better, 0 if equal
        """
        # First compare by rank (lower is better)
        if hasattr(ind1, 'rank') and hasattr(ind2, 'rank'):
            if ind1.rank < ind2.rank:
                return 1
            elif ind1.rank > ind2.rank:
                return -1
        
        # If same rank, compare by crowding distance (higher is better)
        if hasattr(ind1, 'crowding_distance') and hasattr(ind2, 'crowding_distance'):
            if ind1.crowding_distance > ind2.crowding_distance:
                return 1
            elif ind1.crowding_distance < ind2.crowding_distance:
                return -1
        
        return 0
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Simulated binary crossover (SBX)"""
        eta_c = 20.0  # Distribution index
        
        child1_genome = np.zeros_like(parent1.genome)
        child2_genome = np.zeros_like(parent2.genome)
        
        for i in range(len(parent1.genome)):
            if np.random.random() <= 0.5:
                if abs(parent1.genome[i] - parent2.genome[i]) > 1e-14:
                    # Perform SBX
                    y1, y2 = min(parent1.genome[i], parent2.genome[i]), max(parent1.genome[i], parent2.genome[i])
                    
                    # Get bounds
                    lower_bounds, upper_bounds = self.fitness_function.get_bounds()
                    yl, yu = lower_bounds[i], upper_bounds[i]
                    
                    rand = np.random.random()
                    
                    # Calculate beta
                    if rand <= 0.5:
                        beta = (2.0 * rand) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (eta_c + 1.0))
                    
                    # Generate children
                    c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                    
                    # Apply bounds
                    c1 = max(yl, min(yu, c1))
                    c2 = max(yl, min(yu, c2))
                    
                    child1_genome[i] = c1
                    child2_genome[i] = c2
                else:
                    child1_genome[i] = parent1.genome[i]
                    child2_genome[i] = parent2.genome[i]
            else:
                child1_genome[i] = parent1.genome[i]
                child2_genome[i] = parent2.genome[i]
        
        return Individual(genome=child1_genome), Individual(genome=child2_genome)
    
    def _mutate(self, individual: Individual) -> Individual:
        """Polynomial mutation"""
        eta_m = 20.0  # Distribution index
        mutated = individual.copy()
        
        lower_bounds, upper_bounds = self.fitness_function.get_bounds()
        
        for i in range(len(mutated.genome)):
            if np.random.random() < (1.0 / len(mutated.genome)):
                y = mutated.genome[i]
                yl, yu = lower_bounds[i], upper_bounds[i]
                
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                
                rand = np.random.random()
                mut_pow = 1.0 / (eta_m + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                y_new = y + deltaq * (yu - yl)
                mutated.genome[i] = max(yl, min(yu, y_new))
        
        return mutated


class MOEA_D_Algorithm:
    """
    Implementation of MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition).
    
    MOEA/D decomposes a multi-objective problem into scalar optimization subproblems
    using weight vectors and optimizes them simultaneously.
    """
    
    def __init__(
        self,
        fitness_function: FitnessFunction,
        population_size: int = 100,
        genome_length: int = 10,
        neighborhood_size: int = 20,
        max_generations: int = 250,
        replacement_probability: float = 0.9,
        decomposition_method: str = 'tchebycheff'
    ):
        """
        Initialize MOEA/D algorithm.
        
        Args:
            fitness_function: Multi-objective fitness function
            population_size: Size of the population
            genome_length: Length of individual genomes
            neighborhood_size: Size of neighborhood for each subproblem
            max_generations: Maximum number of generations
            replacement_probability: Probability of replacement in neighborhood
            decomposition_method: Decomposition method ('tchebycheff', 'weighted_sum', 'pbi')
        """
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.genome_length = genome_length
        self.neighborhood_size = neighborhood_size
        self.max_generations = max_generations
        self.replacement_probability = replacement_probability
        self.decomposition_method = decomposition_method
        
        # Generate weight vectors
        num_objectives = self._get_num_objectives()
        self.weight_vectors = self._generate_weight_vectors(num_objectives, population_size)
        
        # Calculate neighborhoods
        self.neighborhoods = self._calculate_neighborhoods()
        
        # Algorithm state
        self.population = []
        self.ideal_point = None
        self.generation = 0
        
        logger.info(f"Initialized MOEA/D with {population_size} subproblems")
    
    def _get_num_objectives(self) -> int:
        """Determine number of objectives from fitness function"""
        # Create a random test individual
        lower_bounds, upper_bounds = self.fitness_function.get_bounds()
        test_genome = np.random.uniform(lower_bounds, upper_bounds, self.genome_length)
        test_objectives = self.fitness_function.evaluate(test_genome)
        
        if isinstance(test_objectives, (int, float)):
            return 1
        else:
            return len(test_objectives)
    
    def _generate_weight_vectors(self, num_objectives: int, num_vectors: int) -> np.ndarray:
        """Generate uniformly distributed weight vectors"""
        if num_objectives == 2:
            # For 2D, use uniform spacing
            weights = np.zeros((num_vectors, 2))
            for i in range(num_vectors):
                w1 = i / (num_vectors - 1)
                w2 = 1.0 - w1
                weights[i] = [w1, w2]
            return weights
        
        elif num_objectives == 3:
            # For 3D, use triangular lattice
            weights = []
            step = 1.0 / np.sqrt(num_vectors)
            
            for i in range(int(1.0 / step) + 1):
                for j in range(int(1.0 / step) + 1 - i):
                    w1 = i * step
                    w2 = j * step
                    w3 = 1.0 - w1 - w2
                    if w3 >= 0:
                        weights.append([w1, w2, w3])
            
            # Adjust to exact number of vectors
            weights = np.array(weights)
            if len(weights) > num_vectors:
                indices = np.random.choice(len(weights), num_vectors, replace=False)
                weights = weights[indices]
            
            return weights
        
        else:
            # For higher dimensions, use random weights
            weights = np.random.random((num_vectors, num_objectives))
            # Normalize to sum to 1
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            return weights
    
    def _calculate_neighborhoods(self) -> List[List[int]]:
        """Calculate neighborhoods for each weight vector"""
        neighborhoods = []
        
        for i in range(self.population_size):
            # Calculate Euclidean distances to all other weight vectors
            distances = []
            for j in range(self.population_size):
                if i != j:
                    distance = np.linalg.norm(self.weight_vectors[i] - self.weight_vectors[j])
                    distances.append((distance, j))
            
            # Sort by distance and select closest neighbors
            distances.sort()
            neighborhood = [i] + [idx for _, idx in distances[:self.neighborhood_size-1]]
            neighborhoods.append(neighborhood)
        
        return neighborhoods
    
    def initialize_population(self) -> None:
        """Initialize population with random individuals"""
        self.population = []
        lower_bounds, upper_bounds = self.fitness_function.get_bounds()
        
        for _ in range(self.population_size):
            genome = np.random.uniform(lower_bounds, upper_bounds, self.genome_length)
            individual = Individual(genome=genome)
            
            # Evaluate objectives
            objectives = self.fitness_function.evaluate(genome)
            if isinstance(objectives, (int, float)):
                individual.objectives = np.array([objectives])
            else:
                individual.objectives = np.array(objectives)
            
            self.population.append(individual)
        
        # Initialize ideal point
        all_objectives = np.array([ind.objectives for ind in self.population])
        self.ideal_point = np.min(all_objectives, axis=0)
        
        logger.info(f"Initialized MOEA/D population with {len(self.population)} individuals")
    
    def evolve(self) -> Dict[str, Any]:
        """
        Run MOEA/D evolution.
        
        Returns:
            Evolution results including Pareto approximation
        """
        start_time = time.time()
        
        if not self.population:
            self.initialize_population()
        
        for generation in range(self.max_generations):
            self.generation = generation
            
            for i in range(self.population_size):
                # Select parents from neighborhood
                parents = self._select_parents(i)
                
                # Generate offspring
                offspring = self._generate_offspring(parents)
                
                # Evaluate offspring
                offspring_objectives = self.fitness_function.evaluate(offspring.genome)
                if isinstance(offspring_objectives, (int, float)):
                    offspring.objectives = np.array([offspring_objectives])
                else:
                    offspring.objectives = np.array(offspring_objectives)
                
                # Update ideal point
                self.ideal_point = np.minimum(self.ideal_point, offspring.objectives)
                
                # Update neighboring solutions
                self._update_neighborhood(i, offspring)
            
            # Log progress
            if generation % 50 == 0:
                logger.info(f"MOEA/D Generation {generation}: Ideal point = {self.ideal_point}")
        
        execution_time = time.time() - start_time
        
        # Extract Pareto front approximation
        pareto_front = self._extract_pareto_front()
        
        # Compile results
        results = {
            'algorithm': 'MOEA/D',
            'pareto_front': pareto_front,
            'final_population': self.population,
            'generations_run': self.generation + 1,
            'execution_time': execution_time,
            'ideal_point': self.ideal_point,
            'weight_vectors': self.weight_vectors,
            'decomposition_method': self.decomposition_method
        }
        
        logger.info(
            f"MOEA/D completed in {execution_time:.2f}s. "
            f"Final Pareto approximation: {len(pareto_front.solutions)} solutions"
        )
        
        return results
    
    def _select_parents(self, subproblem_index: int) -> List[Individual]:
        """Select parents from neighborhood"""
        neighborhood = self.neighborhoods[subproblem_index]
        
        # Select two random parents from neighborhood
        parent_indices = random.sample(neighborhood, 2)
        parents = [self.population[idx] for idx in parent_indices]
        
        return parents
    
    def _generate_offspring(self, parents: List[Individual]) -> Individual:
        """Generate offspring from parents"""
        parent1, parent2 = parents
        
        # Differential evolution recombination
        F = 0.5  # Scaling factor
        CR = 0.9  # Crossover probability
        
        # Select random third parent
        random_parent = random.choice(self.population)
        
        # Generate mutant vector
        mutant_genome = parent1.genome + F * (parent2.genome - random_parent.genome)
        
        # Apply bounds
        lower_bounds, upper_bounds = self.fitness_function.get_bounds()
        mutant_genome = np.clip(mutant_genome, lower_bounds, upper_bounds)
        
        # Crossover
        offspring_genome = parent1.genome.copy()
        for i in range(len(offspring_genome)):
            if np.random.random() < CR:
                offspring_genome[i] = mutant_genome[i]
        
        return Individual(genome=offspring_genome)
    
    def _update_neighborhood(self, subproblem_index: int, offspring: Individual) -> None:
        """Update neighborhood solutions if offspring is better"""
        neighborhood = self.neighborhoods[subproblem_index]
        
        for neighbor_idx in neighborhood:
            if np.random.random() < self.replacement_probability:
                # Compare offspring with current solution for this neighbor's subproblem
                weight_vector = self.weight_vectors[neighbor_idx]
                
                current_fitness = self._scalar_fitness(
                    self.population[neighbor_idx].objectives, weight_vector
                )
                offspring_fitness = self._scalar_fitness(offspring.objectives, weight_vector)
                
                # Replace if offspring is better
                if offspring_fitness < current_fitness:
                    self.population[neighbor_idx] = offspring.copy()
    
    def _scalar_fitness(self, objectives: np.ndarray, weight_vector: np.ndarray) -> float:
        """Calculate scalar fitness using decomposition method"""
        if self.decomposition_method == 'weighted_sum':
            return np.sum(weight_vector * objectives)
        
        elif self.decomposition_method == 'tchebycheff':
            # Tchebycheff approach
            return np.max(weight_vector * np.abs(objectives - self.ideal_point))
        
        elif self.decomposition_method == 'pbi':
            # Penalty-based intersection (PBI)
            theta = 5.0  # Penalty parameter
            
            # Normalize objectives
            normalized_obj = objectives - self.ideal_point
            
            # Calculate parallel and perpendicular distances
            d1 = np.abs(np.dot(normalized_obj, weight_vector)) / np.linalg.norm(weight_vector)
            d2 = np.linalg.norm(normalized_obj - d1 * weight_vector / np.linalg.norm(weight_vector))
            
            return d1 + theta * d2
        
        else:
            raise ValueError(f"Unknown decomposition method: {self.decomposition_method}")
    
    def _extract_pareto_front(self) -> ParetoFront:
        """Extract Pareto front from current population"""
        # Perform non-dominated sorting
        fronts = DominanceComparator.fast_non_dominated_sort(self.population)
        
        if fronts:
            pareto_front = ParetoFront(
                solutions=fronts[0].copy(),
                generation=self.generation
            )
            pareto_front.update_quality_metrics()
            return pareto_front
        else:
            return ParetoFront()


class MultiObjectiveEvolution:
    """
    Main multi-objective evolution coordinator.
    
    Provides a unified interface for different multi-objective algorithms
    and advanced analysis capabilities for NFCS optimization.
    """
    
    def __init__(
        self,
        objectives: List[OptimizationObjective],
        objective_weights: Optional[Dict[OptimizationObjective, float]] = None,
        constraints: Optional[List[Callable]] = None,
        algorithm: SelectionMethod = SelectionMethod.NSGA2
    ):
        """
        Initialize multi-objective evolution coordinator.
        
        Args:
            objectives: List of optimization objectives
            objective_weights: Weights for objective prioritization
            constraints: List of constraint functions
            algorithm: Multi-objective algorithm to use
        """
        self.objectives = objectives
        self.objective_weights = objective_weights or {obj: 1.0 for obj in objectives}
        self.constraints = constraints or []
        self.algorithm = algorithm
        
        # Evolution state
        self.current_front = ParetoFront()
        self.evolution_history = []
        self.algorithm_instance = None
        
        logger.info(f"Initialized MultiObjectiveEvolution with {len(objectives)} objectives")
    
    def optimize(
        self,
        fitness_function: FitnessFunction,
        population_size: int = 100,
        max_generations: int = 250,
        **algorithm_kwargs
    ) -> Dict[str, Any]:
        """
        Run multi-objective optimization.
        
        Args:
            fitness_function: Multi-objective fitness function
            population_size: Population size
            max_generations: Maximum generations
            **algorithm_kwargs: Algorithm-specific parameters
            
        Returns:
            Comprehensive optimization results
        """
        start_time = time.time()
        
        # Initialize algorithm
        if self.algorithm == SelectionMethod.NSGA2:
            self.algorithm_instance = NSGA2Algorithm(
                fitness_function=fitness_function,
                population_size=population_size,
                max_generations=max_generations,
                **algorithm_kwargs
            )
        elif self.algorithm == SelectionMethod.MOEAD:
            self.algorithm_instance = MOEA_D_Algorithm(
                fitness_function=fitness_function,
                population_size=population_size,
                max_generations=max_generations,
                **algorithm_kwargs
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Run optimization
        logger.info(f"Starting multi-objective optimization with {self.algorithm.value}")
        algorithm_results = self.algorithm_instance.evolve()
        
        # Extract final Pareto front
        self.current_front = algorithm_results['pareto_front']
        
        # Perform comprehensive analysis
        analysis_results = self._analyze_results(algorithm_results)
        
        execution_time = time.time() - start_time
        
        # Compile comprehensive results
        results = {
            'algorithm_results': algorithm_results,
            'analysis_results': analysis_results,
            'pareto_front': self.current_front,
            'objectives': [obj.value for obj in self.objectives],
            'objective_weights': {obj.value: weight for obj, weight in self.objective_weights.items()},
            'algorithm': self.algorithm.value,
            'total_execution_time': execution_time,
            'optimization_summary': self._create_optimization_summary(algorithm_results, analysis_results)
        }
        
        # Store in evolution history
        self.evolution_history.append(results)
        
        logger.info(
            f"Multi-objective optimization completed in {execution_time:.2f}s. "
            f"Final Pareto front: {len(self.current_front.solutions)} solutions"
        )
        
        return results
    
    def _analyze_results(self, algorithm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of optimization results"""
        analysis = {}
        
        # Pareto front analysis
        analysis['pareto_analysis'] = self._analyze_pareto_front()
        
        # Convergence analysis
        analysis['convergence_analysis'] = self._analyze_convergence(algorithm_results)
        
        # Diversity analysis
        analysis['diversity_analysis'] = self._analyze_diversity()
        
        # Objective trade-off analysis
        analysis['tradeoff_analysis'] = self._analyze_objective_tradeoffs()
        
        # Solution ranking and selection
        analysis['solution_ranking'] = self._rank_solutions()
        
        return analysis
    
    def _analyze_pareto_front(self) -> Dict[str, Any]:
        """Analyze Pareto front characteristics"""
        if len(self.current_front.solutions) == 0:
            return {'error': 'Empty Pareto front'}
        
        objectives_matrix = self.current_front.objectives
        
        analysis = {
            'front_size': len(self.current_front.solutions),
            'objective_ranges': {},
            'objective_statistics': {},
            'coverage_metrics': {},
            'quality_indicators': self.current_front.quality_metrics.copy()
        }
        
        # Analyze each objective
        for i, objective in enumerate(self.objectives):
            obj_values = objectives_matrix[:, i]
            
            analysis['objective_ranges'][objective.value] = {
                'min': float(np.min(obj_values)),
                'max': float(np.max(obj_values)),
                'range': float(np.max(obj_values) - np.min(obj_values))
            }
            
            analysis['objective_statistics'][objective.value] = {
                'mean': float(np.mean(obj_values)),
                'std': float(np.std(obj_values)),
                'median': float(np.median(obj_values)),
                'q25': float(np.percentile(obj_values, 25)),
                'q75': float(np.percentile(obj_values, 75))
            }
        
        # Coverage metrics
        if len(objectives_matrix) > 1:
            # Calculate dominated hypervolume
            if self.current_front.reference_point is not None:
                hv_calculator = HyperVolumeCalculator(self.current_front.reference_point)
                analysis['coverage_metrics']['hypervolume'] = hv_calculator.calculate_hypervolume(objectives_matrix)
            
            # Calculate extent (maximum distance between solutions)
            distances = cdist(objectives_matrix, objectives_matrix)
            analysis['coverage_metrics']['maximum_extent'] = float(np.max(distances))
            analysis['coverage_metrics']['average_distance'] = float(np.mean(distances[distances > 0]))
        
        return analysis
    
    def _analyze_convergence(self, algorithm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze convergence characteristics"""
        convergence_metrics = algorithm_results.get('convergence_metrics', {})
        
        analysis = {
            'hypervolume_trend': 'stable',
            'front_size_trend': 'stable',
            'convergence_rate': 0.0,
            'final_convergence': False
        }
        
        # Analyze hypervolume history
        hv_history = convergence_metrics.get('hypervolume_history', [])
        if len(hv_history) > 1:
            # Calculate trend
            x = np.arange(len(hv_history))
            slope, _, _, _, _ = stats.linregress(x, hv_history)
            
            if slope > 1e-6:
                analysis['hypervolume_trend'] = 'increasing'
            elif slope < -1e-6:
                analysis['hypervolume_trend'] = 'decreasing'
            
            # Calculate convergence rate (improvement per generation)
            if hv_history[0] > 0:
                analysis['convergence_rate'] = (hv_history[-1] - hv_history[0]) / (len(hv_history) * hv_history[0])
            
            # Check for convergence (stable for last 25% of generations)
            last_quarter = hv_history[-len(hv_history)//4:]
            if len(last_quarter) > 1:
                stability = np.std(last_quarter) / np.mean(last_quarter)
                analysis['final_convergence'] = stability < 0.01  # 1% coefficient of variation
        
        # Analyze front size history
        fs_history = convergence_metrics.get('front_size_history', [])
        if len(fs_history) > 1:
            x = np.arange(len(fs_history))
            slope, _, _, _, _ = stats.linregress(x, fs_history)
            
            if slope > 0.1:
                analysis['front_size_trend'] = 'increasing'
            elif slope < -0.1:
                analysis['front_size_trend'] = 'decreasing'
        
        return analysis
    
    def _analyze_diversity(self) -> Dict[str, Any]:
        """Analyze diversity of solutions in objective and decision space"""
        if len(self.current_front.solutions) < 2:
            return {'error': 'Insufficient solutions for diversity analysis'}
        
        objectives_matrix = self.current_front.objectives
        genomes_matrix = np.array([sol.genome for sol in self.current_front.solutions])
        
        analysis = {
            'objective_space_diversity': {},
            'decision_space_diversity': {},
            'cluster_analysis': {}
        }
        
        # Objective space diversity
        obj_distances = cdist(objectives_matrix, objectives_matrix)
        obj_distances = obj_distances[obj_distances > 0]  # Remove diagonal zeros
        
        analysis['objective_space_diversity'] = {
            'mean_distance': float(np.mean(obj_distances)),
            'std_distance': float(np.std(obj_distances)),
            'min_distance': float(np.min(obj_distances)),
            'max_distance': float(np.max(obj_distances)),
            'diversity_index': float(np.std(obj_distances) / np.mean(obj_distances))
        }
        
        # Decision space diversity
        genome_distances = cdist(genomes_matrix, genomes_matrix)
        genome_distances = genome_distances[genome_distances > 0]
        
        analysis['decision_space_diversity'] = {
            'mean_distance': float(np.mean(genome_distances)),
            'std_distance': float(np.std(genome_distances)),
            'min_distance': float(np.min(genome_distances)),
            'max_distance': float(np.max(genome_distances)),
            'diversity_index': float(np.std(genome_distances) / np.mean(genome_distances))
        }
        
        # Simple clustering analysis
        try:
            from sklearn.cluster import KMeans
            
            # Cluster in objective space
            n_clusters = min(5, len(objectives_matrix))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(objectives_matrix)
            
            analysis['cluster_analysis'] = {
                'num_clusters': n_clusters,
                'cluster_sizes': [int(np.sum(cluster_labels == i)) for i in range(n_clusters)],
                'silhouette_score': self._calculate_silhouette_score(objectives_matrix, cluster_labels)
            }
        except ImportError:
            # Fallback if sklearn not available
            analysis['cluster_analysis'] = {'error': 'sklearn not available for clustering'}
        
        return analysis
    
    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score manually"""
        n = len(data)
        if n <= 1:
            return 0.0
        
        silhouette_scores = []
        
        for i in range(n):
            # Calculate a(i): average distance to points in same cluster
            same_cluster = data[labels == labels[i]]
            if len(same_cluster) > 1:
                a_i = np.mean([euclidean(data[i], point) for point in same_cluster if not np.array_equal(data[i], point)])
            else:
                a_i = 0
            
            # Calculate b(i): minimum average distance to points in other clusters
            b_i = float('inf')
            for cluster_label in np.unique(labels):
                if cluster_label != labels[i]:
                    other_cluster = data[labels == cluster_label]
                    if len(other_cluster) > 0:
                        avg_dist = np.mean([euclidean(data[i], point) for point in other_cluster])
                        b_i = min(b_i, avg_dist)
            
            # Calculate silhouette score for this point
            if max(a_i, b_i) > 0:
                s_i = (b_i - a_i) / max(a_i, b_i)
            else:
                s_i = 0
            
            silhouette_scores.append(s_i)
        
        return float(np.mean(silhouette_scores))
    
    def _analyze_objective_tradeoffs(self) -> Dict[str, Any]:
        """Analyze trade-offs between objectives"""
        if len(self.current_front.objectives) < 2:
            return {'error': 'Need at least 2 objectives for trade-off analysis'}
        
        objectives_matrix = self.current_front.objectives
        n_objectives = len(self.objectives)
        
        analysis = {
            'correlation_matrix': {},
            'pairwise_tradeoffs': {},
            'conflict_analysis': {}
        }
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(objectives_matrix.T)
        
        for i, obj1 in enumerate(self.objectives):
            analysis['correlation_matrix'][obj1.value] = {}
            for j, obj2 in enumerate(self.objectives):
                analysis['correlation_matrix'][obj1.value][obj2.value] = float(correlation_matrix[i, j])
        
        # Pairwise trade-off analysis
        for i in range(n_objectives):
            for j in range(i + 1, n_objectives):
                obj1_name = self.objectives[i].value
                obj2_name = self.objectives[j].value
                
                obj1_values = objectives_matrix[:, i]
                obj2_values = objectives_matrix[:, j]
                
                correlation = correlation_matrix[i, j]
                
                # Trade-off strength (negative correlation indicates trade-off)
                tradeoff_key = f"{obj1_name}_vs_{obj2_name}"
                analysis['pairwise_tradeoffs'][tradeoff_key] = {
                    'correlation': float(correlation),
                    'tradeoff_strength': float(-correlation) if correlation < 0 else 0.0,
                    'conflict_level': 'high' if correlation < -0.7 else 'medium' if correlation < -0.3 else 'low'
                }
        
        # Overall conflict analysis
        negative_correlations = [corr for corr in correlation_matrix.flatten() 
                               if corr < 0 and not np.isnan(corr)]
        
        if negative_correlations:
            analysis['conflict_analysis'] = {
                'average_conflict': float(np.mean([-corr for corr in negative_correlations])),
                'max_conflict': float(-np.min(negative_correlations)),
                'num_conflicting_pairs': len(negative_correlations),
                'overall_conflict_level': 'high' if np.mean(negative_correlations) < -0.5 else 'medium' if np.mean(negative_correlations) < -0.2 else 'low'
            }
        else:
            analysis['conflict_analysis'] = {
                'average_conflict': 0.0,
                'max_conflict': 0.0,
                'num_conflicting_pairs': 0,
                'overall_conflict_level': 'none'
            }
        
        return analysis
    
    def _rank_solutions(self) -> Dict[str, Any]:
        """Rank and select best solutions from Pareto front"""
        if len(self.current_front.solutions) == 0:
            return {'error': 'No solutions to rank'}
        
        ranking = {
            'weighted_ranking': [],
            'knee_points': [],
            'extreme_points': {},
            'compromise_solutions': []
        }
        
        objectives_matrix = self.current_front.objectives
        
        # Weighted ranking based on user preferences
        weighted_scores = []
        for i, solution in enumerate(self.current_front.solutions):
            # Normalize objectives (assuming minimization)
            normalized_obj = (objectives_matrix[i] - self.current_front.ideal_point) / \
                           (self.current_front.nadir_point - self.current_front.ideal_point + 1e-10)
            
            # Calculate weighted score
            weighted_score = 0.0
            for j, objective in enumerate(self.objectives):
                weight = self.objective_weights.get(objective, 1.0)
                weighted_score += weight * normalized_obj[j]
            
            weighted_scores.append(weighted_score)
            
            ranking['weighted_ranking'].append({
                'solution_index': i,
                'weighted_score': float(weighted_score),
                'objectives': objectives_matrix[i].tolist(),
                'genome': solution.genome.tolist()
            })
        
        # Sort by weighted score
        ranking['weighted_ranking'].sort(key=lambda x: x['weighted_score'])
        
        # Find extreme points (best in each objective)
        for i, objective in enumerate(self.objectives):
            best_index = np.argmin(objectives_matrix[:, i])
            ranking['extreme_points'][objective.value] = {
                'solution_index': int(best_index),
                'objective_value': float(objectives_matrix[best_index, i]),
                'objectives': objectives_matrix[best_index].tolist()
            }
        
        # Find knee points (solutions with good balance)
        if len(objectives_matrix) >= 2:
            knee_indices = self._find_knee_points(objectives_matrix)
            for idx in knee_indices:
                ranking['knee_points'].append({
                    'solution_index': int(idx),
                    'objectives': objectives_matrix[idx].tolist(),
                    'balance_score': float(self._calculate_balance_score(objectives_matrix[idx]))
                })
        
        # Compromise solutions (closest to ideal point)
        distances_to_ideal = [
            np.linalg.norm(obj - self.current_front.ideal_point)
            for obj in objectives_matrix
        ]
        
        best_compromise_indices = np.argsort(distances_to_ideal)[:min(3, len(distances_to_ideal))]
        
        for idx in best_compromise_indices:
            ranking['compromise_solutions'].append({
                'solution_index': int(idx),
                'distance_to_ideal': float(distances_to_ideal[idx]),
                'objectives': objectives_matrix[idx].tolist()
            })
        
        return ranking
    
    def _find_knee_points(self, objectives_matrix: np.ndarray) -> List[int]:
        """Find knee points in the Pareto front"""
        if len(objectives_matrix) < 3:
            return list(range(len(objectives_matrix)))
        
        # Simple knee point detection using angle analysis
        knee_indices = []
        
        # Normalize objectives for fair comparison
        normalized_obj = (objectives_matrix - np.min(objectives_matrix, axis=0)) / \
                        (np.max(objectives_matrix, axis=0) - np.min(objectives_matrix, axis=0) + 1e-10)
        
        # For 2D case, use curvature analysis
        if objectives_matrix.shape[1] == 2:
            # Sort by first objective
            sorted_indices = np.argsort(normalized_obj[:, 0])
            sorted_obj = normalized_obj[sorted_indices]
            
            # Calculate curvature for each point
            for i in range(1, len(sorted_obj) - 1):
                prev_point = sorted_obj[i - 1]
                curr_point = sorted_obj[i]
                next_point = sorted_obj[i + 1]
                
                # Calculate vectors
                v1 = curr_point - prev_point
                v2 = next_point - curr_point
                
                # Calculate angle (measure of curvature)
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    
                    # Sharp angles indicate knee points
                    if angle < np.pi / 3:  # Less than 60 degrees
                        knee_indices.append(sorted_indices[i])
        
        # For higher dimensions, use distance-based method
        else:
            # Find points that are furthest from line connecting extreme points
            for i in range(len(objectives_matrix)):
                min_distance_to_extremes = float('inf')
                
                # Check distance to all pairs of extreme points
                for j in range(len(objectives_matrix)):
                    for k in range(j + 1, len(objectives_matrix)):
                        if i != j and i != k:
                            # Calculate distance from point i to line jk
                            point = normalized_obj[i]
                            line_start = normalized_obj[j]
                            line_end = normalized_obj[k]
                            
                            distance = self._point_to_line_distance(point, line_start, line_end)
                            min_distance_to_extremes = min(min_distance_to_extremes, distance)
                
                # Points with large distance to extreme pairs are potential knee points
                if min_distance_to_extremes > 0.1:  # Threshold for knee detection
                    knee_indices.append(i)
        
        return knee_indices[:5]  # Return at most 5 knee points
    
    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Calculate distance from point to line segment"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_length_squared = np.dot(line_vec, line_vec)
        
        if line_length_squared == 0:
            return np.linalg.norm(point_vec)
        
        t = np.dot(point_vec, line_vec) / line_length_squared
        t = max(0, min(1, t))  # Clamp to line segment
        
        projection = line_start + t * line_vec
        return np.linalg.norm(point - projection)
    
    def _calculate_balance_score(self, objectives: np.ndarray) -> float:
        """Calculate balance score for a solution (lower variance = better balance)"""
        if len(objectives) <= 1:
            return 0.0
        
        # Normalize objectives
        normalized = (objectives - self.current_front.ideal_point) / \
                    (self.current_front.nadir_point - self.current_front.ideal_point + 1e-10)
        
        # Balance score is inverse of variance (lower variance = better balance)
        variance = np.var(normalized)
        return 1.0 / (1.0 + variance)
    
    def _create_optimization_summary(
        self, 
        algorithm_results: Dict[str, Any], 
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive optimization summary"""
        summary = {
            'success': len(self.current_front.solutions) > 0,
            'pareto_front_size': len(self.current_front.solutions),
            'algorithm_performance': {},
            'objective_achievement': {},
            'recommendations': []
        }
        
        # Algorithm performance summary
        convergence_info = analysis_results.get('convergence_analysis', {})
        summary['algorithm_performance'] = {
            'converged': convergence_info.get('final_convergence', False),
            'convergence_rate': convergence_info.get('convergence_rate', 0.0),
            'hypervolume_trend': convergence_info.get('hypervolume_trend', 'unknown'),
            'execution_time': algorithm_results.get('execution_time', 0.0)
        }
        
        # Objective achievement summary
        pareto_analysis = analysis_results.get('pareto_analysis', {})
        obj_stats = pareto_analysis.get('objective_statistics', {})
        
        for objective in self.objectives:
            obj_name = objective.value
            if obj_name in obj_stats:
                summary['objective_achievement'][obj_name] = {
                    'best_value': pareto_analysis['objective_ranges'][obj_name]['min'],
                    'worst_value': pareto_analysis['objective_ranges'][obj_name]['max'],
                    'average_value': obj_stats[obj_name]['mean'],
                    'improvement_range': pareto_analysis['objective_ranges'][obj_name]['range']
                }
        
        # Generate recommendations
        if not summary['success']:
            summary['recommendations'].append("Optimization failed to find solutions. Check problem formulation and constraints.")
        
        if summary['pareto_front_size'] < 10:
            summary['recommendations'].append("Small Pareto front size. Consider increasing population size or generations.")
        
        if not summary['algorithm_performance']['converged']:
            summary['recommendations'].append("Algorithm did not converge. Consider increasing maximum generations or adjusting parameters.")
        
        # Diversity recommendations
        diversity_info = analysis_results.get('diversity_analysis', {})
        obj_diversity = diversity_info.get('objective_space_diversity', {})
        
        if 'diversity_index' in obj_diversity and obj_diversity['diversity_index'] < 0.1:
            summary['recommendations'].append("Low solution diversity. Consider adjusting selection pressure or mutation rates.")
        
        # Trade-off recommendations
        tradeoff_info = analysis_results.get('tradeoff_analysis', {})
        conflict_info = tradeoff_info.get('conflict_analysis', {})
        
        if 'overall_conflict_level' in conflict_info:
            conflict_level = conflict_info['overall_conflict_level']
            if conflict_level == 'high':
                summary['recommendations'].append("High objective conflicts detected. Consider preference articulation or constraint relaxation.")
            elif conflict_level == 'none':
                summary['recommendations'].append("Low objective conflicts. Problem may benefit from single-objective optimization.")
        
        return summary
    
    def get_best_solutions(self, n: int = 5, criterion: str = 'weighted') -> List[Dict[str, Any]]:
        """
        Get best solutions from Pareto front based on specified criterion.
        
        Args:
            n: Number of solutions to return
            criterion: Selection criterion ('weighted', 'knee', 'compromise', 'extreme')
            
        Returns:
            List of best solutions with metadata
        """
        if len(self.current_front.solutions) == 0:
            return []
        
        # Get ranking from analysis
        if not hasattr(self, '_last_analysis') or not self._last_analysis:
            # Perform quick analysis if not available
            self._last_analysis = self._analyze_results({'convergence_metrics': {}})
        
        ranking = self._last_analysis.get('solution_ranking', {})
        
        if criterion == 'weighted':
            solutions = ranking.get('weighted_ranking', [])
        elif criterion == 'knee':
            solutions = ranking.get('knee_points', [])
        elif criterion == 'compromise':
            solutions = ranking.get('compromise_solutions', [])
        elif criterion == 'extreme':
            solutions = list(ranking.get('extreme_points', {}).values())
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        # Return top n solutions
        return solutions[:n]
    
    def visualize_pareto_front(self, save_path: Optional[str] = None) -> None:
        """
        Visualize Pareto front (for 2D and 3D objectives).
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if len(self.current_front.solutions) == 0:
            logger.warning("No solutions to visualize")
            return
        
        objectives_matrix = self.current_front.objectives
        n_objectives = objectives_matrix.shape[1]
        
        if n_objectives == 2:
            self._plot_2d_pareto_front(objectives_matrix, save_path)
        elif n_objectives == 3:
            self._plot_3d_pareto_front(objectives_matrix, save_path)
        else:
            logger.warning(f"Cannot visualize {n_objectives}D Pareto front directly")
    
    def _plot_2d_pareto_front(self, objectives_matrix: np.ndarray, save_path: Optional[str] = None) -> None:
        """Plot 2D Pareto front"""
        plt.figure(figsize=(10, 8))
        
        plt.scatter(objectives_matrix[:, 0], objectives_matrix[:, 1], 
                   c='blue', alpha=0.7, s=50, label='Pareto Front')
        
        plt.xlabel(f'{self.objectives[0].value}')
        plt.ylabel(f'{self.objectives[1].value}')
        plt.title('Pareto Front Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pareto front plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_3d_pareto_front(self, objectives_matrix: np.ndarray, save_path: Optional[str] = None) -> None:
        """Plot 3D Pareto front"""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(objectives_matrix[:, 0], objectives_matrix[:, 1], objectives_matrix[:, 2],
                  c='blue', alpha=0.7, s=50, label='Pareto Front')
        
        ax.set_xlabel(f'{self.objectives[0].value}')
        ax.set_ylabel(f'{self.objectives[1].value}')
        ax.set_zlabel(f'{self.objectives[2].value}')
        ax.set_title('3D Pareto Front Visualization')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"3D Pareto front plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_pareto_front(self, file_path: str, format: str = 'json') -> None:
        """
        Export Pareto front to file.
        
        Args:
            file_path: Path to save the file
            format: Export format ('json', 'csv')
        """
        if len(self.current_front.solutions) == 0:
            logger.warning("No solutions to export")
            return
        
        if format == 'json':
            export_data = {
                'pareto_front': {
                    'solutions': [
                        {
                            'genome': sol.genome.tolist(),
                            'objectives': sol.objectives.tolist(),
                            'fitness': float(sol.fitness) if isinstance(sol.fitness, (int, float)) else sol.fitness
                        }
                        for sol in self.current_front.solutions
                    ],
                    'quality_metrics': self.current_front.quality_metrics,
                    'ideal_point': self.current_front.ideal_point.tolist() if self.current_front.ideal_point is not None else None,
                    'nadir_point': self.current_front.nadir_point.tolist() if self.current_front.nadir_point is not None else None
                },
                'objectives': [obj.value for obj in self.objectives],
                'objective_weights': {obj.value: weight for obj, weight in self.objective_weights.items()}
            }
            
            import json
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        elif format == 'csv':
            import csv
            
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                header = ['solution_id'] + [f'obj_{obj.value}' for obj in self.objectives] + \
                        [f'genome_{i}' for i in range(len(self.current_front.solutions[0].genome))]
                writer.writerow(header)
                
                # Data
                for i, sol in enumerate(self.current_front.solutions):
                    row = [i] + sol.objectives.tolist() + sol.genome.tolist()
                    writer.writerow(row)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Pareto front exported to {file_path} in {format} format")
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics"""
        return {
            'total_optimizations': len(self.evolution_history),
            'current_front_size': len(self.current_front.solutions),
            'objectives': [obj.value for obj in self.objectives],
            'algorithm_used': self.algorithm.value,
            'quality_metrics': self.current_front.quality_metrics,
            'evolution_history_summary': [
                {
                    'optimization_id': i,
                    'front_size': len(result['pareto_front'].solutions),
                    'execution_time': result['total_execution_time'],
                    'algorithm': result['algorithm']
                }
                for i, result in enumerate(self.evolution_history)
            ]
        }