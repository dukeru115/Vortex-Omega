"""
Genetic Optimizer Core - NFCS Evolutionary Engine
===============================================

Implements the primary genetic algorithm engine for NFCS v2.4.3 system optimization.
Provides advanced evolutionary strategies including differential evolution, particle swarm
optimization, and adaptive genetic operators for neural field control optimization.

Scientific Foundation:
Based on PDF Section 5.13, implements multi-strategy evolutionary algorithms:
- Adaptive Genetic Algorithm (AGA) with dynamic parameters
- Differential Evolution (DE) for continuous optimization
- Particle Swarm Optimization (PSO) for swarm intelligence
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
- Hybrid evolutionary strategies with self-adaptive operators

Mathematical Framework:
- Population evolution: P(t+1) = f(P(t), selection, crossover, mutation)
- Fitness landscape navigation using gradient-free optimization
- Multi-modal optimization with speciation and niching
- Convergence analysis using diversity metrics and fitness variance

Integration:
Seamlessly integrates with NFCS neural field dynamics, Kuramoto synchronization,
and symbolic AI components for comprehensive system optimization.

Created: September 14, 2025
Author: Team Ω - Neural Field Control Systems Research Group
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import time
import json
from collections import defaultdict, deque
import random
import copy
from scipy import stats
from scipy.special import softmax
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Selection strategies for evolutionary algorithms"""

    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"
    ELITIST = "elitist"
    CROWDING_DISTANCE = "crowding_distance"


class CrossoverOperator(Enum):
    """Crossover operators for genetic recombination"""

    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    BLEND_ALPHA = "blend_alpha"
    SIMULATED_BINARY = "simulated_binary"


class MutationOperator(Enum):
    """Mutation operators for genetic variation"""

    GAUSSIAN = "gaussian"
    POLYNOMIAL = "polynomial"
    UNIFORM = "uniform"
    BIT_FLIP = "bit_flip"
    SWAP = "swap"
    INVERSION = "inversion"


class EvolutionaryStrategy(Enum):
    """High-level evolutionary strategies"""

    GENETIC_ALGORITHM = "genetic_algorithm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    PARTICLE_SWARM = "particle_swarm"
    CMA_ES = "cma_es"
    HYBRID_MULTI_STRATEGY = "hybrid_multi_strategy"


@dataclass
class Individual:
    """
    Represents an individual solution in the evolutionary population.

    Attributes:
        genome: Parameter vector representing the solution
        fitness: Fitness value(s) for the individual
        age: Number of generations the individual has survived
        rank: Dominance rank for multi-objective optimization
        crowding_distance: Crowding distance for diversity preservation
        objectives: Multi-objective fitness values
        constraints: Constraint violation measures
        metadata: Additional information about the individual
    """

    genome: np.ndarray
    fitness: Union[float, np.ndarray] = 0.0
    age: int = 0
    rank: int = 0
    crowding_distance: float = 0.0
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    constraints: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived attributes"""
        if len(self.objectives) == 0 and isinstance(self.fitness, (int, float)):
            self.objectives = np.array([self.fitness])

    def copy(self) -> "Individual":
        """Create a deep copy of the individual"""
        return Individual(
            genome=self.genome.copy(),
            fitness=self.fitness,
            age=self.age,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            objectives=self.objectives.copy() if len(self.objectives) > 0 else np.array([]),
            constraints=self.constraints.copy() if len(self.constraints) > 0 else np.array([]),
            metadata=copy.deepcopy(self.metadata),
        )

    def dominates(self, other: "Individual") -> bool:
        """Check if this individual dominates another (Pareto dominance)"""
        if len(self.objectives) != len(other.objectives):
            return False

        better_in_at_least_one = False
        for i in range(len(self.objectives)):
            if self.objectives[i] < other.objectives[i]:  # Assuming minimization
                return False
            elif self.objectives[i] > other.objectives[i]:
                better_in_at_least_one = True

        return better_in_at_least_one


@dataclass
class Population:
    """
    Represents a population of individuals in the evolutionary algorithm.

    Attributes:
        individuals: List of Individual objects
        size: Population size
        generation: Current generation number
        diversity_metrics: Measures of population diversity
        convergence_metrics: Measures of convergence progress
        statistics: Population statistics
    """

    individuals: List[Individual] = field(default_factory=list)
    size: int = 100
    generation: int = 0
    diversity_metrics: Dict[str, float] = field(default_factory=dict)
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.individuals)

    def add_individual(self, individual: Individual) -> None:
        """Add an individual to the population"""
        self.individuals.append(individual)

    def get_best(self, n: int = 1) -> List[Individual]:
        """Get the n best individuals based on fitness"""
        sorted_individuals = sorted(
            self.individuals,
            key=lambda x: x.fitness if isinstance(x.fitness, (int, float)) else np.sum(x.fitness),
            reverse=True,
        )
        return sorted_individuals[:n]

    def get_worst(self, n: int = 1) -> List[Individual]:
        """Get the n worst individuals based on fitness"""
        sorted_individuals = sorted(
            self.individuals,
            key=lambda x: x.fitness if isinstance(x.fitness, (int, float)) else np.sum(x.fitness),
        )
        return sorted_individuals[:n]

    def calculate_diversity(self) -> float:
        """Calculate population diversity using genome distance"""
        if len(self.individuals) < 2:
            return 0.0

        total_distance = 0.0
        count = 0

        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                distance = np.linalg.norm(self.individuals[i].genome - self.individuals[j].genome)
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 0.0

    def update_statistics(self) -> None:
        """Update population statistics"""
        if not self.individuals:
            return

        fitnesses = [
            ind.fitness for ind in self.individuals if isinstance(ind.fitness, (int, float))
        ]

        if fitnesses:
            self.statistics.update(
                {
                    "mean_fitness": np.mean(fitnesses),
                    "std_fitness": np.std(fitnesses),
                    "max_fitness": np.max(fitnesses),
                    "min_fitness": np.min(fitnesses),
                    "median_fitness": np.median(fitnesses),
                }
            )

        self.diversity_metrics["genome_diversity"] = self.calculate_diversity()


class FitnessFunction(ABC):
    """Abstract base class for fitness function evaluation"""

    @abstractmethod
    def evaluate(self, genome: np.ndarray) -> Union[float, np.ndarray]:
        """
        Evaluate the fitness of a genome.

        Args:
            genome: Parameter vector to evaluate

        Returns:
            Fitness value(s) - single float for single-objective,
            array for multi-objective optimization
        """
        pass

    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the bounds for the optimization problem.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        pass

    def evaluate_constraints(self, genome: np.ndarray) -> np.ndarray:
        """
        Evaluate constraint violations.

        Args:
            genome: Parameter vector to evaluate

        Returns:
            Array of constraint violation values (0 = satisfied, >0 = violated)
        """
        return np.array([])


class GeneticOptimizer:
    """
    Main genetic algorithm optimizer for NFCS system optimization.

    Implements multiple evolutionary strategies with adaptive operators
    for neural field control system parameter optimization.
    """

    def __init__(
        self,
        fitness_function: FitnessFunction,
        population_size: int = 100,
        genome_length: int = 10,
        strategy: EvolutionaryStrategy = EvolutionaryStrategy.GENETIC_ALGORITHM,
        selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT,
        crossover_operator: CrossoverOperator = CrossoverOperator.UNIFORM,
        mutation_operator: MutationOperator = MutationOperator.GAUSSIAN,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism_rate: float = 0.1,
        max_generations: int = 1000,
        convergence_threshold: float = 1e-6,
        diversity_threshold: float = 1e-4,
        adaptive_operators: bool = True,
        multi_objective: bool = False,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the genetic optimizer.

        Args:
            fitness_function: Function to evaluate individual fitness
            population_size: Size of the population
            genome_length: Length of individual genome vectors
            strategy: Evolutionary strategy to use
            selection_strategy: Selection method for parent individuals
            crossover_operator: Crossover operator for recombination
            mutation_operator: Mutation operator for variation
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_rate: Fraction of elite individuals to preserve
            max_generations: Maximum number of generations
            convergence_threshold: Threshold for convergence detection
            diversity_threshold: Minimum diversity threshold
            adaptive_operators: Enable adaptive operator parameters
            multi_objective: Enable multi-objective optimization
            random_seed: Random seed for reproducibility
        """
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.genome_length = genome_length
        self.strategy = strategy
        self.selection_strategy = selection_strategy
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator

        # Evolutionary parameters
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        self.diversity_threshold = diversity_threshold

        # Advanced features
        self.adaptive_operators = adaptive_operators
        self.multi_objective = multi_objective

        # Internal state
        self.population = Population(size=population_size)
        self.generation = 0
        self.convergence_history = []
        self.diversity_history = []
        self.best_individual = None
        self.pareto_front = []

        # Adaptive operator parameters
        self.adaptive_params = {
            "crossover_rate_history": deque(maxlen=10),
            "mutation_rate_history": deque(maxlen=10),
            "success_rates": deque(maxlen=10),
            "diversity_window": deque(maxlen=5),
        }

        # Strategy-specific parameters
        self._init_strategy_params()

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        logger.info(f"Initialized GeneticOptimizer with strategy: {strategy}")

    def _init_strategy_params(self) -> None:
        """Initialize strategy-specific parameters"""
        if self.strategy == EvolutionaryStrategy.DIFFERENTIAL_EVOLUTION:
            self.de_params = {
                "F": 0.8,  # Differential weight
                "CR": 0.9,  # Crossover probability
                "strategy": "rand/1/bin",
            }
        elif self.strategy == EvolutionaryStrategy.PARTICLE_SWARM:
            self.pso_params = {
                "w": 0.9,  # Inertia weight
                "c1": 2.0,  # Cognitive parameter
                "c2": 2.0,  # Social parameter
                "velocities": None,
            }
        elif self.strategy == EvolutionaryStrategy.CMA_ES:
            self.cma_params = {
                "sigma": 0.3,  # Step size
                "mean": None,
                "C": None,  # Covariance matrix
                "pc": None,  # Evolution path for covariance
                "ps": None,  # Evolution path for sigma
            }

    def initialize_population(self) -> None:
        """Initialize the population with random individuals"""
        lower_bounds, upper_bounds = self.fitness_function.get_bounds()

        self.population = Population(size=self.population_size)

        for _ in range(self.population_size):
            genome = np.random.uniform(lower_bounds, upper_bounds, size=self.genome_length)
            individual = Individual(genome=genome)
            self.population.add_individual(individual)

        # Evaluate initial population
        self._evaluate_population()

        # Initialize strategy-specific components
        if self.strategy == EvolutionaryStrategy.PARTICLE_SWARM:
            self.pso_params["velocities"] = [
                np.random.uniform(-1, 1, self.genome_length) for _ in range(self.population_size)
            ]
        elif self.strategy == EvolutionaryStrategy.CMA_ES:
            self.cma_params["mean"] = np.mean(
                [ind.genome for ind in self.population.individuals], axis=0
            )
            self.cma_params["C"] = np.eye(self.genome_length)
            self.cma_params["pc"] = np.zeros(self.genome_length)
            self.cma_params["ps"] = np.zeros(self.genome_length)

        logger.info(f"Initialized population of size {self.population_size}")

    def _evaluate_population(self) -> None:
        """Evaluate fitness for all individuals in the population"""
        for individual in self.population.individuals:
            fitness = self.fitness_function.evaluate(individual.genome)
            individual.fitness = fitness

            if self.multi_objective and isinstance(fitness, np.ndarray):
                individual.objectives = fitness
            else:
                individual.objectives = np.array([fitness])

            # Evaluate constraints if applicable
            constraints = self.fitness_function.evaluate_constraints(individual.genome)
            individual.constraints = constraints

        self.population.update_statistics()

        # Update best individual
        if not self.multi_objective:
            current_best = self.population.get_best(1)[0]
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best.copy()

    def evolve(self) -> Dict[str, Any]:
        """
        Run the evolutionary algorithm for optimization.

        Returns:
            Dictionary containing optimization results and statistics
        """
        start_time = time.time()

        # Initialize population if not already done
        if len(self.population) == 0:
            self.initialize_population()

        convergence_counter = 0
        stagnation_counter = 0
        previous_best_fitness = float("-inf")

        for generation in range(self.max_generations):
            self.generation = generation

            # Execute one generation based on strategy
            if self.strategy == EvolutionaryStrategy.GENETIC_ALGORITHM:
                self._evolve_genetic_algorithm()
            elif self.strategy == EvolutionaryStrategy.DIFFERENTIAL_EVOLUTION:
                self._evolve_differential_evolution()
            elif self.strategy == EvolutionaryStrategy.PARTICLE_SWARM:
                self._evolve_particle_swarm()
            elif self.strategy == EvolutionaryStrategy.CMA_ES:
                self._evolve_cma_es()
            elif self.strategy == EvolutionaryStrategy.HYBRID_MULTI_STRATEGY:
                self._evolve_hybrid_strategy()

            # Update population statistics
            self.population.generation = generation
            self.population.update_statistics()

            # Track convergence and diversity
            current_best_fitness = self.population.statistics.get("max_fitness", 0)
            diversity = self.population.diversity_metrics.get("genome_diversity", 0)

            self.convergence_history.append(current_best_fitness)
            self.diversity_history.append(diversity)

            # Check for convergence
            if abs(current_best_fitness - previous_best_fitness) < self.convergence_threshold:
                convergence_counter += 1
            else:
                convergence_counter = 0

            if convergence_counter >= 10:  # Converged for 10 consecutive generations
                logger.info(f"Convergence detected at generation {generation}")
                break

            # Check for stagnation (low diversity)
            if diversity < self.diversity_threshold:
                stagnation_counter += 1
                if stagnation_counter >= 5:
                    self._handle_stagnation()
                    stagnation_counter = 0
            else:
                stagnation_counter = 0

            # Adaptive operator adjustment
            if self.adaptive_operators and generation > 0 and generation % 10 == 0:
                self._adapt_operators()

            previous_best_fitness = current_best_fitness

            # Log progress
            if generation % 50 == 0:
                logger.info(
                    f"Generation {generation}: "
                    f"Best fitness = {current_best_fitness:.6f}, "
                    f"Diversity = {diversity:.6f}"
                )

        execution_time = time.time() - start_time

        # Prepare results
        results = {
            "best_individual": self.best_individual,
            "best_fitness": self.best_individual.fitness if self.best_individual else None,
            "best_genome": self.best_individual.genome if self.best_individual else None,
            "generations_run": self.generation + 1,
            "execution_time": execution_time,
            "convergence_history": self.convergence_history,
            "diversity_history": self.diversity_history,
            "final_population": self.population,
            "population_statistics": self.population.statistics,
            "strategy": self.strategy.value,
            "parameters": {
                "population_size": self.population_size,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "elitism_rate": self.elitism_rate,
            },
        }

        if self.multi_objective:
            self._update_pareto_front()
            results["pareto_front"] = self.pareto_front

        logger.info(
            f"Evolution completed in {execution_time:.2f}s after {self.generation + 1} generations"
        )

        return results

    def _evolve_genetic_algorithm(self) -> None:
        """Execute one generation of genetic algorithm"""
        # Selection
        parents = self._selection()

        # Generate offspring through crossover and mutation
        offspring = []

        # Preserve elites
        elite_count = int(self.population_size * self.elitism_rate)
        elites = self.population.get_best(elite_count)
        offspring.extend([elite.copy() for elite in elites])

        # Generate remaining offspring
        while len(offspring) < self.population_size:
            # Select two parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

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

        # Replace population (truncate to exact size)
        offspring = offspring[: self.population_size]

        # Update population
        self.population.individuals = offspring

        # Evaluate new population
        self._evaluate_population()

    def _evolve_differential_evolution(self) -> None:
        """Execute one generation of differential evolution"""
        F = self.de_params["F"]
        CR = self.de_params["CR"]

        new_population = []

        for i, target in enumerate(self.population.individuals):
            # Select three random individuals different from target
            candidates = [ind for j, ind in enumerate(self.population.individuals) if j != i]
            if len(candidates) < 3:
                new_population.append(target.copy())
                continue

            a, b, c = random.sample(candidates, 3)

            # Create mutant vector: a + F * (b - c)
            mutant_genome = a.genome + F * (b.genome - c.genome)

            # Apply bounds
            lower_bounds, upper_bounds = self.fitness_function.get_bounds()
            mutant_genome = np.clip(mutant_genome, lower_bounds, upper_bounds)

            # Crossover
            trial_genome = target.genome.copy()
            for j in range(len(trial_genome)):
                if np.random.random() < CR or j == np.random.randint(len(trial_genome)):
                    trial_genome[j] = mutant_genome[j]

            # Create trial individual
            trial = Individual(genome=trial_genome)
            trial.fitness = self.fitness_function.evaluate(trial.genome)

            # Selection (keep better individual)
            if trial.fitness > target.fitness:
                new_population.append(trial)
            else:
                new_population.append(target)

        self.population.individuals = new_population

    def _evolve_particle_swarm(self) -> None:
        """Execute one generation of particle swarm optimization"""
        w = self.pso_params["w"]  # Inertia weight
        c1 = self.pso_params["c1"]  # Cognitive parameter
        c2 = self.pso_params["c2"]  # Social parameter

        # Find global best
        global_best = self.population.get_best(1)[0]

        # Update each particle
        for i, particle in enumerate(self.population.individuals):
            velocity = self.pso_params["velocities"][i]

            # Personal best (particle's best position so far)
            personal_best = particle.genome  # Simplified - should track history

            # Update velocity
            r1, r2 = np.random.random(2)
            cognitive_component = c1 * r1 * (personal_best - particle.genome)
            social_component = c2 * r2 * (global_best.genome - particle.genome)

            velocity = w * velocity + cognitive_component + social_component

            # Update position
            particle.genome = particle.genome + velocity

            # Apply bounds
            lower_bounds, upper_bounds = self.fitness_function.get_bounds()
            particle.genome = np.clip(particle.genome, lower_bounds, upper_bounds)

            # Update velocity for next iteration
            self.pso_params["velocities"][i] = velocity

        # Evaluate population
        self._evaluate_population()

    def _evolve_cma_es(self) -> None:
        """Execute one generation of CMA-ES"""
        # Simplified CMA-ES implementation
        sigma = self.cma_params["sigma"]
        mean = self.cma_params["mean"]
        C = self.cma_params["C"]

        # Generate offspring
        offspring = []
        for _ in range(self.population_size):
            genome = np.random.multivariate_normal(mean, sigma * sigma * C)

            # Apply bounds
            lower_bounds, upper_bounds = self.fitness_function.get_bounds()
            genome = np.clip(genome, lower_bounds, upper_bounds)

            individual = Individual(genome=genome)
            offspring.append(individual)

        # Evaluate offspring
        for individual in offspring:
            individual.fitness = self.fitness_function.evaluate(individual.genome)

        # Selection (keep best individuals)
        all_individuals = self.population.individuals + offspring
        all_individuals.sort(key=lambda x: x.fitness, reverse=True)
        self.population.individuals = all_individuals[: self.population_size]

        # Update CMA-ES parameters (simplified)
        best_genomes = [
            ind.genome for ind in self.population.individuals[: self.population_size // 2]
        ]
        self.cma_params["mean"] = np.mean(best_genomes, axis=0)

    def _evolve_hybrid_strategy(self) -> None:
        """Execute hybrid multi-strategy evolution"""
        # Divide population into subgroups for different strategies
        subgroup_size = self.population_size // 4

        # Apply GA to first subgroup
        ga_individuals = self.population.individuals[:subgroup_size]
        temp_population = Population()
        temp_population.individuals = ga_individuals
        temp_pop_backup = self.population
        self.population = temp_population
        self._evolve_genetic_algorithm()
        ga_results = self.population.individuals

        # Apply DE to second subgroup
        de_individuals = temp_pop_backup.individuals[subgroup_size : 2 * subgroup_size]
        self.population.individuals = de_individuals
        self._evolve_differential_evolution()
        de_results = self.population.individuals

        # Apply PSO to third subgroup
        pso_individuals = temp_pop_backup.individuals[2 * subgroup_size : 3 * subgroup_size]
        self.population.individuals = pso_individuals
        if self.pso_params["velocities"] is None or len(self.pso_params["velocities"]) != len(
            pso_individuals
        ):
            self.pso_params["velocities"] = [
                np.random.uniform(-1, 1, self.genome_length) for _ in pso_individuals
            ]
        self._evolve_particle_swarm()
        pso_results = self.population.individuals

        # Keep remaining individuals unchanged
        remaining = temp_pop_backup.individuals[3 * subgroup_size :]

        # Combine results and restore population
        self.population = temp_pop_backup
        self.population.individuals = ga_results + de_results + pso_results + remaining

        # Migration between subgroups (best individuals)
        best_from_each = []
        for subgroup in [ga_results, de_results, pso_results]:
            if subgroup:
                best_from_each.append(max(subgroup, key=lambda x: x.fitness))

        # Replace worst individuals with migrants
        if best_from_each:
            worst_individuals = self.population.get_worst(len(best_from_each))
            for i, worst in enumerate(worst_individuals):
                if i < len(best_from_each):
                    worst.genome = best_from_each[i].genome.copy()
                    worst.fitness = best_from_each[i].fitness

    def _selection(self) -> List[Individual]:
        """Perform selection of parent individuals"""
        if self.selection_strategy == SelectionStrategy.TOURNAMENT:
            return self._tournament_selection()
        elif self.selection_strategy == SelectionStrategy.ROULETTE_WHEEL:
            return self._roulette_wheel_selection()
        elif self.selection_strategy == SelectionStrategy.RANK_BASED:
            return self._rank_based_selection()
        elif self.selection_strategy == SelectionStrategy.ELITIST:
            return self._elitist_selection()
        else:
            return self._tournament_selection()  # Default

    def _tournament_selection(self, tournament_size: int = 3) -> List[Individual]:
        """Tournament selection"""
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(
                self.population.individuals, min(tournament_size, len(self.population.individuals))
            )
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        return selected

    def _roulette_wheel_selection(self) -> List[Individual]:
        """Roulette wheel selection"""
        fitnesses = [ind.fitness for ind in self.population.individuals]
        min_fitness = min(fitnesses)

        # Adjust for negative fitness values
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 1e-10 for f in fitnesses]

        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choices(self.population.individuals, k=self.population_size)

        probabilities = [f / total_fitness for f in fitnesses]

        selected = np.random.choice(
            self.population.individuals, size=self.population_size, p=probabilities, replace=True
        ).tolist()

        return selected

    def _rank_based_selection(self) -> List[Individual]:
        """Rank-based selection"""
        sorted_individuals = sorted(self.population.individuals, key=lambda x: x.fitness)
        ranks = list(range(1, len(sorted_individuals) + 1))
        total_rank = sum(ranks)

        probabilities = [rank / total_rank for rank in ranks]

        selected = np.random.choice(
            sorted_individuals, size=self.population_size, p=probabilities, replace=True
        ).tolist()

        return selected

    def _elitist_selection(self) -> List[Individual]:
        """Elitist selection - select only the best individuals"""
        return self.population.get_best(self.population_size)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents"""
        if self.crossover_operator == CrossoverOperator.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif self.crossover_operator == CrossoverOperator.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif self.crossover_operator == CrossoverOperator.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif self.crossover_operator == CrossoverOperator.ARITHMETIC:
            return self._arithmetic_crossover(parent1, parent2)
        elif self.crossover_operator == CrossoverOperator.BLEND_ALPHA:
            return self._blend_alpha_crossover(parent1, parent2)
        else:
            return self._uniform_crossover(parent1, parent2)  # Default

    def _single_point_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Single-point crossover"""
        point = np.random.randint(1, len(parent1.genome))

        child1_genome = np.concatenate([parent1.genome[:point], parent2.genome[point:]])
        child2_genome = np.concatenate([parent2.genome[:point], parent1.genome[point:]])

        return Individual(genome=child1_genome), Individual(genome=child2_genome)

    def _two_point_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Two-point crossover"""
        point1 = np.random.randint(1, len(parent1.genome) - 1)
        point2 = np.random.randint(point1 + 1, len(parent1.genome))

        child1_genome = parent1.genome.copy()
        child2_genome = parent2.genome.copy()

        child1_genome[point1:point2] = parent2.genome[point1:point2]
        child2_genome[point1:point2] = parent1.genome[point1:point2]

        return Individual(genome=child1_genome), Individual(genome=child2_genome)

    def _uniform_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Uniform crossover"""
        mask = np.random.random(len(parent1.genome)) < 0.5

        child1_genome = np.where(mask, parent1.genome, parent2.genome)
        child2_genome = np.where(mask, parent2.genome, parent1.genome)

        return Individual(genome=child1_genome), Individual(genome=child2_genome)

    def _arithmetic_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Arithmetic crossover"""
        alpha = np.random.random()

        child1_genome = alpha * parent1.genome + (1 - alpha) * parent2.genome
        child2_genome = (1 - alpha) * parent1.genome + alpha * parent2.genome

        return Individual(genome=child1_genome), Individual(genome=child2_genome)

    def _blend_alpha_crossover(
        self, parent1: Individual, parent2: Individual, alpha: float = 0.5
    ) -> Tuple[Individual, Individual]:
        """Blend-alpha (BLX-α) crossover"""
        child1_genome = np.zeros_like(parent1.genome)
        child2_genome = np.zeros_like(parent2.genome)

        for i in range(len(parent1.genome)):
            min_val = min(parent1.genome[i], parent2.genome[i])
            max_val = max(parent1.genome[i], parent2.genome[i])
            range_val = max_val - min_val

            lower = min_val - alpha * range_val
            upper = max_val + alpha * range_val

            child1_genome[i] = np.random.uniform(lower, upper)
            child2_genome[i] = np.random.uniform(lower, upper)

        return Individual(genome=child1_genome), Individual(genome=child2_genome)

    def _mutate(self, individual: Individual) -> Individual:
        """Perform mutation on an individual"""
        mutated = individual.copy()

        if self.mutation_operator == MutationOperator.GAUSSIAN:
            self._gaussian_mutation(mutated)
        elif self.mutation_operator == MutationOperator.POLYNOMIAL:
            self._polynomial_mutation(mutated)
        elif self.mutation_operator == MutationOperator.UNIFORM:
            self._uniform_mutation(mutated)
        else:
            self._gaussian_mutation(mutated)  # Default

        return mutated

    def _gaussian_mutation(self, individual: Individual, sigma: Optional[float] = None) -> None:
        """Gaussian mutation"""
        if sigma is None:
            sigma = 0.1 * np.std(individual.genome) if np.std(individual.genome) > 0 else 0.1

        mutation = np.random.normal(0, sigma, size=len(individual.genome))
        individual.genome += mutation

        # Apply bounds
        lower_bounds, upper_bounds = self.fitness_function.get_bounds()
        individual.genome = np.clip(individual.genome, lower_bounds, upper_bounds)

    def _polynomial_mutation(self, individual: Individual, eta: float = 20.0) -> None:
        """Polynomial mutation"""
        lower_bounds, upper_bounds = self.fitness_function.get_bounds()

        for i in range(len(individual.genome)):
            if np.random.random() < 1.0 / len(individual.genome):
                x = individual.genome[i]
                xl = lower_bounds[i]
                xu = upper_bounds[i]

                delta1 = (x - xl) / (xu - xl)
                delta2 = (xu - x) / (xu - xl)

                rand = np.random.random()
                mut_pow = 1.0 / (eta + 1.0)

                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = val**mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val**mut_pow

                x = x + deltaq * (xu - xl)
                individual.genome[i] = np.clip(x, xl, xu)

    def _uniform_mutation(self, individual: Individual) -> None:
        """Uniform mutation"""
        lower_bounds, upper_bounds = self.fitness_function.get_bounds()

        for i in range(len(individual.genome)):
            if np.random.random() < 1.0 / len(individual.genome):
                individual.genome[i] = np.random.uniform(lower_bounds[i], upper_bounds[i])

    def _adapt_operators(self) -> None:
        """Adapt operator parameters based on performance"""
        if len(self.adaptive_params["success_rates"]) < 5:
            return

        recent_success = np.mean(list(self.adaptive_params["success_rates"])[-5:])
        recent_diversity = np.mean(list(self.adaptive_params["diversity_window"])[-5:])

        # Adapt crossover rate
        if recent_success < 0.3:  # Low success rate
            self.crossover_rate = min(0.95, self.crossover_rate + 0.05)
        elif recent_success > 0.7:  # High success rate
            self.crossover_rate = max(0.3, self.crossover_rate - 0.05)

        # Adapt mutation rate
        if recent_diversity < self.diversity_threshold:  # Low diversity
            self.mutation_rate = min(0.5, self.mutation_rate + 0.02)
        elif recent_diversity > 2 * self.diversity_threshold:  # High diversity
            self.mutation_rate = max(0.01, self.mutation_rate - 0.02)

        # Store parameter history
        self.adaptive_params["crossover_rate_history"].append(self.crossover_rate)
        self.adaptive_params["mutation_rate_history"].append(self.mutation_rate)

        logger.debug(
            f"Adapted parameters - CR: {self.crossover_rate:.3f}, MR: {self.mutation_rate:.3f}"
        )

    def _handle_stagnation(self) -> None:
        """Handle population stagnation by introducing diversity"""
        logger.info("Handling population stagnation")

        # Replace worst 20% of population with random individuals
        worst_count = int(0.2 * self.population_size)
        worst_individuals = self.population.get_worst(worst_count)

        lower_bounds, upper_bounds = self.fitness_function.get_bounds()

        for individual in worst_individuals:
            individual.genome = np.random.uniform(
                lower_bounds, upper_bounds, size=self.genome_length
            )
            individual.fitness = self.fitness_function.evaluate(individual.genome)
            individual.age = 0  # Reset age

        # Increase mutation rate temporarily
        self.mutation_rate = min(0.5, self.mutation_rate * 2.0)

    def _update_pareto_front(self) -> None:
        """Update Pareto front for multi-objective optimization"""
        if not self.multi_objective:
            return

        # Find non-dominated solutions
        pareto_front = []

        for individual in self.population.individuals:
            is_dominated = False
            for other in self.population.individuals:
                if other.dominates(individual):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(individual.copy())

        self.pareto_front = pareto_front
        logger.debug(f"Updated Pareto front with {len(pareto_front)} solutions")

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        stats = {
            "algorithm": {
                "strategy": self.strategy.value,
                "population_size": self.population_size,
                "generations_run": self.generation + 1,
                "selection_strategy": self.selection_strategy.value,
                "crossover_operator": self.crossover_operator.value,
                "mutation_operator": self.mutation_operator.value,
            },
            "performance": {
                "best_fitness": self.best_individual.fitness if self.best_individual else None,
                "convergence_history": self.convergence_history,
                "diversity_history": self.diversity_history,
                "final_diversity": self.diversity_history[-1] if self.diversity_history else None,
            },
            "population": self.population.statistics,
            "parameters": {
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "elitism_rate": self.elitism_rate,
                "adaptive_operators": self.adaptive_operators,
            },
        }

        if self.multi_objective:
            stats["pareto_front_size"] = len(self.pareto_front)

        return stats
