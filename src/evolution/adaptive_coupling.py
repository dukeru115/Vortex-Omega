"""
Adaptive Coupling Evolution - Dynamic Kuramoto Network Optimization
================================================================

Implements evolutionary optimization of Kuramoto synchronization networks
for NFCS v2.4.3. Provides adaptive coupling matrix evolution, dynamic topology
optimization, and synchronized network parameter tuning.

Scientific Foundation:
Based on Kuramoto model dynamics and adaptive synchronization theory:
- Coupling matrix evolution: K_ij(t) = f(topology, synchronization_state)
- Phase dynamics: dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)
- Synchronization order parameter: r(t) = |1/N Σ_j e^(iθ_j)|
- Adaptive coupling rules: dK_ij/dt = η * g(phase_difference, synchronization)

Mathematical Framework:
- Network topology optimization using graph-theoretic measures
- Coupling strength evolution with stability constraints
- Multi-objective optimization: maximize synchronization, minimize coupling cost
- Real-time adaptation based on synchronization performance metrics

Integration:
Optimizes Kuramoto coupling for NFCS semantic synchronization, ESC processing
enhancement, and multi-agent coordination in neural field control systems.

Created: September 14, 2025
Author: Team Ω - Neural Field Control Systems Research Group
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import random
from abc import ABC, abstractmethod
import networkx as nx
from scipy import stats
from scipy.integrate import odeint
from collections import defaultdict

from .genetic_optimizer import Individual, Population, FitnessFunction, GeneticOptimizer
from ..core.kuramoto_solver import KuramotoSolver

logger = logging.getLogger(__name__)


class TopologyType(Enum):
    """Network topology types for Kuramoto systems"""

    FULLY_CONNECTED = "fully_connected"
    RING = "ring"
    RANDOM = "random"
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class SynchronizationMetric(Enum):
    """Metrics for measuring synchronization quality"""

    ORDER_PARAMETER = "order_parameter"
    PHASE_COHERENCE = "phase_coherence"
    FREQUENCY_LOCKING = "frequency_locking"
    CLUSTER_SYNCHRONY = "cluster_synchrony"
    METASTABILITY = "metastability"
    CHIMERA_INDEX = "chimera_index"


class AdaptationRule(Enum):
    """Rules for adaptive coupling evolution"""

    HEBBIAN = "hebbian"
    ANTI_HEBBIAN = "anti_hebbian"
    STDP = "stdp"  # Spike-timing dependent plasticity
    HOMEOSTATIC = "homeostatic"
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"


@dataclass
class CouplingMatrix:
    """
    Represents a coupling matrix for Kuramoto networks.

    Attributes:
        matrix: N×N coupling strength matrix K_ij
        topology_type: Type of network topology
        N: Number of oscillators
        directed: Whether the network is directed
        weighted: Whether connections have variable weights
        dynamic: Whether coupling changes over time
        adaptation_rule: Rule for dynamic adaptation
        metadata: Additional matrix metadata
    """

    matrix: np.ndarray
    topology_type: TopologyType = TopologyType.FULLY_CONNECTED
    N: int = 0
    directed: bool = False
    weighted: bool = True
    dynamic: bool = False
    adaptation_rule: Optional[AdaptationRule] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived attributes"""
        if self.N == 0:
            self.N = self.matrix.shape[0]

        # Validate matrix dimensions
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"Coupling matrix must be square, got shape {self.matrix.shape}")

        if self.matrix.shape[0] != self.N:
            raise ValueError(f"Matrix size {self.matrix.shape[0]} != N {self.N}")

    def get_network_properties(self) -> Dict[str, float]:
        """Compute network topology properties"""
        # Create NetworkX graph for analysis
        G = nx.from_numpy_array(
            np.abs(self.matrix), create_using=nx.DiGraph if self.directed else nx.Graph
        )

        properties = {}

        # Basic properties
        properties["density"] = nx.density(G)
        properties["num_edges"] = G.number_of_edges()

        # Degree statistics
        if self.directed:
            in_degrees = [d for n, d in G.in_degree()]
            out_degrees = [d for n, d in G.out_degree()]
            properties["avg_in_degree"] = np.mean(in_degrees)
            properties["avg_out_degree"] = np.mean(out_degrees)
            properties["degree_variance"] = np.var(in_degrees + out_degrees)
        else:
            degrees = [d for n, d in G.degree()]
            properties["avg_degree"] = np.mean(degrees)
            properties["degree_variance"] = np.var(degrees)

        # Connectivity
        if nx.is_connected(G.to_undirected()):
            properties["is_connected"] = 1.0
            properties["diameter"] = nx.diameter(G.to_undirected())
            properties["avg_path_length"] = nx.average_shortest_path_length(G.to_undirected())
        else:
            properties["is_connected"] = 0.0
            properties["diameter"] = np.inf
            properties["avg_path_length"] = np.inf

        # Clustering
        if not self.directed:
            properties["avg_clustering"] = nx.average_clustering(G)
            properties["transitivity"] = nx.transitivity(G)

            # Small-world properties
            try:
                properties["small_worldness"] = nx.sigma(G)
            except:
                properties["small_worldness"] = 0.0

        # Centrality measures
        try:
            betweenness = nx.betweenness_centrality(G)
            properties["avg_betweenness"] = np.mean(list(betweenness.values()))

            closeness = nx.closeness_centrality(G)
            properties["avg_closeness"] = np.mean(list(closeness.values()))

            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
            properties["avg_eigenvector"] = np.mean(list(eigenvector.values()))
        except:
            properties["avg_betweenness"] = 0.0
            properties["avg_closeness"] = 0.0
            properties["avg_eigenvector"] = 0.0

        # Spectral properties
        try:
            laplacian_eigenvals = nx.laplacian_spectrum(G.to_undirected())
            properties["algebraic_connectivity"] = (
                np.sort(laplacian_eigenvals)[1] if len(laplacian_eigenvals) > 1 else 0.0
            )
            properties["spectral_gap"] = np.max(laplacian_eigenvals) - np.min(
                laplacian_eigenvals[laplacian_eigenvals > 1e-10]
            )
        except:
            properties["algebraic_connectivity"] = 0.0
            properties["spectral_gap"] = 0.0

        return properties

    def get_coupling_statistics(self) -> Dict[str, float]:
        """Compute coupling matrix statistics"""
        stats = {}

        # Basic statistics
        stats["mean_coupling"] = np.mean(self.matrix)
        stats["std_coupling"] = np.std(self.matrix)
        stats["max_coupling"] = np.max(self.matrix)
        stats["min_coupling"] = np.min(self.matrix)

        # Matrix properties
        stats["matrix_norm"] = np.linalg.norm(self.matrix)
        stats["trace"] = np.trace(self.matrix)
        stats["determinant"] = np.linalg.det(self.matrix)

        # Eigenvalues
        try:
            eigenvals = np.linalg.eigvals(self.matrix)
            stats["max_eigenvalue"] = np.max(np.real(eigenvals))
            stats["min_eigenvalue"] = np.min(np.real(eigenvals))
            stats["eigenvalue_spread"] = stats["max_eigenvalue"] - stats["min_eigenvalue"]

            # Stability indicator (for real symmetric matrices)
            if np.allclose(self.matrix, self.matrix.T):
                stats["is_stable"] = float(np.all(np.real(eigenvals) <= 0))
            else:
                stats["is_stable"] = float(np.all(np.real(eigenvals) < 0))
        except:
            stats["max_eigenvalue"] = 0.0
            stats["min_eigenvalue"] = 0.0
            stats["eigenvalue_spread"] = 0.0
            stats["is_stable"] = 0.0

        # Symmetry
        if self.matrix.shape[0] == self.matrix.shape[1]:
            stats["symmetry"] = 1.0 - np.mean(np.abs(self.matrix - self.matrix.T)) / np.mean(
                np.abs(self.matrix)
            )

        # Sparsity
        total_elements = self.matrix.size
        zero_elements = np.sum(np.abs(self.matrix) < 1e-10)
        stats["sparsity"] = zero_elements / total_elements
        stats["density"] = 1.0 - stats["sparsity"]

        return stats

    def copy(self) -> "CouplingMatrix":
        """Create a deep copy of the coupling matrix"""
        return CouplingMatrix(
            matrix=self.matrix.copy(),
            topology_type=self.topology_type,
            N=self.N,
            directed=self.directed,
            weighted=self.weighted,
            dynamic=self.dynamic,
            adaptation_rule=self.adaptation_rule,
            metadata=self.metadata.copy(),
        )


@dataclass
class SynchronizationState:
    """
    Represents the synchronization state of a Kuramoto network.

    Attributes:
        phases: Current phase values θ_i
        frequencies: Natural frequencies ω_i
        order_parameter: Complex order parameter r*e^(iψ)
        sync_level: Synchronization level |r|
        mean_phase: Mean phase ψ
        phase_coherence: Phase coherence measure
        cluster_indices: Cluster membership indices
        metastability: Metastability measure
        time: Current time
    """

    phases: np.ndarray
    frequencies: np.ndarray
    order_parameter: complex = 0.0
    sync_level: float = 0.0
    mean_phase: float = 0.0
    phase_coherence: float = 0.0
    cluster_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    metastability: float = 0.0
    time: float = 0.0

    def __post_init__(self):
        """Compute derived synchronization metrics"""
        self.update_synchronization_metrics()

    def update_synchronization_metrics(self) -> None:
        """Update all synchronization metrics from current phases"""
        N = len(self.phases)

        # Order parameter: r*e^(iψ) = (1/N) * Σ e^(iθ_j)
        complex_phases = np.exp(1j * self.phases)
        self.order_parameter = np.mean(complex_phases)
        self.sync_level = np.abs(self.order_parameter)
        self.mean_phase = np.angle(self.order_parameter)

        # Phase coherence (alternative measure)
        phase_diffs = self.phases[:, np.newaxis] - self.phases[np.newaxis, :]
        cos_diffs = np.cos(phase_diffs)
        self.phase_coherence = np.mean(cos_diffs[np.triu_indices(N, k=1)])

        # Metastability (variance of order parameter over time window)
        # Note: This requires time series data, simplified here
        self.metastability = 1.0 - self.sync_level  # Simplified measure

    def get_cluster_structure(self, threshold: float = 0.9) -> np.ndarray:
        """Identify phase clusters in the network"""
        N = len(self.phases)

        # Compute pairwise phase similarities
        phase_diffs = self.phases[:, np.newaxis] - self.phases[np.newaxis, :]
        similarities = np.cos(phase_diffs)

        # Simple clustering based on similarity threshold
        clusters = np.zeros(N, dtype=int)
        cluster_id = 0

        for i in range(N):
            if clusters[i] == 0:  # Not yet assigned
                cluster_id += 1
                clusters[i] = cluster_id

                # Find similar oscillators
                similar_indices = np.where(similarities[i] > threshold)[0]
                clusters[similar_indices] = cluster_id

        self.cluster_indices = clusters
        return clusters

    def compute_local_order_parameters(self, coupling_matrix: np.ndarray) -> np.ndarray:
        """Compute local order parameters for each oscillator"""
        N = len(self.phases)
        local_orders = np.zeros(N, dtype=complex)

        for i in range(N):
            # Weighted local order parameter based on coupling
            weights = coupling_matrix[i, :]
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                local_orders[i] = np.sum(weights * np.exp(1j * self.phases))
            else:
                local_orders[i] = 0.0

        return local_orders


class KuramotoFitnessFunction(FitnessFunction):
    """
    Fitness function for Kuramoto coupling matrix optimization.

    Evaluates coupling matrices based on synchronization performance,
    network efficiency, and stability criteria.
    """

    def __init__(
        self,
        N: int,
        natural_frequencies: np.ndarray,
        simulation_time: float = 50.0,
        dt: float = 0.01,
        objectives: List[SynchronizationMetric] = None,
        objective_weights: Dict[SynchronizationMetric, float] = None,
        topology_constraints: Dict[str, Any] = None,
        target_sync_level: float = 0.8,
    ):
        """
        Initialize Kuramoto fitness function.

        Args:
            N: Number of oscillators
            natural_frequencies: Natural frequency distribution ω_i
            simulation_time: Duration of synchronization simulation
            dt: Integration time step
            objectives: List of synchronization objectives
            objective_weights: Weights for multi-objective optimization
            topology_constraints: Network topology constraints
            target_sync_level: Target synchronization level
        """
        self.N = N
        self.natural_frequencies = natural_frequencies
        self.simulation_time = simulation_time
        self.dt = dt
        self.target_sync_level = target_sync_level

        # Set default objectives
        if objectives is None:
            objectives = [
                SynchronizationMetric.ORDER_PARAMETER,
                SynchronizationMetric.PHASE_COHERENCE,
                SynchronizationMetric.FREQUENCY_LOCKING,
            ]
        self.objectives = objectives

        # Set default weights
        if objective_weights is None:
            objective_weights = {obj: 1.0 for obj in objectives}
        self.objective_weights = objective_weights

        # Set topology constraints
        if topology_constraints is None:
            topology_constraints = {
                "max_coupling_strength": 10.0,
                "min_coupling_strength": 0.0,
                "max_connections_per_node": N,
                "require_connectivity": True,
                "symmetry_preference": 0.5,
            }
        self.topology_constraints = topology_constraints

        # Initialize Kuramoto solver
        self.kuramoto_solver = KuramotoSolver(N=N, dt=dt)

        logger.info(f"Initialized KuramotoFitnessFunction for {N} oscillators")

    def evaluate(self, genome: np.ndarray) -> float:
        """
        Evaluate fitness of coupling matrix genome.

        Args:
            genome: Flattened coupling matrix

        Returns:
            Fitness value (higher is better)
        """
        try:
            # Reconstruct coupling matrix from genome
            coupling_matrix = self._genome_to_matrix(genome)

            # Check constraints
            constraint_penalty = self._evaluate_constraints(coupling_matrix)
            if constraint_penalty > 1e3:  # Severe constraint violation
                return -constraint_penalty

            # Run Kuramoto simulation
            sync_results = self._simulate_kuramoto_dynamics(coupling_matrix)

            # Evaluate objectives
            objective_values = {}

            for objective in self.objectives:
                if objective == SynchronizationMetric.ORDER_PARAMETER:
                    objective_values[objective] = self._evaluate_order_parameter(sync_results)
                elif objective == SynchronizationMetric.PHASE_COHERENCE:
                    objective_values[objective] = self._evaluate_phase_coherence(sync_results)
                elif objective == SynchronizationMetric.FREQUENCY_LOCKING:
                    objective_values[objective] = self._evaluate_frequency_locking(sync_results)
                elif objective == SynchronizationMetric.CLUSTER_SYNCHRONY:
                    objective_values[objective] = self._evaluate_cluster_synchrony(sync_results)
                elif objective == SynchronizationMetric.METASTABILITY:
                    objective_values[objective] = self._evaluate_metastability(sync_results)

            # Combine objectives
            total_fitness = 0.0
            for objective, value in objective_values.items():
                weight = self.objective_weights.get(objective, 1.0)
                total_fitness += weight * value

            # Apply constraint penalty
            total_fitness -= constraint_penalty

            return total_fitness

        except Exception as e:
            logger.warning(f"Error in Kuramoto fitness evaluation: {e}")
            return -1e6  # Penalty for invalid matrices

    def _genome_to_matrix(self, genome: np.ndarray) -> CouplingMatrix:
        """Convert genome vector to coupling matrix"""
        # Reshape genome to matrix
        matrix = genome.reshape(self.N, self.N)

        # Apply symmetry if preferred
        if self.topology_constraints.get("symmetry_preference", 0) > 0.5:
            matrix = (matrix + matrix.T) / 2

        # Create coupling matrix object
        coupling_matrix = CouplingMatrix(
            matrix=matrix,
            N=self.N,
            weighted=True,
            directed=not (self.topology_constraints.get("symmetry_preference", 0) > 0.5),
        )

        return coupling_matrix

    def _simulate_kuramoto_dynamics(self, coupling_matrix: CouplingMatrix) -> Dict[str, Any]:
        """Simulate Kuramoto dynamics with given coupling matrix"""
        # Set solver parameters
        self.kuramoto_solver.K = 1.0  # Base coupling (will be modulated by matrix)
        self.kuramoto_solver.natural_frequencies = self.natural_frequencies

        # Initialize phases randomly
        phases = np.random.uniform(0, 2 * np.pi, self.N)

        # Storage for results
        time_points = np.arange(0, self.simulation_time, self.dt)
        phase_history = []
        order_param_history = []

        # Simulation loop
        for t in time_points:
            # Compute coupled dynamics
            phase_diffs = phases[:, np.newaxis] - phases[np.newaxis, :]
            coupling_forces = coupling_matrix.matrix * np.sin(phase_diffs)

            # Update phases
            dphases_dt = self.natural_frequencies + np.sum(coupling_forces, axis=1)
            phases += dphases_dt * self.dt

            # Normalize phases to [0, 2π)
            phases = np.mod(phases, 2 * np.pi)

            # Store results (subsample for efficiency)
            if len(phase_history) < 1000:  # Limit storage
                phase_history.append(phases.copy())

                # Compute order parameter
                order_param = np.mean(np.exp(1j * phases))
                order_param_history.append(order_param)

        return {
            "phase_history": phase_history,
            "order_param_history": order_param_history,
            "final_phases": phases,
            "time_points": time_points[: len(phase_history)],
            "coupling_matrix": coupling_matrix,
        }

    def _evaluate_order_parameter(self, results: Dict[str, Any]) -> float:
        """Evaluate synchronization order parameter"""
        order_params = results["order_param_history"]
        if not order_params:
            return 0.0

        # Average order parameter over second half of simulation (steady state)
        steady_state_start = len(order_params) // 2
        steady_state_orders = order_params[steady_state_start:]

        avg_order = np.mean([np.abs(op) for op in steady_state_orders])

        # Reward high synchronization
        return avg_order

    def _evaluate_phase_coherence(self, results: Dict[str, Any]) -> float:
        """Evaluate phase coherence across oscillators"""
        phase_history = results["phase_history"]
        if not phase_history:
            return 0.0

        # Compute coherence over time
        coherence_values = []
        for phases in phase_history[-len(phase_history) // 2 :]:  # Steady state
            N = len(phases)
            phase_diffs = phases[:, np.newaxis] - phases[np.newaxis, :]
            coherence = np.mean(np.cos(phase_diffs))
            coherence_values.append(coherence)

        return np.mean(coherence_values)

    def _evaluate_frequency_locking(self, results: Dict[str, Any]) -> float:
        """Evaluate frequency locking quality"""
        phase_history = results["phase_history"]
        if len(phase_history) < 10:
            return 0.0

        # Compute instantaneous frequencies
        dt = self.dt
        freq_estimates = []

        for i in range(len(phase_history) - 1):
            phase_diffs = np.array(phase_history[i + 1]) - np.array(phase_history[i])
            # Handle phase wrapping
            phase_diffs = np.mod(phase_diffs + np.pi, 2 * np.pi) - np.pi
            frequencies = phase_diffs / dt
            freq_estimates.append(frequencies)

        if not freq_estimates:
            return 0.0

        # Analyze frequency variance (lower variance = better locking)
        steady_state_freqs = freq_estimates[-len(freq_estimates) // 2 :]
        freq_variance = np.var([f for freqs in steady_state_freqs for f in freqs])

        # Return inverse of variance (higher = better locking)
        return 1.0 / (1.0 + freq_variance)

    def _evaluate_cluster_synchrony(self, results: Dict[str, Any]) -> float:
        """Evaluate quality of cluster synchronization"""
        final_phases = results["final_phases"]

        # Create synchronization state
        sync_state = SynchronizationState(phases=final_phases, frequencies=self.natural_frequencies)

        # Get cluster structure
        clusters = sync_state.get_cluster_structure(threshold=0.8)

        # Evaluate cluster quality
        unique_clusters = np.unique(clusters)
        if len(unique_clusters) <= 1:
            return sync_state.sync_level  # Full synchronization

        # Compute within-cluster synchronization
        cluster_sync_levels = []
        for cluster_id in unique_clusters:
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 1:
                cluster_phases = final_phases[cluster_indices]
                cluster_order = np.abs(np.mean(np.exp(1j * cluster_phases)))
                cluster_sync_levels.append(cluster_order)

        # Return average cluster synchronization
        return np.mean(cluster_sync_levels) if cluster_sync_levels else 0.0

    def _evaluate_metastability(self, results: Dict[str, Any]) -> float:
        """Evaluate metastability (dynamic synchronization patterns)"""
        order_params = results["order_param_history"]
        if len(order_params) < 10:
            return 0.0

        # Compute variance of order parameter over time
        order_magnitudes = [np.abs(op) for op in order_params]
        order_variance = np.var(order_magnitudes)

        # Metastability is moderate variance (not too stable, not too chaotic)
        optimal_variance = 0.1
        metastability = 1.0 / (1.0 + abs(order_variance - optimal_variance))

        return metastability

    def _evaluate_constraints(self, coupling_matrix: CouplingMatrix) -> float:
        """Evaluate constraint violations and return penalty"""
        penalty = 0.0

        matrix = coupling_matrix.matrix
        constraints = self.topology_constraints

        # Coupling strength constraints
        max_allowed = constraints.get("max_coupling_strength", 10.0)
        min_allowed = constraints.get("min_coupling_strength", 0.0)

        over_max = np.sum(matrix > max_allowed)
        under_min = np.sum(matrix < min_allowed)

        if over_max > 0:
            penalty += 100 * over_max + 10 * np.sum((matrix - max_allowed)[matrix > max_allowed])
        if under_min > 0:
            penalty += 100 * under_min + 10 * np.sum((min_allowed - matrix)[matrix < min_allowed])

        # Connectivity constraint
        if constraints.get("require_connectivity", False):
            G = nx.from_numpy_array(np.abs(matrix))
            if not nx.is_connected(G):
                penalty += 1000  # Large penalty for disconnected networks

        # Maximum connections per node
        max_connections = constraints.get("max_connections_per_node", self.N)
        for i in range(self.N):
            connections = np.sum(np.abs(matrix[i, :]) > 1e-6)
            if connections > max_connections:
                penalty += 50 * (connections - max_connections)

        return penalty

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds for coupling matrix optimization"""
        matrix_size = self.N * self.N

        lower_bounds = np.full(
            matrix_size, self.topology_constraints.get("min_coupling_strength", -5.0)
        )
        upper_bounds = np.full(
            matrix_size, self.topology_constraints.get("max_coupling_strength", 5.0)
        )

        return lower_bounds, upper_bounds


class AdaptiveCoupling:
    """
    Main adaptive coupling evolution coordinator.

    Implements evolutionary optimization of Kuramoto coupling matrices
    with real-time adaptation capabilities for NFCS systems.
    """

    def __init__(
        self,
        N: int,
        natural_frequencies: Optional[np.ndarray] = None,
        topology_type: TopologyType = TopologyType.ADAPTIVE,
        optimization_config: Dict[str, Any] = None,
        adaptation_config: Dict[str, Any] = None,
    ):
        """
        Initialize adaptive coupling system.

        Args:
            N: Number of oscillators
            natural_frequencies: Natural frequency distribution (generated if None)
            topology_type: Initial network topology type
            optimization_config: Configuration for evolutionary optimization
            adaptation_config: Configuration for real-time adaptation
        """
        self.N = N

        # Set natural frequencies
        if natural_frequencies is None:
            # Generate heterogeneous frequency distribution
            natural_frequencies = np.random.normal(0, 1, N)
        self.natural_frequencies = natural_frequencies

        self.topology_type = topology_type

        # Set default optimization configuration
        if optimization_config is None:
            optimization_config = {
                "population_size": 100,
                "max_generations": 200,
                "crossover_rate": 0.8,
                "mutation_rate": 0.2,
                "elitism_rate": 0.1,
                "simulation_time": 30.0,
                "target_sync_level": 0.8,
            }
        self.optimization_config = optimization_config

        # Set default adaptation configuration
        if adaptation_config is None:
            adaptation_config = {
                "adaptation_rate": 0.1,
                "adaptation_rule": AdaptationRule.HEBBIAN,
                "plasticity_window": 100,
                "stability_threshold": 0.05,
                "enable_real_time": True,
            }
        self.adaptation_config = adaptation_config

        # Initialize fitness function
        self.fitness_function = KuramotoFitnessFunction(
            N=N,
            natural_frequencies=natural_frequencies,
            simulation_time=optimization_config["simulation_time"],
            target_sync_level=optimization_config["target_sync_level"],
        )

        # Initialize genetic optimizer
        genome_length = N * N  # Flattened coupling matrix

        self.optimizer = GeneticOptimizer(
            fitness_function=self.fitness_function,
            population_size=optimization_config["population_size"],
            genome_length=genome_length,
            max_generations=optimization_config["max_generations"],
            crossover_rate=optimization_config["crossover_rate"],
            mutation_rate=optimization_config["mutation_rate"],
            elitism_rate=optimization_config["elitism_rate"],
        )

        # Coupling evolution state
        self.current_coupling = self._initialize_coupling_matrix()
        self.coupling_history = []
        self.sync_performance_history = []
        self.adaptation_events = []

        # Real-time adaptation state
        self.phase_buffer = []
        self.performance_buffer = []
        self.last_adaptation_time = 0

        logger.info(f"Initialized AdaptiveCoupling for {N} oscillators")

    def _initialize_coupling_matrix(self) -> CouplingMatrix:
        """Initialize coupling matrix based on topology type"""
        matrix = np.zeros((self.N, self.N))

        if self.topology_type == TopologyType.FULLY_CONNECTED:
            # Fully connected with random weights
            matrix = np.random.normal(0, 1, (self.N, self.N))
            np.fill_diagonal(matrix, 0)  # No self-coupling

        elif self.topology_type == TopologyType.RING:
            # Ring topology
            for i in range(self.N):
                matrix[i, (i + 1) % self.N] = 1.0
                matrix[i, (i - 1) % self.N] = 1.0

        elif self.topology_type == TopologyType.RANDOM:
            # Random topology
            connection_prob = 0.3
            for i in range(self.N):
                for j in range(self.N):
                    if i != j and np.random.random() < connection_prob:
                        matrix[i, j] = np.random.normal(0, 1)

        elif self.topology_type == TopologyType.SMALL_WORLD:
            # Small-world (Watts-Strogatz)
            G = nx.watts_strogatz_graph(self.N, 4, 0.3)
            matrix = nx.to_numpy_array(G)
            matrix += np.random.normal(0, 0.1, matrix.shape)  # Add weight variation

        elif self.topology_type == TopologyType.SCALE_FREE:
            # Scale-free (Barabási–Albert)
            G = nx.barabasi_albert_graph(self.N, 3)
            matrix = nx.to_numpy_array(G)
            matrix += np.random.normal(0, 0.1, matrix.shape)

        elif self.topology_type == TopologyType.ADAPTIVE:
            # Start with small-world and evolve
            G = nx.watts_strogatz_graph(self.N, 4, 0.1)
            matrix = nx.to_numpy_array(G)
            matrix += np.random.normal(0, 0.5, matrix.shape)

        # Normalize and create coupling matrix
        if np.max(np.abs(matrix)) > 0:
            matrix = matrix / np.max(np.abs(matrix)) * 2.0  # Scale to reasonable range

        return CouplingMatrix(
            matrix=matrix,
            topology_type=self.topology_type,
            N=self.N,
            weighted=True,
            dynamic=True,
            adaptation_rule=self.adaptation_config["adaptation_rule"],
        )

    def evolve_coupling_matrix(
        self,
        objectives: List[SynchronizationMetric] = None,
        objective_weights: Dict[SynchronizationMetric, float] = None,
    ) -> Dict[str, Any]:
        """
        Evolve optimal coupling matrix using genetic algorithm.

        Args:
            objectives: Synchronization objectives to optimize
            objective_weights: Weights for multi-objective optimization

        Returns:
            Evolution results including optimal coupling matrix
        """
        start_time = time.time()

        # Update fitness function objectives
        if objectives is not None:
            self.fitness_function.objectives = objectives
        if objective_weights is not None:
            self.fitness_function.objective_weights = objective_weights

        # Initialize with current coupling matrix
        self.optimizer.initialize_population()

        # Set first individual to current coupling
        current_genome = self.current_coupling.matrix.flatten()
        self.optimizer.population.individuals[0].genome = current_genome

        logger.info("Starting coupling matrix evolution...")

        # Run evolutionary optimization
        evolution_results = self.optimizer.evolve()

        # Extract best coupling matrix
        if evolution_results["best_genome"] is not None:
            best_matrix = evolution_results["best_genome"].reshape(self.N, self.N)

            # Create new coupling matrix
            evolved_coupling = CouplingMatrix(
                matrix=best_matrix,
                topology_type=TopologyType.ADAPTIVE,
                N=self.N,
                weighted=True,
                dynamic=True,
                adaptation_rule=self.adaptation_config["adaptation_rule"],
            )

            # Update current coupling
            old_coupling = self.current_coupling.copy()
            self.current_coupling = evolved_coupling

            # Record evolution event
            self.adaptation_events.append(
                {
                    "type": "evolutionary_optimization",
                    "timestamp": time.time(),
                    "old_coupling": old_coupling,
                    "new_coupling": evolved_coupling,
                    "fitness_improvement": evolution_results["best_fitness"],
                    "generation": evolution_results["generations_run"],
                }
            )

        # Store in history
        self.coupling_history.append(
            {
                "timestamp": time.time(),
                "coupling_matrix": self.current_coupling.copy(),
                "evolution_results": evolution_results,
                "network_properties": self.current_coupling.get_network_properties(),
                "coupling_statistics": self.current_coupling.get_coupling_statistics(),
            }
        )

        execution_time = time.time() - start_time

        # Compile comprehensive results
        results = {
            "evolved_coupling": self.current_coupling,
            "evolution_results": evolution_results,
            "execution_time": execution_time,
            "network_analysis": self._analyze_network_properties(),
            "synchronization_analysis": self._analyze_synchronization_performance(),
            "optimization_statistics": self.optimizer.get_optimization_statistics(),
        }

        logger.info(
            f"Coupling evolution completed in {execution_time:.2f}s. "
            f"Best fitness: {evolution_results['best_fitness']:.4f}"
        )

        return results

    def update_real_time_adaptation(
        self, current_phases: np.ndarray, performance_metrics: Dict[str, float], dt: float = 0.01
    ) -> CouplingMatrix:
        """
        Update coupling matrix using real-time adaptation.

        Args:
            current_phases: Current oscillator phases
            performance_metrics: Current synchronization performance
            dt: Time step

        Returns:
            Updated coupling matrix
        """
        if not self.adaptation_config["enable_real_time"]:
            return self.current_coupling

        # Store current state
        self.phase_buffer.append(current_phases.copy())
        self.performance_buffer.append(performance_metrics.copy())

        # Limit buffer size
        max_buffer_size = self.adaptation_config["plasticity_window"]
        if len(self.phase_buffer) > max_buffer_size:
            self.phase_buffer.pop(0)
            self.performance_buffer.pop(0)

        # Check if adaptation is needed
        current_time = time.time()
        adaptation_interval = 1.0  # Adapt every second

        if (
            current_time - self.last_adaptation_time > adaptation_interval
            and len(self.phase_buffer) >= 10
        ):

            # Apply adaptation rule
            self._apply_adaptation_rule(current_phases, dt)
            self.last_adaptation_time = current_time

        return self.current_coupling

    def _apply_adaptation_rule(self, phases: np.ndarray, dt: float) -> None:
        """Apply adaptation rule to update coupling matrix"""
        adaptation_rule = self.adaptation_config["adaptation_rule"]
        adaptation_rate = self.adaptation_config["adaptation_rate"]

        if adaptation_rule == AdaptationRule.HEBBIAN:
            self._apply_hebbian_adaptation(phases, dt, adaptation_rate)
        elif adaptation_rule == AdaptationRule.ANTI_HEBBIAN:
            self._apply_anti_hebbian_adaptation(phases, dt, adaptation_rate)
        elif adaptation_rule == AdaptationRule.STDP:
            self._apply_stdp_adaptation(phases, dt, adaptation_rate)
        elif adaptation_rule == AdaptationRule.HOMEOSTATIC:
            self._apply_homeostatic_adaptation(phases, dt, adaptation_rate)
        elif adaptation_rule == AdaptationRule.COMPETITIVE:
            self._apply_competitive_adaptation(phases, dt, adaptation_rate)
        elif adaptation_rule == AdaptationRule.COOPERATIVE:
            self._apply_cooperative_adaptation(phases, dt, adaptation_rate)

    def _apply_hebbian_adaptation(self, phases: np.ndarray, dt: float, rate: float) -> None:
        """Apply Hebbian adaptation rule: strengthen connections between synchronized oscillators"""
        # Compute phase synchronization matrix
        phase_diffs = phases[:, np.newaxis] - phases[np.newaxis, :]
        sync_matrix = np.cos(phase_diffs)

        # Hebbian update: dK_ij/dt = η * cos(θ_i - θ_j)
        coupling_update = rate * sync_matrix * dt

        # Apply update
        self.current_coupling.matrix += coupling_update

        # Apply bounds
        self.current_coupling.matrix = np.clip(self.current_coupling.matrix, -5.0, 5.0)

    def _apply_anti_hebbian_adaptation(self, phases: np.ndarray, dt: float, rate: float) -> None:
        """Apply anti-Hebbian adaptation: weaken connections between synchronized oscillators"""
        phase_diffs = phases[:, np.newaxis] - phases[np.newaxis, :]
        sync_matrix = np.cos(phase_diffs)

        # Anti-Hebbian update
        coupling_update = -rate * sync_matrix * dt

        self.current_coupling.matrix += coupling_update
        self.current_coupling.matrix = np.clip(self.current_coupling.matrix, -5.0, 5.0)

    def _apply_stdp_adaptation(self, phases: np.ndarray, dt: float, rate: float) -> None:
        """Apply spike-timing dependent plasticity adaptation"""
        if len(self.phase_buffer) < 2:
            return

        # Compute phase velocities (spike timing)
        prev_phases = self.phase_buffer[-2]
        phase_velocities = (phases - prev_phases) / dt

        # STDP window function (simplified)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    time_diff = phase_velocities[i] - phase_velocities[j]

                    # STDP update rule
                    if time_diff > 0:  # i leads j
                        stdp_update = rate * np.exp(-time_diff / 0.02)
                    else:  # j leads i
                        stdp_update = -rate * np.exp(time_diff / 0.02)

                    self.current_coupling.matrix[i, j] += stdp_update * dt

        self.current_coupling.matrix = np.clip(self.current_coupling.matrix, -5.0, 5.0)

    def _apply_homeostatic_adaptation(self, phases: np.ndarray, dt: float, rate: float) -> None:
        """Apply homeostatic adaptation to maintain target synchronization level"""
        # Compute current synchronization level
        current_sync = np.abs(np.mean(np.exp(1j * phases)))
        target_sync = 0.7  # Target synchronization level

        # Homeostatic scaling
        sync_error = target_sync - current_sync

        # Global scaling of coupling matrix
        scaling_factor = 1.0 + rate * sync_error * dt
        self.current_coupling.matrix *= scaling_factor

        self.current_coupling.matrix = np.clip(self.current_coupling.matrix, -5.0, 5.0)

    def _apply_competitive_adaptation(self, phases: np.ndarray, dt: float, rate: float) -> None:
        """Apply competitive adaptation: strengthen strong connections, weaken weak ones"""
        current_matrix = self.current_coupling.matrix

        # Compute connection strengths
        strengths = np.abs(current_matrix)

        # Winner-take-all dynamics
        for i in range(self.N):
            row_max = np.max(strengths[i, :])
            if row_max > 0:
                # Strengthen strongest connections
                strongest_indices = np.where(strengths[i, :] == row_max)[0]
                for j in strongest_indices:
                    if i != j:
                        self.current_coupling.matrix[i, j] += rate * dt

                # Weaken other connections
                for j in range(self.N):
                    if i != j and j not in strongest_indices:
                        self.current_coupling.matrix[i, j] -= rate * 0.1 * dt

        self.current_coupling.matrix = np.clip(self.current_coupling.matrix, -5.0, 5.0)

    def _apply_cooperative_adaptation(self, phases: np.ndarray, dt: float, rate: float) -> None:
        """Apply cooperative adaptation: enhance global synchronization"""
        # Compute global order parameter
        order_param = np.mean(np.exp(1j * phases))
        sync_level = np.abs(order_param)

        # Enhance connections that contribute to synchronization
        phase_diffs = phases[:, np.newaxis] - phases[np.newaxis, :]
        sync_contributions = np.cos(phase_diffs)

        # Cooperative update proportional to synchronization contribution
        cooperation_matrix = sync_contributions * sync_level
        coupling_update = rate * cooperation_matrix * dt

        self.current_coupling.matrix += coupling_update
        self.current_coupling.matrix = np.clip(self.current_coupling.matrix, -5.0, 5.0)

    def _analyze_network_properties(self) -> Dict[str, Any]:
        """Analyze current network topology properties"""
        return {
            "network_properties": self.current_coupling.get_network_properties(),
            "coupling_statistics": self.current_coupling.get_coupling_statistics(),
            "topology_type": self.current_coupling.topology_type.value,
            "adaptation_rule": (
                self.current_coupling.adaptation_rule.value
                if self.current_coupling.adaptation_rule
                else None
            ),
        }

    def _analyze_synchronization_performance(self) -> Dict[str, Any]:
        """Analyze synchronization performance of current coupling"""
        # Run short simulation to assess performance
        test_phases = np.random.uniform(0, 2 * np.pi, self.N)

        # Simple integration for performance assessment
        dt = 0.01
        for _ in range(100):  # Short simulation
            phase_diffs = test_phases[:, np.newaxis] - test_phases[np.newaxis, :]
            coupling_forces = self.current_coupling.matrix * np.sin(phase_diffs)
            dphases_dt = self.natural_frequencies + np.sum(coupling_forces, axis=1)
            test_phases += dphases_dt * dt
            test_phases = np.mod(test_phases, 2 * np.pi)

        # Compute synchronization metrics
        sync_state = SynchronizationState(phases=test_phases, frequencies=self.natural_frequencies)

        return {
            "order_parameter": sync_state.sync_level,
            "phase_coherence": sync_state.phase_coherence,
            "mean_phase": sync_state.mean_phase,
            "metastability": sync_state.metastability,
            "cluster_structure": sync_state.get_cluster_structure().tolist(),
        }

    def get_current_coupling(self) -> CouplingMatrix:
        """Get current coupling matrix"""
        return self.current_coupling

    def get_adaptation_history(self) -> Dict[str, Any]:
        """Get comprehensive adaptation history"""
        return {
            "coupling_history": self.coupling_history,
            "adaptation_events": self.adaptation_events,
            "sync_performance_history": self.sync_performance_history,
            "adaptation_statistics": {
                "total_adaptations": len(self.adaptation_events),
                "evolutionary_optimizations": len(
                    [
                        event
                        for event in self.adaptation_events
                        if event["type"] == "evolutionary_optimization"
                    ]
                ),
                "real_time_adaptations": len(
                    [
                        event
                        for event in self.adaptation_events
                        if event["type"] == "real_time_adaptation"
                    ]
                ),
            },
        }


class CouplingMatrixEvolution(GeneticOptimizer):
    """Specialized genetic optimizer for coupling matrix evolution"""

    def __init__(self, N: int, **kwargs):
        """Initialize coupling matrix evolution"""
        # Create specialized fitness function for coupling matrices
        fitness_function = KuramotoFitnessFunction(N=N)

        super().__init__(fitness_function=fitness_function, genome_length=N * N, **kwargs)

        self.N = N

    def _create_coupling_individual(self, matrix: np.ndarray) -> Individual:
        """Create individual from coupling matrix"""
        genome = matrix.flatten()
        individual = Individual(genome=genome)
        return individual

    def matrix_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Specialized crossover for coupling matrices"""
        N = parent1.shape[0]

        # Block crossover: exchange submatrices
        split_row = np.random.randint(1, N)
        split_col = np.random.randint(1, N)

        child1 = parent1.copy()
        child2 = parent2.copy()

        # Exchange blocks
        child1[split_row:, split_col:] = parent2[split_row:, split_col:]
        child2[split_row:, split_col:] = parent1[split_row:, split_col:]

        return child1, child2

    def matrix_mutation(self, matrix: np.ndarray, mutation_strength: float = 0.1) -> np.ndarray:
        """Specialized mutation for coupling matrices"""
        N = matrix.shape[0]
        mutated = matrix.copy()

        # Random element mutations
        mutation_mask = np.random.random((N, N)) < 0.1
        noise = np.random.normal(0, mutation_strength, (N, N))
        mutated[mutation_mask] += noise[mutation_mask]

        # Structural mutations (add/remove connections)
        if np.random.random() < 0.05:  # 5% chance of structural mutation
            i, j = np.random.randint(0, N, 2)
            if i != j:
                if np.abs(mutated[i, j]) < 1e-6:  # Add connection
                    mutated[i, j] = np.random.normal(0, 1)
                else:  # Remove connection
                    mutated[i, j] = 0

        return mutated


class SynchronizationOptimizer:
    """High-level synchronization optimizer using multiple strategies"""

    def __init__(self, N: int, strategies: List[str] = None):
        """
        Initialize synchronization optimizer.

        Args:
            N: Number of oscillators
            strategies: List of optimization strategies to use
        """
        self.N = N

        if strategies is None:
            strategies = ["genetic", "adaptive", "hybrid"]
        self.strategies = strategies

        # Initialize components
        self.adaptive_coupling = AdaptiveCoupling(N=N)
        self.optimization_results = {}

        logger.info(f"Initialized SynchronizationOptimizer with strategies: {strategies}")

    def optimize_synchronization(
        self, target_metrics: Dict[SynchronizationMetric, float], time_budget: float = 300.0
    ) -> Dict[str, Any]:
        """
        Optimize synchronization using multiple strategies.

        Args:
            target_metrics: Target synchronization performance
            time_budget: Total optimization time budget

        Returns:
            Comprehensive optimization results
        """
        start_time = time.time()

        results = {
            "strategies_used": self.strategies,
            "target_metrics": target_metrics,
            "strategy_results": {},
            "best_coupling": None,
            "best_performance": {},
            "optimization_summary": {},
        }

        time_per_strategy = time_budget / len(self.strategies)

        for strategy in self.strategies:
            strategy_start = time.time()

            logger.info(f"Running {strategy} optimization strategy...")

            if strategy == "genetic":
                strategy_result = self._run_genetic_optimization(time_per_strategy)
            elif strategy == "adaptive":
                strategy_result = self._run_adaptive_optimization(time_per_strategy)
            elif strategy == "hybrid":
                strategy_result = self._run_hybrid_optimization(time_per_strategy)
            else:
                logger.warning(f"Unknown strategy: {strategy}")
                continue

            strategy_time = time.time() - strategy_start
            strategy_result["execution_time"] = strategy_time

            results["strategy_results"][strategy] = strategy_result

            # Update best coupling if this strategy performed better
            if self._is_better_performance(strategy_result, results["best_performance"]):
                results["best_coupling"] = strategy_result.get("coupling_matrix")
                results["best_performance"] = strategy_result.get("performance_metrics", {})

        total_time = time.time() - start_time

        results["optimization_summary"] = {
            "total_execution_time": total_time,
            "strategies_completed": len(results["strategy_results"]),
            "best_strategy": self._identify_best_strategy(results["strategy_results"]),
            "performance_comparison": self._compare_strategy_performance(
                results["strategy_results"]
            ),
        }

        logger.info(f"Multi-strategy optimization completed in {total_time:.2f}s")

        return results

    def _run_genetic_optimization(self, time_budget: float) -> Dict[str, Any]:
        """Run genetic algorithm optimization"""
        result = self.adaptive_coupling.evolve_coupling_matrix()

        return {
            "method": "genetic_algorithm",
            "coupling_matrix": self.adaptive_coupling.get_current_coupling(),
            "performance_metrics": result["synchronization_analysis"],
            "evolution_statistics": result["optimization_statistics"],
        }

    def _run_adaptive_optimization(self, time_budget: float) -> Dict[str, Any]:
        """Run adaptive real-time optimization"""
        # Simulate real-time adaptation
        dt = 0.01
        steps = int(time_budget / dt)

        performance_history = []

        for step in range(steps):
            # Generate test phases
            test_phases = np.random.uniform(0, 2 * np.pi, self.N)

            # Compute performance metrics
            sync_level = np.abs(np.mean(np.exp(1j * test_phases)))
            performance_metrics = {"synchronization_level": sync_level}

            # Update coupling
            self.adaptive_coupling.update_real_time_adaptation(test_phases, performance_metrics, dt)

            performance_history.append(performance_metrics)

            if step % 1000 == 0:  # Periodic check
                logger.debug(f"Adaptive step {step}/{steps}, sync_level: {sync_level:.4f}")

        return {
            "method": "adaptive_real_time",
            "coupling_matrix": self.adaptive_coupling.get_current_coupling(),
            "performance_metrics": performance_history[-1] if performance_history else {},
            "adaptation_history": self.adaptive_coupling.get_adaptation_history(),
            "performance_evolution": performance_history,
        }

    def _run_hybrid_optimization(self, time_budget: float) -> Dict[str, Any]:
        """Run hybrid optimization combining genetic and adaptive approaches"""
        # First phase: genetic optimization (70% of time)
        genetic_time = 0.7 * time_budget
        genetic_result = self._run_genetic_optimization(genetic_time)

        # Second phase: adaptive refinement (30% of time)
        adaptive_time = 0.3 * time_budget
        adaptive_result = self._run_adaptive_optimization(adaptive_time)

        return {
            "method": "hybrid_genetic_adaptive",
            "coupling_matrix": self.adaptive_coupling.get_current_coupling(),
            "performance_metrics": adaptive_result["performance_metrics"],
            "genetic_phase": genetic_result,
            "adaptive_phase": adaptive_result,
            "hybrid_statistics": {
                "genetic_time_fraction": 0.7,
                "adaptive_time_fraction": 0.3,
                "total_phases": 2,
            },
        }

    def _is_better_performance(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> bool:
        """Compare performance of two optimization results"""
        if not result2:  # First result
            return True

        perf1 = result1.get("performance_metrics", {})
        perf2 = result2.get("performance_metrics", {})

        # Compare synchronization level (primary metric)
        sync1 = perf1.get("order_parameter", perf1.get("synchronization_level", 0))
        sync2 = perf2.get("order_parameter", perf2.get("synchronization_level", 0))

        return sync1 > sync2

    def _identify_best_strategy(self, strategy_results: Dict[str, Dict[str, Any]]) -> str:
        """Identify the best performing strategy"""
        best_strategy = None
        best_performance = -1

        for strategy, result in strategy_results.items():
            perf = result.get("performance_metrics", {})
            sync_level = perf.get("order_parameter", perf.get("synchronization_level", 0))

            if sync_level > best_performance:
                best_performance = sync_level
                best_strategy = strategy

        return best_strategy or "unknown"

    def _compare_strategy_performance(
        self, strategy_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare performance across strategies"""
        comparison = {}

        for strategy, result in strategy_results.items():
            perf = result.get("performance_metrics", {})
            comparison[strategy] = {
                "sync_level": perf.get("order_parameter", perf.get("synchronization_level", 0)),
                "phase_coherence": perf.get("phase_coherence", 0),
                "execution_time": result.get("execution_time", 0),
            }

        return comparison


class DynamicTopologyEvolution:
    """Evolution of dynamic network topologies for adaptive synchronization"""

    def __init__(self, N: int, evolution_rules: List[str] = None):
        """Initialize dynamic topology evolution"""
        self.N = N

        if evolution_rules is None:
            evolution_rules = ["growth", "pruning", "rewiring", "clustering"]
        self.evolution_rules = evolution_rules

        logger.info(f"Initialized DynamicTopologyEvolution with rules: {evolution_rules}")

    def evolve_topology(
        self,
        initial_coupling: CouplingMatrix,
        synchronization_target: float = 0.8,
        evolution_steps: int = 1000,
    ) -> Dict[str, Any]:
        """
        Evolve network topology dynamically.

        Args:
            initial_coupling: Starting coupling matrix
            synchronization_target: Target synchronization level
            evolution_steps: Number of evolution steps

        Returns:
            Evolution results and final topology
        """
        current_coupling = initial_coupling.copy()
        evolution_history = []

        for step in range(evolution_steps):
            # Evaluate current performance
            performance = self._evaluate_synchronization_performance(current_coupling)

            # Apply evolution rules based on performance
            if performance["sync_level"] < synchronization_target:
                # Need to improve synchronization
                current_coupling = self._apply_improvement_rules(current_coupling)
            else:
                # Maintain or optimize further
                current_coupling = self._apply_maintenance_rules(current_coupling)

            # Record evolution step
            if step % 100 == 0:
                evolution_history.append(
                    {
                        "step": step,
                        "performance": performance,
                        "network_properties": current_coupling.get_network_properties(),
                        "coupling_statistics": current_coupling.get_coupling_statistics(),
                    }
                )

        return {
            "final_coupling": current_coupling,
            "evolution_history": evolution_history,
            "final_performance": self._evaluate_synchronization_performance(current_coupling),
        }

    def _evaluate_synchronization_performance(self, coupling: CouplingMatrix) -> Dict[str, float]:
        """Evaluate synchronization performance of coupling matrix"""
        # Simplified synchronization simulation
        phases = np.random.uniform(0, 2 * np.pi, self.N)
        dt = 0.01

        sync_levels = []
        for _ in range(100):  # Short simulation
            phase_diffs = phases[:, np.newaxis] - phases[np.newaxis, :]
            coupling_forces = coupling.matrix * np.sin(phase_diffs)
            dphases_dt = np.sum(coupling_forces, axis=1)
            phases += dphases_dt * dt
            phases = np.mod(phases, 2 * np.pi)

            sync_level = np.abs(np.mean(np.exp(1j * phases)))
            sync_levels.append(sync_level)

        return {
            "sync_level": np.mean(sync_levels[-20:]),  # Average over final steps
            "sync_stability": 1.0 / (1.0 + np.var(sync_levels[-20:])),
            "final_phases": phases,
        }

    def _apply_improvement_rules(self, coupling: CouplingMatrix) -> CouplingMatrix:
        """Apply rules to improve synchronization"""
        new_coupling = coupling.copy()

        # Growth: add new connections
        if "growth" in self.evolution_rules:
            new_coupling = self._apply_growth_rule(new_coupling)

        # Clustering: strengthen local connections
        if "clustering" in self.evolution_rules:
            new_coupling = self._apply_clustering_rule(new_coupling)

        return new_coupling

    def _apply_maintenance_rules(self, coupling: CouplingMatrix) -> CouplingMatrix:
        """Apply rules to maintain/optimize current performance"""
        new_coupling = coupling.copy()

        # Pruning: remove weak connections
        if "pruning" in self.evolution_rules:
            new_coupling = self._apply_pruning_rule(new_coupling)

        # Rewiring: reconnect for efficiency
        if "rewiring" in self.evolution_rules:
            new_coupling = self._apply_rewiring_rule(new_coupling)

        return new_coupling

    def _apply_growth_rule(self, coupling: CouplingMatrix) -> CouplingMatrix:
        """Add new connections to improve synchronization"""
        new_coupling = coupling.copy()

        # Find disconnected pairs and add weak connections
        for i in range(self.N):
            for j in range(self.N):
                if i != j and abs(new_coupling.matrix[i, j]) < 1e-6:
                    if np.random.random() < 0.01:  # 1% chance to add connection
                        new_coupling.matrix[i, j] = np.random.normal(0, 0.5)

        return new_coupling

    def _apply_pruning_rule(self, coupling: CouplingMatrix) -> CouplingMatrix:
        """Remove weak connections to simplify network"""
        new_coupling = coupling.copy()

        # Remove connections below threshold
        threshold = np.percentile(np.abs(new_coupling.matrix[new_coupling.matrix != 0]), 10)

        mask = np.abs(new_coupling.matrix) < threshold
        new_coupling.matrix[mask] = 0

        return new_coupling

    def _apply_rewiring_rule(self, coupling: CouplingMatrix) -> CouplingMatrix:
        """Rewire connections for better synchronization"""
        new_coupling = coupling.copy()

        # Randomly rewire a small fraction of connections
        for _ in range(int(0.05 * self.N * self.N)):  # 5% of possible connections
            i, j = np.random.randint(0, self.N, 2)
            k, l = np.random.randint(0, self.N, 2)

            if i != j and k != l and i != k:
                # Swap connections
                old_ij = new_coupling.matrix[i, j]
                new_coupling.matrix[i, j] = new_coupling.matrix[k, l]
                new_coupling.matrix[k, l] = old_ij

        return new_coupling

    def _apply_clustering_rule(self, coupling: CouplingMatrix) -> CouplingMatrix:
        """Strengthen connections within local clusters"""
        new_coupling = coupling.copy()

        # Identify clusters based on current connectivity
        G = nx.from_numpy_array(np.abs(new_coupling.matrix))

        try:
            communities = nx.algorithms.community.greedy_modularity_communities(G)

            # Strengthen intra-cluster connections
            for community in communities:
                nodes = list(community)
                for i in nodes:
                    for j in nodes:
                        if i != j:
                            new_coupling.matrix[i, j] *= 1.1  # Strengthen by 10%

        except:
            # Fallback: strengthen nearest neighbor connections
            for i in range(self.N):
                neighbors = np.argsort(np.abs(new_coupling.matrix[i, :]))[-3:]  # Top 3 connections
                for j in neighbors:
                    if i != j:
                        new_coupling.matrix[i, j] *= 1.05

        return new_coupling


class KuramotoEvolution:
    """Comprehensive Kuramoto network evolution framework"""

    def __init__(self, N: int, config: Dict[str, Any] = None):
        """Initialize comprehensive Kuramoto evolution"""
        self.N = N

        if config is None:
            config = {
                "enable_adaptive_coupling": True,
                "enable_topology_evolution": True,
                "enable_frequency_adaptation": True,
                "enable_multi_objective": True,
            }
        self.config = config

        # Initialize components
        if config["enable_adaptive_coupling"]:
            self.adaptive_coupling = AdaptiveCoupling(N=N)

        if config["enable_topology_evolution"]:
            self.topology_evolution = DynamicTopologyEvolution(N=N)

        self.synchronization_optimizer = SynchronizationOptimizer(N=N)

        logger.info(f"Initialized comprehensive KuramotoEvolution for {N} oscillators")

    def comprehensive_evolution(self, evolution_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run comprehensive Kuramoto network evolution.

        Args:
            evolution_config: Configuration for evolution process

        Returns:
            Comprehensive evolution results
        """
        if evolution_config is None:
            evolution_config = {
                "total_time_budget": 600.0,  # 10 minutes
                "coupling_evolution_fraction": 0.4,
                "topology_evolution_fraction": 0.3,
                "optimization_fraction": 0.3,
            }

        results = {
            "evolution_config": evolution_config,
            "component_results": {},
            "final_network": {},
            "performance_analysis": {},
            "evolution_summary": {},
        }

        total_budget = evolution_config["total_time_budget"]

        # Phase 1: Coupling matrix evolution
        if self.config["enable_adaptive_coupling"]:
            coupling_budget = total_budget * evolution_config["coupling_evolution_fraction"]
            logger.info(f"Phase 1: Coupling evolution ({coupling_budget:.1f}s)")

            coupling_result = self.adaptive_coupling.evolve_coupling_matrix()
            results["component_results"]["coupling_evolution"] = coupling_result

        # Phase 2: Topology evolution
        if self.config["enable_topology_evolution"]:
            topology_budget = total_budget * evolution_config["topology_evolution_fraction"]
            logger.info(f"Phase 2: Topology evolution ({topology_budget:.1f}s)")

            current_coupling = self.adaptive_coupling.get_current_coupling()
            topology_result = self.topology_evolution.evolve_topology(
                initial_coupling=current_coupling,
                evolution_steps=int(topology_budget * 10),  # 10 steps per second
            )
            results["component_results"]["topology_evolution"] = topology_result

            # Update coupling with evolved topology
            self.adaptive_coupling.current_coupling = topology_result["final_coupling"]

        # Phase 3: Multi-strategy optimization
        optimization_budget = total_budget * evolution_config["optimization_fraction"]
        logger.info(f"Phase 3: Multi-strategy optimization ({optimization_budget:.1f}s)")

        target_metrics = {
            SynchronizationMetric.ORDER_PARAMETER: 0.8,
            SynchronizationMetric.PHASE_COHERENCE: 0.7,
        }

        optimization_result = self.synchronization_optimizer.optimize_synchronization(
            target_metrics=target_metrics, time_budget=optimization_budget
        )
        results["component_results"]["optimization"] = optimization_result

        # Compile final results
        final_coupling = self.adaptive_coupling.get_current_coupling()
        results["final_network"] = {
            "coupling_matrix": final_coupling,
            "network_properties": final_coupling.get_network_properties(),
            "coupling_statistics": final_coupling.get_coupling_statistics(),
        }

        # Performance analysis
        results["performance_analysis"] = self._comprehensive_performance_analysis(final_coupling)

        # Evolution summary
        results["evolution_summary"] = {
            "total_execution_time": sum(
                result.get("execution_time", 0) for result in results["component_results"].values()
            ),
            "components_used": list(results["component_results"].keys()),
            "final_synchronization_level": results["performance_analysis"]["sync_performance"][
                "sync_level"
            ],
            "optimization_success": results["performance_analysis"]["sync_performance"][
                "sync_level"
            ]
            >= 0.7,
        }

        logger.info("Comprehensive Kuramoto evolution completed")
        logger.info(
            f"Final sync level: {results['evolution_summary']['final_synchronization_level']:.4f}"
        )

        return results

    def _comprehensive_performance_analysis(self, coupling: CouplingMatrix) -> Dict[str, Any]:
        """Perform comprehensive performance analysis of final network"""
        # Synchronization performance
        sync_performance = self._evaluate_detailed_synchronization(coupling)

        # Network efficiency metrics
        network_efficiency = self._evaluate_network_efficiency(coupling)

        # Robustness analysis
        robustness_analysis = self._evaluate_network_robustness(coupling)

        return {
            "sync_performance": sync_performance,
            "network_efficiency": network_efficiency,
            "robustness_analysis": robustness_analysis,
            "overall_score": self._compute_overall_performance_score(
                sync_performance, network_efficiency, robustness_analysis
            ),
        }

    def _evaluate_detailed_synchronization(self, coupling: CouplingMatrix) -> Dict[str, float]:
        """Detailed synchronization performance evaluation"""
        # Extended simulation for thorough evaluation
        phases = np.random.uniform(0, 2 * np.pi, self.N)
        natural_freqs = np.random.normal(0, 1, self.N)

        dt = 0.01
        sync_levels = []
        coherence_values = []

        for _ in range(500):  # Extended simulation
            phase_diffs = phases[:, np.newaxis] - phases[np.newaxis, :]
            coupling_forces = coupling.matrix * np.sin(phase_diffs)
            dphases_dt = natural_freqs + np.sum(coupling_forces, axis=1)
            phases += dphases_dt * dt
            phases = np.mod(phases, 2 * np.pi)

            # Compute metrics
            sync_level = np.abs(np.mean(np.exp(1j * phases)))
            coherence = np.mean(np.cos(phase_diffs))

            sync_levels.append(sync_level)
            coherence_values.append(coherence)

        return {
            "sync_level": np.mean(sync_levels[-100:]),  # Steady state average
            "sync_stability": 1.0 / (1.0 + np.var(sync_levels[-100:])),
            "phase_coherence": np.mean(coherence_values[-100:]),
            "convergence_time": self._estimate_convergence_time(sync_levels),
            "metastability": np.var(sync_levels),
        }

    def _evaluate_network_efficiency(self, coupling: CouplingMatrix) -> Dict[str, float]:
        """Evaluate network efficiency metrics"""
        G = nx.from_numpy_array(np.abs(coupling.matrix))

        try:
            efficiency_metrics = {
                "global_efficiency": nx.global_efficiency(G),
                "local_efficiency": nx.local_efficiency(G),
                "edge_density": nx.density(G),
                "average_clustering": nx.average_clustering(G) if not nx.is_directed(G) else 0.0,
            }

            # Communication efficiency
            if nx.is_connected(G):
                efficiency_metrics["communication_efficiency"] = (
                    1.0 / nx.average_shortest_path_length(G)
                )
            else:
                efficiency_metrics["communication_efficiency"] = 0.0

        except:
            efficiency_metrics = {
                "global_efficiency": 0.0,
                "local_efficiency": 0.0,
                "edge_density": 0.0,
                "average_clustering": 0.0,
                "communication_efficiency": 0.0,
            }

        return efficiency_metrics

    def _evaluate_network_robustness(self, coupling: CouplingMatrix) -> Dict[str, float]:
        """Evaluate network robustness to perturbations"""
        # Test robustness to random node removal
        G = nx.from_numpy_array(np.abs(coupling.matrix))
        original_connectivity = 1.0 if nx.is_connected(G) else 0.0

        # Random node removal test
        nodes_to_remove = random.sample(list(G.nodes()), min(5, self.N // 4))
        G_perturbed = G.copy()
        G_perturbed.remove_nodes_from(nodes_to_remove)

        robustness_to_removal = 1.0 if nx.is_connected(G_perturbed) else 0.0

        # Edge perturbation test
        edge_robustness = self._test_edge_perturbation_robustness(coupling)

        return {
            "original_connectivity": original_connectivity,
            "robustness_to_node_removal": robustness_to_removal,
            "robustness_to_edge_perturbation": edge_robustness,
            "structural_robustness": (
                original_connectivity + robustness_to_removal + edge_robustness
            )
            / 3,
        }

    def _test_edge_perturbation_robustness(self, coupling: CouplingMatrix) -> float:
        """Test robustness to edge weight perturbations"""
        original_performance = self._evaluate_detailed_synchronization(coupling)

        # Add noise to coupling matrix
        noise_strength = 0.1 * np.std(coupling.matrix)
        noise = np.random.normal(0, noise_strength, coupling.matrix.shape)
        perturbed_matrix = coupling.matrix + noise

        perturbed_coupling = CouplingMatrix(
            matrix=perturbed_matrix, N=coupling.N, topology_type=coupling.topology_type
        )

        perturbed_performance = self._evaluate_detailed_synchronization(perturbed_coupling)

        # Robustness is preservation of synchronization performance
        performance_ratio = perturbed_performance["sync_level"] / max(
            original_performance["sync_level"], 1e-6
        )

        return min(1.0, performance_ratio)

    def _estimate_convergence_time(self, sync_levels: List[float]) -> float:
        """Estimate convergence time from synchronization level time series"""
        if len(sync_levels) < 10:
            return float("inf")

        # Find when synchronization level stabilizes
        final_level = np.mean(sync_levels[-10:])
        threshold = 0.95 * final_level

        for i, level in enumerate(sync_levels):
            if level >= threshold:
                return i * 0.01  # Convert to time (dt=0.01)

        return float("inf")  # Didn't converge

    def _compute_overall_performance_score(
        self,
        sync_perf: Dict[str, float],
        network_eff: Dict[str, float],
        robustness: Dict[str, float],
    ) -> float:
        """Compute overall performance score"""
        # Weighted combination of metrics
        sync_weight = 0.5
        efficiency_weight = 0.3
        robustness_weight = 0.2

        sync_score = sync_perf["sync_level"] * sync_perf["sync_stability"]
        efficiency_score = network_eff["global_efficiency"]
        robustness_score = robustness["structural_robustness"]

        overall_score = (
            sync_weight * sync_score
            + efficiency_weight * efficiency_score
            + robustness_weight * robustness_score
        )

        return overall_score
