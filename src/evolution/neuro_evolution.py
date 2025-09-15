"""
Neuro-Evolution - Neural Network Architecture Evolution
====================================================

Implements neural architecture search and weight evolution for NFCS v2.4.3.
Provides evolutionary optimization of neural network topologies, connection weights,
and structural parameters for enhanced neuro-symbolic integration.

Scientific Foundation:
Implements advanced neuro-evolutionary algorithms including:
- Topology and Weight Evolving Artificial Neural Networks (TWEANN)
- Neural Architecture Search (NAS) with evolutionary strategies
- Connection weight optimization using genetic algorithms
- Structural evolution with node and connection mutations
- Hybrid evolution for neuro-symbolic architecture optimization

Mathematical Framework:
- Network representation: G = (V, E, W) where V=nodes, E=connections, W=weights
- Fitness evaluation: F(G) = performance on NFCS integration tasks
- Mutations: add/remove nodes, add/remove connections, weight perturbation
- Crossover: structural crossover preserving network functionality

Integration:
Optimizes neural components within NFCS for symbolic-neural transformations,
enhanced ESC processing, and improved Kuramoto synchronization coupling.

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
import copy
from abc import ABC, abstractmethod
import networkx as nx
from collections import defaultdict

from .genetic_optimizer import Individual, Population, FitnessFunction, GeneticOptimizer

logger = logging.getLogger(__name__)


class ActivationFunction(Enum):
    """Neural activation functions"""

    TANH = "tanh"
    SIGMOID = "sigmoid"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SWISH = "swish"
    LINEAR = "linear"


class NodeType(Enum):
    """Types of neural network nodes"""

    INPUT = "input"
    OUTPUT = "output"
    HIDDEN = "hidden"
    BIAS = "bias"
    RECURRENT = "recurrent"


class MutationType(Enum):
    """Types of structural mutations"""

    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    ADD_CONNECTION = "add_connection"
    REMOVE_CONNECTION = "remove_connection"
    MUTATE_WEIGHT = "mutate_weight"
    MUTATE_ACTIVATION = "mutate_activation"
    MUTATE_BIAS = "mutate_bias"


@dataclass
class NetworkNode:
    """
    Represents a node in the neural network.

    Attributes:
        node_id: Unique identifier for the node
        node_type: Type of node (input, output, hidden, etc.)
        activation: Activation function for the node
        bias: Bias value for the node
        layer: Layer index (for feedforward networks)
        metadata: Additional node metadata
    """

    node_id: int
    node_type: NodeType
    activation: ActivationFunction = ActivationFunction.TANH
    bias: float = 0.0
    layer: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "NetworkNode":
        """Create a deep copy of the node"""
        return NetworkNode(
            node_id=self.node_id,
            node_type=self.node_type,
            activation=self.activation,
            bias=self.bias,
            layer=self.layer,
            metadata=copy.deepcopy(self.metadata),
        )


@dataclass
class NetworkConnection:
    """
    Represents a connection between neural network nodes.

    Attributes:
        from_node: Source node ID
        to_node: Target node ID
        weight: Connection weight
        enabled: Whether connection is active
        innovation_number: Innovation number for NEAT-style evolution
        metadata: Additional connection metadata
    """

    from_node: int
    to_node: int
    weight: float = 0.0
    enabled: bool = True
    innovation_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "NetworkConnection":
        """Create a deep copy of the connection"""
        return NetworkConnection(
            from_node=self.from_node,
            to_node=self.to_node,
            weight=self.weight,
            enabled=self.enabled,
            innovation_number=self.innovation_number,
            metadata=copy.deepcopy(self.metadata),
        )


@dataclass
class NetworkTopology:
    """
    Represents the complete neural network topology.

    Attributes:
        nodes: Dictionary of network nodes
        connections: List of network connections
        input_nodes: List of input node IDs
        output_nodes: List of output node IDs
        fitness: Network fitness score
        species_id: Species ID for speciation
        generation: Generation when network was created
        metadata: Additional topology metadata
    """

    nodes: Dict[int, NetworkNode] = field(default_factory=dict)
    connections: List[NetworkConnection] = field(default_factory=list)
    input_nodes: List[int] = field(default_factory=list)
    output_nodes: List[int] = field(default_factory=list)
    fitness: float = 0.0
    species_id: Optional[int] = None
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived attributes"""
        self._connection_dict = {}
        self._update_connection_dict()

    def _update_connection_dict(self):
        """Update internal connection dictionary for fast lookup"""
        self._connection_dict = {}
        for conn in self.connections:
            if conn.enabled:
                key = (conn.from_node, conn.to_node)
                self._connection_dict[key] = conn

    def add_node(self, node: NetworkNode) -> None:
        """Add a node to the network"""
        self.nodes[node.node_id] = node

    def add_connection(self, connection: NetworkConnection) -> None:
        """Add a connection to the network"""
        self.connections.append(connection)
        self._update_connection_dict()

    def remove_node(self, node_id: int) -> None:
        """Remove a node and all its connections"""
        if node_id in self.nodes:
            del self.nodes[node_id]

        # Remove connections involving this node
        self.connections = [
            conn
            for conn in self.connections
            if conn.from_node != node_id and conn.to_node != node_id
        ]
        self._update_connection_dict()

    def remove_connection(self, from_node: int, to_node: int) -> None:
        """Remove a specific connection"""
        self.connections = [
            conn
            for conn in self.connections
            if not (conn.from_node == from_node and conn.to_node == to_node)
        ]
        self._update_connection_dict()

    def has_connection(self, from_node: int, to_node: int) -> bool:
        """Check if connection exists between nodes"""
        return (from_node, to_node) in self._connection_dict

    def get_connection(self, from_node: int, to_node: int) -> Optional[NetworkConnection]:
        """Get connection between nodes"""
        return self._connection_dict.get((from_node, to_node))

    def compute_layers(self) -> None:
        """Compute layer assignments for feedforward networks"""
        # Build adjacency list
        adj_list = defaultdict(list)
        for conn in self.connections:
            if conn.enabled:
                adj_list[conn.from_node].append(conn.to_node)

        # Topological sort to assign layers
        visited = set()
        layers = {}

        def dfs(node_id, current_layer):
            if node_id in visited:
                return layers.get(node_id, current_layer)

            visited.add(node_id)
            layers[node_id] = current_layer

            max_child_layer = current_layer
            for child in adj_list[node_id]:
                child_layer = dfs(child, current_layer + 1)
                max_child_layer = max(max_child_layer, child_layer)

            return max_child_layer

        # Start from input nodes
        for input_id in self.input_nodes:
            dfs(input_id, 0)

        # Update node layers
        for node_id, layer in layers.items():
            if node_id in self.nodes:
                self.nodes[node_id].layer = layer

    def is_valid(self) -> bool:
        """Check if network topology is valid"""
        # Check for cycles in feedforward networks
        if self._has_cycles():
            return False

        # Check connectivity
        if not self.input_nodes or not self.output_nodes:
            return False

        # Check if outputs are reachable from inputs
        if not self._outputs_reachable():
            return False

        return True

    def _has_cycles(self) -> bool:
        """Check for cycles using DFS"""
        adj_list = defaultdict(list)
        for conn in self.connections:
            if conn.enabled:
                adj_list[conn.from_node].append(conn.to_node)

        visited = set()
        rec_stack = set()

        def has_cycle_util(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle_util(node_id):
                    return True

        return False

    def _outputs_reachable(self) -> bool:
        """Check if all outputs are reachable from inputs"""
        adj_list = defaultdict(list)
        for conn in self.connections:
            if conn.enabled:
                adj_list[conn.from_node].append(conn.to_node)

        # BFS from all input nodes
        reachable = set(self.input_nodes)
        queue = list(self.input_nodes)

        while queue:
            current = queue.pop(0)
            for neighbor in adj_list[current]:
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        # Check if all outputs are reachable
        return all(output_id in reachable for output_id in self.output_nodes)

    def copy(self) -> "NetworkTopology":
        """Create a deep copy of the network topology"""
        new_topology = NetworkTopology(
            input_nodes=self.input_nodes.copy(),
            output_nodes=self.output_nodes.copy(),
            fitness=self.fitness,
            species_id=self.species_id,
            generation=self.generation,
            metadata=copy.deepcopy(self.metadata),
        )

        # Copy nodes
        for node_id, node in self.nodes.items():
            new_topology.add_node(node.copy())

        # Copy connections
        for conn in self.connections:
            new_topology.add_connection(conn.copy())

        return new_topology

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for analysis"""
        graph = nx.DiGraph()

        # Add nodes
        for node_id, node in self.nodes.items():
            graph.add_node(
                node_id,
                type=node.node_type.value,
                activation=node.activation.value,
                bias=node.bias,
                layer=node.layer,
            )

        # Add edges
        for conn in self.connections:
            if conn.enabled:
                graph.add_edge(
                    conn.from_node,
                    conn.to_node,
                    weight=conn.weight,
                    innovation=conn.innovation_number,
                )

        return graph

    def get_complexity_metrics(self) -> Dict[str, float]:
        """Compute network complexity metrics"""
        num_nodes = len(self.nodes)
        num_connections = len([c for c in self.connections if c.enabled])

        # Connection density
        max_connections = num_nodes * (num_nodes - 1)
        density = num_connections / max(1, max_connections)

        # Average degree
        degrees = defaultdict(int)
        for conn in self.connections:
            if conn.enabled:
                degrees[conn.from_node] += 1
                degrees[conn.to_node] += 1
        avg_degree = np.mean(list(degrees.values())) if degrees else 0

        # Depth (maximum layer)
        max_layer = max((node.layer for node in self.nodes.values()), default=0)

        return {
            "num_nodes": num_nodes,
            "num_connections": num_connections,
            "connection_density": density,
            "average_degree": avg_degree,
            "network_depth": max_layer,
            "complexity_score": num_nodes * 0.5 + num_connections * 0.3 + max_layer * 0.2,
        }


class InnovationTracker:
    """Tracks innovation numbers for NEAT-style evolution"""

    def __init__(self):
        self.innovation_counter = 0
        self.innovation_history = {}

    def get_innovation_number(self, from_node: int, to_node: int) -> int:
        """Get innovation number for a connection"""
        key = (from_node, to_node)
        if key not in self.innovation_history:
            self.innovation_history[key] = self.innovation_counter
            self.innovation_counter += 1
        return self.innovation_history[key]

    def reset(self) -> None:
        """Reset innovation tracking"""
        self.innovation_counter = 0
        self.innovation_history.clear()


class NEATSpecies:
    """Species for NEAT-style speciation"""

    def __init__(self, species_id: int, representative: NetworkTopology):
        self.species_id = species_id
        self.representative = representative.copy()
        self.members = []
        self.average_fitness = 0.0
        self.max_fitness = 0.0
        self.generations_without_improvement = 0

    def add_member(self, topology: NetworkTopology) -> None:
        """Add member to species"""
        topology.species_id = self.species_id
        self.members.append(topology)

    def update_fitness_statistics(self) -> None:
        """Update species fitness statistics"""
        if not self.members:
            return

        fitnesses = [member.fitness for member in self.members]
        self.average_fitness = np.mean(fitnesses)

        current_max = np.max(fitnesses)
        if current_max > self.max_fitness:
            self.max_fitness = current_max
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

    def should_eliminate(self, stagnation_threshold: int = 15) -> bool:
        """Check if species should be eliminated due to stagnation"""
        return (
            self.generations_without_improvement >= stagnation_threshold and len(self.members) > 1
        )


class NeuroEvolution:
    """
    Main neuro-evolution engine for NFCS neural architecture optimization.

    Implements TWEANN (Topology and Weight Evolving Artificial Neural Networks)
    with speciation, structural mutations, and weight evolution.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        population_size: int = 150,
        max_generations: int = 300,
        compatibility_threshold: float = 3.0,
        survival_rate: float = 0.2,
        mutation_rates: Dict[MutationType, float] = None,
        activation_functions: List[ActivationFunction] = None,
        enable_speciation: bool = True,
        max_nodes: int = 100,
        max_connections: int = 1000,
    ):
        """
        Initialize neuro-evolution engine.

        Args:
            input_size: Number of input nodes
            output_size: Number of output nodes
            population_size: Size of evolution population
            max_generations: Maximum evolution generations
            compatibility_threshold: Threshold for speciation
            survival_rate: Fraction of population that survives
            mutation_rates: Mutation probabilities for different operations
            activation_functions: Available activation functions
            enable_speciation: Enable NEAT-style speciation
            max_nodes: Maximum nodes per network
            max_connections: Maximum connections per network
        """
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        self.max_generations = max_generations
        self.compatibility_threshold = compatibility_threshold
        self.survival_rate = survival_rate
        self.enable_speciation = enable_speciation
        self.max_nodes = max_nodes
        self.max_connections = max_connections

        # Set default mutation rates
        if mutation_rates is None:
            mutation_rates = {
                MutationType.ADD_NODE: 0.05,
                MutationType.REMOVE_NODE: 0.02,
                MutationType.ADD_CONNECTION: 0.3,
                MutationType.REMOVE_CONNECTION: 0.1,
                MutationType.MUTATE_WEIGHT: 0.8,
                MutationType.MUTATE_ACTIVATION: 0.1,
                MutationType.MUTATE_BIAS: 0.3,
            }
        self.mutation_rates = mutation_rates

        # Set available activation functions
        if activation_functions is None:
            activation_functions = [
                ActivationFunction.TANH,
                ActivationFunction.SIGMOID,
                ActivationFunction.RELU,
                ActivationFunction.ELU,
            ]
        self.activation_functions = activation_functions

        # Evolution state
        self.population = []
        self.species_list = []
        self.generation = 0
        self.innovation_tracker = InnovationTracker()
        self.node_counter = 0
        self.best_topology = None
        self.evolution_history = []

        # Statistics
        self.generation_statistics = []

        logger.info(
            f"Initialized NeuroEvolution: {input_size}→{output_size}, pop={population_size}"
        )

    def initialize_population(self) -> None:
        """Initialize population with minimal topologies"""
        self.population = []
        self.node_counter = 0

        for _ in range(self.population_size):
            topology = self._create_minimal_topology()
            self.population.append(topology)

        logger.info(f"Initialized population with {len(self.population)} minimal topologies")

    def _create_minimal_topology(self) -> NetworkTopology:
        """Create minimal network topology with inputs and outputs"""
        topology = NetworkTopology()

        # Add input nodes
        input_nodes = []
        for i in range(self.input_size):
            node_id = self.node_counter
            self.node_counter += 1

            node = NetworkNode(
                node_id=node_id,
                node_type=NodeType.INPUT,
                activation=ActivationFunction.LINEAR,
                layer=0,
            )
            topology.add_node(node)
            input_nodes.append(node_id)

        topology.input_nodes = input_nodes

        # Add bias node
        bias_id = self.node_counter
        self.node_counter += 1
        bias_node = NetworkNode(
            node_id=bias_id,
            node_type=NodeType.BIAS,
            activation=ActivationFunction.LINEAR,
            bias=1.0,
            layer=0,
        )
        topology.add_node(bias_node)

        # Add output nodes
        output_nodes = []
        for i in range(self.output_size):
            node_id = self.node_counter
            self.node_counter += 1

            node = NetworkNode(
                node_id=node_id,
                node_type=NodeType.OUTPUT,
                activation=random.choice(self.activation_functions),
                layer=1,
            )
            topology.add_node(node)
            output_nodes.append(node_id)

        topology.output_nodes = output_nodes

        # Add random connections from inputs to outputs
        for input_id in input_nodes + [bias_id]:
            for output_id in output_nodes:
                if random.random() < 0.5:  # 50% chance of initial connection
                    connection = NetworkConnection(
                        from_node=input_id,
                        to_node=output_id,
                        weight=random.gauss(0, 1),
                        innovation_number=self.innovation_tracker.get_innovation_number(
                            input_id, output_id
                        ),
                    )
                    topology.add_connection(connection)

        topology.compute_layers()
        return topology

    def evolve(
        self,
        fitness_function: Callable[[NetworkTopology], float],
        target_fitness: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run neuro-evolution optimization.

        Args:
            fitness_function: Function to evaluate network fitness
            target_fitness: Target fitness for early termination

        Returns:
            Dictionary containing evolution results
        """
        start_time = time.time()

        if not self.population:
            self.initialize_population()

        best_fitness_history = []
        avg_fitness_history = []
        species_count_history = []

        for generation in range(self.max_generations):
            self.generation = generation

            # Evaluate population fitness
            self._evaluate_population(fitness_function)

            # Update best topology
            current_best = max(self.population, key=lambda t: t.fitness)
            if self.best_topology is None or current_best.fitness > self.best_topology.fitness:
                self.best_topology = current_best.copy()

            # Record statistics
            fitnesses = [t.fitness for t in self.population]
            best_fitness = np.max(fitnesses)
            avg_fitness = np.mean(fitnesses)

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            # Check for target fitness
            if target_fitness is not None and best_fitness >= target_fitness:
                logger.info(f"Target fitness {target_fitness} reached at generation {generation}")
                break

            # Speciation
            if self.enable_speciation:
                self._perform_speciation()
                species_count_history.append(len(self.species_list))
            else:
                species_count_history.append(1)

            # Generate next generation
            self._generate_next_generation()

            # Log progress
            if generation % 50 == 0:
                logger.info(
                    f"Generation {generation}: "
                    f"Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, "
                    f"Species={len(self.species_list) if self.enable_speciation else 'N/A'}"
                )

        execution_time = time.time() - start_time

        # Compile results
        results = {
            "best_topology": self.best_topology,
            "best_fitness": self.best_topology.fitness if self.best_topology else None,
            "generations_run": self.generation + 1,
            "execution_time": execution_time,
            "best_fitness_history": best_fitness_history,
            "avg_fitness_history": avg_fitness_history,
            "species_count_history": species_count_history,
            "final_population": self.population.copy(),
            "evolution_statistics": self._compute_evolution_statistics(),
            "complexity_evolution": self._analyze_complexity_evolution(),
        }

        logger.info(
            f"Neuro-evolution completed in {execution_time:.2f}s. "
            f"Best fitness: {results['best_fitness']}"
        )

        return results

    def _evaluate_population(self, fitness_function: Callable[[NetworkTopology], float]) -> None:
        """Evaluate fitness for all topologies in population"""
        for topology in self.population:
            try:
                topology.fitness = fitness_function(topology)
            except Exception as e:
                logger.warning(f"Error evaluating topology: {e}")
                topology.fitness = -1e6  # Penalty for invalid topologies

    def _perform_speciation(self) -> None:
        """Perform NEAT-style speciation"""
        # Clear existing species memberships
        for species in self.species_list:
            species.members.clear()

        # Assign topologies to species
        for topology in self.population:
            assigned = False

            for species in self.species_list:
                if (
                    self._calculate_compatibility(topology, species.representative)
                    < self.compatibility_threshold
                ):
                    species.add_member(topology)
                    assigned = True
                    break

            if not assigned:
                # Create new species
                new_species = NEATSpecies(len(self.species_list), topology)
                new_species.add_member(topology)
                self.species_list.append(new_species)

        # Update species statistics
        for species in self.species_list:
            species.update_fitness_statistics()

        # Remove empty or stagnant species
        self.species_list = [
            species
            for species in self.species_list
            if species.members and not species.should_eliminate()
        ]

    def _calculate_compatibility(
        self, topology1: NetworkTopology, topology2: NetworkTopology
    ) -> float:
        """Calculate compatibility distance between two topologies"""
        # Get innovation numbers for connections
        innovations1 = set(
            conn.innovation_number
            for conn in topology1.connections
            if conn.innovation_number is not None and conn.enabled
        )
        innovations2 = set(
            conn.innovation_number
            for conn in topology2.connections
            if conn.innovation_number is not None and conn.enabled
        )

        # Count excess and disjoint genes
        all_innovations = innovations1 | innovations2
        max_innovation1 = max(innovations1) if innovations1 else 0
        max_innovation2 = max(innovations2) if innovations2 else 0
        max_innovation = max(max_innovation1, max_innovation2)

        excess = 0
        disjoint = 0
        matching = 0
        weight_diff = 0.0

        for innovation in all_innovations:
            in1 = innovation in innovations1
            in2 = innovation in innovations2

            if in1 and in2:
                # Matching gene - compare weights
                matching += 1
                conn1 = next(c for c in topology1.connections if c.innovation_number == innovation)
                conn2 = next(c for c in topology2.connections if c.innovation_number == innovation)
                weight_diff += abs(conn1.weight - conn2.weight)
            elif innovation > min(max_innovation1, max_innovation2):
                # Excess gene
                excess += 1
            else:
                # Disjoint gene
                disjoint += 1

        # Normalize by genome size
        N = max(len(topology1.connections), len(topology2.connections), 1)
        avg_weight_diff = weight_diff / max(matching, 1)

        # NEAT compatibility formula with coefficients
        c1, c2, c3 = 1.0, 1.0, 0.4  # Coefficients for excess, disjoint, weight differences
        compatibility = (c1 * excess / N) + (c2 * disjoint / N) + (c3 * avg_weight_diff)

        return compatibility

    def _generate_next_generation(self) -> None:
        """Generate next generation through selection, crossover, and mutation"""
        new_population = []

        if self.enable_speciation and self.species_list:
            # Species-based reproduction
            total_avg_fitness = sum(species.average_fitness for species in self.species_list)

            if total_avg_fitness <= 0:
                total_avg_fitness = 1.0

            for species in self.species_list:
                # Calculate offspring allocation for this species
                species_fitness_ratio = species.average_fitness / total_avg_fitness
                offspring_count = max(1, int(species_fitness_ratio * self.population_size))

                # Sort species members by fitness
                species.members.sort(key=lambda t: t.fitness, reverse=True)

                # Keep elite
                elite_count = max(1, int(self.survival_rate * len(species.members)))
                elites = species.members[:elite_count]

                # Add elites to new population
                for elite in elites:
                    if len(new_population) < self.population_size:
                        new_population.append(elite.copy())

                # Generate offspring
                while (
                    len(new_population) < self.population_size
                    and len(new_population) < offspring_count
                ):
                    if len(elites) >= 2 and random.random() < 0.7:  # Crossover
                        parent1 = random.choice(elites)
                        parent2 = random.choice(elites)
                        child = self._crossover(parent1, parent2)
                    else:  # Mutation only
                        parent = random.choice(elites)
                        child = parent.copy()

                    # Apply mutations
                    self._mutate(child)
                    child.generation = self.generation + 1
                    new_population.append(child)

        else:
            # Non-speciated reproduction
            # Sort population by fitness
            self.population.sort(key=lambda t: t.fitness, reverse=True)

            # Keep elite
            elite_count = max(1, int(self.survival_rate * self.population_size))
            elites = self.population[:elite_count]

            # Add elites
            for elite in elites:
                new_population.append(elite.copy())

            # Generate offspring
            while len(new_population) < self.population_size:
                if len(elites) >= 2 and random.random() < 0.7:  # Crossover
                    parent1 = random.choice(elites)
                    parent2 = random.choice(elites)
                    child = self._crossover(parent1, parent2)
                else:  # Mutation only
                    parent = random.choice(elites)
                    child = parent.copy()

                # Apply mutations
                self._mutate(child)
                child.generation = self.generation + 1
                new_population.append(child)

        # Ensure population size
        self.population = new_population[: self.population_size]

    def _crossover(self, parent1: NetworkTopology, parent2: NetworkTopology) -> NetworkTopology:
        """Perform structural crossover between two parent topologies"""
        # Choose the fitter parent as primary
        if parent1.fitness >= parent2.fitness:
            primary, secondary = parent1, parent2
        else:
            primary, secondary = parent2, parent1

        child = NetworkTopology()

        # Copy all nodes from primary parent
        for node_id, node in primary.nodes.items():
            child.add_node(node.copy())

        child.input_nodes = primary.input_nodes.copy()
        child.output_nodes = primary.output_nodes.copy()

        # Inherit connections
        primary_innovations = {
            conn.innovation_number: conn
            for conn in primary.connections
            if conn.innovation_number is not None
        }
        secondary_innovations = {
            conn.innovation_number: conn
            for conn in secondary.connections
            if conn.innovation_number is not None
        }

        for innovation_num, conn in primary_innovations.items():
            if innovation_num in secondary_innovations:
                # Matching gene - randomly choose from either parent
                if random.random() < 0.5:
                    child_conn = conn.copy()
                else:
                    child_conn = secondary_innovations[innovation_num].copy()
            else:
                # Excess/disjoint gene from fitter parent
                child_conn = conn.copy()

            child.add_connection(child_conn)

        child.compute_layers()
        return child

    def _mutate(self, topology: NetworkTopology) -> None:
        """Apply structural and weight mutations to topology"""
        # Weight mutations
        if random.random() < self.mutation_rates[MutationType.MUTATE_WEIGHT]:
            for conn in topology.connections:
                if random.random() < 0.9:  # 90% of weights get mutated
                    if random.random() < 0.1:  # 10% get completely randomized
                        conn.weight = random.gauss(0, 1)
                    else:  # 90% get perturbed
                        conn.weight += random.gauss(0, 0.1)

        # Bias mutations
        if random.random() < self.mutation_rates[MutationType.MUTATE_BIAS]:
            for node in topology.nodes.values():
                if random.random() < 0.1:
                    node.bias += random.gauss(0, 0.1)

        # Activation function mutations
        if random.random() < self.mutation_rates[MutationType.MUTATE_ACTIVATION]:
            hidden_nodes = [
                node
                for node in topology.nodes.values()
                if node.node_type == NodeType.HIDDEN or node.node_type == NodeType.OUTPUT
            ]
            if hidden_nodes:
                node = random.choice(hidden_nodes)
                node.activation = random.choice(self.activation_functions)

        # Structural mutations
        if random.random() < self.mutation_rates[MutationType.ADD_CONNECTION]:
            self._mutate_add_connection(topology)

        if random.random() < self.mutation_rates[MutationType.ADD_NODE]:
            self._mutate_add_node(topology)

        if random.random() < self.mutation_rates[MutationType.REMOVE_CONNECTION]:
            self._mutate_remove_connection(topology)

        if random.random() < self.mutation_rates[MutationType.REMOVE_NODE]:
            self._mutate_remove_node(topology)

    def _mutate_add_connection(self, topology: NetworkTopology) -> None:
        """Add a new connection between existing nodes"""
        if len(topology.connections) >= self.max_connections:
            return

        # Get potential connection pairs
        node_ids = list(topology.nodes.keys())
        attempts = 100

        for _ in range(attempts):
            from_node = random.choice(node_ids)
            to_node = random.choice(node_ids)

            # Avoid self-connections and existing connections
            if (
                from_node != to_node
                and not topology.has_connection(from_node, to_node)
                and from_node not in topology.output_nodes
            ):  # Outputs can't be sources

                connection = NetworkConnection(
                    from_node=from_node,
                    to_node=to_node,
                    weight=random.gauss(0, 1),
                    innovation_number=self.innovation_tracker.get_innovation_number(
                        from_node, to_node
                    ),
                )
                topology.add_connection(connection)
                topology.compute_layers()

                # Check if network is still valid (no cycles for feedforward)
                if topology.is_valid():
                    break
                else:
                    # Remove invalid connection
                    topology.remove_connection(from_node, to_node)

    def _mutate_add_node(self, topology: NetworkTopology) -> None:
        """Add a new node by splitting an existing connection"""
        if len(topology.nodes) >= self.max_nodes:
            return

        enabled_connections = [c for c in topology.connections if c.enabled]
        if not enabled_connections:
            return

        # Choose connection to split
        connection = random.choice(enabled_connections)

        # Disable old connection
        connection.enabled = False

        # Create new node
        new_node_id = self.node_counter
        self.node_counter += 1

        new_node = NetworkNode(
            node_id=new_node_id,
            node_type=NodeType.HIDDEN,
            activation=random.choice(self.activation_functions),
        )
        topology.add_node(new_node)

        # Create two new connections
        conn1 = NetworkConnection(
            from_node=connection.from_node,
            to_node=new_node_id,
            weight=1.0,  # Keep signal strength
            innovation_number=self.innovation_tracker.get_innovation_number(
                connection.from_node, new_node_id
            ),
        )

        conn2 = NetworkConnection(
            from_node=new_node_id,
            to_node=connection.to_node,
            weight=connection.weight,
            innovation_number=self.innovation_tracker.get_innovation_number(
                new_node_id, connection.to_node
            ),
        )

        topology.add_connection(conn1)
        topology.add_connection(conn2)
        topology.compute_layers()

    def _mutate_remove_connection(self, topology: NetworkTopology) -> None:
        """Remove a random connection"""
        enabled_connections = [c for c in topology.connections if c.enabled]
        if len(enabled_connections) <= len(topology.output_nodes):  # Keep minimum connectivity
            return

        connection = random.choice(enabled_connections)
        connection.enabled = False

        # Check if network is still valid
        if not topology._outputs_reachable():
            # Re-enable if it breaks connectivity
            connection.enabled = True

    def _mutate_remove_node(self, topology: NetworkTopology) -> None:
        """Remove a random hidden node"""
        hidden_nodes = [
            node_id for node_id, node in topology.nodes.items() if node.node_type == NodeType.HIDDEN
        ]

        if not hidden_nodes:
            return

        node_to_remove = random.choice(hidden_nodes)
        topology.remove_node(node_to_remove)

        # Check if network is still valid
        if not topology.is_valid():
            # This should not happen with proper removal, but just in case
            logger.warning("Network became invalid after node removal")

    def _compute_evolution_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive evolution statistics"""
        fitnesses = [t.fitness for t in self.population]
        complexities = [t.get_complexity_metrics()["complexity_score"] for t in self.population]

        return {
            "population_size": len(self.population),
            "generations_run": self.generation + 1,
            "fitness_statistics": {
                "mean": np.mean(fitnesses),
                "std": np.std(fitnesses),
                "min": np.min(fitnesses),
                "max": np.max(fitnesses),
                "median": np.median(fitnesses),
            },
            "complexity_statistics": {
                "mean": np.mean(complexities),
                "std": np.std(complexities),
                "min": np.min(complexities),
                "max": np.max(complexities),
            },
            "species_count": len(self.species_list) if self.enable_speciation else 1,
            "innovation_count": self.innovation_tracker.innovation_counter,
            "mutation_rates": self.mutation_rates,
        }

    def _analyze_complexity_evolution(self) -> Dict[str, Any]:
        """Analyze how network complexity evolved"""
        if not self.best_topology:
            return {}

        best_complexity = self.best_topology.get_complexity_metrics()

        # Analyze population complexity distribution
        population_complexities = [t.get_complexity_metrics() for t in self.population]

        node_counts = [c["num_nodes"] for c in population_complexities]
        connection_counts = [c["num_connections"] for c in population_complexities]
        depths = [c["network_depth"] for c in population_complexities]

        return {
            "best_topology_complexity": best_complexity,
            "population_complexity_distribution": {
                "node_counts": {
                    "mean": np.mean(node_counts),
                    "std": np.std(node_counts),
                    "range": [np.min(node_counts), np.max(node_counts)],
                },
                "connection_counts": {
                    "mean": np.mean(connection_counts),
                    "std": np.std(connection_counts),
                    "range": [np.min(connection_counts), np.max(connection_counts)],
                },
                "network_depths": {
                    "mean": np.mean(depths),
                    "std": np.std(depths),
                    "range": [np.min(depths), np.max(depths)],
                },
            },
        }

    def get_best_topology(self) -> Optional[NetworkTopology]:
        """Get the best evolved topology"""
        return self.best_topology

    def export_topology(self, topology: NetworkTopology, format: str = "json") -> Dict[str, Any]:
        """Export topology in specified format"""
        if format == "json":
            return {
                "nodes": [
                    {
                        "id": node.node_id,
                        "type": node.node_type.value,
                        "activation": node.activation.value,
                        "bias": node.bias,
                        "layer": node.layer,
                    }
                    for node in topology.nodes.values()
                ],
                "connections": [
                    {
                        "from": conn.from_node,
                        "to": conn.to_node,
                        "weight": conn.weight,
                        "enabled": conn.enabled,
                        "innovation": conn.innovation_number,
                    }
                    for conn in topology.connections
                ],
                "input_nodes": topology.input_nodes,
                "output_nodes": topology.output_nodes,
                "fitness": topology.fitness,
                "generation": topology.generation,
                "complexity": topology.get_complexity_metrics(),
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")


class WeightEvolution:
    """Specialized weight evolution for existing network topologies"""

    def __init__(
        self,
        topology: NetworkTopology,
        population_size: int = 100,
        max_generations: int = 200,
        mutation_strength: float = 0.1,
        crossover_rate: float = 0.7,
    ):
        """Initialize weight evolution for a fixed topology"""
        self.topology = topology
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate

        # Extract weight structure
        self.weight_positions = []
        for i, conn in enumerate(topology.connections):
            if conn.enabled:
                self.weight_positions.append((i, "weight"))

        for node_id, node in topology.nodes.items():
            if node.node_type in [NodeType.HIDDEN, NodeType.OUTPUT]:
                self.weight_positions.append((node_id, "bias"))

        self.genome_length = len(self.weight_positions)

        logger.info(f"Initialized WeightEvolution with {self.genome_length} parameters")

    def optimize_weights(
        self, fitness_function: Callable[[NetworkTopology], float]
    ) -> Dict[str, Any]:
        """Optimize weights using genetic algorithm"""

        # Create fitness function for weight genomes
        def weight_fitness(genome: np.ndarray) -> float:
            # Apply weights to topology
            test_topology = self.topology.copy()
            self._apply_weights_to_topology(test_topology, genome)
            return fitness_function(test_topology)

        # Create bounds for weights
        lower_bounds = np.full(self.genome_length, -5.0)
        upper_bounds = np.full(self.genome_length, 5.0)

        # Use genetic optimizer for weight evolution
        class WeightFitnessFunction(FitnessFunction):
            def __init__(self, fitness_func, bounds):
                self.fitness_func = fitness_func
                self.bounds = bounds

            def evaluate(self, genome: np.ndarray) -> float:
                return self.fitness_func(genome)

            def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
                return self.bounds

        weight_fitness_func = WeightFitnessFunction(weight_fitness, (lower_bounds, upper_bounds))

        optimizer = GeneticOptimizer(
            fitness_function=weight_fitness_func,
            population_size=self.population_size,
            genome_length=self.genome_length,
            max_generations=self.max_generations,
            crossover_rate=self.crossover_rate,
            mutation_rate=0.8,  # High mutation rate for weight optimization
        )

        results = optimizer.evolve()

        # Apply best weights to original topology
        if results["best_genome"] is not None:
            self._apply_weights_to_topology(self.topology, results["best_genome"])

        return results

    def _apply_weights_to_topology(self, topology: NetworkTopology, genome: np.ndarray) -> None:
        """Apply weight genome to topology"""
        for i, (pos, param_type) in enumerate(self.weight_positions):
            if param_type == "weight":
                topology.connections[pos].weight = genome[i]
            elif param_type == "bias":
                topology.nodes[pos].bias = genome[i]


class StructuralEvolution:
    """Structural evolution for network architecture search"""

    def __init__(self, input_size: int, output_size: int, layer_constraints: Dict[str, Any] = None):
        """Initialize structural evolution with architectural constraints"""
        self.input_size = input_size
        self.output_size = output_size

        if layer_constraints is None:
            layer_constraints = {
                "min_hidden_layers": 1,
                "max_hidden_layers": 5,
                "min_nodes_per_layer": 2,
                "max_nodes_per_layer": 50,
                "allow_skip_connections": True,
                "allow_recurrent_connections": False,
            }
        self.layer_constraints = layer_constraints

    def generate_random_architecture(self) -> NetworkTopology:
        """Generate random network architecture within constraints"""
        topology = NetworkTopology()
        node_counter = 0

        # Add input layer
        input_nodes = []
        for i in range(self.input_size):
            node = NetworkNode(
                node_id=node_counter,
                node_type=NodeType.INPUT,
                activation=ActivationFunction.LINEAR,
                layer=0,
            )
            topology.add_node(node)
            input_nodes.append(node_counter)
            node_counter += 1

        topology.input_nodes = input_nodes

        # Add hidden layers
        num_hidden_layers = random.randint(
            self.layer_constraints["min_hidden_layers"], self.layer_constraints["max_hidden_layers"]
        )

        hidden_layers = []
        for layer_idx in range(1, num_hidden_layers + 1):
            layer_size = random.randint(
                self.layer_constraints["min_nodes_per_layer"],
                self.layer_constraints["max_nodes_per_layer"],
            )

            layer_nodes = []
            for _ in range(layer_size):
                node = NetworkNode(
                    node_id=node_counter,
                    node_type=NodeType.HIDDEN,
                    activation=random.choice(
                        [ActivationFunction.TANH, ActivationFunction.RELU, ActivationFunction.ELU]
                    ),
                    layer=layer_idx,
                )
                topology.add_node(node)
                layer_nodes.append(node_counter)
                node_counter += 1

            hidden_layers.append(layer_nodes)

        # Add output layer
        output_nodes = []
        output_layer_idx = num_hidden_layers + 1
        for i in range(self.output_size):
            node = NetworkNode(
                node_id=node_counter,
                node_type=NodeType.OUTPUT,
                activation=ActivationFunction.TANH,
                layer=output_layer_idx,
            )
            topology.add_node(node)
            output_nodes.append(node_counter)
            node_counter += 1

        topology.output_nodes = output_nodes

        # Add connections
        all_layers = [input_nodes] + hidden_layers + [output_nodes]

        # Feedforward connections
        for layer_idx in range(len(all_layers) - 1):
            current_layer = all_layers[layer_idx]
            next_layer = all_layers[layer_idx + 1]

            for from_node in current_layer:
                for to_node in next_layer:
                    if random.random() < 0.8:  # 80% connection probability
                        connection = NetworkConnection(
                            from_node=from_node, to_node=to_node, weight=random.gauss(0, 1)
                        )
                        topology.add_connection(connection)

        # Skip connections (if allowed)
        if self.layer_constraints["allow_skip_connections"]:
            for i in range(len(all_layers)):
                for j in range(i + 2, len(all_layers)):  # Skip at least one layer
                    if random.random() < 0.1:  # 10% skip connection probability
                        from_layer = all_layers[i]
                        to_layer = all_layers[j]

                        from_node = random.choice(from_layer)
                        to_node = random.choice(to_layer)

                        if not topology.has_connection(from_node, to_node):
                            connection = NetworkConnection(
                                from_node=from_node, to_node=to_node, weight=random.gauss(0, 0.5)
                            )
                            topology.add_connection(connection)

        return topology


class NeuralArchitectureSearch:
    """
    Neural Architecture Search (NAS) for NFCS integration.

    Combines neuro-evolution with architectural constraints for
    discovering optimal neural network topologies for specific tasks.
    """

    def __init__(
        self,
        search_space: Dict[str, Any],
        evaluation_budget: int = 1000,
        population_size: int = 50,
        tournament_size: int = 3,
    ):
        """
        Initialize NAS engine.

        Args:
            search_space: Definition of architectural search space
            evaluation_budget: Maximum number of architecture evaluations
            population_size: Size of evolution population
            tournament_size: Tournament selection size
        """
        self.search_space = search_space
        self.evaluation_budget = evaluation_budget
        self.population_size = population_size
        self.tournament_size = tournament_size

        # Extract search space constraints
        self.input_size = search_space.get("input_size", 10)
        self.output_size = search_space.get("output_size", 1)
        self.max_depth = search_space.get("max_depth", 8)
        self.max_width = search_space.get("max_width", 100)
        self.allowed_operations = search_space.get("operations", ["conv", "fc", "pool"])

        # NAS state
        self.evaluated_architectures = []
        self.evaluation_count = 0
        self.best_architecture = None
        self.search_history = []

        logger.info(f"Initialized NeuralArchitectureSearch with budget {evaluation_budget}")

    def search(
        self,
        fitness_function: Callable[[Dict[str, Any]], float],
        early_stopping: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform neural architecture search.

        Args:
            fitness_function: Function to evaluate architecture performance
            early_stopping: Early stopping criteria

        Returns:
            Search results including best architecture
        """
        start_time = time.time()

        # Initialize population with random architectures
        population = []
        for _ in range(self.population_size):
            arch = self._generate_random_architecture()
            population.append(arch)

        best_fitness = float("-inf")
        no_improvement_count = 0

        generation = 0
        while self.evaluation_count < self.evaluation_budget:
            generation += 1

            # Evaluate population
            for arch in population:
                if self.evaluation_count >= self.evaluation_budget:
                    break

                fitness = self._evaluate_architecture(arch, fitness_function)
                arch["fitness"] = fitness

                if fitness > best_fitness:
                    best_fitness = fitness
                    self.best_architecture = arch.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

            # Check early stopping
            if early_stopping:
                if (
                    early_stopping.get("patience")
                    and no_improvement_count >= early_stopping["patience"]
                ):
                    logger.info(f"Early stopping triggered at generation {generation}")
                    break

                if (
                    early_stopping.get("target_fitness")
                    and best_fitness >= early_stopping["target_fitness"]
                ):
                    logger.info(f"Target fitness reached at generation {generation}")
                    break

            # Generate next generation
            new_population = []

            # Elitism - keep best architectures
            population.sort(key=lambda x: x.get("fitness", float("-inf")), reverse=True)
            elite_count = max(1, self.population_size // 10)
            new_population.extend(population[:elite_count])

            # Generate offspring through mutation and crossover
            while len(new_population) < self.population_size:
                if random.random() < 0.7:  # Crossover
                    parent1 = self._tournament_selection(population)
                    parent2 = self._tournament_selection(population)
                    child = self._crossover_architectures(parent1, parent2)
                else:  # Mutation only
                    parent = self._tournament_selection(population)
                    child = self._mutate_architecture(parent.copy())

                new_population.append(child)

            population = new_population

            # Log progress
            if generation % 10 == 0:
                avg_fitness = np.mean([arch.get("fitness", 0) for arch in population])
                logger.info(
                    f"Generation {generation}: "
                    f"Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, "
                    f"Evaluations={self.evaluation_count}/{self.evaluation_budget}"
                )

        execution_time = time.time() - start_time

        # Compile results
        results = {
            "best_architecture": self.best_architecture,
            "best_fitness": best_fitness,
            "generations_run": generation,
            "evaluations_used": self.evaluation_count,
            "execution_time": execution_time,
            "search_history": self.search_history,
            "final_population": population,
            "search_statistics": {
                "total_architectures_evaluated": len(self.evaluated_architectures),
                "unique_architectures": len(
                    set(str(arch.get("structure", "")) for arch in self.evaluated_architectures)
                ),
                "average_fitness": np.mean(
                    [arch.get("fitness", 0) for arch in self.evaluated_architectures]
                ),
            },
        }

        logger.info(
            f"NAS completed: {self.evaluation_count} evaluations in {execution_time:.2f}s. "
            f"Best fitness: {best_fitness:.4f}"
        )

        return results

    def _generate_random_architecture(self) -> Dict[str, Any]:
        """Generate random architecture within search space constraints"""
        architecture = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "layers": [],
            "connections": [],
            "metadata": {},
        }

        # Random number of layers
        num_layers = random.randint(2, self.max_depth)

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                # Input layer
                layer = {"type": "input", "size": self.input_size, "activation": "linear"}
            elif layer_idx == num_layers - 1:
                # Output layer
                layer = {
                    "type": "output",
                    "size": self.output_size,
                    "activation": random.choice(["tanh", "sigmoid", "linear"]),
                }
            else:
                # Hidden layer
                layer = {
                    "type": "hidden",
                    "size": random.randint(5, min(self.max_width, 100)),
                    "activation": random.choice(["tanh", "relu", "elu", "sigmoid"]),
                    "operation": random.choice(self.allowed_operations),
                }

            architecture["layers"].append(layer)

        # Add connections (simplified - assume feedforward)
        for i in range(len(architecture["layers"]) - 1):
            connection = {"from_layer": i, "to_layer": i + 1, "connection_type": "full"}
            architecture["connections"].append(connection)

        return architecture

    def _evaluate_architecture(
        self, architecture: Dict[str, Any], fitness_function: Callable[[Dict[str, Any]], float]
    ) -> float:
        """Evaluate architecture using fitness function"""
        self.evaluation_count += 1

        try:
            fitness = fitness_function(architecture)

            # Store evaluation
            eval_record = architecture.copy()
            eval_record["fitness"] = fitness
            eval_record["evaluation_id"] = self.evaluation_count
            self.evaluated_architectures.append(eval_record)

            return fitness

        except Exception as e:
            logger.warning(f"Error evaluating architecture: {e}")
            return -1e6  # Penalty for invalid architectures

    def _tournament_selection(self, population: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select individual using tournament selection"""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.get("fitness", float("-inf")))

    def _crossover_architectures(
        self, parent1: Dict[str, Any], parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform crossover between two architectures"""
        # Simple layer-wise crossover
        child = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "layers": [],
            "connections": [],
            "metadata": {},
        }

        # Choose number of layers
        num_layers = random.choice([len(parent1["layers"]), len(parent2["layers"])])

        for layer_idx in range(num_layers):
            if layer_idx < len(parent1["layers"]) and layer_idx < len(parent2["layers"]):
                # Both parents have this layer - choose randomly
                parent_layer = random.choice(
                    [parent1["layers"][layer_idx], parent2["layers"][layer_idx]]
                )
            elif layer_idx < len(parent1["layers"]):
                parent_layer = parent1["layers"][layer_idx]
            elif layer_idx < len(parent2["layers"]):
                parent_layer = parent2["layers"][layer_idx]
            else:
                # Generate new random layer
                parent_layer = {
                    "type": "hidden",
                    "size": random.randint(5, 50),
                    "activation": random.choice(["tanh", "relu", "elu"]),
                    "operation": random.choice(self.allowed_operations),
                }

            child["layers"].append(parent_layer.copy())

        # Ensure proper input/output layers
        if child["layers"]:
            child["layers"][0]["type"] = "input"
            child["layers"][0]["size"] = self.input_size
            child["layers"][-1]["type"] = "output"
            child["layers"][-1]["size"] = self.output_size

        # Add basic connections
        for i in range(len(child["layers"]) - 1):
            connection = {"from_layer": i, "to_layer": i + 1, "connection_type": "full"}
            child["connections"].append(connection)

        return child

    def _mutate_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mutations to architecture"""
        # Layer size mutation
        for layer in architecture["layers"]:
            if layer["type"] == "hidden" and random.random() < 0.3:
                layer["size"] = max(1, layer["size"] + random.randint(-10, 10))
                layer["size"] = min(layer["size"], self.max_width)

        # Activation function mutation
        for layer in architecture["layers"]:
            if layer["type"] in ["hidden", "output"] and random.random() < 0.2:
                layer["activation"] = random.choice(["tanh", "relu", "elu", "sigmoid"])

        # Add/remove layer mutation
        if random.random() < 0.1 and len(architecture["layers"]) > 3:
            # Remove random hidden layer
            hidden_indices = [
                i for i, layer in enumerate(architecture["layers"]) if layer["type"] == "hidden"
            ]
            if hidden_indices:
                remove_idx = random.choice(hidden_indices)
                del architecture["layers"][remove_idx]

                # Update connections
                architecture["connections"] = [
                    conn
                    for conn in architecture["connections"]
                    if conn["from_layer"] != remove_idx and conn["to_layer"] != remove_idx
                ]

                # Adjust connection indices
                for conn in architecture["connections"]:
                    if conn["from_layer"] > remove_idx:
                        conn["from_layer"] -= 1
                    if conn["to_layer"] > remove_idx:
                        conn["to_layer"] -= 1

        elif random.random() < 0.1 and len(architecture["layers"]) < self.max_depth:
            # Add new hidden layer
            insert_pos = random.randint(1, len(architecture["layers"]) - 1)
            new_layer = {
                "type": "hidden",
                "size": random.randint(5, 50),
                "activation": random.choice(["tanh", "relu", "elu"]),
                "operation": random.choice(self.allowed_operations),
            }
            architecture["layers"].insert(insert_pos, new_layer)

            # Update connections
            for conn in architecture["connections"]:
                if conn["from_layer"] >= insert_pos:
                    conn["from_layer"] += 1
                if conn["to_layer"] >= insert_pos:
                    conn["to_layer"] += 1

            # Add connections for new layer
            if insert_pos > 0:
                architecture["connections"].append(
                    {
                        "from_layer": insert_pos - 1,
                        "to_layer": insert_pos,
                        "connection_type": "full",
                    }
                )
            if insert_pos < len(architecture["layers"]) - 1:
                architecture["connections"].append(
                    {
                        "from_layer": insert_pos,
                        "to_layer": insert_pos + 1,
                        "connection_type": "full",
                    }
                )

        return architecture

    def get_best_architecture(self) -> Optional[Dict[str, Any]]:
        """Get the best discovered architecture"""
        return self.best_architecture

    def get_architecture_diversity(self) -> Dict[str, Any]:
        """Analyze diversity of explored architectures"""
        if not self.evaluated_architectures:
            return {}

        # Analyze layer count distribution
        layer_counts = [len(arch.get("layers", [])) for arch in self.evaluated_architectures]

        # Analyze size distribution
        total_params = []
        for arch in self.evaluated_architectures:
            params = 0
            layers = arch.get("layers", [])
            for i in range(len(layers) - 1):
                params += layers[i]["size"] * layers[i + 1]["size"]
            total_params.append(params)

        return {
            "total_architectures_explored": len(self.evaluated_architectures),
            "layer_count_distribution": {
                "mean": np.mean(layer_counts),
                "std": np.std(layer_counts),
                "range": [np.min(layer_counts), np.max(layer_counts)],
            },
            "parameter_count_distribution": {
                "mean": np.mean(total_params),
                "std": np.std(total_params),
                "range": [np.min(total_params), np.max(total_params)],
            },
            "fitness_distribution": {
                "mean": np.mean([arch.get("fitness", 0) for arch in self.evaluated_architectures]),
                "std": np.std([arch.get("fitness", 0) for arch in self.evaluated_architectures]),
                "range": [
                    np.min([arch.get("fitness", 0) for arch in self.evaluated_architectures]),
                    np.max([arch.get("fitness", 0) for arch in self.evaluated_architectures]),
                ],
            },
        }
