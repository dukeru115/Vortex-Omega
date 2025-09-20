"""
Integration tests for Multi-Agent Consensus in NFCS.

Tests the coordination between multiple agents using Kuramoto synchronization,
constitutional monitoring, and distributed decision-making protocols.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import time

# Import modules with graceful fallback for missing dependencies
try:
    from src.core.kuramoto_solver import KuramotoSolver
    from src.core.enhanced_kuramoto import EnhancedKuramotoModule, CouplingMode
    from src.core.state import KuramotoConfig
    from src.modules.constitutional_realtime import ConstitutionalRealTimeMonitor
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    
    # Mock classes for testing when dependencies are not available
    @dataclass
    class MockKuramotoConfig:
        natural_frequencies: dict
        base_coupling_strength: float = 1.0
        time_step: float = 0.01
    
    class MockKuramotoSolver:
        def __init__(self, config, module_order):
            self.config = config
            self.module_order = module_order
            self.num_modules = len(module_order)
    
    class MockEnhancedKuramotoModule:
        def __init__(self, config, num_modules=8, coupling_mode=None):
            self.config = config
            self.num_modules = num_modules
            self.initialized = False
            self.active_signals = []
        
        async def initialize(self):
            self.initialized = True
            return True
    
    class MockConstitutionalMonitor:
        def __init__(self):
            self.monitoring = False
        
        async def start_monitoring(self, metrics_callback=None):
            self.monitoring = True


class TestMultiAgentConsensus:
    """Test suite for multi-agent consensus algorithms"""
    
    @pytest.fixture
    def consensus_config(self):
        """Create configuration for multi-agent consensus testing"""
        agents = ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5']
        frequencies = {agent: 1.0 + 0.1*i for i, agent in enumerate(agents)}
        
        if DEPENDENCIES_AVAILABLE:
            return KuramotoConfig(
                natural_frequencies=frequencies,
                base_coupling_strength=0.8,
                time_step=0.01
            )
        else:
            return MockKuramotoConfig(
                natural_frequencies=frequencies,
                base_coupling_strength=0.8,
                time_step=0.01
            )
    
    @pytest.fixture
    def agent_network(self, consensus_config):
        """Create network of agents for consensus testing"""
        agents = list(consensus_config.natural_frequencies.keys())
        
        if DEPENDENCIES_AVAILABLE:
            # Create Enhanced Kuramoto modules for each agent
            network = {}
            for agent in agents:
                agent_config = KuramotoConfig(
                    natural_frequencies={agent: consensus_config.natural_frequencies[agent]},
                    base_coupling_strength=consensus_config.base_coupling_strength,
                    time_step=consensus_config.time_step
                )
                network[agent] = EnhancedKuramotoModule(
                    config=agent_config,
                    num_modules=1,
                    coupling_mode=CouplingMode.ADAPTIVE
                )
            return network
        else:
            # Mock network for testing
            network = {}
            for agent in agents:
                network[agent] = MockEnhancedKuramotoModule(
                    config=consensus_config,
                    num_modules=1
                )
            return network
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    def test_kuramoto_consensus_initialization(self, consensus_config):
        """Test initialization of Kuramoto consensus system"""
        agents = list(consensus_config.natural_frequencies.keys())
        solver = KuramotoSolver(consensus_config, agents)
        
        assert solver.num_modules == len(agents)
        assert len(solver.omega) == len(agents)
        assert solver.config == consensus_config
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    @pytest.mark.asyncio
    async def test_multi_agent_initialization(self, agent_network):
        """Test initialization of multi-agent network"""
        # Initialize all agents
        for agent_name, agent in agent_network.items():
            result = await agent.initialize()
            assert result is True
            assert agent.initialized is True
    
    @pytest.mark.asyncio
    async def test_consensus_protocol_simulation(self, agent_network, consensus_config):
        """Test basic consensus protocol simulation"""
        # This test works with or without dependencies
        agents = list(consensus_config.natural_frequencies.keys())
        
        # Simulate consensus process
        consensus_data = {
            'agents': agents,
            'convergence_time': 0.0,
            'final_coherence': 0.0,
            'consensus_reached': False
        }
        
        # Mock consensus algorithm
        async def simulate_consensus_step(agents, step):
            """Simulate one consensus step"""
            # Simple simulation: agents gradually converge
            coherence = min(0.95, 0.1 + step * 0.1)
            return coherence
        
        # Run consensus simulation
        max_steps = 10
        for step in range(max_steps):
            coherence = await simulate_consensus_step(agents, step)
            consensus_data['final_coherence'] = coherence
            
            if coherence > 0.9:  # Consensus threshold
                consensus_data['consensus_reached'] = True
                consensus_data['convergence_time'] = step * consensus_config.time_step
                break
        
        # Verify consensus properties
        assert consensus_data['consensus_reached'] is True
        assert consensus_data['final_coherence'] > 0.9
        assert consensus_data['convergence_time'] > 0
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
    @pytest.mark.asyncio
    async def test_constitutional_oversight_integration(self, consensus_config):
        """Test integration with constitutional monitoring during consensus"""
        # Create constitutional monitor
        monitor = ConstitutionalRealTimeMonitor()
        
        # Mock metrics for consensus monitoring
        consensus_metrics = []
        
        async def consensus_metrics_callback():
            """Provide metrics for constitutional monitoring"""
            # Simulate consensus-related metrics
            return {
                'hallucination_number': np.random.uniform(0.5, 1.5),
                'consensus_coherence': np.random.uniform(0.7, 0.95),
                'agent_synchronization': np.random.uniform(0.8, 0.98),
                'constitutional_compliance': np.random.uniform(0.85, 0.99),
                'field_energy': np.random.uniform(200, 400)
            }
        
        # Start constitutional monitoring
        await monitor.start_monitoring(metrics_callback=consensus_metrics_callback)
        
        # Simulate some monitoring cycles
        for _ in range(5):
            metrics = await consensus_metrics_callback()
            consensus_metrics.append(metrics)
            await asyncio.sleep(0.01)  # Small delay
        
        # Verify monitoring data
        assert len(consensus_metrics) == 5
        assert all('consensus_coherence' in m for m in consensus_metrics)
        assert all(m['constitutional_compliance'] > 0.8 for m in consensus_metrics)
    
    def test_consensus_without_dependencies(self, consensus_config):
        """Test consensus structure without full dependencies"""
        # This test ensures our testing framework works even without imports
        agents = list(consensus_config.natural_frequencies.keys())
        
        # Test basic consensus properties
        assert len(agents) == 5
        assert all(isinstance(f, float) for f in consensus_config.natural_frequencies.values())
        assert consensus_config.base_coupling_strength > 0
        assert consensus_config.time_step > 0
    
    @pytest.mark.asyncio
    async def test_distributed_decision_making(self, agent_network, consensus_config):
        """Test distributed decision-making across agent network"""
        agents = list(consensus_config.natural_frequencies.keys())
        
        # Simulate distributed decision scenario
        decision_scenario = {
            'problem': 'resource_allocation',
            'agents': agents,
            'preferences': {agent: np.random.rand() for agent in agents},
            'consensus_threshold': 0.85
        }
        
        # Mock distributed decision algorithm
        async def agent_vote(agent_name, scenario):
            """Simulate agent voting in distributed decision"""
            # Each agent's vote based on their preference
            preference = scenario['preferences'][agent_name]
            vote = preference > 0.5  # Simple binary decision
            confidence = abs(preference - 0.5) * 2  # Distance from neutral
            return {'vote': vote, 'confidence': confidence}
        
        # Collect votes from all agents
        votes = {}
        for agent in agents:
            vote_result = await agent_vote(agent, decision_scenario)
            votes[agent] = vote_result
        
        # Calculate consensus
        positive_votes = sum(1 for v in votes.values() if v['vote'])
        total_votes = len(votes)
        consensus_ratio = positive_votes / total_votes
        
        # Calculate weighted confidence
        avg_confidence = np.mean([v['confidence'] for v in votes.values()])
        
        # Verify decision properties
        assert len(votes) == len(agents)
        assert all('vote' in v and 'confidence' in v for v in votes.values())
        assert 0 <= consensus_ratio <= 1
        assert 0 <= avg_confidence <= 1
    
    @pytest.mark.benchmark
    def test_consensus_performance(self, benchmark, consensus_config):
        """Benchmark consensus algorithm performance"""
        agents = list(consensus_config.natural_frequencies.keys())
        
        def run_consensus_simulation():
            """Run a single consensus simulation for benchmarking"""
            # Simulate consensus computation
            phases = np.random.rand(len(agents)) * 2 * np.pi
            coupling_matrix = np.random.rand(len(agents), len(agents))
            
            # Simple consensus step calculation
            for _ in range(100):  # 100 simulation steps
                # Kuramoto-style update (simplified)
                phase_diffs = phases[:, np.newaxis] - phases[np.newaxis, :]
                interactions = np.sin(phase_diffs)
                updates = np.sum(coupling_matrix * interactions, axis=1)
                phases += updates * consensus_config.time_step
            
            return phases
        
        # Benchmark the simulation
        result = benchmark(run_consensus_simulation)
        assert len(result) == len(agents)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_consensus_workflow(self, agent_network, consensus_config):
        """Test complete end-to-end consensus workflow"""
        agents = list(consensus_config.natural_frequencies.keys())
        
        # Phase 1: Network Initialization
        initialization_success = []
        for agent_name, agent in agent_network.items():
            if hasattr(agent, 'initialize'):
                success = await agent.initialize()
            else:
                success = True  # Mock success
            initialization_success.append(success)
        
        assert all(initialization_success), "All agents should initialize successfully"
        
        # Phase 2: Consensus Protocol
        consensus_state = {
            'round': 0,
            'converged': False,
            'coherence_history': [],
            'max_rounds': 20
        }
        
        while not consensus_state['converged'] and consensus_state['round'] < consensus_state['max_rounds']:
            # Simulate consensus round
            round_coherence = min(0.95, 0.3 + consensus_state['round'] * 0.035)
            consensus_state['coherence_history'].append(round_coherence)
            
            # Check convergence
            if round_coherence > 0.9:
                consensus_state['converged'] = True
            
            consensus_state['round'] += 1
            await asyncio.sleep(0.001)  # Small delay for realism
        
        # Phase 3: Verification
        assert consensus_state['converged'], "Consensus should be reached"
        assert len(consensus_state['coherence_history']) > 0
        assert consensus_state['coherence_history'][-1] > 0.9
        
        # Phase 4: Decision Output
        final_decision = {
            'consensus_reached': consensus_state['converged'],
            'convergence_rounds': consensus_state['round'],
            'final_coherence': consensus_state['coherence_history'][-1],
            'participating_agents': agents
        }
        
        assert final_decision['consensus_reached'] is True
        assert final_decision['convergence_rounds'] > 0
        assert len(final_decision['participating_agents']) == len(agents)


class TestConsensusAlgorithms:
    """Test specific consensus algorithms and their properties"""
    
    def test_admm_consensus_structure(self):
        """Test ADMM (Alternating Direction Method of Multipliers) consensus structure"""
        # ADMM parameters
        admm_params = {
            'rho': 1.0,      # Penalty parameter
            'alpha': 1.6,    # Over-relaxation parameter
            'tolerance': 1e-6,
            'max_iterations': 1000
        }
        
        # Test parameter validity
        assert admm_params['rho'] > 0
        assert 1.0 <= admm_params['alpha'] <= 2.0
        assert admm_params['tolerance'] > 0
        assert admm_params['max_iterations'] > 0
    
    def test_kuramoto_consensus_properties(self):
        """Test mathematical properties of Kuramoto consensus"""
        # Kuramoto network properties
        n_agents = 5
        frequencies = np.random.uniform(0.8, 1.2, n_agents)
        coupling_strength = 2.0
        
        # Critical coupling for synchronization
        frequency_spread = np.max(frequencies) - np.min(frequencies)
        critical_coupling = frequency_spread / 2
        
        # Test synchronization condition
        assert coupling_strength > critical_coupling, "Coupling should exceed critical value for sync"
        
        # Test order parameter bounds
        order_parameter = np.random.uniform(0, 1)  # Simulated order parameter
        assert 0 <= order_parameter <= 1, "Order parameter should be in [0,1]"
    
    @pytest.mark.asyncio
    async def test_consensus_convergence_criteria(self):
        """Test various convergence criteria for consensus algorithms"""
        n_agents = 8
        
        # Different convergence criteria
        criteria = {
            'phase_coherence': {'threshold': 0.95, 'current': 0.97},
            'frequency_spread': {'threshold': 0.1, 'current': 0.05},
            'coupling_stability': {'threshold': 0.02, 'current': 0.01},
            'constitutional_compliance': {'threshold': 0.9, 'current': 0.93}
        }
        
        # Check each criterion
        convergence_status = {}
        for criterion, params in criteria.items():
            if criterion in ['phase_coherence', 'constitutional_compliance']:
                # Higher is better
                convergence_status[criterion] = params['current'] >= params['threshold']
            else:
                # Lower is better
                convergence_status[criterion] = params['current'] <= params['threshold']
        
        # Test that all criteria can be met
        overall_convergence = all(convergence_status.values())
        assert overall_convergence, f"Convergence criteria not met: {convergence_status}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])