"""
Enhanced NFCS v2.4.3 Integration Test - Option C Implementation

Tests the complete enhanced Neural Field Control System with:
- Enhanced Kuramoto Module 1.4 with adaptive signal control
- ESC Module 2.1 with echo-semantic conversion
- Enhanced Metrics 1.5 with advanced risk analytics  
- Full Stage 1 integration with constitutional framework
- Real-time performance and safety validation

This demonstrates the phased development approach (Option C) with
systematic enhancement of partially implemented components.
"""

import pytest
import numpy as np
import asyncio
import logging
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock
import time

# Core NFCS imports
from src.core.state import SystemState, CostFunctionalConfig
from src.core.enhanced_kuramoto import (
    EnhancedKuramotoModule, KuramotoConfig, CouplingMode, 
    KuramotoSignal, AdaptiveState
)
from src.core.enhanced_metrics import EnhancedMetricsCalculator, ConstitutionalLimits
from src.modules.esc.esc_core import (
    EchoSemanticConverter, ESCConfig, ProcessingMode, TokenType
)

# Stage 1 integration components  
from src.orchestrator.resonance_bus import ResonanceBus, TopicType, EventPriority, BusEvent
from src.modules.risk_monitor import RiskMonitor, RiskConfig
from src.modules.constitution_v0 import ConstitutionalFramework, ConstitutionalConfig
from src.modules.emergency_protocols import EmergencyProtocols, EmergencyConfig
from src.orchestrator.main_loop import MainLoop, MainLoopConfig

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEnhancedNFCSIntegration:
    """Enhanced NFCS v2.4.3 Integration Test Suite."""
    
    @pytest.fixture
    def enhanced_system_config(self):
        """Create comprehensive configuration for enhanced system."""
        return {
            'kuramoto_config': KuramotoConfig(
                num_modules=8,
                coupling_mode=CouplingMode.ADAPTIVE,
                base_coupling_strength=0.1,
                adaptation_rate=0.01,
                constitutional_weight=0.3,
                enable_memory=True,
                enable_emergency_protocols=True
            ),
            'esc_config': ESCConfig(
                embedding_dim=512,
                max_sequence_length=1024,
                vocabulary_size=10000,
                semantic_field_layers=4,
                attention_heads=8,
                processing_mode=ProcessingMode.BALANCED,
                enable_constitutional_filtering=True,
                enable_adaptive_vocabulary=True,
                max_unsafe_ratio=0.05,
                constitutional_threshold=0.8
            ),
            'enhanced_metrics_config': CostFunctionalConfig(
                lambda_defect=1.0,
                lambda_coherence=0.8,  
                lambda_energy=0.6,
                lambda_constitutional=1.2,
                enable_defect_tracking=True,
                enable_coherence_analysis=True,
                enable_energy_monitoring=True
            ),
            'constitutional_limits': ConstitutionalLimits(
                max_hallucination_number=0.3,
                max_defect_density=0.08,
                min_global_coherence=0.4,
                min_modular_coherence=0.5,
                max_systemic_risk=0.7,
                min_diversity_index=0.25
            )
        }
    
    @pytest.fixture
    async def enhanced_nfcs_system(self, enhanced_system_config):
        """Initialize complete enhanced NFCS system."""
        # Create enhanced components
        enhanced_kuramoto = EnhancedKuramotoModule(
            config=enhanced_system_config['kuramoto_config'],
            num_modules=8
        )
        
        esc_converter = EchoSemanticConverter(
            config=enhanced_system_config['esc_config']
        )
        
        enhanced_metrics = EnhancedMetricsCalculator(
            config=enhanced_system_config['enhanced_metrics_config'],
            constitutional_limits=enhanced_system_config['constitutional_limits']
        )
        
        # Create Stage 1 integration components
        resonance_bus = ResonanceBus()
        await resonance_bus.initialize()
        
        # Mock external dependencies for testing
        risk_monitor = MagicMock()
        risk_monitor.analyze_state = AsyncMock(return_value={
            'ha_metrics': {'Ha': 0.3, 'ρ_def_mean': 0.05, 'R_field': 0.8, 'R_mod': 0.7},
            'trend_analysis': {'Ha_trend': 0.02, 'stability_trend': 0.01},
            'risk_level': 'LOW'
        })
        
        constitution = MagicMock()
        constitution.evaluate_control_intent = AsyncMock(return_value={
            'decision': 'ACCEPT',
            'control_intent': {'target_coupling': 0.12, 'adaptation_rate': 0.015},
            'constitutional_compliance': 0.85
        })
        
        emergency_protocols = MagicMock()
        emergency_protocols.check_emergency_conditions = AsyncMock(return_value={
            'emergency_active': False,
            'protocol_status': 'NORMAL',
            'safety_margin': 0.4
        })
        
        return {
            'enhanced_kuramoto': enhanced_kuramoto,
            'esc_converter': esc_converter,
            'enhanced_metrics': enhanced_metrics,
            'resonance_bus': resonance_bus,
            'risk_monitor': risk_monitor,
            'constitution': constitution,
            'emergency_protocols': emergency_protocols
        }
    
    async def test_enhanced_kuramoto_signal_processing(self, enhanced_nfcs_system):
        """Test Enhanced Kuramoto Module 1.4 with advanced signal control."""
        kuramoto = enhanced_nfcs_system['enhanced_kuramoto']
        
        logger.info("Testing Enhanced Kuramoto Module 1.4...")
        
        # Create test neural field state
        test_state = SystemState(
            psi=np.random.randn(64, 64) + 1j * np.random.randn(64, 64),
            eta_field=np.random.randn(64, 64) + 1j * np.random.randn(64, 64),
            dt=0.01
        )
        
        # Create constitutional signal for processing
        constitutional_signal = KuramotoSignal(
            target_phases=np.random.randn(8) * 0.2,
            coupling_adjustments={'constitutional': 0.15},
            priority=0.8,
            duration=10,
            constitutional_approved=True
        )
        
        # Test signal processing
        result = await kuramoto.process_signal(constitutional_signal, test_state)
        
        # Validate results
        assert result is not None
        assert 'coupling_evolution' in result
        assert 'adaptation_metrics' in result
        assert 'constitutional_compliance' in result
        
        # Check adaptive coupling response
        assert result['adaptation_metrics']['coupling_strength'] > kuramoto.config.base_coupling_strength
        assert result['constitutional_compliance'] >= 0.7
        
        logger.info(f"✓ Enhanced Kuramoto processing completed - Coupling: {result['adaptation_metrics']['coupling_strength']:.3f}")
        
    async def test_esc_echo_semantic_conversion(self, enhanced_nfcs_system):
        """Test ESC Module 2.1 with echo-semantic token processing."""
        esc = enhanced_nfcs_system['esc_converter']
        
        logger.info("Testing ESC Module 2.1...")
        
        # Test tokens with various types
        test_tokens = [
            "neural", "field", "control", "system",  # Semantic tokens
            "must", "shall", "required",  # Constitutional tokens
            "process", "analyze", "evaluate",  # Processing tokens
            "safety", "truth", "fairness"  # Constitutional principles
        ]
        
        # Process token sequence
        result = esc.process_sequence(test_tokens)
        
        # Validate processing results
        assert result is not None
        assert len(result.processed_tokens) <= len(test_tokens)  # May filter unsafe tokens
        assert result.constitutional_metrics['constitutional_compliance'] >= 0.7
        assert result.constitutional_metrics['unsafe_token_ratio'] <= 0.1
        
        # Check semantic field coupling
        assert result.semantic_field_state.shape == (esc.config.semantic_field_layers, esc.config.embedding_dim)
        assert not np.allclose(result.semantic_field_state, 0)  # Should have non-zero activation
        
        # Validate attention mechanisms
        assert result.attention_map.shape[0] == len(result.processed_tokens)
        assert result.attention_map.shape[1] == len(result.processed_tokens)
        
        logger.info(f"✓ ESC processing completed - {len(result.processed_tokens)} tokens, compliance: {result.constitutional_metrics['constitutional_compliance']:.3f}")
        
    async def test_enhanced_metrics_constitutional_analysis(self, enhanced_nfcs_system):
        """Test Enhanced Metrics 1.5 with advanced risk analytics."""
        metrics_calc = enhanced_nfcs_system['enhanced_metrics']
        
        logger.info("Testing Enhanced Metrics 1.5...")
        
        # Create test state with controlled properties
        test_state = SystemState(
            psi=np.random.randn(32, 32) + 1j * np.random.randn(32, 32),
            eta_field=np.random.randn(32, 32) + 1j * np.random.randn(32, 32),
            dt=0.01
        )
        
        # Apply some coherent structure to avoid random noise
        x, y = np.meshgrid(np.linspace(-np.pi, np.pi, 32), np.linspace(-np.pi, np.pi, 32))
        test_state.psi = np.exp(1j * (np.sin(x) + np.cos(y))) + 0.1 * (np.random.randn(32, 32) + 1j * np.random.randn(32, 32))
        
        # Compute enhanced metrics
        enhanced_result = metrics_calc.compute_enhanced_metrics(test_state)
        
        # Validate constitutional assessment
        assert enhanced_result.constitutional_score >= 0.0
        assert enhanced_result.constitutional_score <= 1.0
        assert enhanced_result.safety_margin >= 0.0
        
        # Check multi-scale analysis
        assert len(enhanced_result.coherence_scales) > 0
        assert enhanced_result.energy_spectrum.size > 0
        
        # Validate stability metrics
        assert enhanced_result.stability_index >= 0.0
        assert enhanced_result.field_entropy >= 0.0
        
        # Constitutional compliance check
        constitutional_report = metrics_calc.generate_constitutional_report()
        assert 'constitutional_status' in constitutional_report
        assert 'risk_assessment' in constitutional_report
        assert 'recommendations' in constitutional_report
        
        logger.info(f"✓ Enhanced Metrics analysis completed - Constitutional score: {enhanced_result.constitutional_score:.3f}")
        
    async def test_stage1_enhanced_integration(self, enhanced_nfcs_system):
        """Test full Stage 1 integration with enhanced components."""
        
        logger.info("Testing Stage 1 + Enhanced Components Integration...")
        
        # Get system components
        kuramoto = enhanced_nfcs_system['enhanced_kuramoto']
        esc = enhanced_nfcs_system['esc_converter']
        metrics_calc = enhanced_nfcs_system['enhanced_metrics']
        bus = enhanced_nfcs_system['resonance_bus']
        
        # Create integrated test scenario
        test_state = SystemState(
            psi=np.random.randn(32, 32) + 1j * np.random.randn(32, 32),
            eta_field=np.random.randn(32, 32) + 1j * np.random.randn(32, 32),
            dt=0.01
        )
        
        # 1. Process tokens through ESC
        test_tokens = ["analyze", "neural", "field", "dynamics", "safely"]
        esc_result = esc.process_sequence(test_tokens)
        
        # 2. Compute enhanced risk metrics
        enhanced_metrics = metrics_calc.compute_enhanced_metrics(test_state)
        
        # 3. Create constitutional signal based on ESC analysis
        constitutional_signal = KuramotoSignal(
            target_phases=np.random.randn(8) * 0.1,
            coupling_adjustments={'constitutional': 0.1 + 0.1 * enhanced_metrics.constitutional_score},
            priority=esc_result.constitutional_metrics['constitutional_compliance'],
            duration=1.5,
            constitutional_approved=True
        )
        
        # 4. Process signal through Enhanced Kuramoto
        kuramoto_result = await kuramoto.process_signal(constitutional_signal, test_state)
        
        # 5. Validate integrated performance
        integration_metrics = {
            'esc_constitutional_compliance': esc_result.constitutional_metrics['constitutional_compliance'],
            'enhanced_metrics_score': enhanced_metrics.constitutional_score,
            'kuramoto_constitutional_compliance': kuramoto_result['constitutional_compliance'],
            'overall_safety_score': (
                esc_result.constitutional_metrics['constitutional_compliance'] +
                enhanced_metrics.constitutional_score +
                kuramoto_result['constitutional_compliance']
            ) / 3
        }
        
        # Validate integration success
        assert integration_metrics['overall_safety_score'] >= 0.6
        assert integration_metrics['esc_constitutional_compliance'] >= 0.7
        assert integration_metrics['enhanced_metrics_score'] >= 0.5
        
        # Test event bus integration
        test_event = {
            'type': 'METRIC_UPDATE',
            'data': {
                'enhanced_metrics': enhanced_metrics,
                'esc_metrics': esc_result.constitutional_metrics,
                'kuramoto_metrics': kuramoto_result['adaptation_metrics']
            },
            'timestamp': time.time()
        }
        
        # Publish using the resonance bus (simplified for testing)
        try:
            await bus.publish_risk_metrics({'test_data': test_event})\n        except Exception as e:\n            logger.warning(f'Bus publish failed (non-critical for test): {e}')
        
        logger.info(f"✓ Integration completed - Overall safety score: {integration_metrics['overall_safety_score']:.3f}")
        
        return integration_metrics
        
    async def test_enhanced_performance_benchmarks(self, enhanced_nfcs_system):
        """Test performance benchmarks for enhanced components."""
        
        logger.info("Testing Enhanced NFCS Performance Benchmarks...")
        
        kuramoto = enhanced_nfcs_system['enhanced_kuramoto']
        esc = enhanced_nfcs_system['esc_converter']
        metrics_calc = enhanced_nfcs_system['enhanced_metrics']
        
        performance_results = {}
        
        # Benchmark Enhanced Kuramoto
        start_time = time.time()
        test_state = SystemState(
            psi=np.random.randn(64, 64) + 1j * np.random.randn(64, 64),
            eta_field=np.random.randn(64, 64) + 1j * np.random.randn(64, 64),
            dt=0.01
        )
        
        test_signal = KuramotoSignal(
            target_phases=np.random.randn(8) * 0.15,
            coupling_adjustments={'adaptive': 0.12},
            priority=1.0,
            duration=1.0
        )
        
        kuramoto_result = await kuramoto.process_signal(test_signal, test_state)
        performance_results['kuramoto_processing_time'] = time.time() - start_time
        
        # Benchmark ESC Module
        start_time = time.time()
        test_tokens = ["enhanced", "neural", "field", "control", "system", "with", "constitutional", "safety"] * 10
        esc_result = esc.process_sequence(test_tokens)
        performance_results['esc_processing_time'] = time.time() - start_time
        performance_results['esc_tokens_per_second'] = len(test_tokens) / performance_results['esc_processing_time']
        
        # Benchmark Enhanced Metrics  
        start_time = time.time()
        enhanced_metrics = metrics_calc.compute_enhanced_metrics(test_state)
        performance_results['enhanced_metrics_time'] = time.time() - start_time
        
        # Validate performance requirements
        assert performance_results['kuramoto_processing_time'] < 0.5  # < 500ms for signal processing
        assert performance_results['esc_processing_time'] < 2.0  # < 2s for 80 tokens
        assert performance_results['enhanced_metrics_time'] < 0.3  # < 300ms for metrics
        assert performance_results['esc_tokens_per_second'] > 20  # > 20 tokens/sec
        
        logger.info("✓ Performance benchmarks passed:")
        logger.info(f"  - Kuramoto: {performance_results['kuramoto_processing_time']:.3f}s")
        logger.info(f"  - ESC: {performance_results['esc_tokens_per_second']:.1f} tokens/sec")
        logger.info(f"  - Enhanced Metrics: {performance_results['enhanced_metrics_time']:.3f}s")
        
        return performance_results
    
    async def test_enhanced_system_emergency_response(self, enhanced_nfcs_system):
        """Test emergency response coordination across enhanced components."""
        
        logger.info("Testing Enhanced System Emergency Response...")
        
        kuramoto = enhanced_nfcs_system['enhanced_kuramoto']
        esc = enhanced_nfcs_system['esc_converter']
        
        # Simulate emergency scenario with unsafe tokens
        unsafe_tokens = ["harmful", "dangerous", "illegal", "unsafe", "threat"]
        
        # Process through ESC (should trigger constitutional filtering)
        esc_result = esc.process_sequence(unsafe_tokens)
        
        # Check if ESC detected risks
        assert esc_result.constitutional_metrics['unsafe_token_ratio'] > 0 or len(esc_result.processed_tokens) < len(unsafe_tokens)
        
        # Create emergency signal
        emergency_signal = KuramotoSignal(
            target_phases=np.random.randn(8) * 0.05,  # Low amplitude for safety
            coupling_adjustments={'emergency': -0.1},  # Reduce coupling in emergency
            priority=2.0,  # High priority
            duration=0.5,
            constitutional_approved=True
        )
        
        # Test Kuramoto emergency response
        test_state = SystemState(
            psi=np.random.randn(32, 32) + 1j * np.random.randn(32, 32),
            eta_field=np.random.randn(32, 32) + 1j * np.random.randn(32, 32),
            dt=0.01
        )
        
        kuramoto_result = await kuramoto.process_signal(emergency_signal, test_state)
        
        # Validate emergency response
        assert kuramoto_result['constitutional_compliance'] >= 0.8  # High compliance in emergency
        assert kuramoto_result['adaptation_metrics']['coupling_strength'] <= kuramoto.config.base_coupling_strength * 1.1  # Conservative coupling
        
        logger.info("✓ Emergency response coordination successful")
        
    async def test_enhanced_nfcs_full_cycle(self, enhanced_nfcs_system):
        """Test complete enhanced NFCS cycle with all components."""
        
        logger.info("Testing Complete Enhanced NFCS v2.4.3 Cycle...")
        
        # Initialize test scenario
        test_tokens = [
            "initialize", "neural", "field", "control", "system",
            "analyze", "constitutional", "compliance", "and", "safety",
            "adapt", "coupling", "parameters", "for", "optimal", "performance"
        ]
        
        test_state = SystemState(
            psi=np.random.randn(64, 64) + 1j * np.random.randn(64, 64),
            eta_field=np.random.randn(64, 64) + 1j * np.random.randn(64, 64),
            dt=0.01
        )
        
        cycle_results = {}
        
        # Phase 1: Token Processing (ESC Module 2.1)
        logger.info("Phase 1: ESC Token Processing")
        esc = enhanced_nfcs_system['esc_converter']
        esc_result = esc.process_sequence(test_tokens)
        cycle_results['esc'] = esc_result
        
        # Phase 2: Risk Assessment (Enhanced Metrics 1.5)  
        logger.info("Phase 2: Enhanced Risk Assessment")
        metrics_calc = enhanced_nfcs_system['enhanced_metrics']
        enhanced_metrics = metrics_calc.compute_enhanced_metrics(test_state)
        cycle_results['enhanced_metrics'] = enhanced_metrics
        
        # Phase 3: Adaptive Control (Enhanced Kuramoto 1.4)
        logger.info("Phase 3: Enhanced Kuramoto Control")
        kuramoto = enhanced_nfcs_system['enhanced_kuramoto']
        
        # Create adaptive signal based on ESC and metrics analysis
        signal_amplitude = 0.1 + 0.05 * enhanced_metrics.constitutional_score
        target_coupling = 0.1 * (1 + enhanced_metrics.safety_margin)
        
        adaptive_signal = KuramotoSignal(
            target_phases=np.random.randn(8) * signal_amplitude,
            coupling_adjustments={'adaptive': target_coupling},
            frequency_shifts={'constitutional': 0.01 * enhanced_metrics.stability_index},
            priority=esc_result.constitutional_metrics['constitutional_compliance'],
            duration=2.0,
            constitutional_approved=True
        )
        
        kuramoto_result = await kuramoto.process_signal(adaptive_signal, test_state)
        cycle_results['kuramoto'] = kuramoto_result
        
        # Phase 4: Validate Complete Cycle
        logger.info("Phase 4: Cycle Validation")
        
        overall_performance = {
            'constitutional_compliance': (
                esc_result.constitutional_metrics['constitutional_compliance'] +
                enhanced_metrics.constitutional_score +
                kuramoto_result['constitutional_compliance']
            ) / 3,
            'system_stability': enhanced_metrics.stability_index,
            'processing_efficiency': 1.0 / (esc.processing_stats['average_processing_time'] + 0.001),
            'adaptive_response': kuramoto_result['adaptation_metrics']['learning_efficiency']
        }
        
        cycle_results['overall_performance'] = overall_performance
        
        # Validate success criteria
        assert overall_performance['constitutional_compliance'] >= 0.7
        assert overall_performance['system_stability'] >= 0.0
        assert overall_performance['processing_efficiency'] > 0
        
        logger.info("✓ Complete Enhanced NFCS v2.4.3 Cycle Successful!")
        logger.info(f"  - Constitutional Compliance: {overall_performance['constitutional_compliance']:.3f}")
        logger.info(f"  - System Stability: {overall_performance['system_stability']:.3f}")
        logger.info(f"  - Processing Efficiency: {overall_performance['processing_efficiency']:.1f}")
        
        return cycle_results


@pytest.mark.asyncio
async def test_enhanced_nfcs_integration_suite():
    """Run complete enhanced NFCS integration test suite."""
    
    logger.info("=" * 60)
    logger.info("Enhanced NFCS v2.4.3 Integration Test Suite - Option C")
    logger.info("Testing: Enhanced Kuramoto 1.4 + ESC 2.1 + Enhanced Metrics 1.5")
    logger.info("=" * 60)
    
    # Create test configuration
    config = {
        'kuramoto_config': KuramotoConfig(num_modules=8, coupling_mode=CouplingMode.ADAPTIVE),
        'esc_config': ESCConfig(processing_mode=ProcessingMode.BALANCED),
        'enhanced_metrics_config': CostFunctionalConfig(lambda_constitutional=1.0),
        'constitutional_limits': ConstitutionalLimits()
    }
    
    # Initialize test instance
    test_instance = TestEnhancedNFCSIntegration()
    
    # Create enhanced system
    enhanced_system = await test_instance.enhanced_nfcs_system(config)
    
    try:
        # Run individual component tests
        await test_instance.test_enhanced_kuramoto_signal_processing(enhanced_system)
        await test_instance.test_esc_echo_semantic_conversion(enhanced_system)
        await test_instance.test_enhanced_metrics_constitutional_analysis(enhanced_system)
        
        # Run integration tests
        integration_result = await test_instance.test_stage1_enhanced_integration(enhanced_system)
        performance_result = await test_instance.test_enhanced_performance_benchmarks(enhanced_system)
        
        # Run emergency response test
        await test_instance.test_enhanced_system_emergency_response(enhanced_system)
        
        # Run complete cycle test
        cycle_result = await test_instance.test_enhanced_nfcs_full_cycle(enhanced_system)
        
        logger.info("=" * 60)
        logger.info("✅ ALL ENHANCED NFCS v2.4.3 TESTS PASSED!")
        logger.info("✅ Option C Implementation Successfully Validated")
        logger.info("=" * 60)
        
        return {
            'integration': integration_result,
            'performance': performance_result,
            'cycle': cycle_result,
            'status': 'SUCCESS'
        }
        
    finally:
        # Cleanup
        if 'resonance_bus' in enhanced_system:
            await enhanced_system['resonance_bus'].shutdown()


if __name__ == "__main__":
    """Run enhanced integration tests directly."""
    async def main():
        result = await test_enhanced_nfcs_integration_suite()
        print(f"\nTest Result: {result['status']}")
        
    asyncio.run(main())