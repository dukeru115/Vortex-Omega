"""
Test Suite for Symbolic-Neural Bridge
=====================================

Comprehensive tests for the Symbolic-Neural Bridge implementation.
Tests S ↔ φ transformations, consistency verification, and integration.

Author: Team Omega (GenSpark AI Implementation)
License: CC BY-NC 4.0
"""

import pytest
import torch
import numpy as np
import asyncio
from typing import List, Dict, Any

# Import components to test
from src.modules.symbolic.neural_bridge import SymbolicNeuralBridge, BasisFunction, SymbolicWeight
from src.modules.symbolic.models import SymClause, ClauseType, VerificationStatus
from src.modules.integration.symbolic_nfcs_integration import SymbolicNFCSIntegration


class TestSymbolicNeuralBridge:
    """Test cases for SymbolicNeuralBridge"""
    
    @pytest.fixture
    def bridge(self):
        """Create bridge instance for testing"""
        return SymbolicNeuralBridge(
            field_dims=(32, 32),
            max_symbols=64,
            config={
                'embedding_dim': 32,
                'learning_rate': 0.01,
                'consistency_threshold': 0.8
            }
        )
    
    @pytest.fixture
    def test_clauses(self) -> List[SymClause]:
        """Create test symbolic clauses"""
        return [
            SymClause(
                cid="test_eq1",
                ctype=ClauseType.EQUATION,
                meta={'domain': 'physics', 'confidence': 0.9}
            ),
            SymClause(
                cid="test_fact1", 
                ctype=ClauseType.FACT,
                meta={'domain': 'physics', 'confidence': 0.8}
            ),
            SymClause(
                cid="test_constraint1",
                ctype=ClauseType.CONSTRAINT,
                meta={'domain': 'math', 'confidence': 0.95}
            )
        ]
    
    def test_initialization(self, bridge):
        """Test proper initialization of bridge"""
        assert bridge.field_dims == (32, 32)
        assert bridge.max_symbols == 64
        assert bridge.spatial_grid.shape == (32, 32, 2)
        assert len(bridge.symbol_registry) == 0
        assert len(bridge.basis_functions) == 0
    
    def test_symbol_registration(self, bridge, test_clauses):
        """Test symbol registration functionality"""
        clause = test_clauses[0]
        
        # Register new symbol
        idx = bridge.register_symbol("test_symbol", clause)
        
        assert idx >= 0
        assert "test_symbol" in bridge.symbol_registry
        assert "test_symbol" in bridge.basis_functions
        assert "test_symbol" in bridge.symbolic_weights
        
        # Register same symbol again - should return existing index
        idx2 = bridge.register_symbol("test_symbol", clause)
        assert idx == idx2
    
    @pytest.mark.asyncio
    async def test_fieldization(self, bridge, test_clauses):
        """Test S → φ transformation (fieldization)"""
        # Fieldize clauses into neural field
        field = await bridge.fieldize(test_clauses)
        
        # Check output properties
        assert field.shape == bridge.field_dims
        assert torch.is_complex(field)
        assert not torch.isnan(field).any()
        assert not torch.isinf(field).any()
        
        # Check that symbols were registered
        assert len(bridge.symbol_registry) == len(test_clauses)
        
        # Check field has non-zero energy
        field_energy = torch.sum(torch.abs(field) ** 2)
        assert field_energy > 0
    
    @pytest.mark.asyncio
    async def test_symbolization(self, bridge):
        """Test φ → S transformation (symbolization)"""
        # Create test field with known structure
        test_field = torch.zeros(32, 32, dtype=torch.complex64)
        
        # Add some structured patterns
        test_field[10:20, 10:20] = 1.0 + 0.5j
        test_field[5:15, 20:30] = 0.8 * torch.exp(1j * torch.pi/4)
        
        # Extract symbols
        symbols = await bridge.symbolize(test_field)
        
        # Check results
        assert len(symbols) > 0
        assert all(isinstance(s, SymClause) for s in symbols)
        
        # Check that symbols contain field information
        for symbol in symbols:
            assert 'amplitude' in symbol.meta
            assert 'phase' in symbol.meta
            assert 'wave_vector' in symbol.meta
    
    @pytest.mark.asyncio
    async def test_consistency_verification(self, bridge, test_clauses):
        """Test S ↔ φ consistency verification"""
        # First fieldization
        field1 = await bridge.fieldize(test_clauses)
        
        # Verify consistency
        consistency_result = await bridge.verify_consistency(test_clauses, field1)
        
        # Check results structure
        assert 'consistency_score' in consistency_result
        assert 'field_mse' in consistency_result
        assert 'symbol_similarity' in consistency_result
        
        # Consistency score should be reasonable
        assert 0 <= consistency_result['consistency_score'] <= 1
        
        # Field MSE should be finite
        assert not np.isnan(consistency_result['field_mse'])
        assert not np.isinf(consistency_result['field_mse'])
    
    @pytest.mark.asyncio
    async def test_round_trip_transformation(self, bridge, test_clauses):
        """Test complete S → φ → S round trip"""
        # Forward: S → φ
        field = await bridge.fieldize(test_clauses)
        
        # Backward: φ → S  
        extracted_symbols = await bridge.symbolize(field)
        
        # Verify we got some symbols back
        assert len(extracted_symbols) > 0
        
        # Forward again: S → φ
        field2 = await bridge.fieldize(extracted_symbols)
        
        # Compare fields (should be similar)
        field_diff = torch.mean(torch.abs(field - field2) ** 2)
        assert field_diff < 1.0  # Reasonable similarity threshold
    
    def test_basis_function_creation(self, bridge, test_clauses):
        """Test basis function creation for different clause types"""
        for clause in test_clauses:
            symbol = f"test_{clause.ctype.value}"
            basis = bridge._create_basis_function(symbol, clause)
            
            assert isinstance(basis, BasisFunction)
            assert basis.symbol == symbol
            assert callable(basis.spatial_pattern)
            assert basis.frequency > 0
            assert basis.amplitude > 0
    
    def test_metrics_tracking(self, bridge):
        """Test metrics collection and updates"""
        initial_metrics = bridge.get_metrics()
        
        # Check initial state
        assert initial_metrics['symbolization_time'] >= 0
        assert initial_metrics['fieldization_time'] >= 0
        assert initial_metrics['total_transformations'] >= 0
    
    def test_state_serialization(self, bridge, test_clauses):
        """Test save/load state functionality"""
        # Register some symbols and create field
        for i, clause in enumerate(test_clauses):
            bridge.register_symbol(f"symbol_{i}", clause)
        
        # Save state
        state = bridge.save_state()
        
        # Check state structure
        assert 'symbol_registry' in state
        assert 'symbolic_weights' in state
        assert 'metrics' in state
        
        # Create new bridge and load state
        bridge2 = SymbolicNeuralBridge(field_dims=(32, 32))
        bridge2.load_state(state)
        
        # Verify state was loaded correctly
        assert bridge2.symbol_registry == bridge.symbol_registry
        assert len(bridge2.symbolic_weights) == len(bridge.symbolic_weights)


class TestSymbolicNFCSIntegration:
    """Test cases for complete NFCS integration"""
    
    @pytest.fixture
    def integration(self):
        """Create integration instance for testing"""
        config = {
            'field_dims': (16, 16),  # Smaller for faster testing
            'max_symbols': 32,
            'emergency_ha_threshold': 2.0
        }
        return SymbolicNFCSIntegration(config)
    
    @pytest.mark.asyncio
    async def test_basic_processing_pipeline(self, integration):
        """Test basic input processing pipeline"""
        test_input = """
        Energy equals mass times the speed of light squared.
        This is a fundamental physics equation.
        """
        
        result = await integration.process_input(
            test_input,
            context={'domain': 'physics'}
        )
        
        # Check result structure
        assert 'success' in result
        assert 'hallucination_number' in result
        assert 'coherence_measure' in result
        assert 'processing_time_ms' in result
        
        # Check values are reasonable
        assert 0 <= result['hallucination_number'] <= 10
        assert 0 <= result['coherence_measure'] <= 2
        assert result['processing_time_ms'] > 0
    
    @pytest.mark.asyncio
    async def test_hallucination_number_calculation(self, integration):
        """Test Hallucination Number calculation"""
        # Process input that should have low Ha
        good_input = "E = mc²"
        result1 = await integration.process_input(good_input)
        
        # Process input that should have higher Ha  
        confusing_input = "Energy maybe sometimes equals something squared or not"
        result2 = await integration.process_input(confusing_input)
        
        # Good input should have lower Ha
        assert result1['hallucination_number'] < result2['hallucination_number']
    
    @pytest.mark.asyncio  
    async def test_emergency_protocols(self, integration):
        """Test emergency protocol activation"""
        # Force high Ha value to trigger emergency
        integration.emergency_threshold = 0.5  # Lower threshold
        
        bad_input = "Contradictory nonsensical invalid mathematical physics energy"
        result = await integration.process_input(bad_input)
        
        # Should activate emergency protocols
        assert result.get('emergency_active', False) or result['hallucination_number'] > 1.0
    
    @pytest.mark.asyncio
    async def test_real_time_metrics(self, integration):
        """Test real-time metrics collection"""
        # Process some inputs first
        await integration.process_input("Test input 1")
        await integration.process_input("Test input 2")
        
        # Get metrics
        metrics = await integration.get_real_time_metrics()
        
        # Check metrics structure
        assert 'ha_current' in metrics
        assert 'coherence_current' in metrics
        assert 'field_energy' in metrics
        assert 'performance' in metrics
        
        # Check performance sub-metrics
        perf = metrics['performance']
        assert perf['total_processed'] >= 2


class TestIntegrationFailureModes:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self):
        """Test handling of empty input"""
        integration = SymbolicNFCSIntegration()
        
        result = await integration.process_input("")
        
        # Should handle gracefully
        assert 'success' in result
        assert 'hallucination_number' in result
    
    @pytest.mark.asyncio
    async def test_very_long_input_handling(self):
        """Test handling of extremely long input"""
        integration = SymbolicNFCSIntegration()
        
        # Create very long input
        long_input = "This is a test sentence. " * 1000
        
        result = await integration.process_input(long_input)
        
        # Should complete without crashing
        assert 'success' in result
    
    @pytest.mark.asyncio
    async def test_invalid_unicode_handling(self):
        """Test handling of invalid or complex unicode"""
        integration = SymbolicNFCSIntegration()
        
        unicode_input = "Mathematical symbols: ∑∫∂∇⊗⟨⟩∈∉⊂⊃ρdefψφηχ"
        
        result = await integration.process_input(unicode_input)
        
        # Should handle without errors
        assert 'success' in result


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])