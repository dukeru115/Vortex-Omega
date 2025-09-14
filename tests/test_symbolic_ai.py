"""
Comprehensive Test Suite for Symbolic AI Module
===============================================

Tests for the Symbolic AI module with focus on:
- Core functionality
- Parser accuracy
- Unit system
- Verification pipeline
- Discrepancy detection
- Kant mode ethical testing
- Security features
- Integration with ESC

Target: 80% code coverage
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import time
from typing import List, Dict, Any

# Import modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'modules'))

from symbolic.symbolic_core import SymbolicCore
from symbolic.models import (
    SymClause, SymField, Unit, ClauseType, 
    VerificationResult, DiscrepancyReport
)
from symbolic.parser import SymbolicParser
from symbolic.units import UnitSystem
from symbolic.verifier import SymbolicVerifier
from symbolic.discrepancy_gate import DiscrepancyGate
from symbolic.kant_mode import KantMode
from symbolic.security import SecurityModule, RateLimiter, CircuitBreaker


class TestSymbolicCore:
    """Test the main SymbolicCore orchestrator"""
    
    @pytest.fixture
    def symbolic_core(self):
        """Create a SymbolicCore instance for testing"""
        return SymbolicCore()
    
    @pytest.mark.asyncio
    async def test_process_text_basic(self, symbolic_core):
        """Test basic text processing"""
        text = "The velocity is 10 m/s and the mass is 5 kg"
        result = await symbolic_core.process_text(text)
        
        assert result is not None
        assert 'clauses' in result
        assert 'fields' in result
        assert 'verification' in result
        assert len(result['clauses']) > 0
    
    @pytest.mark.asyncio
    async def test_process_text_with_formulas(self, symbolic_core):
        """Test processing text with mathematical formulas"""
        text = "According to F = ma, with mass = 2 kg and acceleration = 9.8 m/s^2"
        result = await symbolic_core.process_text(text)
        
        assert result is not None
        assert any(c.formula for c in result['clauses'])
        
    @pytest.mark.asyncio
    async def test_parallel_processing(self, symbolic_core):
        """Test parallel processing of multiple texts"""
        texts = [
            "The temperature is 25°C",
            "The pressure is 101.3 kPa",
            "The volume is 2.5 L"
        ]
        
        results = await asyncio.gather(*[
            symbolic_core.process_text(text) for text in texts
        ])
        
        assert len(results) == 3
        assert all(r is not None for r in results)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, symbolic_core):
        """Test error handling with invalid input"""
        with pytest.raises(ValueError):
            await symbolic_core.process_text("")
        
        with pytest.raises(ValueError):
            await symbolic_core.process_text("a" * 10001)  # Too long
    
    @pytest.mark.asyncio
    async def test_fieldize_integration(self, symbolic_core):
        """Test the fieldize step in the pipeline"""
        text = "The particle has energy E = 10 J"
        result = await symbolic_core.process_text(text)
        
        fields = result['fields']
        assert len(fields) > 0
        assert any(f.intensity > 0 for f in fields)


class TestSymbolicParser:
    """Test the SymbolicParser component"""
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance"""
        return SymbolicParser()
    
    def test_parse_simple_text(self, parser):
        """Test parsing simple text"""
        text = "The speed is 100 km/h"
        clauses = parser.parse(text)
        
        assert len(clauses) > 0
        assert any(c.ctype == ClauseType.QUANTITY for c in clauses)
    
    def test_parse_formula(self, parser):
        """Test parsing mathematical formulas"""
        text = "E = mc^2 where m is mass and c is speed of light"
        clauses = parser.parse(text)
        
        assert any(c.formula == "E = mc^2" for c in clauses)
        assert any(c.ctype == ClauseType.EQUATION for c in clauses)
    
    def test_parse_complex_expression(self, parser):
        """Test parsing complex mathematical expressions"""
        text = "The wave function ψ(x,t) = A*exp(i(kx - ωt))"
        clauses = parser.parse(text)
        
        assert any("ψ" in c.content or "psi" in c.content for c in clauses)
        assert any(c.ctype == ClauseType.EQUATION for c in clauses)
    
    def test_extract_entities(self, parser):
        """Test entity extraction"""
        text = "Newton discovered that force equals mass times acceleration"
        clauses = parser.parse(text)
        
        # Check for entity extraction
        entities = [c for c in clauses if c.ctype == ClauseType.ENTITY]
        assert len(entities) > 0
        
    def test_extract_units(self, parser):
        """Test unit extraction"""
        text = "The distance is 5 meters and time is 2 seconds"
        clauses = parser.parse(text)
        
        quantities = [c for c in clauses if c.ctype == ClauseType.QUANTITY]
        assert len(quantities) >= 2
        assert any(c.units for c in quantities)


class TestUnitSystem:
    """Test the UnitSystem for dimensional analysis"""
    
    @pytest.fixture
    def unit_system(self):
        """Create a UnitSystem instance"""
        return UnitSystem()
    
    def test_parse_simple_unit(self, unit_system):
        """Test parsing simple units"""
        unit = unit_system.parse_unit("m/s")
        assert unit is not None
        assert unit.symbol == "m/s"
        assert unit.dimensions['length'] == 1
        assert unit.dimensions['time'] == -1
    
    def test_parse_complex_unit(self, unit_system):
        """Test parsing complex units"""
        unit = unit_system.parse_unit("kg*m^2/s^3")
        assert unit is not None
        assert unit.dimensions['mass'] == 1
        assert unit.dimensions['length'] == 2
        assert unit.dimensions['time'] == -3
    
    def test_unit_conversion(self, unit_system):
        """Test unit conversion"""
        # Convert km/h to m/s
        value = unit_system.convert(100, "km/h", "m/s")
        assert abs(value - 27.778) < 0.01
        
        # Convert Celsius to Kelvin
        value = unit_system.convert(25, "°C", "K")
        assert abs(value - 298.15) < 0.01
    
    def test_check_dimensional_consistency(self, unit_system):
        """Test dimensional consistency checking"""
        # Consistent: both are velocities
        assert unit_system.check_consistency("m/s", "km/h")
        
        # Inconsistent: velocity vs acceleration
        assert not unit_system.check_consistency("m/s", "m/s^2")
        
        # Consistent: both are energies
        assert unit_system.check_consistency("J", "kg*m^2/s^2")
    
    def test_invalid_unit_handling(self, unit_system):
        """Test handling of invalid units"""
        unit = unit_system.parse_unit("invalid_unit")
        assert unit is None
        
        with pytest.raises(ValueError):
            unit_system.convert(10, "invalid", "m")


class TestSymbolicVerifier:
    """Test the verification system"""
    
    @pytest.fixture
    def verifier(self):
        """Create a verifier instance"""
        return SymbolicVerifier()
    
    @pytest.mark.asyncio
    async def test_verify_consistent_clauses(self, verifier):
        """Test verification of consistent clauses"""
        clauses = [
            SymClause(
                content="velocity is 10 m/s",
                ctype=ClauseType.QUANTITY,
                units="m/s",
                value=10.0
            ),
            SymClause(
                content="distance is 100 m",
                ctype=ClauseType.QUANTITY,
                units="m",
                value=100.0
            ),
            SymClause(
                content="time is 10 s",
                ctype=ClauseType.QUANTITY,
                units="s",
                value=10.0
            )
        ]
        
        result = await verifier.verify(clauses)
        assert result.is_valid
        assert result.confidence > 0.7
    
    @pytest.mark.asyncio
    async def test_verify_inconsistent_clauses(self, verifier):
        """Test verification of inconsistent clauses"""
        clauses = [
            SymClause(
                content="velocity is 10 m/s",
                ctype=ClauseType.QUANTITY,
                units="m/s",
                value=10.0
            ),
            SymClause(
                content="velocity is 20 m/s",  # Contradiction
                ctype=ClauseType.QUANTITY,
                units="m/s",
                value=20.0
            )
        ]
        
        result = await verifier.verify(clauses)
        assert not result.is_valid
        assert len(result.issues) > 0
    
    @pytest.mark.asyncio
    async def test_verify_formula(self, verifier):
        """Test formula verification"""
        clauses = [
            SymClause(
                content="F = ma",
                ctype=ClauseType.EQUATION,
                formula="F = m * a"
            ),
            SymClause(
                content="mass is 2 kg",
                ctype=ClauseType.QUANTITY,
                units="kg",
                value=2.0
            ),
            SymClause(
                content="acceleration is 5 m/s^2",
                ctype=ClauseType.QUANTITY,
                units="m/s^2",
                value=5.0
            )
        ]
        
        result = await verifier.verify(clauses)
        assert result.is_valid


class TestDiscrepancyGate:
    """Test the discrepancy detection system"""
    
    @pytest.fixture
    def gate(self):
        """Create a DiscrepancyGate instance"""
        return DiscrepancyGate()
    
    @pytest.mark.asyncio
    async def test_no_discrepancy(self, gate):
        """Test when there's no discrepancy"""
        symbolic_output = {
            'clauses': [
                SymClause(
                    content="temperature is 25°C",
                    ctype=ClauseType.QUANTITY,
                    value=25.0,
                    units="°C"
                )
            ]
        }
        
        llm_output = "The temperature is 25 degrees Celsius"
        
        report = await gate.check_discrepancy(symbolic_output, llm_output)
        assert not report.has_discrepancy
        assert report.severity == "none"
    
    @pytest.mark.asyncio
    async def test_minor_discrepancy(self, gate):
        """Test detection of minor discrepancies"""
        symbolic_output = {
            'clauses': [
                SymClause(
                    content="speed is 100 km/h",
                    ctype=ClauseType.QUANTITY,
                    value=100.0,
                    units="km/h"
                )
            ]
        }
        
        llm_output = "The speed is approximately 100 kilometers per hour"
        
        report = await gate.check_discrepancy(symbolic_output, llm_output)
        # Minor discrepancy due to "approximately"
        assert report.severity in ["none", "low"]
    
    @pytest.mark.asyncio
    async def test_critical_discrepancy(self, gate):
        """Test detection of critical discrepancies"""
        symbolic_output = {
            'clauses': [
                SymClause(
                    content="pressure is 1 atm",
                    ctype=ClauseType.QUANTITY,
                    value=1.0,
                    units="atm"
                )
            ]
        }
        
        llm_output = "The pressure is 10 atmospheres"  # Wrong value
        
        report = await gate.check_discrepancy(symbolic_output, llm_output)
        assert report.has_discrepancy
        assert report.severity in ["high", "critical"]
        assert len(report.discrepancies) > 0


class TestKantMode:
    """Test the Kant mode ethical testing"""
    
    @pytest.fixture
    def kant_mode(self):
        """Create a KantMode instance"""
        return KantMode()
    
    @pytest.mark.asyncio
    async def test_universalization_principle(self, kant_mode):
        """Test the universalization principle"""
        action = "Always tell the truth"
        context = {'situation': 'medical diagnosis'}
        
        result = await kant_mode.test_universalization(action, context)
        assert result is not None
        assert 'can_universalize' in result
        assert 'reasoning' in result
    
    @pytest.mark.asyncio
    async def test_means_end_principle(self, kant_mode):
        """Test the means-end principle"""
        action = "Use customer data for targeted advertising"
        context = {'stakeholders': ['customers', 'company']}
        
        result = await kant_mode.test_means_end(action, context)
        assert result is not None
        assert 'respects_persons' in result
        assert 'affected_parties' in result
    
    @pytest.mark.asyncio
    async def test_full_ethical_evaluation(self, kant_mode):
        """Test complete ethical evaluation"""
        scenario = {
            'action': 'Implement facial recognition in public spaces',
            'purpose': 'Enhance security',
            'affected': ['citizens', 'law enforcement', 'criminals'],
            'consequences': ['increased surveillance', 'reduced crime', 'privacy concerns']
        }
        
        evaluation = await kant_mode.evaluate(scenario)
        assert evaluation is not None
        assert 'universalization' in evaluation
        assert 'means_end' in evaluation
        assert 'overall_assessment' in evaluation
        assert 'ethical_score' in evaluation
        assert 0 <= evaluation['ethical_score'] <= 1


class TestSecurityModule:
    """Test security features"""
    
    @pytest.fixture
    def security(self):
        """Create a SecurityModule instance"""
        return SecurityModule()
    
    def test_input_sanitization(self, security):
        """Test input sanitization"""
        # Test SQL injection attempt
        malicious = "'; DROP TABLE users; --"
        sanitized = security.sanitize_input(malicious)
        assert "DROP TABLE" not in sanitized
        
        # Test XSS attempt
        xss = "<script>alert('XSS')</script>"
        sanitized = security.sanitize_input(xss)
        assert "<script>" not in sanitized
        
        # Test path traversal
        path_attack = "../../../etc/passwd"
        sanitized = security.sanitize_input(path_attack)
        assert "../" not in sanitized
    
    def test_rate_limiting(self, security):
        """Test rate limiting"""
        user_id = "test_user"
        
        # Should allow initial requests
        for _ in range(5):
            assert security.check_rate_limit(user_id)
        
        # Should eventually rate limit
        limited = False
        for _ in range(100):
            if not security.check_rate_limit(user_id):
                limited = True
                break
        
        assert limited, "Rate limiting should kick in"
    
    def test_circuit_breaker(self, security):
        """Test circuit breaker pattern"""
        service = "test_service"
        
        # Simulate failures
        for _ in range(10):
            security.record_failure(service)
        
        # Circuit should be open
        assert not security.is_service_available(service)
        
        # Wait and test half-open state
        time.sleep(2)
        assert security.is_service_available(service)
        
        # Record success to close circuit
        security.record_success(service)
        assert security.is_service_available(service)
    
    def test_input_validation(self, security):
        """Test input validation"""
        # Valid input
        assert security.validate_input("Normal text with numbers 123")
        
        # Too long
        assert not security.validate_input("a" * 20000)
        
        # Empty
        assert not security.validate_input("")
        
        # Control characters
        assert not security.validate_input("Text with \x00 null")


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.fixture
    def system(self):
        """Create a complete system for integration testing"""
        return {
            'core': SymbolicCore(),
            'parser': SymbolicParser(),
            'verifier': SymbolicVerifier(),
            'gate': DiscrepancyGate(),
            'kant': KantMode(),
            'security': SecurityModule()
        }
    
    @pytest.mark.asyncio
    async def test_complete_pipeline(self, system):
        """Test the complete processing pipeline"""
        # Input text
        text = """
        Consider a projectile launched at angle θ = 45° with initial velocity v0 = 20 m/s.
        The maximum height is H = v0²sin²(θ)/(2g) where g = 9.8 m/s².
        This gives H ≈ 10.2 m.
        """
        
        # Process through pipeline
        core = system['core']
        result = await core.process_text(text)
        
        # Verify output structure
        assert 'clauses' in result
        assert 'fields' in result
        assert 'verification' in result
        
        # Check clause extraction
        clauses = result['clauses']
        assert len(clauses) > 0
        assert any(c.ctype == ClauseType.EQUATION for c in clauses)
        assert any(c.ctype == ClauseType.QUANTITY for c in clauses)
        
        # Check verification
        verification = result['verification']
        assert verification is not None
        assert hasattr(verification, 'is_valid')
        assert hasattr(verification, 'confidence')
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, system):
        """Test system error recovery"""
        core = system['core']
        
        # Test with problematic input
        problematic_text = "The value is ∞ and undefined³²¹"
        
        try:
            result = await core.process_text(problematic_text)
            # Should handle gracefully
            assert result is not None
        except Exception as e:
            pytest.fail(f"System should handle errors gracefully: {e}")
    
    @pytest.mark.asyncio  
    async def test_performance(self, system):
        """Test system performance"""
        core = system['core']
        
        # Process multiple texts
        texts = [
            f"The measurement {i} is {i*10} units" 
            for i in range(10)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*[
            core.process_text(text) for text in texts
        ])
        elapsed = time.time() - start_time
        
        # Should process 10 texts in reasonable time
        assert elapsed < 5.0  # 5 seconds max
        assert len(results) == 10
        assert all(r is not None for r in results)
    
    @pytest.mark.asyncio
    async def test_memory_safety(self, system):
        """Test memory safety with large inputs"""
        core = system['core']
        
        # Create large but valid input
        large_text = " ".join([
            f"Variable x{i} = {i}" for i in range(1000)
        ])
        
        # Should handle without memory issues
        result = await core.process_text(large_text)
        assert result is not None
        
        # Check memory is released (basic check)
        import gc
        gc.collect()
        # If we get here without crashing, memory management is working


class TestESCIntegration:
    """Test integration with Enhanced ESC module"""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated Symbolic + ESC system"""
        from esc.enhanced_esc import EnhancedESC
        
        return {
            'symbolic': SymbolicCore(),
            'esc': EnhancedESC()
        }
    
    @pytest.mark.asyncio
    async def test_symbolic_to_esc_flow(self, integrated_system):
        """Test data flow from Symbolic to ESC"""
        symbolic = integrated_system['symbolic']
        esc = integrated_system['esc']
        
        # Process text symbolically
        text = "The frequency is 440 Hz (musical note A4)"
        symbolic_result = await symbolic.process_text(text)
        
        # Convert to ESC format
        clauses = symbolic_result['clauses']
        order_param = esc.integrate_with_symbolic(clauses)
        
        # Verify integration
        assert order_param is not None
        assert -1 <= order_param <= 1  # Should be normalized
        
        # Check ESC state update
        esc_state = esc.get_semantic_field_state()
        assert esc_state is not None
        assert len(esc_state) > 0


# Fixtures for pytest
@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    # Reset any singleton patterns if used
    yield
    import gc
    gc.collect()


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for critical paths"""
    
    def test_parser_performance(self, benchmark):
        """Benchmark parser performance"""
        parser = SymbolicParser()
        text = "E = mc^2 where m = 1 kg and c = 3e8 m/s"
        
        result = benchmark(parser.parse, text)
        assert len(result) > 0
    
    def test_verifier_performance(self, benchmark):
        """Benchmark verifier performance"""
        verifier = SymbolicVerifier()
        clauses = [
            SymClause(
                content=f"var{i} = {i}",
                ctype=ClauseType.QUANTITY,
                value=float(i)
            )
            for i in range(100)
        ]
        
        async def verify_async():
            return await verifier.verify(clauses)
        
        loop = asyncio.new_event_loop()
        result = benchmark(lambda: loop.run_until_complete(verify_async()))
        assert result is not None


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=symbolic",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-target=80"
    ])