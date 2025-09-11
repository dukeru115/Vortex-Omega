"""
Tests for NFCS Orchestrator core functionality
"""

import pytest
import asyncio
from src.orchestrator.nfcs_orchestrator import create_default_config, OrchestrationConfig


class TestOrchestrationConfig:
    """Test OrchestrationConfig creation and validation"""
    
    def test_create_default_config(self):
        """Test default configuration creation"""
        config = create_default_config()
        
        assert isinstance(config, OrchestrationConfig)
        assert config.max_concurrent_processes > 0
        assert config.update_frequency_hz > 0
        assert config.enable_autonomous_mode is True
        assert config.enable_constitutional_enforcement is True
    
    def test_config_validation(self):
        """Test configuration parameter validation"""
        config = create_default_config()
        
        # Test reasonable defaults
        assert config.max_error_threshold >= 10
        assert config.emergency_shutdown_threshold >= 1
        assert config.resource_limit_cpu_percent <= 100
        assert config.resource_limit_memory_mb > 0


class TestNFCSOrchestrator:
    """Test NFCS Orchestrator functionality"""
    
    def test_orchestrator_import(self):
        """Test that orchestrator can be imported without errors"""
        from src.orchestrator.nfcs_orchestrator import NFCSOrchestrator
        assert NFCSOrchestrator is not None
    
    @pytest.mark.asyncio
    async def test_orchestrator_creation(self):
        """Test orchestrator creation with default config"""
        from src.orchestrator.nfcs_orchestrator import NFCSOrchestrator
        
        config = create_default_config()
        # Reduce settings for test
        config.max_concurrent_processes = 2
        config.update_frequency_hz = 1.0
        
        orchestrator = NFCSOrchestrator(config)
        assert orchestrator is not None
        assert orchestrator.config == config
    
    def test_orchestrator_repr(self):
        """Test orchestrator string representation"""
        from src.orchestrator.nfcs_orchestrator import NFCSOrchestrator
        
        config = create_default_config()
        orchestrator = NFCSOrchestrator(config)
        
        repr_str = repr(orchestrator)
        assert "NFCSOrchestrator" in repr_str
        assert "status=" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])