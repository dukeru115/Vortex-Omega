"""
Tests for Constitutional Framework
"""

import pytest
from src.modules.cognitive.constitution.constitution_core import ConstitutionalFramework


class TestConstitutionalFramework:
    """Test Constitutional Framework functionality"""

    def test_framework_creation(self):
        """Test constitutional framework can be created"""
        framework = ConstitutionalFramework()
        assert framework is not None

    def test_compliance_check(self):
        """Test basic compliance checking"""
        framework = ConstitutionalFramework()

        # Test with empty context (should pass)
        result = framework.check_compliance({})
        assert hasattr(result, "compliant")

        # Should not crash with test context
        test_context = {"action": "test_action", "module": "test_module"}
        result = framework.check_compliance(test_context)
        assert hasattr(result, "compliant")

    def test_get_active_policies(self):
        """Test getting active policies"""
        framework = ConstitutionalFramework()

        # Should return some form of policies (could be empty)
        policies = framework.get_active_policies()
        assert isinstance(policies, (list, dict))


if __name__ == "__main__":
    pytest.main([__file__])
