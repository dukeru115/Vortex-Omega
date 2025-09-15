"""
Constitution Module for NFCS

Constitutional framework and policy enforcement system:
- Constitutional policy definition and management
- Real-time constitutional compliance checking
- Policy evolution and adaptation mechanisms
- Emergency constitutional protocols
- Multi-stakeholder governance frameworks
"""

from .constitution_core import ConstitutionModule, ConstitutionalFramework
from .policy_manager import PolicyManager, PolicyType
from .compliance_monitor import ComplianceMonitor, ViolationType
from .governance_system import GovernanceSystem, StakeholderRole

__version__ = "1.0.0"
__all__ = [
    "ConstitutionModule",
    "ConstitutionalFramework",
    "PolicyManager",
    "PolicyType",
    "ComplianceMonitor",
    "ViolationType",
    "GovernanceSystem",
    "StakeholderRole",
]
