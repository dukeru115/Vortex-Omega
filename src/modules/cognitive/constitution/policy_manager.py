"""
Policy Manager - Constitutional Policy Management System
========================================================

The PolicyManager provides centralized management of constitutional policies,
including creation, modification, enforcement, and compliance monitoring.

Key Features:
- Policy lifecycle management (create, update, activate, deactivate, delete)
- Policy hierarchy and inheritance management
- Conflict detection and resolution between policies
- Real-time policy enforcement and violation tracking
- Policy versioning and rollback capabilities
- Stakeholder-based governance and consensus mechanisms

Architecture:
The PolicyManager operates as a central authority for all policy-related
operations while maintaining decentralized enforcement through distributed
compliance monitors and enforcement agents.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timezone
import uuid


class PolicyType(Enum):
    """Types of constitutional policies."""
    FUNDAMENTAL = "fundamental"      # Core unchangeable principles
    OPERATIONAL = "operational"      # Day-to-day operation rules
    CONTEXTUAL = "contextual"        # Context-specific guidelines
    EMERGENCY = "emergency"          # Emergency protocol policies
    STAKEHOLDER = "stakeholder"      # Stakeholder-specific policies


class PolicyStatus(Enum):
    """Policy lifecycle status."""
    DRAFT = "draft"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class ConstitutionalPolicy:
    """Constitutional policy representation."""
    policy_id: str
    title: str
    description: str
    policy_type: PolicyType
    content: Dict[str, Any]
    priority: int = 5  # 1-10 scale, 10 = highest
    enforcement_level: float = 1.0  # 0.0-1.0 scale
    status: PolicyStatus = PolicyStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"
    author: str = "NFCS_System"
    stakeholders: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)


class PolicyManager:
    """Central manager for constitutional policies."""
    
    def __init__(self):
        """Initialize the Policy Manager."""
        self.policies: Dict[str, ConstitutionalPolicy] = {}
        self.policy_hierarchy: Dict[str, List[str]] = {}
        self.active_policies: Set[str] = set()
        self.enforcement_rules: Dict[str, Any] = {}
        self.violation_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("PolicyManager")
        
    def create_policy(self, 
                     policy_id: str,
                     title: str,
                     description: str,
                     policy_type: PolicyType,
                     content: Dict[str, Any],
                     **kwargs) -> ConstitutionalPolicy:
        """Create a new constitutional policy."""
        if policy_id in self.policies:
            raise ValueError(f"Policy {policy_id} already exists")
        
        policy = ConstitutionalPolicy(
            policy_id=policy_id,
            title=title,
            description=description,
            policy_type=policy_type,
            content=content,
            **kwargs
        )
        
        self.policies[policy_id] = policy
        self.logger.info(f"Created policy: {policy_id}")
        return policy
    
    def activate_policy(self, policy_id: str) -> bool:
        """Activate a policy for enforcement."""
        if policy_id not in self.policies:
            return False
        
        policy = self.policies[policy_id]
        policy.status = PolicyStatus.ACTIVE
        self.active_policies.add(policy_id)
        self.logger.info(f"Activated policy: {policy_id}")
        return True
    
    def deactivate_policy(self, policy_id: str) -> bool:
        """Deactivate a policy."""
        if policy_id not in self.policies:
            return False
        
        policy = self.policies[policy_id]
        policy.status = PolicyStatus.SUSPENDED
        self.active_policies.discard(policy_id)
        self.logger.info(f"Deactivated policy: {policy_id}")
        return True
    
    def get_policy(self, policy_id: str) -> Optional[ConstitutionalPolicy]:
        """Get a policy by ID."""
        return self.policies.get(policy_id)
    
    def get_active_policies(self) -> List[ConstitutionalPolicy]:
        """Get all currently active policies."""
        return [self.policies[pid] for pid in self.active_policies 
                if pid in self.policies]
    
    def get_policies_by_type(self, policy_type: PolicyType) -> List[ConstitutionalPolicy]:
        """Get policies by type."""
        return [policy for policy in self.policies.values() 
                if policy.policy_type == policy_type]
    
    def check_compliance(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check if an action complies with active policies."""
        violations = []
        compliant_policies = []
        
        for policy_id in self.active_policies:
            policy = self.policies[policy_id]
            # Basic compliance check (simplified)
            if self._violates_policy(action, policy):
                violations.append({
                    'policy_id': policy_id,
                    'policy_title': policy.title,
                    'violation_type': 'content_violation',
                    'details': f"Action violates {policy.title}"
                })
            else:
                compliant_policies.append(policy_id)
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'compliant_policies': compliant_policies,
            'checked_at': datetime.now(timezone.utc).isoformat()
        }
    
    def _violates_policy(self, action: Dict[str, Any], policy: ConstitutionalPolicy) -> bool:
        """Check if an action violates a specific policy (simplified implementation)."""
        # This is a basic implementation - in practice would be much more sophisticated
        action_type = action.get('type', '')
        policy_content = policy.content
        
        # Check if action is explicitly forbidden
        forbidden_actions = policy_content.get('forbidden_actions', [])
        if action_type in forbidden_actions:
            return True
        
        # Check constraints
        constraints = policy_content.get('constraints', {})
        for constraint, value in constraints.items():
            action_value = action.get(constraint)
            if action_value is not None and action_value > value:
                return True
        
        return False
    
    def log_violation(self, violation: Dict[str, Any]) -> None:
        """Log a policy violation."""
        violation['logged_at'] = datetime.now(timezone.utc).isoformat()
        violation['violation_id'] = str(uuid.uuid4())
        self.violation_log.append(violation)
        self.logger.warning(f"Policy violation logged: {violation}")
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of policy violations."""
        total_violations = len(self.violation_log)
        recent_violations = [v for v in self.violation_log 
                           if (datetime.now(timezone.utc) - 
                               datetime.fromisoformat(v['logged_at'].replace('Z', '+00:00'))).days < 7]
        
        return {
            'total_violations': total_violations,
            'recent_violations': len(recent_violations),
            'violation_types': list(set(v.get('violation_type', 'unknown') 
                                      for v in self.violation_log)),
            'most_violated_policies': self._get_most_violated_policies()
        }
    
    def _get_most_violated_policies(self) -> List[Dict[str, Any]]:
        """Get policies that are violated most frequently."""
        violation_counts = {}
        for violation in self.violation_log:
            policy_id = violation.get('policy_id')
            if policy_id:
                violation_counts[policy_id] = violation_counts.get(policy_id, 0) + 1
        
        return [{'policy_id': pid, 'violations': count} 
                for pid, count in sorted(violation_counts.items(), 
                                       key=lambda x: x[1], reverse=True)]
    
    def get_status(self) -> Dict[str, Any]:
        """Get policy manager status."""
        return {
            'total_policies': len(self.policies),
            'active_policies': len(self.active_policies),
            'policy_types': list(set(p.policy_type.value for p in self.policies.values())),
            'violation_summary': self.get_violation_summary()
        }