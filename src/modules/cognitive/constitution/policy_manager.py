"""
Policy Manager for Constitutional Framework
=========================================

Manages constitutional policies, enforcement, and compliance monitoring
for the Neural Field Control System (NFCS).
"""

from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field
import logging
import json
from datetime import datetime, timezone


class PolicyType(Enum):
    """Types of constitutional policies."""
    FUNDAMENTAL = "fundamental"
    OPERATIONAL = "operational"
    GOVERNANCE = "governance"
    SAFETY = "safety"
    ETHICAL = "ethical"


class EnforcementLevel(Enum):
    """Policy enforcement levels."""
    ADVISORY = 0.2      # Advisory recommendations
    MODERATE = 0.5      # Moderate enforcement
    STRICT = 0.8        # Strict enforcement
    ABSOLUTE = 1.0      # Absolute enforcement


@dataclass
class ConstitutionalPolicy:
    """Represents a single constitutional policy."""
    policy_id: str
    title: str
    description: str
    policy_type: PolicyType
    content: Dict[str, Any]
    priority: int = 5
    enforcement_level: float = 0.8
    active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)


@dataclass
class PolicyViolation:
    """Represents a policy violation."""
    violation_id: str
    policy_id: str
    description: str
    severity: float
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False


class PolicyManager:
    """
    Manages constitutional policies and enforcement.
    
    This class handles policy storage, retrieval, enforcement,
    and violation tracking for the constitutional framework.
    """
    
    def __init__(self):
        """Initialize the policy manager."""
        self.logger = logging.getLogger("PolicyManager")
        
        # Policy storage
        self.policies: Dict[str, ConstitutionalPolicy] = {}
        self.policy_hierarchy: Dict[PolicyType, List[str]] = {
            policy_type: [] for policy_type in PolicyType
        }
        
        # Violation tracking
        self.violations: Dict[str, PolicyViolation] = {}
        self.violation_counter = 0
        
        # Enforcement settings
        self.enforcement_enabled = True
        self.global_enforcement_level = 0.8
        
        self.logger.info("Policy Manager initialized")
    
    def add_policy(self, policy: ConstitutionalPolicy) -> bool:
        """
        Add a new constitutional policy.
        
        Args:
            policy: The policy to add
            
        Returns:
            True if policy was added successfully
        """
        try:
            if policy.policy_id in self.policies:
                self.logger.warning(f"Policy {policy.policy_id} already exists, updating")
            
            self.policies[policy.policy_id] = policy
            
            # Add to hierarchy
            if policy.policy_id not in self.policy_hierarchy[policy.policy_type]:
                self.policy_hierarchy[policy.policy_type].append(policy.policy_id)
                # Sort by priority
                self.policy_hierarchy[policy.policy_type].sort(
                    key=lambda pid: self.policies[pid].priority, reverse=True
                )
            
            self.logger.info(f"Added policy: {policy.policy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add policy {policy.policy_id}: {e}")
            return False
    
    def get_policy(self, policy_id: str) -> Optional[ConstitutionalPolicy]:
        """Get a policy by ID."""
        return self.policies.get(policy_id)
    
    def get_policies_by_type(self, policy_type: PolicyType) -> List[ConstitutionalPolicy]:
        """Get all policies of a specific type."""
        policy_ids = self.policy_hierarchy.get(policy_type, [])
        return [self.policies[pid] for pid in policy_ids if pid in self.policies]
    
    def get_active_policies(self) -> List[ConstitutionalPolicy]:
        """Get all active policies."""
        return [policy for policy in self.policies.values() if policy.active]
    
    def evaluate_compliance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate compliance with constitutional policies.
        
        Args:
            context: Context for evaluation
            
        Returns:
            Compliance evaluation results
        """
        try:
            results = {
                'compliant': True,
                'violations': [],
                'recommendations': [],
                'enforcement_actions': [],
                'evaluation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Check each active policy
            for policy in self.get_active_policies():
                violation = self._check_policy_compliance(policy, context)
                if violation:
                    results['violations'].append(violation)
                    results['compliant'] = False
                    
                    # Record violation
                    self._record_violation(violation)
                    
                    # Determine enforcement action
                    if policy.enforcement_level >= self.global_enforcement_level:
                        enforcement_action = {
                            'policy_id': policy.policy_id,
                            'action': 'block',
                            'severity': violation['severity']
                        }
                        results['enforcement_actions'].append(enforcement_action)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate compliance: {e}")
            return {
                'compliant': False,
                'error': str(e),
                'evaluation_timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _check_policy_compliance(self, policy: ConstitutionalPolicy, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check compliance with a specific policy."""
        try:
            # Basic policy evaluation logic
            # This is a simplified implementation - in practice would be more sophisticated
            
            policy_content = policy.content
            
            # Check safety constraints
            if 'safety' in context and 'constraints' in policy_content:
                constraints = policy_content['constraints']
                if 'max_risk_level' in constraints:
                    if context['safety'].get('risk_level', 0) > constraints['max_risk_level']:
                        return {
                            'policy_id': policy.policy_id,
                            'violation_type': 'safety_risk_exceeded',
                            'description': f"Risk level {context['safety']['risk_level']} exceeds maximum {constraints['max_risk_level']}",
                            'severity': 0.8,
                            'context': context
                        }
            
            # Check ethical principles
            if 'ethical_concerns' in context and policy.policy_type == PolicyType.ETHICAL:
                concerns = context['ethical_concerns']
                if concerns and len(concerns) > 0:
                    return {
                        'policy_id': policy.policy_id,
                        'violation_type': 'ethical_concern',
                        'description': f"Ethical concerns detected: {', '.join(concerns)}",
                        'severity': 0.6,
                        'context': context
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking policy compliance for {policy.policy_id}: {e}")
            return None
    
    def _record_violation(self, violation_data: Dict[str, Any]) -> None:
        """Record a policy violation."""
        try:
            self.violation_counter += 1
            violation_id = f"VIO_{self.violation_counter:06d}"
            
            violation = PolicyViolation(
                violation_id=violation_id,
                policy_id=violation_data['policy_id'],
                description=violation_data['description'],
                severity=violation_data['severity'],
                context=violation_data['context']
            )
            
            self.violations[violation_id] = violation
            self.logger.warning(f"Policy violation recorded: {violation_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to record violation: {e}")
    
    def get_violations(self, policy_id: Optional[str] = None, unresolved_only: bool = True) -> List[PolicyViolation]:
        """Get policy violations."""
        violations = list(self.violations.values())
        
        if policy_id:
            violations = [v for v in violations if v.policy_id == policy_id]
        
        if unresolved_only:
            violations = [v for v in violations if not v.resolved]
        
        return violations
    
    def resolve_violation(self, violation_id: str) -> bool:
        """Mark a violation as resolved."""
        if violation_id in self.violations:
            self.violations[violation_id].resolved = True
            self.logger.info(f"Violation {violation_id} marked as resolved")
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get policy manager status."""
        return {
            'total_policies': len(self.policies),
            'active_policies': len(self.get_active_policies()),
            'policies_by_type': {
                policy_type.value: len(self.get_policies_by_type(policy_type))
                for policy_type in PolicyType
            },
            'total_violations': len(self.violations),
            'unresolved_violations': len(self.get_violations()),
            'enforcement_enabled': self.enforcement_enabled,
            'global_enforcement_level': self.global_enforcement_level
        }