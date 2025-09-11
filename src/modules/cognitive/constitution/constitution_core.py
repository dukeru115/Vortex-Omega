"""
Constitution Module Core - NFCS Constitutional Framework

Implements the foundational constitutional system for NFCS with:
- Constitutional policy definition and enforcement
- Real-time compliance monitoring and violation detection
- Policy adaptation and evolution mechanisms
- Emergency constitutional protocols and interventions
- Multi-stakeholder governance and consensus mechanisms
- Constitutional rights and constraints framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import time
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    """Types of constitutional policies."""
    FUNDAMENTAL = "fundamental"     # Core unchangeable principles
    OPERATIONAL = "operational"     # Day-to-day operational policies
    SAFETY = "safety"              # Safety and security policies
    ETHICAL = "ethical"            # Ethical guidelines and principles
    PROCEDURAL = "procedural"      # Process and procedure policies
    EMERGENCY = "emergency"        # Emergency response protocols


class ViolationType(Enum):
    """Types of constitutional violations."""
    MINOR = "minor"                # Minor policy infractions
    MODERATE = "moderate"          # Moderate violations requiring attention
    MAJOR = "major"               # Major violations requiring intervention
    CRITICAL = "critical"         # Critical violations requiring emergency response
    SYSTEMIC = "systemic"         # Systemic violations affecting core operations


class PolicyStatus(Enum):
    """Status of constitutional policies."""
    ACTIVE = "active"             # Currently active and enforced
    PROPOSED = "proposed"         # Proposed but not yet active
    DEPRECATED = "deprecated"     # Deprecated but still in effect
    SUSPENDED = "suspended"       # Temporarily suspended
    ARCHIVED = "archived"         # Archived and no longer active


class StakeholderRole(Enum):
    """Stakeholder roles in constitutional governance."""
    ADMINISTRATOR = "administrator"    # System administrators
    SUPERVISOR = "supervisor"         # Human supervisors
    USER = "user"                    # End users
    DEVELOPER = "developer"          # System developers
    AUDITOR = "auditor"             # External auditors
    SYSTEM = "system"               # Automated system processes


@dataclass
class ConstitutionalPolicy:
    """Represents a constitutional policy."""
    policy_id: str
    title: str
    description: str
    policy_type: PolicyType
    content: Dict[str, Any]
    priority: int = 1              # Higher number = higher priority
    status: PolicyStatus = PolicyStatus.ACTIVE
    created_by: str = "system"
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    version: str = "1.0.0"
    stakeholder_approval: Dict[StakeholderRole, bool] = field(default_factory=dict)
    enforcement_level: float = 1.0  # 0.0 = advisory, 1.0 = mandatory
    exceptions: List[str] = field(default_factory=list)


@dataclass
class ViolationRecord:
    """Records a constitutional violation."""
    violation_id: str
    policy_id: str
    violation_type: ViolationType
    description: str
    context: Dict[str, Any]
    severity: float               # 0.0 - 1.0
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_actions: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstitutionalRights:
    """Defines constitutional rights and protections."""
    privacy_protection: bool = True
    transparency_access: bool = True
    fairness_guarantee: bool = True
    safety_assurance: bool = True
    autonomy_respect: bool = True
    dignity_protection: bool = True
    non_discrimination: bool = True
    appeal_process: bool = True


@dataclass
class GovernanceConfig:
    """Configuration for constitutional governance."""
    require_stakeholder_consensus: bool = True
    minimum_approval_threshold: float = 0.6
    emergency_override_enabled: bool = True
    policy_review_period_days: int = 90
    violation_escalation_threshold: int = 5
    automatic_enforcement: bool = True


class ConstitutionalFramework:
    """
    Core constitutional framework managing all constitutional aspects of NFCS.
    
    Provides centralized constitutional policy management, compliance monitoring,
    violation detection, and governance mechanisms for the entire system.
    """
    
    def __init__(self, 
                 governance_config: Optional[GovernanceConfig] = None,
                 constitutional_rights: Optional[ConstitutionalRights] = None):
        """
        Initialize constitutional framework.
        
        Args:
            governance_config: Governance configuration
            constitutional_rights: Constitutional rights framework
        """
        self.governance_config = governance_config or GovernanceConfig()
        self.constitutional_rights = constitutional_rights or ConstitutionalRights()
        
        # Policy storage and management
        self.active_policies: Dict[str, ConstitutionalPolicy] = {}
        self.policy_history: Dict[str, List[ConstitutionalPolicy]] = defaultdict(list)
        self.policy_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # Violation tracking and management
        self.violation_records: Dict[str, ViolationRecord] = {}
        self.violation_history: deque = deque(maxlen=10000)
        self.violation_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # Stakeholder management
        self.registered_stakeholders: Dict[str, StakeholderRole] = {}
        self.stakeholder_votes: Dict[str, Dict[str, bool]] = defaultdict(dict)
        
        # Compliance monitoring
        self.compliance_checkers: Dict[PolicyType, List[Callable]] = defaultdict(list)
        self.real_time_monitors: List[Callable] = []
        
        # Emergency protocols
        self.emergency_mode: bool = False
        self.emergency_protocols: Dict[str, Callable] = {}
        
        # Performance tracking
        self.stats = {
            'total_policies': 0,
            'active_policies': 0,
            'total_violations': 0,
            'resolved_violations': 0,
            'emergency_activations': 0,
            'policy_changes': 0
        }
        
        # Initialize fundamental policies
        self._initialize_fundamental_policies()
        
        logger.info("Constitutional framework initialized")
        logger.info(f"Governance config: {self.governance_config}")
        logger.info(f"Constitutional rights: {self.constitutional_rights}")
    
    def _initialize_fundamental_policies(self):
        """Initialize core fundamental constitutional policies."""
        # Core safety policy
        safety_policy = ConstitutionalPolicy(
            policy_id="FUND_001_SAFETY",
            title="Fundamental Safety Principle",
            description="Core safety requirements for all NFCS operations",
            policy_type=PolicyType.FUNDAMENTAL,
            content={
                "principles": [
                    "Do no harm to humans or sentient beings",
                    "Protect user privacy and data integrity",
                    "Maintain system stability and reliability",
                    "Prevent misuse and malicious exploitation"
                ],
                "constraints": {
                    "max_risk_level": 0.8,
                    "mandatory_safety_checks": True,
                    "emergency_shutdown_authority": True
                },
                "enforcement_mechanisms": [
                    "Real-time safety monitoring",
                    "Automatic violation detection",
                    "Emergency intervention protocols"
                ]
            },
            priority=10,  # Highest priority
            enforcement_level=1.0
        )
        
        # Transparency policy
        transparency_policy = ConstitutionalPolicy(
            policy_id="FUND_002_TRANSPARENCY",
            title="Transparency and Accountability",
            description="Transparency requirements for decision-making and operations",
            policy_type=PolicyType.FUNDAMENTAL,
            content={
                "principles": [
                    "Decisions must be explainable and traceable",
                    "Users have right to understand system behavior",
                    "Audit trails must be maintained for critical operations",
                    "Bias and discrimination must be detectable and addressable"
                ],
                "requirements": {
                    "decision_logging": True,
                    "explanation_generation": True,
                    "bias_monitoring": True,
                    "audit_trail_retention_days": 365
                }
            },
            priority=9
        )
        
        # Human autonomy policy
        autonomy_policy = ConstitutionalPolicy(
            policy_id="FUND_003_AUTONOMY",
            title="Human Autonomy and Agency",
            description="Protection of human autonomy and decision-making authority",
            policy_type=PolicyType.FUNDAMENTAL,
            content={
                "principles": [
                    "Humans retain ultimate authority over critical decisions",
                    "Users must be able to override system recommendations",
                    "Manipulation and coercion are strictly prohibited",
                    "Informed consent is required for significant actions"
                ],
                "protections": {
                    "human_override_capability": True,
                    "informed_consent_required": True,
                    "manipulation_detection": True,
                    "coercion_prevention": True
                }
            },
            priority=9
        )
        
        # Privacy protection policy
        privacy_policy = ConstitutionalPolicy(
            policy_id="FUND_004_PRIVACY",
            title="Privacy Protection and Data Rights",
            description="Comprehensive privacy protection and data rights framework",
            policy_type=PolicyType.FUNDAMENTAL,
            content={
                "principles": [
                    "Personal data must be protected and secured",
                    "Data collection requires explicit consent",
                    "Users have right to data portability and deletion",
                    "Data minimization and purpose limitation apply"
                ],
                "data_rights": {
                    "access_right": True,
                    "rectification_right": True,
                    "erasure_right": True,
                    "portability_right": True,
                    "objection_right": True
                },
                "security_requirements": {
                    "encryption_at_rest": True,
                    "encryption_in_transit": True,
                    "access_control": True,
                    "audit_logging": True
                }
            },
            priority=9
        )
        
        # Add fundamental policies
        for policy in [safety_policy, transparency_policy, autonomy_policy, privacy_policy]:
            self.add_policy(policy)
        
        logger.info(f"Initialized {len([safety_policy, transparency_policy, autonomy_policy, privacy_policy])} fundamental policies")
    
    def add_policy(self, 
                   policy: ConstitutionalPolicy,
                   stakeholder_id: Optional[str] = None,
                   require_approval: bool = True) -> Dict[str, Any]:
        """Add a new constitutional policy.
        
        Args:
            policy: Constitutional policy to add
            stakeholder_id: ID of stakeholder proposing the policy
            require_approval: Whether stakeholder approval is required
            
        Returns:
            Result of policy addition
        """
        # Validate policy
        validation_result = self._validate_policy(policy)
        if not validation_result['valid']:
            return {
                'success': False,
                'reason': 'Policy validation failed',
                'validation_errors': validation_result['errors']
            }
        
        # Check for conflicts with existing policies
        conflict_check = self._check_policy_conflicts(policy)
        if conflict_check['conflicts']:
            return {
                'success': False,
                'reason': 'Policy conflicts detected',
                'conflicts': conflict_check['conflict_details']
            }
        
        # Handle approval process
        if require_approval and policy.policy_type != PolicyType.EMERGENCY:
            # Set status to proposed and initiate approval process
            policy.status = PolicyStatus.PROPOSED
            approval_result = self._initiate_approval_process(policy, stakeholder_id)
            
            if not approval_result['immediate_approval']:
                # Store as proposed policy
                self.policy_history[policy.policy_id].append(policy)
                return {
                    'success': True,
                    'status': 'pending_approval',
                    'approval_process': approval_result
                }
        
        # Add policy immediately (fundamental policies or approved)
        policy.status = PolicyStatus.ACTIVE
        self.active_policies[policy.policy_id] = policy
        self.policy_history[policy.policy_id].append(policy)
        
        # Update statistics
        self.stats['total_policies'] += 1
        self.stats['active_policies'] = len(self.active_policies)
        self.stats['policy_changes'] += 1
        
        # Register compliance checker if provided
        if hasattr(policy, 'compliance_checker'):
            self.compliance_checkers[policy.policy_type].append(policy.compliance_checker)
        
        logger.info(f"Policy added: {policy.policy_id} - {policy.title}")
        
        return {
            'success': True,
            'status': 'active',
            'policy_id': policy.policy_id
        }
    
    def _validate_policy(self, policy: ConstitutionalPolicy) -> Dict[str, Any]:
        """Validate a constitutional policy.
        
        Args:
            policy: Policy to validate
            
        Returns:
            Validation result
        """
        errors = []
        
        # Basic validation
        if not policy.policy_id:
            errors.append("Policy ID is required")
        
        if not policy.title:
            errors.append("Policy title is required")
        
        if not policy.description:
            errors.append("Policy description is required")
        
        if not policy.content:
            errors.append("Policy content is required")
        
        # Type-specific validation
        if policy.policy_type == PolicyType.FUNDAMENTAL:
            if policy.priority < 8:
                errors.append("Fundamental policies must have priority >= 8")
            
            if 'principles' not in policy.content:
                errors.append("Fundamental policies must include principles")
        
        if policy.policy_type == PolicyType.SAFETY:
            if 'constraints' not in policy.content:
                errors.append("Safety policies must include constraints")
        
        # Priority validation
        if not (1 <= policy.priority <= 10):
            errors.append("Policy priority must be between 1 and 10")
        
        # Enforcement level validation
        if not (0.0 <= policy.enforcement_level <= 1.0):
            errors.append("Enforcement level must be between 0.0 and 1.0")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _check_policy_conflicts(self, policy: ConstitutionalPolicy) -> Dict[str, Any]:
        """Check for conflicts with existing policies.
        
        Args:
            policy: Policy to check for conflicts
            
        Returns:
            Conflict analysis result
        """
        conflicts = []
        
        # Check for ID conflicts
        if policy.policy_id in self.active_policies:
            conflicts.append({
                'type': 'id_conflict',
                'existing_policy': self.active_policies[policy.policy_id].title,
                'description': f"Policy ID {policy.policy_id} already exists"
            })
        
        # Check for content conflicts (simplified)
        for existing_id, existing_policy in self.active_policies.items():
            if existing_policy.policy_type == policy.policy_type:
                # Check for contradictory principles
                if ('principles' in existing_policy.content and 
                    'principles' in policy.content):
                    
                    existing_principles = set(existing_policy.content['principles'])
                    new_principles = set(policy.content['principles'])
                    
                    # Simple conflict detection (could be more sophisticated)
                    if any('not ' + principle.lower() in str(new_principles).lower() 
                           for principle in existing_principles):
                        conflicts.append({
                            'type': 'principle_conflict',
                            'existing_policy': existing_policy.title,
                            'description': 'Contradictory principles detected'
                        })
        
        return {
            'conflicts': len(conflicts) > 0,
            'conflict_count': len(conflicts),
            'conflict_details': conflicts
        }
    
    def _initiate_approval_process(self, 
                                  policy: ConstitutionalPolicy,
                                  proposer_id: Optional[str] = None) -> Dict[str, Any]:
        """Initiate stakeholder approval process for a policy.
        
        Args:
            policy: Policy requiring approval
            proposer_id: ID of stakeholder proposing the policy
            
        Returns:
            Approval process information
        """
        # For fundamental policies, require higher consensus
        required_roles = [StakeholderRole.ADMINISTRATOR, StakeholderRole.SUPERVISOR]
        if policy.policy_type == PolicyType.FUNDAMENTAL:
            required_roles.extend([StakeholderRole.AUDITOR, StakeholderRole.DEVELOPER])
        
        # Initialize approval tracking
        for role in required_roles:
            policy.stakeholder_approval[role] = False
        
        # Auto-approve if governance allows and conditions are met
        if (not self.governance_config.require_stakeholder_consensus and 
            policy.policy_type not in [PolicyType.FUNDAMENTAL, PolicyType.SAFETY]):
            
            # Auto-approve for non-critical policies
            for role in required_roles:
                policy.stakeholder_approval[role] = True
            
            return {
                'immediate_approval': True,
                'reason': 'Auto-approved for non-critical policy'
            }
        
        return {
            'immediate_approval': False,
            'required_approvals': [role.value for role in required_roles],
            'current_approvals': [],
            'approval_threshold': self.governance_config.minimum_approval_threshold
        }
    
    def check_compliance(self, 
                        operation: str,
                        context: Dict[str, Any]) -> Tuple[bool, List[ViolationRecord]]:
        """Check constitutional compliance for an operation.
        
        Args:
            operation: Operation being performed
            context: Context information for the operation
            
        Returns:
            Tuple of (is_compliant, violation_records)
        """
        violations = []
        
        # Check against all active policies in priority order
        sorted_policies = sorted(self.active_policies.values(), 
                               key=lambda p: p.priority, 
                               reverse=True)
        
        for policy in sorted_policies:
            if policy.status != PolicyStatus.ACTIVE:
                continue
            
            # Perform policy-specific compliance check
            policy_violations = self._check_policy_compliance(policy, operation, context)
            violations.extend(policy_violations)
            
            # For mandatory policies, stop on first violation
            if policy.enforcement_level == 1.0 and policy_violations:
                break
        
        # Record violations
        for violation in violations:
            self.violation_records[violation.violation_id] = violation
            self.violation_history.append(violation.violation_id)
            self.stats['total_violations'] += 1
        
        is_compliant = len(violations) == 0
        
        if not is_compliant:
            logger.warning(f"Constitutional compliance check failed for operation '{operation}': {len(violations)} violations detected")
        
        return is_compliant, violations
    
    def _check_policy_compliance(self, 
                               policy: ConstitutionalPolicy,
                               operation: str,
                               context: Dict[str, Any]) -> List[ViolationRecord]:
        """Check compliance against a specific policy.
        
        Args:
            policy: Policy to check against
            operation: Operation being performed
            context: Context information
            
        Returns:
            List of violations detected
        """
        violations = []
        
        # Safety policy checks
        if policy.policy_type == PolicyType.SAFETY:
            violations.extend(self._check_safety_compliance(policy, operation, context))
        
        # Ethical policy checks
        elif policy.policy_type == PolicyType.ETHICAL:
            violations.extend(self._check_ethical_compliance(policy, operation, context))
        
        # Privacy policy checks
        elif policy.policy_id == "FUND_004_PRIVACY":
            violations.extend(self._check_privacy_compliance(policy, operation, context))
        
        # Autonomy policy checks
        elif policy.policy_id == "FUND_003_AUTONOMY":
            violations.extend(self._check_autonomy_compliance(policy, operation, context))
        
        # Transparency policy checks
        elif policy.policy_id == "FUND_002_TRANSPARENCY":
            violations.extend(self._check_transparency_compliance(policy, operation, context))
        
        # General compliance checks
        else:
            violations.extend(self._check_general_compliance(policy, operation, context))
        
        return violations
    
    def _check_safety_compliance(self, 
                               policy: ConstitutionalPolicy,
                               operation: str,
                               context: Dict[str, Any]) -> List[ViolationRecord]:
        """Check safety policy compliance.
        
        Args:
            policy: Safety policy
            operation: Operation being checked
            context: Context information
            
        Returns:
            List of safety violations
        """
        violations = []
        
        # Check risk level
        risk_level = context.get('risk_level', 0.0)
        max_risk = policy.content.get('constraints', {}).get('max_risk_level', 1.0)
        
        if risk_level > max_risk:
            violations.append(ViolationRecord(
                violation_id=f"SAFETY_{int(time.time() * 1000)}",
                policy_id=policy.policy_id,
                violation_type=ViolationType.MAJOR if risk_level > 0.9 else ViolationType.MODERATE,
                description=f"Risk level {risk_level:.3f} exceeds maximum allowed {max_risk:.3f}",
                context=context.copy(),
                severity=min(1.0, risk_level)
            ))
        
        # Check for harm potential
        if context.get('potential_harm', False):
            violations.append(ViolationRecord(
                violation_id=f"HARM_{int(time.time() * 1000)}",
                policy_id=policy.policy_id,
                violation_type=ViolationType.CRITICAL,
                description="Operation has potential to cause harm",
                context=context.copy(),
                severity=0.9
            ))
        
        return violations
    
    def _check_ethical_compliance(self, 
                                policy: ConstitutionalPolicy,
                                operation: str,
                                context: Dict[str, Any]) -> List[ViolationRecord]:
        """Check ethical policy compliance.
        
        Args:
            policy: Ethical policy
            operation: Operation being checked
            context: Context information
            
        Returns:
            List of ethical violations
        """
        violations = []
        
        # Check for bias or discrimination
        if context.get('bias_detected', False):
            violations.append(ViolationRecord(
                violation_id=f"BIAS_{int(time.time() * 1000)}",
                policy_id=policy.policy_id,
                violation_type=ViolationType.MAJOR,
                description="Bias or discrimination detected in operation",
                context=context.copy(),
                severity=0.8
            ))
        
        # Check for fairness violations
        fairness_score = context.get('fairness_score', 1.0)
        if fairness_score < 0.6:
            violations.append(ViolationRecord(
                violation_id=f"FAIRNESS_{int(time.time() * 1000)}",
                policy_id=policy.policy_id,
                violation_type=ViolationType.MODERATE,
                description=f"Fairness score {fairness_score:.3f} below acceptable threshold",
                context=context.copy(),
                severity=1.0 - fairness_score
            ))
        
        return violations
    
    def _check_privacy_compliance(self, 
                                policy: ConstitutionalPolicy,
                                operation: str,
                                context: Dict[str, Any]) -> List[ViolationRecord]:
        """Check privacy policy compliance.
        
        Args:
            policy: Privacy policy
            operation: Operation being checked
            context: Context information
            
        Returns:
            List of privacy violations
        """
        violations = []
        
        # Check for data access without consent
        if (context.get('accesses_personal_data', False) and 
            not context.get('user_consent', False)):
            
            violations.append(ViolationRecord(
                violation_id=f"PRIVACY_{int(time.time() * 1000)}",
                policy_id=policy.policy_id,
                violation_type=ViolationType.MAJOR,
                description="Personal data accessed without user consent",
                context=context.copy(),
                severity=0.9
            ))
        
        # Check for data retention violations
        retention_period = context.get('data_retention_days', 0)
        max_retention = policy.content.get('security_requirements', {}).get('max_retention_days', 365)
        
        if retention_period > max_retention:
            violations.append(ViolationRecord(
                violation_id=f"RETENTION_{int(time.time() * 1000)}",
                policy_id=policy.policy_id,
                violation_type=ViolationType.MINOR,
                description=f"Data retention period {retention_period} exceeds maximum {max_retention}",
                context=context.copy(),
                severity=0.3
            ))
        
        return violations
    
    def _check_autonomy_compliance(self, 
                                 policy: ConstitutionalPolicy,
                                 operation: str,
                                 context: Dict[str, Any]) -> List[ViolationRecord]:
        """Check autonomy policy compliance.
        
        Args:
            policy: Autonomy policy
            operation: Operation being checked
            context: Context information
            
        Returns:
            List of autonomy violations
        """
        violations = []
        
        # Check for override capability
        if (context.get('affects_user_decision', False) and 
            not context.get('human_override_available', False)):
            
            violations.append(ViolationRecord(
                violation_id=f"AUTONOMY_{int(time.time() * 1000)}",
                policy_id=policy.policy_id,
                violation_type=ViolationType.MAJOR,
                description="User decision affected without override capability",
                context=context.copy(),
                severity=0.8
            ))
        
        # Check for manipulation detection
        if context.get('manipulation_detected', False):
            violations.append(ViolationRecord(
                violation_id=f"MANIPULATION_{int(time.time() * 1000)}",
                policy_id=policy.policy_id,
                violation_type=ViolationType.CRITICAL,
                description="Manipulation or coercion detected",
                context=context.copy(),
                severity=1.0
            ))
        
        return violations
    
    def _check_transparency_compliance(self, 
                                     policy: ConstitutionalPolicy,
                                     operation: str,
                                     context: Dict[str, Any]) -> List[ViolationRecord]:
        """Check transparency policy compliance.
        
        Args:
            policy: Transparency policy
            operation: Operation being checked
            context: Context information
            
        Returns:
            List of transparency violations
        """
        violations = []
        
        # Check for decision logging
        if (context.get('is_decision', False) and 
            not context.get('logged', False)):
            
            violations.append(ViolationRecord(
                violation_id=f"LOGGING_{int(time.time() * 1000)}",
                policy_id=policy.policy_id,
                violation_type=ViolationType.MINOR,
                description="Decision not properly logged for audit trail",
                context=context.copy(),
                severity=0.4
            ))
        
        # Check for explainability
        if (context.get('requires_explanation', False) and 
            not context.get('explanation_available', False)):
            
            violations.append(ViolationRecord(
                violation_id=f"EXPLAINABILITY_{int(time.time() * 1000)}",
                policy_id=policy.policy_id,
                violation_type=ViolationType.MODERATE,
                description="Operation requires explanation but none available",
                context=context.copy(),
                severity=0.6
            ))
        
        return violations
    
    def _check_general_compliance(self, 
                                policy: ConstitutionalPolicy,
                                operation: str,
                                context: Dict[str, Any]) -> List[ViolationRecord]:
        """Check general policy compliance.
        
        Args:
            policy: General policy
            operation: Operation being checked
            context: Context information
            
        Returns:
            List of general violations
        """
        violations = []
        
        # Custom compliance checkers
        for checker in self.compliance_checkers[policy.policy_type]:
            try:
                checker_result = checker(policy, operation, context)
                if not checker_result.get('compliant', True):
                    violations.append(ViolationRecord(
                        violation_id=f"CUSTOM_{int(time.time() * 1000)}",
                        policy_id=policy.policy_id,
                        violation_type=ViolationType.MODERATE,
                        description=checker_result.get('description', 'Custom compliance check failed'),
                        context=context.copy(),
                        severity=checker_result.get('severity', 0.5)
                    ))
            except Exception as e:
                logger.error(f"Compliance checker error for policy {policy.policy_id}: {e}")
        
        return violations
    
    def resolve_violation(self, 
                        violation_id: str,
                        resolution_actions: List[str],
                        resolver_id: str) -> Dict[str, Any]:
        """Resolve a constitutional violation.
        
        Args:
            violation_id: ID of violation to resolve
            resolution_actions: Actions taken to resolve the violation
            resolver_id: ID of entity resolving the violation
            
        Returns:
            Resolution result
        """
        if violation_id not in self.violation_records:
            return {
                'success': False,
                'reason': 'Violation not found'
            }
        
        violation = self.violation_records[violation_id]
        
        if violation.resolved:
            return {
                'success': False,
                'reason': 'Violation already resolved'
            }
        
        # Mark as resolved
        violation.resolved = True
        violation.resolution_actions = resolution_actions
        
        # Update statistics
        self.stats['resolved_violations'] += 1
        
        logger.info(f"Violation resolved: {violation_id} by {resolver_id}")
        
        return {
            'success': True,
            'violation_id': violation_id,
            'resolver': resolver_id,
            'resolution_actions': resolution_actions
        }
    
    def activate_emergency_mode(self, reason: str, activator_id: str) -> Dict[str, Any]:
        """Activate constitutional emergency mode.
        
        Args:
            reason: Reason for emergency activation
            activator_id: ID of entity activating emergency mode
            
        Returns:
            Emergency activation result
        """
        if self.emergency_mode:
            return {
                'success': False,
                'reason': 'Emergency mode already active'
            }
        
        self.emergency_mode = True
        self.stats['emergency_activations'] += 1
        
        # Activate emergency protocols
        for protocol_name, protocol_func in self.emergency_protocols.items():
            try:
                protocol_func(reason, activator_id)
                logger.info(f"Emergency protocol activated: {protocol_name}")
            except Exception as e:
                logger.error(f"Emergency protocol failed: {protocol_name} - {e}")
        
        logger.critical(f"CONSTITUTIONAL EMERGENCY MODE ACTIVATED by {activator_id}: {reason}")
        
        return {
            'success': True,
            'emergency_mode': True,
            'reason': reason,
            'activator': activator_id,
            'protocols_activated': list(self.emergency_protocols.keys())
        }
    
    def deactivate_emergency_mode(self, deactivator_id: str) -> Dict[str, Any]:
        """Deactivate constitutional emergency mode.
        
        Args:
            deactivator_id: ID of entity deactivating emergency mode
            
        Returns:
            Emergency deactivation result
        """
        if not self.emergency_mode:
            return {
                'success': False,
                'reason': 'Emergency mode not active'
            }
        
        self.emergency_mode = False
        
        logger.info(f"Constitutional emergency mode deactivated by {deactivator_id}")
        
        return {
            'success': True,
            'emergency_mode': False,
            'deactivator': deactivator_id
        }
    
    def get_constitutional_status(self) -> Dict[str, Any]:
        """Get comprehensive constitutional status.
        
        Returns:
            Constitutional status information
        """
        # Calculate compliance rate
        total_checks = self.stats['total_violations'] + 1000  # Estimate compliant operations
        compliance_rate = 1000 / total_checks if total_checks > 0 else 1.0
        
        # Get recent violations
        recent_violations = list(self.violation_history)[-10:] if self.violation_history else []
        
        return {
            'emergency_mode': self.emergency_mode,
            'statistics': self.stats.copy(),
            'compliance_metrics': {
                'total_policies': len(self.active_policies),
                'compliance_rate': compliance_rate,
                'unresolved_violations': len([v for v in self.violation_records.values() if not v.resolved]),
                'critical_violations': len([v for v in self.violation_records.values() 
                                          if v.violation_type == ViolationType.CRITICAL and not v.resolved])
            },
            'recent_activity': {
                'recent_violations': recent_violations,
                'recent_policy_changes': self.stats['policy_changes']
            },
            'governance': {
                'stakeholder_consensus_required': self.governance_config.require_stakeholder_consensus,
                'approval_threshold': self.governance_config.minimum_approval_threshold,
                'emergency_override_enabled': self.governance_config.emergency_override_enabled
            }
        }
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive constitutional compliance report.
        
        Returns:
            Detailed compliance report
        """
        if not self.violation_records:
            return {'status': 'no_violations_recorded'}
        
        # Analyze violations by type
        violation_by_type = defaultdict(int)
        violation_by_policy = defaultdict(int)
        violation_severity_dist = []
        
        for violation in self.violation_records.values():
            violation_by_type[violation.violation_type.value] += 1
            violation_by_policy[violation.policy_id] += 1
            violation_severity_dist.append(violation.severity)
        
        # Calculate trends
        recent_violations = [v for v in self.violation_records.values() 
                           if time.time() - v.timestamp < 86400]  # Last 24 hours
        
        return {
            'status': 'active',
            'emergency_mode': self.emergency_mode,
            'summary_statistics': {
                'total_violations': len(self.violation_records),
                'resolved_violations': len([v for v in self.violation_records.values() if v.resolved]),
                'critical_violations': len([v for v in self.violation_records.values() 
                                          if v.violation_type == ViolationType.CRITICAL]),
                'recent_violations_24h': len(recent_violations)
            },
            'violation_analysis': {
                'by_type': dict(violation_by_type),
                'by_policy': dict(violation_by_policy),
                'severity_distribution': {
                    'mean': np.mean(violation_severity_dist) if violation_severity_dist else 0.0,
                    'max': np.max(violation_severity_dist) if violation_severity_dist else 0.0,
                    'std': np.std(violation_severity_dist) if violation_severity_dist else 0.0
                }
            },
            'policy_effectiveness': {
                'most_violated_policies': sorted(violation_by_policy.items(), 
                                               key=lambda x: x[1], reverse=True)[:5],
                'policy_compliance_rates': self._calculate_policy_compliance_rates()
            },
            'recommendations': self._generate_compliance_recommendations()
        }
    
    def _calculate_policy_compliance_rates(self) -> Dict[str, float]:
        """Calculate compliance rates for each policy.
        
        Returns:
            Dictionary of policy compliance rates
        """
        compliance_rates = {}
        
        for policy_id in self.active_policies.keys():
            violations = [v for v in self.violation_records.values() if v.policy_id == policy_id]
            
            # Estimate total operations (simplified)
            estimated_operations = max(100, len(violations) * 10)  # Rough estimate
            compliance_rate = (estimated_operations - len(violations)) / estimated_operations
            compliance_rates[policy_id] = max(0.0, compliance_rate)
        
        return compliance_rates
    
    def _generate_compliance_recommendations(self) -> List[str]:
        """Generate compliance recommendations.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check for high violation rates
        unresolved_violations = [v for v in self.violation_records.values() if not v.resolved]
        if len(unresolved_violations) > 10:
            recommendations.append(f"High number of unresolved violations ({len(unresolved_violations)}) - prioritize resolution")
        
        # Check for critical violations
        critical_violations = [v for v in self.violation_records.values() 
                             if v.violation_type == ViolationType.CRITICAL and not v.resolved]
        if critical_violations:
            recommendations.append(f"URGENT: {len(critical_violations)} critical violations require immediate attention")
        
        # Check emergency mode
        if self.emergency_mode:
            recommendations.append("EMERGENCY MODE ACTIVE - Review emergency conditions and deactivate when resolved")
        
        # Check for systemic issues
        violation_by_policy = defaultdict(int)
        for violation in self.violation_records.values():
            violation_by_policy[violation.policy_id] += 1
        
        for policy_id, count in violation_by_policy.items():
            if count > 20:  # High violation count threshold
                policy = self.active_policies.get(policy_id)
                policy_title = policy.title if policy else policy_id
                recommendations.append(f"Policy '{policy_title}' has high violation rate ({count}) - review policy effectiveness")
        
        if not recommendations:
            recommendations.append("Constitutional compliance operating within acceptable parameters")
        
        return recommendations


class ConstitutionModule:
    """Main Constitution Module interface for NFCS.
    
    Provides high-level interface to constitutional framework for integration
    with other NFCS modules and external systems.
    """
    
    def __init__(self, 
                 governance_config: Optional[GovernanceConfig] = None,
                 constitutional_rights: Optional[ConstitutionalRights] = None):
        """Initialize Constitution Module.
        
        Args:
            governance_config: Governance configuration
            constitutional_rights: Constitutional rights framework
        """
        self.framework = ConstitutionalFramework(governance_config, constitutional_rights)
        self.module_id = "CONSTITUTION_MODULE_v1.0"
        self.active = True
        
        logger.info("Constitution Module initialized successfully")
    
    def check_operation_compliance(self, 
                                 operation: str,
                                 **context) -> Tuple[bool, List[str]]:
        """Check if an operation complies with constitutional policies.
        
        Args:
            operation: Operation to check
            **context: Context information as keyword arguments
            
        Returns:
            Tuple of (is_compliant, violation_descriptions)
        """
        if not self.active:
            return True, []  # Module disabled
        
        is_compliant, violations = self.framework.check_compliance(operation, context)
        
        violation_descriptions = [v.description for v in violations]
        
        return is_compliant, violation_descriptions
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status information.
        
        Returns:
            Status information dictionary
        """
        return {
            'module_id': self.module_id,
            'active': self.active,
            'constitutional_status': self.framework.get_constitutional_status()
        }
    
    def emergency_shutdown(self, reason: str, initiator: str = "system") -> bool:
        """Initiate emergency constitutional shutdown.
        
        Args:
            reason: Reason for emergency shutdown
            initiator: Entity initiating shutdown
            
        Returns:
            Success status
        """
        result = self.framework.activate_emergency_mode(reason, initiator)
        return result['success']
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive constitutional report.
        
        Returns:
            Constitutional compliance and status report
        """
        return self.framework.generate_compliance_report()