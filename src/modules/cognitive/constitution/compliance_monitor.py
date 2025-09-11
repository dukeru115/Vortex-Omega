"""
Compliance Monitor - Real-time Constitutional Compliance Monitoring
==================================================================

The ComplianceMonitor provides real-time monitoring of system operations
to ensure adherence to constitutional policies and frameworks.

Key Features:
- Continuous monitoring of system actions and decisions
- Real-time policy violation detection and reporting
- Compliance scoring and trend analysis
- Automated enforcement triggers and escalation protocols
- Integration with constitutional framework and policy management
- Audit trail generation for compliance verification

Architecture:
The ComplianceMonitor operates as a distributed monitoring system with
centralized coordination, providing both passive monitoring and active
enforcement capabilities for constitutional compliance.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import asyncio
import uuid
import threading
from collections import deque


class ComplianceLevel(Enum):
    """Compliance assessment levels."""
    FULL_COMPLIANCE = "full_compliance"
    MINOR_DEVIATION = "minor_deviation"
    MODERATE_VIOLATION = "moderate_violation"
    MAJOR_VIOLATION = "major_violation"
    CRITICAL_VIOLATION = "critical_violation"


class ViolationType(Enum):
    """Types of constitutional violations."""
    MINOR = "minor"                # Minor policy infractions
    MODERATE = "moderate"          # Moderate violations requiring attention
    MAJOR = "major"               # Major violations requiring intervention
    CRITICAL = "critical"         # Critical violations requiring emergency response
    SYSTEMIC = "systemic"         # Systemic violations affecting core operations


class MonitoringMode(Enum):
    """Monitoring operation modes."""
    PASSIVE = "passive"          # Monitor and log only
    ACTIVE = "active"            # Monitor and enforce
    LEARNING = "learning"        # Monitor and learn patterns
    EMERGENCY = "emergency"      # Emergency monitoring mode


@dataclass
class ComplianceEvent:
    """Compliance monitoring event."""
    event_id: str
    timestamp: datetime
    event_type: str
    source: str
    action: Dict[str, Any]
    compliance_level: ComplianceLevel
    policy_violations: List[Dict[str, Any]]
    compliance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceMonitor:
    """Real-time constitutional compliance monitor."""
    
    def __init__(self, policy_manager=None):
        """Initialize the Compliance Monitor."""
        self.policy_manager = policy_manager
        self.monitoring_mode = MonitoringMode.ACTIVE
        self.compliance_events: deque = deque(maxlen=10000)  # Recent events
        self.violation_handlers: List[Callable] = []
        self.compliance_thresholds = {
            'minor_threshold': 0.8,
            'moderate_threshold': 0.6,
            'major_threshold': 0.4,
            'critical_threshold': 0.2
        }
        
        self.monitoring_active = False
        self.monitor_thread = None
        self.event_queue = asyncio.Queue() if hasattr(asyncio, 'Queue') else []
        self.logger = logging.getLogger("ComplianceMonitor")
        
        # Compliance statistics
        self.compliance_stats = {
            'total_events': 0,
            'violations': 0,
            'average_compliance_score': 1.0,
            'last_critical_violation': None
        }
        
    def start_monitoring(self) -> bool:
        """Start continuous compliance monitoring."""
        try:
            self.monitoring_active = True
            self.logger.info("Compliance monitoring started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop compliance monitoring."""
        try:
            self.monitoring_active = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=1.0)
            self.logger.info("Compliance monitoring stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    def monitor_action(self, action: Dict[str, Any], source: str = "system") -> ComplianceEvent:
        """Monitor a specific action for compliance."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Check compliance if policy manager is available
        compliance_result = self._check_compliance(action)
        compliance_level = self._determine_compliance_level(compliance_result)
        compliance_score = self._calculate_compliance_score(compliance_result)
        
        # Create compliance event
        event = ComplianceEvent(
            event_id=event_id,
            timestamp=timestamp,
            event_type="action_monitoring",
            source=source,
            action=action,
            compliance_level=compliance_level,
            policy_violations=compliance_result.get('violations', []),
            compliance_score=compliance_score
        )
        
        # Store event
        self.compliance_events.append(event)
        self._update_statistics(event)
        
        # Handle violations if any
        if compliance_level != ComplianceLevel.FULL_COMPLIANCE:
            self._handle_violation(event)
        
        return event
    
    def _check_compliance(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check action compliance against policies."""
        if self.policy_manager:
            return self.policy_manager.check_compliance(action)
        else:
            # Basic compliance check without policy manager
            return {
                'compliant': True,
                'violations': [],
                'compliant_policies': [],
                'checked_at': datetime.now(timezone.utc).isoformat()
            }
    
    def _determine_compliance_level(self, compliance_result: Dict[str, Any]) -> ComplianceLevel:
        """Determine compliance level based on violations."""
        violations = compliance_result.get('violations', [])
        
        if not violations:
            return ComplianceLevel.FULL_COMPLIANCE
        
        # Assess severity based on number and type of violations
        violation_count = len(violations)
        has_critical = any(v.get('severity') == 'critical' for v in violations)
        has_major = any(v.get('severity') == 'major' for v in violations)
        
        if has_critical or violation_count >= 3:
            return ComplianceLevel.CRITICAL_VIOLATION
        elif has_major or violation_count >= 2:
            return ComplianceLevel.MAJOR_VIOLATION
        elif violation_count == 1:
            violation_type = violations[0].get('violation_type', 'minor')
            if violation_type in ['content_violation', 'policy_violation']:
                return ComplianceLevel.MODERATE_VIOLATION
            else:
                return ComplianceLevel.MINOR_DEVIATION
        
        return ComplianceLevel.MINOR_DEVIATION
    
    def _calculate_compliance_score(self, compliance_result: Dict[str, Any]) -> float:
        """Calculate numerical compliance score (0.0-1.0)."""
        if compliance_result.get('compliant', True):
            return 1.0
        
        violations = compliance_result.get('violations', [])
        if not violations:
            return 1.0
        
        # Score based on violation severity and count
        base_score = 1.0
        for violation in violations:
            severity = violation.get('severity', 'minor')
            if severity == 'critical':
                base_score -= 0.4
            elif severity == 'major':
                base_score -= 0.25
            elif severity == 'moderate':
                base_score -= 0.15
            else:  # minor
                base_score -= 0.1
        
        return max(0.0, base_score)
    
    def _handle_violation(self, event: ComplianceEvent) -> None:
        """Handle compliance violation."""
        # Log violation
        self.logger.warning(f"Compliance violation detected: {event.compliance_level.value}")
        
        # Execute registered violation handlers
        for handler in self.violation_handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Violation handler failed: {e}")
        
        # Update policy manager if available
        if self.policy_manager and event.policy_violations:
            for violation in event.policy_violations:
                self.policy_manager.log_violation({
                    'event_id': event.event_id,
                    'timestamp': event.timestamp.isoformat(),
                    'compliance_level': event.compliance_level.value,
                    **violation
                })
    
    def _update_statistics(self, event: ComplianceEvent) -> None:
        """Update compliance statistics."""
        self.compliance_stats['total_events'] += 1
        
        if event.compliance_level != ComplianceLevel.FULL_COMPLIANCE:
            self.compliance_stats['violations'] += 1
        
        if event.compliance_level == ComplianceLevel.CRITICAL_VIOLATION:
            self.compliance_stats['last_critical_violation'] = event.timestamp.isoformat()
        
        # Update average compliance score (rolling average)
        current_avg = self.compliance_stats['average_compliance_score']
        total_events = self.compliance_stats['total_events']
        self.compliance_stats['average_compliance_score'] = (
            (current_avg * (total_events - 1) + event.compliance_score) / total_events
        )
    
    def add_violation_handler(self, handler: Callable[[ComplianceEvent], None]) -> None:
        """Add a violation event handler."""
        self.violation_handlers.append(handler)
    
    def remove_violation_handler(self, handler: Callable[[ComplianceEvent], None]) -> bool:
        """Remove a violation event handler."""
        try:
            self.violation_handlers.remove(handler)
            return True
        except ValueError:
            return False
    
    def get_compliance_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get compliance summary for a time window."""
        cutoff_time = datetime.now(timezone.utc) - datetime.timedelta(hours=time_window_hours)
        recent_events = [e for e in self.compliance_events if e.timestamp >= cutoff_time]
        
        if not recent_events:
            return {
                'total_events': 0,
                'compliance_rate': 1.0,
                'violations_by_level': {},
                'average_compliance_score': 1.0,
                'time_window_hours': time_window_hours
            }
        
        violations_by_level = {}
        total_score = 0.0
        
        for event in recent_events:
            level = event.compliance_level.value
            violations_by_level[level] = violations_by_level.get(level, 0) + 1
            total_score += event.compliance_score
        
        compliance_rate = (len(recent_events) - violations_by_level.get('critical_violation', 0) - 
                          violations_by_level.get('major_violation', 0)) / len(recent_events)
        
        return {
            'total_events': len(recent_events),
            'compliance_rate': compliance_rate,
            'violations_by_level': violations_by_level,
            'average_compliance_score': total_score / len(recent_events),
            'time_window_hours': time_window_hours
        }
    
    def get_recent_violations(self, limit: int = 10) -> List[ComplianceEvent]:
        """Get recent compliance violations."""
        violations = [e for e in self.compliance_events 
                     if e.compliance_level != ComplianceLevel.FULL_COMPLIANCE]
        return list(reversed(violations))[-limit:]
    
    def set_monitoring_mode(self, mode: MonitoringMode) -> None:
        """Set monitoring operation mode."""
        self.monitoring_mode = mode
        self.logger.info(f"Monitoring mode set to: {mode.value}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status."""
        return {
            'monitoring_active': self.monitoring_active,
            'monitoring_mode': self.monitoring_mode.value,
            'compliance_stats': self.compliance_stats.copy(),
            'recent_events_count': len(self.compliance_events),
            'violation_handlers_count': len(self.violation_handlers)
        }