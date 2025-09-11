"""
Boundary Module Core - NFCS Boundary Management System

Implements system boundaries and safety constraints:
- Dynamic boundary detection and enforcement
- Safety constraint management and validation
- Boundary violation detection and response
- Adaptive boundary adjustment based on context
- Integration with constitutional framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class BoundaryType(Enum):
    """Types of system boundaries."""
    SAFETY = "safety"
    PERFORMANCE = "performance"
    ETHICAL = "ethical"
    RESOURCE = "resource"
    CAPABILITY = "capability"
    TEMPORAL = "temporal"


class BoundaryStatus(Enum):
    """Status of boundary constraints."""
    ACTIVE = "active"
    WARNING = "warning"
    VIOLATED = "violated"
    CRITICAL = "critical"


@dataclass
class BoundaryConstraint:
    """Represents a system boundary constraint."""
    constraint_id: str
    name: str
    boundary_type: BoundaryType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    current_value: float = 0.0
    tolerance: float = 0.1
    status: BoundaryStatus = BoundaryStatus.ACTIVE
    enforcement_level: float = 1.0
    created_at: float = field(default_factory=time.time)


class BoundaryModule:
    """
    Boundary Management Module for NFCS.
    
    Manages system boundaries, safety constraints, and boundary violations
    with integration to the constitutional framework.
    """
    
    def __init__(self):
        """Initialize Boundary Module."""
        self.constraints: Dict[str, BoundaryConstraint] = {}
        self.violation_history: List[Dict[str, Any]] = []
        self.boundary_monitors: List[callable] = []
        
        # Initialize default boundaries
        self._initialize_default_boundaries()
        
        logger.info("Boundary Module initialized")
    
    def _initialize_default_boundaries(self):
        """Initialize default system boundaries."""
        default_boundaries = [
            BoundaryConstraint("SAFETY_RISK", "Maximum Risk Level", BoundaryType.SAFETY, max_value=0.8),
            BoundaryConstraint("PERFORMANCE_MIN", "Minimum Performance", BoundaryType.PERFORMANCE, min_value=0.3),
            BoundaryConstraint("RESOURCE_CPU", "CPU Usage Limit", BoundaryType.RESOURCE, max_value=0.9),
            BoundaryConstraint("CAPABILITY_COMPLEXITY", "Maximum Complexity", BoundaryType.CAPABILITY, max_value=0.95),
        ]
        
        for constraint in default_boundaries:
            self.constraints[constraint.constraint_id] = constraint
    
    def check_boundaries(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check all boundaries against current metrics."""
        violations = []
        all_within_bounds = True
        
        for constraint_id, constraint in self.constraints.items():
            if constraint.constraint_id.lower().replace('_', '') in str(metrics).lower():
                # Simple metric matching - in production would be more sophisticated
                current_val = metrics.get(constraint_id.lower(), constraint.current_value)
                constraint.current_value = current_val
                
                violation_detected = False
                
                if constraint.min_value is not None and current_val < constraint.min_value:
                    violation_detected = True
                    constraint.status = BoundaryStatus.VIOLATED
                    
                if constraint.max_value is not None and current_val > constraint.max_value:
                    violation_detected = True
                    constraint.status = BoundaryStatus.VIOLATED
                
                if violation_detected:
                    all_within_bounds = False
                    violations.append(f"Boundary '{constraint.name}' violated: {current_val:.3f}")
                    
                    self.violation_history.append({
                        'constraint_id': constraint_id,
                        'value': current_val,
                        'timestamp': time.time(),
                        'severity': abs(current_val - (constraint.max_value or constraint.min_value or 0))
                    })
                else:
                    constraint.status = BoundaryStatus.ACTIVE
        
        return all_within_bounds, violations
    
    def get_boundary_status(self) -> Dict[str, Any]:
        """Get current boundary status."""
        return {
            'total_constraints': len(self.constraints),
            'active_constraints': len([c for c in self.constraints.values() if c.status == BoundaryStatus.ACTIVE]),
            'violated_constraints': len([c for c in self.constraints.values() if c.status == BoundaryStatus.VIOLATED]),
            'recent_violations': len(self.violation_history[-10:]),
            'constraints': {cid: {
                'name': c.name,
                'type': c.boundary_type.value,
                'status': c.status.value,
                'current_value': c.current_value,
                'min_value': c.min_value,
                'max_value': c.max_value
            } for cid, c in self.constraints.items()}
        }


@dataclass
class BoundaryConstraints:
    """Container for boundary constraint definitions."""
    safety_boundaries: Dict[str, BoundaryConstraint] = field(default_factory=dict)
    performance_boundaries: Dict[str, BoundaryConstraint] = field(default_factory=dict)
    resource_boundaries: Dict[str, BoundaryConstraint] = field(default_factory=dict)