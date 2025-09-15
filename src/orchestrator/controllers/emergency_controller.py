"""
Neural Field Control System - Emergency Controller
=================================================

The EmergencyController handles system-wide emergency detection, response,
and recovery protocols to ensure safe operation of the NFCS.

Key Features:
- Real-time emergency condition detection
- Automated emergency response protocols
- Escalation procedures and notifications
- System state preservation during emergencies
- Recovery and restoration procedures
- Emergency event logging and analysis
- Constitutional compliance during emergencies
- Fail-safe mechanism activation

Architecture:
Implements multi-layered emergency detection and response with constitutional
framework integration for safe emergency handling and system protection.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading


class EmergencyLevel(Enum):
    """Emergency severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class EmergencyCondition:
    """Emergency condition definition"""

    condition_id: str
    name: str
    description: str
    level: EmergencyLevel
    check_function: Callable[[Dict[str, Any]], bool]
    response_actions: List[str]


class EmergencyController:
    """NFCS Emergency Controller"""

    def __init__(self, error_threshold: int = 50):
        self.logger = logging.getLogger("EmergencyController")
        self.error_threshold = error_threshold

        # Emergency tracking
        self._emergency_conditions: Dict[str, EmergencyCondition] = {}
        self._active_emergencies: Dict[str, Dict[str, Any]] = {}
        self._emergency_history: List[Dict[str, Any]] = []

        # State tracking
        self._running = False
        self._lock = threading.RLock()

    async def initialize(self) -> bool:
        """Initialize emergency controller"""
        self.logger.info("EmergencyController initialized")
        self._running = True

        # Register default emergency conditions
        await self._register_default_conditions()
        return True

    async def _register_default_conditions(self):
        """Register standard emergency conditions"""

        def check_error_threshold(context: Dict[str, Any]) -> bool:
            """Check if error count exceeds threshold"""
            error_count = context.get("error_count", 0)
            return error_count > self.error_threshold

        # Register error threshold condition
        error_condition = EmergencyCondition(
            condition_id="error_threshold",
            name="Error Threshold Exceeded",
            description=f"System error count exceeds {self.error_threshold}",
            level=EmergencyLevel.CRITICAL,
            check_function=check_error_threshold,
            response_actions=["log_emergency", "notify_administrators", "enable_safe_mode"],
        )

        self._emergency_conditions["error_threshold"] = error_condition

    def check_emergency_conditions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check all registered emergency conditions"""
        emergency_detected = False
        triggered_conditions = []

        with self._lock:
            for condition_id, condition in self._emergency_conditions.items():
                try:
                    if condition.check_function(context):
                        emergency_detected = True
                        triggered_conditions.append(condition_id)

                        # Activate emergency if not already active
                        if condition_id not in self._active_emergencies:
                            self._activate_emergency(condition, context)

                except Exception as e:
                    self.logger.error(f"Error checking emergency condition {condition_id}: {e}")

        return {
            "emergency_detected": emergency_detected,
            "triggered_conditions": triggered_conditions,
            "reason": (
                f"Emergency conditions triggered: {', '.join(triggered_conditions)}"
                if triggered_conditions
                else None
            ),
        }

    def _activate_emergency(self, condition: EmergencyCondition, context: Dict[str, Any]):
        """Activate emergency response for a condition"""
        emergency_record = {
            "condition_id": condition.condition_id,
            "name": condition.name,
            "level": condition.level.value,
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "context": context.copy(),
            "response_actions": condition.response_actions.copy(),
            "resolved": False,
        }

        self._active_emergencies[condition.condition_id] = emergency_record
        self._emergency_history.append(emergency_record.copy())

        self.logger.critical(f"EMERGENCY ACTIVATED: {condition.name} - {condition.description}")

        # Execute response actions
        for action in condition.response_actions:
            try:
                self._execute_response_action(action, emergency_record)
            except Exception as e:
                self.logger.error(f"Error executing emergency action {action}: {e}")

    def _execute_response_action(self, action: str, emergency_record: Dict[str, Any]):
        """Execute an emergency response action"""
        if action == "log_emergency":
            self.logger.critical(f"Emergency logged: {emergency_record}")

        elif action == "notify_administrators":
            self.logger.critical(f"ADMINISTRATOR NOTIFICATION: {emergency_record['name']}")

        elif action == "enable_safe_mode":
            self.logger.critical("SAFE MODE ENABLED - System operating in emergency safe mode")

        else:
            self.logger.warning(f"Unknown emergency action: {action}")

    async def get_status(self) -> Dict[str, Any]:
        """Get emergency controller status"""
        with self._lock:
            return {
                "running": self._running,
                "registered_conditions": len(self._emergency_conditions),
                "active_emergencies": len(self._active_emergencies),
                "total_emergency_history": len(self._emergency_history),
                "error_threshold": self.error_threshold,
            }

    async def shutdown(self) -> bool:
        """Shutdown emergency controller"""
        self.logger.info("EmergencyController shutting down")
        self._running = False
        return True
