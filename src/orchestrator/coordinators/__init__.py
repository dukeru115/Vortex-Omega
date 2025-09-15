"""
NFCS Orchestrator - Coordination Components
==========================================

Coordination components handle inter-module communication, state synchronization,
and event management for the Neural Field Control System.
"""

from .state_coordinator import StateCoordinator
from .event_system import EventSystem

__all__ = ["StateCoordinator", "EventSystem"]
