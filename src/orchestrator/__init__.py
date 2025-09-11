"""
Neural Field Control System (NFCS) - Orchestrator Package
=======================================================

The Orchestrator package provides the central coordination and management
layer for the Neural Field Control System. It integrates all cognitive modules,
mathematical frameworks, and safety systems into a unified, production-ready system.

Key Components:
- NFCSOrchestrator: Main coordination system
- ModuleManager: Module lifecycle and registration management
- StateCoordinator: Global state synchronization
- EventSystem: Inter-module communication framework
- ConfigurationManager: Dynamic system configuration
- ResourceManager: Memory and compute resource management
- PerformanceMonitor: System-wide performance tracking
- EmergencyController: Safety and emergency protocols

Architecture:
The orchestrator follows a hierarchical control pattern with constitutional
safety frameworks ensuring all operations comply with established policies
and constraints. It provides both autonomous operation modes and manual
control interfaces for production deployment.
"""

__version__ = "1.0.0"
__author__ = "NFCS Development Team"

# Core orchestrator components
from .nfcs_orchestrator import NFCSOrchestrator
from .managers.module_manager import ModuleManager
from .managers.configuration_manager import ConfigurationManager
from .managers.resource_manager import ResourceManager
from .coordinators.state_coordinator import StateCoordinator
from .coordinators.event_system import EventSystem
from .controllers.performance_monitor import PerformanceMonitor
from .controllers.emergency_controller import EmergencyController

__all__ = [
    "NFCSOrchestrator",
    "ModuleManager", 
    "ConfigurationManager",
    "ResourceManager",
    "StateCoordinator",
    "EventSystem",
    "PerformanceMonitor",
    "EmergencyController"
]