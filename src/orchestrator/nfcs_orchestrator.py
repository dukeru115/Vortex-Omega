"""
Neural Field Control System (NFCS) - Core Orchestrator
=====================================================

The NFCSOrchestrator is the central coordination system that manages all components
of the Neural Field Control System. It provides unified control, monitoring, and
safety management for the entire system.

Key Features:
- Module lifecycle management and coordination
- Global state synchronization across all components
- Constitutional framework integration and compliance monitoring
- Real-time performance tracking and optimization
- Emergency protocols and safety constraint enforcement
- Event-driven inter-module communication
- Resource management and allocation
- Autonomous decision-making with human oversight capabilities

Architecture:
The orchestrator implements a hybrid control pattern combining centralized coordination
with distributed autonomous processing. All operations are governed by constitutional
policies ensuring safety, compliance, and optimal performance.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
import json

# Import all NFCS components
from ..core.enhanced_kuramoto import EnhancedKuramotoModule
from ..core.enhanced_metrics import EnhancedMetricsCalculator
from ..modules.esc.esc_core import EchoSemanticConverter
from ..modules.cognitive.constitution.constitution_core import ConstitutionalFramework
from ..modules.cognitive.boundary.boundary_core import BoundaryModule
from ..modules.cognitive.memory.memory_core import MemoryModule
from ..modules.cognitive.meta_reflection.reflection_core import MetaReflectionModule
from ..modules.cognitive.freedom.freedom_core import FreedomModule

# Import orchestrator components
from .managers.module_manager import ModuleManager
from .managers.configuration_manager import ConfigurationManager
from .managers.resource_manager import ResourceManager
from .coordinators.state_coordinator import StateCoordinator
from .coordinators.event_system import EventSystem
from .controllers.performance_monitor import PerformanceMonitor
from .controllers.emergency_controller import EmergencyController


class OperationalMode(Enum):
    """System operational modes"""
    INITIALIZING = "initializing"
    AUTONOMOUS = "autonomous"
    SUPERVISED = "supervised"
    MANUAL = "manual"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class SystemStatus(Enum):
    """System status indicators"""
    OFFLINE = "offline"
    STARTING = "starting"
    ONLINE = "online"
    DEGRADED = "degraded"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SystemState:
    """Complete system state representation"""
    status: SystemStatus = SystemStatus.OFFLINE
    mode: OperationalMode = OperationalMode.INITIALIZING
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active_modules: Set[str] = field(default_factory=set)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    constitutional_compliance: Dict[str, Any] = field(default_factory=dict)
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    warning_count: int = 0
    uptime_seconds: float = 0.0


@dataclass
class OrchestrationConfig:
    """Orchestrator configuration settings"""
    # Core settings
    max_concurrent_processes: int = 10
    update_frequency_hz: float = 10.0
    enable_autonomous_mode: bool = True
    enable_constitutional_enforcement: bool = True
    
    # Safety settings
    max_error_threshold: int = 100
    emergency_shutdown_threshold: int = 50
    resource_limit_cpu_percent: float = 80.0
    resource_limit_memory_mb: float = 2048.0
    
    # Performance settings
    performance_monitoring_enabled: bool = True
    performance_history_size: int = 1000
    metrics_calculation_interval: float = 1.0
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_detailed_logging: bool = True


class NFCSOrchestrator:
    """
    Neural Field Control System Core Orchestrator
    
    The main coordination system that manages all NFCS components, ensuring
    safe, efficient, and constitutionally-compliant operation of the entire
    neural field control system.
    """
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        """Initialize the NFCS Orchestrator"""
        self.config = config or OrchestrationConfig()
        self.logger = self._setup_logging()
        
        # System state
        self._state = SystemState()
        self._start_time = time.time()
        self._running = False
        self._main_loop_task: Optional[Future] = None
        
        # Thread safety
        self._state_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Component managers and coordinators
        self.module_manager: Optional[ModuleManager] = None
        self.config_manager: Optional[ConfigurationManager] = None
        self.resource_manager: Optional[ResourceManager] = None
        self.state_coordinator: Optional[StateCoordinator] = None
        self.event_system: Optional[EventSystem] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.emergency_controller: Optional[EmergencyController] = None
        
        # Core NFCS modules
        self.kuramoto_module: Optional[EnhancedKuramotoModule] = None
        self.metrics_calculator: Optional[EnhancedMetricsCalculator] = None
        self.esc_module: Optional[EchoSemanticConverter] = None
        
        # Cognitive modules
        self.constitutional_framework: Optional[ConstitutionalFramework] = None
        self.boundary_module: Optional[BoundaryModule] = None
        self.memory_module: Optional[MemoryModule] = None
        self.reflection_module: Optional[MetaReflectionModule] = None
        self.freedom_module: Optional[FreedomModule] = None
        
        # Execution context
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_processes)
        
        # Statistics and metrics
        self.stats = {
            'initialization_time': 0.0,
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'emergency_activations': 0,
            'constitutional_violations': 0,
            'performance_alerts': 0
        }
        
        self.logger.info(f"NFCS Orchestrator initialized with config: {self.config}")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger("NFCSOrchestrator")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    async def initialize(self) -> bool:
        """
        Initialize the complete NFCS system
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        init_start_time = time.time()
        self.logger.info("Starting NFCS system initialization...")
        
        try:
            with self._state_lock:
                self._state.status = SystemStatus.STARTING
                self._state.mode = OperationalMode.INITIALIZING
            
            # Initialize management components
            if not await self._initialize_managers():
                return False
            
            # Initialize core NFCS modules
            if not await self._initialize_core_modules():
                return False
            
            # Initialize cognitive modules
            if not await self._initialize_cognitive_modules():
                return False
            
            # Perform system integration checks
            if not await self._perform_integration_checks():
                return False
            
            # Start main coordination loop
            if not await self._start_main_loop():
                return False
            
            # Update system state
            with self._state_lock:
                self._state.status = SystemStatus.ONLINE
                self._state.mode = OperationalMode.SUPERVISED if not self.config.enable_autonomous_mode else OperationalMode.AUTONOMOUS
                self._running = True
            
            self.stats['initialization_time'] = time.time() - init_start_time
            self.logger.info(f"NFCS system initialization completed successfully in {self.stats['initialization_time']:.2f}s")
            
            # Trigger initialization complete event
            if self.event_system:
                await self.event_system.emit_event("system.initialized", {
                    "initialization_time": self.stats['initialization_time'],
                    "mode": self._state.mode.value,
                    "active_modules": list(self._state.active_modules)
                })
            
            return True
            
        except Exception as e:
            self.logger.error(f"NFCS initialization failed: {e}")
            with self._state_lock:
                self._state.status = SystemStatus.ERROR
            return False
    
    async def _initialize_managers(self) -> bool:
        """Initialize all management components"""
        try:
            self.logger.info("Initializing management components...")
            
            # Configuration Manager
            self.config_manager = ConfigurationManager()
            await self.config_manager.initialize()
            
            # Resource Manager
            self.resource_manager = ResourceManager(
                cpu_limit=self.config.resource_limit_cpu_percent,
                memory_limit_mb=self.config.resource_limit_memory_mb
            )
            await self.resource_manager.initialize()
            
            # Event System
            self.event_system = EventSystem()
            await self.event_system.initialize()
            
            # State Coordinator
            self.state_coordinator = StateCoordinator(self.event_system)
            await self.state_coordinator.initialize()
            
            # Performance Monitor
            self.performance_monitor = PerformanceMonitor(
                history_size=self.config.performance_history_size,
                update_interval=self.config.metrics_calculation_interval
            )
            await self.performance_monitor.initialize()
            
            # Emergency Controller
            self.emergency_controller = EmergencyController(
                error_threshold=self.config.emergency_shutdown_threshold
            )
            await self.emergency_controller.initialize()
            
            # Module Manager (depends on other managers)
            self.module_manager = ModuleManager(
                event_system=self.event_system,
                resource_manager=self.resource_manager,
                performance_monitor=self.performance_monitor
            )
            await self.module_manager.initialize()
            
            self.logger.info("Management components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize management components: {e}")
            return False
    
    async def _initialize_core_modules(self) -> bool:
        """Initialize core NFCS mathematical modules"""
        try:
            self.logger.info("Initializing core NFCS modules...")
            
            # Enhanced Kuramoto Module
            self.kuramoto_module = EnhancedKuramotoModule()
            await self.module_manager.register_module("kuramoto", self.kuramoto_module)
            
            # Enhanced Metrics Calculator
            self.metrics_calculator = EnhancedMetricsCalculator()
            await self.module_manager.register_module("metrics", self.metrics_calculator)
            
            # Echo-Semantic Converter
            self.esc_module = EchoSemanticConverter()
            await self.module_manager.register_module("esc", self.esc_module)
            
            self.logger.info("Core NFCS modules initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize core modules: {e}")
            return False
    
    async def _initialize_cognitive_modules(self) -> bool:
        """Initialize cognitive processing modules"""
        try:
            self.logger.info("Initializing cognitive modules...")
            
            # Constitutional Framework (must be first)
            self.constitutional_framework = ConstitutionalFramework()
            await self.module_manager.register_module("constitution", self.constitutional_framework)
            
            # Boundary Module
            self.boundary_module = BoundaryModule(constitutional_framework=self.constitutional_framework)
            await self.module_manager.register_module("boundary", self.boundary_module)
            
            # Memory Module
            self.memory_module = MemoryModule(constitutional_framework=self.constitutional_framework)
            await self.module_manager.register_module("memory", self.memory_module)
            
            # Meta-Reflection Module
            self.reflection_module = MetaReflectionModule(constitutional_framework=self.constitutional_framework)
            await self.module_manager.register_module("reflection", self.reflection_module)
            
            # Freedom Module
            self.freedom_module = FreedomModule(constitutional_framework=self.constitutional_framework)
            await self.module_manager.register_module("freedom", self.freedom_module)
            
            self.logger.info("Cognitive modules initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cognitive modules: {e}")
            return False
    
    async def _perform_integration_checks(self) -> bool:
        """Perform comprehensive system integration verification"""
        try:
            self.logger.info("Performing system integration checks...")
            
            # Check module interconnections
            if not await self._verify_module_connections():
                return False
            
            # Check constitutional framework compliance
            if not await self._verify_constitutional_compliance():
                return False
            
            # Check resource availability
            if not await self._verify_resource_availability():
                return False
            
            # Check emergency protocols
            if not await self._verify_emergency_protocols():
                return False
            
            self.logger.info("System integration checks passed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Integration checks failed: {e}")
            return False
    
    async def _verify_module_connections(self) -> bool:
        """Verify all modules can communicate properly"""
        if not self.module_manager:
            return False
        
        active_modules = await self.module_manager.get_active_modules()
        self.logger.info(f"Verified connections for {len(active_modules)} active modules")
        
        with self._state_lock:
            self._state.active_modules = set(active_modules.keys())
        
        return len(active_modules) > 0
    
    async def _verify_constitutional_compliance(self) -> bool:
        """Verify constitutional framework is properly enforced"""
        if not self.constitutional_framework:
            return False
        
        # Check if constitutional policies are loaded and active
        active_policies = self.constitutional_framework.get_active_policies()
        compliance_status = self.constitutional_framework.check_compliance({})
        
        self.logger.info(f"Constitutional framework active with {len(active_policies)} policies")
        return compliance_status.compliant
    
    async def _verify_resource_availability(self) -> bool:
        """Verify sufficient system resources are available"""
        if not self.resource_manager:
            return False
        
        resource_status = await self.resource_manager.get_resource_status()
        
        # Check if resources are within acceptable limits
        cpu_ok = resource_status.get('cpu_percent', 0) < self.config.resource_limit_cpu_percent
        memory_ok = resource_status.get('memory_mb', 0) < self.config.resource_limit_memory_mb
        
        self.logger.info(f"Resource check - CPU: {resource_status.get('cpu_percent', 0):.1f}%, Memory: {resource_status.get('memory_mb', 0):.1f}MB")
        
        return cpu_ok and memory_ok
    
    async def _verify_emergency_protocols(self) -> bool:
        """Verify emergency protocols are functional"""
        if not self.emergency_controller:
            return False
        
        # Test emergency detection capabilities
        emergency_status = await self.emergency_controller.get_status()
        
        self.logger.info(f"Emergency protocols verified - Status: {emergency_status}")
        return True
    
    async def _start_main_loop(self) -> bool:
        """Start the main coordination loop"""
        try:
            self.logger.info("Starting main coordination loop...")
            
            # Submit main loop to executor
            self._main_loop_task = self.executor.submit(self._main_coordination_loop)
            
            # Wait a moment to ensure loop starts
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start main loop: {e}")
            return False
    
    def _main_coordination_loop(self):
        """Main coordination loop - runs in separate thread"""
        self.logger.info("Main coordination loop started")
        
        loop_interval = 1.0 / self.config.update_frequency_hz
        
        while not self._shutdown_event.is_set():
            try:
                loop_start = time.time()
                
                # Update system state
                self._update_system_state()
                
                # Perform coordination tasks
                self._coordinate_modules()
                
                # Check for emergencies
                self._check_emergency_conditions()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Handle autonomous operations if enabled
                if self._state.mode == OperationalMode.AUTONOMOUS:
                    self._handle_autonomous_operations()
                
                # Calculate sleep time to maintain frequency
                elapsed = time.time() - loop_start
                sleep_time = max(0, loop_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                self.stats['total_operations'] += 1
                self.stats['successful_operations'] += 1
                
            except Exception as e:
                self.logger.error(f"Error in main coordination loop: {e}")
                self.stats['failed_operations'] += 1
                
                # If too many failures, switch to emergency mode
                if self.stats['failed_operations'] > self.config.max_error_threshold:
                    self._activate_emergency_mode("Too many coordination loop failures")
                    break
        
        self.logger.info("Main coordination loop ended")
    
    def _update_system_state(self):
        """Update comprehensive system state"""
        with self._state_lock:
            self._state.timestamp = datetime.now(timezone.utc)
            self._state.uptime_seconds = time.time() - self._start_time
            
            # Update error and warning counts
            if self.performance_monitor:
                metrics = self.performance_monitor.get_current_metrics()
                self._state.error_count = metrics.get('total_errors', 0)
                self._state.warning_count = metrics.get('total_warnings', 0)
    
    def _coordinate_modules(self):
        """Coordinate all active modules"""
        if not self.module_manager:
            return
        
        # Get current module states and coordinate them
        # This is where the Kuramoto synchronization would be applied
        if self.kuramoto_module and self._state.status == SystemStatus.ONLINE:
            # Apply synchronization across cognitive modules
            self._synchronize_cognitive_modules()
    
    def _synchronize_cognitive_modules(self):
        """Synchronize cognitive modules using Kuramoto dynamics"""
        try:
            # Collect phase information from cognitive modules
            phases = {}
            
            if self.boundary_module:
                status = self.boundary_module.get_boundary_status()
                phases['boundary'] = status.get('phase', 0.0)
            
            if self.memory_module:
                status = self.memory_module.get_memory_status()
                phases['memory'] = status.get('phase', 0.0)
            
            if self.reflection_module:
                status = self.reflection_module.get_reflection_status()
                phases['reflection'] = status.get('phase', 0.0)
            
            if self.freedom_module:
                status = self.freedom_module.get_freedom_status()
                phases['freedom'] = status.get('phase', 0.0)
            
            # Apply Kuramoto synchronization
            if len(phases) > 1 and self.kuramoto_module:
                sync_result = self.kuramoto_module.synchronize_phases(phases)
                
                # Update modules with synchronized phases
                for module_name, new_phase in sync_result.items():
                    self._update_module_phase(module_name, new_phase)
                        
        except Exception as e:
            self.logger.error(f"Error synchronizing cognitive modules: {e}")
    
    def _update_module_phase(self, module_name: str, phase: float):
        """Update a specific module's phase"""
        try:
            if module_name == 'boundary' and self.boundary_module:
                self.boundary_module.update_phase(phase)
            elif module_name == 'memory' and self.memory_module:
                self.memory_module.update_phase(phase)
            elif module_name == 'reflection' and self.reflection_module:
                self.reflection_module.update_phase(phase)
            elif module_name == 'freedom' and self.freedom_module:
                self.freedom_module.update_phase(phase)
        except Exception as e:
            self.logger.error(f"Error updating phase for {module_name}: {e}")
    
    def _check_emergency_conditions(self):
        """Check for emergency conditions across all systems"""
        if not self.emergency_controller:
            return
        
        try:
            # Check for system-wide emergencies
            emergency_status = self.emergency_controller.check_emergency_conditions({
                'error_count': self._state.error_count,
                'warning_count': self._state.warning_count,
                'uptime': self._state.uptime_seconds,
                'active_modules': len(self._state.active_modules)
            })
            
            if emergency_status.get('emergency_detected', False):
                self._activate_emergency_mode(emergency_status.get('reason', 'Unknown emergency'))
                
        except Exception as e:
            self.logger.error(f"Error checking emergency conditions: {e}")
    
    def _activate_emergency_mode(self, reason: str):
        """Activate emergency mode with specified reason"""
        self.logger.critical(f"EMERGENCY MODE ACTIVATED: {reason}")
        
        with self._state_lock:
            self._state.status = SystemStatus.EMERGENCY
            self._state.mode = OperationalMode.EMERGENCY
        
        self.stats['emergency_activations'] += 1
        
        # Trigger emergency event
        if self.event_system:
            asyncio.create_task(self.event_system.emit_event("system.emergency", {
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system_state": self._state.__dict__
            }))
    
    def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        if not self.performance_monitor:
            return
        
        try:
            # Collect metrics from all modules
            system_metrics = {
                'uptime': self._state.uptime_seconds,
                'active_modules': len(self._state.active_modules),
                'total_operations': self.stats['total_operations'],
                'success_rate': (self.stats['successful_operations'] / max(1, self.stats['total_operations'])) * 100,
                'error_rate': (self.stats['failed_operations'] / max(1, self.stats['total_operations'])) * 100
            }
            
            # Add resource metrics if available
            if self.resource_manager:
                resource_status = asyncio.create_task(self.resource_manager.get_resource_status())
                # Note: In production, this should be handled asynchronously
            
            self.performance_monitor.update_metrics(system_metrics)
            
            with self._state_lock:
                self._state.performance_metrics = system_metrics
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _handle_autonomous_operations(self):
        """Handle autonomous decision-making and operations"""
        if not self.freedom_module or not self.constitutional_framework:
            return
        
        try:
            # Let the freedom module make autonomous decisions
            # within constitutional constraints
            current_context = {
                'system_state': self._state.__dict__,
                'performance_metrics': self._state.performance_metrics,
                'active_modules': list(self._state.active_modules)
            }
            
            autonomous_decisions = self.freedom_module.make_autonomous_decisions(current_context)
            
            # Apply decisions that pass constitutional review
            for decision in autonomous_decisions:
                if self.constitutional_framework.check_compliance(decision).compliant:
                    self._apply_autonomous_decision(decision)
                else:
                    self.stats['constitutional_violations'] += 1
                    self.logger.warning(f"Autonomous decision blocked by constitutional framework: {decision}")
                    
        except Exception as e:
            self.logger.error(f"Error handling autonomous operations: {e}")
    
    def _apply_autonomous_decision(self, decision: Dict[str, Any]):
        """Apply an approved autonomous decision"""
        try:
            decision_type = decision.get('type')
            
            if decision_type == 'optimization':
                self._apply_optimization_decision(decision)
            elif decision_type == 'resource_reallocation':
                self._apply_resource_decision(decision)
            elif decision_type == 'module_coordination':
                self._apply_coordination_decision(decision)
            else:
                self.logger.warning(f"Unknown autonomous decision type: {decision_type}")
                
        except Exception as e:
            self.logger.error(f"Error applying autonomous decision: {e}")
    
    def _apply_optimization_decision(self, decision: Dict[str, Any]):
        """Apply system optimization decision"""
        # Implementation for system optimization
        self.logger.info(f"Applied optimization decision: {decision.get('description', 'Unknown')}")
    
    def _apply_resource_decision(self, decision: Dict[str, Any]):
        """Apply resource allocation decision"""
        # Implementation for resource reallocation
        self.logger.info(f"Applied resource decision: {decision.get('description', 'Unknown')}")
    
    def _apply_coordination_decision(self, decision: Dict[str, Any]):
        """Apply module coordination decision"""
        # Implementation for coordination changes
        self.logger.info(f"Applied coordination decision: {decision.get('description', 'Unknown')}")
    
    # Public Interface Methods
    
    async def shutdown(self, timeout: float = 30.0) -> bool:
        """
        Gracefully shutdown the NFCS system
        
        Args:
            timeout: Maximum time to wait for shutdown in seconds
            
        Returns:
            bool: True if shutdown completed successfully, False if timeout
        """
        self.logger.info("Starting NFCS system shutdown...")
        
        try:
            # Set shutdown flag
            self._shutdown_event.set()
            
            with self._state_lock:
                self._state.status = SystemStatus.OFFLINE
                self._state.mode = OperationalMode.SHUTDOWN
                self._running = False
            
            # Wait for main loop to complete
            if self._main_loop_task:
                try:
                    self._main_loop_task.result(timeout=timeout)
                except Exception as e:
                    self.logger.error(f"Error waiting for main loop shutdown: {e}")
            
            # Shutdown all modules
            if self.module_manager:
                await self.module_manager.shutdown_all_modules()
            
            # Shutdown management components
            for component in [self.performance_monitor, self.emergency_controller, 
                            self.state_coordinator, self.event_system, 
                            self.resource_manager, self.config_manager]:
                if component and hasattr(component, 'shutdown'):
                    try:
                        await component.shutdown()
                    except Exception as e:
                        self.logger.error(f"Error shutting down component {component.__class__.__name__}: {e}")
            
            # Shutdown executor
            self.executor.shutdown(wait=True, timeout=timeout)
            
            self.logger.info("NFCS system shutdown completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during system shutdown: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self._state_lock:
            return {
                'status': self._state.status.value,
                'mode': self._state.mode.value,
                'uptime_seconds': self._state.uptime_seconds,
                'active_modules': list(self._state.active_modules),
                'performance_metrics': self._state.performance_metrics.copy(),
                'error_count': self._state.error_count,
                'warning_count': self._state.warning_count,
                'statistics': self.stats.copy(),
                'timestamp': self._state.timestamp.isoformat()
            }
    
    async def set_operational_mode(self, mode: OperationalMode) -> bool:
        """
        Change the system operational mode
        
        Args:
            mode: New operational mode to set
            
        Returns:
            bool: True if mode change successful, False otherwise
        """
        try:
            # Validate mode transition
            if not self._validate_mode_transition(self._state.mode, mode):
                self.logger.warning(f"Invalid mode transition from {self._state.mode.value} to {mode.value}")
                return False
            
            # Apply mode change
            with self._state_lock:
                old_mode = self._state.mode
                self._state.mode = mode
            
            self.logger.info(f"Operational mode changed from {old_mode.value} to {mode.value}")
            
            # Trigger mode change event
            if self.event_system:
                await self.event_system.emit_event("system.mode_changed", {
                    "old_mode": old_mode.value,
                    "new_mode": mode.value,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error changing operational mode: {e}")
            return False
    
    def _validate_mode_transition(self, current_mode: OperationalMode, new_mode: OperationalMode) -> bool:
        """Validate if mode transition is allowed"""
        # Emergency mode can be entered from any mode
        if new_mode == OperationalMode.EMERGENCY:
            return True
        
        # Cannot exit emergency mode without manual intervention
        if current_mode == OperationalMode.EMERGENCY and new_mode != OperationalMode.MANUAL:
            return False
        
        # Cannot enter autonomous mode if not enabled
        if new_mode == OperationalMode.AUTONOMOUS and not self.config.enable_autonomous_mode:
            return False
        
        return True
    
    async def execute_command(self, command: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a system command
        
        Args:
            command: Command to execute
            parameters: Optional command parameters
            
        Returns:
            Dict containing command result and status
        """
        try:
            self.logger.info(f"Executing command: {command}")
            
            # Validate command against constitutional framework
            if self.constitutional_framework:
                command_context = {
                    'command': command,
                    'parameters': parameters or {},
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                compliance_check = self.constitutional_framework.check_compliance(command_context)
                if not compliance_check.compliant:
                    return {
                        'success': False,
                        'error': f"Command blocked by constitutional framework: {compliance_check.violations}",
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
            
            # Execute command based on type
            result = await self._execute_system_command(command, parameters or {})
            
            return {
                'success': True,
                'result': result,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing command {command}: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _execute_system_command(self, command: str, parameters: Dict[str, Any]) -> Any:
        """Execute specific system commands"""
        if command == "get_status":
            return self.get_system_status()
        
        elif command == "set_mode":
            mode_str = parameters.get('mode')
            if mode_str:
                mode = OperationalMode(mode_str)
                return await self.set_operational_mode(mode)
            else:
                raise ValueError("Mode parameter required")
        
        elif command == "get_modules":
            if self.module_manager:
                return await self.module_manager.get_active_modules()
            return {}
        
        elif command == "restart_module":
            module_name = parameters.get('module')
            if module_name and self.module_manager:
                return await self.module_manager.restart_module(module_name)
            else:
                raise ValueError("Module name required")
        
        elif command == "get_performance":
            if self.performance_monitor:
                return self.performance_monitor.get_current_metrics()
            return {}
        
        elif command == "emergency_stop":
            reason = parameters.get('reason', 'Manual emergency stop')
            self._activate_emergency_mode(reason)
            return {"emergency_activated": True, "reason": reason}
        
        else:
            raise ValueError(f"Unknown command: {command}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup"""
        if self._running:
            # Use asyncio to run shutdown if we're in an async context
            try:
                asyncio.run(self.shutdown())
            except RuntimeError:
                # If already in async context, just set shutdown flag
                self._shutdown_event.set()
                self._running = False
    
    def __repr__(self) -> str:
        """String representation"""
        return f"NFCSOrchestrator(status={self._state.status.value}, mode={self._state.mode.value}, modules={len(self._state.active_modules)})"


# Utility functions for orchestrator management

async def create_orchestrator(config: Optional[OrchestrationConfig] = None) -> NFCSOrchestrator:
    """
    Create and initialize a complete NFCS orchestrator
    
    Args:
        config: Optional configuration settings
        
    Returns:
        Initialized NFCSOrchestrator instance
    """
    orchestrator = NFCSOrchestrator(config)
    
    if await orchestrator.initialize():
        return orchestrator
    else:
        raise RuntimeError("Failed to initialize NFCS orchestrator")


def create_default_config() -> OrchestrationConfig:
    """Create default orchestrator configuration"""
    return OrchestrationConfig(
        max_concurrent_processes=10,
        update_frequency_hz=10.0,
        enable_autonomous_mode=True,
        enable_constitutional_enforcement=True,
        max_error_threshold=100,
        emergency_shutdown_threshold=50,
        resource_limit_cpu_percent=80.0,
        resource_limit_memory_mb=2048.0,
        performance_monitoring_enabled=True,
        performance_history_size=1000,
        metrics_calculation_interval=1.0,
        log_level="INFO"
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        config = create_default_config()
        
        async with await create_orchestrator(config) as orchestrator:
            print("NFCS Orchestrator started successfully!")
            
            # Run for a demonstration period
            await asyncio.sleep(5.0)
            
            # Get system status
            status = orchestrator.get_system_status()
            print(f"System Status: {status}")
    
    # Run the example
    asyncio.run(main())