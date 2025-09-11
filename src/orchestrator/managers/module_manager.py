"""
Neural Field Control System - Module Manager
===========================================

The ModuleManager handles registration, lifecycle management, and coordination
of all NFCS modules including core mathematical components and cognitive modules.

Key Features:
- Module registration and deregistration
- Lifecycle management (start, stop, restart)
- Health monitoring and status tracking
- Dependency management and resolution
- Resource allocation coordination
- Performance monitoring integration
- Constitutional framework compliance
- Emergency protocols and safety constraints

Architecture:
The module manager implements a service-oriented architecture pattern with
centralized registration and distributed execution. All modules are monitored
for health, performance, and constitutional compliance.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Type, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import traceback


class ModuleStatus(Enum):
    """Module status indicators"""
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    EMERGENCY = "emergency"


class ModulePriority(Enum):
    """Module priority levels"""
    CRITICAL = 1    # Constitutional, Emergency systems
    HIGH = 2        # Core mathematical modules
    MEDIUM = 3      # Cognitive modules
    LOW = 4         # Auxiliary systems


@dataclass
class ModuleInfo:
    """Complete module information"""
    name: str
    module_instance: Any
    status: ModuleStatus = ModuleStatus.UNREGISTERED
    priority: ModulePriority = ModulePriority.MEDIUM
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    # Lifecycle tracking
    registration_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    
    # Performance metrics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_response_time: float = 0.0
    
    # Health metrics
    health_score: float = 1.0
    error_count: int = 0
    warning_count: int = 0
    last_error: Optional[str] = None
    
    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Metadata
    version: str = "1.0.0"
    description: str = ""
    tags: Set[str] = field(default_factory=set)


class ModuleInterface(ABC):
    """Standard interface that all NFCS modules should implement"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the module"""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the module"""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the module"""
        pass
    
    @abstractmethod
    async def pause(self) -> bool:
        """Pause the module"""
        pass
    
    @abstractmethod
    async def resume(self) -> bool:
        """Resume the module"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current module status"""
        pass
    
    @abstractmethod
    async def get_health(self) -> Dict[str, Any]:
        """Get module health information"""
        pass
    
    @abstractmethod
    async def handle_emergency(self, emergency_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency conditions"""
        pass


class ModuleAdapter:
    """Adapter to wrap non-conforming modules with standard interface"""
    
    def __init__(self, module_instance: Any, name: str):
        self.module_instance = module_instance
        self.name = name
        self.logger = logging.getLogger(f"ModuleAdapter-{name}")
        
        # Track if module has standard methods
        self._has_initialize = hasattr(module_instance, 'initialize')
        self._has_start = hasattr(module_instance, 'start')
        self._has_stop = hasattr(module_instance, 'stop')
        self._has_get_status = hasattr(module_instance, 'get_status')
        self._has_get_health = hasattr(module_instance, 'get_health')
    
    async def initialize(self) -> bool:
        """Initialize the wrapped module"""
        try:
            if self._has_initialize:
                result = self.module_instance.initialize()
                if asyncio.iscoroutine(result):
                    return await result
                return bool(result)
            return True
        except Exception as e:
            self.logger.error(f"Error initializing module {self.name}: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the wrapped module"""
        try:
            if self._has_start:
                result = self.module_instance.start()
                if asyncio.iscoroutine(result):
                    return await result
                return bool(result)
            return True
        except Exception as e:
            self.logger.error(f"Error starting module {self.name}: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the wrapped module"""
        try:
            if self._has_stop:
                result = self.module_instance.stop()
                if asyncio.iscoroutine(result):
                    return await result
                return bool(result)
            return True
        except Exception as e:
            self.logger.error(f"Error stopping module {self.name}: {e}")
            return False
    
    async def pause(self) -> bool:
        """Pause the wrapped module"""
        try:
            if hasattr(self.module_instance, 'pause'):
                result = self.module_instance.pause()
                if asyncio.iscoroutine(result):
                    return await result
                return bool(result)
            return True
        except Exception as e:
            self.logger.error(f"Error pausing module {self.name}: {e}")
            return False
    
    async def resume(self) -> bool:
        """Resume the wrapped module"""
        try:
            if hasattr(self.module_instance, 'resume'):
                result = self.module_instance.resume()
                if asyncio.iscoroutine(result):
                    return await result
                return bool(result)
            return True
        except Exception as e:
            self.logger.error(f"Error resuming module {self.name}: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of wrapped module"""
        try:
            if self._has_get_status:
                result = self.module_instance.get_status()
                if asyncio.iscoroutine(result):
                    return await result
                return result if isinstance(result, dict) else {"status": str(result)}
            return {"status": "unknown", "adapter": True}
        except Exception as e:
            return {"status": "error", "error": str(e), "adapter": True}
    
    async def get_health(self) -> Dict[str, Any]:
        """Get health of wrapped module"""
        try:
            if self._has_get_health:
                result = self.module_instance.get_health()
                if asyncio.iscoroutine(result):
                    return await result
                return result if isinstance(result, dict) else {"health": str(result)}
            return {"health": "unknown", "adapter": True}
        except Exception as e:
            return {"health": "error", "error": str(e), "adapter": True}
    
    async def handle_emergency(self, emergency_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency for wrapped module"""
        try:
            if hasattr(self.module_instance, 'handle_emergency'):
                result = self.module_instance.handle_emergency(emergency_type, context)
                if asyncio.iscoroutine(result):
                    return await result
                return result if isinstance(result, dict) else {"handled": bool(result)}
            return {"handled": False, "reason": "No emergency handler", "adapter": True}
        except Exception as e:
            return {"handled": False, "error": str(e), "adapter": True}


class ModuleManager:
    """
    NFCS Module Manager
    
    Manages registration, lifecycle, and coordination of all NFCS modules
    including core mathematical components and cognitive modules.
    """
    
    def __init__(self, event_system=None, resource_manager=None, performance_monitor=None):
        """Initialize the Module Manager"""
        self.logger = logging.getLogger("ModuleManager")
        
        # Dependencies
        self.event_system = event_system
        self.resource_manager = resource_manager
        self.performance_monitor = performance_monitor
        
        # Module registry
        self._modules: Dict[str, ModuleInfo] = {}
        self._module_lock = threading.RLock()
        
        # Lifecycle management
        self._startup_order: List[str] = []
        self._shutdown_order: List[str] = []
        
        # Health monitoring
        self._health_check_interval = 30.0  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistics
        self.stats = {
            'total_registrations': 0,
            'total_starts': 0,
            'total_stops': 0,
            'total_failures': 0,
            'total_health_checks': 0,
            'last_health_check': None
        }
        
        self.logger.info("ModuleManager initialized")
    
    async def initialize(self) -> bool:
        """Initialize the module manager"""
        try:
            self.logger.info("Initializing ModuleManager...")
            
            # Start health monitoring
            if not await self._start_health_monitoring():
                return False
            
            self._running = True
            self.logger.info("ModuleManager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ModuleManager: {e}")
            return False
    
    async def register_module(self, 
                            name: str, 
                            module_instance: Any,
                            priority: ModulePriority = ModulePriority.MEDIUM,
                            dependencies: Optional[Set[str]] = None,
                            description: str = "",
                            tags: Optional[Set[str]] = None,
                            auto_start: bool = True) -> bool:
        """
        Register a module with the manager
        
        Args:
            name: Unique module name
            module_instance: Module instance to register
            priority: Module priority level
            dependencies: Set of module names this module depends on
            description: Module description
            tags: Optional tags for categorization
            auto_start: Whether to automatically start the module
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            self.logger.info(f"Registering module: {name}")
            
            with self._module_lock:
                # Check if module already registered
                if name in self._modules:
                    self.logger.warning(f"Module {name} already registered, updating...")
                    
                    # Stop existing module if running
                    if self._modules[name].status in [ModuleStatus.RUNNING, ModuleStatus.PAUSED]:
                        await self.stop_module(name)
                
                # Wrap module with adapter if it doesn't implement ModuleInterface
                if not isinstance(module_instance, ModuleInterface):
                    module_instance = ModuleAdapter(module_instance, name)
                
                # Create module info
                module_info = ModuleInfo(
                    name=name,
                    module_instance=module_instance,
                    status=ModuleStatus.REGISTERED,
                    priority=priority,
                    dependencies=dependencies or set(),
                    registration_time=datetime.now(timezone.utc),
                    description=description,
                    tags=tags or set()
                )
                
                # Update dependency graph
                self._update_dependency_graph(name, dependencies or set())
                
                # Register the module
                self._modules[name] = module_info
                
                # Update startup/shutdown orders
                self._update_execution_orders()
            
            # Initialize the module
            if not await self._initialize_module(name):
                self.logger.error(f"Failed to initialize module {name}")
                return False
            
            # Auto-start if requested
            if auto_start:
                if not await self.start_module(name):
                    self.logger.error(f"Failed to auto-start module {name}")
                    return False
            
            self.stats['total_registrations'] += 1
            
            # Emit registration event
            if self.event_system:
                await self.event_system.emit_event("module.registered", {
                    "module_name": name,
                    "priority": priority.name,
                    "dependencies": list(dependencies or []),
                    "auto_started": auto_start,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            self.logger.info(f"Module {name} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering module {name}: {e}")
            return False
    
    async def unregister_module(self, name: str) -> bool:
        """
        Unregister a module
        
        Args:
            name: Module name to unregister
            
        Returns:
            bool: True if unregistration successful, False otherwise
        """
        try:
            self.logger.info(f"Unregistering module: {name}")
            
            with self._module_lock:
                if name not in self._modules:
                    self.logger.warning(f"Module {name} not registered")
                    return False
                
                module_info = self._modules[name]
                
                # Stop module if running
                if module_info.status in [ModuleStatus.RUNNING, ModuleStatus.PAUSED]:
                    await self.stop_module(name)
                
                # Check for dependents
                if module_info.dependents:
                    self.logger.warning(f"Module {name} has dependents: {module_info.dependents}")
                    # Could force stop dependents or fail
                    for dependent in list(module_info.dependents):
                        await self.stop_module(dependent)
                
                # Remove from dependency graph
                self._remove_from_dependency_graph(name)
                
                # Remove from registry
                del self._modules[name]
                
                # Update execution orders
                self._update_execution_orders()
            
            # Emit unregistration event
            if self.event_system:
                await self.event_system.emit_event("module.unregistered", {
                    "module_name": name,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            self.logger.info(f"Module {name} unregistered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unregistering module {name}: {e}")
            return False
    
    async def start_module(self, name: str) -> bool:
        """
        Start a registered module
        
        Args:
            name: Module name to start
            
        Returns:
            bool: True if start successful, False otherwise
        """
        try:
            self.logger.info(f"Starting module: {name}")
            
            with self._module_lock:
                if name not in self._modules:
                    self.logger.error(f"Module {name} not registered")
                    return False
                
                module_info = self._modules[name]
                
                # Check current status
                if module_info.status == ModuleStatus.RUNNING:
                    self.logger.warning(f"Module {name} already running")
                    return True
                
                if module_info.status not in [ModuleStatus.REGISTERED, ModuleStatus.STOPPED, ModuleStatus.PAUSED]:
                    self.logger.error(f"Cannot start module {name} in status {module_info.status}")
                    return False
                
                # Check dependencies
                if not await self._check_dependencies(name):
                    self.logger.error(f"Dependencies not met for module {name}")
                    return False
                
                # Update status
                module_info.status = ModuleStatus.INITIALIZING
            
            # Start the module
            start_time = time.time()
            
            try:
                success = await module_info.module_instance.start()
                
                if success:
                    with self._module_lock:
                        module_info.status = ModuleStatus.RUNNING
                        module_info.start_time = datetime.now(timezone.utc)
                        module_info.last_heartbeat = datetime.now(timezone.utc)
                    
                    self.stats['total_starts'] += 1
                    
                    # Record performance metrics
                    if self.performance_monitor:
                        await self.performance_monitor.record_module_event(name, "start", time.time() - start_time)
                    
                    # Emit start event
                    if self.event_system:
                        await self.event_system.emit_event("module.started", {
                            "module_name": name,
                            "start_time": time.time() - start_time,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                    
                    self.logger.info(f"Module {name} started successfully")
                    return True
                else:
                    with self._module_lock:
                        module_info.status = ModuleStatus.ERROR
                        module_info.error_count += 1
                        module_info.last_error = "Failed to start"
                    
                    self.stats['total_failures'] += 1
                    self.logger.error(f"Module {name} failed to start")
                    return False
                    
            except Exception as e:
                with self._module_lock:
                    module_info.status = ModuleStatus.ERROR
                    module_info.error_count += 1
                    module_info.last_error = str(e)
                
                self.stats['total_failures'] += 1
                self.logger.error(f"Exception starting module {name}: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting module {name}: {e}")
            return False
    
    async def stop_module(self, name: str) -> bool:
        """
        Stop a running module
        
        Args:
            name: Module name to stop
            
        Returns:
            bool: True if stop successful, False otherwise
        """
        try:
            self.logger.info(f"Stopping module: {name}")
            
            with self._module_lock:
                if name not in self._modules:
                    self.logger.error(f"Module {name} not registered")
                    return False
                
                module_info = self._modules[name]
                
                # Check current status
                if module_info.status in [ModuleStatus.STOPPED, ModuleStatus.UNREGISTERED]:
                    self.logger.warning(f"Module {name} already stopped")
                    return True
                
                # Check for dependents
                if module_info.dependents:
                    active_dependents = [
                        dep for dep in module_info.dependents 
                        if self._modules.get(dep, {}).status == ModuleStatus.RUNNING
                    ]
                    if active_dependents:
                        self.logger.warning(f"Stopping module {name} with active dependents: {active_dependents}")
                        # Stop dependents first
                        for dependent in active_dependents:
                            await self.stop_module(dependent)
                
                # Update status
                module_info.status = ModuleStatus.STOPPING
            
            # Stop the module
            stop_time = time.time()
            
            try:
                success = await module_info.module_instance.stop()
                
                with self._module_lock:
                    if success:
                        module_info.status = ModuleStatus.STOPPED
                        self.stats['total_stops'] += 1
                    else:
                        module_info.status = ModuleStatus.ERROR
                        module_info.error_count += 1
                        module_info.last_error = "Failed to stop"
                        self.stats['total_failures'] += 1
                
                # Record performance metrics
                if self.performance_monitor:
                    await self.performance_monitor.record_module_event(name, "stop", time.time() - stop_time)
                
                # Emit stop event
                if self.event_system:
                    await self.event_system.emit_event("module.stopped", {
                        "module_name": name,
                        "stop_time": time.time() - stop_time,
                        "success": success,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
                self.logger.info(f"Module {name} {'stopped successfully' if success else 'failed to stop'}")
                return success
                
            except Exception as e:
                with self._module_lock:
                    module_info.status = ModuleStatus.ERROR
                    module_info.error_count += 1
                    module_info.last_error = str(e)
                
                self.stats['total_failures'] += 1
                self.logger.error(f"Exception stopping module {name}: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error stopping module {name}: {e}")
            return False
    
    async def restart_module(self, name: str) -> bool:
        """
        Restart a module (stop then start)
        
        Args:
            name: Module name to restart
            
        Returns:
            bool: True if restart successful, False otherwise
        """
        try:
            self.logger.info(f"Restarting module: {name}")
            
            # Stop first
            if not await self.stop_module(name):
                return False
            
            # Wait a moment for cleanup
            await asyncio.sleep(0.1)
            
            # Start again
            return await self.start_module(name)
            
        except Exception as e:
            self.logger.error(f"Error restarting module {name}: {e}")
            return False
    
    async def pause_module(self, name: str) -> bool:
        """Pause a running module"""
        try:
            with self._module_lock:
                if name not in self._modules:
                    return False
                
                module_info = self._modules[name]
                if module_info.status != ModuleStatus.RUNNING:
                    return False
                
                success = await module_info.module_instance.pause()
                if success:
                    module_info.status = ModuleStatus.PAUSED
                
                return success
                
        except Exception as e:
            self.logger.error(f"Error pausing module {name}: {e}")
            return False
    
    async def resume_module(self, name: str) -> bool:
        """Resume a paused module"""
        try:
            with self._module_lock:
                if name not in self._modules:
                    return False
                
                module_info = self._modules[name]
                if module_info.status != ModuleStatus.PAUSED:
                    return False
                
                success = await module_info.module_instance.resume()
                if success:
                    module_info.status = ModuleStatus.RUNNING
                    module_info.last_heartbeat = datetime.now(timezone.utc)
                
                return success
                
        except Exception as e:
            self.logger.error(f"Error resuming module {name}: {e}")
            return False
    
    async def get_module_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a module"""
        try:
            with self._module_lock:
                if name not in self._modules:
                    return None
                
                module_info = self._modules[name]
                
                # Get current status and health from module
                current_status = await module_info.module_instance.get_status()
                current_health = await module_info.module_instance.get_health()
                
                return {
                    "name": module_info.name,
                    "status": module_info.status.value,
                    "priority": module_info.priority.name,
                    "dependencies": list(module_info.dependencies),
                    "dependents": list(module_info.dependents),
                    "registration_time": module_info.registration_time.isoformat() if module_info.registration_time else None,
                    "start_time": module_info.start_time.isoformat() if module_info.start_time else None,
                    "last_heartbeat": module_info.last_heartbeat.isoformat() if module_info.last_heartbeat else None,
                    "total_operations": module_info.total_operations,
                    "successful_operations": module_info.successful_operations,
                    "failed_operations": module_info.failed_operations,
                    "success_rate": (module_info.successful_operations / max(1, module_info.total_operations)) * 100,
                    "average_response_time": module_info.average_response_time,
                    "health_score": module_info.health_score,
                    "error_count": module_info.error_count,
                    "warning_count": module_info.warning_count,
                    "last_error": module_info.last_error,
                    "cpu_usage_percent": module_info.cpu_usage_percent,
                    "memory_usage_mb": module_info.memory_usage_mb,
                    "version": module_info.version,
                    "description": module_info.description,
                    "tags": list(module_info.tags),
                    "current_status": current_status,
                    "current_health": current_health
                }
                
        except Exception as e:
            self.logger.error(f"Error getting module info for {name}: {e}")
            return None
    
    async def get_active_modules(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active modules"""
        active_modules = {}
        
        with self._module_lock:
            for name, module_info in self._modules.items():
                if module_info.status == ModuleStatus.RUNNING:
                    module_data = await self.get_module_info(name)
                    if module_data:
                        active_modules[name] = module_data
        
        return active_modules
    
    async def get_all_modules(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered modules"""
        all_modules = {}
        
        with self._module_lock:
            for name in self._modules.keys():
                module_data = await self.get_module_info(name)
                if module_data:
                    all_modules[name] = module_data
        
        return all_modules
    
    async def start_all_modules(self) -> Dict[str, bool]:
        """Start all registered modules in dependency order"""
        results = {}
        
        self.logger.info("Starting all modules in dependency order...")
        
        for module_name in self._startup_order:
            if module_name in self._modules:
                results[module_name] = await self.start_module(module_name)
            
        return results
    
    async def stop_all_modules(self) -> Dict[str, bool]:
        """Stop all modules in reverse dependency order"""
        results = {}
        
        self.logger.info("Stopping all modules in reverse dependency order...")
        
        for module_name in reversed(self._shutdown_order):
            if module_name in self._modules:
                results[module_name] = await self.stop_module(module_name)
            
        return results
    
    async def shutdown_all_modules(self) -> bool:
        """Shutdown all modules and the manager"""
        try:
            self.logger.info("Shutting down ModuleManager...")
            
            # Stop health monitoring
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Stop all modules
            await self.stop_all_modules()
            
            self._running = False
            self.logger.info("ModuleManager shutdown completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during ModuleManager shutdown: {e}")
            return False
    
    # Dependency management methods
    
    def _update_dependency_graph(self, module_name: str, dependencies: Set[str]):
        """Update the dependency graph when adding a module"""
        # Add reverse dependencies
        for dep_name in dependencies:
            if dep_name in self._modules:
                self._modules[dep_name].dependents.add(module_name)
    
    def _remove_from_dependency_graph(self, module_name: str):
        """Remove module from dependency graph"""
        module_info = self._modules[module_name]
        
        # Remove from dependencies' dependents
        for dep_name in module_info.dependencies:
            if dep_name in self._modules:
                self._modules[dep_name].dependents.discard(module_name)
        
        # Remove from dependents' dependencies
        for dependent_name in module_info.dependents:
            if dependent_name in self._modules:
                self._modules[dependent_name].dependencies.discard(module_name)
    
    def _update_execution_orders(self):
        """Update startup and shutdown orders based on dependencies"""
        # Topological sort for startup order
        self._startup_order = self._topological_sort()
        self._shutdown_order = list(reversed(self._startup_order))
    
    def _topological_sort(self) -> List[str]:
        """Perform topological sort based on module dependencies"""
        # Simple topological sort implementation
        visited = set()
        temp_mark = set()
        result = []
        
        def visit(module_name: str):
            if module_name in temp_mark:
                # Circular dependency detected - log warning
                self.logger.warning(f"Circular dependency detected involving {module_name}")
                return
            
            if module_name in visited:
                return
            
            temp_mark.add(module_name)
            
            # Visit dependencies first
            module_info = self._modules.get(module_name)
            if module_info:
                for dep_name in module_info.dependencies:
                    if dep_name in self._modules:
                        visit(dep_name)
            
            temp_mark.remove(module_name)
            visited.add(module_name)
            result.append(module_name)
        
        # Sort by priority first, then process
        modules_by_priority = sorted(
            self._modules.keys(),
            key=lambda name: self._modules[name].priority.value
        )
        
        for module_name in modules_by_priority:
            if module_name not in visited:
                visit(module_name)
        
        return result
    
    async def _check_dependencies(self, module_name: str) -> bool:
        """Check if all dependencies are running for a module"""
        module_info = self._modules[module_name]
        
        for dep_name in module_info.dependencies:
            if dep_name not in self._modules:
                self.logger.error(f"Dependency {dep_name} not registered for module {module_name}")
                return False
            
            dep_status = self._modules[dep_name].status
            if dep_status != ModuleStatus.RUNNING:
                self.logger.error(f"Dependency {dep_name} not running for module {module_name} (status: {dep_status})")
                return False
        
        return True
    
    async def _initialize_module(self, name: str) -> bool:
        """Initialize a registered module"""
        try:
            module_info = self._modules[name]
            
            # Check dependencies are initialized
            for dep_name in module_info.dependencies:
                if dep_name in self._modules:
                    dep_status = self._modules[dep_name].status
                    if dep_status == ModuleStatus.UNREGISTERED:
                        self.logger.error(f"Dependency {dep_name} not initialized for module {name}")
                        return False
            
            # Initialize the module
            success = await module_info.module_instance.initialize()
            
            if success:
                module_info.status = ModuleStatus.REGISTERED
                self.logger.info(f"Module {name} initialized successfully")
            else:
                module_info.status = ModuleStatus.ERROR
                module_info.error_count += 1
                module_info.last_error = "Initialization failed"
                self.logger.error(f"Module {name} initialization failed")
            
            return success
            
        except Exception as e:
            module_info = self._modules[name]
            module_info.status = ModuleStatus.ERROR
            module_info.error_count += 1
            module_info.last_error = str(e)
            
            self.logger.error(f"Exception during module {name} initialization: {e}")
            return False
    
    # Health monitoring
    
    async def _start_health_monitoring(self) -> bool:
        """Start the health monitoring task"""
        try:
            self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
            self.logger.info("Health monitoring started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start health monitoring: {e}")
            return False
    
    async def _health_monitoring_loop(self):
        """Main health monitoring loop"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self._health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Brief pause before retry
    
    async def _perform_health_checks(self):
        """Perform health checks on all active modules"""
        self.stats['total_health_checks'] += 1
        self.stats['last_health_check'] = datetime.now(timezone.utc).isoformat()
        
        with self._module_lock:
            active_modules = [
                name for name, info in self._modules.items()
                if info.status == ModuleStatus.RUNNING
            ]
        
        for module_name in active_modules:
            try:
                await self._check_module_health(module_name)
            except Exception as e:
                self.logger.error(f"Error checking health of module {module_name}: {e}")
    
    async def _check_module_health(self, module_name: str):
        """Check health of a specific module"""
        try:
            with self._module_lock:
                module_info = self._modules[module_name]
            
            # Get health from module
            health_data = await module_info.module_instance.get_health()
            
            # Update heartbeat
            with self._module_lock:
                module_info.last_heartbeat = datetime.now(timezone.utc)
                
                # Update health metrics
                if isinstance(health_data, dict):
                    module_info.health_score = health_data.get('score', 1.0)
                    if 'cpu_usage' in health_data:
                        module_info.cpu_usage_percent = health_data['cpu_usage']
                    if 'memory_usage' in health_data:
                        module_info.memory_usage_mb = health_data['memory_usage']
                
                # Check for health issues
                if module_info.health_score < 0.5:
                    self.logger.warning(f"Module {module_name} health score low: {module_info.health_score}")
                    
                    # Emit health warning event
                    if self.event_system:
                        await self.event_system.emit_event("module.health_warning", {
                            "module_name": module_name,
                            "health_score": module_info.health_score,
                            "health_data": health_data,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
            
        except Exception as e:
            self.logger.error(f"Error checking health for module {module_name}: {e}")
            
            # Mark module as having health issues
            with self._module_lock:
                module_info = self._modules[module_name]
                module_info.error_count += 1
                module_info.last_error = f"Health check failed: {str(e)}"
    
    # Emergency handling
    
    async def handle_module_emergency(self, module_name: str, emergency_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency for a specific module"""
        try:
            self.logger.critical(f"Handling emergency for module {module_name}: {emergency_type}")
            
            with self._module_lock:
                if module_name not in self._modules:
                    return {"handled": False, "error": "Module not found"}
                
                module_info = self._modules[module_name]
                
                # Mark module as in emergency state
                old_status = module_info.status
                module_info.status = ModuleStatus.EMERGENCY
            
            # Let module handle its own emergency
            result = await module_info.module_instance.handle_emergency(emergency_type, context)
            
            # Emit emergency event
            if self.event_system:
                await self.event_system.emit_event("module.emergency", {
                    "module_name": module_name,
                    "emergency_type": emergency_type,
                    "context": context,
                    "result": result,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Restore status if emergency handled
            if result.get("handled", False):
                with self._module_lock:
                    module_info.status = old_status
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error handling emergency for module {module_name}: {e}")
            return {"handled": False, "error": str(e)}
    
    # Statistics and reporting
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics"""
        with self._module_lock:
            module_counts = {}
            for status in ModuleStatus:
                count = sum(1 for info in self._modules.values() if info.status == status)
                module_counts[status.value] = count
            
            total_errors = sum(info.error_count for info in self._modules.values())
            total_operations = sum(info.total_operations for info in self._modules.values())
            
            return {
                "total_registered_modules": len(self._modules),
                "module_status_counts": module_counts,
                "total_operations": total_operations,
                "total_errors": total_errors,
                "manager_stats": self.stats.copy(),
                "startup_order": self._startup_order.copy(),
                "shutdown_order": self._shutdown_order.copy(),
                "health_monitoring_active": self._running and self._health_check_task is not None
            }
    
    def __repr__(self) -> str:
        """String representation"""
        with self._module_lock:
            running_count = sum(1 for info in self._modules.values() if info.status == ModuleStatus.RUNNING)
            return f"ModuleManager(registered={len(self._modules)}, running={running_count})"


# Utility functions

async def create_test_module(name: str, auto_fail: bool = False) -> ModuleInterface:
    """Create a test module for development and testing"""
    
    class TestModule(ModuleInterface):
        def __init__(self, name: str, auto_fail: bool = False):
            self.name = name
            self.auto_fail = auto_fail
            self.initialized = False
            self.running = False
            self.paused = False
            
        async def initialize(self) -> bool:
            if self.auto_fail:
                return False
            self.initialized = True
            return True
            
        async def start(self) -> bool:
            if self.auto_fail or not self.initialized:
                return False
            self.running = True
            return True
            
        async def stop(self) -> bool:
            self.running = False
            return True
            
        async def pause(self) -> bool:
            if self.running:
                self.paused = True
                return True
            return False
            
        async def resume(self) -> bool:
            if self.paused:
                self.paused = False
                return True
            return False
            
        async def get_status(self) -> Dict[str, Any]:
            return {
                "name": self.name,
                "initialized": self.initialized,
                "running": self.running,
                "paused": self.paused
            }
            
        async def get_health(self) -> Dict[str, Any]:
            return {
                "score": 0.1 if self.auto_fail else 1.0,
                "cpu_usage": 10.0,
                "memory_usage": 50.0
            }
            
        async def handle_emergency(self, emergency_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "handled": not self.auto_fail,
                "actions_taken": ["emergency_protocol_activated"] if not self.auto_fail else []
            }
    
    return TestModule(name, auto_fail)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create module manager
        manager = ModuleManager()
        await manager.initialize()
        
        # Create and register test modules
        test_module1 = await create_test_module("test1")
        test_module2 = await create_test_module("test2")
        
        await manager.register_module("test1", test_module1, priority=ModulePriority.HIGH)
        await manager.register_module("test2", test_module2, dependencies={"test1"})
        
        # Start all modules
        await manager.start_all_modules()
        
        # Get status
        modules = await manager.get_all_modules()
        print(f"Active modules: {modules}")
        
        # Shutdown
        await manager.shutdown_all_modules()
    
    # Run example
    asyncio.run(main())