"""
Neural Field Control System - State Coordinator
===============================================

The StateCoordinator manages global state synchronization across all NFCS components.
It ensures consistent state representation, coordinates state updates, and maintains
state history for analysis and rollback capabilities.

Key Features:
- Global state management and synchronization
- State versioning and history tracking
- Distributed state consistency protocols
- Conflict resolution and merging
- State persistence and recovery
- Constitutional compliance verification
- Real-time state monitoring and alerts
- Rollback and checkpoint capabilities

Architecture:
Implements a centralized state management pattern with distributed state caching
for performance. All state changes are validated against constitutional policies
and logged for audit and analysis purposes.
"""

import asyncio
import logging
import threading
import time
import json
import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import copy


class StateType(Enum):
    """Types of state managed by the coordinator"""
    GLOBAL = "global"           # System-wide global state
    MODULE = "module"           # Individual module states
    COGNITIVE = "cognitive"     # Cognitive module states
    MATHEMATICAL = "mathematical"  # Mathematical framework states
    CONSTITUTIONAL = "constitutional"  # Constitutional compliance states
    PERFORMANCE = "performance"    # Performance and metrics states
    EMERGENCY = "emergency"     # Emergency and safety states


class StateOperation(Enum):
    """Types of state operations"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    ROLLBACK = "rollback"
    CHECKPOINT = "checkpoint"


class StatePriority(Enum):
    """Priority levels for state updates"""
    CRITICAL = 1    # Emergency, safety-critical
    HIGH = 2        # Constitutional, core system
    MEDIUM = 3      # Module operations
    LOW = 4         # Metrics, logging


@dataclass
class StateEntry:
    """Individual state entry with metadata"""
    key: str
    value: Any
    state_type: StateType
    timestamp: datetime
    version: int = 1
    checksum: str = ""
    source_module: str = "system"
    priority: StatePriority = StatePriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum of the state value"""
        try:
            value_str = json.dumps(self.value, sort_keys=True, default=str)
            return hashlib.sha256(value_str.encode()).hexdigest()[:16]
        except:
            return hashlib.sha256(str(self.value).encode()).hexdigest()[:16]
    
    def is_valid(self) -> bool:
        """Verify state entry integrity"""
        return self.checksum == self._calculate_checksum()


@dataclass
class StateOperation:
    """State operation record"""
    operation: StateOperation
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source_module: str
    success: bool = False
    error: Optional[str] = None


@dataclass
class StateCheckpoint:
    """State checkpoint for rollback capabilities"""
    checkpoint_id: str
    timestamp: datetime
    description: str
    state_snapshot: Dict[str, StateEntry]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving state conflicts"""
    LAST_WRITE_WINS = "last_write_wins"
    PRIORITY_BASED = "priority_based" 
    CONSTITUTIONAL_PRIORITY = "constitutional_priority"
    MERGE_VALUES = "merge_values"
    MANUAL_RESOLUTION = "manual_resolution"


class StateCoordinator:
    """
    NFCS State Coordinator
    
    Manages global state synchronization, consistency, and coordination
    across all NFCS components with constitutional compliance verification.
    """
    
    def __init__(self, event_system=None, constitutional_framework=None):
        """Initialize the State Coordinator"""
        self.logger = logging.getLogger("StateCoordinator")
        
        # Dependencies
        self.event_system = event_system
        self.constitutional_framework = constitutional_framework
        
        # State storage
        self._state_store: Dict[str, StateEntry] = {}
        self._state_lock = threading.RLock()
        
        # State history and versioning
        self._state_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._version_counter = 0
        
        # Checkpoints and rollback
        self._checkpoints: Dict[str, StateCheckpoint] = {}
        self._checkpoint_counter = 0
        
        # Operation tracking
        self._operations_log: deque = deque(maxlen=10000)
        self._pending_operations: Dict[str, List[StateOperation]] = defaultdict(list)
        
        # Conflict resolution
        self._conflict_strategy = ConflictResolutionStrategy.CONSTITUTIONAL_PRIORITY
        self._conflict_handlers: Dict[ConflictResolutionStrategy, Callable] = {
            ConflictResolutionStrategy.LAST_WRITE_WINS: self._resolve_last_write_wins,
            ConflictResolutionStrategy.PRIORITY_BASED: self._resolve_priority_based,
            ConflictResolutionStrategy.CONSTITUTIONAL_PRIORITY: self._resolve_constitutional_priority,
            ConflictResolutionStrategy.MERGE_VALUES: self._resolve_merge_values
        }
        
        # Synchronization settings
        self._sync_interval = 1.0  # seconds
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        
        # State watchers and subscribers
        self._watchers: Dict[str, List[Callable]] = defaultdict(list)  # key -> [callbacks]
        self._type_watchers: Dict[StateType, List[Callable]] = defaultdict(list)  # type -> [callbacks]
        
        # Performance tracking
        self.stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'conflicts_resolved': 0,
            'checkpoints_created': 0,
            'rollbacks_performed': 0,
            'state_entries': 0,
            'sync_cycles': 0,
            'last_sync': None
        }
        
        self.logger.info("StateCoordinator initialized")
    
    async def initialize(self) -> bool:
        """Initialize the state coordinator"""
        try:
            self.logger.info("Initializing StateCoordinator...")
            
            # Create initial global state
            await self._initialize_global_state()
            
            # Start synchronization loop
            if not await self._start_synchronization_loop():
                return False
            
            self._running = True
            self.logger.info("StateCoordinator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize StateCoordinator: {e}")
            return False
    
    async def _initialize_global_state(self):
        """Initialize the global system state"""
        initial_state = {
            "system.status": "initializing",
            "system.mode": "supervised", 
            "system.start_time": datetime.now(timezone.utc).isoformat(),
            "system.version": "1.0.0",
            "modules.active_count": 0,
            "constitutional.compliance_level": 1.0,
            "performance.system_load": 0.0,
            "emergency.status": "normal"
        }
        
        for key, value in initial_state.items():
            await self.set_state(
                key=key,
                value=value,
                state_type=StateType.GLOBAL,
                source_module="system",
                priority=StatePriority.HIGH
            )
    
    async def _start_synchronization_loop(self) -> bool:
        """Start the state synchronization loop"""
        try:
            self._sync_task = asyncio.create_task(self._synchronization_loop())
            self.logger.info("State synchronization loop started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start synchronization loop: {e}")
            return False
    
    async def _synchronization_loop(self):
        """Main state synchronization loop"""
        while self._running:
            try:
                sync_start = time.time()
                
                # Perform synchronization tasks
                await self._perform_sync_cycle()
                
                # Update statistics
                self.stats['sync_cycles'] += 1
                self.stats['last_sync'] = datetime.now(timezone.utc).isoformat()
                
                # Calculate sleep time
                elapsed = time.time() - sync_start
                sleep_time = max(0, self._sync_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in synchronization loop: {e}")
                await asyncio.sleep(1.0)
        
        self.logger.info("State synchronization loop ended")
    
    async def _perform_sync_cycle(self):
        """Perform a complete synchronization cycle"""
        # Clean up old history entries
        self._cleanup_state_history()
        
        # Process pending operations
        await self._process_pending_operations()
        
        # Validate state consistency
        await self._validate_state_consistency()
        
        # Update performance metrics
        await self._update_state_metrics()
    
    async def set_state(self, 
                       key: str, 
                       value: Any, 
                       state_type: StateType = StateType.MODULE,
                       source_module: str = "unknown",
                       priority: StatePriority = StatePriority.MEDIUM,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Set a state value with validation and conflict resolution
        
        Args:
            key: State key
            value: State value
            state_type: Type of state
            source_module: Module setting the state
            priority: Priority level
            metadata: Optional metadata
            
        Returns:
            bool: True if state set successfully, False otherwise
        """
        try:
            self.logger.debug(f"Setting state {key} = {value} (type: {state_type.value}, module: {source_module})")
            
            # Validate against constitutional framework if available
            if self.constitutional_framework:
                if not await self._validate_constitutional_compliance(key, value, source_module):
                    self.logger.warning(f"Constitutional validation failed for state {key}")
                    return False
            
            with self._state_lock:
                # Get current version
                self._version_counter += 1
                current_time = datetime.now(timezone.utc)
                
                # Check for existing state
                old_entry = self._state_store.get(key)
                
                # Create new state entry
                new_entry = StateEntry(
                    key=key,
                    value=copy.deepcopy(value),
                    state_type=state_type,
                    timestamp=current_time,
                    version=self._version_counter,
                    source_module=source_module,
                    priority=priority,
                    metadata=metadata or {}
                )
                
                # Handle conflicts if state already exists
                if old_entry:
                    resolved_entry = await self._resolve_state_conflict(old_entry, new_entry)
                    if not resolved_entry:
                        return False
                    new_entry = resolved_entry
                
                # Store the new state
                self._state_store[key] = new_entry
                
                # Add to history
                self._state_history[key].append(old_entry if old_entry else None)
                
                # Log operation
                operation = StateOperation(
                    operation=StateOperation.UPDATE if old_entry else StateOperation.CREATE,
                    key=key,
                    old_value=old_entry.value if old_entry else None,
                    new_value=value,
                    timestamp=current_time,
                    source_module=source_module,
                    success=True
                )
                self._operations_log.append(operation)
            
            # Update statistics
            self.stats['total_operations'] += 1
            self.stats['successful_operations'] += 1
            self.stats['state_entries'] = len(self._state_store)
            
            # Notify watchers
            await self._notify_watchers(key, new_entry)
            
            # Emit state change event
            if self.event_system:
                await self.event_system.emit_event("state.changed", {
                    "key": key,
                    "value": value,
                    "state_type": state_type.value,
                    "source_module": source_module,
                    "priority": priority.name,
                    "timestamp": current_time.isoformat()
                })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting state {key}: {e}")
            self.stats['failed_operations'] += 1
            return False
    
    async def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get a state value
        
        Args:
            key: State key
            default: Default value if key not found
            
        Returns:
            State value or default
        """
        try:
            with self._state_lock:
                entry = self._state_store.get(key)
                if entry and entry.is_valid():
                    return copy.deepcopy(entry.value)
                return default
                
        except Exception as e:
            self.logger.error(f"Error getting state {key}: {e}")
            return default
    
    async def get_state_entry(self, key: str) -> Optional[StateEntry]:
        """Get complete state entry with metadata"""
        try:
            with self._state_lock:
                entry = self._state_store.get(key)
                return copy.deepcopy(entry) if entry else None
                
        except Exception as e:
            self.logger.error(f"Error getting state entry {key}: {e}")
            return None
    
    async def delete_state(self, key: str, source_module: str = "unknown") -> bool:
        """Delete a state entry"""
        try:
            with self._state_lock:
                if key not in self._state_store:
                    return False
                
                old_entry = self._state_store[key]
                del self._state_store[key]
                
                # Log operation
                operation = StateOperation(
                    operation=StateOperation.DELETE,
                    key=key,
                    old_value=old_entry.value,
                    new_value=None,
                    timestamp=datetime.now(timezone.utc),
                    source_module=source_module,
                    success=True
                )
                self._operations_log.append(operation)
            
            # Update statistics
            self.stats['total_operations'] += 1
            self.stats['successful_operations'] += 1
            self.stats['state_entries'] = len(self._state_store)
            
            # Notify watchers
            await self._notify_watchers(key, None)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting state {key}: {e}")
            self.stats['failed_operations'] += 1
            return False
    
    async def get_states_by_type(self, state_type: StateType) -> Dict[str, Any]:
        """Get all states of a specific type"""
        result = {}
        
        with self._state_lock:
            for key, entry in self._state_store.items():
                if entry.state_type == state_type and entry.is_valid():
                    result[key] = copy.deepcopy(entry.value)
        
        return result
    
    async def get_states_by_module(self, module_name: str) -> Dict[str, Any]:
        """Get all states from a specific module"""
        result = {}
        
        with self._state_lock:
            for key, entry in self._state_store.items():
                if entry.source_module == module_name and entry.is_valid():
                    result[key] = copy.deepcopy(entry.value)
        
        return result
    
    async def get_state_keys(self, pattern: Optional[str] = None, state_type: Optional[StateType] = None) -> List[str]:
        """Get state keys matching optional pattern and type"""
        keys = []
        
        with self._state_lock:
            for key, entry in self._state_store.items():
                # Check type filter
                if state_type and entry.state_type != state_type:
                    continue
                
                # Check pattern filter
                if pattern and pattern not in key:
                    continue
                
                keys.append(key)
        
        return sorted(keys)
    
    # State history and versioning
    
    async def get_state_history(self, key: str, limit: int = 100) -> List[StateEntry]:
        """Get state history for a key"""
        with self._state_lock:
            history = list(self._state_history.get(key, []))
            return [copy.deepcopy(entry) for entry in history[-limit:] if entry]
    
    async def get_state_version(self, key: str, version: int) -> Optional[Any]:
        """Get specific version of a state"""
        history = await self.get_state_history(key)
        for entry in reversed(history):
            if entry and entry.version == version:
                return copy.deepcopy(entry.value)
        return None
    
    # Checkpoints and rollback
    
    async def create_checkpoint(self, description: str = "", metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a state checkpoint for rollback capabilities"""
        try:
            self._checkpoint_counter += 1
            checkpoint_id = f"checkpoint_{self._checkpoint_counter}_{int(time.time())}"
            
            with self._state_lock:
                # Create complete state snapshot
                state_snapshot = {
                    key: copy.deepcopy(entry) 
                    for key, entry in self._state_store.items()
                }
                
                checkpoint = StateCheckpoint(
                    checkpoint_id=checkpoint_id,
                    timestamp=datetime.now(timezone.utc),
                    description=description or f"Checkpoint {self._checkpoint_counter}",
                    state_snapshot=state_snapshot,
                    metadata=metadata or {}
                )
                
                self._checkpoints[checkpoint_id] = checkpoint
            
            self.stats['checkpoints_created'] += 1
            
            # Emit checkpoint event
            if self.event_system:
                await self.event_system.emit_event("state.checkpoint_created", {
                    "checkpoint_id": checkpoint_id,
                    "description": description,
                    "state_count": len(state_snapshot),
                    "timestamp": checkpoint.timestamp.isoformat()
                })
            
            self.logger.info(f"Created checkpoint {checkpoint_id} with {len(state_snapshot)} states")
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Error creating checkpoint: {e}")
            return ""
    
    async def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rollback state to a specific checkpoint"""
        try:
            if checkpoint_id not in self._checkpoints:
                self.logger.error(f"Checkpoint {checkpoint_id} not found")
                return False
            
            checkpoint = self._checkpoints[checkpoint_id]
            
            with self._state_lock:
                # Clear current state
                old_state_count = len(self._state_store)
                self._state_store.clear()
                
                # Restore checkpoint state
                for key, entry in checkpoint.state_snapshot.items():
                    self._state_store[key] = copy.deepcopy(entry)
                
                # Log rollback operation
                operation = StateOperation(
                    operation=StateOperation.ROLLBACK,
                    key=f"system.rollback",
                    old_value=f"states_count_{old_state_count}",
                    new_value=f"checkpoint_{checkpoint_id}",
                    timestamp=datetime.now(timezone.utc),
                    source_module="state_coordinator",
                    success=True
                )
                self._operations_log.append(operation)
            
            self.stats['rollbacks_performed'] += 1
            self.stats['state_entries'] = len(self._state_store)
            
            # Emit rollback event
            if self.event_system:
                await self.event_system.emit_event("state.rollback", {
                    "checkpoint_id": checkpoint_id,
                    "restored_states": len(checkpoint.state_snapshot),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            self.logger.info(f"Rolled back to checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error rolling back to checkpoint {checkpoint_id}: {e}")
            return False
    
    async def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        
        for checkpoint in self._checkpoints.values():
            checkpoints.append({
                "checkpoint_id": checkpoint.checkpoint_id,
                "timestamp": checkpoint.timestamp.isoformat(),
                "description": checkpoint.description,
                "state_count": len(checkpoint.state_snapshot),
                "metadata": checkpoint.metadata
            })
        
        return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)
    
    # Conflict resolution
    
    async def _resolve_state_conflict(self, old_entry: StateEntry, new_entry: StateEntry) -> Optional[StateEntry]:
        """Resolve conflict between existing and new state entries"""
        try:
            self.stats['conflicts_resolved'] += 1
            
            handler = self._conflict_handlers.get(self._conflict_strategy)
            if handler:
                return await handler(old_entry, new_entry)
            else:
                self.logger.warning(f"No handler for conflict strategy {self._conflict_strategy}")
                return new_entry
                
        except Exception as e:
            self.logger.error(f"Error resolving state conflict: {e}")
            return None
    
    async def _resolve_last_write_wins(self, old_entry: StateEntry, new_entry: StateEntry) -> StateEntry:
        """Resolve conflict using last write wins strategy"""
        return new_entry
    
    async def _resolve_priority_based(self, old_entry: StateEntry, new_entry: StateEntry) -> StateEntry:
        """Resolve conflict based on priority levels"""
        if new_entry.priority.value <= old_entry.priority.value:
            return new_entry
        else:
            self.logger.debug(f"Rejected lower priority update for {new_entry.key}")
            return old_entry
    
    async def _resolve_constitutional_priority(self, old_entry: StateEntry, new_entry: StateEntry) -> StateEntry:
        """Resolve conflict with constitutional framework priority"""
        # Constitutional states always take priority
        if new_entry.state_type == StateType.CONSTITUTIONAL:
            return new_entry
        if old_entry.state_type == StateType.CONSTITUTIONAL and new_entry.state_type != StateType.CONSTITUTIONAL:
            return old_entry
        
        # Emergency states take priority over non-emergency
        if new_entry.state_type == StateType.EMERGENCY:
            return new_entry
        if old_entry.state_type == StateType.EMERGENCY and new_entry.state_type != StateType.EMERGENCY:
            return old_entry
        
        # Fall back to priority-based resolution
        return await self._resolve_priority_based(old_entry, new_entry)
    
    async def _resolve_merge_values(self, old_entry: StateEntry, new_entry: StateEntry) -> StateEntry:
        """Attempt to merge state values"""
        try:
            # Handle dictionary merging
            if isinstance(old_entry.value, dict) and isinstance(new_entry.value, dict):
                merged_value = copy.deepcopy(old_entry.value)
                merged_value.update(new_entry.value)
                
                merged_entry = copy.deepcopy(new_entry)
                merged_entry.value = merged_value
                return merged_entry
            
            # Handle list concatenation
            elif isinstance(old_entry.value, list) and isinstance(new_entry.value, list):
                merged_value = old_entry.value + new_entry.value
                
                merged_entry = copy.deepcopy(new_entry)
                merged_entry.value = merged_value
                return merged_entry
            
            # For non-mergeable types, use new value
            else:
                return new_entry
                
        except Exception as e:
            self.logger.error(f"Error merging values: {e}")
            return new_entry
    
    # Watchers and notifications
    
    async def watch_state(self, key: str, callback: Callable[[str, Optional[StateEntry]], None]):
        """Watch for changes to a specific state key"""
        self._watchers[key].append(callback)
        
    async def watch_state_type(self, state_type: StateType, callback: Callable[[str, Optional[StateEntry]], None]):
        """Watch for changes to any state of a specific type"""
        self._type_watchers[state_type].append(callback)
    
    async def _notify_watchers(self, key: str, entry: Optional[StateEntry]):
        """Notify all watchers about state changes"""
        try:
            # Notify key-specific watchers
            for callback in self._watchers.get(key, []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(key, entry)
                    else:
                        callback(key, entry)
                except Exception as e:
                    self.logger.error(f"Error in state watcher callback: {e}")
            
            # Notify type-specific watchers
            if entry:
                for callback in self._type_watchers.get(entry.state_type, []):
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(key, entry)
                        else:
                            callback(key, entry)
                    except Exception as e:
                        self.logger.error(f"Error in state type watcher callback: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error notifying watchers for {key}: {e}")
    
    # Validation and compliance
    
    async def _validate_constitutional_compliance(self, key: str, value: Any, source_module: str) -> bool:
        """Validate state change against constitutional framework"""
        try:
            if not self.constitutional_framework:
                return True
            
            context = {
                "state_key": key,
                "state_value": value,
                "source_module": source_module,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            compliance_result = self.constitutional_framework.check_compliance(context)
            
            if not compliance_result.compliant:
                self.logger.warning(f"State change {key} failed constitutional validation: {compliance_result.violations}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in constitutional validation: {e}")
            return False  # Fail safe
    
    async def _validate_state_consistency(self):
        """Validate overall state consistency"""
        try:
            with self._state_lock:
                # Check for corrupted entries
                corrupted_keys = []
                for key, entry in self._state_store.items():
                    if not entry.is_valid():
                        corrupted_keys.append(key)
                        self.logger.warning(f"Corrupted state entry detected: {key}")
                
                # Remove corrupted entries
                for key in corrupted_keys:
                    del self._state_store[key]
                    self.logger.info(f"Removed corrupted state entry: {key}")
                
                if corrupted_keys and self.event_system:
                    await self.event_system.emit_event("state.corruption_detected", {
                        "corrupted_keys": corrupted_keys,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                        
        except Exception as e:
            self.logger.error(f"Error validating state consistency: {e}")
    
    # Utility methods
    
    def _cleanup_state_history(self):
        """Clean up old state history entries"""
        try:
            # Remove histories for deleted states
            with self._state_lock:
                active_keys = set(self._state_store.keys())
                history_keys = set(self._state_history.keys())
                
                for key in history_keys - active_keys:
                    del self._state_history[key]
                        
        except Exception as e:
            self.logger.error(f"Error cleaning up state history: {e}")
    
    async def _process_pending_operations(self):
        """Process any pending state operations"""
        # Placeholder for batch operations or deferred updates
        pass
    
    async def _update_state_metrics(self):
        """Update state-related performance metrics"""
        try:
            with self._state_lock:
                total_states = len(self._state_store)
                
                # Count states by type
                type_counts = defaultdict(int)
                for entry in self._state_store.values():
                    type_counts[entry.state_type.value] += 1
                
                # Update system state with metrics
                await self.set_state(
                    key="system.state_metrics",
                    value={
                        "total_states": total_states,
                        "states_by_type": dict(type_counts),
                        "checkpoints": len(self._checkpoints),
                        "operations_logged": len(self._operations_log),
                        "last_update": datetime.now(timezone.utc).isoformat()
                    },
                    state_type=StateType.PERFORMANCE,
                    source_module="state_coordinator",
                    priority=StatePriority.LOW
                )
                
        except Exception as e:
            self.logger.error(f"Error updating state metrics: {e}")
    
    # Bulk operations
    
    async def bulk_set_states(self, states: Dict[str, Any], 
                            state_type: StateType = StateType.MODULE,
                            source_module: str = "unknown") -> Dict[str, bool]:
        """Set multiple states in a single operation"""
        results = {}
        
        for key, value in states.items():
            success = await self.set_state(
                key=key,
                value=value,
                state_type=state_type,
                source_module=source_module
            )
            results[key] = success
        
        return results
    
    async def bulk_get_states(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple states in a single operation"""
        results = {}
        
        for key in keys:
            value = await self.get_state(key)
            if value is not None:
                results[key] = value
        
        return results
    
    # Export and import
    
    async def export_state(self, include_history: bool = False) -> Dict[str, Any]:
        """Export complete state for backup or migration"""
        try:
            with self._state_lock:
                export_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "version": "1.0.0",
                    "states": {},
                    "checkpoints": {},
                    "statistics": self.stats.copy()
                }
                
                # Export current states
                for key, entry in self._state_store.items():
                    export_data["states"][key] = {
                        "value": entry.value,
                        "state_type": entry.state_type.value,
                        "timestamp": entry.timestamp.isoformat(),
                        "version": entry.version,
                        "source_module": entry.source_module,
                        "priority": entry.priority.name,
                        "metadata": entry.metadata
                    }
                
                # Export checkpoints
                for checkpoint_id, checkpoint in self._checkpoints.items():
                    export_data["checkpoints"][checkpoint_id] = {
                        "timestamp": checkpoint.timestamp.isoformat(),
                        "description": checkpoint.description,
                        "metadata": checkpoint.metadata,
                        "state_count": len(checkpoint.state_snapshot)
                    }
                
                # Export history if requested
                if include_history:
                    export_data["history"] = {}
                    for key, history in self._state_history.items():
                        export_data["history"][key] = [
                            {
                                "value": entry.value,
                                "timestamp": entry.timestamp.isoformat(),
                                "version": entry.version
                            } for entry in history if entry
                        ]
                
                return export_data
                
        except Exception as e:
            self.logger.error(f"Error exporting state: {e}")
            return {}
    
    async def import_state(self, import_data: Dict[str, Any], merge: bool = False) -> bool:
        """Import state from backup or migration"""
        try:
            if not merge:
                # Clear existing state
                with self._state_lock:
                    self._state_store.clear()
                    self._checkpoints.clear()
                    self._state_history.clear()
            
            # Import states
            states_data = import_data.get("states", {})
            for key, state_info in states_data.items():
                await self.set_state(
                    key=key,
                    value=state_info["value"],
                    state_type=StateType(state_info["state_type"]),
                    source_module=state_info["source_module"],
                    priority=StatePriority[state_info["priority"]],
                    metadata=state_info.get("metadata", {})
                )
            
            self.logger.info(f"Imported {len(states_data)} states")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing state: {e}")
            return False
    
    # System management
    
    async def shutdown(self) -> bool:
        """Shutdown the state coordinator"""
        try:
            self.logger.info("Shutting down StateCoordinator...")
            
            # Stop synchronization loop
            self._running = False
            if self._sync_task:
                self._sync_task.cancel()
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    pass
            
            # Create final checkpoint
            await self.create_checkpoint("shutdown_checkpoint")
            
            self.logger.info("StateCoordinator shutdown completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during StateCoordinator shutdown: {e}")
            return False
    
    def get_coordinator_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coordinator statistics"""
        with self._state_lock:
            return {
                "statistics": self.stats.copy(),
                "state_counts": {
                    "total_states": len(self._state_store),
                    "states_by_type": {
                        state_type.value: sum(
                            1 for entry in self._state_store.values() 
                            if entry.state_type == state_type
                        ) for state_type in StateType
                    }
                },
                "checkpoints": len(self._checkpoints),
                "watchers": {
                    "key_watchers": len(self._watchers),
                    "type_watchers": len(self._type_watchers)
                },
                "configuration": {
                    "sync_interval": self._sync_interval,
                    "conflict_strategy": self._conflict_strategy.value,
                    "running": self._running
                }
            }
    
    def __repr__(self) -> str:
        """String representation"""
        with self._state_lock:
            return f"StateCoordinator(states={len(self._state_store)}, checkpoints={len(self._checkpoints)}, running={self._running})"


# Utility functions

async def create_state_coordinator(event_system=None, constitutional_framework=None) -> StateCoordinator:
    """Create and initialize a state coordinator"""
    coordinator = StateCoordinator(event_system, constitutional_framework)
    
    if await coordinator.initialize():
        return coordinator
    else:
        raise RuntimeError("Failed to initialize state coordinator")


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create state coordinator
        coordinator = StateCoordinator()
        await coordinator.initialize()
        
        # Set some test states
        await coordinator.set_state("test.value1", 42, StateType.MODULE, "test_module")
        await coordinator.set_state("test.value2", "hello", StateType.GLOBAL, "system")
        
        # Get states
        value1 = await coordinator.get_state("test.value1")
        value2 = await coordinator.get_state("test.value2")
        print(f"Values: {value1}, {value2}")
        
        # Create checkpoint
        checkpoint_id = await coordinator.create_checkpoint("test_checkpoint")
        print(f"Created checkpoint: {checkpoint_id}")
        
        # Update state and rollback
        await coordinator.set_state("test.value1", 99, StateType.MODULE, "test_module")
        print(f"Updated value: {await coordinator.get_state('test.value1')}")
        
        await coordinator.rollback_to_checkpoint(checkpoint_id)
        print(f"After rollback: {await coordinator.get_state('test.value1')}")
        
        # Shutdown
        await coordinator.shutdown()
    
    # Run example
    asyncio.run(main())