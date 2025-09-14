"""
Neural Field Control System - Event System
==========================================

The EventSystem provides a robust, asynchronous event-driven communication
framework for all NFCS components. It enables decoupled inter-module communication,
event routing, filtering, and processing with constitutional compliance verification.

Key Features:
- Asynchronous event publishing and subscription
- Event filtering and routing
- Priority-based event processing
- Event history and replay capabilities
- Constitutional compliance verification
- Event aggregation and pattern detection
- Reliable delivery and retry mechanisms
- Performance monitoring and metrics
- Emergency event handling

Architecture:
Implements a publish-subscribe pattern with centralized event routing and
distributed processing. All events are validated against constitutional
policies and logged for audit and analysis purposes.
"""

import asyncio
import logging
import threading
import time
import json
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import weakref
import uuid
import re


class EventType(Enum):
    """Standard NFCS event types"""
    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_MODE_CHANGED = "system.mode_changed"
    
    # Module events
    MODULE_REGISTERED = "module.registered"
    MODULE_UNREGISTERED = "module.unregistered"
    MODULE_STARTED = "module.started"
    MODULE_STOPPED = "module.stopped"
    MODULE_ERROR = "module.error"
    MODULE_HEALTH_WARNING = "module.health_warning"
    MODULE_EMERGENCY = "module.emergency"
    
    # State events
    STATE_CHANGED = "state.changed"
    STATE_CHECKPOINT_CREATED = "state.checkpoint_created"
    STATE_ROLLBACK = "state.rollback"
    STATE_CORRUPTION_DETECTED = "state.corruption_detected"
    
    # Constitutional events
    CONSTITUTIONAL_VIOLATION = "constitutional.violation"
    CONSTITUTIONAL_POLICY_UPDATED = "constitutional.policy_updated"
    CONSTITUTIONAL_COMPLIANCE_CHECK = "constitutional.compliance_check"
    
    # Performance events
    PERFORMANCE_ALERT = "performance.alert"
    PERFORMANCE_THRESHOLD_EXCEEDED = "performance.threshold_exceeded"
    PERFORMANCE_DEGRADATION = "performance.degradation"
    
    # Emergency events
    EMERGENCY_DETECTED = "emergency.detected"
    EMERGENCY_RESOLVED = "emergency.resolved"
    EMERGENCY_PROTOCOL_ACTIVATED = "emergency.protocol_activated"
    
    # Cognitive events
    COGNITIVE_DECISION = "cognitive.decision"
    COGNITIVE_REFLECTION = "cognitive.reflection"
    COGNITIVE_MEMORY_UPDATE = "cognitive.memory_update"
    COGNITIVE_BOUNDARY_VIOLATION = "cognitive.boundary_violation"
    
    # Custom and user-defined events
    CUSTOM = "custom"


class EventPriority(Enum):
    """Event priority levels"""
    CRITICAL = 1    # Emergency, safety-critical
    HIGH = 2        # Constitutional, system errors
    MEDIUM = 3      # Module operations, warnings
    LOW = 4         # Information, metrics


@dataclass
class Event:
    """Event data structure"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    source: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: EventPriority = EventPriority.MEDIUM
    
    # Event content
    data: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    # Processing metadata
    processed: bool = False
    retry_count: int = 0
    max_retries: int = 3
    ttl_seconds: Optional[int] = None
    
    # Tracing and correlation
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate event after creation"""
        if not self.event_type:
            raise ValueError("Event type cannot be empty")
        
        # Set TTL based on priority if not specified
        if self.ttl_seconds is None:
            if self.priority == EventPriority.CRITICAL:
                self.ttl_seconds = 3600  # 1 hour
            elif self.priority == EventPriority.HIGH:
                self.ttl_seconds = 1800  # 30 minutes
            else:
                self.ttl_seconds = 600   # 10 minutes
    
    def is_expired(self) -> bool:
        """Check if event has expired"""
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > self.ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.name,
            "data": self.data,
            "tags": list(self.tags),
            "processed": self.processed,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "ttl_seconds": self.ttl_seconds,
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        event = cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=data["event_type"],
            source=data.get("source", "unknown"),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            priority=EventPriority[data.get("priority", "MEDIUM")],
            data=data.get("data", {}),
            tags=set(data.get("tags", [])),
            processed=data.get("processed", False),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            ttl_seconds=data.get("ttl_seconds"),
            correlation_id=data.get("correlation_id"),
            parent_event_id=data.get("parent_event_id")
        )
        return event


@dataclass
class EventSubscription:
    """Event subscription information"""
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subscriber_name: str = "unknown"
    event_pattern: str = "*"  # Supports wildcards
    callback: Optional[Callable] = None
    filter_function: Optional[Callable[[Event], bool]] = None
    
    # Subscription options
    priority_filter: Optional[EventPriority] = None
    source_filter: Optional[str] = None
    tag_filter: Optional[Set[str]] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_event_received: Optional[datetime] = None
    events_received: int = 0
    
    # Weak reference to avoid circular references
    _callback_ref: Optional[weakref.ref] = field(default=None, init=False)
    
    def __post_init__(self):
        """Set up weak reference for callback"""
        if self.callback:
            try:
                self._callback_ref = weakref.ref(self.callback)
            except TypeError:
                # Callback is not a bound method, store directly
                pass
    
    def get_callback(self) -> Optional[Callable]:
        """Get callback, handling weak references"""
        if self._callback_ref:
            callback = self._callback_ref()
            return callback if callback else None
        return self.callback
    
    def matches_event(self, event: Event) -> bool:
        """Check if subscription matches an event"""
        try:
            # Check event pattern (supports wildcards)
            if not self._matches_pattern(event.event_type, self.event_pattern):
                return False
            
            # Check priority filter
            if self.priority_filter and event.priority != self.priority_filter:
                return False
            
            # Check source filter
            if self.source_filter and event.source != self.source_filter:
                return False
            
            # Check tag filter
            if self.tag_filter and not (self.tag_filter & event.tags):
                return False
            
            # Check custom filter function
            if self.filter_function and not self.filter_function(event):
                return False
            
            return True
            
        except Exception as e:
            logging.getLogger("EventSystem").error(f"Error matching event {event.event_id}: {e}")
            return False
    
    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches pattern (supports wildcards)"""
        if pattern == "*":
            return True
        
        # Convert glob pattern to regex
        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        return bool(re.match(f"^{regex_pattern}$", event_type))


class EventAggregator:
    """Aggregates events for pattern detection and analysis"""
    
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._events: deque = deque()
        self._patterns: Dict[str, List[Event]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def add_event(self, event: Event):
        """Add event to aggregator"""
        with self._lock:
            self._events.append(event)
            self._cleanup_old_events()
            self._update_patterns(event)
    
    def _cleanup_old_events(self):
        """Remove events older than window"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.window_seconds)
        
        while self._events and self._events[0].timestamp < cutoff_time:
            old_event = self._events.popleft()
            # Remove from patterns too
            for pattern_events in self._patterns.values():
                if old_event in pattern_events:
                    pattern_events.remove(old_event)
    
    def _update_patterns(self, event: Event):
        """Update pattern detection with new event"""
        # Simple pattern detection - count events by type
        self._patterns[event.event_type].append(event)
        
        # Could add more sophisticated pattern detection here
    
    def get_event_count(self, event_type: str = None, window_seconds: int = None) -> int:
        """Get count of events in time window"""
        window = window_seconds or self.window_seconds
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=window)
        
        with self._lock:
            if event_type:
                return sum(1 for event in self._patterns.get(event_type, []) 
                          if event.timestamp >= cutoff_time)
            else:
                return sum(1 for event in self._events if event.timestamp >= cutoff_time)
    
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns in recent events"""
        patterns = []
        
        with self._lock:
            # Detect high-frequency events
            for event_type, events in self._patterns.items():
                count = len(events)
                if count > 10:  # Threshold for high frequency
                    patterns.append({
                        "type": "high_frequency",
                        "event_type": event_type,
                        "count": count,
                        "window_seconds": self.window_seconds
                    })
            
            # Could add more pattern detection logic here
            # - Event correlation analysis
            # - Sequence pattern detection
            # - Anomaly detection
        
        return patterns


class EventSystem:
    """
    NFCS Event System
    
    Provides comprehensive event-driven communication framework for all
    NFCS components with constitutional compliance and performance monitoring.
    """
    
    def __init__(self, constitutional_framework=None):
        """Initialize the Event System"""
        self.logger = logging.getLogger("EventSystem")
        
        # Dependencies
        self.constitutional_framework = constitutional_framework
        
        # Event storage and processing
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_history: deque = deque(maxlen=10000)
        self._event_lock = threading.RLock()
        
        # Subscriptions management
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._subscriptions_by_pattern: Dict[str, List[str]] = defaultdict(list)
        self._subscription_lock = threading.Lock()
        
        # Event processing
        self._processing_tasks: List[asyncio.Task] = []
        self._worker_count = 3
        self._running = False
        
        # Event aggregation and pattern detection
        self._aggregator = EventAggregator()
        
        # Retry mechanism
        self._retry_queue: asyncio.Queue = asyncio.Queue()
        self._retry_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.stats = {
            'total_events_published': 0,
            'total_events_processed': 0,
            'total_events_failed': 0,
            'total_subscriptions': 0,
            'active_subscriptions': 0,
            'patterns_detected': 0,
            'events_retried': 0,
            'last_event_time': None,
            'processing_errors': 0
        }
        
        # Event type registry
        self._registered_event_types: Set[str] = set()
        self._register_standard_event_types()
        
        self.logger.info("EventSystem initialized")
    
    def _register_standard_event_types(self):
        """Register all standard NFCS event types"""
        for event_type in EventType:
            self._registered_event_types.add(event_type.value)
    
    async def initialize(self) -> bool:
        """Initialize the event system"""
        try:
            self.logger.info("Initializing EventSystem...")
            
            # Start event processing workers
            if not await self._start_event_processors():
                return False
            
            # Start retry processor
            if not await self._start_retry_processor():
                return False
            
            self._running = True
            self.logger.info("EventSystem initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize EventSystem: {e}")
            return False
    
    async def _start_event_processors(self) -> bool:
        """Start event processing worker tasks"""
        try:
            for i in range(self._worker_count):
                task = asyncio.create_task(self._event_processor_worker(f"worker_{i}"))
                self._processing_tasks.append(task)
            
            self.logger.info(f"Started {self._worker_count} event processor workers")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start event processors: {e}")
            return False
    
    async def _start_retry_processor(self) -> bool:
        """Start retry processing task"""
        try:
            self._retry_task = asyncio.create_task(self._retry_processor_worker())
            self.logger.info("Started retry processor")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start retry processor: {e}")
            return False
    
    async def _event_processor_worker(self, worker_name: str):
        """Event processing worker loop"""
        self.logger.info(f"Event processor {worker_name} started")
        
        while self._running:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                
                # Process the event
                await self._process_event(event)
                
                # Mark task as done
                self._event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue  # No events to process, continue loop
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in event processor {worker_name}: {e}")
                self.stats['processing_errors'] += 1
        
        self.logger.info(f"Event processor {worker_name} stopped")
    
    async def _retry_processor_worker(self):
        """Retry processing worker loop"""
        self.logger.info("Retry processor started")
        
        while self._running:
            try:
                # Get event from retry queue
                event = await asyncio.wait_for(self._retry_queue.get(), timeout=5.0)
                
                # Check if event should be retried
                if event.retry_count < event.max_retries and not event.is_expired():
                    event.retry_count += 1
                    
                    # Wait before retry (exponential backoff)
                    wait_time = min(2 ** event.retry_count, 30)  # Max 30 seconds
                    await asyncio.sleep(wait_time)
                    
                    # Re-queue for processing
                    await self._event_queue.put(event)
                    self.stats['events_retried'] += 1
                    
                    self.logger.debug(f"Retrying event {event.event_id} (attempt {event.retry_count})")
                else:
                    # Event exhausted retries or expired
                    self.logger.warning(f"Event {event.event_id} failed after {event.retry_count} retries")
                    self.stats['total_events_failed'] += 1
                
                # Mark retry task as done
                self._retry_queue.task_done()
                
            except asyncio.TimeoutError:
                continue  # No events to retry, continue loop
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in retry processor: {e}")
        
        self.logger.info("Retry processor stopped")
    
    async def emit_event(self, 
                        event_type: str, 
                        data: Dict[str, Any] = None,
                        source: str = "unknown",
                        priority: EventPriority = EventPriority.MEDIUM,
                        tags: Optional[Set[str]] = None,
                        correlation_id: Optional[str] = None,
                        parent_event_id: Optional[str] = None) -> str:
        """
        Emit an event
        
        Args:
            event_type: Type of event to emit
            data: Event data payload
            source: Source module/component emitting the event
            priority: Event priority level
            tags: Optional tags for event categorization
            correlation_id: Optional correlation ID for event tracing
            parent_event_id: Optional parent event ID for event chaining
            
        Returns:
            str: Event ID of the emitted event
        """
        try:
            # Create event
            event = Event(
                event_type=event_type,
                source=source,
                priority=priority,
                data=data or {},
                tags=tags or set(),
                correlation_id=correlation_id,
                parent_event_id=parent_event_id
            )
            
            # Validate event against constitutional framework
            if self.constitutional_framework:
                if not await self._validate_event_compliance(event):
                    self.logger.warning(f"Event {event.event_id} failed constitutional validation")
                    return ""
            
            # Add to event queue for processing
            await self._event_queue.put(event)
            
            # Update statistics
            self.stats['total_events_published'] += 1
            self.stats['last_event_time'] = datetime.now(timezone.utc).isoformat()
            
            # Add to aggregator for pattern detection
            self._aggregator.add_event(event)
            
            self.logger.debug(f"Emitted event {event.event_id} of type {event_type}")
            return event.event_id
            
        except Exception as e:
            self.logger.error(f"Error emitting event {event_type}: {e}")
            return ""
    
    async def _process_event(self, event: Event):
        """Process a single event by delivering to subscribers"""
        try:
            self.logger.debug(f"Processing event {event.event_id} of type {event.event_type}")
            
            # Add to event history
            with self._event_lock:
                self._event_history.append(event)
            
            # Find matching subscriptions
            matching_subscriptions = []
            
            with self._subscription_lock:
                for subscription in self._subscriptions.values():
                    if subscription.matches_event(event):
                        matching_subscriptions.append(subscription)
            
            # Deliver event to subscribers
            delivery_tasks = []
            for subscription in matching_subscriptions:
                task = asyncio.create_task(self._deliver_event_to_subscriber(event, subscription))
                delivery_tasks.append(task)
            
            # Wait for all deliveries to complete
            if delivery_tasks:
                results = await asyncio.gather(*delivery_tasks, return_exceptions=True)
                
                # Check for delivery failures
                failures = [r for r in results if isinstance(r, Exception)]
                if failures:
                    self.logger.warning(f"Event {event.event_id} had {len(failures)} delivery failures")
                    
                    # Queue for retry if there were failures
                    if len(failures) > 0:
                        await self._retry_queue.put(event)
                        return
            
            # Mark event as processed
            event.processed = True
            self.stats['total_events_processed'] += 1
            
        except Exception as e:
            self.logger.error(f"Error processing event {event.event_id}: {e}")
            # Queue for retry on processing error
            await self._retry_queue.put(event)
    
    async def _deliver_event_to_subscriber(self, event: Event, subscription: EventSubscription):
        """Deliver event to a specific subscriber"""
        try:
            callback = subscription.get_callback()
            if not callback:
                # Callback no longer exists (weak reference expired)
                await self.unsubscribe(subscription.subscription_id)
                return
            
            # Update subscription statistics
            subscription.last_event_received = datetime.now(timezone.utc)
            subscription.events_received += 1
            
            # Call the subscriber callback
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                # Run sync callback in thread pool to avoid blocking
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, callback, event)
            
            self.logger.debug(f"Delivered event {event.event_id} to subscriber {subscription.subscriber_name}")
            
        except Exception as e:
            self.logger.error(f"Error delivering event {event.event_id} to {subscription.subscriber_name}: {e}")
            raise  # Re-raise to trigger retry
    
    async def subscribe(self,
                       event_pattern: str,
                       callback: Callable,
                       subscriber_name: str = "unknown",
                       priority_filter: Optional[EventPriority] = None,
                       source_filter: Optional[str] = None,
                       tag_filter: Optional[Set[str]] = None,
                       filter_function: Optional[Callable[[Event], bool]] = None) -> str:
        """
        Subscribe to events matching a pattern
        
        Args:
            event_pattern: Event type pattern (supports wildcards)
            callback: Callback function to handle events
            subscriber_name: Name of the subscriber
            priority_filter: Optional priority filter
            source_filter: Optional source filter
            tag_filter: Optional tag filter
            filter_function: Optional custom filter function
            
        Returns:
            str: Subscription ID
        """
        try:
            subscription = EventSubscription(
                subscriber_name=subscriber_name,
                event_pattern=event_pattern,
                callback=callback,
                priority_filter=priority_filter,
                source_filter=source_filter,
                tag_filter=tag_filter,
                filter_function=filter_function
            )
            
            with self._subscription_lock:
                self._subscriptions[subscription.subscription_id] = subscription
                self._subscriptions_by_pattern[event_pattern].append(subscription.subscription_id)
            
            # Update statistics
            self.stats['total_subscriptions'] += 1
            self.stats['active_subscriptions'] = len(self._subscriptions)
            
            self.logger.info(f"Created subscription {subscription.subscription_id} for pattern '{event_pattern}' by {subscriber_name}")
            return subscription.subscription_id
            
        except Exception as e:
            self.logger.error(f"Error creating subscription: {e}")
            return ""
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events
        
        Args:
            subscription_id: Subscription ID to remove
            
        Returns:
            bool: True if unsubscribed successfully, False otherwise
        """
        try:
            with self._subscription_lock:
                if subscription_id not in self._subscriptions:
                    return False
                
                subscription = self._subscriptions[subscription_id]
                
                # Remove from pattern mapping
                pattern_subscriptions = self._subscriptions_by_pattern.get(subscription.event_pattern, [])
                if subscription_id in pattern_subscriptions:
                    pattern_subscriptions.remove(subscription_id)
                
                # Remove subscription
                del self._subscriptions[subscription_id]
            
            # Update statistics
            self.stats['active_subscriptions'] = len(self._subscriptions)
            
            self.logger.info(f"Unsubscribed {subscription_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unsubscribing {subscription_id}: {e}")
            return False
    
    async def get_subscription_info(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a subscription"""
        with self._subscription_lock:
            subscription = self._subscriptions.get(subscription_id)
            if not subscription:
                return None
            
            return {
                "subscription_id": subscription.subscription_id,
                "subscriber_name": subscription.subscriber_name,
                "event_pattern": subscription.event_pattern,
                "priority_filter": subscription.priority_filter.name if subscription.priority_filter else None,
                "source_filter": subscription.source_filter,
                "tag_filter": list(subscription.tag_filter) if subscription.tag_filter else None,
                "created_at": subscription.created_at.isoformat(),
                "last_event_received": subscription.last_event_received.isoformat() if subscription.last_event_received else None,
                "events_received": subscription.events_received
            }
    
    async def list_subscriptions(self, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all subscriptions, optionally filtered by pattern"""
        subscriptions = []
        
        with self._subscription_lock:
            for subscription in self._subscriptions.values():
                if pattern and pattern not in subscription.event_pattern:
                    continue
                
                info = await self.get_subscription_info(subscription.subscription_id)
                if info:
                    subscriptions.append(info)
        
        return subscriptions
    
    # Event history and querying
    
    async def get_event_history(self, 
                              limit: int = 100,
                              event_type_filter: Optional[str] = None,
                              source_filter: Optional[str] = None,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get event history with optional filtering"""
        events = []
        
        with self._event_lock:
            for event in reversed(list(self._event_history)):
                # Apply filters
                if event_type_filter and event_type_filter not in event.event_type:
                    continue
                
                if source_filter and event.source != source_filter:
                    continue
                
                if start_time and event.timestamp < start_time:
                    continue
                
                if end_time and event.timestamp > end_time:
                    continue
                
                events.append(event.to_dict())
                
                if len(events) >= limit:
                    break
        
        return events
    
    async def get_events_by_correlation_id(self, correlation_id: str) -> List[Dict[str, Any]]:
        """Get all events with a specific correlation ID"""
        events = []
        
        with self._event_lock:
            for event in self._event_history:
                if event.correlation_id == correlation_id:
                    events.append(event.to_dict())
        
        return sorted(events, key=lambda x: x["timestamp"])
    
    async def replay_events(self, 
                          start_time: datetime,
                          end_time: Optional[datetime] = None,
                          event_type_filter: Optional[str] = None) -> int:
        """Replay events from history to current subscribers"""
        replayed_count = 0
        end_time = end_time or datetime.now(timezone.utc)
        
        with self._event_lock:
            for event in self._event_history:
                if (event.timestamp >= start_time and 
                    event.timestamp <= end_time):
                    
                    if event_type_filter and event_type_filter not in event.event_type:
                        continue
                    
                    # Create a copy of the event for replay
                    replay_event = Event.from_dict(event.to_dict())
                    replay_event.event_id = str(uuid.uuid4())  # New ID for replay
                    replay_event.parent_event_id = event.event_id
                    replay_event.data["replayed"] = True
                    replay_event.data["original_timestamp"] = event.timestamp.isoformat()
                    
                    # Process the replay event
                    await self._process_event(replay_event)
                    replayed_count += 1
        
        self.logger.info(f"Replayed {replayed_count} events from {start_time} to {end_time}")
        return replayed_count
    
    # Pattern detection and analysis
    
    async def detect_event_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns in recent events"""
        patterns = self._aggregator.detect_patterns()
        
        # Update statistics
        self.stats['patterns_detected'] += len(patterns)
        
        # Emit pattern detection events
        for pattern in patterns:
            await self.emit_event(
                event_type="system.pattern_detected",
                data=pattern,
                source="event_system",
                priority=EventPriority.MEDIUM,
                tags={"pattern_detection", "analysis"}
            )
        
        return patterns
    
    async def get_event_statistics(self, window_seconds: int = 300) -> Dict[str, Any]:
        """Get event statistics for a time window"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        
        stats = {
            "window_seconds": window_seconds,
            "total_events": 0,
            "events_by_type": defaultdict(int),
            "events_by_source": defaultdict(int),
            "events_by_priority": defaultdict(int),
            "avg_events_per_second": 0.0
        }
        
        with self._event_lock:
            for event in self._event_history:
                if event.timestamp >= cutoff_time:
                    stats["total_events"] += 1
                    stats["events_by_type"][event.event_type] += 1
                    stats["events_by_source"][event.source] += 1
                    stats["events_by_priority"][event.priority.name] += 1
        
        # Calculate average events per second
        if stats["total_events"] > 0:
            stats["avg_events_per_second"] = stats["total_events"] / window_seconds
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats["events_by_type"] = dict(stats["events_by_type"])
        stats["events_by_source"] = dict(stats["events_by_source"])
        stats["events_by_priority"] = dict(stats["events_by_priority"])
        
        return stats
    
    # Constitutional compliance
    
    async def _validate_event_compliance(self, event: Event) -> bool:
        """Validate event against constitutional framework"""
        try:
            if not self.constitutional_framework:
                return True
            
            context = {
                "event_type": event.event_type,
                "event_data": event.data,
                "event_source": event.source,
                "event_priority": event.priority.name,
                "timestamp": event.timestamp.isoformat()
            }
            
            compliance_result = self.constitutional_framework.check_compliance(context)
            
            if not compliance_result.compliant:
                # Emit constitutional violation event
                await self.emit_event(
                    event_type=EventType.CONSTITUTIONAL_VIOLATION.value,
                    data={
                        "original_event_id": event.event_id,
                        "violations": compliance_result.violations,
                        "blocked": True
                    },
                    source="event_system",
                    priority=EventPriority.HIGH
                )
                
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in constitutional validation: {e}")
            return False  # Fail safe
    
    # Emergency handling
    
    async def emit_emergency_event(self, 
                                 emergency_type: str,
                                 data: Dict[str, Any],
                                 source: str) -> str:
        """Emit a high-priority emergency event"""
        return await self.emit_event(
            event_type=f"emergency.{emergency_type}",
            data=data,
            source=source,
            priority=EventPriority.CRITICAL,
            tags={"emergency", emergency_type}
        )
    
    # System management
    
    async def shutdown(self) -> bool:
        """Shutdown the event system"""
        try:
            self.logger.info("Shutting down EventSystem...")
            
            # Stop processing
            self._running = False
            
            # Wait for queue to empty
            await self._event_queue.join()
            await self._retry_queue.join()
            
            # Cancel processing tasks
            for task in self._processing_tasks:
                task.cancel()
            
            if self._retry_task:
                self._retry_task.cancel()
            
            # Wait for tasks to complete
            if self._processing_tasks:
                await asyncio.gather(*self._processing_tasks, return_exceptions=True)
            
            if self._retry_task:
                try:
                    await self._retry_task
                except asyncio.CancelledError:
                    pass
            
            # Emit shutdown event
            await self.emit_event(
                event_type=EventType.SYSTEM_SHUTDOWN.value,
                data={"shutdown_time": datetime.now(timezone.utc).isoformat()},
                source="event_system",
                priority=EventPriority.HIGH
            )
            
            self.logger.info("EventSystem shutdown completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during EventSystem shutdown: {e}")
            return False
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        with self._subscription_lock:
            active_subscriptions = len(self._subscriptions)
        
        with self._event_lock:
            history_size = len(self._event_history)
        
        return {
            "statistics": self.stats.copy(),
            "system_state": {
                "running": self._running,
                "active_subscriptions": active_subscriptions,
                "event_history_size": history_size,
                "worker_count": self._worker_count,
                "queue_size": self._event_queue.qsize(),
                "retry_queue_size": self._retry_queue.qsize()
            },
            "registered_event_types": list(self._registered_event_types),
            "aggregator_stats": {
                "window_seconds": self._aggregator.window_seconds,
                "total_events_in_window": self._aggregator.get_event_count()
            }
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return f"EventSystem(subscriptions={len(self._subscriptions)}, running={self._running})"


# Utility functions

async def create_event_system(constitutional_framework=None) -> EventSystem:
    """Create and initialize an event system"""
    event_system = EventSystem(constitutional_framework)
    
    if await event_system.initialize():
        return event_system
    else:
        raise RuntimeError("Failed to initialize event system")


# Decorators for easy event handling

def event_handler(event_pattern: str, priority_filter: Optional[EventPriority] = None):
    """Decorator to mark a function as an event handler"""
    def decorator(func: Callable):
        func._event_pattern = event_pattern
        func._priority_filter = priority_filter
        func._is_event_handler = True
        return func
    return decorator


class EventHandlerRegistry:
    """Registry for automatic event handler subscription"""
    
    def __init__(self, event_system: EventSystem):
        self.event_system = event_system
        self.registered_handlers: Dict[str, str] = {}  # function -> subscription_id
    
    async def register_handlers(self, obj: Any, name_prefix: str = "") -> List[str]:
        """Register all event handlers in an object"""
        subscription_ids = []
        
        for attr_name in dir(obj):
            attr = getattr(obj, attr_name)
            
            if (callable(attr) and 
                hasattr(attr, '_is_event_handler') and 
                attr._is_event_handler):
                
                subscription_id = await self.event_system.subscribe(
                    event_pattern=attr._event_pattern,
                    callback=attr,
                    subscriber_name=f"{name_prefix}{obj.__class__.__name__}.{attr_name}",
                    priority_filter=getattr(attr, '_priority_filter', None)
                )
                
                if subscription_id:
                    self.registered_handlers[f"{obj}.{attr_name}"] = subscription_id
                    subscription_ids.append(subscription_id)
        
        return subscription_ids


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create event system
        event_system = EventSystem()
        await event_system.initialize()
        
        # Subscribe to events
        async def handle_system_events(event: Event):
            print(f"Received system event: {event.event_type} from {event.source}")
        
        subscription_id = await event_system.subscribe(
            event_pattern="system.*",
            callback=handle_system_events,
            subscriber_name="test_subscriber"
        )
        
        # Emit some test events
        await event_system.emit_event(
            event_type="system.test",
            data={"message": "Hello World"},
            source="test_module"
        )
        
        # Wait a moment for processing
        await asyncio.sleep(0.5)
        
        # Get statistics
        stats = event_system.get_system_statistics()
        print(f"System stats: {stats}")
        
        # Cleanup
        await event_system.unsubscribe(subscription_id)
        await event_system.shutdown()
    
    # Run example
    asyncio.run(main())