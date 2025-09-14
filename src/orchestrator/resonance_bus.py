"""
Resonance Bus for NFCS (Neural Field Control System)
===================================================

High-performance typed Pub/Sub system for inter-module communication.
Provides thread-safe event exchange between all NFCS components with buffering
and delivery guarantees.

Key Event Topics:
- signals.eta: ESC → System (order parameters)  
- metrics.risk: RiskMonitor → Constitution (risks and threats)
- control.intent: Constitution → CGL/Kuramoto (control intentions)
- orchestration.emergency: Emergency signals (critical events)
- telemetry.event: System telemetry (monitoring and logs)
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone  
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np


class EventPriority(Enum):
    """Event priorities in the resonance bus"""
    LOW = 1
    NORMAL = 2  
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class TopicType(Enum):
    """Topic types in the resonance bus"""
    SIGNALS_ETA = "signals.eta"
    METRICS_RISK = "metrics.risk" 
    CONTROL_INTENT = "control.intent"
    ORCHESTRATION_EMERGENCY = "orchestration.emergency"
    TELEMETRY_EVENT = "telemetry.event"
    SYSTEM_STATUS = "system.status"
    MODULE_SYNC = "module.sync"


@dataclass
class EventPayload:
    """Base class for event payload data"""
    timestamp: float = field(default_factory=time.time)
    source_module: str = "unknown"
    event_id: str = field(default_factory=lambda: f"evt_{int(time.time() * 1000000)}")


@dataclass 
class EtaSignalPayload(EventPayload):
    """Payload for ESC signals (signals.eta)"""
    eta_value: float = 0.0
    resonance_frequencies: List[float] = field(default_factory=list)
    active_oscillators: Dict[str, Any] = field(default_factory=dict)
    semantic_coherence: float = 0.0


@dataclass
class RiskMetricsPayload(EventPayload):
    """Payload for risk metrics (metrics.risk)"""
    hallucination_number: float = 0.0
    defect_density_mean: float = 0.0
    coherence_global: float = 0.0
    coherence_modular: float = 0.0
    systemic_risk: float = 0.0
    risk_level: str = "NORMAL"  # NORMAL, WARNING, CRITICAL, EMERGENCY
    risk_trend: str = "STABLE"  # INCREASING, DECREASING, STABLE
    violations_count: int = 0


@dataclass
class ControlIntentPayload(EventPayload):
    """Payload for control intentions (control.intent)"""
    decision_type: str = "ACCEPT"  # ACCEPT, REJECT, EMERGENCY
    u_field_limits: Dict[str, float] = field(default_factory=dict)
    kuramoto_masks: Dict[str, Any] = field(default_factory=dict) 
    esc_normalization_mode: str = "standard"
    freedom_window: float = 0.1
    emergency_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmergencyPayload(EventPayload):
    """Payload for emergency events (orchestration.emergency)"""
    emergency_type: str = "UNKNOWN"
    severity_level: int = 1  # 1-5, where 5 is critical
    trigger_reason: str = ""
    affected_modules: List[str] = field(default_factory=list)
    required_actions: List[str] = field(default_factory=list)
    recovery_estimated_time: Optional[float] = None


@dataclass
class TelemetryPayload(EventPayload):
    """Полезная нагрузка для телеметрии (telemetry.event)"""
    metric_name: str = ""
    metric_value: Union[float, int, str] = 0
    metric_unit: str = ""
    module_state: Dict[str, Any] = field(default_factory=dict)
    performance_data: Dict[str, float] = field(default_factory=dict)


@dataclass
class BusEvent:
    """Event in the resonance bus with enhanced data handling"""
    topic: TopicType
    payload: Optional[EventPayload] = None
    priority: EventPriority = EventPriority.NORMAL
    ttl_seconds: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    
    # Enhanced compatibility parameters
    data: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        \"\"\"Initialize with backward compatibility for different input formats\"\"\"
        # Handle legacy data parameter
        if self.data is not None and self.payload is None:
            self.payload = self.data
            
        # Set timestamp if provided
        if self.timestamp is not None:
            self.created_at = self.timestamp
    
    def is_expired(self) -> bool:
        """Check if the event TTL has expired"""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds


class EventHandler:
    """Обработчик событий для подписчиков"""
    
    def __init__(self, 
                 callback: Callable[[BusEvent], None],
                 handler_id: str,
                 topic_filter: Optional[Set[TopicType]] = None,
                 priority_filter: Optional[Set[EventPriority]] = None):
        self.callback = callback
        self.handler_id = handler_id
        self.topic_filter = topic_filter or set()
        self.priority_filter = priority_filter or set()
        self.created_at = time.time()
        self.events_processed = 0
        self.last_activity = time.time()
        self.is_active = True
        
    def should_handle(self, event: BusEvent) -> bool:
        """Определить, должен ли обработчик process событие"""
        if not self.is_active:
            return False
            
        if self.topic_filter and event.topic not in self.topic_filter:
            return False
            
        if self.priority_filter and event.priority not in self.priority_filter:
            return False
            
        return True
        
    def handle_event(self, event: BusEvent) -> bool:
        """Process событие"""
        try:
            if not self.should_handle(event):
                return False
                
            self.callback(event)
            self.events_processed += 1
            self.last_activity = time.time()
            return True
            
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Error in event handler {self.handler_id}: {e}"
            )
            return False


class ResonanceBus:
    """
    Резонансная шина для NFCS
    
    Высокопроизводительная типизированная system Pub/Sub с:
    - Thread-safe операциями
    - Приоритизацией событий
    - Буферизацией и TTL
    - Метриками производительности
    - Graceful degradation при перегрузке
    """
    
    def __init__(self, 
                 max_buffer_size: int = 10000,
                 max_handlers: int = 1000,
                 cleanup_interval_seconds: float = 60.0,
                 enable_telemetry: bool = True):
        
        self.max_buffer_size = max_buffer_size
        self.max_handlers = max_handlers
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.enable_telemetry = enable_telemetry
        
        # Основные структуры данных
        self._event_buffer: deque = deque(maxlen=max_buffer_size)
        self._handlers: Dict[str, EventHandler] = {}
        self._topic_handlers: Dict[TopicType, Set[str]] = defaultdict(set)
        
        # Thread safety
        self._buffer_lock = threading.RLock()
        self._handlers_lock = threading.RLock() 
        self._stats_lock = threading.RLock()
        
        # Async processing
        self._processing_queue = asyncio.Queue(maxsize=max_buffer_size)
        self._processing_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Executor для callback'ов
        self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="resonance_bus")
        
        # Метрики и статистика
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_dropped': 0,
            'handlers_count': 0,
            'avg_processing_time_ms': 0.0,
            'peak_buffer_size': 0,
            'errors_count': 0
        }
        
        # Логгер
        self.logger = logging.getLogger(f"{__name__}.ResonanceBus")
        
        # Настройка внутренней телеметрии
        if self.enable_telemetry:
            self._setup_internal_telemetry()
    
    def _setup_internal_telemetry(self):
        """Настроить внутреннюю телеметрию шины"""
        def telemetry_handler(event: BusEvent):
            if event.topic != TopicType.TELEMETRY_EVENT:
                with self._stats_lock:
                    self._stats['events_processed'] += 1
        
        self.subscribe(
            "internal_telemetry",
            telemetry_handler,
            topic_filter={TopicType.TELEMETRY_EVENT}
        )
    
    async def initialize(self):
        """Initialization резонансной шины (алиас для start)"""
        return await self.start()
        
    async def start(self):
        """Start резонансную шину"""
        if self._is_running:
            return
            
        self._is_running = True
        
        # Start задач обработки
        self._processing_task = asyncio.create_task(self._process_events_async())
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_events())
        
        self.logger.info("Резонансная шина started")
        
        if self.enable_telemetry:
            await self._publish_telemetry("bus_started", {"status": "running"})
    
    async def stop(self):
        """Stop резонансную шину"""
        if not self._is_running:
            return
            
        self._is_running = False
        
        # Stop задач
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
                
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Закрытие executor'а
        self._executor.shutdown(wait=True, timeout=5.0)
        
        if self.enable_telemetry:
            await self._publish_telemetry("bus_stopped", {"final_stats": self._stats.copy()})
        
        self.logger.info("Резонансная шина stopped")
    
    def publish(self, 
                topic: TopicType, 
                payload: EventPayload,
                priority: EventPriority = EventPriority.NORMAL,
                ttl_seconds: Optional[float] = None) -> bool:
        """
        Опубликовать событие в шину
        
        Args:
            topic: Топик события
            payload: Полезная нагрузка
            priority: Приоритет события
            ttl_seconds: Time жизни события в секундах
            
        Returns:
            bool: True если событие successfully опубликовано
        """
        try:
            # Check лимитов
            with self._buffer_lock:
                if len(self._event_buffer) >= self.max_buffer_size:
                    with self._stats_lock:
                        self._stats['events_dropped'] += 1
                    
                    self.logger.warning(f"Event buffer full, dropping event on topic {topic.value}")
                    return False
                
                # Creation события
                event = BusEvent(
                    topic=topic,
                    payload=payload, 
                    priority=priority,
                    ttl_seconds=ttl_seconds
                )
                
                # Добавление в буфер с учетом приоритета
                if priority in [EventPriority.CRITICAL, EventPriority.EMERGENCY]:
                    self._event_buffer.appendleft(event)  # Высокий приоритет в начало
                else:
                    self._event_buffer.append(event)  # Обычный приоритет в конец
                
                # Update статистики
                with self._stats_lock:
                    self._stats['events_published'] += 1
                    self._stats['peak_buffer_size'] = max(
                        self._stats['peak_buffer_size'], 
                        len(self._event_buffer)
                    )
            
            # Асинхронная processing если возможно
            if self._is_running:
                try:
                    self._processing_queue.put_nowait(event)
                except asyncio.QueueFull:
                    # Fallback на синхронную обработку
                    self._process_event_sync(event)
            else:
                # Синхронная processing если шина не started
                self._process_event_sync(event)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error publishing event to {topic.value}: {e}")
            with self._stats_lock:
                self._stats['errors_count'] += 1
            return False
    
    def subscribe(self,
                  handler_id: str,
                  callback: Callable[[BusEvent], None],
                  topic_filter: Optional[Set[TopicType]] = None,
                  priority_filter: Optional[Set[EventPriority]] = None) -> bool:
        """
        Подписаться на события
        
        Args:
            handler_id: Уникальный ID обработчика
            callback: Function обработки событий
            topic_filter: Фильтр по топикам (None = все топики)
            priority_filter: Фильтр по приоритетам (None = все приоритеты)
            
        Returns:
            bool: True если подписка успешна
        """
        try:
            with self._handlers_lock:
                if handler_id in self._handlers:
                    self.logger.warning(f"Handler {handler_id} already exists, replacing")
                
                if len(self._handlers) >= self.max_handlers:
                    self.logger.error(f"Maximum handlers limit ({self.max_handlers}) reached")
                    return False
                
                # Creation обработчика
                handler = EventHandler(
                    callback=callback,
                    handler_id=handler_id,
                    topic_filter=topic_filter,
                    priority_filter=priority_filter
                )
                
                self._handlers[handler_id] = handler
                
                # Индексация по топикам для быстрого поиска
                if topic_filter:
                    for topic in topic_filter:
                        self._topic_handlers[topic].add(handler_id)
                else:
                    # Подписка на все топики
                    for topic in TopicType:
                        self._topic_handlers[topic].add(handler_id)
                
                # Update статистики
                with self._stats_lock:
                    self._stats['handlers_count'] = len(self._handlers)
                
                self.logger.debug(f"Handler {handler_id} subscribed to topics: {topic_filter}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error subscribing handler {handler_id}: {e}")
            return False
    
    def unsubscribe(self, handler_id: str) -> bool:
        """
        Отписаться от событий
        
        Args:
            handler_id: ID обработчика для отписки
            
        Returns:
            bool: True если отписка успешна
        """
        try:
            with self._handlers_lock:
                if handler_id not in self._handlers:
                    self.logger.warning(f"Handler {handler_id} not found")
                    return False
                
                handler = self._handlers[handler_id]
                
                # Deletion из индексов топиков
                for topic_handlers in self._topic_handlers.values():
                    topic_handlers.discard(handler_id)
                
                # Deletion обработчика
                del self._handlers[handler_id]
                
                # Update статистики  
                with self._stats_lock:
                    self._stats['handlers_count'] = len(self._handlers)
                
                self.logger.debug(f"Handler {handler_id} unsubscribed")
                return True
                
        except Exception as e:
            self.logger.error(f"Error unsubscribing handler {handler_id}: {e}")
            return False
    
    def _process_event_sync(self, event: BusEvent):
        """Синхронная processing события"""
        start_time = time.time()
        
        try:
            # Check TTL
            if event.is_expired():
                return
            
            # Получение подходящих обработчиков
            relevant_handlers = self._get_relevant_handlers(event.topic)
            
            # Processing события каждым подходящим обработчиком
            processed_count = 0
            for handler_id in relevant_handlers:
                with self._handlers_lock:
                    if handler_id in self._handlers:
                        handler = self._handlers[handler_id]
                        if handler.handle_event(event):
                            processed_count += 1
            
            # Update статистики
            processing_time = (time.time() - start_time) * 1000  # мс
            with self._stats_lock:
                self._stats['avg_processing_time_ms'] = (
                    self._stats['avg_processing_time_ms'] * 0.9 + processing_time * 0.1
                )
            
            if processed_count == 0:
                self.logger.debug(f"No handlers processed event on topic {event.topic.value}")
                
        except Exception as e:
            self.logger.error(f"Error processing event {event.payload.event_id}: {e}")
            with self._stats_lock:
                self._stats['errors_count'] += 1
    
    async def _process_events_async(self):
        """Асинхронная processing событий из очереди"""
        while self._is_running:
            try:
                # Получение события из очереди с timeout
                event = await asyncio.wait_for(
                    self._processing_queue.get(), 
                    timeout=1.0
                )
                
                # Processing в executor'е для не блокирования (исправлено для Python 3.10+)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self._executor, 
                    self._process_event_sync, 
                    event
                )
                
                self._processing_queue.task_done()
                
            except asyncio.TimeoutError:
                continue  # Нормальная ситуация, продолжаем ожидание
                
            except Exception as e:
                self.logger.error(f"Error in async event processing: {e}")
                await asyncio.sleep(0.1)  # Небольшая pause при ошибке
    
    async def _cleanup_expired_events(self):
        """Периодическая очистка устаревших событий"""
        while self._is_running:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                
                with self._buffer_lock:
                    # Deletion устаревших событий
                    original_size = len(self._event_buffer)
                    
                    # Фильтрация активных событий
                    active_events = [
                        event for event in self._event_buffer 
                        if not event.is_expired()
                    ]
                    
                    self._event_buffer.clear()
                    self._event_buffer.extend(active_events)
                    
                    expired_count = original_size - len(active_events)
                    if expired_count > 0:
                        self.logger.debug(f"Cleaned up {expired_count} expired events")
                
                # Очистка неактивных обработчиков
                await self._cleanup_inactive_handlers()
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
    
    async def _cleanup_inactive_handlers(self):
        """Очистка неактивных обработчиков"""
        inactive_timeout = 300.0  # 5 минут
        current_time = time.time()
        
        with self._handlers_lock:
            inactive_handlers = [
                handler_id for handler_id, handler in self._handlers.items()
                if (current_time - handler.last_activity) > inactive_timeout
            ]
            
            for handler_id in inactive_handlers:
                self.logger.info(f"Removing inactive handler: {handler_id}")
                self.unsubscribe(handler_id)
    
    def _get_relevant_handlers(self, topic: TopicType) -> List[str]:
        """Получить список подходящих обработчиков для топика"""
        with self._handlers_lock:
            return list(self._topic_handlers.get(topic, set()))
    
    async def _publish_telemetry(self, event_name: str, data: Dict[str, Any]):
        """Опубликовать внутреннюю телеметрию"""
        if not self.enable_telemetry:
            return
            
        telemetry_payload = TelemetryPayload(
            source_module="resonance_bus",
            metric_name=event_name,
            module_state=data,
            performance_data=self._stats.copy()
        )
        
        # Публикуем без рекурсии
        self.publish(
            TopicType.TELEMETRY_EVENT,
            telemetry_payload,
            priority=EventPriority.LOW
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику работы шины"""
        with self._stats_lock:
            stats = self._stats.copy()
        
        with self._buffer_lock:
            stats['current_buffer_size'] = len(self._event_buffer)
        
        with self._handlers_lock:
            stats['active_handlers'] = len([
                h for h in self._handlers.values() if h.is_active
            ])
        
        stats['is_running'] = self._is_running
        return stats
    
    def get_handler_info(self) -> Dict[str, Dict[str, Any]]:
        """Получить информацию о всех обработчиках"""
        with self._handlers_lock:
            return {
                handler_id: {
                    'events_processed': handler.events_processed,
                    'last_activity': handler.last_activity,
                    'is_active': handler.is_active,
                    'topics': list(handler.topic_filter) if handler.topic_filter else 'ALL',
                    'priorities': list(handler.priority_filter) if handler.priority_filter else 'ALL'
                }
                for handler_id, handler in self._handlers.items()
            }
    
    def __repr__(self) -> str:
        """Строковое представление шины"""
        return (f"ResonanceBus(running={self._is_running}, "
                f"handlers={len(self._handlers)}, "
                f"buffer_size={len(self._event_buffer)})")


# Глобальная инстанция резонансной шины
_global_bus: Optional[ResonanceBus] = None
_bus_lock = threading.Lock()


def get_global_bus() -> ResonanceBus:
    """Получить глобальную инстанцию резонансной шины"""
    global _global_bus
    
    with _bus_lock:
        if _global_bus is None:
            _global_bus = ResonanceBus()
        return _global_bus


async def initialize_global_bus(**kwargs) -> ResonanceBus:
    """Initialize глобальную резонансную шину"""
    global _global_bus
    
    with _bus_lock:
        if _global_bus is not None:
            await _global_bus.stop()
        
        _global_bus = ResonanceBus(**kwargs)
        await _global_bus.start()
        
        return _global_bus


async def shutdown_global_bus():
    """Stop глобальную резонансную шину"""
    global _global_bus
    
    with _bus_lock:
        if _global_bus is not None:
            await _global_bus.stop()
            _global_bus = None


# Удобные функции для быстрого использования
def publish_eta_signal(eta_value: float, 
                      resonance_frequencies: List[float],
                      source_module: str = "esc") -> bool:
    """Быстрая публикация ETA сигнала"""
    payload = EtaSignalPayload(
        source_module=source_module,
        eta_value=eta_value,
        resonance_frequencies=resonance_frequencies
    )
    return get_global_bus().publish(TopicType.SIGNALS_ETA, payload)


def publish_risk_metrics(hallucination_number: float,
                        defect_density_mean: float,
                        risk_level: str = "NORMAL",
                        source_module: str = "risk_monitor") -> bool:
    """Быстрая публикация метрик риска"""
    payload = RiskMetricsPayload(
        source_module=source_module,
        hallucination_number=hallucination_number,
        defect_density_mean=defect_density_mean,
        risk_level=risk_level
    )
    return get_global_bus().publish(TopicType.METRICS_RISK, payload)


def publish_emergency(emergency_type: str,
                     severity_level: int,
                     trigger_reason: str,
                     source_module: str = "emergency_controller") -> bool:
    """Быстрая публикация аварийного события"""
    payload = EmergencyPayload(
        source_module=source_module,
        emergency_type=emergency_type,
        severity_level=severity_level,
        trigger_reason=trigger_reason
    )
    return get_global_bus().publish(
        TopicType.ORCHESTRATION_EMERGENCY, 
        payload, 
        priority=EventPriority.EMERGENCY
    )


if __name__ == "__main__":
    # Пример использования
    import asyncio
    
    async def example_usage():
        # Initialization
        bus = await initialize_global_bus()
        
        # Подписка на события
        def risk_handler(event: BusEvent):
            if isinstance(event.payload, RiskMetricsPayload):
                print(f"Risk alert: Ha={event.payload.hallucination_number:.4f}")
        
        bus.subscribe("risk_handler", risk_handler, {TopicType.METRICS_RISK})
        
        # Публикация событий
        publish_risk_metrics(0.85, 0.12, "CRITICAL")
        
        # Ожидание обработки
        await asyncio.sleep(0.1)
        
        # Статистика
        stats = bus.get_statistics()
        print(f"Bus stats: {stats}")
        
        # Stop
        await shutdown_global_bus()
    
    asyncio.run(example_usage())