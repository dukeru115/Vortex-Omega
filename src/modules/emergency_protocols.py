"""
Emergency Protocols for NFCS - Protection System
===============================================

High-level emergency state management system with detailed
ENTER/EXIT EMERGENCY protocols, smooth transitions and system recovery.

Key Capabilities:
- Detection of emergency states and automatic activation of protective measures
- ENTER EMERGENCY protocols: isolation, connection strengthening, cross-link suppression
- EXIT EMERGENCY protocols: smooth recovery, stability verification
- Specialized measures for each component: Boundary, Kuramoto, ESC, CGL
- Telemetry and logging of all emergency operations
- Integration with resonance bus and constitution
"""

import asyncio
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np

from ..core.state import SystemState, RiskMetrics
from ..modules.risk_monitor import RiskLevel
from ..modules.constitution_v0 import ControlIntent, DecisionType
from ..orchestrator.resonance_bus import (
    get_global_bus,
    TopicType,
    EventPriority,
    BusEvent,
    EmergencyPayload,
    TelemetryPayload,
    publish_emergency,
)


class EmergencyPhase(Enum):
    """Emergency mode phases"""

    NORMAL = "NORMAL"  # Normal operation
    DETECTION = "DETECTION"  # Emergency situation detection
    ENTERING = "ENTERING"  # Entering emergency mode
    ACTIVE = "ACTIVE"  # Active emergency mode
    RECOVERY_ASSESSMENT = "RECOVERY_ASSESSMENT"  # Recovery possibility assessment
    EXITING = "EXITING"  # Exiting emergency mode
    STABILIZATION = "STABILIZATION"  # Stabilization after exit


class EmergencyTrigger(Enum):
    """Emergency situation triggers"""

    HIGH_HALLUCINATION_NUMBER = "HIGH_HALLUCINATION_NUMBER"
    EXCESSIVE_DEFECT_DENSITY = "EXCESSIVE_DEFECT_DENSITY"
    COHERENCE_COLLAPSE = "COHERENCE_COLLAPSE"
    SYSTEM_INSTABILITY = "SYSTEM_INSTABILITY"
    CONSTITUTIONAL_VIOLATION = "CONSTITUTIONAL_VIOLATION"
    MANUAL_TRIGGER = "MANUAL_TRIGGER"
    CASCADING_FAILURE = "CASCADING_FAILURE"
    TIMEOUT_RECOVERY = "TIMEOUT_RECOVERY"


class ComponentProtocol(Enum):
    """Protocols for specific components"""

    BOUNDARY_ISOLATION = "BOUNDARY_ISOLATION"
    KURAMOTO_CLUSTERING = "KURAMOTO_CLUSTERING"
    ESC_NORMALIZATION = "ESC_NORMALIZATION"
    CGL_STABILIZATION = "CGL_STABILIZATION"
    CROSS_TALK_SUPPRESSION = "CROSS_TALK_SUPPRESSION"
    COHERENCE_ENFORCEMENT = "COHERENCE_ENFORCEMENT"


@dataclass
class EmergencyAction:
    """Action in emergency protocol"""

    protocol: ComponentProtocol
    target_component: str
    action_type: str  # activate, deactivate, adjust, monitor
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1-5, where 5 is critical
    estimated_duration: float = 10.0  # Expected duration (sec)
    prerequisites: List[str] = field(default_factory=list)  # Dependencies
    success_criteria: Dict[str, float] = field(default_factory=dict)
    rollback_action: Optional["EmergencyAction"] = None


@dataclass
class ProtocolExecution:
    """Protocol execution information"""

    action: EmergencyAction
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None

    def get_duration(self) -> float:
        """Get execution duration"""
        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    def is_completed(self) -> bool:
        """Check if execution is completed"""
        return self.completed_at is not None

    def is_successful(self) -> bool:
        """Check if execution was successful"""
        return self.success is True


@dataclass
class EmergencyState:
    """Emergency protocols system state"""

    current_phase: EmergencyPhase = EmergencyPhase.NORMAL
    active_triggers: List[EmergencyTrigger] = field(default_factory=list)
    emergency_start_time: Optional[float] = None
    last_phase_change: float = field(default_factory=time.time)

    # Active protocols and their execution
    active_protocols: Dict[ComponentProtocol, ProtocolExecution] = field(default_factory=dict)
    completed_protocols: List[ProtocolExecution] = field(default_factory=list)
    failed_protocols: List[ProtocolExecution] = field(default_factory=list)

    # State metrics
    stabilization_progress: float = 0.0  # Stabilization progress [0-1]
    recovery_readiness: float = 0.0  # Recovery readiness [0-1]
    system_coherence_target: float = 0.7  # Target coherence for recovery

    # Counters and statistics
    total_emergencies: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    avg_recovery_time: float = 0.0

    def get_emergency_duration(self) -> float:
        """Get current emergency mode duration"""
        if self.emergency_start_time is None:
            return 0.0
        return time.time() - self.emergency_start_time

    def get_phase_duration(self) -> float:
        """Get current phase duration"""
        return time.time() - self.last_phase_change

    def is_in_emergency(self) -> bool:
        """Check if system is in emergency mode"""
        return self.current_phase not in [EmergencyPhase.NORMAL, EmergencyPhase.STABILIZATION]


class EmergencyProtocols:
    """
    Emergency protocols system for NFCS

    Manages detection, processing and recovery from emergency states
    with coordination of all system components through detailed protocols.
    """

    def __init__(
        self,
        enable_auto_detection: bool = True,
        enable_auto_recovery: bool = True,
        max_emergency_duration: float = 600.0,
        stabilization_timeout: float = 120.0,
        recovery_assessment_interval: float = 30.0,
    ):

        self.enable_auto_detection = enable_auto_detection
        self.enable_auto_recovery = enable_auto_recovery
        self.max_emergency_duration = max_emergency_duration
        self.stabilization_timeout = stabilization_timeout
        self.recovery_assessment_interval = recovery_assessment_interval

        # System state
        self.state = EmergencyState()

        # Thread safety
        self._lock = threading.RLock()

        # Resonance bus
        self.bus = get_global_bus()

        # Logger
        self.logger = logging.getLogger(f"{__name__}.EmergencyProtocols")

        # Predefined protocols
        self._initialize_emergency_protocols()

        # Event subscriptions
        if self.enable_auto_detection:
            self._subscribe_to_events()

        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

        self.logger.info("Emergency protocols system initialized")

    def _initialize_emergency_protocols(self):
        """Initialize predefined emergency protocols"""

        self.emergency_protocols = {
            # Boundary isolation protocol
            ComponentProtocol.BOUNDARY_ISOLATION: EmergencyAction(
                protocol=ComponentProtocol.BOUNDARY_ISOLATION,
                target_component="boundary",
                action_type="adjust",
                parameters={
                    "permeability_multiplier": 0.1,  # Reduce permeability to 10%
                    "trust_threshold_increase": 0.3,  # Increase trust threshold
                    "novelty_suppression": 0.8,  # Novelty suppression
                    "threat_sensitivity": 2.0,  # Increase threat sensitivity
                },
                priority=5,
                estimated_duration=5.0,
                success_criteria={"permeability_achieved": 0.1, "isolation_effective": True},
            ),
            # Kuramoto clustering protocol
            ComponentProtocol.KURAMOTO_CLUSTERING: EmergencyAction(
                protocol=ComponentProtocol.KURAMOTO_CLUSTERING,
                target_component="kuramoto",
                action_type="adjust",
                parameters={
                    "intra_cluster_boost": 2.0,  # Strengthen intra-cluster connections
                    "inter_cluster_suppression": 0.3,  # Weaken inter-cluster connections
                    "self_coupling_boost": 1.5,  # Strengthen self-coupling
                    "frequency_lock": True,  # Frequency locking
                },
                priority=4,
                estimated_duration=10.0,
                success_criteria={"clustering_coefficient": 0.8, "synchronization_stable": True},
            ),
            # ESC normalization protocol
            ComponentProtocol.ESC_NORMALIZATION: EmergencyAction(
                protocol=ComponentProtocol.ESC_NORMALIZATION,
                target_component="esc",
                action_type="adjust",
                parameters={
                    "normalization_mode": "strict",  # Strict normalization
                    "order_parameter_limit": 0.7,  # Order parameter limitation
                    "resonance_damping": 0.4,  # Resonance damping
                    "semantic_filtering": True,  # Semantic filtering
                    "alpha_lock": True,  # α-locking
                },
                priority=3,
                estimated_duration=8.0,
                success_criteria={"order_param_stable": True, "resonance_controlled": True},
            ),
            # CGL stabilization protocol
            ComponentProtocol.CGL_STABILIZATION: EmergencyAction(
                protocol=ComponentProtocol.CGL_STABILIZATION,
                target_component="cgl",
                action_type="adjust",
                parameters={
                    "energy_penalty_multiplier": 5.0,  # Increase energy penalties
                    "spatial_smoothing": 0.3,  # Spatial smoothing
                    "temporal_damping": 0.2,  # Temporal damping
                    "amplitude_clipping": 0.8,  # Amplitude clipping
                    "diffusion_boost": 1.5,  # Diffusion boost
                },
                priority=4,
                estimated_duration=15.0,
                success_criteria={"field_stable": True, "energy_controlled": True},
            ),
            # Cross-talk suppression protocol
            ComponentProtocol.CROSS_TALK_SUPPRESSION: EmergencyAction(
                protocol=ComponentProtocol.CROSS_TALK_SUPPRESSION,
                target_component="system",
                action_type="adjust",
                parameters={
                    "cross_talk_multiplier": 0.2,  # Suppress to 20%
                    "module_isolation": True,  # Module isolation
                    "communication_filtering": True,  # Communication filtering
                    "signal_attenuation": 0.7,  # Signal attenuation
                },
                priority=3,
                estimated_duration=12.0,
                success_criteria={"cross_talk_reduced": True, "isolation_effective": True},
            ),
            # Forced coherence protocol
            ComponentProtocol.COHERENCE_ENFORCEMENT: EmergencyAction(
                protocol=ComponentProtocol.COHERENCE_ENFORCEMENT,
                target_component="system",
                action_type="activate",
                parameters={
                    "coherence_target": 0.7,  # Целевая coherence
                    "enforcement_strength": 0.8,  # Сила принуждения
                    "gradient_suppression": True,  # Подавление градиентов
                    "phase_locking": True,  # Блокировка фаз
                },
                priority=2,
                estimated_duration=20.0,
                success_criteria={"coherence_achieved": 0.7, "stability_maintained": True},
            ),
        }

    def _subscribe_to_events(self):
        """Подписаться на события для автоматической детекции"""

        try:
            # Подписка на аварийные события
            self.bus.subscribe(
                handler_id="emergency_protocols_handler",
                callback=self._handle_emergency_event,
                topic_filter={TopicType.ORCHESTRATION_EMERGENCY},
                priority_filter={EventPriority.EMERGENCY, EventPriority.CRITICAL},
            )

            # Подписка на события рисков для предиктивной детекции
            self.bus.subscribe(
                handler_id="emergency_risk_handler",
                callback=self._handle_risk_event,
                topic_filter={TopicType.METRICS_RISK},
                priority_filter={EventPriority.CRITICAL, EventPriority.EMERGENCY},
            )

            self.logger.info("Подписка на аварийные события активирована")

        except Exception as e:
            self.logger.error(f"Error подписки на события: {e}")

    def _handle_emergency_event(self, event: BusEvent):
        """Обработчик аварийных событий"""

        try:
            if isinstance(event.payload, EmergencyPayload):
                emergency_type = event.payload.emergency_type
                severity = event.payload.severity_level
                reason = event.payload.trigger_reason

                self.logger.critical(
                    f"Получено аварийное событие: {emergency_type} "
                    f"(severity={severity}) - {reason}"
                )

                # Определение триггера на основе типа события
                trigger = self._map_emergency_type_to_trigger(emergency_type)

                # Start аварийных протоколов если не активны
                if not self.state.is_in_emergency():
                    asyncio.create_task(self.enter_emergency_mode(trigger, reason))

        except Exception as e:
            self.logger.error(f"Error обработки аварийного события: {e}")

    def _handle_risk_event(self, event: BusEvent):
        """Обработчик событий рисков для предиктивной детекции"""

        try:
            from ..orchestrator.resonance_bus import RiskMetricsPayload

            if isinstance(event.payload, RiskMetricsPayload):
                risk_level = RiskLevel(event.payload.risk_level)

                # Предиктивная детекция на основе критических рисков
                if risk_level == RiskLevel.EMERGENCY and not self.state.is_in_emergency():

                    # Определение триггера на основе метрик
                    trigger = self._analyze_risk_trigger(event.payload)
                    reason = f"Predictive detection: {risk_level.value}"

                    asyncio.create_task(self.enter_emergency_mode(trigger, reason))

        except Exception as e:
            self.logger.error(f"Error обработки события риска: {e}")

    def _map_emergency_type_to_trigger(self, emergency_type: str) -> EmergencyTrigger:
        """Сопоставить тип аварийного события с триггером"""

        mapping = {
            "HIGH_HALLUCINATION_NUMBER": EmergencyTrigger.HIGH_HALLUCINATION_NUMBER,
            "EXCESSIVE_DEFECT_DENSITY": EmergencyTrigger.EXCESSIVE_DEFECT_DENSITY,
            "COHERENCE_COLLAPSE": EmergencyTrigger.COHERENCE_COLLAPSE,
            "SYSTEMIC_RISK_CRITICAL": EmergencyTrigger.SYSTEM_INSTABILITY,
            "CONSTITUTIONAL_VIOLATION": EmergencyTrigger.CONSTITUTIONAL_VIOLATION,
            "MANUAL_EMERGENCY": EmergencyTrigger.MANUAL_TRIGGER,
        }

        return mapping.get(emergency_type, EmergencyTrigger.SYSTEM_INSTABILITY)

    def _analyze_risk_trigger(self, risk_payload) -> EmergencyTrigger:
        """Анализировать триггер на основе метрик риска"""

        # Приоритет по серьезности угрозы
        if risk_payload.hallucination_number > 0.9:
            return EmergencyTrigger.HIGH_HALLUCINATION_NUMBER
        elif risk_payload.defect_density_mean > 0.2:
            return EmergencyTrigger.EXCESSIVE_DEFECT_DENSITY
        elif risk_payload.coherence_global < 0.2 or risk_payload.coherence_modular < 0.2:
            return EmergencyTrigger.COHERENCE_COLLAPSE
        else:
            return EmergencyTrigger.SYSTEM_INSTABILITY

    async def start_monitoring(self):
        """Start фоновый monitoring аварийных состояний"""

        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Monitoring аварийных состояний started")

    async def stop_monitoring(self):
        """Stop фоновый monitoring"""

        if not self._running:
            return

        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Monitoring аварийных состояний stopped")

    async def _monitoring_loop(self):
        """Основной цикл мониторинга"""

        while self._running:
            try:
                with self._lock:
                    current_phase = self.state.current_phase
                    phase_duration = self.state.get_phase_duration()

                # Processing в зависимости от текущей фазы
                if current_phase == EmergencyPhase.ACTIVE:
                    await self._monitor_active_emergency()

                elif current_phase == EmergencyPhase.RECOVERY_ASSESSMENT:
                    await self._assess_recovery_readiness()

                elif current_phase == EmergencyPhase.STABILIZATION:
                    await self._monitor_stabilization()

                # Check тайм-аутов
                await self._check_timeouts()

                # Pause
                await asyncio.sleep(5.0)

            except Exception as e:
                self.logger.error(f"Error в цикле мониторинга: {e}")
                await asyncio.sleep(1.0)

    async def _monitor_active_emergency(self):
        """Monitoring активного аварийного режима"""

        with self._lock:
            # Check завершения активных протоколов
            completed_protocols = []
            for protocol, execution in self.state.active_protocols.items():
                if execution.is_completed():
                    completed_protocols.append(protocol)

                    if execution.is_successful():
                        self.state.completed_protocols.append(execution)
                        self.logger.info(f"Protocol {protocol.value} completed successfully")
                    else:
                        self.state.failed_protocols.append(execution)
                        self.logger.error(
                            f"Protocol {protocol.value} completed с ошибкой: {execution.error_message}"
                        )

            # Deletion завершенных протоколов из активных
            for protocol in completed_protocols:
                del self.state.active_protocols[protocol]

            # Если все протоколы завершены, переход к оценке восстановления
            if not self.state.active_protocols and self.enable_auto_recovery:
                await self._transition_to_recovery_assessment()

    async def _assess_recovery_readiness(self):
        """Оценить readiness к восстановлению"""

        # Имитация оценки состояния системы
        # В реальной реализации здесь должны быть проверки метрик

        with self._lock:
            # Простая эвристика на основе времени и завершенных протоколов
            successful_protocols = len(self.state.completed_protocols)
            total_protocols = successful_protocols + len(self.state.failed_protocols)

            if total_protocols > 0:
                success_rate = successful_protocols / total_protocols
                self.state.recovery_readiness = min(success_rate, 1.0)

            # Если readiness высокая, начинаем выход
            if self.state.recovery_readiness >= 0.7:
                await self.exit_emergency_mode("Recovery readiness achieved")

    async def _monitor_stabilization(self):
        """Monitoring стабилизации после выхода из аварийного режима"""

        with self._lock:
            phase_duration = self.state.get_phase_duration()

            # Имитация прогресса стабилизации
            self.state.stabilization_progress = min(
                phase_duration / self.stabilization_timeout, 1.0
            )

            # Если stabilization completed
            if self.state.stabilization_progress >= 1.0:
                await self._transition_to_normal()

    async def _check_timeouts(self):
        """Check различные тайм-ауты"""

        with self._lock:
            emergency_duration = self.state.get_emergency_duration()

            # Тайм-аут аварийного режима
            if self.state.is_in_emergency() and emergency_duration > self.max_emergency_duration:

                self.logger.critical(
                    f"Тайм-аут аварийного режима ({emergency_duration:.1f}с), "
                    "принудительное recovery"
                )

                await self.force_emergency_exit("Emergency timeout")

    async def enter_emergency_mode(
        self,
        trigger: EmergencyTrigger,
        reason: str,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Войти в emergency режим

        Args:
            trigger: Триггер аварийной ситуации
            reason: Reason входа в emergency режим
            additional_context: Дополнительный контекст

        Returns:
            bool: True если вход в emergency режим успешен
        """

        with self._lock:
            try:
                # Check, что не находимся уже в аварийном режиме
                if self.state.is_in_emergency():
                    self.logger.warning(
                        f"Попытка входа в emergency режим, но уже активен "
                        f"режим {self.state.current_phase.value}"
                    )
                    return False

                # Update состояния
                self.state.current_phase = EmergencyPhase.ENTERING
                self.state.emergency_start_time = time.time()
                self.state.last_phase_change = time.time()
                self.state.total_emergencies += 1

                # Добавление триггера
                if trigger not in self.state.active_triggers:
                    self.state.active_triggers.append(trigger)

                # Логгирование
                self.logger.critical(
                    f"🚨 ВХОД В АВАРИЙНЫЙ РЕЖИМ 🚨\n"
                    f"Триггер: {trigger.value}\n"
                    f"Reason: {reason}\n"
                    f"Контекст: {additional_context}"
                )

                # Публикация аварийного события
                await self._publish_emergency_status("EMERGENCY_ENTERING", reason, trigger.value)

                # Определение и start необходимых протоколов
                protocols_to_activate = self._determine_emergency_protocols(trigger)

                # Активация протоколов
                success = await self._activate_emergency_protocols(protocols_to_activate)

                if success:
                    # Переход в active emergency режим
                    self.state.current_phase = EmergencyPhase.ACTIVE
                    self.state.last_phase_change = time.time()

                    self.logger.critical(
                        f"Emergency режим активирован с {len(protocols_to_activate)} протоколами"
                    )

                    return True
                else:
                    # Откат при неудаче
                    await self._rollback_emergency_entry("Protocol activation failed")
                    return False

            except Exception as e:
                self.logger.error(f"Error входа в emergency режим: {e}")
                await self._rollback_emergency_entry(f"Entry error: {str(e)}")
                return False

    def _determine_emergency_protocols(self, trigger: EmergencyTrigger) -> List[ComponentProtocol]:
        """Определить необходимые протоколы для конкретного триггера"""

        protocol_map = {
            EmergencyTrigger.HIGH_HALLUCINATION_NUMBER: [
                ComponentProtocol.CGL_STABILIZATION,
                ComponentProtocol.ESC_NORMALIZATION,
                ComponentProtocol.COHERENCE_ENFORCEMENT,
            ],
            EmergencyTrigger.EXCESSIVE_DEFECT_DENSITY: [
                ComponentProtocol.CGL_STABILIZATION,
                ComponentProtocol.KURAMOTO_CLUSTERING,
                ComponentProtocol.CROSS_TALK_SUPPRESSION,
            ],
            EmergencyTrigger.COHERENCE_COLLAPSE: [
                ComponentProtocol.COHERENCE_ENFORCEMENT,
                ComponentProtocol.KURAMOTO_CLUSTERING,
                ComponentProtocol.ESC_NORMALIZATION,
            ],
            EmergencyTrigger.SYSTEM_INSTABILITY: [
                ComponentProtocol.BOUNDARY_ISOLATION,
                ComponentProtocol.KURAMOTO_CLUSTERING,
                ComponentProtocol.CROSS_TALK_SUPPRESSION,
                ComponentProtocol.CGL_STABILIZATION,
            ],
            EmergencyTrigger.CONSTITUTIONAL_VIOLATION: [
                ComponentProtocol.BOUNDARY_ISOLATION,
                ComponentProtocol.CROSS_TALK_SUPPRESSION,
            ],
            EmergencyTrigger.MANUAL_TRIGGER: [
                ComponentProtocol.BOUNDARY_ISOLATION,
                ComponentProtocol.KURAMOTO_CLUSTERING,
                ComponentProtocol.ESC_NORMALIZATION,
                ComponentProtocol.CGL_STABILIZATION,
            ],
            EmergencyTrigger.CASCADING_FAILURE: [
                ComponentProtocol.BOUNDARY_ISOLATION,
                ComponentProtocol.CROSS_TALK_SUPPRESSION,
                ComponentProtocol.COHERENCE_ENFORCEMENT,
                ComponentProtocol.CGL_STABILIZATION,
            ],
        }

        return protocol_map.get(
            trigger, [ComponentProtocol.BOUNDARY_ISOLATION, ComponentProtocol.CGL_STABILIZATION]
        )

    async def _activate_emergency_protocols(self, protocols: List[ComponentProtocol]) -> bool:
        """Активировать список аварийных протоколов"""

        try:
            # Сортировка протоколов по приоритету
            sorted_protocols = sorted(
                protocols, key=lambda p: self.emergency_protocols[p].priority, reverse=True
            )

            successful_activations = 0

            for protocol in sorted_protocols:
                action = self.emergency_protocols[protocol]

                self.logger.info(f"Активация протокола {protocol.value}...")

                # Creation записи выполнения
                execution = ProtocolExecution(action=action)
                self.state.active_protocols[protocol] = execution

                # Имитация выполнения протокола
                success = await self._execute_protocol_action(action)

                # Update результата
                execution.completed_at = time.time()
                execution.success = success

                if success:
                    successful_activations += 1
                    self.logger.info(
                        f"Protocol {protocol.value} активирован successfully "
                        f"({execution.get_duration():.1f}с)"
                    )
                else:
                    execution.error_message = "Activation failed"
                    self.logger.error(f"Error активации протокола {protocol.value}")

                # Публикация телеметрии
                await self._publish_protocol_telemetry(protocol, execution)

            # Check общего успеха (требуется минимум 50% успешных активаций)
            success_rate = successful_activations / len(protocols) if protocols else 0
            return success_rate >= 0.5

        except Exception as e:
            self.logger.error(f"Error активации протоколов: {e}")
            return False

    async def _execute_protocol_action(self, action: EmergencyAction) -> bool:
        """Execute действие протокола"""

        try:
            # Имитация выполнения протокола с задержкой
            await asyncio.sleep(min(action.estimated_duration * 0.1, 2.0))  # Ускоренная имитация

            # В реальной реализации здесь должно быть:
            # 1. Взаимодействие с конкретными компонентами (CGL, Kuramoto, ESC, Boundary)
            # 2. Применение параметров из action.parameters
            # 3. Check критериев успеха из action.success_criteria

            component = action.target_component
            parameters = action.parameters

            self.logger.debug(f"Execution {action.action_type} для {component}: {parameters}")

            # Имитация успеха (в реальности зависит от состояния системы)
            import random

            success_probability = 0.8  # 80% вероятность успеха
            return random.random() < success_probability

        except Exception as e:
            self.logger.error(f"Error выполнения действия протокола: {e}")
            return False

    async def _rollback_emergency_entry(self, reason: str):
        """Откатить вход в emergency режим"""

        self.logger.error(f"Откат входа в emergency режим: {reason}")

        with self._lock:
            # Сброс состояния
            self.state.current_phase = EmergencyPhase.NORMAL
            self.state.emergency_start_time = None
            self.state.active_triggers.clear()
            self.state.active_protocols.clear()
            self.state.last_phase_change = time.time()

        # Публикация события отката
        await self._publish_emergency_status("EMERGENCY_ROLLBACK", reason, "ENTRY_FAILED")

    async def exit_emergency_mode(self, reason: str) -> bool:
        """
        Выйти из аварийного режима

        Args:
            reason: Reason выхода из аварийного режима

        Returns:
            bool: True если выход успешен
        """

        with self._lock:
            try:
                # Check, что находимся в аварийном режиме
                if not self.state.is_in_emergency():
                    self.logger.warning("Попытка выхода из аварийного режима, но режим не активен")
                    return False

                # Переход в фазу выхода
                self.state.current_phase = EmergencyPhase.EXITING
                self.state.last_phase_change = time.time()

                emergency_duration = self.state.get_emergency_duration()

                self.logger.info(
                    f"🔄 ВЫХОД ИЗ АВАРИЙНОГО РЕЖИМА 🔄\n"
                    f"Reason: {reason}\n"
                    f"Duration: {emergency_duration:.1f}с"
                )

                # Публикация события выхода
                await self._publish_emergency_status(
                    "EMERGENCY_EXITING", reason, "RECOVERY_INITIATED"
                )

                # Деактивация аварийных протоколов
                success = await self._deactivate_emergency_protocols()

                if success:
                    # Переход к стабилизации
                    await self._transition_to_stabilization()

                    # Update статистики
                    self.state.successful_recoveries += 1
                    self._update_recovery_time_stats(emergency_duration)

                    return True
                else:
                    # Возврат к активному аварийному режиму при неудаче деактивации
                    self.state.current_phase = EmergencyPhase.ACTIVE
                    self.state.last_phase_change = time.time()

                    self.logger.error(
                        "Error выхода из аварийного режима, возврат к активному режиму"
                    )
                    return False

            except Exception as e:
                self.logger.error(f"Error выхода из аварийного режима: {e}")
                return False

    async def _deactivate_emergency_protocols(self) -> bool:
        """Деактивировать аварийные протоколы"""

        try:
            successful_deactivations = 0
            total_protocols = len(self.state.active_protocols)

            # Деактивация в обратном порядке приоритета
            protocols_to_deactivate = sorted(
                self.state.active_protocols.keys(),
                key=lambda p: self.emergency_protocols[p].priority,
            )

            for protocol in protocols_to_deactivate:
                execution = self.state.active_protocols[protocol]
                action = execution.action

                self.logger.info(f"Деактивация протокола {protocol.value}...")

                # Имитация деактивации
                success = await self._deactivate_protocol_action(action)

                if success:
                    successful_deactivations += 1
                    self.logger.info(f"Protocol {protocol.value} деактивирован successfully")
                else:
                    self.logger.error(f"Error деактивации протокола {protocol.value}")

                # Публикация телеметрии
                await self._publish_protocol_telemetry(protocol, execution, deactivating=True)

            # Очистка активных протоколов
            self.state.active_protocols.clear()

            # Check общего успеха
            if total_protocols == 0:
                return True

            success_rate = successful_deactivations / total_protocols
            return success_rate >= 0.7  # Требуется 70% успешных деактиваций

        except Exception as e:
            self.logger.error(f"Error деактивации протоколов: {e}")
            return False

    async def _deactivate_protocol_action(self, action: EmergencyAction) -> bool:
        """Деактивировать действие протокола"""

        try:
            # Имитация деактивации
            await asyncio.sleep(0.5)

            # В реальной реализации здесь должен быть откат изменений,
            # произведенных при активации протокола

            component = action.target_component
            self.logger.debug(f"Деактивация протокола для {component}")

            # Имитация успеха деактивации
            import random

            return random.random() < 0.9  # 90% вероятность успешной деактивации

        except Exception as e:
            self.logger.error(f"Error деактивации действия протокола: {e}")
            return False

    async def _transition_to_recovery_assessment(self):
        """Переход к оценке восстановления"""

        with self._lock:
            self.state.current_phase = EmergencyPhase.RECOVERY_ASSESSMENT
            self.state.last_phase_change = time.time()

        self.logger.info("Переход к оценке готовности восстановления")
        await self._publish_emergency_status("RECOVERY_ASSESSMENT", "Assessing system recovery", "")

    async def _transition_to_stabilization(self):
        """Переход к стабилизации"""

        with self._lock:
            self.state.current_phase = EmergencyPhase.STABILIZATION
            self.state.last_phase_change = time.time()
            self.state.stabilization_progress = 0.0

        self.logger.info("Переход к фазе стабилизации")
        await self._publish_emergency_status("STABILIZATION", "System stabilization", "")

    async def _transition_to_normal(self):
        """Переход к нормальному режиму"""

        with self._lock:
            emergency_duration = self.state.get_emergency_duration()

            # Сброс состояния
            self.state.current_phase = EmergencyPhase.NORMAL
            self.state.emergency_start_time = None
            self.state.active_triggers.clear()
            self.state.recovery_readiness = 0.0
            self.state.stabilization_progress = 0.0
            self.state.last_phase_change = time.time()

            # Очистка завершенных и неудачных протоколов
            self.state.completed_protocols.clear()
            self.state.failed_protocols.clear()

        self.logger.info(
            f"✅ ВОССТАНОВЛЕНИЕ ЗАВЕРШЕНО ✅\n"
            f"Полная duration аварийного режима: {emergency_duration:.1f}с\n"
            f"System возвращена к нормальной работе"
        )

        await self._publish_emergency_status(
            "NORMAL", "Recovery completed successfully", "SYSTEM_RESTORED"
        )

    def _update_recovery_time_stats(self, duration: float):
        """Update статистику времени восстановления"""

        # Экспоненциально взвешенное среднее
        if self.state.avg_recovery_time == 0.0:
            self.state.avg_recovery_time = duration
        else:
            alpha = 0.2  # Фактор сглаживания
            self.state.avg_recovery_time = (
                alpha * duration + (1 - alpha) * self.state.avg_recovery_time
            )

    async def _publish_emergency_status(self, status: str, reason: str, context: str):
        """Опубликовать статус аварийного режима"""

        try:
            payload = EmergencyPayload(
                source_module="emergency_protocols",
                emergency_type=status,
                severity_level=3 if status in ["EMERGENCY_ENTERING", "EMERGENCY_ACTIVE"] else 1,
                trigger_reason=reason,
                affected_modules=["system"],
                required_actions=[context] if context else [],
            )

            priority = EventPriority.EMERGENCY if "EMERGENCY" in status else EventPriority.HIGH

            self.bus.publish(TopicType.ORCHESTRATION_EMERGENCY, payload, priority)

        except Exception as e:
            self.logger.error(f"Error публикации статуса аварийного режима: {e}")

    async def _publish_protocol_telemetry(
        self, protocol: ComponentProtocol, execution: ProtocolExecution, deactivating: bool = False
    ):
        """Опубликовать телеметрию протокола"""

        try:
            action_type = "deactivation" if deactivating else "activation"

            payload = TelemetryPayload(
                source_module="emergency_protocols",
                metric_name=f"protocol_{action_type}",
                metric_value=1 if execution.is_successful() else 0,
                module_state={
                    "protocol": protocol.value,
                    "target_component": execution.action.target_component,
                    "duration": execution.get_duration(),
                    "success": execution.is_successful(),
                    "error": execution.error_message,
                },
                performance_data={
                    "execution_time": execution.get_duration(),
                    "estimated_time": execution.action.estimated_duration,
                    "efficiency": execution.action.estimated_duration
                    / max(execution.get_duration(), 0.1),
                },
            )

            self.bus.publish(TopicType.TELEMETRY_EVENT, payload, EventPriority.NORMAL)

        except Exception as e:
            self.logger.error(f"Error публикации телеметрии протокола: {e}")

    async def force_emergency_exit(self, reason: str) -> bool:
        """
        Принудительный выход из аварийного режима

        Args:
            reason: Reason принудительного выхода

        Returns:
            bool: True если выход успешен
        """

        self.logger.critical(f"🔴 ПРИНУДИТЕЛЬНЫЙ ВЫХОД ИЗ АВАРИЙНОГО РЕЖИМА: {reason}")

        with self._lock:
            # Принудительный сброс всех состояний
            emergency_duration = self.state.get_emergency_duration()

            self.state.current_phase = EmergencyPhase.NORMAL
            self.state.emergency_start_time = None
            self.state.active_triggers.clear()
            self.state.active_protocols.clear()
            self.state.completed_protocols.clear()
            self.state.failed_protocols.clear()
            self.state.recovery_readiness = 0.0
            self.state.stabilization_progress = 0.0
            self.state.last_phase_change = time.time()

            # Update статистики как неудачное recovery
            self.state.failed_recoveries += 1

        await self._publish_emergency_status("FORCE_EXIT", reason, "FORCED_RECOVERY")

        self.logger.warning(
            f"Принудительный выход completed. "
            f"Duration аварийного режима: {emergency_duration:.1f}с"
        )

        return True

    def manual_trigger_emergency(
        self, reason: str, additional_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Ручной start аварийного режима

        Args:
            reason: Reason ручного запуска
            additional_context: Дополнительный контекст

        Returns:
            bool: True если start успешен
        """

        self.logger.warning(f"Ручной start аварийного режима: {reason}")

        # Асинхронный start
        asyncio.create_task(
            self.enter_emergency_mode(EmergencyTrigger.MANUAL_TRIGGER, reason, additional_context)
        )

        return True

    def get_current_status(self) -> Dict[str, Any]:
        """Получить текущий статус системы аварийных протоколов"""

        with self._lock:
            status = {
                "current_phase": self.state.current_phase.value,
                "is_in_emergency": self.state.is_in_emergency(),
                "emergency_duration": self.state.get_emergency_duration(),
                "phase_duration": self.state.get_phase_duration(),
                "active_triggers": [t.value for t in self.state.active_triggers],
                "active_protocols": len(self.state.active_protocols),
                "completed_protocols": len(self.state.completed_protocols),
                "failed_protocols": len(self.state.failed_protocols),
                "recovery_readiness": self.state.recovery_readiness,
                "stabilization_progress": self.state.stabilization_progress,
                "statistics": {
                    "total_emergencies": self.state.total_emergencies,
                    "successful_recoveries": self.state.successful_recoveries,
                    "failed_recoveries": self.state.failed_recoveries,
                    "avg_recovery_time": self.state.avg_recovery_time,
                    "success_rate": (
                        self.state.successful_recoveries
                        / max(1, self.state.successful_recoveries + self.state.failed_recoveries)
                    ),
                },
            }

            # Детали активных протоколов
            if self.state.active_protocols:
                status["active_protocol_details"] = {
                    protocol.value: {
                        "target": execution.action.target_component,
                        "duration": execution.get_duration(),
                        "completed": execution.is_completed(),
                        "success": execution.is_successful(),
                    }
                    for protocol, execution in self.state.active_protocols.items()
                }

            return status

    def get_protocol_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Получить историю выполнения протоколов"""

        with self._lock:
            all_executions = (
                self.state.completed_protocols
                + self.state.failed_protocols
                + list(self.state.active_protocols.values())
            )

            # Сортировка по времени начала
            sorted_executions = sorted(all_executions, key=lambda x: x.started_at, reverse=True)

            return [
                {
                    "protocol": exec.action.protocol.value,
                    "target_component": exec.action.target_component,
                    "action_type": exec.action.action_type,
                    "started_at": exec.started_at,
                    "duration": exec.get_duration(),
                    "completed": exec.is_completed(),
                    "success": exec.is_successful(),
                    "error": exec.error_message,
                }
                for exec in sorted_executions[:limit]
            ]

    def __repr__(self) -> str:
        """Строковое представление системы аварийных протоколов"""
        return (
            f"EmergencyProtocols(phase={self.state.current_phase.value}, "
            f"active_protocols={len(self.state.active_protocols)}, "
            f"emergencies={self.state.total_emergencies})"
        )


# Удобные функции для создания системы аварийных протоколов
def create_default_emergency_protocols(**kwargs) -> EmergencyProtocols:
    """Create систему аварийных протоколов с настройками по умолчанию"""
    return EmergencyProtocols(**kwargs)


def create_strict_emergency_protocols() -> EmergencyProtocols:
    """Create строгую систему аварийных протоколов с быстрым реагированием"""
    return EmergencyProtocols(
        enable_auto_detection=True,
        enable_auto_recovery=True,
        max_emergency_duration=300.0,  # 5 минут максимум
        stabilization_timeout=60.0,  # 1 minute стабилизации
        recovery_assessment_interval=15.0,  # Оценка каждые 15 секунд
    )


def create_permissive_emergency_protocols() -> EmergencyProtocols:
    """Create мягкую систему аварийных протоколов с длительными интервалами"""
    return EmergencyProtocols(
        enable_auto_detection=True,
        enable_auto_recovery=True,
        max_emergency_duration=1200.0,  # 20 минут максимум
        stabilization_timeout=300.0,  # 5 минут стабилизации
        recovery_assessment_interval=60.0,  # Оценка каждую минуту
    )


if __name__ == "__main__":
    # Пример использования
    import asyncio
    from ..orchestrator.resonance_bus import initialize_global_bus

    async def demo_emergency_protocols():
        # Initialization шины
        await initialize_global_bus()

        # Creation системы аварийных протоколов
        emergency_system = create_default_emergency_protocols()

        # Start мониторинга
        await emergency_system.start_monitoring()

        # Testing ручного запуска аварийного режима
        success = emergency_system.manual_trigger_emergency("Test emergency", {"test_mode": True})

        if success:
            print("Emergency режим started")

            # Ожидание обработки
            for i in range(10):
                status = emergency_system.get_current_status()
                print(
                    f"Статус: {status['current_phase']} "
                    f"(duration: {status['emergency_duration']:.1f}с)"
                )
                await asyncio.sleep(2.0)

            # Принудительный выход для тестирования
            await emergency_system.force_emergency_exit("Test completed")

        # Финальный статус
        final_status = emergency_system.get_current_status()
        print(f"\nФинальный статус: {final_status['current_phase']}")
        print(f"Статистика: {final_status['statistics']}")

        # Stop мониторинга
        await emergency_system.stop_monitoring()

    # Start демо
    if __name__ == "__main__":
        asyncio.run(demo_emergency_protocols())
