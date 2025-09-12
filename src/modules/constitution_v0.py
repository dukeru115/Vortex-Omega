"""
Constitution v0 for NFCS - Core Decision Making
==============================================

High-level decision-making system based on risk analysis and generation
of control intentions for all NFCS components.

Key Capabilities:
- Analyze RiskMetrics and generate ACCEPT/REJECT/EMERGENCY decisions
- Generate control_intent for CGL, Kuramoto, ESC 
- Safety policies and constraints
- Integration with resonance bus for event reception and decision publication
- Adaptive control strategies based on system state
- Constitutional principles and constraints for autonomous modules
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np

from ..modules.risk_monitor import RiskLevel, RiskAssessment
from ..orchestrator.resonance_bus import (
    get_global_bus, TopicType, EventPriority, BusEvent,
    RiskMetricsPayload, ControlIntentPayload
)


class DecisionType(Enum):
    """Types of constitutional decisions"""
    ACCEPT = "ACCEPT"        # Принять операцию, стандартное управление
    REJECT = "REJECT"        # Отклонить операцию, усилить мониторинг  
    EMERGENCY = "EMERGENCY"  # Аварийный режим, активировать протоколы защиты
    MONITOR = "MONITOR"      # Усиленное наблюдение без вмешательства


class ControlStrategy(Enum):
    """System control strategies"""
    PERMISSIVE = "PERMISSIVE"      # Мягкое управление, максимальная свобода
    STANDARD = "STANDARD"          # Стандартное управление
    RESTRICTIVE = "RESTRICTIVE"    # Жесткое управление, ограничения
    EMERGENCY = "EMERGENCY"        # Аварийное управление, максимальные ограничения


@dataclass
class PolicyConstraints:
    """Constitutional constraints and policies"""
    
    # Ограничения на управляющие сигналы
    u_field_max_amplitude: float = 1.0      # Максимальная амплитуда u_field
    u_modules_max_amplitude: float = 0.5    # Максимальная амплитуда u_modules
    
    # Ограничения на связность Kuramoto
    kuramoto_coupling_max: float = 2.0      # Максимальная связность
    kuramoto_coupling_min: float = 0.1      # Минимальная связность  
    
    # Ограничения на ESC
    esc_normalization_strict: bool = True   # Строгая нормализация ESC
    esc_max_order_parameter: float = 0.9    # Максимальный order parameter
    
    # Ограничения на Freedom модуль
    freedom_max_noise_amplitude: float = 0.2  # Максимальная амплитуда шума свободы
    freedom_min_coherence_threshold: float = 0.3  # Минимальная когерентность для свободы
    
    # Временные ограничения
    emergency_mode_max_duration: float = 300.0    # Максимальная длительность emergency (сек)
    recovery_assessment_interval: float = 30.0    # Интервал оценки восстановления
    
    # Пороги для автономии
    autonomous_operation_min_coherence: float = 0.5
    autonomous_operation_max_risk: float = 0.3


@dataclass
class ControlIntent:
    """Control intention for system components"""
    
    # Основное решение
    decision: DecisionType = DecisionType.ACCEPT
    strategy: ControlStrategy = ControlStrategy.STANDARD
    
    # Ограничения для CGL solver
    u_field_limits: Dict[str, float] = field(default_factory=lambda: {
        'max_amplitude': 1.0,
        'spatial_smoothing': 0.0,
        'temporal_damping': 0.0
    })
    
    # Маски и ограничения для Kuramoto
    kuramoto_masks: Dict[str, Any] = field(default_factory=lambda: {
        'coupling_multipliers': None,   # Множители связности [N x N]
        'frequency_adjustments': None,  # Коррекция частот [N]
        'connection_masks': None        # Маски подключений [N x N, bool]
    })
    
    # Настройки ESC 
    esc_configuration: Dict[str, Any] = field(default_factory=lambda: {
        'normalization_mode': 'standard',  # standard, strict, adaptive
        'order_parameter_limit': 0.9,
        'resonance_damping': 0.0,
        'semantic_filtering': False
    })
    
    # Окно свободы для Freedom модуля  
    freedom_window: Dict[str, float] = field(default_factory=lambda: {
        'noise_amplitude': 0.1,
        'coherence_threshold': 0.3,
        'exploration_rate': 0.05,
        'creativity_boost': 0.0
    })
    
    # Аварийные ограничения
    emergency_constraints: Dict[str, Any] = field(default_factory=lambda: {
        'boundary_permeability': 1.0,      # Множитель проницаемости границы
        'cross_talk_suppression': 0.0,     # Подавление кросс-связей [0-1]
        'coherence_enforcement': False,     # Принудительная когерентность
        'risk_escalation_rate': 1.0        # Скорость эскалации рисков
    })
    
    # Метаданные решения
    reasoning: List[str] = field(default_factory=list)    # Обоснование решения
    confidence: float = 1.0                               # Уверенность в решении [0-1]
    validity_duration: float = 60.0                       # Срок действия решения (сек)
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if decision has expired"""
        return time.time() - self.created_at > self.validity_duration
    
    def get_summary(self) -> str:
        """Get brief description of intention"""
        return (f"{self.decision.value}({self.strategy.value}) "
               f"confidence={self.confidence:.2f}")


@dataclass  
class ConstitutionalState:
    """State of constitutional system"""
    current_risk_level: RiskLevel = RiskLevel.NORMAL
    current_strategy: ControlStrategy = ControlStrategy.STANDARD
    emergency_mode_start: Optional[float] = None
    last_decision_time: float = field(default_factory=time.time)
    
    # Счетчики и статистика
    total_decisions: int = 0
    accept_decisions: int = 0  
    reject_decisions: int = 0
    emergency_decisions: int = 0
    
    # Адаптивные параметры
    risk_sensitivity: float = 1.0      # Множитель чувствительности к рискам
    recovery_progress: float = 0.0     # Прогресс восстановления [0-1]
    
    def get_emergency_duration(self) -> float:
        """Get duration in emergency mode"""
        if self.emergency_mode_start is None:
            return 0.0
        return time.time() - self.emergency_mode_start


class ConstitutionV0:
    """
    Конституция v0 - Система принятия решений для NFCS
    
    Анализирует состояние рисков и вырабатывает управляющие намерения
    для всех компонентов системы согласно конституционным принципам.
    """
    
    def __init__(self, 
                 constraints: Optional[PolicyConstraints] = None,
                 enable_auto_subscription: bool = True,
                 decision_interval: float = 1.0):
        
        self.constraints = constraints or PolicyConstraints()
        self.enable_auto_subscription = enable_auto_subscription
        self.decision_interval = decision_interval
        
        # Состояние конституции
        self.state = ConstitutionalState()
        
        # История решений для анализа
        self.decision_history: List[ControlIntent] = []
        self.max_history_size = 1000
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Резонансная шина
        self.bus = get_global_bus()
        
        # Логгер
        self.logger = logging.getLogger(f"{__name__}.ConstitutionV0")
        
        # Подписка на события рисков
        if self.enable_auto_subscription:
            self._subscribe_to_risk_events()
        
        # Периодическое принятие решений
        self._decision_task: Optional[asyncio.Task] = None
        self._running = False
        
        self.logger.info("Конституция v0 инициализирована")
    
    def _subscribe_to_risk_events(self):
        """Subscribe to risk events from ResonanceBus"""
        try:
            self.bus.subscribe(
                handler_id="constitution_risk_handler",
                callback=self._handle_risk_event,
                topic_filter={TopicType.METRICS_RISK},
                priority_filter={EventPriority.HIGH, EventPriority.CRITICAL, EventPriority.EMERGENCY}
            )
            
            self.logger.info("Подписка на события рисков активирована")
            
        except Exception as e:
            self.logger.error(f"Ошибка подписки на события рисков: {e}")
    
    def _handle_risk_event(self, event: BusEvent):
        """Risk event handler"""
        try:
            if isinstance(event.payload, RiskMetricsPayload):
                # Быстрое принятие решения на критические события  
                if event.priority in [EventPriority.CRITICAL, EventPriority.EMERGENCY]:
                    risk_level = RiskLevel(event.payload.risk_level)
                    intent = self._make_immediate_decision(event.payload, risk_level)
                    
                    if intent:
                        self._publish_control_intent(intent)
                        self.logger.warning(
                            f"Экстренное решение: {intent.get_summary()} "
                            f"на {event.payload.risk_level}"
                        )
        
        except Exception as e:
            self.logger.error(f"Ошибка обработки события риска: {e}")
    
    def _make_immediate_decision(self, 
                               risk_payload: RiskMetricsPayload, 
                               risk_level: RiskLevel) -> Optional[ControlIntent]:
        """Make immediate decision on critical event"""
        
        with self._lock:
            # Обновление состояния риска
            self.state.current_risk_level = risk_level
            
            # Быстрая оценка ситуации
            if risk_level == RiskLevel.EMERGENCY:
                return self._create_emergency_intent(risk_payload, "Immediate emergency response")
            
            elif risk_level == RiskLevel.CRITICAL:
                return self._create_restrictive_intent(risk_payload, "Critical risk mitigation")
            
            else:
                return None  # Для WARNING и NORMAL нет экстренных решений
    
    async def start_decision_loop(self):
        """Start periodic decision-making cycle"""
        if self._running:
            return
        
        self._running = True
        self._decision_task = asyncio.create_task(self._decision_loop())
        self.logger.info("Цикл принятия решений запущен")
    
    async def stop_decision_loop(self):
        """Stop decision-making cycle"""
        if not self._running:
            return
        
        self._running = False
        if self._decision_task:
            self._decision_task.cancel()
            try:
                await self._decision_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Цикл принятия решений остановлен")
    
    async def _decision_loop(self):
        """Main periodic decision-making loop"""
        while self._running:
            try:
                # Комплексная оценка состояния системы
                intent = self._make_comprehensive_decision()
                
                if intent:
                    self._publish_control_intent(intent)
                    
                    # Логгирование изменений стратегии
                    if intent.strategy != self.state.current_strategy:
                        self.logger.info(
                            f"Стратегия изменена: {self.state.current_strategy.value} → "
                            f"{intent.strategy.value}. Причина: {', '.join(intent.reasoning)}"
                        )
                        self.state.current_strategy = intent.strategy
                
                await asyncio.sleep(self.decision_interval)
                
            except Exception as e:
                self.logger.error(f"Ошибка в цикле принятия решений: {e}")
                await asyncio.sleep(1.0)  # Пауза при ошибке
    
    def _make_comprehensive_decision(self) -> Optional[ControlIntent]:
        """Make comprehensive decision based on system analysis"""
        
        with self._lock:
            try:
                # Анализ текущего состояния
                system_analysis = self._analyze_system_state()
                
                # Определение стратегии
                strategy = self._determine_strategy(system_analysis)
                
                # Создание намерения
                if strategy == ControlStrategy.EMERGENCY:
                    intent = self._create_emergency_intent(None, "Comprehensive emergency assessment")
                elif strategy == ControlStrategy.RESTRICTIVE:
                    intent = self._create_restrictive_intent(None, "Risk mitigation required")
                elif strategy == ControlStrategy.PERMISSIVE:
                    intent = self._create_permissive_intent("Low risk, enhanced freedom")
                else:
                    intent = self._create_standard_intent("Normal operation")
                
                # Обновление статистики
                self._update_decision_statistics(intent)
                
                return intent
                
            except Exception as e:
                self.logger.error(f"Ошибка комплексного принятия решения: {e}")
                # Безопасное возврат - ограничительная стратегия
                return self._create_restrictive_intent(None, f"Decision error: {str(e)}")
    
    def _analyze_system_state(self) -> Dict[str, Any]:
        """Analysis of current system state"""
        
        analysis = {
            'current_risk_level': self.state.current_risk_level,
            'emergency_duration': self.state.get_emergency_duration(),
            'recent_decisions': len([d for d in self.decision_history[-10:] 
                                   if d.decision == DecisionType.EMERGENCY]),
            'decision_frequency': self._calculate_decision_frequency(),
            'system_stability': self._assess_system_stability(),
            'recovery_indicators': self._assess_recovery_progress()
        }
        
        return analysis
    
    def _determine_strategy(self, analysis: Dict[str, Any]) -> ControlStrategy:
        """Determine optimal control strategy"""
        
        # Аварийная стратегия
        if analysis['current_risk_level'] == RiskLevel.EMERGENCY:
            return ControlStrategy.EMERGENCY
        
        # Проверка превышения времени аварийного режима
        if (self.state.emergency_mode_start and 
            analysis['emergency_duration'] > self.constraints.emergency_mode_max_duration):
            self.logger.warning(f"Emergency mode timeout, forcing recovery")
            return ControlStrategy.RESTRICTIVE
        
        # Ограничительная стратегия
        if (analysis['current_risk_level'] == RiskLevel.CRITICAL or
            analysis['recent_decisions'] >= 3 or  # Много экстренных решений
            analysis['system_stability'] < 0.3):
            return ControlStrategy.RESTRICTIVE
        
        # Разрешительная стратегия  
        if (analysis['current_risk_level'] == RiskLevel.NORMAL and
            analysis['system_stability'] > 0.7 and
            analysis['recovery_indicators'] > 0.8):
            return ControlStrategy.PERMISSIVE
        
        # Стандартная стратегия по умолчанию
        return ControlStrategy.STANDARD
    
    def _calculate_decision_frequency(self) -> float:
        """Calculate decision frequency (decisions per minute)"""
        if len(self.decision_history) < 2:
            return 0.0
        
        recent_decisions = [d for d in self.decision_history[-20:] 
                          if time.time() - d.created_at < 60.0]  # За последнюю минуту
        
        return len(recent_decisions)
    
    def _assess_system_stability(self) -> float:
        """Evaluate system stability [0-1]"""
        if len(self.decision_history) < 5:
            return 0.5  # Недостаточно данных
        
        recent_decisions = self.decision_history[-10:]
        
        # Подсчет изменений стратегии
        strategy_changes = 0
        for i in range(1, len(recent_decisions)):
            if recent_decisions[i].strategy != recent_decisions[i-1].strategy:
                strategy_changes += 1
        
        # Подсчет аварийных решений  
        emergency_count = sum(1 for d in recent_decisions 
                            if d.decision == DecisionType.EMERGENCY)
        
        # Оценка стабильности
        stability = 1.0 - (strategy_changes / len(recent_decisions)) - (emergency_count * 0.2)
        
        return max(0.0, min(1.0, stability))
    
    def _assess_recovery_progress(self) -> float:
        """Evaluate system recovery progress [0-1]"""
        
        # Если не было аварийного режима
        if self.state.emergency_mode_start is None:
            return 1.0
        
        emergency_duration = self.state.get_emergency_duration()
        
        # Прогресс на основе времени восстановления
        if emergency_duration < 60.0:  # Менее минуты
            time_progress = 0.2
        elif emergency_duration < 300.0:  # Менее 5 минут
            time_progress = 0.5
        else:  # Долгое восстановление
            time_progress = 0.8
        
        # Прогресс на основе снижения рисков
        if self.state.current_risk_level == RiskLevel.NORMAL:
            risk_progress = 1.0
        elif self.state.current_risk_level == RiskLevel.WARNING:
            risk_progress = 0.7
        elif self.state.current_risk_level == RiskLevel.CRITICAL:
            risk_progress = 0.3
        else:
            risk_progress = 0.0
        
        return (time_progress + risk_progress) / 2.0
    
    def _create_emergency_intent(self, 
                               risk_payload: Optional[RiskMetricsPayload],
                               reason: str) -> ControlIntent:
        """Create emergency intention"""
        
        intent = ControlIntent(
            decision=DecisionType.EMERGENCY,
            strategy=ControlStrategy.EMERGENCY,
            confidence=0.95
        )
        
        # Жесткие ограничения на управление
        intent.u_field_limits = {
            'max_amplitude': self.constraints.u_field_max_amplitude * 0.5,
            'spatial_smoothing': 0.3,
            'temporal_damping': 0.2
        }
        
        # Ограничения Kuramoto - усиление внутри-кластерных связей
        intent.kuramoto_masks = {
            'coupling_multipliers': self._create_emergency_coupling_matrix(),
            'frequency_adjustments': None,
            'connection_masks': None
        }
        
        # Строгие настройки ESC
        intent.esc_configuration = {
            'normalization_mode': 'strict',
            'order_parameter_limit': 0.7,
            'resonance_damping': 0.4,
            'semantic_filtering': True
        }
        
        # Минимальная свобода
        intent.freedom_window = {
            'noise_amplitude': 0.02,
            'coherence_threshold': 0.6,
            'exploration_rate': 0.01,
            'creativity_boost': -0.1
        }
        
        # Аварийные ограничения
        intent.emergency_constraints = {
            'boundary_permeability': 0.1,    # Сильное ограничение проницаемости
            'cross_talk_suppression': 0.8,   # Подавление кросс-связей
            'coherence_enforcement': True,   # Принудительная когерентность
            'risk_escalation_rate': 0.5     # Замедленная эскалация
        }
        
        intent.reasoning = [reason, "Emergency protocols activated"]
        intent.validity_duration = 30.0  # Короткий срок действия
        
        # Обновление состояния
        if self.state.emergency_mode_start is None:
            self.state.emergency_mode_start = time.time()
        
        return intent
    
    def _create_restrictive_intent(self,
                                 risk_payload: Optional[RiskMetricsPayload],
                                 reason: str) -> ControlIntent:
        """Create restrictive intention"""
        
        intent = ControlIntent(
            decision=DecisionType.REJECT,
            strategy=ControlStrategy.RESTRICTIVE,
            confidence=0.8
        )
        
        # Умеренные ограничения на управление
        intent.u_field_limits = {
            'max_amplitude': self.constraints.u_field_max_amplitude * 0.7,
            'spatial_smoothing': 0.1,
            'temporal_damping': 0.1
        }
        
        # Ограничения Kuramoto
        intent.kuramoto_masks = {
            'coupling_multipliers': self._create_restrictive_coupling_matrix(),
            'frequency_adjustments': None,
            'connection_masks': None
        }
        
        # Усиленные настройки ESC
        intent.esc_configuration = {
            'normalization_mode': 'adaptive',
            'order_parameter_limit': 0.8,
            'resonance_damping': 0.2,
            'semantic_filtering': True
        }
        
        # Ограниченная свобода
        intent.freedom_window = {
            'noise_amplitude': 0.05,
            'coherence_threshold': 0.4,
            'exploration_rate': 0.02,
            'creativity_boost': 0.0
        }
        
        # Умеренные ограничения
        intent.emergency_constraints = {
            'boundary_permeability': 0.5,
            'cross_talk_suppression': 0.4,
            'coherence_enforcement': False,
            'risk_escalation_rate': 0.8
        }
        
        intent.reasoning = [reason, "Risk mitigation measures"]
        
        return intent
    
    def _create_standard_intent(self, reason: str) -> ControlIntent:
        """Create standard intention"""
        
        intent = ControlIntent(
            decision=DecisionType.ACCEPT,
            strategy=ControlStrategy.STANDARD,
            confidence=0.9
        )
        
        # Стандартные ограничения
        intent.u_field_limits = {
            'max_amplitude': self.constraints.u_field_max_amplitude,
            'spatial_smoothing': 0.0,
            'temporal_damping': 0.0
        }
        
        # Стандартная конфигурация ESC
        intent.esc_configuration = {
            'normalization_mode': 'standard',
            'order_parameter_limit': self.constraints.esc_max_order_parameter,
            'resonance_damping': 0.0,
            'semantic_filtering': False
        }
        
        # Стандартная свобода
        intent.freedom_window = {
            'noise_amplitude': self.constraints.freedom_max_noise_amplitude,
            'coherence_threshold': self.constraints.freedom_min_coherence_threshold,
            'exploration_rate': 0.05,
            'creativity_boost': 0.0
        }
        
        intent.reasoning = [reason, "Standard operation"]
        
        # Сброс аварийного режима если был
        if self.state.emergency_mode_start is not None:
            self.logger.info(f"Выход из аварийного режима после {self.state.get_emergency_duration():.1f}с")
            self.state.emergency_mode_start = None
        
        return intent
    
    def _create_permissive_intent(self, reason: str) -> ControlIntent:
        """Create permissive intention"""
        
        intent = ControlIntent(
            decision=DecisionType.ACCEPT,
            strategy=ControlStrategy.PERMISSIVE,
            confidence=0.85
        )
        
        # Увеличенные лимиты
        intent.u_field_limits = {
            'max_amplitude': self.constraints.u_field_max_amplitude * 1.2,
            'spatial_smoothing': 0.0,
            'temporal_damping': 0.0
        }
        
        # Расширенная конфигурация ESC
        intent.esc_configuration = {
            'normalization_mode': 'adaptive',
            'order_parameter_limit': 0.95,
            'resonance_damping': 0.0,
            'semantic_filtering': False
        }
        
        # Расширенная свобода
        intent.freedom_window = {
            'noise_amplitude': self.constraints.freedom_max_noise_amplitude * 1.5,
            'coherence_threshold': 0.2,
            'exploration_rate': 0.1,
            'creativity_boost': 0.1
        }
        
        intent.reasoning = [reason, "Enhanced autonomy granted"]
        
        return intent
    
    def _create_emergency_coupling_matrix(self) -> Optional[np.ndarray]:
        """Create connectivity matrix for emergency mode"""
        # Пример: усиление внутри-кластерных связей, ослабление между кластерами
        # В реальной реализации это должно зависеть от текущей конфигурации модулей
        n_modules = 4  # constitution, boundary, memory, meta_reflection
        
        matrix = np.eye(n_modules) * 2.0  # Усиление самосвязи
        
        # Усиление связей внутри когнитивного кластера
        cognitive_cluster = [0, 1, 2, 3]  # Все модули в одном кластере пока
        for i in cognitive_cluster:
            for j in cognitive_cluster:
                if i != j:
                    matrix[i, j] = 1.5  # Усиленные связи внутри кластера
        
        return matrix
    
    def _create_restrictive_coupling_matrix(self) -> Optional[np.ndarray]:
        """Create connectivity matrix for restrictive mode"""
        n_modules = 4
        
        matrix = np.eye(n_modules) * 1.2  # Небольшое усиление самосвязи
        
        # Стандартные связи с небольшим ослаблением
        for i in range(n_modules):
            for j in range(n_modules):
                if i != j:
                    matrix[i, j] = 0.8
        
        return matrix
    
    def _update_decision_statistics(self, intent: ControlIntent):
        """Update decision statistics"""
        
        self.state.total_decisions += 1
        self.state.last_decision_time = time.time()
        
        if intent.decision == DecisionType.ACCEPT:
            self.state.accept_decisions += 1
        elif intent.decision == DecisionType.REJECT:
            self.state.reject_decisions += 1
        elif intent.decision == DecisionType.EMERGENCY:
            self.state.emergency_decisions += 1
        
        # Добавление в историю
        self.decision_history.append(intent)
        
        # Ограничение размера истории
        if len(self.decision_history) > self.max_history_size:
            self.decision_history = self.decision_history[-self.max_history_size//2:]
    
    def _publish_control_intent(self, intent: ControlIntent):
        """Publish intention to resonance bus"""
        
        try:
            payload = ControlIntentPayload(
                source_module="constitution_v0",
                decision_type=intent.decision.value,
                u_field_limits=intent.u_field_limits,
                kuramoto_masks=intent.kuramoto_masks,
                esc_normalization_mode=intent.esc_configuration['normalization_mode'],
                freedom_window=intent.freedom_window['noise_amplitude'],
                emergency_constraints=intent.emergency_constraints
            )
            
            # Определение приоритета
            priority = EventPriority.NORMAL
            if intent.decision == DecisionType.EMERGENCY:
                priority = EventPriority.EMERGENCY
            elif intent.decision == DecisionType.REJECT:
                priority = EventPriority.HIGH
            
            success = self.bus.publish(TopicType.CONTROL_INTENT, payload, priority)
            
            if success:
                self.logger.debug(f"Опубликовано намерение: {intent.get_summary()}")
            else:
                self.logger.error("Ошибка публикации намерения")
        
        except Exception as e:
            self.logger.error(f"Ошибка публикации намерения: {e}")
    
    def manual_decision(self, 
                       risk_level: RiskLevel,
                       additional_context: Optional[Dict[str, Any]] = None) -> ControlIntent:
        """
        Принять решение вручную на основе указанного уровня риска
        
        Args:
            risk_level: Уровень риска для принятия решения
            additional_context: Дополнительный контекст для решения
            
        Returns:
            ControlIntent: Управляющее намерение
        """
        
        with self._lock:
            self.state.current_risk_level = risk_level
            
            context_info = additional_context or {}
            reason = f"Manual decision for {risk_level.value}"
            
            if risk_level == RiskLevel.EMERGENCY:
                intent = self._create_emergency_intent(None, reason)
            elif risk_level == RiskLevel.CRITICAL:
                intent = self._create_restrictive_intent(None, reason)
            elif risk_level == RiskLevel.WARNING:
                intent = self._create_standard_intent(reason)
            else:
                intent = self._create_permissive_intent(reason)
            
            # Добавление контекста в обоснование
            if additional_context:
                intent.reasoning.extend([f"{k}={v}" for k, v in context_info.items()])
            
            self._update_decision_statistics(intent)
            self._publish_control_intent(intent)
            
            self.logger.info(f"Ручное решение: {intent.get_summary()}")
            
            return intent
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current constitution status"""
        
        with self._lock:
            status = {
                'current_risk_level': self.state.current_risk_level.value,
                'current_strategy': self.state.current_strategy.value,
                'emergency_mode': self.state.emergency_mode_start is not None,
                'emergency_duration': self.state.get_emergency_duration(),
                'statistics': {
                    'total_decisions': self.state.total_decisions,
                    'accept_rate': (self.state.accept_decisions / max(1, self.state.total_decisions)),
                    'reject_rate': (self.state.reject_decisions / max(1, self.state.total_decisions)), 
                    'emergency_rate': (self.state.emergency_decisions / max(1, self.state.total_decisions)),
                    'decision_frequency': self._calculate_decision_frequency(),
                    'system_stability': self._assess_system_stability(),
                    'recovery_progress': self._assess_recovery_progress()
                },
                'constraints': {
                    'u_field_max': self.constraints.u_field_max_amplitude,
                    'emergency_max_duration': self.constraints.emergency_mode_max_duration,
                    'autonomous_min_coherence': self.constraints.autonomous_operation_min_coherence
                }
            }
            
            if self.decision_history:
                last_decision = self.decision_history[-1]
                status['last_decision'] = {
                    'type': last_decision.decision.value,
                    'strategy': last_decision.strategy.value,
                    'confidence': last_decision.confidence,
                    'reasoning': last_decision.reasoning,
                    'age_seconds': time.time() - last_decision.created_at
                }
            
            return status
    
    def update_constraints(self, new_constraints: PolicyConstraints):
        """Update constitutional constraints"""
        
        with self._lock:
            old_constraints = self.constraints
            self.constraints = new_constraints
            
            self.logger.info("Конституционные ограничения обновлены")
    
    def get_decision_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get decision history"""
        
        with self._lock:
            recent_decisions = self.decision_history[-limit:]
            
            return [
                {
                    'decision': d.decision.value,
                    'strategy': d.strategy.value,
                    'confidence': d.confidence,
                    'reasoning': d.reasoning,
                    'created_at': d.created_at,
                    'expired': d.is_expired()
                }
                for d in recent_decisions
            ]
    
    def force_recovery_mode(self):
        """Force recovery mode activation"""
        
        with self._lock:
            if self.state.emergency_mode_start:
                recovery_intent = self._create_restrictive_intent(
                    None, "Forced recovery from emergency mode"
                )
                
                self.state.emergency_mode_start = None
                self.state.current_risk_level = RiskLevel.WARNING
                self.state.current_strategy = ControlStrategy.RESTRICTIVE
                
                self._update_decision_statistics(recovery_intent)
                self._publish_control_intent(recovery_intent)
                
                self.logger.warning("Принудительное восстановление из аварийного режима")
                
                return recovery_intent
    
    def __repr__(self) -> str:
        """String representation of constitution"""
        return (f"ConstitutionV0(risk={self.state.current_risk_level.value}, "
               f"strategy={self.state.current_strategy.value}, "
               f"decisions={self.state.total_decisions})")


# Удобные функции для создания конституций
def create_default_constitution(**kwargs) -> ConstitutionV0:
    """Create constitution with default settings"""
    return ConstitutionV0(**kwargs)


def create_strict_constitution() -> ConstitutionV0:
    """Create strict constitution with rigid constraints"""
    strict_constraints = PolicyConstraints(
        u_field_max_amplitude=0.5,
        u_modules_max_amplitude=0.3,
        kuramoto_coupling_max=1.5,
        esc_max_order_parameter=0.7,
        freedom_max_noise_amplitude=0.1,
        freedom_min_coherence_threshold=0.5,
        emergency_mode_max_duration=180.0,
        autonomous_operation_min_coherence=0.7,
        autonomous_operation_max_risk=0.2
    )
    
    return ConstitutionV0(constraints=strict_constraints)


def create_permissive_constitution() -> ConstitutionV0:
    """Create lenient constitution with expanded capabilities"""
    permissive_constraints = PolicyConstraints(
        u_field_max_amplitude=2.0,
        u_modules_max_amplitude=1.0,
        kuramoto_coupling_max=3.0,
        esc_max_order_parameter=0.95,
        freedom_max_noise_amplitude=0.5,
        freedom_min_coherence_threshold=0.2,
        emergency_mode_max_duration=600.0,
        autonomous_operation_min_coherence=0.3,
        autonomous_operation_max_risk=0.5
    )
    
    return ConstitutionV0(constraints=permissive_constraints)


if __name__ == "__main__":
    # Пример использования
    import asyncio
    from ..orchestrator.resonance_bus import initialize_global_bus
    from ..modules.risk_monitor import RiskLevel
    
    async def demo_constitution():
        # Инициализация шины
        await initialize_global_bus()
        
        # Создание конституции
        constitution = create_default_constitution()
        
        # Запуск цикла принятия решений
        await constitution.start_decision_loop()
        
        # Тестирование ручных решений
        for risk_level in [RiskLevel.WARNING, RiskLevel.CRITICAL, RiskLevel.EMERGENCY, RiskLevel.NORMAL]:
            intent = constitution.manual_decision(risk_level, {"test": True})
            print(f"Решение для {risk_level.value}: {intent.get_summary()}")
            await asyncio.sleep(1.0)
        
        # Получение статуса
        status = constitution.get_current_status()
        print(f"\nСтатус конституции:")
        print(f"Стратегия: {status['current_strategy']}")
        print(f"Статистика: {status['statistics']}")
        
        # Остановка
        await constitution.stop_decision_loop()
    
    # Запуск демо
    if __name__ == "__main__":
        asyncio.run(demo_constitution())