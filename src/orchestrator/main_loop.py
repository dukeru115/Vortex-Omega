"""
Main NFCS Orchestrator - Integrator of All Components
===================================================

Central coordination system that unifies all NFCS components into a single
working organism. Implements main control loop with DI, event processing,
control intent application and emergency protocol coordination.

Key Capabilities:
- Main cycle: Collect → Publish → Wait → Apply → Handle Emergency → Update
- DI (Dependency Injection) for all system modules
- Seamless integration: CGL, Kuramoto, ESC, Metrics, Regulator, RiskMonitor, Constitution, EmergencyProtocols
- Graceful shutdown and error handling
- Configurable cycle parameters and timeouts
- Telemetry and detailed logging
- Automatic component lifecycle management
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple, Type
import numpy as np

# NFCS component imports
from ..core.state import SystemState, SystemConfig, create_empty_system_state, validate_system_state
from ..core.cgl_solver import CGLSolver
from ..core.kuramoto_solver import KuramotoSolver
from ..core.metrics import MetricsCalculator
from ..core.regulator import Regulator
from ..utils.config_loader import load_config

# New Stage 1 components
from ..modules.risk_monitor import RiskMonitor, RiskAssessment, create_default_risk_monitor
from ..modules.constitution_v0 import ConstitutionV0, ControlIntent, create_default_constitution
from ..modules.emergency_protocols import EmergencyProtocols, create_default_emergency_protocols
from ..orchestrator.resonance_bus import (
    get_global_bus, initialize_global_bus, TopicType, EventPriority, BusEvent,
    ControlIntentPayload, TelemetryPayload
)


class OrchestratorState(Enum):
    """Состояния главного оркестратора"""
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


class CyclePhase(Enum):
    """Main cycle phases"""
    COLLECT_STATE = "COLLECT_STATE"           # Сбор состояний от всех модулей
    PUBLISH_METRICS = "PUBLISH_METRICS"       # Публикация метрик в шину
    WAIT_DECISIONS = "WAIT_DECISIONS"         # Ожидание решений от конституции
    APPLY_CONTROL = "APPLY_CONTROL"           # Применение управляющих воздействий
    HANDLE_EMERGENCY = "HANDLE_EMERGENCY"     # Processing аварийных ситуаций
    UPDATE_TELEMETRY = "UPDATE_TELEMETRY"     # Update телеметрии


@dataclass
class OrchestratorConfig:
    """Configuration главного оркестратора"""
    
    # Основные параметры цикла
    cycle_frequency_hz: float = 10.0          # Main cycle frequency (Hz)
    max_cycle_time_ms: float = 100.0         # Максимальное time цикла (мс)
    
    # Тайм-ауты
    decision_timeout_ms: float = 50.0         # Тайм-аут ожидания решений
    module_response_timeout_ms: float = 30.0  # Тайм-аут ответа модулей
    emergency_response_timeout_ms: float = 20.0  # Тайм-аут аварийного реагирования
    
    # Performance parameters
    enable_parallel_processing: bool = True   # Parallel module processing
    max_concurrent_modules: int = 4          # Maximum parallel modules
    
    # Error control
    max_consecutive_errors: int = 10         # Maximum consecutive errors
    error_recovery_delay_ms: float = 100.0  # Delay during error recovery
    
    # Telemetry and logging
    enable_detailed_telemetry: bool = True   # Detailed telemetry
    telemetry_interval_cycles: int = 10      # Telemetry interval (cycles)
    
    # Graceful shutdown
    shutdown_timeout_seconds: float = 30.0   # Timeout for graceful shutdown
    
    # Automatic lifecycle control
    auto_start_components: bool = True       # Automatic component startup
    auto_recovery_mode: bool = True          # Автоматическое recovery


@dataclass
class CycleMetrics:
    """Метрики одного цикла выполнения"""
    cycle_number: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Time выполнения фаз (мс)
    collect_time_ms: float = 0.0
    publish_time_ms: float = 0.0
    decision_wait_time_ms: float = 0.0
    apply_control_time_ms: float = 0.0
    emergency_time_ms: float = 0.0
    telemetry_time_ms: float = 0.0
    
    # Результаты
    success: bool = True
    error_message: Optional[str] = None
    current_phase: CyclePhase = CyclePhase.COLLECT_STATE
    
    def get_total_time_ms(self) -> float:
        """Получить общее time цикла в миллисекундах"""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000.0
        return (self.end_time - self.start_time) * 1000.0
    
    def complete_cycle(self, success: bool = True, error: Optional[str] = None):
        """Завершить цикл"""
        self.end_time = time.time()
        self.success = success
        self.error_message = error


@dataclass
class OrchestratorStatistics:
    """Статистика работы оркестратора"""
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    
    # Временные метрики
    avg_cycle_time_ms: float = 0.0
    min_cycle_time_ms: float = float('inf')
    max_cycle_time_ms: float = 0.0
    
    # Производительность
    avg_frequency_hz: float = 0.0
    target_frequency_hz: float = 10.0
    
    # Ошибки
    consecutive_errors: int = 0
    total_errors: int = 0
    last_error_time: Optional[float] = None
    
    # Компоненты
    active_modules: int = 0
    emergency_activations: int = 0
    
    def update_cycle_metrics(self, cycle_metrics: CycleMetrics):
        """Update метрики на основе завершенного цикла"""
        self.total_cycles += 1
        
        cycle_time = cycle_metrics.get_total_time_ms()
        
        if cycle_metrics.success:
            self.successful_cycles += 1
            self.consecutive_errors = 0
            
            # Update временных метрик
            self.avg_cycle_time_ms = (
                self.avg_cycle_time_ms * 0.9 + cycle_time * 0.1
            )
            self.min_cycle_time_ms = min(self.min_cycle_time_ms, cycle_time)
            self.max_cycle_time_ms = max(self.max_cycle_time_ms, cycle_time)
            
            # Update частоты
            if cycle_time > 0:
                current_freq = 1000.0 / cycle_time
                self.avg_frequency_hz = (
                    self.avg_frequency_hz * 0.9 + current_freq * 0.1
                )
        else:
            self.failed_cycles += 1
            self.consecutive_errors += 1
            self.total_errors += 1
            self.last_error_time = time.time()


class NFCSMainOrchestrator:
    """
    Главный оркестратор NFCS
    
    Central coordination system that unifies all system components
    в единый работающий организм через основной цикл управления.
    """
    
    def __init__(self, 
                 config: Optional[OrchestratorConfig] = None,
                 system_config: Optional[SystemConfig] = None):
        
        self.config = config or OrchestratorConfig()
        self.system_config = system_config or load_config()
        
        # State оркестратора
        self.state = OrchestratorState.INITIALIZING
        self.statistics = OrchestratorStatistics()
        self.statistics.target_frequency_hz = self.config.cycle_frequency_hz
        
        # Asyncio-safe synchronization
        self._lock = None  # Will be initialized in async context
        
    async def _ensure_lock(self):
        """Ensure asyncio lock is initialized"""
        if self._lock is None:
            self._lock = asyncio.Lock()
        
        # Основной цикл
        self._main_task: Optional[asyncio.Task] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Компоненты системы (DI контейнер)
        self.components: Dict[str, Any] = {}
        self._component_initialization_order = [
            'bus', 'cgl_solver', 'kuramoto_solver', 'metrics_calculator', 
            'regulator', 'risk_monitor', 'constitution', 'emergency_protocols'
        ]
        
        # Текущее state системы
        self.current_system_state: Optional[SystemState] = None
        self.last_control_intent: Optional[ControlIntent] = None
        
        # Метрики текущего цикла
        self.current_cycle: Optional[CycleMetrics] = None
        
        # Логгер
        self.logger = logging.getLogger(f"{__name__}.NFCSMainOrchestrator")
        
        self.logger.info("Главный оркестратор NFCS initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize all system components
        
        Returns:
            bool: True если initialization успешна
        """
        
        try:
            self.logger.info("🚀 Starting NFCS system initialization...")
            
            # Ensure asyncio lock is initialized
            await self._ensure_lock()
            
            async with self._lock:
                self.state = OrchestratorState.INITIALIZING
            
            # Initialize components in correct order
            success = await self._initialize_components()
            
            if not success:
                self.logger.error("Component initialization error")
                return False
            
            # Creation начального состояния системы
            self._create_initial_system_state()
            
            # Подписка на события
            self._subscribe_to_control_intent()
            
            async with self._lock:
                self.state = OrchestratorState.RUNNING
            
            self.logger.info("✅ NFCS system initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Critical initialization error: {e}")
            async with self._lock:
                self.state = OrchestratorState.ERROR
            return False
    
    async def _initialize_components(self) -> bool:
        """Initialize all system components"""
        
        try:
            # 1. Resonance bus (communication foundation)
            self.logger.info("Initializing resonance bus...")
            self.components['bus'] = await initialize_global_bus(
                max_buffer_size=10000,
                enable_telemetry=True
            )
            
            # 2. Mathematical solvers (system core)
            self.logger.info("Initializing mathematical solvers...")
            self.components['cgl_solver'] = CGLSolver(self.system_config.cgl)
            self.components['kuramoto_solver'] = KuramotoSolver(
                self.system_config.kuramoto,
                module_order=['constitution', 'boundary', 'memory', 'meta_reflection']
            )
            
            # 3. Metrics and regulator
            self.logger.info("Initializing metrics and regulator...")
            self.components['metrics_calculator'] = MetricsCalculator(self.system_config.cost_functional)
            self.components['regulator'] = Regulator(self.system_config.cost_functional)
            
            # 4. Risk monitor (critical security component)
            self.logger.info("Initializing risk monitor...")
            self.components['risk_monitor'] = create_default_risk_monitor(
                enable_auto_publication=True,
                enable_trend_analysis=True,
                enable_anomaly_detection=True
            )
            
            # 5. Constitution (decision-making core)
            self.logger.info("Initializing constitution...")
            self.components['constitution'] = create_default_constitution(
                enable_auto_subscription=True,
                decision_interval=1.0 / self.config.cycle_frequency_hz
            )
            
            # 6. Emergency protocols (system protection)
            self.logger.info("Initialization аварийных протоколов...")
            self.components['emergency_protocols'] = create_default_emergency_protocols(
                enable_auto_detection=True,
                enable_auto_recovery=True
            )
            
            # Start autonomous components if auto-start is enabled
            if self.config.auto_start_components:
                await self._start_autonomous_components()
            
            self.logger.info(f"✅ Initialized {len(self.components)} components")
            
            # Update статистики
            async with self._lock:
                self.statistics.active_modules = len(self.components)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Component initialization error: {e}")
            return False
    
    async def _start_autonomous_components(self):
        """Start autonomous components"""
        
        try:
            # Start цикла принятия решений конституции
            if 'constitution' in self.components:
                await self.components['constitution'].start_decision_loop()
                self.logger.info("Constitution decision-making cycle started")
            
            # Start emergency protocols monitoring
            if 'emergency_protocols' in self.components:
                await self.components['emergency_protocols'].start_monitoring()
                self.logger.info("Emergency protocols monitoring started")
        
        except Exception as e:
            self.logger.error(f"Autonomous components startup error: {e}")
    
    def _create_initial_system_state(self):
        """Create initial system state"""
        
        try:
            # Create empty state
            self.current_system_state = create_empty_system_state(
                grid_size=self.system_config.cgl.grid_size,
                n_modules=len(self.system_config.kuramoto.natural_frequencies)
            )
            
            # Set initial conditions for CGL
            if 'cgl_solver' in self.components:
                initial_field = self.components['cgl_solver'].create_initial_condition(
                    pattern="random_noise", 
                    amplitude=0.1
                )
                self.current_system_state.neural_field = initial_field
            
            self.logger.info(
                f"Created initial state: "
                f"field {self.current_system_state.neural_field.shape}, "
                f"modules {len(self.current_system_state.module_phases)}"
            )
            
        except Exception as e:
            self.logger.error(f"Initial state creation error: {e}")
            raise
    
    def _subscribe_to_control_intent(self):
        """Subscribe to control intent events"""
        
        try:
            bus = self.components.get('bus')
            if bus:
                bus.subscribe(
                    handler_id="orchestrator_control_handler",
                    callback=self._handle_control_intent_event,
                    topic_filter={TopicType.CONTROL_INTENT},
                    priority_filter={EventPriority.HIGH, EventPriority.CRITICAL, EventPriority.EMERGENCY}
                )
                
                self.logger.info("Подписка на управляющие намерения активирована")
        
        except Exception as e:
            self.logger.error(f"Control intent subscription error: {e}")
    
    def _handle_control_intent_event(self, event: BusEvent):
        """Обработчик событий управляющих намерений"""
        
        try:
            if isinstance(event.payload, ControlIntentPayload):
                # Save last intent for application in next cycle
                # В полной реализации здесь должно быть преобразование payload в ControlIntent
                
                self.logger.debug(f"Получено управляющее намерение: {event.payload.decision_type}")
                
                # Simple logging for demonstration
                if event.payload.decision_type == "EMERGENCY":
                    self.logger.critical("🚨 ПОЛУЧЕНО АВАРИЙНОЕ УПРАВЛЯЮЩЕЕ НАМЕРЕНИЕ")
        
        except Exception as e:
            self.logger.error(f"Control intent processing error: {e}")
    
    async def start_main_loop(self):
        """Start основной цикл координации"""
        
        if self._running:
            self.logger.warning("Основной цикл уже started")
            return
        
        try:
            self._running = True
            self._main_task = asyncio.create_task(self._main_coordination_loop())
            
            self.logger.info(
                f"🔄 Основной цикл started с частотой {self.config.cycle_frequency_hz:.1f} Гц"
            )
            
        except Exception as e:
            self.logger.error(f"Main loop startup error: {e}")
            self._running = False
            raise
    
    async def stop_main_loop(self):
        """Stop основной цикл координации"""
        
        if not self._running:
            return
        
        try:
            self.logger.info("Stopping main loop...")
            
            self._running = False
            self._shutdown_event.set()
            
            if self._main_task:
                # Ожидание завершения с тайм-аутом
                try:
                    await asyncio.wait_for(
                        self._main_task, 
                        timeout=self.config.shutdown_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Main loop stop timeout, forcing stop")
                    self._main_task.cancel()
            
            self.logger.info("✅ Основной цикл stopped")
            
        except Exception as e:
            self.logger.error(f"Main loop stop error: {e}")
    
    async def _main_coordination_loop(self):
        """Основной цикл координации"""
        
        self.logger.info("Основной цикл координации started")
        
        cycle_interval = 1.0 / self.config.cycle_frequency_hz
        cycle_number = 0
        
        while self._running:
            cycle_start_time = time.time()
            cycle_number += 1
            
            # Initialization метрик цикла
            self.current_cycle = CycleMetrics(cycle_number=cycle_number)
            
            try:
                # === ФАЗА 1: СБОР СОСТОЯНИЙ ===
                phase_start = time.time()
                self.current_cycle.current_phase = CyclePhase.COLLECT_STATE
                
                success = await self._collect_system_state()
                if not success:
                    raise RuntimeError("System state collection error")
                
                self.current_cycle.collect_time_ms = (time.time() - phase_start) * 1000.0
                
                # === ФАЗА 2: ПУБЛИКАЦИЯ МЕТРИК ===
                phase_start = time.time()
                self.current_cycle.current_phase = CyclePhase.PUBLISH_METRICS
                
                await self._publish_system_metrics()
                
                self.current_cycle.publish_time_ms = (time.time() - phase_start) * 1000.0
                
                # === ФАЗА 3: ОЖИДАНИЕ РЕШЕНИЙ ===
                phase_start = time.time()
                self.current_cycle.current_phase = CyclePhase.WAIT_DECISIONS
                
                # Short pause for event processing in resonance bus
                await asyncio.sleep(self.config.decision_timeout_ms / 1000.0)
                
                self.current_cycle.decision_wait_time_ms = (time.time() - phase_start) * 1000.0
                
                # === ФАЗА 4: ПРИМЕНЕНИЕ УПРАВЛЕНИЯ ===
                phase_start = time.time()
                self.current_cycle.current_phase = CyclePhase.APPLY_CONTROL
                
                await self._apply_control_actions()
                
                self.current_cycle.apply_control_time_ms = (time.time() - phase_start) * 1000.0
                
                # === ФАЗА 5: ОБРАБОТКА АВАРИЙНЫХ СИТУАЦИЙ ===
                phase_start = time.time()
                self.current_cycle.current_phase = CyclePhase.HANDLE_EMERGENCY
                
                await self._handle_emergency_conditions()
                
                self.current_cycle.emergency_time_ms = (time.time() - phase_start) * 1000.0
                
                # === ФАЗА 6: ОБНОВЛЕНИЕ ТЕЛЕМЕТРИИ ===
                phase_start = time.time()
                self.current_cycle.current_phase = CyclePhase.UPDATE_TELEMETRY
                
                if self.config.enable_detailed_telemetry and cycle_number % self.config.telemetry_interval_cycles == 0:
                    await self._update_system_telemetry()
                
                self.current_cycle.telemetry_time_ms = (time.time() - phase_start) * 1000.0
                
                # Завершение успешного цикла
                self.current_cycle.complete_cycle(success=True)
                
                # Update статистики
                async with self._lock:
                    self.statistics.update_cycle_metrics(self.current_cycle)
                
                # Логгирование периодических отчетов
                if cycle_number % 100 == 0:
                    self._log_periodic_status()
                
            except Exception as e:
                # Processing ошибки цикла
                error_msg = f"Error in cycle {cycle_number}: {str(e)}"
                self.logger.error(error_msg)
                
                self.current_cycle.complete_cycle(success=False, error=error_msg)
                
                async with self._lock:
                    self.statistics.update_cycle_metrics(self.current_cycle)
                    
                    # Check критического количества ошибок
                    if self.statistics.consecutive_errors >= self.config.max_consecutive_errors:
                        self.logger.critical(
                            f"Critical number of errors ({self.statistics.consecutive_errors}), "
                            "switching to recovery mode"
                        )
                        
                        if self.config.auto_recovery_mode:
                            await self._attempt_error_recovery()
                        else:
                            break
                
                # Pause при ошибке
                await asyncio.sleep(self.config.error_recovery_delay_ms / 1000.0)
            
            # Регулирование частоты цикла
            cycle_elapsed = time.time() - cycle_start_time
            sleep_time = max(0, cycle_interval - cycle_elapsed)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            elif cycle_elapsed > cycle_interval * 1.5:
                self.logger.warning(
                    f"Цикл {cycle_number} превысил целевое time: "
                    f"{cycle_elapsed*1000:.1f}мс > {cycle_interval*1000:.1f}мс"
                )
        
        self.logger.info("Основной цикл координации completed")
    
    async def _collect_system_state(self) -> bool:
        """Собрать текущее state всех модулей системы"""
        
        try:
            if self.current_system_state is None:
                return False
            
            # Update временных меток
            self.current_system_state.simulation_time += self.system_config.cgl.time_step
            self.current_system_state.current_step += 1
            
            # Вычисление метрик текущего состояния
            if 'metrics_calculator' in self.components:
                self.current_system_state.risk_metrics = (
                    self.components['metrics_calculator'].calculate_all_metrics(
                        self.current_system_state
                    )
                )
            
            # Check валидности состояния
            if not validate_system_state(self.current_system_state):
                self.logger.error("Detected invalid system state")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"System state collection error: {e}")
            return False
    
    async def _publish_system_metrics(self):
        """Опубликовать метрики системы в резонансную шину"""
        
        try:
            if not self.current_system_state or 'risk_monitor' not in self.components:
                return
            
            # Отправка метрик в монитор рисков для анализа
            risk_monitor = self.components['risk_monitor']
            assessment = risk_monitor.assess_risks(self.current_system_state.risk_metrics)
            
            # Публикация уже происходит автоматически в RiskMonitor при enable_auto_publication=True
            
            # Дополнительная телеметрия производительности
            if 'bus' in self.components:
                bus = self.components['bus']
                
                performance_payload = TelemetryPayload(
                    source_module="main_orchestrator",
                    metric_name="system_performance",
                    module_state={
                        'cycle_number': self.current_cycle.cycle_number if self.current_cycle else 0,
                        'system_mode': self.current_system_state.system_mode,
                        'risk_level': assessment.current_level.value,
                        'active_modules': len(self.components)
                    },
                    performance_data={
                        'avg_cycle_time_ms': self.statistics.avg_cycle_time_ms,
                        'current_frequency_hz': self.statistics.avg_frequency_hz,
                        'success_rate': (
                            self.statistics.successful_cycles / 
                            max(1, self.statistics.total_cycles)
                        )
                    }
                )
                
                bus.publish(TopicType.TELEMETRY_EVENT, performance_payload, EventPriority.LOW)
            
        except Exception as e:
            self.logger.error(f"Error публикации метрик системы: {e}")
    
    async def _apply_control_actions(self):
        """Применить управляющие воздействия к системе"""
        
        try:
            if not self.current_system_state or 'regulator' not in self.components:
                return
            
            # Вычисление оптимального управления
            regulator = self.components['regulator']
            control_signals = regulator.compute_optimal_control(self.current_system_state)
            
            # Применение управляющих сигналов к модулям
            await self._apply_control_to_modules(control_signals)
            
            # Сохранение управляющих сигналов в состоянии
            self.current_system_state.last_control_signals = control_signals
            
        except Exception as e:
            self.logger.error(f"Control application error: {e}")
    
    async def _apply_control_to_modules(self, control_signals):
        """Применить управляющие сигналы к конкретным модулям"""
        
        try:
            # Эволюция нейронного поля (CGL)
            if 'cgl_solver' in self.components and self.current_system_state:
                cgl_solver = self.components['cgl_solver']
                
                self.current_system_state.neural_field = cgl_solver.step(
                    self.current_system_state.neural_field,
                    control_signals.u_field
                )
            
            # Эволюция фаз модулей (Kuramoto)
            if 'kuramoto_solver' in self.components and self.current_system_state:
                kuramoto_solver = self.components['kuramoto_solver']
                
                self.current_system_state.module_phases = kuramoto_solver.step(
                    self.current_system_state.module_phases,
                    self.current_system_state.kuramoto_coupling_matrix,
                    control_signals.u_modules
                )
            
        except Exception as e:
            self.logger.error(f"Module control application error: {e}")
    
    async def _handle_emergency_conditions(self):
        """Process аварийные условия"""
        
        try:
            if not self.current_system_state:
                return
            
            # Check аварийных условий через системную конфигурацию
            ha_threshold = self.system_config.emergency_threshold_ha
            defect_threshold = self.system_config.emergency_threshold_defects
            
            risk_metrics = self.current_system_state.risk_metrics
            
            # Check критических условий
            emergency_detected = (
                risk_metrics.hallucination_number > ha_threshold or
                risk_metrics.rho_def_mean > defect_threshold or
                risk_metrics.coherence_global < 0.2 or
                risk_metrics.coherence_modular < 0.2
            )
            
            # Update режима системы
            if emergency_detected and self.current_system_state.system_mode == "NORMAL":
                self.current_system_state.system_mode = "EMERGENCY_MODE"
                
                self.logger.critical(
                    f"🚨 ДЕТЕКЦИЯ АВАРИЙНОГО СОСТОЯНИЯ 🚨\n"
                    f"Ha: {risk_metrics.hallucination_number:.4f} > {ha_threshold}\n"
                    f"Defects: {risk_metrics.rho_def_mean:.4f} > {defect_threshold}\n"
                    f"R_global: {risk_metrics.coherence_global:.4f}\n"
                    f"R_modular: {risk_metrics.coherence_modular:.4f}"
                )
                
                # Update статистики
                async with self._lock:
                    self.statistics.emergency_activations += 1
                
            elif not emergency_detected and self.current_system_state.system_mode == "EMERGENCY_MODE":
                self.current_system_state.system_mode = "NORMAL"
                
                self.logger.info("✅ Выход из аварийного режима - system стабилизирована")
                
        except Exception as e:
            self.logger.error(f"Emergency condition processing error: {e}")
    
    async def _update_system_telemetry(self):
        """Update системную телеметрию"""
        
        try:
            if 'bus' not in self.components or not self.current_system_state:
                return
            
            bus = self.components['bus']
            
            # Detailed system telemetry
            detailed_payload = TelemetryPayload(
                source_module="main_orchestrator",
                metric_name="detailed_system_state",
                module_state={
                    'simulation_time': self.current_system_state.simulation_time,
                    'current_step': self.current_system_state.current_step,
                    'system_mode': self.current_system_state.system_mode,
                    'field_mean_amplitude': float(np.mean(np.abs(self.current_system_state.neural_field))),
                    'field_max_amplitude': float(np.max(np.abs(self.current_system_state.neural_field))),
                    'module_phases_mean': float(np.mean(self.current_system_state.module_phases)),
                    'orchestrator_state': self.state.value
                },
                performance_data={
                    'total_cycles': self.statistics.total_cycles,
                    'success_rate': (
                        self.statistics.successful_cycles / max(1, self.statistics.total_cycles)
                    ),
                    'avg_cycle_time_ms': self.statistics.avg_cycle_time_ms,
                    'frequency_deviation': abs(
                        self.statistics.avg_frequency_hz - self.statistics.target_frequency_hz
                    ),
                    'consecutive_errors': self.statistics.consecutive_errors,
                    'active_modules': self.statistics.active_modules,
                    'emergency_activations': self.statistics.emergency_activations
                }
            )
            
            bus.publish(TopicType.TELEMETRY_EVENT, detailed_payload, EventPriority.LOW)
            
        except Exception as e:
            self.logger.error(f"Telemetry update error: {e}")
    
    async def _attempt_error_recovery(self):
        """Attempt recovery after critical errors"""
        
        self.logger.warning("🔧 Starting automatic system recovery...")
        
        try:
            # Reset error statistics
            async with self._lock:
                self.statistics.consecutive_errors = 0
                self.state = OrchestratorState.RUNNING
            
            # Recreate initial system state
            self._create_initial_system_state()
            
            # Small pause for stabilization
            await asyncio.sleep(1.0)
            
            self.logger.info("✅ Automatic recovery completed")
            
        except Exception as e:
            self.logger.error(f"❌ Automatic recovery error: {e}")
            async with self._lock:
                self.state = OrchestratorState.ERROR
    
    def _log_periodic_status(self):
        """Log periodic system status"""
        
        stats = self.statistics
        
        self.logger.info(
            f"📊 СТАТУС СИСТЕМЫ (цикл {stats.total_cycles}):\n"
            f"├─ Производительность: {stats.avg_frequency_hz:.1f} Hz "
            f"(цель: {stats.target_frequency_hz:.1f} Hz)\n"
            f"├─ Среднее time цикла: {stats.avg_cycle_time_ms:.1f} мс\n"
            f"├─ Успешность: {stats.successful_cycles}/{stats.total_cycles} "
            f"({100 * stats.successful_cycles / max(1, stats.total_cycles):.1f}%)\n"
            f"├─ Активные модули: {stats.active_modules}\n"
            f"├─ Аварийные активации: {stats.emergency_activations}\n"
            f"└─ Текущий режим: {self.current_system_state.system_mode if self.current_system_state else 'N/A'}"
        )
    
    async def shutdown(self):
        """Graceful shutdown всей системы"""
        
        self.logger.info("🔄 Начало graceful shutdown системы NFCS...")
        
        try:
            async with self._lock:
                self.state = OrchestratorState.SHUTDOWN
            
            # Stop main loop
            await self.stop_main_loop()
            
            # Stop autonomous components
            await self._shutdown_autonomous_components()
            
            # Финальная телеметрия
            await self._publish_shutdown_telemetry()
            
            # Stop резонансной шины (последней)
            if 'bus' in self.components:
                from ..orchestrator.resonance_bus import shutdown_global_bus
                await shutdown_global_bus()
            
            self.logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error graceful shutdown: {e}")
    
    async def _shutdown_autonomous_components(self):
        """Stop autonomous components"""
        
        try:
            # Stop цикла принятия решений
            if 'constitution' in self.components:
                await self.components['constitution'].stop_decision_loop()
                self.logger.info("Цикл принятия решений stopped")
            
            # Stop мониторинга аварийных протоколов
            if 'emergency_protocols' in self.components:
                await self.components['emergency_protocols'].stop_monitoring()
                self.logger.info("Monitoring аварийных протоколов stopped")
        
        except Exception as e:
            self.logger.error(f"Autonomous components stop error: {e}")
    
    async def _publish_shutdown_telemetry(self):
        """Опубликовать финальную телеметрию при остановке"""
        
        try:
            if 'bus' in self.components:
                bus = self.components['bus']
                
                shutdown_payload = TelemetryPayload(
                    source_module="main_orchestrator",
                    metric_name="system_shutdown",
                    module_state={
                        'shutdown_reason': 'graceful',
                        'total_runtime_cycles': self.statistics.total_cycles,
                        'final_state': self.state.value
                    },
                    performance_data={
                        'final_success_rate': (
                            self.statistics.successful_cycles / 
                            max(1, self.statistics.total_cycles)
                        ),
                        'total_emergency_activations': self.statistics.emergency_activations,
                        'total_errors': self.statistics.total_errors
                    }
                )
                
                bus.publish(TopicType.TELEMETRY_EVENT, shutdown_payload, EventPriority.HIGH)
                
                # Небольшая pause для отправки финального сообщения
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Final telemetry publication error: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Получить полный статус системы"""
        
        async with self._lock:
            status = {
                'orchestrator_state': self.state.value,
                'is_running': self._running,
                'statistics': {
                    'total_cycles': self.statistics.total_cycles,
                    'successful_cycles': self.statistics.successful_cycles,
                    'failed_cycles': self.statistics.failed_cycles,
                    'success_rate': (
                        self.statistics.successful_cycles / 
                        max(1, self.statistics.total_cycles)
                    ),
                    'avg_cycle_time_ms': self.statistics.avg_cycle_time_ms,
                    'avg_frequency_hz': self.statistics.avg_frequency_hz,
                    'target_frequency_hz': self.statistics.target_frequency_hz,
                    'consecutive_errors': self.statistics.consecutive_errors,
                    'total_errors': self.statistics.total_errors,
                    'emergency_activations': self.statistics.emergency_activations,
                    'active_modules': self.statistics.active_modules
                },
                'components': {
                    name: type(component).__name__ 
                    for name, component in self.components.items()
                },
                'configuration': {
                    'cycle_frequency_hz': self.config.cycle_frequency_hz,
                    'max_cycle_time_ms': self.config.max_cycle_time_ms,
                    'auto_recovery_mode': self.config.auto_recovery_mode,
                    'enable_detailed_telemetry': self.config.enable_detailed_telemetry
                }
            }
            
            # Текущее state системы
            if self.current_system_state:
                status['current_system_state'] = {
                    'simulation_time': self.current_system_state.simulation_time,
                    'current_step': self.current_system_state.current_step,
                    'system_mode': self.current_system_state.system_mode,
                    'field_shape': list(self.current_system_state.neural_field.shape),
                    'n_modules': len(self.current_system_state.module_phases),
                    'last_risk_metrics': {
                        'hallucination_number': self.current_system_state.risk_metrics.hallucination_number,
                        'defect_density_mean': self.current_system_state.risk_metrics.rho_def_mean,
                        'coherence_global': self.current_system_state.risk_metrics.coherence_global,
                        'coherence_modular': self.current_system_state.risk_metrics.coherence_modular,
                        'systemic_risk': self.current_system_state.risk_metrics.systemic_risk
                    }
                }
            
            # Текущий цикл
            if self.current_cycle:
                status['current_cycle'] = {
                    'cycle_number': self.current_cycle.cycle_number,
                    'current_phase': self.current_cycle.current_phase.value,
                    'elapsed_time_ms': self.current_cycle.get_total_time_ms(),
                    'success': self.current_cycle.success
                }
            
            return status
    
    def get_component_status(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get specific component status"""
        
        component = self.components.get(component_name)
        if not component:
            return None
        
        try:
            # Attempt to get status if component has corresponding method
            if hasattr(component, 'get_current_status'):
                return component.get_current_status()
            elif hasattr(component, 'get_statistics'):
                return component.get_statistics()
            else:
                return {
                    'type': type(component).__name__,
                    'available': True,
                    'methods': [method for method in dir(component) if not method.startswith('_')]
                }
        
        except Exception as e:
            self.logger.error(f"Component status retrieval error {component_name}: {e}")
            return {'error': str(e)}
    
    def __repr__(self) -> str:
        """Строковое представление оркестратора"""
        return (f"NFCSMainOrchestrator(state={self.state.value}, "
               f"cycles={self.statistics.total_cycles}, "
               f"components={len(self.components)})")


# Convenient functions for creating and managing orchestrator
async def create_nfcs_orchestrator(config: Optional[OrchestratorConfig] = None,
                                  system_config: Optional[SystemConfig] = None) -> NFCSMainOrchestrator:
    """
    Create и initialize оркестратор NFCS
    
    Args:
        config: Configuration оркестратора
        system_config: Системная configuration NFCS
        
    Returns:
        NFCSMainOrchestrator: Инициализированный оркестратор
    """
    
    orchestrator = NFCSMainOrchestrator(config, system_config)
    
    success = await orchestrator.initialize()
    if not success:
        raise RuntimeError("NFCS orchestrator initialization error")
    
    return orchestrator


def create_default_orchestrator_config() -> OrchestratorConfig:
    """Create конфигурацию оркестратора по умолчанию"""
    return OrchestratorConfig(
        cycle_frequency_hz=10.0,
        max_cycle_time_ms=100.0,
        enable_parallel_processing=True,
        auto_start_components=True,
        auto_recovery_mode=True,
        enable_detailed_telemetry=True
    )


def create_high_performance_config() -> OrchestratorConfig:
    """Create высокопроизводительную конфигурацию"""
    return OrchestratorConfig(
        cycle_frequency_hz=50.0,          # Высокая frequency
        max_cycle_time_ms=20.0,           # Жесткие временные ограничения
        decision_timeout_ms=10.0,         # Быстрые решения
        enable_parallel_processing=True,
        max_concurrent_modules=8,         # Больше параллелизма
        auto_recovery_mode=True,
        enable_detailed_telemetry=False,  # Минимальная телеметрия для производительности
        telemetry_interval_cycles=50      # Редкая телеметрия
    )


def create_safe_config() -> OrchestratorConfig:
    """Create безопасную конфигурацию с консервативными настройками"""
    return OrchestratorConfig(
        cycle_frequency_hz=5.0,           # Низкая frequency для стабильности
        max_cycle_time_ms=200.0,          # Большие тайм-ауты
        decision_timeout_ms=100.0,        # Больше времени на решения
        max_consecutive_errors=3,         # Fast problem detection
        error_recovery_delay_ms=500.0,    # Slow recovery
        enable_parallel_processing=False, # Последовательная processing
        auto_recovery_mode=True,
        enable_detailed_telemetry=True,   # Полная телеметрия
        telemetry_interval_cycles=5       # Частая телеметрия
    )


if __name__ == "__main__":
    # Пример использования
    import asyncio
    
    async def demo_main_orchestrator():
        # Creation оркестратора
        orchestrator = await create_nfcs_orchestrator()
        
        try:
            # Start main loop
            await orchestrator.start_main_loop()
            
            # Работа в течение некоторого времени
            print("🚀 Оркестратор started, monitoring за циклами...")
            
            for i in range(20):
                await asyncio.sleep(1.0)
                
                status = orchestrator.get_system_status()
                print(f"Цикл {status['statistics']['total_cycles']}: "
                      f"{status['statistics']['avg_frequency_hz']:.1f} Hz, "
                      f"успешность {status['statistics']['success_rate']*100:.1f}%")
            
            # Получение детального статуса
            final_status = orchestrator.get_system_status()
            print(f"\n📊 Финальный статус:")
            print(f"Total cycles: {final_status['statistics']['total_cycles']}")
            print(f"Средняя frequency: {final_status['statistics']['avg_frequency_hz']:.2f} Hz")
            print(f"Успешность: {final_status['statistics']['success_rate']*100:.1f}%")
            print(f"Компоненты: {list(final_status['components'].keys())}")
            
        finally:
            # Graceful shutdown
            await orchestrator.shutdown()
    
    # Start демо
    if __name__ == "__main__":
        asyncio.run(demo_main_orchestrator())