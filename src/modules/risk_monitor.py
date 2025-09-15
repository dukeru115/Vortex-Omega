"""
RiskMonitor for NFCS - Critical Safety Component
===============================================

High-performance risk monitoring system with hysteresis, trend analysis
and resonance bus integration. Provides early threat detection and
automatic escalation of critical states.

Key Capabilities:
- Monitor Ha, ρ_def_mean, R_field, R_mod with configurable thresholds
- Hysteresis analysis to prevent false positives
- Derivative analysis (trends) on sliding windows
- Threat classification: NORMAL, WARNING, CRITICAL, EMERGENCY
- Automatic publication to resonance bus
- Detailed logging and telemetry
"""

import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..core.state import RiskMetrics
from ..orchestrator.resonance_bus import (
    get_global_bus,
    publish_risk_metrics,
    publish_emergency,
    RiskMetricsPayload,
    TopicType,
    EventPriority,
)


class RiskLevel(Enum):
    """Risk levels in the system"""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class TrendDirection(Enum):
    """Direction of metric trend"""

    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    STABLE = "STABLE"
    OSCILLATING = "OSCILLATING"


@dataclass
class RiskThresholds:
    """Risk thresholds with hysteresis"""

    # Основные пороги входа в state риска
    ha_warning: float = 0.6
    ha_critical: float = 0.8
    ha_emergency: float = 0.95

    defect_warning: float = 0.05
    defect_critical: float = 0.1
    defect_emergency: float = 0.2

    coherence_global_warning: float = 0.4
    coherence_global_critical: float = 0.3
    coherence_global_emergency: float = 0.2

    coherence_modular_warning: float = 0.5
    coherence_modular_critical: float = 0.3
    coherence_modular_emergency: float = 0.2

    # Гистерезисные множители для выхода из состояния (должны быть < 1.0)
    warning_exit_factor: float = 0.85  # -15% для выхода из WARNING
    critical_exit_factor: float = 0.75  # -25% для выхода из CRITICAL
    emergency_exit_factor: float = 0.7  # -30% для выхода из EMERGENCY

    # Пороги для производных (тренд-анализ)
    derivative_warning_threshold: float = 0.05  # За шаг
    derivative_critical_threshold: float = 0.1  # За шаг

    # Количество нарушений для эскалации
    violations_warning_threshold: int = 3
    violations_critical_threshold: int = 10
    violations_emergency_threshold: int = 25


@dataclass
class MetricHistory:
    """History of metric values for trend analysis"""

    values: deque = field(default_factory=lambda: deque(maxlen=50))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=50))
    window_size: int = 10

    def add_value(self, value: float, timestamp: float = None):
        """Add value to history"""
        if timestamp is None:
            timestamp = time.time()

        self.values.append(value)
        self.timestamps.append(timestamp)

    def get_recent_values(self, count: int = None) -> List[float]:
        """Get recent values"""
        if count is None:
            count = self.window_size
        return list(self.values)[-count:]

    def get_derivative(self) -> float:
        """Calculate derivative (trend) for recent values"""
        if len(self.values) < 2:
            return 0.0

        recent_values = self.get_recent_values()
        recent_times = list(self.timestamps)[-len(recent_values) :]

        if len(recent_values) < 2:
            return 0.0

        # Простая линейная регрессия для тренда
        n = len(recent_values)
        sum_t = sum(recent_times)
        sum_v = sum(recent_values)
        sum_tv = sum(t * v for t, v in zip(recent_times, recent_values))
        sum_t2 = sum(t * t for t in recent_times)

        denominator = n * sum_t2 - sum_t * sum_t
        if abs(denominator) < 1e-10:
            return 0.0

        slope = (n * sum_tv - sum_t * sum_v) / denominator
        return slope

    def get_trend_direction(self) -> TrendDirection:
        """Determine trend direction"""
        derivative = self.get_derivative()
        abs_derivative = abs(derivative)

        # Пороги для определения тренда
        stable_threshold = 1e-6
        oscillation_threshold = 0.01

        if abs_derivative < stable_threshold:
            return TrendDirection.STABLE

        # Check на осцилляции
        if len(self.values) >= 4:
            recent = self.get_recent_values(4)
            direction_changes = 0
            for i in range(1, len(recent) - 1):
                if ((recent[i] > recent[i - 1]) and (recent[i] > recent[i + 1])) or (
                    (recent[i] < recent[i - 1]) and (recent[i] < recent[i + 1])
                ):
                    direction_changes += 1

            if direction_changes >= 2:
                return TrendDirection.OSCILLATING

        return TrendDirection.INCREASING if derivative > 0 else TrendDirection.DECREASING

    def get_volatility(self) -> float:
        """Calculate metric volatility"""
        if len(self.values) < 2:
            return 0.0

        recent_values = self.get_recent_values()
        return float(np.std(recent_values))

    def is_anomaly(self, current_value: float, z_threshold: float = 3.0) -> bool:
        """Anomaly detection via Z-score"""
        if len(self.values) < 5:  # Недостаточно данных
            return False

        recent_values = self.get_recent_values()
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)

        if std_val < 1e-10:  # Нет изменчивости
            return False

        z_score = abs(current_value - mean_val) / std_val
        return z_score > z_threshold


@dataclass
class RiskAssessment:
    """Risk assessment result"""

    current_level: RiskLevel = RiskLevel.NORMAL
    previous_level: RiskLevel = RiskLevel.NORMAL
    level_changed: bool = False

    # Детализированные риски по метрикам
    ha_risk: RiskLevel = RiskLevel.NORMAL
    defect_risk: RiskLevel = RiskLevel.NORMAL
    coherence_global_risk: RiskLevel = RiskLevel.NORMAL
    coherence_modular_risk: RiskLevel = RiskLevel.NORMAL

    # Information о трендах
    ha_trend: TrendDirection = TrendDirection.STABLE
    defect_trend: TrendDirection = TrendDirection.STABLE
    coherence_global_trend: TrendDirection = TrendDirection.STABLE
    coherence_modular_trend: TrendDirection = TrendDirection.STABLE

    # Дополнительная information
    violations_count: int = 0
    anomalies_detected: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    systemic_risk_score: float = 0.0

    # Временные метки
    assessment_time: float = field(default_factory=time.time)

    def get_max_risk_level(self) -> RiskLevel:
        """Get maximum risk level among all metrics"""
        risk_levels = [
            self.ha_risk,
            self.defect_risk,
            self.coherence_global_risk,
            self.coherence_modular_risk,
        ]

        # Порядок серьезности
        level_order = {
            RiskLevel.NORMAL: 0,
            RiskLevel.WARNING: 1,
            RiskLevel.CRITICAL: 2,
            RiskLevel.EMERGENCY: 3,
        }

        max_level = max(risk_levels, key=lambda x: level_order[x])
        return max_level

    def get_risk_summary(self) -> str:
        """Get brief description of risks"""
        if self.current_level == RiskLevel.NORMAL:
            return "System в нормальном состоянии"

        active_risks = []
        if self.ha_risk != RiskLevel.NORMAL:
            active_risks.append(f"Ha:{self.ha_risk.value}")
        if self.defect_risk != RiskLevel.NORMAL:
            active_risks.append(f"Defects:{self.defect_risk.value}")
        if self.coherence_global_risk != RiskLevel.NORMAL:
            active_risks.append(f"R_global:{self.coherence_global_risk.value}")
        if self.coherence_modular_risk != RiskLevel.NORMAL:
            active_risks.append(f"R_modular:{self.coherence_modular_risk.value}")

        return f"Риски: {', '.join(active_risks)}"


class RiskMonitor:
    """
    Монитор рисков для NFCS

    Высокопроизводительная system непрерывного мониторинга состояния NFCS
    с детекцией аномалий, анализом трендов и автоматическим управлением рисками.
    """

    def __init__(
        self,
        thresholds: Optional[RiskThresholds] = None,
        history_window: int = 50,
        enable_trend_analysis: bool = True,
        enable_anomaly_detection: bool = True,
        enable_auto_publication: bool = True,
    ):

        self.thresholds = thresholds or RiskThresholds()
        self.history_window = history_window
        self.enable_trend_analysis = enable_trend_analysis
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_auto_publication = enable_auto_publication

        # История метрик для анализа трендов
        self.ha_history = MetricHistory()
        self.defect_history = MetricHistory()
        self.coherence_global_history = MetricHistory()
        self.coherence_modular_history = MetricHistory()

        # Текущее state
        self.current_risk_level = RiskLevel.NORMAL
        self.last_assessment: Optional[RiskAssessment] = None
        self.total_violations = 0
        self.consecutive_violations = 0
        self.last_violation_reset = time.time()

        # Thread safety
        self._lock = threading.RLock()

        # Статистика
        self.stats = {
            "assessments_count": 0,
            "escalations_count": 0,
            "anomalies_detected": 0,
            "emergency_triggers": 0,
            "avg_assessment_time_ms": 0.0,
            "last_assessment_time": 0.0,
        }

        # Логгер
        self.logger = logging.getLogger(f"{__name__}.RiskMonitor")

        # Резонансная шина
        self.bus = get_global_bus()

        self.logger.info("RiskMonitor initialized")

    def assess_risks(self, risk_metrics: RiskMetrics) -> RiskAssessment:
        """
        Провести комплексную оценку рисков

        Args:
            risk_metrics: Текущие метрики риска из системы

        Returns:
            RiskAssessment: Полная оценка текущего состояния рисков
        """
        start_time = time.time()

        with self._lock:
            try:
                # Update истории метрик
                current_time = time.time()
                self._update_metric_histories(risk_metrics, current_time)

                # Creation новой оценки
                assessment = RiskAssessment()
                assessment.previous_level = self.current_risk_level

                # Оценка рисков по каждой метрике
                assessment.ha_risk = self._assess_hallucination_risk(
                    risk_metrics.hallucination_number
                )
                assessment.defect_risk = self._assess_defect_risk(risk_metrics.rho_def_mean)
                assessment.coherence_global_risk = self._assess_coherence_global_risk(
                    risk_metrics.coherence_global
                )
                assessment.coherence_modular_risk = self._assess_coherence_modular_risk(
                    risk_metrics.coherence_modular
                )

                # Анализ трендов если включен
                if self.enable_trend_analysis:
                    assessment.ha_trend = self.ha_history.get_trend_direction()
                    assessment.defect_trend = self.defect_history.get_trend_direction()
                    assessment.coherence_global_trend = (
                        self.coherence_global_history.get_trend_direction()
                    )
                    assessment.coherence_modular_trend = (
                        self.coherence_modular_history.get_trend_direction()
                    )

                # Детекция аномалий если включена
                if self.enable_anomaly_detection:
                    assessment.anomalies_detected = self._detect_anomalies(risk_metrics)

                # Определение общего уровня риска
                assessment.current_level = assessment.get_max_risk_level()

                # Учет нарушений и трендов для эскалации
                assessment.current_level = self._apply_escalation_logic(
                    assessment.current_level, assessment
                )

                # Update счетчиков нарушений
                assessment.violations_count = self._update_violation_counters(assessment)

                # Вычисление системного риска
                assessment.systemic_risk_score = self._calculate_systemic_risk(
                    risk_metrics, assessment
                )

                # Составление списка факторов риска
                assessment.risk_factors = self._identify_risk_factors(assessment, risk_metrics)

                # Check изменения уровня
                assessment.level_changed = assessment.current_level != assessment.previous_level

                # Update состояния
                self.current_risk_level = assessment.current_level
                self.last_assessment = assessment

                # Автоматическая публикация в шину если включена
                if self.enable_auto_publication:
                    self._publish_assessment(assessment, risk_metrics)

                # Логгирование изменений
                if assessment.level_changed:
                    self.logger.warning(
                        f"Уровень риска изменен: {assessment.previous_level.value} → "
                        f"{assessment.current_level.value}. "
                        f"Причины: {', '.join(assessment.risk_factors)}"
                    )
                    self.stats["escalations_count"] += 1

                    # Критическая эскалация
                    if assessment.current_level == RiskLevel.EMERGENCY:
                        self.stats["emergency_triggers"] += 1
                        self._trigger_emergency(assessment, risk_metrics)

                # Update статистики
                processing_time = (time.time() - start_time) * 1000
                self.stats["assessments_count"] += 1
                self.stats["avg_assessment_time_ms"] = (
                    self.stats["avg_assessment_time_ms"] * 0.9 + processing_time * 0.1
                )
                self.stats["last_assessment_time"] = current_time

                return assessment

            except Exception as e:
                self.logger.error(f"Error при оценке рисков: {e}")
                # Возвращаем безопасную оценку с максимальным риском
                emergency_assessment = RiskAssessment()
                emergency_assessment.current_level = RiskLevel.EMERGENCY
                emergency_assessment.risk_factors = [f"Error монитора: {str(e)}"]
                return emergency_assessment

    def _update_metric_histories(self, risk_metrics: RiskMetrics, timestamp: float):
        """Update history of all metrics"""
        self.ha_history.add_value(risk_metrics.hallucination_number, timestamp)
        self.defect_history.add_value(risk_metrics.rho_def_mean, timestamp)
        self.coherence_global_history.add_value(risk_metrics.coherence_global, timestamp)
        self.coherence_modular_history.add_value(risk_metrics.coherence_modular, timestamp)

    def _assess_hallucination_risk(self, ha_value: float) -> RiskLevel:
        """Assess risk based on hallucination number"""
        return self._assess_metric_with_hysteresis(
            ha_value,
            self.thresholds.ha_warning,
            self.thresholds.ha_critical,
            self.thresholds.ha_emergency,
            increasing_is_bad=True,
        )

    def _assess_defect_risk(self, defect_density: float) -> RiskLevel:
        """Assess risk based on defect density"""
        return self._assess_metric_with_hysteresis(
            defect_density,
            self.thresholds.defect_warning,
            self.thresholds.defect_critical,
            self.thresholds.defect_emergency,
            increasing_is_bad=True,
        )

    def _assess_coherence_global_risk(self, coherence: float) -> RiskLevel:
        """Assess risk based on global coherence"""
        return self._assess_metric_with_hysteresis(
            coherence,
            self.thresholds.coherence_global_warning,
            self.thresholds.coherence_global_critical,
            self.thresholds.coherence_global_emergency,
            increasing_is_bad=False,  # Низкая coherence = плохо
        )

    def _assess_coherence_modular_risk(self, coherence: float) -> RiskLevel:
        """Assess risk based on modular coherence"""
        return self._assess_metric_with_hysteresis(
            coherence,
            self.thresholds.coherence_modular_warning,
            self.thresholds.coherence_modular_critical,
            self.thresholds.coherence_modular_emergency,
            increasing_is_bad=False,  # Низкая coherence = плохо
        )

    def _assess_metric_with_hysteresis(
        self,
        value: float,
        warning_threshold: float,
        critical_threshold: float,
        emergency_threshold: float,
        increasing_is_bad: bool = True,
    ) -> RiskLevel:
        """
        Оценить метрику с учетом гистерезиса для предотвращения дребезга

        Args:
            value: Текущее value метрики
            warning_threshold: Порог WARNING
            critical_threshold: Порог CRITICAL
            emergency_threshold: Порог EMERGENCY
            increasing_is_bad: True если рост метрики = ухудшение
        """

        if increasing_is_bad:
            # Для метрик где увеличение = плохо (Ha, defects)
            if value >= emergency_threshold:
                return RiskLevel.EMERGENCY
            elif value >= critical_threshold:
                return RiskLevel.CRITICAL
            elif value >= warning_threshold:
                return RiskLevel.WARNING
            else:
                # Check гистерезиса для выхода из состояний риска
                if self.current_risk_level == RiskLevel.EMERGENCY:
                    exit_threshold = emergency_threshold * self.thresholds.emergency_exit_factor
                    return RiskLevel.EMERGENCY if value > exit_threshold else RiskLevel.CRITICAL
                elif self.current_risk_level == RiskLevel.CRITICAL:
                    exit_threshold = critical_threshold * self.thresholds.critical_exit_factor
                    return RiskLevel.CRITICAL if value > exit_threshold else RiskLevel.WARNING
                elif self.current_risk_level == RiskLevel.WARNING:
                    exit_threshold = warning_threshold * self.thresholds.warning_exit_factor
                    return RiskLevel.WARNING if value > exit_threshold else RiskLevel.NORMAL
                else:
                    return RiskLevel.NORMAL
        else:
            # Для метрик где уменьшение = плохо (coherence)
            if value <= emergency_threshold:
                return RiskLevel.EMERGENCY
            elif value <= critical_threshold:
                return RiskLevel.CRITICAL
            elif value <= warning_threshold:
                return RiskLevel.WARNING
            else:
                # Check гистерезиса для выхода из состояний риска
                if self.current_risk_level == RiskLevel.EMERGENCY:
                    exit_threshold = emergency_threshold / self.thresholds.emergency_exit_factor
                    return RiskLevel.EMERGENCY if value < exit_threshold else RiskLevel.CRITICAL
                elif self.current_risk_level == RiskLevel.CRITICAL:
                    exit_threshold = critical_threshold / self.thresholds.critical_exit_factor
                    return RiskLevel.CRITICAL if value < exit_threshold else RiskLevel.WARNING
                elif self.current_risk_level == RiskLevel.WARNING:
                    exit_threshold = warning_threshold / self.thresholds.warning_exit_factor
                    return RiskLevel.WARNING if value < exit_threshold else RiskLevel.NORMAL
                else:
                    return RiskLevel.NORMAL

    def _detect_anomalies(self, risk_metrics: RiskMetrics) -> List[str]:
        """Anomaly detection in metrics"""
        anomalies = []

        try:
            # Check аномалий по каждой метрике
            if self.ha_history.is_anomaly(risk_metrics.hallucination_number):
                anomalies.append("Ha_anomaly")

            if self.defect_history.is_anomaly(risk_metrics.rho_def_mean):
                anomalies.append("Defects_anomaly")

            if self.coherence_global_history.is_anomaly(risk_metrics.coherence_global):
                anomalies.append("Coherence_global_anomaly")

            if self.coherence_modular_history.is_anomaly(risk_metrics.coherence_modular):
                anomalies.append("Coherence_modular_anomaly")

            # Update статистики
            if anomalies:
                self.stats["anomalies_detected"] += len(anomalies)
                self.logger.warning(f"Обнаружены аномалии: {', '.join(anomalies)}")

        except Exception as e:
            self.logger.error(f"Error детекции аномалий: {e}")

        return anomalies

    def _apply_escalation_logic(
        self, base_level: RiskLevel, assessment: RiskAssessment
    ) -> RiskLevel:
        """Apply escalation logic based on trends and violations"""

        # Check неблагоприятных трендов
        negative_trends = [
            assessment.ha_trend == TrendDirection.INCREASING,
            assessment.defect_trend == TrendDirection.INCREASING,
            assessment.coherence_global_trend == TrendDirection.DECREASING,
            assessment.coherence_modular_trend == TrendDirection.DECREASING,
        ]

        negative_trend_count = sum(negative_trends)

        # Эскалация на основе множественных негативных трендов
        if negative_trend_count >= 3:  # 3+ неблагоприятных тренда
            if base_level == RiskLevel.WARNING:
                return RiskLevel.CRITICAL
            elif base_level == RiskLevel.CRITICAL:
                return RiskLevel.EMERGENCY
        elif negative_trend_count >= 2:  # 2 неблагоприятных тренда
            if base_level == RiskLevel.NORMAL:
                return RiskLevel.WARNING

        # Эскалация на основе количества нарушений
        if self.consecutive_violations >= self.thresholds.violations_emergency_threshold:
            return RiskLevel.EMERGENCY
        elif self.consecutive_violations >= self.thresholds.violations_critical_threshold:
            return max(base_level, RiskLevel.CRITICAL, key=lambda x: x.value)
        elif self.consecutive_violations >= self.thresholds.violations_warning_threshold:
            return max(base_level, RiskLevel.WARNING, key=lambda x: x.value)

        return base_level

    def _update_violation_counters(self, assessment: RiskAssessment) -> int:
        """Update violation counters"""
        current_time = time.time()

        # Считаем нарушением любое state выше NORMAL
        if assessment.current_level != RiskLevel.NORMAL:
            self.total_violations += 1
            self.consecutive_violations += 1
        else:
            # Сброс последовательных нарушений при возврате к норме
            if self.consecutive_violations > 0:
                self.logger.info(
                    f"System восстановлена после {self.consecutive_violations} нарушений"
                )
            self.consecutive_violations = 0

        # Сброс счетчика раз в hour
        if current_time - self.last_violation_reset > 3600:
            self.total_violations = 0
            self.last_violation_reset = current_time

        return self.total_violations

    def _calculate_systemic_risk(
        self, risk_metrics: RiskMetrics, assessment: RiskAssessment
    ) -> float:
        """Calculate integral systemic risk assessment [0.0 - 1.0]"""

        # Веса для разных компонентов риска
        weights = {
            "ha": 0.3,
            "defects": 0.25,
            "coherence_global": 0.2,
            "coherence_modular": 0.15,
            "trends": 0.05,
            "anomalies": 0.05,
        }

        # Нормализация метрик к [0, 1]
        ha_normalized = min(risk_metrics.hallucination_number / 1.0, 1.0)
        defects_normalized = min(risk_metrics.rho_def_mean / 0.5, 1.0)
        coherence_global_normalized = max(1.0 - risk_metrics.coherence_global, 0.0)
        coherence_modular_normalized = max(1.0 - risk_metrics.coherence_modular, 0.0)

        # Оценка трендов
        trend_risk = 0.0
        if assessment.ha_trend == TrendDirection.INCREASING:
            trend_risk += 0.25
        if assessment.defect_trend == TrendDirection.INCREASING:
            trend_risk += 0.25
        if assessment.coherence_global_trend == TrendDirection.DECREASING:
            trend_risk += 0.25
        if assessment.coherence_modular_trend == TrendDirection.DECREASING:
            trend_risk += 0.25

        # Оценка аномалий
        anomaly_risk = min(len(assessment.anomalies_detected) / 4.0, 1.0)

        # Интегральная оценка
        systemic_risk = (
            weights["ha"] * ha_normalized
            + weights["defects"] * defects_normalized
            + weights["coherence_global"] * coherence_global_normalized
            + weights["coherence_modular"] * coherence_modular_normalized
            + weights["trends"] * trend_risk
            + weights["anomalies"] * anomaly_risk
        )

        return min(max(systemic_risk, 0.0), 1.0)

    def _identify_risk_factors(
        self, assessment: RiskAssessment, risk_metrics: RiskMetrics
    ) -> List[str]:
        """Identify active risk factors"""
        factors = []

        # Факторы по метрикам
        if assessment.ha_risk != RiskLevel.NORMAL:
            factors.append(f"Ha={risk_metrics.hallucination_number:.3f}")

        if assessment.defect_risk != RiskLevel.NORMAL:
            factors.append(f"Defects={risk_metrics.rho_def_mean:.3f}")

        if assessment.coherence_global_risk != RiskLevel.NORMAL:
            factors.append(f"R_global={risk_metrics.coherence_global:.3f}")

        if assessment.coherence_modular_risk != RiskLevel.NORMAL:
            factors.append(f"R_modular={risk_metrics.coherence_modular:.3f}")

        # Факторы по трендам
        if assessment.ha_trend == TrendDirection.INCREASING:
            factors.append("Ha_trending_up")
        if assessment.defect_trend == TrendDirection.INCREASING:
            factors.append("Defects_trending_up")
        if assessment.coherence_global_trend == TrendDirection.DECREASING:
            factors.append("R_global_trending_down")
        if assessment.coherence_modular_trend == TrendDirection.DECREASING:
            factors.append("R_modular_trending_down")

        # Факторы по аномалиям
        factors.extend(assessment.anomalies_detected)

        # Факторы по нарушениям
        if self.consecutive_violations > self.thresholds.violations_warning_threshold:
            factors.append(f"Violations={self.consecutive_violations}")

        return factors

    def _publish_assessment(self, assessment: RiskAssessment, risk_metrics: RiskMetrics):
        """Publish assessment to resonance bus"""
        try:
            # Публикация базовых метрик риска
            publish_risk_metrics(
                hallucination_number=risk_metrics.hallucination_number,
                defect_density_mean=risk_metrics.rho_def_mean,
                risk_level=assessment.current_level.value,
                source_module="risk_monitor",
            )

            # Дополнительная детализированная публикация
            detailed_payload = RiskMetricsPayload(
                source_module="risk_monitor",
                hallucination_number=risk_metrics.hallucination_number,
                defect_density_mean=risk_metrics.rho_def_mean,
                coherence_global=risk_metrics.coherence_global,
                coherence_modular=risk_metrics.coherence_modular,
                systemic_risk=assessment.systemic_risk_score,
                risk_level=assessment.current_level.value,
                risk_trend=assessment.ha_trend.value,  # Основной тренд
                violations_count=assessment.violations_count,
            )

            priority = EventPriority.NORMAL
            if assessment.current_level == RiskLevel.CRITICAL:
                priority = EventPriority.HIGH
            elif assessment.current_level == RiskLevel.EMERGENCY:
                priority = EventPriority.EMERGENCY

            self.bus.publish(TopicType.METRICS_RISK, detailed_payload, priority)

        except Exception as e:
            self.logger.error(f"Error публикации оценки рисков: {e}")

    def _trigger_emergency(self, assessment: RiskAssessment, risk_metrics: RiskMetrics):
        """Launch emergency protocols"""
        try:
            emergency_reason = f"Критическое state системы: {assessment.get_risk_summary()}"

            publish_emergency(
                emergency_type="SYSTEMIC_RISK_CRITICAL",
                severity_level=5,
                trigger_reason=emergency_reason,
                source_module="risk_monitor",
            )

            self.logger.critical(
                f"🚨 EMERGENCY TRIGGERED: {emergency_reason}\n"
                f"Системный risk: {assessment.systemic_risk_score:.3f}\n"
                f"Факторы: {', '.join(assessment.risk_factors)}\n"
                f"Ha: {risk_metrics.hallucination_number:.4f}, "
                f"Defects: {risk_metrics.rho_def_mean:.4f}, "
                f"R_global: {risk_metrics.coherence_global:.4f}, "
                f"R_modular: {risk_metrics.coherence_modular:.4f}"
            )

        except Exception as e:
            self.logger.error(f"Error запуска аварийных протоколов: {e}")

    def get_current_status(self) -> Dict[str, Any]:
        """Get current risk monitor status"""
        with self._lock:
            status = {
                "current_risk_level": self.current_risk_level.value,
                "total_violations": self.total_violations,
                "consecutive_violations": self.consecutive_violations,
                "statistics": self.stats.copy(),
                "thresholds": {
                    "ha_emergency": self.thresholds.ha_emergency,
                    "defect_emergency": self.thresholds.defect_emergency,
                    "coherence_global_emergency": self.thresholds.coherence_global_emergency,
                    "coherence_modular_emergency": self.thresholds.coherence_modular_emergency,
                },
            }

            if self.last_assessment:
                status["last_assessment"] = {
                    "level": self.last_assessment.current_level.value,
                    "systemic_risk": self.last_assessment.systemic_risk_score,
                    "risk_factors": self.last_assessment.risk_factors,
                    "anomalies": self.last_assessment.anomalies_detected,
                    "time": self.last_assessment.assessment_time,
                }

            return status

    def update_thresholds(self, new_thresholds: RiskThresholds):
        """Update risk thresholds"""
        with self._lock:
            old_thresholds = self.thresholds
            self.thresholds = new_thresholds

            self.logger.info(f"Пороги риска обновлены")

            # Можно добавить логику для плавного перехода

    def reset_violation_counters(self):
        """Reset violation counters"""
        with self._lock:
            self.total_violations = 0
            self.consecutive_violations = 0
            self.last_violation_reset = time.time()

            self.logger.info("Счетчики нарушений сброшены")

    def get_metric_histories(self) -> Dict[str, List[float]]:
        """Get history of all metrics for analysis"""
        return {
            "ha_values": list(self.ha_history.values),
            "defect_values": list(self.defect_history.values),
            "coherence_global_values": list(self.coherence_global_history.values),
            "coherence_modular_values": list(self.coherence_modular_history.values),
            "timestamps": list(self.ha_history.timestamps),
        }

    def __repr__(self) -> str:
        """String representation of monitor"""
        return (
            f"RiskMonitor(level={self.current_risk_level.value}, "
            f"violations={self.consecutive_violations}, "
            f"assessments={self.stats['assessments_count']})"
        )


# Удобные функции для быстрого использования
def create_default_risk_monitor(**kwargs) -> RiskMonitor:
    """Create risk monitor with default settings"""
    return RiskMonitor(**kwargs)


def create_strict_risk_monitor() -> RiskMonitor:
    """Create strict risk monitor with low thresholds"""
    strict_thresholds = RiskThresholds(
        ha_warning=0.4,
        ha_critical=0.6,
        ha_emergency=0.8,
        defect_warning=0.03,
        defect_critical=0.07,
        defect_emergency=0.15,
        coherence_global_warning=0.5,
        coherence_global_critical=0.4,
        coherence_global_emergency=0.3,
        coherence_modular_warning=0.6,
        coherence_modular_critical=0.4,
        coherence_modular_emergency=0.3,
    )

    return RiskMonitor(thresholds=strict_thresholds)


def create_relaxed_risk_monitor() -> RiskMonitor:
    """Create lenient risk monitor with high thresholds"""
    relaxed_thresholds = RiskThresholds(
        ha_warning=0.8,
        ha_critical=0.9,
        ha_emergency=0.99,
        defect_warning=0.1,
        defect_critical=0.2,
        defect_emergency=0.4,
        coherence_global_warning=0.3,
        coherence_global_critical=0.2,
        coherence_global_emergency=0.1,
        coherence_modular_warning=0.4,
        coherence_modular_critical=0.2,
        coherence_modular_emergency=0.1,
    )

    return RiskMonitor(thresholds=relaxed_thresholds)


if __name__ == "__main__":
    # Пример использования
    import asyncio
    from ..orchestrator.resonance_bus import initialize_global_bus

    async def demo_risk_monitor():
        # Initialization шины
        await initialize_global_bus()

        # Creation монитора
        monitor = create_default_risk_monitor()

        # Creation тестовых метрик
        from ..core.state import RiskMetrics
        import numpy as np

        # Симуляция нарастающего риска
        for step in range(20):
            # Постепенно ухудшающиеся метрики
            test_metrics = RiskMetrics(
                rho_def_field=np.zeros((10, 10)),
                rho_def_mean=0.02 + step * 0.01,
                hallucination_number=0.3 + step * 0.04,
                systemic_risk=0.1 + step * 0.03,
                coherence_global=0.8 - step * 0.03,
                coherence_modular=0.7 - step * 0.025,
            )

            # Оценка рисков
            assessment = monitor.assess_risks(test_metrics)

            print(
                f"Шаг {step}: {assessment.current_level.value} "
                f"(risk: {assessment.systemic_risk_score:.3f}) - "
                f"{assessment.get_risk_summary()}"
            )

            # Pause
            await asyncio.sleep(0.1)

        # Статистика
        status = monitor.get_current_status()
        print(f"\nСтатистика: {status['statistics']}")

    # Start демо
    if __name__ == "__main__":
        asyncio.run(demo_risk_monitor())
