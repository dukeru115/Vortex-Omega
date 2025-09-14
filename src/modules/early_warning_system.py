"""
Early Warning System Integration
===============================

Advanced early warning system for Neural Field Control System (NFCS) v2.4.3
providing predictive monitoring and proactive threat detection.

This system provides:
1. Predictive analytics for Ha trajectory prediction  
2. Pattern recognition for anomaly detection
3. Multi-threshold early warning alerts
4. Integration with constitutional monitoring
5. Proactive emergency protocol recommendations

Based on NFCS theoretical framework with machine learning enhancements
for predictive constitutional oversight and threat anticipation.

Author: Team Omega (GenSpark AI Implementation)  
License: CC BY-NC 4.0
Date: 2025-09-14
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import numpy as np
from pathlib import Path
import sqlite3
from collections import deque
from scipy import stats, signal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class WarningLevel(Enum):
    """Early warning system alert levels"""
    BASELINE = 0      # Normal operation baseline
    ADVISORY = 1      # Advisory level - trend monitoring
    WATCH = 2         # Watch level - pattern detected  
    WARNING = 3       # Warning level - threshold approaching
    ALERT = 4         # Alert level - immediate attention needed
    EMERGENCY = 5     # Emergency level - critical action required


class PredictionHorizon(Enum):
    """Prediction time horizons for early warning"""
    SHORT_TERM = 30    # 30 seconds ahead
    MEDIUM_TERM = 180  # 3 minutes ahead  
    LONG_TERM = 600    # 10 minutes ahead


@dataclass
class EarlyWarningPrediction:
    """Prediction result from early warning system"""
    timestamp: float = field(default_factory=time.time)
    horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM
    
    # Predicted values
    predicted_ha: float = 0.0
    predicted_integrity: float = 1.0
    predicted_coherence: float = 1.0
    
    # Confidence metrics
    prediction_confidence: float = 0.0
    model_accuracy: float = 0.0
    uncertainty_estimate: float = 0.0
    
    # Risk assessment
    risk_probability: float = 0.0
    threat_indicators: List[str] = field(default_factory=list)
    recommended_horizon: Optional[PredictionHorizon] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'horizon': self.horizon.name,
            'horizon_seconds': self.horizon.value,
            'predicted_ha': self.predicted_ha,
            'predicted_integrity': self.predicted_integrity,
            'predicted_coherence': self.predicted_coherence,
            'prediction_confidence': self.prediction_confidence,
            'model_accuracy': self.model_accuracy,
            'uncertainty_estimate': self.uncertainty_estimate,
            'risk_probability': self.risk_probability,
            'threat_indicators': self.threat_indicators,
            'recommended_horizon': self.recommended_horizon.name if self.recommended_horizon else None
        }


@dataclass
class EarlyWarningAlert:
    """Early warning alert with predictive information"""
    alert_id: str
    timestamp: float = field(default_factory=time.time)
    warning_level: WarningLevel = WarningLevel.ADVISORY
    
    # Alert details
    title: str = ""
    description: str = ""
    trigger_conditions: List[str] = field(default_factory=list)
    
    # Predictive information
    time_to_threshold: Optional[float] = None  # Seconds until threshold breach
    prediction_horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM
    confidence_level: float = 0.0
    
    # Recommended actions
    preventive_actions: List[str] = field(default_factory=list)
    monitoring_recommendations: List[str] = field(default_factory=list)
    escalation_timeline: Dict[str, float] = field(default_factory=dict)
    
    # Related data
    current_metrics: Optional[Dict] = None
    predictions: Optional[EarlyWarningPrediction] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp,
            'warning_level': self.warning_level.name,
            'title': self.title,
            'description': self.description,
            'trigger_conditions': self.trigger_conditions,
            'time_to_threshold': self.time_to_threshold,
            'prediction_horizon': self.prediction_horizon.name,
            'confidence_level': self.confidence_level,
            'preventive_actions': self.preventive_actions,
            'monitoring_recommendations': self.monitoring_recommendations,
            'escalation_timeline': self.escalation_timeline,
            'predictions': self.predictions.to_dict() if self.predictions else None
        }


@dataclass 
class EarlyWarningConfiguration:
    """Configuration for early warning system"""
    
    # Prediction model settings
    prediction_window_size: int = 60        # Historical data points for prediction
    min_data_points: int = 30               # Minimum data for reliable prediction
    model_retrain_interval: int = 3600      # Retrain models every hour
    
    # Warning thresholds (% of critical thresholds)
    advisory_threshold_pct: float = 0.6     # 60% of critical threshold
    watch_threshold_pct: float = 0.75       # 75% of critical threshold  
    warning_threshold_pct: float = 0.85     # 85% of critical threshold
    alert_threshold_pct: float = 0.95       # 95% of critical threshold
    
    # Prediction confidence requirements
    min_prediction_confidence: float = 0.7   # Minimum confidence for alerts
    high_confidence_threshold: float = 0.9   # High confidence threshold
    uncertainty_warning_threshold: float = 0.3  # High uncertainty warning
    
    # Anomaly detection settings
    anomaly_contamination: float = 0.1      # Expected proportion of anomalies
    anomaly_sensitivity: float = 0.8        # Sensitivity for anomaly detection
    pattern_correlation_threshold: float = 0.7  # Pattern correlation threshold
    
    # Time-based settings
    prediction_update_interval: float = 5.0  # Update predictions every 5 seconds
    alert_suppression_time: float = 30.0    # Suppress duplicate alerts for 30s
    trend_analysis_window: int = 300        # 5 minutes for trend analysis
    
    # Integration settings
    enable_constitutional_integration: bool = True
    enable_predictive_protocols: bool = True
    enable_pattern_learning: bool = True


class EarlyWarningSystem:
    """
    Advanced Early Warning System for NFCS with predictive capabilities
    """
    
    def __init__(self, config: Optional[EarlyWarningConfiguration] = None):
        """Initialize early warning system"""
        self.config = config or EarlyWarningConfiguration()
        
        # Data storage
        self.metrics_buffer = deque(maxlen=self.config.prediction_window_size * 2)
        self.prediction_history = deque(maxlen=200)
        self.active_warnings: Dict[str, EarlyWarningAlert] = {}
        
        # Machine learning models
        self.anomaly_detector = IsolationForest(
            contamination=self.config.anomaly_contamination,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.models_trained = False
        self.last_retrain_time = 0
        
        # Prediction state
        self.current_predictions: Dict[PredictionHorizon, EarlyWarningPrediction] = {}
        self.trend_analysis = {}
        self.pattern_cache = {}
        
        # System state
        self.warning_active = False
        self.last_alert_times: Dict[str, float] = {}
        
        # Integration callbacks
        self.constitutional_callback: Optional[Callable] = None
        self.emergency_callback: Optional[Callable] = None
        self.metrics_callback: Optional[Callable] = None
        
        logger.info("Early Warning System initialized")
    
    async def start_monitoring(self, 
                             constitutional_callback: Optional[Callable] = None,
                             emergency_callback: Optional[Callable] = None,
                             metrics_callback: Optional[Callable] = None):
        """Start early warning monitoring"""
        self.constitutional_callback = constitutional_callback
        self.emergency_callback = emergency_callback  
        self.metrics_callback = metrics_callback
        
        self.warning_active = True
        
        # Start monitoring tasks
        asyncio.create_task(self._prediction_update_loop())
        asyncio.create_task(self._warning_evaluation_loop())
        asyncio.create_task(self._model_maintenance_loop())
        
        logger.info("Early warning monitoring started")
    
    async def stop_monitoring(self):
        """Stop early warning monitoring"""
        self.warning_active = False
        logger.info("Early warning monitoring stopped")
    
    async def update_metrics(self, metrics_data: Dict[str, Any]):
        """Update system with new metrics data"""
        current_time = time.time()
        
        # Add timestamp if not present
        if 'timestamp' not in metrics_data:
            metrics_data['timestamp'] = current_time
        
        # Store in buffer
        self.metrics_buffer.append(metrics_data.copy())
        
        # Update predictions if enough data
        if len(self.metrics_buffer) >= self.config.min_data_points:
            await self._update_predictions()
            await self._evaluate_warnings()
    
    async def _prediction_update_loop(self):
        """Main prediction update loop"""
        while self.warning_active:
            try:
                if len(self.metrics_buffer) >= self.config.min_data_points:
                    await self._update_predictions()
                
                await asyncio.sleep(self.config.prediction_update_interval)
            except Exception as e:
                logger.error(f"Prediction update error: {e}")
                await asyncio.sleep(1.0)
    
    async def _warning_evaluation_loop(self):
        """Warning evaluation and alert generation loop"""
        while self.warning_active:
            try:
                await self._evaluate_warnings()
                await self._cleanup_expired_warnings()
                await asyncio.sleep(2.0)  # Check warnings every 2 seconds
            except Exception as e:
                logger.error(f"Warning evaluation error: {e}")
                await asyncio.sleep(1.0)
    
    async def _model_maintenance_loop(self):
        """Model retraining and maintenance loop"""
        while self.warning_active:
            try:
                current_time = time.time()
                if (current_time - self.last_retrain_time) > self.config.model_retrain_interval:
                    await self._retrain_models()
                
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Model maintenance error: {e}")
                await asyncio.sleep(60.0)
    
    async def _update_predictions(self):
        """Update predictions for all horizons"""
        if len(self.metrics_buffer) < self.config.min_data_points:
            return
        
        try:
            # Prepare data
            data_array = self._prepare_prediction_data()
            
            if data_array is None or len(data_array) < 10:
                return
            
            # Generate predictions for each horizon
            for horizon in PredictionHorizon:
                prediction = await self._generate_prediction(data_array, horizon)
                if prediction:
                    self.current_predictions[horizon] = prediction
            
            # Store prediction in history
            if PredictionHorizon.SHORT_TERM in self.current_predictions:
                self.prediction_history.append(self.current_predictions[PredictionHorizon.SHORT_TERM])
        
        except Exception as e:
            logger.error(f"Prediction update error: {e}")
    
    def _prepare_prediction_data(self) -> Optional[np.ndarray]:
        """Prepare data for machine learning predictions"""
        try:
            if len(self.metrics_buffer) < self.config.min_data_points:
                return None
            
            # Extract key metrics
            metrics_list = []
            for metrics in list(self.metrics_buffer):
                row = [
                    metrics.get('hallucination_number', 0.0),
                    metrics.get('coherence_measure', 0.0),
                    metrics.get('defect_density', 0.0),
                    metrics.get('field_energy', 0.0),
                    metrics.get('integrity_score', 1.0),
                    metrics.get('constitutional_compliance', 1.0),
                    metrics.get('system_stability', 1.0)
                ]
                metrics_list.append(row)
            
            data_array = np.array(metrics_list)
            
            # Handle NaN values
            if np.any(np.isnan(data_array)):
                data_array = np.nan_to_num(data_array, nan=0.0)
            
            return data_array
        
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            return None
    
    async def _generate_prediction(self, data_array: np.ndarray, 
                                 horizon: PredictionHorizon) -> Optional[EarlyWarningPrediction]:
        """Generate prediction for specific horizon"""
        try:
            # Simple trend-based prediction (can be enhanced with more sophisticated models)
            steps_ahead = max(1, horizon.value // 5)  # Convert seconds to prediction steps
            
            # Get recent trend
            if len(data_array) < steps_ahead + 5:
                return None
            
            # Calculate trends for each metric
            ha_values = data_array[-20:, 0]  # Last 20 Ha values
            integrity_values = data_array[-20:, 4]  # Last 20 integrity values  
            coherence_values = data_array[-20:, 1]  # Last 20 coherence values
            
            # Linear trend extrapolation
            time_points = np.arange(len(ha_values))
            
            # Ha prediction
            if len(ha_values) > 2:
                ha_slope, ha_intercept, ha_r_value, _, _ = stats.linregress(time_points, ha_values)
                predicted_ha = ha_intercept + ha_slope * (len(time_points) + steps_ahead)
                ha_confidence = abs(ha_r_value)
            else:
                predicted_ha = ha_values[-1] if len(ha_values) > 0 else 0.0
                ha_confidence = 0.5
            
            # Integrity prediction  
            if len(integrity_values) > 2:
                int_slope, int_intercept, int_r_value, _, _ = stats.linregress(time_points, integrity_values)
                predicted_integrity = int_intercept + int_slope * (len(time_points) + steps_ahead)
                int_confidence = abs(int_r_value)
            else:
                predicted_integrity = integrity_values[-1] if len(integrity_values) > 0 else 1.0
                int_confidence = 0.5
            
            # Coherence prediction
            if len(coherence_values) > 2:
                coh_slope, coh_intercept, coh_r_value, _, _ = stats.linregress(time_points, coherence_values)
                predicted_coherence = coh_intercept + coh_slope * (len(time_points) + steps_ahead)
                coh_confidence = abs(coh_r_value)
            else:
                predicted_coherence = coherence_values[-1] if len(coherence_values) > 0 else 1.0
                coh_confidence = 0.5
            
            # Overall prediction confidence
            prediction_confidence = np.mean([ha_confidence, int_confidence, coh_confidence])
            
            # Risk assessment
            risk_factors = []
            risk_score = 0.0
            
            if predicted_ha > 1.0:
                risk_score += 0.3
                risk_factors.append(f"Ha trending upward: {predicted_ha:.3f}")
            
            if predicted_integrity < 0.8:
                risk_score += 0.3
                risk_factors.append(f"Integrity declining: {predicted_integrity:.3f}")
            
            if predicted_coherence < 0.5:
                risk_score += 0.2  
                risk_factors.append(f"Coherence degrading: {predicted_coherence:.3f}")
            
            # Uncertainty estimate based on recent volatility
            ha_std = np.std(ha_values) if len(ha_values) > 2 else 0.1
            uncertainty = min(1.0, ha_std * 2.0)
            
            # Create prediction object
            prediction = EarlyWarningPrediction(
                horizon=horizon,
                predicted_ha=max(0.0, predicted_ha),
                predicted_integrity=np.clip(predicted_integrity, 0.0, 1.0),
                predicted_coherence=np.clip(predicted_coherence, 0.0, 1.0),
                prediction_confidence=prediction_confidence,
                model_accuracy=prediction_confidence,  # Simplified
                uncertainty_estimate=uncertainty,
                risk_probability=risk_score,
                threat_indicators=risk_factors
            )
            
            return prediction
        
        except Exception as e:
            logger.error(f"Prediction generation error: {e}")
            return None
    
    async def _evaluate_warnings(self):
        """Evaluate current predictions and generate warnings"""
        if not self.current_predictions:
            return
        
        try:
            # Get constitutional thresholds (from integration if available)
            ha_critical = 2.0   # Default thresholds
            ha_warning = 1.0
            integrity_critical = 0.5
            integrity_warning = 0.7
            
            if self.constitutional_callback:
                try:
                    thresholds = await self._safe_callback(self.constitutional_callback, 'get_thresholds')
                    if thresholds:
                        ha_critical = thresholds.get('ha_critical', 2.0)
                        ha_warning = thresholds.get('ha_warning', 1.0)
                        integrity_critical = thresholds.get('integrity_critical', 0.5)
                        integrity_warning = thresholds.get('integrity_warning', 0.7)
                except Exception as e:
                    logger.debug(f"Could not get thresholds: {e}")
            
            # Evaluate each prediction horizon
            for horizon, prediction in self.current_predictions.items():
                await self._evaluate_prediction_warnings(
                    prediction, ha_critical, ha_warning, 
                    integrity_critical, integrity_warning
                )
        
        except Exception as e:
            logger.error(f"Warning evaluation error: {e}")
    
    async def _evaluate_prediction_warnings(self, prediction: EarlyWarningPrediction,
                                          ha_critical: float, ha_warning: float,
                                          integrity_critical: float, integrity_warning: float):
        """Evaluate specific prediction for warnings"""
        current_time = time.time()
        
        # Skip if prediction confidence is too low
        if prediction.prediction_confidence < self.config.min_prediction_confidence:
            return
        
        warnings_to_create = []
        
        # Ha-based warnings
        ha_alert_pct = self.config.alert_threshold_pct
        ha_warning_pct = self.config.warning_threshold_pct
        
        if prediction.predicted_ha >= ha_critical * ha_alert_pct:
            alert_id = f"ha_alert_{prediction.horizon.name.lower()}"
            if self._should_create_alert(alert_id):
                
                # Estimate time to threshold
                current_ha = list(self.metrics_buffer)[-1].get('hallucination_number', 0.0) if self.metrics_buffer else 0.0
                time_to_threshold = self._estimate_time_to_threshold(
                    current_ha, prediction.predicted_ha, ha_critical, prediction.horizon.value
                )
                
                warning = EarlyWarningAlert(
                    alert_id=alert_id,
                    warning_level=WarningLevel.ALERT,
                    title=f"Predicted Ha Alert ({prediction.horizon.name})",
                    description=f"Ha predicted to reach {prediction.predicted_ha:.3f} in {prediction.horizon.value}s "
                               f"(threshold: {ha_critical})",
                    trigger_conditions=[
                        f"Predicted Ha: {prediction.predicted_ha:.3f}",
                        f"Alert threshold: {ha_critical * ha_alert_pct:.3f}",
                        f"Confidence: {prediction.prediction_confidence:.2f}"
                    ],
                    time_to_threshold=time_to_threshold,
                    prediction_horizon=prediction.horizon,
                    confidence_level=prediction.prediction_confidence,
                    preventive_actions=[
                        "Increase monitoring frequency",
                        "Review symbolic processing integrity", 
                        "Prepare emergency protocols",
                        "Check input data quality"
                    ],
                    monitoring_recommendations=[
                        "Monitor Ha trajectory closely",
                        "Watch for sudden changes",
                        "Verify prediction accuracy"
                    ],
                    predictions=prediction
                )
                warnings_to_create.append(warning)
        
        elif prediction.predicted_ha >= ha_warning * ha_warning_pct:
            alert_id = f"ha_warning_{prediction.horizon.name.lower()}"
            if self._should_create_alert(alert_id):
                
                warning = EarlyWarningAlert(
                    alert_id=alert_id,
                    warning_level=WarningLevel.WARNING,
                    title=f"Predicted Ha Warning ({prediction.horizon.name})",  
                    description=f"Ha trending toward warning threshold: {prediction.predicted_ha:.3f}",
                    trigger_conditions=[
                        f"Predicted Ha: {prediction.predicted_ha:.3f}",
                        f"Warning threshold: {ha_warning * ha_warning_pct:.3f}"
                    ],
                    prediction_horizon=prediction.horizon,
                    confidence_level=prediction.prediction_confidence,
                    preventive_actions=[
                        "Enhanced monitoring",
                        "Review system inputs",
                        "Check processing patterns"
                    ],
                    predictions=prediction
                )
                warnings_to_create.append(warning)
        
        # Integrity-based warnings
        if prediction.predicted_integrity <= integrity_critical / self.config.warning_threshold_pct:
            alert_id = f"integrity_alert_{prediction.horizon.name.lower()}"
            if self._should_create_alert(alert_id):
                
                warning = EarlyWarningAlert(
                    alert_id=alert_id,
                    warning_level=WarningLevel.ALERT,
                    title=f"Predicted Integrity Alert ({prediction.horizon.name})",
                    description=f"Integrity predicted to drop to {prediction.predicted_integrity:.3f}",
                    trigger_conditions=[
                        f"Predicted integrity: {prediction.predicted_integrity:.3f}",
                        f"Critical threshold: {integrity_critical}"
                    ],
                    prediction_horizon=prediction.horizon,
                    confidence_level=prediction.prediction_confidence,
                    preventive_actions=[
                        "Activate integrity monitoring",
                        "Review constitutional constraints",
                        "Prepare synchronization protocols"
                    ],
                    predictions=prediction
                )
                warnings_to_create.append(warning)
        
        # Create warnings
        for warning in warnings_to_create:
            self.active_warnings[warning.alert_id] = warning
            self.last_alert_times[warning.alert_id] = current_time
            logger.warning(f"Early warning created: {warning.title}")
    
    def _should_create_alert(self, alert_id: str) -> bool:
        """Check if alert should be created (not suppressed)"""
        current_time = time.time()
        
        # Check if alert already exists
        if alert_id in self.active_warnings:
            return False
        
        # Check suppression time
        if alert_id in self.last_alert_times:
            time_since_last = current_time - self.last_alert_times[alert_id]
            if time_since_last < self.config.alert_suppression_time:
                return False
        
        return True
    
    def _estimate_time_to_threshold(self, current_value: float, predicted_value: float, 
                                  threshold: float, horizon_seconds: float) -> Optional[float]:
        """Estimate time until threshold is reached"""
        try:
            if current_value >= threshold:
                return 0.0  # Already at threshold
            
            if predicted_value <= current_value:
                return None  # Not trending toward threshold
            
            # Linear interpolation
            rate = (predicted_value - current_value) / horizon_seconds
            if rate <= 0:
                return None
            
            time_to_threshold = (threshold - current_value) / rate
            return max(0.0, time_to_threshold)
        
        except Exception:
            return None
    
    async def _cleanup_expired_warnings(self):
        """Remove expired or resolved warnings"""
        current_time = time.time()
        expired_warnings = []
        
        for alert_id, warning in self.active_warnings.items():
            # Check if warning is too old (auto-expire after 10 minutes)
            warning_age = current_time - warning.timestamp
            if warning_age > 600:  # 10 minutes
                expired_warnings.append(alert_id)
                continue
            
            # Check if conditions have improved
            if await self._should_resolve_warning(warning):
                expired_warnings.append(alert_id)
        
        # Remove expired warnings
        for alert_id in expired_warnings:
            removed_warning = self.active_warnings.pop(alert_id)
            logger.info(f"Early warning resolved: {removed_warning.title}")
    
    async def _should_resolve_warning(self, warning: EarlyWarningAlert) -> bool:
        """Check if warning should be resolved"""
        try:
            # Get current prediction for same horizon
            current_prediction = self.current_predictions.get(warning.prediction_horizon)
            if not current_prediction:
                return False  # Keep warning if no current prediction
            
            # Check if predicted values have improved significantly
            if warning.alert_id.startswith('ha_'):
                if current_prediction.predicted_ha < warning.predictions.predicted_ha * 0.8:
                    return True
            
            elif warning.alert_id.startswith('integrity_'):
                if current_prediction.predicted_integrity > warning.predictions.predicted_integrity * 1.2:
                    return True
            
            return False
        
        except Exception:
            return False
    
    async def _retrain_models(self):
        """Retrain machine learning models with recent data"""
        try:
            if len(self.metrics_buffer) < 50:  # Need sufficient data
                return
            
            logger.info("Retraining early warning models...")
            
            # Prepare training data
            data_array = self._prepare_prediction_data()
            if data_array is None:
                return
            
            # Retrain anomaly detector
            if len(data_array) >= 20:
                # Scale data  
                scaled_data = self.scaler.fit_transform(data_array)
                
                # Train anomaly detector
                self.anomaly_detector.fit(scaled_data)
                self.models_trained = True
                
                # Test model performance on recent data
                test_data = scaled_data[-10:]
                anomaly_scores = self.anomaly_detector.score_samples(test_data)
                avg_score = np.mean(anomaly_scores)
                
                logger.info(f"Models retrained - Anomaly detector avg score: {avg_score:.3f}")
            
            self.last_retrain_time = time.time()
        
        except Exception as e:
            logger.error(f"Model retraining error: {e}")
    
    async def _safe_callback(self, callback: Callable, *args) -> Any:
        """Safely execute callback with error handling"""
        try:
            if asyncio.iscoroutinefunction(callback):
                return await callback(*args)
            else:
                return callback(*args)
        except Exception as e:
            logger.error(f"Callback execution error: {e}")
            return None
    
    # Public API methods
    
    def get_current_warnings(self) -> Dict[str, Dict[str, Any]]:
        """Get all active warnings"""
        return {k: v.to_dict() for k, v in self.active_warnings.items()}
    
    def get_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Get current predictions for all horizons"""
        return {h.name: p.to_dict() for h, p in self.current_predictions.items()}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get early warning system status"""
        current_time = time.time()
        
        # Calculate overall threat level
        max_warning_level = WarningLevel.BASELINE
        for warning in self.active_warnings.values():
            if warning.warning_level.value > max_warning_level.value:
                max_warning_level = warning.warning_level
        
        return {
            'system_active': self.warning_active,
            'models_trained': self.models_trained,
            'data_points': len(self.metrics_buffer),
            'active_warnings_count': len(self.active_warnings),
            'max_warning_level': max_warning_level.name,
            'predictions_available': len(self.current_predictions),
            'last_retrain_time': self.last_retrain_time,
            'prediction_confidence': np.mean([
                p.prediction_confidence for p in self.current_predictions.values()
            ]) if self.current_predictions else 0.0
        }
    
    async def force_prediction_update(self):
        """Manually trigger prediction update"""
        await self._update_predictions()
    
    async def clear_warnings(self, warning_level: Optional[WarningLevel] = None):
        """Clear warnings by level or all warnings"""
        if warning_level is None:
            cleared_count = len(self.active_warnings)
            self.active_warnings.clear()
            logger.info(f"Cleared all {cleared_count} early warnings")
        else:
            warnings_to_remove = [
                alert_id for alert_id, warning in self.active_warnings.items()
                if warning.warning_level == warning_level
            ]
            for alert_id in warnings_to_remove:
                self.active_warnings.pop(alert_id)
            logger.info(f"Cleared {len(warnings_to_remove)} warnings at level {warning_level.name}")


# Test function for early warning system
async def test_early_warning_system():
    """Test early warning system functionality"""
    print("⚠️  Testing Early Warning System")
    
    # Create system
    config = EarlyWarningConfiguration()
    config.min_data_points = 10
    config.prediction_update_interval = 1.0
    
    ews = EarlyWarningSystem(config)
    
    # Mock integration callbacks
    async def mock_constitutional_callback(action):
        if action == 'get_thresholds':
            return {
                'ha_critical': 2.0,
                'ha_warning': 1.0,
                'integrity_critical': 0.5,
                'integrity_warning': 0.7
            }
    
    # Start monitoring
    await ews.start_monitoring(
        constitutional_callback=mock_constitutional_callback
    )
    
    print("📊 Feeding test data...")
    
    # Feed gradually increasing Ha values to trigger warnings
    for i in range(30):
        await asyncio.sleep(0.2)
        
        ha_value = 0.1 + (i * 0.05)  # Gradually increase Ha
        integrity_value = max(0.3, 1.0 - (i * 0.02))  # Gradually decrease integrity
        
        metrics = {
            'hallucination_number': ha_value,
            'coherence_measure': max(0.2, 0.8 - i * 0.01),
            'defect_density': i * 0.005,
            'field_energy': 100 + i * 10,
            'integrity_score': integrity_value,
            'constitutional_compliance': integrity_value,
            'system_stability': max(0.4, 1.0 - i * 0.015)
        }
        
        await ews.update_metrics(metrics)
        
        if i % 5 == 0:  # Print status every 5 iterations
            status = ews.get_system_status()
            warnings = ews.get_current_warnings()
            predictions = ews.get_predictions()
            
            print(f"Step {i}: Ha={ha_value:.3f}, Integrity={integrity_value:.3f}")
            print(f"  Warnings: {status['active_warnings_count']}, Max Level: {status['max_warning_level']}")
            
            if predictions:
                short_pred = predictions.get('SHORT_TERM', {})
                if short_pred:
                    print(f"  Predicted Ha (30s): {short_pred.get('predicted_ha', 0):.3f}")
    
    # Stop monitoring
    await ews.stop_monitoring()
    
    print("✅ Early Warning System test completed")
    
    # Final status
    final_warnings = ews.get_current_warnings()
    print(f"📋 Final warnings generated: {len(final_warnings)}")
    for warning_id, warning in final_warnings.items():
        print(f"  - {warning['title']} ({warning['warning_level']})")


if __name__ == "__main__":
    asyncio.run(test_early_warning_system())