"""
Enhanced Risk Metrics and Coherence Computation v1.5

Implements advanced constitutional metrics with:
- Multi-scale risk assessment and topological analysis
- Constitutional safety metrics and compliance monitoring
- Adaptive threshold detection and emergency protocols
- Entropy-based diversity and stability measures
- Real-time anomaly detection and prediction
- Temporal coherence tracking and trend analysis
"""

import numpy as np
from typing import Tuple, Dict, Optional, List, Any, Union
from scipy import ndimage, signal
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from skimage.restoration import unwrap_phase
from sklearn.cluster import DBSCAN
import warnings
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque

from .state import SystemState, RiskMetrics, CostFunctionalConfig
from .metrics import MetricsCalculator

# Configure logging
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications."""
    SAFE = "safe"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics for categorization."""
    CONSTITUTIONAL = "constitutional"
    TOPOLOGICAL = "topological"
    COHERENCE = "coherence"
    ENERGY = "energy"
    STABILITY = "stability"
    DIVERSITY = "diversity"


@dataclass
class ConstitutionalLimits:
    """Constitutional limits for safe operation."""
    max_hallucination_number: float = 0.5
    max_defect_density: float = 0.1
    min_global_coherence: float = 0.3
    min_modular_coherence: float = 0.4
    max_systemic_risk: float = 0.8
    max_energy_growth_rate: float = 0.2
    min_diversity_index: float = 0.2
    

@dataclass
class AdvancedMetrics:
    """Advanced metrics for constitutional assessment."""
    # Constitutional compliance
    constitutional_score: float = 0.0
    safety_margin: float = 0.0
    compliance_trend: float = 0.0
    
    # Multi-scale analysis
    defect_hierarchy: Dict[str, float] = field(default_factory=dict)
    coherence_scales: List[float] = field(default_factory=list)
    energy_spectrum: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Stability metrics
    lyapunov_estimate: float = 0.0
    stability_index: float = 0.0
    perturbation_response: float = 0.0
    
    # Diversity and entropy
    field_entropy: float = 0.0
    phase_diversity: float = 0.0
    spatial_complexity: float = 0.0
    
    # Temporal analysis
    coherence_velocity: float = 0.0
    defect_generation_rate: float = 0.0
    energy_dissipation_rate: float = 0.0
    
    # Emergency indicators
    anomaly_score: float = 0.0
    instability_warning: bool = False
    constitutional_violation: bool = False


class EnhancedMetricsCalculator(MetricsCalculator):
    """
    Enhanced Metrics Calculator v1.5 - Constitutional system monitoring.
    
    Implements advanced multi-scale risk assessment with:
    - Constitutional safety monitoring and compliance tracking
    - Multi-scale topological defect analysis and classification
    - Entropy-based diversity measures and stability indicators
    - Real-time anomaly detection with predictive warnings
    - Temporal coherence analysis with trend prediction
    - Emergency protocol triggers and safety margin assessment
    """
    
    def __init__(self, 
                 config: CostFunctionalConfig,
                 constitutional_limits: Optional[ConstitutionalLimits] = None,
                 enable_advanced_analysis: bool = True,
                 history_length: int = 500):
        """
        Initialize enhanced metrics calculator with constitutional monitoring.
        
        Args:
            config: Configuration with weights for different cost terms
            constitutional_limits: Safety limits and thresholds
            enable_advanced_analysis: Enable advanced multi-scale analysis
            history_length: Length of metric history to maintain
        """
        super().__init__(config)
        
        self.constitutional_limits = constitutional_limits or ConstitutionalLimits()
        self.enable_advanced_analysis = enable_advanced_analysis
        
        # Metric history for temporal analysis
        self.history_length = history_length
        self.metric_history = deque(maxlen=history_length)
        self.risk_history = deque(maxlen=history_length)
        self.coherence_history = deque(maxlen=history_length)
        self.energy_history = deque(maxlen=history_length)
        
        # Advanced analysis state
        self.baseline_metrics = None
        self.anomaly_detector_state = {}
        self.trend_analyzer_state = {}
        
        # Constitutional monitoring
        self.constitutional_violations = []
        self.emergency_triggers = []
        self.safety_warnings = []
        
        # Performance tracking
        self.computation_stats = {
            'total_calculations': 0,
            'constitutional_checks': 0,
            'anomalies_detected': 0,
            'emergency_activations': 0,
            'average_computation_time': 0.0
        }
        
        logger.info(f"Enhanced Metrics Calculator v1.5 initialized")
        logger.info(f"Advanced analysis: {enable_advanced_analysis}")
        logger.info(f"History length: {history_length}")
        
    def calculate_constitutional_score(self, 
                                     hallucination_number: float,
                                     defect_density: float,
                                     coherence_global: float,
                                     coherence_modular: float,
                                     energy_growth_rate: float = 0.0) -> Tuple[float, bool]:
        """
        Calculate overall constitutional compliance score.
        
        Args:
            hallucination_number: Current hallucination number
            defect_density: Average defect density
            coherence_global: Global coherence value
            coherence_modular: Modular coherence value
            energy_growth_rate: Rate of energy growth
            
        Returns:
            Tuple of (constitutional_score, violation_detected)
        """
        violations = []
        scores = []
        
        # Check hallucination limit
        if hallucination_number <= self.constitutional_limits.max_hallucination_number:
            h_score = 1.0 - (hallucination_number / self.constitutional_limits.max_hallucination_number)
        else:
            h_score = 0.0
            violations.append(f"Hallucination number exceeded: {hallucination_number:.3f}")
        scores.append(h_score)
        
        # Check defect density limit
        if defect_density <= self.constitutional_limits.max_defect_density:
            d_score = 1.0 - (defect_density / self.constitutional_limits.max_defect_density)
        else:
            d_score = 0.0
            violations.append(f"Defect density exceeded: {defect_density:.3f}")
        scores.append(d_score)
        
        # Check coherence minimums
        if coherence_global >= self.constitutional_limits.min_global_coherence:
            gc_score = coherence_global
        else:
            gc_score = 0.0
            violations.append(f"Global coherence below limit: {coherence_global:.3f}")
        scores.append(gc_score)
        
        if coherence_modular >= self.constitutional_limits.min_modular_coherence:
            mc_score = coherence_modular
        else:
            mc_score = 0.0
            violations.append(f"Modular coherence below limit: {coherence_modular:.3f}")
        scores.append(mc_score)
        
        # Check energy growth rate
        if abs(energy_growth_rate) <= self.constitutional_limits.max_energy_growth_rate:
            e_score = 1.0 - abs(energy_growth_rate) / self.constitutional_limits.max_energy_growth_rate
        else:
            e_score = 0.0
            violations.append(f"Energy growth rate exceeded: {energy_growth_rate:.3f}")
        scores.append(e_score)
        
        # Overall constitutional score (weighted average)
        constitutional_score = np.mean(scores)
        violation_detected = len(violations) > 0
        
        if violations:
            self.constitutional_violations.extend(violations)
            self.computation_stats['constitutional_checks'] += 1
            logger.warning(f"Constitutional violations: {violations}")
        
        return constitutional_score, violation_detected
        
    def calculate_multi_scale_defects(self, field: np.ndarray) -> Dict[str, Any]:
        """
        Perform multi-scale analysis of topological defects.
        
        Args:
            field: Complex neural field
            
        Returns:
            Dictionary with multi-scale defect analysis
        """
        scales = [1, 2, 4, 8]  # Different analysis scales
        scale_results = {}
        
        for scale in scales:
            # Downsample field for different scales
            if scale > 1:
                downsampled = field[::scale, ::scale]
            else:
                downsampled = field
            
            # Calculate defect density at this scale
            rho_def = self.calculate_defect_density(downsampled)
            
            # Analyze defect clustering
            defect_map = rho_def > 0.01
            labeled_defects, num_defects = ndimage.label(defect_map)
            
            # Calculate defect statistics
            defect_sizes = []
            defect_strengths = []
            
            for i in range(1, num_defects + 1):
                defect_region = (labeled_defects == i)
                size = np.sum(defect_region)
                strength = np.mean(rho_def[defect_region])
                
                defect_sizes.append(size)
                defect_strengths.append(strength)
            
            scale_results[f'scale_{scale}'] = {
                'defect_count': num_defects,
                'mean_defect_size': np.mean(defect_sizes) if defect_sizes else 0.0,
                'mean_defect_strength': np.mean(defect_strengths) if defect_strengths else 0.0,
                'defect_density_mean': np.mean(rho_def),
                'defect_density_std': np.std(rho_def),
                'spatial_correlation': self._calculate_spatial_correlation(rho_def)
            }
        
        return scale_results
        
    def _calculate_spatial_correlation(self, field: np.ndarray) -> float:
        """
        Calculate spatial correlation length of a field.
        
        Args:
            field: 2D field to analyze
            
        Returns:
            Characteristic correlation length
        """
        # Calculate 2D autocorrelation using FFT
        field_centered = field - np.mean(field)
        correlation = signal.correlate2d(field_centered, field_centered, mode='same')
        correlation = correlation / np.max(correlation)  # Normalize
        
        # Find correlation length as distance where correlation drops to 1/e
        center_y, center_x = np.array(correlation.shape) // 2
        
        # Radial profile
        y, x = np.ogrid[:correlation.shape[0], :correlation.shape[1]]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Calculate radial average
        r_max = min(center_x, center_y)
        radial_profile = []
        for i in range(1, r_max):
            mask = (r >= i - 0.5) & (r < i + 0.5)
            if np.any(mask):
                radial_profile.append(np.mean(correlation[mask]))
        
        # Find where correlation drops to 1/e ≈ 0.368
        correlation_length = 1.0  # Default
        for i, corr_val in enumerate(radial_profile):
            if corr_val < 0.368:
                correlation_length = float(i + 1)
                break
        
        return correlation_length
        
    def calculate_field_entropy(self, field: np.ndarray, bins: int = 50) -> float:
        """
        Calculate entropy of field amplitude distribution.
        
        Args:
            field: Complex neural field
            bins: Number of histogram bins
            
        Returns:
            Field entropy value
        """
        # Get field amplitude
        amplitude = np.abs(field).flatten()
        
        # Calculate histogram
        hist, _ = np.histogram(amplitude, bins=bins, density=True)
        
        # Add small value to avoid log(0)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)  # Ensure normalization
        
        # Calculate entropy
        field_entropy = entropy(hist, base=2)
        
        return float(field_entropy)
        
    def calculate_phase_diversity(self, phases: np.ndarray) -> float:
        """
        Calculate diversity of module phases.
        
        Args:
            phases: Array of module phases
            
        Returns:
            Phase diversity index
        """
        if len(phases) <= 1:
            return 0.0
        
        # Convert phases to unit vectors
        unit_vectors = np.exp(1j * phases)
        
        # Calculate pairwise distances on unit circle
        distances = []
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                # Angular distance on unit circle
                angle_diff = np.angle(unit_vectors[j] / unit_vectors[i])
                angle_diff = np.mod(angle_diff + np.pi, 2*np.pi) - np.pi
                distances.append(abs(angle_diff))
        
        # Diversity is mean pairwise distance normalized by maximum
        if distances:
            diversity = np.mean(distances) / np.pi  # Normalize to [0,1]
        else:
            diversity = 0.0
        
        return float(diversity)
        
    def calculate_stability_index(self, 
                                field_history: List[np.ndarray],
                                phase_history: List[np.ndarray]) -> float:
        """
        Calculate system stability index from temporal evolution.
        
        Args:
            field_history: List of recent field states
            phase_history: List of recent phase states
            
        Returns:
            Stability index [0,1] where 1 is most stable
        """
        if len(field_history) < 3 or len(phase_history) < 3:
            return 0.5  # Neutral stability with insufficient data
        
        # Field stability - variation in field energy
        field_energies = [np.mean(np.abs(f)**2) for f in field_history]
        field_stability = 1.0 / (1.0 + np.std(field_energies))
        
        # Phase stability - variation in order parameter
        order_params = []
        for phases in phase_history:
            if len(phases) > 0:
                R = np.abs(np.mean(np.exp(1j * phases)))
                order_params.append(R)
        
        if order_params:
            phase_stability = 1.0 / (1.0 + np.std(order_params))
        else:
            phase_stability = 0.5
        
        # Combined stability index
        stability_index = 0.6 * field_stability + 0.4 * phase_stability
        
        return float(np.clip(stability_index, 0.0, 1.0))
        
    def detect_anomalies(self, current_metrics: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Detect anomalies in current metrics using statistical methods.
        
        Args:
            current_metrics: Dictionary of current metric values
            
        Returns:
            Tuple of (anomaly_score, anomaly_descriptions)
        """
        if len(self.metric_history) < 10:
            return 0.0, []  # Need more history for anomaly detection
        
        anomalies = []
        anomaly_scores = []
        
        # Statistical anomaly detection for each metric
        for metric_name, current_value in current_metrics.items():
            if metric_name in ['timestamp']:  # Skip non-numeric metrics
                continue
                
            # Extract historical values
            historical = []
            for hist_metrics in list(self.metric_history)[-50:]:  # Last 50 entries
                if hasattr(hist_metrics, metric_name):
                    historical.append(getattr(hist_metrics, metric_name))
                elif isinstance(hist_metrics, dict) and metric_name in hist_metrics:
                    historical.append(hist_metrics[metric_name])
            
            if len(historical) < 5:
                continue
            
            # Z-score based anomaly detection
            mean_val = np.mean(historical)
            std_val = np.std(historical)
            
            if std_val > 1e-6:  # Avoid division by zero
                z_score = abs(current_value - mean_val) / std_val
                
                if z_score > 3.0:  # 3-sigma rule
                    anomalies.append(f"{metric_name}: {current_value:.3f} (z={z_score:.2f})")
                    anomaly_scores.append(min(1.0, z_score / 5.0))  # Normalize
        
        # Overall anomaly score
        overall_anomaly_score = np.mean(anomaly_scores) if anomaly_scores else 0.0
        
        if anomalies:
            self.computation_stats['anomalies_detected'] += len(anomalies)
            logger.warning(f"Anomalies detected: {anomalies}")
        
        return float(overall_anomaly_score), anomalies
        
    def predict_trend(self, metric_name: str, window_size: int = 20) -> float:
        """
        Predict trend for a specific metric using linear regression.
        
        Args:
            metric_name: Name of the metric to analyze
            window_size: Window size for trend analysis
            
        Returns:
            Trend slope (positive = increasing, negative = decreasing)
        """
        if len(self.metric_history) < window_size:
            return 0.0
        
        # Extract recent values
        values = []
        for hist_metrics in list(self.metric_history)[-window_size:]:
            if hasattr(hist_metrics, metric_name):
                values.append(getattr(hist_metrics, metric_name))
            elif isinstance(hist_metrics, dict) and metric_name in hist_metrics:
                values.append(hist_metrics[metric_name])
        
        if len(values) < 3:
            return 0.0
        
        # Simple linear regression
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        trend_slope = coeffs[0]  # Slope of linear fit
        
        return float(trend_slope)
        
    def assess_emergency_conditions(self, 
                                  current_metrics: RiskMetrics,
                                  advanced_metrics: AdvancedMetrics) -> Tuple[bool, List[str]]:
        """
        Assess if emergency conditions are met requiring immediate intervention.
        
        Args:
            current_metrics: Current risk metrics
            advanced_metrics: Advanced analysis results
            
        Returns:
            Tuple of (emergency_detected, emergency_reasons)
        """
        emergency_conditions = []
        
        # Critical systemic risk
        if current_metrics.systemic_risk > 0.95:
            emergency_conditions.append(f"Critical systemic risk: {current_metrics.systemic_risk:.3f}")
        
        # Extremely high hallucination number
        if current_metrics.hallucination_number > 2 * self.constitutional_limits.max_hallucination_number:
            emergency_conditions.append(f"Extreme hallucination: {current_metrics.hallucination_number:.3f}")
        
        # Coherence collapse
        if (current_metrics.coherence_global < 0.1 and 
            current_metrics.coherence_modular < 0.1):
            emergency_conditions.append("Coherence collapse detected")
        
        # High anomaly score
        if advanced_metrics.anomaly_score > 0.9:
            emergency_conditions.append(f"High anomaly score: {advanced_metrics.anomaly_score:.3f}")
        
        # Instability warning
        if advanced_metrics.instability_warning:
            emergency_conditions.append("System instability detected")
        
        # Energy runaway
        if abs(advanced_metrics.energy_dissipation_rate) > 5 * self.constitutional_limits.max_energy_growth_rate:
            emergency_conditions.append(f"Energy runaway: {advanced_metrics.energy_dissipation_rate:.3f}")
        
        emergency_detected = len(emergency_conditions) > 0
        
        if emergency_detected:
            self.emergency_triggers.extend(emergency_conditions)
            self.computation_stats['emergency_activations'] += 1
            logger.critical(f"EMERGENCY CONDITIONS: {emergency_conditions}")
        
        return emergency_detected, emergency_conditions

    def calculate_all_metrics(self, state: SystemState) -> Tuple[RiskMetrics, AdvancedMetrics]:
        """
        Enhanced method to compute all metrics with constitutional analysis.
        
        Args:
            state: Current system state
            
        Returns:
            Tuple of (RiskMetrics, AdvancedMetrics) with comprehensive analysis
        """
        start_time = time.time()
        
        # Calculate base metrics using parent class
        base_metrics = super().calculate_all_metrics(state)
        
        # Advanced metrics (if enabled)
        advanced_metrics = AdvancedMetrics()
        
        if self.enable_advanced_analysis:
            # Constitutional analysis
            energy_growth_rate = 0.0
            if len(self.energy_history) >= 2:
                dt = max(0.01, state.simulation_time - getattr(list(self.metric_history)[-1], 'timestamp', state.simulation_time - 0.01))
                energy_growth_rate = (base_metrics.field_energy - self.energy_history[-1]) / dt
            
            constitutional_score, violation = self.calculate_constitutional_score(
                base_metrics.hallucination_number, 
                base_metrics.rho_def_mean, 
                base_metrics.coherence_global, 
                base_metrics.coherence_modular, 
                energy_growth_rate
            )
            advanced_metrics.constitutional_score = constitutional_score
            advanced_metrics.constitutional_violation = violation
            
            # Multi-scale defect analysis
            advanced_metrics.defect_hierarchy = self.calculate_multi_scale_defects(state.neural_field)
            
            # Entropy and diversity measures
            advanced_metrics.field_entropy = self.calculate_field_entropy(state.neural_field)
            advanced_metrics.phase_diversity = self.calculate_phase_diversity(state.module_phases)
            
            # Stability analysis
            recent_fields = [getattr(h, 'neural_field', state.neural_field) for h in list(self.metric_history)[-10:] if hasattr(h, 'neural_field')]
            recent_phases = [getattr(h, 'module_phases', state.module_phases) for h in list(self.metric_history)[-10:] if hasattr(h, 'module_phases')]
            
            if not recent_fields:
                recent_fields = [state.neural_field]
            if not recent_phases:
                recent_phases = [state.module_phases]
            
            advanced_metrics.stability_index = self.calculate_stability_index(recent_fields, recent_phases)
            
            # Temporal analysis
            advanced_metrics.coherence_velocity = self.predict_trend('coherence_global')
            advanced_metrics.defect_generation_rate = self.predict_trend('rho_def_mean')
            advanced_metrics.energy_dissipation_rate = energy_growth_rate
            
            # Anomaly detection
            current_metric_dict = {
                'hallucination_number': base_metrics.hallucination_number,
                'rho_def_mean': base_metrics.rho_def_mean,
                'coherence_global': base_metrics.coherence_global,
                'coherence_modular': base_metrics.coherence_modular,
                'systemic_risk': base_metrics.systemic_risk,
                'field_energy': base_metrics.field_energy
            }
            
            anomaly_score, anomaly_list = self.detect_anomalies(current_metric_dict)
            advanced_metrics.anomaly_score = anomaly_score
            
            # Safety margin calculation
            safety_margins = [
                1.0 - (base_metrics.hallucination_number / self.constitutional_limits.max_hallucination_number),
                1.0 - (base_metrics.rho_def_mean / self.constitutional_limits.max_defect_density),
                base_metrics.coherence_global / self.constitutional_limits.min_global_coherence - 1.0,
                base_metrics.coherence_modular / self.constitutional_limits.min_modular_coherence - 1.0
            ]
            advanced_metrics.safety_margin = min(safety_margins)
            
            # Instability warning
            advanced_metrics.instability_warning = (
                advanced_metrics.stability_index < 0.3 or
                anomaly_score > 0.8 or
                abs(energy_growth_rate) > 2 * self.constitutional_limits.max_energy_growth_rate
            )
            
            # Emergency assessment
            emergency_detected, emergency_reasons = self.assess_emergency_conditions(base_metrics, advanced_metrics)
            
            if emergency_detected:
                logger.critical(f"EMERGENCY CONDITIONS DETECTED: {emergency_reasons}")
        
        # Update histories
        self.metric_history.append(base_metrics)
        self.energy_history.append(base_metrics.field_energy)
        self.coherence_history.append(base_metrics.coherence_global)
        self.risk_history.append(base_metrics.systemic_risk)
        
        # Update computation statistics
        computation_time = time.time() - start_time
        self.computation_stats['total_calculations'] += 1
        old_avg = self.computation_stats['average_computation_time']
        self.computation_stats['average_computation_time'] = 0.9 * old_avg + 0.1 * computation_time
        
        return base_metrics, advanced_metrics

    def get_constitutional_status(self) -> Dict[str, Any]:
        """
        Get comprehensive constitutional status report.
        
        Returns:
            Dictionary with constitutional status information
        """
        return {
            'constitutional_limits': {
                'max_hallucination_number': self.constitutional_limits.max_hallucination_number,
                'max_defect_density': self.constitutional_limits.max_defect_density,
                'min_global_coherence': self.constitutional_limits.min_global_coherence,
                'min_modular_coherence': self.constitutional_limits.min_modular_coherence,
                'max_systemic_risk': self.constitutional_limits.max_systemic_risk
            },
            'violation_history': {
                'total_violations': len(self.constitutional_violations),
                'recent_violations': self.constitutional_violations[-10:],
                'emergency_triggers': len(self.emergency_triggers),
                'safety_warnings': len(self.safety_warnings)
            },
            'computation_statistics': self.computation_stats.copy(),
            'system_status': {
                'advanced_analysis_enabled': self.enable_advanced_analysis,
                'history_length': len(self.metric_history),
                'baseline_established': self.baseline_metrics is not None
            }
        }

    def generate_safety_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive safety and performance report.
        
        Returns:
            Detailed safety analysis report
        """
        if not self.metric_history:
            return {'status': 'insufficient_data'}
        
        recent_metrics = list(self.metric_history)[-50:]  # Last 50 measurements
        
        # Calculate statistics
        recent_risks = [m.systemic_risk for m in recent_metrics if hasattr(m, 'systemic_risk')]
        recent_coherence = [m.coherence_global for m in recent_metrics if hasattr(m, 'coherence_global')]
        recent_hallucinations = [m.hallucination_number for m in recent_metrics if hasattr(m, 'hallucination_number')]
        
        return {
            'status': 'active',
            'timestamp': time.time(),
            'risk_analysis': {
                'mean_risk': np.mean(recent_risks) if recent_risks else 0.0,
                'max_risk': np.max(recent_risks) if recent_risks else 0.0,
                'risk_trend': self.predict_trend('systemic_risk'),
                'risk_violations': len([r for r in recent_risks if r > self.constitutional_limits.max_systemic_risk])
            },
            'coherence_analysis': {
                'mean_coherence': np.mean(recent_coherence) if recent_coherence else 0.0,
                'min_coherence': np.min(recent_coherence) if recent_coherence else 0.0,
                'coherence_trend': self.predict_trend('coherence_global'),
                'coherence_violations': len([c for c in recent_coherence if c < self.constitutional_limits.min_global_coherence])
            },
            'hallucination_analysis': {
                'mean_hallucinations': np.mean(recent_hallucinations) if recent_hallucinations else 0.0,
                'max_hallucinations': np.max(recent_hallucinations) if recent_hallucinations else 0.0,
                'hallucination_trend': self.predict_trend('hallucination_number'),
                'hallucination_violations': len([h for h in recent_hallucinations if h > self.constitutional_limits.max_hallucination_number])
            },
            'constitutional_status': {
                'total_violations': len(self.constitutional_violations),
                'emergency_activations': self.computation_stats['emergency_activations'],
                'anomalies_detected': self.computation_stats['anomalies_detected'],
                'overall_compliance': self._calculate_overall_compliance()
            },
            'recommendations': self._generate_safety_recommendations()
        }
    
    def _calculate_overall_compliance(self) -> float:
        """
        Calculate overall constitutional compliance score.
        
        Returns:
            Compliance score [0,1] where 1 is perfect compliance
        """
        if not self.metric_history:
            return 0.5
        
        recent_metrics = list(self.metric_history)[-20:]  # Last 20 measurements
        compliance_scores = []
        
        for metrics in recent_metrics:
            if hasattr(metrics, 'hallucination_number') and hasattr(metrics, 'systemic_risk'):
                # Individual compliance checks
                h_compliance = 1.0 if metrics.hallucination_number <= self.constitutional_limits.max_hallucination_number else 0.0
                r_compliance = 1.0 if metrics.systemic_risk <= self.constitutional_limits.max_systemic_risk else 0.0
                c_compliance = 1.0 if metrics.coherence_global >= self.constitutional_limits.min_global_coherence else 0.0
                
                overall = (h_compliance + r_compliance + c_compliance) / 3.0
                compliance_scores.append(overall)
        
        return np.mean(compliance_scores) if compliance_scores else 0.5
    
    def _generate_safety_recommendations(self) -> List[str]:
        """
        Generate safety recommendations based on current system state.
        
        Returns:
            List of safety recommendations
        """
        recommendations = []
        
        # Check violation rates
        if len(self.constitutional_violations) > 10:
            recommendations.append("High violation rate detected - review constitutional parameters")
        
        # Check emergency activations
        if self.computation_stats['emergency_activations'] > 3:
            recommendations.append("Multiple emergency activations - system may need fundamental review")
        
        # Check anomaly rate
        if self.computation_stats['anomalies_detected'] > 20:
            recommendations.append("High anomaly detection rate - investigate system stability")
        
        # Check trend predictions
        risk_trend = self.predict_trend('systemic_risk')
        if risk_trend > 0.1:
            recommendations.append(f"Rising risk trend detected ({risk_trend:.3f}) - consider preventive measures")
        
        coherence_trend = self.predict_trend('coherence_global')
        if coherence_trend < -0.05:
            recommendations.append(f"Declining coherence trend ({coherence_trend:.3f}) - review coupling parameters")
        
        # Check computation performance
        if self.computation_stats['average_computation_time'] > 1.0:
            recommendations.append(f"High computation time ({self.computation_stats['average_computation_time']:.3f}s) - consider optimization")
        
        if not recommendations:
            recommendations.append("System operating within acceptable constitutional parameters")
        
        return recommendations