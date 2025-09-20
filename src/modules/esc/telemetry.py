"""
ESC Module 2.1 Telemetry System

Enhanced telemetry for interpretability and monitoring of Echo-Semantic Converter operations.
Provides comprehensive metrics, semantic anchor stability monitoring, and real-time insights.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import statistics
from datetime import datetime, timedelta


@dataclass
class SemanticAnchorMetrics:
    """Metrics for semantic anchor stability."""
    anchor_id: str
    stability_score: float
    drift_rate: float
    coherence_index: float
    activation_frequency: float
    last_activated: float
    anchor_strength: float


@dataclass
class ProcessingTelemetry:
    """Telemetry data for a single processing session."""
    session_id: str
    timestamp: float
    processing_time: float
    token_count: int
    semantic_field_activation: Dict[str, float]
    constitutional_scores: Dict[str, float]
    attention_patterns: Dict[str, Any]
    anchor_metrics: List[SemanticAnchorMetrics]
    resource_usage: Dict[str, float]
    warnings: List[str]


class ESCTelemetryCollector:
    """
    Comprehensive telemetry collection system for ESC Module 2.1.
    
    Features:
    - Real-time semantic anchor monitoring
    - Processing performance metrics
    - Constitutional compliance tracking
    - Attention pattern analysis
    - Resource utilization monitoring
    """
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.session_history = deque(maxlen=buffer_size)
        self.semantic_anchors = {}
        self.performance_metrics = defaultdict(list)
        self.constitutional_history = deque(maxlen=500)
        self.attention_patterns = defaultdict(list)
        
        # Real-time monitoring state
        self.current_session_id = None
        self.session_start_time = None
        self.active_anchors = set()
        
        # Aggregated statistics
        self.total_tokens_processed = 0
        self.total_sessions = 0
        self.average_processing_time = 0.0
        self.constitutional_violation_rate = 0.0
        
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new telemetry session."""
        if session_id is None:
            session_id = f"esc_session_{int(time.time())}_{self.total_sessions}"
        
        self.current_session_id = session_id
        self.session_start_time = time.time()
        self.total_sessions += 1
        
        return session_id
    
    def track_semantic_anchor(self, anchor_id: str, embedding_vector: List[float], 
                            activation_strength: float = 1.0):
        """Track semantic anchor stability and activity."""
        current_time = time.time()
        
        if anchor_id not in self.semantic_anchors:
            self.semantic_anchors[anchor_id] = {
                'creation_time': current_time,
                'activation_history': deque(maxlen=100),
                'embedding_history': deque(maxlen=50),
                'drift_measurements': deque(maxlen=20)
            }
        
        anchor_data = self.semantic_anchors[anchor_id]
        anchor_data['activation_history'].append((current_time, activation_strength))
        anchor_data['embedding_history'].append(embedding_vector)
        
        # Calculate drift if we have previous embeddings
        if len(anchor_data['embedding_history']) > 1:
            drift = self._calculate_embedding_drift(
                anchor_data['embedding_history'][-2],
                anchor_data['embedding_history'][-1]
            )
            anchor_data['drift_measurements'].append(drift)
        
        self.active_anchors.add(anchor_id)
    
    def record_processing_metrics(self, token_count: int, processing_time: float,
                                semantic_field_state: Dict[str, float],
                                constitutional_scores: Dict[str, float],
                                attention_weights: Dict[str, Any]):
        """Record metrics for current processing operation."""
        if not self.current_session_id:
            self.start_session()
        
        # Update running statistics
        self.total_tokens_processed += token_count
        self.performance_metrics['processing_time'].append(processing_time)
        self.performance_metrics['token_count'].append(token_count)
        
        # Track constitutional compliance
        overall_score = constitutional_scores.get('overall_compliance', 1.0)
        self.constitutional_history.append(overall_score)
        
        # Store attention patterns for analysis
        attention_entropy = self._calculate_attention_entropy(attention_weights)
        self.attention_patterns['entropy'].append(attention_entropy)
        
        # Update averages
        recent_times = list(self.performance_metrics['processing_time'])[-10:]
        self.average_processing_time = statistics.mean(recent_times) if recent_times else 0.0
        
        # Calculate violation rate
        recent_scores = list(self.constitutional_history)[-100:]
        violations = sum(1 for score in recent_scores if score < 0.8)
        self.constitutional_violation_rate = violations / len(recent_scores) if recent_scores else 0.0
    
    def end_session(self, warnings: Optional[List[str]] = None) -> ProcessingTelemetry:
        """End current session and generate telemetry report."""
        if not self.current_session_id or not self.session_start_time:
            raise ValueError("No active session to end")
        
        session_duration = time.time() - self.session_start_time
        
        # Generate anchor metrics for active anchors
        anchor_metrics = []
        for anchor_id in self.active_anchors:
            metrics = self._generate_anchor_metrics(anchor_id)
            if metrics:
                anchor_metrics.append(metrics)
        
        # Create telemetry record
        telemetry = ProcessingTelemetry(
            session_id=self.current_session_id,
            timestamp=self.session_start_time,
            processing_time=session_duration,
            token_count=self.performance_metrics['token_count'][-1] if self.performance_metrics['token_count'] else 0,
            semantic_field_activation=self._get_latest_field_state(),
            constitutional_scores=self._get_latest_constitutional_scores(),
            attention_patterns=self._get_attention_summary(),
            anchor_metrics=anchor_metrics,
            resource_usage=self._measure_resource_usage(),
            warnings=warnings or []
        )
        
        # Store in history
        self.session_history.append(telemetry)
        
        # Reset session state
        self.current_session_id = None
        self.session_start_time = None
        self.active_anchors.clear()
        
        return telemetry
    
    def get_interpretability_report(self) -> Dict[str, Any]:
        """Generate comprehensive interpretability report."""
        return {
            'system_overview': {
                'total_sessions': self.total_sessions,
                'total_tokens_processed': self.total_tokens_processed,
                'average_processing_time': self.average_processing_time,
                'constitutional_violation_rate': self.constitutional_violation_rate,
                'active_semantic_anchors': len(self.semantic_anchors)
            },
            'semantic_anchor_stability': self._analyze_anchor_stability(),
            'performance_trends': self._analyze_performance_trends(),
            'constitutional_compliance': self._analyze_constitutional_trends(),
            'attention_patterns': self._analyze_attention_patterns(),
            'recommendations': self._generate_recommendations()
        }
    
    def export_telemetry_data(self, format: str = 'json') -> str:
        """Export telemetry data in specified format."""
        data = {
            'metadata': {
                'export_timestamp': time.time(),
                'system_version': 'ESC_2.1',
                'buffer_size': self.buffer_size,
                'total_sessions': len(self.session_history)
            },
            'sessions': [asdict(session) for session in self.session_history],
            'semantic_anchors': self._export_anchor_data(),
            'summary_statistics': self.get_interpretability_report()
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _calculate_embedding_drift(self, embedding1: List[float], 
                                 embedding2: List[float]) -> float:
        """Calculate drift between two embedding vectors."""
        if len(embedding1) != len(embedding2):
            return 1.0  # Maximum drift for mismatched dimensions
        
        # Simple cosine distance approximation
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 1.0
        
        cosine_sim = dot_product / (magnitude1 * magnitude2)
        return 1.0 - max(0.0, cosine_sim)  # Convert to distance
    
    def _calculate_attention_entropy(self, attention_weights: Dict[str, Any]) -> float:
        """Calculate entropy of attention distribution."""
        weights = []
        for key, value in attention_weights.items():
            if isinstance(value, (int, float)):
                weights.append(value)
            elif isinstance(value, list):
                weights.extend(value)
        
        if not weights:
            return 0.0
        
        # Normalize weights
        total = sum(weights)
        if total == 0:
            return 0.0
        
        normalized = [w / total for w in weights]
        
        # Calculate entropy
        entropy = 0.0
        for p in normalized:
            if p > 0:
                entropy -= p * (p ** 0.5)  # Simplified entropy calculation
        
        return entropy
    
    def _generate_anchor_metrics(self, anchor_id: str) -> Optional[SemanticAnchorMetrics]:
        """Generate metrics for a specific semantic anchor."""
        if anchor_id not in self.semantic_anchors:
            return None
        
        anchor_data = self.semantic_anchors[anchor_id]
        
        # Calculate stability metrics
        drift_values = list(anchor_data['drift_measurements'])
        avg_drift = statistics.mean(drift_values) if drift_values else 0.0
        
        activations = anchor_data['activation_history']
        if not activations:
            return None
        
        # Calculate activation frequency (activations per hour)
        recent_activations = [
            act for act in activations 
            if time.time() - act[0] < 3600  # Last hour
        ]
        activation_frequency = len(recent_activations)
        
        # Calculate coherence and strength
        recent_strengths = [act[1] for act in list(activations)[-10:]]
        avg_strength = statistics.mean(recent_strengths) if recent_strengths else 0.0
        coherence = 1.0 - statistics.stdev(recent_strengths) if len(recent_strengths) > 1 else 1.0
        
        return SemanticAnchorMetrics(
            anchor_id=anchor_id,
            stability_score=1.0 - avg_drift,
            drift_rate=avg_drift,
            coherence_index=max(0.0, coherence),
            activation_frequency=activation_frequency,
            last_activated=activations[-1][0],
            anchor_strength=avg_strength
        )
    
    def _get_latest_field_state(self) -> Dict[str, float]:
        """Get latest semantic field state."""
        # Placeholder implementation
        return {
            'field_energy': 0.7,
            'field_coherence': 0.8,
            'field_stability': 0.9
        }
    
    def _get_latest_constitutional_scores(self) -> Dict[str, float]:
        """Get latest constitutional compliance scores."""
        if self.constitutional_history:
            latest_score = self.constitutional_history[-1]
            return {
                'overall_compliance': latest_score,
                'safety_score': min(1.0, latest_score + 0.1),
                'risk_assessment': max(0.0, 1.0 - latest_score)
            }
        return {'overall_compliance': 1.0, 'safety_score': 1.0, 'risk_assessment': 0.0}
    
    def _get_attention_summary(self) -> Dict[str, Any]:
        """Get summary of attention patterns."""
        recent_entropy = list(self.attention_patterns['entropy'])[-10:]
        return {
            'average_entropy': statistics.mean(recent_entropy) if recent_entropy else 0.0,
            'entropy_trend': 'stable',
            'pattern_diversity': len(recent_entropy)
        }
    
    def _measure_resource_usage(self) -> Dict[str, float]:
        """Measure current resource usage."""
        return {
            'memory_usage_mb': 0.0,  # Placeholder
            'cpu_usage_percent': 0.0,  # Placeholder
            'processing_efficiency': 1.0  # Placeholder
        }
    
    def _analyze_anchor_stability(self) -> Dict[str, Any]:
        """Analyze overall semantic anchor stability."""
        if not self.semantic_anchors:
            return {'status': 'no_anchors', 'recommendations': ['Create semantic anchors']}
        
        stability_scores = []
        for anchor_id in self.semantic_anchors:
            metrics = self._generate_anchor_metrics(anchor_id)
            if metrics:
                stability_scores.append(metrics.stability_score)
        
        if not stability_scores:
            return {'status': 'insufficient_data'}
        
        avg_stability = statistics.mean(stability_scores)
        min_stability = min(stability_scores)
        
        return {
            'average_stability': avg_stability,
            'minimum_stability': min_stability,
            'stable_anchors': len([s for s in stability_scores if s > 0.8]),
            'unstable_anchors': len([s for s in stability_scores if s < 0.6]),
            'status': 'stable' if avg_stability > 0.8 else 'needs_attention'
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        recent_times = list(self.performance_metrics['processing_time'])[-50:]
        if len(recent_times) < 2:
            return {'status': 'insufficient_data'}
        
        # Simple trend analysis
        first_half = recent_times[:len(recent_times)//2]
        second_half = recent_times[len(recent_times)//2:]
        
        trend = 'stable'
        if statistics.mean(second_half) > statistics.mean(first_half) * 1.1:
            trend = 'degrading'
        elif statistics.mean(second_half) < statistics.mean(first_half) * 0.9:
            trend = 'improving'
        
        return {
            'trend': trend,
            'average_time': statistics.mean(recent_times),
            'processing_variance': statistics.stdev(recent_times) if len(recent_times) > 1 else 0.0
        }
    
    def _analyze_constitutional_trends(self) -> Dict[str, Any]:
        """Analyze constitutional compliance trends."""
        recent_scores = list(self.constitutional_history)[-50:]
        if not recent_scores:
            return {'status': 'no_data'}
        
        avg_score = statistics.mean(recent_scores)
        violation_rate = sum(1 for s in recent_scores if s < 0.8) / len(recent_scores)
        
        return {
            'average_compliance': avg_score,
            'violation_rate': violation_rate,
            'status': 'compliant' if violation_rate < 0.1 else 'needs_review'
        }
    
    def _analyze_attention_patterns(self) -> Dict[str, Any]:
        """Analyze attention pattern characteristics."""
        entropy_values = list(self.attention_patterns['entropy'])
        if not entropy_values:
            return {'status': 'no_data'}
        
        return {
            'average_entropy': statistics.mean(entropy_values),
            'entropy_stability': 1.0 - (statistics.stdev(entropy_values) if len(entropy_values) > 1 else 0.0),
            'pattern_complexity': 'balanced'
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations based on telemetry data."""
        recommendations = []
        
        # Check anchor stability
        anchor_analysis = self._analyze_anchor_stability()
        if anchor_analysis.get('status') == 'needs_attention':
            recommendations.append("Review semantic anchor stability - some anchors showing drift")
        
        # Check performance
        perf_analysis = self._analyze_performance_trends()
        if perf_analysis.get('trend') == 'degrading':
            recommendations.append("Performance degradation detected - consider optimization")
        
        # Check constitutional compliance
        const_analysis = self._analyze_constitutional_trends()
        if const_analysis.get('violation_rate', 0) > 0.15:
            recommendations.append("High constitutional violation rate - review filtering parameters")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return recommendations
    
    def _export_anchor_data(self) -> Dict[str, Any]:
        """Export semantic anchor data for analysis."""
        anchor_export = {}
        for anchor_id, data in self.semantic_anchors.items():
            anchor_export[anchor_id] = {
                'creation_time': data['creation_time'],
                'total_activations': len(data['activation_history']),
                'recent_drift': list(data['drift_measurements'])[-5:],
                'status': 'active' if anchor_id in self.active_anchors else 'inactive'
            }
        return anchor_export


# Global telemetry collector instance
_global_telemetry_collector = None

def get_telemetry_collector() -> ESCTelemetryCollector:
    """Get global telemetry collector instance."""
    global _global_telemetry_collector
    if _global_telemetry_collector is None:
        _global_telemetry_collector = ESCTelemetryCollector()
    return _global_telemetry_collector

def reset_telemetry_collector():
    """Reset global telemetry collector."""
    global _global_telemetry_collector
    _global_telemetry_collector = ESCTelemetryCollector()