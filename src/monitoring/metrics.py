"""
Prometheus Metrics Module for Vortex-Omega NFCS
===============================================

Provides comprehensive metrics collection and exposure for monitoring.
"""

from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST, Info
)
from prometheus_client.core import GaugeMetricFamily
import asyncio
import time
import psutil
import logging
from typing import Dict, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

# Create a custom registry
registry = CollectorRegistry()

# System Information
system_info = Info(
    'vortex_omega_system',
    'System information',
    registry=registry
)

# Request metrics
request_counter = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

# Symbolic AI metrics
symbolic_processing_duration = Histogram(
    'symbolic_processing_duration_seconds',
    'Symbolic AI processing duration',
    ['operation'],
    registry=registry
)

symbolic_clauses_processed = Counter(
    'symbolic_clauses_processed_total',
    'Total symbolic clauses processed',
    ['clause_type'],
    registry=registry
)

symbolic_verification_results = Counter(
    'symbolic_verification_results_total',
    'Symbolic verification results',
    ['result'],
    registry=registry
)

# ESC metrics
esc_buffer_size = Gauge(
    'esc_buffer_size',
    'Current ESC buffer size',
    ['buffer_type'],
    registry=registry
)

esc_memory_usage = Gauge(
    'esc_memory_usage_mb',
    'ESC memory usage in MB',
    registry=registry
)

esc_oscillator_count = Gauge(
    'esc_oscillator_count',
    'Number of active ESC oscillators',
    registry=registry
)

esc_echo_activations = Counter(
    'esc_echo_activations_total',
    'Total ESC echo activations',
    ['scale'],
    registry=registry
)

# Kuramoto metrics
kuramoto_order_parameter = Gauge(
    'kuramoto_order_parameter',
    'Kuramoto model order parameter (coherence)',
    registry=registry
)

kuramoto_frequency_variance = Gauge(
    'kuramoto_frequency_variance',
    'Variance of Kuramoto oscillator frequencies',
    registry=registry
)

kuramoto_sync_events = Counter(
    'kuramoto_sync_events_total',
    'Total Kuramoto synchronization events',
    registry=registry
)

# Discrepancy Gate metrics
discrepancy_detected = Counter(
    'discrepancy_detected_total',
    'Total discrepancies detected',
    ['severity'],
    registry=registry
)

discrepancy_resolution_time = Histogram(
    'discrepancy_resolution_time_seconds',
    'Time to resolve discrepancies',
    ['severity'],
    registry=registry
)

# Kant Mode metrics
kant_evaluations = Counter(
    'kant_evaluations_total',
    'Total Kant mode ethical evaluations',
    ['principle'],
    registry=registry
)

kant_ethical_score = Histogram(
    'kant_ethical_score',
    'Kant mode ethical scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry
)

# Security metrics
rate_limit_exceeded = Counter(
    'rate_limit_exceeded_total',
    'Total rate limit violations',
    ['user_type'],
    registry=registry
)

circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=half-open, 2=open)',
    ['service'],
    registry=registry
)

security_violations = Counter(
    'security_violations_total',
    'Total security violations detected',
    ['violation_type'],
    registry=registry
)

# Database metrics
db_connections_active = Gauge(
    'db_connections_active',
    'Active database connections',
    registry=registry
)

db_query_duration = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['query_type'],
    registry=registry
)

# Cache metrics
cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type'],
    registry=registry
)

cache_evictions = Counter(
    'cache_evictions_total',
    'Total cache evictions',
    ['cache_type', 'reason'],
    registry=registry
)

# System metrics
cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=registry
)

memory_usage = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage',
    registry=registry
)

disk_usage = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage',
    ['mount_point'],
    registry=registry
)


class MetricsCollector:
    """Collects and exposes metrics for Prometheus"""
    
    def __init__(self):
        self.registry = registry
        self._start_time = time.time()
        self._update_system_info()
        
    def _update_system_info(self):
        """Update system information"""
        system_info.info({
            'version': '2.5.0',
            'environment': 'production',
            'component': 'nfcs',
            'python_version': '3.11'
        })
    
    def track_request(self, method: str, endpoint: str, status: int, duration: float):
        """Track HTTP request metrics"""
        request_counter.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def track_symbolic_processing(self, operation: str, duration: float, clauses: Dict[str, int]):
        """Track Symbolic AI processing metrics"""
        symbolic_processing_duration.labels(operation=operation).observe(duration)
        for clause_type, count in clauses.items():
            symbolic_clauses_processed.labels(clause_type=clause_type).inc(count)
    
    def track_verification(self, result: str):
        """Track verification results"""
        symbolic_verification_results.labels(result=result).inc()
    
    def update_esc_metrics(self, buffer_sizes: Dict[str, int], memory_mb: float, oscillator_count: int):
        """Update ESC metrics"""
        for buffer_type, size in buffer_sizes.items():
            esc_buffer_size.labels(buffer_type=buffer_type).set(size)
        esc_memory_usage.set(memory_mb)
        esc_oscillator_count.set(oscillator_count)
    
    def track_echo_activation(self, scale: str):
        """Track ESC echo activation"""
        esc_echo_activations.labels(scale=scale).inc()
    
    def update_kuramoto_metrics(self, order_param: float, freq_variance: float):
        """Update Kuramoto model metrics"""
        kuramoto_order_parameter.set(order_param)
        kuramoto_frequency_variance.set(freq_variance)
        kuramoto_sync_events.inc()
    
    def track_discrepancy(self, severity: str, resolution_time: Optional[float] = None):
        """Track discrepancy detection"""
        discrepancy_detected.labels(severity=severity).inc()
        if resolution_time:
            discrepancy_resolution_time.labels(severity=severity).observe(resolution_time)
    
    def track_kant_evaluation(self, principle: str, ethical_score: float):
        """Track Kant mode evaluation"""
        kant_evaluations.labels(principle=principle).inc()
        kant_ethical_score.observe(ethical_score)
    
    def track_security_event(self, event_type: str, **kwargs):
        """Track security events"""
        if event_type == 'rate_limit':
            rate_limit_exceeded.labels(user_type=kwargs.get('user_type', 'unknown')).inc()
        elif event_type == 'circuit_breaker':
            state = kwargs.get('state', 0)
            service = kwargs.get('service', 'unknown')
            circuit_breaker_state.labels(service=service).set(state)
        elif event_type == 'violation':
            violation_type = kwargs.get('violation_type', 'unknown')
            security_violations.labels(violation_type=violation_type).inc()
    
    def update_database_metrics(self, active_connections: int):
        """Update database metrics"""
        db_connections_active.set(active_connections)
    
    def track_query(self, query_type: str, duration: float):
        """Track database query"""
        db_query_duration.labels(query_type=query_type).observe(duration)
    
    def track_cache_operation(self, operation: str, cache_type: str, **kwargs):
        """Track cache operations"""
        if operation == 'hit':
            cache_hits.labels(cache_type=cache_type).inc()
        elif operation == 'miss':
            cache_misses.labels(cache_type=cache_type).inc()
        elif operation == 'eviction':
            reason = kwargs.get('reason', 'unknown')
            cache_evictions.labels(cache_type=cache_type, reason=reason).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage.set(memory.percent)
        
        # Disk usage
        for partition in psutil.disk_partitions():
            if partition.mountpoint:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage.labels(mount_point=partition.mountpoint).set(usage.percent)
    
    async def collect_metrics_periodically(self, interval: int = 15):
        """Collect system metrics periodically"""
        while True:
            try:
                self.update_system_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(interval)
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry)


# Decorators for metric tracking
def track_processing_time(operation: str):
    """Decorator to track processing time"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                symbolic_processing_duration.labels(operation=operation).observe(duration)
                return result
            except Exception as e:
                duration = time.time() - start
                symbolic_processing_duration.labels(operation=f"{operation}_error").observe(duration)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                symbolic_processing_duration.labels(operation=operation).observe(duration)
                return result
            except Exception as e:
                duration = time.time() - start
                symbolic_processing_duration.labels(operation=f"{operation}_error").observe(duration)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def track_cache(cache_type: str):
    """Decorator to track cache operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would be integrated with actual cache logic
            result = func(*args, **kwargs)
            if result is not None:
                cache_hits.labels(cache_type=cache_type).inc()
            else:
                cache_misses.labels(cache_type=cache_type).inc()
            return result
        return wrapper
    return decorator


# Global metrics collector instance
metrics_collector = MetricsCollector()


# FastAPI/Flask integration endpoint
async def metrics_endpoint():
    """Endpoint to expose metrics"""
    metrics_data = metrics_collector.get_metrics()
    return metrics_data, 200, {'Content-Type': CONTENT_TYPE_LATEST}