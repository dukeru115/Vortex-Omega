# NFCS Monitoring Module

## Overview

The monitoring module provides comprehensive metrics collection, performance tracking, and system health monitoring for the Neural Field Control System (NFCS). This module implements the core monitoring instrumentation that integrates with the external monitoring infrastructure.

**Purpose**: Internal metrics collection and system health monitoring for all NFCS components.

## 📁 Module Structure

```
src/monitoring/
├── README.md          # This documentation
├── metrics.py         # Core metrics collection and instrumentation
└── __init__.py       # Module initialization
```

## 🎯 Core Components

### 1. **Metrics Collection** (`metrics.py`)

**Purpose**: Centralized metrics collection and instrumentation for all NFCS components.

**Key Features**:
- **Constitutional Metrics**: Real-time compliance monitoring and violation tracking
- **Performance Metrics**: Response times, throughput, and resource utilization
- **Kuramoto Synchronization**: Phase coherence and coupling strength metrics  
- **ESC Token Processing**: Token throughput and semantic field stability
- **System Health**: Memory usage, CPU utilization, and error rates
- **Custom Metrics**: Domain-specific metrics for research and optimization

**Main Classes**:
```python
class NFCSMetricsCollector:
    """Main metrics collection class for NFCS system."""
    
    def __init__(self):
        self.constitutional_metrics = ConstitutionalMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.kuramoto_metrics = KuramotoMetrics()
        self.esc_metrics = ESCMetrics()
        self.system_metrics = SystemMetrics()
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect all system metrics."""
        
    async def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        
    async def get_health_status(self) -> HealthStatus:
        """Get current system health status."""
```

## 📊 Metrics Categories

### Constitutional Monitoring
```python
# Constitutional compliance metrics
constitutional_violations_total = Counter(
    'nfcs_constitutional_violations_total',
    'Total constitutional violations detected'
)

constitutional_compliance_score = Gauge(
    'nfcs_constitutional_compliance_score',
    'Current constitutional compliance score (0-1)'
)

ha_number_current = Gauge(
    'nfcs_ha_number_current',
    'Current hallucination number (Ha)'
)

policy_enforcement_rate = Gauge(
    'nfcs_policy_enforcement_rate',
    'Policy enforcement success rate'
)
```

### Performance Monitoring
```python
# Request and response metrics
request_duration_seconds = Histogram(
    'nfcs_request_duration_seconds',
    'Request duration in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

requests_total = Counter(
    'nfcs_requests_total',
    'Total number of requests processed',
    ['method', 'endpoint', 'status']
)

# Memory and CPU metrics
memory_usage_bytes = Gauge(
    'nfcs_memory_usage_bytes',
    'Current memory usage in bytes'
)

cpu_utilization_percent = Gauge(
    'nfcs_cpu_utilization_percent',
    'Current CPU utilization percentage'
)
```

### Kuramoto Synchronization
```python
# Synchronization metrics
kuramoto_sync_level = Gauge(
    'nfcs_kuramoto_sync_level',
    'Current Kuramoto synchronization level (0-1)'
)

kuramoto_coupling_strength = Gauge(
    'nfcs_kuramoto_coupling_strength',
    'Current coupling strength'
)

oscillator_phases = Gauge(
    'nfcs_oscillator_phases',
    'Individual oscillator phases',
    ['oscillator_id']
)

phase_coherence = Gauge(
    'nfcs_phase_coherence',
    'Global phase coherence measure'
)
```

### ESC System Monitoring
```python
# Token processing metrics
esc_tokens_processed_total = Counter(
    'nfcs_esc_tokens_processed_total',
    'Total tokens processed by ESC system'
)

esc_semantic_field_stability = Gauge(
    'nfcs_esc_semantic_field_stability',
    'Semantic field stability measure'
)

esc_attention_weights = Gauge(
    'nfcs_esc_attention_weights',
    'Attention mechanism weights',
    ['head', 'layer']
)

esc_processing_latency = Histogram(
    'nfcs_esc_processing_latency_seconds',
    'ESC token processing latency in seconds'
)
```

## 🚀 Usage Examples

### Basic Metrics Collection
```python
from src.monitoring.metrics import NFCSMetricsCollector

# Initialize metrics collector
metrics = NFCSMetricsCollector()

# Collect all metrics
current_metrics = await metrics.collect_metrics()
print(f"System health: {current_metrics['health_status']}")
print(f"Constitutional compliance: {current_metrics['constitutional_compliance']}")

# Get Prometheus-formatted metrics
prometheus_data = await metrics.export_prometheus_metrics()
```

### Constitutional Monitoring
```python
from src.monitoring.metrics import ConstitutionalMetrics

# Initialize constitutional monitoring
constitutional = ConstitutionalMetrics()

# Record violation
await constitutional.record_violation(
    violation_type="policy_breach",
    severity="high",
    context={"policy_id": "safety_001", "trigger": "unsafe_output"}
)

# Update compliance score
await constitutional.update_compliance_score(0.95)

# Check current status
status = await constitutional.get_compliance_status()
```

### Performance Monitoring
```python
from src.monitoring.metrics import PerformanceMetrics
import time

# Initialize performance monitoring
performance = PerformanceMetrics()

# Track request
start_time = time.time()
# ... process request ...
duration = time.time() - start_time

await performance.record_request(
    method="POST",
    endpoint="/api/process",
    status_code=200,
    duration=duration
)

# Update resource usage
await performance.update_resource_usage(
    memory_bytes=1024*1024*512,  # 512MB
    cpu_percent=45.2
)
```

### Kuramoto Monitoring
```python
from src.monitoring.metrics import KuramotoMetrics
import numpy as np

# Initialize Kuramoto monitoring
kuramoto = KuramotoMetrics()

# Update synchronization metrics
phases = np.array([0.1, 0.3, 0.2, 0.4, 0.15])  # 5 oscillators
sync_level = calculate_sync_level(phases)

await kuramoto.update_synchronization(
    phases=phases,
    sync_level=sync_level,
    coupling_strength=2.0
)

# Get synchronization status
sync_status = await kuramoto.get_sync_status()
```

## 🔧 Configuration

### Metrics Configuration
```python
# Metrics collection configuration
METRICS_CONFIG = {
    'collection_interval': 10,  # seconds
    'retention_period': 86400,  # 24 hours
    'export_format': 'prometheus',
    'health_check_interval': 30,
    
    'constitutional': {
        'enabled': True,
        'violation_threshold': 0.1,
        'compliance_target': 0.95
    },
    
    'performance': {
        'enabled': True,
        'histogram_buckets': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        'resource_monitoring': True
    },
    
    'kuramoto': {
        'enabled': True,
        'oscillator_count': 5,
        'sync_threshold': 0.8
    },
    
    'esc': {
        'enabled': True,
        'token_tracking': True,
        'semantic_field_monitoring': True
    }
}
```

### Alerting Thresholds
```python
# Alert threshold configuration
ALERT_THRESHOLDS = {
    'constitutional_violations': {
        'warning': 1,
        'critical': 5
    },
    'response_time_p95': {
        'warning': 1.0,  # seconds
        'critical': 2.0
    },
    'memory_usage': {
        'warning': 0.8,  # 80%
        'critical': 0.9  # 90%
    },
    'kuramoto_sync_level': {
        'warning': 0.7,
        'critical': 0.5
    },
    'error_rate': {
        'warning': 0.05,  # 5%
        'critical': 0.1   # 10%
    }
}
```

## 📈 Integration with External Monitoring

### Prometheus Integration
```python
from prometheus_client import start_http_server, generate_latest

class PrometheusExporter:
    """Prometheus metrics exporter for NFCS."""
    
    def __init__(self, port=8000):
        self.port = port
        self.metrics_collector = NFCSMetricsCollector()
    
    async def start_server(self):
        """Start Prometheus metrics HTTP server."""
        start_http_server(self.port)
        
    async def update_metrics(self):
        """Update all metrics for Prometheus scraping."""
        metrics = await self.metrics_collector.collect_metrics()
        # Update Prometheus metrics...
        
# Usage
exporter = PrometheusExporter(port=8000)
await exporter.start_server()
```

### Grafana Dashboard Data
```python
class GrafanaDashboardData:
    """Prepare data for Grafana dashboard consumption."""
    
    def __init__(self):
        self.metrics = NFCSMetricsCollector()
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for Grafana dashboards."""
        metrics = await self.metrics.collect_metrics()
        
        return {
            'system_health': {
                'status': metrics['health_status'],
                'uptime': metrics['uptime_seconds'],
                'version': metrics['system_version']
            },
            'performance': {
                'response_time_p50': metrics['response_time_p50'],
                'response_time_p95': metrics['response_time_p95'],
                'requests_per_second': metrics['requests_per_second'],
                'error_rate': metrics['error_rate']
            },
            'constitutional': {
                'compliance_score': metrics['constitutional_compliance'],
                'violations_total': metrics['constitutional_violations'],
                'ha_number': metrics['ha_number_current']
            },
            'kuramoto': {
                'sync_level': metrics['kuramoto_sync_level'],
                'coupling_strength': metrics['kuramoto_coupling'],
                'oscillator_phases': metrics['oscillator_phases']
            }
        }
```

## 🔍 Health Checks

### System Health Assessment
```python
class HealthChecker:
    """System health assessment and reporting."""
    
    def __init__(self):
        self.metrics = NFCSMetricsCollector()
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        return {
            'overall_status': await self._calculate_overall_status(),
            'components': {
                'constitutional': await self._check_constitutional_health(),
                'kuramoto': await self._check_kuramoto_health(),
                'esc': await self._check_esc_health(),
                'memory': await self._check_memory_health(),
                'cpu': await self._check_cpu_health()
            },
            'alerts': await self._get_active_alerts(),
            'timestamp': time.time()
        }
    
    async def _calculate_overall_status(self) -> str:
        """Calculate overall system health status."""
        # Implementation for overall health calculation
        pass
```

## 🛠️ Development and Testing

### Testing Metrics Collection
```python
import pytest
from src.monitoring.metrics import NFCSMetricsCollector

@pytest.mark.asyncio
async def test_metrics_collection():
    """Test basic metrics collection functionality."""
    collector = NFCSMetricsCollector()
    
    # Test metrics collection
    metrics = await collector.collect_metrics()
    assert 'health_status' in metrics
    assert 'constitutional_compliance' in metrics
    assert 'kuramoto_sync_level' in metrics
    
    # Test Prometheus export
    prometheus_data = await collector.export_prometheus_metrics()
    assert 'nfcs_' in prometheus_data

@pytest.mark.asyncio
async def test_constitutional_metrics():
    """Test constitutional monitoring functionality."""
    from src.monitoring.metrics import ConstitutionalMetrics
    
    constitutional = ConstitutionalMetrics()
    
    # Test violation recording
    await constitutional.record_violation(
        violation_type="test_violation",
        severity="low",
        context={"test": True}
    )
    
    # Test compliance score update
    await constitutional.update_compliance_score(0.95)
    
    # Verify status
    status = await constitutional.get_compliance_status()
    assert status['compliance_score'] == 0.95
```

## 📚 Related Documentation

- [System Monitoring Setup](../../monitoring/README.md)
- [Production Deployment](../../DEPLOYMENT.md)
- [Configuration Management](../../config/README.md)
- [Performance Optimization](../performance/README.md)

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-09-15 | Initial monitoring module documentation | Team Ω |

---

*This monitoring module provides the foundational metrics collection and health monitoring capabilities for the NFCS system. For external monitoring infrastructure setup, see the [monitoring directory](../../monitoring/README.md).*

_Last updated: 2025-09-15 by Team Ω_