# 📊 Monitoring & Observability - NFCS v2.5.0

## 🎯 **PRODUCTION MONITORING SYSTEM**

This directory contains **enterprise-grade monitoring and observability** configurations for the Neural Field Control System (NFCS) v2.5.0, including Prometheus, Grafana, and custom alerting.

**Status**: ✅ **PRODUCTION READY** - Updated September 2025

## 🏗️ Architecture

### Current MVP Monitoring
The MVP system includes built-in monitoring via:
- **MVP Controller**: Real-time system health tracking
- **Web Dashboard**: Live metrics visualization at http://localhost:5000
- **Socket.IO Integration**: Real-time data streaming and updates
- **Supervisor Monitoring**: Service health and auto-restart capabilities
- **Live Logging**: Real-time event streaming and analysis
- **Constitutional Monitoring**: Real-time Ha (hallucination) monitoring

### Production Monitoring Stack
```
📊 Grafana (Visualization & Dashboards)
    ↓
📈 Prometheus (Metrics Collection & Storage)
    ↓
🎯 NFCS Applications (Instrumented)
    ├── Constitutional Monitoring (Algorithm 1)
    ├── ESC-Kuramoto Integration (64 oscillators)
    ├── Cognitive Modules (5 systems)
    ├── Empirical Validation Pipeline
    └── MVP Production Interface
    ↓
📝 Log Aggregation (ELK Stack)
    ├── Elasticsearch (Search & Analytics)
    ├── Logstash (Processing)
    └── Kibana (Visualization)
```

## 📁 Directory Structure

### 📈 `prometheus/`
Prometheus monitoring configuration:
- **prometheus.yml**: Main Prometheus configuration with NFCS targets
- **rules/**: Alerting and recording rules for NFCS metrics
- **targets/**: Service discovery configuration for dynamic environments
- **alerts.yml**: NFCS-specific alerting rules

### 📊 `grafana/`
Grafana dashboard and configuration:
- **dashboards/**: JSON dashboard definitions for NFCS components
- **datasources/**: Data source configurations (Prometheus, Loki)
- **provisioning/**: Automated setup scripts and configurations
- **plugins/**: Custom Grafana plugins for NFCS visualizations

### 📝 `logging/`
Centralized logging configuration:
- **logstash/**: Log processing pipelines
- **filebeat/**: Log shipping configuration
- **elasticsearch/**: Index templates and mappings
- **kibana/**: Dashboard and visualization configs

## 🎯 Key Metrics

### System Health Metrics
- **Constitutional Status**: Real-time compliance monitoring (0.0-1.0)
- **Kuramoto Sync Levels**: 64-oscillator synchronization metrics
- **Ha Monitoring**: Hallucination number tracking with thresholds
- **Validation Scores**: Empirical validation pipeline metrics
- **Safety Violations**: Security and safety event counts
- **Cognitive Module Status**: Active module tracking (5/5 operational)
- **ESC Token Processing**: Token throughput and semantic field stability

### Performance Metrics
- **API Response Time**: Request latency percentiles (p50, p95, p99)
- **Memory Usage**: System memory consumption patterns
- **CPU Utilization**: Processing load across all cores
- **Neural Field Computation**: FPS and mathematical accuracy
- **Database Performance**: Query execution times and connection pool status
- **Coordination Frequency**: Real-time 10Hz orchestration monitoring

### Business & Research Metrics
- **User Sessions**: Active user tracking and session analytics
- **System Uptime**: Availability monitoring with SLA tracking
- **Error Rates**: System error frequency and categorization
- **Prediction Accuracy**: ML model performance metrics
- **Research Throughput**: Simulation and analysis completion rates
- **Constitutional Compliance**: Policy enforcement effectiveness

## 🚨 Alerting Rules

### Critical Alerts
- **System Downtime**: Service unavailability (immediate escalation)
- **Constitutional Violations**: Safety breaches requiring intervention
- **Memory Leaks**: Excessive memory usage patterns
- **High Error Rates**: Above threshold errors (>5% error rate)
- **Kuramoto Desynchronization**: Sync levels below critical threshold
- **Ha Threshold Exceeded**: Hallucination detection beyond safe limits

### Warning Alerts
- **Performance Degradation**: Slow response times (>1s p95)
- **Low Synchronization**: Kuramoto sync below optimal threshold
- **Resource Usage**: High CPU/memory utilization (>80%)
- **Queue Backlog**: Processing queue depth warnings
- **Configuration Drift**: Unexpected configuration changes

### Info Alerts
- **System Events**: Planned maintenance, deployments, configuration updates
- **Research Milestones**: Experiment completion notifications
- **Capacity Planning**: Resource usage trend notifications

## 🚀 Quick Start

### MVP Monitoring Access
```bash
# Start MVP with built-in monitoring
./start_mvp.sh

# Access dashboard with live metrics
# Local: http://localhost:5000
# Live Demo: https://5000-i3xy7hm4ybz4gfsijjc3h-6532622b.e2b.dev/

# View real-time constitutional monitoring
# Constitutional monitor runs on port 8765
# WebSocket connection for live Ha monitoring
```

### Production Monitoring Setup
```bash
# Start complete monitoring stack
docker-compose -f monitoring/docker-compose.yml up -d

# Access Grafana dashboards
# http://localhost:3000 (admin/admin)

# Access Prometheus metrics
# http://localhost:9090

# Access Kibana logs
# http://localhost:5601
```

### Monitoring Configuration
```bash
# Configure monitoring targets
cp monitoring/prometheus/prometheus.yml.example monitoring/prometheus/prometheus.yml
nano monitoring/prometheus/prometheus.yml

# Setup Grafana datasources
cp monitoring/grafana/datasources.yml.example monitoring/grafana/datasources.yml

# Configure alerting
cp monitoring/prometheus/alerts.yml.example monitoring/prometheus/alerts.yml
```

## 📈 Dashboard Access

### MVP Dashboard (Currently Active)
- **URL**: http://localhost:5000
- **Features**: 
  - Real-time NFCS system monitoring
  - Constitutional compliance tracking
  - Kuramoto synchronization visualization
  - Performance metrics and system health
  - Interactive system controls
- **Live Demo**: https://5000-i3xy7hm4ybz4gfsijjc3h-6532622b.e2b.dev/

### Grafana Dashboards (Production)
- **NFCS System Overview**: High-level system health and performance
- **Constitutional Monitoring**: Detailed compliance and safety metrics
- **ESC-Kuramoto Analysis**: Synchronization and coupling dynamics
- **Performance Analytics**: Detailed performance analysis and optimization
- **Research Dashboard**: Experiment tracking and results visualization
- **Alert Management**: Alert status, history, and escalation tracking

### Kibana Dashboards (Logging)
- **Application Logs**: Centralized application log analysis
- **Security Events**: Security-related event monitoring
- **Error Analysis**: Error pattern analysis and troubleshooting
- **Audit Logs**: Compliance and audit trail visualization

## 🔧 Configuration

### Prometheus Targets
```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"
  - "rules/*.yml"

scrape_configs:
  # NFCS MVP application
  - job_name: 'nfcs-mvp'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  # NFCS API servers
  - job_name: 'nfcs-api'
    static_configs:
      - targets: 
        - 'localhost:8000'
        - 'localhost:8001'
        - 'localhost:8002'
    
  # Constitutional monitoring
  - job_name: 'nfcs-constitutional'
    static_configs:
      - targets: ['localhost:8765']
    metrics_path: '/constitutional/metrics'
    
  # System metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
      
  # Database metrics
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['localhost:9187']
```

### Grafana Data Sources
```yaml
# monitoring/grafana/datasources.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
    editable: true
    
  - name: Loki
    type: loki
    access: proxy
    url: http://localhost:3100
    editable: true
    
  - name: Elasticsearch
    type: elasticsearch
    access: proxy
    url: http://localhost:9200
    database: nfcs-logs
    editable: true
```

### Alert Rules
```yaml
# monitoring/prometheus/alerts.yml
groups:
  - name: nfcs.rules
    rules:
      # Constitutional violations
      - alert: ConstitutionalViolation
        expr: nfcs_constitutional_violations_total > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Constitutional violation detected"
          description: "NFCS constitutional framework has detected {{ $value }} violations"
          
      # High response time
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(nfcs_request_duration_seconds_bucket[5m])) > 1.0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s"
          
      # Kuramoto desynchronization
      - alert: KuramotoDesync
        expr: nfcs_kuramoto_sync_level < 0.7
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Kuramoto synchronization degraded"
          description: "Synchronization level is {{ $value }}, below threshold"
          
      # Memory usage
      - alert: HighMemoryUsage
        expr: (process_resident_memory_bytes / 1024 / 1024 / 1024) > 4.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}GB"
```

## 📝 Maintenance

### Log Rotation
```bash
# Automated log rotation configuration
# /etc/logrotate.d/nfcs
/var/log/nfcs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 nfcs nfcs
    postrotate
        systemctl reload nfcs-api
    endscript
}
```

### Backup Strategy
```bash
# Daily backup script for monitoring data
#!/bin/bash
# monitoring/scripts/backup-monitoring.sh

BACKUP_DIR="/backup/monitoring/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup Grafana dashboards
cp -r /var/lib/grafana/dashboards "$BACKUP_DIR/"

# Backup Prometheus configuration
cp -r /etc/prometheus "$BACKUP_DIR/"

# Export Prometheus data (last 7 days)
promtool query range \
  --start=$(date -d '7 days ago' +%s) \
  --end=$(date +%s) \
  --step=300 \
  'up' > "$BACKUP_DIR/prometheus_data.json"

# Compress backup
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

# Cleanup old backups (keep 30 days)
find /backup/monitoring -name "*.tar.gz" -mtime +30 -delete
```

### Performance Tuning
```bash
# Prometheus retention and performance
# /etc/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  
# Storage retention (adjust based on disk space)
# --storage.tsdb.retention.time=15d
# --storage.tsdb.retention.size=10GB

# Grafana performance optimization
# /etc/grafana/grafana.ini
[database]
max_open_conn = 100
max_idle_conn = 100
conn_max_lifetime = 14400

[server]
enable_gzip = true
```

## 🔍 Troubleshooting

### Common Issues

1. **Missing Metrics**: 
   ```bash
   # Check service instrumentation
   curl http://localhost:5000/metrics
   curl http://localhost:8765/constitutional/metrics
   
   # Verify Prometheus targets
   curl http://localhost:9090/api/v1/targets
   ```

2. **Dashboard Errors**: 
   ```bash
   # Verify Grafana data source connections
   curl http://localhost:3000/api/datasources
   
   # Check Grafana logs
   journalctl -u grafana-server -f
   ```

3. **Alert Flooding**: 
   ```bash
   # Review alert thresholds in alerts.yml
   # Check alert inhibition rules
   # Verify notification channels
   ```

4. **Performance Issues**: 
   ```bash
   # Check resource allocation
   docker stats
   
   # Monitor Prometheus performance
   curl http://localhost:9090/api/v1/status/tsdb
   ```

### Debug Commands
```bash
# Check MVP monitoring status
supervisorctl -c supervisord.conf status

# Test constitutional monitoring WebSocket
wscat -c ws://localhost:8765/constitutional/ws

# Validate Prometheus configuration
promtool check config monitoring/prometheus/prometheus.yml

# Test Grafana API
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
     http://localhost:3000/api/health

# Check log aggregation
curl http://localhost:9200/_cat/indices?v
```

## 📊 Metrics Exposition

### NFCS Custom Metrics
```python
# Example NFCS metrics exposition
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Constitutional metrics
constitutional_violations = Counter('nfcs_constitutional_violations_total', 
                                  'Total constitutional violations detected')
ha_current = Gauge('nfcs_ha_current', 'Current hallucination number')

# Performance metrics
request_duration = Histogram('nfcs_request_duration_seconds',
                            'Request duration in seconds')
kuramoto_sync = Gauge('nfcs_kuramoto_sync_level', 
                     'Current Kuramoto synchronization level')

# System metrics
cognitive_modules_active = Gauge('nfcs_cognitive_modules_active',
                                'Number of active cognitive modules')
esc_tokens_processed = Counter('nfcs_esc_tokens_processed_total',
                              'Total tokens processed by ESC system')

# Start metrics server
start_http_server(8000)
```

## 🤝 Integration Points

### CI/CD Integration
```yaml
# .github/workflows/monitoring-tests.yml
name: Monitoring Tests
on: [push, pull_request]

jobs:
  monitoring-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Start monitoring stack
        run: docker-compose -f monitoring/docker-compose.yml up -d
        
      - name: Wait for services
        run: sleep 30
        
      - name: Test Prometheus targets
        run: |
          curl -f http://localhost:9090/api/v1/targets
          
      - name: Test Grafana health
        run: |
          curl -f http://localhost:3000/api/health
          
      - name: Validate dashboards
        run: |
          for dashboard in monitoring/grafana/dashboards/*.json; do
            curl -X POST \
              -H "Content-Type: application/json" \
              -d @"$dashboard" \
              http://admin:admin@localhost:3000/api/dashboards/db
          done
```

### API Integration
```python
# NFCS monitoring API integration
import asyncio
from monitoring.client import MonitoringClient

class NFCSMonitoring:
    def __init__(self):
        self.client = MonitoringClient(
            prometheus_url="http://localhost:9090",
            grafana_url="http://localhost:3000"
        )
    
    async def get_system_health(self):
        """Get current system health metrics."""
        metrics = await self.client.query_prometheus({
            'constitutional_status': 'nfcs_constitutional_compliance',
            'kuramoto_sync': 'nfcs_kuramoto_sync_level',
            'response_time': 'rate(nfcs_request_duration_seconds_sum[5m])',
            'error_rate': 'rate(nfcs_errors_total[5m])'
        })
        return metrics
    
    async def create_alert(self, severity, message, context=None):
        """Create monitoring alert."""
        await self.client.send_alert({
            'severity': severity,
            'message': message,
            'context': context or {},
            'timestamp': time.time()
        })
```

## 🔗 Related Documentation
- [MVP Web Interface](../mvp_web_interface.py)
- [System Architecture](../docs/README.md)
- [Production Deployment](../DEPLOYMENT.md)
- [Configuration Guide](../config/README.md)

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.2 | 2025-09-15 | Comprehensive monitoring documentation overhaul | GitHub Copilot |
| 1.1 | 2025-09-14 | Added production monitoring stack details | Team Ω |
| 1.0 | 2025-09-12 | Initial monitoring system documentation | Team Ω |

---

*This monitoring system provides comprehensive observability for the Vortex-Omega NFCS across all deployment scenarios. For more information, see our [main documentation](../docs/README.md) and [architecture guide](../ARCHITECTURE.md).*

_Last updated: 2025-09-15 by GitHub Copilot Assistant_