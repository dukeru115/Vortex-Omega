# 📊 NFCS Monitoring System

## Overview
Comprehensive monitoring infrastructure for the Neural Field Control System (NFCS) v2.4.3, providing real-time observability and alerting capabilities.

## 🏗️ Architecture

### Current MVP Monitoring
The MVP system includes built-in monitoring via:
- **MVP Controller**: Real-time system health tracking
- **Web Dashboard**: Live metrics visualization
- **Supervisor Monitoring**: Service health and auto-restart
- **Live Logging**: Real-time event streaming

### Production Monitoring Stack
```
📊 Grafana (Visualization)
    ↓
📈 Prometheus (Metrics Collection)
    ↓
🎯 NFCS Applications (Instrumented)
    ├── Constitutional Monitoring
    ├── ESC-Kuramoto Integration
    ├── Cognitive Modules
    └── Empirical Validation
```

## 📁 Directory Structure

### 📈 `prometheus/`
Prometheus monitoring configuration:
- **prometheus.yml**: Main Prometheus configuration
- **Rules**: Alerting and recording rules
- **Targets**: Service discovery configuration

### 📊 `grafana/`
Grafana dashboard and configuration:
- **dashboards/**: JSON dashboard definitions
- **datasources/**: Data source configurations
- **provisioning/**: Automated setup scripts

## 🎯 Key Metrics

### System Health Metrics
- **Constitutional Status**: Real-time compliance monitoring
- **Kuramoto Sync Levels**: Oscillator synchronization (0.0-1.0)
- **Validation Scores**: System performance metrics
- **Safety Violations**: Security and safety event counts
- **Cognitive Module Status**: Active module tracking (5/5)

### Performance Metrics
- **Request Latency**: API response times
- **Memory Usage**: System memory consumption
- **CPU Utilization**: Processing load metrics
- **Neural Field Computation**: FPS and throughput
- **Database Performance**: Query execution times

### Business Metrics
- **User Sessions**: Active user tracking
- **System Uptime**: Availability monitoring
- **Error Rates**: System error frequency
- **Prediction Accuracy**: ML model performance

## 🚨 Alerting Rules

### Critical Alerts
- **System Downtime**: Service unavailability
- **Constitutional Violations**: Safety breaches
- **Memory Leaks**: Excessive memory usage
- **High Error Rates**: Above threshold errors

### Warning Alerts
- **Performance Degradation**: Slow response times
- **Low Synchronization**: Kuramoto sync below threshold
- **Resource Usage**: High CPU/memory utilization

## 🚀 Quick Start

### MVP Monitoring
```bash
# Start MVP with built-in monitoring
./start_mvp.sh

# Access dashboard with live metrics
# http://localhost:5000
```

### Production Monitoring Setup
```bash
# Start monitoring stack
docker-compose -f monitoring/docker-compose.yml up -d

# Access Grafana
# http://localhost:3000
# Default: admin/admin

# Access Prometheus
# http://localhost:9090
```

## 📈 Dashboard Access

### MVP Dashboard (Active)
- **URL**: http://localhost:5000
- **Features**: Real-time NFCS monitoring
- **Live Demo**: https://5000-i3xy7hm4ybz4gfsijjc3h-6532622b.e2b.dev/

### Grafana Dashboards (Production)
- **NFCS Overview**: System-wide health dashboard
- **Constitutional Monitoring**: Compliance and safety metrics
- **Performance Analytics**: Detailed performance analysis
- **Alert Management**: Alert status and history

## 🔧 Configuration

### Prometheus Targets
```yaml
# NFCS application endpoints
- targets:
  - "localhost:8000"  # Main API
  - "localhost:5000"  # MVP Dashboard
  - "localhost:8765"  # Constitutional Monitor
```

### Grafana Data Sources
- **Prometheus**: Primary metrics source
- **Loki**: Log aggregation (optional)
- **Jaeger**: Distributed tracing (optional)

## 📝 Maintenance

### Log Rotation
- Logs are rotated daily
- Retention period: 30 days
- Compressed storage for historical logs

### Backup Strategy
- Dashboard configurations backed up daily
- Metrics retention: 15 days (default)
- Long-term storage via remote storage

## 🔍 Troubleshooting

### Common Issues
1. **Missing Metrics**: Check service instrumentation
2. **Dashboard Errors**: Verify data source connections
3. **Alert Flooding**: Review alert thresholds
4. **Performance Issues**: Check resource allocation

### Debug Commands
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Validate Grafana health
curl http://localhost:3000/api/health

# Test MVP monitoring
supervisorctl -c supervisord.conf status
```

## 🔗 Related Documentation
- [MVP Web Interface](../mvp_web_interface.py)
- [System Architecture](../docs/README.md)
- [Production Deployment](../README.md#deployment)

---
*Part of Vortex-Omega Neural Field Control System v2.4.3*