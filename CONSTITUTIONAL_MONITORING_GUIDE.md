# Constitutional Monitoring System Guide

## Overview

The Neural Field Control System (NFCS) v2.4.3 Constitutional Monitoring System provides real-time oversight and early warning capabilities for AI systems implementing the Costly Coherence theory and Symbolic-Neural Bridge architecture.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Constitutional Monitoring System          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────────┐  │
│  │ Constitutional      │    │ Early Warning System    │  │
│  │ Real-time Monitor   │◄──►│                         │  │
│  │ (Algorithm 1)       │    │ (Predictive Analytics)  │  │
│  └─────────────────────┘    └─────────────────────────┘  │
│              │                            │              │
│              ▼                            ▼              │
│  ┌─────────────────────────────────────────────────────┐  │
│  │          WebSocket Dashboard                        │  │
│  │     (Real-time Monitoring Interface)                │  │
│  └─────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                    NFCS Core Modules                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │   Memory    │ │     ESC     │ │ Symbolic-Neural     │ │
│  │   Module    │ │   Module    │ │     Bridge          │ │
│  └─────────────┘ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Production Deployment

```bash
# Clone repository
git clone https://github.com/dukeru115/Vortex-Omega.git
cd Vortex-Omega

# Start production deployment
chmod +x quick_start_production.sh
./quick_start_production.sh
```

### 2. Docker Deployment

```bash
# Build and start with Docker Compose
docker-compose up --build -d

# Check service health
docker-compose logs -f vortex-omega
```

### 3. Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install additional constitutional monitoring dependencies
pip install scikit-learn websockets matplotlib

# Run constitutional monitoring service
python src/modules/constitutional_integration_manager.py
```

## Configuration

### Environment Variables

```bash
# Core NFCS Settings
NFCS_ENV=production
LOG_LEVEL=INFO

# Constitutional Monitoring
ENABLE_CONSTITUTIONAL_MONITORING=true
CONSTITUTIONAL_DB_PATH=/app/data/constitutional_monitoring.db
CONSTITUTIONAL_DASHBOARD_PORT=8765
DASHBOARD_HOST=0.0.0.0

# Hallucination Number Thresholds
HA_WARNING_THRESHOLD=1.0
HA_CRITICAL_THRESHOLD=2.0
HA_EMERGENCY_THRESHOLD=4.0
HA_FAILURE_THRESHOLD=7.0

# Integrity Thresholds
INTEGRITY_WARNING_THRESHOLD=0.7
INTEGRITY_CRITICAL_THRESHOLD=0.5
INTEGRITY_FAILURE_THRESHOLD=0.3

# Early Warning System
ENABLE_EARLY_WARNING=true
WEBSOCKET_DASHBOARD=true
MAX_WEBSOCKET_CONNECTIONS=50

# Performance Settings
MAX_PROCESSING_LATENCY_MS=2000
MAX_MEMORY_USAGE_MB=4096
MAX_CPU_USAGE_PERCENT=85
```

### Configuration Files

Create `.env` file in project root:

```bash
cp .env.example .env
# Edit .env with your specific settings
```

## Service Endpoints

After deployment, the following endpoints will be available:

### Core Services
- **Vortex-Omega API**: `http://0.0.0.0:8080`
- **API Documentation**: `http://0.0.0.0:8080/docs`  
- **Health Check**: `http://0.0.0.0:8080/health`

### Monitoring Services
- **Constitutional Monitor**: `http://0.0.0.0:8765`
- **Grafana Dashboard**: `http://0.0.0.0:3000` (admin/vortex123)
- **Prometheus Metrics**: `http://0.0.0.0:9090`

### Dashboard Access
- **WebSocket Dashboard**: Open `dashboard/constitutional_monitor.html` in browser
- **Direct Dashboard File**: `file://$(pwd)/dashboard/constitutional_monitor.html`

## Features

### Constitutional Real-time Monitor

**Algorithm 1 Implementation**
- Real-time Hallucination Number (Ha) monitoring
- Constitutional compliance scoring
- Emergency desynchronization protocols
- Module control signal generation

**Key Capabilities:**
- ✅ Threshold-based monitoring (Warning → Critical → Emergency)
- ✅ Automated emergency protocol activation
- ✅ Real-time WebSocket dashboard
- ✅ SQLite persistence for metrics history
- ✅ Integration with core NFCS modules

### Early Warning System

**Predictive Analytics**
- Machine learning-based Ha trajectory prediction
- Multi-horizon forecasting (30s, 3min, 10min)
- Pattern recognition and anomaly detection
- Proactive alert generation

**Key Capabilities:**
- ✅ Trend-based prediction with confidence scoring
- ✅ Risk assessment with threat indicators
- ✅ Integration with constitutional monitoring
- ✅ Configurable alert thresholds

### WebSocket Dashboard

**Real-time Monitoring Interface**
- Live metrics visualization
- Ha trajectory charts with threshold lines
- Active alerts panel with severity indicators
- System performance monitoring

**Key Features:**
- ✅ Responsive design for desktop/mobile
- ✅ Real-time WebSocket updates
- ✅ Interactive controls for emergency management
- ✅ Historical metrics display

## Usage Examples

### 1. Interactive Demo

```bash
# Run interactive constitutional monitoring demo
python demo_constitutional_system.py
```

The demo showcases:
- Normal operation → Gradual degradation → Crisis → Recovery scenarios
- Real-time Algorithm 1 execution
- Emergency protocol demonstration
- WebSocket dashboard integration

### 2. Integration with NFCS Modules

```python
from src.modules.constitutional_integration_manager import ConstitutionalIntegrationManager

# Initialize integration manager
manager = ConstitutionalIntegrationManager()
await manager.initialize()

# Register NFCS module callbacks
manager.register_nfcs_callback('memory', memory_module_callback)
manager.register_nfcs_callback('esc', esc_module_callback)
manager.register_nfcs_callback('symbolic', symbolic_bridge_callback)

# Start monitoring
await manager.start_monitoring()

# Update system metrics
manager.update_system_metrics({
    'hallucination_number': 1.2,
    'integrity_score': 0.8,
    'coherence_measure': 0.75
})

# Get system status
status = await manager.get_integration_status()
print(f"Status: {status['integration_status']}")
```

### 3. Custom Threshold Configuration

```python
from src.config.constitutional_config import get_production_config

# Load configuration
const_config, ews_config = get_production_config()

# Customize thresholds
const_config.ha_warning_threshold = 0.8
const_config.ha_critical_threshold = 1.5
const_config.ha_emergency_threshold = 3.0

# Validate configuration
const_config.validate()
```

## Monitoring and Alerts

### Alert Levels

**Threat Levels:**
- `MINIMAL` (Ha < 0.5): Normal operation
- `LOW` (Ha 0.5-1.0): Enhanced monitoring
- `MODERATE` (Ha 1.0-2.0): Warning alerts
- `HIGH` (Ha 2.0-4.0): Critical alerts
- `CRITICAL` (Ha 4.0-7.0): Emergency protocols
- `EXTREME` (Ha > 7.0): System failure prevention

**Constitutional Status:**
- `NORMAL`: All systems operating within parameters
- `MONITORING`: Enhanced monitoring active
- `WARNING`: Warning thresholds exceeded
- `CRITICAL`: Critical intervention required
- `EMERGENCY`: Emergency protocols active
- `RECOVERY`: System recovery in progress
- `FAILURE`: System failure detected

### Emergency Protocols

When Ha exceeds emergency threshold (4.0), Algorithm 1 activates:

1. **Emergency Desynchronization**: All modules receive desync signals
2. **Module Control**: Targeted intervention signals sent to affected modules
3. **Dashboard Alerts**: Real-time notifications via WebSocket
4. **Recovery Assessment**: Continuous monitoring for recovery conditions

## Testing

### Unit Tests

```bash
# Run constitutional monitoring tests
python -m pytest tests/test_constitutional_integration.py -v

# Run with coverage
python -m pytest tests/test_constitutional_integration.py --cov=src/modules --cov-report=html
```

### Integration Tests

```bash
# Test complete system integration
python tests/test_constitutional_integration.py

# Test WebSocket dashboard
python -m pytest tests/test_constitutional_integration.py::TestWebSocketIntegration -v
```

### Performance Testing

```bash
# Stress test monitoring system
python -m pytest tests/test_constitutional_integration.py::TestStressAndResilience -v
```

## Production Deployment

### Docker Production Setup

1. **Build Production Image**:
```bash
docker build -t vortex-omega:latest .
```

2. **Deploy with Compose**:
```bash
# Production profile with all services
docker-compose --profile production up -d
```

3. **Health Checks**:
```bash
# Check service health
curl http://0.0.0.0:8080/health
curl http://0.0.0.0:8765

# Monitor logs
docker-compose logs -f vortex-omega
```

### Kubernetes Deployment

```yaml
# constitutional-monitoring-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vortex-omega-constitutional
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vortex-omega-constitutional
  template:
    metadata:
      labels:
        app: vortex-omega-constitutional
    spec:
      containers:
      - name: constitutional-monitor
        image: vortex-omega:latest
        ports:
        - containerPort: 8080
        - containerPort: 8765
        env:
        - name: ENABLE_CONSTITUTIONAL_MONITORING
          value: "true"
        - name: DASHBOARD_HOST
          value: "0.0.0.0"
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
          requests:
            memory: "1Gi"
            cpu: "500m"
```

### Service Discovery

```yaml
# constitutional-monitoring-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: constitutional-monitoring-service
spec:
  selector:
    app: vortex-omega-constitutional
  ports:
  - name: api
    port: 8080
    targetPort: 8080
  - name: dashboard
    port: 8765
    targetPort: 8765
  type: LoadBalancer
```

## Troubleshooting

### Common Issues

**WebSocket Connection Failed**
```bash
# Check if dashboard service is running
curl http://0.0.0.0:8765

# Verify WebSocket configuration
docker-compose logs vortex-omega | grep "WebSocket"
```

**High Memory Usage**
```bash
# Check metrics retention settings
grep "METRICS_RETENTION" .env

# Reduce retention period
export METRICS_RETENTION_HOURS=6
```

**Emergency Protocols Not Triggering**
```bash
# Verify thresholds in configuration
curl http://0.0.0.0:8080/config/thresholds

# Check Ha values in dashboard
curl http://0.0.0.0:8080/metrics | grep hallucination_number
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with detailed output
python src/modules/constitutional_integration_manager.py
```

### Health Checks

```bash
# System health endpoint
curl http://0.0.0.0:8080/health/constitutional

# Detailed status
curl http://0.0.0.0:8080/status/detailed
```

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Setup pre-commit hooks
pre-commit install

# Run development server
python src/modules/constitutional_integration_manager.py
```

### Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/constitutional-enhancement`
3. Make changes with tests
4. Run test suite: `pytest`
5. Submit pull request

### Code Structure

```
src/
├── config/
│   └── constitutional_config.py           # Configuration management
├── modules/
│   ├── constitutional_realtime.py         # Algorithm 1 implementation
│   ├── early_warning_system.py            # Predictive analytics
│   └── constitutional_integration_manager.py  # Main coordinator
dashboard/
└── constitutional_monitor.html            # WebSocket dashboard
tests/
└── test_constitutional_integration.py     # Comprehensive test suite
```

## API Reference

### REST Endpoints

```bash
# Get system status
GET /status/constitutional

# Get current metrics  
GET /metrics/constitutional

# Get active alerts
GET /alerts/constitutional

# Force emergency mode (POST)
POST /emergency/activate
```

### WebSocket Events

```javascript
// Connect to dashboard
const ws = new WebSocket('ws://0.0.0.0:8765');

// Event types
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    switch(data.type) {
        case 'initial_state':
        case 'metrics_update':
        case 'alert_created':
        case 'alert_resolved':
        case 'emergency_activated':
    }
};
```

## License

CC BY-NC 4.0 - See LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: https://github.com/dukeru115/Vortex-Omega/issues
- Documentation: https://github.com/dukeru115/Vortex-Omega/wiki

---

**Constitutional Monitoring System v2.4.3**  
*Real-time oversight for Neural Field Control Systems*