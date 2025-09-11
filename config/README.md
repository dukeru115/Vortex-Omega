# Configuration Files - NFCS System Configuration

## Overview

This directory contains configuration files, templates, and settings for the Neural Field Control System (NFCS). These files define system parameters, operational modes, module configurations, and deployment settings.

**Purpose**: Centralized configuration management for all NFCS components and deployment scenarios.

## 📁 Configuration Structure

```
config/
├── production/              # 🚀 Production deployment configurations
│   ├── production.yaml      # Main production configuration
│   ├── security.yaml       # Security and safety settings
│   └── monitoring.yaml     # Performance monitoring setup
├── development/            # 🔧 Development environment settings
│   ├── development.yaml    # Development configuration
│   ├── debug.yaml         # Debug mode settings
│   └── testing.yaml       # Testing configuration
├── examples/               # 📋 Example configurations and templates
│   ├── basic_config.yaml  # Basic setup example
│   ├── advanced_config.yaml # Advanced features example
│   └── custom_template.yaml # Template for custom configs
└── README.md              # 📄 This documentation
```

## ⚙️ Configuration Categories

### 1. **System Configuration**
```yaml
# Basic system settings
system:
  name: "NFCS-Production"
  version: "1.0.0"
  environment: "production"
  
orchestrator:
  coordination_frequency: 10.0  # Hz
  operational_mode: "supervised"  # autonomous, supervised, manual
  safety_level: 0.8
  max_concurrent_operations: 100

logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "structured"
  output: ["console", "file", "syslog"]
  retention_days: 30
```

### 2. **Module Configuration**
```yaml
cognitive_modules:
  constitution:
    enabled: true
    enforcement_level: 0.9
    policy_update_frequency: 3600  # seconds
    
  boundary:
    enabled: true
    adaptive_boundaries: true
    safety_margin: 0.1
    
  memory:
    enabled: true
    max_memory_size: "4GB"
    retention_policy: "semantic_priority"
    compression: true
    
  meta_reflection:
    enabled: true
    reflection_interval: 60  # seconds
    adaptation_threshold: 0.7
    
  freedom:
    enabled: true
    autonomy_level: 0.6
    human_oversight: true
```

### 3. **Mathematical Core Settings**
```yaml
mathematical_core:
  cgl_solver:
    grid_size: [128, 128]
    domain_size: [10.0, 10.0]
    time_step: 0.01
    boundary_conditions: "periodic"
    
  kuramoto:
    coupling_strength: 2.0
    adaptation_rate: 0.1
    synchronization_threshold: 0.8
    
  metrics:
    calculation_interval: 0.1
    topological_analysis: true
    stability_monitoring: true
```

### 4. **ESC System Configuration**
```yaml
esc_system:
  token_processor:
    max_sequence_length: 8192
    batch_size: 32
    parallel_processing: true
    
  attention_mechanisms:
    num_heads: 8
    attention_dropout: 0.1
    local_attention_window: 256
    
  semantic_fields:
    embedding_dimension: 512
    field_stability_threshold: 0.7
    cross_field_interactions: true
    
  constitutional_filter:
    enforcement_mode: "strict"
    violation_threshold: 0.1
    escalation_policy: "immediate"
```

## 🚀 Quick Configuration Examples

### Basic Startup Configuration
```yaml
# config/basic_startup.yaml
system:
  name: "NFCS-Basic"
  operational_mode: "supervised"
  
orchestrator:
  coordination_frequency: 5.0
  safety_level: 0.9
  
modules:
  - constitution
  - boundary
  - memory
  
logging:
  level: "INFO"
  output: ["console"]
```

### Development Configuration
```yaml
# config/development.yaml  
system:
  name: "NFCS-Dev"
  environment: "development"
  debug_mode: true
  
orchestrator:
  coordination_frequency: 1.0  # Slower for debugging
  operational_mode: "manual"
  
logging:
  level: "DEBUG"
  output: ["console", "file"]
  file_path: "./logs/nfcs_dev.log"
  
performance:
  profiling_enabled: true
  memory_monitoring: true
  execution_tracing: false
```

### High-Performance Configuration
```yaml
# config/high_performance.yaml
system:
  name: "NFCS-HighPerf"
  
orchestrator:
  coordination_frequency: 20.0  # High frequency
  max_concurrent_operations: 500
  thread_pool_size: 16
  
mathematical_core:
  cgl_solver:
    grid_size: [256, 256]  # High resolution
    parallel_computation: true
    gpu_acceleration: true
    
performance_optimization:
  memory_pooling: true
  computation_caching: true
  predictive_loading: true
```

## 📋 Configuration Templates

### Template Usage
```bash
# Copy template for customization
cp config/examples/custom_template.yaml config/my_config.yaml

# Edit configuration
nano config/my_config.yaml

# Validate configuration
python src/main.py --validate-config config/my_config.yaml

# Use configuration
python src/main.py --config config/my_config.yaml
```

### Environment-Specific Configurations
```bash
# Development
python src/main.py --config config/development/development.yaml

# Testing
python src/main.py --config config/development/testing.yaml

# Production
python src/main.py --config config/production/production.yaml
```

## 🔧 Configuration Management

### Loading Configurations
```python
from orchestrator.managers.configuration_manager import ConfigurationManager

# Load configuration
config_manager = ConfigurationManager()
config = config_manager.load_config("config/production/production.yaml")

# Merge multiple configurations
base_config = config_manager.load_config("config/base.yaml")
env_config = config_manager.load_config("config/production/production.yaml")
merged_config = config_manager.merge_configs(base_config, env_config)

# Validate configuration
validation_result = config_manager.validate_config(config)
if not validation_result.is_valid:
    print("Configuration errors:", validation_result.errors)
```

### Dynamic Configuration Updates
```python
# Update configuration at runtime
await orchestrator.update_configuration({
    "orchestrator": {
        "coordination_frequency": 15.0,
        "safety_level": 0.85
    }
})

# Get current configuration
current_config = await orchestrator.get_configuration()
```

## 🔐 Security Configuration

### Security Settings Example
```yaml
# config/production/security.yaml
security:
  authentication:
    enabled: true
    method: "token_based"
    token_expiry: 3600
    
  encryption:
    data_at_rest: true
    data_in_transit: true
    key_rotation_interval: 86400
    
  access_control:
    role_based_access: true
    audit_logging: true
    session_timeout: 1800
    
  constitutional_enforcement:
    strict_mode: true
    violation_reporting: true
    automatic_shutdown: true
```

## 📊 Monitoring Configuration

### Monitoring Setup
```yaml
# config/production/monitoring.yaml
monitoring:
  performance:
    enabled: true
    metrics_collection_interval: 10  # seconds
    retention_period: "30d"
    
    metrics:
      - response_time
      - throughput
      - memory_usage
      - cpu_utilization
      - error_rate
      
  health_checks:
    enabled: true
    check_interval: 30  # seconds
    timeout: 5  # seconds
    
    endpoints:
      - "/health"
      - "/metrics"  
      - "/status"
      
  alerting:
    enabled: true
    
    rules:
      - name: "high_response_time"
        condition: "avg_response_time > 1.0"
        severity: "warning"
        
      - name: "memory_usage_high" 
        condition: "memory_usage > 0.8"
        severity: "critical"
        
      - name: "constitutional_violation"
        condition: "violation_detected == true"
        severity: "critical"
        action: "immediate_shutdown"
```

---

## Russian Translation / Русский перевод

# Файлы конфигурации - Конфигурация системы NFCS

## Обзор

Данная директория содержит файлы конфигурации, шаблоны и настройки для Системы управления нейронными полями (NFCS). Эти файлы определяют параметры системы, операционные режимы, конфигурации модулей и настройки развертывания.

**Назначение**: Централизованное управление конфигурацией для всех компонентов NFCS и сценариев развертывания.

---

*This README provides comprehensive documentation for NFCS configuration management, covering all aspects from basic setup to advanced production deployments.*

*Данный README предоставляет исчерпывающую документацию для управления конфигурацией NFCS, охватывающую все аспекты от базовой настройки до продвинутых производственных развертываний.*