# ⚙️ NFCS Configuration Management

## Overview
Centralized configuration system for the Neural Field Control System (NFCS) v2.4.3, managing all system settings, parameters, and environment-specific configurations.

## 🏗️ Configuration Architecture

### Configuration Hierarchy
```
Production Config (Environment Variables)
    ↓
Application Config (YAML/JSON files)
    ↓
Module Defaults (Python classes)
    ↓
System Defaults (Hardcoded values)
```

## 📁 Directory Structure

### Current Files
- **settings.py**: Main configuration management
- **database.py**: Database connection settings
- **security.py**: Security and authentication config
- **logging.py**: Logging configuration
- **environments/**: Environment-specific settings

## 🎯 Configuration Categories

### 🧠 Core System Configuration

#### NFCS Parameters
```python
class NFCSConfig:
    # Neural Field Parameters
    field_resolution: int = 128
    time_step: float = 0.01
    cuda_enabled: bool = True
    
    # Constitutional Monitoring
    ha_threshold: float = 0.85
    violation_sensitivity: float = 0.1
    emergency_shutdown: bool = True
```

#### ESC-Kuramoto Settings
```python
class KuramotoConfig:
    # Oscillator Network
    n_oscillators: int = 64
    coupling_strength: float = 0.5
    natural_frequency_std: float = 0.1
    
    # Semantic Integration
    semantic_embedding_dim: int = 768
    attention_heads: int = 12
    transformer_layers: int = 6
```

### 🏛️ Constitutional Framework
```python
class ConstitutionalConfig:
    # Monitoring Settings
    real_time_monitoring: bool = True
    compliance_check_interval: float = 0.1
    violation_log_retention: int = 30
    
    # Emergency Protocols
    auto_shutdown_enabled: bool = True
    escalation_thresholds: Dict[str, float] = {
        'minor': 0.3,
        'moderate': 0.6,
        'major': 0.8,
        'critical': 0.95
    }
```

### 🧠 Cognitive Module Settings
```python
class CognitiveConfig:
    # Module Activation
    constitution_module: bool = True
    symbolic_ai_module: bool = True
    memory_module: bool = True
    reflection_module: bool = True
    freedom_module: bool = True
    
    # Memory System
    memory_capacity: int = 10000
    forgetting_factor: float = 0.01
    consolidation_threshold: float = 0.7
```

### 🌐 Web Interface Configuration
```python
class WebConfig:
    # Flask Settings
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    
    # Socket.IO
    cors_allowed_origins: str = "*"
    async_mode: str = "threading"
    
    # Security
    secret_key: str = "nfcs-secret-key"
    session_timeout: int = 3600
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/nfcs
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=jwt-secret-key

# NFCS Settings
NFCS_CUDA_ENABLED=true
NFCS_DEBUG_MODE=false
CONSTITUTIONAL_MONITORING=true

# Web Interface
WEB_HOST=0.0.0.0
WEB_PORT=5000
FLASK_ENV=production
```

### Configuration Files

#### `nfcs.yaml` - Main Configuration
```yaml
nfcs:
  core:
    field_resolution: 128
    time_step: 0.01
    cuda_enabled: true
  
  constitutional:
    monitoring_enabled: true
    ha_threshold: 0.85
    real_time_alerts: true
  
  kuramoto:
    n_oscillators: 64
    coupling_strength: 0.5
    semantic_integration: true
  
  cognitive:
    modules_enabled:
      - constitution
      - symbolic_ai
      - memory
      - reflection
      - freedom
```

#### `environments/development.yaml`
```yaml
debug: true
log_level: DEBUG
database_pool_size: 5
cache_ttl: 60
```

#### `environments/production.yaml`
```yaml
debug: false
log_level: INFO
database_pool_size: 20
cache_ttl: 3600
ssl_required: true
```

## 🚀 Usage Examples

### Loading Configuration
```python
from src.config.settings import get_settings

# Get current configuration
config = get_settings()

# Access nested settings
nfcs_config = config.nfcs
web_config = config.web
db_config = config.database
```

### Dynamic Configuration Updates
```python
from src.config.manager import ConfigManager

# Update configuration at runtime
config_manager = ConfigManager()
config_manager.update_setting('nfcs.ha_threshold', 0.90)

# Reload configuration
config_manager.reload_config()
```

### Environment-Specific Loading
```python
import os
from src.config.settings import load_config

# Load based on environment
env = os.getenv('ENVIRONMENT', 'development')
config = load_config(environment=env)
```

## 🔒 Security Configuration

### Secrets Management
```python
class SecurityConfig:
    # Encryption
    encryption_key: SecretStr
    database_password: SecretStr
    redis_password: SecretStr
    
    # JWT Configuration
    jwt_secret_key: SecretStr
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    
    # API Security
    rate_limit_per_minute: int = 60
    cors_origins: List[str] = ["http://localhost:3000"]
```

### SSL/TLS Settings
```python
class SSLConfig:
    ssl_enabled: bool = True
    ssl_cert_path: str = "/etc/ssl/certs/nfcs.crt"
    ssl_key_path: str = "/etc/ssl/private/nfcs.key"
    ssl_ca_path: str = "/etc/ssl/certs/ca.crt"
```

## 📊 Performance Configuration

### Resource Management
```python
class PerformanceConfig:
    # Threading
    max_workers: int = 4
    worker_timeout: int = 30
    
    # Memory Management
    max_memory_usage: str = "2GB"
    gc_threshold: int = 1000
    
    # Caching
    cache_size: int = 1000
    cache_ttl: int = 3600
```

### Database Optimization
```python
class DatabaseConfig:
    # Connection Pool
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    
    # Query Optimization
    query_timeout: int = 30
    batch_size: int = 100
    enable_autocommit: bool = False
```

## 🔧 Development Tools

### Configuration Validation
```python
from src.config.validator import validate_config

# Validate configuration on startup
config = get_settings()
validation_result = validate_config(config)

if not validation_result.is_valid:
    raise ConfigurationError(validation_result.errors)
```

### Configuration Testing
```python
# Test configuration loading
def test_config_loading():
    config = get_settings()
    assert config.nfcs.field_resolution == 128
    assert config.web.port == 5000
```

### Hot Reloading (Development)
```python
from src.config.watcher import ConfigWatcher

# Watch for configuration changes
watcher = ConfigWatcher()
watcher.start()  # Auto-reload on file changes
```

## 📈 Monitoring Configuration

### Logging Configuration
```python
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "/var/log/nfcs/app.log"
    max_file_size: str = "10MB"
    backup_count: int = 5
```

### Metrics Configuration
```python
class MetricsConfig:
    enabled: bool = True
    prometheus_port: int = 9090
    collection_interval: int = 30
    retention_days: int = 30
```

## 🔗 Integration Points

### MVP Integration
The current MVP system uses configuration for:
- **Web Interface Settings**: Host, port, debug mode
- **Supervisor Configuration**: Service management
- **System Parameters**: NFCS core settings

### Production Integration
Full production system will use:
- **Database Configuration**: Connection strings and pool settings
- **Security Settings**: Authentication and authorization
- **Performance Tuning**: Resource limits and optimization
- **Monitoring Settings**: Observability configuration

## 📝 Best Practices

### Configuration Guidelines
1. **Use Environment Variables**: For sensitive data
2. **Validate Settings**: Check configuration on startup
3. **Document Defaults**: Clear documentation for all settings
4. **Environment Separation**: Different configs per environment
5. **Version Control**: Track configuration changes

### Security Best Practices
1. **Never Commit Secrets**: Use environment variables or secret management
2. **Encrypt Sensitive Data**: Use proper encryption for secrets
3. **Least Privilege**: Minimal permissions for each component
4. **Regular Updates**: Keep security settings current

## 🔗 Related Documentation
- [System Core](../core/README.md)
- [API Configuration](../api/README.md)
- [Production Deployment](../../README.md#deployment)

---
*Part of Vortex-Omega Neural Field Control System v2.4.3*