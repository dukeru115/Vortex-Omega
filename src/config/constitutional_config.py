"""
Constitutional Monitoring Configuration
=====================================

Configuration module for Constitutional Real-time Monitoring System
with production-ready defaults and environment-based overrides.

Author: Team Omega (GenSpark AI Implementation)
License: CC BY-NC 4.0
Date: 2025-09-14
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ProductionConstitutionalConfig:
    """Production-ready constitutional monitoring configuration"""

    # Core monitoring settings
    monitoring_interval_ms: int = 1000  # 1 Hz for production stability
    alert_evaluation_interval_ms: int = 2000  # 0.5 Hz alert evaluation
    metrics_retention_hours: int = 24

    # Hallucination Number thresholds
    ha_warning_threshold: float = float(os.getenv("HA_WARNING_THRESHOLD", "1.0"))
    ha_critical_threshold: float = float(os.getenv("HA_CRITICAL_THRESHOLD", "2.0"))
    ha_emergency_threshold: float = float(os.getenv("HA_EMERGENCY_THRESHOLD", "4.0"))
    ha_failure_threshold: float = float(os.getenv("HA_FAILURE_THRESHOLD", "7.0"))

    # Integrity thresholds
    integrity_warning_threshold: float = float(os.getenv("INTEGRITY_WARNING_THRESHOLD", "0.7"))
    integrity_critical_threshold: float = float(os.getenv("INTEGRITY_CRITICAL_THRESHOLD", "0.5"))
    integrity_failure_threshold: float = float(os.getenv("INTEGRITY_FAILURE_THRESHOLD", "0.3"))

    # Emergency protocol settings
    emergency_desync_amplitude: float = -1.0
    emergency_timeout_seconds: float = 30.0
    recovery_assessment_interval: float = 5.0

    # Performance thresholds
    max_processing_latency_ms: float = 2000.0  # Relaxed for production
    max_memory_usage_mb: float = 4096.0
    max_cpu_usage_percent: float = 85.0

    # WebSocket dashboard configuration
    enable_websocket_dashboard: bool = os.getenv("WEBSOCKET_DASHBOARD", "true").lower() == "true"
    dashboard_port: int = int(os.getenv("CONSTITUTIONAL_DASHBOARD_PORT", "8765"))
    dashboard_host: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")  # Bind to all interfaces
    max_websocket_connections: int = int(os.getenv("MAX_WEBSOCKET_CONNECTIONS", "50"))

    # Database and persistence
    enable_metrics_persistence: bool = (
        os.getenv("ENABLE_CONSTITUTIONAL_MONITORING", "true").lower() == "true"
    )
    database_path: str = os.getenv(
        "CONSTITUTIONAL_DB_PATH", "/app/data/constitutional_monitoring.db"
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Security settings
    websocket_origin_whitelist: Optional[list] = None  # Allow all origins in production
    enable_cors: bool = True
    max_message_size: int = 1024 * 1024  # 1MB max message size

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "monitoring_interval_ms": self.monitoring_interval_ms,
            "alert_evaluation_interval_ms": self.alert_evaluation_interval_ms,
            "ha_warning_threshold": self.ha_warning_threshold,
            "ha_critical_threshold": self.ha_critical_threshold,
            "ha_emergency_threshold": self.ha_emergency_threshold,
            "integrity_warning_threshold": self.integrity_warning_threshold,
            "integrity_critical_threshold": self.integrity_critical_threshold,
            "enable_websocket_dashboard": self.enable_websocket_dashboard,
            "dashboard_port": self.dashboard_port,
            "dashboard_host": self.dashboard_host,
            "database_path": self.database_path,
            "enable_metrics_persistence": self.enable_metrics_persistence,
        }

    @classmethod
    def from_environment(cls) -> "ProductionConstitutionalConfig":
        """Create configuration from environment variables"""
        return cls()

    def validate(self) -> bool:
        """Validate configuration values"""
        # Validate thresholds are in correct order
        if not (
            0 < self.ha_warning_threshold < self.ha_critical_threshold < self.ha_emergency_threshold
        ):
            raise ValueError("Ha thresholds must be in ascending order")

        if not (0 < self.integrity_critical_threshold < self.integrity_warning_threshold < 1.0):
            raise ValueError("Integrity thresholds must be in descending order")

        # Validate ports
        if not (1024 <= self.dashboard_port <= 65535):
            raise ValueError("Dashboard port must be between 1024-65535")

        return True


@dataclass
class EarlyWarningProductionConfig:
    """Production configuration for Early Warning System"""

    # Prediction model settings
    prediction_window_size: int = 120  # 2 minutes of data points
    min_data_points: int = 30
    model_retrain_interval: int = 7200  # Retrain every 2 hours in production

    # Warning thresholds (% of critical thresholds)
    advisory_threshold_pct: float = 0.7  # 70% of critical threshold
    watch_threshold_pct: float = 0.8  # 80% of critical threshold
    warning_threshold_pct: float = 0.9  # 90% of critical threshold
    alert_threshold_pct: float = 0.95  # 95% of critical threshold

    # Prediction confidence requirements
    min_prediction_confidence: float = 0.8  # Higher confidence for production
    high_confidence_threshold: float = 0.95
    uncertainty_warning_threshold: float = 0.25

    # Anomaly detection settings
    anomaly_contamination: float = 0.05  # Lower contamination for production
    anomaly_sensitivity: float = 0.9  # Higher sensitivity
    pattern_correlation_threshold: float = 0.8

    # Time-based settings
    prediction_update_interval: float = 10.0  # Update every 10 seconds
    alert_suppression_time: float = 60.0  # Suppress duplicates for 1 minute
    trend_analysis_window: int = 600  # 10 minutes for trend analysis

    # Integration settings
    enable_constitutional_integration: bool = True
    enable_predictive_protocols: bool = os.getenv("ENABLE_EARLY_WARNING", "true").lower() == "true"
    enable_pattern_learning: bool = True

    def validate(self) -> bool:
        """Validate early warning configuration"""
        if not (0 < self.min_prediction_confidence < 1.0):
            raise ValueError("Prediction confidence must be between 0-1")

        if self.min_data_points < 10:
            raise ValueError("Minimum data points must be at least 10")

        return True


def get_production_config() -> tuple:
    """Get production-ready configuration for both systems"""

    # Validate environment
    required_env_vars = ["NFCS_ENV", "CONSTITUTIONAL_DB_PATH"]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"⚠️  Missing environment variables: {missing_vars}")
        print("Using default values...")

    # Create configurations
    const_config = ProductionConstitutionalConfig.from_environment()
    ews_config = EarlyWarningProductionConfig()

    # Validate configurations
    try:
        const_config.validate()
        ews_config.validate()
        print("✅ Configuration validation passed")
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
        raise

    return const_config, ews_config


def create_docker_environment_template() -> str:
    """Create Docker environment template"""
    template = """
# Constitutional Monitoring Docker Environment
# Copy this to .env and customize as needed

# NFCS Environment
NFCS_ENV=production
LOG_LEVEL=INFO

# Constitutional Monitoring
ENABLE_CONSTITUTIONAL_MONITORING=true
CONSTITUTIONAL_DB_PATH=/app/data/constitutional_monitoring.db
CONSTITUTIONAL_DASHBOARD_PORT=8765
DASHBOARD_HOST=0.0.0.0

# Thresholds
HA_WARNING_THRESHOLD=1.0
HA_CRITICAL_THRESHOLD=2.0  
HA_EMERGENCY_THRESHOLD=4.0
HA_FAILURE_THRESHOLD=7.0

INTEGRITY_WARNING_THRESHOLD=0.7
INTEGRITY_CRITICAL_THRESHOLD=0.5
INTEGRITY_FAILURE_THRESHOLD=0.3

# Early Warning System
ENABLE_EARLY_WARNING=true
WEBSOCKET_DASHBOARD=true
MAX_WEBSOCKET_CONNECTIONS=50

# Security
ENABLE_CORS=true
MAX_MESSAGE_SIZE=1048576

# Performance
MAX_PROCESSING_LATENCY_MS=2000
MAX_MEMORY_USAGE_MB=4096
MAX_CPU_USAGE_PERCENT=85
"""
    return template.strip()


if __name__ == "__main__":
    # Test configuration loading
    print("🧪 Testing Constitutional Monitoring Configuration")

    try:
        const_config, ews_config = get_production_config()

        print("\n📋 Constitutional Monitor Configuration:")
        for key, value in const_config.to_dict().items():
            print(f"  {key}: {value}")

        print("\n📋 Early Warning System Configuration:")
        print(f"  min_data_points: {ews_config.min_data_points}")
        print(f"  prediction_update_interval: {ews_config.prediction_update_interval}")
        print(f"  min_prediction_confidence: {ews_config.min_prediction_confidence}")
        print(f"  enable_predictive_protocols: {ews_config.enable_predictive_protocols}")

        print("\n✅ Configuration test completed successfully")

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

    print("\n📄 Docker Environment Template:")
    print(create_docker_environment_template())
