"""
Constitutional Integration Manager
===============================

Main integration manager for Constitutional Monitoring System that coordinates
between Constitutional Real-time Monitor, Early Warning System, and core NFCS modules.

This module provides:
1. Unified initialization and configuration management
2. Cross-system communication and coordination
3. Production deployment integration
4. Health monitoring and system status reporting
5. Integration with Docker and service discovery

Author: Team Omega (GenSpark AI Implementation)
License: CC BY-NC 4.0
Date: 2025-09-14
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import time
from dataclasses import asdict

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.constitutional_config import (
    get_production_config,
    ProductionConstitutionalConfig,
    EarlyWarningProductionConfig
)

from modules.constitutional_realtime import (
    ConstitutionalRealTimeMonitor,
    ConstitutionalConfiguration,
    ConstitutionalStatus,
    ThreatLevel
)

from modules.early_warning_system import (
    EarlyWarningSystem,
    EarlyWarningConfiguration,
    WarningLevel
)

logger = logging.getLogger(__name__)


class ConstitutionalIntegrationManager:
    """
    Main integration manager for Constitutional Monitoring System
    """
    
    def __init__(self):
        """Initialize integration manager"""
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Core components
        self.constitutional_monitor: Optional[ConstitutionalRealTimeMonitor] = None
        self.early_warning_system: Optional[EarlyWarningSystem] = None
        
        # Configuration
        self.const_config: Optional[ProductionConstitutionalConfig] = None
        self.ews_config: Optional[EarlyWarningProductionConfig] = None
        
        # Integration state
        self.system_metrics = {}
        self.integration_status = "initializing"
        self.start_time = time.time()
        
        # Health check state
        self.last_health_check = 0
        self.health_check_interval = 30.0  # 30 seconds
        
        # NFCS integration callbacks
        self.nfcs_callbacks = {}
        
        logger.info("Constitutional Integration Manager initialized")
    
    async def initialize(self):
        """Initialize all constitutional monitoring components"""
        try:
            print("🏛️  Initializing Constitutional Monitoring System...")
            
            # Load production configuration
            self.const_config, self.ews_config = get_production_config()
            
            # Convert to component-specific configurations
            const_monitor_config = self._create_constitutional_config()
            ews_system_config = self._create_early_warning_config()
            
            # Initialize Constitutional Monitor
            print("📊 Initializing Constitutional Real-time Monitor...")
            self.constitutional_monitor = ConstitutionalRealTimeMonitor(const_monitor_config)
            
            # Initialize Early Warning System
            print("⚠️  Initializing Early Warning System...")
            self.early_warning_system = EarlyWarningSystem(ews_system_config)
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            self.integration_status = "initialized"
            print("✅ Constitutional Monitoring System initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.integration_status = "failed"
            raise
    
    def _create_constitutional_config(self) -> ConstitutionalConfiguration:
        """Create ConstitutionalConfiguration from production config"""
        config = ConstitutionalConfiguration()
        
        # Copy settings from production config
        config.ha_warning_threshold = self.const_config.ha_warning_threshold
        config.ha_critical_threshold = self.const_config.ha_critical_threshold
        config.ha_emergency_threshold = self.const_config.ha_emergency_threshold
        config.ha_failure_threshold = self.const_config.ha_failure_threshold
        
        config.integrity_warning_threshold = self.const_config.integrity_warning_threshold
        config.integrity_critical_threshold = self.const_config.integrity_critical_threshold
        config.integrity_failure_threshold = self.const_config.integrity_failure_threshold
        
        config.monitoring_interval_ms = self.const_config.monitoring_interval_ms
        config.alert_evaluation_interval_ms = self.const_config.alert_evaluation_interval_ms
        config.metrics_retention_hours = self.const_config.metrics_retention_hours
        
        config.enable_websocket_dashboard = self.const_config.enable_websocket_dashboard
        config.dashboard_port = self.const_config.dashboard_port
        config.max_websocket_connections = self.const_config.max_websocket_connections
        
        config.enable_metrics_persistence = self.const_config.enable_metrics_persistence
        config.database_path = self.const_config.database_path
        config.log_level = self.const_config.log_level
        
        return config
    
    def _create_early_warning_config(self) -> EarlyWarningConfiguration:
        """Create EarlyWarningConfiguration from production config"""
        config = EarlyWarningConfiguration()
        
        # Copy settings from production config
        config.prediction_window_size = self.ews_config.prediction_window_size
        config.min_data_points = self.ews_config.min_data_points
        config.model_retrain_interval = self.ews_config.model_retrain_interval
        
        config.advisory_threshold_pct = self.ews_config.advisory_threshold_pct
        config.watch_threshold_pct = self.ews_config.watch_threshold_pct
        config.warning_threshold_pct = self.ews_config.warning_threshold_pct
        config.alert_threshold_pct = self.ews_config.alert_threshold_pct
        
        config.min_prediction_confidence = self.ews_config.min_prediction_confidence
        config.prediction_update_interval = self.ews_config.prediction_update_interval
        config.alert_suppression_time = self.ews_config.alert_suppression_time
        
        config.enable_constitutional_integration = self.ews_config.enable_constitutional_integration
        config.enable_predictive_protocols = self.ews_config.enable_predictive_protocols
        
        return config
    
    async def start_monitoring(self):
        """Start all constitutional monitoring systems"""
        if self.integration_status != "initialized":
            raise RuntimeError("System not properly initialized")
        
        try:
            print("🚀 Starting Constitutional Monitoring Systems...")
            
            # Start Constitutional Monitor
            await self.constitutional_monitor.start_monitoring(
                module_control_callback=self.handle_module_control,
                emergency_callback=self.handle_emergency_protocol,
                metrics_callback=self.get_system_metrics
            )
            
            # Start Early Warning System
            await self.early_warning_system.start_monitoring(
                constitutional_callback=self.handle_constitutional_query,
                emergency_callback=self.handle_emergency_protocol
            )
            
            self.running = True
            self.integration_status = "running"
            
            # Start health monitoring task
            asyncio.create_task(self.health_monitoring_loop())
            
            # Start metrics integration task
            asyncio.create_task(self.metrics_integration_loop())
            
            print("✅ All constitutional monitoring systems started successfully")
            print(f"📊 WebSocket Dashboard: http://{self.const_config.dashboard_host}:{self.const_config.dashboard_port}")
            print(f"📈 Dashboard File: {Path(__file__).parent.parent.parent}/dashboard/constitutional_monitor.html")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.integration_status = "failed"
            raise
    
    async def stop_monitoring(self):
        """Stop all constitutional monitoring systems"""
        print("🛑 Stopping Constitutional Monitoring Systems...")
        
        self.running = False
        self.shutdown_event.set()
        
        # Stop components
        if self.constitutional_monitor:
            await self.constitutional_monitor.stop_monitoring()
        
        if self.early_warning_system:
            await self.early_warning_system.stop_monitoring()
        
        self.integration_status = "stopped"
        print("✅ All constitutional monitoring systems stopped")
    
    # Integration callback methods
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Provide current system metrics to constitutional monitor"""
        # This would integrate with actual NFCS metrics collection
        # For now, return stored metrics or defaults
        current_time = time.time()
        
        if not self.system_metrics:
            # Default metrics when no NFCS integration
            return {
                'hallucination_number': 0.1,
                'coherence_measure': 0.95,
                'defect_density': 0.005,
                'field_energy': 120.0,
                'integrity_score': 0.98,
                'constitutional_compliance': 0.96,
                'system_stability': 0.94,
                'processing_latency_ms': 45.0,
                'memory_usage_mb': 256.0,
                'cpu_usage_percent': 15.0,
                'timestamp': current_time
            }
        
        # Update timestamp
        self.system_metrics['timestamp'] = current_time
        return self.system_metrics
    
    async def handle_module_control(self, control_signals: Dict[str, Any]):
        """Handle module control signals from constitutional monitor"""
        logger.info(f"Received module control signals: {list(control_signals.keys())}")
        
        # Forward to NFCS modules if callbacks are registered
        for module_name, signal in control_signals.items():
            if module_name in self.nfcs_callbacks:
                try:
                    callback = self.nfcs_callbacks[module_name]
                    if asyncio.iscoroutinefunction(callback):
                        await callback(signal)
                    else:
                        callback(signal)
                except Exception as e:
                    logger.error(f"Module control callback error for {module_name}: {e}")
        
        # Log control action for monitoring
        control_log = {
            'timestamp': time.time(),
            'action': 'module_control',
            'signals': list(control_signals.keys()),
            'emergency': 'kuramoto_all' in control_signals
        }
        
        # Store control history for analysis
        self._log_control_action(control_log)
    
    async def handle_emergency_protocol(self, action: Dict[str, Any]):
        """Handle emergency protocol activation/deactivation"""
        if action.get('activate'):
            logger.critical("🚨 EMERGENCY PROTOCOLS ACTIVATED")
            # Notify external systems, trigger alerts, etc.
            await self._activate_emergency_procedures()
        else:
            logger.info("✅ Emergency protocols deactivated - entering recovery mode")
            await self._deactivate_emergency_procedures()
    
    async def handle_constitutional_query(self, query: str) -> Any:
        """Handle queries from early warning system"""
        if query == 'get_thresholds':
            return {
                'ha_critical': self.const_config.ha_critical_threshold,
                'ha_warning': self.const_config.ha_warning_threshold,
                'integrity_critical': self.const_config.integrity_critical_threshold,
                'integrity_warning': self.const_config.integrity_warning_threshold
            }
        
        elif query == 'get_system_status':
            return await self.get_integration_status()
        
        return None
    
    # Integration loops
    
    async def health_monitoring_loop(self):
        """Monitor system health and report status"""
        while self.running:
            try:
                current_time = time.time()
                
                if (current_time - self.last_health_check) >= self.health_check_interval:
                    await self._perform_health_check()
                    self.last_health_check = current_time
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10.0)
    
    async def metrics_integration_loop(self):
        """Integrate metrics between systems"""
        while self.running:
            try:
                # Update early warning system with latest metrics
                if self.early_warning_system:
                    metrics = await self.get_system_metrics()
                    await self.early_warning_system.update_metrics(metrics)
                
                await asyncio.sleep(2.0)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Metrics integration error: {e}")
                await asyncio.sleep(5.0)
    
    # Public API methods
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Get component statuses
        const_status = None
        ews_status = None
        
        if self.constitutional_monitor:
            const_status = self.constitutional_monitor.get_current_status()
        
        if self.early_warning_system:
            ews_status = self.early_warning_system.get_system_status()
        
        return {
            'integration_status': self.integration_status,
            'uptime_seconds': uptime,
            'running': self.running,
            'constitutional_monitor': const_status,
            'early_warning_system': ews_status,
            'configuration': {
                'ha_thresholds': {
                    'warning': self.const_config.ha_warning_threshold,
                    'critical': self.const_config.ha_critical_threshold,
                    'emergency': self.const_config.ha_emergency_threshold
                },
                'dashboard_port': self.const_config.dashboard_port,
                'websocket_enabled': self.const_config.enable_websocket_dashboard
            },
            'last_health_check': self.last_health_check,
            'timestamp': current_time
        }
    
    def register_nfcs_callback(self, module_name: str, callback):
        """Register NFCS module callback for control signals"""
        self.nfcs_callbacks[module_name] = callback
        logger.info(f"Registered NFCS callback for module: {module_name}")
    
    def update_system_metrics(self, metrics: Dict[str, Any]):
        """Update system metrics from external source"""
        self.system_metrics.update(metrics)
    
    # Private helper methods
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop_monitoring())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        health_issues = []
        
        # Check Constitutional Monitor
        if not self.constitutional_monitor or not self.constitutional_monitor.monitoring_active:
            health_issues.append("Constitutional monitor not active")
        
        # Check Early Warning System
        if not self.early_warning_system or not self.early_warning_system.warning_active:
            health_issues.append("Early warning system not active")
        
        # Check WebSocket dashboard
        if (self.const_config.enable_websocket_dashboard and 
            self.constitutional_monitor and 
            not self.constitutional_monitor.websocket_server):
            health_issues.append("WebSocket dashboard not running")
        
        # Log health status
        if health_issues:
            logger.warning(f"Health check issues: {health_issues}")
        else:
            logger.debug("Health check passed - all systems operational")
    
    async def _activate_emergency_procedures(self):
        """Activate emergency procedures"""
        # This would integrate with external alerting systems
        logger.critical("Activating emergency procedures")
        
        # Could send notifications, trigger external systems, etc.
        pass
    
    async def _deactivate_emergency_procedures(self):
        """Deactivate emergency procedures"""
        logger.info("Deactivating emergency procedures")
        pass
    
    def _log_control_action(self, action_log: Dict[str, Any]):
        """Log control actions for analysis"""
        # Could store in database, send to external logging system, etc.
        logger.info(f"Control action logged: {action_log}")


# Production deployment functions

async def start_constitutional_monitoring_service():
    """Start constitutional monitoring as a service"""
    print("🏛️  Starting Constitutional Monitoring Service")
    print("=" * 50)
    
    # Initialize integration manager
    manager = ConstitutionalIntegrationManager()
    
    try:
        # Initialize and start
        await manager.initialize()
        await manager.start_monitoring()
        
        print("🎯 Constitutional Monitoring Service started successfully")
        print("📋 System Status:")
        
        status = await manager.get_integration_status()
        print(f"   • Integration Status: {status['integration_status']}")
        print(f"   • Dashboard Port: {status['configuration']['dashboard_port']}")
        print(f"   • WebSocket Enabled: {status['configuration']['websocket_enabled']}")
        
        # Keep service running
        await manager.shutdown_event.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Service interrupted by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        raise
    finally:
        await manager.stop_monitoring()
        print("✅ Constitutional Monitoring Service stopped")


if __name__ == "__main__":
    # Run as standalone service
    asyncio.run(start_constitutional_monitoring_service())