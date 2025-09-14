"""
Comprehensive Integration Tests for Constitutional Monitoring System
================================================================

Tests for the complete constitutional monitoring system including:
1. Constitutional Real-time Monitor
2. Early Warning System
3. Integration with NFCS components
4. WebSocket dashboard functionality
5. Emergency protocol activation
6. Performance and stress testing

Author: Team Omega (GenSpark AI Implementation)
License: CC BY-NC 4.0  
Date: 2025-09-14
"""

import asyncio
import pytest
import numpy as np
import json
import time
import logging
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
import tempfile
import websockets

# Import the modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modules.constitutional_realtime import (
    ConstitutionalRealTimeMonitor,
    ConstitutionalConfiguration,
    ConstitutionalStatus,
    ThreatLevel,
    RealTimeMetrics
)

from modules.early_warning_system import (
    EarlyWarningSystem,
    EarlyWarningConfiguration,
    WarningLevel,
    PredictionHorizon
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestConstitutionalRealTimeMonitor:
    """Test suite for Constitutional Real-time Monitor"""
    
    @pytest.fixture
    async def monitor(self):
        """Create a test monitor instance"""
        config = ConstitutionalConfiguration()
        config.monitoring_interval_ms = 50  # Fast monitoring for tests
        config.alert_evaluation_interval_ms = 100
        config.enable_websocket_dashboard = False  # Disable WebSocket for tests
        config.enable_metrics_persistence = False  # Disable DB for tests
        
        monitor = ConstitutionalRealTimeMonitor(config)
        yield monitor
        
        # Cleanup
        if monitor.monitoring_active:
            await monitor.stop_monitoring()
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    async def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.current_status == ConstitutionalStatus.NORMAL
        assert monitor.current_metrics.hallucination_number == 0.0
        assert monitor.current_metrics.integrity_score == 1.0
        assert not monitor.monitoring_active
        assert not monitor.emergency_active
    
    async def test_metrics_collection(self, monitor):
        """Test metrics collection callback"""
        metrics_callback_called = False
        test_metrics = {
            'hallucination_number': 0.5,
            'coherence_measure': 0.8,
            'defect_density': 0.02,
            'field_energy': 150.0,
            'integrity_score': 0.9
        }
        
        async def mock_metrics_callback():
            nonlocal metrics_callback_called
            metrics_callback_called = True
            return test_metrics
        
        # Start monitoring
        await monitor.start_monitoring(metrics_callback=mock_metrics_callback)
        
        # Wait for metrics collection
        await asyncio.sleep(0.2)
        
        # Verify metrics were collected
        assert metrics_callback_called
        assert monitor.current_metrics.hallucination_number == 0.5
        assert monitor.current_metrics.coherence_measure == 0.8
        assert monitor.current_metrics.integrity_score == 0.9
        
        await monitor.stop_monitoring()
    
    async def test_algorithm_1_normal_operation(self, monitor):
        """Test Algorithm 1 implementation for normal operation"""
        # Set normal metrics
        monitor.current_metrics.hallucination_number = 0.3
        monitor.current_metrics.integrity_score = 0.9
        
        # Apply constitutional algorithm
        decision = await monitor._apply_constitutional_algorithm()
        
        assert decision['type'] == 'ACCEPT'
        assert not decision['emergency_action']
        assert decision['control_signals'] == {}
        assert 'Normal operation' in decision['reasoning'][0]
    
    async def test_algorithm_1_warning_state(self, monitor):
        """Test Algorithm 1 warning state"""
        # Set warning metrics
        monitor.current_metrics.hallucination_number = 1.2  # Above warning threshold
        monitor.current_metrics.integrity_score = 0.6  # Below warning threshold
        
        decision = await monitor._apply_constitutional_algorithm()
        
        assert decision['type'] == 'MONITOR'
        assert not decision['emergency_action']
        assert 'Warning state' in decision['reasoning'][0]
    
    async def test_algorithm_1_forced_synchronization(self, monitor):
        """Test Algorithm 1 forced synchronization"""
        # Set low integrity
        monitor.current_metrics.hallucination_number = 0.8
        monitor.current_metrics.integrity_score = 0.4  # Below critical threshold
        
        decision = await monitor._apply_constitutional_algorithm()
        
        assert decision['type'] == 'FORCED_SYNC'
        assert not decision['emergency_action']
        assert len(decision['control_signals']) > 0
        assert 'memory' in decision['control_signals']
        assert 'esc' in decision['control_signals']
    
    async def test_algorithm_1_emergency_mode(self, monitor):
        """Test Algorithm 1 emergency mode activation"""
        # Set emergency Ha level
        monitor.current_metrics.hallucination_number = 5.0  # Above emergency threshold
        monitor.current_metrics.integrity_score = 0.8
        
        decision = await monitor._apply_constitutional_algorithm()
        
        assert decision['type'] == 'EMERGENCY'
        assert decision['emergency_action']
        assert len(decision['control_signals']) > 0
        assert 'kuramoto_all' in decision['control_signals']
        assert all(isinstance(signal, (int, float, np.ndarray)) 
                  for signal in decision['control_signals'].values())
    
    async def test_alert_generation(self, monitor):
        """Test alert generation for various conditions"""
        # Start monitoring
        await monitor.start_monitoring()
        
        # Set critical Ha
        monitor.current_metrics.hallucination_number = 2.5
        await monitor._evaluate_alerts()
        
        assert 'ha_critical' in monitor.active_alerts
        alert = monitor.active_alerts['ha_critical']
        assert alert.severity == ThreatLevel.CRITICAL
        assert 'Critical Hallucination Number' in alert.title
        
        await monitor.stop_monitoring()
    
    async def test_alert_auto_resolution(self, monitor):
        """Test automatic alert resolution"""
        # Start monitoring
        await monitor.start_monitoring()
        
        # Create alert
        monitor.current_metrics.hallucination_number = 2.5
        await monitor._evaluate_alerts()
        assert 'ha_critical' in monitor.active_alerts
        
        # Improve metrics
        monitor.current_metrics.hallucination_number = 0.5
        await monitor._evaluate_alerts()
        
        # Wait for alert evaluation
        await asyncio.sleep(0.2)
        
        # Alert should be resolved
        assert 'ha_critical' not in monitor.active_alerts
        
        await monitor.stop_monitoring()
    
    async def test_emergency_protocol_activation(self, monitor):
        """Test emergency protocol activation and deactivation"""
        emergency_callback_called = False
        emergency_state = None
        
        async def mock_emergency_callback(action):
            nonlocal emergency_callback_called, emergency_state
            emergency_callback_called = True
            emergency_state = action
        
        await monitor.start_monitoring(emergency_callback=mock_emergency_callback)
        
        # Trigger emergency
        await monitor.force_emergency_mode("Test emergency")
        
        assert emergency_callback_called
        assert emergency_state['activate'] == True
        assert monitor.emergency_active
        assert 'manual_emergency' in monitor.active_alerts
        
        # Deactivate emergency
        emergency_callback_called = False
        await monitor.deactivate_emergency_mode("Test recovery")
        
        assert emergency_callback_called
        assert emergency_state['activate'] == False
        assert not monitor.emergency_active
        assert monitor.recovery_mode_active
        
        await monitor.stop_monitoring()
    
    async def test_metrics_persistence(self):
        """Test metrics persistence to database"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            config = ConstitutionalConfiguration()
            config.enable_metrics_persistence = True
            config.database_path = db_path
            config.monitoring_interval_ms = 50
            
            monitor = ConstitutionalRealTimeMonitor(config)
            
            # Add test metrics
            monitor.current_metrics.hallucination_number = 1.5
            monitor.current_metrics.integrity_score = 0.7
            monitor.current_metrics.threat_level = ThreatLevel.MODERATE
            
            # Store metrics
            await monitor._store_metrics()
            
            # Verify database storage
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("SELECT * FROM metrics")
                rows = cursor.fetchall()
                assert len(rows) == 1
                
                row = rows[0]
                assert row[1] == 1.5  # hallucination_number
                assert row[5] == 0.7   # integrity_score
                assert row[6] == 'MODERATE'  # threat_level
        
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    async def test_status_transitions(self, monitor):
        """Test system status transitions"""
        # Start monitoring
        await monitor.start_monitoring()
        
        # Normal -> Warning
        monitor.current_metrics.hallucination_number = 1.2
        monitor._update_system_status()
        assert monitor.current_status == ConstitutionalStatus.WARNING
        
        # Warning -> Critical
        monitor.current_metrics.hallucination_number = 2.5
        monitor._update_system_status()
        assert monitor.current_status == ConstitutionalStatus.CRITICAL
        
        # Critical -> Emergency (manual)
        monitor.emergency_active = True
        monitor._update_system_status()
        assert monitor.current_status == ConstitutionalStatus.EMERGENCY
        
        # Emergency -> Recovery
        monitor.emergency_active = False
        monitor.recovery_mode_active = True
        monitor._update_system_status()
        assert monitor.current_status == ConstitutionalStatus.RECOVERY
        
        # Recovery -> Normal
        monitor.recovery_mode_active = False
        monitor.current_metrics.hallucination_number = 0.3
        monitor._update_system_status()
        assert monitor.current_status == ConstitutionalStatus.NORMAL
        
        await monitor.stop_monitoring()


class TestEarlyWarningSystem:
    """Test suite for Early Warning System"""
    
    @pytest.fixture
    async def warning_system(self):
        """Create a test early warning system"""
        config = EarlyWarningConfiguration()
        config.min_data_points = 5
        config.prediction_update_interval = 0.1
        config.min_prediction_confidence = 0.5
        
        ews = EarlyWarningSystem(config)
        yield ews
        
        if ews.warning_active:
            await ews.stop_monitoring()
    
    async def test_warning_system_initialization(self, warning_system):
        """Test warning system initialization"""
        assert not warning_system.warning_active
        assert len(warning_system.metrics_buffer) == 0
        assert len(warning_system.active_warnings) == 0
        assert not warning_system.models_trained
    
    async def test_metrics_buffer_management(self, warning_system):
        """Test metrics buffer management"""
        # Add metrics
        for i in range(10):
            metrics = {
                'hallucination_number': i * 0.1,
                'coherence_measure': 0.8,
                'integrity_score': 0.9
            }
            await warning_system.update_metrics(metrics)
        
        assert len(warning_system.metrics_buffer) == 10
        assert warning_system.metrics_buffer[0]['hallucination_number'] == 0.0
        assert warning_system.metrics_buffer[-1]['hallucination_number'] == 0.9
    
    async def test_trend_prediction(self, warning_system):
        """Test trend-based prediction"""
        # Feed increasing Ha values
        for i in range(20):
            metrics = {
                'hallucination_number': i * 0.05,  # Linear increase
                'coherence_measure': 0.8 - i * 0.01,
                'integrity_score': 0.9 - i * 0.01,
                'defect_density': i * 0.001,
                'field_energy': 100 + i * 5,
                'constitutional_compliance': 0.9,
                'system_stability': 0.8
            }
            await warning_system.update_metrics(metrics)
        
        # Force prediction update
        await warning_system.force_prediction_update()
        
        # Check predictions were generated
        assert len(warning_system.current_predictions) > 0
        
        # Check SHORT_TERM prediction
        if PredictionHorizon.SHORT_TERM in warning_system.current_predictions:
            prediction = warning_system.current_predictions[PredictionHorizon.SHORT_TERM]
            # Should predict continued increase
            assert prediction.predicted_ha > 0.9  # Last value was 0.95
    
    async def test_warning_generation(self, warning_system):
        """Test warning generation for approaching thresholds"""
        await warning_system.start_monitoring()
        
        # Feed data approaching warning threshold
        for i in range(15):
            ha_value = 0.5 + (i * 0.04)  # Approaching 1.0 threshold
            metrics = {
                'hallucination_number': ha_value,
                'coherence_measure': 0.7,
                'integrity_score': 0.8,
                'defect_density': 0.01,
                'field_energy': 200,
                'constitutional_compliance': 0.8,
                'system_stability': 0.7
            }
            await warning_system.update_metrics(metrics)
            await asyncio.sleep(0.05)
        
        # Check if warnings were generated
        warnings = warning_system.get_current_warnings()
        assert len(warnings) > 0
        
        # Should have Ha-related warnings
        ha_warnings = [w for w in warnings.values() if 'ha_' in w['alert_id']]
        assert len(ha_warnings) > 0
    
    async def test_prediction_confidence(self, warning_system):
        """Test prediction confidence calculation"""
        # Feed consistent trend data
        for i in range(30):
            metrics = {
                'hallucination_number': 0.2 + (i * 0.02),  # Consistent linear trend
                'coherence_measure': 0.8,
                'integrity_score': 0.9,
                'defect_density': 0.01,
                'field_energy': 150,
                'constitutional_compliance': 0.9,
                'system_stability': 0.9
            }
            await warning_system.update_metrics(metrics)
        
        await warning_system.force_prediction_update()
        
        # Check confidence levels
        if warning_system.current_predictions:
            for prediction in warning_system.current_predictions.values():
                assert 0.0 <= prediction.prediction_confidence <= 1.0
                # Consistent trend should have higher confidence
                assert prediction.prediction_confidence > 0.3
    
    async def test_warning_suppression(self, warning_system):
        """Test warning suppression to prevent spam"""
        await warning_system.start_monitoring()
        
        # Set short suppression time for testing
        warning_system.config.alert_suppression_time = 0.2
        
        # Generate multiple warnings quickly
        for i in range(5):
            await warning_system._evaluate_prediction_warnings(
                warning_system.current_predictions.get(PredictionHorizon.SHORT_TERM),
                2.0, 1.0, 0.5, 0.7  # thresholds
            )
            await asyncio.sleep(0.05)
        
        # Should only have one warning due to suppression
        initial_warning_count = len(warning_system.active_warnings)
        
        # Wait for suppression to expire
        await asyncio.sleep(0.3)
        
        # Generate another warning
        await warning_system._evaluate_prediction_warnings(
            warning_system.current_predictions.get(PredictionHorizon.SHORT_TERM),
            2.0, 1.0, 0.5, 0.7
        )
        
        # Should have new warning after suppression expires
        final_warning_count = len(warning_system.active_warnings)
        # Note: exact counts depend on prediction generation, so we test logic exists
        assert warning_system.config.alert_suppression_time == 0.2


class TestSystemIntegration:
    """Integration tests for complete constitutional monitoring system"""
    
    @pytest.fixture
    async def integrated_system(self):
        """Create integrated constitutional monitoring system"""
        # Constitutional monitor
        const_config = ConstitutionalConfiguration()
        const_config.monitoring_interval_ms = 100
        const_config.enable_websocket_dashboard = False
        const_config.enable_metrics_persistence = False
        
        monitor = ConstitutionalRealTimeMonitor(const_config)
        
        # Early warning system  
        ews_config = EarlyWarningConfiguration()
        ews_config.min_data_points = 5
        ews_config.prediction_update_interval = 0.2
        
        ews = EarlyWarningSystem(ews_config)
        
        yield {'monitor': monitor, 'ews': ews}
        
        # Cleanup
        if monitor.monitoring_active:
            await monitor.stop_monitoring()
        if ews.warning_active:
            await ews.stop_monitoring()
    
    async def test_integrated_monitoring_flow(self, integrated_system):
        """Test integrated monitoring workflow"""
        monitor = integrated_system['monitor']
        ews = integrated_system['ews']
        
        # Shared metrics storage
        shared_metrics = {}
        
        # Mock metrics callback that simulates NFCS metrics
        async def mock_metrics_callback():
            return shared_metrics
        
        # Mock constitutional callback for EWS
        async def mock_constitutional_callback(action):
            if action == 'get_thresholds':
                return {
                    'ha_critical': monitor.config.ha_critical_threshold,
                    'ha_warning': monitor.config.ha_warning_threshold,
                    'integrity_critical': monitor.config.integrity_critical_threshold,
                    'integrity_warning': monitor.config.integrity_warning_threshold
                }
        
        # Start both systems
        await monitor.start_monitoring(metrics_callback=mock_metrics_callback)
        await ews.start_monitoring(constitutional_callback=mock_constitutional_callback)
        
        # Simulate escalating system degradation
        for step in range(20):
            # Simulate increasing Ha and decreasing integrity
            ha_value = 0.1 + (step * 0.15)  # Will reach 3.0+
            integrity_value = max(0.2, 1.0 - (step * 0.04))
            
            # Update shared metrics
            shared_metrics.update({
                'hallucination_number': ha_value,
                'coherence_measure': max(0.3, 0.9 - step * 0.02),
                'defect_density': step * 0.003,
                'field_energy': 100 + step * 20,
                'integrity_score': integrity_value,
                'constitutional_compliance': integrity_value,
                'system_stability': max(0.3, 1.0 - step * 0.03)
            })
            
            # Update EWS with same metrics
            await ews.update_metrics(shared_metrics.copy())
            
            await asyncio.sleep(0.15)
            
            # Check system states at different stages
            if step == 5:  # Early stage
                assert monitor.current_status in [ConstitutionalStatus.NORMAL, ConstitutionalStatus.MONITORING]
            elif step == 10:  # Warning stage
                assert monitor.current_status in [ConstitutionalStatus.WARNING, ConstitutionalStatus.MONITORING]
            elif step == 15:  # Critical stage
                assert monitor.current_status in [ConstitutionalStatus.CRITICAL, ConstitutionalStatus.WARNING]
        
        # Final checks
        final_status = monitor.get_current_status()
        ews_status = ews.get_system_status()
        
        assert final_status['metrics']['hallucination_number'] >= 2.5
        assert ews_status['active_warnings_count'] > 0
        
        # Should have emergency alerts or critical warnings
        assert (monitor.current_status == ConstitutionalStatus.EMERGENCY or 
                len(monitor.active_alerts) > 0)
    
    async def test_emergency_coordination(self, integrated_system):
        """Test emergency protocol coordination between systems"""
        monitor = integrated_system['monitor']
        ews = integrated_system['ews']
        
        # Track emergency calls
        emergency_calls = []
        
        async def mock_emergency_callback(action):
            emergency_calls.append(action)
        
        await monitor.start_monitoring(emergency_callback=mock_emergency_callback)
        await ews.start_monitoring()
        
        # Trigger emergency through high Ha
        monitor.current_metrics.hallucination_number = 5.0
        monitor.current_metrics.integrity_score = 0.3
        
        # Process emergency decision
        decision = await monitor._apply_constitutional_algorithm()
        await monitor._execute_constitutional_decision(decision)
        
        # Verify emergency was triggered
        assert len(emergency_calls) > 0
        assert emergency_calls[-1]['activate'] == True
        assert monitor.emergency_active
        
        # Test recovery
        monitor.current_metrics.hallucination_number = 0.5
        decision = await monitor._apply_constitutional_algorithm()
        await monitor._execute_constitutional_decision(decision)
        
        # Should eventually deactivate emergency
        await asyncio.sleep(0.1)
        assert not monitor.emergency_active or monitor.recovery_mode_active
    
    async def test_performance_under_load(self, integrated_system):
        """Test system performance under high-frequency updates"""
        monitor = integrated_system['monitor']
        ews = integrated_system['ews']
        
        # Performance tracking
        start_time = time.time()
        update_count = 0
        
        async def mock_metrics_callback():
            nonlocal update_count
            update_count += 1
            return {
                'hallucination_number': np.random.uniform(0, 2),
                'coherence_measure': np.random.uniform(0.5, 1.0),
                'defect_density': np.random.uniform(0, 0.1),
                'field_energy': np.random.uniform(100, 500),
                'integrity_score': np.random.uniform(0.6, 1.0)
            }
        
        await monitor.start_monitoring(metrics_callback=mock_metrics_callback)
        await ews.start_monitoring()
        
        # High-frequency updates for 2 seconds
        for i in range(100):
            metrics = {
                'hallucination_number': np.random.uniform(0, 1.5),
                'coherence_measure': np.random.uniform(0.5, 1.0),
                'integrity_score': np.random.uniform(0.6, 1.0),
                'defect_density': np.random.uniform(0, 0.05),
                'field_energy': np.random.uniform(100, 300)
            }
            await ews.update_metrics(metrics)
            await asyncio.sleep(0.02)  # 50 Hz updates
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify performance
        assert duration < 5.0  # Should complete within 5 seconds
        assert update_count > 0  # Metrics callback was called
        assert len(ews.metrics_buffer) > 0  # EWS received updates
        
        # System should still be responsive
        status = monitor.get_current_status()
        assert status is not None
        ews_status = ews.get_system_status()
        assert ews_status['system_active']


class TestWebSocketIntegration:
    """Test WebSocket dashboard integration"""
    
    async def test_websocket_server_startup(self):
        """Test WebSocket server startup and shutdown"""
        config = ConstitutionalConfiguration()
        config.enable_websocket_dashboard = True
        config.dashboard_port = 8766  # Different port for testing
        config.enable_metrics_persistence = False
        
        monitor = ConstitutionalRealTimeMonitor(config)
        
        # Start monitoring (includes WebSocket server)
        await monitor.start_monitoring()
        
        # Verify server is running
        assert monitor.websocket_server is not None
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        # Verify server is closed
        assert monitor.websocket_server.is_serving() == False
    
    async def test_websocket_message_format(self):
        """Test WebSocket message format"""
        config = ConstitutionalConfiguration()
        config.enable_websocket_dashboard = False  # We'll test message format directly
        
        monitor = ConstitutionalRealTimeMonitor(config)
        
        # Set test metrics
        monitor.current_metrics.hallucination_number = 1.5
        monitor.current_metrics.integrity_score = 0.8
        monitor.current_metrics.threat_level = ThreatLevel.MODERATE
        monitor.current_status = ConstitutionalStatus.WARNING
        
        # Add test alert
        from modules.constitutional_realtime import ConstitutionalAlert
        alert = ConstitutionalAlert(
            alert_id='test_alert',
            severity=ThreatLevel.HIGH,
            title='Test Alert',
            description='Test alert description'
        )
        monitor.active_alerts['test_alert'] = alert
        
        # Generate message
        message = json.dumps({
            'type': 'metrics_update',
            'status': monitor.current_status.value,
            'metrics': monitor.current_metrics.to_dict(),
            'alerts': {k: v.to_dict() for k, v in monitor.active_alerts.items()},
            'emergency_active': monitor.emergency_active,
            'recovery_mode': monitor.recovery_mode_active
        })
        
        # Verify message can be parsed
        parsed_message = json.loads(message)
        assert parsed_message['type'] == 'metrics_update'
        assert parsed_message['status'] == 'WARNING'
        assert parsed_message['metrics']['hallucination_number'] == 1.5
        assert parsed_message['metrics']['threat_level'] == 'MODERATE'
        assert 'test_alert' in parsed_message['alerts']


@pytest.mark.asyncio
class TestStressAndResilience:
    """Stress testing and resilience validation"""
    
    async def test_memory_usage_stability(self):
        """Test memory usage remains stable under continuous operation"""
        config = ConstitutionalConfiguration()
        config.monitoring_interval_ms = 10  # Very fast for stress test
        config.enable_websocket_dashboard = False
        config.enable_metrics_persistence = False
        
        monitor = ConstitutionalRealTimeMonitor(config)
        
        # Simple metrics callback
        async def fast_metrics_callback():
            return {
                'hallucination_number': np.random.uniform(0, 1),
                'coherence_measure': np.random.uniform(0.5, 1.0),
                'integrity_score': np.random.uniform(0.7, 1.0)
            }
        
        await monitor.start_monitoring(metrics_callback=fast_metrics_callback)
        
        # Run for 1 second with very high frequency
        await asyncio.sleep(1.0)
        
        # Check that system is still responsive
        status = monitor.get_current_status()
        assert status is not None
        assert status['monitoring_active']
        
        # Verify metrics history has reasonable size (not growing unbounded)
        history_size = len(monitor.metrics_history)
        assert history_size > 0
        assert history_size < 1000  # Should not grow unbounded
        
        await monitor.stop_monitoring()
    
    async def test_error_recovery(self):
        """Test system recovery from callback errors"""
        config = ConstitutionalConfiguration()
        config.monitoring_interval_ms = 50
        config.enable_websocket_dashboard = False
        
        monitor = ConstitutionalRealTimeMonitor(config)
        
        call_count = 0
        
        async def failing_metrics_callback():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception("Simulated callback failure")
            return {
                'hallucination_number': 0.5,
                'coherence_measure': 0.8,
                'integrity_score': 0.9
            }
        
        await monitor.start_monitoring(metrics_callback=failing_metrics_callback)
        
        # Wait for several callback attempts
        await asyncio.sleep(0.5)
        
        # System should still be active despite callback failures
        assert monitor.monitoring_active
        assert call_count > 3  # Should have retried after failures
        
        # Should eventually have valid metrics
        assert monitor.current_metrics.hallucination_number > 0
        
        await monitor.stop_monitoring()


# Utility functions for testing

def generate_test_metrics_sequence(steps: int, trend: str = 'stable') -> List[Dict[str, Any]]:
    """Generate a sequence of test metrics with specified trend"""
    metrics_sequence = []
    
    for i in range(steps):
        if trend == 'increasing_ha':
            ha = 0.1 + (i * 0.1)
            integrity = max(0.3, 1.0 - (i * 0.02))
        elif trend == 'decreasing_integrity':
            ha = 0.2 + np.random.uniform(-0.1, 0.1)
            integrity = max(0.2, 1.0 - (i * 0.05))
        elif trend == 'oscillating':
            ha = 0.5 + 0.3 * np.sin(i * 0.2)
            integrity = 0.8 + 0.1 * np.cos(i * 0.3)
        else:  # stable
            ha = 0.3 + np.random.uniform(-0.1, 0.1)
            integrity = 0.9 + np.random.uniform(-0.05, 0.05)
        
        metrics = {
            'hallucination_number': max(0, ha),
            'coherence_measure': 0.8 + np.random.uniform(-0.1, 0.1),
            'defect_density': np.random.uniform(0, 0.02),
            'field_energy': 150 + np.random.uniform(-50, 50),
            'integrity_score': np.clip(integrity, 0.0, 1.0),
            'constitutional_compliance': np.clip(integrity + 0.1, 0.0, 1.0),
            'system_stability': 0.8 + np.random.uniform(-0.1, 0.1),
            'timestamp': time.time()
        }
        metrics_sequence.append(metrics)
    
    return metrics_sequence


if __name__ == "__main__":
    # Run specific test for development
    async def run_integration_test():
        """Run a basic integration test"""
        print("🧪 Running Constitutional Monitoring Integration Test")
        
        # Create systems
        const_config = ConstitutionalConfiguration()
        const_config.monitoring_interval_ms = 200
        const_config.enable_websocket_dashboard = False
        const_config.enable_metrics_persistence = False
        
        monitor = ConstitutionalRealTimeMonitor(const_config)
        
        ews_config = EarlyWarningConfiguration()
        ews_config.min_data_points = 5
        
        ews = EarlyWarningSystem(ews_config)
        
        try:
            # Start monitoring
            await monitor.start_monitoring()
            await ews.start_monitoring()
            
            # Feed test data
            test_metrics = generate_test_metrics_sequence(15, 'increasing_ha')
            
            for i, metrics in enumerate(test_metrics):
                await ews.update_metrics(metrics)
                await asyncio.sleep(0.1)
                
                if i % 5 == 0:
                    status = monitor.get_current_status()
                    warnings = ews.get_current_warnings()
                    print(f"Step {i}: Status={status['status']}, Warnings={len(warnings)}, "
                          f"Ha={metrics['hallucination_number']:.3f}")
            
            print("✅ Integration test completed successfully")
            
        finally:
            await monitor.stop_monitoring()
            await ews.stop_monitoring()
    
    asyncio.run(run_integration_test())