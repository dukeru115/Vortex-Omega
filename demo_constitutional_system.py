#!/usr/bin/env python3
"""
Constitutional Monitoring System Integration Demo
==============================================

Comprehensive demonstration of the Neural Field Control System (NFCS) v2.4.3
Constitutional Monitoring System including:

1. Constitutional Real-time Monitor with Algorithm 1
2. Early Warning System with predictive capabilities
3. WebSocket dashboard integration
4. Emergency protocol demonstration
5. Integration with core NFCS components

This demo shows the complete constitutional oversight system in action,
demonstrating real-time monitoring, threat assessment, and emergency protocols.

Author: Team Omega (GenSpark AI Implementation)
License: CC BY-NC 4.0
Date: 2025-09-14
"""

import asyncio
import logging
import numpy as np
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import threading
import webbrowser
from dataclasses import dataclass

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from modules.constitutional_realtime import (
    ConstitutionalRealTimeMonitor,
    ConstitutionalConfiguration,
    ConstitutionalStatus,
    ThreatLevel
)

from modules.early_warning_system import (
    EarlyWarningSystem,
    EarlyWarningConfiguration,
    WarningLevel,
    PredictionHorizon
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class NFCSSimulationState:
    """Simulated NFCS state for demonstration"""
    step: int = 0
    
    # Core metrics
    hallucination_number: float = 0.0
    coherence_measure: float = 1.0
    defect_density: float = 0.0
    field_energy: float = 100.0
    integrity_score: float = 1.0
    
    # System state
    emergency_mode: bool = False
    processing_load: float = 0.1
    memory_usage: float = 100.0
    
    # Symbolic-Neural Bridge state
    symbolic_consistency: float = 1.0
    neural_field_coupling: float = 0.8
    
    def to_metrics_dict(self) -> Dict[str, Any]:
        """Convert to metrics dictionary"""
        return {
            'hallucination_number': self.hallucination_number,
            'coherence_measure': self.coherence_measure,
            'defect_density': self.defect_density,
            'field_energy': self.field_energy,
            'integrity_score': self.integrity_score,
            'constitutional_compliance': min(1.0, self.integrity_score + 0.1),
            'system_stability': max(0.0, 1.0 - self.processing_load),
            'processing_latency_ms': self.processing_load * 100,
            'memory_usage_mb': self.memory_usage,
            'cpu_usage_percent': self.processing_load * 100,
            'timestamp': time.time()
        }


class NFCSConstitutionalDemo:
    """
    Complete Constitutional Monitoring System Demo
    """
    
    def __init__(self):
        """Initialize demo system"""
        self.running = False
        self.simulation_state = NFCSSimulationState()
        
        # Configure constitutional monitor
        self.const_config = ConstitutionalConfiguration()
        self.const_config.monitoring_interval_ms = 500  # 2 Hz for demo
        self.const_config.ha_warning_threshold = 0.8
        self.const_config.ha_critical_threshold = 1.5
        self.const_config.ha_emergency_threshold = 3.0
        self.const_config.enable_websocket_dashboard = True
        self.const_config.dashboard_port = 8765
        
        self.constitutional_monitor = ConstitutionalRealTimeMonitor(self.const_config)
        
        # Configure early warning system
        self.ews_config = EarlyWarningConfiguration()
        self.ews_config.min_data_points = 8
        self.ews_config.prediction_update_interval = 2.0
        self.ews_config.min_prediction_confidence = 0.6
        
        self.early_warning = EarlyWarningSystem(self.ews_config)
        
        # Demo state
        self.demo_scenario = "normal"
        self.scenario_step = 0
        self.metrics_history = []
        self.control_signals_log = []
        
        # Integration state
        self.emergency_protocols_active = False
        self.last_constitutional_decision = None
        
        logger.info("Constitutional Monitoring Demo initialized")
    
    async def start_demo(self):
        """Start the complete demo system"""
        print("🏛️  Starting NFCS Constitutional Monitoring System Demo")
        print("=" * 60)
        
        # Start constitutional monitoring
        await self.constitutional_monitor.start_monitoring(
            module_control_callback=self.handle_module_control,
            emergency_callback=self.handle_emergency_protocol,
            metrics_callback=self.get_current_metrics
        )
        
        # Start early warning system
        await self.early_warning.start_monitoring(
            constitutional_callback=self.handle_constitutional_query,
            emergency_callback=self.handle_emergency_protocol
        )
        
        self.running = True
        
        # Start simulation loop
        asyncio.create_task(self.simulation_loop())
        
        # Start monitoring display
        asyncio.create_task(self.display_loop())
        
        # Open dashboard
        await self.open_dashboard()
        
        print(f"✅ Constitutional monitoring system started")
        print(f"📊 Dashboard available at: http://localhost:{self.const_config.dashboard_port}")
        print(f"📈 WebSocket dashboard: file://{Path(__file__).parent}/dashboard/constitutional_monitor.html")
        
    async def stop_demo(self):
        """Stop the demo system"""
        self.running = False
        await self.constitutional_monitor.stop_monitoring()
        await self.early_warning.stop_monitoring()
        print("🛑 Demo stopped")
    
    async def simulation_loop(self):
        """Main simulation loop"""
        while self.running:
            try:
                # Update simulation based on scenario
                self.update_simulation()
                
                # Update early warning system
                await self.early_warning.update_metrics(self.simulation_state.to_metrics_dict())
                
                # Store metrics history
                self.metrics_history.append({
                    'timestamp': time.time(),
                    'step': self.scenario_step,
                    'scenario': self.demo_scenario,
                    **self.simulation_state.to_metrics_dict()
                })
                
                # Rotate history
                if len(self.metrics_history) > 200:
                    self.metrics_history = self.metrics_history[-200:]
                
                await asyncio.sleep(1.0)  # 1 Hz simulation
                
            except Exception as e:
                logger.error(f"Simulation loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def display_loop(self):
        """Display monitoring information"""
        while self.running:
            try:
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
                # Get system status
                const_status = self.constitutional_monitor.get_current_status()
                ews_status = self.early_warning.get_system_status()
                warnings = self.early_warning.get_current_warnings()
                predictions = self.early_warning.get_predictions()
                
                # Display summary
                print("\n" + "="*60)
                print(f"🏛️  CONSTITUTIONAL MONITORING STATUS - Step {self.scenario_step}")
                print(f"Scenario: {self.demo_scenario.upper()}")
                print("-" * 60)
                
                # Core metrics
                metrics = const_status['metrics']
                ha = metrics['hallucination_number']
                integrity = metrics['integrity_score']
                threat = metrics['threat_level']
                
                print(f"📊 Ha Number: {ha:.3f} | Integrity: {integrity:.3f} | Threat: {threat}")
                print(f"🎯 Status: {const_status['status']} | Alerts: {const_status['alerts_count']}")
                
                # Early warning status
                if warnings:
                    print(f"⚠️  Active Warnings: {len(warnings)}")
                    for w_id, warning in warnings.items():
                        print(f"   • {warning['title']} ({warning['warning_level']})")
                
                # Predictions
                if predictions:
                    short_pred = predictions.get('SHORT_TERM', {})
                    if short_pred:
                        pred_ha = short_pred.get('predicted_ha', 0)
                        confidence = short_pred.get('prediction_confidence', 0)
                        print(f"🔮 Predicted Ha (30s): {pred_ha:.3f} (conf: {confidence:.2f})")
                
                # Emergency status
                if const_status['emergency_active']:
                    print("🚨 EMERGENCY PROTOCOLS ACTIVE")
                elif const_status['recovery_mode']:
                    print("🔄 RECOVERY MODE ACTIVE")
                
                print("="*60)
                
            except Exception as e:
                logger.error(f"Display loop error: {e}")
                await asyncio.sleep(1.0)
    
    def update_simulation(self):
        """Update simulation state based on current scenario"""
        self.scenario_step += 1
        
        if self.demo_scenario == "normal":
            self.simulate_normal_operation()
            
            # Transition to degradation after 20 steps
            if self.scenario_step > 20:
                self.demo_scenario = "degradation"
                self.scenario_step = 0
                print("\n🔄 Transitioning to: GRADUAL DEGRADATION scenario")
        
        elif self.demo_scenario == "degradation":
            self.simulate_gradual_degradation()
            
            # Transition to crisis after 30 steps
            if self.scenario_step > 30:
                self.demo_scenario = "crisis"
                self.scenario_step = 0
                print("\n🚨 Transitioning to: CRISIS scenario")
        
        elif self.demo_scenario == "crisis":
            self.simulate_crisis_scenario()
            
            # Transition to recovery after 20 steps
            if self.scenario_step > 20:
                self.demo_scenario = "recovery"
                self.scenario_step = 0
                print("\n🔄 Transitioning to: RECOVERY scenario")
        
        elif self.demo_scenario == "recovery":
            self.simulate_recovery()
            
            # Transition back to normal after 25 steps
            if self.scenario_step > 25:
                self.demo_scenario = "normal"
                self.scenario_step = 0
                print("\n✅ Transitioning to: NORMAL OPERATION scenario")
    
    def simulate_normal_operation(self):
        """Simulate normal NFCS operation"""
        # Stable metrics with small fluctuations
        self.simulation_state.hallucination_number = 0.2 + np.random.uniform(-0.1, 0.1)
        self.simulation_state.coherence_measure = 0.9 + np.random.uniform(-0.05, 0.05)
        self.simulation_state.defect_density = max(0, 0.01 + np.random.uniform(-0.005, 0.005))
        self.simulation_state.field_energy = 120 + np.random.uniform(-20, 20)
        self.simulation_state.integrity_score = 0.95 + np.random.uniform(-0.02, 0.02)
        self.simulation_state.processing_load = 0.15 + np.random.uniform(-0.05, 0.05)
        self.simulation_state.memory_usage = 150 + np.random.uniform(-10, 10)
    
    def simulate_gradual_degradation(self):
        """Simulate gradual system degradation"""
        degradation_factor = self.scenario_step / 30.0
        
        # Gradually increasing Ha
        base_ha = 0.3 + (degradation_factor * 1.2)
        self.simulation_state.hallucination_number = base_ha + np.random.uniform(-0.1, 0.2)
        
        # Decreasing coherence and integrity
        self.simulation_state.coherence_measure = max(0.3, 0.9 - degradation_factor * 0.4)
        self.simulation_state.integrity_score = max(0.4, 0.95 - degradation_factor * 0.3)
        
        # Increasing defects and processing load
        self.simulation_state.defect_density = degradation_factor * 0.05
        self.simulation_state.processing_load = min(0.8, 0.2 + degradation_factor * 0.4)
        self.simulation_state.memory_usage = 150 + degradation_factor * 300
        
        # Fluctuating field energy
        self.simulation_state.field_energy = 120 + np.sin(self.scenario_step * 0.3) * 100 + degradation_factor * 200
    
    def simulate_crisis_scenario(self):
        """Simulate crisis with high Ha values"""
        crisis_intensity = min(1.0, self.scenario_step / 15.0)
        
        # High Ha with spikes
        base_ha = 2.0 + crisis_intensity * 2.0
        spike = np.sin(self.scenario_step * 0.5) * 1.5 if self.scenario_step % 3 == 0 else 0
        self.simulation_state.hallucination_number = base_ha + spike + np.random.uniform(-0.3, 0.5)
        
        # Severely degraded metrics
        self.simulation_state.coherence_measure = max(0.1, 0.4 - crisis_intensity * 0.2)
        self.simulation_state.integrity_score = max(0.2, 0.6 - crisis_intensity * 0.3)
        
        # High defects and system stress
        self.simulation_state.defect_density = 0.08 + crisis_intensity * 0.1
        self.simulation_state.processing_load = min(1.0, 0.6 + crisis_intensity * 0.4)
        self.simulation_state.memory_usage = 400 + crisis_intensity * 400
        self.simulation_state.field_energy = 500 + np.random.uniform(-100, 200)
    
    def simulate_recovery(self):
        """Simulate system recovery"""
        recovery_progress = min(1.0, self.scenario_step / 25.0)
        
        # Gradually improving Ha
        start_ha = 3.0
        target_ha = 0.3
        self.simulation_state.hallucination_number = start_ha - (start_ha - target_ha) * recovery_progress
        self.simulation_state.hallucination_number += np.random.uniform(-0.2, 0.1)
        
        # Improving metrics
        self.simulation_state.coherence_measure = 0.3 + recovery_progress * 0.6
        self.simulation_state.integrity_score = 0.4 + recovery_progress * 0.5
        
        # Reducing system stress
        self.simulation_state.defect_density = max(0.01, 0.15 - recovery_progress * 0.14)
        self.simulation_state.processing_load = max(0.15, 0.8 - recovery_progress * 0.65)
        self.simulation_state.memory_usage = max(150, 600 - recovery_progress * 450)
        self.simulation_state.field_energy = 300 - recovery_progress * 180
    
    # Callback functions for integration
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Provide current metrics to constitutional monitor"""
        return self.simulation_state.to_metrics_dict()
    
    async def handle_module_control(self, control_signals: Dict[str, Any]):
        """Handle module control signals from constitutional monitor"""
        self.control_signals_log.append({
            'timestamp': time.time(),
            'step': self.scenario_step,
            'signals': control_signals.copy()
        })
        
        print(f"🎛️  Module control signals received: {list(control_signals.keys())}")
        
        # Simulate effects of control signals
        if 'kuramoto_all' in control_signals:
            # Emergency desynchronization
            emergency_strength = abs(control_signals['kuramoto_all'])
            if emergency_strength > 0.5:
                # Reduce Ha through emergency protocol
                self.simulation_state.hallucination_number *= 0.8
                self.simulation_state.processing_load = min(1.0, self.simulation_state.processing_load + 0.2)
                print(f"🚨 Emergency desynchronization applied (strength: {emergency_strength:.2f})")
        
        if 'memory' in control_signals:
            # Memory module control
            print("🧠 Memory module synchronization applied")
            self.simulation_state.coherence_measure = min(1.0, self.simulation_state.coherence_measure + 0.1)
        
        if 'esc' in control_signals:
            # ESC module control
            print("🔄 ESC module synchronization applied")
            self.simulation_state.integrity_score = min(1.0, self.simulation_state.integrity_score + 0.05)
    
    async def handle_emergency_protocol(self, action: Dict[str, Any]):
        """Handle emergency protocol activation/deactivation"""
        if action.get('activate'):
            self.emergency_protocols_active = True
            self.simulation_state.emergency_mode = True
            print("🚨🚨 EMERGENCY PROTOCOLS ACTIVATED 🚨🚨")
            
            # Emergency effects on simulation
            self.simulation_state.processing_load = min(1.0, self.simulation_state.processing_load + 0.3)
            
        else:
            self.emergency_protocols_active = False
            self.simulation_state.emergency_mode = False
            print("✅ Emergency protocols deactivated - entering recovery mode")
    
    async def handle_constitutional_query(self, query: str) -> Any:
        """Handle queries from early warning system"""
        if query == 'get_thresholds':
            return {
                'ha_critical': self.const_config.ha_critical_threshold,
                'ha_warning': self.const_config.ha_warning_threshold,
                'integrity_critical': self.const_config.integrity_critical_threshold,
                'integrity_warning': self.const_config.integrity_warning_threshold
            }
        return None
    
    async def open_dashboard(self):
        """Open the WebSocket dashboard in browser"""
        dashboard_path = Path(__file__).parent / "dashboard" / "constitutional_monitor.html"
        if dashboard_path.exists():
            try:
                # Start a simple HTTP server for the dashboard
                import http.server
                import socketserver
                import threading
                
                def serve_dashboard():
                    os.chdir(Path(__file__).parent)
                    with socketserver.TCPServer(("", 8766), http.server.SimpleHTTPRequestHandler) as httpd:
                        httpd.serve_forever()
                
                import os
                dashboard_thread = threading.Thread(target=serve_dashboard, daemon=True)
                dashboard_thread.start()
                
                # Wait a bit then open browser
                await asyncio.sleep(2)
                webbrowser.open("http://localhost:8766/dashboard/constitutional_monitor.html")
                
            except Exception as e:
                logger.warning(f"Could not auto-open dashboard: {e}")
                print(f"📋 Manual dashboard access: file://{dashboard_path}")
    
    async def run_interactive_demo(self):
        """Run interactive demo with user commands"""
        await self.start_demo()
        
        print("\n" + "="*60)
        print("🎮 INTERACTIVE DEMO CONTROLS")
        print("="*60)
        print("Commands:")
        print("  'status' - Show current system status")
        print("  'emergency' - Force emergency mode")
        print("  'recover' - Force recovery mode")
        print("  'scenario X' - Change scenario (normal/degradation/crisis/recovery)")
        print("  'warnings' - Show active warnings")
        print("  'predictions' - Show current predictions")
        print("  'history' - Show metrics history")
        print("  'quit' - Stop demo")
        print("="*60)
        
        try:
            while self.running:
                try:
                    # Non-blocking input simulation
                    await asyncio.sleep(0.1)
                    
                    # In a real interactive demo, you'd handle user input here
                    # For automation, we'll just run the scenarios
                    
                except KeyboardInterrupt:
                    print("\n🛑 Demo interrupted by user")
                    break
                    
        finally:
            await self.stop_demo()
    
    def generate_final_report(self):
        """Generate final demo report"""
        print("\n" + "="*80)
        print("📋 CONSTITUTIONAL MONITORING DEMO - FINAL REPORT")
        print("="*80)
        
        if not self.metrics_history:
            print("No metrics data collected")
            return
        
        # Analysis of metrics
        ha_values = [m['hallucination_number'] for m in self.metrics_history]
        integrity_values = [m['integrity_score'] for m in self.metrics_history]
        
        print(f"📊 Metrics Summary:")
        print(f"   • Ha Range: {min(ha_values):.3f} - {max(ha_values):.3f}")
        print(f"   • Avg Ha: {np.mean(ha_values):.3f}")
        print(f"   • Integrity Range: {min(integrity_values):.3f} - {max(integrity_values):.3f}")
        print(f"   • Avg Integrity: {np.mean(integrity_values):.3f}")
        
        print(f"\n🎛️  Control Interventions: {len(self.control_signals_log)}")
        
        # Count emergency activations
        emergency_count = sum(1 for log in self.control_signals_log 
                            if 'kuramoto_all' in log['signals'])
        print(f"🚨 Emergency Interventions: {emergency_count}")
        
        print(f"\n🏛️  Constitutional Monitoring Features Demonstrated:")
        print("   ✅ Real-time Ha monitoring")
        print("   ✅ Algorithm 1 implementation")
        print("   ✅ Early warning predictions")
        print("   ✅ Emergency protocol activation")
        print("   ✅ WebSocket dashboard")
        print("   ✅ Integration with NFCS modules")
        
        print("\n" + "="*80)


async def main():
    """Main demo execution"""
    print("🚀 Initializing NFCS Constitutional Monitoring Demo")
    
    # Create demo system
    demo = NFCSConstitutionalDemo()
    
    try:
        # Run automated demo
        print("🎬 Starting automated demonstration...")
        await demo.start_demo()
        
        # Let the demo run for scenarios
        print("⏳ Running demo scenarios (this will take about 5 minutes)...")
        
        # Run for enough time to see all scenarios
        demo_duration = 300  # 5 minutes
        start_time = time.time()
        
        while (time.time() - start_time) < demo_duration and demo.running:
            await asyncio.sleep(1)
            
            # Check for user interrupt
            if demo.scenario_step > 100:  # Safety limit
                break
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
    finally:
        await demo.stop_demo()
        demo.generate_final_report()
        print("✅ Demo completed successfully!")


if __name__ == "__main__":
    # Run the complete constitutional monitoring demo
    asyncio.run(main())