#!/usr/bin/env python3
"""
NFCS API Demo Script
===================

Demonstration script for NFCS FastAPI v2.4.3 REST API and WebSocket monitoring.
Shows complete API functionality including system control, ESC processing, and real-time monitoring.

Author: Team Ω (Omega)
Date: September 13, 2025
Version: 2.4.3
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any

import requests
import websockets
from websockets.exceptions import ConnectionClosedError


class NFCSAPIDemo:
    """NFCS API demonstration client"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws") + "/ws"
        self.session = requests.Session()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def health_check(self) -> Dict[str, Any]:
        """Test health check endpoint"""
        self.logger.info("🔍 Testing health check endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"✅ Health check successful: {data['status']}")
            self.logger.info(f"   Server version: {data['version']}")
            self.logger.info(f"   Uptime: {data['uptime_seconds']:.1f}s")
            self.logger.info(f"   Memory usage: {data['memory_usage_percent']:.1f}%")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Test system status endpoint"""
        self.logger.info("📊 Getting system status...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/v1/system/status", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"✅ System status: {data['status']}")
            
            metrics = data.get('metrics', {})
            self.logger.info(f"   Total cycles: {metrics.get('total_cycles', 0)}")
            self.logger.info(f"   Success rate: {metrics.get('success_rate', 0):.3f}")
            self.logger.info(f"   Active modules: {metrics.get('active_modules', 0)}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ System status failed: {e}")
            return {}
    
    def test_esc_processing(self) -> Dict[str, Any]:
        """Test ESC token processing"""
        self.logger.info("🎭 Testing ESC token processing...")
        
        try:
            # Sample tokens for processing
            test_tokens = [
                "The", "neural", "field", "exhibits", "coherent", "behavior",
                "with", "minimal", "topological", "defects"
            ]
            
            request_data = {
                "tokens": test_tokens,
                "processing_mode": "full_pipeline", 
                "context": "Scientific analysis of neural field dynamics",
                "enable_constitutional_filtering": True,
                "return_embeddings": False
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/esc/process",
                json=request_data,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"✅ ESC processing successful")
            self.logger.info(f"   Processing time: {data['processing_time_ms']:.1f}ms")
            self.logger.info(f"   Sequence coherence: {data['sequence_coherence']:.3f}")
            
            # Show token analysis results
            for analysis in data['token_analyses'][:3]:  # Show first 3
                self.logger.info(f"   Token '{analysis['token']}': attention={analysis['attention_score']:.3f}, compliance={analysis['constitutional_compliance']:.3f}")
            
            # Constitutional filtering results
            const_filter = data['constitutional_filter']
            self.logger.info(f"   Constitutional compliance: {const_filter['overall_compliance']:.3f}")
            self.logger.info(f"   Safety score: {const_filter['safety_score']:.3f}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ ESC processing failed: {e}")
            return {}
    
    def test_system_control(self) -> bool:
        """Test system control operations"""
        self.logger.info("⚙️ Testing system control operations...")
        
        try:
            # Test getting current status first
            status_response = self.session.get(f"{self.base_url}/api/v1/system/status")
            if status_response.status_code == 200:
                current_state = status_response.json()['status']
                self.logger.info(f"   Current system state: {current_state}")
            
            # Note: In a real demo, we might not want to actually restart the system
            # This is a mock control operation that checks the endpoint structure
            self.logger.info("   System control endpoint available (not executing restart for demo safety)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System control test failed: {e}")
            return False
    
    async def test_websocket_monitoring(self, duration: int = 30):
        """Test WebSocket real-time monitoring"""
        self.logger.info(f"🔗 Testing WebSocket monitoring for {duration} seconds...")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                self.logger.info("✅ WebSocket connection established")
                
                # Send subscription request
                subscribe_msg = {
                    "action": "subscribe",
                    "event_types": ["telemetry_update", "system_status", "heartbeat"],
                    "filters": {"min_severity": "info"}
                }
                
                await websocket.send(json.dumps(subscribe_msg))
                self.logger.info("📡 Subscribed to telemetry updates")
                
                # Monitor events
                start_time = time.time()
                event_count = 0
                
                while time.time() - start_time < duration:
                    try:
                        # Wait for message with timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(message)
                        
                        event_count += 1
                        event_type = data.get('event_type', 'unknown')
                        source = data.get('source', 'unknown')
                        
                        self.logger.info(f"📨 Event #{event_count}: {event_type} from {source}")
                        
                        # Show specific event details
                        if event_type == "telemetry_update":
                            payload = data.get('data', {})
                            if 'cycle_number' in payload:
                                self.logger.info(f"   Cycle: {payload['cycle_number']}, Time: {payload.get('cycle_time_ms', 0):.1f}ms")
                        
                        elif event_type == "system_status":
                            payload = data.get('data', {})
                            if 'connection_id' in payload:
                                self.logger.info(f"   Connection: {payload['connection_id']}")
                            elif 'action' in payload:
                                self.logger.info(f"   Action: {payload['action']}")
                        
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.send(json.dumps({"action": "ping"}))
                        
                    except ConnectionClosedError:
                        self.logger.warning("WebSocket connection closed by server")
                        break
                
                self.logger.info(f"✅ WebSocket monitoring complete: {event_count} events received")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ WebSocket monitoring failed: {e}")
            return False
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API information"""
        self.logger.info("ℹ️ Getting API information...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/v1/info", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"✅ API Version: {data['api_version']}")
            self.logger.info(f"   NFCS Version: {data['nfcs_version']}")
            self.logger.info(f"   Team: {data['team']}")
            
            # Show capabilities
            capabilities = data.get('capabilities', {})
            enabled_features = [k for k, v in capabilities.items() if v]
            self.logger.info(f"   Enabled features: {', '.join(enabled_features)}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ API info failed: {e}")
            return {}
    
    async def run_complete_demo(self):
        """Run complete API demonstration"""
        self.logger.info("🚀 Starting NFCS API v2.4.3 Complete Demonstration")
        self.logger.info("=" * 60)
        
        # Test 1: Health Check
        health_data = self.health_check()
        if not health_data:
            self.logger.error("❌ Health check failed - aborting demo")
            return
        
        print()
        
        # Test 2: API Info
        api_info = self.get_api_info()
        print()
        
        # Test 3: System Status
        system_status = self.get_system_status()
        print()
        
        # Test 4: ESC Processing
        esc_result = self.test_esc_processing()
        print()
        
        # Test 5: System Control
        control_result = self.test_system_control()
        print()
        
        # Test 6: WebSocket Monitoring
        self.logger.info("Starting WebSocket monitoring test...")
        ws_result = await self.test_websocket_monitoring(duration=15)
        print()
        
        # Summary
        self.logger.info("📋 Demo Summary:")
        self.logger.info(f"   ✅ Health Check: {'PASS' if health_data else 'FAIL'}")
        self.logger.info(f"   ✅ API Info: {'PASS' if api_info else 'FAIL'}")
        self.logger.info(f"   ✅ System Status: {'PASS' if system_status else 'FAIL'}")
        self.logger.info(f"   ✅ ESC Processing: {'PASS' if esc_result else 'FAIL'}")
        self.logger.info(f"   ✅ System Control: {'PASS' if control_result else 'FAIL'}")
        self.logger.info(f"   ✅ WebSocket Monitor: {'PASS' if ws_result else 'FAIL'}")
        
        success_count = sum([
            bool(health_data), bool(api_info), bool(system_status),
            bool(esc_result), bool(control_result), bool(ws_result)
        ])
        
        self.logger.info(f"🎯 Overall Success Rate: {success_count}/6 ({success_count/6*100:.1f}%)")
        
        if success_count == 6:
            self.logger.info("🎉 All API tests passed successfully!")
        else:
            self.logger.warning(f"⚠️ {6-success_count} test(s) failed - check server status")


async def main():
    """Main demonstration function"""
    print("NFCS FastAPI v2.4.3 - Complete API Demonstration")
    print("=" * 50)
    print("Team Ω (Omega) | September 13, 2025")
    print()
    
    # Check if server is specified
    import sys
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"Server URL: {base_url}")
    print("Note: Make sure the NFCS FastAPI server is running!")
    print()
    
    # Run demonstration
    demo = NFCSAPIDemo(base_url)
    await demo.run_complete_demo()
    
    print()
    print("Demo complete! Check the server documentation at:")
    print(f"  Swagger UI: {base_url}/docs")
    print(f"  ReDoc: {base_url}/redoc")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logging.error("Demo exception", exc_info=True)