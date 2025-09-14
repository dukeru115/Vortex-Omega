"""
WebSocket Event Handlers
=======================

Event handlers for WebSocket real-time monitoring integration with NFCS orchestrator.
Bridges system events to WebSocket notifications.

Author: Team Ω (Omega)
Date: September 13, 2025
Version: 2.4.3
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import WebSocket, WebSocketDisconnect
from .connection_manager import websocket_manager
from ..models.websocket_models import (
    WebSocketMessage, EventType, AlertLevel, WebSocketSubscriptionRequest,
    WebSocketSubscriptionResponse, TelemetryUpdate, EmergencyAlert
)


class WebSocketEventHandler:
    """WebSocket event handling and integration with NFCS"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.WebSocketEventHandler")
        self._nfcs_orchestrator = None  # Will be injected by dependency
    
    async def handle_websocket_connection(self, websocket: WebSocket, client_ip: Optional[str] = None):
        """Handle new WebSocket connection lifecycle"""
        connection_id = None
        
        try:
            # Accept connection
            connection_id = await websocket_manager.connect(websocket, client_ip)
            self.logger.info(f"WebSocket connection established: {connection_id}")
            
            # Handle messages
            while True:
                try:
                    # Receive message from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Process client message
                    await self._process_client_message(connection_id, message)
                    
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON from {connection_id}: {e}")
                    await self._send_error(connection_id, "INVALID_JSON", "Invalid JSON format")
                except Exception as e:
                    self.logger.error(f"Error processing message from {connection_id}: {e}")
                    await self._send_error(connection_id, "PROCESSING_ERROR", str(e))
                    
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
        finally:
            if connection_id:
                await websocket_manager.disconnect(connection_id)
                self.logger.info(f"WebSocket connection closed: {connection_id}")
    
    async def _process_client_message(self, connection_id: str, message: Dict[str, Any]):
        """Process incoming client message"""
        action = message.get("action")
        
        if action == "subscribe":
            await self._handle_subscription_request(connection_id, message)
        elif action == "unsubscribe":
            await self._handle_unsubscription_request(connection_id, message)
        elif action == "ping":
            await self._handle_ping(connection_id)
        elif action == "get_status":
            await self._handle_status_request(connection_id)
        elif action == "get_metrics":
            await self._handle_metrics_request(connection_id)
        else:
            await self._send_error(connection_id, "UNKNOWN_ACTION", f"Unknown action: {action}")
    
    async def _handle_subscription_request(self, connection_id: str, message: Dict[str, Any]):
        """Handle subscription request"""
        try:
            event_types = [EventType(et) for et in message.get("event_types", [])]
            filters = message.get("filters", {})
            
            success = await websocket_manager.update_subscription(
                connection_id, event_types, filters
            )
            
            if success:
                self.logger.info(f"Updated subscriptions for {connection_id}: {event_types}")
            else:
                await self._send_error(connection_id, "SUBSCRIPTION_FAILED", "Failed to update subscriptions")
                
        except ValueError as e:
            await self._send_error(connection_id, "INVALID_EVENT_TYPE", str(e))
        except Exception as e:
            self.logger.error(f"Subscription error for {connection_id}: {e}")
            await self._send_error(connection_id, "SUBSCRIPTION_ERROR", str(e))
    
    async def _handle_unsubscription_request(self, connection_id: str, message: Dict[str, Any]):
        """Handle unsubscription request"""
        try:
            # Unsubscribe from all events by passing empty list
            success = await websocket_manager.update_subscription(connection_id, [])
            
            if not success:
                await self._send_error(connection_id, "UNSUBSCRIBE_FAILED", "Failed to unsubscribe")
                
        except Exception as e:
            self.logger.error(f"Unsubscription error for {connection_id}: {e}")
            await self._send_error(connection_id, "UNSUBSCRIBE_ERROR", str(e))
    
    async def _handle_ping(self, connection_id: str):
        """Handle ping request"""
        pong_msg = WebSocketMessage(
            event_type=EventType.HEARTBEAT,
            source="websocket_handler",
            data={
                "type": "pong",
                "server_time": datetime.utcnow().isoformat(),
                "connection_id": connection_id
            }
        )
        
        await websocket_manager.send_to_connection(connection_id, pong_msg)
    
    async def _handle_status_request(self, connection_id: str):
        """Handle system status request"""
        try:
            # Get system status from orchestrator if available
            status_data = {"status": "unknown", "message": "NFCS orchestrator not available"}
            
            if self._nfcs_orchestrator:
                try:
                    system_status = self._nfcs_orchestrator.get_system_status()
                    status_data = {
                        "status": system_status.get("orchestrator_state", "unknown"),
                        "statistics": system_status.get("statistics", {}),
                        "components": system_status.get("components", {}),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    self.logger.error(f"Failed to get system status: {e}")
                    status_data = {"status": "error", "message": f"Failed to get status: {e}"}
            
            status_msg = WebSocketMessage(
                event_type=EventType.SYSTEM_STATUS,
                source="nfcs_orchestrator",
                data=status_data
            )
            
            await websocket_manager.send_to_connection(connection_id, status_msg)
            
        except Exception as e:
            self.logger.error(f"Status request error for {connection_id}: {e}")
            await self._send_error(connection_id, "STATUS_ERROR", str(e))
    
    async def _handle_metrics_request(self, connection_id: str):
        """Handle metrics request"""
        try:
            # Get WebSocket manager statistics
            ws_stats = websocket_manager.get_connection_stats()
            
            # Combine with system metrics if available
            metrics_data = {
                "websocket_stats": ws_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if self._nfcs_orchestrator:
                try:
                    system_status = self._nfcs_orchestrator.get_system_status()
                    metrics_data["system_metrics"] = system_status.get("statistics", {})
                except Exception as e:
                    self.logger.error(f"Failed to get system metrics: {e}")
            
            metrics_msg = WebSocketMessage(
                event_type=EventType.TELEMETRY_UPDATE,
                source="websocket_handler",
                data=metrics_data
            )
            
            await websocket_manager.send_to_connection(connection_id, metrics_msg)
            
        except Exception as e:
            self.logger.error(f"Metrics request error for {connection_id}: {e}")
            await self._send_error(connection_id, "METRICS_ERROR", str(e))
    
    async def _send_error(self, connection_id: str, error_code: str, message: str):
        """Send error message to client"""
        error_msg = WebSocketMessage(
            event_type=EventType.ERROR_OCCURRED,
            source="websocket_handler",
            data={
                "error_code": error_code,
                "error_message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        await websocket_manager.send_to_connection(connection_id, error_msg)
    
    def set_nfcs_orchestrator(self, orchestrator):
        """Inject NFCS orchestrator dependency"""
        self._nfcs_orchestrator = orchestrator
    
    # Integration methods for NFCS events
    async def on_cycle_complete(self, cycle_metrics: Dict[str, Any]):
        """Handle NFCS cycle completion event"""
        telemetry = TelemetryUpdate(
            cycle_number=cycle_metrics.get("cycle_number", 0),
            cycle_time_ms=cycle_metrics.get("total_time_ms", 0.0),
            coherence_global=cycle_metrics.get("coherence_global"),
            coherence_modular=cycle_metrics.get("coherence_modular"),
            hallucination_risk=cycle_metrics.get("hallucination_risk"),
            defect_density=cycle_metrics.get("defect_density"),
            active_modules=cycle_metrics.get("active_modules", 0)
        )
        
        await websocket_manager.send_telemetry_update(telemetry)
    
    async def on_emergency_detected(self, emergency_info: Dict[str, Any]):
        """Handle emergency condition detection"""
        alert = EmergencyAlert(
            alert_id=emergency_info.get("alert_id", f"EMG_{int(datetime.utcnow().timestamp())}"),
            alert_type=emergency_info.get("alert_type", "unknown"),
            message=emergency_info.get("message", "Emergency condition detected"),
            severity=AlertLevel(emergency_info.get("severity", "critical")),
            trigger_values=emergency_info.get("trigger_values"),
            thresholds=emergency_info.get("thresholds"),
            auto_actions=emergency_info.get("auto_actions", []),
            requires_intervention=emergency_info.get("requires_intervention", True)
        )
        
        await websocket_manager.send_emergency_alert(alert)
    
    async def on_system_state_change(self, old_state: str, new_state: str, reason: Optional[str] = None):
        """Handle system state changes"""
        await websocket_manager.send_system_event(
            event_name="system_state_change",
            description=f"System state changed from {old_state} to {new_state}" + 
                       (f": {reason}" if reason else ""),
            level=AlertLevel.INFO,
            affected_components=["orchestrator"]
        )
    
    async def on_configuration_change(self, component: str, changes: Dict[str, Any]):
        """Handle configuration changes"""
        await websocket_manager.broadcast_event(
            EventType.CONFIGURATION_CHANGE,
            component,
            {
                "component": component,
                "changes": changes,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def on_error_occurred(self, error_info: Dict[str, Any]):
        """Handle system errors"""
        await websocket_manager.send_system_event(
            event_name="error_occurred",
            description=error_info.get("message", "System error occurred"),
            level=AlertLevel.ERROR,
            affected_components=error_info.get("affected_components", [])
        )
    
    async def broadcast_custom_event(self, event_type: EventType, source: str, 
                                   data: Dict[str, Any], target_connections: Optional[List[str]] = None):
        """Broadcast custom event"""
        await websocket_manager.broadcast_event(event_type, source, data, target_connections)


# Global event handler instance
websocket_event_handler = WebSocketEventHandler()