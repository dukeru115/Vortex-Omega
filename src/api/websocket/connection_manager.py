"""
WebSocket Connection Manager
===========================

Advanced WebSocket connection management for NFCS real-time monitoring.
Handles connection lifecycle, event broadcasting, and subscription management.

Author: Team Ω (Omega)
Date: September 13, 2025
Version: 2.4.3
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from uuid import uuid4
import weakref

from fastapi import WebSocket, WebSocketDisconnect
from ..models.websocket_models import (
    WebSocketMessage, EventType, AlertLevel, WebSocketConnectionInfo,
    TelemetryUpdate, EmergencyAlert, HeartbeatMessage, ErrorMessage
)


class WebSocketConnection:
    """Individual WebSocket connection wrapper"""
    
    def __init__(self, websocket: WebSocket, connection_id: str, client_ip: Optional[str] = None):
        self.websocket = websocket
        self.connection_id = connection_id
        self.client_ip = client_ip
        self.connected_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.subscriptions: Set[EventType] = set()
        self.filters: Dict[str, Any] = {}
        self.is_authenticated = False
        self.metadata: Dict[str, Any] = {}
        self.message_count = 0
        
        self.logger = logging.getLogger(f"{__name__}.WebSocketConnection")
        
    async def send_message(self, message: WebSocketMessage) -> bool:
        """Send message to WebSocket client"""
        try:
            await self.websocket.send_text(message.model_dump_json())
            self.last_activity = datetime.utcnow()
            self.message_count += 1
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message to {self.connection_id}: {e}")
            return False
    
    async def send_error(self, error_code: str, message: str, details: Optional[str] = None):
        """Send error message to client"""
        error_msg = ErrorMessage(
            error_code=error_code,
            error_message=message,
            details=details
        )
        
        websocket_msg = WebSocketMessage(
            event_type=EventType.ERROR_OCCURRED,
            source="websocket_manager",
            data=error_msg.model_dump()
        )
        
        await self.send_message(websocket_msg)
    
    def update_subscriptions(self, event_types: List[EventType], filters: Optional[Dict[str, Any]] = None):
        """Update subscription preferences"""
        self.subscriptions = set(event_types)
        if filters:
            self.filters.update(filters)
        self.last_activity = datetime.utcnow()
    
    def should_receive_event(self, event_type: EventType, event_data: Dict[str, Any]) -> bool:
        """Check if connection should receive this event"""
        if event_type not in self.subscriptions:
            return False
            
        # Apply filters if any
        if "min_severity" in self.filters:
            severity_levels = ["info", "warning", "error", "critical", "emergency"]
            min_level = self.filters["min_severity"]
            event_level = event_data.get("level", "info")
            
            try:
                if severity_levels.index(event_level) < severity_levels.index(min_level):
                    return False
            except ValueError:
                pass  # Invalid severity level, allow the event
        
        return True
    
    def get_connection_info(self) -> WebSocketConnectionInfo:
        """Get connection information"""
        return WebSocketConnectionInfo(
            connection_id=self.connection_id,
            client_ip=self.client_ip,
            connected_at=self.connected_at,
            last_activity=self.last_activity,
            subscription_filters=list(self.subscriptions),
            is_authenticated=self.is_authenticated,
            metadata=self.metadata
        )


class ConnectionManager:
    """WebSocket connection manager with event broadcasting"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.heartbeat_interval = 30.0  # seconds
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_interval = 300.0  # 5 minutes
        self.cleanup_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(f"{__name__}.ConnectionManager")
        
        # Statistics
        self.total_connections = 0
        self.total_messages_sent = 0
        self.start_time = datetime.utcnow()
        
    async def connect(self, websocket: WebSocket, client_ip: Optional[str] = None) -> str:
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        connection_id = str(uuid4())
        connection = WebSocketConnection(websocket, connection_id, client_ip)
        
        self.connections[connection_id] = connection
        self.total_connections += 1
        
        self.logger.info(f"New WebSocket connection: {connection_id} from {client_ip}")
        
        # Start background tasks if this is the first connection
        if len(self.connections) == 1:
            await self._start_background_tasks()
        
        # Send welcome message
        welcome_msg = WebSocketMessage(
            event_type=EventType.SYSTEM_STATUS,
            source="websocket_manager",
            data={
                "connection_id": connection_id,
                "message": "Connected to NFCS WebSocket API",
                "server_time": datetime.utcnow().isoformat(),
                "available_events": [e.value for e in EventType]
            }
        )
        await connection.send_message(welcome_msg)
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            del self.connections[connection_id]
            
            self.logger.info(f"WebSocket disconnected: {connection_id}")
            
            # Stop background tasks if no connections remain
            if len(self.connections) == 0:
                await self._stop_background_tasks()
    
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Send message to specific connection"""
        connection = self.connections.get(connection_id)
        if connection:
            success = await connection.send_message(message)
            if success:
                self.total_messages_sent += 1
            return success
        return False
    
    async def broadcast_event(self, event_type: EventType, source: str, data: Dict[str, Any], 
                             target_connections: Optional[List[str]] = None):
        """Broadcast event to all subscribed connections"""
        message = WebSocketMessage(
            event_type=event_type,
            source=source,
            data=data,
            sequence_id=self.total_messages_sent
        )
        
        connections_to_notify = []
        
        if target_connections:
            # Send to specific connections
            connections_to_notify = [
                conn for conn_id, conn in self.connections.items()
                if conn_id in target_connections and conn.should_receive_event(event_type, data)
            ]
        else:
            # Send to all subscribed connections
            connections_to_notify = [
                conn for conn in self.connections.values()
                if conn.should_receive_event(event_type, data)
            ]
        
        # Send messages concurrently
        send_tasks = []
        for connection in connections_to_notify:
            task = asyncio.create_task(connection.send_message(message))
            send_tasks.append(task)
        
        if send_tasks:
            results = await asyncio.gather(*send_tasks, return_exceptions=True)
            successful_sends = sum(1 for result in results if result is True)
            self.total_messages_sent += successful_sends
            
            self.logger.debug(
                f"Broadcasted {event_type.value} to {successful_sends}/{len(send_tasks)} connections"
            )
    
    async def update_subscription(self, connection_id: str, event_types: List[EventType], 
                                filters: Optional[Dict[str, Any]] = None) -> bool:
        """Update connection subscription preferences"""
        connection = self.connections.get(connection_id)
        if connection:
            connection.update_subscriptions(event_types, filters)
            
            # Send confirmation
            response_msg = WebSocketMessage(
                event_type=EventType.SYSTEM_STATUS,
                source="websocket_manager",
                data={
                    "action": "subscription_updated",
                    "active_subscriptions": [et.value for et in event_types],
                    "filters": filters or {}
                }
            )
            await connection.send_message(response_msg)
            return True
        return False
    
    async def send_telemetry_update(self, telemetry: TelemetryUpdate):
        """Send telemetry update to subscribed connections"""
        await self.broadcast_event(
            EventType.TELEMETRY_UPDATE,
            "nfcs_orchestrator", 
            telemetry.model_dump()
        )
    
    async def send_emergency_alert(self, alert: EmergencyAlert):
        """Send emergency alert to all connections"""
        await self.broadcast_event(
            EventType.EMERGENCY_ALERT,
            "emergency_controller",
            alert.model_dump()
        )
    
    async def send_system_event(self, event_name: str, description: str, 
                              level: AlertLevel = AlertLevel.INFO, 
                              affected_components: Optional[List[str]] = None):
        """Send system event notification"""
        event_data = {
            "event_id": f"sys_{int(datetime.utcnow().timestamp())}",
            "event_name": event_name,
            "description": description,
            "level": level.value,
            "affected_components": affected_components or []
        }
        
        await self.broadcast_event(
            EventType.SYSTEM_STATUS,
            "system",
            event_data
        )
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "active_connections": len(self.connections),
            "total_connections": self.total_connections,
            "total_messages_sent": self.total_messages_sent,
            "uptime_seconds": uptime,
            "messages_per_second": self.total_messages_sent / max(uptime, 1),
            "connection_details": [
                conn.get_connection_info().model_dump() for conn in self.connections.values()
            ]
        }
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if not self.heartbeat_task:
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _stop_background_tasks(self):
        """Stop background maintenance tasks"""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None
            
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if self.connections:
                    uptime = (datetime.utcnow() - self.start_time).total_seconds()
                    heartbeat = HeartbeatMessage(
                        uptime_seconds=uptime,
                        active_connections=len(self.connections)
                    )
                    
                    await self.broadcast_event(
                        EventType.HEARTBEAT,
                        "websocket_manager",
                        heartbeat.model_dump()
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of stale connections"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Remove stale connections (no activity for 30 minutes)
                stale_threshold = datetime.utcnow() - timedelta(minutes=30)
                stale_connections = [
                    conn_id for conn_id, conn in self.connections.items()
                    if conn.last_activity < stale_threshold
                ]
                
                for conn_id in stale_connections:
                    self.logger.warning(f"Removing stale WebSocket connection: {conn_id}")
                    await self.disconnect(conn_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")


# Global WebSocket manager instance
websocket_manager = ConnectionManager()

# Alias for backward compatibility
WebSocketManager = ConnectionManager