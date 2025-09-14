"""
WebSocket Real-time Monitoring Models
====================================

Pydantic models for WebSocket real-time monitoring, events, and notifications.
Supports live telemetry streaming and emergency alerts.

Author: Team Ω (Omega)
Date: September 13, 2025
Version: 2.4.3
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime


class EventType(str, Enum):
    """WebSocket event types"""
    SYSTEM_STATUS = "system_status"
    TELEMETRY_UPDATE = "telemetry_update"
    EMERGENCY_ALERT = "emergency_alert"
    CYCLE_COMPLETE = "cycle_complete"
    ESC_PROCESSING = "esc_processing"
    CONFIGURATION_CHANGE = "configuration_change"
    ERROR_OCCURRED = "error_occurred"
    USER_ACTION = "user_action"
    HEARTBEAT = "heartbeat"


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class WebSocketMessage(BaseModel):
    """Base WebSocket message structure"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event_type": "telemetry_update",
                "timestamp": "2025-09-13T10:30:00Z",
                "source": "orchestrator",
                "data": {"cycle_number": 1547, "coherence": 0.89}
            }
        }
    )
    
    event_type: EventType = Field(..., description="Type of WebSocket event")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    source: str = Field(..., description="Source component generating the event")
    data: Dict[str, Any] = Field(..., description="Event-specific data payload")
    sequence_id: Optional[int] = Field(default=None, description="Message sequence identifier")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation ID")


class SystemEvent(BaseModel):
    """System-level event notification"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event_id": "sys_001547",
                "event_name": "orchestrator_state_change", 
                "description": "System transitioned to running state",
                "level": "info",
                "affected_components": ["orchestrator", "esc"]
            }
        }
    )
    
    event_id: str = Field(..., description="Unique event identifier")
    event_name: str = Field(..., description="Event name/type")
    description: str = Field(..., description="Human-readable event description")
    level: AlertLevel = Field(..., description="Event severity level")
    affected_components: List[str] = Field(
        default_factory=list,
        description="System components affected by this event"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional event metadata"
    )


class TelemetryUpdate(BaseModel):
    """Real-time telemetry data update"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cycle_number": 1547,
                "cycle_time_ms": 8.2,
                "coherence_global": 0.89,
                "coherence_modular": 0.92,
                "hallucination_risk": 0.05,
                "active_modules": 8
            }
        }
    )
    
    cycle_number: int = Field(..., ge=0, description="Current processing cycle number")
    cycle_time_ms: float = Field(..., ge=0.0, description="Last cycle execution time")
    coherence_global: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Global neural field coherence"
    )
    coherence_modular: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Modular coherence measure"
    )
    hallucination_risk: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Current hallucination risk level"
    )
    defect_density: Optional[float] = Field(
        default=None, ge=0.0,
        description="Topological defect density"
    )
    active_modules: int = Field(..., ge=0, description="Number of active modules")
    memory_usage_mb: Optional[float] = Field(
        default=None, ge=0.0,
        description="Current memory usage"
    )
    cpu_usage_percent: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="CPU utilization percentage"
    )


class EmergencyAlert(BaseModel):
    """Emergency system alert"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "alert_id": "EMG_001",
                "alert_type": "high_hallucination_risk",
                "message": "Hallucination risk exceeded threshold (0.95)",
                "severity": "critical",
                "triggered_at": "2025-09-13T10:30:00Z",
                "auto_actions": ["emergency_protocols_activated"]
            }
        }
    )
    
    alert_id: str = Field(..., description="Unique alert identifier")
    alert_type: str = Field(..., description="Type of emergency condition")
    message: str = Field(..., description="Alert description")
    severity: AlertLevel = Field(..., description="Alert severity level")
    triggered_at: datetime = Field(default_factory=datetime.utcnow, description="Alert trigger time")
    trigger_values: Optional[Dict[str, float]] = Field(
        default=None,
        description="Metric values that triggered the alert"
    )
    thresholds: Optional[Dict[str, float]] = Field(
        default=None,
        description="Threshold values that were exceeded"
    )
    auto_actions: List[str] = Field(
        default_factory=list,
        description="Automated actions taken in response"
    )
    requires_intervention: bool = Field(
        default=False,
        description="Whether manual intervention is required"
    )


class WebSocketConnectionInfo(BaseModel):
    """WebSocket connection information"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "connection_id": "ws_12345",
                "client_ip": "192.168.1.100",
                "connected_at": "2025-09-13T10:30:00Z",
                "subscription_filters": ["telemetry_update", "emergency_alert"]
            }
        }
    )
    
    connection_id: str = Field(..., description="Unique connection identifier")
    client_ip: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="Client user agent")
    connected_at: datetime = Field(default_factory=datetime.utcnow, description="Connection timestamp")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")
    subscription_filters: List[EventType] = Field(
        default_factory=list,
        description="Event types this connection is subscribed to"
    )
    is_authenticated: bool = Field(default=False, description="Authentication status")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional connection metadata"
    )


class WebSocketSubscriptionRequest(BaseModel):
    """WebSocket subscription management request"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action": "subscribe",
                "event_types": ["telemetry_update", "emergency_alert"],
                "filters": {"min_severity": "warning"}
            }
        }
    )
    
    action: str = Field(..., description="Action: subscribe, unsubscribe, or update")
    event_types: List[EventType] = Field(
        default_factory=list,
        description="Event types to subscribe to"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Event filtering criteria"
    )


class WebSocketSubscriptionResponse(BaseModel):
    """WebSocket subscription management response"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": true,
                "message": "Successfully subscribed to 2 event types",
                "active_subscriptions": ["telemetry_update", "emergency_alert"],
                "total_connections": 5
            }
        }
    )
    
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Result message")
    active_subscriptions: List[EventType] = Field(
        default_factory=list,
        description="Currently active subscriptions"
    )
    total_connections: int = Field(..., ge=0, description="Total active WebSocket connections")


class HeartbeatMessage(BaseModel):
    """WebSocket heartbeat message"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2025-09-13T10:30:00Z",
                "server_time": "2025-09-13T10:30:00Z",
                "uptime_seconds": 3600,
                "active_connections": 12
            }
        }
    )
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Heartbeat timestamp")
    server_time: datetime = Field(default_factory=datetime.utcnow, description="Server current time")
    uptime_seconds: float = Field(..., ge=0.0, description="Server uptime in seconds")
    active_connections: int = Field(..., ge=0, description="Number of active WebSocket connections")
    system_status: Optional[str] = Field(default=None, description="Brief system status")


class ErrorMessage(BaseModel):
    """WebSocket error message"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error_code": "WS_AUTH_FAILED",
                "error_message": "Authentication failed for WebSocket connection",
                "details": "Invalid or expired authentication token",
                "timestamp": "2025-09-13T10:30:00Z"
            }
        }
    )
    
    error_code: str = Field(..., description="Error code identifier")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    retry_after: Optional[int] = Field(
        default=None, ge=0,
        description="Suggested retry delay in seconds"
    )