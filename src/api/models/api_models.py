"""
Core API Models for NFCS v2.4.3
===============================

Pydantic models for system control, health checks, and telemetry endpoints.
Follows NFCS architecture with constitutional safety frameworks.

Author: Team Ω (Omega)
Date: September 13, 2025
Version: 2.4.3
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime


class SystemStatus(str, Enum):
    """System operational status enumeration"""
    INITIALIZING = "initializing"
    RUNNING = "running" 
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


class SystemControlAction(str, Enum):
    """System control actions"""
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    PAUSE = "pause"
    RESUME = "resume"
    EMERGENCY_STOP = "emergency_stop"


class HealthCheckResponse(BaseModel):
    """Health check endpoint response"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2025-09-13T10:30:00Z",
                "version": "2.4.3",
                "uptime_seconds": 3600,
                "system_load": 0.75,
                "memory_usage_percent": 42.5
            }
        }
    )
    
    status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
    timestamp: datetime = Field(..., description="Current server timestamp")
    version: str = Field(..., description="NFCS version")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    system_load: float = Field(..., ge=0.0, le=1.0, description="System load (0-1)")
    memory_usage_percent: float = Field(..., ge=0.0, le=100.0, description="Memory usage percentage")
    dependencies: Optional[Dict[str, str]] = Field(
        default=None, 
        description="Dependency health status"
    )


class SystemMetrics(BaseModel):
    """Comprehensive system metrics"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "orchestrator_state": "running",
                "total_cycles": 15420,
                "success_rate": 0.987,
                "avg_cycle_time_ms": 8.5,
                "emergency_activations": 0,
                "active_modules": 8
            }
        }
    )
    
    orchestrator_state: SystemStatus = Field(..., description="Current orchestrator state")
    total_cycles: int = Field(..., ge=0, description="Total processing cycles completed")
    successful_cycles: int = Field(..., ge=0, description="Successfully completed cycles") 
    failed_cycles: int = Field(..., ge=0, description="Failed processing cycles")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate (0-1)")
    avg_cycle_time_ms: float = Field(..., ge=0.0, description="Average cycle time in milliseconds")
    avg_frequency_hz: float = Field(..., ge=0.0, description="Average processing frequency")
    target_frequency_hz: float = Field(..., ge=0.0, description="Target processing frequency")
    consecutive_errors: int = Field(..., ge=0, description="Consecutive error count")
    emergency_activations: int = Field(..., ge=0, description="Emergency protocol activations")
    active_modules: int = Field(..., ge=0, description="Number of active modules")
    
    # Neural field metrics
    neural_field_coherence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, 
        description="Global neural field coherence"
    )
    hallucination_risk: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Current hallucination risk level"
    )
    defect_density: Optional[float] = Field(
        default=None, ge=0.0,
        description="Topological defect density"
    )


class SystemControlRequest(BaseModel):
    """System control operation request"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action": "restart", 
                "reason": "Performance optimization",
                "force": false,
                "timeout_seconds": 30
            }
        }
    )
    
    action: SystemControlAction = Field(..., description="Control action to perform")
    reason: Optional[str] = Field(default=None, description="Reason for the action")
    force: bool = Field(default=False, description="Force action even if system is busy")
    timeout_seconds: Optional[float] = Field(
        default=30.0, ge=1.0, le=300.0,
        description="Operation timeout in seconds"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional parameters for the action"
    )


class SystemControlResponse(BaseModel):
    """System control operation response"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": true,
                "action": "restart",
                "message": "System restarted successfully",
                "execution_time_ms": 2450.5,
                "new_state": "running"
            }
        }
    )
    
    success: bool = Field(..., description="Whether the operation succeeded")
    action: SystemControlAction = Field(..., description="Action that was performed")
    message: str = Field(..., description="Human-readable result message")
    execution_time_ms: float = Field(..., ge=0.0, description="Operation execution time")
    new_state: Optional[SystemStatus] = Field(default=None, description="New system state")
    error_details: Optional[str] = Field(default=None, description="Error details if failed")


class NFCSSystemResponse(BaseModel):
    """Complete NFCS system status response"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "running",
                "timestamp": "2025-09-13T10:30:00Z", 
                "metrics": "...",
                "components": {"orchestrator": "running", "esc": "running"},
                "alerts": []
            }
        }
    )
    
    status: SystemStatus = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Response timestamp")
    metrics: SystemMetrics = Field(..., description="Detailed system metrics")
    components: Dict[str, str] = Field(..., description="Component status mapping")
    configuration: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Current system configuration"
    )
    alerts: List[str] = Field(default_factory=list, description="Active system alerts")
    performance_summary: Optional[Dict[str, float]] = Field(
        default=None,
        description="Performance summary statistics"
    )


class ErrorResponse(BaseModel):
    """Standard API error response"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Invalid input parameters",
                "details": "Field 'tokens' is required",
                "timestamp": "2025-09-13T10:30:00Z"
            }
        }
    )
    
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request tracking ID")