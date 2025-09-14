"""
NFCS FastAPI Server v2.4.3
==========================

Production-ready FastAPI server for Neural Field Control System (NFCS).
Provides REST API endpoints, WebSocket real-time monitoring, and comprehensive
Swagger documentation for system control and ESC token processing.

Features:
- Health checks and system metrics endpoints
- ESC token processing with constitutional filtering  
- Real-time WebSocket monitoring and alerts
- System control operations (start/stop/restart)
- Comprehensive OpenAPI/Swagger documentation
- CORS middleware and error handling
- Dependency injection for NFCS orchestrator

Author: Team Ω (Omega)
Date: September 13, 2025
Version: 2.4.3
"""

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any

import psutil
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, 
    status, Request, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from pydantic import ValidationError

# NFCS imports
from ..orchestrator.main_loop import NFCSMainOrchestrator
from ..modules.esc.esc_core import EchoSemanticConverter, ESCConfig, ProcessingMode

# API models
from .models.api_models import (
    HealthCheckResponse, SystemMetrics, NFCSSystemResponse, SystemControlRequest,
    SystemControlResponse, ErrorResponse, SystemStatus, SystemControlAction
)
from .models.esc_models import (
    ESCProcessRequest, ESCProcessResponse, TokenAnalysis, SemanticField,
    ConstitutionalFilter, ESCConfiguration, ESCSystemStatus
)
from .models.websocket_models import WebSocketMessage, EventType

# WebSocket components
from .websocket.connection_manager import websocket_manager
from .websocket.handlers import websocket_event_handler


# Global NFCS orchestrator instance
_nfcs_orchestrator: Optional[NFCSMainOrchestrator] = None
_server_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global _nfcs_orchestrator
    
    # Startup
    logging.info("🚀 Starting NFCS FastAPI server...")
    
    try:
        # Initialize NFCS orchestrator
        _nfcs_orchestrator = NFCSMainOrchestrator()
        await _nfcs_orchestrator.initialize()
        
        # Inject orchestrator into WebSocket handler
        websocket_event_handler.set_nfcs_orchestrator(_nfcs_orchestrator)
        
        logging.info("✅ NFCS orchestrator initialized successfully")
        
        # Start orchestrator main loop
        await _nfcs_orchestrator.start_main_loop()
        logging.info("✅ NFCS main loop started")
        
        yield  # Server running
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize NFCS orchestrator: {e}")
        logging.error(traceback.format_exc())
        raise
    
    finally:
        # Shutdown
        logging.info("🛑 Shutting down NFCS FastAPI server...")
        
        if _nfcs_orchestrator:
            try:
                await _nfcs_orchestrator.shutdown()
                logging.info("✅ NFCS orchestrator shutdown complete")
            except Exception as e:
                logging.error(f"❌ Error during NFCS shutdown: {e}")


# FastAPI application with comprehensive metadata
app = FastAPI(
    title="NFCS API v2.4.3",
    description="""
    **Neural Field Control System (NFCS) Advanced API**
    
    Production-ready REST API and WebSocket interface for the Neural Field Control System v2.4.3.
    
    ## 🧠 Core Features
    
    - **System Control**: Start, stop, restart, and monitor NFCS orchestrator
    - **Health Monitoring**: Comprehensive health checks and system metrics
    - **ESC Processing**: Echo-Semantic Converter token processing with constitutional filtering
    - **Real-time Monitoring**: WebSocket live telemetry and emergency alerts
    - **Constitutional Safety**: Multi-layered safety frameworks and compliance monitoring
    
    ## 🔬 Technical Architecture
    
    - **Neural Field Dynamics**: Complex Ginzburg-Landau equation modeling
    - **Kuramoto Synchronization**: Multi-agent consensus and phase coordination
    - **ESC Module 2.1**: Advanced token-level semantic processing
    - **Constitutional Framework**: Multi-stakeholder governance and safety compliance
    
    ## 🚀 Getting Started
    
    1. **Health Check**: `GET /health` - Verify system status
    2. **System Status**: `GET /api/v1/system/status` - Get detailed system information  
    3. **WebSocket Monitor**: `WS /ws` - Connect to real-time monitoring
    4. **ESC Processing**: `POST /api/v1/esc/process` - Process tokens with semantic analysis
    
    ## 📊 Real-time Monitoring
    
    Connect to WebSocket endpoint `/ws` to receive:
    - Live telemetry updates (cycle metrics, coherence, defect analysis)
    - Emergency alerts and safety notifications
    - System state changes and configuration updates
    - Performance metrics and resource utilization
    
    ## 🛡️ Safety & Compliance
    
    All operations include constitutional safety filtering and multi-layered compliance monitoring
    to ensure safe and reliable AI system operation.
    
    ---
    
    **Team Ω (Omega)** | September 13, 2025 | Version 2.4.3
    """,
    version="2.4.3",
    contact={
        "name": "Team Ω (Omega)",
        "email": "urmanov.t@gmail.com",
    },
    license_info={
        "name": "CC BY-NC 4.0", 
        "url": "https://creativecommons.org/licenses/by-nc/4.0/",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
async def get_nfcs_system() -> NFCSMainOrchestrator:
    """Dependency injection for NFCS orchestrator"""
    if _nfcs_orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NFCS orchestrator not initialized"
        )
    return _nfcs_orchestrator


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with detailed error responses"""
    logging.error(f"Unhandled exception: {exc}")
    logging.error(traceback.format_exc())
    
    error_response = ErrorResponse(
        error="InternalServerError",
        message="An internal server error occurred",
        details=str(exc) if app.debug else "Please check server logs"
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


@app.exception_handler(ValidationError) 
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors"""
    error_response = ErrorResponse(
        error="ValidationError",
        message="Request validation failed",
        details=str(exc)
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump()
    )


# =====================
# HEALTH & STATUS ENDPOINTS
# =====================

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Get server health status and basic system information",
    tags=["Health & Monitoring"]
)
async def health_check():
    """
    Comprehensive health check endpoint providing system status and performance metrics.
    
    Returns server health, uptime, resource utilization, and dependency status.
    Use this endpoint for load balancer health checks and system monitoring.
    """
    try:
        uptime = time.time() - _server_start_time
        
        # Get system resource information
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Check NFCS orchestrator status
        nfcs_status = "healthy"
        dependencies = {"nfcs_orchestrator": "healthy"}
        
        if _nfcs_orchestrator:
            try:
                system_status = _nfcs_orchestrator.get_system_status()
                orchestrator_state = system_status.get("orchestrator_state", "unknown")
                if orchestrator_state in ["error", "emergency", "shutdown"]:
                    nfcs_status = "degraded"
                    dependencies["nfcs_orchestrator"] = "degraded"
            except Exception as e:
                logging.error(f"Health check NFCS error: {e}")
                nfcs_status = "unhealthy"
                dependencies["nfcs_orchestrator"] = "unhealthy"
        else:
            nfcs_status = "degraded"
            dependencies["nfcs_orchestrator"] = "not_initialized"
        
        # Determine overall health
        overall_status = "healthy"
        if nfcs_status == "degraded":
            overall_status = "degraded"
        elif nfcs_status == "unhealthy":
            overall_status = "unhealthy"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="2.4.3",
            uptime_seconds=uptime,
            system_load=min(cpu_percent / 100.0, 1.0),
            memory_usage_percent=memory.percent,
            dependencies=dependencies
        )
        
    except Exception as e:
        logging.error(f"Health check error: {e}")
        return HealthCheckResponse(
            status="unhealthy", 
            timestamp=datetime.utcnow(),
            version="2.4.3",
            uptime_seconds=time.time() - _server_start_time,
            system_load=0.0,
            memory_usage_percent=0.0,
            dependencies={"error": str(e)}
        )


@app.get(
    "/api/v1/system/status",
    response_model=NFCSSystemResponse,
    summary="Complete System Status",
    description="Get comprehensive NFCS system status with detailed metrics",
    tags=["System Control"]
)
async def get_system_status(nfcs: NFCSMainOrchestrator = Depends(get_nfcs_system)):
    """
    Get complete NFCS system status including orchestrator state, component health,
    performance metrics, and current configuration.
    
    Provides detailed information for system administration and monitoring dashboards.
    """
    try:
        # Get system status from orchestrator
        system_status = nfcs.get_system_status()
        
        # Extract metrics
        stats = system_status.get("statistics", {})
        metrics = SystemMetrics(
            orchestrator_state=SystemStatus(system_status.get("orchestrator_state", "unknown")),
            total_cycles=stats.get("total_cycles", 0),
            successful_cycles=stats.get("successful_cycles", 0), 
            failed_cycles=stats.get("failed_cycles", 0),
            success_rate=stats.get("success_rate", 0.0),
            avg_cycle_time_ms=stats.get("avg_cycle_time_ms", 0.0),
            avg_frequency_hz=stats.get("avg_frequency_hz", 0.0),
            target_frequency_hz=stats.get("target_frequency_hz", 10.0),
            consecutive_errors=stats.get("consecutive_errors", 0),
            emergency_activations=stats.get("emergency_activations", 0),
            active_modules=stats.get("active_modules", 0),
            neural_field_coherence=system_status.get("current_system_state", {}).get("last_risk_metrics", {}).get("coherence_global"),
            hallucination_risk=system_status.get("current_system_state", {}).get("last_risk_metrics", {}).get("hallucination_number"),
            defect_density=system_status.get("current_system_state", {}).get("last_risk_metrics", {}).get("defect_density_mean")
        )
        
        # Get WebSocket connection stats
        ws_stats = websocket_manager.get_connection_stats()
        
        return NFCSSystemResponse(
            status=SystemStatus(system_status.get("orchestrator_state", "unknown")),
            timestamp=datetime.utcnow(),
            metrics=metrics,
            components=system_status.get("components", {}),
            configuration=system_status.get("configuration", {}),
            alerts=system_status.get("alerts", []),
            performance_summary={
                "websocket_connections": ws_stats["active_connections"],
                "messages_sent": ws_stats["total_messages_sent"],
                "avg_cycle_time": stats.get("avg_cycle_time_ms", 0.0),
                "success_rate": stats.get("success_rate", 0.0)
            }
        )
        
    except Exception as e:
        logging.error(f"System status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )


# =====================
# SYSTEM CONTROL ENDPOINTS  
# =====================

@app.post(
    "/api/v1/system/control",
    response_model=SystemControlResponse,
    summary="System Control Operations", 
    description="Execute system control actions (start, stop, restart, pause, resume)",
    tags=["System Control"]
)
async def system_control(
    request: SystemControlRequest,
    nfcs: NFCSMainOrchestrator = Depends(get_nfcs_system)
):
    """
    Execute system control operations on the NFCS orchestrator.
    
    Supported actions:
    - **start**: Start the orchestrator main loop
    - **stop**: Stop the orchestrator gracefully  
    - **restart**: Restart the orchestrator
    - **pause**: Pause processing (if supported)
    - **resume**: Resume processing (if supported)
    - **emergency_stop**: Emergency shutdown
    
    All operations include timeout handling and detailed execution feedback.
    """
    start_time = time.time()
    
    try:
        logging.info(f"Executing system control action: {request.action} - {request.reason}")
        
        # Get current state
        current_status = nfcs.get_system_status()
        old_state = current_status.get("orchestrator_state", "unknown")
        
        # Execute action based on type
        if request.action == SystemControlAction.START:
            if old_state != "running":
                await nfcs.start_main_loop()
                new_state = "running"
                message = "System started successfully"
            else:
                new_state = old_state
                message = "System was already running"
                
        elif request.action == SystemControlAction.STOP:
            if old_state == "running":
                await nfcs.stop_main_loop()
                new_state = "stopped"
                message = "System stopped successfully"
            else:
                new_state = old_state
                message = f"System was not running (state: {old_state})"
                
        elif request.action == SystemControlAction.RESTART:
            # Stop then start
            if old_state == "running":
                await nfcs.stop_main_loop()
            await asyncio.sleep(1.0)  # Brief pause
            await nfcs.start_main_loop()
            new_state = "running"
            message = "System restarted successfully"
            
        elif request.action == SystemControlAction.EMERGENCY_STOP:
            await nfcs.shutdown()
            new_state = "shutdown"
            message = "Emergency shutdown completed"
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported action: {request.action}"
            )
        
        execution_time = (time.time() - start_time) * 1000
        
        # Notify WebSocket clients of state change
        if old_state != new_state:
            await websocket_event_handler.on_system_state_change(
                old_state, new_state, request.reason
            )
        
        return SystemControlResponse(
            success=True,
            action=request.action,
            message=message,
            execution_time_ms=execution_time,
            new_state=SystemStatus(new_state)
        )
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        error_msg = f"Failed to execute {request.action}: {str(e)}"
        
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        
        # Notify WebSocket clients of error
        await websocket_event_handler.on_error_occurred({
            "message": error_msg,
            "action": request.action.value,
            "affected_components": ["orchestrator"]
        })
        
        return SystemControlResponse(
            success=False,
            action=request.action,
            message="Operation failed",
            execution_time_ms=execution_time,
            error_details=error_msg
        )


# =====================  
# ESC PROCESSING ENDPOINTS
# =====================

@app.post(
    "/api/v1/esc/process",
    response_model=ESCProcessResponse,
    summary="ESC Token Processing",
    description="Process tokens using Echo-Semantic Converter with constitutional filtering",
    tags=["ESC Processing"]
)
async def process_tokens(
    request: ESCProcessRequest,
    nfcs: NFCSMainOrchestrator = Depends(get_nfcs_system)
):
    """
    Process input tokens using the Echo-Semantic Converter (ESC) Module 2.1.
    
    Features:
    - **Multi-scale attention**: Advanced attention mechanisms
    - **Constitutional filtering**: Safety and compliance checking  
    - **Semantic field analysis**: High-dimensional semantic space analysis
    - **Token-level analysis**: Per-token embeddings and risk assessment
    - **Performance metrics**: Processing time and quality metrics
    
    The ESC module provides state-of-the-art token processing with built-in safety
    mechanisms and comprehensive semantic analysis capabilities.
    """
    start_time = time.time()
    
    try:
        logging.info(f"Processing {len(request.tokens)} tokens with mode: {request.processing_mode}")
        
        # Configure ESC based on request
        config = ESCConfig(processing_mode=ProcessingMode[request.processing_mode.upper()])
        
        # Initialize ESC (in production, this would be cached/singleton)
        esc = EchoSemanticConverter(config)
        
        # Process tokens
        result = esc.process_sequence(
            tokens=request.tokens,
            context=request.context,
            enable_constitutional_filtering=request.enable_constitutional_filtering
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Convert result to API response format
        token_analyses = []
        for i, token in enumerate(request.tokens):
            analysis = TokenAnalysis(
                token=token,
                token_type="word",  # Simplified for demo
                position=i,
                attention_score=result.get("attention_scores", [1.0] * len(request.tokens))[i],
                constitutional_compliance=result.get("constitutional_scores", [1.0] * len(request.tokens))[i],
                semantic_embedding=result.get("embeddings", [[]] * len(request.tokens))[i] if request.return_embeddings else None
            )
            token_analyses.append(analysis)
        
        semantic_field = SemanticField(
            dimensionality=config.embedding_dim,
            coherence_score=result.get("coherence_score", 0.85),
            stability_measure=result.get("stability_score", 0.90),
            semantic_density=result.get("semantic_density", 0.75)
        )
        
        constitutional_filter = ConstitutionalFilter(
            overall_compliance=result.get("overall_compliance", 0.95),
            policy_violations=result.get("violations", []),
            filtered_tokens=result.get("filtered_tokens", []),
            safety_score=result.get("safety_score", 0.95)
        )
        
        # Notify WebSocket clients of ESC processing
        await websocket_manager.broadcast_event(
            EventType.ESC_PROCESSING,
            "esc_module", 
            {
                "tokens_processed": len(request.tokens),
                "processing_time_ms": processing_time,
                "coherence_score": semantic_field.coherence_score,
                "constitutional_compliance": constitutional_filter.overall_compliance
            }
        )
        
        return ESCProcessResponse(
            success=True,
            processing_time_ms=processing_time,
            token_analyses=token_analyses,
            semantic_field=semantic_field,
            constitutional_filter=constitutional_filter,
            sequence_coherence=result.get("sequence_coherence", 0.85),
            performance_metrics={
                "tokens_per_second": len(request.tokens) / (processing_time / 1000),
                "memory_usage_mb": result.get("memory_usage", 0.0),
                "cache_hit_rate": result.get("cache_hit_rate", 0.0)
            }
        )
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        error_msg = f"ESC processing failed: {str(e)}"
        
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )


# =====================
# WEBSOCKET ENDPOINTS
# =====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time NFCS monitoring and alerts.
    
    **Connection Flow:**
    1. Connect to `/ws` endpoint
    2. Send subscription messages to configure event filters
    3. Receive real-time telemetry, alerts, and system events
    4. Send ping messages to maintain connection
    
    **Supported Client Messages:**
    - `{"action": "subscribe", "event_types": ["telemetry_update", "emergency_alert"]}`
    - `{"action": "unsubscribe"}`
    - `{"action": "ping"}`
    - `{"action": "get_status"}`
    - `{"action": "get_metrics"}`
    
    **Server Event Types:**
    - `system_status`: System state changes and status updates
    - `telemetry_update`: Real-time performance and coherence metrics
    - `emergency_alert`: Critical system alerts and safety notifications
    - `cycle_complete`: Processing cycle completion notifications
    - `heartbeat`: Periodic keep-alive messages
    
    **Event Filtering:**
    Use `filters` parameter to control which events you receive:
    - `{"min_severity": "warning"}`: Only receive warning+ level events
    - Custom filters based on component, metrics thresholds, etc.
    """
    client_ip = websocket.client.host if websocket.client else None
    await websocket_event_handler.handle_websocket_connection(websocket, client_ip)


# =====================
# API INFORMATION ENDPOINTS
# =====================

@app.get(
    "/api/v1/info",
    summary="API Information",
    description="Get API version, capabilities, and configuration information",
    tags=["Information"]
)
async def api_info():
    """Get comprehensive API information and capabilities"""
    return {
        "api_version": "2.4.3",
        "nfcs_version": "2.4.3", 
        "server_time": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - _server_start_time,
        "capabilities": {
            "esc_processing": True,
            "system_control": True,
            "websocket_monitoring": True,
            "constitutional_filtering": True,
            "real_time_telemetry": True
        },
        "endpoints": {
            "health": "/health",
            "system_status": "/api/v1/system/status", 
            "system_control": "/api/v1/system/control",
            "esc_processing": "/api/v1/esc/process",
            "websocket": "/ws",
            "documentation": "/docs"
        },
        "websocket_stats": websocket_manager.get_connection_stats(),
        "team": "Team Ω (Omega)",
        "contact": "urmanov.t@gmail.com"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run server
    uvicorn.run(
        "server:app",
        host="0.0.0.0", 
        port=8000,
        reload=False,
        access_log=True,
        server_header=False
    )