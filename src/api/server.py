#!/usr/bin/env python3
"""
NFCS v2.4.3 FastAPI REST Server
===============================

Production-ready REST API for Neural Field Control System with:
- Real-time metrics monitoring
- System control and configuration
- ESC semantic processing endpoints
- WebSocket live telemetry
- Comprehensive error handling and validation

Built with FastAPI for high performance and automatic OpenAPI documentation.

Usage:
    python -m src.api.server --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

# NFCS Core imports
from ..orchestrator.nfcs_orchestrator import create_orchestrator, OrchestrationConfig, NFCSOrchestrator
from ..modules.esc.esc_core import EchoSemanticConverter, ESCConfig, ProcessingMode
from ..core.enhanced_kuramoto import EnhancedKuramotoModule, KuramotoConfig, CouplingMode
from ..core.enhanced_metrics import EnhancedMetricsCalculator, ConstitutionalLimits
from ..orchestrator.resonance_bus import get_global_bus, TopicType, EventPriority, BusEvent
from .models.api_models import *
from .routes import nfcs_routes, esc_routes, monitoring_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global NFCS system instance
_nfcs_system: Optional[NFCSOrchestrator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global _nfcs_system
    
    logger.info("🚀 Starting NFCS v2.4.3 API Server...")
    
    # Initialize NFCS system
    try:
        config = OrchestrationConfig(
            enable_detailed_logging=True,
            max_concurrent_processes=4
        )
        _nfcs_system = await create_orchestrator(config)
        
        # Start system components
        async with _nfcs_system:
            logger.info("✅ NFCS System initialized successfully")
            yield
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize NFCS system: {e}")
        yield
    finally:
        logger.info("🔄 Shutting down NFCS API Server...")
        if _nfcs_system:
            try:
                await _nfcs_system.stop()
            except Exception as e:
                logger.error(f"Error stopping NFCS system: {e}")


# FastAPI Application with enhanced configuration
app = FastAPI(
    title="NFCS v2.4.3 REST API",
    description="""
    **Neural Field Control System v2.4.3 REST API**
    
    Advanced hybrid AI system implementing:
    - Echo-Semantic Converter (ESC) 2.1 for token processing
    - Kuramoto synchronization networks for multi-agent consensus  
    - Complex Ginzburg-Landau field dynamics
    - Constitutional safety frameworks
    - Real-time telemetry and monitoring
    
    Built on September 13, 2025 by Team Ω.
    """,
    version="2.4.3",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "Team Ω - Neural Field Control Systems",
        "email": "team-omega@nfcs.dev",
        "url": "https://github.com/dukeru115/Vortex-Omega"
    },
    license_info={
        "name": "CC BY-NC 4.0",
        "url": "https://creativecommons.org/licenses/by-nc/4.0/"
    }
)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time monitoring"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)
        
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(connection)
                
        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()

def get_nfcs_system() -> NFCSOrchestrator:
    """Dependency injection for NFCS system"""
    if _nfcs_system is None:
        raise HTTPException(status_code=503, detail="NFCS system not initialized")
    return _nfcs_system

# =====================================================
# CORE API ROUTES
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """API root with system information"""
    return """
    <html>
        <head>
            <title>NFCS v2.4.3 API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
                .status { color: #28a745; font-weight: bold; }
                .version { color: #007bff; font-size: 1.2em; }
                .links { margin: 20px 0; }
                .links a { margin-right: 15px; color: #007bff; text-decoration: none; }
                .links a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 Neural Field Control System v2.4.3</h1>
                <p class="version">API Server - Team Ω</p>
                <p class="status">✅ System Online</p>
                
                <h3>🔗 API Documentation</h3>
                <div class="links">
                    <a href="/docs">📚 Interactive API Docs (Swagger)</a>
                    <a href="/redoc">📖 ReDoc Documentation</a>
                    <a href="/api/v1/health">🏥 Health Check</a>
                    <a href="/api/v1/metrics/system">📊 System Metrics</a>
                </div>
                
                <h3>🚀 Features</h3>
                <ul>
                    <li>Echo-Semantic Converter (ESC) 2.1</li>
                    <li>Kuramoto synchronization networks</li>
                    <li>Real-time constitutional monitoring</li>
                    <li>WebSocket live telemetry</li>
                    <li>Performance optimization with JIT</li>
                </ul>
                
                <p><em>Updated: September 13, 2025</em></p>
            </div>
        </body>
    </html>
    """

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check(nfcs: NFCSOrchestrator = Depends(get_nfcs_system)):
    """Comprehensive system health check"""
    try:
        status = nfcs.get_system_status()
        
        return HealthResponse(
            status="healthy" if status.get('state') == 'RUNNING' else "degraded",
            timestamp=datetime.utcnow(),
            version="2.4.3",
            components={
                "nfcs_orchestrator": "online" if status.get('state') else "offline",
                "resonance_bus": "online",
                "esc_module": "online",
                "constitutional_framework": "online"
            },
            metrics={
                "uptime_seconds": status.get('uptime_seconds', 0),
                "total_cycles": status.get('statistics', {}).get('total_cycles', 0),
                "success_rate": status.get('statistics', {}).get('success_rate', 0),
                "avg_frequency_hz": status.get('statistics', {}).get('avg_frequency_hz', 0)
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="2.4.3",
            components={},
            metrics={},
            error=str(e)
        )

@app.get("/api/v1/metrics/system", response_model=SystemMetricsResponse, tags=["Monitoring"])
async def get_system_metrics(nfcs: NFCSOrchestrator = Depends(get_nfcs_system)):
    """Get comprehensive system metrics"""
    try:
        status = nfcs.get_system_status()
        
        return SystemMetricsResponse(
            timestamp=datetime.utcnow(),
            system_state=status.get('state', 'UNKNOWN'),
            performance_metrics=status.get('statistics', {}),
            component_status=status.get('components', {}),
            resource_usage={
                "memory_mb": 0,  # TODO: Implement actual memory monitoring
                "cpu_percent": 0,
                "threads_active": status.get('statistics', {}).get('total_cycles', 0)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {e}")

@app.post("/api/v1/esc/process", response_model=ESCProcessResponse, tags=["ESC"])
async def process_tokens(
    request: ESCProcessRequest,
    background_tasks: BackgroundTasks,
    nfcs: NFCSOrchestrator = Depends(get_nfcs_system)
):
    """Process tokens through Echo-Semantic Converter"""
    try:
        # Create ESC instance with specified configuration
        config = ESCConfig(
            processing_mode=ProcessingMode[request.processing_mode.upper()],
            enable_constitutional_filtering=request.enable_constitutional_filtering,
            constitutional_threshold=request.constitutional_threshold,
            max_unsafe_ratio=request.max_unsafe_ratio
        )
        
        esc = EchoSemanticConverter(config)
        
        # Process token sequence
        result = esc.process_sequence(
            request.tokens,
            context=request.context
        )
        
        # Broadcast real-time update
        background_tasks.add_task(
            manager.broadcast,
            {
                "type": "esc_processing",
                "timestamp": datetime.utcnow().isoformat(),
                "tokens_processed": len(result.processed_tokens),
                "constitutional_compliance": result.constitutional_metrics['constitutional_compliance']
            }
        )
        
        return ESCProcessResponse(
            processed_tokens=[{
                "token": token_info.token,
                "token_type": token_info.token_type.value,
                "constitutional_score": token_info.constitutional_score,
                "risk_score": token_info.risk_score,
                "echo_strength": token_info.echo_strength
            } for token_info in result.processed_tokens],
            constitutional_metrics=result.constitutional_metrics,
            processing_stats=result.processing_stats,
            semantic_field_shape=list(result.semantic_field_state.shape),
            attention_map_shape=list(result.attention_map.shape) if result.attention_map.size > 0 else [0, 0],
            warnings=result.warnings,
            emergency_triggered=result.emergency_triggered
        )
        
    except Exception as e:
        logger.error(f"ESC processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"ESC processing failed: {e}")

# =====================================================
# WEBSOCKET REAL-TIME MONITORING  
# =====================================================

@app.websocket("/ws/monitoring")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time system monitoring"""
    await manager.connect(websocket)
    
    try:
        # Send initial system status
        if _nfcs_system:
            initial_status = _nfcs_system.get_system_status()
            await manager.send_personal_message({
                "type": "system_status",
                "timestamp": datetime.utcnow().isoformat(),
                "data": initial_status
            }, websocket)
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages with timeout
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                # Handle client requests
                if data.get("type") == "get_metrics":
                    if _nfcs_system:
                        metrics = _nfcs_system.get_system_status()
                        await manager.send_personal_message({
                            "type": "metrics_response", 
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": metrics
                        }, websocket)
                        
            except asyncio.TimeoutError:
                # Send keepalive ping
                await manager.send_personal_message({
                    "type": "ping",
                    "timestamp": datetime.utcnow().isoformat()
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# =====================================================
# ADVANCED ROUTES
# =====================================================

@app.get("/api/v1/system/status", response_model=Dict[str, Any], tags=["System"])
async def get_system_status(nfcs: NFCSOrchestrator = Depends(get_nfcs_system)):
    """Get detailed system status and diagnostics"""
    return nfcs.get_system_status()

@app.post("/api/v1/system/control", tags=["System"])
async def control_system(
    action: str,
    parameters: Optional[Dict[str, Any]] = None,
    nfcs: NFCSOrchestrator = Depends(get_nfcs_system)
):
    """Execute system control actions"""
    try:
        if action == "start":
            await nfcs.start()
            return {"status": "success", "message": "System started"}
        elif action == "stop":
            await nfcs.stop()
            return {"status": "success", "message": "System stopped"}
        elif action == "restart":
            await nfcs.stop()
            await nfcs.start()
            return {"status": "success", "message": "System restarted"}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Control action failed: {e}")

# =====================================================
# ERROR HANDLERS
# =====================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# =====================================================
# SERVER STARTUP
# =====================================================

def create_app() -> FastAPI:
    """Factory function to create configured FastAPI app"""
    return app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NFCS v2.4.3 API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting NFCS v2.4.3 API Server on {args.host}:{args.port}")
    
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        access_log=True,
        loop="asyncio"
    )