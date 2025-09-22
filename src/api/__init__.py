"""
NFCS FastAPI Interface
=====================

Advanced Neural Field Control System (NFCS) REST API with WebSocket real-time monitoring.
Provides comprehensive endpoints for system control, ESC token processing, and telemetry.

Features:
- Production-ready FastAPI server with async support
- Real-time WebSocket monitoring and notifications
- Constitutional filtering and safety compliance
- Comprehensive Swagger/OpenAPI documentation
- System health checks and metrics endpoints
- ESC token processing with semantic analysis

Author: Team Ω (Omega)
Date: September 13, 2025
Version: 2.4.3
"""

__version__ = "2.4.3"
__author__ = "Team Ω (Omega)"
__date__ = "2025-09-13"

from .server import app
from .models import *

__all__ = [
    "app",
    "NFCSSystemResponse",
    "ESCProcessRequest", 
    "ESCProcessResponse",
    "SystemMetrics",
    "HealthCheckResponse",
    "WebSocketMessage",
]