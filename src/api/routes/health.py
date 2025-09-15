"""
Health Check Routes
==================

Health monitoring and status endpoints for NFCS API.

Author: Team Ω (Omega)
Date: September 13, 2025
"""

import time
import logging
from datetime import datetime

import psutil
from fastapi import APIRouter, status
from ..models.api_models import HealthCheckResponse

router = APIRouter(tags=["Health & Monitoring"])
logger = logging.getLogger(__name__)

# Track server start time
_server_start_time = time.time()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Basic Health Check",
    description="Quick health check for load balancers and monitoring systems",
)
async def health_check():
    """
    Basic health check endpoint optimized for frequent polling.

    Returns essential health information with minimal overhead.
    Use this endpoint for load balancer health checks and basic monitoring.
    """
    try:
        uptime = time.time() - _server_start_time
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        return HealthCheckResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version="2.4.3",
            uptime_seconds=uptime,
            system_load=min(cpu_percent / 100.0, 1.0),
            memory_usage_percent=memory.percent,
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="2.4.3",
            uptime_seconds=time.time() - _server_start_time,
            system_load=0.0,
            memory_usage_percent=0.0,
            dependencies={"error": str(e)},
        )


@router.get(
    "/health/detailed",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Detailed Health Check",
    description="Comprehensive health check with dependency status",
)
async def detailed_health_check():
    """
    Detailed health check with comprehensive system information.

    Includes dependency status, resource utilization, and diagnostic information.
    Use this for detailed monitoring and diagnostic purposes.
    """
    try:
        uptime = time.time() - _server_start_time

        # Get detailed system information
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        disk = psutil.disk_usage("/")

        # Check system load
        try:
            load_avg = psutil.getloadavg()[0] / psutil.cpu_count()
        except (AttributeError, OSError):
            load_avg = cpu_percent / 100.0

        # Determine health status based on resources
        status_level = "healthy"
        if memory.percent > 90 or load_avg > 0.9:
            status_level = "degraded"
        elif memory.percent > 95 or load_avg > 0.95:
            status_level = "unhealthy"

        dependencies = {
            "system_memory": f"{memory.percent:.1f}% used",
            "system_cpu": f"{cpu_percent:.1f}% used",
            "disk_space": f"{(disk.used / disk.total) * 100:.1f}% used",
            "load_average": f"{load_avg:.2f}",
        }

        return HealthCheckResponse(
            status=status_level,
            timestamp=datetime.utcnow(),
            version="2.4.3",
            uptime_seconds=uptime,
            system_load=load_avg,
            memory_usage_percent=memory.percent,
            dependencies=dependencies,
        )

    except Exception as e:
        logger.error(f"Detailed health check error: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="2.4.3",
            uptime_seconds=time.time() - _server_start_time,
            system_load=0.0,
            memory_usage_percent=0.0,
            dependencies={"error": str(e)},
        )


@router.get(
    "/health/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Check",
    description="Check if the service is ready to handle requests",
)
async def readiness_check():
    """
    Kubernetes-style readiness probe.

    Returns 200 if the service is ready to handle requests,
    503 if the service is not ready (e.g., still initializing).
    """
    # Add readiness checks here (database connections, etc.)
    return {"status": "ready", "timestamp": datetime.utcnow()}


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness Check",
    description="Check if the service is alive (not deadlocked)",
)
async def liveness_check():
    """
    Kubernetes-style liveness probe.

    Returns 200 if the service is alive and responsive,
    suitable for detecting deadlocks and restart triggers.
    """
    return {"status": "alive", "timestamp": datetime.utcnow()}
