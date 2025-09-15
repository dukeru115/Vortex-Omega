"""
API Routes Module
================

Modular route definitions for NFCS FastAPI server.
Organizes endpoints by functionality for better maintainability.

Author: Team Ω (Omega)
Date: September 13, 2025
"""

from .health import router as health_router
from .system import router as system_router
from .esc import router as esc_router

__all__ = [
    "health_router",
    "system_router",
    "esc_router",
]
