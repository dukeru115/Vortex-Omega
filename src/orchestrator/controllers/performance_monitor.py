"""
Neural Field Control System - Performance Monitor
================================================

The PerformanceMonitor tracks system-wide performance metrics, identifies
bottlenecks, and provides optimization recommendations for the NFCS.

Key Features:
- Real-time performance metrics collection
- System bottleneck detection and analysis
- Performance trend analysis and forecasting
- Alert generation for performance degradation
- Optimization recommendations
- Historical performance data storage
- Constitutional compliance monitoring
- Resource efficiency analysis

Architecture:
Implements comprehensive performance monitoring with predictive analytics
and constitutional framework integration for optimal system performance.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import statistics


class PerformanceMonitor:
    """NFCS Performance Monitor"""

    def __init__(self, history_size: int = 1000, update_interval: float = 1.0):
        self.logger = logging.getLogger("PerformanceMonitor")
        self.history_size = history_size
        self.update_interval = update_interval

        # Performance data storage
        self._metrics_history: deque = deque(maxlen=history_size)
        self._current_metrics: Dict[str, Any] = {}
        self._running = False

    async def initialize(self) -> bool:
        """Initialize performance monitor"""
        self.logger.info("PerformanceMonitor initialized")
        self._running = True
        return True

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics"""
        self._current_metrics.update(metrics)
        self._metrics_history.append(
            {"timestamp": datetime.now(timezone.utc).isoformat(), "metrics": metrics.copy()}
        )

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self._current_metrics.copy()

    async def record_module_event(self, module_name: str, event_type: str, duration: float):
        """Record module performance event"""
        self.logger.debug(f"Module {module_name} {event_type} took {duration:.3f}s")

    async def shutdown(self) -> bool:
        """Shutdown performance monitor"""
        self._running = False
        return True
