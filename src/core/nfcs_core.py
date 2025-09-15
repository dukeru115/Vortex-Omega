"""
NFCS Core Module - Neural Field Control System Core
==================================================

Main orchestration core for the Neural Field Control System.
Coordinates between different solver components and manages system state.

Author: Team Omega
License: CC BY-NC 4.0
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .cgl_solver import CGLSolver
from .kuramoto_solver import KuramotoSolver
from .metrics import MetricsCalculator
from .regulator import Regulator

logger = logging.getLogger(__name__)


@dataclass
class NFCSConfig:
    """Configuration for NFCS Core"""

    field_dims: Tuple[int, int] = (64, 64)
    dt: float = 0.01
    max_workers: int = 4
    enable_monitoring: bool = True
    logging_level: str = "INFO"


class NFCSCore:
    """
    Neural Field Control System Core

    Main coordination class that manages the interaction between:
    - CGL (Complex Ginzburg-Landau) solver for field dynamics
    - Kuramoto solver for oscillator synchronization
    - Metrics calculation and monitoring
    - Control regulation and feedback
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize NFCS Core

        Args:
            config: Configuration dictionary
        """
        self.config = NFCSConfig(**(config or {}))

        # Initialize components
        self.cgl_solver = CGLSolver(grid_size=self.config.field_dims, dt=self.config.dt)

        self.kuramoto_solver = KuramotoSolver(
            n_oscillators=np.prod(self.config.field_dims), dt=self.config.dt
        )

        self.metrics_calculator = MetricsCalculator()
        self.regulator = Regulator()

        # State management
        self._field_state = None
        self._oscillator_state = None
        self._running = False

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        logger.info(f"NFCS Core initialized with dimensions {self.config.field_dims}")

    def initialize_field(self, initial_conditions: Optional[np.ndarray] = None) -> np.ndarray:
        """Initialize the complex field with given or random conditions"""
        if initial_conditions is not None:
            self._field_state = initial_conditions
        else:
            self._field_state = self.cgl_solver.create_initial_conditions()
        return self._field_state

    def initialize_oscillators(self, initial_phases: Optional[np.ndarray] = None) -> np.ndarray:
        """Initialize oscillator phases"""
        if initial_phases is not None:
            self._oscillator_state = initial_phases
        else:
            n_osc = np.prod(self.config.field_dims)
            self._oscillator_state = np.random.uniform(0, 2 * np.pi, n_osc)
        return self._oscillator_state

    def step(self, control_field: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform one integration step

        Args:
            control_field: Optional control field to apply

        Returns:
            Dictionary with state information and metrics
        """
        if self._field_state is None:
            self.initialize_field()
        if self._oscillator_state is None:
            self.initialize_oscillators()

        # Update field dynamics
        self._field_state = self.cgl_solver.step(self._field_state, control_field=control_field)

        # Update oscillators (simplified coupling)
        self._oscillator_state = self.kuramoto_solver.step(self._oscillator_state)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_system_state(
            self._field_state, self._oscillator_state
        )

        return {
            "field_state": self._field_state,
            "oscillator_state": self._oscillator_state,
            "metrics": metrics,
            "timestamp": asyncio.get_event_loop().time(),
        }

    async def async_step(self, control_field: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Async version of step"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.step, control_field)

    def get_state(self) -> Dict[str, Any]:
        """Get current system state"""
        return {
            "field_state": self._field_state,
            "oscillator_state": self._oscillator_state,
            "running": self._running,
            "config": self.config.__dict__,
        }

    def reset(self):
        """Reset system to initial state"""
        self._field_state = None
        self._oscillator_state = None
        self._running = False
        logger.info("NFCS Core reset")

    def shutdown(self):
        """Shutdown and cleanup"""
        self._running = False
        self.executor.shutdown(wait=True)
        logger.info("NFCS Core shutdown")

    def __repr__(self):
        return f"NFCSCore(dims={self.config.field_dims}, running={self._running})"
