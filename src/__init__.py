"""
Vortex-Omega Neural Field Control System (NFCS)
Main package module
"""

__version__ = "2.5.0"
__author__ = "Team Omega"
__description__ = "Neural Field Control System - Advanced AI orchestration framework"

# Basic module initialization for CI/CD compatibility
import sys
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"Vortex-Omega NFCS v{__version__} module loaded")