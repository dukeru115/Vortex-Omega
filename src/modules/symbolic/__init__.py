"""
Symbolic AI Module for Vortex-Omega NFCS
=========================================

This module implements the Symbolic AI boundary interface between the Field and the Other,
managing discrete-continuous transformations and symbolic reasoning.

Based on Technical Specification v2.4.3 for Neural Field Control System.

Components:
- Symbolizer: Extract and canonicalize symbolic clauses from text
- Fieldizer: Group clauses into semantic fields with invariants
- Verifier: Check dimensional, numeric, logical, and ethical consistency
- Discrepancy Gate: Detect and resolve inconsistencies
- Kant Mode: Ethical universalization and means-end testing

Author: Team Omega
License: CC BY-NC 4.0
"""

from .symbolic_core import SymbolicAI
from .parser import SymbolicParser
from .units import UnitSystem
from .verifier import SymbolicVerifier
from .discrepancy_gate import DiscrepancyGate
from .kant_mode import KantMode

__all__ = [
    'SymbolicAI',
    'SymbolicParser',
    'UnitSystem',
    'SymbolicVerifier',
    'DiscrepancyGate',
    'KantMode'
]

__version__ = '1.0.0'