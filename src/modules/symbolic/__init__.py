"""
Symbolic AI Module for Vortex-Omega NFCS
=========================================

This module implements the Symbolic AI boundary interface between the Field and the Other,
managing discrete-continuous transformations and symbolic reasoning.

Based on Technical Specification v2.4.3 for Neural Field Control System.

Components:
- SymbolicAI: Main symbolic reasoning engine
- SymbolicNeuralBridge: S ↔ φ transformation interface (NEW)
- SymbolicParser: Extract and canonicalize symbolic clauses from text
- UnitSystem: Physical unit handling and dimensional analysis
- SymbolicVerifier: Check dimensional, numeric, logical, and ethical consistency
- DiscrepancyGate: Detect and resolve inconsistencies
- KantMode: Ethical universalization and means-end testing

NFCS Integration Features:
- Implements Equation 25: φ_symbolic(x,t) = Σ w_s(t) · Ψ_s(x) · δ_logic[s]
- Real-time symbolic-neural field transformations
- Constitutional oversight integration
- Hallucination Number (Ha) contribution monitoring

Author: Team Omega (GenSpark AI Implementation)
License: CC BY-NC 4.0
"""

from .symbolic_core import SymbolicAI
from .neural_bridge import SymbolicNeuralBridge, BasisFunction, SymbolicWeight
from .models import (
    SymClause,
    SymField,
    VerificationReport,
    ClauseType,
    VerificationStatus,
    Unit,
    Quantity,
    Discrepancy,
    Suggestion,
    Term,
    Expression,
)
from .parser import SymbolicParser
from .units import UnitSystem
from .verifier import SymbolicVerifier
from .discrepancy_gate import DiscrepancyGate
from .kant_mode import KantMode

__all__ = [
    # Core Components
    "SymbolicAI",
    "SymbolicNeuralBridge",
    # Bridge Components (NEW)
    "BasisFunction",
    "SymbolicWeight",
    # Data Models
    "SymClause",
    "SymField",
    "VerificationReport",
    "ClauseType",
    "VerificationStatus",
    "Unit",
    "Quantity",
    "Discrepancy",
    "Suggestion",
    "Term",
    "Expression",
    # Processing Components
    "SymbolicParser",
    "UnitSystem",
    "SymbolicVerifier",
    "DiscrepancyGate",
    "KantMode",
]

__version__ = "2.4.3"
