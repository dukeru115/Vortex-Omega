"""
Symbolic AI Module - NFCS Hybrid Neuro-Symbolic Architecture
===========================================================

This module implements the critical symbolic reasoning component of NFCS v2.4.3,
enabling hybrid neuro-symbolic processing as described in the scientific foundation.

Key Components:
- SymbolicAI Core: Main symbolic reasoning engine
- Knowledge Graph: Structured knowledge representation
- Logic Engine: First-order logic reasoning and inference
- Neuro-Symbolic Bridge: Integration with neural field dynamics

Scientific Foundation:
Based on Section 5.4 of the NFCS research paper, implementing:
- Discrete-continuous transformations
- Symbolization: Φ(field) → symbolic_representation
- Fieldization: symbolic_input → neural_field_modulation
- Verification: consistency_check(symbolic, neural)

Created: September 14, 2025
Author: Team Ω - Neural Field Control Systems Research Group
"""

from .symbolic_ai import (
    SymbolicAI,
    SymbolicQuery,
    SymbolicRepresentation,
    ConsistencyScore,
    FieldModulation,
)

from .knowledge_graph import KnowledgeGraph, ConceptNode, RelationEdge, KnowledgeGraphConfig

from .logic_engine import LogicEngine, LogicalRule, InferenceResult, LogicEngineConfig

from .neuro_symbolic_bridge import (
    NeuroSymbolicBridge,
    SymbolicFieldConverter,
    FieldSymbolicConverter,
)

__all__ = [
    "SymbolicAI",
    "SymbolicQuery",
    "SymbolicRepresentation",
    "ConsistencyScore",
    "FieldModulation",
    "KnowledgeGraph",
    "ConceptNode",
    "RelationEdge",
    "LogicEngine",
    "LogicalRule",
    "InferenceResult",
    "NeuroSymbolicBridge",
]

__version__ = "2.4.3"
__author__ = "Team Ω"
__date__ = "September 14, 2025"
