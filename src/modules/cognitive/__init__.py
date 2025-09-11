"""
Cognitive Modules for NFCS

Advanced cognitive system implementations:
- Constitution Module: Constitutional framework and policy enforcement
- Boundary Module: System boundaries and safety constraints
- Memory Module: Long-term memory and experience integration  
- Meta-Reflection Module: Self-reflection and meta-cognitive awareness
- Freedom Module: Autonomous decision-making and creative expression

All cognitive modules implement constitutional safety frameworks
and integrate with the NFCS core for coherent operation.
"""

from .constitution.constitution_core import ConstitutionModule, ConstitutionalFramework
from .boundary.boundary_core import BoundaryModule, BoundaryConstraints
from .memory.memory_core import MemoryModule, MemorySystem
from .meta_reflection.reflection_core import MetaReflectionModule, ReflectionFramework
from .freedom.freedom_core import FreedomModule, AutonomousDecisionMaking

__version__ = "1.0.0"
__all__ = [
    "ConstitutionModule",
    "ConstitutionalFramework",
    "BoundaryModule", 
    "BoundaryConstraints",
    "MemoryModule",
    "MemorySystem",
    "MetaReflectionModule",
    "ReflectionFramework",
    "FreedomModule",
    "AutonomousDecisionMaking"
]