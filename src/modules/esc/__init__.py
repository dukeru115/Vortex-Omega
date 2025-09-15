"""
Echo-Semantic Converter (ESC) Module 2.1

Advanced token processing system for NFCS with:
- Echo-semantic token transformation and encoding
- Multi-scale attention mechanisms
- Semantic field coupling with neural fields
- Constitutional content filtering and safety policies
- Dynamic vocabulary adaptation and learning
"""

from .esc_core import EchoSemanticConverter, ESCConfig
from .token_processor import TokenProcessor, TokenType
from .semantic_fields import SemanticFieldCoupler
from .attention_mechanisms import MultiScaleAttention
from .constitutional_filter import ConstitutionalContentFilter
from .adaptive_vocabulary import AdaptiveVocabulary

__version__ = "2.1.0"
__all__ = [
    "EchoSemanticConverter",
    "ESCConfig",
    "TokenProcessor",
    "TokenType",
    "SemanticFieldCoupler",
    "MultiScaleAttention",
    "ConstitutionalContentFilter",
    "AdaptiveVocabulary",
]
