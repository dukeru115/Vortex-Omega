"""
RAG (Retrieval-Augmented Generation) Module

Advanced RAG system with:
- External knowledge base integration (Wikipedia, scientific articles)
- Conformal abstention for uncertain answers
- Hallucination reduction mechanisms
- Constitutional safety integration
"""

from .rag_core import RAGProcessor, RAGConfig, RetrievalResult
from .knowledge_integrator import KnowledgeBaseIntegrator, WikipediaConnector, ScientificArticleConnector
from .conformal_abstention import ConformalAbstentionSystem, UncertaintyEstimator
from .hallucination_detector import HallucinationDetector, HallucinationMetrics

__all__ = [
    'RAGProcessor',
    'RAGConfig', 
    'RetrievalResult',
    'KnowledgeBaseIntegrator',
    'WikipediaConnector',
    'ScientificArticleConnector',
    'ConformalAbstentionSystem',
    'UncertaintyEstimator',
    'HallucinationDetector',
    'HallucinationMetrics'
]