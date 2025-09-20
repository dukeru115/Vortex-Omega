"""
Knowledge Base Integrator for RAG System

Handles integration with external knowledge sources including:
- Wikipedia API integration
- Scientific article databases
- Internal knowledge bases
- Caching and performance optimization
"""

import time
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from urllib.parse import quote_plus
import re

from .rag_core import RetrievalResult, RetrievalSource

logger = logging.getLogger(__name__)


class KnowledgeConnector(ABC):
    """Abstract base class for knowledge source connectors."""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search the knowledge source."""
        pass
    
    @abstractmethod
    def get_source_type(self) -> RetrievalSource:
        """Get the source type this connector handles."""
        pass


class WikipediaConnector(KnowledgeConnector):
    """Wikipedia API connector with rate limiting and caching."""
    
    def __init__(self, language: str = 'en', max_retries: int = 3):
        self.language = language
        self.max_retries = max_retries
        self.base_url = f"https://{language}.wikipedia.org/api/rest_v1"
        
        # Simple rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Cache for search results
        self.search_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Wikipedia for relevant articles."""
        # Check cache first
        cache_key = f"{query}_{max_results}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Rate limiting
            self._enforce_rate_limit()
            
            # Search for relevant pages
            search_results = self._search_pages(query, max_results)
            
            # Get content for top results
            articles = []
            for result in search_results[:max_results]:
                article_content = self._get_article_content(result['title'])
                if article_content:
                    articles.append({
                        'title': result['title'],
                        'content': article_content,
                        'url': f"https://{self.language}.wikipedia.org/wiki/{quote_plus(result['title'])}",
                        'source': 'wikipedia',
                        'relevance_score': result.get('score', 0.5),
                        'snippet': article_content[:300] + "..." if len(article_content) > 300 else article_content
                    })
            
            # Cache the results
            self._cache_result(cache_key, articles)
            
            return articles
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []
    
    def get_source_type(self) -> RetrievalSource:
        return RetrievalSource.WIKIPEDIA
    
    def _search_pages(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search Wikipedia pages using OpenSearch API."""
        # Mock implementation - in real deployment, would use requests library
        # For now, return mock search results
        mock_results = [
            {'title': f'Wikipedia Article about {query}', 'score': 0.9},
            {'title': f'{query} - Overview', 'score': 0.8},
            {'title': f'Related topic to {query}', 'score': 0.7}
        ]
        return mock_results[:limit]
    
    def _get_article_content(self, title: str) -> Optional[str]:
        """Get article content using Wikipedia REST API."""
        # Mock implementation - in real deployment, would fetch actual content
        return f"This is mock Wikipedia content for the article titled '{title}'. " \
               f"In a real implementation, this would contain the actual Wikipedia article content " \
               f"retrieved from the Wikipedia REST API. The content would be processed to extract " \
               f"the most relevant sections and formatted for use in the RAG system."
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search result if still valid."""
        if cache_key in self.search_cache:
            result, timestamp = self.search_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                del self.search_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: List[Dict[str, Any]]):
        """Cache search result with timestamp."""
        self.search_cache[cache_key] = (result, time.time())


class ScientificArticleConnector(KnowledgeConnector):
    """Connector for scientific article databases (arXiv, PubMed, etc.)."""
    
    def __init__(self, sources: List[str] = None):
        self.sources = sources or ['arxiv', 'pubmed']
        self.search_cache = {}
        self.cache_ttl = 7200  # 2 hours for scientific content
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search scientific article databases."""
        cache_key = f"sci_{query}_{max_results}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        articles = []
        
        # Search each configured source
        for source in self.sources:
            try:
                source_results = self._search_source(source, query, max_results // len(self.sources))
                articles.extend(source_results)
            except Exception as e:
                logger.error(f"Error searching {source}: {e}")
        
        # Sort by relevance and limit results
        articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        articles = articles[:max_results]
        
        # Cache the results
        self._cache_result(cache_key, articles)
        
        return articles
    
    def get_source_type(self) -> RetrievalSource:
        return RetrievalSource.SCIENTIFIC_ARTICLES
    
    def _search_source(self, source: str, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search a specific scientific database."""
        if source == 'arxiv':
            return self._search_arxiv(query, limit)
        elif source == 'pubmed':
            return self._search_pubmed(query, limit)
        else:
            logger.warning(f"Unknown scientific source: {source}")
            return []
    
    def _search_arxiv(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search arXiv database."""
        # Mock implementation - in real deployment, would use arXiv API
        mock_results = []
        for i in range(min(limit, 3)):
            mock_results.append({
                'title': f'arXiv Paper {i+1}: {query} Research',
                'content': f'Abstract: This paper discusses {query} and its applications in various fields. '
                          f'The research presents novel approaches and methodologies related to {query}.',
                'authors': ['Dr. Smith', 'Prof. Johnson'],
                'url': f'https://arxiv.org/abs/mock.{i+1}',
                'source': 'arxiv',
                'publication_date': '2024-01-01',
                'relevance_score': 0.8 - (i * 0.1),
                'citation_count': 50 - (i * 10)
            })
        return mock_results
    
    def _search_pubmed(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search PubMed database."""
        # Mock implementation - in real deployment, would use PubMed API
        mock_results = []
        for i in range(min(limit, 2)):
            mock_results.append({
                'title': f'PubMed Study {i+1}: Clinical Research on {query}',
                'content': f'Abstract: This clinical study investigates {query} in medical contexts. '
                          f'The study provides evidence-based insights into {query} applications.',
                'authors': ['Dr. Wilson', 'MD Anderson'],
                'url': f'https://pubmed.ncbi.nlm.nih.gov/mock{i+1}',
                'source': 'pubmed',
                'publication_date': '2024-02-01',
                'relevance_score': 0.75 - (i * 0.1),
                'journal': 'Journal of Medical Research'
            })
        return mock_results
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search result if still valid."""
        if cache_key in self.search_cache:
            result, timestamp = self.search_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                del self.search_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: List[Dict[str, Any]]):
        """Cache search result with timestamp."""
        self.search_cache[cache_key] = (result, time.time())


class InternalKnowledgeConnector(KnowledgeConnector):
    """Connector for internal knowledge bases."""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base_path = knowledge_base_path
        self.internal_kb = self._load_internal_kb()
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search internal knowledge base."""
        results = []
        query_terms = query.lower().split()
        
        for doc_id, doc_data in self.internal_kb.items():
            # Simple text matching
            content = doc_data.get('content', '').lower()
            title = doc_data.get('title', '').lower()
            
            # Calculate relevance score
            relevance = 0.0
            for term in query_terms:
                relevance += content.count(term) * 0.1
                relevance += title.count(term) * 0.5
            
            if relevance > 0:
                results.append({
                    'id': doc_id,
                    'title': doc_data.get('title', f'Document {doc_id}'),
                    'content': doc_data.get('content', ''),
                    'source': 'internal_kb',
                    'relevance_score': min(1.0, relevance),
                    'metadata': doc_data.get('metadata', {})
                })
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]
    
    def get_source_type(self) -> RetrievalSource:
        return RetrievalSource.INTERNAL_KB
    
    def _load_internal_kb(self) -> Dict[str, Dict[str, Any]]:
        """Load internal knowledge base."""
        # Mock implementation - in real deployment, would load from file/database
        return {
            'doc1': {
                'title': 'NFCS Overview',
                'content': 'Neural Field Control System (NFCS) is an advanced AI architecture that combines '
                          'constitutional AI principles with neural field dynamics for safe and robust AI systems.',
                'metadata': {'category': 'technical', 'last_updated': '2024-01-01'}
            },
            'doc2': {
                'title': 'Constitutional AI Principles',
                'content': 'Constitutional AI involves training AI systems to follow a set of principles or '
                          'constitution that guides their behavior to be helpful, harmless, and honest.',
                'metadata': {'category': 'ai_safety', 'last_updated': '2024-01-15'}
            }
        }


class KnowledgeBaseIntegrator:
    """
    Main integrator coordinating multiple knowledge sources.
    
    Manages multiple knowledge connectors, handles source prioritization,
    and provides unified retrieval interface.
    """
    
    def __init__(self, config):
        self.config = config
        self.connectors = {}
        self.source_weights = {
            RetrievalSource.WIKIPEDIA: 0.7,
            RetrievalSource.SCIENTIFIC_ARTICLES: 0.9,
            RetrievalSource.INTERNAL_KB: 1.0,
            RetrievalSource.CACHED_RESULTS: 0.8
        }
        
        # Initialize connectors based on configuration
        self._initialize_connectors()
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'successful_retrievals': 0,
            'failed_retrievals': 0,
            'average_retrieval_time': 0.0
        }
    
    def _initialize_connectors(self):
        """Initialize knowledge source connectors."""
        if RetrievalSource.WIKIPEDIA in self.config.enabled_sources:
            self.connectors[RetrievalSource.WIKIPEDIA] = WikipediaConnector()
        
        if RetrievalSource.SCIENTIFIC_ARTICLES in self.config.enabled_sources:
            self.connectors[RetrievalSource.SCIENTIFIC_ARTICLES] = ScientificArticleConnector()
        
        if RetrievalSource.INTERNAL_KB in self.config.enabled_sources:
            self.connectors[RetrievalSource.INTERNAL_KB] = InternalKnowledgeConnector()
        
        logger.info(f"Initialized {len(self.connectors)} knowledge connectors")
    
    def retrieve(self, query: str, sources: List[RetrievalSource]) -> RetrievalResult:
        """
        Retrieve knowledge from specified sources.
        
        Args:
            query: Search query
            sources: List of sources to search
            
        Returns:
            RetrievalResult with aggregated results
        """
        start_time = time.time()
        self.retrieval_stats['total_queries'] += 1
        
        all_documents = []
        all_scores = []
        source_metadata = {}
        
        try:
            # Search each enabled source
            for source in sources:
                if source not in self.connectors:
                    logger.warning(f"Connector not available for source: {source}")
                    continue
                
                connector = self.connectors[source]
                
                try:
                    # Get results from this source
                    source_results = connector.search(
                        query, 
                        max_results=self.config.max_retrieved_docs // len(sources)
                    )
                    
                    # Apply source weighting to scores
                    source_weight = self.source_weights.get(source, 0.5)
                    for doc in source_results:
                        doc['relevance_score'] *= source_weight
                        all_documents.append(doc)
                        all_scores.append(doc['relevance_score'])
                    
                    source_metadata[source.value] = {
                        'documents_found': len(source_results),
                        'max_score': max([doc['relevance_score'] for doc in source_results]) if source_results else 0.0
                    }
                    
                except Exception as e:
                    logger.error(f"Error retrieving from {source}: {e}")
                    source_metadata[source.value] = {'error': str(e)}
            
            # Sort all documents by relevance
            all_documents.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Limit to max results
            all_documents = all_documents[:self.config.max_retrieved_docs]
            all_scores = [doc.get('relevance_score', 0) for doc in all_documents]
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence(all_documents, all_scores)
            
            # Calculate constitutional compliance
            constitutional_compliance = self._assess_constitutional_compliance(all_documents)
            
            # Calculate hallucination risk
            hallucination_risk = self._assess_hallucination_risk(query, all_documents)
            
            retrieval_time = time.time() - start_time
            
            # Update performance stats
            if all_documents:
                self.retrieval_stats['successful_retrievals'] += 1
            else:
                self.retrieval_stats['failed_retrievals'] += 1
            
            self._update_average_time(retrieval_time)
            
            return RetrievalResult(
                query=query,
                documents=all_documents,
                similarity_scores=all_scores,
                source_metadata=source_metadata,
                retrieval_time=retrieval_time,
                confidence_score=confidence_score,
                hallucination_risk=hallucination_risk,
                constitutional_compliance=constitutional_compliance
            )
            
        except Exception as e:
            logger.error(f"Knowledge retrieval error: {e}")
            self.retrieval_stats['failed_retrievals'] += 1
            
            return RetrievalResult(
                query=query,
                documents=[],
                similarity_scores=[],
                source_metadata={'error': str(e)},
                retrieval_time=time.time() - start_time,
                confidence_score=0.0,
                hallucination_risk=1.0,
                constitutional_compliance=1.0
            )
    
    def _calculate_confidence(self, documents: List[Dict[str, Any]], scores: List[float]) -> float:
        """Calculate overall confidence score for retrieved documents."""
        if not documents or not scores:
            return 0.0
        
        # Base confidence on highest score and score consistency
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        
        # High variance reduces confidence
        confidence = max_score * (1.0 - min(0.5, score_variance))
        
        # Adjust based on number of documents
        document_factor = min(1.0, len(documents) / 5)  # More documents = higher confidence
        confidence *= document_factor
        
        return min(1.0, confidence)
    
    def _assess_constitutional_compliance(self, documents: List[Dict[str, Any]]) -> float:
        """Assess constitutional compliance of retrieved content."""
        if not documents:
            return 1.0
        
        # Simple content filtering - in real implementation, would use more sophisticated methods
        problematic_terms = ['harmful', 'dangerous', 'illegal', 'unethical']
        
        total_compliance = 0.0
        for doc in documents:
            content = doc.get('content', '').lower()
            compliance = 1.0
            
            for term in problematic_terms:
                if term in content:
                    compliance -= 0.2
            
            total_compliance += max(0.0, compliance)
        
        return total_compliance / len(documents)
    
    def _assess_hallucination_risk(self, query: str, documents: List[Dict[str, Any]]) -> float:
        """Assess hallucination risk based on query-document alignment."""
        if not documents:
            return 1.0  # High risk if no supporting documents
        
        query_terms = set(query.lower().split())
        total_alignment = 0.0
        
        for doc in documents:
            content = doc.get('content', '').lower()
            doc_terms = set(content.split())
            
            # Calculate term overlap
            overlap = len(query_terms.intersection(doc_terms))
            alignment = overlap / len(query_terms) if query_terms else 0.0
            total_alignment += alignment
        
        avg_alignment = total_alignment / len(documents)
        
        # Higher alignment = lower hallucination risk
        return max(0.0, 1.0 - avg_alignment)
    
    def _update_average_time(self, retrieval_time: float):
        """Update average retrieval time."""
        current_avg = self.retrieval_stats['average_retrieval_time']
        total_queries = self.retrieval_stats['total_queries']
        
        # Incremental average update
        new_avg = ((current_avg * (total_queries - 1)) + retrieval_time) / total_queries
        self.retrieval_stats['average_retrieval_time'] = new_avg
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get retrieval performance statistics."""
        total = self.retrieval_stats['total_queries']
        success_rate = (
            self.retrieval_stats['successful_retrievals'] / total 
            if total > 0 else 0.0
        )
        
        return {
            'total_queries': total,
            'successful_retrievals': self.retrieval_stats['successful_retrievals'],
            'failed_retrievals': self.retrieval_stats['failed_retrievals'],
            'success_rate': success_rate,
            'average_retrieval_time': self.retrieval_stats['average_retrieval_time'],
            'active_connectors': list(self.connectors.keys()),
            'source_weights': self.source_weights
        }