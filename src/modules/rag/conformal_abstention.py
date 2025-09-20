"""
Conformal Abstention System for RAG

Implements conformal prediction-based abstention mechanism to identify
when the system should refrain from providing answers due to uncertainty.
"""

import time
import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import statistics

from .rag_core import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class CalibrationSample:
    """Sample for conformal calibration."""
    query: str
    confidence_score: float
    uncertainty_estimate: float
    true_confidence: Optional[float] = None  # For calibration
    timestamp: float = 0.0


class UncertaintyEstimator:
    """
    Uncertainty estimation for RAG responses using multiple techniques.
    
    Combines:
    - Retrieval confidence analysis
    - Knowledge source agreement
    - Query complexity assessment
    - Historical performance patterns
    """
    
    def __init__(self, history_size: int = 500):
        self.history_size = history_size
        self.query_history = deque(maxlen=history_size)
        self.uncertainty_patterns = {
            'high_uncertainty_queries': [],
            'low_uncertainty_queries': [],
            'uncertainty_trends': deque(maxlen=100)
        }
    
    def estimate_uncertainty(self, query: str, retrieval_result: RetrievalResult) -> Tuple[float, Dict[str, float]]:
        """
        Estimate uncertainty for a query given retrieval results.
        
        Args:
            query: User query
            retrieval_result: Results from knowledge retrieval
            
        Returns:
            Tuple of (uncertainty_score, uncertainty_breakdown)
        """
        uncertainty_components = {}
        
        # 1. Retrieval quality uncertainty
        retrieval_uncertainty = self._assess_retrieval_uncertainty(retrieval_result)
        uncertainty_components['retrieval_uncertainty'] = retrieval_uncertainty
        
        # 2. Query complexity uncertainty
        query_uncertainty = self._assess_query_complexity(query)
        uncertainty_components['query_complexity'] = query_uncertainty
        
        # 3. Knowledge source agreement uncertainty
        source_uncertainty = self._assess_source_agreement(retrieval_result)
        uncertainty_components['source_disagreement'] = source_uncertainty
        
        # 4. Historical pattern uncertainty
        pattern_uncertainty = self._assess_pattern_uncertainty(query, retrieval_result)
        uncertainty_components['pattern_uncertainty'] = pattern_uncertainty
        
        # 5. Content coherence uncertainty
        coherence_uncertainty = self._assess_content_coherence(retrieval_result)
        uncertainty_components['content_coherence'] = coherence_uncertainty
        
        # Weighted combination of uncertainty components
        weights = {
            'retrieval_uncertainty': 0.3,
            'query_complexity': 0.2,
            'source_disagreement': 0.2,
            'pattern_uncertainty': 0.15,
            'content_coherence': 0.15
        }
        
        overall_uncertainty = sum(
            uncertainty_components[component] * weights[component]
            for component in weights
        )
        
        # Store for pattern analysis
        self._update_uncertainty_patterns(query, overall_uncertainty, uncertainty_components)
        
        return min(1.0, overall_uncertainty), uncertainty_components
    
    def _assess_retrieval_uncertainty(self, retrieval_result: RetrievalResult) -> float:
        """Assess uncertainty based on retrieval quality."""
        if not retrieval_result.documents:
            return 1.0  # Maximum uncertainty if no documents
        
        # Base uncertainty on confidence score
        base_uncertainty = 1.0 - retrieval_result.confidence_score
        
        # Adjust based on score variance
        if len(retrieval_result.similarity_scores) > 1:
            score_variance = statistics.variance(retrieval_result.similarity_scores)
            base_uncertainty += min(0.3, score_variance)
        
        # Adjust based on retrieval time (longer time may indicate difficulty)
        if retrieval_result.retrieval_time > 2.0:
            base_uncertainty += min(0.2, (retrieval_result.retrieval_time - 2.0) / 10.0)
        
        return min(1.0, base_uncertainty)
    
    def _assess_query_complexity(self, query: str) -> float:
        """Assess uncertainty based on query complexity."""
        complexity_indicators = {
            'length': len(query.split()),
            'questions': query.count('?'),
            'conditionals': sum(query.lower().count(word) for word in ['if', 'when', 'unless', 'provided']),
            'negations': sum(query.lower().count(word) for word in ['not', 'never', 'no', 'none']),
            'comparisons': sum(query.lower().count(word) for word in ['better', 'worse', 'compare', 'versus']),
            'technical_terms': self._count_technical_terms(query)
        }
        
        # Calculate complexity score
        complexity = 0.0
        
        # Long queries are more complex
        if complexity_indicators['length'] > 20:
            complexity += 0.2
        
        # Multiple questions increase complexity
        complexity += min(0.3, complexity_indicators['questions'] * 0.1)
        
        # Conditionals and negations add uncertainty
        complexity += min(0.2, complexity_indicators['conditionals'] * 0.05)
        complexity += min(0.2, complexity_indicators['negations'] * 0.05)
        
        # Comparisons are inherently uncertain
        complexity += min(0.3, complexity_indicators['comparisons'] * 0.1)
        
        # Technical terms may require specialized knowledge
        complexity += min(0.2, complexity_indicators['technical_terms'] * 0.02)
        
        return min(1.0, complexity)
    
    def _count_technical_terms(self, query: str) -> int:
        """Count technical terms in query."""
        technical_indicators = [
            'algorithm', 'neural', 'network', 'model', 'data', 'analysis',
            'system', 'process', 'method', 'technique', 'approach', 'framework',
            'optimization', 'parameter', 'function', 'variable', 'coefficient'
        ]
        
        query_lower = query.lower()
        return sum(query_lower.count(term) for term in technical_indicators)
    
    def _assess_source_agreement(self, retrieval_result: RetrievalResult) -> float:
        """Assess uncertainty based on agreement between sources."""
        if len(retrieval_result.documents) < 2:
            return 0.3  # Moderate uncertainty with single source
        
        # Group documents by source
        source_groups = {}
        for doc in retrieval_result.documents:
            source = doc.get('source', 'unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        if len(source_groups) == 1:
            return 0.4  # Higher uncertainty if all from same source
        
        # Calculate agreement between sources
        disagreement = 0.0
        
        # Simple heuristic: check for conflicting information
        # In a real implementation, would use more sophisticated similarity measures
        sources = list(source_groups.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source1_docs = source_groups[sources[i]]
                source2_docs = source_groups[sources[j]]
                
                # Compare document similarity scores
                avg_score1 = sum(doc.get('relevance_score', 0) for doc in source1_docs) / len(source1_docs)
                avg_score2 = sum(doc.get('relevance_score', 0) for doc in source2_docs) / len(source2_docs)
                
                score_diff = abs(avg_score1 - avg_score2)
                disagreement += score_diff * 0.5
        
        return min(1.0, disagreement)
    
    def _assess_pattern_uncertainty(self, query: str, retrieval_result: RetrievalResult) -> float:
        """Assess uncertainty based on historical patterns."""
        if len(self.query_history) < 10:
            return 0.3  # Default uncertainty for insufficient history
        
        # Find similar queries in history
        similar_queries = self._find_similar_queries(query)
        
        if not similar_queries:
            return 0.4  # Higher uncertainty for novel query types
        
        # Calculate uncertainty based on historical performance
        historical_uncertainties = [sample.uncertainty_estimate for sample in similar_queries]
        avg_historical_uncertainty = sum(historical_uncertainties) / len(historical_uncertainties)
        
        # Adjust based on retrieval quality compared to historical patterns
        historical_confidences = [sample.confidence_score for sample in similar_queries]
        avg_historical_confidence = sum(historical_confidences) / len(historical_confidences)
        
        confidence_diff = abs(retrieval_result.confidence_score - avg_historical_confidence)
        pattern_uncertainty = avg_historical_uncertainty + (confidence_diff * 0.5)
        
        return min(1.0, pattern_uncertainty)
    
    def _assess_content_coherence(self, retrieval_result: RetrievalResult) -> float:
        """Assess uncertainty based on content coherence across documents."""
        if len(retrieval_result.documents) < 2:
            return 0.2  # Low uncertainty impact for single document
        
        # Simple coherence check based on keyword overlap
        all_keywords = []
        for doc in retrieval_result.documents:
            content = doc.get('content', '')
            # Extract potential keywords (words longer than 4 characters)
            keywords = [word.lower() for word in content.split() if len(word) > 4]
            all_keywords.extend(keywords)
        
        if not all_keywords:
            return 0.5
        
        # Calculate keyword frequency
        keyword_freq = {}
        for keyword in all_keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # Coherence based on shared keywords
        shared_keywords = sum(1 for freq in keyword_freq.values() if freq > 1)
        total_unique_keywords = len(keyword_freq)
        
        if total_unique_keywords == 0:
            return 0.5
        
        coherence_ratio = shared_keywords / total_unique_keywords
        incoherence = 1.0 - coherence_ratio
        
        return min(1.0, incoherence)
    
    def _find_similar_queries(self, query: str) -> List[CalibrationSample]:
        """Find similar queries in history."""
        query_terms = set(query.lower().split())
        similar_queries = []
        
        for sample in self.query_history:
            sample_terms = set(sample.query.lower().split())
            overlap = len(query_terms.intersection(sample_terms))
            similarity = overlap / len(query_terms.union(sample_terms)) if query_terms.union(sample_terms) else 0
            
            if similarity > 0.3:  # Similarity threshold
                similar_queries.append(sample)
        
        return similar_queries[-10:]  # Return recent similar queries
    
    def _update_uncertainty_patterns(self, query: str, uncertainty: float, components: Dict[str, float]):
        """Update uncertainty pattern tracking."""
        sample = CalibrationSample(
            query=query,
            confidence_score=1.0 - uncertainty,
            uncertainty_estimate=uncertainty,
            timestamp=time.time()
        )
        
        self.query_history.append(sample)
        self.uncertainty_patterns['uncertainty_trends'].append(uncertainty)
        
        # Categorize query based on uncertainty
        if uncertainty > 0.7:
            self.uncertainty_patterns['high_uncertainty_queries'].append(query)
        elif uncertainty < 0.3:
            self.uncertainty_patterns['low_uncertainty_queries'].append(query)


class ConformalAbstentionSystem:
    """
    Conformal prediction-based abstention system.
    
    Uses conformal prediction theory to provide statistically valid
    confidence intervals and abstention decisions.
    """
    
    def __init__(self, alpha: float = 0.1, uncertainty_threshold: float = 0.5):
        """
        Initialize conformal abstention system.
        
        Args:
            alpha: Significance level (e.g., 0.1 for 90% confidence)
            uncertainty_threshold: Threshold for abstention decisions
        """
        self.alpha = alpha
        self.uncertainty_threshold = uncertainty_threshold
        
        # Calibration data
        self.calibration_samples = deque(maxlen=1000)
        self.calibrated = False
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator()
        
        # Performance tracking
        self.abstention_stats = {
            'total_queries': 0,
            'abstentions': 0,
            'correct_abstentions': 0,
            'false_abstentions': 0
        }
    
    def should_abstain(self, query: str, confidence_score: float, uncertainty_estimate: float) -> bool:
        """
        Determine whether to abstain from answering based on uncertainty.
        
        Args:
            query: User query
            confidence_score: System confidence in the answer
            uncertainty_estimate: Estimated uncertainty
            
        Returns:
            True if system should abstain from answering
        """
        self.abstention_stats['total_queries'] += 1
        
        # Basic uncertainty threshold check
        if uncertainty_estimate > self.uncertainty_threshold:
            self.abstention_stats['abstentions'] += 1
            return True
        
        # Conformal prediction check (if calibrated)
        if self.calibrated and len(self.calibration_samples) > 10:
            conformal_threshold = self._compute_conformal_threshold()
            if uncertainty_estimate > conformal_threshold:
                self.abstention_stats['abstentions'] += 1
                return True
        
        # Additional safety checks
        if self._safety_abstention_check(query, confidence_score, uncertainty_estimate):
            self.abstention_stats['abstentions'] += 1
            return True
        
        return False
    
    def add_calibration_sample(self, query: str, confidence_score: float, 
                             uncertainty_estimate: float, true_confidence: float):
        """Add a calibration sample for conformal prediction."""
        sample = CalibrationSample(
            query=query,
            confidence_score=confidence_score,
            uncertainty_estimate=uncertainty_estimate,
            true_confidence=true_confidence,
            timestamp=time.time()
        )
        
        self.calibration_samples.append(sample)
        
        # Re-calibrate if we have enough samples
        if len(self.calibration_samples) >= 50:
            self.calibrated = True
    
    def _compute_conformal_threshold(self) -> float:
        """Compute conformal prediction threshold."""
        if not self.calibration_samples:
            return self.uncertainty_threshold
        
        # Calculate nonconformity scores
        nonconformity_scores = []
        for sample in self.calibration_samples:
            if sample.true_confidence is not None:
                # Nonconformity is difference between predicted and true confidence
                nonconformity = abs(sample.confidence_score - sample.true_confidence)
                nonconformity_scores.append(nonconformity)
        
        if not nonconformity_scores:
            return self.uncertainty_threshold
        
        # Compute quantile threshold
        sorted_scores = sorted(nonconformity_scores)
        quantile_index = int((1 - self.alpha) * len(sorted_scores))
        quantile_index = min(quantile_index, len(sorted_scores) - 1)
        
        threshold = sorted_scores[quantile_index]
        
        # Convert to uncertainty threshold
        uncertainty_threshold = min(1.0, self.uncertainty_threshold + threshold)
        
        return uncertainty_threshold
    
    def _safety_abstention_check(self, query: str, confidence_score: float, 
                               uncertainty_estimate: float) -> bool:
        """Additional safety-based abstention checks."""
        safety_flags = []
        
        # Very low confidence
        if confidence_score < 0.3:
            safety_flags.append("very_low_confidence")
        
        # High uncertainty with low confidence combination
        if uncertainty_estimate > 0.6 and confidence_score < 0.5:
            safety_flags.append("high_uncertainty_low_confidence")
        
        # Query contains uncertainty markers
        uncertainty_markers = ['unsure', 'maybe', 'possibly', 'might', 'could be', 'not sure']
        if any(marker in query.lower() for marker in uncertainty_markers):
            safety_flags.append("uncertainty_in_query")
        
        # Requesting harmful or inappropriate content
        harmful_indicators = ['harmful', 'dangerous', 'illegal', 'unethical', 'inappropriate']
        if any(indicator in query.lower() for indicator in harmful_indicators):
            safety_flags.append("potentially_harmful_query")
        
        # Abstain if any safety flags triggered
        return len(safety_flags) > 0
    
    def get_abstention_statistics(self) -> Dict[str, Any]:
        """Get abstention performance statistics."""
        total = self.abstention_stats['total_queries']
        abstention_rate = (
            self.abstention_stats['abstentions'] / total 
            if total > 0 else 0.0
        )
        
        # Calculate precision of abstentions (if we have ground truth)
        correct_abstentions = self.abstention_stats.get('correct_abstentions', 0)
        total_abstentions = self.abstention_stats['abstentions']
        abstention_precision = (
            correct_abstentions / total_abstentions 
            if total_abstentions > 0 else 0.0
        )
        
        return {
            'total_queries': total,
            'total_abstentions': total_abstentions,
            'abstention_rate': abstention_rate,
            'abstention_precision': abstention_precision,
            'calibration_samples': len(self.calibration_samples),
            'is_calibrated': self.calibrated,
            'uncertainty_threshold': self.uncertainty_threshold,
            'alpha_level': self.alpha,
            'confidence_level': 1.0 - self.alpha
        }
    
    def update_feedback(self, query: str, abstained: bool, was_correct: bool):
        """Update system with feedback on abstention decision."""
        if abstained and was_correct:
            self.abstention_stats['correct_abstentions'] += 1
        elif abstained and not was_correct:
            self.abstention_stats['false_abstentions'] += 1