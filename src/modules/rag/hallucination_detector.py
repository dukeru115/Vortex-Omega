"""
Hallucination Detection System for RAG

Advanced hallucination detection with multiple validation techniques:
- Fact-checking against retrieved knowledge
- Consistency verification across sources
- Content plausibility analysis
- Statistical validation
"""

import time
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics

from .rag_core import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class HallucinationMetrics:
    """Metrics for hallucination detection and prevention."""
    query: str
    response: str
    hallucination_score: float
    fact_check_score: float
    consistency_score: float
    plausibility_score: float
    sources_agreement: float
    detection_confidence: float
    flagged_content: List[str]
    timestamp: float


class FactChecker:
    """Fact-checking component for verifying response accuracy."""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.fact_patterns = self._initialize_fact_patterns()
    
    def check_facts(self, response: str, retrieval_result: RetrievalResult) -> Tuple[float, List[str]]:
        """
        Check factual accuracy of response against retrieved knowledge.
        
        Args:
            response: Generated response to check
            retrieval_result: Retrieved knowledge for verification
            
        Returns:
            Tuple of (fact_check_score, flagged_statements)
        """
        if not retrieval_result.documents:
            return 0.5, ["No supporting documents available for fact-checking"]
        
        # Extract factual claims from response
        claims = self._extract_claims(response)
        
        if not claims:
            return 0.8, []  # No specific claims to verify
        
        # Verify each claim against retrieved documents
        verified_claims = 0
        flagged_statements = []
        
        for claim in claims:
            is_supported = self._verify_claim(claim, retrieval_result.documents)
            if is_supported:
                verified_claims += 1
            else:
                flagged_statements.append(f"Unverified claim: {claim}")
        
        # Calculate fact-check score
        fact_score = verified_claims / len(claims) if claims else 0.8
        
        return fact_score, flagged_statements
    
    def _extract_claims(self, response: str) -> List[str]:
        """Extract factual claims from response."""
        # Split response into sentences
        sentences = re.split(r'[.!?]+', response)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Identify factual statements vs opinions/questions
            if self._is_factual_claim(sentence):
                claims.append(sentence)
        
        return claims
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """Determine if sentence contains a factual claim."""
        # Opinion indicators
        opinion_markers = ['I think', 'I believe', 'in my opinion', 'it seems', 'arguably', 'possibly']
        if any(marker in sentence.lower() for marker in opinion_markers):
            return False
        
        # Question indicators
        if '?' in sentence:
            return False
        
        # Factual patterns
        factual_patterns = [
            r'\b\d+\b',  # Contains numbers
            r'\bin \d{4}\b',  # Contains years
            r'\bis\b|\bare\b|\bwas\b|\bwere\b',  # Definitive statements
            r'\baccording to\b|\bstudies show\b|\bresearch indicates\b'  # Citation patterns
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        return False
    
    def _verify_claim(self, claim: str, documents: List[Dict[str, Any]]) -> bool:
        """Verify a claim against retrieved documents."""
        claim_terms = set(claim.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        claim_terms = claim_terms - stop_words
        
        if not claim_terms:
            return True  # No specific terms to verify
        
        # Check support in documents
        total_support = 0.0
        
        for doc in documents:
            content = doc.get('content', '').lower()
            doc_terms = set(content.split())
            
            # Calculate term overlap
            overlap = len(claim_terms.intersection(doc_terms))
            support_ratio = overlap / len(claim_terms)
            
            # Weight by document relevance
            doc_relevance = doc.get('relevance_score', 0.5)
            weighted_support = support_ratio * doc_relevance
            
            total_support += weighted_support
        
        # Normalize by number of documents
        avg_support = total_support / len(documents) if documents else 0.0
        
        return avg_support > 0.5  # Threshold for verification
    
    def _initialize_fact_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for fact detection."""
        return {
            'numerical': [r'\b\d+\.?\d*\s*(percent|%|million|billion|thousand|years?|days?|hours?)\b'],
            'temporal': [r'\bin \d{4}\b', r'\bon \w+ \d{1,2},? \d{4}\b', r'\b\d{1,2}/\d{1,2}/\d{4}\b'],
            'geographical': [r'\bin [A-Z][a-z]+ [A-Z][a-z]+\b', r'\b[A-Z][a-z]+ is located\b'],
            'scientific': [r'\bstudies? show\b', r'\bresearch indicates?\b', r'\baccording to\b']
        }


class ConsistencyChecker:
    """Check consistency of information across sources and within response."""
    
    def __init__(self):
        self.consistency_patterns = self._initialize_consistency_patterns()
    
    def check_consistency(self, response: str, retrieval_result: RetrievalResult) -> Tuple[float, List[str]]:
        """
        Check consistency of response with retrieved knowledge.
        
        Args:
            response: Generated response
            retrieval_result: Retrieved knowledge
            
        Returns:
            Tuple of (consistency_score, inconsistency_flags)
        """
        inconsistencies = []
        
        # Check internal consistency within response
        internal_score, internal_issues = self._check_internal_consistency(response)
        inconsistencies.extend(internal_issues)
        
        # Check consistency with retrieved documents
        external_score, external_issues = self._check_external_consistency(response, retrieval_result)
        inconsistencies.extend(external_issues)
        
        # Check cross-source consistency
        source_score, source_issues = self._check_source_consistency(retrieval_result)
        inconsistencies.extend(source_issues)
        
        # Weighted average of consistency scores
        overall_score = (internal_score * 0.4 + external_score * 0.4 + source_score * 0.2)
        
        return overall_score, inconsistencies
    
    def _check_internal_consistency(self, response: str) -> Tuple[float, List[str]]:
        """Check for internal contradictions in response."""
        sentences = re.split(r'[.!?]+', response)
        contradictions = []
        
        # Simple contradiction detection
        contradiction_pairs = [
            (['always', 'never'], ['sometimes', 'occasionally']),
            (['all', 'every'], ['some', 'few']),
            (['increase', 'rise', 'grow'], ['decrease', 'fall', 'shrink']),
            (['positive', 'beneficial'], ['negative', 'harmful'])
        ]
        
        for i, sentence1 in enumerate(sentences):
            for j, sentence2 in enumerate(sentences[i+1:], i+1):
                if self._detect_contradiction(sentence1, sentence2, contradiction_pairs):
                    contradictions.append(f"Contradiction between statements {i+1} and {j+1}")
        
        # Score based on contradictions found
        consistency_score = max(0.0, 1.0 - (len(contradictions) * 0.3))
        
        return consistency_score, contradictions
    
    def _check_external_consistency(self, response: str, retrieval_result: RetrievalResult) -> Tuple[float, List[str]]:
        """Check consistency with retrieved documents."""
        if not retrieval_result.documents:
            return 0.5, ["No documents available for consistency checking"]
        
        inconsistencies = []
        response_terms = set(response.lower().split())
        
        # Check for contradictory information
        doc_terms_sets = []
        for doc in retrieval_result.documents:
            content = doc.get('content', '')
            doc_terms = set(content.lower().split())
            doc_terms_sets.append(doc_terms)
        
        # Simple consistency check based on term overlap
        total_overlap = 0.0
        for doc_terms in doc_terms_sets:
            overlap = len(response_terms.intersection(doc_terms))
            total_overlap += overlap / len(response_terms.union(doc_terms)) if response_terms.union(doc_terms) else 0
        
        consistency_score = total_overlap / len(doc_terms_sets) if doc_terms_sets else 0.5
        
        if consistency_score < 0.3:
            inconsistencies.append("Low term overlap with source documents")
        
        return min(1.0, consistency_score), inconsistencies
    
    def _check_source_consistency(self, retrieval_result: RetrievalResult) -> Tuple[float, List[str]]:
        """Check consistency across different sources."""
        if len(retrieval_result.documents) < 2:
            return 0.8, []  # Cannot check with single source
        
        inconsistencies = []
        
        # Group documents by source
        source_groups = defaultdict(list)
        for doc in retrieval_result.documents:
            source = doc.get('source', 'unknown')
            source_groups[source].append(doc)
        
        if len(source_groups) < 2:
            return 0.7, ["All documents from same source"]
        
        # Check agreement between sources
        source_agreement = 0.0
        comparisons = 0
        
        sources = list(source_groups.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                agreement = self._calculate_source_agreement(
                    source_groups[sources[i]], 
                    source_groups[sources[j]]
                )
                source_agreement += agreement
                comparisons += 1
        
        if comparisons > 0:
            avg_agreement = source_agreement / comparisons
        else:
            avg_agreement = 0.5
        
        if avg_agreement < 0.4:
            inconsistencies.append("Low agreement between different sources")
        
        return avg_agreement, inconsistencies
    
    def _detect_contradiction(self, sentence1: str, sentence2: str, 
                            contradiction_pairs: List[Tuple[List[str], List[str]]]) -> bool:
        """Detect contradictions between two sentences."""
        s1_lower = sentence1.lower()
        s2_lower = sentence2.lower()
        
        for positive_terms, negative_terms in contradiction_pairs:
            s1_has_positive = any(term in s1_lower for term in positive_terms)
            s1_has_negative = any(term in s1_lower for term in negative_terms)
            s2_has_positive = any(term in s2_lower for term in positive_terms)
            s2_has_negative = any(term in s2_lower for term in negative_terms)
            
            # Check for contradiction patterns
            if (s1_has_positive and s2_has_negative) or (s1_has_negative and s2_has_positive):
                return True
        
        return False
    
    def _calculate_source_agreement(self, docs1: List[Dict], docs2: List[Dict]) -> float:
        """Calculate agreement between two groups of documents."""
        # Extract key terms from each group
        terms1 = set()
        terms2 = set()
        
        for doc in docs1:
            content = doc.get('content', '')
            terms1.update(content.lower().split())
        
        for doc in docs2:
            content = doc.get('content', '')
            terms2.update(content.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(terms1.intersection(terms2))
        union = len(terms1.union(terms2))
        
        return intersection / union if union > 0 else 0.0
    
    def _initialize_consistency_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for consistency checking."""
        return {
            'contradiction_indicators': ['however', 'but', 'although', 'despite', 'on the other hand'],
            'agreement_indicators': ['similarly', 'likewise', 'in addition', 'furthermore', 'moreover'],
            'uncertainty_indicators': ['might', 'could', 'possibly', 'perhaps', 'maybe']
        }


class PlausibilityAnalyzer:
    """Analyze plausibility of generated content."""
    
    def __init__(self):
        self.plausibility_rules = self._initialize_plausibility_rules()
    
    def analyze_plausibility(self, response: str, query: str) -> Tuple[float, List[str]]:
        """
        Analyze plausibility of response content.
        
        Args:
            response: Generated response
            query: Original query
            
        Returns:
            Tuple of (plausibility_score, implausible_elements)
        """
        implausible_elements = []
        
        # Check for implausible claims
        implausible_elements.extend(self._check_numerical_plausibility(response))
        implausible_elements.extend(self._check_temporal_plausibility(response))
        implausible_elements.extend(self._check_logical_plausibility(response))
        implausible_elements.extend(self._check_context_plausibility(response, query))
        
        # Calculate plausibility score
        plausibility_score = max(0.0, 1.0 - (len(implausible_elements) * 0.2))
        
        return plausibility_score, implausible_elements
    
    def _check_numerical_plausibility(self, response: str) -> List[str]:
        """Check for implausible numerical claims."""
        issues = []
        
        # Extract numbers with units
        number_patterns = [
            (r'(\d+(?:\.\d+)?)\s*%', 'percentage', 0, 100),
            (r'(\d+(?:\.\d+)?)\s*(?:million|billion)', 'large_number', 0, 1000000),
            (r'(\d{4})', 'year', 1800, 2030),
            (r'(\d+(?:\.\d+)?)\s*(?:kg|pounds?)', 'weight', 0, 1000)
        ]
        
        for pattern, category, min_val, max_val in number_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match)
                    if not (min_val <= value <= max_val):
                        issues.append(f"Implausible {category}: {value}")
                except ValueError:
                    continue
        
        return issues
    
    def _check_temporal_plausibility(self, response: str) -> List[str]:
        """Check for implausible temporal claims."""
        issues = []
        
        # Check for impossible dates or time periods
        year_pattern = r'\b((?:19|20)\d{2})\b'
        years = re.findall(year_pattern, response)
        
        current_year = 2024  # In real implementation, would use datetime.now().year
        for year in years:
            year_int = int(year)
            if year_int > current_year + 1:  # Allow for near future
                issues.append(f"Future year mentioned: {year}")
            elif year_int < 1800:  # Very old dates might be suspicious
                issues.append(f"Very old year mentioned: {year}")
        
        return issues
    
    def _check_logical_plausibility(self, response: str) -> List[str]:
        """Check for logical inconsistencies."""
        issues = []
        
        # Check for impossible claims
        impossible_patterns = [
            r'\b(?:100%|completely)\s+(?:free|certain|impossible)\b',
            r'\balways\s+(?:never|impossible)\b',
            r'\bnever\s+(?:always|certain)\b'
        ]
        
        for pattern in impossible_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append("Logically inconsistent absolute statement detected")
        
        return issues
    
    def _check_context_plausibility(self, response: str, query: str) -> List[str]:
        """Check if response is contextually appropriate for query."""
        issues = []
        
        # Extract key terms from query and response
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        
        # Check for topic drift
        overlap = len(query_terms.intersection(response_terms))
        total_terms = len(query_terms.union(response_terms))
        
        if total_terms > 0:
            relevance = overlap / total_terms
            if relevance < 0.1:
                issues.append("Response topic significantly differs from query")
        
        return issues
    
    def _initialize_plausibility_rules(self) -> Dict[str, Any]:
        """Initialize plausibility checking rules."""
        return {
            'numerical_bounds': {
                'percentage': (0, 100),
                'temperature_celsius': (-273, 10000),
                'human_age': (0, 150),
                'year': (1800, 2030)
            },
            'impossible_combinations': [
                ['always', 'never'],
                ['everyone', 'nobody'],
                ['everywhere', 'nowhere']
            ]
        }


class HallucinationDetector:
    """
    Main hallucination detection system coordinating all detection components.
    """
    
    def __init__(self, threshold: float = 0.3, fact_checking_enabled: bool = True):
        self.threshold = threshold
        self.fact_checking_enabled = fact_checking_enabled
        
        # Initialize components
        self.fact_checker = FactChecker()
        self.consistency_checker = ConsistencyChecker()
        self.plausibility_analyzer = PlausibilityAnalyzer()
        
        # Tracking
        self.detection_history = deque(maxlen=1000)
        self.detection_stats = {
            'total_checks': 0,
            'hallucinations_detected': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
    
    def detect_hallucinations(self, query: str, response: str, 
                            retrieval_result: RetrievalResult) -> float:
        """
        Detect hallucinations in generated response.
        
        Args:
            query: Original user query
            response: Generated response to check
            retrieval_result: Retrieved knowledge for verification
            
        Returns:
            Hallucination score (0.0 = no hallucination, 1.0 = definite hallucination)
        """
        self.detection_stats['total_checks'] += 1
        
        # Component scores
        fact_score = 1.0
        fact_flags = []
        consistency_score = 1.0
        consistency_flags = []
        plausibility_score = 1.0
        plausibility_flags = []
        
        # Fact checking
        if self.fact_checking_enabled and retrieval_result.documents:
            fact_score, fact_flags = self.fact_checker.check_facts(response, retrieval_result)
        
        # Consistency checking
        consistency_score, consistency_flags = self.consistency_checker.check_consistency(
            response, retrieval_result
        )
        
        # Plausibility analysis
        plausibility_score, plausibility_flags = self.plausibility_analyzer.analyze_plausibility(
            response, query
        )
        
        # Combine scores (lower scores indicate higher hallucination risk)
        weights = {
            'fact_score': 0.4,
            'consistency_score': 0.35,
            'plausibility_score': 0.25
        }
        
        # Convert to hallucination scores (invert since lower fact/consistency scores = higher hallucination)
        hallucination_components = {
            'fact_hallucination': 1.0 - fact_score,
            'consistency_hallucination': 1.0 - consistency_score,
            'plausibility_hallucination': 1.0 - plausibility_score
        }
        
        overall_hallucination_score = sum(
            hallucination_components[component] * weights[list(weights.keys())[i]]
            for i, component in enumerate(hallucination_components)
        )
        
        # Apply additional penalties for specific issues
        all_flags = fact_flags + consistency_flags + plausibility_flags
        if len(all_flags) > 5:  # Many issues detected
            overall_hallucination_score += 0.2
        
        overall_hallucination_score = min(1.0, overall_hallucination_score)
        
        # Track detection
        metrics = HallucinationMetrics(
            query=query,
            response=response,
            hallucination_score=overall_hallucination_score,
            fact_check_score=fact_score,
            consistency_score=consistency_score,
            plausibility_score=plausibility_score,
            sources_agreement=self._calculate_sources_agreement(retrieval_result),
            detection_confidence=self._calculate_detection_confidence(overall_hallucination_score),
            flagged_content=all_flags,
            timestamp=time.time()
        )
        
        self.detection_history.append(metrics)
        
        # Update statistics
        if overall_hallucination_score > self.threshold:
            self.detection_stats['hallucinations_detected'] += 1
        
        return overall_hallucination_score
    
    def _calculate_sources_agreement(self, retrieval_result: RetrievalResult) -> float:
        """Calculate agreement between sources."""
        if len(retrieval_result.documents) < 2:
            return 1.0
        
        # Simple agreement based on score variance
        scores = retrieval_result.similarity_scores
        if len(scores) < 2:
            return 1.0
        
        score_variance = statistics.variance(scores)
        agreement = max(0.0, 1.0 - score_variance)
        
        return agreement
    
    def _calculate_detection_confidence(self, hallucination_score: float) -> float:
        """Calculate confidence in hallucination detection."""
        # Higher confidence for extreme scores
        if hallucination_score > 0.8 or hallucination_score < 0.2:
            return 0.9
        elif hallucination_score > 0.6 or hallucination_score < 0.4:
            return 0.7
        else:
            return 0.5  # Lower confidence for middle range
    
    def get_detection_report(self) -> Dict[str, Any]:
        """Get comprehensive hallucination detection report."""
        total_checks = self.detection_stats['total_checks']
        detection_rate = (
            self.detection_stats['hallucinations_detected'] / total_checks 
            if total_checks > 0 else 0.0
        )
        
        # Analyze recent trends
        recent_scores = [m.hallucination_score for m in list(self.detection_history)[-50:]]
        avg_recent_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
        
        return {
            'detection_statistics': self.detection_stats.copy(),
            'detection_rate': detection_rate,
            'average_hallucination_score': avg_recent_score,
            'threshold': self.threshold,
            'recent_trends': {
                'improving': avg_recent_score < 0.3,
                'stable': 0.3 <= avg_recent_score <= 0.5,
                'concerning': avg_recent_score > 0.5
            },
            'component_performance': {
                'fact_checking_enabled': self.fact_checking_enabled,
                'consistency_checking': True,
                'plausibility_analysis': True
            }
        }
    
    def update_feedback(self, query: str, response: str, was_hallucination: bool):
        """Update detector with feedback on detection accuracy."""
        # Find corresponding detection in history
        for metrics in reversed(self.detection_history):
            if metrics.query == query and metrics.response == response:
                detected_hallucination = metrics.hallucination_score > self.threshold
                
                if was_hallucination and not detected_hallucination:
                    self.detection_stats['false_negatives'] += 1
                elif not was_hallucination and detected_hallucination:
                    self.detection_stats['false_positives'] += 1
                
                break