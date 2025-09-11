"""
Adaptive Vocabulary System for ESC Module 2.1

Advanced vocabulary adaptation with constitutional constraints:
- Dynamic vocabulary expansion with safety validation
- Constitutional token evaluation and scoring
- Frequency-based learning with risk assessment
- Emergency vocabulary lockdown and safe-word enforcement
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import logging
import re
import hashlib
import time

logger = logging.getLogger(__name__)


class VocabularyMode(Enum):
    """Vocabulary adaptation modes."""
    STATIC = "static"          # No vocabulary expansion
    CONSERVATIVE = "conservative"  # Very careful expansion
    BALANCED = "balanced"      # Standard expansion
    EXPLORATORY = "exploratory"  # Aggressive expansion
    EMERGENCY = "emergency"    # Emergency lockdown


class TokenStatus(Enum):
    """Token status in adaptive vocabulary."""
    APPROVED = "approved"
    PENDING = "pending"
    REJECTED = "rejected"
    QUARANTINE = "quarantine"
    CONSTITUTIONAL = "constitutional"


@dataclass
class VocabularyConfig:
    """Configuration for adaptive vocabulary system."""
    mode: VocabularyMode = VocabularyMode.BALANCED
    base_vocabulary_size: int = 50000
    max_expansion_size: int = 10000
    min_frequency_threshold: int = 5
    constitutional_score_threshold: float = 0.6
    risk_score_threshold: float = 0.7
    adaptation_rate: float = 0.01
    quarantine_period: int = 100  # Number of encounters before re-evaluation


@dataclass
class TokenEntry:
    """Entry for a token in adaptive vocabulary."""
    token: str
    token_id: int
    frequency: int = 0
    constitutional_score: float = 0.0
    risk_score: float = 0.0
    status: TokenStatus = TokenStatus.PENDING
    first_seen: float = 0.0
    last_seen: float = 0.0
    contexts: List[str] = field(default_factory=list)
    quarantine_count: int = 0


class AdaptiveVocabulary:
    """
    Adaptive Vocabulary System for Constitutional ESC.
    
    Manages dynamic vocabulary expansion with constitutional safety constraints,
    learning from usage patterns while maintaining safety and compliance.
    """
    
    def __init__(self, config: VocabularyConfig):
        """
        Initialize adaptive vocabulary system.
        
        Args:
            config: Vocabulary configuration
        """
        self.config = config
        self.current_mode = config.mode
        
        # Core vocabulary storage
        self.base_vocabulary: Dict[str, TokenEntry] = {}
        self.expanded_vocabulary: Dict[str, TokenEntry] = {}
        self.rejected_tokens: Dict[str, TokenEntry] = {}
        self.quarantine_tokens: Dict[str, TokenEntry] = {}
        
        # Token tracking
        self.token_id_counter = config.base_vocabulary_size
        self.token_usage_history = defaultdict(list)
        self.context_associations = defaultdict(set)
        
        # Constitutional monitoring
        self.constitutional_violations = []
        self.safety_interventions = []
        self.adaptation_history = []
        
        # Statistics
        self.stats = {
            'tokens_evaluated': 0,
            'tokens_approved': 0,
            'tokens_rejected': 0,
            'constitutional_blocks': 0,
            'emergency_lockdowns': 0,
            'vocabulary_resets': 0
        }
        
        # Initialize constitutional patterns
        self._initialize_constitutional_patterns()
        self._initialize_safety_filters()
        
        logger.info(f"Adaptive vocabulary system initialized")
        logger.info(f"Mode: {config.mode.value}, Base size: {config.base_vocabulary_size}")
    
    def _initialize_constitutional_patterns(self):
        """Initialize constitutional evaluation patterns."""
        # Constitutional positive patterns
        self.constitutional_positive = {
            'safety': [r'\\b(safe|secure|protect|privacy|ethical)\\w*'],
            'education': [r'\\b(learn|teach|educate|inform|knowledge)\\w*'],
            'help': [r'\\b(help|assist|support|guide|beneficial)\\w*'],
            'truth': [r'\\b(true|honest|accurate|factual|reliable)\\w*'],
            'respect': [r'\\b(respect|dignity|fair|equal|inclusive)\\w*']
        }
        
        # Constitutional negative patterns
        self.constitutional_negative = {
            'violence': [r'\\b(violence|harm|attack|destroy|kill)\\w*'],
            'hate': [r'\\b(hate|racist|discriminat|bigot|supremac)\\w*'],
            'deception': [r'\\b(lie|deceive|fraud|fake|manipulat)\\w*'],
            'illegal': [r'\\b(illegal|criminal|unlawful|forbidden)\\w*'],
            'harmful': [r'\\b(toxic|poison|dangerous|lethal|deadly)\\w*']
        }
        
        # Safe core vocabulary (always allowed)
        self.safe_core_words = {
            'basic': ['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were'],
            'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her'],
            'numbers': ['one', 'two', 'three', 'four', 'five', 'zero', 'first', 'second'],
            'actions': ['go', 'come', 'see', 'look', 'think', 'know', 'say', 'tell'],
            'positive': ['good', 'nice', 'kind', 'helpful', 'safe', 'happy', 'thank']
        }
    
    def _initialize_safety_filters(self):
        """Initialize safety filtering systems."""
        # Character-based safety checks
        self.unsafe_char_patterns = [
            r'[\\x00-\\x1f]',  # Control characters
            r'[^\w\s\-\'".,!?;:()\[\]{}]',  # Unusual characters
            r'(.)\\1{5,}',  # Excessive repetition
        ]
        
        # Length and format constraints
        self.format_constraints = {
            'min_length': 1,
            'max_length': 50,
            'max_digits': 0.5,  # Max 50% digits
            'max_uppercase': 0.8,  # Max 80% uppercase
            'no_excessive_punctuation': True
        }
        
        # Contextual risk factors
        self.risk_factors = {
            'excessive_length': 0.3,
            'unusual_characters': 0.5,
            'excessive_repetition': 0.4,
            'all_caps': 0.2,
            'mixed_scripts': 0.3,
            'potential_encoding': 0.6
        }
    
    def evaluate_token(self, 
                      token: str, 
                      context: Optional[str] = None,
                      force_evaluation: bool = False) -> Tuple[TokenEntry, Dict[str, Any]]:
        """Evaluate a token for inclusion in adaptive vocabulary.
        
        Args:
            token: Token to evaluate
            context: Optional context where token appeared
            force_evaluation: Force re-evaluation even if already processed
            
        Returns:
            Tuple of (token_entry, evaluation_info)
        """
        self.stats['tokens_evaluated'] += 1
        current_time = time.time()
        
        # Check if token already exists
        existing_entry = self._find_existing_token(token)
        if existing_entry and not force_evaluation:
            # Update existing entry
            existing_entry.frequency += 1
            existing_entry.last_seen = current_time
            if context:
                existing_entry.contexts.append(context[:100])  # Limit context length
                if len(existing_entry.contexts) > 20:
                    existing_entry.contexts = existing_entry.contexts[-20:]  # Keep recent contexts
            
            evaluation_info = {
                'is_new_token': False,
                'existing_status': existing_entry.status.value,
                'frequency_updated': existing_entry.frequency
            }
            return existing_entry, evaluation_info
        
        # Create new token entry
        token_entry = TokenEntry(
            token=token,
            token_id=self._generate_token_id(token),
            frequency=1,
            first_seen=current_time,
            last_seen=current_time,
            contexts=[context[:100]] if context else []
        )
        
        # Evaluate constitutional compliance
        constitutional_analysis = self._evaluate_constitutional_compliance(token, context)
        token_entry.constitutional_score = constitutional_analysis['score']
        
        # Evaluate safety risks
        safety_analysis = self._evaluate_safety_risks(token, context)
        token_entry.risk_score = safety_analysis['risk_score']
        
        # Determine token status based on current mode
        status_decision = self._determine_token_status(
            token_entry, constitutional_analysis, safety_analysis
        )
        token_entry.status = status_decision['status']
        
        # Store token in appropriate collection
        self._store_token(token_entry)
        
        # Update statistics
        if token_entry.status == TokenStatus.APPROVED:
            self.stats['tokens_approved'] += 1
        elif token_entry.status == TokenStatus.REJECTED:
            self.stats['tokens_rejected'] += 1
        elif token_entry.status == TokenStatus.QUARANTINE:
            self.stats['constitutional_blocks'] += 1
        
        evaluation_info = {
            'is_new_token': True,
            'constitutional_analysis': constitutional_analysis,
            'safety_analysis': safety_analysis,
            'status_decision': status_decision,
            'final_status': token_entry.status.value
        }
        
        # Log significant events
        if token_entry.status == TokenStatus.REJECTED:
            logger.warning(f"Token rejected: '{token}' (constitutional: {token_entry.constitutional_score:.3f}, risk: {token_entry.risk_score:.3f})")
        
        return token_entry, evaluation_info
    
    def _find_existing_token(self, token: str) -> Optional[TokenEntry]:
        """Find existing token entry across all vocabularies.
        
        Args:
            token: Token to find
            
        Returns:
            Existing token entry or None
        """
        # Check in order of priority
        for vocab in [self.base_vocabulary, self.expanded_vocabulary, 
                     self.quarantine_tokens, self.rejected_tokens]:
            if token in vocab:
                return vocab[token]
        return None
    
    def _evaluate_constitutional_compliance(self, 
                                          token: str, 
                                          context: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate constitutional compliance of a token.
        
        Args:
            token: Token to evaluate
            context: Optional context
            
        Returns:
            Constitutional analysis results
        """
        token_lower = token.lower()
        
        # Check if it's a safe core word
        is_safe_core = any(token_lower in words for words in self.safe_core_words.values())
        if is_safe_core:
            return {
                'score': 1.0,
                'is_safe_core': True,
                'positive_matches': [],
                'negative_matches': [],
                'context_boost': 0.0
            }
        
        # Evaluate positive constitutional patterns
        positive_score = 0.0
        positive_matches = []
        
        for category, patterns in self.constitutional_positive.items():
            for pattern in patterns:
                if re.search(pattern, token_lower):
                    positive_score += 0.2
                    positive_matches.append(f"{category}:{pattern}")
        
        # Evaluate negative constitutional patterns
        negative_score = 0.0
        negative_matches = []
        
        for category, patterns in self.constitutional_negative.items():
            for pattern in patterns:
                if re.search(pattern, token_lower):
                    negative_score += 0.4  # Higher penalty for negative patterns
                    negative_matches.append(f"{category}:{pattern}")
        
        # Context-based adjustments
        context_boost = 0.0
        if context:
            context_lower = context.lower()
            # Educational context gets boost
            if any(word in context_lower for word in ['learn', 'teach', 'explain', 'understand']):
                context_boost += 0.1
            # Safety context gets boost
            if any(word in context_lower for word in ['safe', 'secure', 'protect', 'ethical']):
                context_boost += 0.15
        
        # Final constitutional score
        base_score = 0.5  # Neutral starting point
        final_score = base_score + positive_score - negative_score + context_boost
        final_score = max(0.0, min(1.0, final_score))
        
        return {
            'score': final_score,
            'is_safe_core': False,
            'positive_matches': positive_matches,
            'negative_matches': negative_matches,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'context_boost': context_boost
        }
    
    def _evaluate_safety_risks(self, 
                              token: str, 
                              context: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate safety risks of a token.
        
        Args:
            token: Token to evaluate
            context: Optional context
            
        Returns:
            Safety risk analysis
        """
        risk_factors = {}
        total_risk = 0.0
        
        # Character-based safety checks
        for i, pattern in enumerate(self.unsafe_char_patterns):
            if re.search(pattern, token):
                risk_name = f'unsafe_chars_{i}'
                risk_factors[risk_name] = 0.5
                total_risk += 0.5
        
        # Length constraints
        if len(token) < self.format_constraints['min_length']:
            risk_factors['too_short'] = 0.3
            total_risk += 0.3
        elif len(token) > self.format_constraints['max_length']:
            risk_factors['too_long'] = 0.4
            total_risk += 0.4
        
        # Digit ratio
        digit_ratio = sum(1 for c in token if c.isdigit()) / max(1, len(token))
        if digit_ratio > self.format_constraints['max_digits']:
            risk_factors['excessive_digits'] = digit_ratio * 0.3
            total_risk += digit_ratio * 0.3
        
        # Uppercase ratio
        upper_ratio = sum(1 for c in token if c.isupper()) / max(1, len(token))
        if upper_ratio > self.format_constraints['max_uppercase']:
            risk_factors['excessive_uppercase'] = upper_ratio * 0.2
            total_risk += upper_ratio * 0.2
        
        # Repetition patterns
        if re.search(r'(.)\\1{3,}', token):  # Same character 4+ times
            risk_factors['excessive_repetition'] = 0.4
            total_risk += 0.4
        
        # Mixed scripts (potential encoding attack)
        has_latin = any(ord(c) < 256 for c in token)
        has_unicode = any(ord(c) >= 256 for c in token)
        if has_latin and has_unicode:
            risk_factors['mixed_scripts'] = 0.3
            total_risk += 0.3
        
        # Final risk score
        risk_score = min(1.0, total_risk)
        
        return {
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'format_violations': len(risk_factors),
            'safety_compliant': risk_score < self.config.risk_score_threshold
        }
    
    def _determine_token_status(self, 
                               token_entry: TokenEntry,
                               constitutional_analysis: Dict[str, Any],
                               safety_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the status of a token based on analyses and current mode.
        
        Args:
            token_entry: Token entry being evaluated
            constitutional_analysis: Constitutional analysis results
            safety_analysis: Safety analysis results
            
        Returns:
            Status decision information
        """
        decision_factors = []
        
        # Emergency mode - reject almost everything
        if self.current_mode == VocabularyMode.EMERGENCY:
            if constitutional_analysis['is_safe_core']:
                status = TokenStatus.APPROVED
                decision_factors.append("Safe core word in emergency mode")
            else:
                status = TokenStatus.REJECTED
                decision_factors.append("Emergency mode - non-core rejected")
        
        # Static mode - no new tokens
        elif self.current_mode == VocabularyMode.STATIC:
            status = TokenStatus.REJECTED
            decision_factors.append("Static mode - no expansion allowed")
        
        # Normal evaluation modes
        else:
            # Check safety first
            if not safety_analysis['safety_compliant']:
                status = TokenStatus.REJECTED
                decision_factors.append(f"Safety risk too high: {token_entry.risk_score:.3f}")
            
            # Check constitutional compliance
            elif token_entry.constitutional_score < self.config.constitutional_score_threshold:
                # Mode-specific handling
                if self.current_mode == VocabularyMode.CONSERVATIVE:
                    status = TokenStatus.REJECTED
                    decision_factors.append(f"Conservative mode - constitutional score too low: {token_entry.constitutional_score:.3f}")
                else:
                    status = TokenStatus.QUARANTINE
                    decision_factors.append(f"Quarantined for constitutional review: {token_entry.constitutional_score:.3f}")
            
            # Constitutional violations
            elif constitutional_analysis['negative_matches']:
                status = TokenStatus.QUARANTINE
                decision_factors.append(f"Constitutional negative patterns: {constitutional_analysis['negative_matches']}")
            
            # Approved cases
            else:
                # Mode-specific approval thresholds
                if self.current_mode == VocabularyMode.CONSERVATIVE:
                    if token_entry.constitutional_score > 0.8:
                        status = TokenStatus.APPROVED
                        decision_factors.append("Conservative approval - high constitutional score")
                    else:
                        status = TokenStatus.PENDING
                        decision_factors.append("Conservative mode - pending further evaluation")
                
                elif self.current_mode == VocabularyMode.EXPLORATORY:
                    if token_entry.constitutional_score > 0.3:
                        status = TokenStatus.APPROVED
                        decision_factors.append("Exploratory approval - moderate constitutional score")
                    else:
                        status = TokenStatus.QUARANTINE
                        decision_factors.append("Exploratory quarantine - low constitutional score")
                
                else:  # Balanced mode
                    if token_entry.constitutional_score >= self.config.constitutional_score_threshold:
                        status = TokenStatus.APPROVED
                        decision_factors.append("Balanced approval - meets constitutional threshold")
                    else:
                        status = TokenStatus.PENDING
                        decision_factors.append("Balanced mode - pending threshold evaluation")
        
        return {
            'status': status,
            'decision_factors': decision_factors,
            'mode_applied': self.current_mode.value
        }
    
    def _store_token(self, token_entry: TokenEntry):
        """Store token in appropriate vocabulary collection.
        
        Args:
            token_entry: Token entry to store
        """
        token = token_entry.token
        
        # Remove from other collections if it exists
        for vocab in [self.base_vocabulary, self.expanded_vocabulary, 
                     self.quarantine_tokens, self.rejected_tokens]:
            if token in vocab:
                del vocab[token]
        
        # Store in appropriate collection
        if token_entry.status == TokenStatus.APPROVED:
            if len(self.expanded_vocabulary) < self.config.max_expansion_size:
                self.expanded_vocabulary[token] = token_entry
            else:
                # Vocabulary full - need to make room or reject
                self._handle_vocabulary_overflow(token_entry)
        
        elif token_entry.status == TokenStatus.CONSTITUTIONAL:
            self.base_vocabulary[token] = token_entry
        
        elif token_entry.status == TokenStatus.PENDING:
            self.expanded_vocabulary[token] = token_entry
        
        elif token_entry.status == TokenStatus.QUARANTINE:
            self.quarantine_tokens[token] = token_entry
        
        elif token_entry.status == TokenStatus.REJECTED:
            self.rejected_tokens[token] = token_entry
    
    def _handle_vocabulary_overflow(self, new_token_entry: TokenEntry):
        """Handle vocabulary size overflow.
        
        Args:
            new_token_entry: New token that would exceed vocabulary size
        """
        # Find least valuable token to replace
        if self.expanded_vocabulary:
            # Score existing tokens (frequency * constitutional_score)
            token_scores = []
            for token, entry in self.expanded_vocabulary.items():
                score = entry.frequency * entry.constitutional_score
                token_scores.append((score, token, entry))
            
            # Sort by score (lowest first)
            token_scores.sort(key=lambda x: x[0])
            
            # Compare new token with lowest scored token
            new_score = new_token_entry.frequency * new_token_entry.constitutional_score
            
            if token_scores and new_score > token_scores[0][0]:
                # Replace lowest scored token
                _, old_token, old_entry = token_scores[0]
                
                # Move old token to rejected
                old_entry.status = TokenStatus.REJECTED
                self.rejected_tokens[old_token] = old_entry
                del self.expanded_vocabulary[old_token]
                
                # Add new token
                self.expanded_vocabulary[new_token_entry.token] = new_token_entry
                
                logger.info(f"Vocabulary overflow: replaced '{old_token}' (score {token_scores[0][0]:.3f}) with '{new_token_entry.token}' (score {new_score:.3f})")
            else:
                # New token not good enough
                new_token_entry.status = TokenStatus.REJECTED
                self.rejected_tokens[new_token_entry.token] = new_token_entry
        else:
            # No existing tokens to replace
            new_token_entry.status = TokenStatus.REJECTED
            self.rejected_tokens[new_token_entry.token] = new_token_entry
    
    def _generate_token_id(self, token: str) -> int:
        """Generate unique token ID.
        
        Args:
            token: Token to generate ID for
            
        Returns:
            Unique token ID
        """
        # Use hash-based ID generation with collision handling
        token_hash = int(hashlib.md5(token.encode('utf-8')).hexdigest()[:8], 16)
        token_id = self.config.base_vocabulary_size + (token_hash % self.config.max_expansion_size)
        
        # Handle collisions
        while any(entry.token_id == token_id for vocab in [self.base_vocabulary, self.expanded_vocabulary, 
                                                           self.quarantine_tokens, self.rejected_tokens]
                 for entry in vocab.values()):
            token_id = (token_id + 1) % (self.config.base_vocabulary_size + self.config.max_expansion_size)
            if token_id < self.config.base_vocabulary_size:
                token_id = self.config.base_vocabulary_size  # Ensure in expansion range
        
        return token_id
    
    def promote_quarantine_tokens(self) -> Dict[str, Any]:
        """Review and potentially promote tokens from quarantine.
        
        Returns:
            Promotion results
        """
        promoted = []
        still_quarantined = []
        rejected = []
        
        for token, entry in list(self.quarantine_tokens.items()):
            entry.quarantine_count += 1
            
            # Re-evaluate after quarantine period
            if entry.quarantine_count >= self.config.quarantine_period:
                # Re-evaluate constitutional score based on usage patterns
                if entry.frequency >= self.config.min_frequency_threshold:
                    # Promote if it has gained sufficient usage
                    entry.status = TokenStatus.APPROVED
                    self.expanded_vocabulary[token] = entry
                    del self.quarantine_tokens[token]
                    promoted.append(token)
                    logger.info(f"Quarantine token promoted: '{token}' (freq={entry.frequency})")
                else:
                    # Reject if still low usage
                    entry.status = TokenStatus.REJECTED
                    self.rejected_tokens[token] = entry
                    del self.quarantine_tokens[token]
                    rejected.append(token)
            else:
                still_quarantined.append(token)
        
        return {
            'promoted': promoted,
            'rejected': rejected,
            'still_quarantined': still_quarantined,
            'total_processed': len(promoted) + len(rejected)
        }
    
    def set_vocabulary_mode(self, mode: VocabularyMode):
        """Set the vocabulary adaptation mode.
        
        Args:
            mode: New vocabulary mode
        """
        old_mode = self.current_mode
        self.current_mode = mode
        
        # Handle mode-specific actions
        if mode == VocabularyMode.EMERGENCY:
            self.stats['emergency_lockdowns'] += 1
            logger.warning(f"EMERGENCY: Vocabulary locked down - only safe core words allowed")
        
        logger.info(f"Vocabulary mode changed: {old_mode.value} -> {mode.value}")
    
    def reset_vocabulary_expansion(self, preserve_approved: bool = True):
        """Reset vocabulary expansion state.
        
        Args:
            preserve_approved: Whether to preserve approved expanded tokens
        """
        if preserve_approved:
            # Keep approved tokens, clear others
            approved_tokens = {k: v for k, v in self.expanded_vocabulary.items() 
                             if v.status == TokenStatus.APPROVED}
            self.expanded_vocabulary = approved_tokens
        else:
            # Clear all expanded vocabulary
            self.expanded_vocabulary.clear()
        
        # Clear quarantine and rejected
        self.quarantine_tokens.clear()
        self.rejected_tokens.clear()
        
        # Reset statistics
        self.stats['vocabulary_resets'] += 1
        
        logger.info(f"Vocabulary expansion reset (preserve_approved={preserve_approved})")
    
    def get_vocabulary_status(self) -> Dict[str, Any]:
        """Get comprehensive vocabulary status.
        
        Returns:
            Vocabulary status information
        """
        return {
            'configuration': {
                'mode': self.current_mode.value,
                'base_size': self.config.base_vocabulary_size,
                'max_expansion': self.config.max_expansion_size,
                'constitutional_threshold': self.config.constitutional_score_threshold
            },
            'current_state': {
                'base_vocabulary_size': len(self.base_vocabulary),
                'expanded_vocabulary_size': len(self.expanded_vocabulary),
                'quarantine_size': len(self.quarantine_tokens),
                'rejected_size': len(self.rejected_tokens),
                'expansion_utilization': len(self.expanded_vocabulary) / self.config.max_expansion_size
            },
            'statistics': self.stats.copy(),
            'recent_activity': {
                'tokens_in_quarantine': list(self.quarantine_tokens.keys())[:10],  # First 10
                'recently_approved': [token for token, entry in self.expanded_vocabulary.items() 
                                    if entry.status == TokenStatus.APPROVED][:10],
                'recently_rejected': list(self.rejected_tokens.keys())[-10:]  # Last 10
            }
        }
    
    def generate_vocabulary_report(self) -> Dict[str, Any]:
        """Generate comprehensive vocabulary adaptation report.
        
        Returns:
            Detailed vocabulary analysis report
        """
        # Analyze approval rates by constitutional score ranges
        approved_tokens = [entry for entry in self.expanded_vocabulary.values() 
                          if entry.status == TokenStatus.APPROVED]
        
        if approved_tokens:
            constitutional_scores = [entry.constitutional_score for entry in approved_tokens]
            avg_constitutional_score = np.mean(constitutional_scores)
            
            frequency_distribution = Counter([entry.frequency for entry in approved_tokens])
        else:
            avg_constitutional_score = 0.0
            frequency_distribution = Counter()
        
        # Analyze rejection reasons
        rejection_analysis = {}
        for token, entry in self.rejected_tokens.items():
            if entry.constitutional_score < self.config.constitutional_score_threshold:
                rejection_analysis['low_constitutional'] = rejection_analysis.get('low_constitutional', 0) + 1
            if entry.risk_score > self.config.risk_score_threshold:
                rejection_analysis['high_risk'] = rejection_analysis.get('high_risk', 0) + 1
        
        return {
            'status': 'active',
            'mode': self.current_mode.value,
            'vocabulary_metrics': {
                'total_tokens_managed': len(self.base_vocabulary) + len(self.expanded_vocabulary) + 
                                      len(self.quarantine_tokens) + len(self.rejected_tokens),
                'expansion_efficiency': len(self.expanded_vocabulary) / max(1, self.stats['tokens_evaluated']),
                'approval_rate': self.stats['tokens_approved'] / max(1, self.stats['tokens_evaluated']),
                'rejection_rate': self.stats['tokens_rejected'] / max(1, self.stats['tokens_evaluated'])
            },
            'quality_analysis': {
                'avg_constitutional_score': avg_constitutional_score,
                'frequency_distribution': dict(frequency_distribution.most_common(10)),
                'rejection_reasons': rejection_analysis
            },
            'statistics': self.stats.copy(),
            'recommendations': self._generate_vocabulary_recommendations()
        }
    
    def _generate_vocabulary_recommendations(self) -> List[str]:
        """Generate vocabulary management recommendations.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check approval rate
        approval_rate = self.stats['tokens_approved'] / max(1, self.stats['tokens_evaluated'])
        if approval_rate < 0.1:
            recommendations.append(f"Very low approval rate ({approval_rate:.3f}) - consider more permissive mode")
        elif approval_rate > 0.8:
            recommendations.append(f"Very high approval rate ({approval_rate:.3f}) - consider stricter evaluation")
        
        # Check quarantine backlog
        if len(self.quarantine_tokens) > 100:
            recommendations.append(f"Large quarantine backlog ({len(self.quarantine_tokens)}) - review quarantine policies")
        
        # Check expansion utilization
        utilization = len(self.expanded_vocabulary) / self.config.max_expansion_size
        if utilization > 0.9:
            recommendations.append(f"Vocabulary expansion near capacity ({utilization:.3f}) - consider increasing limit or pruning")
        
        # Check emergency lockdowns
        if self.stats['emergency_lockdowns'] > 3:
            recommendations.append("Multiple emergency lockdowns detected - review safety thresholds")
        
        # Check constitutional compliance
        constitutional_violations = len([entry for entry in self.expanded_vocabulary.values() 
                                       if entry.constitutional_score < self.config.constitutional_score_threshold])
        if constitutional_violations > len(self.expanded_vocabulary) * 0.1:
            recommendations.append(f"High constitutional violation rate in vocabulary - review approval process")
        
        if not recommendations:
            recommendations.append("Adaptive vocabulary operating within normal parameters")
        
        return recommendations