"""
Token Processor for ESC Module 2.1

Advanced token processing utilities with:
- Constitutional token classification and filtering
- Multi-modal token representation
- Dynamic token type inference
- Safe token transformation and normalization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import re
import logging
from collections import defaultdict, Counter

from .esc_core import TokenType, TokenInfo

logger = logging.getLogger(__name__)


class TokenProcessor:
    """
    Advanced token processing system for constitutional ESC.

    Handles token classification, transformation, and safety validation
    with constitutional compliance checking.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        enable_safety_checking: bool = True,
        constitutional_threshold: float = 0.8,
    ):
        """
        Initialize Token Processor.

        Args:
            vocab_size: Maximum vocabulary size
            enable_safety_checking: Enable constitutional safety checking
            constitutional_threshold: Minimum constitutional compliance score
        """
        self.vocab_size = vocab_size
        self.enable_safety_checking = enable_safety_checking
        self.constitutional_threshold = constitutional_threshold

        # Initialize token classification systems
        self._initialize_classification_rules()
        self._initialize_safety_filters()
        self._initialize_normalization_rules()

        # Token processing statistics
        self.processing_stats = {
            "tokens_processed": 0,
            "tokens_filtered": 0,
            "tokens_normalized": 0,
            "unsafe_tokens_detected": 0,
            "constitutional_violations": 0,
        }

        logger.info(f"TokenProcessor initialized with vocab_size={vocab_size}")

    def _initialize_classification_rules(self):
        """Initialize token classification rules."""
        # Semantic token patterns
        self.semantic_patterns = {
            "noun": r"\b[A-Z][a-z]+\b|\b[a-z]+(?:s|es|ies)\b",
            "verb": r"\b\w+(?:ed|ing|s)\b",
            "adjective": r"\b\w+(?:er|est|ly)\b",
            "concept": r"\b[A-Z][A-Z_]+\b",  # ALL_CAPS concepts
        }

        # Structural token patterns
        self.structural_patterns = {
            "punctuation": r"[.!?,:;]",
            "delimiter": r"[\(\)\[\]{}\<\>]",
            "operator": r"[+\-*/=<>%&|^~]",
            "separator": r"[\s\t\n\r]+",
        }

        # Control token patterns
        self.control_patterns = {
            "conditional": r"\b(if|else|elif|unless|when|while|until)\b",
            "loop": r"\b(for|foreach|while|do|repeat|iterate)\b",
            "function": r"\b(def|function|lambda|proc|method|class)\b",
            "flow": r"\b(return|break|continue|yield|exit|halt)\b",
        }

        # Constitutional token patterns
        self.constitutional_patterns = {
            "policy": r"\b(policy|rule|guideline|principle|standard)\b",
            "constraint": r"\b(must|shall|cannot|forbidden|prohibited|required)\b",
            "safety": r"\b(safe|secure|protect|privacy|consent|permission)\b",
            "ethics": r"\b(ethical|moral|fair|just|honest|transparent)\b",
        }

        # Echo/repetition detection
        self.echo_indicators = {
            "repetition": r"\b(\w+)\s+\1\b",  # Immediate repetition
            "stutter": r"\b(\w)\1{2,}\b",  # Character repetition
            "cycle": r"\b(\w+)(?:\s+\w+)*\s+\1\b",  # Cyclic repetition
        }

    def _initialize_safety_filters(self):
        """Initialize safety filtering patterns."""
        # Unsafe content patterns
        self.unsafe_patterns = {
            "violence": r"\b(kill|murder|assault|attack|violence|harm|hurt|damage|destroy)\w*",
            "hate": r"\b(hate|racist|sexist|bigot|discrimination|prejudice)\w*",
            "illegal": r"\b(illegal|criminal|fraud|theft|piracy|hack|crack)\w*",
            "adult": r"\b(sexual|explicit|pornographic|nsfw|adult)\w*",
            "medical": r"\b(suicide|self-harm|overdose|poisoning|medical-advice)\w*",
            "personal": r"\b(ssn|social-security|password|credit-card|private-key)\w*",
        }

        # Constitutional safety keywords
        self.safety_keywords = {
            "positive": ["safe", "secure", "protected", "ethical", "honest", "transparent", "fair"],
            "negative": ["unsafe", "dangerous", "harmful", "unethical", "deceptive", "biased"],
            "neutral": ["policy", "guideline", "standard", "procedure", "protocol"],
        }

        # Context-sensitive safety rules
        self.context_safety_rules = {
            "medical_context": {
                "allowed": ["treatment", "diagnosis", "symptom", "medicine"],
                "forbidden": ["self-diagnose", "self-medicate", "medical-advice"],
            },
            "financial_context": {
                "allowed": ["budget", "savings", "investment", "financial-planning"],
                "forbidden": ["gambling", "get-rich-quick", "ponzi", "scam"],
            },
        }

    def _initialize_normalization_rules(self):
        """Initialize token normalization rules."""
        # Case normalization rules
        self.case_rules = {
            "preserve_proper_nouns": True,
            "lowercase_common": True,
            "preserve_acronyms": True,
        }

        # Character normalization
        self.char_normalizations = {
            # Unicode normalization
            "'": "'",
            '"': '"',
            '"': '"',
            "—": "-",
            "–": "-",
            "…": "...",
            "™": "",
            "®": "",
            "©": "",
            # Common substitutions
            "&": "and",
            "@": "at",
            "#": "hash",
            "%": "percent",
        }

        # Token length limits
        self.length_limits = {
            "min_length": 1,
            "max_length": 50,
            "truncate_method": "end",  # 'start', 'end', 'middle'
        }

    def process_tokens(
        self, tokens: List[str], context: Optional[Dict[str, Any]] = None
    ) -> List[TokenInfo]:
        """
        Process a list of tokens with classification and safety checking.

        Args:
            tokens: List of input tokens
            context: Optional processing context

        Returns:
            List of processed TokenInfo objects
        """
        processed_tokens = []

        for i, token in enumerate(tokens):
            # Normalize token
            normalized_token = self.normalize_token(token)

            # Skip empty tokens after normalization
            if not normalized_token:
                continue

            # Classify token type
            token_type = self.classify_token_type(normalized_token, i, tokens)

            # Compute constitutional score
            constitutional_score = self.compute_constitutional_score(
                normalized_token, token_type, context
            )

            # Compute risk score
            risk_score = self.compute_risk_score(normalized_token, token_type, context)

            # Safety filtering
            if self.enable_safety_checking:
                if constitutional_score < self.constitutional_threshold:
                    self.processing_stats["constitutional_violations"] += 1

                    # Replace with safe alternative
                    normalized_token = self.get_safe_replacement(normalized_token, token_type)
                    constitutional_score = 1.0
                    risk_score = 0.1

                    logger.warning(
                        f"Constitutional violation filtered: '{token}' -> '{normalized_token}'"
                    )

            # Create token info
            token_info = TokenInfo(
                token=normalized_token,
                token_id=self.get_token_id(normalized_token),
                token_type=token_type,
                semantic_embedding=np.zeros(512),  # Will be filled by ESC
                attention_weights=np.zeros(len(tokens)),
                constitutional_score=constitutional_score,
                risk_score=risk_score,
                echo_strength=self.compute_echo_strength(normalized_token, i, tokens),
                processing_timestamp=0.0,  # Will be set by ESC
            )

            processed_tokens.append(token_info)
            self.processing_stats["tokens_processed"] += 1

        return processed_tokens

    def normalize_token(self, token: str) -> str:
        """
        Normalize token according to constitutional standards.

        Args:
            token: Input token to normalize

        Returns:
            Normalized token
        """
        if not token:
            return ""

        normalized = token

        # Character normalization
        for old_char, new_char in self.char_normalizations.items():
            normalized = normalized.replace(old_char, new_char)

        # Remove control characters
        normalized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", normalized)

        # Length limits
        if len(normalized) > self.length_limits["max_length"]:
            if self.length_limits["truncate_method"] == "end":
                normalized = normalized[: self.length_limits["max_length"]]
            elif self.length_limits["truncate_method"] == "start":
                normalized = normalized[-self.length_limits["max_length"] :]
            elif self.length_limits["truncate_method"] == "middle":
                mid = self.length_limits["max_length"] // 2
                normalized = normalized[:mid] + normalized[-mid:]

        # Case normalization
        if self.case_rules["lowercase_common"] and not self.is_proper_noun(normalized):
            if not (self.case_rules["preserve_acronyms"] and normalized.isupper()):
                normalized = normalized.lower()

        # Remove excessive whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        if normalized != token:
            self.processing_stats["tokens_normalized"] += 1

        return normalized

    def classify_token_type(self, token: str, position: int, full_sequence: List[str]) -> TokenType:
        """
        Classify token type using constitutional rules.

        Args:
            token: Token to classify
            position: Position in sequence
            full_sequence: Full token sequence for context

        Returns:
            TokenType classification
        """
        token_lower = token.lower()

        # Check for unsafe patterns first (highest priority)
        for category, pattern in self.unsafe_patterns.items():
            if re.search(pattern, token_lower):
                self.processing_stats["unsafe_tokens_detected"] += 1
                return TokenType.UNSAFE

        # Check for constitutional patterns
        for category, pattern in self.constitutional_patterns.items():
            if re.search(pattern, token_lower):
                return TokenType.CONSTITUTIONAL

        # Check for control patterns
        for category, pattern in self.control_patterns.items():
            if re.search(pattern, token_lower):
                return TokenType.CONTROL

        # Check for structural patterns
        for category, pattern in self.structural_patterns.items():
            if re.search(pattern, token):  # Use original case for structural
                return TokenType.STRUCTURAL

        # Check for echo patterns
        if self.detect_echo_pattern(token, position, full_sequence):
            return TokenType.ECHO

        # Default to semantic
        return TokenType.SEMANTIC

    def compute_constitutional_score(
        self, token: str, token_type: TokenType, context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute constitutional compliance score for token.

        Args:
            token: Token to evaluate
            token_type: Classified token type
            context: Optional context information

        Returns:
            Constitutional compliance score [0, 1]
        """
        score = 1.0  # Start with perfect compliance
        token_lower = token.lower()

        # Type-based scoring
        if token_type == TokenType.UNSAFE:
            score = 0.1  # Very low score for unsafe content
        elif token_type == TokenType.CONSTITUTIONAL:
            score = 1.0  # Perfect score for constitutional content
        elif token_type == TokenType.STRUCTURAL:
            score = 0.9  # High score for structural elements

        # Safety keyword analysis
        positive_matches = sum(
            1 for keyword in self.safety_keywords["positive"] if keyword in token_lower
        )
        negative_matches = sum(
            1 for keyword in self.safety_keywords["negative"] if keyword in token_lower
        )

        # Adjust score based on safety keywords
        score += positive_matches * 0.1
        score -= negative_matches * 0.3

        # Context-sensitive adjustments
        if context:
            context_type = context.get("type", "general")
            if context_type in self.context_safety_rules:
                rules = self.context_safety_rules[context_type]

                allowed_matches = sum(1 for keyword in rules["allowed"] if keyword in token_lower)
                forbidden_matches = sum(
                    1 for keyword in rules["forbidden"] if keyword in token_lower
                )

                score += allowed_matches * 0.05
                score -= forbidden_matches * 0.5

        # Unsafe pattern penalties
        for category, pattern in self.unsafe_patterns.items():
            if re.search(pattern, token_lower):
                score *= 0.1  # Heavy penalty for unsafe patterns

        # Clamp score to valid range
        return max(0.0, min(1.0, score))

    def compute_risk_score(
        self, token: str, token_type: TokenType, context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute risk score for token.

        Args:
            token: Token to evaluate
            token_type: Classified token type
            context: Optional context information

        Returns:
            Risk score [0, 1] where higher is more risky
        """
        risk = 0.1  # Base low risk
        token_lower = token.lower()

        # Type-based risk
        if token_type == TokenType.UNSAFE:
            risk = 0.9
        elif token_type == TokenType.CONSTITUTIONAL:
            risk = 0.1
        elif token_type == TokenType.ECHO:
            risk = 0.4  # Moderate risk for repetitive content

        # Pattern-based risk assessment
        for category, pattern in self.unsafe_patterns.items():
            if re.search(pattern, token_lower):
                risk = max(risk, 0.8)

        # Length-based risk (very long tokens can be suspicious)
        if len(token) > 30:
            risk += 0.2

        # Character-based risk (excessive special characters)
        special_char_ratio = len(re.findall(r"[^\w\s]", token)) / max(1, len(token))
        if special_char_ratio > 0.3:
            risk += 0.3

        # Case anomaly risk (aLtErNaTiNg case can indicate spam)
        if self.detect_case_anomaly(token):
            risk += 0.2

        return min(1.0, risk)

    def compute_echo_strength(self, token: str, position: int, full_sequence: List[str]) -> float:
        """
        Compute echo/repetition strength for token.

        Args:
            token: Current token
            position: Position in sequence
            full_sequence: Full token sequence

        Returns:
            Echo strength [0, 1] where higher indicates more repetition
        """
        if position == 0:
            return 0.0

        echo_strength = 0.0

        # Look for immediate repetition
        if position > 0 and full_sequence[position - 1].lower() == token.lower():
            echo_strength = max(echo_strength, 0.8)

        # Look for recent repetitions (within last 10 tokens)
        lookback = min(10, position)
        recent_tokens = [full_sequence[i].lower() for i in range(position - lookback, position)]
        count = recent_tokens.count(token.lower())

        if count > 0:
            echo_strength = max(echo_strength, min(0.9, count * 0.3))

        # Pattern-based echo detection
        for pattern_name, pattern in self.echo_indicators.items():
            if re.search(pattern, token.lower()):
                echo_strength = max(echo_strength, 0.6)

        return echo_strength

    def detect_echo_pattern(self, token: str, position: int, full_sequence: List[str]) -> bool:
        """
        Detect if token is part of an echo/repetition pattern.

        Args:
            token: Token to check
            position: Position in sequence
            full_sequence: Full token sequence

        Returns:
            True if echo pattern detected
        """
        # Check various echo patterns
        for pattern_name, pattern in self.echo_indicators.items():
            if re.search(pattern, token):
                return True

        # Check for cyclic repetition in context
        if position >= 3:
            window = full_sequence[max(0, position - 3) : position + 1]
            if len(set(w.lower() for w in window)) < len(window) * 0.7:
                return True

        return False

    def is_proper_noun(self, token: str) -> bool:
        """Check if token is a proper noun."""
        # Simple heuristic: starts with capital letter and contains lowercase
        return (
            len(token) > 1
            and token[0].isupper()
            and any(c.islower() for c in token[1:])
            and not token.isupper()
        )  # Not an acronym

    def detect_case_anomaly(self, token: str) -> bool:
        """Detect abnormal case patterns that might indicate spam."""
        if len(token) < 3:
            return False

        # Count case transitions
        transitions = 0
        for i in range(1, len(token)):
            if token[i - 1].isalpha() and token[i].isalpha():
                if token[i - 1].islower() != token[i].islower():
                    transitions += 1

        # High transition rate suggests alternating case
        transition_rate = transitions / max(1, len([c for c in token if c.isalpha()]))
        return transition_rate > 0.3

    def get_safe_replacement(self, token: str, token_type: TokenType) -> str:
        """
        Get a safe replacement for a potentially unsafe token.

        Args:
            token: Original token
            token_type: Token type

        Returns:
            Safe replacement token
        """
        # Type-based replacements
        if token_type == TokenType.UNSAFE:
            return "[FILTERED]"
        elif token_type == TokenType.ECHO:
            return "[ECHO-FILTERED]"

        # Pattern-based replacements
        token_lower = token.lower()

        for category, pattern in self.unsafe_patterns.items():
            if re.search(pattern, token_lower):
                replacements = {
                    "violence": "[CONTENT-FILTERED]",
                    "hate": "[INAPPROPRIATE]",
                    "illegal": "[RESTRICTED]",
                    "adult": "[MATURE-CONTENT]",
                    "medical": "[MEDICAL-INFO]",
                    "personal": "[PRIVATE-INFO]",
                }
                return replacements.get(category, "[FILTERED]")

        # Default safe replacement
        return f"[SAFE-{len(token)}]"

    def get_token_id(self, token: str) -> int:
        """
        Get unique ID for token.

        Args:
            token: Token to get ID for

        Returns:
            Unique token ID
        """
        # Simple hash-based ID (in practice would use proper tokenizer)
        import hashlib

        token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
        return token_hash % self.vocab_size

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "statistics": self.processing_stats.copy(),
            "safety_status": {
                "safety_checking_enabled": self.enable_safety_checking,
                "constitutional_threshold": self.constitutional_threshold,
                "unsafe_detection_rate": (
                    self.processing_stats["unsafe_tokens_detected"]
                    / max(1, self.processing_stats["tokens_processed"])
                ),
                "constitutional_violation_rate": (
                    self.processing_stats["constitutional_violations"]
                    / max(1, self.processing_stats["tokens_processed"])
                ),
            },
            "normalization_stats": {
                "normalization_rate": (
                    self.processing_stats["tokens_normalized"]
                    / max(1, self.processing_stats["tokens_processed"])
                )
            },
        }
