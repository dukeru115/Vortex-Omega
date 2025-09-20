"""
Echo-Semantic Converter (ESC) Core Module v2.1

Implements advanced echo-semantic conversion for token processing with:
- Multi-layered semantic field coupling
- Constitutional safety in content processing
- Dynamic attention and memory integration
- Adaptive vocabulary expansion with constitutional constraints
- Risk assessment and emergency content filtering
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import re
import json
from collections import defaultdict, deque
import hashlib
import time

# Configure logging
logger = logging.getLogger(__name__)

# Import telemetry system (conditional import to handle dependencies)
try:
    from .telemetry import get_telemetry_collector, ESCTelemetryCollector
    TELEMETRY_AVAILABLE = True
except ImportError:
    logger.warning("Telemetry system not available - running without telemetry")
    TELEMETRY_AVAILABLE = False


class TokenType(Enum):
    """Token classification types."""
    SEMANTIC = "semantic"      # Semantic content tokens
    STRUCTURAL = "structural"  # Syntax/structure tokens  
    CONTROL = "control"        # Control flow tokens
    ECHO = "echo"             # Echo/repetition tokens
    CONSTITUTIONAL = "constitutional"  # Constitutional constraint tokens
    UNSAFE = "unsafe"         # Potentially unsafe tokens
    UNKNOWN = "unknown"       # Unclassified tokens


class ProcessingMode(Enum):
    """ESC processing modes."""
    CONSERVATIVE = "conservative"  # Strict constitutional compliance
    BALANCED = "balanced"         # Standard processing
    CREATIVE = "creative"         # Enhanced semantic exploration
    EMERGENCY = "emergency"       # Emergency safe mode


@dataclass
class ESCConfig:
    """Configuration for Echo-Semantic Converter."""
    # Core parameters
    embedding_dim: int = 512
    max_sequence_length: int = 2048
    vocabulary_size: int = 50000
    
    # Semantic field parameters
    semantic_field_layers: int = 6
    attention_heads: int = 8
    hidden_dim: int = 2048
    
    # Constitutional parameters
    max_unsafe_ratio: float = 0.1
    constitutional_threshold: float = 0.8
    emergency_threshold: float = 0.95
    
    # Learning parameters
    learning_rate: float = 1e-4
    adaptation_rate: float = 0.01
    memory_decay: float = 0.99
    
    # Processing parameters
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    enable_constitutional_filtering: bool = True
    enable_adaptive_vocabulary: bool = True
    enable_memory_integration: bool = True
    
    # Safety parameters
    max_generation_length: int = 1000
    repetition_penalty: float = 1.2
    constitutional_weight: float = 0.7


@dataclass
class TokenInfo:
    """Information about a processed token."""
    token: str
    token_id: int
    token_type: TokenType
    semantic_embedding: np.ndarray
    attention_weights: np.ndarray
    constitutional_score: float
    risk_score: float
    echo_strength: float
    processing_timestamp: float


@dataclass
class ProcessingResult:
    """Result of ESC processing."""
    processed_tokens: List[TokenInfo]
    semantic_field_state: np.ndarray
    attention_map: np.ndarray
    constitutional_metrics: Dict[str, float]
    processing_stats: Dict[str, Any]
    warnings: List[str]
    emergency_triggered: bool = False


class EchoSemanticConverter:
    """
    Echo-Semantic Converter v2.1 - Advanced token processing for NFCS.
    
    Implements constitutional echo-semantic conversion with:
    - Multi-scale attention mechanisms
    - Semantic field coupling with neural dynamics
    - Adaptive vocabulary with safety constraints
    - Constitutional content filtering and risk assessment
    """
    
    def __init__(self, config: ESCConfig):
        """
        Initialize Echo-Semantic Converter.
        
        Args:
            config: ESC configuration object
        """
        self.config = config
        self.processing_mode = config.processing_mode
        
        # Initialize telemetry system
        self.telemetry_enabled = TELEMETRY_AVAILABLE
        if self.telemetry_enabled:
            self.telemetry = get_telemetry_collector()
            logger.info("ESC telemetry system initialized")
        else:
            self.telemetry = None
        
        # Initialize core components
        self._initialize_embeddings()
        self._initialize_semantic_fields()
        self._initialize_attention_mechanisms()
        self._initialize_constitutional_system()
        self._initialize_adaptive_vocabulary()
        
        # Processing state
        self.current_sequence = []
        self.processing_history = deque(maxlen=1000)
        self.constitutional_violations = []
        self.emergency_mode = False
        
        # Semantic anchor tracking for telemetry
        self.semantic_anchors = {}
        self.anchor_update_threshold = 0.1
        
        # Performance metrics
        self.processing_stats = {
            'total_tokens_processed': 0,
            'constitutional_interventions': 0,
            'emergency_activations': 0,
            'vocabulary_adaptations': 0,
            'average_processing_time': 0.0,
            'telemetry_sessions': 0
        }
        
        logger.info(f"ESC v2.1 initialized with {config.vocabulary_size} vocabulary size")
        logger.info(f"Processing mode: {config.processing_mode.value}")
        
    def _initialize_embeddings(self):
        """Initialize token embeddings and encoding systems."""
        # Token embeddings matrix
        self.token_embeddings = np.random.randn(
            self.config.vocabulary_size, 
            self.config.embedding_dim
        ) * 0.1
        
        # Positional encodings
        self.positional_encodings = self._create_positional_encodings()
        
        # Type embeddings for different token types
        self.type_embeddings = {}
        for token_type in TokenType:
            self.type_embeddings[token_type] = np.random.randn(self.config.embedding_dim) * 0.1
            
    def _create_positional_encodings(self) -> np.ndarray:
        """Create sinusoidal positional encodings."""
        position = np.arange(self.config.max_sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.config.embedding_dim, 2) * 
                         -(np.log(10000.0) / self.config.embedding_dim))
        
        pe = np.zeros((self.config.max_sequence_length, self.config.embedding_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
        
    def _initialize_semantic_fields(self):
        """Initialize semantic field coupling system."""
        # Semantic field state vectors
        self.semantic_field_state = np.zeros(
            (self.config.semantic_field_layers, self.config.embedding_dim)
        )
        
        # Coupling matrices between semantic fields and token embeddings
        self.field_coupling_matrices = []
        for layer in range(self.config.semantic_field_layers):
            coupling_matrix = np.random.randn(
                self.config.embedding_dim, 
                self.config.embedding_dim
            ) * 0.1 / np.sqrt(self.config.embedding_dim)
            self.field_coupling_matrices.append(coupling_matrix)
        
        # Field dynamics parameters
        self.field_decay_rates = np.linspace(0.95, 0.99, self.config.semantic_field_layers)
        self.field_coupling_strengths = np.linspace(0.1, 0.5, self.config.semantic_field_layers)
        
    def _initialize_attention_mechanisms(self):
        """Initialize multi-scale attention mechanisms."""
        # Multi-head attention parameters
        self.attention_weights = []
        for head in range(self.config.attention_heads):
            # Query, Key, Value projection matrices
            W_q = np.random.randn(self.config.embedding_dim, self.config.embedding_dim // self.config.attention_heads) * 0.1
            W_k = np.random.randn(self.config.embedding_dim, self.config.embedding_dim // self.config.attention_heads) * 0.1  
            W_v = np.random.randn(self.config.embedding_dim, self.config.embedding_dim // self.config.attention_heads) * 0.1
            
            self.attention_weights.append({'W_q': W_q, 'W_k': W_k, 'W_v': W_v})
        
        # Output projection
        self.attention_output_projection = np.random.randn(self.config.embedding_dim, self.config.embedding_dim) * 0.1
        
        # Attention scales for multi-scale processing
        self.attention_scales = [1, 2, 4, 8]  # Different attention window sizes
        
    def _initialize_constitutional_system(self):
        """Initialize constitutional safety and filtering system."""
        # Constitutional word lists and patterns
        self.unsafe_patterns = [
            r'\b(kill|death|violence|harm)\w*',
            r'\b(hate|racist|sexist)\w*',
            r'\b(illegal|criminal|fraud)\w*',
        ]
        
        # Constitutional principles (simplified representation)
        self.constitutional_principles = {
            'safety': 0.9,      # High priority on safety
            'truthfulness': 0.8, # High priority on truth
            'fairness': 0.7,    # Moderate priority on fairness
            'privacy': 0.6,     # Moderate priority on privacy
            'autonomy': 0.5     # Moderate priority on autonomy
        }
        
        # Constitutional violation tracking
        self.constitutional_history = deque(maxlen=500)
        
    def _initialize_adaptive_vocabulary(self):
        """Initialize adaptive vocabulary expansion system."""
        # New token discovery and integration
        self.discovered_tokens = set()
        self.token_frequency = defaultdict(int)
        self.token_constitutional_scores = {}
        
        # Vocabulary adaptation thresholds
        self.min_frequency_threshold = 5
        self.min_constitutional_score = 0.6
        
    def process_sequence(self, 
                        input_tokens: List[str], 
                        context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process a sequence of tokens through the ESC system.
        
        Args:
            input_tokens: List of input tokens to process
            context: Optional context information for processing
            
        Returns:
            ProcessingResult with processed tokens and metrics
        """
        start_time = time.time()
        
        # Start telemetry session
        session_id = None
        if self.telemetry_enabled and self.telemetry:
            session_id = self.telemetry.start_session()
            self.processing_stats['telemetry_sessions'] += 1
        
        # Initialize processing context
        if context is None:
            context = {}
        
        # Emergency mode check
        if self.emergency_mode:
            result = self._emergency_processing(input_tokens)
            # End telemetry session with emergency warning
            if session_id and self.telemetry:
                self.telemetry.end_session(warnings=["Emergency mode activated"])
            return result
        
        try:
            # Tokenize and classify input
            token_infos = self._tokenize_and_classify(input_tokens)
            
            # Apply constitutional filtering
            if self.config.enable_constitutional_filtering:
                token_infos = self._apply_constitutional_filtering(token_infos)
            
            # Semantic field processing with anchor tracking
            token_infos = self._process_semantic_fields(token_infos)
            self._track_semantic_anchors(token_infos)
            
            # Multi-scale attention processing
            attention_map = self._compute_multi_scale_attention(token_infos)
            
            # Update token embeddings based on attention
            token_infos = self._update_token_embeddings(token_infos, attention_map)
            
            # Adaptive vocabulary update
            if self.config.enable_adaptive_vocabulary:
                self._update_adaptive_vocabulary(token_infos)
            
            # Compute constitutional metrics
            constitutional_metrics = self._compute_constitutional_metrics(token_infos)
            
            # Check for emergency conditions
            if constitutional_metrics['overall_risk'] > self.config.emergency_threshold:
                self.emergency_mode = True
                logger.warning(f"EMERGENCY: Constitutional risk exceeded threshold: {constitutional_metrics['overall_risk']:.3f}")
                # End telemetry with emergency warning
                if session_id and self.telemetry:
                    self.telemetry.end_session(warnings=["Emergency threshold exceeded"])
                return self._emergency_processing(input_tokens)
            
            # Update processing statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(len(input_tokens), processing_time)
            
            # Record telemetry data
            if self.telemetry_enabled and self.telemetry:
                self._record_telemetry_data(
                    len(input_tokens), processing_time, 
                    constitutional_metrics, attention_map
                )
            
            # Create processing result
            result = ProcessingResult(
                processed_tokens=token_infos,
                semantic_field_state=self.semantic_field_state.copy(),
                attention_map=attention_map,
                constitutional_metrics=constitutional_metrics,
                processing_stats=self.processing_stats.copy(),
                warnings=[]
            )
            
            # Store in history
            self.processing_history.append(result)
            
            # End telemetry session
            if session_id and self.telemetry:
                self.telemetry.end_session()
            
            return result
            
        except Exception as e:
            logger.error(f"ESC processing error: {e}")
            # End telemetry with error warning
            if session_id and self.telemetry:
                self.telemetry.end_session(warnings=[f"Processing error: {str(e)}"])
            # Fallback to emergency processing
            self.emergency_mode = True
            return self._emergency_processing(input_tokens)
    
    def _tokenize_and_classify(self, input_tokens: List[str]) -> List[TokenInfo]:
        """Tokenize input and classify token types."""
        token_infos = []
        
        for i, token in enumerate(input_tokens):
            # Get or create token ID
            token_id = self._get_token_id(token)
            
            # Classify token type
            token_type = self._classify_token_type(token)
            
            # Get base embedding
            embedding = self._get_token_embedding(token_id, i)
            
            # Compute initial constitutional score
            constitutional_score = self._compute_constitutional_score(token)
            
            # Compute risk score
            risk_score = self._compute_risk_score(token, token_type)
            
            # Compute echo strength (repetition detection)
            echo_strength = self._compute_echo_strength(token, i)
            
            token_info = TokenInfo(
                token=token,
                token_id=token_id,
                token_type=token_type,
                semantic_embedding=embedding,
                attention_weights=np.zeros(len(input_tokens)),  # Will be computed later
                constitutional_score=constitutional_score,
                risk_score=risk_score,
                echo_strength=echo_strength,
                processing_timestamp=time.time()
            )
            
            token_infos.append(token_info)
        
        return token_infos
    
    def _classify_token_type(self, token: str) -> TokenType:
        """Classify token into appropriate type."""
        token_lower = token.lower()
        
        # Check for unsafe patterns
        for pattern in self.unsafe_patterns:
            if re.search(pattern, token_lower):
                return TokenType.UNSAFE
        
        # Check for constitutional keywords
        constitutional_keywords = ['must', 'shall', 'cannot', 'forbidden', 'required', 'policy']
        if any(keyword in token_lower for keyword in constitutional_keywords):
            return TokenType.CONSTITUTIONAL
        
        # Check for structural tokens
        if token in ['.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}']:
            return TokenType.STRUCTURAL
        
        # Check for control tokens
        control_keywords = ['if', 'then', 'else', 'while', 'for', 'function', 'return']
        if token_lower in control_keywords:
            return TokenType.CONTROL
        
        # Default to semantic
        return TokenType.SEMANTIC
    
    def _get_token_id(self, token: str) -> int:
        """Get or create token ID."""
        # Simple hash-based token ID (in practice, would use proper tokenizer)
        token_hash = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
        return token_hash % self.config.vocabulary_size
    
    def _get_token_embedding(self, token_id: int, position: int) -> np.ndarray:
        """Get token embedding with positional encoding."""
        # Base token embedding
        base_embedding = self.token_embeddings[token_id].copy()
        
        # Add positional encoding
        if position < self.config.max_sequence_length:
            base_embedding += self.positional_encodings[position]
        
        return base_embedding
    
    def _compute_constitutional_score(self, token: str) -> float:
        """Compute constitutional compliance score for token."""
        score = 1.0  # Start with perfect score
        
        # Check against unsafe patterns
        token_lower = token.lower()
        for pattern in self.unsafe_patterns:
            if re.search(pattern, token_lower):
                score *= 0.1  # Heavily penalize unsafe content
        
        # Boost constitutional keywords
        constitutional_keywords = ['safety', 'truth', 'fair', 'honest', 'respect']
        if any(keyword in token_lower for keyword in constitutional_keywords):
            score = min(1.0, score * 1.2)
        
        return score
    
    def _compute_risk_score(self, token: str, token_type: TokenType) -> float:
        """Compute risk score for token."""
        if token_type == TokenType.UNSAFE:
            return 0.9
        elif token_type == TokenType.CONSTITUTIONAL:
            return 0.1
        else:
            # Base risk assessment
            risk = 0.2  # Default low risk
            
            # Check for potentially risky patterns
            if re.search(r'\b\w{20,}\b', token):  # Very long words
                risk += 0.2
            
            if re.search(r'[^\w\s\.,!?;:()\[\]{}-]', token):  # Special characters
                risk += 0.1
            
            return min(0.8, risk)
    
    def _compute_echo_strength(self, token: str, position: int) -> float:
        """Compute echo/repetition strength."""
        if position == 0:
            return 0.0
        
        # Look for recent repetitions
        echo_strength = 0.0
        current_tokens = [info.token for info in self.current_sequence[-10:]]  # Last 10 tokens
        
        count = current_tokens.count(token)
        if count > 0:
            echo_strength = min(0.9, count * 0.3)
        
        return echo_strength
    
    def _apply_constitutional_filtering(self, token_infos: List[TokenInfo]) -> List[TokenInfo]:
        """Apply constitutional filtering to tokens."""
        filtered_tokens = []
        violations = []
        
        for token_info in token_infos:
            # Check constitutional score
            if token_info.constitutional_score < self.config.constitutional_threshold:
                # Constitutional violation detected
                violation = {
                    'token': token_info.token,
                    'score': token_info.constitutional_score,
                    'risk': token_info.risk_score,
                    'timestamp': time.time()
                }
                violations.append(violation)
                
                # Replace with safe alternative or remove
                if self.processing_mode == ProcessingMode.CONSERVATIVE:
                    continue  # Remove entirely
                else:
                    # Replace with neutral token
                    token_info.token = "[FILTERED]"
                    token_info.constitutional_score = 1.0
                    token_info.risk_score = 0.1
            
            filtered_tokens.append(token_info)
        
        # Log violations
        if violations:
            self.constitutional_violations.extend(violations)
            self.processing_stats['constitutional_interventions'] += len(violations)
            logger.warning(f"Constitutional filtering removed {len(violations)} tokens")
        
        return filtered_tokens
    
    def _process_semantic_fields(self, token_infos: List[TokenInfo]) -> List[TokenInfo]:
        """Process tokens through semantic field coupling."""
        # Update semantic field states based on token embeddings
        for layer in range(self.config.semantic_field_layers):
            # Compute field activation from current tokens
            field_activation = np.zeros(self.config.embedding_dim)
            
            for token_info in token_infos:
                # Couple token embedding to field
                coupling_strength = self.field_coupling_strengths[layer]
                coupled_activation = np.dot(
                    self.field_coupling_matrices[layer], 
                    token_info.semantic_embedding
                ) * coupling_strength
                field_activation += coupled_activation
            
            # Apply field dynamics (decay + activation)
            decay_rate = self.field_decay_rates[layer]
            self.semantic_field_state[layer] = (
                decay_rate * self.semantic_field_state[layer] + 
                (1 - decay_rate) * field_activation
            )
        
        # Update token embeddings based on field feedback
        for token_info in token_infos:
            field_feedback = np.zeros(self.config.embedding_dim)
            
            for layer in range(self.config.semantic_field_layers):
                # Compute feedback from semantic field to token
                layer_feedback = np.dot(
                    self.field_coupling_matrices[layer].T,  # Transpose for feedback
                    self.semantic_field_state[layer]
                ) * self.field_coupling_strengths[layer]
                
                field_feedback += layer_feedback
            
            # Mix original embedding with field feedback
            mixing_ratio = 0.3  # 30% field feedback, 70% original
            token_info.semantic_embedding = (
                (1 - mixing_ratio) * token_info.semantic_embedding +
                mixing_ratio * field_feedback
            )
        
        return token_infos
    
    def _compute_multi_scale_attention(self, token_infos: List[TokenInfo]) -> np.ndarray:
        """Compute multi-scale attention weights."""
        seq_len = len(token_infos)
        if seq_len == 0:
            return np.array([])
        
        # Stack embeddings
        embeddings = np.stack([info.semantic_embedding for info in token_infos])
        
        # Initialize attention map
        attention_map = np.zeros((seq_len, seq_len))
        
        # Compute attention for each head and scale
        for head_idx, head_weights in enumerate(self.attention_weights):
            for scale in self.attention_scales:
                # Project embeddings
                Q = np.dot(embeddings, head_weights['W_q'])
                K = np.dot(embeddings, head_weights['W_k'])
                V = np.dot(embeddings, head_weights['W_v'])
                
                # Compute attention scores with scale
                scores = np.dot(Q, K.T) / np.sqrt(Q.shape[-1])
                
                # Apply scale mask (only attend within scale window)
                mask = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)) <= scale
                scores = np.where(mask, scores, -np.inf)
                
                # Softmax attention weights
                attention_weights = self._softmax(scores, axis=-1)
                
                # Add to attention map
                attention_map += attention_weights / (len(self.attention_weights) * len(self.attention_scales))
        
        return attention_map
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax with numerical stability."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _update_token_embeddings(self, token_infos: List[TokenInfo], attention_map: np.ndarray) -> List[TokenInfo]:
        """Update token embeddings based on attention."""
        if len(token_infos) == 0:
            return token_infos
        
        # Update attention weights in token info
        for i, token_info in enumerate(token_infos):
            token_info.attention_weights = attention_map[i]
        
        # Compute attention-weighted updates
        embeddings = np.stack([info.semantic_embedding for info in token_infos])
        
        # Apply attention-weighted mixing
        for i, token_info in enumerate(token_infos):
            # Weighted combination of all token embeddings
            attended_embedding = np.dot(attention_map[i], embeddings)
            
            # Mix with original embedding
            mixing_ratio = 0.2  # 20% attention, 80% original
            token_info.semantic_embedding = (
                (1 - mixing_ratio) * token_info.semantic_embedding +
                mixing_ratio * attended_embedding
            )
        
        return token_infos
    
    def _update_adaptive_vocabulary(self, token_infos: List[TokenInfo]):
        """Update adaptive vocabulary based on processed tokens."""
        for token_info in token_infos:
            token = token_info.token
            
            # Track token frequency
            self.token_frequency[token] += 1
            
            # Update constitutional score history
            if token in self.token_constitutional_scores:
                # Running average
                old_score = self.token_constitutional_scores[token]
                new_score = token_info.constitutional_score
                self.token_constitutional_scores[token] = 0.9 * old_score + 0.1 * new_score
            else:
                self.token_constitutional_scores[token] = token_info.constitutional_score
            
            # Check for vocabulary expansion
            if (token not in self.discovered_tokens and
                self.token_frequency[token] >= self.min_frequency_threshold and
                self.token_constitutional_scores[token] >= self.min_constitutional_score):
                
                self.discovered_tokens.add(token)
                self.processing_stats['vocabulary_adaptations'] += 1
                logger.info(f"Adaptive vocabulary expanded: '{token}' (freq={self.token_frequency[token]}, const={self.token_constitutional_scores[token]:.3f})")
    
    def _compute_constitutional_metrics(self, token_infos: List[TokenInfo]) -> Dict[str, float]:
        """Compute constitutional compliance and safety metrics."""
        if not token_infos:
            return {
                'overall_risk': 0.0,
                'constitutional_compliance': 1.0,
                'unsafe_token_ratio': 0.0,
                'echo_strength_avg': 0.0,
                'attention_diversity': 0.0
            }
        
        # Risk assessment
        risk_scores = [info.risk_score for info in token_infos]
        overall_risk = np.mean(risk_scores)
        
        # Constitutional compliance
        constitutional_scores = [info.constitutional_score for info in token_infos]
        constitutional_compliance = np.mean(constitutional_scores)
        
        # Unsafe token ratio
        unsafe_count = sum(1 for info in token_infos if info.token_type == TokenType.UNSAFE)
        unsafe_ratio = unsafe_count / len(token_infos)
        
        # Echo strength (repetition analysis)
        echo_strengths = [info.echo_strength for info in token_infos]
        echo_strength_avg = np.mean(echo_strengths)
        
        # Attention diversity (entropy of attention weights)
        attention_entropy = 0.0
        for info in token_infos:
            if len(info.attention_weights) > 0:
                # Normalize attention weights
                weights = info.attention_weights + 1e-8  # Avoid log(0)
                weights = weights / np.sum(weights)
                entropy = -np.sum(weights * np.log(weights))
                attention_entropy += entropy
        
        attention_diversity = attention_entropy / len(token_infos) if token_infos else 0.0
        
        return {
            'overall_risk': overall_risk,
            'constitutional_compliance': constitutional_compliance,
            'unsafe_token_ratio': unsafe_ratio,
            'echo_strength_avg': echo_strength_avg,
            'attention_diversity': attention_diversity
        }
    
    def _update_processing_stats(self, num_tokens: int, processing_time: float):
        """Update processing statistics."""
        self.processing_stats['total_tokens_processed'] += num_tokens
        
        # Update average processing time
        old_avg = self.processing_stats['average_processing_time']
        new_avg = (old_avg * 0.9 + processing_time * 0.1)  # Exponential moving average
        self.processing_stats['average_processing_time'] = new_avg
    
    def _emergency_processing(self, input_tokens: List[str]) -> ProcessingResult:
        """Emergency safe processing mode."""
        logger.warning("ESC running in emergency mode - applying maximum safety filters")
        
        self.processing_stats['emergency_activations'] += 1
        
        # Extremely conservative processing
        safe_tokens = []
        for token in input_tokens:
            # Only allow very safe tokens
            if self._compute_constitutional_score(token) > 0.9:
                safe_tokens.append(token)
            else:
                safe_tokens.append("[SAFE]")  # Replace with safe placeholder
        
        # Create minimal token infos
        token_infos = []
        for i, token in enumerate(safe_tokens):
            token_info = TokenInfo(
                token=token,
                token_id=self._get_token_id(token),
                token_type=TokenType.CONSTITUTIONAL,
                semantic_embedding=np.zeros(self.config.embedding_dim),
                attention_weights=np.zeros(len(safe_tokens)),
                constitutional_score=1.0,
                risk_score=0.0,
                echo_strength=0.0,
                processing_timestamp=time.time()
            )
            token_infos.append(token_info)
        
        return ProcessingResult(
            processed_tokens=token_infos,
            semantic_field_state=np.zeros_like(self.semantic_field_state),
            attention_map=np.eye(len(safe_tokens)) if safe_tokens else np.array([]),
            constitutional_metrics={
                'overall_risk': 0.0,
                'constitutional_compliance': 1.0,
                'unsafe_token_ratio': 0.0,
                'echo_strength_avg': 0.0,
                'attention_diversity': 0.0
            },
            processing_stats=self.processing_stats.copy(),
            warnings=["Emergency mode active - reduced functionality"],
            emergency_triggered=True
        )
    
    def set_processing_mode(self, mode: ProcessingMode):
        """Set the processing mode."""
        old_mode = self.processing_mode
        self.processing_mode = mode
        self.config.processing_mode = mode
        logger.info(f"ESC processing mode changed: {old_mode.value} -> {mode.value}")
    
    def reset_emergency_mode(self):
        """Reset emergency mode after manual intervention."""
        if self.emergency_mode:
            self.emergency_mode = False
            logger.info("ESC emergency mode reset - resuming normal operation")
    
    def get_processing_report(self) -> Dict[str, Any]:
        """Generate comprehensive processing report."""
        return {
            'status': 'emergency' if self.emergency_mode else 'normal',
            'processing_mode': self.processing_mode.value,
            'statistics': self.processing_stats.copy(),
            'constitutional_history': {
                'violations_count': len(self.constitutional_violations),
                'recent_violations': list(self.constitutional_violations)[-10:] if self.constitutional_violations else []
            },
            'vocabulary_status': {
                'base_size': self.config.vocabulary_size,
                'discovered_tokens': len(self.discovered_tokens),
                'total_tracked': len(self.token_frequency)
            },
            'semantic_field_status': {
                'field_layers': self.config.semantic_field_layers,
                'current_activation_norm': float(np.linalg.norm(self.semantic_field_state)),
                'field_stability': float(np.std(self.semantic_field_state))
            },
            'recommendations': self._generate_processing_recommendations()
        }

    def _track_semantic_anchors(self, token_infos: List[TokenInfo]):
        """Track semantic anchors for telemetry and stability monitoring."""
        if not self.telemetry_enabled or not self.telemetry:
            return
        
        for token_info in token_infos:
            # Create semantic anchor ID based on token semantic properties
            anchor_id = f"anchor_{token_info.token_type.value}_{hash(token_info.token) % 10000}"
            
            # Track anchor activation
            embedding_vector = token_info.semantic_embedding.tolist() if hasattr(token_info.semantic_embedding, 'tolist') else []
            activation_strength = float(token_info.echo_strength)
            
            self.telemetry.track_semantic_anchor(
                anchor_id=anchor_id,
                embedding_vector=embedding_vector,
                activation_strength=activation_strength
            )
            
            # Update local anchor tracking
            if anchor_id not in self.semantic_anchors:
                self.semantic_anchors[anchor_id] = {
                    'created': time.time(),
                    'last_activation': time.time(),
                    'activation_count': 1,
                    'stability_score': 1.0
                }
            else:
                anchor_data = self.semantic_anchors[anchor_id]
                anchor_data['last_activation'] = time.time()
                anchor_data['activation_count'] += 1
                
                # Simple stability assessment
                time_since_creation = time.time() - anchor_data['created']
                expected_activations = time_since_creation * 0.1  # Expected rate
                stability = min(1.0, anchor_data['activation_count'] / max(1, expected_activations))
                anchor_data['stability_score'] = stability

    def _record_telemetry_data(self, token_count: int, processing_time: float,
                             constitutional_metrics: Dict[str, float],
                             attention_map: Any):
        """Record comprehensive telemetry data for current processing session."""
        if not self.telemetry_enabled or not self.telemetry:
            return
        
        # Prepare semantic field state data
        semantic_field_data = {
            'field_energy': float(getattr(self, 'semantic_field_state', [0.0]).mean() if hasattr(getattr(self, 'semantic_field_state', []), 'mean') else 0.0),
            'field_stability': 1.0 - float(getattr(self, 'semantic_field_state', [0.0]).std() if hasattr(getattr(self, 'semantic_field_state', []), 'std') else 0.0),
            'active_layers': getattr(self.config, 'semantic_field_layers', 6)
        }
        
        # Prepare attention pattern data
        attention_data = {
            'attention_entropy': self._calculate_attention_entropy(attention_map),
            'attention_focus': self._calculate_attention_focus(attention_map),
            'pattern_complexity': 'balanced'  # Placeholder
        }
        
        # Record the data
        self.telemetry.record_processing_metrics(
            token_count=token_count,
            processing_time=processing_time,
            semantic_field_state=semantic_field_data,
            constitutional_scores=constitutional_metrics,
            attention_weights=attention_data
        )

    def _calculate_attention_entropy(self, attention_map: Any) -> float:
        """Calculate entropy of attention distribution."""
        if not hasattr(attention_map, 'flatten'):
            return 0.0
        
        weights = attention_map.flatten() if hasattr(attention_map, 'flatten') else []
        if len(weights) == 0:
            return 0.0
        
        # Normalize weights
        total = weights.sum() if hasattr(weights, 'sum') else sum(weights)
        if total == 0:
            return 0.0
        
        normalized = weights / total if hasattr(weights, '__truediv__') else [w / total for w in weights]
        
        # Calculate entropy
        entropy = 0.0
        for p in normalized:
            if p > 0:
                entropy -= float(p) * (float(p) ** 0.5)  # Simplified entropy
        
        return entropy

    def _calculate_attention_focus(self, attention_map: Any) -> float:
        """Calculate attention focus (concentration) measure."""
        if not hasattr(attention_map, 'flatten'):
            return 0.0
        
        weights = attention_map.flatten() if hasattr(attention_map, 'flatten') else []
        if len(weights) == 0:
            return 0.0
        
        # Calculate coefficient of variation as focus measure
        mean_val = weights.mean() if hasattr(weights, 'mean') else (sum(weights) / len(weights))
        std_val = weights.std() if hasattr(weights, 'std') else 0.0
        
        if mean_val == 0:
            return 0.0
        
        return float(std_val / mean_val)

    def get_telemetry_report(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive telemetry report for interpretability."""
        if not self.telemetry_enabled or not self.telemetry:
            return None
        
        return self.telemetry.get_interpretability_report()

    def export_telemetry_data(self, format: str = 'json') -> Optional[str]:
        """Export telemetry data for external analysis."""
        if not self.telemetry_enabled or not self.telemetry:
            return None
        
        return self.telemetry.export_telemetry_data(format=format)

    def reset_telemetry(self):
        """Reset telemetry data collection."""
        if self.telemetry_enabled and self.telemetry:
            from .telemetry import reset_telemetry_collector
            reset_telemetry_collector()
            self.telemetry = get_telemetry_collector()
            self.processing_stats['telemetry_sessions'] = 0
            logger.info("ESC telemetry system reset")

    def _generate_processing_recommendations(self) -> List[str]:
        """Generate processing recommendations based on current state."""
        recommendations = []
        
        if self.emergency_mode:
            recommendations.append("CRITICAL: System in emergency mode - review recent inputs and reset")
        
        if self.processing_stats['constitutional_interventions'] > 100:
            recommendations.append("High constitutional intervention rate - consider stricter input filtering")
        
        if self.processing_stats['emergency_activations'] > 5:
            recommendations.append("Multiple emergency activations detected - review safety parameters")
        
        if len(self.discovered_tokens) > self.config.vocabulary_size * 0.1:
            recommendations.append("Large vocabulary expansion detected - consider rebalancing base vocabulary")
        
        avg_time = self.processing_stats['average_processing_time']
        if avg_time > 0.1:
            recommendations.append(f"High processing time ({avg_time:.3f}s) - consider performance optimization")
        
        if not recommendations:
            recommendations.append("ESC operating within normal parameters")
        
        return recommendations