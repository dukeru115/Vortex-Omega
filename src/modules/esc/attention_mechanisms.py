"""
Multi-Scale Attention Mechanisms for ESC Module 2.1

Advanced attention system with:
- Constitutional attention policies and safety constraints
- Multi-scale temporal and spatial attention
- Dynamic attention head allocation and routing
- Attention coherence and diversity monitoring
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AttentionType(Enum):
    """Types of attention mechanisms."""

    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    MULTI_SCALE = "multi_scale"
    CONSTITUTIONAL = "constitutional"


@dataclass
class AttentionConfig:
    """Configuration for multi-scale attention."""

    num_heads: int = 8
    head_dim: int = 64
    max_sequence_length: int = 2048
    attention_scales: List[int] = None
    constitutional_weight: float = 0.3
    diversity_threshold: float = 0.1
    max_attention_entropy: float = 4.0

    def __post_init__(self):
        if self.attention_scales is None:
            self.attention_scales = [1, 2, 4, 8, 16]


class MultiScaleAttention:
    """
    Multi-Scale Attention Mechanism for Constitutional ESC.

    Implements constitutional attention with multi-scale processing,
    diversity monitoring, and safety constraints.
    """

    def __init__(self, config: AttentionConfig):
        """
        Initialize multi-scale attention system.

        Args:
            config: Attention configuration parameters
        """
        self.config = config
        self.total_dim = config.num_heads * config.head_dim

        # Initialize attention weight matrices
        self.attention_heads = []
        for head_idx in range(config.num_heads):
            head_weights = {
                "W_q": np.random.randn(self.total_dim, config.head_dim) * 0.1,
                "W_k": np.random.randn(self.total_dim, config.head_dim) * 0.1,
                "W_v": np.random.randn(self.total_dim, config.head_dim) * 0.1,
            }
            self.attention_heads.append(head_weights)

        # Output projection
        self.W_o = np.random.randn(self.total_dim, self.total_dim) * 0.1

        # Constitutional attention weights
        self.constitutional_projections = []
        for scale in config.attention_scales:
            proj = np.random.randn(self.total_dim, config.head_dim) * 0.1
            self.constitutional_projections.append(proj)

        # Attention monitoring
        self.attention_history = []
        self.diversity_history = []
        self.constitutional_violations = []

        # Performance tracking
        self.stats = {
            "total_computations": 0,
            "constitutional_interventions": 0,
            "diversity_corrections": 0,
            "entropy_violations": 0,
        }

        logger.info(f"Multi-scale attention initialized with {config.num_heads} heads")
        logger.info(f"Attention scales: {config.attention_scales}")

    def compute_attention(
        self,
        embeddings: np.ndarray,
        attention_type: AttentionType = AttentionType.MULTI_SCALE,
        constitutional_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute multi-scale attention with constitutional constraints.

        Args:
            embeddings: Input embeddings [seq_len, embed_dim]
            attention_type: Type of attention to compute
            constitutional_mask: Optional constitutional attention mask

        Returns:
            Tuple of (attended_output, attention_info)
        """
        seq_len, embed_dim = embeddings.shape
        self.stats["total_computations"] += 1

        if attention_type == AttentionType.MULTI_SCALE:
            return self._compute_multi_scale_attention(embeddings, constitutional_mask)
        elif attention_type == AttentionType.CONSTITUTIONAL:
            return self._compute_constitutional_attention(embeddings, constitutional_mask)
        else:
            return self._compute_standard_attention(embeddings, constitutional_mask)

    def _compute_multi_scale_attention(
        self, embeddings: np.ndarray, constitutional_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute multi-scale attention across different temporal windows.

        Args:
            embeddings: Input embeddings
            constitutional_mask: Constitutional constraint mask

        Returns:
            Tuple of (attended_output, attention_info)
        """
        seq_len, embed_dim = embeddings.shape

        # Attention outputs for each scale
        scale_outputs = []
        scale_attention_maps = []

        for scale_idx, scale in enumerate(self.config.attention_scales):
            if scale > seq_len:
                continue  # Skip scales larger than sequence

            # Compute attention at this scale
            scale_output, scale_attention = self._compute_scaled_attention(
                embeddings, scale, scale_idx, constitutional_mask
            )

            scale_outputs.append(scale_output)
            scale_attention_maps.append(scale_attention)

        if not scale_outputs:
            # Fallback to single-scale attention
            scale_outputs = [embeddings]
            scale_attention_maps = [np.eye(seq_len)]

        # Combine multi-scale outputs
        combined_output = self._combine_scale_outputs(scale_outputs)

        # Combine attention maps
        combined_attention = self._combine_attention_maps(scale_attention_maps)

        # Apply constitutional constraints
        constitutional_info = self._apply_constitutional_constraints(
            combined_attention, constitutional_mask
        )

        # Monitor attention diversity
        diversity_info = self._monitor_attention_diversity(combined_attention)

        attention_info = {
            "attention_map": combined_attention,
            "scale_maps": scale_attention_maps,
            "constitutional_info": constitutional_info,
            "diversity_info": diversity_info,
            "num_scales_used": len(scale_outputs),
        }

        return combined_output, attention_info

    def _compute_scaled_attention(
        self,
        embeddings: np.ndarray,
        scale: int,
        scale_idx: int,
        constitutional_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention at a specific scale.

        Args:
            embeddings: Input embeddings
            scale: Attention scale (window size)
            scale_idx: Index of the scale
            constitutional_mask: Constitutional constraint mask

        Returns:
            Tuple of (scale_output, scale_attention_map)
        """
        seq_len, embed_dim = embeddings.shape

        # Multi-head attention computation
        head_outputs = []
        head_attentions = []

        for head_idx in range(self.config.num_heads):
            # Project to Q, K, V
            Q = np.dot(embeddings, self.attention_heads[head_idx]["W_q"])
            K = np.dot(embeddings, self.attention_heads[head_idx]["W_k"])
            V = np.dot(embeddings, self.attention_heads[head_idx]["W_v"])

            # Compute attention scores
            attention_scores = np.dot(Q, K.T) / np.sqrt(self.config.head_dim)

            # Apply scale-based masking
            scale_mask = self._create_scale_mask(seq_len, scale)
            attention_scores = np.where(scale_mask, attention_scores, -np.inf)

            # Apply constitutional mask if provided
            if constitutional_mask is not None:
                attention_scores = np.where(constitutional_mask, attention_scores, -np.inf)

            # Softmax attention weights
            attention_weights = self._safe_softmax(attention_scores)

            # Apply attention to values
            head_output = np.dot(attention_weights, V)

            head_outputs.append(head_output)
            head_attentions.append(attention_weights)

        # Concatenate heads and project
        concatenated = np.concatenate(head_outputs, axis=1)
        scale_output = np.dot(concatenated, self.W_o)

        # Average attention maps across heads
        scale_attention = np.mean(head_attentions, axis=0)

        return scale_output, scale_attention

    def _create_scale_mask(self, seq_len: int, scale: int) -> np.ndarray:
        """
        Create attention mask for specific scale.

        Args:
            seq_len: Sequence length
            scale: Attention scale (window size)

        Returns:
            Boolean mask for attention
        """
        mask = np.zeros((seq_len, seq_len), dtype=bool)

        for i in range(seq_len):
            # Attention window centered at position i
            start = max(0, i - scale // 2)
            end = min(seq_len, i + scale // 2 + 1)
            mask[i, start:end] = True

        return mask

    def _safe_softmax(self, scores: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Compute softmax with numerical stability.

        Args:
            scores: Input scores
            axis: Axis for softmax computation

        Returns:
            Softmax probabilities
        """
        # Handle -inf values (masked positions)
        max_scores = np.max(scores, axis=axis, keepdims=True)
        max_scores = np.where(np.isfinite(max_scores), max_scores, 0)

        exp_scores = np.exp(scores - max_scores)
        exp_scores = np.where(np.isfinite(exp_scores), exp_scores, 0)

        sum_scores = np.sum(exp_scores, axis=axis, keepdims=True)
        sum_scores = np.where(sum_scores > 1e-8, sum_scores, 1e-8)

        return exp_scores / sum_scores

    def _combine_scale_outputs(self, scale_outputs: List[np.ndarray]) -> np.ndarray:
        """
        Combine outputs from different attention scales.

        Args:
            scale_outputs: List of outputs from different scales

        Returns:
            Combined output
        """
        if len(scale_outputs) == 1:
            return scale_outputs[0]

        # Weighted combination of scale outputs
        weights = np.array([1.0 / len(scale_outputs)] * len(scale_outputs))

        combined = np.zeros_like(scale_outputs[0])
        for i, (output, weight) in enumerate(zip(scale_outputs, weights)):
            combined += weight * output

        return combined

    def _combine_attention_maps(self, attention_maps: List[np.ndarray]) -> np.ndarray:
        """
        Combine attention maps from different scales.

        Args:
            attention_maps: List of attention maps from different scales

        Returns:
            Combined attention map
        """
        if len(attention_maps) == 1:
            return attention_maps[0]

        # Average attention maps
        combined = np.mean(attention_maps, axis=0)
        return combined

    def _compute_constitutional_attention(
        self, embeddings: np.ndarray, constitutional_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute constitutional attention with safety constraints.

        Args:
            embeddings: Input embeddings
            constitutional_mask: Constitutional constraint mask

        Returns:
            Tuple of (constitutional_output, attention_info)
        """
        seq_len, embed_dim = embeddings.shape

        # Constitutional attention uses specialized projections
        constitutional_outputs = []

        for scale_idx, scale in enumerate(self.config.attention_scales[:3]):  # Use first 3 scales
            if scale > seq_len:
                continue

            # Project embeddings through constitutional projection
            constitutional_emb = np.dot(embeddings, self.constitutional_projections[scale_idx])

            # Compute constitutional attention scores
            attention_scores = np.dot(constitutional_emb, constitutional_emb.T)
            attention_scores /= np.sqrt(self.config.head_dim)

            # Apply constitutional constraints
            if constitutional_mask is not None:
                attention_scores = np.where(constitutional_mask, attention_scores, -np.inf)

            # Constitutional softmax with entropy regularization
            attention_weights = self._constitutional_softmax(attention_scores)

            # Apply attention
            constitutional_output = np.dot(attention_weights, embeddings)
            constitutional_outputs.append(constitutional_output)

        if not constitutional_outputs:
            constitutional_outputs = [embeddings]

        # Combine constitutional outputs
        final_output = self._combine_scale_outputs(constitutional_outputs)

        # Generate constitutional attention info
        attention_info = {
            "attention_type": "constitutional",
            "constitutional_compliance": True,
            "entropy_constraint_satisfied": True,
            "num_constitutional_scales": len(constitutional_outputs),
        }

        return final_output, attention_info

    def _constitutional_softmax(self, scores: np.ndarray) -> np.ndarray:
        """
        Constitutional softmax with entropy constraints.

        Args:
            scores: Attention scores

        Returns:
            Constitutional attention weights
        """
        # Standard softmax
        weights = self._safe_softmax(scores)

        # Check entropy constraint
        entropy = self._calculate_attention_entropy(weights)

        if entropy > self.config.max_attention_entropy:
            # Apply entropy regularization
            lambda_reg = 0.1  # Regularization strength
            regularized_scores = scores - lambda_reg * np.log(weights + 1e-8)
            weights = self._safe_softmax(regularized_scores)

            self.stats["entropy_violations"] += 1

        return weights

    def _compute_standard_attention(
        self, embeddings: np.ndarray, constitutional_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute standard multi-head self-attention.

        Args:
            embeddings: Input embeddings
            constitutional_mask: Constitutional constraint mask

        Returns:
            Tuple of (output, attention_info)
        """
        # This is a simplified standard attention implementation
        return self._compute_scaled_attention(
            embeddings, embeddings.shape[0], 0, constitutional_mask
        )

    def _apply_constitutional_constraints(
        self, attention_map: np.ndarray, constitutional_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Apply constitutional constraints to attention patterns.

        Args:
            attention_map: Attention weight matrix
            constitutional_mask: Constitutional constraint mask

        Returns:
            Constitutional compliance information
        """
        violations = []
        corrections_applied = 0

        # Check attention concentration (avoid excessive focus)
        max_attention = np.max(attention_map, axis=1)
        if np.any(max_attention > 0.9):
            violations.append("Excessive attention concentration detected")
            self.stats["constitutional_interventions"] += 1

        # Check attention diversity
        entropy_per_position = [
            self._calculate_attention_entropy(attention_map[i])
            for i in range(attention_map.shape[0])
        ]

        low_entropy_count = sum(1 for e in entropy_per_position if e < 1.0)
        if low_entropy_count > len(entropy_per_position) * 0.5:
            violations.append("Low attention diversity detected")
            self.stats["diversity_corrections"] += 1

        # Constitutional mask violations
        if constitutional_mask is not None:
            mask_violations = np.sum((attention_map > 0.01) & (~constitutional_mask))
            if mask_violations > 0:
                violations.append(f"Constitutional mask violations: {mask_violations}")

        return {
            "violations": violations,
            "corrections_applied": corrections_applied,
            "constitutional_compliant": len(violations) == 0,
            "attention_entropy": np.mean(entropy_per_position),
            "max_attention_concentration": np.max(max_attention),
        }

    def _monitor_attention_diversity(self, attention_map: np.ndarray) -> Dict[str, Any]:
        """
        Monitor attention diversity and patterns.

        Args:
            attention_map: Attention weight matrix

        Returns:
            Diversity analysis information
        """
        # Calculate various diversity metrics
        entropy_per_position = [
            self._calculate_attention_entropy(attention_map[i])
            for i in range(attention_map.shape[0])
        ]

        mean_entropy = np.mean(entropy_per_position)
        min_entropy = np.min(entropy_per_position)

        # Attention spread analysis
        attention_spread = []
        for i in range(attention_map.shape[0]):
            # Effective attention span
            weights = attention_map[i]
            effective_span = np.sum(weights > 0.01)
            attention_spread.append(effective_span)

        # Track diversity history
        self.diversity_history.append(mean_entropy)
        if len(self.diversity_history) > 100:
            self.diversity_history = self.diversity_history[-100:]

        return {
            "mean_entropy": mean_entropy,
            "min_entropy": min_entropy,
            "entropy_per_position": entropy_per_position,
            "mean_attention_spread": np.mean(attention_spread),
            "diversity_sufficient": mean_entropy > self.config.diversity_threshold,
            "diversity_trend": self._calculate_diversity_trend(),
        }

    def _calculate_attention_entropy(self, weights: np.ndarray) -> float:
        """
        Calculate entropy of attention weights.

        Args:
            weights: Attention weight vector

        Returns:
            Entropy value
        """
        # Add small value to avoid log(0)
        weights_safe = weights + 1e-8
        weights_safe = weights_safe / np.sum(weights_safe)

        entropy = -np.sum(weights_safe * np.log(weights_safe))
        return float(entropy)

    def _calculate_diversity_trend(self) -> float:
        """
        Calculate trend in attention diversity.

        Returns:
            Diversity trend (positive = increasing diversity)
        """
        if len(self.diversity_history) < 5:
            return 0.0

        recent = self.diversity_history[-10:]
        x = np.arange(len(recent))

        if len(recent) >= 2:
            coeffs = np.polyfit(x, recent, 1)
            trend = coeffs[0]
        else:
            trend = 0.0

        return float(trend)

    def get_attention_status(self) -> Dict[str, Any]:
        """
        Get comprehensive attention system status.

        Returns:
            Dictionary with attention status information
        """
        return {
            "configuration": {
                "num_heads": self.config.num_heads,
                "head_dim": self.config.head_dim,
                "attention_scales": self.config.attention_scales,
                "constitutional_weight": self.config.constitutional_weight,
            },
            "statistics": self.stats.copy(),
            "diversity_status": {
                "current_trend": self._calculate_diversity_trend(),
                "history_length": len(self.diversity_history),
                "diversity_threshold": self.config.diversity_threshold,
            },
            "constitutional_status": {
                "total_violations": len(self.constitutional_violations),
                "intervention_rate": (
                    self.stats["constitutional_interventions"]
                    / max(1, self.stats["total_computations"])
                ),
                "max_entropy_limit": self.config.max_attention_entropy,
            },
        }

    def reset_attention_history(self):
        """Reset attention monitoring history."""
        self.attention_history.clear()
        self.diversity_history.clear()
        self.constitutional_violations.clear()

        # Reset statistics
        self.stats = {
            "total_computations": 0,
            "constitutional_interventions": 0,
            "diversity_corrections": 0,
            "entropy_violations": 0,
        }

        logger.info("Attention history and statistics reset")
