"""
Semantic Field Coupler for ESC Module 2.1

Advanced semantic field coupling system with:
- Dynamic field-token interaction and coupling
- Multi-layer semantic representation
- Constitutional field dynamics and safety constraints
- Field resonance and interference patterns
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FieldConfig:
    """Configuration for semantic field coupling."""
    num_layers: int = 6
    field_dim: int = 512
    coupling_strength: float = 0.3
    decay_rate: float = 0.95
    resonance_threshold: float = 0.7
    max_field_amplitude: float = 10.0


class SemanticFieldCoupler:
    """
    Semantic Field Coupling System for ESC.
    
    Manages dynamic coupling between semantic fields and token embeddings
    with constitutional safety constraints and field dynamics.
    """
    
    def __init__(self, config: FieldConfig):
        """
        Initialize semantic field coupler.
        
        Args:
            config: Field configuration parameters
        """
        self.config = config
        
        # Initialize semantic field states
        self.field_states = np.zeros((config.num_layers, config.field_dim))
        
        # Coupling matrices between layers
        self.coupling_matrices = []
        for i in range(config.num_layers):
            matrix = np.random.randn(config.field_dim, config.field_dim) * 0.1
            self.coupling_matrices.append(matrix)
        
        # Field dynamics parameters
        self.layer_frequencies = np.linspace(0.1, 1.0, config.num_layers)
        self.layer_decay_rates = np.linspace(0.90, 0.99, config.num_layers)
        
        # Resonance tracking
        self.resonance_history = []
        self.field_energy_history = []
        
        logger.info(f"Semantic field coupler initialized with {config.num_layers} layers")
    
    def couple_tokens_to_fields(self, 
                              token_embeddings: List[np.ndarray],
                              coupling_strengths: Optional[List[float]] = None) -> np.ndarray:
        """
        Couple token embeddings to semantic fields.
        
        Args:
            token_embeddings: List of token embedding vectors
            coupling_strengths: Optional coupling strength per token
            
        Returns:
            Updated field states
        """
        if not token_embeddings:
            return self.field_states.copy()
        
        # Default coupling strengths
        if coupling_strengths is None:
            coupling_strengths = [self.config.coupling_strength] * len(token_embeddings)
        
        # Aggregate token influence on each layer
        for layer_idx in range(self.config.num_layers):
            layer_activation = np.zeros(self.config.field_dim)
            
            for token_emb, strength in zip(token_embeddings, coupling_strengths):
                # Project token through coupling matrix
                coupled_activation = np.dot(self.coupling_matrices[layer_idx], token_emb)
                layer_activation += strength * coupled_activation
            
            # Update field state with decay and new activation
            decay_rate = self.layer_decay_rates[layer_idx]
            self.field_states[layer_idx] = (
                decay_rate * self.field_states[layer_idx] + 
                (1 - decay_rate) * layer_activation
            )
            
            # Apply amplitude limits for stability
            amplitude = np.linalg.norm(self.field_states[layer_idx])
            if amplitude > self.config.max_field_amplitude:
                self.field_states[layer_idx] *= self.config.max_field_amplitude / amplitude
        
        return self.field_states.copy()
    
    def generate_field_feedback(self, token_positions: List[int]) -> List[np.ndarray]:
        """
        Generate feedback from fields to tokens.
        
        Args:
            token_positions: Positional indices of tokens
            
        Returns:
            List of feedback vectors for each token
        """
        feedback_vectors = []
        
        for pos in token_positions:
            token_feedback = np.zeros(self.config.field_dim)
            
            # Aggregate feedback from all layers
            for layer_idx in range(self.config.num_layers):
                # Apply position-dependent weighting
                position_weight = self._compute_position_weight(pos, layer_idx)
                
                # Feedback through transpose of coupling matrix
                layer_feedback = np.dot(
                    self.coupling_matrices[layer_idx].T,
                    self.field_states[layer_idx]
                )
                
                token_feedback += position_weight * layer_feedback
            
            feedback_vectors.append(token_feedback)
        
        return feedback_vectors
    
    def _compute_position_weight(self, position: int, layer_idx: int) -> float:
        """
        Compute position-dependent coupling weight.
        
        Args:
            position: Token position in sequence
            layer_idx: Layer index
            
        Returns:
            Position weight for coupling
        """
        # Different layers have different position sensitivities
        layer_period = 2.0 ** (layer_idx + 1)  # Exponential scaling
        
        # Sinusoidal position encoding
        weight = 0.5 * (1.0 + np.cos(2 * np.pi * position / layer_period))
        
        return float(weight)
    
    def detect_field_resonances(self) -> Dict[str, Any]:
        """
        Detect resonance patterns between field layers.
        
        Returns:
            Dictionary with resonance analysis
        """
        resonances = {}
        
        # Calculate cross-correlations between layers
        for i in range(self.config.num_layers):
            for j in range(i + 1, self.config.num_layers):
                # Normalized cross-correlation
                field_i = self.field_states[i]
                field_j = self.field_states[j]
                
                correlation = np.dot(field_i, field_j) / (
                    np.linalg.norm(field_i) * np.linalg.norm(field_j) + 1e-8
                )
                
                resonances[f'layer_{i}_layer_{j}'] = float(correlation)
        
        # Overall resonance strength
        resonance_values = list(resonances.values())
        max_resonance = max(resonance_values) if resonance_values else 0.0
        mean_resonance = np.mean(resonance_values) if resonance_values else 0.0
        
        # Track resonance history
        self.resonance_history.append(max_resonance)
        if len(self.resonance_history) > 100:
            self.resonance_history = self.resonance_history[-100:]
        
        return {
            'pairwise_resonances': resonances,
            'max_resonance': max_resonance,
            'mean_resonance': mean_resonance,
            'resonance_trend': self._compute_resonance_trend(),
            'high_resonance_detected': max_resonance > self.config.resonance_threshold
        }
    
    def _compute_resonance_trend(self) -> float:
        """
        Compute trend in resonance strength.
        
        Returns:
            Resonance trend (positive = increasing)
        """
        if len(self.resonance_history) < 5:
            return 0.0
        
        # Simple linear trend over recent history
        recent = self.resonance_history[-10:]
        x = np.arange(len(recent))
        
        if len(recent) >= 2:
            coeffs = np.polyfit(x, recent, 1)
            trend = coeffs[0]
        else:
            trend = 0.0
        
        return float(trend)
    
    def calculate_field_energy(self) -> Dict[str, float]:
        """
        Calculate energy in each semantic field layer.
        
        Returns:
            Dictionary with energy analysis
        """
        layer_energies = {}
        total_energy = 0.0
        
        for layer_idx in range(self.config.num_layers):
            energy = np.sum(self.field_states[layer_idx]**2)
            layer_energies[f'layer_{layer_idx}'] = float(energy)
            total_energy += energy
        
        # Track energy history
        self.field_energy_history.append(total_energy)
        if len(self.field_energy_history) > 100:
            self.field_energy_history = self.field_energy_history[-100:]
        
        # Energy distribution analysis
        if total_energy > 1e-8:
            energy_distribution = {
                f'layer_{i}_fraction': layer_energies[f'layer_{i}'] / total_energy
                for i in range(self.config.num_layers)
            }
        else:
            energy_distribution = {
                f'layer_{i}_fraction': 0.0
                for i in range(self.config.num_layers)
            }
        
        return {
            'layer_energies': layer_energies,
            'total_energy': total_energy,
            'energy_distribution': energy_distribution,
            'energy_trend': self._compute_energy_trend(),
            'max_layer_energy': max(layer_energies.values()) if layer_energies else 0.0
        }
    
    def _compute_energy_trend(self) -> float:
        """
        Compute trend in total field energy.
        
        Returns:
            Energy trend (positive = increasing)
        """
        if len(self.field_energy_history) < 5:
            return 0.0
        
        # Linear trend over recent history
        recent = self.field_energy_history[-10:]
        x = np.arange(len(recent))
        
        if len(recent) >= 2:
            coeffs = np.polyfit(x, recent, 1)
            trend = coeffs[0]
        else:
            trend = 0.0
        
        return float(trend)
    
    def apply_constitutional_constraints(self) -> Dict[str, Any]:
        """
        Apply constitutional constraints to field dynamics.
        
        Returns:
            Dictionary with constraint application results
        """
        violations = []
        corrections_applied = 0
        
        # Energy constraints
        energy_analysis = self.calculate_field_energy()
        if energy_analysis['total_energy'] > 100.0:  # Energy limit
            # Scale down all fields proportionally
            scale_factor = np.sqrt(100.0 / energy_analysis['total_energy'])
            self.field_states *= scale_factor
            violations.append("Excessive field energy - scaled down")
            corrections_applied += 1
        
        # Amplitude constraints per layer
        for layer_idx in range(self.config.num_layers):
            amplitude = np.linalg.norm(self.field_states[layer_idx])
            if amplitude > self.config.max_field_amplitude:
                self.field_states[layer_idx] *= self.config.max_field_amplitude / amplitude
                violations.append(f"Layer {layer_idx} amplitude exceeded limit")
                corrections_applied += 1
        
        # Resonance constraints
        resonance_analysis = self.detect_field_resonances()
        if resonance_analysis['max_resonance'] > 0.95:  # Avoid perfect resonance
            # Add small noise to break perfect resonance
            noise_strength = 0.01
            for layer_idx in range(self.config.num_layers):
                noise = np.random.randn(self.config.field_dim) * noise_strength
                self.field_states[layer_idx] += noise
            violations.append("Excessive resonance - noise added")
            corrections_applied += 1
        
        return {
            'violations_detected': len(violations),
            'violation_descriptions': violations,
            'corrections_applied': corrections_applied,
            'field_states_modified': corrections_applied > 0
        }
    
    def reset_fields(self, preserve_structure: bool = True):
        """
        Reset field states to initial conditions.
        
        Args:
            preserve_structure: Whether to preserve coupling structure
        """
        if preserve_structure:
            # Reset states but keep coupling matrices
            self.field_states = np.zeros((self.config.num_layers, self.config.field_dim))
        else:
            # Full reset including coupling matrices
            self.field_states = np.zeros((self.config.num_layers, self.config.field_dim))
            
            self.coupling_matrices = []
            for i in range(self.config.num_layers):
                matrix = np.random.randn(self.config.field_dim, self.config.field_dim) * 0.1
                self.coupling_matrices.append(matrix)
        
        # Clear histories
        self.resonance_history.clear()
        self.field_energy_history.clear()
        
        logger.info(f"Semantic fields reset (preserve_structure={preserve_structure})")
    
    def get_field_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of semantic field system.
        
        Returns:
            Dictionary with field status information
        """
        energy_analysis = self.calculate_field_energy()
        resonance_analysis = self.detect_field_resonances()
        
        return {
            'configuration': {
                'num_layers': self.config.num_layers,
                'field_dim': self.config.field_dim,
                'coupling_strength': self.config.coupling_strength,
                'max_field_amplitude': self.config.max_field_amplitude
            },
            'current_state': {
                'field_norms': [float(np.linalg.norm(self.field_states[i])) 
                              for i in range(self.config.num_layers)],
                'total_energy': energy_analysis['total_energy'],
                'max_resonance': resonance_analysis['max_resonance'],
                'mean_resonance': resonance_analysis['mean_resonance']
            },
            'dynamics': {
                'energy_trend': energy_analysis['energy_trend'],
                'resonance_trend': resonance_analysis['resonance_trend'],
                'high_resonance_active': resonance_analysis['high_resonance_detected']
            },
            'history_lengths': {
                'resonance_history': len(self.resonance_history),
                'energy_history': len(self.field_energy_history)
            }
        }