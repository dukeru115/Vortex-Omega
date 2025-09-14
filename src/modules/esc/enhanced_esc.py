"""
Enhanced Echo-Semantic Converter (ESC) Module v2.2
====================================================

Advanced implementation with multi-scale echo and adaptive frequencies
based on NFCS v2.4.3 specifications.

Key Enhancements:
- Multi-scale temporal echo with different decay types
- Adaptive frequency modulation based on context
- Integration with Symbolic AI module
- Improved Kuramoto synchronization interface

Author: Team Omega
License: CC BY-NC 4.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from scipy.signal import hilbert, chirp
from scipy.integrate import odeint
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class MultiScaleEchoConfig:
    """Configuration for multi-scale echo system"""
    # Echo scales (in seconds)
    working_memory_scale: float = 1.0      # ~1 second
    episodic_memory_scale: float = 60.0    # ~1 minute
    semantic_memory_scale: float = 3600.0  # ~1 hour
    procedural_memory_scale: float = 86400.0  # ~1 day
    
    # Decay types
    working_decay_type: str = 'exponential'  # e^(-t/τ)
    episodic_decay_type: str = 'power'      # (t/τ)^(-α)
    semantic_decay_type: str = 'hyperbolic'  # sech²(t/τ)
    procedural_decay_type: str = 'persistent'  # δ(t - τ)
    
    # Echo weights
    working_weight: float = 0.4
    episodic_weight: float = 0.3
    semantic_weight: float = 0.2
    procedural_weight: float = 0.1


@dataclass
class AdaptiveFrequencyConfig:
    """Configuration for adaptive frequency modulation"""
    base_frequency_range: Tuple[float, float] = (0.1, 100.0)  # Hz
    context_sensitivity: float = 0.5
    frequency_drift_rate: float = 0.01
    phase_noise_level: float = 0.001
    coupling_strength: float = 0.1


class EnhancedESC:
    """
    Enhanced Echo-Semantic Converter with multi-scale echo and adaptive frequencies
    
    Implements the mathematical model from NFCS v2.4.3:
    - Si(t) = si * sin(2πfi(t - ti) + φi) * e^(-λ(t - ti)) * H(t - ti)
    - Multi-scale echo: E(t) = Σ γj ∫ S(τ) * e^(-μj(t-τ)^δj) dτ
    - Adaptive frequencies: fi(t) = fi^0 + Δfi * φ'(C(t))
    """
    
    def __init__(self, 
                 echo_config: Optional[MultiScaleEchoConfig] = None,
                 freq_config: Optional[AdaptiveFrequencyConfig] = None):
        """
        Initialize Enhanced ESC
        
        Args:
            echo_config: Multi-scale echo configuration
            freq_config: Adaptive frequency configuration
        """
        self.echo_config = echo_config or MultiScaleEchoConfig()
        self.freq_config = freq_config or AdaptiveFrequencyConfig()
        
        # Initialize echo buffers for different scales
        self.echo_buffers = {
            'working': deque(maxlen=1000),
            'episodic': deque(maxlen=5000),
            'semantic': deque(maxlen=10000),
            'procedural': deque(maxlen=20000)
        }
        
        # Frequency adaptation state
        self.oscillator_frequencies = {}  # Token ID -> frequency
        self.oscillator_phases = {}       # Token ID -> phase
        self.context_state = np.zeros(512)  # Global context vector
        
        # Kuramoto coupling matrix (for synchronization)
        self.coupling_matrix = None
        self.natural_frequencies = None
        
        # Performance metrics
        self.metrics = {
            'tokens_processed': 0,
            'echo_activations': 0,
            'frequency_adaptations': 0,
            'synchronization_events': 0
        }
        
        logger.info("Enhanced ESC initialized with multi-scale echo and adaptive frequencies")
    
    def process_token(self, 
                     token_id: int,
                     token_embedding: np.ndarray,
                     timestamp: float) -> Dict[str, Any]:
        """
        Process a single token with echo and frequency modulation
        
        Args:
            token_id: Unique token identifier
            token_embedding: Token embedding vector
            timestamp: Current timestamp
        
        Returns:
            Processing result with oscillatory signal
        """
        # Get or initialize oscillator for this token
        frequency = self._get_adaptive_frequency(token_id, timestamp)
        phase = self._get_oscillator_phase(token_id, timestamp)
        
        # Generate oscillatory signal (Si(t) from the paper)
        amplitude = np.linalg.norm(token_embedding)
        decay_rate = 0.1  # λ in the paper
        
        signal = amplitude * np.sin(2 * np.pi * frequency * timestamp + phase)
        signal *= np.exp(-decay_rate * timestamp)
        
        # Add multi-scale echo
        echo_contribution = self._compute_multi_scale_echo(
            signal, token_embedding, timestamp
        )
        
        # Combine signal with echo
        enhanced_signal = signal + echo_contribution
        
        # Update echo buffers
        self._update_echo_buffers(token_id, enhanced_signal, timestamp)
        
        # Update metrics
        self.metrics['tokens_processed'] += 1
        
        return {
            'token_id': token_id,
            'signal': enhanced_signal,
            'frequency': frequency,
            'phase': phase,
            'echo_strength': np.abs(echo_contribution),
            'timestamp': timestamp
        }
    
    def _get_adaptive_frequency(self, token_id: int, timestamp: float) -> float:
        """
        Get adaptive frequency for token oscillator
        
        Implements: fi(t) = fi^0 + Δfi * φ'(C(t))
        """
        # Get base frequency or initialize
        if token_id not in self.oscillator_frequencies:
            # Initialize with hash-based frequency in valid range
            base_freq = self.freq_config.base_frequency_range[0] + \
                       (hash(token_id) % 100) / 100.0 * \
                       (self.freq_config.base_frequency_range[1] - 
                        self.freq_config.base_frequency_range[0])
            self.oscillator_frequencies[token_id] = base_freq
        
        base_freq = self.oscillator_frequencies[token_id]
        
        # Compute context-dependent modulation
        context_modulation = self._compute_context_modulation()
        
        # Apply adaptive frequency change
        delta_freq = self.freq_config.frequency_drift_rate * context_modulation
        adapted_freq = base_freq + delta_freq
        
        # Constrain to valid range
        adapted_freq = np.clip(
            adapted_freq,
            self.freq_config.base_frequency_range[0],
            self.freq_config.base_frequency_range[1]
        )
        
        # Update stored frequency
        self.oscillator_frequencies[token_id] = adapted_freq
        self.metrics['frequency_adaptations'] += 1
        
        return adapted_freq
    
    def _get_oscillator_phase(self, token_id: int, timestamp: float) -> float:
        """Get or initialize oscillator phase"""
        if token_id not in self.oscillator_phases:
            # Initialize with random phase
            self.oscillator_phases[token_id] = np.random.uniform(0, 2 * np.pi)
        
        # Add phase noise
        phase = self.oscillator_phases[token_id]
        phase += np.random.normal(0, self.freq_config.phase_noise_level)
        
        # Wrap phase to [0, 2π]
        phase = phase % (2 * np.pi)
        
        self.oscillator_phases[token_id] = phase
        return phase
    
    def _compute_multi_scale_echo(self,
                                 signal: float,
                                 embedding: np.ndarray,
                                 timestamp: float) -> float:
        """
        Compute multi-scale echo contribution
        
        Implements: E(t) = Σ γj ∫ S(τ) * e^(-μj(t-τ)^δj) dτ
        """
        total_echo = 0.0
        
        # Working memory echo (exponential decay)
        working_echo = self._compute_echo_scale(
            'working',
            signal,
            timestamp,
            self.echo_config.working_memory_scale,
            self.echo_config.working_decay_type,
            self.echo_config.working_weight
        )
        total_echo += working_echo
        
        # Episodic memory echo (power law decay)
        episodic_echo = self._compute_echo_scale(
            'episodic',
            signal,
            timestamp,
            self.echo_config.episodic_memory_scale,
            self.echo_config.episodic_decay_type,
            self.echo_config.episodic_weight
        )
        total_echo += episodic_echo
        
        # Semantic memory echo (hyperbolic decay)
        semantic_echo = self._compute_echo_scale(
            'semantic',
            signal,
            timestamp,
            self.echo_config.semantic_memory_scale,
            self.echo_config.semantic_decay_type,
            self.echo_config.semantic_weight
        )
        total_echo += semantic_echo
        
        # Procedural memory echo (persistent)
        procedural_echo = self._compute_echo_scale(
            'procedural',
            signal,
            timestamp,
            self.echo_config.procedural_memory_scale,
            self.echo_config.procedural_decay_type,
            self.echo_config.procedural_weight
        )
        total_echo += procedural_echo
        
        self.metrics['echo_activations'] += 1
        
        return total_echo
    
    def _compute_echo_scale(self,
                          scale_name: str,
                          signal: float,
                          timestamp: float,
                          time_scale: float,
                          decay_type: str,
                          weight: float) -> float:
        """Compute echo for a specific temporal scale"""
        echo_buffer = self.echo_buffers[scale_name]
        
        if len(echo_buffer) == 0:
            return 0.0
        
        echo_sum = 0.0
        for past_signal, past_time in echo_buffer:
            time_diff = timestamp - past_time
            
            if time_diff <= 0:
                continue
            
            # Apply decay kernel
            if decay_type == 'exponential':
                kernel = np.exp(-time_diff / time_scale)
            elif decay_type == 'power':
                kernel = (time_scale / (time_diff + time_scale)) ** 2
            elif decay_type == 'hyperbolic':
                kernel = 1.0 / np.cosh(time_diff / time_scale) ** 2
            elif decay_type == 'persistent':
                kernel = 1.0 if abs(time_diff - time_scale) < 0.1 else 0.0
            else:
                kernel = 0.0
            
            echo_sum += past_signal * kernel
        
        return weight * echo_sum / max(len(echo_buffer), 1)
    
    def _update_echo_buffers(self, token_id: int, signal: float, timestamp: float):
        """Update echo buffers with new signal"""
        # Add to all buffers
        entry = (signal, timestamp)
        self.echo_buffers['working'].append(entry)
        self.echo_buffers['episodic'].append(entry)
        self.echo_buffers['semantic'].append(entry)
        self.echo_buffers['procedural'].append(entry)
    
    def _compute_context_modulation(self) -> float:
        """Compute context-dependent frequency modulation"""
        # Simple implementation - can be enhanced
        context_energy = np.linalg.norm(self.context_state)
        modulation = np.tanh(self.freq_config.context_sensitivity * context_energy)
        return modulation
    
    def update_context(self, new_context: np.ndarray):
        """Update global context state"""
        # Exponential moving average
        alpha = 0.1
        self.context_state = alpha * new_context + (1 - alpha) * self.context_state
    
    def synchronize_with_kuramoto(self, 
                                 kuramoto_phases: np.ndarray,
                                 kuramoto_frequencies: np.ndarray) -> Dict[str, Any]:
        """
        Synchronize ESC oscillators with Kuramoto model
        
        Args:
            kuramoto_phases: Current phases from Kuramoto model
            kuramoto_frequencies: Natural frequencies from Kuramoto
        
        Returns:
            Synchronization metrics
        """
        if self.coupling_matrix is None:
            n_oscillators = len(kuramoto_phases)
            self.coupling_matrix = np.random.randn(n_oscillators, n_oscillators) * 0.1
            np.fill_diagonal(self.coupling_matrix, 0)
        
        # Compute order parameter (coherence)
        order_param = np.abs(np.mean(np.exp(1j * kuramoto_phases)))
        
        # Adjust oscillator frequencies based on Kuramoto sync
        for i, (token_id, freq) in enumerate(self.oscillator_frequencies.items()):
            if i < len(kuramoto_frequencies):
                # Couple to Kuramoto frequency
                target_freq = kuramoto_frequencies[i]
                coupling_strength = self.freq_config.coupling_strength
                
                new_freq = freq + coupling_strength * (target_freq - freq)
                self.oscillator_frequencies[token_id] = new_freq
        
        self.metrics['synchronization_events'] += 1
        
        return {
            'order_parameter': order_param,
            'mean_frequency': np.mean(list(self.oscillator_frequencies.values())),
            'frequency_variance': np.var(list(self.oscillator_frequencies.values())),
            'coupling_strength': self.freq_config.coupling_strength
        }
    
    def integrate_with_symbolic(self, symbolic_clauses: List[Any]) -> np.ndarray:
        """
        Integrate symbolic clauses into semantic field
        
        Args:
            symbolic_clauses: List of SymClause objects from Symbolic AI
        
        Returns:
            Order parameter η(t) for NFCS
        """
        # Convert symbolic information to oscillatory patterns
        signals = []
        
        for clause in symbolic_clauses:
            # Extract semantic information
            clause_embedding = self._encode_clause(clause)
            
            # Generate oscillatory representation
            timestamp = time.time()
            result = self.process_token(
                token_id=hash(str(clause)),
                token_embedding=clause_embedding,
                timestamp=timestamp
            )
            
            signals.append(result['signal'])
        
        # Combine signals into order parameter
        if signals:
            order_parameter = np.mean(signals)
        else:
            order_parameter = 0.0
        
        # Normalize
        order_parameter = np.tanh(order_parameter)
        
        return order_parameter
    
    def _encode_clause(self, clause: Any) -> np.ndarray:
        """Encode symbolic clause as embedding vector"""
        # Simplified encoding - in production would use proper embeddings
        embedding = np.random.randn(512)
        
        # Modulate based on clause type
        if hasattr(clause, 'ctype'):
            type_hash = hash(str(clause.ctype))
            embedding *= 1.0 + 0.1 * np.sin(type_hash)
        
        return embedding
    
    def get_semantic_field_state(self) -> np.ndarray:
        """Get current semantic field state"""
        # Combine all oscillator states
        if not self.oscillator_frequencies:
            return np.zeros(100)
        
        frequencies = list(self.oscillator_frequencies.values())
        phases = list(self.oscillator_phases.values())
        
        # Create state vector
        state = np.concatenate([
            frequencies[:50],  # First 50 frequencies
            phases[:50]        # First 50 phases
        ])
        
        # Pad if necessary
        if len(state) < 100:
            state = np.pad(state, (0, 100 - len(state)))
        
        return state[:100]
    
    def reset_echo_buffers(self):
        """Clear all echo buffers"""
        for buffer in self.echo_buffers.values():
            buffer.clear()
        logger.info("Echo buffers reset")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()


# Utility class for echo analysis
class EchoAnalyzer:
    """Analyze echo patterns in ESC processing"""
    
    @staticmethod
    def compute_echo_spectrum(esc: EnhancedESC) -> Dict[str, np.ndarray]:
        """
        Compute frequency spectrum of echo patterns
        
        Args:
            esc: Enhanced ESC instance
        
        Returns:
            Spectrum for each echo scale
        """
        spectra = {}
        
        for scale_name, buffer in esc.echo_buffers.items():
            if len(buffer) > 0:
                signals = [s for s, t in buffer]
                # Compute FFT
                if len(signals) > 1:
                    spectrum = np.abs(np.fft.fft(signals))
                    spectra[scale_name] = spectrum
                else:
                    spectra[scale_name] = np.array([0.0])
        
        return spectra
    
    @staticmethod
    def measure_coherence(esc: EnhancedESC) -> float:
        """
        Measure coherence of oscillator ensemble
        
        Args:
            esc: Enhanced ESC instance
        
        Returns:
            Coherence value (0-1)
        """
        if not esc.oscillator_phases:
            return 0.0
        
        phases = np.array(list(esc.oscillator_phases.values()))
        
        # Compute Kuramoto order parameter
        order_param = np.abs(np.mean(np.exp(1j * phases)))
        
        return order_param


# Integration helper for NFCS
class ESCIntegrator:
    """Helper class for integrating ESC with NFCS components"""
    
    @staticmethod
    async def process_with_timeout(esc: EnhancedESC,
                                  tokens: List[int],
                                  embeddings: np.ndarray,
                                  timeout: float = 1.0) -> List[Dict]:
        """
        Process tokens with timeout
        
        Args:
            esc: Enhanced ESC instance
            tokens: List of token IDs
            embeddings: Token embeddings matrix
            timeout: Processing timeout in seconds
        
        Returns:
            List of processing results
        """
        results = []
        
        try:
            # Process with asyncio timeout
            async def process_all():
                timestamp = time.time()
                for i, token_id in enumerate(tokens):
                    result = esc.process_token(
                        token_id=token_id,
                        token_embedding=embeddings[i],
                        timestamp=timestamp + i * 0.001
                    )
                    results.append(result)
                return results
            
            results = await asyncio.wait_for(process_all(), timeout=timeout)
            
        except asyncio.TimeoutError:
            logger.warning(f"ESC processing timeout after {timeout}s")
        
        return results