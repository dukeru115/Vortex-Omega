"""
Symbolic-Neural Bridge Module
============================

Implementation of the boundary interface between discrete symbolic space (S) and 
continuous neural field dynamics (φ) as specified in NFCS v2.4.3.

This module implements Equation 25 from the NFCS paper:
φ_symbolic(x,t) = Σ w_s(t) · Ψ_s(x) · δ_logic[s]
s∈S

Where:
- S is the discrete symbolic space
- φ is the continuous neural field
- Ψ_s are basis functions for symbol s
- w_s are dynamic weights
- δ_logic is the logical consistency function

Author: Team Omega (GenSpark AI Implementation)
License: CC BY-NC 4.0
Date: 2025-09-14
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from .models import (
    SymField, SymClause, VerificationReport,
    ClauseType, VerificationStatus, Term, Expression
)

logger = logging.getLogger(__name__)


@dataclass
class BasisFunction:
    """Basis function for symbolic representation in neural field"""
    symbol: str
    spatial_pattern: Callable[[np.ndarray], np.ndarray]  # Ψ_s(x)
    frequency: float = 1.0  # Characteristic frequency
    amplitude: float = 1.0  # Base amplitude
    phase: float = 0.0  # Phase offset
    
    def __call__(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Evaluate basis function at spatial positions x and time t"""
        spatial = self.spatial_pattern(x)
        temporal = np.cos(2 * np.pi * self.frequency * t + self.phase)
        return self.amplitude * spatial * temporal


@dataclass 
class SymbolicWeight:
    """Dynamic weight for symbolic element in neural field"""
    symbol: str
    value: float = 0.0
    gradient: float = 0.0
    momentum: float = 0.0
    consistency_score: float = 1.0  # δ_logic[s] component
    
    def update(self, delta: float, learning_rate: float = 0.01, momentum_coeff: float = 0.9):
        """Update weight with gradient-based dynamics"""
        self.momentum = momentum_coeff * self.momentum + (1 - momentum_coeff) * delta
        self.value += learning_rate * self.momentum
        self.gradient = delta


class SymbolicNeuralBridge(nn.Module):
    """
    Main bridge class implementing S ↔ φ transformations
    
    Core functions:
    1. Symbolization: φ → S (extract symbolic content from field)
    2. Fieldization: S → φ (embed symbols into neural field)
    3. Verification: S ↔ φ (check consistency between representations)
    """
    
    def __init__(self, 
                 field_dims: Tuple[int, ...] = (64, 64),
                 max_symbols: int = 256,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Symbolic-Neural Bridge
        
        Args:
            field_dims: Spatial dimensions of neural field (H, W) or (H, W, D)
            max_symbols: Maximum number of symbolic elements
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config or self._default_config()
        self.field_dims = field_dims
        self.max_symbols = max_symbols
        
        # Spatial grid for field evaluation
        self.register_buffer('spatial_grid', self._create_spatial_grid())
        
        # Symbolic registry
        self.symbol_registry: Dict[str, int] = {}
        self.basis_functions: Dict[str, BasisFunction] = {}
        self.symbolic_weights: Dict[str, SymbolicWeight] = {}
        
        # Neural field state
        self.register_buffer('field_state', torch.zeros(*field_dims, dtype=torch.complex64))
        
        # Transformation matrices
        self.symbol_embedding = nn.Embedding(max_symbols, self.config['embedding_dim'])
        self.field_projector = nn.Linear(self.config['embedding_dim'], np.prod(field_dims))
        
        # Consistency verification networks
        self.consistency_net = nn.Sequential(
            nn.Linear(self.config['embedding_dim'] + np.prod(field_dims), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Performance metrics
        self.metrics = {
            'symbolization_time': 0.0,
            'fieldization_time': 0.0,
            'verification_accuracy': 0.0,
            'consistency_score': 1.0,
            'total_transformations': 0
        }
        
        logger.info(f"Symbolic-Neural Bridge initialized: {field_dims} field, {max_symbols} max symbols")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'embedding_dim': 64,
            'learning_rate': 0.001,
            'consistency_threshold': 0.8,
            'basis_type': 'gaussian',  # gaussian, sine, wavelet
            'temporal_coupling': True,
            'spatial_coupling': True,
            'max_iterations': 100,
            'convergence_tol': 1e-6
        }
    
    def _create_spatial_grid(self) -> torch.Tensor:
        """Create spatial coordinate grid for field evaluation"""
        if len(self.field_dims) == 2:
            h, w = self.field_dims
            y = torch.linspace(-1, 1, h)
            x = torch.linspace(-1, 1, w)
            Y, X = torch.meshgrid(y, x, indexing='ij')
            return torch.stack([X, Y], dim=-1)  # Shape: (H, W, 2)
        elif len(self.field_dims) == 3:
            h, w, d = self.field_dims
            z = torch.linspace(-1, 1, h)
            y = torch.linspace(-1, 1, w)
            x = torch.linspace(-1, 1, d)
            Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
            return torch.stack([X, Y, Z], dim=-1)  # Shape: (H, W, D, 3)
        else:
            raise ValueError(f"Unsupported field dimensions: {self.field_dims}")
    
    def register_symbol(self, symbol: str, clause: SymClause) -> int:
        """Register new symbol in the bridge"""
        if symbol not in self.symbol_registry:
            if len(self.symbol_registry) >= self.max_symbols:
                logger.warning(f"Symbol registry full, cannot register '{symbol}'")
                return -1
            
            # Assign new index
            idx = len(self.symbol_registry)
            self.symbol_registry[symbol] = idx
            
            # Create basis function
            basis_func = self._create_basis_function(symbol, clause)
            self.basis_functions[symbol] = basis_func
            
            # Initialize weight
            self.symbolic_weights[symbol] = SymbolicWeight(symbol=symbol, value=0.1)
            
            logger.debug(f"Registered symbol '{symbol}' with index {idx}")
            return idx
        
        return self.symbol_registry[symbol]
    
    def _create_basis_function(self, symbol: str, clause: SymClause) -> BasisFunction:
        """Create basis function for symbol based on clause context"""
        
        def gaussian_pattern(x: np.ndarray) -> np.ndarray:
            """Gaussian spatial pattern"""
            # Random center and width for now
            # In production, would use semantic positioning
            center = np.random.randn(x.shape[-1]) * 0.3
            width = 0.2
            if len(x.shape) == 3:  # 2D field
                dist = np.linalg.norm(x - center, axis=-1)
            else:  # 3D field
                dist = np.linalg.norm(x - center, axis=-1)
            return np.exp(-dist**2 / (2 * width**2))
        
        def sine_pattern(x: np.ndarray) -> np.ndarray:
            """Sinusoidal spatial pattern"""
            k = np.random.randn(x.shape[-1]) * 2  # Wave vector
            if len(x.shape) == 3:  # 2D field
                phase = np.sum(x * k, axis=-1)
            else:  # 3D field
                phase = np.sum(x * k, axis=-1)
            return np.sin(phase)
        
        # Choose pattern type based on clause
        if clause.ctype == ClauseType.EQUATION:
            pattern = gaussian_pattern
            freq = 1.0
        elif clause.ctype == ClauseType.CONSTRAINT:
            pattern = sine_pattern
            freq = 2.0
        else:
            pattern = gaussian_pattern
            freq = 0.5
        
        return BasisFunction(
            symbol=symbol,
            spatial_pattern=pattern,
            frequency=freq,
            amplitude=1.0
        )
    
    async def symbolize(self, field: torch.Tensor) -> List[SymClause]:
        """
        Symbolization: φ → S
        Extract symbolic content from neural field state
        
        Args:
            field: Neural field tensor of shape (*field_dims,)
            
        Returns:
            List of extracted symbolic clauses
        """
        start_time = time.time()
        
        try:
            # Ensure field is correct shape
            if field.shape != self.field_dims:
                field = field.reshape(self.field_dims)
            
            # Convert to complex if needed
            if not torch.is_complex(field):
                field = field.to(torch.complex64)
            
            clauses = []
            
            # Extract symbols based on field patterns
            field_np = field.detach().cpu().numpy()
            
            # Analyze dominant frequencies and patterns
            fft_field = np.fft.fftn(field_np)
            dominant_modes = self._find_dominant_modes(fft_field)
            
            # Convert dominant modes to symbolic clauses
            for i, (k_vec, amplitude, phase) in enumerate(dominant_modes[:10]):  # Top 10 modes
                symbol = f"mode_{i}"
                
                # Create symbolic clause from mode
                clause = SymClause(
                    cid=f"extracted_{symbol}",
                    ctype=ClauseType.EQUATION,
                    meta={
                        'wave_vector': k_vec.tolist(),
                        'amplitude': float(amplitude),
                        'phase': float(phase),
                        'extraction_method': 'fft_analysis'
                    }
                )
                
                clauses.append(clause)
                
                # Register symbol if not exists
                self.register_symbol(symbol, clause)
            
            # Update metrics
            self.metrics['symbolization_time'] = time.time() - start_time
            self.metrics['total_transformations'] += 1
            
            logger.debug(f"Symbolized field into {len(clauses)} clauses")
            return clauses
            
        except Exception as e:
            logger.error(f"Symbolization error: {e}")
            return []
    
    async def fieldize(self, symbols: List[SymClause]) -> torch.Tensor:
        """
        Fieldization: S → φ
        Embed symbolic clauses into neural field
        
        Args:
            symbols: List of symbolic clauses to embed
            
        Returns:
            Neural field tensor with embedded symbols
        """
        start_time = time.time()
        
        try:
            # Initialize field
            field = torch.zeros(*self.field_dims, dtype=torch.complex64, device=self.spatial_grid.device)
            
            # Current time (could be passed as parameter)
            t = 0.0
            
            # Implement Equation 25: φ_symbolic(x,t) = Σ w_s(t) · Ψ_s(x) · δ_logic[s]
            for clause in symbols:
                symbol = self._extract_symbol_from_clause(clause)
                
                # Register symbol if needed
                self.register_symbol(symbol, clause)
                
                # Get weight and basis function
                if symbol in self.symbolic_weights and symbol in self.basis_functions:
                    weight = self.symbolic_weights[symbol]
                    basis = self.basis_functions[symbol]
                    
                    # Calculate consistency factor δ_logic[s]
                    delta_logic = self._calculate_logical_consistency(clause)
                    
                    # Evaluate basis function on spatial grid
                    spatial_grid_np = self.spatial_grid.detach().cpu().numpy()
                    basis_values = basis(spatial_grid_np, t)
                    
                    # Convert to tensor
                    basis_tensor = torch.from_numpy(basis_values).to(field.device).to(torch.complex64)
                    
                    # Add weighted contribution: w_s(t) · Ψ_s(x) · δ_logic[s]
                    contribution = weight.value * weight.consistency_score * delta_logic * basis_tensor
                    field += contribution
                    
                    logger.debug(f"Added symbol '{symbol}' with weight {weight.value:.3f}")
            
            # Store field state
            self.field_state = field.detach()
            
            # Update metrics
            self.metrics['fieldization_time'] = time.time() - start_time
            
            logger.debug(f"Fieldized {len(symbols)} symbols into neural field")
            return field
            
        except Exception as e:
            logger.error(f"Fieldization error: {e}")
            return torch.zeros(*self.field_dims, dtype=torch.complex64)
    
    async def verify_consistency(self, 
                               symbols: List[SymClause],
                               field: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Verify consistency between symbolic and field representations
        
        Args:
            symbols: Symbolic clauses
            field: Neural field (uses self.field_state if None)
            
        Returns:
            Consistency verification results
        """
        if field is None:
            field = self.field_state
        
        try:
            # Forward pass: S → φ
            reconstructed_field = await self.fieldize(symbols)
            
            # Reverse pass: φ → S  
            extracted_symbols = await self.symbolize(field)
            
            # Calculate consistency metrics
            field_mse = torch.mean((field - reconstructed_field).abs() ** 2).item()
            symbol_similarity = self._calculate_symbol_similarity(symbols, extracted_symbols)
            
            # Overall consistency score
            consistency = 1.0 / (1.0 + field_mse) * symbol_similarity
            
            # Update weights based on consistency
            await self._update_weights_from_consistency(symbols, consistency)
            
            results = {
                'field_mse': field_mse,
                'symbol_similarity': symbol_similarity,
                'consistency_score': consistency,
                'verified_symbols': len(symbols),
                'extracted_symbols': len(extracted_symbols)
            }
            
            self.metrics['verification_accuracy'] = consistency
            self.metrics['consistency_score'] = consistency
            
            logger.debug(f"Consistency verification: {consistency:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Consistency verification error: {e}")
            return {'consistency_score': 0.0, 'error': str(e)}
    
    def _find_dominant_modes(self, fft_field: np.ndarray, n_modes: int = 10) -> List[Tuple]:
        """Find dominant Fourier modes in field"""
        magnitude = np.abs(fft_field)
        
        # Find peaks
        flat_indices = np.argpartition(magnitude.ravel(), -n_modes)[-n_modes:]
        indices = np.unravel_index(flat_indices, magnitude.shape)
        
        modes = []
        for i in range(len(flat_indices)):
            idx = tuple(ind[i] for ind in indices)
            k_vec = np.array(idx) - np.array(magnitude.shape) // 2  # Center k-space
            amplitude = magnitude[idx]
            phase = np.angle(fft_field[idx])
            modes.append((k_vec, amplitude, phase))
        
        # Sort by amplitude
        modes.sort(key=lambda x: x[1], reverse=True)
        return modes
    
    def _extract_symbol_from_clause(self, clause: SymClause) -> str:
        """Extract primary symbol identifier from clause"""
        # Simple heuristic - use clause ID as symbol
        # In production, would parse expressions for symbols
        return clause.cid
    
    def _calculate_logical_consistency(self, clause: SymClause) -> float:
        """Calculate δ_logic[s] consistency factor for clause"""
        # Base consistency from clause verification status
        base_score = 1.0
        
        if clause.units_ok is False:
            base_score *= 0.5
        if clause.numeric_ok is False:
            base_score *= 0.7
        if clause.logic_ok is False:
            base_score *= 0.3
        
        # Clause type modifiers
        type_scores = {
            ClauseType.EQUATION: 1.0,
            ClauseType.FACT: 0.9,
            ClauseType.ASSUMPTION: 0.7,
            ClauseType.CLAIM: 0.5,
            ClauseType.CONSTRAINT: 0.8
        }
        
        return base_score * type_scores.get(clause.ctype, 0.5)
    
    def _calculate_symbol_similarity(self, 
                                   symbols1: List[SymClause],
                                   symbols2: List[SymClause]) -> float:
        """Calculate similarity between two sets of symbols"""
        if not symbols1 and not symbols2:
            return 1.0
        if not symbols1 or not symbols2:
            return 0.0
        
        # Simple overlap-based similarity
        ids1 = set(s.cid for s in symbols1)
        ids2 = set(s.cid for s in symbols2)
        
        intersection = len(ids1 & ids2)
        union = len(ids1 | ids2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _update_weights_from_consistency(self, 
                                             symbols: List[SymClause],
                                             consistency: float):
        """Update symbolic weights based on consistency feedback"""
        learning_rate = self.config['learning_rate']
        
        # Reward/penalty based on consistency
        delta = (consistency - 0.5) * 2  # Scale to [-1, 1]
        
        for clause in symbols:
            symbol = self._extract_symbol_from_clause(clause)
            
            if symbol in self.symbolic_weights:
                weight = self.symbolic_weights[symbol]
                weight.update(delta, learning_rate)
                weight.consistency_score = consistency
    
    def get_interface_functions(self) -> Dict[str, Callable]:
        """
        Get the three main interface functions as specified in NFCS Table 7
        
        Returns:
            Dictionary with symbolization, fieldization, and verification functions
        """
        return {
            'symbolization': self.symbolize,  # φ → S: Extract logic from field
            'fieldization': self.fieldize,    # S → φ: Embed rules into field  
            'verification': self.verify_consistency  # S ↔ φ: Check consistency
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def save_state(self) -> Dict[str, Any]:
        """Save bridge state for serialization"""
        return {
            'symbol_registry': self.symbol_registry,
            'symbolic_weights': {
                k: {
                    'value': v.value,
                    'consistency_score': v.consistency_score
                }
                for k, v in self.symbolic_weights.items()
            },
            'field_state': self.field_state.detach().cpu().numpy().tolist(),
            'metrics': self.metrics
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load bridge state from serialization"""
        self.symbol_registry = state['symbol_registry']
        
        # Restore weights
        for symbol, weight_data in state['symbolic_weights'].items():
            self.symbolic_weights[symbol] = SymbolicWeight(
                symbol=symbol,
                value=weight_data['value'],
                consistency_score=weight_data['consistency_score']
            )
        
        # Restore field state
        if 'field_state' in state:
            field_array = np.array(state['field_state'])
            self.field_state = torch.from_numpy(field_array).to(torch.complex64)
        
        self.metrics.update(state.get('metrics', {}))
        
        logger.info("Symbolic-Neural Bridge state loaded successfully")


async def test_bridge():
    """Test function for the bridge"""
    # Create bridge
    bridge = SymbolicNeuralBridge(field_dims=(32, 32))
    
    # Create test symbols
    test_symbols = [
        SymClause(cid="test_eq1", ctype=ClauseType.EQUATION),
        SymClause(cid="test_fact1", ctype=ClauseType.FACT)
    ]
    
    # Test fieldization
    field = await bridge.fieldize(test_symbols)
    print(f"Generated field shape: {field.shape}")
    
    # Test symbolization
    extracted = await bridge.symbolize(field)
    print(f"Extracted {len(extracted)} symbols")
    
    # Test consistency
    consistency = await bridge.verify_consistency(test_symbols, field)
    print(f"Consistency: {consistency}")


if __name__ == "__main__":
    asyncio.run(test_bridge())