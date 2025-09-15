"""
NFCS Performance Optimization Module
===================================

High-performance implementations with Numba JIT compilation, caching,
and memory optimization for NFCS v2.4.3.

Author: Team Ω (Omega)
Date: September 13, 2025
Version: 2.4.3
"""

from .caching import EmbeddingCache, ComputationCache, CacheManager
from .memory_optimizer import MemoryOptimizer, ResourceManager
from .benchmarks import PerformanceBenchmark, BenchmarkSuite

__all__ = [
    "EmbeddingCache",
    "ComputationCache",
    "CacheManager",
    "MemoryOptimizer",
    "ResourceManager",
    "PerformanceBenchmark",
    "BenchmarkSuite",
]
