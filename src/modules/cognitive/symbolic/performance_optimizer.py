"""
Performance Optimizer for Kamil Symbolic AI
===========================================

High-performance optimization and SLO compliance verification module for
deterministic LLM-free symbolic AI implementation.

Features:
- Real-time SLO monitoring (≤300ms latency, ≥0.98 dimensional accuracy)
- Adaptive caching and memoization strategies
- Performance profiling and bottleneck identification
- Memory optimization and garbage collection
- Concurrent processing optimization
- Benchmark suite for continuous performance validation

Created: September 14, 2025
Author: Team Ω - Performance Optimization for Kamil Specification
License: Apache 2.0
"""

import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import weakref
import gc
import psutil
import numpy as np

# Import symbolic AI components
from .symbolic_ai_kamil import SymbolicAIKamil, SymField, VerificationReport
from .kamil_integration import KamilSymbolicIntegration

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics tracking."""

    # SLO Compliance Tracking
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    accuracy_samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    slo_violations: int = 0
    total_operations: int = 0

    # Latency Distribution
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    # Accuracy Statistics
    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0
    accuracy_min: float = 1.0

    # Memory Usage
    memory_peak_mb: float = 0.0
    memory_current_mb: float = 0.0
    gc_collections: int = 0

    # Cache Performance
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0

    # Throughput Metrics
    operations_per_second: float = 0.0
    concurrent_operations: int = 0

    def update_latency(self, latency_ms: float) -> None:
        """Update latency metrics."""
        self.latency_samples.append(latency_ms)
        self.total_operations += 1

        if latency_ms > 300:  # SLO violation
            self.slo_violations += 1

        # Update percentiles
        if len(self.latency_samples) > 10:
            sorted_samples = sorted(self.latency_samples)
            n = len(sorted_samples)
            self.latency_p50 = sorted_samples[int(n * 0.5)]
            self.latency_p95 = sorted_samples[int(n * 0.95)]
            self.latency_p99 = sorted_samples[int(n * 0.99)]

    def update_accuracy(self, accuracy: float) -> None:
        """Update accuracy metrics."""
        self.accuracy_samples.append(accuracy)

        if accuracy < 0.98:  # SLO violation
            self.slo_violations += 1

        # Update statistics
        if len(self.accuracy_samples) > 0:
            samples = list(self.accuracy_samples)
            self.accuracy_mean = np.mean(samples)
            self.accuracy_std = np.std(samples)
            self.accuracy_min = np.min(samples)

    def get_slo_compliance_rate(self) -> float:
        """Calculate SLO compliance rate."""
        if self.total_operations == 0:
            return 1.0
        return max(0.0, 1.0 - (self.slo_violations / self.total_operations))


@dataclass
class CacheEntry:
    """Cache entry with TTL and access tracking."""

    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    ttl_seconds: float = 300.0  # 5 minutes default TTL

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds

    def access(self) -> Any:
        """Access cached value and update metrics."""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.value


class AdaptiveCache:
    """High-performance adaptive cache with TTL and LRU eviction."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: deque = deque()
        self._lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check expiration
                if entry.is_expired():
                    del self._cache[key]
                    self._remove_from_access_order(key)
                    self.misses += 1
                    return None

                # Update access order
                self._update_access_order(key)
                self.hits += 1
                return entry.access()

            self.misses += 1
            return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self._lock:
            ttl = ttl or self.default_ttl

            # Evict if necessary
            if key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_lru()

            # Store entry
            self._cache[key] = CacheEntry(value, ttl_seconds=ttl)
            self._update_access_order(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
            }

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_order:
            return

        lru_key = self._access_order.popleft()
        if lru_key in self._cache:
            del self._cache[lru_key]
            self.evictions += 1

    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU tracking."""
        self._remove_from_access_order(key)
        self._access_order.append(key)

    def _remove_from_access_order(self, key: str) -> None:
        """Remove key from access order tracking."""
        try:
            self._access_order.remove(key)
        except ValueError:
            pass  # Key not in deque


class PerformanceOptimizer:
    """
    High-performance optimization layer for Symbolic AI.

    Provides adaptive caching, memory optimization, concurrent processing,
    and real-time SLO monitoring for maximum performance.
    """

    def __init__(
        self,
        symbolic_ai: SymbolicAIKamil,
        integration: Optional[KamilSymbolicIntegration] = None,
        enable_caching: bool = True,
        enable_profiling: bool = True,
        max_workers: int = 4,
        cache_size: int = 1000,
    ):
        """
        Initialize performance optimizer.

        Args:
            symbolic_ai: Symbolic AI engine to optimize
            integration: Optional integration layer to optimize
            enable_caching: Enable adaptive caching
            enable_profiling: Enable performance profiling
            max_workers: Maximum concurrent workers
            cache_size: Maximum cache size
        """
        self.symbolic_ai = symbolic_ai
        self.integration = integration
        self.enable_caching = enable_caching
        self.enable_profiling = enable_profiling
        self.max_workers = max_workers

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.start_time = time.time()

        # Adaptive caching
        if enable_caching:
            self.symbolize_cache = AdaptiveCache(max_size=cache_size, default_ttl=300)
            self.verify_cache = AdaptiveCache(max_size=cache_size // 2, default_ttl=600)
            self.field_cache = AdaptiveCache(max_size=cache_size // 4, default_ttl=180)
        else:
            self.symbolize_cache = None
            self.verify_cache = None
            self.field_cache = None

        # Concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.concurrent_operations = 0
        self._concurrent_lock = threading.Lock()

        # Profiling data
        self.profiling_data = defaultdict(list)
        self.bottleneck_analysis = {}

        # Memory monitoring
        self.process = psutil.Process()
        self._monitor_memory()

        logger.info(
            f"Performance optimizer initialized with caching: {enable_caching}, "
            f"profiling: {enable_profiling}, max_workers: {max_workers}"
        )

    def optimized_symbolize(
        self, input_text: str, context: Optional[Dict[str, Any]] = None
    ) -> List[SymField]:
        """
        Optimized symbolization with caching and profiling.

        Args:
            input_text: Input text to symbolize
            context: Optional context for symbolization

        Returns:
            Symbolized fields with performance optimization
        """
        start_time = time.time()

        # Generate cache key
        cache_key = self._generate_cache_key(input_text, context)

        # Try cache first
        if self.enable_caching and self.symbolize_cache:
            cached_result = self.symbolize_cache.get(cache_key)
            if cached_result is not None:
                self._update_cache_metrics(hit=True)
                return cached_result

        self._update_cache_metrics(hit=False)

        # Perform symbolization with profiling
        try:
            with self._concurrent_context():
                if self.enable_profiling:
                    prof_start = time.perf_counter()

                # Call original symbolization
                result = self.symbolic_ai.symbolize(input_text, context)

                if self.enable_profiling:
                    prof_time = (time.perf_counter() - prof_start) * 1000
                    self.profiling_data["symbolize"].append(prof_time)

                # Cache result
                if self.enable_caching and self.symbolize_cache:
                    self.symbolize_cache.put(cache_key, result)

                # Update performance metrics
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.update_latency(latency_ms)

                return result

        except Exception as e:
            logger.error(f"Optimized symbolization failed: {e}")
            raise

    def optimized_verify(self, sym_fields: List[SymField]) -> VerificationReport:
        """
        Optimized verification with caching and profiling.

        Args:
            sym_fields: Symbolic fields to verify

        Returns:
            Verification report with performance optimization
        """
        start_time = time.time()

        # Generate cache key based on field content
        cache_key = self._generate_fields_cache_key(sym_fields)

        # Try cache first
        if self.enable_caching and self.verify_cache:
            cached_result = self.verify_cache.get(cache_key)
            if cached_result is not None:
                self._update_cache_metrics(hit=True)
                return cached_result

        self._update_cache_metrics(hit=False)

        # Perform verification with profiling
        try:
            with self._concurrent_context():
                if self.enable_profiling:
                    prof_start = time.perf_counter()

                # Call original verification
                result = self.symbolic_ai.verify(sym_fields)

                if self.enable_profiling:
                    prof_time = (time.perf_counter() - prof_start) * 1000
                    self.profiling_data["verify"].append(prof_time)

                # Cache result with shorter TTL for verification
                if self.enable_caching and self.verify_cache:
                    self.verify_cache.put(cache_key, result, ttl=300)

                # Update performance metrics
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.update_latency(latency_ms)
                self.metrics.update_accuracy(result.dimensional_accuracy)

                return result

        except Exception as e:
            logger.error(f"Optimized verification failed: {e}")
            raise

    def batch_process_semantic_inputs(
        self, inputs: List[Tuple[str, Optional[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple semantic inputs concurrently for maximum throughput.

        Args:
            inputs: List of (input_text, context) tuples

        Returns:
            List of processing results
        """
        start_time = time.time()

        if not self.integration:
            raise ValueError("Integration layer required for batch processing")

        # Submit concurrent tasks
        futures = []
        for input_text, context in inputs:
            future = self.executor.submit(self._process_single_semantic_input, input_text, context)
            futures.append(future)

        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=5.0)  # 5 second timeout per task
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing task failed: {e}")
                results.append({"error": str(e), "slo_compliant": False})

        # Update throughput metrics
        total_time = time.time() - start_time
        if total_time > 0:
            self.metrics.operations_per_second = len(inputs) / total_time

        return results

    def run_performance_benchmark(
        self, num_operations: int = 100, warmup_operations: int = 10
    ) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmark.

        Args:
            num_operations: Number of benchmark operations
            warmup_operations: Number of warmup operations

        Returns:
            Detailed benchmark results
        """
        logger.info(f"Starting performance benchmark: {num_operations} operations")

        # Test cases
        test_cases = [
            "F = m * a where force equals mass times acceleration",
            "Energy conservation: E_initial = E_final in isolated systems",
            "Momentum p = m * v must be conserved in all interactions",
            "The oscillation frequency f = ω/(2π) for harmonic motion",
            "Temperature T must remain below 373 K to prevent phase transition",
        ]

        # Warmup phase
        logger.info("Running warmup operations...")
        for _ in range(warmup_operations):
            test_case = test_cases[np.random.randint(len(test_cases))]
            try:
                sym_fields = self.optimized_symbolize(test_case)
                self.optimized_verify(sym_fields)
            except Exception as e:
                logger.warning(f"Warmup operation failed: {e}")

        # Clear metrics for clean benchmark
        self.metrics = PerformanceMetrics()

        # Benchmark phase
        benchmark_start = time.time()
        successful_operations = 0
        failed_operations = 0

        latencies = []
        accuracies = []

        logger.info("Running benchmark operations...")
        for i in range(num_operations):
            test_case = test_cases[i % len(test_cases)]

            try:
                op_start = time.time()

                sym_fields = self.optimized_symbolize(test_case)
                report = self.optimized_verify(sym_fields)

                op_latency = (time.time() - op_start) * 1000
                latencies.append(op_latency)
                accuracies.append(report.dimensional_accuracy)

                successful_operations += 1

            except Exception as e:
                failed_operations += 1
                logger.warning(f"Benchmark operation {i} failed: {e}")

        benchmark_time = time.time() - benchmark_start

        # Calculate statistics
        if latencies:
            latencies = np.array(latencies)
            accuracies = np.array(accuracies)

            benchmark_results = {
                "total_operations": num_operations,
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "success_rate": successful_operations / num_operations,
                # Latency statistics
                "latency_mean_ms": np.mean(latencies),
                "latency_median_ms": np.median(latencies),
                "latency_p95_ms": np.percentile(latencies, 95),
                "latency_p99_ms": np.percentile(latencies, 99),
                "latency_std_ms": np.std(latencies),
                "latency_min_ms": np.min(latencies),
                "latency_max_ms": np.max(latencies),
                # Accuracy statistics
                "accuracy_mean": np.mean(accuracies),
                "accuracy_median": np.median(accuracies),
                "accuracy_std": np.std(accuracies),
                "accuracy_min": np.min(accuracies),
                "accuracy_max": np.max(accuracies),
                # SLO compliance
                "latency_slo_violations": np.sum(latencies > 300),
                "accuracy_slo_violations": np.sum(accuracies < 0.98),
                "slo_compliance_rate": np.mean((latencies <= 300) & (accuracies >= 0.98)),
                # Throughput
                "operations_per_second": successful_operations / benchmark_time,
                "total_benchmark_time_s": benchmark_time,
                # Cache performance
                "cache_stats": self._get_all_cache_stats(),
                # Memory usage
                "memory_peak_mb": self.metrics.memory_peak_mb,
                "memory_current_mb": self.metrics.memory_current_mb,
                # Profiling data
                "profiling_summary": self._get_profiling_summary(),
            }
        else:
            benchmark_results = {
                "error": "No successful operations completed",
                "failed_operations": failed_operations,
                "total_benchmark_time_s": benchmark_time,
            }

        logger.info(f"Benchmark completed: {successful_operations}/{num_operations} successful")
        return benchmark_results

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage by clearing caches and running garbage collection.

        Returns:
            Memory optimization results
        """
        start_memory = self._get_memory_usage_mb()

        # Clear caches
        cache_entries_cleared = 0
        if self.enable_caching:
            if self.symbolize_cache:
                cache_entries_cleared += len(self.symbolize_cache._cache)
                self.symbolize_cache.clear()

            if self.verify_cache:
                cache_entries_cleared += len(self.verify_cache._cache)
                self.verify_cache.clear()

            if self.field_cache:
                cache_entries_cleared += len(self.field_cache._cache)
                self.field_cache.clear()

        # Run garbage collection
        gc_collected = gc.collect()
        self.metrics.gc_collections += 1

        end_memory = self._get_memory_usage_mb()
        memory_freed = start_memory - end_memory

        optimization_results = {
            "cache_entries_cleared": cache_entries_cleared,
            "gc_objects_collected": gc_collected,
            "memory_before_mb": start_memory,
            "memory_after_mb": end_memory,
            "memory_freed_mb": memory_freed,
            "optimization_effective": memory_freed > 0,
        }

        logger.info(
            f"Memory optimization: {memory_freed:.2f}MB freed, "
            f"{cache_entries_cleared} cache entries cleared"
        )

        return optimization_results

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics and analysis."""

        uptime = time.time() - self.start_time

        return {
            # Core metrics
            "performance_metrics": {
                "slo_compliance_rate": self.metrics.get_slo_compliance_rate(),
                "total_operations": self.metrics.total_operations,
                "slo_violations": self.metrics.slo_violations,
                "operations_per_second": self.metrics.operations_per_second,
                "uptime_seconds": uptime,
            },
            # Latency analysis
            "latency_metrics": {
                "p50_ms": self.metrics.latency_p50,
                "p95_ms": self.metrics.latency_p95,
                "p99_ms": self.metrics.latency_p99,
                "samples_count": len(self.metrics.latency_samples),
            },
            # Accuracy analysis
            "accuracy_metrics": {
                "mean": self.metrics.accuracy_mean,
                "std": self.metrics.accuracy_std,
                "min": self.metrics.accuracy_min,
                "samples_count": len(self.metrics.accuracy_samples),
            },
            # Memory metrics
            "memory_metrics": {
                "current_mb": self.metrics.memory_current_mb,
                "peak_mb": self.metrics.memory_peak_mb,
                "gc_collections": self.metrics.gc_collections,
            },
            # Cache metrics
            "cache_metrics": self._get_all_cache_stats(),
            # Concurrency metrics
            "concurrency_metrics": {
                "max_workers": self.max_workers,
                "current_concurrent_operations": self.concurrent_operations,
            },
            # Profiling analysis
            "profiling_analysis": self._get_profiling_summary(),
            # Bottleneck analysis
            "bottleneck_analysis": self._analyze_bottlenecks(),
        }

    def cleanup(self) -> None:
        """Clean up resources and shutdown optimizer."""
        try:
            # Shutdown thread pool
            self.executor.shutdown(wait=True, timeout=5.0)

            # Clear caches
            if self.enable_caching:
                if self.symbolize_cache:
                    self.symbolize_cache.clear()
                if self.verify_cache:
                    self.verify_cache.clear()
                if self.field_cache:
                    self.field_cache.clear()

            logger.info("Performance optimizer cleaned up successfully")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    # =========================================================================
    # PRIVATE IMPLEMENTATION METHODS
    # =========================================================================

    def _process_single_semantic_input(
        self, input_text: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process single semantic input for batch processing."""
        try:
            return self.integration.process_semantic_input(input_text, context=context)
        except Exception as e:
            return {"error": str(e), "slo_compliant": False}

    def _concurrent_context(self):
        """Context manager for tracking concurrent operations."""

        class ConcurrentContext:
            def __init__(self, optimizer):
                self.optimizer = optimizer

            def __enter__(self):
                with self.optimizer._concurrent_lock:
                    self.optimizer.concurrent_operations += 1
                    self.optimizer.metrics.concurrent_operations = max(
                        self.optimizer.metrics.concurrent_operations,
                        self.optimizer.concurrent_operations,
                    )
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                with self.optimizer._concurrent_lock:
                    self.optimizer.concurrent_operations -= 1

        return ConcurrentContext(self)

    def _generate_cache_key(self, input_text: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for input."""
        key_parts = [input_text]

        if context:
            # Include relevant context parts in key
            for key, value in sorted(context.items()):
                if isinstance(value, (str, int, float, bool)):
                    key_parts.append(f"{key}:{value}")

        return hash(tuple(key_parts))

    def _generate_fields_cache_key(self, sym_fields: List[SymField]) -> str:
        """Generate cache key for symbolic fields."""
        key_parts = []

        for field in sym_fields:
            key_parts.append(field.field_id)
            key_parts.append(str(len(field.clauses)))
            key_parts.append(str(len(field.expressions)))

        return hash(tuple(key_parts))

    def _update_cache_metrics(self, hit: bool) -> None:
        """Update cache performance metrics."""
        if hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1

        total_requests = self.metrics.cache_hits + self.metrics.cache_misses
        if total_requests > 0:
            self.metrics.cache_hit_rate = self.metrics.cache_hits / total_requests

    def _get_all_cache_stats(self) -> Dict[str, Any]:
        """Get statistics from all caches."""
        stats = {}

        if self.enable_caching:
            if self.symbolize_cache:
                stats["symbolize_cache"] = self.symbolize_cache.get_stats()

            if self.verify_cache:
                stats["verify_cache"] = self.verify_cache.get_stats()

            if self.field_cache:
                stats["field_cache"] = self.field_cache.get_stats()

            # Overall cache stats
            total_hits = self.metrics.cache_hits
            total_misses = self.metrics.cache_misses
            total_requests = total_hits + total_misses

            stats["overall"] = {
                "total_hits": total_hits,
                "total_misses": total_misses,
                "total_requests": total_requests,
                "hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
            }

        return stats

    def _get_profiling_summary(self) -> Dict[str, Any]:
        """Get summary of profiling data."""
        summary = {}

        if self.enable_profiling:
            for operation, times in self.profiling_data.items():
                if times:
                    times_array = np.array(times)
                    summary[operation] = {
                        "count": len(times),
                        "mean_ms": np.mean(times_array),
                        "median_ms": np.median(times_array),
                        "p95_ms": np.percentile(times_array, 95),
                        "std_ms": np.std(times_array),
                        "min_ms": np.min(times_array),
                        "max_ms": np.max(times_array),
                    }

        return summary

    def _analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks."""
        bottlenecks = {}

        if self.enable_profiling and self.profiling_data:
            # Find slowest operations
            operation_means = {}
            for operation, times in self.profiling_data.items():
                if times:
                    operation_means[operation] = np.mean(times)

            if operation_means:
                slowest_operation = max(operation_means, key=operation_means.get)
                bottlenecks["slowest_operation"] = {
                    "operation": slowest_operation,
                    "mean_latency_ms": operation_means[slowest_operation],
                }

            # Check SLO violations
            slo_violation_rate = self.metrics.slo_violations / max(self.metrics.total_operations, 1)
            if slo_violation_rate > 0.05:  # >5% violation rate
                bottlenecks["slo_violation_concern"] = {
                    "violation_rate": slo_violation_rate,
                    "total_violations": self.metrics.slo_violations,
                }

            # Check cache effectiveness
            if self.metrics.cache_hit_rate < 0.3:  # <30% hit rate
                bottlenecks["cache_effectiveness_low"] = {
                    "hit_rate": self.metrics.cache_hit_rate,
                    "recommendation": "Consider adjusting cache size or TTL",
                }

        return bottlenecks

    def _monitor_memory(self) -> None:
        """Monitor memory usage."""
        current_memory = self._get_memory_usage_mb()
        self.metrics.memory_current_mb = current_memory

        if current_memory > self.metrics.memory_peak_mb:
            self.metrics.memory_peak_mb = current_memory

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert bytes to MB
        except Exception:
            return 0.0


# =============================================================================
# FACTORY AND UTILITY FUNCTIONS
# =============================================================================


def create_optimized_symbolic_system(
    symbolic_ai: Optional[SymbolicAIKamil] = None,
    integration: Optional[KamilSymbolicIntegration] = None,
    enable_caching: bool = True,
    enable_profiling: bool = True,
    max_workers: int = 4,
) -> PerformanceOptimizer:
    """
    Create optimized symbolic AI system with performance enhancements.

    Args:
        symbolic_ai: Optional symbolic AI engine
        integration: Optional integration layer
        enable_caching: Enable performance caching
        enable_profiling: Enable performance profiling
        max_workers: Maximum concurrent workers

    Returns:
        High-performance optimized system
    """
    from .symbolic_ai_kamil import create_test_symbolic_ai
    from .kamil_integration import create_integrated_symbolic_system

    if symbolic_ai is None:
        symbolic_ai = create_test_symbolic_ai()

    if integration is None:
        integration = create_integrated_symbolic_system(debug_mode=False)

    optimizer = PerformanceOptimizer(
        symbolic_ai=symbolic_ai,
        integration=integration,
        enable_caching=enable_caching,
        enable_profiling=enable_profiling,
        max_workers=max_workers,
    )

    logger.info("High-performance optimized symbolic system created")
    return optimizer


def run_slo_validation_suite(
    optimizer: PerformanceOptimizer,
    num_samples: int = 200,
    target_latency_ms: float = 300.0,
    target_accuracy: float = 0.98,
) -> Dict[str, Any]:
    """
    Run comprehensive SLO validation suite.

    Args:
        optimizer: Performance optimizer to validate
        num_samples: Number of validation samples
        target_latency_ms: Target latency SLO
        target_accuracy: Target accuracy SLO

    Returns:
        SLO validation results
    """
    logger.info(f"Running SLO validation suite with {num_samples} samples")

    # Run benchmark
    benchmark_results = optimizer.run_performance_benchmark(
        num_operations=num_samples, warmup_operations=20
    )

    # Extract SLO compliance metrics
    latency_compliance = benchmark_results.get("latency_slo_violations", num_samples) == 0
    accuracy_compliance = benchmark_results.get("accuracy_slo_violations", num_samples) == 0

    slo_validation = {
        "slo_compliant": latency_compliance and accuracy_compliance,
        "latency_slo_met": latency_compliance,
        "accuracy_slo_met": accuracy_compliance,
        "latency_statistics": {
            "mean_ms": benchmark_results.get("latency_mean_ms", 0),
            "p95_ms": benchmark_results.get("latency_p95_ms", 0),
            "p99_ms": benchmark_results.get("latency_p99_ms", 0),
            "target_ms": target_latency_ms,
            "violations": benchmark_results.get("latency_slo_violations", 0),
        },
        "accuracy_statistics": {
            "mean": benchmark_results.get("accuracy_mean", 0),
            "min": benchmark_results.get("accuracy_min", 0),
            "target": target_accuracy,
            "violations": benchmark_results.get("accuracy_slo_violations", 0),
        },
        "performance_summary": {
            "operations_per_second": benchmark_results.get("operations_per_second", 0),
            "success_rate": benchmark_results.get("success_rate", 0),
            "cache_hit_rate": benchmark_results.get("cache_stats", {})
            .get("overall", {})
            .get("hit_rate", 0),
        },
        "full_benchmark_results": benchmark_results,
    }

    logger.info(f"SLO validation complete: compliant = {slo_validation['slo_compliant']}")
    return slo_validation


if __name__ == "__main__":
    # Performance optimization demonstration
    print("Symbolic AI Performance Optimizer")
    print("=" * 40)

    # Create optimized system
    optimizer = create_optimized_symbolic_system(
        enable_caching=True, enable_profiling=True, max_workers=4
    )

    print("Running performance benchmark...")

    # Run benchmark
    benchmark_results = optimizer.run_performance_benchmark(num_operations=50, warmup_operations=5)

    print(f"Benchmark Results:")
    print(f"  Success Rate: {benchmark_results.get('success_rate', 0):.1%}")
    print(f"  Mean Latency: {benchmark_results.get('latency_mean_ms', 0):.2f}ms")
    print(f"  P95 Latency: {benchmark_results.get('latency_p95_ms', 0):.2f}ms")
    print(f"  Mean Accuracy: {benchmark_results.get('accuracy_mean', 0):.4f}")
    print(f"  SLO Compliance: {benchmark_results.get('slo_compliance_rate', 0):.1%}")
    print(f"  Throughput: {benchmark_results.get('operations_per_second', 0):.1f} ops/sec")

    # Run SLO validation
    print("\nRunning SLO validation...")
    slo_results = run_slo_validation_suite(optimizer, num_samples=30)

    print(f"SLO Validation Results:")
    print(f"  Overall Compliant: {slo_results['slo_compliant']}")
    print(f"  Latency SLO Met: {slo_results['latency_slo_met']}")
    print(f"  Accuracy SLO Met: {slo_results['accuracy_slo_met']}")

    # Memory optimization
    print("\nOptimizing memory...")
    memory_results = optimizer.optimize_memory_usage()
    print(f"Memory freed: {memory_results['memory_freed_mb']:.2f}MB")

    # Cleanup
    optimizer.cleanup()
    print("\nPerformance optimization demonstration complete")
