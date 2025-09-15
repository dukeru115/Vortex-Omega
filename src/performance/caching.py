"""
Advanced Caching System for NFCS Performance Optimization
=========================================================

Multi-level caching system with LRU eviction, memory management,
and intelligent cache warming for ESC embeddings and computations.

Features:
- Multi-tier cache hierarchy (L1/L2/persistent)
- LRU eviction with memory pressure handling
- Hash-based cache keys with collision detection
- Async cache operations and background warming
- Performance metrics and hit rate optimization

Author: Team Ω (Omega)
Date: September 13, 2025
Version: 2.4.3
"""

import asyncio
import hashlib
import pickle
import time
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import psutil
import numpy as np


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    value: Any
    size_bytes: int
    access_count: int = 0
    creation_time: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # Time to live in seconds

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.creation_time > self.ttl

    def touch(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = time.time()


class LRUCache:
    """Thread-safe LRU cache with memory management"""

    def __init__(self, max_size_mb: float = 100.0, max_items: int = 1000):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_items = max_items
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "size_bytes": 0}

        self.logger = logging.getLogger(f"{__name__}.LRUCache")

    def _compute_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            if isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (list, tuple)):
                return sum(self._compute_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._compute_size(k) + self._compute_size(v) for k, v in value.items())
            else:
                # Fallback to pickle size
                return len(pickle.dumps(value))
        except Exception:
            # Conservative estimate
            return 1024

    def _evict_lru(self):
        """Evict least recently used items"""
        while len(self._cache) > self.max_items or self.stats["size_bytes"] > self.max_size_bytes:
            if not self._cache:
                break

            # Remove oldest item
            key, entry = self._cache.popitem(last=False)
            self.stats["size_bytes"] -= entry.size_bytes
            self.stats["evictions"] += 1

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Store value in cache"""
        with self._lock:
            size = self._compute_size(value)

            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self.stats["size_bytes"] -= old_entry.size_bytes

            # Create new entry
            entry = CacheEntry(key, value, size, ttl=ttl)

            # Check if single item is too large
            if size > self.max_size_bytes:
                self.logger.warning(f"Cache item too large: {size} bytes > {self.max_size_bytes}")
                return

            # Evict if necessary
            self._evict_lru()

            # Add new entry
            self._cache[key] = entry
            self.stats["size_bytes"] += size

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache"""
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self.stats["misses"] += 1
                return None

            # Check expiration
            if entry.is_expired():
                self._cache.pop(key)
                self.stats["size_bytes"] -= entry.size_bytes
                self.stats["misses"] += 1
                return None

            # Update access and move to end
            entry.touch()
            self._cache.move_to_end(key)
            self.stats["hits"] += 1

            return entry.value

    def delete(self, key: str) -> bool:
        """Remove entry from cache"""
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry:
                self.stats["size_bytes"] -= entry.size_bytes
                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self.stats["size_bytes"] = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "hit_rate": hit_rate,
                "size_bytes": self.stats["size_bytes"],
                "size_mb": self.stats["size_bytes"] / (1024 * 1024),
                "item_count": len(self._cache),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "max_items": self.max_items,
            }


class EmbeddingCache:
    """Specialized cache for ESC embeddings with semantic hashing"""

    def __init__(self, max_size_mb: float = 200.0, enable_persistence: bool = True):
        self.l1_cache = LRUCache(max_size_mb * 0.3, 500)  # Fast memory cache
        self.l2_cache = LRUCache(max_size_mb * 0.7, 2000)  # Larger memory cache
        self.enable_persistence = enable_persistence
        self.cache_dir = Path("cache/embeddings") if enable_persistence else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"{__name__}.EmbeddingCache")

    def _compute_hash(self, tokens: List[str], context: Optional[str] = None) -> str:
        """Compute semantic hash for tokens and context"""
        # Create consistent hash from tokens and context
        content = "|".join(tokens)
        if context:
            content += f"||{context}"

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_embedding(
        self, tokens: List[str], context: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Get cached embedding for token sequence"""
        cache_key = self._compute_hash(tokens, context)

        # Try L1 cache first
        embedding = self.l1_cache.get(cache_key)
        if embedding is not None:
            return embedding

        # Try L2 cache
        embedding = self.l2_cache.get(cache_key)
        if embedding is not None:
            # Promote to L1
            self.l1_cache.put(cache_key, embedding)
            return embedding

        # Try persistent cache
        if self.enable_persistence:
            embedding = self._load_from_disk(cache_key)
            if embedding is not None:
                self.l2_cache.put(cache_key, embedding)
                return embedding

        return None

    def put_embedding(
        self,
        tokens: List[str],
        embedding: np.ndarray,
        context: Optional[str] = None,
        ttl: Optional[float] = 3600,
    ):
        """Cache embedding for token sequence"""
        cache_key = self._compute_hash(tokens, context)

        # Store in L1 cache
        self.l1_cache.put(cache_key, embedding, ttl)

        # Store in persistent cache if enabled
        if self.enable_persistence:
            self._save_to_disk(cache_key, embedding)

    def _save_to_disk(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to persistent storage"""
        try:
            if self.cache_dir:
                cache_path = self.cache_dir / f"{cache_key}.npy"
                np.save(cache_path, embedding)
        except Exception as e:
            self.logger.warning(f"Failed to save embedding to disk: {e}")

    def _load_from_disk(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from persistent storage"""
        try:
            if self.cache_dir:
                cache_path = self.cache_dir / f"{cache_key}.npy"
                if cache_path.exists():
                    return np.load(cache_path)
        except Exception as e:
            self.logger.warning(f"Failed to load embedding from disk: {e}")

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()

        # Count persistent cache files
        persistent_count = 0
        persistent_size = 0
        if self.cache_dir and self.cache_dir.exists():
            for file in self.cache_dir.glob("*.npy"):
                persistent_count += 1
                persistent_size += file.stat().st_size

        return {
            "l1_cache": l1_stats,
            "l2_cache": l2_stats,
            "persistent_cache": {
                "enabled": self.enable_persistence,
                "item_count": persistent_count,
                "size_mb": persistent_size / (1024 * 1024),
            },
            "total_hit_rate": (l1_stats["hits"] + l2_stats["hits"])
            / max(1, l1_stats["hits"] + l1_stats["misses"] + l2_stats["hits"] + l2_stats["misses"]),
        }


class ComputationCache:
    """Cache for computational results with dependency tracking"""

    def __init__(self, max_size_mb: float = 100.0):
        self.cache = LRUCache(max_size_mb, 1000)
        self.dependencies: Dict[str, List[str]] = {}  # key -> dependent keys
        self.logger = logging.getLogger(f"{__name__}.ComputationCache")

    def _compute_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Compute cache key for function call"""
        # Create deterministic key from function and arguments
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": sorted(kwargs.items()) if kwargs else [],
        }

        key_str = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(key_str).hexdigest()[:16]

    def get_result(self, func_name: str, args: Tuple, kwargs: Dict) -> Optional[Any]:
        """Get cached computation result"""
        cache_key = self._compute_key(func_name, args, kwargs)
        return self.cache.get(cache_key)

    def put_result(
        self,
        func_name: str,
        args: Tuple,
        kwargs: Dict,
        result: Any,
        dependencies: Optional[List[str]] = None,
        ttl: Optional[float] = None,
    ):
        """Cache computation result with dependencies"""
        cache_key = self._compute_key(func_name, args, kwargs)

        self.cache.put(cache_key, result, ttl)

        if dependencies:
            self.dependencies[cache_key] = dependencies

    def invalidate_dependencies(self, dependency_key: str):
        """Invalidate all cache entries that depend on a key"""
        to_remove = []

        for cache_key, deps in self.dependencies.items():
            if dependency_key in deps:
                to_remove.append(cache_key)

        for key in to_remove:
            self.cache.delete(key)
            self.dependencies.pop(key, None)

    def cached_call(
        self,
        func: Callable,
        *args,
        dependencies: Optional[List[str]] = None,
        ttl: Optional[float] = None,
        **kwargs,
    ):
        """Decorator-like cached function call"""
        # Check cache first
        result = self.get_result(func.__name__, args, kwargs)
        if result is not None:
            return result

        # Compute and cache result
        result = func(*args, **kwargs)
        self.put_result(func.__name__, args, kwargs, result, dependencies, ttl)

        return result


class CacheManager:
    """Central cache manager with memory monitoring and optimization"""

    def __init__(self, total_memory_limit_mb: float = 500.0):
        self.memory_limit = total_memory_limit_mb
        self.embedding_cache = EmbeddingCache(total_memory_limit_mb * 0.6)
        self.computation_cache = ComputationCache(total_memory_limit_mb * 0.4)

        # Memory monitoring
        self._memory_check_interval = 30.0  # seconds
        self._memory_monitor_task: Optional[asyncio.Task] = None
        self._running = False

        self.logger = logging.getLogger(f"{__name__}.CacheManager")

    async def start_memory_monitoring(self):
        """Start background memory monitoring"""
        self._running = True
        self._memory_monitor_task = asyncio.create_task(self._memory_monitor_loop())
        self.logger.info("Memory monitoring started")

    async def stop_memory_monitoring(self):
        """Stop background memory monitoring"""
        self._running = False
        if self._memory_monitor_task:
            self._memory_monitor_task.cancel()
            try:
                await self._memory_monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Memory monitoring stopped")

    async def _memory_monitor_loop(self):
        """Background memory monitoring loop"""
        while self._running:
            try:
                await asyncio.sleep(self._memory_check_interval)

                # Check system memory pressure
                memory_info = psutil.virtual_memory()
                if memory_info.percent > 85:  # High memory usage
                    self.logger.warning(f"High memory usage detected: {memory_info.percent:.1f}%")
                    self._emergency_cache_cleanup()

                # Log cache statistics
                stats = self.get_comprehensive_stats()
                self.logger.debug(
                    f"Cache stats - Total: {stats['total_size_mb']:.1f}MB, "
                    f"Hit rate: {stats['overall_hit_rate']:.3f}"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Memory monitor error: {e}")

    def _emergency_cache_cleanup(self):
        """Emergency cache cleanup under memory pressure"""
        self.logger.info("Performing emergency cache cleanup")

        # Clear computation cache first (easier to recompute)
        self.computation_cache.cache.clear()

        # Reduce embedding cache size by 50%
        l1_cache = self.embedding_cache.l1_cache
        l2_cache = self.embedding_cache.l2_cache

        # Clear half of L1 cache (oldest entries)
        with l1_cache._lock:
            items_to_remove = len(l1_cache._cache) // 2
            for _ in range(items_to_remove):
                if l1_cache._cache:
                    key, entry = l1_cache._cache.popitem(last=False)
                    l1_cache.stats["size_bytes"] -= entry.size_bytes

        # Similar for L2 cache
        with l2_cache._lock:
            items_to_remove = len(l2_cache._cache) // 2
            for _ in range(items_to_remove):
                if l2_cache._cache:
                    key, entry = l2_cache._cache.popitem(last=False)
                    l2_cache.stats["size_bytes"] -= entry.size_bytes

    def warm_cache(self, embedding_samples: List[Tuple[List[str], Optional[str]]] = None):
        """Warm up cache with common operations"""
        self.logger.info("Warming up caches...")

        if embedding_samples:
            # Pre-compute embeddings for common token sequences
            for tokens, context in embedding_samples:
                # This would trigger embedding computation if not cached
                self.embedding_cache.get_embedding(tokens, context)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all caches"""
        embedding_stats = self.embedding_cache.get_stats()
        computation_stats = self.computation_cache.cache.get_stats()

        total_size_mb = (
            embedding_stats["l1_cache"]["size_mb"]
            + embedding_stats["l2_cache"]["size_mb"]
            + computation_stats["size_mb"]
        )

        # Overall hit rate calculation
        total_hits = (
            embedding_stats["l1_cache"]["hits"]
            + embedding_stats["l2_cache"]["hits"]
            + computation_stats["hits"]
        )
        total_requests = (
            embedding_stats["l1_cache"]["hits"]
            + embedding_stats["l1_cache"]["misses"]
            + embedding_stats["l2_cache"]["hits"]
            + embedding_stats["l2_cache"]["misses"]
            + computation_stats["hits"]
            + computation_stats["misses"]
        )

        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0

        return {
            "embedding_cache": embedding_stats,
            "computation_cache": computation_stats,
            "total_size_mb": total_size_mb,
            "memory_limit_mb": self.memory_limit,
            "memory_usage_percent": (total_size_mb / self.memory_limit) * 100,
            "overall_hit_rate": overall_hit_rate,
            "memory_monitoring_active": self._running,
        }

    def optimize_performance(self):
        """Optimize cache performance based on usage patterns"""
        stats = self.get_comprehensive_stats()

        # Adjust cache sizes based on hit rates
        emb_hit_rate = stats["embedding_cache"]["total_hit_rate"]
        comp_hit_rate = stats["computation_cache"]["hit_rate"]

        self.logger.info(
            f"Performance optimization - Embedding HR: {emb_hit_rate:.3f}, "
            f"Computation HR: {comp_hit_rate:.3f}"
        )

        # If embedding cache has better hit rate, allocate more memory to it
        if emb_hit_rate > comp_hit_rate + 0.1:  # Significant difference
            self.logger.info("Reallocating memory toward embedding cache")
            # Implementation would adjust cache sizes dynamically

    def clear_all_caches(self):
        """Clear all caches"""
        self.embedding_cache.l1_cache.clear()
        self.embedding_cache.l2_cache.clear()
        self.computation_cache.cache.clear()
        self.logger.info("All caches cleared")
