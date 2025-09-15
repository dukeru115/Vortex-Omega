"""
Memory Module Core - NFCS Memory Management System

Implements long-term memory and experience integration:
- Episodic memory for experience storage and retrieval
- Semantic memory for knowledge representation
- Working memory for active information processing
- Memory consolidation and forgetting mechanisms
- Constitutional memory governance and privacy protection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from collections import deque, defaultdict
import json

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory systems."""

    EPISODIC = "episodic"  # Specific experiences and events
    SEMANTIC = "semantic"  # General knowledge and facts
    PROCEDURAL = "procedural"  # Skills and procedures
    WORKING = "working"  # Active information processing
    CONSTITUTIONAL = "constitutional"  # Constitutional compliance history


class MemoryImportance(Enum):
    """Importance levels for memory entries."""

    CRITICAL = "critical"  # Never forget
    HIGH = "high"  # Long retention
    MEDIUM = "medium"  # Standard retention
    LOW = "low"  # Short retention
    TEMPORARY = "temporary"  # Very short retention


@dataclass
class MemoryEntry:
    """Represents a memory entry."""

    memory_id: str
    content: Dict[str, Any]
    memory_type: MemoryType
    importance: MemoryImportance = MemoryImportance.MEDIUM
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    tags: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    retention_score: float = 1.0
    constitutional_protected: bool = False


class MemorySystem:
    """
    Comprehensive Memory System for NFCS.

    Manages multiple types of memory with constitutional protection,
    intelligent retention policies, and efficient retrieval mechanisms.
    """

    def __init__(self, max_memory_size: int = 100000):
        """Initialize Memory System."""
        self.max_memory_size = max_memory_size

        # Memory stores by type
        self.episodic_memory: Dict[str, MemoryEntry] = {}
        self.semantic_memory: Dict[str, MemoryEntry] = {}
        self.procedural_memory: Dict[str, MemoryEntry] = {}
        self.working_memory: deque = deque(maxlen=1000)
        self.constitutional_memory: Dict[str, MemoryEntry] = {}

        # Memory management
        self.memory_index: Dict[str, str] = {}  # Tag -> Memory ID mapping
        self.retention_policies: Dict[MemoryImportance, int] = {
            MemoryImportance.CRITICAL: -1,  # Never expire
            MemoryImportance.HIGH: 365 * 24 * 3600,  # 1 year
            MemoryImportance.MEDIUM: 30 * 24 * 3600,  # 30 days
            MemoryImportance.LOW: 7 * 24 * 3600,  # 7 days
            MemoryImportance.TEMPORARY: 3600,  # 1 hour
        }

        # Statistics
        self.stats = {
            "total_memories": 0,
            "memories_created": 0,
            "memories_retrieved": 0,
            "memories_forgotten": 0,
            "constitutional_protections": 0,
        }

        logger.info("Memory System initialized")

    def store_memory(
        self,
        content: Dict[str, Any],
        memory_type: MemoryType,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        tags: Set[str] = None,
        context: Dict[str, Any] = None,
        constitutional_protected: bool = False,
    ) -> str:
        """Store a new memory entry."""

        memory_id = f"{memory_type.value}_{int(time.time() * 1000)}"

        memory_entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or set(),
            context=context or {},
            constitutional_protected=constitutional_protected,
        )

        # Store in appropriate memory system
        if memory_type == MemoryType.EPISODIC:
            self.episodic_memory[memory_id] = memory_entry
        elif memory_type == MemoryType.SEMANTIC:
            self.semantic_memory[memory_id] = memory_entry
        elif memory_type == MemoryType.PROCEDURAL:
            self.procedural_memory[memory_id] = memory_entry
        elif memory_type == MemoryType.CONSTITUTIONAL:
            self.constitutional_memory[memory_id] = memory_entry
            self.stats["constitutional_protections"] += 1
        elif memory_type == MemoryType.WORKING:
            self.working_memory.append(memory_entry)

        # Update index
        for tag in memory_entry.tags:
            self.memory_index[tag] = memory_id

        # Update statistics
        self.stats["memories_created"] += 1
        self.stats["total_memories"] += 1

        # Perform memory consolidation if needed
        if self.stats["total_memories"] > self.max_memory_size:
            self._consolidate_memory()

        logger.debug(f"Memory stored: {memory_id} ({memory_type.value})")

        return memory_id

    def retrieve_memory(
        self,
        query: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Retrieve memories based on query criteria."""

        candidate_memories = []

        # Collect memories from appropriate stores
        if memory_type is None:
            all_memories = (
                list(self.episodic_memory.values())
                + list(self.semantic_memory.values())
                + list(self.procedural_memory.values())
                + list(self.constitutional_memory.values())
                + list(self.working_memory)
            )
        else:
            if memory_type == MemoryType.EPISODIC:
                all_memories = list(self.episodic_memory.values())
            elif memory_type == MemoryType.SEMANTIC:
                all_memories = list(self.semantic_memory.values())
            elif memory_type == MemoryType.PROCEDURAL:
                all_memories = list(self.procedural_memory.values())
            elif memory_type == MemoryType.CONSTITUTIONAL:
                all_memories = list(self.constitutional_memory.values())
            elif memory_type == MemoryType.WORKING:
                all_memories = list(self.working_memory)
            else:
                all_memories = []

        # Filter by tags if provided
        if tags:
            all_memories = [m for m in all_memories if tags.intersection(m.tags)]

        # Simple content matching if query provided
        if query:
            query_lower = query.lower()
            all_memories = [m for m in all_memories if query_lower in str(m.content).lower()]

        # Sort by relevance (recency + importance + access frequency)
        for memory in all_memories:
            recency_score = 1.0 / (1.0 + (time.time() - memory.last_accessed) / 86400)  # Days
            importance_score = {
                "critical": 1.0,
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4,
                "temporary": 0.2,
            }[memory.importance.value]
            frequency_score = min(1.0, memory.access_count / 10.0)

            memory.retention_score = (recency_score + importance_score + frequency_score) / 3.0

        # Sort and limit
        all_memories.sort(key=lambda m: m.retention_score, reverse=True)
        retrieved_memories = all_memories[:limit]

        # Update access statistics
        for memory in retrieved_memories:
            memory.last_accessed = time.time()
            memory.access_count += 1

        self.stats["memories_retrieved"] += len(retrieved_memories)

        return retrieved_memories

    def _consolidate_memory(self):
        """Consolidate memory by removing low-importance, old memories."""
        current_time = time.time()
        memories_to_remove = []

        # Check all memory stores for consolidation candidates
        all_stores = [
            (self.episodic_memory, "episodic"),
            (self.semantic_memory, "semantic"),
            (self.procedural_memory, "procedural"),
        ]

        for memory_store, store_name in all_stores:
            for memory_id, memory in memory_store.items():
                # Skip constitutionally protected memories
                if memory.constitutional_protected:
                    continue

                # Check retention policy
                retention_time = self.retention_policies[memory.importance]
                if retention_time > 0:  # -1 means never expire
                    age = current_time - memory.created_at
                    if age > retention_time:
                        memories_to_remove.append((memory_store, memory_id))

        # Remove expired memories
        for memory_store, memory_id in memories_to_remove:
            del memory_store[memory_id]
            self.stats["memories_forgotten"] += 1
            self.stats["total_memories"] -= 1

        if memories_to_remove:
            logger.info(f"Memory consolidation: removed {len(memories_to_remove)} expired memories")

    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory system status."""
        return {
            "statistics": self.stats.copy(),
            "memory_distribution": {
                "episodic": len(self.episodic_memory),
                "semantic": len(self.semantic_memory),
                "procedural": len(self.procedural_memory),
                "working": len(self.working_memory),
                "constitutional": len(self.constitutional_memory),
            },
            "memory_utilization": self.stats["total_memories"] / self.max_memory_size,
            "protection_rate": self.stats["constitutional_protections"]
            / max(1, self.stats["total_memories"]),
        }


class MemoryModule:
    """Main Memory Module interface for NFCS integration."""

    def __init__(self, max_memory_size: int = 100000):
        """Initialize Memory Module."""
        self.memory_system = MemorySystem(max_memory_size)
        self.module_id = "MEMORY_MODULE_v1.0"
        self.active = True

        logger.info("Memory Module initialized")

    def remember(self, content: Dict[str, Any], **kwargs) -> str:
        """Store a memory with constitutional protection."""
        # Default to episodic memory with constitutional protection for important events
        memory_type = kwargs.get("memory_type", MemoryType.EPISODIC)
        importance = kwargs.get("importance", MemoryImportance.MEDIUM)
        constitutional_protected = kwargs.get(
            "constitutional_protected",
            importance in [MemoryImportance.CRITICAL, MemoryImportance.HIGH],
        )

        return self.memory_system.store_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            constitutional_protected=constitutional_protected,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["memory_type", "importance", "constitutional_protected"]
            },
        )

    def recall(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve memories based on query."""
        memories = self.memory_system.retrieve_memory(query=query, **kwargs)
        return [
            {"id": m.memory_id, "content": m.content, "type": m.memory_type.value} for m in memories
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "module_id": self.module_id,
            "active": self.active,
            "memory_status": self.memory_system.get_memory_status(),
        }
