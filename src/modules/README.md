# Cognitive Modules & ESC System

## Overview

The Modules directory contains the complete modular architecture of the Neural Field Control System (NFCS), implementing both cognitive processing modules and the advanced Echo-Semantic Converter (ESC) system for token-level processing.

**Architecture**: Hybrid cognitive system with 5 major cognitive modules + ESC processing pipeline, totaling 80,000+ lines of specialized code.

## 🧠 Modular Architecture

```
Cognitive Architecture
         ↓
┌─────────────────────────────────────┐
│          Orchestrator               │ 
│       Coordination Layer            │
└─────────────────────────────────────┘
         ↓
┌─────────────┬─────────────────────────┐
│  Cognitive  │     ESC System          │
│  Modules    │   (Token Processing)    │
├─────────────┼─────────────────────────┤
│Constitution │ • Token Processor       │
│Boundary     │ • Attention Mechanisms  │
│Memory       │ • Semantic Fields       │
│Meta-Reflect.│ • Constitutional Filter │
│Freedom      │ • Adaptive Vocabulary   │
└─────────────┴─────────────────────────┘
```

## 📁 Directory Structure

```
modules/
├── __init__.py                   # Module system initialization
├── cognitive/                    # 🧠 Core cognitive architecture (5 modules)
│   ├── constitution/             # 🏛️ Constitutional framework (47k+ lines)
│   ├── boundary/                 # 🛡️ Boundary management and safety
│   ├── memory/                   # 💾 Multi-type memory systems
│   ├── meta_reflection/          # 🔄 Self-monitoring and adaptation (21k+ lines)
│   └── freedom/                  # 🕊️ Autonomous decision making (25k+ lines)
└── esc/                          # 🎭 Echo-Semantic Converter (6 components)
    ├── esc_core.py              # Core ESC processing (33k+ lines)
    ├── token_processor.py       # Advanced token processing
    ├── attention_mechanisms.py  # Multi-scale attention systems
    ├── semantic_fields.py       # Semantic field analysis
    ├── constitutional_filter.py # Safety and compliance filtering
    └── adaptive_vocabulary.py   # Dynamic vocabulary learning
```

## 🧠 Cognitive Modules

### 1. **Constitutional Framework** (`cognitive/constitution/`)

**Purpose**: Provides constitutional safety, policy management, and compliance monitoring for the entire NFCS.

**Key Components**:
- **Constitutional Core**: 47,000+ lines of policy management logic
- **Policy Manager**: Dynamic policy creation, modification, and enforcement
- **Compliance Monitor**: Real-time violation detection and response
- **Governance System**: Multi-stakeholder consensus mechanisms

**Features**:
- Hierarchical policy structures with inheritance
- Real-time constitutional compliance checking
- Violation tracking and automated response
- Multi-level enforcement (advisory → mandatory)
- Democratic governance with stakeholder voting

**Usage Example**:
```python
from modules.cognitive.constitution.constitution_core import ConstitutionalFramework

# Initialize constitutional framework
constitution = ConstitutionalFramework()
await constitution.initialize()

# Add fundamental safety policy
safety_policy = {
    "policy_id": "SAFETY_001",
    "title": "Core Safety Principles",
    "principles": [
        "Do no harm to humans",
        "Protect user privacy",
        "Maintain system stability"
    ],
    "enforcement_level": 1.0
}

await constitution.add_policy(safety_policy)

# Check compliance
decision = {"action": "process_data", "context": user_context}
compliance = await constitution.check_compliance(decision)

if compliance["compliant"]:
    proceed_with_action()
else:
    handle_violation(compliance["violations"])
```

### 2. **Boundary Management** (`cognitive/boundary/`)

**Purpose**: Dynamic boundary detection, management, and adaptation for safe system operation.

**Key Features**:
- Adaptive boundary detection using topological analysis
- Dynamic safety perimeter adjustment
- Multi-dimensional boundary management (ethical, technical, legal)
- Integration with constitutional framework for policy enforcement

**Usage Example**:
```python
from modules.cognitive.boundary.boundary_core import BoundaryModule

boundary = BoundaryModule()
await boundary.initialize()

# Define operational boundaries
boundaries = {
    "safety": {"risk_threshold": 0.1, "auto_adjust": True},
    "ethics": {"principles": ["autonomy", "beneficence", "non-maleficence"]},
    "legal": {"jurisdiction": "EU", "compliance_frameworks": ["GDPR", "AI_ACT"]}
}

await boundary.set_boundaries(boundaries)

# Check if operation is within bounds
operation = {"type": "data_processing", "risk_level": 0.05}
within_bounds = await boundary.check_boundaries(operation)
```

### 3. **Memory System** (`cognitive/memory/`)

**Purpose**: Multi-type memory architecture supporting episodic, semantic, and procedural memory.

**Memory Types**:
- **Episodic Memory**: Time-ordered event sequences
- **Semantic Memory**: Factual knowledge and relationships
- **Procedural Memory**: Learned skills and procedures  
- **Working Memory**: Active processing buffer
- **Constitutional Memory**: Policy and compliance history

**Usage Example**:
```python
from modules.cognitive.memory.memory_core import MemoryModule

memory = MemoryModule()
await memory.initialize()

# Store episodic memory
episode = {
    "timestamp": time.time(),
    "event": "user_interaction",
    "context": interaction_data,
    "outcome": "successful_response"
}
await memory.store_episodic(episode)

# Retrieve relevant memories
query = {"topic": "similar_interactions", "timeframe": "last_week"}
relevant_memories = await memory.recall(query)

# Store procedural knowledge
procedure = {
    "name": "handle_user_question",
    "steps": [...],
    "success_rate": 0.95
}
await memory.store_procedural(procedure)
```

### 4. **Meta-Reflection** (`cognitive/meta_reflection/`)

**Purpose**: Self-monitoring, adaptation, and meta-cognitive awareness for system improvement.

**Key Features** (21,000+ lines):
- Real-time performance monitoring
- Adaptive strategy selection
- Meta-cognitive reasoning about system state
- Learning from errors and successes
- Dynamic module configuration adjustment

**Usage Example**:
```python
from modules.cognitive.meta_reflection.meta_reflection_core import MetaReflectionModule

meta_reflection = MetaReflectionModule()
await meta_reflection.initialize()

# Monitor system performance
performance_data = await meta_reflection.assess_performance()
print(f"System efficiency: {performance_data['efficiency']:.2f}")
print(f"Error rate: {performance_data['error_rate']:.3f}")

# Trigger adaptation if needed
if performance_data['efficiency'] < 0.8:
    adaptations = await meta_reflection.generate_adaptations()
    await meta_reflection.apply_adaptations(adaptations)
```

### 5. **Freedom Module** (`cognitive/freedom/`)

**Purpose**: Autonomous decision-making capabilities with constitutional constraints.

**Key Features** (25,000+ lines):
- Autonomous decision trees
- Constitutional constraint satisfaction
- Creative problem-solving capabilities
- Risk assessment and mitigation
- Human-AI collaboration protocols

**Usage Example**:
```python
from modules.cognitive.freedom.freedom_core import FreedomModule

freedom = FreedomModule()
await freedom.initialize()

# Autonomous decision making
problem = {
    "type": "resource_allocation",
    "constraints": ["budget_limit", "time_constraint", "safety_requirements"],
    "objectives": ["maximize_efficiency", "minimize_risk"]
}

decision = await freedom.make_autonomous_decision(problem)

# Verify constitutional compliance
if decision["constitutional_compliant"]:
    execute_decision(decision["actions"])
else:
    escalate_to_human(decision["conflicts"])
```

## 🎭 Echo-Semantic Converter (ESC) System

The ESC system implements advanced token-level processing with semantic anchoring and constitutional filtering.

### Core Components

### 1. **ESC Core** (`esc/esc_core.py`)

**Purpose**: Central coordination for all ESC processing (33,000+ lines of code).

**Key Features**:
- Token-level semantic processing
- Multi-scale attention mechanisms
- Semantic field analysis and manipulation
- Integration with cognitive modules
- Real-time processing optimization

### 2. **Token Processor** (`esc/token_processor.py`)

**Purpose**: Advanced tokenization and token-level analysis.

**Features**:
- Adaptive tokenization strategies
- Semantic token classification
- Context-aware token weighting
- Token relationship mapping
- Multi-language support

### 3. **Attention Mechanisms** (`esc/attention_mechanisms.py`)

**Purpose**: Multi-scale attention for semantic processing.

**Attention Types**:
- **Local Attention**: Token-to-token relationships
- **Global Attention**: Document-level semantic coherence
- **Constitutional Attention**: Policy-aware processing
- **Temporal Attention**: Sequential pattern recognition
- **Cross-Modal Attention**: Multi-modal information fusion

### 4. **Semantic Fields** (`esc/semantic_fields.py`)

**Purpose**: Semantic field analysis and manipulation.

**Field Operations**:
- Semantic vector field construction
- Field topology analysis
- Semantic gradient computation
- Field stability assessment
- Cross-field interactions

### 5. **Constitutional Filter** (`esc/constitutional_filter.py`)

**Purpose**: Real-time constitutional compliance filtering for all ESC operations.

**Filtering Stages**:
- Pre-processing safety checks
- Token-level policy validation
- Semantic coherence verification
- Post-processing compliance audit
- Violation flagging and remediation

### 6. **Adaptive Vocabulary** (`esc/adaptive_vocabulary.py`)

**Purpose**: Dynamic vocabulary learning and adaptation.

**Adaptive Features**:
- Context-sensitive vocabulary expansion
- Semantic similarity learning
- Domain-specific vocabulary adaptation
- Vocabulary pruning and optimization
- Cross-domain vocabulary transfer

## ⚡ Quick Start

### Prerequisites
```bash
# Core dependencies for module system
pip install numpy>=1.24.0 scipy>=1.11.0 PyYAML>=6.0
pip install dataclasses-json>=0.5.9 pydantic>=2.0.0

# Optional for enhanced performance
pip install numba>=0.57.0  # For computational optimization
```

### Basic Module Usage
```python
import asyncio
from modules.cognitive.constitution.constitution_core import ConstitutionalFramework
from modules.esc.esc_core import EchoSemanticConverter

async def main():
    # Initialize cognitive modules
    constitution = ConstitutionalFramework()
    await constitution.initialize()
    
    # Initialize ESC system
    esc = EchoSemanticConverter()
    await esc.initialize()
    
    # Process input with constitutional safety
    input_text = "Process this user request safely"
    
    # ESC processing with constitutional filtering
    processed = await esc.process_tokens(
        text=input_text,
        constitutional_filter=True,
        semantic_analysis=True
    )
    
    # Verify constitutional compliance
    compliance = await constitution.verify_output(processed)
    
    if compliance["safe"]:
        return processed["output"]
    else:
        return f"Processing blocked: {compliance['reason']}"

# Run example
result = asyncio.run(main())
print(result)
```

### Module Integration Example
```python
from modules.cognitive import (
    ConstitutionalFramework, BoundaryModule, MemoryModule,
    MetaReflectionModule, FreedomModule
)

class CognitiveSystem:
    def __init__(self):
        self.constitution = ConstitutionalFramework()
        self.boundary = BoundaryModule()
        self.memory = MemoryModule()
        self.meta_reflection = MetaReflectionModule()
        self.freedom = FreedomModule()
    
    async def initialize(self):
        """Initialize all cognitive modules."""
        modules = [
            self.constitution, self.boundary, self.memory,
            self.meta_reflection, self.freedom
        ]
        
        for module in modules:
            await module.initialize()
        
        # Configure inter-module communication
        await self.setup_module_connections()
    
    async def process_complex_decision(self, decision_context):
        """Process complex decision using all cognitive modules."""
        
        # 1. Constitutional check
        constitutional_ok = await self.constitution.check_compliance(decision_context)
        if not constitutional_ok["compliant"]:
            return {"decision": "blocked", "reason": "constitutional_violation"}
        
        # 2. Boundary validation
        within_bounds = await self.boundary.check_boundaries(decision_context)
        if not within_bounds:
            return {"decision": "blocked", "reason": "boundary_violation"}
        
        # 3. Memory retrieval
        relevant_memory = await self.memory.recall(decision_context)
        decision_context["memory"] = relevant_memory
        
        # 4. Meta-reflection assessment
        reflection = await self.meta_reflection.assess_context(decision_context)
        decision_context["meta_analysis"] = reflection
        
        # 5. Autonomous decision (if appropriate)
        if reflection["confidence"] > 0.8:
            decision = await self.freedom.make_autonomous_decision(decision_context)
        else:
            decision = {"decision": "escalate", "reason": "low_confidence"}
        
        # 6. Store outcome in memory
        await self.memory.store_episodic({
            "context": decision_context,
            "decision": decision,
            "timestamp": time.time()
        })
        
        return decision
```

## 🔧 System Requirements

### Computational Requirements

**Per Module Estimates**:
- **Constitutional Framework**: 512 MB RAM, moderate CPU
- **Boundary Module**: 256 MB RAM, low CPU  
- **Memory System**: 1-2 GB RAM (depends on history size)
- **Meta-Reflection**: 512 MB RAM, moderate CPU
- **Freedom Module**: 1 GB RAM, high CPU (decision trees)
- **ESC System**: 2-4 GB RAM, high CPU (token processing)

**Total System Requirements**:
- **Minimum**: 4 GB RAM, quad-core 2.0 GHz CPU
- **Recommended**: 16 GB RAM, octa-core 3.0 GHz CPU
- **Production**: 32 GB RAM, high-performance CPU cluster

### Dependencies by Module

**Constitutional Framework**:
```python
# Core policy management
pydantic >= 2.0.0      # Data validation
dataclasses-json        # Serialization
PyYAML >= 6.0          # Configuration files
```

**Memory System**:
```python
# Memory storage and retrieval
numpy >= 1.24.0        # Efficient arrays
scipy >= 1.11.0        # Similarity calculations  
sqlite3                # Persistent storage (built-in)
```

**ESC System**:
```python
# Token processing and NLP
numpy >= 1.24.0        # Vectorized operations
scipy >= 1.11.0        # Mathematical functions
numba >= 0.57.0        # JIT acceleration (optional)
```

## 🧪 Testing and Validation

### Module Testing Strategy

**Unit Testing**:
```python
# Test individual module functionality
import pytest
from modules.cognitive.constitution.constitution_core import ConstitutionalFramework

@pytest.mark.asyncio
async def test_constitutional_framework():
    constitution = ConstitutionalFramework()
    await constitution.initialize()
    
    # Test policy addition
    policy = {"policy_id": "TEST_001", "title": "Test Policy"}
    result = await constitution.add_policy(policy)
    assert result == True
    
    # Test compliance checking
    decision = {"action": "test_action"}
    compliance = await constitution.check_compliance(decision)
    assert "compliant" in compliance
```

**Integration Testing**:
```python
# Test module interactions
@pytest.mark.asyncio  
async def test_module_integration():
    cognitive_system = CognitiveSystem()
    await cognitive_system.initialize()
    
    # Test complex decision processing
    decision_context = {
        "type": "user_request",
        "content": "test request",
        "risk_level": 0.1
    }
    
    result = await cognitive_system.process_complex_decision(decision_context)
    assert "decision" in result
    assert result["decision"] in ["approved", "blocked", "escalate"]
```

**Performance Testing**:
```python
import time
import asyncio

async def test_processing_speed():
    """Test module processing speed under load."""
    esc = EchoSemanticConverter()
    await esc.initialize()
    
    # Process multiple inputs
    inputs = ["test input " + str(i) for i in range(100)]
    
    start_time = time.time()
    tasks = [esc.process_tokens(text) for text in inputs]
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    processing_time = end_time - start_time
    throughput = len(inputs) / processing_time
    
    print(f"Processed {len(inputs)} inputs in {processing_time:.2f}s")
    print(f"Throughput: {throughput:.1f} inputs/second")
    
    # Performance assertions
    assert throughput > 10  # Should process at least 10 inputs/second
    assert processing_time < 30  # Should complete within 30 seconds
```

### Load Testing

**Concurrent Module Usage**:
```python
async def test_concurrent_modules():
    """Test multiple modules working simultaneously."""
    modules = [
        ConstitutionalFramework(),
        BoundaryModule(), 
        MemoryModule(),
        MetaReflectionModule(),
        FreedomModule()
    ]
    
    # Initialize all modules concurrently
    await asyncio.gather(*[module.initialize() for module in modules])
    
    # Test concurrent processing
    test_contexts = [{"id": i, "data": f"test_{i}"} for i in range(10)]
    
    async def process_with_module(module, context):
        return await module.process(context)
    
    # Run all modules on all contexts concurrently
    tasks = []
    for module in modules:
        for context in test_contexts:
            tasks.append(process_with_module(module, context))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify no exceptions occurred
    exceptions = [r for r in results if isinstance(r, Exception)]
    assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
```

## 🚀 Advanced Configuration

### Module-Specific Configuration

**Constitutional Framework Configuration**:
```yaml
# config/constitution_config.yaml
constitutional_framework:
  enforcement_level: 0.8
  policy_hierarchy: "strict"
  governance:
    voting_mechanism: "consensus"
    stakeholder_weights:
      system: 0.3
      user: 0.4
      developer: 0.3
  
  fundamental_policies:
    - policy_id: "SAFETY_001"
      title: "Core Safety"
      enforcement_level: 1.0
    - policy_id: "PRIVACY_001"
      title: "Privacy Protection"
      enforcement_level: 0.9
```

**ESC System Configuration**:
```yaml
# config/esc_config.yaml
esc_system:
  token_processor:
    max_tokens: 8192
    adaptive_tokenization: true
    multi_language: ["en", "ru", "zh"]
  
  attention_mechanisms:
    local_attention_radius: 5
    global_attention_layers: 3
    constitutional_attention_weight: 0.2
  
  semantic_fields:
    field_dimensions: 512
    stability_threshold: 0.7
    cross_field_interactions: true
  
  constitutional_filter:
    pre_processing_checks: true
    token_level_validation: true
    post_processing_audit: true
```

### Dynamic Configuration
```python
# Runtime configuration updates
async def update_module_config(module, new_config):
    """Update module configuration at runtime."""
    current_config = await module.get_configuration()
    merged_config = {**current_config, **new_config}
    
    # Validate configuration
    validation_result = await module.validate_configuration(merged_config)
    if not validation_result["valid"]:
        raise ValueError(f"Invalid configuration: {validation_result['errors']}")
    
    # Apply configuration
    await module.update_configuration(merged_config)
    
    # Verify application
    updated_config = await module.get_configuration()
    assert updated_config == merged_config
```

## 🤝 Contributing

### Adding New Cognitive Modules

**Module Interface Standard**:
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class CognitiveModuleInterface(ABC):
    """Standard interface for all cognitive modules."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the module."""
        pass
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data."""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current module status."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the module gracefully."""
        pass

class NewCognitiveModule(CognitiveModuleInterface):
    """Template for new cognitive module."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.initialized = False
        self.active = False
    
    async def initialize(self) -> bool:
        # Module-specific initialization logic
        self.initialized = True
        return True
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.initialized:
            raise RuntimeError("Module not initialized")
        
        # Module-specific processing logic
        result = {"processed": True, "data": input_data}
        return result
    
    async def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": self.initialized,
            "active": self.active,
            "config": self.config
        }
    
    async def shutdown(self) -> bool:
        self.active = False
        return True
```

### Code Standards for Modules

**Documentation Requirements**:
```python
class ExampleModule:
    """
    Brief module description.
    
    This module implements [specific functionality] for the NFCS cognitive
    architecture. It provides [key capabilities] and integrates with
    [other modules].
    
    Mathematical Foundation:
        [Any relevant mathematical models or equations]
    
    Performance Characteristics:
        - Memory usage: [typical RAM usage]
        - Processing speed: [operations/second]
        - Scalability: [concurrent operation limits]
    
    Args:
        config: Configuration dictionary with keys:
            - param1 (type): Description of parameter
            - param2 (type): Description of parameter
    
    Example:
        >>> module = ExampleModule(config={"param1": "value"})
        >>> await module.initialize()
        >>> result = await module.process({"input": "data"})
    """
```

**Testing Requirements**:
- Unit tests for all public methods
- Integration tests with other modules
- Performance benchmarks
- Error handling validation
- Configuration validation tests

**Performance Guidelines**:
- Async/await for all I/O operations
- Efficient memory usage (< 1GB per module)
- Response time < 100ms for standard operations
- Support for concurrent operations
- Graceful degradation under resource constraints

## 📚 Documentation

### Module Documentation Structure

Each module should include:
- **README.md**: Module overview and usage
- **API.md**: Detailed API documentation
- **EXAMPLES.md**: Usage examples and tutorials
- **ARCHITECTURE.md**: Internal architecture details
- **PERFORMANCE.md**: Performance characteristics and benchmarks

### Research References

**Cognitive Architecture**:
1. **Anderson, J. R.** (2007). "How Can the Human Mind Occur in the Physical Universe?"
2. **Laird, J. E.** (2012). "The Soar Cognitive Architecture"
3. **Langley, P. et al.** (2009). "Cognitive architectures: Research issues and challenges"

**Constitutional AI**:
1. **Anthropic** (2022). "Constitutional AI: Harmlessness from AI Feedback"
2. **Russell, S.** (2019). "Human Compatible: Artificial Intelligence and the Problem of Control"

**Token Processing**:
1. **Vaswani, A. et al.** (2017). "Attention Is All You Need"
2. **Devlin, J. et al.** (2018). "BERT: Pre-training of Deep Bidirectional Transformers"

---

## Russian Translation / Русский перевод

# Когнитивные модули и система ESC

## Обзор

Директория Modules содержит полную модульную архитектуру Системы управления нейронными полями (NFCS), реализующую как когнитивные модули обработки, так и продвинутую систему Echo-Semantic Converter (ESC) для обработки на уровне токенов.

**Архитектура**: Гибридная когнитивная система с 5 основными когнитивными модулями + конвейер обработки ESC, в общей сложности 80,000+ строк специализированного кода.

## 🧠 Модульная архитектура

```
Когнитивная архитектура
         ↓
┌─────────────────────────────────────┐
│          Оркестратор                │ 
│     Уровень координации             │
└─────────────────────────────────────┘
         ↓
┌─────────────┬─────────────────────────┐
│ Когнитивные │     Система ESC         │
│   модули    │  (Обработка токенов)    │
├─────────────┼─────────────────────────┤
│Конституция  │ • Обработчик токенов    │
│Границы      │ • Механизмы внимания    │
│Память       │ • Семантические поля    │
│Мета-рефлекс.│ • Конституционный фильтр│
│Свобода      │ • Адаптивный словарь    │
└─────────────┴─────────────────────────┘
```

## 📁 Структура директорий

```
modules/
├── __init__.py                   # Инициализация системы модулей
├── cognitive/                    # 🧠 Базовая когнитивная архитектура (5 модулей)
│   ├── constitution/             # 🏛️ Конституционная структура (47k+ строк)
│   ├── boundary/                 # 🛡️ Управление границами и безопасность
│   ├── memory/                   # 💾 Многотипные системы памяти
│   ├── meta_reflection/          # 🔄 Самомониторинг и адаптация (21k+ строк)
│   └── freedom/                  # 🕊️ Автономное принятие решений (25k+ строк)
└── esc/                          # 🎭 Echo-Semantic Converter (6 компонентов)
    ├── esc_core.py              # Основная обработка ESC (33k+ строк)
    ├── token_processor.py       # Продвинутая обработка токенов
    ├── attention_mechanisms.py  # Многомасштабные системы внимания
    ├── semantic_fields.py       # Анализ семантических полей
    ├── constitutional_filter.py # Фильтрация безопасности и соответствия
    └── adaptive_vocabulary.py   # Динамическое изучение словаря
```

---

*This README provides comprehensive documentation for the NFCS modular system, covering both cognitive modules and the ESC processing pipeline. Each module implements specialized cognitive functions while maintaining constitutional safety and inter-module coordination.*

*Данный README предоставляет исчерпывающую документацию для модульной системы NFCS, охватывающую как когнитивные модули, так и конвейер обработки ESC. Каждый модуль реализует специализированные когнитивные функции, поддерживая конституционную безопасность и межмодульную координацию.*