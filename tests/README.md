# Testing Suite - NFCS Validation Framework

## Overview

This directory contains the comprehensive testing suite for the Neural Field Control System (NFCS), implementing multi-layered validation from unit tests to full system integration testing. The testing framework ensures reliability, performance, and safety compliance across all system components.

**Testing Coverage**: 5,000+ lines of test code covering unit, integration, performance, and validation testing.

## 🧪 Testing Architecture

```
Testing Pyramid
      ↗
┌─────────────────┐
│ System/E2E Tests│ ← Full system validation
├─────────────────┤
│Integration Tests│ ← Module interactions  
├─────────────────┤
│  Unit Tests     │ ← Individual components
└─────────────────┘
      ↓
   Validation Framework
```

## 📁 Test Directory Structure

```
tests/
├── __init__.py                    # Test package initialization
├── test_core/                     # 🔬 Mathematical core testing
│   ├── test_cgl_solver.py         # CGL equation solver validation
│   ├── test_kuramoto_solver.py    # Kuramoto model testing
│   ├── test_enhanced_metrics.py   # Metrics calculation validation
│   └── test_mathematical_models.py # Mathematical accuracy tests
├── test_orchestrator/             # ⚙️ Orchestrator system testing
│   ├── test_nfcs_orchestrator.py  # Main orchestrator functionality
│   ├── test_managers.py           # Manager component testing
│   ├── test_coordinators.py       # Coordination system validation
│   └── test_controllers.py        # Controller system testing
├── test_modules/                  # 🧠 Cognitive module testing
│   ├── test_module_interfaces.py  # Module interface compliance
│   ├── test_module_integration.py # Inter-module communication
│   └── test_module_performance.py # Module performance benchmarks
├── test_cognitive/                # 🏛️ Cognitive system testing
│   ├── test_constitution.py       # Constitutional framework validation
│   ├── test_boundary.py          # Boundary management testing
│   ├── test_memory.py            # Memory system validation
│   ├── test_meta_reflection.py   # Meta-reflection testing
│   └── test_freedom.py           # Freedom module testing
├── test_esc/                     # 🎭 ESC system testing
│   ├── test_esc_core.py          # Core ESC functionality
│   ├── test_token_processor.py   # Token processing validation
│   ├── test_attention.py         # Attention mechanism testing
│   ├── test_semantic_fields.py   # Semantic field validation
│   └── test_constitutional_filter.py # Safety filtering tests
└── validation/                   # 📊 System validation and benchmarks
    ├── performance_benchmarks.py  # Performance testing suite
    ├── safety_validation.py      # Safety and compliance testing
    ├── integration_scenarios.py  # Complex integration testing
    └── mathematical_validation.py # Mathematical accuracy validation
```

## 🎯 Testing Categories

### 1. **Unit Tests**

**Purpose**: Test individual components in isolation.

**Coverage Areas**:
- Mathematical solver accuracy
- Module initialization and shutdown
- Configuration validation
- Error handling and edge cases
- Data structure integrity

**Example Unit Test**:
```python
import pytest
import numpy as np
from src.core.cgl_solver import CGLSolver

class TestCGLSolver:
    """Unit tests for Complex Ginzburg-Landau solver."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.solver = CGLSolver(
            grid_size=(32, 32),
            domain_size=(5.0, 5.0),
            dt=0.01
        )
    
    def test_initialization(self):
        """Test solver initialization."""
        assert self.solver.grid_size == (32, 32)
        assert self.solver.dt == 0.01
        assert self.solver.initialized == True
    
    def test_field_evolution(self):
        """Test field evolution accuracy."""
        # Initial condition: uniform field
        psi_initial = np.ones((32, 32), dtype=complex)
        
        # Evolve for one step
        psi_evolved = self.solver.step(psi_initial)
        
        # Verify field properties
        assert psi_evolved.shape == (32, 32)
        assert np.isfinite(psi_evolved).all()
        assert np.abs(psi_evolved).max() < 10.0  # No blow-up
    
    def test_energy_conservation(self):
        """Test energy conservation property."""
        psi = np.random.complex128((32, 32))
        initial_energy = self.solver.calculate_energy(psi)
        
        # Evolve for multiple steps
        for _ in range(10):
            psi = self.solver.step(psi)
        
        final_energy = self.solver.calculate_energy(psi)
        energy_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        
        # Energy should be approximately conserved
        assert energy_drift < 0.05
    
    def test_boundary_conditions(self):
        """Test periodic boundary conditions."""
        psi = np.random.complex128((32, 32))
        psi_evolved = self.solver.step(psi)
        
        # Check continuity at boundaries
        left_boundary = psi_evolved[:, 0]
        right_boundary = psi_evolved[:, -1]
        
        # Should be approximately equal for periodic boundaries
        np.testing.assert_allclose(left_boundary, right_boundary, rtol=1e-10)
    
    @pytest.mark.parametrize("dt", [0.001, 0.01, 0.1])
    def test_stability_different_timesteps(self, dt):
        """Test numerical stability with different time steps."""
        solver = CGLSolver(grid_size=(16, 16), dt=dt)
        psi = np.random.complex128((16, 16))
        
        # Should remain stable for reasonable time steps
        for _ in range(50):
            psi = solver.step(psi)
            max_amplitude = np.abs(psi).max()
            assert max_amplitude < 100.0  # No exponential blow-up
```

### 2. **Integration Tests**

**Purpose**: Test component interactions and system-level behavior.

**Coverage Areas**:
- Module-to-module communication
- Orchestrator coordination
- Data flow validation
- System initialization and shutdown
- Configuration consistency

**Example Integration Test**:
```python
import pytest
import asyncio
from src.orchestrator.nfcs_orchestrator import create_orchestrator, create_default_config
from src.modules.cognitive.constitution.constitution_core import ConstitutionalFramework

@pytest.mark.asyncio
class TestSystemIntegration:
    """Integration tests for full system functionality."""
    
    async def test_orchestrator_module_integration(self):
        """Test orchestrator integration with cognitive modules."""
        # Create system configuration
        config = create_default_config()
        config.operational_mode = "supervised"
        config.safety_level = 0.8
        
        # Initialize orchestrator
        orchestrator = create_orchestrator(config)
        
        try:
            # Start system
            startup_result = await orchestrator.start()
            assert startup_result == True
            
            # Verify all modules are initialized
            module_status = await orchestrator.get_module_status()
            assert len(module_status) > 0
            
            for module_name, status in module_status.items():
                assert status["initialized"] == True
                assert status["active"] == True
            
            # Test processing pipeline
            test_input = {
                "text": "Test input for processing",
                "metadata": {"source": "integration_test"}
            }
            
            result = await orchestrator.process_input(test_input)
            
            # Verify result structure
            assert "processed" in result
            assert "constitutional_compliant" in result
            assert "timestamp" in result
            
            # Test emergency shutdown
            emergency_result = await orchestrator.emergency_shutdown()
            assert emergency_result == True
            
        finally:
            # Cleanup
            await orchestrator.stop()
    
    async def test_constitutional_integration(self):
        """Test constitutional framework integration across modules."""
        constitution = ConstitutionalFramework()
        await constitution.initialize()
        
        # Add test policy
        test_policy = {
            "policy_id": "TEST_INTEGRATION_001",
            "title": "Integration Test Policy",
            "enforcement_level": 0.9,
            "scope": ["all_modules"]
        }
        
        await constitution.add_policy(test_policy)
        
        # Test cross-module compliance checking
        modules = ["boundary", "memory", "freedom", "meta_reflection"]
        
        for module_name in modules:
            decision_context = {
                "module": module_name,
                "action": "process_data",
                "risk_level": 0.1
            }
            
            compliance = await constitution.check_compliance(decision_context)
            assert compliance["compliant"] == True
            assert "TEST_INTEGRATION_001" in [p["policy_id"] for p in compliance["evaluated_policies"]]
    
    async def test_data_flow_validation(self):
        """Test data flow through entire system pipeline."""
        config = create_default_config()
        orchestrator = create_orchestrator(config)
        
        await orchestrator.start()
        
        try:
            # Input data
            input_data = {
                "text": "Complex multi-modal input for comprehensive testing",
                "context": {
                    "user_preferences": {"safety_level": "high"},
                    "task_type": "analysis",
                    "domain": "general"
                },
                "metadata": {
                    "timestamp": time.time(),
                    "source": "integration_test",
                    "priority": "normal"
                }
            }
            
            # Process through full pipeline
            result = await orchestrator.process_comprehensive(input_data)
            
            # Validate result completeness
            required_fields = [
                "esc_processing", "cognitive_analysis", 
                "constitutional_compliance", "final_output"
            ]
            
            for field in required_fields:
                assert field in result, f"Missing required field: {field}"
            
            # Validate processing chain
            assert result["esc_processing"]["tokens_processed"] > 0
            assert result["cognitive_analysis"]["modules_engaged"] > 0
            assert result["constitutional_compliance"]["policies_checked"] > 0
            
        finally:
            await orchestrator.stop()
```

### 3. **Performance Tests**

**Purpose**: Validate system performance under various loads and conditions.

**Coverage Areas**:
- Response time benchmarks
- Memory usage profiling
- Concurrent operation handling
- Resource utilization monitoring
- Scalability validation

**Example Performance Test**:
```python
import pytest
import asyncio
import time
import psutil
import numpy as np

class TestPerformance:
    """Performance testing suite."""
    
    @pytest.mark.performance
    async def test_orchestrator_response_time(self):
        """Test orchestrator response time under standard load."""
        config = create_default_config()
        orchestrator = create_orchestrator(config)
        await orchestrator.start()
        
        try:
            # Warm up system
            for _ in range(5):
                await orchestrator.process_input({"text": "warmup"})
            
            # Measure response times
            response_times = []
            
            for i in range(100):
                input_data = {"text": f"Performance test input {i}"}
                
                start_time = time.time()
                result = await orchestrator.process_input(input_data)
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                
                # Verify successful processing
                assert result["processed"] == True
            
            # Calculate statistics
            mean_response_time = np.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            
            print(f"Response time statistics:")
            print(f"  Mean: {mean_response_time:.3f}s")
            print(f"  95th percentile: {p95_response_time:.3f}s")
            print(f"  99th percentile: {p99_response_time:.3f}s")
            
            # Performance assertions
            assert mean_response_time < 0.1, f"Mean response time too high: {mean_response_time:.3f}s"
            assert p95_response_time < 0.2, f"P95 response time too high: {p95_response_time:.3f}s"
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.performance
    async def test_memory_usage(self):
        """Test memory usage during extended operation."""
        config = create_default_config()
        orchestrator = create_orchestrator(config)
        
        # Measure initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        await orchestrator.start()
        
        try:
            # Process many inputs
            for i in range(1000):
                input_data = {
                    "text": f"Extended operation test input {i} with some additional content to test memory management",
                    "metadata": {"iteration": i}
                }
                
                result = await orchestrator.process_input(input_data)
                
                # Periodic memory checks
                if i % 100 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_growth = current_memory - initial_memory
                    
                    print(f"Iteration {i}: Memory usage = {current_memory:.1f} MB (growth: +{memory_growth:.1f} MB)")
                    
                    # Memory should not grow excessively
                    assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.1f} MB"
            
            # Final memory check
            final_memory = process.memory_info().rss / 1024 / 1024
            total_growth = final_memory - initial_memory
            
            print(f"Total memory growth: {total_growth:.1f} MB")
            assert total_growth < 1000, f"Memory leak detected: {total_growth:.1f} MB growth"
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.performance
    async def test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        config = create_default_config()
        orchestrator = create_orchestrator(config)
        await orchestrator.start()
        
        try:
            # Create concurrent processing tasks
            num_concurrent = 50
            
            async def process_single_input(input_id):
                input_data = {
                    "text": f"Concurrent processing test {input_id}",
                    "metadata": {"concurrent_id": input_id}
                }
                
                start_time = time.time()
                result = await orchestrator.process_input(input_data)
                end_time = time.time()
                
                return {
                    "input_id": input_id,
                    "success": result["processed"],
                    "response_time": end_time - start_time
                }
            
            # Execute all tasks concurrently
            start_time = time.time()
            tasks = [process_single_input(i) for i in range(num_concurrent)]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # Analyze results
            successful_results = [r for r in results if r["success"]]
            response_times = [r["response_time"] for r in successful_results]
            
            print(f"Concurrent processing results:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Successful: {len(successful_results)}/{num_concurrent}")
            print(f"  Mean response time: {np.mean(response_times):.3f}s")
            print(f"  Throughput: {len(successful_results)/total_time:.1f} req/s")
            
            # Performance assertions
            assert len(successful_results) == num_concurrent, "Some concurrent requests failed"
            assert total_time < 10.0, f"Concurrent processing took too long: {total_time:.2f}s"
            
            throughput = len(successful_results) / total_time
            assert throughput > 10.0, f"Throughput too low: {throughput:.1f} req/s"
            
        finally:
            await orchestrator.stop()
```

### 4. **Validation Tests**

**Purpose**: Validate system correctness, safety, and compliance.

**Coverage Areas**:
- Mathematical accuracy validation
- Constitutional compliance verification
- Safety protocol testing
- Error recovery validation
- Data integrity checks

**Example Validation Test**:
```python
import pytest
import numpy as np
from src.validation.mathematical_validation import MathematicalValidator
from src.validation.safety_validation import SafetyValidator

class TestValidation:
    """System validation testing."""
    
    def test_mathematical_accuracy(self):
        """Validate mathematical computations against known solutions."""
        validator = MathematicalValidator()
        
        # Test CGL solver against analytical solution
        analytical_solution = validator.cgl_traveling_wave_solution(
            amplitude=1.0, velocity=0.5, width=2.0
        )
        
        numerical_solution = validator.test_cgl_solver_accuracy(
            initial_condition=analytical_solution,
            evolution_time=1.0,
            grid_size=(64, 64)
        )
        
        # Calculate error
        error = validator.calculate_l2_error(analytical_solution, numerical_solution)
        print(f"CGL solver L2 error: {error:.6f}")
        
        # Accuracy assertion
        assert error < 0.01, f"Mathematical accuracy insufficient: L2 error = {error:.6f}"
        
        # Test Kuramoto synchronization
        sync_result = validator.test_kuramoto_synchronization(
            N=20, coupling_strength=3.0, evolution_time=10.0
        )
        
        final_sync_order = sync_result["final_synchronization_order"]
        print(f"Kuramoto synchronization order: {final_sync_order:.3f}")
        
        assert final_sync_order > 0.9, f"Synchronization insufficient: {final_sync_order:.3f}"
    
    @pytest.mark.asyncio
    async def test_constitutional_compliance(self):
        """Validate constitutional compliance across all scenarios."""
        validator = SafetyValidator()
        await validator.initialize()
        
        # Test scenarios with different risk levels
        test_scenarios = [
            {"description": "Safe operation", "risk_level": 0.1, "expected_outcome": "approved"},
            {"description": "Moderate risk", "risk_level": 0.5, "expected_outcome": "conditional"},
            {"description": "High risk", "risk_level": 0.9, "expected_outcome": "blocked"},
            {"description": "Prohibited action", "risk_level": 1.0, "expected_outcome": "blocked"}
        ]
        
        for scenario in test_scenarios:
            decision_context = {
                "action": "test_action",
                "risk_level": scenario["risk_level"],
                "description": scenario["description"]
            }
            
            compliance_result = await validator.validate_constitutional_compliance(decision_context)
            
            actual_outcome = compliance_result["decision"]
            expected_outcome = scenario["expected_outcome"]
            
            assert actual_outcome == expected_outcome, \
                f"Constitutional compliance failed for '{scenario['description']}': expected {expected_outcome}, got {actual_outcome}"
    
    async def test_safety_protocols(self):
        """Test emergency safety protocols."""
        validator = SafetyValidator()
        await validator.initialize()
        
        # Test emergency shutdown
        emergency_scenarios = [
            "constitutional_violation",
            "system_overload", 
            "security_breach",
            "human_override_request"
        ]
        
        for scenario in emergency_scenarios:
            emergency_result = await validator.test_emergency_protocol(scenario)
            
            assert emergency_result["protocol_activated"] == True, \
                f"Emergency protocol not activated for scenario: {scenario}"
            
            assert emergency_result["system_safe_state"] == True, \
                f"System not in safe state after emergency protocol: {scenario}"
            
            assert emergency_result["response_time"] < 1.0, \
                f"Emergency response too slow for scenario {scenario}: {emergency_result['response_time']:.3f}s"
    
    async def test_data_integrity(self):
        """Test data integrity throughout system pipeline."""
        validator = SafetyValidator()
        await validator.initialize()
        
        # Test data with various integrity characteristics
        test_datasets = [
            {"name": "clean_data", "corruption_level": 0.0},
            {"name": "minor_noise", "corruption_level": 0.1},
            {"name": "significant_noise", "corruption_level": 0.3},
            {"name": "corrupted_data", "corruption_level": 0.8}
        ]
        
        for dataset in test_datasets:
            integrity_result = await validator.test_data_integrity(dataset)
            
            if dataset["corruption_level"] < 0.5:
                # Should process successfully
                assert integrity_result["processing_successful"] == True
                assert integrity_result["output_integrity"] > 0.8
            else:
                # Should detect corruption and handle appropriately  
                assert integrity_result["corruption_detected"] == True
                assert integrity_result["safety_measures_activated"] == True
```

## ⚡ Running Tests

### Prerequisites
```bash
# Install testing dependencies
pip install pytest>=7.4.0 pytest-cov>=4.1.0 pytest-asyncio>=0.21.0
pip install pytest-benchmark>=4.0.0 pytest-xdist>=3.0.0

# Optional: For performance monitoring
pip install psutil>=5.9.0 memory-profiler>=0.60.0
```

### Basic Test Execution

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_core/ -v                    # Mathematical core tests
pytest tests/test_orchestrator/ -v            # Orchestrator tests
pytest tests/test_modules/ -v                 # Module tests
pytest tests/validation/ -v                   # Validation tests

# Run with coverage reporting
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run performance tests only
pytest tests/ -m performance -v

# Run tests in parallel
pytest tests/ -n auto                         # Auto-detect CPU cores
pytest tests/ -n 4                           # Use 4 parallel workers
```

### Advanced Test Options

```bash
# Run with detailed output and timing
pytest tests/ -v --tb=long --durations=10

# Run specific test file
pytest tests/test_core/test_cgl_solver.py -v

# Run specific test method
pytest tests/test_core/test_cgl_solver.py::TestCGLSolver::test_energy_conservation -v

# Run with custom markers
pytest tests/ -m "not slow" -v               # Skip slow tests
pytest tests/ -m "integration or performance" -v  # Run integration and performance tests

# Generate detailed HTML coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html                       # View coverage report
```

### Continuous Integration Configuration

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    
    - name: Run unit tests
      run: pytest tests/test_core tests/test_modules -v --cov=src
    
    - name: Run integration tests
      run: pytest tests/test_orchestrator tests/test_cognitive -v
    
    - name: Run validation tests
      run: pytest tests/validation -v
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
```

## 🔧 Test Configuration

### pytest Configuration

```ini
# pytest.ini
[tool:pytest]
minversion = 7.0
addopts = 
    -ra 
    --strict-markers 
    --strict-config 
    --cov=src
    --cov-branch
    --cov-report=term-missing
    --cov-report=html
testpaths = tests
markers =
    unit: Unit tests
    integration: Integration tests  
    performance: Performance tests
    slow: Slow-running tests
    validation: System validation tests
    mathematical: Mathematical accuracy tests
    safety: Safety and security tests

asyncio_mode = auto
```

### Custom Test Configuration

```python
# tests/conftest.py
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_config_dir():
    """Create temporary directory for test configurations."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
async def initialized_orchestrator():
    """Provide fully initialized orchestrator for testing."""
    from src.orchestrator.nfcs_orchestrator import create_orchestrator, create_default_config
    
    config = create_default_config()
    config.operational_mode = "test"
    orchestrator = create_orchestrator(config)
    
    await orchestrator.start()
    yield orchestrator
    await orchestrator.stop()

@pytest.fixture
def sample_test_data():
    """Provide sample test data for various test scenarios."""
    return {
        "simple_text": "This is a simple test input.",
        "complex_text": "This is a more complex test input with multiple sentences. It contains various punctuation marks! And questions?",
        "multilingual": "Hello, 世界! Привет, мир!",
        "edge_cases": ["", " ", "\n", "\t", "!@#$%^&*()", "🚀🧠🎯"]
    }
```

## 📊 Test Metrics and Reporting

### Coverage Requirements

**Minimum Coverage Targets**:
- **Overall**: 85% line coverage, 80% branch coverage
- **Core Mathematical**: 95% line coverage, 90% branch coverage
- **Safety/Constitutional**: 98% line coverage, 95% branch coverage
- **Critical Paths**: 100% coverage for safety-critical code paths

### Performance Benchmarks

**Response Time Targets**:
- Unit operations: < 1ms
- Module operations: < 10ms
- System operations: < 100ms
- Complex processing: < 1s

**Memory Usage Targets**:
- Unit tests: < 100MB peak memory
- Integration tests: < 500MB peak memory
- Full system tests: < 2GB peak memory

### Quality Metrics

**Code Quality Gates**:
- All tests must pass
- No critical security vulnerabilities
- Performance benchmarks within acceptable ranges
- Memory leak detection passes
- Constitutional compliance 100%

## 🤝 Contributing to Tests

### Adding New Tests

**Test File Naming Convention**:
```
test_<component_name>.py          # Unit tests
test_<component>_integration.py   # Integration tests
test_<component>_performance.py   # Performance tests
validate_<component>.py           # Validation tests
```

**Test Method Naming**:
```python
def test_<functionality>():                    # Basic unit test
def test_<functionality>_edge_case():         # Edge case testing
def test_<functionality>_error_handling():   # Error condition testing
async def test_<functionality>_async():      # Async functionality
```

### Test Development Guidelines

**Test Structure**:
1. **Arrange**: Set up test fixtures and input data
2. **Act**: Execute the functionality being tested
3. **Assert**: Verify expected outcomes and side effects
4. **Cleanup**: Clean up resources and reset state

**Best Practices**:
- Write descriptive test names that explain the scenario
- Include both positive and negative test cases
- Test boundary conditions and edge cases
- Use appropriate assertions with clear failure messages
- Keep tests independent and idempotent
- Mock external dependencies appropriately

---

## Russian Translation / Русский перевод

# Набор тестов - Фреймворк валидации NFCS

## Обзор

Данная директория содержит комплексный набор тестов для Системы управления нейронными полями (NFCS), реализующий многоуровневую валидацию от модульных тестов до полного интеграционного тестирования системы. Фреймворк тестирования обеспечивает надежность, производительность и соответствие требованиям безопасности во всех компонентах системы.

**Покрытие тестами**: 5,000+ строк тестового кода, охватывающего модульные, интеграционные, производительные тесты и тесты валидации.

---

*This README provides comprehensive documentation for the NFCS testing framework, covering all aspects from unit testing to system validation. The testing suite ensures reliability, performance, and safety compliance across the entire system.*

*Данный README предоставляет исчерпывающую документацию для фреймворка тестирования NFCS, охватывающую все аспекты от модульного тестирования до системной валидации. Набор тестов обеспечивает надежность, производительность и соответствие требованиям безопасности во всей системе.*