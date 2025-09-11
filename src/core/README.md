# Mathematical Core - NFCS Foundation

## Overview

The Mathematical Core provides the fundamental mathematical framework for the Neural Field Control System (NFCS). It implements advanced mathematical models including Complex Ginzburg-Landau (CGL) equations, Kuramoto oscillator networks, and sophisticated metrics for topological analysis.

**Mathematical Foundation**: Based on cutting-edge research in neural field dynamics, phase synchronization, and topological defect analysis.

## 🧮 Mathematical Models

```
Neural Field Dynamics (CGL) ←→ Cognitive Module Synchronization (Kuramoto)
           ↓                                    ↓
    Topological Analysis  ←────────→  Phase Coordination
           ↓                                    ↓
     Defect Detection    ←────────→    Stability Metrics
```

## 📁 Core Components

```
core/
├── cgl_solver.py              # 🌊 Complex Ginzburg-Landau equation solver
├── kuramoto_solver.py         # 🔄 Kuramoto oscillator network dynamics  
├── enhanced_kuramoto.py       # ⚡ Enhanced Kuramoto with adaptive coupling
├── metrics.py                 # 📊 Basic metrics and measurements
├── enhanced_metrics.py        # 📈 Advanced topological and stability metrics
├── state.py                   # 🎯 System state management and tracking
├── regulator.py              # ⚖️ Regulatory mechanisms and control
└── __init__.py               # Package exports and initialization
```

## 🔬 Scientific Foundation

### 1. **Complex Ginzburg-Landau (CGL) Equations**
Fundamental model for neural field dynamics and pattern formation.

**Mathematical Form**:
```
∂ψ/∂t = (1 + ic₁)∇²ψ + (1 + ic₂)ψ - (1 + ic₃)|ψ|²ψ
```

**Physical Interpretation**:
- `ψ(x,t)`: Complex neural field amplitude
- `c₁, c₂, c₃`: System parameters controlling dynamics
- Describes pattern formation, stability, and topological defects

### 2. **Kuramoto Model**
Synchronization dynamics for cognitive module coordination.

**Mathematical Form**:
```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
```

**Parameters**:
- `θᵢ`: Phase of oscillator i
- `ωᵢ`: Natural frequency of oscillator i  
- `K`: Coupling strength
- `N`: Number of oscillators

### 3. **Topological Analysis**
Advanced metrics for pattern stability and defect detection.

**Key Metrics**:
- **Topological Charge**: `Q = (1/2π) ∮ ∇φ · dl`
- **Lyapunov Exponents**: Stability and chaos characterization
- **Correlation Functions**: Spatial and temporal correlations
- **Entropy Measures**: Information-theoretic complexity

## 🎯 Component Details

### 1. **CGL Solver** (`cgl_solver.py`)

**Purpose**: Solves Complex Ginzburg-Landau equations for neural field dynamics.

**Key Features**:
- Finite difference numerical integration
- Adaptive time stepping for stability
- Periodic and Neumann boundary conditions
- Efficient NumPy-based implementation

**Usage Example**:
```python
from core.cgl_solver import CGLSolver
import numpy as np

# Initialize solver
solver = CGLSolver(
    grid_size=(128, 128),
    domain_size=(10.0, 10.0),
    dt=0.01,
    c1=0.5, c2=1.0, c3=0.8
)

# Set initial conditions
x, y = solver.get_coordinates()
psi_0 = np.exp(1j * np.random.uniform(0, 2*np.pi, x.shape))

# Evolve system
for step in range(1000):
    psi_0 = solver.step(psi_0)
    
    # Analyze every 100 steps
    if step % 100 == 0:
        defects = solver.find_topological_defects(psi_0)
        print(f"Step {step}: {len(defects)} topological defects")
```

### 2. **Kuramoto Solver** (`kuramoto_solver.py`)

**Purpose**: Simulates Kuramoto oscillator networks for module synchronization.

**Key Features**:
- Variable coupling topologies (all-to-all, network, custom)
- Adaptive coupling strength
- Synchronization order parameter calculation
- Phase clustering analysis

**Usage Example**:
```python
from core.kuramoto_solver import KuramotoSolver
import numpy as np

# Initialize network
N = 10  # Number of oscillators (cognitive modules)
solver = KuramotoSolver(
    N=N,
    natural_frequencies=np.random.normal(1.0, 0.1, N),
    coupling_strength=2.0,
    topology='all_to_all'
)

# Simulate dynamics
phases = np.random.uniform(0, 2*np.pi, N)
for t in range(1000):
    phases = solver.step(phases, dt=0.01)
    
    # Check synchronization
    if t % 100 == 0:
        sync_order = solver.synchronization_order(phases)
        print(f"Time {t*0.01:.2f}: Sync = {sync_order:.3f}")
```

### 3. **Enhanced Kuramoto** (`enhanced_kuramoto.py`)

**Purpose**: Advanced Kuramoto implementation with adaptive features.

**Enhanced Features**:
- Adaptive coupling based on synchronization state
- Hierarchical oscillator networks
- Dynamic frequency adjustment
- Integration with NFCS orchestrator

**Usage Example**:
```python
from core.enhanced_kuramoto import EnhancedKuramotoModule

# Initialize enhanced module
kuramoto = EnhancedKuramotoModule()
await kuramoto.initialize()

# Synchronize phases with adaptive coupling
phases = {
    'constitution': 0.5,
    'boundary': 1.2, 
    'memory': 2.1,
    'meta_reflection': 0.8,
    'freedom': 1.8
}

synchronized_phases = kuramoto.synchronize_phases(phases)
print("Synchronized phases:", synchronized_phases)
```

### 4. **Enhanced Metrics** (`enhanced_metrics.py`)

**Purpose**: Advanced mathematical analysis and topological metrics.

**Metrics Included**:
- **Topological Charge Density**: Local defect strength
- **Lyapunov Spectrum**: System stability analysis  
- **Correlation Functions**: Spatial/temporal correlations
- **Entropy Measures**: Kolmogorov-Sinai entropy
- **Fractal Dimensions**: Pattern complexity analysis

**Usage Example**:
```python
from core.enhanced_metrics import EnhancedMetricsCalculator
import numpy as np

# Initialize metrics calculator
calculator = EnhancedMetricsCalculator()

# Generate sample neural field
field = np.random.complex128((64, 64))
field = field / np.abs(field)  # Normalize to unit circle

# Calculate comprehensive metrics
metrics = calculator.calculate_comprehensive_metrics(field)

print("Topological Analysis:")
print(f"  Total charge: {metrics['topological']['total_charge']}")
print(f"  Defect count: {metrics['topological']['defect_count']}")
print(f"  Stability index: {metrics['stability']['lyapunov_exponent']}")
print(f"  Spatial correlation: {metrics['correlation']['spatial_correlation_length']}")
```

### 5. **State Management** (`state.py`)

**Purpose**: Manages system state, history, and rollback capabilities.

**Features**:
- State versioning and history tracking
- Rollback to previous stable states
- State validation and consistency checks
- Efficient serialization/deserialization

**Usage Example**:
```python
from core.state import SystemState, StateManager

# Initialize state manager
state_manager = StateManager(max_history=100)

# Create and store state
current_state = SystemState(
    neural_field=psi,
    phases=phases,
    metrics=metrics_dict,
    timestamp=time.time()
)

state_id = state_manager.store_state(current_state)

# Later: retrieve or rollback
retrieved_state = state_manager.get_state(state_id)
state_manager.rollback_to_state(state_id)
```

## ⚡ Performance Optimization

### Numerical Efficiency

**NumPy Vectorization**:
```python
# Efficient CGL evolution using NumPy
def evolve_cgl_vectorized(psi, dt, c1, c2, c3):
    # Laplacian using FFT for periodic boundaries
    k = fftfreq(psi.shape[0])
    kx, ky = np.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    
    psi_fft = fft2(psi)
    laplacian = ifft2(-k2 * psi_fft)
    
    # CGL evolution
    dpsi_dt = (1 + 1j*c1) * laplacian + (1 + 1j*c2) * psi - (1 + 1j*c3) * np.abs(psi)**2 * psi
    
    return psi + dt * dpsi_dt
```

**Kuramoto Optimization**:
```python
# Vectorized Kuramoto dynamics
def kuramoto_step_vectorized(phases, omega, K, dt):
    # Phase differences matrix
    phase_diff = phases[:, None] - phases[None, :]
    
    # Coupling term
    coupling = K * np.mean(np.sin(phase_diff), axis=1)
    
    # Evolution
    return phases + dt * (omega + coupling)
```

### Memory Management

**Efficient Storage**:
- Use `numpy.float32` for reduced memory footprint when precision allows
- Implement sliding window for time series data
- Use memory mapping for large field arrays
- Compress stored states using `numpy.savez_compressed`

## 🧪 Testing and Validation

### Mathematical Validation

**Conservation Laws**:
```python
def test_energy_conservation():
    """Test energy conservation in CGL dynamics."""
    solver = CGLSolver(grid_size=(64, 64))
    psi = initialize_random_field()
    
    initial_energy = calculate_energy(psi)
    
    # Evolve for many steps
    for _ in range(1000):
        psi = solver.step(psi)
    
    final_energy = calculate_energy(psi)
    energy_drift = abs(final_energy - initial_energy) / initial_energy
    
    assert energy_drift < 0.01, f"Energy drift too large: {energy_drift}"
```

**Synchronization Tests**:
```python
def test_kuramoto_synchronization():
    """Test synchronization in Kuramoto model."""
    solver = KuramotoSolver(N=20, coupling_strength=5.0)
    phases = np.random.uniform(0, 2*np.pi, 20)
    
    # Should synchronize with strong coupling
    for _ in range(1000):
        phases = solver.step(phases, dt=0.01)
    
    sync_order = solver.synchronization_order(phases)
    assert sync_order > 0.9, f"Synchronization failed: {sync_order}"
```

### Numerical Stability

**Stability Analysis**:
```python
def test_numerical_stability():
    """Test numerical stability of solvers."""
    # Test different time steps
    dt_values = [0.001, 0.01, 0.1]
    
    for dt in dt_values:
        solver = CGLSolver(dt=dt)
        psi = initialize_test_field()
        
        # Check for blow-up
        for step in range(100):
            psi = solver.step(psi)
            max_amplitude = np.max(np.abs(psi))
            
            assert max_amplitude < 10.0, f"Numerical blow-up at dt={dt}, step={step}"
```

## 🔧 System Requirements

### Mathematical Libraries
```bash
# Required for core functionality
numpy >= 1.24.0          # Vectorized numerical operations
scipy >= 1.11.0          # Scientific computing and optimization  

# Optional for enhanced performance
numba >= 0.57.0          # JIT compilation for speed
```

### Computational Requirements

**Minimum Configuration**:
- **RAM**: 2 GB (for 128×128 field simulations)
- **CPU**: Dual-core 2.0 GHz 
- **Storage**: 100 MB for state history

**Recommended Configuration**:
- **RAM**: 8 GB+ (for large-scale simulations)
- **CPU**: Quad-core 3.0 GHz with vectorization support
- **GPU**: Optional CUDA support for large fields
- **Storage**: SSD for fast state I/O

### Performance Scaling

**Field Size vs. Memory**:
```python
# Memory usage estimation
def estimate_memory_usage(grid_size):
    """Estimate memory usage for given grid size."""
    complex_field_size = grid_size[0] * grid_size[1] * 16  # bytes for complex128
    history_factor = 100  # Keep 100 time steps
    
    total_mb = (complex_field_size * history_factor) / (1024**2)
    return total_mb

print(f"64×64 grid: {estimate_memory_usage((64, 64)):.1f} MB")
print(f"128×128 grid: {estimate_memory_usage((128, 128)):.1f} MB") 
print(f"256×256 grid: {estimate_memory_usage((256, 256)):.1f} MB")
```

## 🚀 Advanced Usage

### Custom Solvers

**Creating Custom CGL Solver**:
```python
from core.cgl_solver import CGLSolver

class CustomCGLSolver(CGLSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_parameter = kwargs.get('custom_param', 1.0)
    
    def custom_evolution_step(self, psi):
        # Implement custom dynamics
        standard_step = super().step(psi)
        custom_modification = self.apply_custom_dynamics(psi)
        return standard_step + custom_modification
    
    def apply_custom_dynamics(self, psi):
        # Custom mathematical model
        return self.custom_parameter * np.gradient(np.abs(psi)**2)
```

### Integration with Orchestrator

**Orchestrator Integration**:
```python
from core.enhanced_kuramoto import EnhancedKuramotoModule
from orchestrator.nfcs_orchestrator import NFCSOrchestrator

class IntegratedMathCore:
    def __init__(self, orchestrator: NFCSOrchestrator):
        self.orchestrator = orchestrator
        self.kuramoto = EnhancedKuramotoModule()
        self.cgl_solver = CGLSolver()
        
    async def update_system_dynamics(self):
        # Get current module phases from orchestrator
        module_states = await self.orchestrator.get_module_states()
        phases = {mod.name: mod.phase for mod in module_states}
        
        # Synchronize phases
        new_phases = self.kuramoto.synchronize_phases(phases)
        
        # Update orchestrator with synchronized phases
        await self.orchestrator.update_module_phases(new_phases)
```

## 🤝 Contributing

### Mathematical Extensions

**Adding New Equations**:
1. Inherit from base solver classes
2. Implement mathematical model in `evolve_step()` method
3. Add comprehensive unit tests
4. Validate against known analytical solutions

**Performance Improvements**:
1. Profile code using `cProfile`
2. Identify bottlenecks in numerical loops
3. Implement vectorized NumPy operations
4. Consider Numba JIT compilation for critical sections

### Code Standards

**Mathematical Documentation**:
- Include mathematical formulation in docstrings
- Provide references to scientific literature
- Document parameter ranges and stability conditions
- Include convergence and accuracy analysis

**Testing Requirements**:
- Validate against analytical solutions where available
- Test conservation laws and symmetries
- Check numerical stability across parameter ranges
- Benchmark performance with different configurations

## 📚 Scientific References

### Core Literature
1. **Kuramoto, Y.** (1984). "Chemical Oscillations, Waves, and Turbulence"
2. **Aranson, I. S. & Kramer, L.** (2002). "The world of the complex Ginzburg-Landau equation"
3. **Cross, M. C. & Hohenberg, P. C.** (1993). "Pattern formation outside of equilibrium"
4. **Acebrón, J. A. et al.** (2005). "The Kuramoto model: A simple paradigm for synchronization phenomena"

### NFCS-Specific Research
1. **Urmanov, T. et al.** (2025). "Neural Field Control Systems with Constitutional Safety"
2. **Gadeev, K. et al.** (2025). "Topological Defect Analysis in Cognitive Architectures"
3. **Yusupov, B. et al.** (2025). "Phase Synchronization in Multi-Agent Cognitive Systems"

---

## Russian Translation / Русский перевод

# Математическое ядро - Основа NFCS

## Обзор

Математическое ядро обеспечивает фундаментальную математическую основу для Системы управления нейронными полями (NFCS). Оно реализует продвинутые математические модели, включая уравнения комплексного Гинзбурга-Ландау (CGL), сети осцилляторов Курамото и сложные метрики для топологического анализа.

**Математическая основа**: Базируется на передовых исследованиях в области динамики нейронных полей, фазовой синхронизации и анализа топологических дефектов.

## 🧮 Математические модели

```
Динамика нейронных полей (CGL) ←→ Синхронизация когнитивных модулей (Kuramoto)
           ↓                                         ↓
   Топологический анализ      ←────────→      Координация фаз
           ↓                                         ↓
   Обнаружение дефектов      ←────────→      Метрики стабильности
```

## 📁 Основные компоненты

```
core/
├── cgl_solver.py              # 🌊 Решатель уравнений комплексного Гинзбурга-Ландау
├── kuramoto_solver.py         # 🔄 Динамика сети осцилляторов Курамото
├── enhanced_kuramoto.py       # ⚡ Расширенный Курамото с адаптивной связью
├── metrics.py                 # 📊 Базовые метрики и измерения
├── enhanced_metrics.py        # 📈 Продвинутые топологические метрики и метрики стабильности
├── state.py                   # 🎯 Управление состоянием системы и отслеживание
├── regulator.py              # ⚖️ Регуляторные механизмы и управление
└── __init__.py               # Экспорты пакета и инициализация
```

## 🔬 Научная основа

### 1. **Уравнения комплексного Гинзбурга-Ландау (CGL)**
Фундаментальная модель для динамики нейронных полей и формирования паттернов.

**Математическая форма**:
```
∂ψ/∂t = (1 + ic₁)∇²ψ + (1 + ic₂)ψ - (1 + ic₃)|ψ|²ψ
```

**Физическая интерпретация**:
- `ψ(x,t)`: Амплитуда комплексного нейронного поля
- `c₁, c₂, c₃`: Параметры системы, управляющие динамикой
- Описывает формирование паттернов, стабильность и топологические дефекты

### 2. **Модель Курамото**
Динамика синхронизации для координации когнитивных модулей.

**Математическая форма**:
```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
```

**Параметры**:
- `θᵢ`: Фаза осциллятора i
- `ωᵢ`: Собственная частота осциллятора i
- `K`: Сила связи
- `N`: Количество осцилляторов

## 🔧 Системные требования

### Математические библиотеки
```bash
# Обязательные для основной функциональности
numpy >= 1.24.0          # Векторизованные численные операции
scipy >= 1.11.0          # Научные вычисления и оптимизация

# Опционально для повышения производительности
numba >= 0.57.0          # JIT-компиляция для ускорения
```

### Вычислительные требования

**Минимальная конфигурация**:
- **RAM**: 2 GB (для симуляций поля 128×128)
- **CPU**: Двухъядерный 2.0 GHz
- **Хранилище**: 100 MB для истории состояний

**Рекомендуемая конфигурация**:
- **RAM**: 8 GB+ (для крупномасштабных симуляций)
- **CPU**: Четырехъядерный 3.0 GHz с поддержкой векторизации
- **GPU**: Опционально поддержка CUDA для больших полей
- **Хранилище**: SSD для быстрого I/O состояний

---

*This README provides comprehensive documentation for the Mathematical Core of NFCS. The implementation combines rigorous mathematical foundations with efficient numerical methods.*

*Данный README предоставляет исчерпывающую документацию для Математического ядра NFCS. Реализация объединяет строгие математические основы с эффективными численными методами.*