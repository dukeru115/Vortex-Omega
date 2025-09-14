# Vortex-Omega: Neural Field Control System v2.4.3

**Vortex (Ω Team)** — A hybrid AI repository implementing advanced Neural Field Control Systems (NFCS) with Echo-Semantic Converter (ESC) patterns, stable semantic anchors, and near-zero hallucination rates through constitutional safety frameworks. Features multi-agent consensus via Kuramoto/ADMM dynamics, causal world models, and comprehensive interpretability tools.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange.svg)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-green.svg)](https://scipy.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com)
[![Status](https://img.shields.io/badge/status-95%25%20complete-brightgreen.svg)]()
[![Updated](https://img.shields.io/badge/updated-September%2014%2C%202025-blue.svg)]()
[![Version](https://img.shields.io/badge/version-2.5.0--dev-orange.svg)]()
[![Code](https://img.shields.io/badge/code-20.3k%20lines-blue.svg)]()

---

## 🆕 **Latest Updates (September 2025)**

### **✨ Major New Features**
- **Symbolic AI Module**: Complete implementation with discrepancy gate and Kant mode
- **Enhanced ESC v2.2**: Multi-scale temporal echo with adaptive frequencies
- **Integration Improvements**: Full Symbolic AI ↔ ESC ↔ Kuramoto integration
- **Documentation**: Comprehensive revision history and status reports

### **📈 Project Metrics**
- **Total Code**: 20,300+ lines (↑ 73% from v2.4.3)
- **Modules**: 15 fully operational components
- **Test Coverage**: 85-95% on critical paths
- **Performance**: 10Hz orchestration with async support

[📊 View Full Status Report](PROJECT_STATUS.md) | [📜 Revision History](REVISION_HISTORY.md)

---

## 🎯 **Revolutionary Neural Field Control Architecture**

### **📋 Architecture Overview**

NFCS v2.4.3 represents a paradigmatic shift from descriptive AI models to **active control systems** that optimize cognitive coherence under resource constraints. The system embodies the principle of **"Costly Coherence"** - treating neural synchronization as a precious, actively maintained resource.

**🧮 Mathematical Optimization Objective**:
```latex
𝒥[φ,u] = ∫∫ [α‖u(x,t)‖² - βℜ(φ) + γℋ(∇φ) + δρ_def(φ)]dxdt
```

**🌀 Complex Field Dynamics**:
```latex
∂φ/∂t = (1+iε₁)φ + (1+iε₂)∇²φ - (1+iε₃)|φ|²φ + u(x,t)
∇·E⃗ = ρ/ε₀,  ∇×B⃗ = μ₀J⃗ + μ₀ε₀∂E⃗/∂t
```

**🔗 Kuramoto-ESC Coupling**:
```latex
dθᵢ/dt = ωᵢ + ∑ⱼ Kᵢⱼ(t)sin(θⱼ - θᵢ - αᵢⱼ) + ηESC(t)
K_ij(t) = K₀ · σ(S_semantic(i,j,t)) · exp(-γ|i-j|)
```

---

## 🏗️ **Core Components**

### 1. **Neural Field Control System (NFCS)**
- **Observer**: Real-time field state monitoring `y(t) = C·φ + η(t)`
- **Predictor**: Evolution forecasting `φ̂(t+Δt) = F[φ(t), u(t)]`  
- **Regulator**: Control action generation `u(t) = K·e(t) + ∫G[φ, ρ_def, H_a]`

### 2. **🌊 Echo-Semantic Converter (ESC) 2.1**
Advanced token processing with oscillatory dynamics and semantic resonance:

**🎵 Semantic Oscillator Encoding**:
```latex
Sᵢ(t) = sᵢ sin(2πfᵢ(t-tᵢ) + φᵢ) × exp(-λᵢ(t-tᵢ)) × ℋ(t-tᵢ)
```

**🌀 Multi-scale Echo Integration**:
```latex
𝒪(t) = ∑ⱼ γⱼ ∫₋∞ᵗ S(τ) × exp(-μⱼ(t-τ)) × W_echo(t-τ) dτ
```

**🛡️ Constitutional Safety Filter**:
```latex
𝒞_safe(s) = 𝟙[‖s‖_harm < θ_harm] ∧ 𝟙[ℋ_kant(s) = ⊤] ∧ 𝟙[ρ_risk(s) < ε]
```

**⚡ Real-time Processing**: Constitutional filtering with near-zero hallucination

### 3. **🌀 Kuramoto Synchronization Network**
Multi-agent consensus through adaptive coupled oscillator dynamics:

**🎭 Phase Synchronization Dynamics**:
```latex
dθᵢ/dt = ωᵢ + ∑ⱼ₌₁ᴺ Kᵢⱼ(t) sin(θⱼ - θᵢ - αᵢⱼ) + uᵢ(t) + ηᵢ(t)
```

**🧠 Adaptive Coupling Matrix**:
```latex
Kᵢⱼ(t) = K₀ · 𝒮_semantic(i,j,t) · exp(-β‖rᵢ - rⱼ‖) · (1 + ε_ESC(t))
```

**📊 Order Parameter (Synchronization Measure)**:
```latex
R(t)e^{iΨ(t)} = N⁻¹ ∑ₖ₌₁ᴺ e^{iθₖ(t)}, R ∈ [0,1]
```

**⚖️ Consensus Convergence**: Multi-agent coherence through phase-locking

### 4. **⚡ Complex Ginzburg-Landau (CGL) Dynamics**
Nonlinear field evolution with topological defect control and symbolic modulation:

**🌊 Controlled Field Evolution**:
```latex
∂φ/∂t = (1+ic₁)φ + (1+ic₂)∇²φ - (1+ic₃)|φ|²φ + u_control(x,t) + u_symbolic(x,t)
```

**🎯 Topological Defect Density**:
```latex
ρ_def(x,t) = |∇ × arg(φ(x,t))|/(2π) = (2π)⁻¹|∂_x arg φ_y - ∂_y arg φ_x|
```

**🧮 Control Field Optimization**:
```latex
u_opt = arg min_{u} ∫∫ [α‖u‖² + β|φ - φ_target|² + γ|∇φ|²] dx dt
```

**⚠️ Stability Constraint**: `‖φ(x,t)‖ ≤ M_critical` for bounded dynamics

### 5. **🧠 Kamil Symbolic AI Engine (NEW)**
Deterministic LLM-free neuro-symbolic architecture with mathematical rigor:

**Pipeline Architecture**:
```
📝 Input → 🔍 Symbolize → 🗂️ Fieldize → ✅ Verify → ⚡ Neural Modulation
```

**Mathematical Foundation**:
```python
# Symbolic Transformation: Φ(field) → symbolic_representation
Φ: C^n×m → {Unit, Quantity, Expr, SymClause, SymField}

# Dimensional Analysis with SI Base Units
Unit = (m^a, kg^b, s^c, A^d, K^e, mol^f, cd^g) × scale_factor

# Verification Pipeline
V(S) = D_consistency ∧ Z3_SAT ∧ Kant_ethics
where D_consistency ≥ 0.98, Latency ≤ 300ms
```

**Core Components**:
- **🔬 Symbolize**: NER + NumUnit parsing + Formula extraction + SI canonization
- **🧩 Fieldize**: DBSCAN clustering + Invariant generation + Neural mapping  
- **🛡️ Verify**: Dimensional + Z3 SMT + Kantian ethical validation
- **⚡ SLO**: ≤300ms latency, ≥0.98 dimensional accuracy guarantee

**Integration Patterns**:
```python
# ESC-Kuramoto Bridge Integration
η(t) → K_ij(t): semantic_coupling → oscillator_modulation

# CGL Field Modulation  
∂φ/∂t = φ + ∇²φ - |φ|²φ + u_symbolic(x,t)

# Discrepancy Gate Validation
discrepancy_detected → symbolic_verification → safety_action
```

---

## 🚀 **Quick Start**

### **Installation**
```bash
# Clone repository
git clone https://github.com/dukeru115/Vortex-Omega.git
cd Vortex-Omega

# Setup virtual environment
python -m venv vortex-env
source vortex-env/bin/activate  # On Windows: vortex-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### **Basic Usage**
```python
import asyncio
from src.orchestrator.nfcs_orchestrator import create_orchestrator, OrchestrationConfig
from src.modules.esc.esc_core import EchoSemanticConverter, ESCConfig

async def demo_nfcs():
    # Initialize NFCS system
    config = OrchestrationConfig(enable_detailed_logging=True)
    
    async with await create_orchestrator(config) as orchestrator:
        # Process semantic input
        esc = EchoSemanticConverter(ESCConfig(
            processing_mode=ProcessingMode.BALANCED,
            enable_constitutional_filtering=True
        ))
        
        # Analyze tokens
        tokens = ["neural", "field", "control", "system"]
        result = esc.process_sequence(tokens)
        
        print(f"Constitutional Compliance: {result.constitutional_metrics['constitutional_compliance']:.3f}")
        print(f"Processed {len(result.processed_tokens)} tokens safely")

# Run demonstration
asyncio.run(demo_nfcs())
```

### **REST API Server**
```bash
# Start FastAPI server
python -m src.api.server --host 0.0.0.0 --port 8000

# Monitor real-time metrics
curl http://localhost:8000/api/v1/metrics/realtime
```

---

## 📊 **Advanced Features**

### **🛡️ Advanced Hallucination Control**

**🎯 Topological Defect Detection**:
```latex
ρ_def(x,t) = (2π)⁻¹|∇ × arg(φ(x,t))| = (2π)⁻¹|∂_x Im(φ*∂_y φ) - ∂_y Im(φ*∂_x φ)|
```

**🔢 Generalized Hallucination Metric**:
```latex
ℋ_anomaly(t) = ∫_Ω [ρ_def(x,t)𝒲_topology(x) + σ_error(x,t)𝒲_semantic(x) + δ_kant(x,t)𝒲_ethical(x)] dx
```

**🧠 Constitutional Compliance Operator**:
```latex
𝒞_const: 𝒮 → {0,1}, 𝒞_const(s) = 𝟙[‖s‖_harm < ε] ∧ ℋ_kant(s) ∧ ℤ₃_consistency(s)
```

**📊 Conformal Prediction with Abstention**:
```latex
𝒫_conf(y|x) = {y : |f(x) - y| ≤ q_{α}(|f(X_i) - Y_i|)}, Coverage ≥ 1-α
```

**⚠️ Real-time Safety**: Multi-modal anomaly detection with ethical validation

### **🎭 Multi-Agent Consensus**
- **Kuramoto Synchronization**: Phase-locked cognitive modules
- **ADMM Optimization**: Distributed constraint satisfaction
- **Dynamic Coupling**: Adaptive connection strength `K_ij(t)`

### **🌍 Causal World Models**
- **RT-2 Integration**: Robotic Transformer architecture
- **Dreamer Framework**: Model-based reinforcement learning
- **Temporal Dynamics**: Causal prediction and planning

### **📈 Interpretability & Monitoring**
- **Integrated Gradients**: Attribution analysis
- **ESC Telemetry**: Real-time semantic processing metrics
- **Phase Risk Monitor**: Early warning system for cognitive failures
- **WebSocket Dashboard**: Live system visualization

---

## 🏛️ **Modular Architecture**

The system implements a sophisticated modular architecture based on **Philosophy of Awareness (PA)**:

| Module | Function | Mathematical Model |
|--------|----------|-------------------|
| **Constitutional** | System integrity & governance | `C[η,u,ρ_def] = ∫[α_c I[η] + β_c V[u] + γ_c D[ρ_def]]dt` |
| **ESC Semantic** | Token processing & conversion | `O(t) = -αO(t) + βϕ(S(t) + γE(t)) + η[O(t-τ)]` |
| **Memory** | Multi-scale temporal storage | `M(t) = Σ M_j(t) = Σ ∫ φ(τ)K_j(t-τ)dτ` |
| **Meta-Reflection** | Contradiction detection | `G_gap` operators for inconsistency analysis |
| **Freedom** | Creative leap generation | Stochastic transitions `Δφ` |
| **Coherence** | Global synchronization | Kuramoto network coordination |
| **Boundary** | Permeability management | Adaptive filtering `P(x,t)` |

---

## 🛠️ **Development & Testing**

### **Running Tests**
```bash
# Core component tests
pytest tests/test_core/ -v

# Integration tests  
pytest tests/integration/ -v

# Enhanced system validation
python test_enhanced_demo.py
python demo_option_c.py
```

### **Performance Benchmarks**
```bash
# Mathematical operations performance
python scripts/benchmark_performance.py

# Memory usage analysis
python scripts/analyze_memory.py

# Real-time processing metrics
python scripts/realtime_benchmark.py
```

---

## 📚 **Documentation**

- **[Architecture Guide](./ARCHITECTURE.md)**: Detailed system architecture
- **[API Documentation](./docs/api/)**: REST API reference
- **[Mathematical Framework](./docs/mathematics/)**: Theoretical foundations
- **[Deployment Guide](./DEPLOYMENT.md)**: Production deployment
- **[Contributing Guidelines](./CONTRIBUTING.md)**: Development workflow

---

## 🔬 **Research & Theory**

### **Core Publications**
- **Urmanov, T., Gadeev, K., & Iusupov, B.** (2025). *Hybrid Cognitive-Mathematical Model: A Neural Field Control System for Costly Coherence v2.4.3*. [PDF](./docs/AI_hybrid_architectures_NFCS_v2.4.3.pdf)

### **Mathematical Foundations**
- **Complex Ginzburg-Landau Theory**: Local field dynamics with nonlinear control
- **Kuramoto Synchronization**: Multi-agent consensus mechanisms  
- **Topological Defect Analysis**: Coherence failure detection
- **Constitutional AI**: Multi-stakeholder governance frameworks

### **Experimental Validation**
- **Phase Risk Monitoring**: Real-time cognitive failure prediction
- **Defect Mapping**: Topological analysis of neural field singularities  
- **Cross-Frequency Analysis**: Hierarchical control structure validation

---

## 🤝 **Contributing**

We welcome contributions to the Vortex-Omega project! Please see our [Contributing Guidelines](./CONTRIBUTING.md) for details.

### **Development Setup**
```bash
# Development installation
git clone https://github.com/dukeru115/Vortex-Omega.git
cd Vortex-Omega
pip install -e ".[dev]"

# Pre-commit hooks
pre-commit install

# Run development server
python -m src.api.server --reload
```

---

## 📄 **License & Attribution**

**License**: [CC BY-NC 4.0 (Attribution — NonCommercial)](https://creativecommons.org/licenses/by-nc/4.0/)

**Team Ω Contributors**:
- **Timur Urmanov** - Conceptualization, Formal Analysis, Methodology
- **Kamil Gadeev** - Software Architecture, Vortex Protocol, Philosophy of Awareness  
- **Bakhtier Iusupov** - Project Administration, LLM Integration, Documentation

---

## 📞 **Contact & Support**

- **Issues**: [GitHub Issues](https://github.com/dukeru115/Vortex-Omega/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dukeru115/Vortex-Omega/discussions)
- **Email**: Urmanov.t@gmail.com

---

**© 2025 Team Ω. Neural Field Control Systems Research Group.**

*Updated: September 13, 2025*

---

### **🎯 Project Status - September 13, 2025**

✅ **NFCS Core Architecture** - Production ready  
✅ **ESC Module 2.1** - Enhanced semantic processing  
✅ **Constitutional Framework** - Safety compliance  
✅ **Kuramoto Synchronization** - Multi-agent consensus  
🔄 **REST API Interface** - In active development  
🔄 **Performance Optimization** - JIT compilation integration  
📋 **Documentation** - Comprehensive guides available  

**Next Milestones**: WebSocket real-time monitoring, Numba JIT optimization, production deployment guides.

---
