# Source Code Overview

Core for NFCS (PDF 2-5). 80,000+ lines.

## Subfolders
- **core/**: cgl_solver.py (CGL, PDF 3.2); kuramoto_solver.py (sync, PDF 3.4).
- **modules/**: esc_system.py (ESC, PDF 4); constitutional_realtime.py (PDF 5.3); enhanced_kuramoto.py.
- **api/**: LLM integration (PDF 5.10).
- **i18n/**: Localization.

Example:
```python
from core.cgl_solver import simulate_cgl
result = simulate_cgl(params={'c1': 1, 'c3': 1})
```

Updated: Sept 21, 2025.

## 📁 Directory Structure

```
src/
├── main.py                    # 🚀 Main entry point and CLI interface
├── test_nfcs_integration.py   # 🧪 Comprehensive integration tests
├── core/                      # 🔬 Mathematical core (Kuramoto, CGL equations)
├── orchestrator/              # ⚙️ Central coordination system (8 components)
├── modules/                   # 🧠 Cognitive modules and ESC system
├── evolution/                 # 🔄 Evolution algorithms and optimization
├── utils/                     # 🛠️ Utility functions and helpers
└── __init__.py               # Package initialization
```

## 🎯 Core Components

### 1. **Main Entry Point** (`main.py`)
- **Purpose**: Primary application entry with full CLI interface
- **Features**: Multiple operational modes, configuration management, daemon support
- **Usage**: `python src/main.py --help`

### 2. **Mathematical Core** (`core/`)
- **Complex Ginzburg-Landau Solver**: Neural field dynamics simulation
- **Enhanced Kuramoto Module**: Phase synchronization between cognitive modules
- **Metrics Calculator**: Topological defect analysis and stability assessment

### 3. **Orchestrator System** (`orchestrator/`)
- **NFCS Orchestrator**: 40,000+ lines of central coordination logic
- **Module Manager**: Lifecycle management for all cognitive modules
- **State Coordinator**: Global state synchronization and versioning
- **Event System**: Real-time inter-module communication

### 4. **Cognitive Modules** (`modules/`)
- **Constitutional Framework**: 47,000+ lines of policy management
- **ESC System**: Echo-Semantic Converter for token processing
- **Boundary Module**: Dynamic boundary management
- **Memory System**: Multi-type memory architecture
- **Freedom Module**: Autonomous decision-making capabilities

### 5. **Integration Testing** (`test_nfcs_integration.py`)
- **Comprehensive Tests**: Full system integration validation
- **Performance Benchmarks**: Coordination frequency and response time testing
- **Safety Validation**: Constitutional compliance verification

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- NumPy >= 1.24.0
- SciPy >= 1.11.0
- PyYAML >= 6.0
- psutil >= 5.9.0

### Basic Usage
```bash
# Navigate to project root
cd /path/to/Vortex-Omega

# Run test mode (5 seconds)
python src/main.py --test --log-level INFO

# Run autonomous mode
python src/main.py --mode autonomous

# Run integration tests
python src/test_nfcs_integration.py
```

### Python API Usage
```python
from orchestrator.nfcs_orchestrator import create_orchestrator, create_default_config

# Create configuration
config = create_default_config()
config.operational_mode = "supervised"

# Initialize and start system
orchestrator = create_orchestrator(config)
await orchestrator.start()

# Process input
result = await orchestrator.process_input("Your data here")
```

## 🔧 System Requirements

### Minimum Requirements
- **RAM**: 4 GB (8 GB recommended)
- **CPU**: Dual-core 2.0 GHz
- **Disk**: 500 MB free space
- **Python**: 3.8+ with async/await support

### Production Requirements
- **RAM**: 16 GB or more
- **CPU**: Quad-core 4.0 GHz or better
- **GPU**: Optional (NVIDIA CUDA 11.7+ for acceleration)
- **Network**: Low latency for real-time coordination

## 📊 Performance Characteristics

- **Coordination Frequency**: 10Hz real-time orchestration
- **Module Count**: 5 cognitive modules + ESC system
- **Response Time**: <100ms for standard operations
- **Memory Usage**: ~2-4 GB during active processing
- **Scalability**: Supports distributed multi-node deployment

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    NFCS Orchestrator                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   Module    │ │    State    │ │   Event     │          │
│  │  Manager    │ │ Coordinator │ │  System     │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
            │                    │                    │
┌───────────▼──────────┐ ┌───────▼────────┐ ┌────────▼─────────┐
│   Cognitive Modules  │ │ Mathematical   │ │   ESC System     │
│ ┌─────┐ ┌─────────┐ │ │     Core       │ │ ┌──────────────┐ │
│ │Const│ │Boundary │ │ │ ┌───────────┐  │ │ │Token Processor│ │
│ │itut.│ │ Memory  │ │ │ │  Kuramoto │  │ │ │Semantic Fields│ │
│ │ Free│ │Meta-Ref │ │ │ │    CGL    │  │ │ │Constitutional │ │
│ └─────┘ └─────────┘ │ │ │  Metrics  │  │ │ │   Filter     │ │
└─────────────────────┘ │ └───────────┘  │ │ └──────────────┘ │
                        └────────────────┘ └──────────────────┘
```

## 🧪 Testing and Validation

### Integration Tests
```bash
# Run comprehensive integration tests
python src/test_nfcs_integration.py

# Expected output: All 15+ test cases should pass
# Test coverage includes:
# - System initialization and startup
# - Module coordination and synchronization  
# - Constitutional compliance verification
# - Emergency protocols and safety systems
# - Performance and stability benchmarks
```

### Unit Tests
```bash
# Navigate to project root
cd /path/to/Vortex-Omega

# Run all unit tests
pytest tests/ -v

# Test specific components
pytest tests/test_core/ -v           # Mathematical core
pytest tests/test_orchestrator/ -v   # Orchestrator system
pytest tests/test_modules/ -v        # Cognitive modules
```

## 📚 Documentation

- **[../ARCHITECTURE.md](../ARCHITECTURE.md)**: Detailed system architecture
- **[../QUICK_START.md](../QUICK_START.md)**: Getting started guide  
- **[../DEPLOYMENT.md](../DEPLOYMENT.md)**: Production deployment guide
- **[core/README.md](core/README.md)**: Mathematical core documentation
- **[orchestrator/README.md](orchestrator/README.md)**: Orchestrator system guide
- **[modules/README.md](modules/README.md)**: Cognitive modules overview

## 🚀 Deployment

### Development Mode
```bash
# Set development environment
export PYTHONPATH="/path/to/Vortex-Omega/src:$PYTHONPATH"
export NFCS_ENV="development"

# Run with debug logging
python src/main.py --log-level DEBUG --test
```

### Production Mode
```bash
# Set production environment
export NFCS_ENV="production"

# Run as daemon with configuration
python src/main.py --daemon --mode autonomous --config production_config.yaml
```

### Docker Deployment
```bash
# Build container (from project root)
docker build -t vortex-omega-nfcs .

# Run container
docker run -d --name nfcs-prod -p 8080:8080 vortex-omega-nfcs
```

## 🤝 Contributing

### Code Standards
- **Style**: PEP 8 compliance (use `black` formatter)
- **Documentation**: All public functions require docstrings
- **Testing**: New code must include comprehensive tests
- **Type Hints**: Use type annotations for all function signatures

### Development Workflow
1. Fork repository and create feature branch
2. Make changes with proper tests
3. Run full test suite: `pytest tests/`
4. Format code: `black src/`
5. Submit pull request with detailed description

### Adding New Modules
1. Create module in appropriate subdirectory
2. Follow existing module interface patterns
3. Add integration tests in `test_nfcs_integration.py`
4. Update orchestrator registration
5. Document in module README

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/dukeru115/Vortex-Omega/issues)
- **Documentation**: [Project Docs](../docs/)
- **Contact**: urmanov.t@gmail.com

---

## Russian Translation / Русский перевод

# Исходный код - System управления нейронными полями (NFCS)

## Обзор

Данная директория содержит полную реализацию исходного кода Системы управления нейронными полями (NFCS), готовую к производству когнитивную архитектуру с продвинутыми математическими моделями и комплексными рамками конституционной безопасности.

**Статистика кода**: 11,772+ строк готового к производству Python кода в 6 основных подсистемах.

## 📁 Структура директорий

```
src/
├── main.py                    # 🚀 Главная точка входа и CLI интерфейс
├── test_nfcs_integration.py   # 🧪 Комплексные интеграционные тесты
├── core/                      # 🔬 Математическое ядро (Kuramoto, CGL уравнения)
├── orchestrator/              # ⚙️ Центральная system координации (8 компонентов)
├── modules/                   # 🧠 Когнитивные модули и ESC system
├── evolution/                 # 🔄 Алгоритмы эволюции и оптимизации
├── utils/                     # 🛠️ Утилиты и вспомогательные функции
└── __init__.py               # Initialization пакета
```

## 🎯 Основные компоненты

### 1. **Главная точка входа** (`main.py`)
- **Назначение**: Основной вход приложения с полным CLI интерфейсом
- **Функции**: Несколько операционных режимов, control конфигурацией, поддержка демона
- **Использование**: `python src/main.py --help`

### 2. **Математическое ядро** (`core/`)
- **Решатель комплексного Гинзбурга-Ландау**: Симуляция динамики нейронных полей
- **Расширенный module Kuramoto**: Фазовая synchronization между когнитивными модулями
- **Калькулятор метрик**: Анализ топологических дефектов и оценка стабильности

### 3. **System оркестратора** (`orchestrator/`)
- **NFCS Оркестратор**: 40,000+ строк логики центральной координации
- **Менеджер модулей**: Control жизненным циклом всех когнитивных модулей
- **Координатор состояний**: Глобальная synchronization состояний и версионирование
- **System событий**: Межмодульная коммуникация в реальном времени

### 4. **Когнитивные модули** (`modules/`)
- **Конституционная структура**: 47,000+ строк управления политиками
- **ESC System**: Echo-Semantic Converter для обработки токенов
- **Module границ**: Динамическое control границами
- **System памяти**: Многотипная архитектура памяти
- **Module свободы**: Возможности автономного принятия решений

### 5. **Интеграционное testing** (`test_nfcs_integration.py`)
- **Комплексные тесты**: Валидация полной системной интеграции
- **Бенчмарки производительности**: Testing частоты координации и времени отклика
- **Валидация безопасности**: Check соответствия конституционным требованиям

## ⚡ Быстрый старт

### Предварительные требования
- Python 3.8+
- NumPy >= 1.24.0
- SciPy >= 1.11.0
- PyYAML >= 6.0
- psutil >= 5.9.0

### Базовое использование
```bash
# Переход в корень проекта
cd /path/to/Vortex-Omega

# Start тестового режима (5 секунд)
python src/main.py --test --log-level INFO

# Start автономного режима
python src/main.py --mode autonomous

# Start интеграционных тестов
python src/test_nfcs_integration.py
```

### Использование Python API
```python
from orchestrator.nfcs_orchestrator import create_orchestrator, create_default_config

# Creation конфигурации
config = create_default_config()
config.operational_mode = "supervised"

# Initialization и start системы
orchestrator = create_orchestrator(config)
await orchestrator.start()

# Processing входных данных
result = await orchestrator.process_input("Ваши data здесь")
```

## 🔧 Системные требования

### Минимальные требования
- **RAM**: 4 GB (рекомендуется 8 GB)
- **CPU**: Двухъядерный 2.0 GHz
- **Диск**: 500 MB свободного места
- **Python**: 3.8+ с поддержкой async/await

### Требования для продакшена
- **RAM**: 16 GB или больше
- **CPU**: Четырехъядерный 4.0 GHz или лучше
- **GPU**: Опционально (NVIDIA CUDA 11.7+ для ускорения)
- **Сеть**: Низкая delay для координации в реальном времени

## 📊 Характеристики производительности

- **Frequency координации**: 10Hz оркестрация в реальном времени
- **Количество модулей**: 5 когнитивных модулей + ESC system
- **Time отклика**: <100мс для стандартных операций
- **Использование памяти**: ~2-4 GB во time активной обработки
- **Масштабируемость**: Поддержка распределенного развертывания на нескольких узлах

## 🚀 Развертывание

### Режим разработки
```bash
# Установка среды разработки
export PYTHONPATH="/path/to/Vortex-Omega/src:$PYTHONPATH"
export NFCS_ENV="development"

# Start с отладочным логированием
python src/main.py --log-level DEBUG --test
```

### Продакшн режим
```bash
# Установка продакшн среды
export NFCS_ENV="production"

# Start как демон с конфигурацией
python src/main.py --daemon --mode autonomous --config production_config.yaml
```

## 🤝 Участие в разработке

### Стандарты кода
- **Стиль**: Соответствие PEP 8 (используйте форматтер `black`)
- **Документация**: Все публичные функции требуют docstrings
- **Testing**: Новый код должен включать комплексные тесты
- **Аннотации типов**: Используйте аннотации типов для всех сигнатур функций

### Рабочий process разработки
1. Форкните репозиторий и создайте feature branch
2. Внесите изменения с соответствующими тестами
3. Запустите полный набор тестов: `pytest tests/`
4. Отформатируйте код: `black src/`
5. Отправьте pull request с подробным описанием

---

*This README provides comprehensive information for developers working with the NFCS source code. For user-facing documentation, see the main project README.*

*Данный README предоставляет исчерпывающую информацию для разработчиков, работающих с исходным кодом NFCS. Для пользовательской документации см. основной README проекта.*