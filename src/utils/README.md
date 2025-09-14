# Utilities Module

© 2025 Команда «Ω». Вспомогательные утилиты для NFCS

---

## 📋 Содержание модуля

### 🔧 `config_loader.py`
Загрузчик конфигурации системы:
```python
from src.utils.config_loader import ConfigLoader

config = ConfigLoader.load("config/nfcs_parameters.yml")
cgl_params = config['cgl']
```

### 📊 `data_manager.py`
Менеджер данных для входа/выхода:
```python
from src.utils.data_manager import DataManager

# Загрузка
data = DataManager.load_input("test_data.json")

# Сохранение
DataManager.save_output(results, "results.json")
```

### 📝 `logger.py`
System логирования:
```python
from src.utils.logger import get_logger

logger = get_logger("NFCS.Core")
logger.info("System initialized")
```

### 🧮 `math_utils.py`
Математические утилиты:
```python
from src.utils.math_utils import (
    normalize_complex_field,
    calculate_phase_gradient,
    detect_topological_defects
)

defects = detect_topological_defects(phi_field)
```

### ⚡ `performance.py`
Утилиты производительности:
```python
from src.utils.performance import Timer, MemoryProfiler

with Timer("CGL Solve"):
    result = solver.solve(phi_0)

profiler = MemoryProfiler()
profiler.start()
# ... код ...
memory_usage = profiler.stop()
```

### 🔒 `validation.py`
Валидация входных данных:
```python
from src.utils.validation import (
    validate_config,
    validate_field_state,
    validate_control_input
)

is_valid = validate_config(config_dict)
```

---

## 🚀 Статус реализации

| Module | Статус | Описание |
|---------|--------|----------|
| **config_loader** | ✅ Ready | Полная поддержка YAML/JSON |
| **data_manager** | ✅ Ready | Загрузка/сохранение данных |
| **logger** | ✅ Ready | Структурированное логирование |
| **math_utils** | ✅ Ready | Математические операции |
| **performance** | ✅ Ready | Профилирование производительности |
| **validation** | ✅ Ready | Валидация входных данных |

---

## 📚 Примеры использования

### Полная initialization системы
```python
from src.utils import ConfigLoader, get_logger, validate_config

# Настройка логирования
logger = get_logger("NFCS.Main")

# Загрузка конфигурации
config = ConfigLoader.load("config/production.yml")

# Валидация
if not validate_config(config):
    logger.error("Invalid configuration")
    exit(1)

logger.info("System configuration loaded successfully")
```

### Профилирование производительности
```python
from src.utils.performance import Timer, MemoryProfiler
from src.core.cgl_solver import CGLSolver

profiler = MemoryProfiler()
profiler.start()

with Timer("Full CGL Simulation"):
    solver = CGLSolver(**config['cgl'])
    
    with Timer("Field Initialization"):
        phi_0 = initialize_field(config['grid_size'])
    
    with Timer("Evolution"):
        result = solver.solve(phi_0, steps=1000)

memory_stats = profiler.stop()
logger.info(f"Peak memory usage: {memory_stats['peak_mb']} MB")
```

---

*Последнее update: 11 сентября 2025 г.*