# NFCS API Reference

© 2025 Команда «Ω». Полная документация API для Neural Field Control System

---

## 🔌 Основные интерфейсы API

### 1. 🧮 Core Mathematical API

#### CGL Solver API
```python
from src.core.cgl_solver import CGLSolver

# Initialization решателя
solver = CGLSolver(
    grid_size=(128, 128),
    c1=0.8,
    c3=1.5,
    dt=0.01
)

# Решение уравнения
result = solver.solve(
    initial_field=phi_0,
    control_input=u_t,
    time_steps=1000
)
```

#### Kuramoto Model API
```python
from src.core.enhanced_kuramoto import EnhancedKuramoto

# Creation модели синхронизации
kuramoto = EnhancedKuramoto(
    n_modules=8,
    natural_frequencies=[2.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0],
    coupling_strength=1.0
)

# Synchronization модулей
phases = kuramoto.synchronize(
    control_signals=u_control,
    time_horizon=100
)
```

### 2. 🧠 Cognitive Modules API

#### Constitutional Module API
```python
from src.modules.constitution_core import ConstitutionCore

# Initialization конституционного ядра
constitution = ConstitutionCore(
    integrity_threshold=0.7,
    safety_protocols=True
)

# Check конституционности
result = constitution.validate_action(
    proposed_action=action,
    system_state=current_state,
    context=context_info
)
```

#### ESC Module API
```python
from src.modules.esc.esc_core import ESCCore

# Creation эхо-семантического конвертера
esc = ESCCore(
    oscillator_count=1024,
    frequency_range=(0.1, 100.0),
    echo_scales=[0.1, 1.0, 10.0, 100.0]
)

# Processing токенов
eta_t = esc.process_tokens(
    tokens=["hello", "world", "NFCS"],
    context=conversation_context
)
```

### 3. ⚙️ Orchestrator API

#### Main Orchestrator API
```python
from src.orchestrator.nfcs_orchestrator import NFCSOrchestrator

# Initialization системы
nfcs = NFCSOrchestrator(
    config_path="config/nfcs_parameters.yml"
)

# Start системы
nfcs.initialize()

# Processing входа
response = nfcs.process_input(
    input_data={
        "tokens": ["input", "text"],
        "modality": "text",
        "context": {"user_id": "123"}
    }
)

# Получение метрик
metrics = nfcs.get_system_metrics()
print(f"Ha: {metrics['hallucination_number']}")
print(f"Coherence: {metrics['coherence_level']}")
```

---

## 📊 Configuration API

### Загрузка конфигурации
```python
from src.utils.config_loader import ConfigLoader

# Загрузка параметров
config = ConfigLoader.load("config/nfcs_parameters.yml")

# Доступ к параметрам
cgl_params = config['cgl']
kuramoto_params = config['kuramoto']
safety_params = config['safety']
```

### Динамическое change параметров
```python
# Change параметров во time выполнения
nfcs.update_parameters({
    'kuramoto.base_coupling_strength': 1.5,
    'safety.hallucination_threshold': 3.0
})

# Сохранение новой конфигурации
nfcs.save_configuration("config/updated_params.yml")
```

---

## 🔍 Monitoring и метрики API

### Системные метрики
```python
# Получение полных метрик
all_metrics = nfcs.get_comprehensive_metrics()

# Основные показатели
print(f"System Status: {all_metrics['status']}")
print(f"Uptime: {all_metrics['uptime']}")
print(f"Ha Number: {all_metrics['ha_number']}")
print(f"Coherence R: {all_metrics['coherence_measure']}")
print(f"Defect Density: {all_metrics['defect_density']}")

# Метрики модулей
module_metrics = all_metrics['modules']
for module_name, metrics in module_metrics.items():
    print(f"{module_name}: {metrics['status']} (freq: {metrics['frequency']})")
```

### Real-time monitoring
```python
# Подписка на события
def on_metrics_update(metrics):
    if metrics['ha_number'] > 2.5:
        print("⚠️ WARNING: High hallucination risk!")
    
    if metrics['coherence_measure'] < 0.3:
        print("🔄 INFO: Low coherence detected")

nfcs.subscribe_to_metrics(on_metrics_update)

# Start мониторинга
nfcs.start_monitoring(update_frequency=10)  # 10 Hz
```

---

## 🛡️ Safety и валидация API

### Constitutional Validation
```python
# Check действий на конституционность
validation_result = nfcs.constitutional_check(
    action_type="generate_response",
    content=proposed_response,
    context=current_context
)

if validation_result.is_valid:
    execute_action()
else:
    handle_violation(validation_result.violation_reason)
```

### Risk Assessment
```python
# Оценка рисков в реальном времени
risk_assessment = nfcs.assess_risks(
    current_state=system_state,
    proposed_changes=planned_actions
)

print(f"Risk Level: {risk_assessment.level}")
print(f"Risk Factors: {risk_assessment.factors}")
print(f"Mitigation: {risk_assessment.mitigation_strategies}")
```

---

## 🧪 Экспериментальные функции API

### Vortex Protocol Integration
```python
from документы.разработчик.vortex_protocol import VortexProtocol

# Creation Vortex экземпляра
vortex = VortexProtocol(
    boundary_sensitivity=0.7,
    meta_reflection_threshold=0.5,
    freedom_creativity_level=0.8
)

# Processing через Vortex
vortex_response = vortex.process_dialogue(
    user_input="Может ли машина быть творческой?",
    conversation_history=history,
    system_context=context
)
```

### Evolution and Learning API
```python
# Эволюционное обучение параметров
evolution_result = nfcs.evolve_parameters(
    generations=100,
    population_size=50,
    fitness_function=custom_fitness,
    mutation_rate=0.1
)

# Cultural Evolution между экземплярами
nfcs.share_patterns(
    target_instance=other_nfcs,
    pattern_types=['successful_responses', 'safety_protocols'],
    confidence_threshold=0.8
)
```

---

## 📝 Примеры использования

### Полный workflow
```python
import asyncio
from src.main import NFCS

async def main():
    # Initialization
    nfcs = NFCS("config/production.yml")
    await nfcs.initialize()
    
    # Processing входа
    result = await nfcs.process_input({
        "text": "Объясни принцип работы NFCS",
        "user_context": {"expertise": "expert"},
        "safety_level": "high"
    })
    
    # Check результата
    if result.constitutional_valid and result.ha_number < 1.0:
        print(f"Response: {result.content}")
        print(f"Confidence: {result.confidence}")
        print(f"Coherence: {result.coherence}")
    else:
        print("Response rejected by safety systems")
    
    # Завершение
    await nfcs.shutdown()

# Start
asyncio.run(main())
```

---

## 🔧 Статус реализации API

| Module | Статус | Покрытие | Документация |
|--------|---------|----------|--------------|
| **Core Math** | ✅ Ready | 100% | ✅ Полная |
| **Cognitive Modules** | ✅ Ready | 100% | ✅ Полная |
| **Orchestrator** | ✅ Ready | 100% | ✅ Полная |
| **ESC System** | ✅ Ready | 95% | ✅ Полная |
| **Safety & Constitution** | ✅ Ready | 100% | ✅ Полная |
| **Monitoring** | ✅ Ready | 90% | 🔄 В процессе |
| **Evolution** | 🔄 Beta | 80% | 📋 Планируется |

---

## 📚 Дополнительные ресурсы

- [ARCHITECTURE.md](../../ARCHITECTURE.md) — Техническая архитектура
- [документы/разработчик/архитектура.md](../../документы/разработчик/архитектура.md) — Полная архитектура (RU)
- [docs/testing/](../testing/) — Руководство по тестированию
- [src/](../../src/) — Исходный код
- [tests/](../../tests/) — Тесты API

---

*Последнее update: 11 сентября 2025 г.*  
*Версия API: 2.4.3*  
*Статус: Production Ready* ✅