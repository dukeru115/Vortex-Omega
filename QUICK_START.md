# Быстрый старт NFCS

## Базовый пример симуляции

```python
import numpy as np
from src.utils.config_loader import load_config
from src.core.cgl_solver import CGLSolver
from src.core.kuramoto_solver import KuramotoSolver
from src.core.metrics import MetricsCalculator
from src.core.regulator import Regulator
from src.core.state import create_empty_system_state

# Загрузка конфигурации
config = load_config()

# Initialization решателей
cgl_solver = CGLSolver(config.cgl)
kuramoto_solver = KuramotoSolver(
    config.kuramoto, 
    module_order=['constitution', 'boundary', 'memory', 'meta_reflection']
)

# Initialization метрик и регулятора  
metrics_calc = MetricsCalculator(config.cost_functional)
regulator = Regulator(config.cost_functional)

# Creation начального состояния
initial_state = create_empty_system_state(
    grid_size=config.cgl.grid_size,
    n_modules=len(config.kuramoto.natural_frequencies)
)

# Установка начальных условий
initial_state.neural_field = cgl_solver.create_initial_condition(
    pattern="random_noise", amplitude=0.1
)

# Основной цикл симуляции
current_state = initial_state
for step in range(100):
    # Расчет метрик текущего состояния
    current_state.risk_metrics = metrics_calc.calculate_all_metrics(current_state)
    
    # Вычисление оптимального управления
    control_signals = regulator.compute_optimal_control(current_state)
    
    # Эволюция нейронного поля
    current_state.neural_field = cgl_solver.step(
        current_state.neural_field, 
        control_signals.u_field
    )
    
    # Эволюция фаз модулей
    current_state.module_phases = kuramoto_solver.step(
        current_state.module_phases,
        current_state.kuramoto_coupling_matrix,
        control_signals.u_modules
    )
    
    # Update времени
    current_state.simulation_time += config.cgl.time_step
    current_state.current_step = step
    
    # Вывод прогресса
    if step % 10 == 0:
        print(f"Шаг {step}: H_a = {current_state.risk_metrics.hallucination_number:.4f}, "
              f"R_mod = {current_state.risk_metrics.coherence_modular:.4f}")

print("Симуляция completed!")
```

## Анализ результатов

```python
import matplotlib.pyplot as plt

# Визуализация нейронного поля
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Amplitude поля
im1 = axes[0].imshow(np.abs(current_state.neural_field), cmap='viridis')
axes[0].set_title('Amplitude нейронного поля |φ|')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
plt.colorbar(im1, ax=axes[0])

# Плотность дефектов
im2 = axes[1].imshow(current_state.risk_metrics.rho_def_field, cmap='hot')
axes[1].set_title('Плотность топологических дефектов ρ_def')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# Вывод ключевых метрик
print("\n=== РЕЗУЛЬТАТЫ СИМУЛЯЦИИ ===")
print(f"Число Галлюцинаций H_a: {current_state.risk_metrics.hallucination_number:.6f}")
print(f"Модульная coherence R_mod: {current_state.risk_metrics.coherence_modular:.6f}")
print(f"Глобальная coherence R_glob: {current_state.risk_metrics.coherence_global:.6f}")
print(f"Системный risk: {current_state.risk_metrics.systemic_risk:.6f}")
print(f"Средняя плотность дефектов: {current_state.risk_metrics.rho_def_mean:.6f}")
```