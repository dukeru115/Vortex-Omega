#!/usr/bin/env python3
"""
Basic NFCS demonstration script.

This script shows a minimal working example of the Neural Field Control System,
demonstrating the interaction between CGL field dynamics, Kuramoto module 
synchronization, risk metrics calculation, and optimal control.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config_loader import load_config
from core.cgl_solver import CGLSolver
from core.kuramoto_solver import KuramotoSolver
from core.metrics import MetricsCalculator
from core.regulator import Regulator
from core.state import create_empty_system_state


def run_basic_simulation(num_steps=50, save_plots=True):
    """Run a basic NFCS simulation."""
    
    print("🚀 Запуск базовой симуляции NFCS...")
    
    # Load configuration
    try:
        config = load_config()
        print(f"✅ Конфигурация загружена: grid {config.cgl.grid_size}, dt={config.cgl.time_step}")
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        return
    
    # Initialize solvers
    try:
        cgl_solver = CGLSolver(config.cgl)
        print(f"✅ CGL решатель инициализирован")
        
        module_names = ['constitution', 'boundary', 'memory', 'meta_reflection']
        kuramoto_solver = KuramotoSolver(config.kuramoto, module_names)
        print(f"✅ Kuramoto решатель инициализирован для {len(module_names)} модулей")
        
        metrics_calc = MetricsCalculator(config.cost_functional)
        regulator = Regulator(config.cost_functional)
        print(f"✅ Метрики и регулятор инициализированы")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации решателей: {e}")
        return
    
    # Create initial state
    try:
        state = create_empty_system_state(
            grid_size=config.cgl.grid_size,
            n_modules=len(module_names)
        )
        
        # Set interesting initial conditions
        state.neural_field = cgl_solver.create_initial_condition(
            pattern="spiral", amplitude=0.5, m=1
        )
        
        # Random initial phases for modules
        state.module_phases = 2 * np.pi * np.random.rand(len(module_names)) - np.pi
        
        print(f"✅ Начальное состояние создано")
        
    except Exception as e:
        print(f"❌ Ошибка создания начального состояния: {e}")
        return
    
    # Storage for time series
    time_series = {
        'time': [],
        'hallucination_number': [],
        'coherence_modular': [],
        'coherence_global': [],
        'systemic_risk': [],
        'field_energy': []
    }
    
    # Main simulation loop
    print(f"\n🔄 Запуск симуляции на {num_steps} шагов...")
    
    for step in range(num_steps):
        try:
            # Calculate current metrics
            state.risk_metrics = metrics_calc.calculate_all_metrics(state)
            
            # Compute optimal control
            control_signals = regulator.compute_feedback_control(state, target_coherence=0.8)
            
            # Evolve neural field
            state.neural_field = cgl_solver.step(
                state.neural_field, 
                control_signals.u_field
            )
            
            # Update coupling matrix (simple model)
            coupling_strength = config.kuramoto.base_coupling_strength
            state.kuramoto_coupling_matrix = kuramoto_solver.build_coupling_matrix(
                base_strength=coupling_strength,
                symmetrize=True
            )
            
            # Evolve module phases
            state.module_phases = kuramoto_solver.step(
                state.module_phases,
                state.kuramoto_coupling_matrix,
                control_signals.u_modules
            )
            
            # Update state metadata
            state.simulation_time += config.cgl.time_step
            state.current_step = step
            state.last_control_signals = control_signals
            
            # Store time series data
            time_series['time'].append(state.simulation_time)
            time_series['hallucination_number'].append(state.risk_metrics.hallucination_number)
            time_series['coherence_modular'].append(state.risk_metrics.coherence_modular)
            time_series['coherence_global'].append(state.risk_metrics.coherence_global)
            time_series['systemic_risk'].append(state.risk_metrics.systemic_risk)
            time_series['field_energy'].append(state.risk_metrics.field_energy)
            
            # Progress output
            if step % 10 == 0:
                print(f"  Шаг {step:3d}: H_a={state.risk_metrics.hallucination_number:.4f}, "
                      f"R_mod={state.risk_metrics.coherence_modular:.4f}, "
                      f"Risk={state.risk_metrics.systemic_risk:.4f}")
                
        except Exception as e:
            print(f"❌ Ошибка на шаге {step}: {e}")
            break
    
    print(f"✅ Симуляция завершена!")
    
    # Final results
    print(f"\n📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"  Число Галлюцинаций H_a: {state.risk_metrics.hallucination_number:.6f}")
    print(f"  Модульная когерентность: {state.risk_metrics.coherence_modular:.6f}")
    print(f"  Глобальная когерентность: {state.risk_metrics.coherence_global:.6f}")
    print(f"  Системный риск: {state.risk_metrics.systemic_risk:.6f}")
    print(f"  Энергия поля: {state.risk_metrics.field_energy:.6f}")
    
    # Create visualizations
    if save_plots:
        create_visualization_plots(state, time_series, module_names)
    
    return state, time_series


def create_visualization_plots(state, time_series, module_names):
    """Create visualization plots of the simulation results."""
    
    print("\n📈 Создание графиков...")
    
    # Figure 1: Field visualization
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle('NFCS Neural Field State', fontsize=16)
    
    # Field amplitude
    im1 = axes[0,0].imshow(np.abs(state.neural_field), cmap='viridis')
    axes[0,0].set_title('Field Amplitude |φ|')
    axes[0,0].set_xlabel('X')
    axes[0,0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Field phase
    im2 = axes[0,1].imshow(np.angle(state.neural_field), cmap='hsv')
    axes[0,1].set_title('Field Phase arg(φ)')
    axes[0,1].set_xlabel('X')
    axes[0,1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Defect density
    im3 = axes[1,0].imshow(state.risk_metrics.rho_def_field, cmap='hot')
    axes[1,0].set_title('Defect Density ρ_def')
    axes[1,0].set_xlabel('X')
    axes[1,0].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Module phases
    theta = np.array([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])  # Circle
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    
    axes[1,1].plot(x_circle, y_circle, 'k--', alpha=0.3, label='Unit circle')
    
    for i, (name, phase) in enumerate(zip(module_names, state.module_phases)):
        x = np.cos(phase)
        y = np.sin(phase)
        axes[1,1].scatter(x, y, s=100, label=f'{name}: {phase:.2f}')
        axes[1,1].arrow(0, 0, 0.8*x, 0.8*y, head_width=0.05, 
                       head_length=0.05, fc=f'C{i}', ec=f'C{i}')
    
    axes[1,1].set_xlim(-1.2, 1.2)
    axes[1,1].set_ylim(-1.2, 1.2)
    axes[1,1].set_aspect('equal')
    axes[1,1].set_title('Module Phases θ_i')
    axes[1,1].legend(fontsize=8)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nfcs_field_state.png', dpi=300, bbox_inches='tight')
    print("  ✅ Графики состояния поля сохранены: nfcs_field_state.png")
    
    # Figure 2: Time series
    fig2, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig2.suptitle('NFCS Simulation Time Series', fontsize=16)
    
    time = np.array(time_series['time'])
    
    # Hallucination number
    axes[0,0].plot(time, time_series['hallucination_number'], 'r-', linewidth=2)
    axes[0,0].set_title('Hallucination Number H_a')
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('H_a')
    axes[0,0].grid(True, alpha=0.3)
    
    # Coherence metrics
    axes[0,1].plot(time, time_series['coherence_modular'], 'b-', 
                   linewidth=2, label='Modular R_mod')
    axes[0,1].plot(time, time_series['coherence_global'], 'g-', 
                   linewidth=2, label='Global R_glob')
    axes[0,1].set_title('Coherence Metrics')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Coherence')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Systemic risk
    axes[1,0].plot(time, time_series['systemic_risk'], 'orange', linewidth=2)
    axes[1,0].set_title('Systemic Risk')
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Risk')
    axes[1,0].grid(True, alpha=0.3)
    
    # Field energy
    axes[1,1].plot(time, time_series['field_energy'], 'purple', linewidth=2)
    axes[1,1].set_title('Field Energy')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Energy')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nfcs_time_series.png', dpi=300, bbox_inches='tight')
    print("  ✅ Графики временных рядов сохранены: nfcs_time_series.png")
    
    plt.close('all')  # Close figures to save memory


if __name__ == "__main__":
    print("=" * 60)
    print("   NEURAL FIELD CONTROL SYSTEM (NFCS) v2.4.3")
    print("   Базовая демонстрация системы")
    print("=" * 60)
    
    try:
        final_state, time_data = run_basic_simulation(num_steps=50)
        print(f"\n🎉 Демонстрация успешно завершена!")
        print(f"📁 Результаты сохранены в текущей директории")
        
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        sys.exit(1)