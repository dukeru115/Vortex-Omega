#!/usr/bin/env python3
"""
NFCS Basic Simulation Runner

This script demonstrates a basic simulation of the Neural Field Control System.
It shows how to:
1. Load configuration
2. Initialize solvers and components
3. Run simulation loop
4. Analyze and visualize results
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_loader import load_config
from core.cgl_solver import CGLSolver
from core.kuramoto_solver import KuramotoSolver
from core.metrics import MetricsCalculator
from core.regulator import Regulator
from core.state import create_empty_system_state


def run_basic_simulation(config_path=None, n_steps=200, plot_results=True, save_data=True):
    """
    Run a basic NFCS simulation.
    
    Args:
        config_path: Path to configuration file (None for default)
        n_steps: Number of simulation steps
        plot_results: Whether to create plots
        save_data: Whether to save results to file
        
    Returns:
        Dictionary with simulation results
    """
    print("🧠 Neural Field Control System - Basic Simulation")
    print("=" * 50)
    
    # Load configuration
    print("📋 Loading configuration...")
    try:
        config = load_config(config_path)
        print(f"✅ Configuration loaded successfully")
        print(f"   Grid size: {config.cgl.grid_size}")
        print(f"   Time step: {config.cgl.time_step}")
        print(f"   Modules: {len(config.kuramoto.natural_frequencies)}")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return None
    
    # Initialize solvers
    print("\n🔧 Initializing system components...")
    try:
        # Mathematical solvers
        cgl_solver = CGLSolver(config.cgl)
        module_names = list(config.kuramoto.natural_frequencies.keys())
        kuramoto_solver = KuramotoSolver(config.kuramoto, module_names)
        
        # Analysis and control
        metrics_calculator = MetricsCalculator(config.cost_functional)
        regulator = Regulator(config.cost_functional)
        
        print(f"✅ CGL solver initialized ({config.cgl.grid_size} grid)")
        print(f"✅ Kuramoto solver initialized ({len(module_names)} modules)")
        print(f"✅ Metrics calculator ready")
        print(f"✅ Regulator ready")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None
    
    # Create initial system state
    print("\n🌱 Setting up initial conditions...")
    try:
        state = create_empty_system_state(
            grid_size=config.cgl.grid_size,
            n_modules=len(module_names)
        )
        
        # Initialize neural field with interesting pattern
        state.neural_field = cgl_solver.create_initial_condition(
            pattern="spiral",
            amplitude=0.3,
            m=1,  # Topological charge
            k_radial=0.5
        )
        
        # Add some noise for dynamics
        noise = cgl_solver.create_initial_condition(
            pattern="random_noise",
            amplitude=0.05
        )
        state.neural_field += noise
        
        # Initialize module phases (slightly out of sync)
        state.module_phases = np.array([0.1 * i for i in range(len(module_names))])
        
        # Initialize coupling matrix
        state.kuramoto_coupling_matrix = kuramoto_solver.build_coupling_matrix()
        
        print(f"✅ Initial neural field created (spiral + noise)")
        print(f"✅ Module phases initialized: {state.module_phases}")
        
    except Exception as e:
        print(f"❌ Initial conditions setup failed: {e}")
        return None
    
    # Simulation storage
    history = {
        'time': [],
        'hallucination_number': [],
        'coherence_global': [],
        'coherence_modular': [],
        'systemic_risk': [],
        'field_energy': [],
        'control_energy': [],
        'defect_density_mean': []
    }
    
    # Main simulation loop
    print(f"\n🚀 Running simulation for {n_steps} steps...")
    try:
        for step in range(n_steps):
            # Set spatial extent for metrics calculation
            state.spatial_extent = config.cgl.spatial_extent
            
            # Compute current metrics
            state.risk_metrics = metrics_calculator.calculate_all_metrics(state)
            
            # Compute optimal control
            control_signals = regulator.compute_feedback_control(
                state, target_coherence=config.coherence_target
            )
            
            # Update coupling matrix (simple time-varying example)
            modulation_factor = 1.0 + 0.1 * np.sin(0.1 * step)
            state.kuramoto_coupling_matrix = kuramoto_solver.build_coupling_matrix(
                base_strength=config.kuramoto.base_coupling_strength * modulation_factor
            )
            
            # Evolve neural field
            state.neural_field = cgl_solver.step(
                state.neural_field,
                control_signals.u_field
            )
            
            # Evolve module phases
            state.module_phases = kuramoto_solver.step(
                state.module_phases,
                state.kuramoto_coupling_matrix,
                control_signals.u_modules
            )
            
            # Update state metadata
            state.last_control_signals = control_signals
            state.simulation_time += config.cgl.time_step
            state.current_step = step
            
            # Store history
            history['time'].append(state.simulation_time)
            history['hallucination_number'].append(state.risk_metrics.hallucination_number)
            history['coherence_global'].append(state.risk_metrics.coherence_global)
            history['coherence_modular'].append(state.risk_metrics.coherence_modular)
            history['systemic_risk'].append(state.risk_metrics.systemic_risk)
            history['field_energy'].append(state.risk_metrics.field_energy)
            history['control_energy'].append(state.risk_metrics.control_energy)
            history['defect_density_mean'].append(state.risk_metrics.rho_def_mean)
            
            # Progress reporting
            if step % (n_steps // 10) == 0 or step == n_steps - 1:
                progress = (step + 1) / n_steps * 100
                print(f"   Step {step+1:4d}/{n_steps} ({progress:5.1f}%) | "
                      f"H_a={state.risk_metrics.hallucination_number:.4f} | "
                      f"R_mod={state.risk_metrics.coherence_modular:.3f} | "
                      f"Risk={state.risk_metrics.systemic_risk:.3f}")
        
        print("✅ Simulation completed successfully!")
        
    except Exception as e:
        print(f"❌ Simulation failed at step {step}: {e}")
        return None
    
    # Convert history to numpy arrays
    for key in history:
        history[key] = np.array(history[key])
    
    # Analysis and results
    print("\n📊 Simulation Analysis:")
    print(f"   Final Hallucination Number: {history['hallucination_number'][-1]:.6f}")
    print(f"   Final Modular Coherence: {history['coherence_modular'][-1]:.3f}")
    print(f"   Final Global Coherence: {history['coherence_global'][-1]:.3f}")
    print(f"   Final Systemic Risk: {history['systemic_risk'][-1]:.3f}")
    print(f"   Mean Defect Density: {history['defect_density_mean'][-1]:.6f}")
    
    # Emergency conditions check
    if history['hallucination_number'][-1] > config.emergency_threshold_ha:
        print("⚠️  WARNING: High Hallucination Number detected!")
    
    if history['defect_density_mean'][-1] > config.emergency_threshold_defects:
        print("⚠️  WARNING: High Defect Density detected!")
    
    if history['coherence_modular'][-1] < config.coherence_target * 0.5:
        print("⚠️  WARNING: Low Modular Coherence!")
    
    # Create visualization
    if plot_results:
        print("\n📈 Creating visualization...")
        create_simulation_plots(state, history, config)
    
    # Save results
    if save_data:
        print("\n💾 Saving results...")
        save_simulation_data(state, history, config)
    
    return {
        'final_state': state,
        'history': history,
        'config': config
    }


def create_simulation_plots(state, history, config):
    """Create comprehensive visualization of simulation results."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Neural field visualization
    ax1 = plt.subplot(3, 4, 1)
    im1 = ax1.imshow(np.abs(state.neural_field), cmap='viridis', origin='lower')
    ax1.set_title('Neural Field Amplitude |φ|')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Neural field phase
    ax2 = plt.subplot(3, 4, 2)
    im2 = ax2.imshow(np.angle(state.neural_field), cmap='hsv', origin='lower')
    ax2.set_title('Neural Field Phase arg(φ)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Defect density
    ax3 = plt.subplot(3, 4, 3)
    im3 = ax3.imshow(state.risk_metrics.rho_def_field, cmap='hot', origin='lower')
    ax3.set_title('Defect Density ρ_def')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3)
    
    # 4. Module phases (polar plot)
    ax4 = plt.subplot(3, 4, 4, projection='polar')
    module_names = list(config.kuramoto.natural_frequencies.keys())
    angles = state.module_phases
    colors = plt.cm.Set3(np.linspace(0, 1, len(angles)))
    
    for i, (name, angle) in enumerate(zip(module_names, angles)):
        ax4.plot([0, angle], [0, 1], 'o-', color=colors[i], 
                linewidth=3, markersize=8, label=name)
    ax4.set_title('Module Phases')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # 5. Hallucination Number over time
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(history['time'], history['hallucination_number'], 'r-', linewidth=2)
    ax5.axhline(y=config.emergency_threshold_ha, color='r', linestyle='--', 
               label='Emergency Threshold')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('H_a(t)')
    ax5.set_title('Hallucination Number')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Coherence metrics over time
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(history['time'], history['coherence_global'], 'b-', 
             linewidth=2, label='Global R_global')
    ax6.plot(history['time'], history['coherence_modular'], 'g-', 
             linewidth=2, label='Modular R_modular')
    ax6.axhline(y=config.coherence_target, color='k', linestyle='--', 
               label='Target Coherence')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Coherence')
    ax6.set_title('Coherence Metrics')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Systemic risk over time
    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(history['time'], history['systemic_risk'], 'm-', linewidth=2)
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Risk_total(t)')
    ax7.set_title('Systemic Risk')
    ax7.grid(True, alpha=0.3)
    
    # 8. Energy metrics
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(history['time'], history['field_energy'], 'b-', 
             linewidth=2, label='Field Energy')
    ax8.plot(history['time'], history['control_energy'], 'r-', 
             linewidth=2, label='Control Energy')
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Energy')
    ax8.set_title('Energy Metrics')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Defect density over time
    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(history['time'], history['defect_density_mean'], 'orange', linewidth=2)
    ax9.axhline(y=config.emergency_threshold_defects, color='r', linestyle='--',
               label='Emergency Threshold')
    ax9.set_xlabel('Time')
    ax9.set_ylabel('Mean ρ_def')
    ax9.set_title('Mean Defect Density')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10-12. Module frequency analysis
    freqs = list(config.kuramoto.natural_frequencies.values())
    for i in range(3):
        if i + 10 <= 12:
            ax = plt.subplot(3, 4, 10 + i)
            if i < len(module_names):
                # Show frequency spectrum or phase evolution
                ax.text(0.5, 0.5, f'{module_names[i]}\nω = {freqs[i]:.1f} Hz\nφ = {angles[i]:.3f}',
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7))
                ax.set_title(f'Module {i+1}')
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent.parent / "data" / "output"
    output_dir.mkdir(exist_ok=True)
    
    plot_path = output_dir / "simulation_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   📈 Plots saved to: {plot_path}")
    
    plt.show()


def save_simulation_data(state, history, config):
    """Save simulation data to files."""
    output_dir = Path(__file__).parent.parent / "data" / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save history as CSV
    import pandas as pd
    
    df = pd.DataFrame(history)
    csv_path = output_dir / "simulation_history.csv"
    df.to_csv(csv_path, index=False)
    print(f"   💾 History saved to: {csv_path}")
    
    # Save final state as NPZ
    npz_path = output_dir / "final_state.npz"
    np.savez(
        npz_path,
        neural_field=state.neural_field,
        module_phases=state.module_phases,
        coupling_matrix=state.kuramoto_coupling_matrix,
        defect_field=state.risk_metrics.rho_def_field
    )
    print(f"   💾 Final state saved to: {npz_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NFCS basic simulation")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--no-save", action="store_true", help="Disable saving data")
    
    args = parser.parse_args()
    
    # Run simulation
    results = run_basic_simulation(
        config_path=args.config,
        n_steps=args.steps,
        plot_results=not args.no_plot,
        save_data=not args.no_save
    )
    
    if results is None:
        print("❌ Simulation failed!")
        sys.exit(1)
    else:
        print("\n🎉 Simulation completed successfully!")
        print("   Check data/output/ for saved results.")