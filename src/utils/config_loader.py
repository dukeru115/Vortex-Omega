"""
Configuration loader for NFCS system.

Handles loading and validation of system parameters from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from ..core.state import SystemConfig, CGLConfig, KuramotoConfig, CostFunctionalConfig


def load_config(config_path: str = None) -> SystemConfig:
    """
    Load system configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default path.
        
    Returns:
        SystemConfig object with loaded parameters.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If configuration is invalid.
    """
    if config_path is None:
        # Default path relative to project root
        config_path = Path(__file__).parent.parent.parent / "config" / "parameters.yml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['cgl', 'kuramoto', 'cost_functional', 'system']
    for section in required_sections:
        if section not in config_data:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Build configuration objects
    cgl_config = CGLConfig(
        c1=config_data['cgl']['c1'],
        c3=config_data['cgl']['c3'],
        grid_size=tuple(config_data['cgl']['grid_size']),
        time_step=config_data['cgl']['time_step'],
        spatial_extent=tuple(config_data['cgl']['spatial_extent']),
        boundary_conditions=config_data['cgl'].get('boundary_conditions', 'periodic')
    )
    
    kuramoto_config = KuramotoConfig(
        natural_frequencies=config_data['kuramoto']['natural_frequencies'],
        base_coupling_strength=config_data['kuramoto']['base_coupling_strength'],
        time_step=config_data['kuramoto']['time_step']
    )
    
    cost_functional_config = CostFunctionalConfig(
        w_field_energy=config_data['cost_functional']['w_field_energy'],
        w_field_gradient=config_data['cost_functional']['w_field_gradient'],
        w_control_energy=config_data['cost_functional']['w_control_energy'],
        w_coherence_penalty=config_data['cost_functional']['w_coherence_penalty'],
        w_hallucinations=config_data['cost_functional']['w_hallucinations'],
        w_defect_density=config_data['cost_functional']['w_defect_density'],
        w_coherence_loss=config_data['cost_functional']['w_coherence_loss'],
        w_violations=config_data['cost_functional']['w_violations']
    )
    
    system_config = SystemConfig(
        cgl=cgl_config,
        kuramoto=kuramoto_config,
        cost_functional=cost_functional_config,
        max_simulation_steps=config_data['system']['max_simulation_steps'],
        emergency_threshold_ha=config_data['system']['emergency_threshold_ha'],
        emergency_threshold_defects=config_data['system']['emergency_threshold_defects'],
        coherence_target=config_data['system']['coherence_target']
    )
    
    return system_config


def save_config(config: SystemConfig, config_path: str):
    """
    Save system configuration to YAML file.
    
    Args:
        config: SystemConfig object to save.
        config_path: Path where to save the configuration.
    """
    config_data = {
        'cgl': {
            'c1': config.cgl.c1,
            'c3': config.cgl.c3,
            'grid_size': list(config.cgl.grid_size),
            'time_step': config.cgl.time_step,
            'spatial_extent': list(config.cgl.spatial_extent),
            'boundary_conditions': config.cgl.boundary_conditions
        },
        'kuramoto': {
            'natural_frequencies': config.kuramoto.natural_frequencies,
            'base_coupling_strength': config.kuramoto.base_coupling_strength,
            'time_step': config.kuramoto.time_step
        },
        'cost_functional': {
            'w_field_energy': config.cost_functional.w_field_energy,
            'w_field_gradient': config.cost_functional.w_field_gradient,
            'w_control_energy': config.cost_functional.w_control_energy,
            'w_coherence_penalty': config.cost_functional.w_coherence_penalty,
            'w_hallucinations': config.cost_functional.w_hallucinations,
            'w_defect_density': config.cost_functional.w_defect_density,
            'w_coherence_loss': config.cost_functional.w_coherence_loss,
            'w_violations': config.cost_functional.w_violations
        },
        'system': {
            'max_simulation_steps': config.max_simulation_steps,
            'emergency_threshold_ha': config.emergency_threshold_ha,
            'emergency_threshold_defects': config.emergency_threshold_defects,
            'coherence_target': config.coherence_target
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)


def validate_config(config: SystemConfig) -> bool:
    """
    Validate configuration parameters for physical and numerical consistency.
    
    Args:
        config: SystemConfig to validate.
        
    Returns:
        True if configuration is valid.
        
    Raises:
        ValueError: If configuration contains invalid values.
    """
    # Validate CGL parameters
    if config.cgl.time_step <= 0:
        raise ValueError("CGL time step must be positive")
    
    if any(dim <= 0 for dim in config.cgl.grid_size):
        raise ValueError("Grid dimensions must be positive")
    
    if any(ext <= 0 for ext in config.cgl.spatial_extent):
        raise ValueError("Spatial extent must be positive")
    
    # Validate Kuramoto parameters
    if config.kuramoto.time_step <= 0:
        raise ValueError("Kuramoto time step must be positive")
    
    if config.kuramoto.base_coupling_strength < 0:
        raise ValueError("Base coupling strength must be non-negative")
    
    if not config.kuramoto.natural_frequencies:
        raise ValueError("At least one module frequency must be specified")
    
    for freq_name, freq_value in config.kuramoto.natural_frequencies.items():
        if not isinstance(freq_name, str):
            raise ValueError(f"Module name must be string: {freq_name}")
        if freq_value <= 0:
            raise ValueError(f"Frequency must be positive for module {freq_name}: {freq_value}")
    
    # Validate cost functional weights (should be non-negative)
    weight_attrs = ['w_field_energy', 'w_field_gradient', 'w_control_energy', 
                   'w_coherence_penalty', 'w_hallucinations', 'w_defect_density',
                   'w_coherence_loss', 'w_violations']
    
    for attr in weight_attrs:
        value = getattr(config.cost_functional, attr)
        if value < 0:
            raise ValueError(f"Weight {attr} must be non-negative: {value}")
    
    # Validate system parameters
    if config.max_simulation_steps <= 0:
        raise ValueError("Maximum simulation steps must be positive")
    
    if not 0 <= config.emergency_threshold_ha <= 1:
        raise ValueError("Emergency threshold for hallucinations must be in [0,1]")
    
    if config.emergency_threshold_defects < 0:
        raise ValueError("Emergency threshold for defects must be non-negative")
    
    if not 0 <= config.coherence_target <= 1:
        raise ValueError("Coherence target must be in [0,1]")
    
    return True