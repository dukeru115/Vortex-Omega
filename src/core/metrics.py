"""
Risk metrics and coherence computation for NFCS system.

Implements computation of:
- Cost functional J[φ,u] (Equation 3)
- Topological defect density ρ_def(x,t) (Equation 5) 
- Hallucination Number H_a(t) (Equation 6)
- Modular coherence R_modular(t) (Equation 9)
- Total systemic risk (Equation 42)

These metrics serve as the "nervous system" of NFCS, providing quantitative
assessment of system health, efficiency and stability.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy import ndimage
from skimage.restoration import unwrap_phase
import warnings

from .state import SystemState, RiskMetrics, CostFunctionalConfig


class MetricsCalculator:
    """
    Computes all key metrics for system state, risk and coherence assessment.
    
    This class acts as the "Observer" in NFCS architecture, continuously
    computing health indicators that drive control decisions.
    """
    
    def __init__(self, config: CostFunctionalConfig):
        """
        Initialize metrics calculator with cost functional weights.
        
        Args:
            config: Configuration with weights for different cost terms
        """
        self.config = config
        self.dx = None  # Will be set when processing fields
        self.dy = None
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        weights = [
            self.config.w_field_energy,
            self.config.w_field_gradient, 
            self.config.w_control_energy,
            self.config.w_coherence_penalty,
            self.config.w_hallucinations,
            self.config.w_defect_density,
            self.config.w_coherence_loss,
            self.config.w_violations
        ]
        
        if any(w < 0 for w in weights):
            raise ValueError("All weight parameters must be non-negative")
    
    def set_spatial_resolution(self, dx: float, dy: float):
        """
        Set spatial resolution for numerical derivatives and integration.
        
        Args:
            dx: Grid spacing in x direction
            dy: Grid spacing in y direction
        """
        if dx <= 0 or dy <= 0:
            raise ValueError("Spatial resolution must be positive")
        
        self.dx = dx
        self.dy = dy
    
    def calculate_defect_density(self, field: np.ndarray) -> np.ndarray:
        """
        Calculate topological defect density field ρ_def(x,t).
        
        Implements: ρ_def(x,t) = |∇ × arg(φ(x,t))|/(2π)
        
        Args:
            field: Complex neural field φ(x,t)
            
        Returns:
            Defect density field ρ_def
        """
        if field.dtype != np.complex128 and field.dtype != np.complex64:
            raise ValueError("Field must be complex")
        
        if self.dx is None or self.dy is None:
            raise ValueError("Spatial resolution not set. Call set_spatial_resolution() first.")
        
        # Extract phase field
        phase = np.angle(field)
        
        # Phase unwrapping to handle 2π discontinuities  
        try:
            # Use skimage unwrap_phase for better handling of branch cuts
            phase_unwrapped = unwrap_phase(phase)
        except Exception as e:
            warnings.warn(f"Phase unwrapping failed: {e}. Using raw phase.", UserWarning)
            phase_unwrapped = phase
        
        # Compute gradients using central differences
        grad_y, grad_x = np.gradient(phase_unwrapped, self.dy, self.dx)
        
        # Compute curl (vorticity) of the phase gradient 
        # For 2D: curl = ∂(grad_y)/∂x - ∂(grad_x)/∂y
        curl_gy, _ = np.gradient(grad_y, self.dy, self.dx)
        _, curl_gx = np.gradient(grad_x, self.dy, self.dx)
        
        curl = curl_gy - curl_gx
        
        # Defect density: |curl| / (2π)
        rho_def = np.abs(curl) / (2 * np.pi)
        
        return rho_def
    
    def calculate_hallucination_number(self, 
                                     rho_def_field: np.ndarray,
                                     prediction_error: float = 0.0,
                                     ontological_drift: float = 0.0,
                                     weight_prediction: float = 1.0,
                                     weight_ontology: float = 1.0) -> float:
        """
        Calculate Hallucination Number H_a(t).
        
        Implements: H_a(t) = ∫ dx [w_p·ρ_def + w_e·σ_e + w_o·drift_ont]
        
        Args:
            rho_def_field: Topological defect density field
            prediction_error: Prediction error σ_e from predictor module
            ontological_drift: Ontological drift from symbolic AI
            weight_prediction: Weight for prediction error term
            weight_ontology: Weight for ontological drift term
            
        Returns:
            Scalar Hallucination Number H_a
        """
        if self.dx is None or self.dy is None:
            raise ValueError("Spatial resolution not set")
        
        # Spatial integration of defect density
        defect_integral = np.sum(rho_def_field) * self.dx * self.dy
        
        # Weighted combination of all hallucination sources
        H_a = (defect_integral + 
               weight_prediction * prediction_error +
               weight_ontology * ontological_drift)
        
        return float(H_a)
    
    def calculate_coherence(self, 
                          field: np.ndarray,
                          phases: np.ndarray) -> Tuple[float, float]:
        """
        Calculate global field coherence and modular coherence.
        
        Args:
            field: Complex neural field φ(x,t)
            phases: Module phases θ_i(t)
            
        Returns:
            Tuple of (R_global, R_modular)
        """
        # Global field coherence: spatial average of field magnitude
        field_magnitude = np.abs(field)
        R_global = np.mean(field_magnitude)
        
        # Modular coherence from Kuramoto order parameter
        # R_modular = |1/N Σ_i exp(iθ_i)|²
        N = len(phases)
        if N == 0:
            R_modular = 0.0
        else:
            complex_sum = np.mean(np.exp(1j * phases))
            R_modular = np.abs(complex_sum)**2
        
        return float(R_global), float(R_modular)
    
    def calculate_field_energy(self, field: np.ndarray) -> float:
        """
        Calculate field energy ∫|φ|² dx.
        
        Args:
            field: Complex neural field
            
        Returns:
            Field energy
        """
        if self.dx is None or self.dy is None:
            raise ValueError("Spatial resolution not set")
        
        energy_density = np.abs(field)**2
        energy = np.sum(energy_density) * self.dx * self.dy
        
        return float(energy)
    
    def calculate_gradient_energy(self, field: np.ndarray) -> float:
        """
        Calculate gradient energy ∫|∇φ|² dx using Fourier method.
        
        Args:
            field: Complex neural field
            
        Returns:
            Gradient energy
        """
        if self.dx is None or self.dy is None:
            raise ValueError("Spatial resolution not set")
        
        # Fourier transform
        field_fft = np.fft.fft2(field)
        
        # Wave numbers
        ny, nx = field.shape
        kx = 2 * np.pi * np.fft.fftfreq(nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, self.dy)
        
        # 2D wave vector grids
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        k_squared = KX**2 + KY**2
        
        # Gradient energy in Fourier space
        gradient_energy_fft = k_squared * np.abs(field_fft)**2
        gradient_energy = np.sum(gradient_energy_fft) * self.dx * self.dy / (nx * ny)
        
        return float(gradient_energy)
    
    def calculate_control_energy(self, control_field: np.ndarray) -> float:
        """
        Calculate control energy ∫|u|² dx.
        
        Args:
            control_field: Control field u(x,t)
            
        Returns:
            Control energy
        """
        if self.dx is None or self.dy is None:
            raise ValueError("Spatial resolution not set")
        
        control_magnitude = np.abs(control_field)**2
        energy = np.sum(control_magnitude) * self.dx * self.dy
        
        return float(energy)
    
    def calculate_systemic_risk(self, 
                              hallucination_number: float,
                              rho_def_mean: float,
                              coherence_global: float,
                              coherence_modular: float,
                              violations_count: int) -> float:
        """
        Calculate total systemic risk (Equation 42).
        
        Risk_total = w₁·H_a + w₂·ρ_def + w₃·(1-R_global) + w₄·(1-R_modular) + w₅·violations
        
        Args:
            hallucination_number: H_a(t)
            rho_def_mean: Average defect density  
            coherence_global: Global field coherence
            coherence_modular: Modular coherence
            violations_count: Number of constitutional violations
            
        Returns:
            Total systemic risk score
        """
        risk_components = [
            self.config.w_hallucinations * hallucination_number,
            self.config.w_defect_density * rho_def_mean,
            self.config.w_coherence_loss * (1.0 - coherence_global),
            self.config.w_coherence_loss * (1.0 - coherence_modular),  # Reuse weight
            self.config.w_violations * violations_count
        ]
        
        total_risk = sum(risk_components)
        return float(total_risk)
    
    def calculate_all_metrics(self, state: SystemState) -> RiskMetrics:
        """
        Main method to compute all metrics and return structured object.
        
        Args:
            state: Current system state
            
        Returns:
            RiskMetrics object with all computed metrics
        """
        # Set spatial resolution from field dimensions
        # Assume field represents spatial extent from config
        ny, nx = state.neural_field.shape
        if hasattr(state, 'spatial_extent'):
            Ly, Lx = state.spatial_extent
        else:
            # Default assumption: unit domain
            Lx, Ly = 1.0, 1.0
        
        self.set_spatial_resolution(Lx/nx, Ly/ny)
        
        # Calculate defect density field
        rho_def_field = self.calculate_defect_density(state.neural_field)
        rho_def_mean = np.mean(rho_def_field)
        
        # Calculate coherence metrics
        coherence_global, coherence_modular = self.calculate_coherence(
            state.neural_field, state.module_phases
        )
        
        # Calculate Hallucination Number
        # Note: prediction_error and ontological_drift should come from
        # respective modules in a complete implementation
        hallucination_number = self.calculate_hallucination_number(
            rho_def_field,
            prediction_error=0.0,  # Placeholder
            ontological_drift=0.0  # Placeholder
        )
        
        # Calculate energy metrics
        field_energy = self.calculate_field_energy(state.neural_field)
        control_energy = self.calculate_control_energy(state.last_control_signals.u_field)
        
        # Calculate total systemic risk
        systemic_risk = self.calculate_systemic_risk(
            hallucination_number=hallucination_number,
            rho_def_mean=rho_def_mean,
            coherence_global=coherence_global,
            coherence_modular=coherence_modular,
            violations_count=0  # Should come from Constitution module
        )
        
        return RiskMetrics(
            rho_def_field=rho_def_field,
            rho_def_mean=rho_def_mean,
            hallucination_number=hallucination_number,
            systemic_risk=systemic_risk,
            coherence_global=coherence_global,
            coherence_modular=coherence_modular,
            field_energy=field_energy,
            control_energy=control_energy,
            violations_count=0,
            timestamp=state.simulation_time
        )
    
    def calculate_cost_functional(self, state: SystemState) -> float:
        """
        Calculate scalar cost functional J[φ,u] (Equation 3).
        
        Used by Regulator for optimization.
        
        J = ∫ [w₁|φ|² + w₂|∇φ|² + w₃|u|² + w₄(1-R)] dx dt
        
        Args:
            state: System state
            
        Returns:
            Scalar cost functional value
        """
        # Set spatial resolution
        ny, nx = state.neural_field.shape
        if hasattr(state, 'spatial_extent'):
            Ly, Lx = state.spatial_extent
        else:
            Lx, Ly = 1.0, 1.0
        
        self.set_spatial_resolution(Lx/nx, Ly/ny)
        
        # Energy terms
        field_energy = self.calculate_field_energy(state.neural_field)
        gradient_energy = self.calculate_gradient_energy(state.neural_field)
        control_energy = self.calculate_control_energy(state.last_control_signals.u_field)
        
        # Coherence penalty term
        coherence_global, coherence_modular = self.calculate_coherence(
            state.neural_field, state.module_phases
        )
        coherence_penalty = 1.0 - 0.5 * (coherence_global + coherence_modular)
        
        # Total cost functional
        J = (self.config.w_field_energy * field_energy +
             self.config.w_field_gradient * gradient_energy +
             self.config.w_control_energy * control_energy +
             self.config.w_coherence_penalty * coherence_penalty)
        
        return float(J)
    
    def analyze_defect_topology(self, 
                               rho_def_field: np.ndarray,
                               threshold: float = 0.01) -> Dict[str, int]:
        """
        Analyze topological structure of defects in the field.
        
        Args:
            rho_def_field: Defect density field
            threshold: Threshold for defect detection
            
        Returns:
            Dictionary with defect statistics
        """
        # Binary defect map
        defect_map = rho_def_field > threshold
        
        # Connected component analysis to count individual defects
        labeled_defects, num_defects = ndimage.label(defect_map)
        
        # Calculate defect areas
        defect_areas = []
        for i in range(1, num_defects + 1):
            area = np.sum(labeled_defects == i) * self.dx * self.dy
            defect_areas.append(area)
        
        return {
            'num_defects': num_defects,
            'total_defect_area': float(np.sum(defect_areas)) if defect_areas else 0.0,
            'mean_defect_size': float(np.mean(defect_areas)) if defect_areas else 0.0,
            'max_defect_size': float(np.max(defect_areas)) if defect_areas else 0.0
        }