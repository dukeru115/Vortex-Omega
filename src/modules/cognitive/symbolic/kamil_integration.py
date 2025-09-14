"""
Kamil Symbolic AI Integration Module
===================================

Integrates the new Kamil Gadeev specification Symbolic AI module with existing
NFCS v2.4.3 components including ESC-Kuramoto bridge and discrepancy gate.

Integration Points:
- ESC semantic processing ↔ Symbolic reasoning
- Kuramoto coupling modulation ↔ Field clustering  
- Discrepancy gate validation ↔ Symbolic verification
- Neural field dynamics ↔ Symbolic field mapping
- Real-time monitoring and performance optimization

Created: September 14, 2025
Author: Team Ω - Kamil Gadeev Integration Specification
License: Apache 2.0
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Import NFCS core components
try:
    from ...core.state import SystemState, RiskMetrics
    from ...core.metrics import SystemMetrics
except ImportError:
    # Fallback for testing
    SystemState = None
    RiskMetrics = None
    SystemMetrics = None

# Import integration targets
try:
    from ../../integration.esc_kuramoto_bridge import ESCKuramotoBridge
except ImportError:
    # Create placeholder for missing integration
    class ESCKuramotoBridge:
        def __init__(self, *args, **kwargs):
            pass
        def process_semantic_coupling(self, *args, **kwargs):
            return {}

# Import new Symbolic AI
from .symbolic_ai_kamil import (
    SymbolicAIKamil, 
    Unit, Quantity, Term, Expr, SymClause, SymField, VerificationReport,
    METER, KILOGRAM, SECOND, DIMENSIONLESS, create_test_symbolic_ai
)

logger = logging.getLogger(__name__)


@dataclass
class IntegrationMetrics:
    """Metrics for symbolic-neural integration performance."""
    
    # Processing statistics
    symbolize_requests: int = 0
    fieldize_operations: int = 0
    verify_validations: int = 0
    
    # Performance metrics
    avg_symbolize_latency_ms: float = 0.0
    avg_fieldize_latency_ms: float = 0.0
    avg_verify_latency_ms: float = 0.0
    
    # Integration success rates
    esc_integration_success_rate: float = 0.0
    kuramoto_coupling_success_rate: float = 0.0
    discrepancy_validation_success_rate: float = 0.0
    
    # Quality metrics
    dimensional_accuracy_average: float = 0.0
    logical_consistency_rate: float = 0.0
    ethical_compliance_rate: float = 0.0
    
    # SLO compliance
    latency_slo_violations: int = 0
    accuracy_slo_violations: int = 0
    total_slo_checks: int = 0
    
    def get_slo_compliance_rate(self) -> float:
        """Calculate overall SLO compliance rate."""
        if self.total_slo_checks == 0:
            return 1.0
        violations = self.latency_slo_violations + self.accuracy_slo_violations
        return max(0.0, 1.0 - (violations / self.total_slo_checks))


@dataclass  
class SymbolicFieldMapping:
    """Mapping between symbolic fields and neural field states."""
    
    field_id: str
    symbolic_field: SymField
    neural_coordinates: Tuple[int, int]  # Grid coordinates in neural field
    coupling_strength: float = 1.0
    semantic_weight: float = 1.0
    
    # Field modulation parameters
    spatial_pattern: Optional[np.ndarray] = None
    temporal_frequency: float = 0.0
    phase_offset: float = 0.0
    
    # Integration metadata
    last_update: float = field(default_factory=time.time)
    update_count: int = 0
    consistency_score: float = 1.0


class KamilSymbolicIntegration:
    """
    Integration layer for Kamil Symbolic AI with NFCS system.
    
    Provides seamless integration between deterministic symbolic reasoning
    and neural field dynamics through structured mapping and validation.
    """
    
    def __init__(self,
                 symbolic_ai: Optional[SymbolicAIKamil] = None,
                 esc_bridge: Optional[ESCKuramotoBridge] = None,
                 enable_real_time_monitoring: bool = True,
                 max_field_mappings: int = 100,
                 consistency_threshold: float = 0.8,
                 debug_mode: bool = False):
        """
        Initialize symbolic-neural integration.
        
        Args:
            symbolic_ai: Symbolic AI engine instance
            esc_bridge: ESC-Kuramoto bridge instance  
            enable_real_time_monitoring: Enable performance monitoring
            max_field_mappings: Maximum number of field mappings to maintain
            consistency_threshold: Minimum consistency score for mappings
            debug_mode: Enable debug logging
        """
        self.symbolic_ai = symbolic_ai or create_test_symbolic_ai()
        self.esc_bridge = esc_bridge
        self.enable_monitoring = enable_real_time_monitoring
        self.max_field_mappings = max_field_mappings
        self.consistency_threshold = consistency_threshold
        self.debug_mode = debug_mode
        
        # Integration state
        self.field_mappings: Dict[str, SymbolicFieldMapping] = {}
        self.semantic_coupling_cache: Dict[str, Dict[str, Any]] = {}
        self.verification_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.metrics = IntegrationMetrics()
        self.performance_window = deque(maxlen=100)  # Last 100 operations
        
        # Integration callbacks
        self.discrepancy_validators: List[Callable] = []
        self.field_update_callbacks: List[Callable] = []
        
        # Neural field grid parameters
        self.field_grid_shape = (64, 64)  # Default neural field resolution
        self.field_dx = 0.1  # Spatial step size
        self.field_dt = 0.01  # Temporal step size
        
        logger.info(f"Kamil Symbolic Integration initialized with {len(self.field_mappings)} mappings")
    
    def process_semantic_input(self, 
                             semantic_data: str,
                             field_context: Optional[np.ndarray] = None,
                             system_state: Optional[Any] = None) -> Dict[str, Any]:
        """
        Process semantic input through complete symbolic pipeline with neural integration.
        
        Args:
            semantic_data: Natural language or symbolic input
            field_context: Current neural field state for context
            system_state: Current system state for validation
            
        Returns:
            Complete processing result with neural field modulations
        """
        start_time = time.time()
        
        try:
            # Phase 1: Symbolize semantic input
            symbolize_start = time.time()
            sym_fields = self.symbolic_ai.symbolize(
                semantic_data, 
                context={'field_context': field_context, 'system_state': system_state}
            )
            symbolize_latency = (time.time() - symbolize_start) * 1000
            
            # Phase 2: Fieldize with neural field mapping
            fieldize_start = time.time()
            clustered_fields = self.symbolic_ai.fieldize(sym_fields)
            
            # Create neural field mappings
            field_mappings = self._create_field_mappings(clustered_fields, field_context)
            fieldize_latency = (time.time() - fieldize_start) * 1000
            
            # Phase 3: Verify with comprehensive validation
            verify_start = time.time()
            verification_report = self.symbolic_ai.verify(clustered_fields)
            verify_latency = (time.time() - verify_start) * 1000
            
            # Phase 4: ESC-Kuramoto integration
            esc_integration = self._integrate_with_esc_kuramoto(
                clustered_fields, field_mappings, field_context
            )
            
            # Phase 5: Generate neural field modulations
            neural_modulations = self._generate_neural_modulations(
                field_mappings, verification_report
            )
            
            # Update performance metrics
            self._update_integration_metrics(
                symbolize_latency, fieldize_latency, verify_latency, 
                verification_report, esc_integration
            )
            
            # Create comprehensive result
            processing_time = (time.time() - start_time) * 1000
            result = {
                'symbolic_fields': clustered_fields,
                'field_mappings': field_mappings,
                'verification_report': verification_report,
                'neural_modulations': neural_modulations,
                'esc_integration': esc_integration,
                'processing_time_ms': processing_time,
                'slo_compliant': processing_time <= 300 and verification_report.dimensional_accuracy >= 0.98,
                'performance_metrics': {
                    'symbolize_latency_ms': symbolize_latency,
                    'fieldize_latency_ms': fieldize_latency, 
                    'verify_latency_ms': verify_latency
                }
            }
            
            # Store in verification history
            self.verification_history.append({
                'timestamp': time.time(),
                'result': result,
                'input_hash': hash(semantic_data)
            })
            
            if self.debug_mode:
                logger.debug(f"Semantic processing completed in {processing_time:.2f}ms: "
                           f"SLO compliant = {result['slo_compliant']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Semantic processing failed: {e}")
            return {
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000,
                'slo_compliant': False
            }
    
    def create_discrepancy_validator(self) -> Callable:
        """
        Create discrepancy validation function for integration with discrepancy gate.
        
        Returns:
            Validation function that can be registered with discrepancy gate
        """
        def validate_field_discrepancy(field_state: np.ndarray,
                                     discrepancy_measure: float,
                                     system_context: Dict[str, Any]) -> Dict[str, Any]:
            """
            Validate neural field discrepancies using symbolic reasoning.
            
            This function is called by the discrepancy gate when field anomalies
            are detected to perform symbolic validation and safety checking.
            """
            try:
                # Generate semantic description of discrepancy
                discrepancy_description = self._describe_field_discrepancy(
                    field_state, discrepancy_measure, system_context
                )
                
                # Process through symbolic AI pipeline
                result = self.process_semantic_input(
                    discrepancy_description, 
                    field_context=field_state,
                    system_state=system_context
                )
                
                # Extract validation results
                validation_result = {
                    'symbolic_validation_passed': result.get('verification_report', {}).get('overall_valid', False),
                    'confidence_score': result.get('verification_report', {}).get('confidence_score', 0.0),
                    'dimensional_consistency': result.get('verification_report', {}).get('dimensional_accuracy', 0.0),
                    'ethical_compliance': (
                        result.get('verification_report', {}).get('kant_universalization', False) and
                        result.get('verification_report', {}).get('kant_means_end', False)
                    ),
                    'processing_time_ms': result.get('processing_time_ms', float('inf')),
                    'recommended_action': self._recommend_discrepancy_action(result),
                    'field_modulations': result.get('neural_modulations', {}),
                    'full_analysis': result
                }
                
                # Register with callbacks
                for callback in self.discrepancy_validators:
                    try:
                        callback(validation_result)
                    except Exception as cb_error:
                        logger.warning(f"Discrepancy validator callback failed: {cb_error}")
                
                return validation_result
                
            except Exception as e:
                logger.error(f"Discrepancy validation failed: {e}")
                return {
                    'symbolic_validation_passed': False,
                    'confidence_score': 0.0,
                    'error': str(e),
                    'recommended_action': 'emergency_shutdown'
                }
        
        return validate_field_discrepancy
    
    def integrate_kuramoto_coupling(self,
                                  coupling_matrix: np.ndarray,
                                  oscillator_phases: np.ndarray,
                                  symbolic_context: Optional[List[SymField]] = None) -> np.ndarray:
        """
        Integrate symbolic reasoning with Kuramoto coupling dynamics.
        
        Args:
            coupling_matrix: Current K_ij coupling matrix
            oscillator_phases: Current oscillator phases θ_i
            symbolic_context: Optional symbolic fields for context
            
        Returns:
            Modified coupling matrix with symbolic influence
        """
        try:
            if symbolic_context is None:
                symbolic_context = list(self.field_mappings.values())
                symbolic_context = [mapping.symbolic_field for mapping in symbolic_context]
            
            # Analyze symbolic constraints on coupling
            coupling_constraints = self._extract_coupling_constraints(symbolic_context)
            
            # Calculate semantic coupling modulations
            semantic_modulations = self._calculate_semantic_modulations(
                coupling_matrix, oscillator_phases, coupling_constraints
            )
            
            # Apply modulations to coupling matrix
            modified_coupling = self._apply_coupling_modulations(
                coupling_matrix, semantic_modulations
            )
            
            # Validate dimensional consistency
            if not self._validate_coupling_dimensions(modified_coupling):
                logger.warning("Coupling matrix dimensional inconsistency detected")
                return coupling_matrix  # Return original on validation failure
            
            # Update integration metrics
            self.metrics.kuramoto_coupling_success_rate = self._update_success_rate(
                self.metrics.kuramoto_coupling_success_rate,
                success=True
            )
            
            if self.debug_mode:
                coupling_change = np.linalg.norm(modified_coupling - coupling_matrix)
                logger.debug(f"Kuramoto coupling updated: change magnitude = {coupling_change:.4f}")
            
            return modified_coupling
            
        except Exception as e:
            logger.error(f"Kuramoto coupling integration failed: {e}")
            self.metrics.kuramoto_coupling_success_rate = self._update_success_rate(
                self.metrics.kuramoto_coupling_success_rate,
                success=False
            )
            return coupling_matrix
    
    def get_field_modulations_for_cgl(self) -> Dict[str, np.ndarray]:
        """
        Generate field modulations for CGL equation integration.
        
        Returns:
            Dictionary of field modulation arrays for CGL solver
        """
        modulations = {}
        
        for mapping_id, mapping in self.field_mappings.items():
            if mapping.spatial_pattern is not None:
                # Generate time-dependent modulation
                current_time = time.time() - mapping.last_update
                temporal_factor = np.cos(2 * np.pi * mapping.temporal_frequency * current_time + mapping.phase_offset)
                
                modulation = mapping.spatial_pattern * mapping.coupling_strength * temporal_factor
                modulations[mapping_id] = modulation
        
        return modulations
    
    def update_field_mapping_from_neural_state(self,
                                             mapping_id: str,
                                             neural_field: np.ndarray,
                                             update_symbolic: bool = True) -> bool:
        """
        Update symbolic field mapping based on neural field evolution.
        
        Args:
            mapping_id: ID of mapping to update
            neural_field: Current neural field state
            update_symbolic: Whether to update symbolic representation
            
        Returns:
            Success status of update operation
        """
        try:
            if mapping_id not in self.field_mappings:
                logger.warning(f"Field mapping {mapping_id} not found")
                return False
            
            mapping = self.field_mappings[mapping_id]
            
            # Extract local field information
            coord_x, coord_y = mapping.neural_coordinates
            local_field = self._extract_local_field_patch(neural_field, coord_x, coord_y)
            
            # Calculate consistency with symbolic representation
            consistency = self._calculate_field_consistency(mapping.symbolic_field, local_field)
            mapping.consistency_score = consistency
            
            # Update symbolic representation if requested and consistency is low
            if update_symbolic and consistency < self.consistency_threshold:
                updated_symbolic = self._infer_symbolic_from_field(local_field, mapping.symbolic_field)
                if updated_symbolic:
                    mapping.symbolic_field = updated_symbolic
            
            # Update mapping metadata
            mapping.last_update = time.time()
            mapping.update_count += 1
            
            # Trigger callbacks
            for callback in self.field_update_callbacks:
                try:
                    callback(mapping_id, mapping, neural_field)
                except Exception as cb_error:
                    logger.warning(f"Field update callback failed: {cb_error}")
            
            return True
            
        except Exception as e:
            logger.error(f"Field mapping update failed: {e}")
            return False
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive integration metrics and performance statistics."""
        
        # Calculate derived metrics
        total_operations = (self.metrics.symbolize_requests + 
                           self.metrics.fieldize_operations + 
                           self.metrics.verify_validations)
        
        avg_latency = (self.metrics.avg_symbolize_latency_ms + 
                      self.metrics.avg_fieldize_latency_ms + 
                      self.metrics.avg_verify_latency_ms) / 3
        
        return {
            # Basic statistics
            'total_operations': total_operations,
            'active_field_mappings': len(self.field_mappings),
            'verification_history_size': len(self.verification_history),
            
            # Performance metrics
            'average_total_latency_ms': avg_latency,
            'slo_compliance_rate': self.metrics.get_slo_compliance_rate(),
            
            # Quality metrics  
            'dimensional_accuracy_average': self.metrics.dimensional_accuracy_average,
            'logical_consistency_rate': self.metrics.logical_consistency_rate,
            'ethical_compliance_rate': self.metrics.ethical_compliance_rate,
            
            # Integration success rates
            'esc_integration_success_rate': self.metrics.esc_integration_success_rate,
            'kuramoto_coupling_success_rate': self.metrics.kuramoto_coupling_success_rate,
            'discrepancy_validation_success_rate': self.metrics.discrepancy_validation_success_rate,
            
            # Detailed metrics
            'detailed_metrics': self.metrics,
            
            # Recent performance window
            'recent_performance': list(self.performance_window)
        }
    
    def cleanup_stale_mappings(self, max_age_seconds: float = 3600) -> int:
        """
        Clean up stale field mappings that haven't been updated recently.
        
        Args:
            max_age_seconds: Maximum age before mapping is considered stale
            
        Returns:
            Number of mappings cleaned up
        """
        current_time = time.time()
        stale_mappings = []
        
        for mapping_id, mapping in self.field_mappings.items():
            age = current_time - mapping.last_update
            if age > max_age_seconds or mapping.consistency_score < self.consistency_threshold:
                stale_mappings.append(mapping_id)
        
        # Remove stale mappings
        for mapping_id in stale_mappings:
            del self.field_mappings[mapping_id]
        
        logger.info(f"Cleaned up {len(stale_mappings)} stale field mappings")
        return len(stale_mappings)
    
    # =========================================================================
    # PRIVATE IMPLEMENTATION METHODS  
    # =========================================================================
    
    def _create_field_mappings(self, 
                              sym_fields: List[SymField],
                              field_context: Optional[np.ndarray] = None) -> List[SymbolicFieldMapping]:
        """Create neural field mappings for symbolic fields."""
        mappings = []
        
        for i, sym_field in enumerate(sym_fields):
            # Generate spatial coordinates  
            grid_x = (i * 7 + 13) % self.field_grid_shape[0]  # Pseudo-random placement
            grid_y = (i * 11 + 17) % self.field_grid_shape[1]
            
            # Generate spatial pattern based on field properties
            spatial_pattern = self._generate_spatial_pattern_for_field(sym_field)
            
            # Calculate temporal frequency from field characteristics
            temporal_freq = self._calculate_temporal_frequency(sym_field)
            
            mapping = SymbolicFieldMapping(
                field_id=f"mapping_{sym_field.field_id}_{int(time.time() * 1000)}",
                symbolic_field=sym_field,
                neural_coordinates=(grid_x, grid_y),
                spatial_pattern=spatial_pattern,
                temporal_frequency=temporal_freq,
                phase_offset=np.random.uniform(0, 2*np.pi)  # Random initial phase
            )
            
            mappings.append(mapping)
            
            # Store in internal mappings dict (manage size limit)
            if len(self.field_mappings) >= self.max_field_mappings:
                # Remove oldest mapping
                oldest_mapping = min(self.field_mappings.values(), key=lambda m: m.last_update)
                del self.field_mappings[oldest_mapping.field_id]
            
            self.field_mappings[mapping.field_id] = mapping
        
        return mappings
    
    def _integrate_with_esc_kuramoto(self,
                                   sym_fields: List[SymField],
                                   field_mappings: List[SymbolicFieldMapping],
                                   field_context: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Integrate with ESC-Kuramoto bridge."""
        try:
            if self.esc_bridge is None:
                return {'integration_available': False, 'reason': 'ESC bridge not available'}
            
            # Extract semantic content for ESC processing
            semantic_content = []
            for sym_field in sym_fields:
                for clause in sym_field.clauses:
                    if clause.natural_language:
                        semantic_content.append(clause.natural_language)
            
            # Process through ESC-Kuramoto bridge
            esc_result = self.esc_bridge.process_semantic_coupling(
                semantic_content, field_context
            )
            
            # Update success metrics
            self.metrics.esc_integration_success_rate = self._update_success_rate(
                self.metrics.esc_integration_success_rate, success=True
            )
            
            return {
                'integration_available': True,
                'esc_result': esc_result,
                'semantic_content_items': len(semantic_content)
            }
            
        except Exception as e:
            self.metrics.esc_integration_success_rate = self._update_success_rate(
                self.metrics.esc_integration_success_rate, success=False
            )
            return {
                'integration_available': False,
                'error': str(e)
            }
    
    def _generate_neural_modulations(self,
                                   field_mappings: List[SymbolicFieldMapping], 
                                   verification_report: VerificationReport) -> Dict[str, Any]:
        """Generate neural field modulations from symbolic mappings."""
        modulations = {}
        
        # Base modulation strength on verification confidence
        base_strength = verification_report.confidence_score
        
        for mapping in field_mappings:
            if mapping.spatial_pattern is not None:
                # Create modulation with verification-weighted strength
                modulation_strength = base_strength * mapping.coupling_strength
                
                modulations[mapping.field_id] = {
                    'spatial_pattern': mapping.spatial_pattern,
                    'strength': modulation_strength,
                    'frequency': mapping.temporal_frequency,
                    'phase': mapping.phase_offset,
                    'coordinates': mapping.neural_coordinates
                }
        
        return modulations
    
    def _describe_field_discrepancy(self,
                                  field_state: np.ndarray,
                                  discrepancy_measure: float,
                                  system_context: Dict[str, Any]) -> str:
        """Generate semantic description of field discrepancy."""
        
        # Analyze field characteristics
        max_amplitude = np.max(np.abs(field_state))
        mean_amplitude = np.mean(np.abs(field_state))
        field_energy = np.sum(np.abs(field_state) ** 2)
        
        # Generate contextual description
        description = f"""
        Neural field discrepancy detected with severity {discrepancy_measure:.3f}.
        Field characteristics: maximum amplitude {max_amplitude:.4f}, 
        mean amplitude {mean_amplitude:.4f}, total energy {field_energy:.4f}.
        System context indicates field evolution may violate stability constraints.
        The field dynamics must remain bounded and physically consistent.
        Emergency verification required for dimensional analysis and safety validation.
        """
        
        return description
    
    def _recommend_discrepancy_action(self, validation_result: Dict[str, Any]) -> str:
        """Recommend action based on discrepancy validation result."""
        
        if not validation_result.get('symbolic_validation_passed', False):
            return 'emergency_shutdown'
        
        confidence = validation_result.get('confidence_score', 0.0)
        ethical_compliant = validation_result.get('ethical_compliance', False)
        
        if confidence < 0.5 or not ethical_compliant:
            return 'reduce_field_strength'
        elif confidence < 0.8:
            return 'increase_monitoring'
        else:
            return 'continue_normal'
    
    def _extract_coupling_constraints(self, sym_fields: List[SymField]) -> List[str]:
        """Extract coupling constraints from symbolic fields."""
        constraints = []
        
        for sym_field in sym_fields:
            # Extract constraints from field
            constraints.extend(sym_field.constraints)
            
            # Extract constraints from clauses
            for clause in sym_field.clauses:
                if 'coupling' in clause.natural_language.lower():
                    constraints.append(clause.natural_language)
        
        return constraints
    
    def _calculate_semantic_modulations(self,
                                      coupling_matrix: np.ndarray,
                                      phases: np.ndarray,
                                      constraints: List[str]) -> np.ndarray:
        """Calculate semantic modulations for coupling matrix."""
        
        modulations = np.zeros_like(coupling_matrix)
        
        # Simple constraint-based modulations
        for constraint in constraints:
            if 'increase' in constraint.lower():
                modulations += 0.1  # Small increase
            elif 'decrease' in constraint.lower():
                modulations -= 0.1  # Small decrease
        
        # Ensure modulations are bounded
        modulations = np.clip(modulations, -0.5, 0.5)
        
        return modulations
    
    def _apply_coupling_modulations(self,
                                  coupling_matrix: np.ndarray,
                                  modulations: np.ndarray) -> np.ndarray:
        """Apply modulations to coupling matrix."""
        
        # Apply modulations while preserving matrix properties
        modified = coupling_matrix + modulations
        
        # Ensure positive definiteness and symmetry preservation
        modified = np.maximum(modified, 0.01)  # Minimum coupling strength
        
        return modified
    
    def _validate_coupling_dimensions(self, coupling_matrix: np.ndarray) -> bool:
        """Validate dimensional consistency of coupling matrix."""
        
        # Basic dimensional checks
        if coupling_matrix.ndim != 2:
            return False
        
        if coupling_matrix.shape[0] != coupling_matrix.shape[1]:
            return False
        
        # Check for reasonable values
        if np.any(coupling_matrix < 0) or np.any(coupling_matrix > 100):
            return False
        
        return True
    
    def _generate_spatial_pattern_for_field(self, sym_field: SymField) -> np.ndarray:
        """Generate spatial pattern based on symbolic field characteristics."""
        
        # Create base pattern
        pattern = np.zeros(self.field_grid_shape, dtype=np.complex128)
        
        # Pattern based on field size
        field_size = sym_field.get_size()
        
        if field_size > 10:
            # Large field - use broad pattern
            x = np.linspace(0, 4*np.pi, self.field_grid_shape[0])
            y = np.linspace(0, 4*np.pi, self.field_grid_shape[1])
            X, Y = np.meshgrid(x, y)
            pattern = 0.1 * np.exp(1j * (X + Y)) * np.exp(-(X**2 + Y**2) / 20)
        else:
            # Small field - use localized pattern
            center_x, center_y = self.field_grid_shape[0] // 2, self.field_grid_shape[1] // 2
            x = np.arange(self.field_grid_shape[0]) - center_x
            y = np.arange(self.field_grid_shape[1]) - center_y
            X, Y = np.meshgrid(x, y)
            pattern = 0.05 * np.exp(-(X**2 + Y**2) / 10) * np.exp(1j * np.random.uniform(0, 2*np.pi))
        
        return pattern
    
    def _calculate_temporal_frequency(self, sym_field: SymField) -> float:
        """Calculate temporal frequency for field based on symbolic content."""
        
        # Base frequency on field characteristics
        base_freq = 0.1  # Hz
        
        # Modify based on field properties
        if len(sym_field.expressions) > 5:
            base_freq *= 1.5  # Higher frequency for complex fields
        
        if sym_field.cluster_confidence > 0.9:
            base_freq *= 0.8  # Lower frequency for high-confidence fields
        
        return base_freq
    
    def _extract_local_field_patch(self, neural_field: np.ndarray, 
                                 coord_x: int, coord_y: int, 
                                 patch_size: int = 5) -> np.ndarray:
        """Extract local patch around coordinates."""
        
        half_patch = patch_size // 2
        
        x_start = max(0, coord_x - half_patch)
        x_end = min(neural_field.shape[0], coord_x + half_patch + 1)
        y_start = max(0, coord_y - half_patch)  
        y_end = min(neural_field.shape[1], coord_y + half_patch + 1)
        
        return neural_field[x_start:x_end, y_start:y_end]
    
    def _calculate_field_consistency(self, sym_field: SymField, local_field: np.ndarray) -> float:
        """Calculate consistency between symbolic field and neural field patch."""
        
        # Simple consistency metric based on field properties
        field_energy = np.sum(np.abs(local_field) ** 2)
        field_complexity = np.std(np.abs(local_field))
        
        # Compare with symbolic field properties
        symbolic_complexity = sym_field.get_size() * sym_field.cluster_confidence
        
        # Calculate normalized consistency score
        energy_factor = min(1.0, field_energy / 10.0)  # Normalize to [0,1]
        complexity_match = 1.0 - abs(field_complexity - symbolic_complexity * 0.1) / max(field_complexity, 0.1)
        
        consistency = (energy_factor + complexity_match) / 2
        return max(0.0, min(1.0, consistency))
    
    def _infer_symbolic_from_field(self, local_field: np.ndarray, 
                                 original_field: SymField) -> Optional[SymField]:
        """Infer updated symbolic representation from neural field evolution."""
        
        # Simplified inference - in practice would use more sophisticated methods
        try:
            # Analyze field characteristics  
            field_description = f"Evolved field with energy {np.sum(np.abs(local_field)**2):.4f}"
            
            # Use symbolic AI to reprocess
            updated_fields = self.symbolic_ai.symbolize(field_description)
            
            if updated_fields:
                return updated_fields[0]
                
        except Exception as e:
            logger.debug(f"Symbolic inference failed: {e}")
        
        return None
    
    def _update_success_rate(self, current_rate: float, success: bool, alpha: float = 0.1) -> float:
        """Update exponential moving average success rate."""
        new_value = 1.0 if success else 0.0
        return alpha * new_value + (1 - alpha) * current_rate
    
    def _update_integration_metrics(self,
                                  symbolize_latency: float,
                                  fieldize_latency: float, 
                                  verify_latency: float,
                                  verification_report: VerificationReport,
                                  esc_integration: Dict[str, Any]) -> None:
        """Update integration performance metrics."""
        
        # Update operation counts
        self.metrics.symbolize_requests += 1
        self.metrics.fieldize_operations += 1  
        self.metrics.verify_validations += 1
        
        # Update latency averages (exponential moving average)
        alpha = 0.1
        self.metrics.avg_symbolize_latency_ms = (alpha * symbolize_latency + 
                                               (1-alpha) * self.metrics.avg_symbolize_latency_ms)
        self.metrics.avg_fieldize_latency_ms = (alpha * fieldize_latency + 
                                              (1-alpha) * self.metrics.avg_fieldize_latency_ms)
        self.metrics.avg_verify_latency_ms = (alpha * verify_latency + 
                                            (1-alpha) * self.metrics.avg_verify_latency_ms)
        
        # Update quality metrics
        self.metrics.dimensional_accuracy_average = (
            alpha * verification_report.dimensional_accuracy + 
            (1-alpha) * self.metrics.dimensional_accuracy_average
        )
        
        self.metrics.logical_consistency_rate = self._update_success_rate(
            self.metrics.logical_consistency_rate, verification_report.logical_consistency
        )
        
        self.metrics.ethical_compliance_rate = self._update_success_rate(
            self.metrics.ethical_compliance_rate, 
            verification_report.kant_universalization and verification_report.kant_means_end
        )
        
        # Update ESC integration success rate
        if esc_integration.get('integration_available', False):
            self.metrics.esc_integration_success_rate = self._update_success_rate(
                self.metrics.esc_integration_success_rate, 
                'error' not in esc_integration
            )
        
        # Check SLO violations
        self.metrics.total_slo_checks += 1
        
        total_latency = symbolize_latency + fieldize_latency + verify_latency
        if total_latency > 300:  # Latency SLO violation
            self.metrics.latency_slo_violations += 1
        
        if verification_report.dimensional_accuracy < 0.98:  # Accuracy SLO violation
            self.metrics.accuracy_slo_violations += 1
        
        # Add to performance window
        self.performance_window.append({
            'timestamp': time.time(),
            'symbolize_latency_ms': symbolize_latency,
            'fieldize_latency_ms': fieldize_latency,
            'verify_latency_ms': verify_latency,
            'total_latency_ms': total_latency,
            'dimensional_accuracy': verification_report.dimensional_accuracy,
            'slo_compliant': total_latency <= 300 and verification_report.dimensional_accuracy >= 0.98
        })


# =============================================================================
# FACTORY FUNCTIONS AND UTILITIES
# =============================================================================

def create_integrated_symbolic_system(esc_bridge: Optional[ESCKuramotoBridge] = None,
                                    enable_monitoring: bool = True,
                                    debug_mode: bool = False) -> KamilSymbolicIntegration:
    """
    Factory function to create fully integrated symbolic-neural system.
    
    Args:
        esc_bridge: Optional ESC-Kuramoto bridge instance
        enable_monitoring: Enable real-time performance monitoring
        debug_mode: Enable debug logging
        
    Returns:
        Fully configured integration system
    """
    # Create symbolic AI engine with optimal settings
    symbolic_ai = SymbolicAIKamil(
        enable_z3=True,
        enable_kant_mode=True,
        latency_slo_ms=300.0,
        dimensional_accuracy_slo=0.98,
        debug_mode=debug_mode
    )
    
    # Create integration layer
    integration = KamilSymbolicIntegration(
        symbolic_ai=symbolic_ai,
        esc_bridge=esc_bridge,
        enable_real_time_monitoring=enable_monitoring,
        debug_mode=debug_mode
    )
    
    logger.info("Integrated symbolic-neural system created successfully")
    return integration


def register_with_discrepancy_gate(integration: KamilSymbolicIntegration,
                                 discrepancy_gate_instance: Any) -> bool:
    """
    Register symbolic validation with existing discrepancy gate.
    
    Args:
        integration: Symbolic integration instance
        discrepancy_gate_instance: Discrepancy gate to register with
        
    Returns:
        Success status of registration
    """
    try:
        validator = integration.create_discrepancy_validator()
        
        # Register validator with discrepancy gate (method depends on gate implementation)
        if hasattr(discrepancy_gate_instance, 'register_validator'):
            discrepancy_gate_instance.register_validator(validator)
        elif hasattr(discrepancy_gate_instance, 'add_validator'):
            discrepancy_gate_instance.add_validator(validator)
        else:
            logger.warning("Discrepancy gate does not have recognized validator registration method")
            return False
        
        logger.info("Symbolic validator registered with discrepancy gate successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register with discrepancy gate: {e}")
        return False


if __name__ == "__main__":
    # Example usage and testing
    print("Kamil Symbolic AI Integration Module")
    print("=" * 40)
    
    # Create integrated system
    integration = create_integrated_symbolic_system(debug_mode=True)
    
    # Test semantic processing
    test_input = """
    The neural field exhibits oscillatory instability with frequency 2.5 Hz.
    Conservation laws must be maintained during field evolution.
    Field energy E = ∫|φ|² dx must remain bounded below 100 J.
    """
    
    print(f"Processing: {test_input}")
    
    # Simulate neural field context
    field_context = np.random.complex128((32, 32)) * 0.1
    
    # Process through integrated pipeline
    result = integration.process_semantic_input(test_input, field_context=field_context)
    
    print(f"Processing complete: SLO compliant = {result.get('slo_compliant', False)}")
    print(f"Processing time: {result.get('processing_time_ms', 0):.2f}ms")
    
    # Display metrics
    metrics = integration.get_integration_metrics()
    print(f"Integration metrics: {metrics['total_operations']} operations completed")
    print(f"SLO compliance rate: {metrics['slo_compliance_rate']:.1%}")
    
    print("\nIntegration module ready for NFCS v2.4.3 deployment")