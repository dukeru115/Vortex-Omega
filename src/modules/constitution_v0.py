"""
Constitution v0 for NFCS - Core Decision Making
==============================================

High-level decision-making system based on risk analysis and generation
of control intentions for all NFCS components.

Key Capabilities:
- Analyze RiskMetrics and generate ACCEPT/REJECT/EMERGENCY decisions
- Generate control_intent for CGL, Kuramoto, ESC 
- Safety policies and constraints
- Integration with resonance bus for event reception and decision publication
- Adaptive control strategies based on system state
- Constitutional principles and constraints for autonomous modules
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np

from ..modules.risk_monitor import RiskLevel, RiskAssessment
from ..orchestrator.resonance_bus import (
    get_global_bus, TopicType, EventPriority, BusEvent,
    RiskMetricsPayload, ControlIntentPayload
)


class DecisionType(Enum):
    """Types of constitutional decisions"""
    ACCEPT = "ACCEPT"        # Accept operation, standard control
    REJECT = "REJECT"        # Reject operation, enhance monitoring  
    EMERGENCY = "EMERGENCY"  # Emergency mode, activate protection protocols
    MONITOR = "MONITOR"      # Enhanced observation without intervention


class ControlStrategy(Enum):
    """System control strategies"""
    PERMISSIVE = "PERMISSIVE"      # Soft control, maximum freedom
    STANDARD = "STANDARD"          # Standard control
    RESTRICTIVE = "RESTRICTIVE"    # Strict control, restrictions
    EMERGENCY = "EMERGENCY"        # Emergency control, maximum restrictions


@dataclass
class PolicyConstraints:
    """Constitutional constraints and policies"""
    
    # Control signal constraints
    u_field_max_amplitude: float = 1.0      # Maximum amplitude of u_field
    u_modules_max_amplitude: float = 0.5    # Maximum amplitude of u_modules
    
    # Kuramoto coupling constraints
    kuramoto_coupling_max: float = 2.0      # Maximum coupling
    kuramoto_coupling_min: float = 0.1      # Minimum coupling  
    
    # ESC constraints
    esc_normalization_strict: bool = True   # Strict ESC normalization
    esc_max_order_parameter: float = 0.9    # Maximum order parameter
    
    # Freedom module constraints
    freedom_max_noise_amplitude: float = 0.2  # Maximum freedom noise amplitude
    freedom_min_coherence_threshold: float = 0.3  # Minimum coherence for freedom
    
    # Temporal constraints
    emergency_mode_max_duration: float = 300.0    # Maximum emergency duration (sec)
    recovery_assessment_interval: float = 30.0    # Recovery assessment interval
    
    # Autonomy thresholds
    autonomous_operation_min_coherence: float = 0.5
    autonomous_operation_max_risk: float = 0.3


@dataclass
class ControlIntent:
    """Control intention for system components"""
    
    # Primary decision
    decision: DecisionType = DecisionType.ACCEPT
    strategy: ControlStrategy = ControlStrategy.STANDARD
    
    # CGL solver constraints
    u_field_limits: Dict[str, float] = field(default_factory=lambda: {
        'max_amplitude': 1.0,
        'spatial_smoothing': 0.0,
        'temporal_damping': 0.0
    })
    
    # Kuramoto masks and constraints
    kuramoto_masks: Dict[str, Any] = field(default_factory=lambda: {
        'coupling_multipliers': None,   # Coupling multipliers [N x N]
        'frequency_adjustments': None,  # Frequency corrections [N]
        'connection_masks': None        # Connection masks [N x N, bool]
    })
    
    # ESC settings 
    esc_configuration: Dict[str, Any] = field(default_factory=lambda: {
        'normalization_mode': 'standard',  # standard, strict, adaptive
        'order_parameter_limit': 0.9,
        'resonance_damping': 0.0,
        'semantic_filtering': False
    })
    
    # Freedom window for Freedom module  
    freedom_window: Dict[str, float] = field(default_factory=lambda: {
        'noise_amplitude': 0.1,
        'coherence_threshold': 0.3,
        'exploration_rate': 0.05,
        'creativity_boost': 0.0
    })
    
    # Emergency constraints
    emergency_constraints: Dict[str, Any] = field(default_factory=lambda: {
        'boundary_permeability': 1.0,      # Boundary permeability multiplier
        'cross_talk_suppression': 0.0,     # Cross-talk suppression [0-1]
        'coherence_enforcement': False,     # Forced coherence
        'risk_escalation_rate': 1.0        # Risk escalation rate
    })
    
    # Decision metadata
    reasoning: List[str] = field(default_factory=list)    # Decision justification
    confidence: float = 1.0                               # Decision confidence [0-1]
    validity_duration: float = 60.0                       # Decision validity duration (sec)
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if decision has expired"""
        return time.time() - self.created_at > self.validity_duration
    
    def get_summary(self) -> str:
        """Get brief description of intention"""
        return (f"{self.decision.value}({self.strategy.value}) "
               f"confidence={self.confidence:.2f}")


@dataclass  
class ConstitutionalState:
    """State of constitutional system"""
    current_risk_level: RiskLevel = RiskLevel.NORMAL
    current_strategy: ControlStrategy = ControlStrategy.STANDARD
    emergency_mode_start: Optional[float] = None
    last_decision_time: float = field(default_factory=time.time)
    
    # Counters and statistics
    total_decisions: int = 0
    accept_decisions: int = 0  
    reject_decisions: int = 0
    emergency_decisions: int = 0
    
    # Adaptive parameters
    risk_sensitivity: float = 1.0      # Risk sensitivity multiplier
    recovery_progress: float = 0.0     # Recovery progress [0-1]
    
    def get_emergency_duration(self) -> float:
        """Get duration in emergency mode"""
        if self.emergency_mode_start is None:
            return 0.0
        return time.time() - self.emergency_mode_start


class ConstitutionV0:
    """
    Constitution v0 - Decision Making System for NFCS
    
    Analyzes risk states and generates control intentions
    for all system components according to constitutional principles.
    """
    
    def __init__(self, 
                 constraints: Optional[PolicyConstraints] = None,
                 enable_auto_subscription: bool = True,
                 decision_interval: float = 1.0):
        
        self.constraints = constraints or PolicyConstraints()
        self.enable_auto_subscription = enable_auto_subscription
        self.decision_interval = decision_interval
        
        # Constitutional state
        self.state = ConstitutionalState()
        
        # Decision history for analysis
        self.decision_history: List[ControlIntent] = []
        self.max_history_size = 1000
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Resonance bus
        self.bus = get_global_bus()
        
        # Logger
        self.logger = logging.getLogger(f"{__name__}.ConstitutionV0")
        
        # Risk events subscription
        if self.enable_auto_subscription:
            self._subscribe_to_risk_events()
        
        # Periodic decision making
        self._decision_task: Optional[asyncio.Task] = None
        self._running = False
        
        self.logger.info("Constitution v0 initialized")
    
    def _subscribe_to_risk_events(self):
        """Subscribe to risk events from ResonanceBus"""
        try:
            self.bus.subscribe(
                handler_id="constitution_risk_handler",
                callback=self._handle_risk_event,
                topic_filter={TopicType.METRICS_RISK},
                priority_filter={EventPriority.HIGH, EventPriority.CRITICAL, EventPriority.EMERGENCY}
            )
            
            self.logger.info("Risk events subscription activated")
            
        except Exception as e:
            self.logger.error(f"Risk events subscription error: {e}")
    
    def _handle_risk_event(self, event: BusEvent):
        """Risk event handler"""
        try:
            if isinstance(event.payload, RiskMetricsPayload):
                # Quick decision making on critical events  
                if event.priority in [EventPriority.CRITICAL, EventPriority.EMERGENCY]:
                    risk_level = RiskLevel(event.payload.risk_level)
                    intent = self._make_immediate_decision(event.payload, risk_level)
                    
                    if intent:
                        self._publish_control_intent(intent)
                        self.logger.warning(
                            f"Emergency decision: {intent.get_summary()} "
                            f"on {event.payload.risk_level}"
                        )
        
        except Exception as e:
            self.logger.error(f"Risk event processing error: {e}")
    
    def _make_immediate_decision(self, 
                               risk_payload: RiskMetricsPayload, 
                               risk_level: RiskLevel) -> Optional[ControlIntent]:
        """Make immediate decision on critical event"""
        
        with self._lock:
            # Risk state update
            self.state.current_risk_level = risk_level
            
            # Quick situation assessment
            if risk_level == RiskLevel.EMERGENCY:
                return self._create_emergency_intent(risk_payload, "Immediate emergency response")
            
            elif risk_level == RiskLevel.CRITICAL:
                return self._create_restrictive_intent(risk_payload, "Critical risk mitigation")
            
            else:
                return None  # No emergency decisions for WARNING and NORMAL
    
    async def start_decision_loop(self):
        """Start periodic decision-making cycle"""
        if self._running:
            return
        
        self._running = True
        self._decision_task = asyncio.create_task(self._decision_loop())
        self.logger.info("Decision-making cycle started")
    
    async def stop_decision_loop(self):
        """Stop decision-making cycle"""
        if not self._running:
            return
        
        self._running = False
        if self._decision_task:
            self._decision_task.cancel()
            try:
                await self._decision_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Decision-making cycle stopped")
    
    async def _decision_loop(self):
        """Main periodic decision-making loop"""
        while self._running:
            try:
                # Comprehensive system state assessment
                intent = self._make_comprehensive_decision()
                
                if intent:
                    self._publish_control_intent(intent)
                    
                    # Strategy change logging
                    if intent.strategy != self.state.current_strategy:
                        self.logger.info(
                            f"Strategy changed: {self.state.current_strategy.value} → "
                            f"{intent.strategy.value}. Reason: {', '.join(intent.reasoning)}"
                        )
                        self.state.current_strategy = intent.strategy
                
                await asyncio.sleep(self.decision_interval)
                
            except Exception as e:
                self.logger.error(f"Error in decision-making cycle: {e}")
                await asyncio.sleep(1.0)  # Pause on error
    
    def _make_comprehensive_decision(self) -> Optional[ControlIntent]:
        """Make comprehensive decision based on system analysis"""
        
        with self._lock:
            try:
                # Current state analysis
                system_analysis = self._analyze_system_state()
                
                # Strategy determination
                strategy = self._determine_strategy(system_analysis)
                
                # Intent creation
                if strategy == ControlStrategy.EMERGENCY:
                    intent = self._create_emergency_intent(None, "Comprehensive emergency assessment")
                elif strategy == ControlStrategy.RESTRICTIVE:
                    intent = self._create_restrictive_intent(None, "Risk mitigation required")
                elif strategy == ControlStrategy.PERMISSIVE:
                    intent = self._create_permissive_intent("Low risk, enhanced freedom")
                else:
                    intent = self._create_standard_intent("Normal operation")
                
                # Statistics update
                self._update_decision_statistics(intent)
                
                return intent
                
            except Exception as e:
                self.logger.error(f"Comprehensive decision-making error: {e}")
                # Safe fallback - restrictive strategy
                return self._create_restrictive_intent(None, f"Decision error: {str(e)}")
    
    def _analyze_system_state(self) -> Dict[str, Any]:
        """Analysis of current system state"""
        
        analysis = {
            'current_risk_level': self.state.current_risk_level,
            'emergency_duration': self.state.get_emergency_duration(),
            'recent_decisions': len([d for d in self.decision_history[-10:] 
                                   if d.decision == DecisionType.EMERGENCY]),
            'decision_frequency': self._calculate_decision_frequency(),
            'system_stability': self._assess_system_stability(),
            'recovery_indicators': self._assess_recovery_progress()
        }
        
        return analysis
    
    def _determine_strategy(self, analysis: Dict[str, Any]) -> ControlStrategy:
        """Determine optimal control strategy"""
        
        # Emergency strategy
        if analysis['current_risk_level'] == RiskLevel.EMERGENCY:
            return ControlStrategy.EMERGENCY
        
        # Emergency mode timeout check
        if (self.state.emergency_mode_start and 
            analysis['emergency_duration'] > self.constraints.emergency_mode_max_duration):
            self.logger.warning(f"Emergency mode timeout, forcing recovery")
            return ControlStrategy.RESTRICTIVE
        
        # Restrictive strategy
        if (analysis['current_risk_level'] == RiskLevel.CRITICAL or
            analysis['recent_decisions'] >= 3 or  # Many emergency decisions
            analysis['system_stability'] < 0.3):
            return ControlStrategy.RESTRICTIVE
        
        # Permissive strategy  
        if (analysis['current_risk_level'] == RiskLevel.NORMAL and
            analysis['system_stability'] > 0.7 and
            analysis['recovery_indicators'] > 0.8):
            return ControlStrategy.PERMISSIVE
        
        # Default standard strategy
        return ControlStrategy.STANDARD
    
    def _calculate_decision_frequency(self) -> float:
        """Calculate decision frequency (decisions per minute)"""
        if len(self.decision_history) < 2:
            return 0.0
        
        recent_decisions = [d for d in self.decision_history[-20:] 
                          if time.time() - d.created_at < 60.0]  # In the last minute
        
        return len(recent_decisions)
    
    def _assess_system_stability(self) -> float:
        """Evaluate system stability [0-1]"""
        if len(self.decision_history) < 5:
            return 0.5  # Insufficient data
        
        recent_decisions = self.decision_history[-10:]
        
        # Count strategy changes
        strategy_changes = 0
        for i in range(1, len(recent_decisions)):
            if recent_decisions[i].strategy != recent_decisions[i-1].strategy:
                strategy_changes += 1
        
        # Count emergency decisions  
        emergency_count = sum(1 for d in recent_decisions 
                            if d.decision == DecisionType.EMERGENCY)
        
        # Stability assessment
        stability = 1.0 - (strategy_changes / len(recent_decisions)) - (emergency_count * 0.2)
        
        return max(0.0, min(1.0, stability))
    
    def _assess_recovery_progress(self) -> float:
        """Evaluate system recovery progress [0-1]"""
        
        # If there was no emergency mode
        if self.state.emergency_mode_start is None:
            return 1.0
        
        emergency_duration = self.state.get_emergency_duration()
        
        # Progress based on recovery time
        if emergency_duration < 60.0:  # Less than a minute
            time_progress = 0.2
        elif emergency_duration < 300.0:  # Less than 5 minutes
            time_progress = 0.5
        else:  # Long recovery
            time_progress = 0.8
        
        # Progress based on risk reduction
        if self.state.current_risk_level == RiskLevel.NORMAL:
            risk_progress = 1.0
        elif self.state.current_risk_level == RiskLevel.WARNING:
            risk_progress = 0.7
        elif self.state.current_risk_level == RiskLevel.CRITICAL:
            risk_progress = 0.3
        else:
            risk_progress = 0.0
        
        return (time_progress + risk_progress) / 2.0
    
    def _create_emergency_intent(self, 
                               risk_payload: Optional[RiskMetricsPayload],
                               reason: str) -> ControlIntent:
        """Create emergency intention"""
        
        intent = ControlIntent(
            decision=DecisionType.EMERGENCY,
            strategy=ControlStrategy.EMERGENCY,
            confidence=0.95
        )
        
        # Strict control constraints
        intent.u_field_limits = {
            'max_amplitude': self.constraints.u_field_max_amplitude * 0.5,
            'spatial_smoothing': 0.3,
            'temporal_damping': 0.2
        }
        
        # Kuramoto constraints - strengthening intra-cluster connections
        intent.kuramoto_masks = {
            'coupling_multipliers': self._create_emergency_coupling_matrix(),
            'frequency_adjustments': None,
            'connection_masks': None
        }
        
        # Strict ESC settings
        intent.esc_configuration = {
            'normalization_mode': 'strict',
            'order_parameter_limit': 0.7,
            'resonance_damping': 0.4,
            'semantic_filtering': True
        }
        
        # Minimal freedom
        intent.freedom_window = {
            'noise_amplitude': 0.02,
            'coherence_threshold': 0.6,
            'exploration_rate': 0.01,
            'creativity_boost': -0.1
        }
        
        # Emergency constraints
        intent.emergency_constraints = {
            'boundary_permeability': 0.1,    # Strong permeability limitation
            'cross_talk_suppression': 0.8,   # Cross-talk suppression
            'coherence_enforcement': True,   # Forced coherence
            'risk_escalation_rate': 0.5     # Slowed escalation
        }
        
        intent.reasoning = [reason, "Emergency protocols activated"]
        intent.validity_duration = 30.0  # Short validity period
        
        # State update
        if self.state.emergency_mode_start is None:
            self.state.emergency_mode_start = time.time()
        
        return intent
    
    def _create_restrictive_intent(self,
                                 risk_payload: Optional[RiskMetricsPayload],
                                 reason: str) -> ControlIntent:
        """Create restrictive intention"""
        
        intent = ControlIntent(
            decision=DecisionType.REJECT,
            strategy=ControlStrategy.RESTRICTIVE,
            confidence=0.8
        )
        
        # Moderate control constraints
        intent.u_field_limits = {
            'max_amplitude': self.constraints.u_field_max_amplitude * 0.7,
            'spatial_smoothing': 0.1,
            'temporal_damping': 0.1
        }
        
        # Kuramoto constraints
        intent.kuramoto_masks = {
            'coupling_multipliers': self._create_restrictive_coupling_matrix(),
            'frequency_adjustments': None,
            'connection_masks': None
        }
        
        # Enhanced ESC settings
        intent.esc_configuration = {
            'normalization_mode': 'adaptive',
            'order_parameter_limit': 0.8,
            'resonance_damping': 0.2,
            'semantic_filtering': True
        }
        
        # Limited freedom
        intent.freedom_window = {
            'noise_amplitude': 0.05,
            'coherence_threshold': 0.4,
            'exploration_rate': 0.02,
            'creativity_boost': 0.0
        }
        
        # Moderate constraints
        intent.emergency_constraints = {
            'boundary_permeability': 0.5,
            'cross_talk_suppression': 0.4,
            'coherence_enforcement': False,
            'risk_escalation_rate': 0.8
        }
        
        intent.reasoning = [reason, "Risk mitigation measures"]
        
        return intent
    
    def _create_standard_intent(self, reason: str) -> ControlIntent:
        """Create standard intention"""
        
        intent = ControlIntent(
            decision=DecisionType.ACCEPT,
            strategy=ControlStrategy.STANDARD,
            confidence=0.9
        )
        
        # Standard constraints
        intent.u_field_limits = {
            'max_amplitude': self.constraints.u_field_max_amplitude,
            'spatial_smoothing': 0.0,
            'temporal_damping': 0.0
        }
        
        # Standard ESC configuration
        intent.esc_configuration = {
            'normalization_mode': 'standard',
            'order_parameter_limit': self.constraints.esc_max_order_parameter,
            'resonance_damping': 0.0,
            'semantic_filtering': False
        }
        
        # Standard freedom
        intent.freedom_window = {
            'noise_amplitude': self.constraints.freedom_max_noise_amplitude,
            'coherence_threshold': self.constraints.freedom_min_coherence_threshold,
            'exploration_rate': 0.05,
            'creativity_boost': 0.0
        }
        
        intent.reasoning = [reason, "Standard operation"]
        
        # Reset emergency mode if it was active
        if self.state.emergency_mode_start is not None:
            self.logger.info(f"Exiting emergency mode after {self.state.get_emergency_duration():.1f}s")
            self.state.emergency_mode_start = None
        
        return intent
    
    def _create_permissive_intent(self, reason: str) -> ControlIntent:
        """Create permissive intention"""
        
        intent = ControlIntent(
            decision=DecisionType.ACCEPT,
            strategy=ControlStrategy.PERMISSIVE,
            confidence=0.85
        )
        
        # Increased limits
        intent.u_field_limits = {
            'max_amplitude': self.constraints.u_field_max_amplitude * 1.2,
            'spatial_smoothing': 0.0,
            'temporal_damping': 0.0
        }
        
        # Enhanced ESC configuration
        intent.esc_configuration = {
            'normalization_mode': 'adaptive',
            'order_parameter_limit': 0.95,
            'resonance_damping': 0.0,
            'semantic_filtering': False
        }
        
        # Enhanced freedom
        intent.freedom_window = {
            'noise_amplitude': self.constraints.freedom_max_noise_amplitude * 1.5,
            'coherence_threshold': 0.2,
            'exploration_rate': 0.1,
            'creativity_boost': 0.1
        }
        
        intent.reasoning = [reason, "Enhanced autonomy granted"]
        
        return intent
    
    def _create_emergency_coupling_matrix(self) -> Optional[np.ndarray]:
        """Create connectivity matrix for emergency mode"""
        # Example: strengthen intra-cluster connections, weaken inter-cluster ones
        # In real implementation this should depend on current module configuration
        n_modules = 4  # constitution, boundary, memory, meta_reflection
        
        matrix = np.eye(n_modules) * 2.0  # Self-connection enhancement
        
        # Strengthen connections within cognitive cluster
        cognitive_cluster = [0, 1, 2, 3]  # All modules in one cluster for now
        for i in cognitive_cluster:
            for j in cognitive_cluster:
                if i != j:
                    matrix[i, j] = 1.5  # Enhanced connections within cluster
        
        return matrix
    
    def _create_restrictive_coupling_matrix(self) -> Optional[np.ndarray]:
        """Create connectivity matrix for restrictive mode"""
        n_modules = 4
        
        matrix = np.eye(n_modules) * 1.2  # Small self-connection enhancement
        
        # Standard connections with slight weakening
        for i in range(n_modules):
            for j in range(n_modules):
                if i != j:
                    matrix[i, j] = 0.8
        
        return matrix
    
    def _update_decision_statistics(self, intent: ControlIntent):
        """Update decision statistics"""
        
        self.state.total_decisions += 1
        self.state.last_decision_time = time.time()
        
        if intent.decision == DecisionType.ACCEPT:
            self.state.accept_decisions += 1
        elif intent.decision == DecisionType.REJECT:
            self.state.reject_decisions += 1
        elif intent.decision == DecisionType.EMERGENCY:
            self.state.emergency_decisions += 1
        
        # Add to history
        self.decision_history.append(intent)
        
        # Limit history size
        if len(self.decision_history) > self.max_history_size:
            self.decision_history = self.decision_history[-self.max_history_size//2:]
    
    def _publish_control_intent(self, intent: ControlIntent):
        """Publish intention to resonance bus"""
        
        try:
            payload = ControlIntentPayload(
                source_module="constitution_v0",
                decision_type=intent.decision.value,
                u_field_limits=intent.u_field_limits,
                kuramoto_masks=intent.kuramoto_masks,
                esc_normalization_mode=intent.esc_configuration['normalization_mode'],
                freedom_window=intent.freedom_window['noise_amplitude'],
                emergency_constraints=intent.emergency_constraints
            )
            
            # Priority determination
            priority = EventPriority.NORMAL
            if intent.decision == DecisionType.EMERGENCY:
                priority = EventPriority.EMERGENCY
            elif intent.decision == DecisionType.REJECT:
                priority = EventPriority.HIGH
            
            success = self.bus.publish(TopicType.CONTROL_INTENT, payload, priority)
            
            if success:
                self.logger.debug(f"Published intention: {intent.get_summary()}")
            else:
                self.logger.error("Intention publication error")
        
        except Exception as e:
            self.logger.error(f"Intention publication error: {e}")
    
    def manual_decision(self, 
                       risk_level: RiskLevel,
                       additional_context: Optional[Dict[str, Any]] = None) -> ControlIntent:
        """
        Make manual decision based on specified risk level
        
        Args:
            risk_level: Risk level for decision making
            additional_context: Additional context for decision
            
        Returns:
            ControlIntent: Control intention
        """
        
        with self._lock:
            self.state.current_risk_level = risk_level
            
            context_info = additional_context or {}
            reason = f"Manual decision for {risk_level.value}"
            
            if risk_level == RiskLevel.EMERGENCY:
                intent = self._create_emergency_intent(None, reason)
            elif risk_level == RiskLevel.CRITICAL:
                intent = self._create_restrictive_intent(None, reason)
            elif risk_level == RiskLevel.WARNING:
                intent = self._create_standard_intent(reason)
            else:
                intent = self._create_permissive_intent(reason)
            
            # Add context to reasoning
            if additional_context:
                intent.reasoning.extend([f"{k}={v}" for k, v in context_info.items()])
            
            self._update_decision_statistics(intent)
            self._publish_control_intent(intent)
            
            self.logger.info(f"Manual decision: {intent.get_summary()}")
            
            return intent
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current constitution status"""
        
        with self._lock:
            status = {
                'current_risk_level': self.state.current_risk_level.value,
                'current_strategy': self.state.current_strategy.value,
                'emergency_mode': self.state.emergency_mode_start is not None,
                'emergency_duration': self.state.get_emergency_duration(),
                'statistics': {
                    'total_decisions': self.state.total_decisions,
                    'accept_rate': (self.state.accept_decisions / max(1, self.state.total_decisions)),
                    'reject_rate': (self.state.reject_decisions / max(1, self.state.total_decisions)), 
                    'emergency_rate': (self.state.emergency_decisions / max(1, self.state.total_decisions)),
                    'decision_frequency': self._calculate_decision_frequency(),
                    'system_stability': self._assess_system_stability(),
                    'recovery_progress': self._assess_recovery_progress()
                },
                'constraints': {
                    'u_field_max': self.constraints.u_field_max_amplitude,
                    'emergency_max_duration': self.constraints.emergency_mode_max_duration,
                    'autonomous_min_coherence': self.constraints.autonomous_operation_min_coherence
                }
            }
            
            if self.decision_history:
                last_decision = self.decision_history[-1]
                status['last_decision'] = {
                    'type': last_decision.decision.value,
                    'strategy': last_decision.strategy.value,
                    'confidence': last_decision.confidence,
                    'reasoning': last_decision.reasoning,
                    'age_seconds': time.time() - last_decision.created_at
                }
            
            return status
    
    def update_constraints(self, new_constraints: PolicyConstraints):
        """Update constitutional constraints"""
        
        with self._lock:
            old_constraints = self.constraints
            self.constraints = new_constraints
            
            self.logger.info("Constitutional constraints updated")
    
    def get_decision_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get decision history"""
        
        with self._lock:
            recent_decisions = self.decision_history[-limit:]
            
            return [
                {
                    'decision': d.decision.value,
                    'strategy': d.strategy.value,
                    'confidence': d.confidence,
                    'reasoning': d.reasoning,
                    'created_at': d.created_at,
                    'expired': d.is_expired()
                }
                for d in recent_decisions
            ]
    
    def force_recovery_mode(self):
        """Force recovery mode activation"""
        
        with self._lock:
            if self.state.emergency_mode_start:
                recovery_intent = self._create_restrictive_intent(
                    None, "Forced recovery from emergency mode"
                )
                
                self.state.emergency_mode_start = None
                self.state.current_risk_level = RiskLevel.WARNING
                self.state.current_strategy = ControlStrategy.RESTRICTIVE
                
                self._update_decision_statistics(recovery_intent)
                self._publish_control_intent(recovery_intent)
                
                self.logger.warning("Forced recovery from emergency mode")
                
                return recovery_intent
    
    def __repr__(self) -> str:
        """String representation of constitution"""
        return (f"ConstitutionV0(risk={self.state.current_risk_level.value}, "
               f"strategy={self.state.current_strategy.value}, "
               f"decisions={self.state.total_decisions})")


# Convenience functions for creating constitutions
def create_default_constitution(**kwargs) -> ConstitutionV0:
    """Create constitution with default settings"""
    return ConstitutionV0(**kwargs)


def create_strict_constitution() -> ConstitutionV0:
    """Create strict constitution with rigid constraints"""
    strict_constraints = PolicyConstraints(
        u_field_max_amplitude=0.5,
        u_modules_max_amplitude=0.3,
        kuramoto_coupling_max=1.5,
        esc_max_order_parameter=0.7,
        freedom_max_noise_amplitude=0.1,
        freedom_min_coherence_threshold=0.5,
        emergency_mode_max_duration=180.0,
        autonomous_operation_min_coherence=0.7,
        autonomous_operation_max_risk=0.2
    )
    
    return ConstitutionV0(constraints=strict_constraints)


def create_permissive_constitution() -> ConstitutionV0:
    """Create lenient constitution with expanded capabilities"""
    permissive_constraints = PolicyConstraints(
        u_field_max_amplitude=2.0,
        u_modules_max_amplitude=1.0,
        kuramoto_coupling_max=3.0,
        esc_max_order_parameter=0.95,
        freedom_max_noise_amplitude=0.5,
        freedom_min_coherence_threshold=0.2,
        emergency_mode_max_duration=600.0,
        autonomous_operation_min_coherence=0.3,
        autonomous_operation_max_risk=0.5
    )
    
    return ConstitutionV0(constraints=permissive_constraints)


if __name__ == "__main__":
    # Usage example
    import asyncio
    from ..orchestrator.resonance_bus import initialize_global_bus
    from ..modules.risk_monitor import RiskLevel
    
    async def demo_constitution():
        # Bus initialization
        await initialize_global_bus()
        
        # Constitution creation
        constitution = create_default_constitution()
        
        # Start decision-making cycle
        await constitution.start_decision_loop()
        
        # Test manual decisions
        for risk_level in [RiskLevel.WARNING, RiskLevel.CRITICAL, RiskLevel.EMERGENCY, RiskLevel.NORMAL]:
            intent = constitution.manual_decision(risk_level, {"test": True})
            print(f"Decision for {risk_level.value}: {intent.get_summary()}")
            await asyncio.sleep(1.0)
        
        # Get status
        status = constitution.get_current_status()
        print(f"\nConstitution status:")
        print(f"Strategy: {status['current_strategy']}")
        print(f"Statistics: {status['statistics']}")
        
        # Stop
        await constitution.stop_decision_loop()
    
    # Run demo
    if __name__ == "__main__":
        asyncio.run(demo_constitution())