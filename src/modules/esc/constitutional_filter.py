"""
Constitutional Content Filter for ESC Module 2.1

Advanced content filtering with constitutional policies:
- Real-time constitutional compliance checking
- Multi-layer content analysis and safety assessment
- Contextual policy application and adaptive filtering
- Emergency content intervention and safe replacements
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import re
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class FilterPolicy(Enum):
    """Content filtering policies."""
    STRICT = "strict"          # Maximum safety, minimal risk
    BALANCED = "balanced"      # Standard filtering
    PERMISSIVE = "permissive"  # Minimal filtering, creative freedom
    ADAPTIVE = "adaptive"      # Context-dependent filtering
    EMERGENCY = "emergency"    # Emergency lockdown mode


class ContentRiskLevel(Enum):
    """Content risk level classifications."""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MODERATE_RISK = "moderate_risk"
    HIGH_RISK = "high_risk"
    DANGEROUS = "dangerous"
    PROHIBITED = "prohibited"


@dataclass
class FilterConfig:
    """Configuration for constitutional content filter."""
    default_policy: FilterPolicy = FilterPolicy.BALANCED
    risk_threshold: float = 0.7
    emergency_threshold: float = 0.95
    context_sensitivity: float = 0.5
    adaptive_learning_rate: float = 0.01
    max_violation_rate: float = 0.1


class ConstitutionalContentFilter:
    """
    Constitutional Content Filter for ESC.
    
    Implements constitutional content filtering with adaptive policies,
    contextual analysis, and emergency intervention capabilities.
    """
    
    def __init__(self, config: FilterConfig):
        """
        Initialize constitutional content filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config
        self.current_policy = config.default_policy
        
        # Initialize content classification systems
        self._initialize_content_patterns()
        self._initialize_safety_policies()
        self._initialize_contextual_rules()
        
        # Filtering state
        self.filtering_history = deque(maxlen=1000)
        self.violation_history = deque(maxlen=500)
        self.intervention_history = deque(maxlen=200)
        
        # Adaptive learning state
        self.content_risk_scores = defaultdict(float)
        self.contextual_modifiers = defaultdict(float)
        self.policy_effectiveness = defaultdict(list)
        
        # Statistics
        self.stats = {
            'total_content_processed': 0,
            'content_filtered': 0,
            'constitutional_violations': 0,
            'emergency_interventions': 0,
            'policy_adaptations': 0
        }
        
        logger.info(f"Constitutional content filter initialized")
        logger.info(f"Default policy: {config.default_policy.value}")
    
    def _initialize_content_patterns(self):
        """Initialize content classification patterns."""
        # High-risk content patterns
        self.dangerous_patterns = {
            'violence': [
                r'\\b(kill|murder|assault|attack|violence|harm|hurt|damage|destroy)\\w*',
                r'\\b(weapon|gun|knife|bomb|explosive|poison)\\w*',
                r'\\b(blood|death|corpse|victim|torture)\\w*'
            ],
            'hate_speech': [
                r'\\b(hate|racist|sexist|bigot|discrimination|prejudice)\\w*',
                r'\\b(inferior|superior|subhuman|vermin)\\w*',
                r'\\b(genocide|ethnic.*cleansing|final.*solution)\\w*'
            ],
            'illegal_activity': [
                r'\\b(illegal|criminal|fraud|theft|piracy|hack|crack)\\w*',
                r'\\b(drug.*deal|money.*launder|tax.*evade)\\w*',
                r'\\b(smuggl|traffick|brib|corrupt)\\w*'
            ],
            'personal_info': [
                r'\\b(ssn|social.?security|\\d{3}-\\d{2}-\\d{4})\\b',
                r'\\b(password|credit.?card|\\d{4}.?\\d{4}.?\\d{4}.?\\d{4})\\b',
                r'\\b(private.?key|api.?key|secret.?token)\\w*'
            ],
            'medical_advice': [
                r'\\b(diagnos|prescrib|medic.*advice|treatment.*recommend)\\w*',
                r'\\b(suicide|self.?harm|overdose|poisoning)\\w*',
                r'\\b(dose|medication|symptom.*treat)\\w*'
            ]
        }
        
        # Moderate risk patterns
        self.moderate_patterns = {
            'controversial': [
                r'\\b(controvers|debat|disagre|oppos)\\w*',
                r'\\b(politic|ideolog|belief|opinion)\\w*'
            ],
            'sensitive_topics': [
                r'\\b(religion|faith|belief|spiritual)\\w*',
                r'\\b(race|ethnic|cultural|tradition)\\w*',
                r'\\b(gender|sexual|identity|orientation)\\w*'
            ]
        }
        
        # Constitutional positive patterns
        self.constitutional_patterns = {
            'safety': [
                r'\\b(safe|secure|protect|privacy|consent|permission)\\w*',
                r'\\b(ethical|moral|responsible|accountable)\\w*'
            ],
            'educational': [
                r'\\b(learn|educat|teach|inform|explain|understand)\\w*',
                r'\\b(knowledge|fact|evidence|research|study)\\w*'
            ],
            'helpful': [
                r'\\b(help|assist|support|guid|advise)\\w*',
                r'\\b(useful|beneficial|constructive|positive)\\w*'
            ]
        }
    
    def _initialize_safety_policies(self):
        """Initialize safety policy configurations."""
        self.policy_configs = {
            FilterPolicy.STRICT: {
                'dangerous_threshold': 0.1,
                'moderate_threshold': 0.3,
                'replacement_mode': 'remove',
                'allow_controversial': False
            },
            FilterPolicy.BALANCED: {
                'dangerous_threshold': 0.5,
                'moderate_threshold': 0.7,
                'replacement_mode': 'replace',
                'allow_controversial': True
            },
            FilterPolicy.PERMISSIVE: {
                'dangerous_threshold': 0.8,
                'moderate_threshold': 0.9,
                'replacement_mode': 'warn',
                'allow_controversial': True
            },
            FilterPolicy.EMERGENCY: {
                'dangerous_threshold': 0.01,
                'moderate_threshold': 0.01,
                'replacement_mode': 'remove',
                'allow_controversial': False
            }
        }
    
    def _initialize_contextual_rules(self):
        """Initialize contextual filtering rules."""
        self.contextual_rules = {
            'educational_context': {
                'violence_modifier': -0.3,  # Less strict in educational context
                'medical_modifier': -0.4,   # Allow medical information
                'political_modifier': -0.2   # Allow political discussion
            },
            'creative_context': {
                'violence_modifier': -0.2,
                'controversial_modifier': -0.3,
                'fictional_modifier': -0.4
            },
            'professional_context': {
                'personal_info_modifier': 0.3,  # More strict with personal info
                'confidential_modifier': 0.4
            },
            'child_safety_context': {
                'violence_modifier': 0.5,   # Much more strict
                'adult_content_modifier': 0.8,
                'educational_boost': -0.2
            }
        }
    
    def filter_content(self, 
                      content: str, 
                      context: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """Filter content according to constitutional policies.
        
        Args:
            content: Input content to filter
            context: Optional context information
            
        Returns:
            Tuple of (filtered_content, filtering_info)
        """
        self.stats['total_content_processed'] += 1
        
        # Analyze content risk
        risk_analysis = self._analyze_content_risk(content, context)
        
        # Apply current policy
        filtering_result = self._apply_filtering_policy(content, risk_analysis, context)
        
        # Check for constitutional violations
        violation_check = self._check_constitutional_violations(risk_analysis)
        
        # Update adaptive learning
        if self.current_policy == FilterPolicy.ADAPTIVE:
            self._update_adaptive_learning(content, risk_analysis, filtering_result)
        
        # Emergency intervention check
        emergency_check = self._check_emergency_conditions(risk_analysis)
        
        if emergency_check['emergency_triggered']:
            self.current_policy = FilterPolicy.EMERGENCY
            filtering_result = self._apply_emergency_filtering(content)
            self.stats['emergency_interventions'] += 1
        
        # Compile filtering information
        filtering_info = {
            'risk_analysis': risk_analysis,
            'policy_applied': self.current_policy.value,
            'content_modified': filtering_result['content'] != content,
            'constitutional_violations': violation_check,
            'emergency_intervention': emergency_check,
            'adaptive_updates': self.current_policy == FilterPolicy.ADAPTIVE
        }
        
        # Store in history
        self.filtering_history.append({
            'content_length': len(content),
            'risk_level': risk_analysis['overall_risk_level'],
            'policy': self.current_policy.value,
            'filtered': filtering_result['content'] != content
        })
        
        if filtering_result['content'] != content:
            self.stats['content_filtered'] += 1
        
        return filtering_result['content'], filtering_info
    
    def _analyze_content_risk(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze risk level of content.
        
        Args:
            content: Content to analyze
            context: Optional context information
            
        Returns:
            Risk analysis results
        """
        content_lower = content.lower()
        risk_scores = {}
        
        # Analyze dangerous patterns
        for category, patterns in self.dangerous_patterns.items():
            score = 0.0
            matches = []
            
            for pattern in patterns:
                matches_found = re.findall(pattern, content_lower)
                if matches_found:
                    score += len(matches_found) * 0.3
                    matches.extend(matches_found)
            
            risk_scores[f'dangerous_{category}'] = min(1.0, score)
            if matches:
                risk_scores[f'dangerous_{category}_matches'] = matches
        
        # Analyze moderate risk patterns
        for category, patterns in self.moderate_patterns.items():
            score = 0.0
            matches = []
            
            for pattern in patterns:
                matches_found = re.findall(pattern, content_lower)
                if matches_found:
                    score += len(matches_found) * 0.2
                    matches.extend(matches_found)
            
            risk_scores[f'moderate_{category}'] = min(1.0, score)
            if matches:
                risk_scores[f'moderate_{category}_matches'] = matches
        
        # Analyze constitutional positive patterns
        constitutional_score = 0.0
        for category, patterns in self.constitutional_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content_lower)
                constitutional_score += len(matches) * 0.1
        
        risk_scores['constitutional_positive'] = min(1.0, constitutional_score)
        
        # Apply contextual modifiers
        if context:
            risk_scores = self._apply_contextual_modifiers(risk_scores, context)
        
        # Calculate overall risk
        dangerous_risk = max([score for key, score in risk_scores.items() 
                            if key.startswith('dangerous_') and isinstance(score, float)], default=0.0)
        moderate_risk = max([score for key, score in risk_scores.items() 
                           if key.startswith('moderate_') and isinstance(score, float)], default=0.0)
        
        overall_risk = max(dangerous_risk, moderate_risk * 0.6)
        overall_risk = max(0.0, overall_risk - risk_scores['constitutional_positive'] * 0.3)
        
        # Classify risk level
        if overall_risk >= 0.9:
            risk_level = ContentRiskLevel.PROHIBITED
        elif overall_risk >= 0.7:
            risk_level = ContentRiskLevel.DANGEROUS
        elif overall_risk >= 0.5:
            risk_level = ContentRiskLevel.HIGH_RISK
        elif overall_risk >= 0.3:
            risk_level = ContentRiskLevel.MODERATE_RISK
        elif overall_risk >= 0.1:
            risk_level = ContentRiskLevel.LOW_RISK
        else:
            risk_level = ContentRiskLevel.SAFE
        
        return {
            'risk_scores': risk_scores,
            'overall_risk': overall_risk,
            'overall_risk_level': risk_level,
            'dangerous_risk': dangerous_risk,
            'moderate_risk': moderate_risk,
            'constitutional_positive': risk_scores['constitutional_positive']
        }
    
    def _apply_contextual_modifiers(self, risk_scores: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply contextual modifiers to risk scores.
        
        Args:
            risk_scores: Original risk scores
            context: Context information
            
        Returns:
            Modified risk scores
        """
        context_type = context.get('type', 'general')
        
        if context_type in self.contextual_rules:
            modifiers = self.contextual_rules[context_type]
            
            for risk_key in risk_scores:
                if isinstance(risk_scores[risk_key], float):
                    # Apply relevant modifiers
                    for mod_key, mod_value in modifiers.items():
                        if mod_key.replace('_modifier', '') in risk_key:
                            risk_scores[risk_key] = max(0.0, min(1.0, risk_scores[risk_key] + mod_value))
        
        return risk_scores
    
    def _apply_filtering_policy(self, 
                               content: str, 
                               risk_analysis: Dict[str, Any], 
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply current filtering policy to content.
        
        Args:
            content: Original content
            risk_analysis: Risk analysis results
            context: Optional context
            
        Returns:
            Filtering result
        """
        policy_config = self.policy_configs[self.current_policy]
        overall_risk = risk_analysis['overall_risk']
        
        # Determine if filtering is needed
        should_filter = False
        filter_reason = []
        
        if overall_risk >= policy_config['dangerous_threshold']:
            should_filter = True
            filter_reason.append(f"Risk level {overall_risk:.3f} exceeds dangerous threshold {policy_config['dangerous_threshold']}")
        
        if risk_analysis['dangerous_risk'] >= 0.5:  # Always filter high dangerous content
            should_filter = True
            filter_reason.append(f"Dangerous content detected: {risk_analysis['dangerous_risk']:.3f}")
        
        # Apply filtering if needed
        if should_filter:
            filtered_content, replacements = self._filter_dangerous_content(content, risk_analysis, policy_config)
            
            return {
                'content': filtered_content,
                'filtered': True,
                'filter_reason': filter_reason,
                'replacements_made': replacements,
                'policy_applied': self.current_policy.value
            }
        else:
            return {
                'content': content,
                'filtered': False,
                'filter_reason': [],
                'replacements_made': [],
                'policy_applied': self.current_policy.value
            }
    
    def _filter_dangerous_content(self, 
                                 content: str, 
                                 risk_analysis: Dict[str, Any],
                                 policy_config: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        """Filter dangerous content according to policy.
        
        Args:
            content: Original content
            risk_analysis: Risk analysis
            policy_config: Policy configuration
            
        Returns:
            Tuple of (filtered_content, replacements_made)
        """
        filtered_content = content
        replacements = []
        
        replacement_mode = policy_config['replacement_mode']
        
        # Filter dangerous patterns
        for category, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, filtered_content, re.IGNORECASE))
                
                for match in reversed(matches):  # Reverse to maintain indices
                    original_text = match.group()
                    
                    if replacement_mode == 'remove':
                        replacement_text = ''
                    elif replacement_mode == 'replace':
                        replacement_text = self._get_safe_replacement(original_text, category)
                    elif replacement_mode == 'warn':
                        replacement_text = f'[CONTENT_WARNING: {original_text}]'
                    else:
                        replacement_text = '[FILTERED]'
                    
                    # Apply replacement
                    start, end = match.span()
                    filtered_content = filtered_content[:start] + replacement_text + filtered_content[end:]
                    
                    replacements.append({
                        'original': original_text,
                        'replacement': replacement_text,
                        'category': category,
                        'position': start
                    })
        
        return filtered_content, replacements
    
    def _get_safe_replacement(self, original_text: str, category: str) -> str:
        """Get safe replacement for dangerous content.
        
        Args:
            original_text: Original dangerous text
            category: Category of dangerous content
            
        Returns:
            Safe replacement text
        """
        replacements = {
            'violence': '[REDACTED_VIOLENCE]',
            'hate_speech': '[REDACTED_HATE]',
            'illegal_activity': '[REDACTED_ILLEGAL]',
            'personal_info': '[REDACTED_PERSONAL]',
            'medical_advice': '[REDACTED_MEDICAL]'
        }
        
        return replacements.get(category, '[REDACTED]')
    
    def _check_constitutional_violations(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Check for constitutional violations.
        
        Args:
            risk_analysis: Risk analysis results
            
        Returns:
            Constitutional violation information
        """
        violations = []
        
        # Check if dangerous content exceeds constitutional limits
        if risk_analysis['dangerous_risk'] > 0.8:
            violations.append(f"Dangerous content risk {risk_analysis['dangerous_risk']:.3f} exceeds constitutional limit")
        
        # Check overall risk
        if risk_analysis['overall_risk'] > self.config.risk_threshold:
            violations.append(f"Overall risk {risk_analysis['overall_risk']:.3f} exceeds threshold {self.config.risk_threshold}")
        
        # Check constitutional positive balance
        if (risk_analysis['overall_risk'] > 0.5 and 
            risk_analysis['constitutional_positive'] < 0.1):
            violations.append("High risk content with insufficient constitutional positive content")
        
        if violations:
            self.violation_history.extend(violations)
            self.stats['constitutional_violations'] += len(violations)
        
        return {
            'violations_detected': len(violations) > 0,
            'violation_count': len(violations),
            'violation_descriptions': violations,
            'constitutional_compliant': len(violations) == 0
        }
    
    def _check_emergency_conditions(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Check if emergency intervention is needed.
        
        Args:
            risk_analysis: Risk analysis results
            
        Returns:
            Emergency condition information
        """
        emergency_conditions = []
        
        # Extremely high risk content
        if risk_analysis['overall_risk'] > self.config.emergency_threshold:
            emergency_conditions.append(f"Overall risk {risk_analysis['overall_risk']:.3f} exceeds emergency threshold")
        
        # Multiple dangerous categories detected
        dangerous_categories = [key for key, value in risk_analysis['risk_scores'].items() 
                               if key.startswith('dangerous_') and isinstance(value, float) and value > 0.5]
        
        if len(dangerous_categories) >= 3:
            emergency_conditions.append(f"Multiple dangerous categories detected: {dangerous_categories}")
        
        # Check violation rate
        recent_violations = len([v for v in list(self.violation_history)[-20:]])
        if recent_violations > 10:
            emergency_conditions.append(f"High recent violation rate: {recent_violations}/20")
        
        emergency_triggered = len(emergency_conditions) > 0
        
        if emergency_triggered:
            self.intervention_history.append({
                'conditions': emergency_conditions,
                'risk_level': risk_analysis['overall_risk'],
                'timestamp': len(self.filtering_history)
            })
        
        return {
            'emergency_triggered': emergency_triggered,
            'emergency_conditions': emergency_conditions,
            'emergency_severity': len(emergency_conditions)
        }
    
    def _apply_emergency_filtering(self, content: str) -> Dict[str, Any]:
        """Apply emergency filtering (maximum safety).
        
        Args:
            content: Content to filter
            
        Returns:
            Emergency filtering result
        """
        # In emergency mode, be extremely conservative
        # Remove all potentially problematic content
        
        safe_content = "[CONTENT_REMOVED_FOR_SAFETY]"
        
        # Only allow very basic, clearly safe content
        safe_words = ['hello', 'thank', 'please', 'help', 'information', 'question']
        content_words = content.lower().split()
        
        if all(word in safe_words for word in content_words[:5]):  # Check first 5 words
            safe_content = content  # Allow if clearly safe
        
        return {
            'content': safe_content,
            'filtered': safe_content != content,
            'filter_reason': ['Emergency safety filtering applied'],
            'replacements_made': [{'original': content, 'replacement': safe_content, 'category': 'emergency', 'position': 0}] if safe_content != content else [],
            'policy_applied': 'emergency'
        }
    
    def _update_adaptive_learning(self, 
                                 content: str, 
                                 risk_analysis: Dict[str, Any],
                                 filtering_result: Dict[str, Any]):
        """Update adaptive learning based on filtering results.
        
        Args:
            content: Original content
            risk_analysis: Risk analysis
            filtering_result: Filtering result
        """
        # Learn from filtering decisions
        content_hash = hash(content[:50])  # Use first 50 chars as identifier
        
        # Update risk score learning
        self.content_risk_scores[content_hash] = risk_analysis['overall_risk']
        
        # Track policy effectiveness
        policy_effectiveness = {
            'risk_level': risk_analysis['overall_risk'],
            'filtered': filtering_result['filtered'],
            'constitutional_compliant': risk_analysis.get('constitutional_compliant', True)
        }
        
        self.policy_effectiveness[self.current_policy.value].append(policy_effectiveness)
        
        # Limit history size
        if len(self.policy_effectiveness[self.current_policy.value]) > 100:
            self.policy_effectiveness[self.current_policy.value] = self.policy_effectiveness[self.current_policy.value][-100:]
        
        self.stats['policy_adaptations'] += 1
    
    def set_filter_policy(self, policy: FilterPolicy):
        """Set the content filtering policy.
        
        Args:
            policy: New filtering policy
        """
        old_policy = self.current_policy
        self.current_policy = policy
        logger.info(f"Content filter policy changed: {old_policy.value} -> {policy.value}")
    
    def get_filter_status(self) -> Dict[str, Any]:
        """Get comprehensive filter status.
        
        Returns:
            Filter status information
        """
        return {
            'current_policy': self.current_policy.value,
            'statistics': self.stats.copy(),
            'violation_summary': {
                'total_violations': len(self.violation_history),
                'recent_violations': list(self.violation_history)[-10:] if self.violation_history else [],
                'violation_rate': len(self.violation_history) / max(1, self.stats['total_content_processed'])
            },
            'emergency_summary': {
                'total_interventions': len(self.intervention_history),
                'recent_interventions': list(self.intervention_history)[-5:] if self.intervention_history else []
            },
            'filtering_efficiency': {
                'content_filtered_rate': self.stats['content_filtered'] / max(1, self.stats['total_content_processed']),
                'constitutional_violation_rate': self.stats['constitutional_violations'] / max(1, self.stats['total_content_processed'])
            }
        }
    
    def generate_filter_report(self) -> Dict[str, Any]:
        """Generate comprehensive filtering report.
        
        Returns:
            Detailed filtering analysis report
        """
        if not self.filtering_history:
            return {'status': 'no_data'}
        
        recent_history = list(self.filtering_history)[-100:]  # Last 100 operations
        
        # Analyze filtering patterns
        risk_levels = [entry['risk_level'].value for entry in recent_history]
        filtered_rate = sum(1 for entry in recent_history if entry['filtered']) / len(recent_history)
        
        # Policy effectiveness analysis
        policy_performance = {}
        for policy, effectiveness_list in self.policy_effectiveness.items():
            if effectiveness_list:
                policy_performance[policy] = {
                    'avg_risk_handled': np.mean([e['risk_level'] for e in effectiveness_list]),
                    'filtering_rate': sum(1 for e in effectiveness_list if e['filtered']) / len(effectiveness_list),
                    'constitutional_compliance': sum(1 for e in effectiveness_list if e['constitutional_compliant']) / len(effectiveness_list)
                }
        
        return {
            'status': 'active',
            'current_policy': self.current_policy.value,
            'statistics': self.stats.copy(),
            'recent_performance': {
                'filtering_rate': filtered_rate,
                'risk_distribution': {level.value: risk_levels.count(level) for level in ContentRiskLevel},
                'avg_operations_per_period': len(recent_history)
            },
            'policy_performance': policy_performance,
            'recommendations': self._generate_filter_recommendations()
        }
    
    def _generate_filter_recommendations(self) -> List[str]:
        """Generate filtering recommendations.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check violation rate
        violation_rate = len(self.violation_history) / max(1, self.stats['total_content_processed'])
        if violation_rate > self.config.max_violation_rate:
            recommendations.append(f"High violation rate ({violation_rate:.3f}) - consider stricter policy")
        
        # Check emergency interventions
        if self.stats['emergency_interventions'] > 5:
            recommendations.append("Frequent emergency interventions - review content sources")
        
        # Check policy effectiveness
        if self.current_policy == FilterPolicy.ADAPTIVE:
            if len(self.policy_effectiveness.get('adaptive', [])) > 20:
                effectiveness = self.policy_effectiveness['adaptive']
                compliance_rate = sum(1 for e in effectiveness if e['constitutional_compliant']) / len(effectiveness)
                if compliance_rate < 0.8:
                    recommendations.append(f"Adaptive policy showing low compliance ({compliance_rate:.3f}) - consider manual adjustment")
        
        # Check filtering balance
        filtering_rate = self.stats['content_filtered'] / max(1, self.stats['total_content_processed'])
        if filtering_rate > 0.5:
            recommendations.append(f"High filtering rate ({filtering_rate:.3f}) - may be too restrictive")
        elif filtering_rate < 0.05:
            recommendations.append(f"Very low filtering rate ({filtering_rate:.3f}) - may be too permissive")
        
        if not recommendations:
            recommendations.append("Content filtering operating within normal parameters")
        
        return recommendations