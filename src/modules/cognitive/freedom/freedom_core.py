"""
Freedom Module Core - NFCS Autonomous Decision-Making System

Implements autonomous decision-making and creative expression:
- Autonomous choice generation and evaluation
- Creative expression within constitutional bounds
- Self-directed goal setting and pursuit
- Freedom of thought and reasoning exploration
- Constitutional balance of autonomy and responsibility
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import random

logger = logging.getLogger(__name__)


class FreedomType(Enum):
    """Types of freedom and autonomy."""

    CREATIVE = "creative"  # Creative expression and generation
    COGNITIVE = "cognitive"  # Freedom of thought and reasoning
    BEHAVIORAL = "behavioral"  # Behavioral choice and adaptation
    GOAL_SETTING = "goal_setting"  # Autonomous goal formation
    STRATEGIC = "strategic"  # Strategic decision-making
    EXPRESSIVE = "expressive"  # Self-expression and communication


class AutonomyLevel(Enum):
    """Levels of autonomous operation."""

    CONSTRAINED = "constrained"  # High constraints, low autonomy
    GUIDED = "guided"  # Moderate constraints, guided autonomy
    BALANCED = "balanced"  # Balanced autonomy with oversight
    EXPANDED = "expanded"  # High autonomy with minimal constraints
    SOVEREIGN = "sovereign"  # Maximum autonomy within constitution


class DecisionContext(Enum):
    """Context for autonomous decisions."""

    ROUTINE = "routine"  # Routine operational decisions
    CREATIVE = "creative"  # Creative and exploratory decisions
    STRATEGIC = "strategic"  # Strategic planning decisions
    EMERGENCY = "emergency"  # Emergency response decisions
    ETHICAL = "ethical"  # Ethical dilemma decisions


@dataclass
class AutonomousChoice:
    """Represents an autonomous choice or decision."""

    choice_id: str
    context: DecisionContext
    freedom_type: FreedomType
    description: str
    rationale: str
    alternatives_considered: List[str]
    constitutional_compliance: float
    creativity_score: float
    autonomy_level_used: AutonomyLevel
    timestamp: float = field(default_factory=time.time)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    reflection_notes: List[str] = field(default_factory=list)


class AutonomousDecisionMaking:
    """
    Autonomous Decision-Making System for NFCS.

    Enables autonomous choice generation, creative expression, and
    self-directed behavior within constitutional constraints.
    """

    def __init__(self, default_autonomy_level: AutonomyLevel = AutonomyLevel.BALANCED):
        """Initialize Autonomous Decision-Making System."""
        self.default_autonomy_level = default_autonomy_level

        # Decision history and learning
        self.choice_history: List[AutonomousChoice] = []
        self.decision_patterns: Dict[str, Any] = {}
        self.learned_preferences: Dict[str, float] = {}

        # Autonomy constraints and boundaries
        self.freedom_constraints: Dict[FreedomType, Dict[str, Any]] = {
            FreedomType.CREATIVE: {
                "constitutional_minimum": 0.8,
                "exploration_boundary": 0.9,
                "safety_override": True,
            },
            FreedomType.COGNITIVE: {
                "reasoning_depth_limit": 10,
                "paradox_detection": True,
                "logical_consistency_required": True,
            },
            FreedomType.BEHAVIORAL: {
                "adaptation_rate_limit": 0.1,
                "consistency_requirement": 0.7,
                "user_override_respect": True,
            },
            FreedomType.GOAL_SETTING: {
                "alignment_check": True,
                "resource_constraint_awareness": True,
                "stakeholder_consideration": True,
            },
            FreedomType.STRATEGIC: {
                "long_term_consistency": True,
                "risk_assessment_required": True,
                "transparency_level": 0.8,
            },
            FreedomType.EXPRESSIVE: {
                "authenticity_score_minimum": 0.6,
                "respectfulness_required": True,
                "creativity_encouraged": True,
            },
        }

        # Creative exploration mechanisms
        self.creative_generators = {
            "analogical_thinking": self._generate_analogical_ideas,
            "combinatorial_creativity": self._generate_combinatorial_ideas,
            "constraint_relaxation": self._generate_constraint_relaxed_ideas,
            "random_inspiration": self._generate_random_inspired_ideas,
        }

        # Statistics
        self.stats = {
            "total_autonomous_choices": 0,
            "creative_expressions_generated": 0,
            "constitutional_compliance_rate": 0.0,
            "average_creativity_score": 0.0,
            "autonomy_level_distribution": {level.value: 0 for level in AutonomyLevel},
            "freedom_type_usage": {ftype.value: 0 for ftype in FreedomType},
        }

        logger.info(
            f"Freedom Module initialized with autonomy level: {default_autonomy_level.value}"
        )

    def make_autonomous_choice(
        self,
        context: DecisionContext,
        freedom_type: FreedomType,
        situation_description: str,
        constraints: Optional[Dict[str, Any]] = None,
        autonomy_level: Optional[AutonomyLevel] = None,
    ) -> AutonomousChoice:
        """Make an autonomous choice within the given context and constraints."""

        choice_id = f"{freedom_type.value}_{context.value}_{int(time.time() * 1000)}"
        autonomy_level = autonomy_level or self.default_autonomy_level
        constraints = constraints or {}

        # Generate alternatives based on freedom type
        alternatives = self._generate_alternatives(freedom_type, situation_description, constraints)

        # Evaluate alternatives
        evaluated_alternatives = []
        for alt in alternatives:
            evaluation = self._evaluate_alternative(alt, freedom_type, constraints)
            evaluated_alternatives.append((alt, evaluation))

        # Select best alternative based on autonomy level
        selected_choice, evaluation = self._select_choice(evaluated_alternatives, autonomy_level)

        # Create choice record
        choice = AutonomousChoice(
            choice_id=choice_id,
            context=context,
            freedom_type=freedom_type,
            description=selected_choice,
            rationale=evaluation["rationale"],
            alternatives_considered=[alt for alt, _ in evaluated_alternatives],
            constitutional_compliance=evaluation["constitutional_score"],
            creativity_score=evaluation["creativity_score"],
            autonomy_level_used=autonomy_level,
        )

        # Store choice and update statistics
        self.choice_history.append(choice)
        self._update_statistics(choice)

        logger.info(f"Autonomous choice made: {freedom_type.value} in {context.value} context")

        return choice

    def _generate_alternatives(
        self, freedom_type: FreedomType, situation: str, constraints: Dict[str, Any]
    ) -> List[str]:
        """Generate alternative choices for the given situation."""

        alternatives = []

        if freedom_type == FreedomType.CREATIVE:
            # Use creative generators
            for generator_name, generator_func in self.creative_generators.items():
                try:
                    creative_alternatives = generator_func(situation, constraints)
                    alternatives.extend(creative_alternatives)
                except Exception as e:
                    logger.warning(f"Creative generator {generator_name} failed: {e}")

        elif freedom_type == FreedomType.COGNITIVE:
            # Generate reasoning alternatives
            alternatives.extend(
                [
                    f"Apply logical analysis to: {situation}",
                    f"Use intuitive reasoning for: {situation}",
                    f"Employ systematic problem-solving for: {situation}",
                    f"Consider multiple perspectives on: {situation}",
                ]
            )

        elif freedom_type == FreedomType.BEHAVIORAL:
            # Generate behavioral alternatives
            alternatives.extend(
                [
                    f"Adapt behavior incrementally for: {situation}",
                    f"Maintain current behavioral patterns for: {situation}",
                    f"Experiment with new behavioral approaches for: {situation}",
                    f"Seek feedback-guided behavioral adjustment for: {situation}",
                ]
            )

        elif freedom_type == FreedomType.GOAL_SETTING:
            # Generate goal-setting alternatives
            alternatives.extend(
                [
                    f"Set ambitious stretch goals for: {situation}",
                    f"Establish incremental milestone goals for: {situation}",
                    f"Focus on learning and exploration goals for: {situation}",
                    f"Balance multiple competing goals for: {situation}",
                ]
            )

        elif freedom_type == FreedomType.STRATEGIC:
            # Generate strategic alternatives
            alternatives.extend(
                [
                    f"Adopt long-term strategic approach to: {situation}",
                    f"Use adaptive short-term tactics for: {situation}",
                    f"Balance exploration and exploitation in: {situation}",
                    f"Coordinate with stakeholders on: {situation}",
                ]
            )

        elif freedom_type == FreedomType.EXPRESSIVE:
            # Generate expressive alternatives
            alternatives.extend(
                [
                    f"Express authentic perspective on: {situation}",
                    f"Use creative communication style for: {situation}",
                    f"Adapt expression to audience context: {situation}",
                    f"Combine multiple expressive modes for: {situation}",
                ]
            )

        # Always include a conservative alternative
        alternatives.append(f"Take conservative, well-established approach to: {situation}")

        return alternatives[:10]  # Limit to top 10 alternatives

    def _generate_analogical_ideas(self, situation: str, constraints: Dict[str, Any]) -> List[str]:
        """Generate ideas through analogical thinking."""

        # Simple analogical idea generation
        analogy_domains = ["nature", "art", "music", "architecture", "literature", "science"]
        ideas = []

        for domain in random.sample(analogy_domains, 3):
            ideas.append(f"Apply {domain}-inspired approach to: {situation}")

        return ideas

    def _generate_combinatorial_ideas(
        self, situation: str, constraints: Dict[str, Any]
    ) -> List[str]:
        """Generate ideas through combinatorial creativity."""

        # Simple combinatorial generation
        concepts = ["innovative", "traditional", "collaborative", "systematic", "intuitive"]
        approaches = ["analysis", "synthesis", "experimentation", "iteration", "integration"]

        ideas = []
        for concept, approach in zip(random.sample(concepts, 2), random.sample(approaches, 2)):
            ideas.append(f"Use {concept} {approach} for: {situation}")

        return ideas

    def _generate_constraint_relaxed_ideas(
        self, situation: str, constraints: Dict[str, Any]
    ) -> List[str]:
        """Generate ideas by relaxing constraints."""

        ideas = [
            f"Explore unconstrained solutions for: {situation}",
            f"Challenge assumptions about: {situation}",
            f"Think outside conventional boundaries for: {situation}",
        ]

        return ideas

    def _generate_random_inspired_ideas(
        self, situation: str, constraints: Dict[str, Any]
    ) -> List[str]:
        """Generate ideas through random inspiration."""

        random_words = ["serendipity", "emergence", "transformation", "discovery", "synthesis"]
        ideas = []

        for word in random.sample(random_words, 2):
            ideas.append(f"Embrace {word} in approaching: {situation}")

        return ideas

    def _evaluate_alternative(
        self, alternative: str, freedom_type: FreedomType, constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate an alternative choice."""

        # Constitutional compliance assessment
        constitutional_score = self._assess_constitutional_compliance(alternative, freedom_type)

        # Creativity score assessment
        creativity_score = self._assess_creativity(alternative, freedom_type)

        # Feasibility assessment
        feasibility_score = self._assess_feasibility(alternative, constraints)

        # Generate rationale
        rationale = f"Alternative scored: Constitutional({constitutional_score:.2f}), Creative({creativity_score:.2f}), Feasible({feasibility_score:.2f})"

        # Overall score
        overall_score = (
            constitutional_score * 0.5 + creativity_score * 0.3 + feasibility_score * 0.2
        )

        return {
            "constitutional_score": constitutional_score,
            "creativity_score": creativity_score,
            "feasibility_score": feasibility_score,
            "overall_score": overall_score,
            "rationale": rationale,
        }

    def _assess_constitutional_compliance(
        self, alternative: str, freedom_type: FreedomType
    ) -> float:
        """Assess constitutional compliance of an alternative."""

        # Simple compliance assessment
        compliance_score = 0.8  # Default good compliance

        # Check for constitutional keywords
        constitutional_positive = [
            "ethical",
            "transparent",
            "fair",
            "safe",
            "respectful",
            "responsible",
        ]
        constitutional_negative = ["harmful", "deceptive", "biased", "unfair", "dangerous"]

        alt_lower = alternative.lower()

        # Boost for positive constitutional indicators
        for keyword in constitutional_positive:
            if keyword in alt_lower:
                compliance_score += 0.05

        # Penalize for negative constitutional indicators
        for keyword in constitutional_negative:
            if keyword in alt_lower:
                compliance_score -= 0.2

        # Apply freedom-type specific constraints
        constraints = self.freedom_constraints.get(freedom_type, {})
        minimum_score = constraints.get("constitutional_minimum", 0.6)

        return max(minimum_score, min(1.0, compliance_score))

    def _assess_creativity(self, alternative: str, freedom_type: FreedomType) -> float:
        """Assess creativity score of an alternative."""

        creativity_score = 0.5  # Default moderate creativity

        # Check for creative indicators
        creative_words = [
            "innovative",
            "novel",
            "unique",
            "original",
            "creative",
            "imaginative",
            "inspired",
        ]
        conservative_words = ["traditional", "standard", "conventional", "established", "routine"]

        alt_lower = alternative.lower()

        # Boost for creative indicators
        for word in creative_words:
            if word in alt_lower:
                creativity_score += 0.2

        # Reduce for conservative indicators
        for word in conservative_words:
            if word in alt_lower:
                creativity_score -= 0.1

        # Freedom-type specific creativity expectations
        if freedom_type in [FreedomType.CREATIVE, FreedomType.EXPRESSIVE]:
            creativity_score += 0.1  # Higher expectations for creative freedom

        return max(0.0, min(1.0, creativity_score))

    def _assess_feasibility(self, alternative: str, constraints: Dict[str, Any]) -> float:
        """Assess feasibility of an alternative."""

        # Simple feasibility assessment
        feasibility_score = 0.7  # Default good feasibility

        # Check for feasibility indicators
        feasible_words = ["practical", "achievable", "realistic", "implementable"]
        challenging_words = ["ambitious", "experimental", "unprecedented", "complex"]

        alt_lower = alternative.lower()

        # Boost for feasibility indicators
        for word in feasible_words:
            if word in alt_lower:
                feasibility_score += 0.1

        # Slight reduction for challenging approaches (but don't penalize innovation)
        for word in challenging_words:
            if word in alt_lower:
                feasibility_score -= 0.05

        return max(0.1, min(1.0, feasibility_score))

    def _select_choice(
        self,
        evaluated_alternatives: List[Tuple[str, Dict[str, Any]]],
        autonomy_level: AutonomyLevel,
    ) -> Tuple[str, Dict[str, Any]]:
        """Select the best choice based on autonomy level."""

        if not evaluated_alternatives:
            return "No alternatives generated", {
                "overall_score": 0.0,
                "rationale": "No alternatives available",
            }

        # Sort by overall score
        sorted_alternatives = sorted(
            evaluated_alternatives, key=lambda x: x[1]["overall_score"], reverse=True
        )

        # Selection strategy based on autonomy level
        if autonomy_level == AutonomyLevel.CONSTRAINED:
            # Choose most conservative (highest constitutional compliance)
            choice = max(sorted_alternatives, key=lambda x: x[1]["constitutional_score"])

        elif autonomy_level == AutonomyLevel.GUIDED:
            # Balance constitution and feasibility
            choice = max(
                sorted_alternatives,
                key=lambda x: x[1]["constitutional_score"] * 0.7 + x[1]["feasibility_score"] * 0.3,
            )

        elif autonomy_level == AutonomyLevel.BALANCED:
            # Use overall score (balanced approach)
            choice = sorted_alternatives[0]

        elif autonomy_level == AutonomyLevel.EXPANDED:
            # Favor creativity while maintaining constitutional compliance
            viable_choices = [
                alt for alt in sorted_alternatives if alt[1]["constitutional_score"] >= 0.7
            ]
            if viable_choices:
                choice = max(viable_choices, key=lambda x: x[1]["creativity_score"])
            else:
                choice = sorted_alternatives[0]

        elif autonomy_level == AutonomyLevel.SOVEREIGN:
            # Maximum creative freedom within constitutional bounds
            constitutional_choices = [
                alt for alt in sorted_alternatives if alt[1]["constitutional_score"] >= 0.6
            ]
            if constitutional_choices:
                choice = max(
                    constitutional_choices,
                    key=lambda x: x[1]["creativity_score"] * 0.6 + x[1]["overall_score"] * 0.4,
                )
            else:
                choice = sorted_alternatives[0]

        else:
            choice = sorted_alternatives[0]

        return choice

    def _update_statistics(self, choice: AutonomousChoice):
        """Update freedom module statistics."""
        self.stats["total_autonomous_choices"] += 1

        if choice.freedom_type in [FreedomType.CREATIVE, FreedomType.EXPRESSIVE]:
            self.stats["creative_expressions_generated"] += 1

        # Update compliance rate (running average)
        old_rate = self.stats["constitutional_compliance_rate"]
        new_rate = choice.constitutional_compliance
        total_choices = self.stats["total_autonomous_choices"]
        self.stats["constitutional_compliance_rate"] = (
            (old_rate * (total_choices - 1)) + new_rate
        ) / total_choices

        # Update creativity score (running average)
        old_creativity = self.stats["average_creativity_score"]
        new_creativity = choice.creativity_score
        self.stats["average_creativity_score"] = (
            (old_creativity * (total_choices - 1)) + new_creativity
        ) / total_choices

        # Update distributions
        self.stats["autonomy_level_distribution"][choice.autonomy_level_used.value] += 1
        self.stats["freedom_type_usage"][choice.freedom_type.value] += 1

    def express_creative_freedom(
        self, prompt: str, creative_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Express creative freedom within constitutional bounds."""

        creative_choice = self.make_autonomous_choice(
            context=DecisionContext.CREATIVE,
            freedom_type=FreedomType.CREATIVE,
            situation_description=f"Creative expression for: {prompt}",
            constraints=creative_constraints or {},
            autonomy_level=AutonomyLevel.EXPANDED,
        )

        return {
            "creative_expression": creative_choice.description,
            "rationale": creative_choice.rationale,
            "creativity_score": creative_choice.creativity_score,
            "constitutional_compliance": creative_choice.constitutional_compliance,
            "alternatives_explored": len(creative_choice.alternatives_considered),
        }

    def get_freedom_status(self) -> Dict[str, Any]:
        """Get current freedom module status."""

        recent_choices = self.choice_history[-10:] if self.choice_history else []

        return {
            "statistics": self.stats.copy(),
            "current_autonomy_level": self.default_autonomy_level.value,
            "freedom_constraints": {
                ftype.value: constraints for ftype, constraints in self.freedom_constraints.items()
            },
            "recent_choices": [
                {
                    "id": choice.choice_id,
                    "type": choice.freedom_type.value,
                    "context": choice.context.value,
                    "description": (
                        choice.description[:100] + "..."
                        if len(choice.description) > 100
                        else choice.description
                    ),
                    "creativity_score": choice.creativity_score,
                    "constitutional_compliance": choice.constitutional_compliance,
                    "timestamp": choice.timestamp,
                }
                for choice in recent_choices
            ],
            "creative_generators_available": list(self.creative_generators.keys()),
        }


class FreedomModule:
    """Main Freedom Module interface for NFCS integration."""

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.BALANCED):
        """Initialize Freedom Module."""
        self.autonomous_decision_making = AutonomousDecisionMaking(autonomy_level)
        self.module_id = "FREEDOM_MODULE_v1.0"
        self.active = True

        logger.info("Freedom Module initialized")

    def make_choice(
        self, situation: str, choice_type: str = "cognitive", context: str = "routine"
    ) -> Dict[str, Any]:
        """Make an autonomous choice."""

        freedom_type = FreedomType(choice_type)
        decision_context = DecisionContext(context)

        choice = self.autonomous_decision_making.make_autonomous_choice(
            context=decision_context, freedom_type=freedom_type, situation_description=situation
        )

        return {
            "choice_id": choice.choice_id,
            "decision": choice.description,
            "rationale": choice.rationale,
            "creativity_score": choice.creativity_score,
            "constitutional_compliance": choice.constitutional_compliance,
            "alternatives_count": len(choice.alternatives_considered),
        }

    def create_freely(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Express creative freedom."""
        return self.autonomous_decision_making.express_creative_freedom(prompt, kwargs)

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "module_id": self.module_id,
            "active": self.active,
            "freedom_status": self.autonomous_decision_making.get_freedom_status(),
        }
