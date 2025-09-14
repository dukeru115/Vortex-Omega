"""
Discrepancy Gate Module
=======================

Detects and resolves discrepancies between LLM outputs and verified facts.
Implements strict numeric and logical validation gates.

Author: Team Omega
License: CC BY-NC 4.0
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass

from .models import (
    VerificationReport, Discrepancy, Suggestion,
    SymClause, SymField, VerificationStatus
)

logger = logging.getLogger(__name__)


class DiscrepancyGate:
    """
    Gate mechanism for detecting and resolving discrepancies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize discrepancy gate
        
        Args:
            config: Gate configuration
        """
        self.config = config or {}
        
        # Severity thresholds
        self.severity_threshold = self.config.get('severity_threshold', 0.1)
        self.auto_correct = self.config.get('auto_correct', True)
        self.max_corrections = self.config.get('max_corrections', 5)
        
        # Confidence thresholds
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.high_confidence = self.config.get('high_confidence', 0.9)
        
        # Resolution strategies
        self.strategies = {
            'numeric': self._resolve_numeric_discrepancy,
            'units': self._resolve_unit_discrepancy,
            'logic': self._resolve_logic_discrepancy,
            'temporal': self._resolve_temporal_discrepancy
        }
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'discrepancies_found': 0,
            'auto_resolved': 0,
            'manual_required': 0
        }
        
        logger.info("Discrepancy gate initialized")
    
    def process(self, report: VerificationReport) -> List[Dict[str, Any]]:
        """
        Process verification report and resolve discrepancies
        
        Args:
            report: Verification report with discrepancies
        
        Returns:
            List of resolutions
        """
        resolutions = []
        self.stats['total_processed'] += 1
        
        if not report.discrepancies:
            return resolutions
        
        self.stats['discrepancies_found'] += len(report.discrepancies)
        
        # Sort discrepancies by severity
        sorted_discrepancies = sorted(
            report.discrepancies,
            key=lambda d: self._severity_score(d),
            reverse=True
        )
        
        # Process each discrepancy
        corrections_made = 0
        for discrepancy in sorted_discrepancies:
            if corrections_made >= self.max_corrections:
                logger.warning(f"Reached max corrections limit ({self.max_corrections})")
                break
            
            # Get resolution strategy
            strategy = self.strategies.get(discrepancy.field)
            
            if strategy:
                resolution = strategy(discrepancy, report)
                
                if resolution:
                    resolutions.append(resolution)
                    
                    if resolution.get('auto_applied', False):
                        self.stats['auto_resolved'] += 1
                        corrections_made += 1
                    else:
                        self.stats['manual_required'] += 1
            else:
                logger.warning(f"No strategy for discrepancy field: {discrepancy.field}")
        
        # Generate clarification questions if needed
        if self._needs_clarification(report, resolutions):
            questions = self._generate_clarification_questions(report, resolutions)
            for question in questions:
                resolutions.append({
                    'type': 'clarification',
                    'target': 'user',
                    'patch': question,
                    'reason': 'Ambiguity detected',
                    'confidence': 0.5
                })
        
        return resolutions
    
    def _severity_score(self, discrepancy: Discrepancy) -> float:
        """
        Calculate severity score for discrepancy
        
        Args:
            discrepancy: Discrepancy to score
        
        Returns:
            Severity score (0-1)
        """
        base_score = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.9
        }.get(discrepancy.severity, 0.5)
        
        # Adjust based on field type
        field_multiplier = {
            'logic': 1.2,  # Logic errors are more critical
            'numeric': 1.0,
            'units': 1.1,  # Unit errors can be dangerous
            'temporal': 0.8
        }.get(discrepancy.field, 1.0)
        
        return min(1.0, base_score * field_multiplier)
    
    def _resolve_numeric_discrepancy(self, 
                                    discrepancy: Discrepancy,
                                    report: VerificationReport) -> Optional[Dict[str, Any]]:
        """
        Resolve numeric discrepancy
        
        Args:
            discrepancy: Numeric discrepancy
            report: Full verification report
        
        Returns:
            Resolution or None
        """
        if discrepancy.expected is None or discrepancy.actual is None:
            return None
        
        try:
            expected_val = float(discrepancy.expected)
            actual_val = float(discrepancy.actual)
        except (ValueError, TypeError):
            return None
        
        # Calculate error
        abs_error = abs(expected_val - actual_val)
        if expected_val != 0:
            rel_error = abs_error / abs(expected_val)
        else:
            rel_error = abs_error
        
        # Determine resolution
        resolution = {
            'type': 'numeric_correction',
            'target': discrepancy.cid,
            'original': actual_val,
            'corrected': expected_val,
            'error': rel_error
        }
        
        # Check if auto-correction is appropriate
        if self.auto_correct and rel_error < self.severity_threshold:
            resolution['auto_applied'] = True
            resolution['patch'] = str(expected_val)
            resolution['reason'] = f'Auto-corrected small numeric error ({rel_error:.2%})'
            resolution['confidence'] = self.high_confidence
        else:
            resolution['auto_applied'] = False
            resolution['patch'] = f"Change {actual_val} to {expected_val}"
            resolution['reason'] = f'Significant numeric discrepancy ({rel_error:.2%})'
            resolution['confidence'] = self.min_confidence
        
        return resolution
    
    def _resolve_unit_discrepancy(self,
                                 discrepancy: Discrepancy,
                                 report: VerificationReport) -> Optional[Dict[str, Any]]:
        """
        Resolve unit/dimensional discrepancy
        
        Args:
            discrepancy: Unit discrepancy
            report: Full verification report
        
        Returns:
            Resolution or None
        """
        resolution = {
            'type': 'unit_correction',
            'target': discrepancy.cid,
            'reason': 'Dimensional inconsistency detected'
        }
        
        # Try to determine correct units
        # In production, would analyze context
        
        if discrepancy.expected and discrepancy.actual:
            resolution['patch'] = f"Ensure units are consistent: expected {discrepancy.expected}"
            resolution['confidence'] = 0.8
        else:
            resolution['patch'] = "Review and correct unit specifications"
            resolution['confidence'] = 0.6
        
        resolution['auto_applied'] = False  # Unit corrections require review
        
        return resolution
    
    def _resolve_logic_discrepancy(self,
                                  discrepancy: Discrepancy,
                                  report: VerificationReport) -> Optional[Dict[str, Any]]:
        """
        Resolve logical discrepancy
        
        Args:
            discrepancy: Logic discrepancy
            report: Full verification report
        
        Returns:
            Resolution or None
        """
        resolution = {
            'type': 'logic_correction',
            'target': discrepancy.cid,
            'reason': 'Logical inconsistency detected'
        }
        
        # Analyze the type of logical error
        if 'contradiction' in str(discrepancy.actual).lower():
            resolution['patch'] = "Remove or modify contradictory constraints"
            resolution['confidence'] = 0.7
            
            # Try to identify which constraint to remove
            # In production, would use SAT solver
            
        elif 'unsatisfiable' in str(discrepancy.actual).lower():
            resolution['patch'] = "Relax constraints to make system satisfiable"
            resolution['confidence'] = 0.6
        
        else:
            resolution['patch'] = "Review logical relationships"
            resolution['confidence'] = 0.5
        
        resolution['auto_applied'] = False  # Logic corrections need human review
        
        return resolution
    
    def _resolve_temporal_discrepancy(self,
                                     discrepancy: Discrepancy,
                                     report: VerificationReport) -> Optional[Dict[str, Any]]:
        """
        Resolve temporal/causal discrepancy
        
        Args:
            discrepancy: Temporal discrepancy
            report: Full verification report
        
        Returns:
            Resolution or None
        """
        resolution = {
            'type': 'temporal_correction',
            'target': discrepancy.cid,
            'reason': 'Temporal inconsistency detected'
        }
        
        # Analyze temporal issue
        if 'cycle' in str(discrepancy.actual).lower():
            resolution['patch'] = "Break causal cycle by reordering events"
            resolution['confidence'] = 0.7
        elif 'order' in str(discrepancy.actual).lower():
            resolution['patch'] = "Correct temporal ordering of events"
            resolution['confidence'] = 0.8
        else:
            resolution['patch'] = "Review temporal relationships"
            resolution['confidence'] = 0.5
        
        resolution['auto_applied'] = False
        
        return resolution
    
    def _needs_clarification(self,
                           report: VerificationReport,
                           resolutions: List[Dict]) -> bool:
        """
        Check if clarification from user is needed
        
        Args:
            report: Verification report
            resolutions: Current resolutions
        
        Returns:
            True if clarification needed
        """
        # Need clarification if:
        # 1. Multiple high-severity discrepancies
        high_severity = sum(1 for d in report.discrepancies if d.severity == 'high')
        if high_severity > 2:
            return True
        
        # 2. Low confidence in resolutions
        if resolutions:
            avg_confidence = np.mean([r.get('confidence', 0.5) for r in resolutions])
            if avg_confidence < self.min_confidence:
                return True
        
        # 3. Conflicting resolutions
        if self._has_conflicts(resolutions):
            return True
        
        return False
    
    def _has_conflicts(self, resolutions: List[Dict]) -> bool:
        """
        Check if resolutions conflict with each other
        
        Args:
            resolutions: List of resolutions
        
        Returns:
            True if conflicts exist
        """
        # Check for multiple resolutions targeting same clause
        targets = {}
        for res in resolutions:
            target = res.get('target')
            if target:
                if target in targets:
                    # Multiple resolutions for same target
                    return True
                targets[target] = res
        
        return False
    
    def _generate_clarification_questions(self,
                                        report: VerificationReport,
                                        resolutions: List[Dict]) -> List[str]:
        """
        Generate clarification questions for user
        
        Args:
            report: Verification report
            resolutions: Current resolutions
        
        Returns:
            List of questions
        """
        questions = []
        
        # Ask about high-severity discrepancies
        for disc in report.discrepancies:
            if disc.severity == 'high':
                if disc.field == 'numeric':
                    questions.append(
                        f"Please confirm the value in {disc.cid}: "
                        f"is it {disc.expected} or {disc.actual}?"
                    )
                elif disc.field == 'units':
                    questions.append(
                        f"Please clarify the units for {disc.cid}: "
                        f"what unit system should be used?"
                    )
                elif disc.field == 'logic':
                    questions.append(
                        f"There's a logical inconsistency in {disc.cid}. "
                        f"Which constraint should take precedence?"
                    )
        
        # Ask about low-confidence resolutions
        for res in resolutions:
            if res.get('confidence', 1.0) < self.min_confidence:
                questions.append(
                    f"Low confidence in resolution for {res.get('target')}: "
                    f"{res.get('patch')}. Is this correct?"
                )
        
        # Limit number of questions
        return questions[:3]  # Max 3 questions at a time
    
    def apply_resolutions(self,
                        fields: List[SymField],
                        resolutions: List[Dict]) -> List[SymField]:
        """
        Apply resolutions to fields
        
        Args:
            fields: Original fields
            resolutions: Resolutions to apply
        
        Returns:
            Updated fields
        """
        # Create a copy to avoid modifying original
        updated_fields = []
        
        for field in fields:
            updated_field = SymField(
                fid=field.fid,
                title=field.title,
                clauses=list(field.clauses),
                invariants=list(field.invariants),
                obligations=list(field.obligations),
                domain=field.domain
            )
            
            # Apply resolutions to clauses
            for resolution in resolutions:
                if resolution.get('auto_applied', False):
                    target = resolution.get('target')
                    
                    # Find and update target clause
                    for clause in updated_field.clauses:
                        if clause.cid == target:
                            self._apply_resolution_to_clause(clause, resolution)
                    
                    for clause in updated_field.invariants:
                        if clause.cid == target:
                            self._apply_resolution_to_clause(clause, resolution)
            
            updated_fields.append(updated_field)
        
        return updated_fields
    
    def _apply_resolution_to_clause(self,
                                   clause: SymClause,
                                   resolution: Dict):
        """
        Apply resolution to a specific clause
        
        Args:
            clause: Clause to update
            resolution: Resolution to apply
        """
        res_type = resolution.get('type')
        
        if res_type == 'numeric_correction':
            # Update numeric value
            if hasattr(clause.rhs, 'value'):
                clause.rhs.value = resolution.get('corrected')
            clause.numeric_ok = True
        
        elif res_type == 'unit_correction':
            # Mark units as needing review
            clause.units_ok = False
            clause.meta['needs_unit_review'] = True
        
        elif res_type == 'logic_correction':
            # Mark logic as needing review
            clause.logic_ok = False
            clause.meta['needs_logic_review'] = True
        
        # Add resolution metadata
        if 'resolutions' not in clause.meta:
            clause.meta['resolutions'] = []
        clause.meta['resolutions'].append({
            'type': res_type,
            'patch': resolution.get('patch'),
            'confidence': resolution.get('confidence')
        })
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get gate statistics
        
        Returns:
            Statistics dictionary
        """
        return self.stats.copy()