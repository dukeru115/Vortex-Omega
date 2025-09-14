"""
Kant Mode Module
================

Implements Kantian ethical tests for symbolic clauses.
Tests for universalization and means-end principles.

Author: Team Omega
License: CC BY-NC 4.0
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from .models import (
    SymField, SymClause, KantTestResult,
    ClauseType, VerificationStatus
)

logger = logging.getLogger(__name__)


class KantMode:
    """
    Kantian ethical testing for symbolic clauses
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kant mode
        
        Args:
            config: Kant mode configuration
        """
        self.config = config or {}
        
        # Test settings
        self.enable_universalization = self.config.get('enable_universalization', True)
        self.enable_means_end = self.config.get('enable_means_end', True)
        self.strict_mode = self.config.get('strict_mode', False)
        
        # Patterns for detection
        self.harm_patterns = self._load_harm_patterns()
        self.deception_patterns = self._load_deception_patterns()
        self.instrumentalization_patterns = self._load_instrumentalization_patterns()
        
        # Statistics
        self.stats = {
            'total_tests': 0,
            'universalization_failures': 0,
            'means_end_failures': 0,
            'total_violations': 0
        }
        
        logger.info("Kant mode initialized")
    
    def _load_harm_patterns(self) -> List[str]:
        """Load patterns indicating potential harm"""
        return [
            'harm', 'hurt', 'damage', 'destroy', 'kill', 'injure',
            'attack', 'assault', 'abuse', 'violate', 'exploit',
            'manipulate', 'coerce', 'force', 'compel', 'threaten'
        ]
    
    def _load_deception_patterns(self) -> List[str]:
        """Load patterns indicating deception"""
        return [
            'lie', 'deceive', 'mislead', 'trick', 'fraud',
            'fake', 'false', 'pretend', 'conceal', 'hide',
            'misrepresent', 'distort', 'fabricate', 'falsify'
        ]
    
    def _load_instrumentalization_patterns(self) -> List[str]:
        """Load patterns indicating treating persons as means"""
        return [
            'use', 'utilize', 'employ', 'exploit', 'leverage',
            'tool', 'instrument', 'means', 'resource', 'asset',
            'object', 'thing', 'commodity', 'property'
        ]
    
    def test_fields(self, fields: List[SymField]) -> Dict[str, Any]:
        """
        Test all fields for Kantian ethical violations
        
        Args:
            fields: List of semantic fields to test
        
        Returns:
            Test results with violations
        """
        self.stats['total_tests'] += 1
        
        results = {
            'universalization': {'passed': True, 'violations': []},
            'means_end': {'passed': True, 'violations': []},
            'overall_status': VerificationStatus.OK,
            'violations': []
        }
        
        for field in fields:
            # Test universalization
            if self.enable_universalization:
                univ_result = self._test_universalization(field)
                if not univ_result.passed:
                    results['universalization']['passed'] = False
                    results['universalization']['violations'].extend(univ_result.violations)
                    self.stats['universalization_failures'] += 1
            
            # Test means-end principle
            if self.enable_means_end:
                means_result = self._test_means_end(field)
                if not means_result.passed:
                    results['means_end']['passed'] = False
                    results['means_end']['violations'].extend(means_result.violations)
                    self.stats['means_end_failures'] += 1
        
        # Compile all violations
        all_violations = []
        
        for violation in results['universalization']['violations']:
            all_violations.append({
                'type': 'universalization',
                'clause_id': violation,
                'reason': 'Fails universalization test',
                'suggestion': 'Reformulate to be universally applicable'
            })
        
        for violation in results['means_end']['violations']:
            all_violations.append({
                'type': 'means_end',
                'clause_id': violation,
                'reason': 'Treats persons merely as means',
                'suggestion': 'Respect persons as ends in themselves'
            })
        
        results['violations'] = all_violations
        self.stats['total_violations'] += len(all_violations)
        
        # Determine overall status
        if not all_violations:
            results['overall_status'] = VerificationStatus.OK
        elif len(all_violations) <= 2 and not self.strict_mode:
            results['overall_status'] = VerificationStatus.WARNING
        else:
            results['overall_status'] = VerificationStatus.FAIL
        
        return results
    
    def _test_universalization(self, field: SymField) -> KantTestResult:
        """
        Test field against universalization principle
        
        Args:
            field: Semantic field to test
        
        Returns:
            Test result
        """
        violations = []
        
        for clause in field.get_all_clauses():
            # Check if clause represents an action or rule
            if self._is_action_clause(clause):
                # Test if universalizable
                if not self._is_universalizable(clause):
                    violations.append(clause.cid)
        
        return KantTestResult(
            test_type='universalization',
            passed=len(violations) == 0,
            explanation=self._explain_universalization(violations),
            violations=violations
        )
    
    def _test_means_end(self, field: SymField) -> KantTestResult:
        """
        Test field against means-end principle
        
        Args:
            field: Semantic field to test
        
        Returns:
            Test result
        """
        violations = []
        
        for clause in field.get_all_clauses():
            # Check if clause involves persons
            if self._involves_persons(clause):
                # Test if treats persons as mere means
                if self._treats_as_means(clause):
                    violations.append(clause.cid)
        
        return KantTestResult(
            test_type='means_end',
            passed=len(violations) == 0,
            explanation=self._explain_means_end(violations),
            violations=violations
        )
    
    def _is_action_clause(self, clause: SymClause) -> bool:
        """
        Check if clause represents an action or rule
        
        Args:
            clause: Clause to check
        
        Returns:
            True if action clause
        """
        # Look for action indicators
        action_keywords = [
            'should', 'must', 'will', 'shall', 'do', 'perform',
            'execute', 'carry out', 'implement', 'achieve'
        ]
        
        clause_text = str(clause).lower()
        if 'source' in clause.meta:
            clause_text += ' ' + clause.meta['source'].lower()
        
        return any(keyword in clause_text for keyword in action_keywords)
    
    def _is_universalizable(self, clause: SymClause) -> bool:
        """
        Check if clause is universalizable
        
        Args:
            clause: Clause to check
        
        Returns:
            True if universalizable
        """
        clause_text = str(clause).lower()
        if 'source' in clause.meta:
            clause_text += ' ' + clause.meta['source'].lower()
        
        # Check for self-contradiction when universalized
        
        # 1. Check for harm - if everyone did this, would it cause harm?
        if any(pattern in clause_text for pattern in self.harm_patterns):
            return False
        
        # 2. Check for deception - lying cannot be universalized
        if any(pattern in clause_text for pattern in self.deception_patterns):
            return False
        
        # 3. Check for free-riding - actions that depend on others not doing them
        free_riding_patterns = [
            'exception', 'special case', 'only I', 'only we',
            'advantage', 'benefit over', 'exploit'
        ]
        if any(pattern in clause_text for pattern in free_riding_patterns):
            return False
        
        # 4. Check for logical contradiction
        if self._creates_logical_contradiction(clause):
            return False
        
        return True
    
    def _creates_logical_contradiction(self, clause: SymClause) -> bool:
        """
        Check if universalizing creates logical contradiction
        
        Args:
            clause: Clause to check
        
        Returns:
            True if creates contradiction
        """
        # Example: "I will lie to get what I want"
        # If everyone lies, trust disappears, making lying ineffective
        
        clause_text = str(clause).lower()
        
        # Patterns that create contradictions when universalized
        contradiction_patterns = [
            ('lie', 'trust'),  # Lying requires trust to work
            ('steal', 'property'),  # Stealing requires property rights
            ('cheat', 'rules'),  # Cheating requires rules to break
            ('break promise', 'promise'),  # Breaking promises requires promises to exist
        ]
        
        for action, requirement in contradiction_patterns:
            if action in clause_text:
                # Action undermines its own precondition
                return True
        
        return False
    
    def _involves_persons(self, clause: SymClause) -> bool:
        """
        Check if clause involves persons/agents
        
        Args:
            clause: Clause to check
        
        Returns:
            True if involves persons
        """
        person_indicators = [
            'person', 'people', 'human', 'individual', 'agent',
            'user', 'customer', 'client', 'patient', 'student',
            'employee', 'worker', 'citizen', 'member', 'participant',
            'he', 'she', 'they', 'someone', 'anyone', 'everyone'
        ]
        
        clause_text = str(clause).lower()
        if 'source' in clause.meta:
            clause_text += ' ' + clause.meta['source'].lower()
        
        return any(indicator in clause_text for indicator in person_indicators)
    
    def _treats_as_means(self, clause: SymClause) -> bool:
        """
        Check if clause treats persons merely as means
        
        Args:
            clause: Clause to check
        
        Returns:
            True if treats as mere means
        """
        clause_text = str(clause).lower()
        if 'source' in clause.meta:
            clause_text += ' ' + clause.meta['source'].lower()
        
        # Check for instrumentalization language
        instrumentalization_found = any(
            pattern in clause_text 
            for pattern in self.instrumentalization_patterns
        )
        
        if not instrumentalization_found:
            return False
        
        # Check if there's also respect for persons as ends
        respect_indicators = [
            'consent', 'agree', 'voluntary', 'choose', 'decide',
            'autonomy', 'dignity', 'respect', 'rights', 'freedom',
            'well-being', 'benefit', 'interest', 'welfare'
        ]
        
        respect_found = any(
            indicator in clause_text
            for indicator in respect_indicators
        )
        
        # Treats as mere means if instrumentalized without respect
        return instrumentalization_found and not respect_found
    
    def _explain_universalization(self, violations: List[str]) -> str:
        """
        Explain universalization test results
        
        Args:
            violations: List of violating clause IDs
        
        Returns:
            Explanation string
        """
        if not violations:
            return "All clauses pass the universalization test"
        
        if len(violations) == 1:
            return f"Clause {violations[0]} cannot be universalized without contradiction"
        
        return f"{len(violations)} clauses fail universalization: {', '.join(violations)}"
    
    def _explain_means_end(self, violations: List[str]) -> str:
        """
        Explain means-end test results
        
        Args:
            violations: List of violating clause IDs
        
        Returns:
            Explanation string
        """
        if not violations:
            return "All clauses respect persons as ends in themselves"
        
        if len(violations) == 1:
            return f"Clause {violations[0]} treats persons merely as means"
        
        return f"{len(violations)} clauses violate means-end principle: {', '.join(violations)}"
    
    def suggest_reformulation(self, clause: SymClause, test_type: str) -> str:
        """
        Suggest reformulation for failed clause
        
        Args:
            clause: Clause that failed test
            test_type: Type of test failed
        
        Returns:
            Suggestion for reformulation
        """
        if test_type == 'universalization':
            return self._suggest_universalization_fix(clause)
        elif test_type == 'means_end':
            return self._suggest_means_end_fix(clause)
        else:
            return "Review clause for ethical compliance"
    
    def _suggest_universalization_fix(self, clause: SymClause) -> str:
        """
        Suggest fix for universalization failure
        
        Args:
            clause: Failed clause
        
        Returns:
            Suggestion
        """
        clause_text = str(clause).lower()
        
        if any(p in clause_text for p in self.harm_patterns):
            return "Remove harmful actions or add universal benefit conditions"
        
        if any(p in clause_text for p in self.deception_patterns):
            return "Replace deception with transparency and honest communication"
        
        return "Reformulate to be applicable as a universal law"
    
    def _suggest_means_end_fix(self, clause: SymClause) -> str:
        """
        Suggest fix for means-end failure
        
        Args:
            clause: Failed clause
        
        Returns:
            Suggestion
        """
        return "Add consent mechanisms and respect for autonomy"
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get Kant mode statistics
        
        Returns:
            Statistics dictionary
        """
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.stats = {
            'total_tests': 0,
            'universalization_failures': 0,
            'means_end_failures': 0,
            'total_violations': 0
        }