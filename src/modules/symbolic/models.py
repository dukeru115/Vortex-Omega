"""
Data Models for Symbolic AI Module
===================================

Pydantic models for symbolic representations and verification.

Author: Team Omega
License: CC BY-NC 4.0
"""

from typing import Dict, List, Optional, Literal, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ClauseType(str, Enum):
    """Types of symbolic clauses"""
    EQUATION = 'Equation'
    INEQUALITY = 'Inequality'
    DEFINITION = 'Def'
    ASSUMPTION = 'Assumption'
    FACT = 'Fact'
    CLAIM = 'Claim'
    GOAL = 'Goal'
    CONSTRAINT = 'Constraint'


class OperatorType(str, Enum):
    """Mathematical operators"""
    EQUAL = '='
    LESS = '<'
    LESS_EQUAL = '<='
    GREATER = '>'
    GREATER_EQUAL = '>='
    IN = '∈'
    APPROX = '≈'


class TermKind(str, Enum):
    """Types of symbolic terms"""
    ENTITY = 'Entity'
    PREDICATE = 'Predicate'
    VARIABLE = 'Var'
    CONSTANT = 'Const'


class VerificationStatus(str, Enum):
    """Verification status levels"""
    OK = 'ok'
    WARNING = 'warn'
    FAIL = 'fail'
    NOT_APPLICABLE = 'n/a'


@dataclass
class Unit:
    """Physical unit representation"""
    dimensions: Dict[str, float]  # {'m': 1, 's': -1} for m/s
    scale: float = 1.0  # Conversion factor to SI base units
    
    def __str__(self) -> str:
        """String representation of unit"""
        parts = []
        for dim, power in sorted(self.dimensions.items()):
            if power == 1:
                parts.append(dim)
            elif power != 0:
                parts.append(f"{dim}^{power}")
        return '·'.join(parts) if parts else '1'
    
    def is_dimensionless(self) -> bool:
        """Check if unit is dimensionless"""
        return all(p == 0 for p in self.dimensions.values())
    
    def is_compatible(self, other: 'Unit') -> bool:
        """Check if units have same dimensions"""
        return self.dimensions == other.dimensions


@dataclass
class Quantity:
    """Physical quantity with value and units"""
    name: str
    value: Optional[float] = None
    unit: Optional[Unit] = None
    bounds: Optional[Tuple[float, float]] = None
    uncertainty: Optional[float] = None
    
    def __str__(self) -> str:
        """String representation"""
        s = self.name
        if self.value is not None:
            s += f" = {self.value}"
        if self.unit:
            s += f" {self.unit}"
        if self.bounds:
            s += f" ∈ [{self.bounds[0]}, {self.bounds[1]}]"
        return s


@dataclass
class Term:
    """Symbolic term representation"""
    symbol: str  # Canonical name
    kind: TermKind
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash((self.symbol, self.kind))


@dataclass
class Expression:
    """Symbolic expression with AST"""
    ast: Any  # SymPy expression or custom AST
    free_symbols: List[str] = field(default_factory=list)
    is_numeric: bool = False
    
    def evaluate(self, bindings: Dict[str, float]) -> Optional[float]:
        """Evaluate expression with variable bindings"""
        try:
            # Placeholder for actual evaluation logic
            # Would use SymPy or custom evaluator
            return None
        except:
            return None


@dataclass
class SymClause:
    """Symbolic clause representation"""
    cid: str  # Clause ID
    ctype: ClauseType
    lhs: Optional[Expression] = None
    rhs: Optional[Expression] = None
    op: Optional[OperatorType] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    units_ok: Optional[bool] = None
    numeric_ok: Optional[bool] = None
    logic_ok: Optional[bool] = None
    
    def __str__(self) -> str:
        """String representation"""
        if self.lhs and self.rhs and self.op:
            return f"{self.ctype}[{self.cid}]: {self.lhs} {self.op} {self.rhs}"
        return f"{self.ctype}[{self.cid}]"
    
    def is_verified(self) -> bool:
        """Check if clause passed all verifications"""
        checks = [self.units_ok, self.numeric_ok, self.logic_ok]
        return all(c is True for c in checks if c is not None)


@dataclass
class SymField:
    """Semantic field grouping related clauses"""
    fid: str  # Field ID
    title: str
    clauses: List[SymClause] = field(default_factory=list)
    invariants: List[SymClause] = field(default_factory=list)
    obligations: List[str] = field(default_factory=list)
    domain: str = 'general'  # physics, finance, bio, etc.
    
    def add_clause(self, clause: SymClause):
        """Add clause to field"""
        self.clauses.append(clause)
    
    def add_invariant(self, invariant: SymClause):
        """Add invariant constraint"""
        self.invariants.append(invariant)
    
    def get_all_clauses(self) -> List[SymClause]:
        """Get all clauses including invariants"""
        return self.clauses + self.invariants


@dataclass
class Discrepancy:
    """Discrepancy between expected and actual values"""
    cid: str  # Clause ID where discrepancy found
    field: str  # Field of discrepancy (numeric, units, logic)
    expected: Any
    actual: Any
    tolerance: Optional[float] = None
    severity: Literal['low', 'medium', 'high'] = 'medium'
    
    def __str__(self) -> str:
        return f"Discrepancy in {self.cid}: {self.field} - expected {self.expected}, got {self.actual}"


@dataclass
class Suggestion:
    """Suggested correction or clarification"""
    target: str  # What to fix (clause ID or field)
    patch: str  # Suggested replacement
    reason: str  # Why this suggestion
    confidence: float = 0.5
    
    def __str__(self) -> str:
        return f"Suggest {self.target}: {self.patch} ({self.reason}, conf={self.confidence})"


@dataclass
class VerificationReport:
    """Complete verification report"""
    fields: List[SymField]
    dim_status: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    num_status: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    logic_status: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    kant_status: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    tool_conf: float = 0.0  # Confidence in external tools
    answer_conf: float = 0.0  # Overall answer confidence
    discrepancies: List[Discrepancy] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    
    def add_discrepancy(self, discrepancy: Discrepancy):
        """Add discrepancy to report"""
        self.discrepancies.append(discrepancy)
        # Update status based on severity
        if discrepancy.severity == 'high':
            if discrepancy.field == 'numeric':
                self.num_status = VerificationStatus.FAIL
            elif discrepancy.field == 'units':
                self.dim_status = VerificationStatus.FAIL
            elif discrepancy.field == 'logic':
                self.logic_status = VerificationStatus.FAIL
    
    def add_suggestion(self, suggestion: Suggestion):
        """Add suggestion to report"""
        self.suggestions.append(suggestion)
    
    def calculate_confidence(self) -> float:
        """Calculate overall confidence score"""
        scores = []
        
        # Status scores
        status_map = {
            VerificationStatus.OK: 1.0,
            VerificationStatus.WARNING: 0.7,
            VerificationStatus.FAIL: 0.3,
            VerificationStatus.NOT_APPLICABLE: None
        }
        
        for status in [self.dim_status, self.num_status, self.logic_status]:
            score = status_map.get(status)
            if score is not None:
                scores.append(score)
        
        # Discrepancy penalty
        if self.discrepancies:
            penalty = len(self.discrepancies) * 0.1
            scores.append(max(0, 1.0 - penalty))
        
        # Tool confidence weight
        if self.tool_conf > 0:
            scores.append(self.tool_conf)
        
        # Calculate weighted average
        if scores:
            self.answer_conf = sum(scores) / len(scores)
        else:
            self.answer_conf = 0.5
        
        return self.answer_conf
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'dim_status': self.dim_status.value,
            'num_status': self.num_status.value,
            'logic_status': self.logic_status.value,
            'kant_status': self.kant_status.value,
            'tool_conf': self.tool_conf,
            'answer_conf': self.answer_conf,
            'discrepancies': [
                {
                    'cid': d.cid,
                    'field': d.field,
                    'expected': str(d.expected),
                    'actual': str(d.actual),
                    'severity': d.severity
                }
                for d in self.discrepancies
            ],
            'suggestions': [
                {
                    'target': s.target,
                    'patch': s.patch,
                    'reason': s.reason,
                    'confidence': s.confidence
                }
                for s in self.suggestions
            ],
            'fields_count': len(self.fields),
            'clauses_count': sum(len(f.clauses) for f in self.fields)
        }


@dataclass
class KantTestResult:
    """Result of Kantian ethical test"""
    test_type: Literal['universalization', 'means_end']
    passed: bool
    explanation: str
    violations: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"Kant {self.test_type}: {status} - {self.explanation}"