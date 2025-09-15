"""
Symbolic Parser Module
======================

Parses text and formulas into symbolic clauses.
Handles NER, formula parsing, and term extraction.

Author: Team Omega
License: CC BY-NC 4.0
"""

import re
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import json

from .models import SymClause, Expression, Term, Quantity, Unit, ClauseType, OperatorType, TermKind

logger = logging.getLogger(__name__)


class SymbolicParser:
    """
    Parser for extracting symbolic clauses from text
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize parser

        Args:
            config: Parser configuration
        """
        self.config = config or {}

        # Patterns for different elements
        self.patterns = self._compile_patterns()

        # Domain lexicons
        self.domain_lexicon = self.config.get("domain_lexicon", {})

        # Synonym mappings
        self.synonyms = self._load_synonyms()

        # Unit abbreviations
        self.unit_abbrev = self._load_unit_abbreviations()

        logger.info("Symbolic parser initialized")

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for parsing"""
        return {
            # Numbers with optional units
            "number": re.compile(
                r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([a-zA-Z°℃℉%‰]+(?:/[a-zA-Z]+)?)?"
            ),
            # Equations and inequalities
            "equation": re.compile(r"([^=<>≤≥≈∈]+)\s*(=|<|>|<=|>=|≤|≥|≈|∈)\s*([^=<>≤≥≈∈,;\.]+)"),
            # Variables and symbols
            "variable": re.compile(r"\b([a-zA-Z_]\w*)\b"),
            # Functions
            "function": re.compile(r"([a-zA-Z_]\w*)\s*\(([^)]*)\)"),
            # Intervals/ranges
            "interval": re.compile(r"\[([^,\]]+),\s*([^,\]]+)\]"),
            # Mathematical operators
            "operator": re.compile(r"(\+|-|\*|/|\^|\*\*|√|∫|∑|∏|∂)"),
            # Greek letters
            "greek": re.compile(
                r"(α|β|γ|δ|ε|ζ|η|θ|ι|κ|λ|μ|ν|ξ|ο|π|ρ|σ|τ|υ|φ|χ|ψ|ω|Γ|Δ|Θ|Λ|Ξ|Π|Σ|Φ|Ψ|Ω)"
            ),
        }

    def _load_synonyms(self) -> Dict[str, str]:
        """Load synonym mappings"""
        return {
            "velocity": "speed",
            "mass": "weight",  # Context-dependent
            "temperature": "temp",
            "pressure": "P",
            "volume": "V",
            "energy": "E",
            "force": "F",
            "acceleration": "a",
            "time": "t",
            "distance": "d",
            "length": "L",
            "height": "h",
            "width": "w",
            "radius": "r",
            "diameter": "d",
            "frequency": "f",
            "wavelength": "λ",
            "period": "T",
        }

    def _load_unit_abbreviations(self) -> Dict[str, str]:
        """Load unit abbreviation mappings"""
        return {
            # Length
            "m": "meter",
            "km": "kilometer",
            "cm": "centimeter",
            "mm": "millimeter",
            "ft": "foot",
            "in": "inch",
            "mi": "mile",
            # Time
            "s": "second",
            "ms": "millisecond",
            "min": "minute",
            "h": "hour",
            "hr": "hour",
            "d": "day",
            "y": "year",
            # Mass
            "kg": "kilogram",
            "g": "gram",
            "mg": "milligram",
            "lb": "pound",
            "oz": "ounce",
            # Temperature
            "K": "kelvin",
            "C": "celsius",
            "°C": "celsius",
            "F": "fahrenheit",
            "°F": "fahrenheit",
            # Energy
            "J": "joule",
            "kJ": "kilojoule",
            "cal": "calorie",
            "kcal": "kilocalorie",
            "eV": "electronvolt",
            # Force
            "N": "newton",
            "kN": "kilonewton",
            "lbf": "pound-force",
            # Pressure
            "Pa": "pascal",
            "kPa": "kilopascal",
            "MPa": "megapascal",
            "bar": "bar",
            "atm": "atmosphere",
            "psi": "pound-per-square-inch",
            # Frequency
            "Hz": "hertz",
            "kHz": "kilohertz",
            "MHz": "megahertz",
            "GHz": "gigahertz",
        }

    def parse(self, text: str, domain_hint: Optional[str] = None) -> List[SymClause]:
        """
        Parse text into symbolic clauses

        Args:
            text: Input text to parse
            domain_hint: Optional domain hint

        Returns:
            List of symbolic clauses
        """
        clauses = []
        clause_id = 0

        # Split text into sentences/statements
        statements = self._split_statements(text)

        for statement in statements:
            # Try to parse as equation/inequality
            equation_clauses = self._parse_equations(statement, clause_id)
            if equation_clauses:
                clauses.extend(equation_clauses)
                clause_id += len(equation_clauses)
                continue

            # Try to parse as definition
            def_clause = self._parse_definition(statement, clause_id)
            if def_clause:
                clauses.append(def_clause)
                clause_id += 1
                continue

            # Try to parse as fact/claim
            fact_clause = self._parse_fact(statement, clause_id, domain_hint)
            if fact_clause:
                clauses.append(fact_clause)
                clause_id += 1

        logger.debug(f"Parsed {len(clauses)} clauses from text")
        return clauses

    def _split_statements(self, text: str) -> List[str]:
        """Split text into individual statements"""
        # Split by common delimiters
        statements = re.split(r"[.;]\s+|\n", text)

        # Filter empty statements
        statements = [s.strip() for s in statements if s.strip()]

        return statements

    def _parse_equations(self, statement: str, start_id: int) -> List[SymClause]:
        """Parse equations and inequalities from statement"""
        clauses = []

        matches = self.patterns["equation"].findall(statement)

        for i, match in enumerate(matches):
            lhs_str, op_str, rhs_str = match

            # Parse expressions
            lhs_expr = self._parse_expression(lhs_str.strip())
            rhs_expr = self._parse_expression(rhs_str.strip())

            # Determine operator type
            op_map = {
                "=": OperatorType.EQUAL,
                "<": OperatorType.LESS,
                ">": OperatorType.GREATER,
                "<=": OperatorType.LESS_EQUAL,
                "≤": OperatorType.LESS_EQUAL,
                ">=": OperatorType.GREATER_EQUAL,
                "≥": OperatorType.GREATER_EQUAL,
                "≈": OperatorType.APPROX,
                "∈": OperatorType.IN,
            }

            op = op_map.get(op_str, OperatorType.EQUAL)

            # Determine clause type
            if op == OperatorType.EQUAL:
                ctype = ClauseType.EQUATION
            elif op in [
                OperatorType.LESS,
                OperatorType.GREATER,
                OperatorType.LESS_EQUAL,
                OperatorType.GREATER_EQUAL,
            ]:
                ctype = ClauseType.INEQUALITY
            else:
                ctype = ClauseType.CONSTRAINT

            clause = SymClause(
                cid=f"c{start_id + i}",
                ctype=ctype,
                lhs=lhs_expr,
                rhs=rhs_expr,
                op=op,
                meta={"source": statement},
            )

            clauses.append(clause)

        return clauses

    def _parse_definition(self, statement: str, clause_id: int) -> Optional[SymClause]:
        """Parse definition from statement"""
        # Look for patterns like "X is defined as Y" or "Let X = Y"
        def_patterns = [
            r"(?:Let|Define|Given)\s+([^=]+)\s*=\s*(.+)",
            r"([^:]+):\s*(.+)",
            r"([^,]+)\s+is\s+defined\s+as\s+(.+)",
            r"([^,]+)\s+is\s+(.+)",
        ]

        for pattern in def_patterns:
            match = re.match(pattern, statement, re.IGNORECASE)
            if match:
                term_str, def_str = match.groups()

                term = self._parse_term(term_str.strip())
                definition = self._parse_expression(def_str.strip())

                return SymClause(
                    cid=f"c{clause_id}",
                    ctype=ClauseType.DEFINITION,
                    lhs=Expression(ast=term, free_symbols=[term_str.strip()]),
                    rhs=definition,
                    op=OperatorType.EQUAL,
                    meta={"source": statement},
                )

        return None

    def _parse_fact(
        self, statement: str, clause_id: int, domain_hint: Optional[str]
    ) -> Optional[SymClause]:
        """Parse fact or claim from statement"""
        # Extract quantities and relationships
        quantities = self._extract_quantities(statement)

        if quantities:
            # Create fact clause with extracted information
            return SymClause(
                cid=f"c{clause_id}",
                ctype=ClauseType.FACT,
                meta={
                    "source": statement,
                    "quantities": [q.__dict__ for q in quantities],
                    "domain": domain_hint or "general",
                },
            )

        # Check if it's a claim or assumption
        claim_keywords = ["assume", "suppose", "if", "when", "given that"]
        is_claim = any(keyword in statement.lower() for keyword in claim_keywords)

        if is_claim:
            return SymClause(
                cid=f"c{clause_id}",
                ctype=ClauseType.ASSUMPTION,
                meta={"source": statement, "domain": domain_hint or "general"},
            )

        return None

    def _parse_expression(self, expr_str: str) -> Expression:
        """Parse string into Expression object"""
        # Extract free symbols
        variables = self.patterns["variable"].findall(expr_str)

        # Filter out function names and keywords
        keywords = {"sin", "cos", "tan", "log", "ln", "exp", "sqrt", "abs"}
        free_symbols = [v for v in variables if v not in keywords]

        # Normalize symbols using synonyms
        normalized_symbols = []
        for sym in free_symbols:
            normalized = self.synonyms.get(sym.lower(), sym)
            normalized_symbols.append(normalized)

        # Check if expression is purely numeric
        is_numeric = bool(re.match(r"^[\d\.\+\-\*\/\^\s]+$", expr_str))

        return Expression(
            ast=expr_str,  # In production, would parse to actual AST
            free_symbols=normalized_symbols,
            is_numeric=is_numeric,
        )

    def _parse_term(self, term_str: str) -> Term:
        """Parse string into Term object"""
        # Determine term kind
        if term_str[0].isupper():
            kind = TermKind.ENTITY
        elif "(" in term_str:
            kind = TermKind.PREDICATE
        elif re.match(r"^\d+(\.\d+)?$", term_str):
            kind = TermKind.CONSTANT
        else:
            kind = TermKind.VARIABLE

        # Normalize using synonyms
        normalized = self.synonyms.get(term_str.lower(), term_str)

        return Term(symbol=normalized, kind=kind)

    def _extract_quantities(self, text: str) -> List[Quantity]:
        """Extract quantities with values and units from text"""
        quantities = []

        # Find all number-unit pairs
        matches = self.patterns["number"].findall(text)

        for value_str, unit_str in matches:
            try:
                value = float(value_str)

                # Parse unit if present
                unit = None
                if unit_str:
                    unit = self._parse_unit(unit_str)

                # Try to find associated variable name
                # Look for patterns like "T = 2.1s" or "period is 2.1s"
                context = text[: text.find(value_str)]
                var_match = self.patterns["variable"].findall(context)

                if var_match:
                    name = var_match[-1]  # Use last variable before the number
                    name = self.synonyms.get(name.lower(), name)
                else:
                    name = f"quantity_{len(quantities)}"

                quantity = Quantity(name=name, value=value, unit=unit)

                quantities.append(quantity)

            except ValueError:
                continue

        return quantities

    def _parse_unit(self, unit_str: str) -> Unit:
        """Parse unit string into Unit object"""
        # Handle compound units like m/s or kg*m/s^2
        dimensions = {}

        # Split by multiplication and division
        parts = re.split(r"[*/]", unit_str)

        for i, part in enumerate(parts):
            # Check if it's in denominator (after /)
            is_denominator = "/" in unit_str and unit_str.split("/").index(part) > 0

            # Extract base unit and power
            match = re.match(r"([a-zA-Z°℃℉%‰]+)(?:\^([-\d]+))?", part)
            if match:
                base_unit, power = match.groups()
                power = int(power) if power else 1

                if is_denominator:
                    power = -power

                # Map to standard unit
                standard = self.unit_abbrev.get(base_unit, base_unit)

                # Add to dimensions
                if standard in dimensions:
                    dimensions[standard] += power
                else:
                    dimensions[standard] = power

        return Unit(dimensions=dimensions)

    def parse_formula(self, formula: str) -> Optional[Expression]:
        """
        Parse mathematical formula into Expression

        Args:
            formula: Mathematical formula string

        Returns:
            Parsed Expression or None if parsing fails
        """
        try:
            # In production, would use proper formula parser (SymPy, etc.)
            # For now, just extract basic structure

            # Extract variables
            variables = self.patterns["variable"].findall(formula)

            # Extract operators
            operators = self.patterns["operator"].findall(formula)

            # Extract functions
            functions = self.patterns["function"].findall(formula)

            # Build expression
            expr = Expression(ast=formula, free_symbols=variables, is_numeric=False)

            return expr

        except Exception as e:
            logger.error(f"Failed to parse formula: {e}")
            return None
