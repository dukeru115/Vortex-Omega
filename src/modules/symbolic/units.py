"""
Unit System Module
==================

Handles unit conversions, dimensional analysis, and canonicalization.

Author: Team Omega
License: CC BY-NC 4.0
"""

import logging
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
from dataclasses import dataclass

from .models import Unit, Quantity, Expression

logger = logging.getLogger(__name__)


class UnitSystem:
    """
    Manages physical units and dimensional analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize unit system

        Args:
            config: Configuration with system type and tolerance
        """
        self.config = config or {}
        self.system = self.config.get("system", "SI")
        self.tolerance = self.config.get("tolerance", 1e-6)

        # Base SI units
        self.base_units = {
            "meter": {"m": 1},
            "kilogram": {"kg": 1},
            "second": {"s": 1},
            "ampere": {"A": 1},
            "kelvin": {"K": 1},
            "mole": {"mol": 1},
            "candela": {"cd": 1},
        }

        # Derived units with their base unit decomposition
        self.derived_units = self._load_derived_units()

        # Conversion factors to SI
        self.conversions = self._load_conversions()

        # Common prefixes
        self.prefixes = {
            "yotta": 1e24,
            "Y": 1e24,
            "zetta": 1e21,
            "Z": 1e21,
            "exa": 1e18,
            "E": 1e18,
            "peta": 1e15,
            "P": 1e15,
            "tera": 1e12,
            "T": 1e12,
            "giga": 1e9,
            "G": 1e9,
            "mega": 1e6,
            "M": 1e6,
            "kilo": 1e3,
            "k": 1e3,
            "hecto": 1e2,
            "h": 1e2,
            "deca": 1e1,
            "da": 1e1,
            "deci": 1e-1,
            "d": 1e-1,
            "centi": 1e-2,
            "c": 1e-2,
            "milli": 1e-3,
            "m": 1e-3,
            "micro": 1e-6,
            "μ": 1e-6,
            "u": 1e-6,
            "nano": 1e-9,
            "n": 1e-9,
            "pico": 1e-12,
            "p": 1e-12,
            "femto": 1e-15,
            "f": 1e-15,
            "atto": 1e-18,
            "a": 1e-18,
            "zepto": 1e-21,
            "z": 1e-21,
            "yocto": 1e-24,
            "y": 1e-24,
        }

        logger.info(f"Unit system initialized with {self.system} units")

    def _load_derived_units(self) -> Dict[str, Dict[str, float]]:
        """Load derived SI units"""
        return {
            # Mechanics
            "newton": {"kg": 1, "m": 1, "s": -2},  # N = kg⋅m/s²
            "joule": {"kg": 1, "m": 2, "s": -2},  # J = kg⋅m²/s²
            "watt": {"kg": 1, "m": 2, "s": -3},  # W = kg⋅m²/s³
            "pascal": {"kg": 1, "m": -1, "s": -2},  # Pa = kg/(m⋅s²)
            "hertz": {"s": -1},  # Hz = 1/s
            # Electromagnetism
            "coulomb": {"A": 1, "s": 1},  # C = A⋅s
            "volt": {"kg": 1, "m": 2, "s": -3, "A": -1},  # V = kg⋅m²/(s³⋅A)
            "farad": {"kg": -1, "m": -2, "s": 4, "A": 2},  # F = s⁴⋅A²/(kg⋅m²)
            "ohm": {"kg": 1, "m": 2, "s": -3, "A": -2},  # Ω = kg⋅m²/(s³⋅A²)
            "siemens": {"kg": -1, "m": -2, "s": 3, "A": 2},  # S = s³⋅A²/(kg⋅m²)
            "weber": {"kg": 1, "m": 2, "s": -2, "A": -1},  # Wb = kg⋅m²/(s²⋅A)
            "tesla": {"kg": 1, "s": -2, "A": -1},  # T = kg/(s²⋅A)
            "henry": {"kg": 1, "m": 2, "s": -2, "A": -2},  # H = kg⋅m²/(s²⋅A²)
            # Other
            "lumen": {"cd": 1},  # lm = cd
            "lux": {"cd": 1, "m": -2},  # lx = cd/m²
            "becquerel": {"s": -1},  # Bq = 1/s
            "gray": {"m": 2, "s": -2},  # Gy = m²/s²
            "sievert": {"m": 2, "s": -2},  # Sv = m²/s²
            "katal": {"mol": 1, "s": -1},  # kat = mol/s
        }

    def _load_conversions(self) -> Dict[str, Tuple[Dict[str, float], float]]:
        """Load conversion factors to SI units"""
        return {
            # Length
            "meter": ({"m": 1}, 1.0),
            "kilometer": ({"m": 1}, 1000.0),
            "centimeter": ({"m": 1}, 0.01),
            "millimeter": ({"m": 1}, 0.001),
            "micrometer": ({"m": 1}, 1e-6),
            "nanometer": ({"m": 1}, 1e-9),
            "angstrom": ({"m": 1}, 1e-10),
            "mile": ({"m": 1}, 1609.344),
            "yard": ({"m": 1}, 0.9144),
            "foot": ({"m": 1}, 0.3048),
            "inch": ({"m": 1}, 0.0254),
            # Mass
            "kilogram": ({"kg": 1}, 1.0),
            "gram": ({"kg": 1}, 0.001),
            "milligram": ({"kg": 1}, 1e-6),
            "microgram": ({"kg": 1}, 1e-9),
            "tonne": ({"kg": 1}, 1000.0),
            "pound": ({"kg": 1}, 0.45359237),
            "ounce": ({"kg": 1}, 0.028349523125),
            # Time
            "second": ({"s": 1}, 1.0),
            "millisecond": ({"s": 1}, 0.001),
            "microsecond": ({"s": 1}, 1e-6),
            "nanosecond": ({"s": 1}, 1e-9),
            "minute": ({"s": 1}, 60.0),
            "hour": ({"s": 1}, 3600.0),
            "day": ({"s": 1}, 86400.0),
            "week": ({"s": 1}, 604800.0),
            "year": ({"s": 1}, 31536000.0),  # 365 days
            # Temperature (special handling needed for non-linear conversions)
            "kelvin": ({"K": 1}, 1.0),
            "celsius": ({"K": 1}, 1.0),  # Offset of 273.15
            "fahrenheit": ({"K": 1}, 5 / 9),  # Offset and scale
            # Energy
            "joule": ({"kg": 1, "m": 2, "s": -2}, 1.0),
            "kilojoule": ({"kg": 1, "m": 2, "s": -2}, 1000.0),
            "calorie": ({"kg": 1, "m": 2, "s": -2}, 4.184),
            "kilocalorie": ({"kg": 1, "m": 2, "s": -2}, 4184.0),
            "electronvolt": ({"kg": 1, "m": 2, "s": -2}, 1.602176634e-19),
            # Pressure
            "pascal": ({"kg": 1, "m": -1, "s": -2}, 1.0),
            "kilopascal": ({"kg": 1, "m": -1, "s": -2}, 1000.0),
            "megapascal": ({"kg": 1, "m": -1, "s": -2}, 1e6),
            "bar": ({"kg": 1, "m": -1, "s": -2}, 1e5),
            "atmosphere": ({"kg": 1, "m": -1, "s": -2}, 101325.0),
            "torr": ({"kg": 1, "m": -1, "s": -2}, 133.322),
            "psi": ({"kg": 1, "m": -1, "s": -2}, 6894.757),
            # Frequency
            "hertz": ({"s": -1}, 1.0),
            "kilohertz": ({"s": -1}, 1000.0),
            "megahertz": ({"s": -1}, 1e6),
            "gigahertz": ({"s": -1}, 1e9),
            # Angle (dimensionless but tracked)
            "radian": ({}, 1.0),
            "degree": ({}, np.pi / 180),
            "arcminute": ({}, np.pi / 10800),
            "arcsecond": ({}, np.pi / 648000),
        }

    def canonicalize_unit(self, unit: Unit) -> Unit:
        """
        Convert unit to canonical SI form

        Args:
            unit: Unit to canonicalize

        Returns:
            Canonicalized unit
        """
        canonical_dims = {}
        total_scale = unit.scale

        for dim, power in unit.dimensions.items():
            # Check if it's a known unit
            if dim in self.conversions:
                base_dims, scale = self.conversions[dim]
                total_scale *= scale**power

                for base_dim, base_power in base_dims.items():
                    if base_dim in canonical_dims:
                        canonical_dims[base_dim] += base_power * power
                    else:
                        canonical_dims[base_dim] = base_power * power

            elif dim in self.derived_units:
                # Expand derived unit
                for base_dim, base_power in self.derived_units[dim].items():
                    if base_dim in canonical_dims:
                        canonical_dims[base_dim] += base_power * power
                    else:
                        canonical_dims[base_dim] = base_power * power

            else:
                # Keep as is (unknown unit)
                canonical_dims[dim] = power

        # Remove zero powers
        canonical_dims = {k: v for k, v in canonical_dims.items() if abs(v) > self.tolerance}

        return Unit(dimensions=canonical_dims, scale=total_scale)

    def canonicalize_quantity(self, quantity: Quantity) -> Quantity:
        """
        Convert quantity to canonical SI form

        Args:
            quantity: Quantity to canonicalize

        Returns:
            Canonicalized quantity
        """
        if quantity.unit:
            canonical_unit = self.canonicalize_unit(quantity.unit)

            # Scale the value
            if quantity.value is not None:
                canonical_value = quantity.value * canonical_unit.scale
            else:
                canonical_value = None

            # Scale bounds if present
            if quantity.bounds:
                canonical_bounds = (
                    quantity.bounds[0] * canonical_unit.scale,
                    quantity.bounds[1] * canonical_unit.scale,
                )
            else:
                canonical_bounds = None

            return Quantity(
                name=quantity.name,
                value=canonical_value,
                unit=Unit(dimensions=canonical_unit.dimensions, scale=1.0),
                bounds=canonical_bounds,
                uncertainty=quantity.uncertainty,
            )

        return quantity

    def canonicalize_expression(self, expr: Expression) -> Expression:
        """
        Canonicalize units in expression

        Args:
            expr: Expression to canonicalize

        Returns:
            Expression with canonicalized units
        """
        # In a full implementation, would parse AST and canonicalize all units
        # For now, just return as is with a flag
        expr.meta = expr.meta if hasattr(expr, "meta") else {}
        expr.meta["canonicalized"] = True
        return expr

    def check_dimensional_consistency(self, unit1: Unit, unit2: Unit) -> bool:
        """
        Check if two units have the same dimensions

        Args:
            unit1: First unit
            unit2: Second unit

        Returns:
            True if dimensionally consistent
        """
        # Canonicalize both units
        canonical1 = self.canonicalize_unit(unit1)
        canonical2 = self.canonicalize_unit(unit2)

        # Compare dimensions
        return canonical1.dimensions == canonical2.dimensions

    def multiply_units(self, unit1: Unit, unit2: Unit) -> Unit:
        """
        Multiply two units

        Args:
            unit1: First unit
            unit2: Second unit

        Returns:
            Product unit
        """
        result_dims = {}

        # Add dimensions from first unit
        for dim, power in unit1.dimensions.items():
            result_dims[dim] = power

        # Add dimensions from second unit
        for dim, power in unit2.dimensions.items():
            if dim in result_dims:
                result_dims[dim] += power
            else:
                result_dims[dim] = power

        # Remove zero powers
        result_dims = {k: v for k, v in result_dims.items() if abs(v) > self.tolerance}

        return Unit(dimensions=result_dims, scale=unit1.scale * unit2.scale)

    def divide_units(self, unit1: Unit, unit2: Unit) -> Unit:
        """
        Divide two units

        Args:
            unit1: Numerator unit
            unit2: Denominator unit

        Returns:
            Quotient unit
        """
        result_dims = {}

        # Add dimensions from numerator
        for dim, power in unit1.dimensions.items():
            result_dims[dim] = power

        # Subtract dimensions from denominator
        for dim, power in unit2.dimensions.items():
            if dim in result_dims:
                result_dims[dim] -= power
            else:
                result_dims[dim] = -power

        # Remove zero powers
        result_dims = {k: v for k, v in result_dims.items() if abs(v) > self.tolerance}

        return Unit(dimensions=result_dims, scale=unit1.scale / unit2.scale)

    def power_unit(self, unit: Unit, exponent: float) -> Unit:
        """
        Raise unit to a power

        Args:
            unit: Base unit
            exponent: Power to raise to

        Returns:
            Result unit
        """
        result_dims = {}

        for dim, power in unit.dimensions.items():
            new_power = power * exponent
            if abs(new_power) > self.tolerance:
                result_dims[dim] = new_power

        return Unit(dimensions=result_dims, scale=unit.scale**exponent)

    def parse_unit_string(self, unit_str: str) -> Unit:
        """
        Parse unit string into Unit object

        Args:
            unit_str: String representation of unit

        Returns:
            Parsed Unit object
        """
        # Handle compound units like kg*m/s^2
        dimensions = {}
        scale = 1.0

        # Split by multiplication and division
        parts = unit_str.replace("·", "*").replace("⋅", "*").split("*")

        for part in parts:
            # Check for division
            if "/" in part:
                num_parts, denom_parts = part.split("/", 1)
                # Process numerator
                self._add_unit_part(num_parts, dimensions, scale, 1)
                # Process denominator
                self._add_unit_part(denom_parts, dimensions, scale, -1)
            else:
                self._add_unit_part(part, dimensions, scale, 1)

        return Unit(dimensions=dimensions, scale=scale)

    def _add_unit_part(self, part: str, dimensions: Dict[str, float], scale: float, sign: int):
        """Helper to add unit part to dimensions"""
        # Extract unit and exponent
        import re

        match = re.match(r"([a-zA-Z]+)(?:\^([-\d]+))?", part.strip())

        if match:
            unit_name, exponent = match.groups()
            exponent = float(exponent) if exponent else 1.0
            exponent *= sign

            # Check for prefix
            for prefix, prefix_scale in self.prefixes.items():
                if unit_name.startswith(prefix):
                    scale *= prefix_scale**exponent
                    unit_name = unit_name[len(prefix) :]
                    break

            # Add to dimensions
            if unit_name in dimensions:
                dimensions[unit_name] += exponent
            else:
                dimensions[unit_name] = exponent

    def format_unit(self, unit: Unit) -> str:
        """
        Format unit as human-readable string

        Args:
            unit: Unit to format

        Returns:
            Formatted string
        """
        if not unit.dimensions:
            return "dimensionless"

        # Separate positive and negative powers
        numerator = []
        denominator = []

        for dim, power in sorted(unit.dimensions.items()):
            if power > 0:
                if power == 1:
                    numerator.append(dim)
                else:
                    numerator.append(f"{dim}^{power}")
            elif power < 0:
                if power == -1:
                    denominator.append(dim)
                else:
                    denominator.append(f"{dim}^{-power}")

        # Build string
        result = "·".join(numerator) if numerator else "1"

        if denominator:
            if len(denominator) == 1:
                result += f"/{denominator[0]}"
            else:
                result += f"/({'.'.join(denominator)})"

        # Add scale if not 1
        if abs(unit.scale - 1.0) > self.tolerance:
            result = f"{unit.scale} × {result}"

        return result
