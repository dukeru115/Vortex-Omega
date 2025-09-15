"""
Symbolic AI Core Module
========================

Main implementation of the Symbolic AI boundary interface for NFCS.
Manages the transformation between discrete symbolic space and continuous field dynamics.

Author: Team Omega
License: CC BY-NC 4.0
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from .models import (
    SymClause,
    SymField,
    VerificationReport,
    ClauseType,
    VerificationStatus,
    Discrepancy,
    Suggestion,
)
from .parser import SymbolicParser
from .units import UnitSystem
from .verifier import SymbolicVerifier
from .discrepancy_gate import DiscrepancyGate
from .kant_mode import KantMode

logger = logging.getLogger(__name__)


class SymbolicAI:
    """
    Main Symbolic AI module implementing the boundary interface
    between the Field and the Other.

    Pipeline: Symbolize → Fieldize → Verify
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Symbolic AI module

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()

        # Initialize components
        self.parser = SymbolicParser(self.config.get("parser", {}))
        self.unit_system = UnitSystem(self.config.get("units", {}))
        self.verifier = SymbolicVerifier(self.config.get("verifier", {}))
        self.discrepancy_gate = DiscrepancyGate(self.config.get("discrepancy", {}))
        self.kant_mode = KantMode(self.config.get("kant", {}))

        # Performance settings
        self.max_clauses_per_cycle = self.config.get("max_clauses_per_cycle", 64)
        self.timeout_ms = self.config.get("timeout_ms", 300)
        self.use_parallel = self.config.get("use_parallel", True)

        # Metrics
        self.metrics = {
            "total_processed": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "avg_processing_time": 0.0,
            "discrepancies_detected": 0,
            "kant_violations": 0,
        }

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4) if self.use_parallel else None

        logger.info("Symbolic AI module initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "max_clauses_per_cycle": 64,
            "timeout_ms": 300,
            "use_parallel": True,
            "parser": {"enable_ner": True, "enable_formula_parser": True, "domain_lexicon": {}},
            "units": {"system": "SI", "tolerance": 1e-6},
            "verifier": {
                "use_wolfram": False,  # Disabled by default
                "use_cas": True,
                "tolerance_abs": 1e-6,
                "tolerance_rel": 1e-3,
            },
            "discrepancy": {"severity_threshold": 0.1, "auto_correct": True},
            "kant": {
                "enable_universalization": True,
                "enable_means_end": True,
                "strict_mode": False,
            },
        }

    async def process(
        self, input_text: str, llm_draft: Optional[str] = None, domain_hint: Optional[str] = None
    ) -> VerificationReport:
        """
        Main processing pipeline: Symbolize → Fieldize → Verify

        Args:
            input_text: Input text to process
            llm_draft: Optional LLM-generated draft for comparison
            domain_hint: Domain hint (physics, finance, bio, etc.)

        Returns:
            VerificationReport with results
        """
        start_time = time.time()

        try:
            # Step 1: Symbolize - Extract and canonicalize
            logger.debug("Starting symbolization")
            clauses, symbol_env = await self._symbolize(input_text, llm_draft, domain_hint)

            if len(clauses) > self.max_clauses_per_cycle:
                logger.warning(
                    f"Truncating clauses from {len(clauses)} to {self.max_clauses_per_cycle}"
                )
                clauses = clauses[: self.max_clauses_per_cycle]

            # Step 2: Fieldize - Group into semantic fields
            logger.debug("Starting fieldization")
            fields = await self._fieldize(clauses, symbol_env)

            # Step 3: Verify - Check consistency and generate report
            logger.debug("Starting verification")
            report = await self._verify(fields)

            # Step 4: Apply discrepancy gate if needed
            if report.discrepancies:
                logger.debug(f"Processing {len(report.discrepancies)} discrepancies")
                await self._apply_discrepancy_gate(report)

            # Step 5: Apply Kant mode if configured
            if (
                self.config["kant"]["enable_universalization"]
                or self.config["kant"]["enable_means_end"]
            ):
                logger.debug("Applying Kant mode checks")
                await self._apply_kant_mode(fields, report)

            # Calculate final confidence
            report.calculate_confidence()

            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(report, processing_time)

            logger.info(f"Symbolic AI processing complete: confidence={report.answer_conf:.2f}")

            return report

        except asyncio.TimeoutError:
            logger.error(f"Processing timeout after {self.timeout_ms}ms")
            return self._create_error_report("Processing timeout")
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return self._create_error_report(str(e))

    async def _symbolize(
        self, input_text: str, llm_draft: Optional[str], domain_hint: Optional[str]
    ) -> Tuple[List[SymClause], Dict]:
        """
        Symbolize phase: Extract and canonicalize symbolic clauses
        """
        # Parse input text
        clauses = self.parser.parse(input_text, domain_hint)

        # Parse LLM draft if provided
        if llm_draft:
            draft_clauses = self.parser.parse(llm_draft, domain_hint)
            # Merge and deduplicate
            clauses = self._merge_clauses(clauses, draft_clauses)

        # Canonicalize units
        for clause in clauses:
            if clause.lhs:
                clause.lhs = self.unit_system.canonicalize_expression(clause.lhs)
            if clause.rhs:
                clause.rhs = self.unit_system.canonicalize_expression(clause.rhs)

        # Build symbol environment
        symbol_env = self._build_symbol_environment(clauses)

        return clauses, symbol_env

    async def _fieldize(self, clauses: List[SymClause], symbol_env: Dict) -> List[SymField]:
        """
        Fieldize phase: Group clauses into semantic fields
        """
        fields = []

        # Group by domain and temporal context
        grouped = self._group_clauses(clauses)

        for group_key, group_clauses in grouped.items():
            field = SymField(
                fid=f"field_{len(fields)}",
                title=group_key,
                clauses=group_clauses,
                domain=self._infer_domain(group_clauses),
            )

            # Generate invariants for the field
            invariants = self._generate_invariants(group_clauses, symbol_env)
            field.invariants = invariants

            # Generate obligations
            obligations = self._generate_obligations(field)
            field.obligations = obligations

            fields.append(field)

        return fields

    async def _verify(self, fields: List[SymField]) -> VerificationReport:
        """
        Verify phase: Check consistency and generate report
        """
        report = VerificationReport(fields=fields)

        # Run verifications in parallel if enabled
        if self.use_parallel and self.executor:
            futures = []

            # Dimensional verification
            futures.append(self.executor.submit(self.verifier.verify_dimensions, fields))

            # Numeric verification
            futures.append(self.executor.submit(self.verifier.verify_numeric, fields))

            # Logical verification
            futures.append(self.executor.submit(self.verifier.verify_logic, fields))

            # Wait for results
            results = [f.result(timeout=self.timeout_ms / 1000) for f in futures]

            dim_result, num_result, logic_result = results

        else:
            # Sequential verification
            dim_result = self.verifier.verify_dimensions(fields)
            num_result = self.verifier.verify_numeric(fields)
            logic_result = self.verifier.verify_logic(fields)

        # Update report with results
        report.dim_status = dim_result["status"]
        report.num_status = num_result["status"]
        report.logic_status = logic_result["status"]

        # Add discrepancies
        for disc in dim_result.get("discrepancies", []):
            report.add_discrepancy(disc)
        for disc in num_result.get("discrepancies", []):
            report.add_discrepancy(disc)
        for disc in logic_result.get("discrepancies", []):
            report.add_discrepancy(disc)

        # Add suggestions
        for sugg in dim_result.get("suggestions", []):
            report.add_suggestion(sugg)
        for sugg in num_result.get("suggestions", []):
            report.add_suggestion(sugg)
        for sugg in logic_result.get("suggestions", []):
            report.add_suggestion(sugg)

        return report

    async def _apply_discrepancy_gate(self, report: VerificationReport):
        """Apply discrepancy gate to resolve conflicts"""
        resolved = self.discrepancy_gate.process(report)

        # Update report with resolutions
        for resolution in resolved:
            report.add_suggestion(
                Suggestion(
                    target=resolution["target"],
                    patch=resolution["patch"],
                    reason=f"Discrepancy resolution: {resolution['reason']}",
                    confidence=resolution.get("confidence", 0.8),
                )
            )

    async def _apply_kant_mode(self, fields: List[SymField], report: VerificationReport):
        """Apply Kantian ethical checks"""
        kant_results = self.kant_mode.test_fields(fields)

        if kant_results["universalization"]["passed"] and kant_results["means_end"]["passed"]:
            report.kant_status = VerificationStatus.OK
        elif kant_results["universalization"]["passed"] or kant_results["means_end"]["passed"]:
            report.kant_status = VerificationStatus.WARNING
        else:
            report.kant_status = VerificationStatus.FAIL

        # Add violations as suggestions
        for violation in kant_results.get("violations", []):
            report.add_suggestion(
                Suggestion(
                    target=violation["clause_id"],
                    patch=violation["suggestion"],
                    reason=f"Kantian ethics violation: {violation['reason']}",
                    confidence=0.9,
                )
            )

    def _merge_clauses(
        self, clauses1: List[SymClause], clauses2: List[SymClause]
    ) -> List[SymClause]:
        """Merge and deduplicate clauses"""
        # Simple deduplication by clause ID
        seen = set()
        merged = []

        for clause in clauses1 + clauses2:
            if clause.cid not in seen:
                seen.add(clause.cid)
                merged.append(clause)

        return merged

    def _build_symbol_environment(self, clauses: List[SymClause]) -> Dict:
        """Build symbol environment from clauses"""
        env = {}

        for clause in clauses:
            # Extract symbols from expressions
            if clause.lhs and hasattr(clause.lhs, "free_symbols"):
                for sym in clause.lhs.free_symbols:
                    if sym not in env:
                        env[sym] = {"type": "variable", "clauses": []}
                    env[sym]["clauses"].append(clause.cid)

            if clause.rhs and hasattr(clause.rhs, "free_symbols"):
                for sym in clause.rhs.free_symbols:
                    if sym not in env:
                        env[sym] = {"type": "variable", "clauses": []}
                    env[sym]["clauses"].append(clause.cid)

        return env

    def _group_clauses(self, clauses: List[SymClause]) -> Dict[str, List[SymClause]]:
        """Group clauses by semantic context"""
        groups = {}

        for clause in clauses:
            # Simple grouping by clause type for now
            # In production, would use more sophisticated clustering
            key = clause.ctype.value
            if key not in groups:
                groups[key] = []
            groups[key].append(clause)

        return groups

    def _infer_domain(self, clauses: List[SymClause]) -> str:
        """Infer domain from clauses"""
        # Simple heuristic based on clause metadata
        domains = []
        for clause in clauses:
            if "domain" in clause.meta:
                domains.append(clause.meta["domain"])

        if domains:
            # Return most common domain
            from collections import Counter

            return Counter(domains).most_common(1)[0][0]

        return "general"

    def _generate_invariants(self, clauses: List[SymClause], symbol_env: Dict) -> List[SymClause]:
        """Generate invariant constraints for a field"""
        invariants = []

        # Example: Conservation laws, boundary conditions, etc.
        # This would be domain-specific in production

        return invariants

    def _generate_obligations(self, field: SymField) -> List[str]:
        """Generate verification obligations for a field"""
        obligations = []

        # Dimensional consistency
        obligations.append("verify_dimensional_consistency")

        # Numeric accuracy
        if any(c.ctype == ClauseType.EQUATION for c in field.clauses):
            obligations.append("verify_numeric_accuracy")

        # Logical consistency
        if len(field.clauses) > 1:
            obligations.append("verify_logical_consistency")

        return obligations

    def _update_metrics(self, report: VerificationReport, processing_time: float):
        """Update internal metrics"""
        self.metrics["total_processed"] += 1

        if report.answer_conf > 0.7:
            self.metrics["successful_verifications"] += 1
        else:
            self.metrics["failed_verifications"] += 1

        self.metrics["discrepancies_detected"] += len(report.discrepancies)

        if report.kant_status == VerificationStatus.FAIL:
            self.metrics["kant_violations"] += 1

        # Update average processing time
        n = self.metrics["total_processed"]
        prev_avg = self.metrics["avg_processing_time"]
        self.metrics["avg_processing_time"] = (prev_avg * (n - 1) + processing_time) / n

    def _create_error_report(self, error_msg: str) -> VerificationReport:
        """Create error report"""
        report = VerificationReport(fields=[])
        report.dim_status = VerificationStatus.FAIL
        report.num_status = VerificationStatus.FAIL
        report.logic_status = VerificationStatus.FAIL
        report.answer_conf = 0.0
        report.add_suggestion(
            Suggestion(
                target="system", patch="", reason=f"System error: {error_msg}", confidence=0.0
            )
        )
        return report

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()

    async def shutdown(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("Symbolic AI module shutdown complete")
