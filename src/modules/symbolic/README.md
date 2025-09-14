# Symbolic AI Module

## Overview

The Symbolic AI module implements the boundary interface between the discrete symbolic space (the "Other") and the continuous field dynamics of the Neural Field Control System (NFCS). This module manages the transformation of text and LLM outputs into typed symbolic statements, verifies their consistency, and provides feedback mechanisms for correction.

## Architecture

### Pipeline: Symbolize → Fieldize → Verify

```
Input Text/LLM Draft
        ↓
    [Symbolize]
        ↓
  List<SymClause>
        ↓
    [Fieldize]
        ↓
  List<SymField>
        ↓
     [Verify]
        ↓
VerificationReport
```

## Components

### 1. **Symbolic Core** (`symbolic_core.py`)
- Main orchestrator for the symbolic AI pipeline
- Manages the complete processing flow
- Handles parallel processing and timeouts
- Maintains performance metrics

### 2. **Parser** (`parser.py`)
- Extracts symbolic clauses from natural language text
- Parses mathematical formulas and equations
- Performs Named Entity Recognition (NER)
- Handles unit extraction and normalization

### 3. **Unit System** (`units.py`)
- Manages physical units and dimensional analysis
- Converts between different unit systems (SI, Imperial, etc.)
- Performs dimensional consistency checking
- Canonicalizes units to SI base units

### 4. **Verifier** (`verifier.py`)
- **Dimensional Verification**: Checks unit consistency
- **Numeric Verification**: Validates numerical accuracy
- **Logical Verification**: Ensures logical consistency
- **Temporal Verification**: Validates causal and temporal ordering

### 5. **Discrepancy Gate** (`discrepancy_gate.py`)
- Detects discrepancies between LLM outputs and verified facts
- Implements strict validation gates for numbers and units
- Provides automatic correction for minor discrepancies
- Generates clarification questions for ambiguous cases

### 6. **Kant Mode** (`kant_mode.py`)
- Tests for Kantian ethical principles:
  - **Universalization Test**: Can the action be a universal law?
  - **Means-End Test**: Are persons treated as ends in themselves?
- Detects harmful, deceptive, or exploitative patterns
- Provides reformulation suggestions for ethical compliance

### 7. **Models** (`models.py`)
- Data structures for symbolic representations:
  - `SymClause`: Typed symbolic statements
  - `SymField`: Semantic groupings of clauses
  - `Unit`: Physical unit representation
  - `Quantity`: Physical quantities with units
  - `Expression`: Mathematical expressions
  - `VerificationReport`: Complete verification results

## Usage

### Basic Example

```python
from src.modules.symbolic import SymbolicAI

# Initialize the module
symbolic_ai = SymbolicAI()

# Process input text
input_text = "The pendulum period T = 2π√(L/g). With L=1m and g=9.81m/s², T≈2.1s"
report = await symbolic_ai.process(input_text)

# Check results
print(f"Verification status: {report.dim_status}, {report.num_status}, {report.logic_status}")
print(f"Confidence: {report.answer_conf:.2f}")

# Handle discrepancies
for discrepancy in report.discrepancies:
    print(f"Discrepancy in {discrepancy.cid}: {discrepancy.field}")
    
# Review suggestions
for suggestion in report.suggestions:
    print(f"Suggestion: {suggestion.patch} ({suggestion.reason})")
```

### Advanced Usage with LLM Draft

```python
# Compare LLM output with ground truth
llm_draft = "The calculated value is approximately 2.1 seconds"
domain_hint = "physics"

report = await symbolic_ai.process(
    input_text=input_text,
    llm_draft=llm_draft,
    domain_hint=domain_hint
)

# Kant mode ethical testing
if report.kant_status != VerificationStatus.OK:
    print("Ethical concerns detected - review required")
```

## Configuration

```python
config = {
    'max_clauses_per_cycle': 64,        # Max clauses to process at once
    'timeout_ms': 300,                  # Processing timeout
    'use_parallel': True,                # Enable parallel processing
    
    'parser': {
        'enable_ner': True,              # Named entity recognition
        'enable_formula_parser': True,    # Mathematical formula parsing
    },
    
    'units': {
        'system': 'SI',                  # Unit system (SI, CGS, Imperial)
        'tolerance': 1e-6,               # Numerical tolerance
    },
    
    'verifier': {
        'use_wolfram': False,            # Wolfram Alpha integration
        'use_cas': True,                 # Computer Algebra System
        'tolerance_abs': 1e-6,           # Absolute tolerance
        'tolerance_rel': 1e-3,           # Relative tolerance
    },
    
    'discrepancy': {
        'severity_threshold': 0.1,       # Auto-correction threshold
        'auto_correct': True,            # Enable auto-correction
    },
    
    'kant': {
        'enable_universalization': True,  # Universalization test
        'enable_means_end': True,        # Means-end test
        'strict_mode': False,            # Strict ethical checking
    }
}

symbolic_ai = SymbolicAI(config)
```

## Integration with NFCS

### Connection to ESC Module

The Symbolic AI module provides structured input for the Echo-Semantic Converter:

```python
# Symbolic AI output feeds into ESC
clauses, env = symbolic_ai._symbolize(text)
semantic_signal = esc_module.process_symbolic(clauses, env)
```

### Feedback to Kuramoto Synchronization

Symbolic constraints influence module coupling:

```python
# Extract constraints for Kuramoto coupling
constraints = symbolic_ai._extract_constraints(field)
kuramoto_module.update_coupling_from_constraints(constraints)
```

### Constitutional Compliance

Integration with the Constitutional module for safety:

```python
# Verify constitutional compliance
if report.kant_status == VerificationStatus.FAIL:
    constitution_module.trigger_safety_protocol()
```

## API Reference

### Main Functions

#### `symbolize(text, draft, domain_hint) -> (List[SymClause], Dict)`
Parse text into typed symbolic clauses.

#### `fieldize(clauses, env) -> List[SymField]`
Group clauses into semantic fields with invariants.

#### `verify(fields) -> VerificationReport`
Perform comprehensive verification checks.

### Helper Functions

#### `numeric_check(expr, tol_abs, tol_rel) -> Dict`
Check numeric consistency with specified tolerances.

#### `kant_check(clauses) -> Dict`
Perform Kantian ethical testing.

## Performance Metrics

The module tracks performance metrics:

```python
metrics = symbolic_ai.get_metrics()
print(f"Total processed: {metrics['total_processed']}")
print(f"Success rate: {metrics['successful_verifications'] / metrics['total_processed']:.2%}")
print(f"Discrepancies detected: {metrics['discrepancies_detected']}")
print(f"Kant violations: {metrics['kant_violations']}")
```

## Testing

### Unit Tests

```bash
pytest tests/test_symbolic/ -v
```

### Integration Tests

```bash
python tests/test_symbolic_integration.py
```

### Performance Benchmarks

```bash
python benchmarks/symbolic_performance.py
```

## Technical Specifications

Based on NFCS v2.4.3 specifications:

- **Clause Types**: Equation, Inequality, Definition, Assumption, Fact, Claim, Goal, Constraint
- **Verification Levels**: Dimensional, Numeric, Logical, Temporal, Ethical
- **Confidence Scoring**: Weighted average of verification results
- **Parallel Processing**: Thread pool executor for concurrent verification
- **Caching**: CAS results cached for performance

## Future Enhancements

- [ ] Wolfram Alpha integration for advanced CAS
- [ ] SAT/SMT solver integration (Z3)
- [ ] Extended domain lexicons
- [ ] Machine learning for pattern recognition
- [ ] Real-time streaming processing
- [ ] Distributed verification across nodes

## License

CC BY-NC 4.0 (Attribution — NonCommercial)

## Authors

**Team Omega**
- Mathematical Framework & Verification Logic
- Symbolic-Field Interface Design
- Ethical Testing Implementation

---

*Part of the Vortex-Omega Neural Field Control System v2.4.3*