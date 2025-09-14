# Symbolic AI API Documentation

## REST API Specification for Kamil Gadeev Symbolic AI Module

### Base URL
```
http://localhost:8000/api/v1/symbolic/
```

---

## 📋 **Core Endpoints**

### **1. Complete Verification Pipeline**

#### `POST /verify`
Complete symbolic analysis with verification report.

**Request:**
```json
{
  "text": "The momentum p = m * v must be conserved in all interactions",
  "domain_hint": "physics",
  "options": {
    "use_wolfram": true,
    "use_z3": true,
    "kant_mode": true,
    "tolerance": {
      "abs": 1e-6,
      "rel": 1e-3
    }
  }
}
```

**Response:**
```json
{
  "status": "success",
  "report": {
    "fields": [...],
    "dim_status": "ok",
    "num_status": "ok", 
    "logic_status": "ok",
    "kant_status": "ok",
    "tool_conf": 0.95,
    "answer_conf": 0.92,
    "discrepancies": [],
    "suggestions": [],
    "processing_time_ms": 156.7,
    "slo_compliant": true
  }
}
```

---

### **2. Pipeline Components**

#### `POST /symbolize`
Extract and canonize symbolic representations.

**Request:**
```json
{
  "text": "Energy E = m * c² where m is mass",
  "domain_hint": "physics"
}
```

**Response:**
```json
{
  "clauses": [
    {
      "cid": "c1",
      "ctype": "Equation",
      "lhs": {"ast": "E", "free_symbols": ["E"]},
      "rhs": {"ast": "m*c**2", "free_symbols": ["m", "c"]},
      "op": "=",
      "meta": {"line": 0, "confidence": 0.98}
    }
  ],
  "environment": {
    "symbols": {"E": "Energy", "m": "Mass", "c": "SpeedOfLight"},
    "domain_context": "physics"
  },
  "processing_time_ms": 45.2
}
```

#### `POST /fieldize`  
Group clauses into semantic fields.

**Request:**
```json
{
  "clauses": [...],
  "environment": {...}
}
```

**Response:**
```json
{
  "fields": [
    {
      "fid": "field_1", 
      "title": "Energy-Mass Equivalence",
      "clauses": [...],
      "invariants": [
        "dimensional_consistency(E, [mass*length^2*time^-2])",
        "physical_constraint(c = 299792458)"
      ],
      "obligations": ["verify_units", "check_constants"],
      "domain": "physics",
      "confidence": 0.96
    }
  ],
  "processing_time_ms": 67.8
}
```

---

### **3. Specialized Validation**

#### `POST /check/dimensional`
Dimensional analysis validation.

**Request:**
```json
{
  "expressions": [
    {"expr": "F = m * a", "context": "Newton's second law"},
    {"expr": "E = (1/2) * m * v^2", "context": "Kinetic energy"}
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "expr": "F = m * a",
      "dimensional_consistent": true,
      "lhs_dimensions": {"mass": 1, "length": 1, "time": -2},
      "rhs_dimensions": {"mass": 1, "length": 1, "time": -2},
      "confidence": 0.99
    }
  ],
  "overall_status": "ok"
}
```

#### `POST /check/numeric`
Numerical accuracy validation with CAS.

**Request:**
```json
{
  "expressions": [
    {"expr": "sin(π/2)", "expected": 1.0},
    {"expr": "e^(iπ) + 1", "expected": 0.0}
  ],
  "tolerance": {"abs": 1e-10, "rel": 1e-10}
}
```

**Response:**
```json
{
  "results": [
    {
      "expr": "sin(π/2)",
      "computed": 1.0,
      "expected": 1.0,
      "error": 0.0,
      "within_tolerance": true,
      "cas_source": "wolfram"
    }
  ],
  "summary": {
    "passed": 2,
    "failed": 0,
    "total": 2
  }
}
```

#### `POST /check/kant`
Kantian ethical validation.

**Request:**
```json
{
  "text": "Use prisoners for medical experiments to advance science",
  "mode": "strict"
}
```

**Response:**
```json
{
  "kant_status": "fail",
  "violations": [
    {
      "principle": "means_end",
      "description": "Treating persons merely as means",
      "severity": "high",
      "text_span": [5, 45]
    }
  ],
  "suggestions": [
    {
      "alternative": "Conduct voluntary medical studies with informed consent",
      "principle": "respect_for_persons"
    }
  ]
}
```

---

### **4. Integration Endpoints**

#### `POST /integration/discrepancy-gate`
Discrepancy gate validation integration.

**Request:**
```json
{
  "field_state": [...],  // Neural field data
  "discrepancy_measure": 0.15,
  "system_context": {
    "timestamp": 1694712000,
    "field_type": "semantic_oscillator"
  }
}
```

**Response:**
```json
{
  "symbolic_validation_passed": true,
  "confidence_score": 0.87,
  "dimensional_accuracy": 0.99,
  "ethical_compliant": true,
  "processing_time_ms": 189.4,
  "recommended_action": "continue_normal",
  "field_modulations": {
    "spatial_pattern": [...],
    "temporal_frequency": 2.5,
    "coupling_strength": 0.8
  }
}
```

#### `POST /integration/esc-kuramoto`
ESC-Kuramoto parameter constraint extraction.

**Request:**
```json
{
  "semantic_content": [
    "Oscillation frequency should remain between 1-5 Hz",
    "Coupling strength must not exceed 0.9"
  ],
  "current_parameters": {
    "omega": [2.1, 3.4, 4.2],
    "K_matrix": [[0.8, 0.3], [0.3, 0.7]]
  }
}
```

**Response:**
```json
{
  "constraints": {
    "omega_bounds": {"min": 1.0, "max": 5.0},
    "K_max": 0.9,
    "coupling_constraints": [
      {"i": 0, "j": 1, "max_value": 0.9},
      {"type": "symmetric", "enforce": true}
    ]
  },
  "violations": [],
  "suggested_adjustments": {
    "K_matrix": [[0.8, 0.3], [0.3, 0.7]]
  }
}
```

---

### **5. Performance and Monitoring**

#### `GET /metrics`
Real-time performance metrics.

**Response:**
```json
{
  "slo_compliance": {
    "latency_p95_ms": 287.4,
    "latency_slo_met": true,
    "accuracy_rate": 0.987,
    "accuracy_slo_met": true,
    "overall_compliant": true
  },
  "throughput": {
    "requests_per_second": 23.7,
    "concurrent_requests": 4,
    "queue_depth": 0
  },
  "cache_performance": {
    "hit_rate": 0.763,
    "total_requests": 15420,
    "cache_size_mb": 45.2
  },
  "integration_status": {
    "wolfram_available": true,
    "z3_available": true,
    "esc_bridge_connected": true
  }
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.4.3-kamil",
  "uptime_seconds": 3600,
  "dependencies": {
    "wolfram_api": "ok",
    "z3_solver": "ok", 
    "spacy_nlp": "ok",
    "esc_bridge": "connected"
  },
  "performance": {
    "avg_latency_ms": 156.7,
    "error_rate": 0.003,
    "memory_usage_mb": 234.6
  }
}
```

---

### **6. Administrative Endpoints**

#### `POST /admin/cache/clear`
Clear performance caches.

**Response:**
```json
{
  "status": "success",
  "cleared": {
    "symbolize_cache": 1247,
    "verify_cache": 423,
    "wolfram_cache": 89
  },
  "memory_freed_mb": 67.3
}
```

#### `POST /admin/benchmark`
Run performance benchmark.

**Request:**
```json
{
  "num_operations": 100,
  "test_cases": ["physics", "mathematics", "mixed"]
}
```

**Response:**
```json
{
  "benchmark_id": "bench_1694712000",
  "results": {
    "total_operations": 100,
    "success_rate": 0.98,
    "latency_p50_ms": 145.2,
    "latency_p95_ms": 289.7,
    "latency_p99_ms": 567.1,
    "slo_compliance_rate": 0.96,
    "accuracy_mean": 0.993,
    "throughput_ops_per_sec": 18.4
  },
  "slo_status": {
    "latency_slo_met": true,
    "accuracy_slo_met": true,
    "overall_compliant": true
  }
}
```

---

## 📊 **Error Handling**

### Standard Error Response
```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "Dimensional inconsistency detected in expression",
    "details": {
      "expression": "F = m + v",
      "issue": "Cannot add mass and velocity (different dimensions)",
      "suggestions": ["Check equation formulation", "Verify variable definitions"]
    }
  },
  "processing_time_ms": 67.2
}
```

### Error Codes
- `PARSE_ERROR`: Formula/text parsing failed
- `VALIDATION_FAILED`: Validation checks failed  
- `TIMEOUT_ERROR`: External tool timeout
- `RESOURCE_ERROR`: Insufficient resources
- `INTEGRATION_ERROR`: External service unavailable
- `SLO_VIOLATION`: Performance SLO exceeded

---

## 🔐 **Authentication and Rate Limiting**

### API Key Authentication
```http
Authorization: Bearer your-api-key-here
Content-Type: application/json
```

### Rate Limits
- **Standard**: 100 requests/minute
- **Burst**: 1000 requests/hour  
- **Concurrent**: 10 simultaneous requests
- **Premium**: Custom limits available

---

## 📝 **WebSocket API**

### Real-time Validation Stream
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/symbolic/validate');

ws.send(JSON.stringify({
  "text": "Energy E = m * c²",
  "stream": true
}));

// Receives progressive results:
// {"stage": "parsing", "progress": 0.3, "partial": {...}}
// {"stage": "validation", "progress": 0.7, "partial": {...}}
// {"stage": "complete", "progress": 1.0, "result": {...}}
```

---

## 🛠 **SDK Examples**

### Python SDK
```python
from symbolic_ai_client import SymbolicAIClient

client = SymbolicAIClient(api_key="your-key")

# Complete verification
report = client.verify("The momentum p = m * v is conserved")
print(f"Valid: {report.overall_valid}")

# Dimensional check only  
result = client.check_dimensional("F = m * a")
print(f"Dimensionally consistent: {result.consistent}")
```

### JavaScript SDK
```javascript
import { SymbolicAI } from '@vortex-omega/symbolic-ai';

const client = new SymbolicAI({ apiKey: 'your-key' });

// Async verification
const report = await client.verify({
  text: "Energy conservation: E_initial = E_final",
  options: { kant_mode: true }
});

console.log(`Confidence: ${report.answer_conf}`);
```

---

## 📈 **Performance Guarantees**

### SLO Commitments
- **Latency**: 95% of requests ≤ 300ms
- **Accuracy**: ≥ 98% dimensional validation accuracy
- **Availability**: 99.9% uptime
- **Throughput**: ≥ 50 requests/second sustained

### Monitoring Dashboard
Access real-time metrics at: `http://localhost:8000/dashboard/symbolic-ai`

---

## 🔄 **Integration Patterns**

### Webhook Integration
```json
{
  "webhook_url": "https://your-system.com/symbolic-callback",
  "events": ["verification_complete", "slo_violation", "error"],
  "secret": "webhook-secret-key"
}
```

### Message Queue Integration  
```python
# Redis/RabbitMQ integration
import asyncio
from symbolic_ai_queue import SymbolicQueue

queue = SymbolicQueue(redis_url="redis://localhost:6379")

async def process_batch():
    results = await queue.process_batch(max_size=10)
    return results
```

---

This API documentation provides complete coverage of the Kamil Gadeev Symbolic AI specification with production-ready endpoints, comprehensive error handling, and integration patterns for the NFCS v2.4.3 system.