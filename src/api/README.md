# 🌐 NFCS API Layer

## Overview
RESTful and WebSocket API interface for the Neural Field Control System (NFCS) v2.4.3, providing programmatic access to all system capabilities.

## 🏗️ Architecture

### API Structure
```
src/api/
├── __init__.py              # API package initialization
├── main.py                  # FastAPI application setup
├── dependencies.py          # Dependency injection
├── middleware/              # Custom middleware components
├── models/                  # Pydantic data models
├── routes/                  # API route definitions
└── websocket/               # WebSocket handlers
```

## 🎯 Current Implementation Status

### MVP API Features (Active)
The MVP system provides API access through:
- **Flask Web Interface**: Real-time dashboard API
- **Socket.IO WebSocket**: Live system communication
- **REST Endpoints**: System status and control

### Production API (Planned)
Full FastAPI implementation will provide:
- **OpenAPI Documentation**: Auto-generated API docs
- **Async Operations**: High-performance async handlers
- **Authentication**: JWT-based security
- **Rate Limiting**: DDoS protection
- **Validation**: Pydantic model validation

## 📁 Directory Components

### 🚪 `routes/`
API route definitions organized by functionality:
- **health.py**: System health and status endpoints
- **constitutional.py**: Constitutional monitoring API
- **kuramoto.py**: ESC-Kuramoto integration endpoints
- **cognitive.py**: Cognitive module interfaces
- **validation.py**: Empirical validation API
- **websocket.py**: Real-time WebSocket handlers

### 🎭 `middleware/`
Custom middleware for cross-cutting concerns:
- **auth.py**: Authentication and authorization
- **cors.py**: CORS policy management
- **logging.py**: Request/response logging
- **rate_limit.py**: Rate limiting implementation
- **error_handler.py**: Global error handling

### 📊 `models/`
Pydantic data models for API contracts:
- **constitutional.py**: Constitutional monitoring models
- **kuramoto.py**: Kuramoto synchronization models
- **cognitive.py**: Cognitive module data structures
- **validation.py**: Validation result models
- **common.py**: Shared model definitions

### 🔌 `websocket/`
Real-time WebSocket communication:
- **handlers.py**: WebSocket event handlers
- **events.py**: Event type definitions
- **broadcast.py**: Message broadcasting utilities

## 🚀 API Endpoints

### Current MVP Endpoints

#### System Status
- `GET /api/status` - Current system status
- `GET /api/metrics` - Performance metrics summary

#### WebSocket Events
- `connect` - Client connection
- `start_mvp` - Start MVP system
- `stop_mvp` - Stop MVP system
- `get_status` - Request status update
- `demo_capabilities` - Demonstrate system capabilities

### Future Production Endpoints

#### Constitutional Monitoring
- `GET /api/constitutional/status` - Monitoring status
- `POST /api/constitutional/check` - Run compliance check
- `GET /api/constitutional/violations` - Recent violations
- `WebSocket /ws/constitutional` - Real-time monitoring

#### ESC-Kuramoto Integration  
- `GET /api/kuramoto/sync-state` - Current synchronization state
- `POST /api/kuramoto/configure` - Update configuration
- `GET /api/kuramoto/predictions` - Multi-horizon predictions
- `WebSocket /ws/kuramoto` - Live synchronization data

#### Cognitive Modules
- `GET /api/cognitive/modules` - List active modules
- `POST /api/cognitive/symbolic/query` - Symbolic AI queries
- `GET /api/cognitive/memory/recall` - Memory system access
- `POST /api/cognitive/reflection/analyze` - Meta-reflection

#### Validation & Testing
- `POST /api/validation/run` - Execute validation pipeline
- `GET /api/validation/results/{id}` - Validation results
- `GET /api/validation/benchmarks` - Performance benchmarks

## 🔐 Security Features

### Authentication
- **JWT Tokens**: Stateless authentication
- **Refresh Tokens**: Secure token renewal
- **Role-Based Access**: Permission management
- **API Keys**: Service-to-service auth

### Security Middleware
- **Rate Limiting**: Per-client request limits
- **CORS Policy**: Cross-origin request management
- **Input Validation**: Pydantic model validation
- **SQL Injection Protection**: Parameterized queries
- **XSS Protection**: Output sanitization

## 📊 Performance Features

### Async Operations
- **Async/Await**: Non-blocking request handling
- **Connection Pooling**: Database connection management
- **Background Tasks**: Long-running operations
- **Streaming Responses**: Large data transfers

### Caching Strategy
- **Redis Integration**: In-memory caching
- **Response Caching**: Frequently accessed data
- **Cache Invalidation**: Smart cache management
- **CDN Support**: Static asset delivery

## 🔧 Development Setup

### Local Development
```bash
# Install dependencies
pip install fastapi uvicorn

# Start development server
uvicorn src.api.main:app --reload --port 8000

# Access API documentation
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

### Testing
```bash
# Run API tests
pytest tests/test_api/ -v

# Test specific endpoints
pytest tests/test_api/test_routes.py::test_status_endpoint

# Load testing
locust -f tests/load/api_load_test.py
```

## 📈 Monitoring & Observability

### Metrics Collection
- **Request Metrics**: Latency, throughput, errors
- **Business Metrics**: NFCS-specific KPIs
- **Infrastructure Metrics**: CPU, memory, disk usage

### Logging
- **Structured Logging**: JSON-formatted logs
- **Request Tracing**: Distributed tracing support
- **Error Logging**: Exception capture and reporting

## 🔗 Integration Points

### NFCS Core Systems
- **Constitutional Monitor**: Real-time oversight API
- **Kuramoto Network**: Synchronization state API
- **Cognitive Modules**: Intelligence interface API
- **Validation Pipeline**: Testing and benchmarking API

### External Services
- **Database**: PostgreSQL integration
- **Cache**: Redis integration
- **Message Queue**: Async task processing
- **Monitoring**: Prometheus metrics export

## 📝 Usage Examples

### Python Client
```python
import httpx

# Get system status
async with httpx.AsyncClient() as client:
    response = await client.get("http://localhost:8000/api/status")
    status = response.json()
    print(f"System health: {status['system_health']}")
```

### JavaScript Client
```javascript
// WebSocket connection
const socket = io("http://localhost:5000");

socket.on('status_update', (data) => {
    console.log('System status:', data);
});

socket.emit('get_status');
```

### curl Examples
```bash
# Get system metrics
curl http://localhost:8000/api/metrics

# Start MVP system (current)
curl -X POST http://localhost:5000/start_mvp
```

## 🔗 Related Documentation
- [MVP Web Interface](../../mvp_web_interface.py)
- [System Core](../core/README.md)
- [Testing Guide](../../docs/testing/README.md)

---
*Part of Vortex-Omega Neural Field Control System v2.4.3*