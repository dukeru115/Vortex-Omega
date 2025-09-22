# NFCS API Documentation & Swagger Integration

## Overview

Comprehensive API documentation for the Neural Field Control System (NFCS) with OpenAPI 3.0 specification, interactive Swagger UI, and extensive usage examples. This document provides complete API reference for developers integrating with NFCS.

**API Version**: 2.4.3  
**OpenAPI Version**: 3.0.3  
**Last Updated**: 2025-09-15  
**Maintained by**: Team Ω - Neural Field Control Systems Research Group

## 🎯 API Architecture

### API Design Principles

1. **RESTful Design**: Standard HTTP methods and status codes
2. **Constitutional Compliance**: All API calls subject to constitutional framework
3. **Real-time Capabilities**: WebSocket support for live updates
4. **Security First**: Authentication, authorization, and rate limiting
5. **Comprehensive Documentation**: Interactive examples and testing
6. **Backward Compatibility**: Versioned APIs with deprecation policies

### Base URL Structure

```
Production:  https://api.nfcs.example.com/v2
Development: http://localhost:5000/api/v2
Local MVP:   http://localhost:5000/api
```

## 📋 OpenAPI 3.0 Specification

### Core API Specification

```yaml
# openapi-spec.yaml - NFCS API OpenAPI 3.0 Specification
openapi: 3.0.3
info:
  title: Neural Field Control System (NFCS) API
  description: |
    Comprehensive API for the Neural Field Control System (NFCS) v2.4.3.
    
    The NFCS API provides access to:
    - Constitutional monitoring and compliance
    - Cognitive module interactions
    - Mathematical core operations (CGL, Kuramoto)
    - ESC token processing
    - System orchestration and coordination
    - Real-time monitoring and analytics
    
    ## Authentication
    All API endpoints require authentication unless otherwise specified.
    Use Bearer token authentication with JWT tokens.
    
    ## Rate Limiting
    API calls are rate limited to:
    - 100 requests per minute for authenticated users
    - 10 requests per minute for unauthenticated endpoints
    
    ## Constitutional Compliance
    All API operations are subject to the NFCS constitutional framework.
    Requests violating constitutional policies will be rejected with 403 status.
    
  version: 2.4.3
  contact:
    name: NFCS API Support
    email: api-support@nfcs.internal
    url: https://docs.nfcs.internal
  license:
    name: CC BY-NC 4.0
    url: https://creativecommons.org/licenses/by-nc/4.0/

servers:
  - url: https://api.nfcs.example.com/v2
    description: Production server
  - url: https://staging-api.nfcs.example.com/v2
    description: Staging server
  - url: http://localhost:5000/api/v2
    description: Development server

paths:
  # === System Health and Status ===
  /health:
    get:
      summary: System Health Check
      description: Get current system health status and operational metrics
      tags:
        - System
      security: []
      responses:
        '200':
          description: System health information
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthStatus'
              example:
                status: "healthy"
                version: "2.4.3"
                uptime: 86400
                components:
                  constitutional: "active"
                  kuramoto: "synchronized"
                  cognitive_modules: "5/5 active"
                  esc_system: "operational"
                metrics:
                  coordination_frequency: 10.2
                  response_time_p95: 0.045
                  memory_usage_mb: 2048
                  constitutional_compliance: 0.95

  /status:
    get:
      summary: Detailed System Status
      description: Get comprehensive system status including all components
      tags:
        - System
      security:
        - BearerAuth: []
      responses:
        '200':
          description: Detailed system status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SystemStatus'

  # === Authentication and Authorization ===
  /auth/login:
    post:
      summary: User Authentication
      description: Authenticate user and receive JWT token
      tags:
        - Authentication
      security: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/LoginRequest'
            example:
              username: "researcher"
              password: "secure_password_123"
              mfa_token: "123456"
      responses:
        '200':
          description: Authentication successful
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuthResponse'
        '401':
          description: Authentication failed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '429':
          description: Rate limit exceeded

  /auth/refresh:
    post:
      summary: Refresh JWT Token
      description: Refresh an existing JWT token
      tags:
        - Authentication
      security:
        - BearerAuth: []
      responses:
        '200':
          description: Token refreshed successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuthResponse'

  # === Constitutional Framework ===
  /constitutional/status:
    get:
      summary: Constitutional Framework Status
      description: Get current constitutional framework status and compliance metrics
      tags:
        - Constitutional
      security:
        - BearerAuth: []
      responses:
        '200':
          description: Constitutional status information
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ConstitutionalStatus'
              example:
                status: "active"
                enforcement_level: "strict"
                compliance_score: 0.95
                ha_number: 0.65
                ha_threshold: 0.8
                violations_24h: 2
                policies_active: 15

  /constitutional/policies:
    get:
      summary: List Constitutional Policies
      description: Get list of active constitutional policies
      tags:
        - Constitutional
      security:
        - BearerAuth: []
      parameters:
        - name: category
          in: query
          description: Filter by policy category
          schema:
            type: string
            enum: [safety, security, ethics, governance]
        - name: active_only
          in: query
          description: Return only active policies
          schema:
            type: boolean
            default: true
      responses:
        '200':
          description: List of constitutional policies
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/ConstitutionalPolicy'

    post:
      summary: Create Constitutional Policy
      description: Create a new constitutional policy
      tags:
        - Constitutional
      security:
        - BearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreatePolicyRequest'
      responses:
        '201':
          description: Policy created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ConstitutionalPolicy'
        '403':
          description: Insufficient privileges
        '422':
          description: Invalid policy definition

  /constitutional/violations:
    get:
      summary: Constitutional Violations Log
      description: Get log of constitutional violations
      tags:
        - Constitutional
      security:
        - BearerAuth: []
      parameters:
        - name: since
          in: query
          description: Get violations since timestamp (ISO 8601)
          schema:
            type: string
            format: date-time
        - name: severity
          in: query
          description: Filter by violation severity
          schema:
            type: string
            enum: [low, medium, high, critical]
        - name: limit
          in: query
          description: Maximum number of violations to return
          schema:
            type: integer
            default: 100
            maximum: 1000
      responses:
        '200':
          description: List of constitutional violations
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/ConstitutionalViolation'

  # === Cognitive Modules ===
  /modules:
    get:
      summary: List Cognitive Modules
      description: Get list of available cognitive modules and their status
      tags:
        - Cognitive Modules
      security:
        - BearerAuth: []
      responses:
        '200':
          description: List of cognitive modules
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/CognitiveModule'
              example:
                - name: "constitutional"
                  status: "active"
                  version: "2.4.3"
                  capabilities: ["policy_enforcement", "compliance_monitoring"]
                  metrics:
                    response_time_ms: 15
                    success_rate: 0.99
                - name: "boundary"
                  status: "active"
                  version: "2.4.3"
                  capabilities: ["adaptive_boundaries", "safety_margins"]

  /modules/{module_name}/status:
    get:
      summary: Get Module Status
      description: Get detailed status information for a specific cognitive module
      tags:
        - Cognitive Modules
      security:
        - BearerAuth: []
      parameters:
        - name: module_name
          in: path
          required: true
          description: Name of the cognitive module
          schema:
            type: string
            enum: [constitutional, boundary, memory, meta_reflection, freedom]
      responses:
        '200':
          description: Module status information
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ModuleStatus'
        '404':
          description: Module not found

  /modules/{module_name}/process:
    post:
      summary: Process Input with Module
      description: Send input to a specific cognitive module for processing
      tags:
        - Cognitive Modules
      security:
        - BearerAuth: []
      parameters:
        - name: module_name
          in: path
          required: true
          description: Name of the cognitive module
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ModuleProcessRequest'
            example:
              input: "Analyze this text for constitutional compliance"
              context:
                user_id: "researcher_001"
                session_id: "sess_12345"
                priority: "normal"
              parameters:
                analysis_depth: "deep"
                include_recommendations: true
      responses:
        '200':
          description: Processing completed successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ModuleProcessResponse'
        '403':
          description: Constitutional violation detected
        '422':
          description: Invalid input parameters

  # === Mathematical Core ===
  /mathematical/cgl/solve:
    post:
      summary: Solve CGL Equation
      description: Solve Complex Ginzburg-Landau equation with specified parameters
      tags:
        - Mathematical Core
      security:
        - BearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CGLSolveRequest'
            example:
              parameters:
                c1: 0.5
                c2: 1.0
                c3: 0.8
              grid_size: [128, 128]
              domain_size: [10.0, 10.0]
              time_steps: 1000
              dt: 0.01
              initial_conditions: "random"
      responses:
        '200':
          description: CGL solution computed successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CGLSolveResponse'
        '422':
          description: Invalid parameters

  /mathematical/kuramoto/synchronize:
    post:
      summary: Kuramoto Synchronization
      description: Perform Kuramoto synchronization with specified oscillators
      tags:
        - Mathematical Core
      security:
        - BearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/KuramotoSyncRequest'
            example:
              oscillators: 5
              coupling_strength: 2.0
              natural_frequencies: [0.8, 0.9, 1.0, 1.1, 1.2]
              initial_phases: [0.0, 0.5, 1.0, 1.5, 2.0]
              integration_time: 10.0
              dt: 0.01
      responses:
        '200':
          description: Synchronization completed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/KuramotoSyncResponse'

  # === ESC System ===
  /esc/process:
    post:
      summary: Process Tokens with ESC
      description: Process tokens through Echo-Semantic Converter system
      tags:
        - ESC System
      security:
        - BearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ESCProcessRequest'
            example:
              tokens: ["Hello", "world", "from", "NFCS"]
              context:
                semantic_field: "conversational"
                attention_mode: "focused"
                constitutional_filter: true
              parameters:
                max_sequence_length: 512
                attention_heads: 8
                semantic_depth: "standard"
      responses:
        '200':
          description: Token processing completed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ESCProcessResponse'

  /esc/semantic-fields:
    get:
      summary: Get Semantic Fields Status
      description: Get current status of semantic fields in ESC system
      tags:
        - ESC System
      security:
        - BearerAuth: []
      responses:
        '200':
          description: Semantic fields status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SemanticFieldsStatus'

  # === System Orchestration ===
  /orchestrator/coordination:
    get:
      summary: Coordination Status
      description: Get current coordination status and frequency metrics
      tags:
        - Orchestration
      security:
        - BearerAuth: []
      responses:
        '200':
          description: Coordination status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CoordinationStatus'
              example:
                status: "active"
                frequency_hz: 10.2
                sync_level: 0.85
                active_modules: 5
                coordination_latency_ms: 12
                last_coordination: "2025-09-15T10:30:45Z"

  /orchestrator/process:
    post:
      summary: Process Input through Orchestrator
      description: Send input through the complete NFCS orchestration pipeline
      tags:
        - Orchestration
      security:
        - BearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/OrchestratorProcessRequest'
            example:
              input:
                text: "Analyze this complex scenario"
                type: "natural_language"
                priority: "normal"
              context:
                user_id: "researcher_001"
                session_id: "sess_12345"
                domain: "research"
              options:
                include_constitutional_analysis: true
                include_mathematical_modeling: true
                include_esc_processing: true
                response_format: "detailed"
      responses:
        '200':
          description: Processing completed successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/OrchestratorProcessResponse'

  # === Real-time WebSocket Endpoints ===
  /ws/constitutional:
    get:
      summary: Constitutional Monitoring WebSocket
      description: WebSocket endpoint for real-time constitutional monitoring updates
      tags:
        - WebSocket
      security:
        - BearerAuth: []
      responses:
        '101':
          description: WebSocket connection established

  /ws/metrics:
    get:
      summary: System Metrics WebSocket
      description: WebSocket endpoint for real-time system metrics
      tags:
        - WebSocket
      security:
        - BearerAuth: []
      responses:
        '101':
          description: WebSocket connection established

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: JWT token obtained from /auth/login endpoint

  schemas:
    # === System Schemas ===
    HealthStatus:
      type: object
      properties:
        status:
          type: string
          enum: [healthy, degraded, critical]
        version:
          type: string
          example: "2.4.3"
        uptime:
          type: integer
          description: System uptime in seconds
        components:
          type: object
          properties:
            constitutional:
              type: string
              enum: [active, inactive, error]
            kuramoto:
              type: string
              enum: [synchronized, desynchronized, error]
            cognitive_modules:
              type: string
              example: "5/5 active"
            esc_system:
              type: string
              enum: [operational, degraded, error]
        metrics:
          type: object
          properties:
            coordination_frequency:
              type: number
              example: 10.2
            response_time_p95:
              type: number
              example: 0.045
            memory_usage_mb:
              type: integer
              example: 2048
            constitutional_compliance:
              type: number
              minimum: 0
              maximum: 1
              example: 0.95

    SystemStatus:
      allOf:
        - $ref: '#/components/schemas/HealthStatus'
        - type: object
          properties:
            detailed_metrics:
              type: object
            active_sessions:
              type: integer
            resource_usage:
              type: object
            alerts:
              type: array
              items:
                $ref: '#/components/schemas/Alert'

    # === Authentication Schemas ===
    LoginRequest:
      type: object
      required:
        - username
        - password
      properties:
        username:
          type: string
          example: "researcher"
        password:
          type: string
          format: password
          example: "secure_password_123"
        mfa_token:
          type: string
          description: Multi-factor authentication token
          example: "123456"

    AuthResponse:
      type: object
      properties:
        access_token:
          type: string
          description: JWT access token
        token_type:
          type: string
          example: "Bearer"
        expires_in:
          type: integer
          description: Token expiration time in seconds
          example: 3600
        refresh_token:
          type: string
          description: Refresh token for obtaining new access tokens
        user:
          $ref: '#/components/schemas/User'

    User:
      type: object
      properties:
        id:
          type: string
          example: "researcher_001"
        username:
          type: string
          example: "researcher"
        role:
          type: string
          enum: [admin, researcher, operator, viewer]
        permissions:
          type: array
          items:
            type: string
        last_login:
          type: string
          format: date-time

    # === Constitutional Schemas ===
    ConstitutionalStatus:
      type: object
      properties:
        status:
          type: string
          enum: [active, inactive, error]
        enforcement_level:
          type: string
          enum: [strict, moderate, advisory]
        compliance_score:
          type: number
          minimum: 0
          maximum: 1
        ha_number:
          type: number
          description: Current hallucination number
        ha_threshold:
          type: number
          description: Hallucination threshold
        violations_24h:
          type: integer
          description: Violations in last 24 hours
        policies_active:
          type: integer
          description: Number of active policies

    ConstitutionalPolicy:
      type: object
      properties:
        id:
          type: string
          example: "policy_001"
        name:
          type: string
          example: "Safety Constraint Policy"
        category:
          type: string
          enum: [safety, security, ethics, governance]
        description:
          type: string
        rules:
          type: array
          items:
            type: object
        priority:
          type: integer
          minimum: 1
          maximum: 10
        active:
          type: boolean
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time

    ConstitutionalViolation:
      type: object
      properties:
        id:
          type: string
        timestamp:
          type: string
          format: date-time
        severity:
          type: string
          enum: [low, medium, high, critical]
        policy_id:
          type: string
        violation_type:
          type: string
        description:
          type: string
        context:
          type: object
        action_taken:
          type: string
        resolved:
          type: boolean

    # === Cognitive Module Schemas ===
    CognitiveModule:
      type: object
      properties:
        name:
          type: string
        status:
          type: string
          enum: [active, inactive, error, maintenance]
        version:
          type: string
        capabilities:
          type: array
          items:
            type: string
        metrics:
          type: object
          properties:
            response_time_ms:
              type: number
            success_rate:
              type: number
            throughput:
              type: number

    ModuleProcessRequest:
      type: object
      required:
        - input
      properties:
        input:
          type: string
          description: Input data to process
        context:
          type: object
          description: Processing context
        parameters:
          type: object
          description: Module-specific parameters

    ModuleProcessResponse:
      type: object
      properties:
        result:
          type: object
          description: Processing result
        metadata:
          type: object
          properties:
            processing_time_ms:
              type: number
            constitutional_compliance:
              type: number
            confidence_score:
              type: number
        warnings:
          type: array
          items:
            type: string

    # === Mathematical Core Schemas ===
    CGLSolveRequest:
      type: object
      required:
        - parameters
        - grid_size
      properties:
        parameters:
          type: object
          properties:
            c1:
              type: number
            c2:
              type: number
            c3:
              type: number
        grid_size:
          type: array
          items:
            type: integer
          minItems: 2
          maxItems: 3
        domain_size:
          type: array
          items:
            type: number
        time_steps:
          type: integer
          minimum: 1
        dt:
          type: number
          minimum: 0.001
        initial_conditions:
          type: string
          enum: [random, gaussian, custom]

    CGLSolveResponse:
      type: object
      properties:
        solution:
          type: object
          description: Solution data
        metrics:
          type: object
          properties:
            computation_time_ms:
              type: number
            stability_index:
              type: number
            defect_count:
              type: integer
        visualization_data:
          type: object
          description: Data for visualization

    KuramotoSyncRequest:
      type: object
      required:
        - oscillators
        - coupling_strength
      properties:
        oscillators:
          type: integer
          minimum: 2
        coupling_strength:
          type: number
        natural_frequencies:
          type: array
          items:
            type: number
        initial_phases:
          type: array
          items:
            type: number
        integration_time:
          type: number
          minimum: 0.1
        dt:
          type: number
          minimum: 0.001

    KuramotoSyncResponse:
      type: object
      properties:
        final_phases:
          type: array
          items:
            type: number
        synchronization_level:
          type: number
          minimum: 0
          maximum: 1
        time_series:
          type: object
        metrics:
          type: object

    # === ESC System Schemas ===
    ESCProcessRequest:
      type: object
      required:
        - tokens
      properties:
        tokens:
          type: array
          items:
            type: string
        context:
          type: object
        parameters:
          type: object

    ESCProcessResponse:
      type: object
      properties:
        processed_tokens:
          type: array
          items:
            type: object
        semantic_fields:
          type: object
        attention_weights:
          type: array
        constitutional_compliance:
          type: number
        processing_metadata:
          type: object

    # === Error Schemas ===
    ErrorResponse:
      type: object
      properties:
        error:
          type: string
          description: Error type
        message:
          type: string
          description: Human-readable error message
        code:
          type: integer
          description: Specific error code
        details:
          type: object
          description: Additional error details
        timestamp:
          type: string
          format: date-time

    Alert:
      type: object
      properties:
        id:
          type: string
        severity:
          type: string
          enum: [info, warning, error, critical]
        message:
          type: string
        timestamp:
          type: string
          format: date-time
        resolved:
          type: boolean

  # === Response Examples ===
  examples:
    HealthySystemResponse:
      summary: Healthy system response
      value:
        status: "healthy"
        version: "2.4.3"
        uptime: 86400
        components:
          constitutional: "active"
          kuramoto: "synchronized"
          cognitive_modules: "5/5 active"
          esc_system: "operational"

    ConstitutionalViolationExample:
      summary: Constitutional violation example
      value:
        id: "violation_001"
        timestamp: "2025-09-15T10:30:45Z"
        severity: "high"
        policy_id: "safety_001"
        violation_type: "unsafe_output"
        description: "Generated content violated safety constraints"
        context:
          input: "user request..."
          output: "system response..."
        action_taken: "output_blocked"
        resolved: true

# === Tags ===
tags:
  - name: System
    description: System health and status endpoints
  - name: Authentication
    description: Authentication and authorization
  - name: Constitutional
    description: Constitutional framework operations
  - name: Cognitive Modules
    description: Cognitive module interactions
  - name: Mathematical Core
    description: Mathematical operations and solvers
  - name: ESC System
    description: Echo-Semantic Converter operations
  - name: Orchestration
    description: System orchestration and coordination
  - name: WebSocket
    description: Real-time WebSocket endpoints
```

## 🚀 Swagger UI Integration

### Swagger UI Setup

```python
#!/usr/bin/env python3
# swagger_integration.py - Swagger UI integration for NFCS API

from flask import Flask, render_template, jsonify, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint
import yaml
import os

app = Flask(__name__)

# Swagger UI configuration
SWAGGER_URL = '/docs'  # URL for exposing Swagger UI
API_URL = '/api/spec'  # URL for OpenAPI spec

# Create Swagger UI blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "NFCS API Documentation",
        'supportedSubmitMethods': ['get', 'post', 'put', 'delete', 'patch'],
        'validatorUrl': None,
        'oauth': {
            'clientId': 'nfcs-api-client',
            'realm': 'nfcs-realm',
            'appName': 'NFCS API'
        },
        'persistAuthorization': True,
        'displayRequestDuration': True,
        'filter': True,
        'deepLinking': True,
        'displayOperationId': True,
        'showExtensions': True,
        'showCommonExtensions': True
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/api/spec')
def get_api_spec():
    """Serve OpenAPI specification."""
    try:
        with open('openapi-spec.yaml', 'r') as file:
            spec = yaml.safe_load(file)
        return jsonify(spec)
    except Exception as e:
        return jsonify({'error': 'Failed to load API specification'}), 500

@app.route('/api/spec/download')
def download_api_spec():
    """Download OpenAPI specification file."""
    return send_from_directory('.', 'openapi-spec.yaml', as_attachment=True)

@app.route('/docs/examples')
def api_examples():
    """API usage examples page."""
    return render_template('api_examples.html')

@app.route('/docs/postman')
def postman_collection():
    """Generate Postman collection from OpenAPI spec."""
    # Convert OpenAPI spec to Postman collection
    # Implementation would convert the YAML spec to Postman format
    pass

if __name__ == '__main__':
    app.run(debug=True, port=5001)
```

### Enhanced Swagger Configuration

```javascript
// swagger-config.js - Enhanced Swagger UI configuration

window.onload = function() {
  const ui = SwaggerUIBundle({
    url: '/api/spec',
    dom_id: '#swagger-ui',
    deepLinking: true,
    presets: [
      SwaggerUIBundle.presets.apis,
      SwaggerUIStandalonePreset
    ],
    plugins: [
      SwaggerUIBundle.plugins.DownloadUrl
    ],
    layout: "StandaloneLayout",
    
    // Authentication configuration
    initOAuth: {
      clientId: 'nfcs-api-client',
      realm: 'nfcs-realm',
      appName: 'NFCS API Documentation',
      scopeSeparator: ' ',
      additionalQueryStringParams: {}
    },
    
    // Request interceptor for authentication
    requestInterceptor: (request) => {
      const token = localStorage.getItem('nfcs_auth_token');
      if (token) {
        request.headers.Authorization = `Bearer ${token}`;
      }
      return request;
    },
    
    // Response interceptor for error handling
    responseInterceptor: (response) => {
      if (response.status === 401) {
        // Handle unauthorized access
        localStorage.removeItem('nfcs_auth_token');
        alert('Authentication required. Please login.');
      }
      return response;
    },
    
    // Custom validators
    validatorUrl: null,
    
    // UI customization
    docExpansion: 'list',
    filter: true,
    showRequestHeaders: true,
    showExtensions: true,
    showCommonExtensions: true,
    tryItOutEnabled: true,
    
    // NFCS-specific customizations
    onComplete: () => {
      // Add custom CSS for NFCS branding
      const style = document.createElement('style');
      style.textContent = `
        .swagger-ui .topbar { 
          background-color: #2c3e50; 
        }
        .swagger-ui .topbar .download-url-wrapper { 
          display: none; 
        }
        .swagger-ui .info { 
          margin: 50px 0; 
        }
        .swagger-ui .info hgroup.main { 
          margin: 0 0 20px 0; 
        }
        .swagger-ui .info hgroup.main a { 
          color: #3498db; 
        }
      `;
      document.head.appendChild(style);
      
      // Add authentication helper
      addAuthenticationHelper();
    }
  });
  
  function addAuthenticationHelper() {
    // Add custom authentication UI
    const authContainer = document.createElement('div');
    authContainer.innerHTML = `
      <div style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px;">
        <h4>Quick Authentication</h4>
        <div style="display: flex; gap: 10px; align-items: center;">
          <input type="text" id="quick-auth-username" placeholder="Username" style="padding: 5px;">
          <input type="password" id="quick-auth-password" placeholder="Password" style="padding: 5px;">
          <button onclick="quickAuthenticate()" style="padding: 5px 10px;">Login</button>
          <button onclick="clearAuth()" style="padding: 5px 10px;">Logout</button>
        </div>
        <div id="auth-status" style="margin-top: 10px; font-size: 12px;"></div>
      </div>
    `;
    
    const infoSection = document.querySelector('.swagger-ui .info');
    if (infoSection) {
      infoSection.appendChild(authContainer);
    }
  }
  
  // Global authentication functions
  window.quickAuthenticate = async function() {
    const username = document.getElementById('quick-auth-username').value;
    const password = document.getElementById('quick-auth-password').value;
    const statusElement = document.getElementById('auth-status');
    
    try {
      const response = await fetch('/api/v2/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, password })
      });
      
      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('nfcs_auth_token', data.access_token);
        statusElement.innerHTML = '<span style="color: green;">✓ Authenticated successfully</span>';
        
        // Update Swagger UI with token
        ui.preauthorizeApiKey('BearerAuth', `Bearer ${data.access_token}`);
      } else {
        statusElement.innerHTML = '<span style="color: red;">✗ Authentication failed</span>';
      }
    } catch (error) {
      statusElement.innerHTML = '<span style="color: red;">✗ Network error</span>';
    }
  };
  
  window.clearAuth = function() {
    localStorage.removeItem('nfcs_auth_token');
    document.getElementById('auth-status').innerHTML = 'Logged out';
    ui.preauthorizeApiKey('BearerAuth', null);
  };
};
```

## 📚 API Usage Examples

### Python Client Examples

```python
#!/usr/bin/env python3
# nfcs_api_client.py - Python client examples for NFCS API

import requests
import json
import websocket
import threading
from typing import Dict, Any, Optional

class NFCSClient:
    """Python client for NFCS API."""
    
    def __init__(self, base_url: str = "http://localhost:5000/api/v2"):
        self.base_url = base_url
        self.token = None
        self.session = requests.Session()
    
    def authenticate(self, username: str, password: str, mfa_token: str = None) -> bool:
        """Authenticate with NFCS API."""
        data = {
            "username": username,
            "password": password
        }
        if mfa_token:
            data["mfa_token"] = mfa_token
        
        response = self.session.post(f"{self.base_url}/auth/login", json=data)
        
        if response.status_code == 200:
            auth_data = response.json()
            self.token = auth_data["access_token"]
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}"
            })
            return True
        return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_constitutional_status(self) -> Dict[str, Any]:
        """Get constitutional framework status."""
        response = self.session.get(f"{self.base_url}/constitutional/status")
        response.raise_for_status()
        return response.json()
    
    def process_with_module(self, module_name: str, input_data: str, 
                          context: Dict = None, parameters: Dict = None) -> Dict[str, Any]:
        """Process input with specific cognitive module."""
        data = {
            "input": input_data,
            "context": context or {},
            "parameters": parameters or {}
        }
        
        response = self.session.post(
            f"{self.base_url}/modules/{module_name}/process",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def solve_cgl(self, parameters: Dict[str, float], grid_size: list,
                  time_steps: int = 1000) -> Dict[str, Any]:
        """Solve Complex Ginzburg-Landau equation."""
        data = {
            "parameters": parameters,
            "grid_size": grid_size,
            "time_steps": time_steps,
            "dt": 0.01
        }
        
        response = self.session.post(f"{self.base_url}/mathematical/cgl/solve", json=data)
        response.raise_for_status()
        return response.json()
    
    def kuramoto_sync(self, oscillators: int, coupling_strength: float,
                     natural_frequencies: list = None) -> Dict[str, Any]:
        """Perform Kuramoto synchronization."""
        data = {
            "oscillators": oscillators,
            "coupling_strength": coupling_strength,
            "integration_time": 10.0
        }
        
        if natural_frequencies:
            data["natural_frequencies"] = natural_frequencies
        
        response = self.session.post(
            f"{self.base_url}/mathematical/kuramoto/synchronize",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def process_with_esc(self, tokens: list, context: Dict = None) -> Dict[str, Any]:
        """Process tokens with ESC system."""
        data = {
            "tokens": tokens,
            "context": context or {}
        }
        
        response = self.session.post(f"{self.base_url}/esc/process", json=data)
        response.raise_for_status()
        return response.json()
    
    def orchestrator_process(self, input_data: Dict, context: Dict = None,
                           options: Dict = None) -> Dict[str, Any]:
        """Process input through complete orchestration pipeline."""
        data = {
            "input": input_data,
            "context": context or {},
            "options": options or {}
        }
        
        response = self.session.post(f"{self.base_url}/orchestrator/process", json=data)
        response.raise_for_status()
        return response.json()

# Example usage
def main():
    # Initialize client
    client = NFCSClient()
    
    # Authenticate
    if client.authenticate("researcher", "secure_password_123"):
        print("✓ Authentication successful")
    else:
        print("✗ Authentication failed")
        return
    
    # Check system health
    health = client.get_system_health()
    print(f"System status: {health['status']}")
    
    # Check constitutional status
    constitutional = client.get_constitutional_status()
    print(f"Constitutional compliance: {constitutional['compliance_score']}")
    
    # Process text with constitutional module
    result = client.process_with_module(
        "constitutional",
        "Analyze this text for safety compliance",
        context={"user_id": "researcher_001"},
        parameters={"analysis_depth": "deep"}
    )
    print(f"Constitutional analysis: {result['result']}")
    
    # Solve CGL equation
    cgl_result = client.solve_cgl(
        parameters={"c1": 0.5, "c2": 1.0, "c3": 0.8},
        grid_size=[64, 64],
        time_steps=500
    )
    print(f"CGL solution computed in {cgl_result['metrics']['computation_time_ms']}ms")
    
    # Kuramoto synchronization
    kuramoto_result = client.kuramoto_sync(
        oscillators=5,
        coupling_strength=2.0,
        natural_frequencies=[0.8, 0.9, 1.0, 1.1, 1.2]
    )
    print(f"Synchronization level: {kuramoto_result['synchronization_level']}")
    
    # ESC token processing
    esc_result = client.process_with_esc(
        tokens=["Hello", "world", "from", "NFCS"],
        context={"semantic_field": "conversational"}
    )
    print(f"ESC processing completed with {len(esc_result['processed_tokens'])} tokens")

if __name__ == "__main__":
    main()
```

### JavaScript Client Examples

```javascript
// nfcs-api-client.js - JavaScript client for NFCS API

class NFCSClient {
    constructor(baseUrl = 'http://localhost:5000/api/v2') {
        this.baseUrl = baseUrl;
        this.token = null;
    }

    async authenticate(username, password, mfaToken = null) {
        const data = { username, password };
        if (mfaToken) data.mfa_token = mfaToken;

        try {
            const response = await fetch(`${this.baseUrl}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (response.ok) {
                const authData = await response.json();
                this.token = authData.access_token;
                return true;
            }
            return false;
        } catch (error) {
            console.error('Authentication error:', error);
            return false;
        }
    }

    async apiCall(endpoint, method = 'GET', data = null) {
        const headers = { 'Content-Type': 'application/json' };
        if (this.token) {
            headers.Authorization = `Bearer ${this.token}`;
        }

        const config = { method, headers };
        if (data) {
            config.body = JSON.stringify(data);
        }

        const response = await fetch(`${this.baseUrl}${endpoint}`, config);
        
        if (!response.ok) {
            throw new Error(`API call failed: ${response.status} ${response.statusText}`);
        }
        
        return response.json();
    }

    async getSystemHealth() {
        return this.apiCall('/health');
    }

    async getConstitutionalStatus() {
        return this.apiCall('/constitutional/status');
    }

    async processWithModule(moduleName, input, context = {}, parameters = {}) {
        return this.apiCall(`/modules/${moduleName}/process`, 'POST', {
            input,
            context,
            parameters
        });
    }

    async solveCGL(parameters, gridSize, timeSteps = 1000) {
        return this.apiCall('/mathematical/cgl/solve', 'POST', {
            parameters,
            grid_size: gridSize,
            time_steps: timeSteps,
            dt: 0.01
        });
    }

    async kuramotoSync(oscillators, couplingStrength, naturalFrequencies = null) {
        const data = {
            oscillators,
            coupling_strength: couplingStrength,
            integration_time: 10.0
        };
        
        if (naturalFrequencies) {
            data.natural_frequencies = naturalFrequencies;
        }

        return this.apiCall('/mathematical/kuramoto/synchronize', 'POST', data);
    }

    async processWithESC(tokens, context = {}) {
        return this.apiCall('/esc/process', 'POST', { tokens, context });
    }

    async orchestratorProcess(input, context = {}, options = {}) {
        return this.apiCall('/orchestrator/process', 'POST', {
            input,
            context,
            options
        });
    }

    // WebSocket connections
    connectConstitutionalWebSocket(onMessage) {
        const wsUrl = this.baseUrl.replace('http', 'ws') + '/ws/constitutional';
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('Constitutional WebSocket connected');
            // Send authentication
            ws.send(JSON.stringify({ type: 'auth', token: this.token }));
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        return ws;
    }

    connectMetricsWebSocket(onMessage) {
        const wsUrl = this.baseUrl.replace('http', 'ws') + '/ws/metrics';
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('Metrics WebSocket connected');
            ws.send(JSON.stringify({ type: 'auth', token: this.token }));
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };
        
        return ws;
    }
}

// Example usage
async function example() {
    const client = new NFCSClient();
    
    // Authenticate
    const authenticated = await client.authenticate('researcher', 'secure_password_123');
    if (!authenticated) {
        console.error('Authentication failed');
        return;
    }
    
    console.log('✓ Authentication successful');
    
    // Get system health
    const health = await client.getSystemHealth();
    console.log(`System status: ${health.status}`);
    
    // Constitutional analysis
    const constitutionalResult = await client.processWithModule(
        'constitutional',
        'Analyze this for safety compliance',
        { user_id: 'web_user_001' },
        { analysis_depth: 'standard' }
    );
    console.log('Constitutional analysis:', constitutionalResult);
    
    // Real-time constitutional monitoring
    const constitutionalWS = client.connectConstitutionalWebSocket((data) => {
        console.log('Constitutional update:', data);
        
        if (data.type === 'violation') {
            console.warn('Constitutional violation detected:', data.violation);
        } else if (data.type === 'compliance_update') {
            console.log(`Compliance score: ${data.compliance_score}`);
        }
    });
    
    // Real-time metrics monitoring
    const metricsWS = client.connectMetricsWebSocket((data) => {
        if (data.type === 'system_metrics') {
            console.log(`Coordination frequency: ${data.coordination_frequency}Hz`);
            console.log(`Response time P95: ${data.response_time_p95}ms`);
        }
    });
}

// Run example
example().catch(console.error);
```

### cURL Examples

```bash
#!/bin/bash
# nfcs-api-examples.sh - cURL examples for NFCS API

# Configuration
BASE_URL="http://localhost:5000/api/v2"
USERNAME="researcher"
PASSWORD="secure_password_123"

# === Authentication ===
echo "=== Authentication ==="

# Login and get token
AUTH_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}")

TOKEN=$(echo "$AUTH_RESPONSE" | jq -r '.access_token')
echo "Token obtained: ${TOKEN:0:20}..."

# === System Health ===
echo -e "\n=== System Health ==="

curl -s "$BASE_URL/health" | jq '.'

# === Constitutional Framework ===
echo -e "\n=== Constitutional Status ==="

curl -s -H "Authorization: Bearer $TOKEN" \
  "$BASE_URL/constitutional/status" | jq '.'

# === Cognitive Modules ===
echo -e "\n=== Cognitive Modules List ==="

curl -s -H "Authorization: Bearer $TOKEN" \
  "$BASE_URL/modules" | jq '.'

echo -e "\n=== Process with Constitutional Module ==="

curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "$BASE_URL/modules/constitutional/process" \
  -d '{
    "input": "Analyze this text for constitutional compliance",
    "context": {
      "user_id": "api_test_user",
      "session_id": "test_session_001"
    },
    "parameters": {
      "analysis_depth": "deep",
      "include_recommendations": true
    }
  }' | jq '.'

# === Mathematical Core ===
echo -e "\n=== CGL Equation Solving ==="

curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "$BASE_URL/mathematical/cgl/solve" \
  -d '{
    "parameters": {
      "c1": 0.5,
      "c2": 1.0,
      "c3": 0.8
    },
    "grid_size": [64, 64],
    "domain_size": [10.0, 10.0],
    "time_steps": 500,
    "dt": 0.01
  }' | jq '.metrics'

echo -e "\n=== Kuramoto Synchronization ==="

curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "$BASE_URL/mathematical/kuramoto/synchronize" \
  -d '{
    "oscillators": 5,
    "coupling_strength": 2.0,
    "natural_frequencies": [0.8, 0.9, 1.0, 1.1, 1.2],
    "integration_time": 10.0,
    "dt": 0.01
  }' | jq '.synchronization_level'

# === ESC System ===
echo -e "\n=== ESC Token Processing ==="

curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "$BASE_URL/esc/process" \
  -d '{
    "tokens": ["Hello", "world", "from", "NFCS", "API"],
    "context": {
      "semantic_field": "conversational",
      "attention_mode": "focused"
    },
    "parameters": {
      "max_sequence_length": 512,
      "constitutional_filter": true
    }
  }' | jq '.processing_metadata'

# === Orchestrator ===
echo -e "\n=== Orchestrator Processing ==="

curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "$BASE_URL/orchestrator/process" \
  -d '{
    "input": {
      "text": "Analyze this complex research scenario",
      "type": "natural_language",
      "priority": "normal"
    },
    "context": {
      "user_id": "api_test_user",
      "session_id": "test_session_001",
      "domain": "research"
    },
    "options": {
      "include_constitutional_analysis": true,
      "include_mathematical_modeling": true,
      "include_esc_processing": true,
      "response_format": "detailed"
    }
  }' | jq '.metadata'

echo -e "\n=== API Examples Completed ==="
```

## 📊 API Testing and Validation

### Postman Collection

```json
{
  "info": {
    "name": "NFCS API Collection",
    "description": "Complete collection of NFCS API endpoints for testing",
    "version": "2.4.3"
  },
  "variable": [
    {
      "key": "baseUrl",
      "value": "http://localhost:5000/api/v2"
    },
    {
      "key": "token",
      "value": ""
    }
  ],
  "auth": {
    "type": "bearer",
    "bearer": [
      {
        "key": "token",
        "value": "{{token}}",
        "type": "string"
      }
    ]
  },
  "item": [
    {
      "name": "Authentication",
      "item": [
        {
          "name": "Login",
          "request": {
            "method": "POST",
            "header": [
              {
                "key": "Content-Type",
                "value": "application/json"
              }
            ],
            "body": {
              "mode": "raw",
              "raw": "{\n  \"username\": \"researcher\",\n  \"password\": \"secure_password_123\"\n}"
            },
            "url": {
              "raw": "{{baseUrl}}/auth/login",
              "host": ["{{baseUrl}}"],
              "path": ["auth", "login"]
            }
          },
          "event": [
            {
              "listen": "test",
              "script": {
                "exec": [
                  "if (pm.response.code === 200) {",
                  "    const response = pm.response.json();",
                  "    pm.collectionVariables.set('token', response.access_token);",
                  "    pm.test('Token received', function() {",
                  "        pm.expect(response.access_token).to.be.a('string');",
                  "    });",
                  "}"
                ]
              }
            }
          ]
        }
      ]
    },
    {
      "name": "System",
      "item": [
        {
          "name": "Health Check",
          "request": {
            "method": "GET",
            "url": {
              "raw": "{{baseUrl}}/health",
              "host": ["{{baseUrl}}"],
              "path": ["health"]
            }
          },
          "event": [
            {
              "listen": "test",
              "script": {
                "exec": [
                  "pm.test('Status code is 200', function() {",
                  "    pm.response.to.have.status(200);",
                  "});",
                  "",
                  "pm.test('Health status is present', function() {",
                  "    const response = pm.response.json();",
                  "    pm.expect(response).to.have.property('status');",
                  "});"
                ]
              }
            }
          ]
        }
      ]
    }
  ]
}
```

## 🔧 Development Tools

### API Client Generator

```python
#!/usr/bin/env python3
# generate_api_client.py - Generate API clients from OpenAPI spec

import yaml
import json
from jinja2 import Template
from pathlib import Path

def generate_python_client(spec_file: str, output_dir: str):
    """Generate Python client from OpenAPI specification."""
    
    with open(spec_file, 'r') as f:
        spec = yaml.safe_load(f)
    
    # Python client template
    python_template = Template("""
# Generated NFCS API Client
# Version: {{ spec.info.version }}
# Generated from: {{ spec_file }}

import requests
from typing import Dict, Any, Optional, List

class NFCSAPIClient:
    def __init__(self, base_url: str = "{{ spec.servers[0].url }}"):
        self.base_url = base_url
        self.session = requests.Session()
        self.token = None
    
    def set_auth_token(self, token: str):
        self.token = token
        self.session.headers.update({"Authorization": f"Bearer {token}"})
    
    {% for path, methods in spec.paths.items() %}
    {% for method, details in methods.items() %}
    def {{ details.operationId | default(method + '_' + path.replace('/', '_').replace('{', '').replace('}', '')) }}(self{% for param in details.parameters | default([]) %}, {{ param.name }}: {{ param.schema.type | default('Any') }}{% endfor %}{% if details.requestBody %}, data: Dict[str, Any] = None{% endif %}) -> Dict[str, Any]:
        \"\"\"{{ details.summary | default('API method') }}\"\"\"
        url = f"{self.base_url}{{ path }}"
        {% if details.parameters %}
        # Handle path parameters
        {% for param in details.parameters %}
        {% if param.in == 'path' %}
        url = url.replace("{{ '{' + param.name + '}' }}", str({{ param.name }}))
        {% endif %}
        {% endfor %}
        {% endif %}
        
        response = self.session.{{ method }}(url{% if details.requestBody %}, json=data{% endif %})
        response.raise_for_status()
        return response.json()
    
    {% endfor %}
    {% endfor %}
""")
    
    output_path = Path(output_dir) / "nfcs_api_client.py"
    with open(output_path, 'w') as f:
        f.write(python_template.render(spec=spec, spec_file=spec_file))
    
    print(f"Python client generated: {output_path}")

if __name__ == "__main__":
    generate_python_client("openapi-spec.yaml", "./generated")
```

## 📚 Documentation and Support

### API Documentation Website

Create a comprehensive documentation website with:

1. **Interactive API Explorer**: Swagger UI with authentication
2. **Code Examples**: Multi-language examples for all endpoints
3. **Tutorials**: Step-by-step guides for common use cases
4. **SDK Downloads**: Generated client libraries
5. **Changelog**: API version history and migration guides
6. **Support Forum**: Community support and discussions

### Integration Guides

- **Python Integration**: Flask, Django, FastAPI examples
- **JavaScript Integration**: React, Vue, Angular examples  
- **Mobile Integration**: iOS and Android SDK examples
- **Webhook Integration**: Real-time event processing
- **Batch Processing**: Bulk operations and data processing

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-09-15 | Initial API documentation and Swagger integration | Team Ω |

---

*This comprehensive API documentation provides everything needed to integrate with the NFCS system. For additional support, contact the API team at api-support@nfcs.internal.*

_Last updated: 2025-09-15 by Team Ω_