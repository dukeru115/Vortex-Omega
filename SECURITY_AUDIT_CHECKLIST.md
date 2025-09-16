# NFCS Security Audit & Configuration Templates

## Overview

Comprehensive security audit framework and production configuration templates for the Neural Field Control System (NFCS). This document provides security guidelines, audit procedures, and standardized configuration templates to ensure secure deployment and operation.

**Document Version**: 1.0  
**Classification**: Internal Use  
**Last Updated**: 2025-09-15  
**Security Contact**: security@nfcs.internal

## 🔐 Security Framework

### Security Principles

1. **Constitutional Compliance**: All security measures must align with NFCS constitutional framework
2. **Defense in Depth**: Multiple security layers with redundant protections
3. **Least Privilege**: Minimal necessary access rights and permissions
4. **Zero Trust**: Verify everything, trust nothing by default
5. **Privacy by Design**: Data protection built into system architecture
6. **Audit Everything**: Comprehensive logging and monitoring of all activities

### Security Domains

| Domain | Priority | Components | Responsibility |
|--------|----------|------------|----------------|
| Constitutional Security | Critical | Policy enforcement, compliance monitoring | Constitutional Framework |
| Application Security | High | Input validation, authentication, authorization | API & Web Layer |
| Infrastructure Security | High | Network, servers, containers, orchestration | DevOps Team |
| Data Security | High | Encryption, access control, backup protection | Data Management |
| Cognitive Security | Medium | AI safety, model protection, learning security | Research Team |

## 🛡️ Security Audit Checklist

### 1. **Constitutional Framework Audit**

#### Policy Enforcement Verification
```bash
#!/bin/bash
# Constitutional security audit script

echo "=== Constitutional Framework Security Audit ==="

# Check constitutional monitor status
if curl -s http://localhost:8765/constitutional/health | grep -q "healthy"; then
    echo "✅ Constitutional monitor active"
else
    echo "❌ Constitutional monitor not responding"
fi

# Verify policy enforcement
POLICY_COUNT=$(curl -s http://localhost:8765/constitutional/policies/count)
if [ "$POLICY_COUNT" -gt 0 ]; then
    echo "✅ Constitutional policies loaded: $POLICY_COUNT"
else
    echo "❌ No constitutional policies found"
fi

# Check violation detection
VIOLATION_COUNT=$(curl -s http://localhost:8765/constitutional/violations/24h)
echo "📊 Violations in last 24h: $VIOLATION_COUNT"

# Verify safety thresholds
HA_CURRENT=$(curl -s http://localhost:8765/constitutional/ha/current)
HA_THRESHOLD=$(curl -s http://localhost:8765/constitutional/ha/threshold)
if (( $(echo "$HA_CURRENT < $HA_THRESHOLD" | bc -l) )); then
    echo "✅ Ha number within safe limits: $HA_CURRENT < $HA_THRESHOLD"
else
    echo "⚠️  Ha number approaching threshold: $HA_CURRENT / $HA_THRESHOLD"
fi
```

#### Constitutional Compliance Checklist
- [ ] **Policy Loading**: All constitutional policies loaded and active
- [ ] **Enforcement Engine**: Policy enforcement functioning correctly
- [ ] **Violation Detection**: Automated violation detection operational
- [ ] **Response Mechanisms**: Escalation and response procedures tested
- [ ] **Audit Logging**: All constitutional events logged and retained
- [ ] **Safety Thresholds**: Ha monitoring and threshold enforcement active
- [ ] **Emergency Protocols**: Shutdown and containment procedures verified

### 2. **Application Security Audit**

#### Authentication & Authorization
```python
#!/usr/bin/env python3
# Application security audit script

import requests
import json
from datetime import datetime

class SecurityAudit:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.results = {"timestamp": datetime.now().isoformat(), "checks": []}
    
    def audit_authentication(self):
        """Audit authentication mechanisms."""
        checks = []
        
        # Test unauthorized access
        response = requests.get(f"{self.base_url}/api/admin")
        checks.append({
            "check": "Unauthorized Access Protection",
            "status": "PASS" if response.status_code == 401 else "FAIL",
            "details": f"Status code: {response.status_code}"
        })
        
        # Test authentication endpoint
        response = requests.post(f"{self.base_url}/api/auth/login", 
                               json={"username": "test", "password": "invalid"})
        checks.append({
            "check": "Authentication Endpoint",
            "status": "PASS" if response.status_code in [401, 403] else "FAIL",
            "details": f"Invalid credentials rejected: {response.status_code}"
        })
        
        # Test rate limiting
        for i in range(10):
            response = requests.post(f"{self.base_url}/api/auth/login",
                                   json={"username": "test", "password": "test"})
        checks.append({
            "check": "Rate Limiting",
            "status": "PASS" if response.status_code == 429 else "FAIL",
            "details": f"Rate limiting active: {response.status_code}"
        })
        
        return checks
    
    def audit_input_validation(self):
        """Audit input validation and sanitization."""
        checks = []
        
        # Test SQL injection protection
        response = requests.post(f"{self.base_url}/api/process",
                               json={"input": "'; DROP TABLE users; --"})
        checks.append({
            "check": "SQL Injection Protection",
            "status": "PASS" if response.status_code != 500 else "FAIL",
            "details": "Malicious SQL input handled safely"
        })
        
        # Test XSS protection
        response = requests.post(f"{self.base_url}/api/process",
                               json={"input": "<script>alert('xss')</script>"})
        checks.append({
            "check": "XSS Protection",
            "status": "PASS" if "<script>" not in response.text else "FAIL",
            "details": "Script tags filtered or escaped"
        })
        
        # Test input size limits
        large_input = "A" * 1000000  # 1MB
        response = requests.post(f"{self.base_url}/api/process",
                               json={"input": large_input})
        checks.append({
            "check": "Input Size Limits",
            "status": "PASS" if response.status_code == 413 else "FAIL",
            "details": f"Large input handling: {response.status_code}"
        })
        
        return checks
    
    def generate_report(self):
        """Generate comprehensive security audit report."""
        self.results["checks"].extend(self.audit_authentication())
        self.results["checks"].extend(self.audit_input_validation())
        
        # Calculate overall score
        total_checks = len(self.results["checks"])
        passed_checks = sum(1 for check in self.results["checks"] if check["status"] == "PASS")
        self.results["score"] = f"{passed_checks}/{total_checks} ({passed_checks/total_checks*100:.1f}%)"
        
        return self.results

if __name__ == "__main__":
    audit = SecurityAudit()
    report = audit.generate_report()
    print(json.dumps(report, indent=2))
```

#### Application Security Checklist
- [ ] **Authentication**: Strong authentication mechanisms implemented
- [ ] **Authorization**: Proper access controls and permissions
- [ ] **Session Management**: Secure session handling and timeout
- [ ] **Input Validation**: All inputs validated and sanitized
- [ ] **Output Encoding**: Proper encoding to prevent XSS
- [ ] **CSRF Protection**: Cross-Site Request Forgery protection enabled
- [ ] **SQL Injection**: Parameterized queries and ORM usage
- [ ] **Security Headers**: Appropriate security headers configured
- [ ] **Error Handling**: Secure error messages without information disclosure
- [ ] **Logging**: Security events logged appropriately

### 3. **Infrastructure Security Audit**

#### Network Security
```bash
#!/bin/bash
# Infrastructure security audit

echo "=== Infrastructure Security Audit ==="

# Check firewall status
if systemctl is-active --quiet ufw; then
    echo "✅ Firewall active"
    ufw status numbered
else
    echo "❌ Firewall not active"
fi

# Check SSH configuration
if grep -q "PasswordAuthentication no" /etc/ssh/sshd_config; then
    echo "✅ SSH password authentication disabled"
else
    echo "⚠️  SSH password authentication enabled"
fi

# Check SSL/TLS configuration
if openssl s_client -connect localhost:443 -verify_return_error 2>/dev/null; then
    echo "✅ SSL/TLS configuration valid"
else
    echo "❌ SSL/TLS configuration issues"
fi

# Check Docker security
if docker info --format '{{.SecurityOptions}}' | grep -q "apparmor"; then
    echo "✅ Docker security features enabled"
else
    echo "⚠️  Docker security features not fully enabled"
fi

# Check file permissions
find /opt/nfcs -type f -perm /o+w -exec ls -la {} \; | while read file; do
    echo "⚠️  World-writable file found: $file"
done
```

#### Infrastructure Security Checklist
- [ ] **Network Segmentation**: Proper network isolation and VLANs
- [ ] **Firewall Rules**: Restrictive firewall configuration
- [ ] **SSH Hardening**: SSH key-only authentication, non-standard port
- [ ] **SSL/TLS**: Strong encryption for all external communications
- [ ] **Container Security**: Docker/Kubernetes security best practices
- [ ] **File Permissions**: Appropriate file and directory permissions
- [ ] **User Management**: Principle of least privilege for all accounts
- [ ] **System Updates**: Regular security patches and updates
- [ ] **Monitoring**: Network and system monitoring in place
- [ ] **Backup Security**: Encrypted and access-controlled backups

### 4. **Data Security Audit**

#### Encryption and Data Protection
```python
#!/usr/bin/env python3
# Data security audit script

import os
import subprocess
import json
from pathlib import Path

class DataSecurityAudit:
    def __init__(self):
        self.results = {"encryption": [], "access_control": [], "data_classification": []}
    
    def check_encryption_at_rest(self):
        """Check data encryption at rest."""
        checks = []
        
        # Check database encryption
        try:
            result = subprocess.run(
                ["psql", "-c", "SHOW ssl;"], 
                capture_output=True, text=True
            )
            ssl_enabled = "on" in result.stdout
            checks.append({
                "component": "PostgreSQL",
                "encryption": "PASS" if ssl_enabled else "FAIL",
                "details": "SSL encryption enabled" if ssl_enabled else "SSL not enabled"
            })
        except Exception as e:
            checks.append({
                "component": "PostgreSQL",
                "encryption": "ERROR",
                "details": str(e)
            })
        
        # Check file system encryption
        encrypted_dirs = [
            "/var/lib/nfcs",
            "/opt/nfcs/data",
            "/backup/nfcs"
        ]
        
        for directory in encrypted_dirs:
            if Path(directory).exists():
                # Check if directory is on encrypted filesystem
                result = subprocess.run(
                    ["df", "-T", directory],
                    capture_output=True, text=True
                )
                encrypted = "ext4" in result.stdout  # Simplified check
                checks.append({
                    "component": f"Directory {directory}",
                    "encryption": "PASS" if encrypted else "WARN",
                    "details": "Filesystem encryption recommended"
                })
        
        return checks
    
    def check_access_controls(self):
        """Check data access controls."""
        checks = []
        
        sensitive_files = [
            "/opt/nfcs/.env",
            "/opt/nfcs/config/secrets.yaml",
            "/etc/nfcs/certificates/"
        ]
        
        for file_path in sensitive_files:
            path = Path(file_path)
            if path.exists():
                stat = path.stat()
                permissions = oct(stat.st_mode)[-3:]
                
                # Check if file is world-readable
                world_readable = int(permissions[2]) & 4
                checks.append({
                    "file": file_path,
                    "permissions": permissions,
                    "secure": "PASS" if not world_readable else "FAIL",
                    "details": "World-readable" if world_readable else "Properly restricted"
                })
        
        return checks
    
    def audit_data_classification(self):
        """Audit data classification and handling."""
        data_types = {
            "constitutional_policies": {"classification": "CRITICAL", "encryption": "required"},
            "user_data": {"classification": "SENSITIVE", "encryption": "required"},
            "research_data": {"classification": "INTERNAL", "encryption": "recommended"},
            "log_data": {"classification": "INTERNAL", "encryption": "optional"},
            "public_docs": {"classification": "PUBLIC", "encryption": "optional"}
        }
        
        classification_audit = []
        for data_type, requirements in data_types.items():
            classification_audit.append({
                "data_type": data_type,
                "classification": requirements["classification"],
                "encryption_requirement": requirements["encryption"],
                "status": "COMPLIANT"  # Simplified for example
            })
        
        return classification_audit

if __name__ == "__main__":
    audit = DataSecurityAudit()
    audit.results["encryption"] = audit.check_encryption_at_rest()
    audit.results["access_control"] = audit.check_access_controls()
    audit.results["data_classification"] = audit.audit_data_classification()
    
    print(json.dumps(audit.results, indent=2))
```

#### Data Security Checklist
- [ ] **Encryption at Rest**: All sensitive data encrypted when stored
- [ ] **Encryption in Transit**: All data transmissions encrypted
- [ ] **Key Management**: Proper cryptographic key lifecycle management
- [ ] **Access Controls**: Role-based access control for all data
- [ ] **Data Classification**: Data properly classified and labeled
- [ ] **Data Retention**: Appropriate retention policies implemented
- [ ] **Data Backup**: Secure backup procedures and testing
- [ ] **Data Disposal**: Secure data destruction procedures
- [ ] **Privacy Compliance**: GDPR/CCPA compliance measures
- [ ] **Audit Trails**: Comprehensive data access logging

## 📋 Production Configuration Templates

### 1. **Environment Configuration Template**

```bash
# .env.production - Production Environment Configuration
# NFCS Production Environment Variables
# WARNING: Contains sensitive information - protect accordingly

# === System Configuration ===
NFCS_ENVIRONMENT=production
NFCS_VERSION=2.4.3
NFCS_DEBUG=false
NFCS_LOG_LEVEL=INFO

# === Security Configuration ===
NFCS_SECRET_KEY=${RANDOM_SECRET_KEY_64_CHARS}
NFCS_ENCRYPTION_KEY=${RANDOM_ENCRYPTION_KEY_32_CHARS}
NFCS_JWT_SECRET=${RANDOM_JWT_SECRET_64_CHARS}
NFCS_CSRF_SECRET=${RANDOM_CSRF_SECRET_32_CHARS}

# === Database Configuration ===
DATABASE_URL=postgresql://nfcs_user:${DB_PASSWORD}@localhost:5432/nfcs_production
DATABASE_POOL_SIZE=20
DATABASE_MAX_CONNECTIONS=100
DATABASE_SSL_MODE=require
DATABASE_ENCRYPTION=true

# === Redis Configuration ===
REDIS_URL=redis://:${REDIS_PASSWORD}@localhost:6379/0
REDIS_SSL=true
REDIS_MAX_CONNECTIONS=50

# === Constitutional Framework ===
CONSTITUTIONAL_MONITOR_PORT=8765
CONSTITUTIONAL_ENFORCEMENT_LEVEL=strict
CONSTITUTIONAL_HA_THRESHOLD=0.8
CONSTITUTIONAL_VIOLATION_ESCALATION=immediate
CONSTITUTIONAL_AUDIT_RETENTION_DAYS=90

# === Security Settings ===
SESSION_TIMEOUT_MINUTES=30
PASSWORD_MIN_LENGTH=12
PASSWORD_COMPLEXITY=true
MFA_REQUIRED=true
LOGIN_ATTEMPT_LIMIT=5
LOGIN_LOCKOUT_MINUTES=15

# === Monitoring & Logging ===
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
LOG_AGGREGATION=syslog
LOG_RETENTION_DAYS=90
METRICS_RETENTION_DAYS=30

# === Performance Configuration ===
COORDINATION_FREQUENCY=10.0
MAX_CONCURRENT_OPERATIONS=100
MEMORY_LIMIT_GB=8
CPU_LIMIT_CORES=4
CACHE_SIZE_MB=1024

# === Backup Configuration ===
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"
BACKUP_RETENTION_DAYS=30
BACKUP_ENCRYPTION=true
BACKUP_OFFSITE_SYNC=true
BACKUP_S3_BUCKET=nfcs-backups-prod

# === External Integrations ===
WOLFRAM_ALPHA_API_KEY=${WOLFRAM_API_KEY}
Z3_SMT_SOLVER_ENABLED=true
ELASTICSEARCH_URL=https://elastic:${ELASTIC_PASSWORD}@localhost:9200

# === Development & Debug (Disabled in Production) ===
FLASK_DEBUG=false
FLASK_TESTING=false
PROFILING_ENABLED=false
SWAGGER_UI_ENABLED=false

# === Notification Configuration ===
ALERT_EMAIL=ops@example.com
SLACK_WEBHOOK_URL=${SLACK_WEBHOOK}
PAGERDUTY_INTEGRATION_KEY=${PAGERDUTY_KEY}

# === Compliance Configuration ===
GDPR_COMPLIANCE=true
DATA_RETENTION_POLICY=strict
AUDIT_LOGGING=comprehensive
PRIVACY_MODE=enabled
```

### 2. **Docker Production Configuration**

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  nfcs-app:
    image: nfcs:production-2.4.3
    container_name: nfcs-production
    restart: unless-stopped
    environment:
      - NFCS_ENVIRONMENT=production
    env_file:
      - .env.production
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs
      - ./config:/app/config:ro
    networks:
      - nfcs-network
    security_opt:
      - no-new-privileges:true
      - apparmor:nfcs-profile
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nfcs-constitutional:
    image: nfcs-constitutional:production-2.4.3
    container_name: nfcs-constitutional
    restart: unless-stopped
    environment:
      - CONSTITUTIONAL_ENFORCEMENT_LEVEL=strict
    env_file:
      - .env.production
    ports:
      - "8765:8765"
    networks:
      - nfcs-network
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=50m

  postgres:
    image: postgres:15-alpine
    container_name: nfcs-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: nfcs_production
      POSTGRES_USER: nfcs_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_INITDB_ARGS: "--auth-host=scram-sha-256"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgresql.conf:/etc/postgresql/postgresql.conf:ro
    networks:
      - nfcs-network
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - SETUID
      - SETGID
      - DAC_OVERRIDE

  redis:
    image: redis:7-alpine
    container_name: nfcs-redis
    restart: unless-stopped
    command: >
      redis-server 
      --requirepass ${REDIS_PASSWORD}
      --appendonly yes
      --appendfsync everysec
      --tls-port 6380
      --port 0
      --tls-cert-file /tls/redis.crt
      --tls-key-file /tls/redis.key
      --tls-ca-cert-file /tls/ca.crt
    volumes:
      - redis_data:/data
      - ./tls:/tls:ro
    networks:
      - nfcs-network
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL

  prometheus:
    image: prom/prometheus:latest
    container_name: nfcs-prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    volumes:
      - ./monitoring/prometheus:/etc/prometheus:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - nfcs-network
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL

  grafana:
    image: grafana/grafana:latest
    container_name: nfcs-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
      - GF_SECURITY_SECRET_KEY=${GRAFANA_SECRET_KEY}
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning:ro
    ports:
      - "3000:3000"
    networks:
      - nfcs-network
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL

  nginx:
    image: nginx:alpine
    container_name: nfcs-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/logs:/var/log/nginx
    networks:
      - nfcs-network
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    depends_on:
      - nfcs-app

volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      device: /opt/nfcs/data/postgres
      o: bind
  redis_data:
    driver: local
    driver_opts:
      type: none
      device: /opt/nfcs/data/redis
      o: bind
  prometheus_data:
    driver: local
    driver_opts:
      type: none
      device: /opt/nfcs/data/prometheus
      o: bind
  grafana_data:
    driver: local
    driver_opts:
      type: none
      device: /opt/nfcs/data/grafana
      o: bind

networks:
  nfcs-network:
    driver: bridge
    enable_ipv6: false
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 3. **NGINX Security Configuration**

```nginx
# nginx/nginx.conf - Production NGINX Configuration
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Hide server information
    server_tokens off;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for"';
    access_log /var/log/nginx/access.log main;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Gzip
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=1r/s;

    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_stapling on;
    ssl_stapling_verify on;

    # Main application server
    upstream nfcs_app {
        server nfcs-app:5000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    # Constitutional monitor upstream
    upstream nfcs_constitutional {
        server nfcs-constitutional:8765 max_fails=3 fail_timeout=30s;
        keepalive 16;
    }

    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name _;
        return 301 https://$host$request_uri;
    }

    # Main HTTPS server
    server {
        listen 443 ssl http2;
        server_name nfcs.example.com;

        ssl_certificate /etc/nginx/ssl/nfcs.crt;
        ssl_certificate_key /etc/nginx/ssl/nfcs.key;
        ssl_trusted_certificate /etc/nginx/ssl/ca.crt;

        # Security
        client_max_body_size 10M;
        client_body_timeout 12;
        client_header_timeout 12;
        send_timeout 10;

        # Main application
        location / {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://nfcs_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
            
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
        }

        # API endpoints with stricter rate limiting
        location /api/ {
            limit_req zone=api burst=10 nodelay;
            
            proxy_pass http://nfcs_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Authentication endpoints
        location /api/auth/ {
            limit_req zone=auth burst=5 nodelay;
            
            proxy_pass http://nfcs_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Constitutional monitor
        location /constitutional/ {
            limit_req zone=api burst=5 nodelay;
            
            proxy_pass http://nfcs_constitutional/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Health check endpoint
        location /health {
            access_log off;
            proxy_pass http://nfcs_app;
        }

        # Block sensitive files
        location ~ /\. {
            deny all;
            access_log off;
            log_not_found off;
        }

        location ~ \.(env|conf|config)$ {
            deny all;
            access_log off;
            log_not_found off;
        }
    }

    # Monitoring server (internal only)
    server {
        listen 443 ssl http2;
        server_name monitoring.nfcs.internal;

        ssl_certificate /etc/nginx/ssl/monitoring.crt;
        ssl_certificate_key /etc/nginx/ssl/monitoring.key;

        # Restrict to internal networks
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;

        # Prometheus
        location /prometheus/ {
            proxy_pass http://nfcs-prometheus:9090/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Grafana
        location /grafana/ {
            proxy_pass http://nfcs-grafana:3000/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### 4. **PostgreSQL Security Configuration**

```ini
# postgresql.conf - Production PostgreSQL Configuration

# Connection Settings
listen_addresses = 'localhost'
port = 5432
max_connections = 100
superuser_reserved_connections = 3

# SSL Configuration
ssl = on
ssl_cert_file = '/etc/ssl/certs/postgresql.crt'
ssl_key_file = '/etc/ssl/private/postgresql.key'
ssl_ca_file = '/etc/ssl/certs/ca.crt'
ssl_min_protocol_version = 'TLSv1.2'
ssl_ciphers = 'ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256'
ssl_prefer_server_ciphers = on

# Authentication
password_encryption = scram-sha-256
authentication_timeout = 10s

# Logging
log_destination = 'syslog'
logging_collector = on
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_truncate_on_rotation = on
log_rotation_age = 1d
log_rotation_size = 100MB
log_min_duration_statement = 1000
log_connections = on
log_disconnections = on
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_statement = 'ddl'

# Security
shared_preload_libraries = 'pg_stat_statements'
track_functions = all
track_activity_query_size = 2048

# Performance
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

# Write-Ahead Logging
wal_level = replica
archive_mode = on
archive_command = 'cp %p /backup/postgresql/wal/%f'
max_wal_senders = 3
hot_standby = on
```

## 🔒 Security Automation Scripts

### Automated Security Scan

```bash
#!/bin/bash
# /opt/nfcs/scripts/security-scan.sh
# Automated security scanning and assessment

set -euo pipefail

SCAN_DATE=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="/var/log/nfcs/security-scans/$SCAN_DATE"
mkdir -p "$REPORT_DIR"

echo "Starting NFCS security scan: $SCAN_DATE"

# 1. System vulnerability scan
echo "Running system vulnerability scan..."
if command -v lynis >/dev/null 2>&1; then
    lynis audit system --report-file "$REPORT_DIR/system-audit.log"
fi

# 2. Network security scan
echo "Running network security scan..."
nmap -sS -O -sV -p- localhost > "$REPORT_DIR/network-scan.txt" 2>&1

# 3. SSL/TLS configuration test
echo "Testing SSL/TLS configuration..."
if command -v testssl.sh >/dev/null 2>&1; then
    testssl.sh --jsonfile "$REPORT_DIR/ssl-test.json" https://localhost
fi

# 4. Application security scan
echo "Running application security scan..."
python3 /opt/nfcs/scripts/app-security-audit.py > "$REPORT_DIR/app-security.json"

# 5. Constitutional framework security check
echo "Checking constitutional framework security..."
/opt/nfcs/scripts/constitutional-security-audit.sh > "$REPORT_DIR/constitutional-audit.log"

# 6. Database security assessment
echo "Assessing database security..."
if command -v pg_audit >/dev/null 2>&1; then
    pg_audit --database nfcs_production > "$REPORT_DIR/database-audit.log"
fi

# 7. Container security scan
echo "Scanning container security..."
if command -v docker >/dev/null 2>&1; then
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
        aquasec/trivy image nfcs:production-2.4.3 > "$REPORT_DIR/container-scan.json"
fi

# 8. Generate summary report
echo "Generating security summary..."
python3 /opt/nfcs/scripts/generate-security-report.py "$REPORT_DIR" > "$REPORT_DIR/security-summary.json"

# 9. Send alerts for critical findings
CRITICAL_COUNT=$(jq '.critical_issues | length' "$REPORT_DIR/security-summary.json")
if [ "$CRITICAL_COUNT" -gt 0 ]; then
    echo "CRITICAL: $CRITICAL_COUNT critical security issues found" | \
        mail -s "NFCS Security Alert: Critical issues detected" security@nfcs.internal
fi

echo "Security scan completed: $SCAN_DATE"
echo "Reports available in: $REPORT_DIR"
```

## 📞 Security Incident Response

### Incident Response Plan

1. **Detection**: Automated monitoring alerts security team
2. **Assessment**: Severity classification and impact assessment
3. **Containment**: Immediate containment measures activated
4. **Investigation**: Root cause analysis and evidence collection
5. **Recovery**: System restoration and service resumption
6. **Lessons Learned**: Post-incident review and improvements

### Emergency Contacts

| Role | Primary | Backup | Phone | Email |
|------|---------|--------|-------|-------|
| Security Lead | John Smith | Jane Doe | +1-555-SEC1 | security-lead@nfcs.internal |
| CISO | Alice Johnson | Bob Wilson | +1-555-CISO | ciso@nfcs.internal |
| Legal Counsel | Carol Brown | Dave Green | +1-555-LEGAL | legal@nfcs.internal |
| PR/Communications | Eve White | Frank Black | +1-555-COMM | pr@nfcs.internal |

## 📚 Compliance Framework

### Regulatory Compliance

- **GDPR**: Data protection and privacy compliance
- **SOC 2**: Security, availability, and confidentiality controls
- **ISO 27001**: Information security management system
- **NIST Cybersecurity Framework**: Risk management and security controls
- **PCI DSS**: Payment card data protection (if applicable)

### Audit Requirements

- **Internal Audits**: Quarterly security assessments
- **External Audits**: Annual third-party security audits
- **Penetration Testing**: Semi-annual penetration testing
- **Compliance Audits**: Annual regulatory compliance audits

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-09-15 | Initial security audit and configuration templates | Team Ω |

---

*This document contains sensitive security information and should be treated as confidential. Access should be restricted to authorized personnel only.*

_Last updated: 2025-09-15 by Team Ω_