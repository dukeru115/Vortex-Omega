# 🌐 NGINX Configuration for NFCS

## Overview
NGINX reverse proxy and load balancing configuration for the Neural Field Control System (NFCS) v2.4.3 production deployment.

## 🏗️ Architecture

### Current MVP Setup
```
Internet → NGINX → Flask MVP (Port 5000)
```

### Production Architecture  
```
Internet → NGINX Load Balancer
    ├── API Servers (Port 8000-8003)
    ├── WebSocket Servers (Port 8765-8768) 
    ├── Dashboard (Port 5000)
    └── Static Assets (CDN)
```

## 📁 Configuration Files

### Core Configuration
- **nginx.conf**: Main NGINX configuration
- **sites-available/nfcs**: NFCS site configuration
- **sites-available/nfcs-api**: API-specific configuration
- **sites-available/nfcs-ws**: WebSocket configuration

### SSL/TLS
- **ssl/**: SSL certificates and configuration
- **dhparam.pem**: Diffie-Hellman parameters
- **ssl-params.conf**: SSL security parameters

### Security
- **security-headers.conf**: Security headers
- **rate-limiting.conf**: DDoS protection
- **ip-whitelist.conf**: IP access control

## 🚀 MVP Configuration

### Basic MVP Proxy (`sites-available/nfcs-mvp`)
```nginx
server {
    listen 80;
    server_name nfcs-mvp.local localhost;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    
    # Proxy to MVP Flask application
    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Socket.IO WebSocket support
    location /socket.io/ {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket specific timeouts
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

## 🔒 Production SSL Configuration

### HTTPS Redirect
```nginx
server {
    listen 80;
    server_name nfcs.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### SSL Termination
```nginx
server {
    listen 443 ssl http2;
    server_name nfcs.yourdomain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/nfcs.crt;
    ssl_certificate_key /etc/ssl/private/nfcs.key;
    ssl_dhparam /etc/ssl/certs/dhparam.pem;
    
    # SSL Security Parameters
    include /etc/nginx/ssl-params.conf;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # API Gateway
    location /api/ {
        proxy_pass http://nfcs_api_upstream;
        include /etc/nginx/proxy-params.conf;
    }
    
    # WebSocket Gateway
    location /ws/ {
        proxy_pass http://nfcs_ws_upstream;
        include /etc/nginx/websocket-params.conf;
    }
    
    # Dashboard
    location / {
        proxy_pass http://nfcs_web_upstream;
        include /etc/nginx/proxy-params.conf;
    }
}
```

## ⚖️ Load Balancing Configuration

### Upstream Definitions
```nginx
# API Server Pool
upstream nfcs_api_upstream {
    least_conn;
    server localhost:8000 max_fails=3 fail_timeout=30s;
    server localhost:8001 max_fails=3 fail_timeout=30s;
    server localhost:8002 max_fails=3 fail_timeout=30s;
    server localhost:8003 max_fails=3 fail_timeout=30s;
    
    # Health checks (NGINX Plus)
    # health_check;
}

# WebSocket Server Pool
upstream nfcs_ws_upstream {
    ip_hash; # Sticky sessions for WebSocket
    server localhost:8765 max_fails=2 fail_timeout=30s;
    server localhost:8766 max_fails=2 fail_timeout=30s;
    server localhost:8767 max_fails=2 fail_timeout=30s;
    server localhost:8768 max_fails=2 fail_timeout=30s;
}

# Web Dashboard Pool
upstream nfcs_web_upstream {
    server localhost:5000 max_fails=3 fail_timeout=30s;
    server localhost:5001 backup; # Backup instance
}
```

### Health Checks
```nginx
# Custom health check location
location /nginx-health {
    access_log off;
    return 200 "healthy\n";
    add_header Content-Type text/plain;
    
    # Only allow internal access
    allow 127.0.0.1;
    allow 10.0.0.0/8;
    deny all;
}

# Upstream health monitoring
location /upstream-status {
    access_log off;
    allow 127.0.0.1;
    deny all;
    
    # Return upstream status (requires lua module)
    content_by_lua_block {
        local upstream = require "ngx.upstream"
        local servers = upstream.get_servers("nfcs_api_upstream")
        
        ngx.header["Content-Type"] = "application/json"
        ngx.say(cjson.encode(servers))
    }
}
```

## 🛡️ Security Configuration

### Rate Limiting
```nginx
# Rate limiting zones
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=1r/s;
    limit_req_zone $binary_remote_addr zone=general:10m rate=5r/s;
    
    # Connection limiting
    limit_conn_zone $binary_remote_addr zone=addr:10m;
}

# Apply rate limiting
server {
    # API endpoints
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        limit_conn addr 10;
        proxy_pass http://nfcs_api_upstream;
    }
    
    # Authentication endpoints
    location /api/auth/ {
        limit_req zone=auth burst=5 nodelay;
        limit_conn addr 5;
        proxy_pass http://nfcs_api_upstream;
    }
    
    # General web traffic
    location / {
        limit_req zone=general burst=10 nodelay;
        limit_conn addr 10;
        proxy_pass http://nfcs_web_upstream;
    }
}
```

### Security Headers
```nginx
# /etc/nginx/security-headers.conf
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' fonts.gstatic.com; connect-src 'self' wss:;" always;
add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;

# Remove server signature
server_tokens off;
```

### IP Whitelisting (Optional)
```nginx
# /etc/nginx/ip-whitelist.conf
# Admin interface access restriction
location /admin/ {
    allow 192.168.1.0/24;  # Internal network
    allow 10.0.0.0/8;      # Private network
    deny all;
    
    proxy_pass http://nfcs_web_upstream;
}
```

## 📊 Monitoring & Logging

### Access Logging
```nginx
# Custom log format for NFCS
log_format nfcs_combined '$remote_addr - $remote_user [$time_local] '
                         '"$request" $status $body_bytes_sent '
                         '"$http_referer" "$http_user_agent" '
                         '$request_time $upstream_response_time '
                         '$upstream_addr $upstream_status';

server {
    access_log /var/log/nginx/nfcs.access.log nfcs_combined;
    error_log /var/log/nginx/nfcs.error.log warn;
}
```

### Status Module Configuration
```nginx
# NGINX status page
server {
    listen 127.0.0.1:8080;
    server_name localhost;
    
    location /nginx_status {
        stub_status on;
        access_log off;
        allow 127.0.0.1;
        deny all;
    }
    
    location /upstream_status {
        upstream_status;
        access_log off;
        allow 127.0.0.1;
        deny all;
    }
}
```

## 🚀 Performance Optimization

### Caching Configuration
```nginx
# Cache zones
proxy_cache_path /var/cache/nginx/nfcs levels=1:2 keys_zone=nfcs_cache:10m max_size=1g inactive=60m use_temp_path=off;

server {
    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header Vary Accept-Encoding;
        
        # Enable compression
        gzip_static on;
    }
    
    # Cache API responses (selective)
    location /api/status {
        proxy_cache nfcs_cache;
        proxy_cache_valid 200 30s;
        proxy_cache_key "$scheme$request_method$host$request_uri";
        proxy_cache_bypass $http_cache_control;
        
        add_header X-Cache-Status $upstream_cache_status;
        proxy_pass http://nfcs_api_upstream;
    }
}
```

### Compression
```nginx
# Gzip compression
gzip on;
gzip_vary on;
gzip_min_length 1024;
gzip_proxied any;
gzip_comp_level 6;
gzip_types
    text/plain
    text/css
    text/xml
    text/javascript
    application/json
    application/javascript
    application/xml+rss
    application/atom+xml
    image/svg+xml;
```

## 🔧 Deployment Scripts

### Basic Setup Script
```bash
#!/bin/bash
# NGINX setup for NFCS MVP

# Copy configuration files
sudo cp nginx/sites-available/nfcs-mvp /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/nfcs-mvp /etc/nginx/sites-enabled/

# Copy security configurations
sudo cp nginx/security-headers.conf /etc/nginx/
sudo cp nginx/ssl-params.conf /etc/nginx/

# Test configuration
sudo nginx -t

# Reload NGINX
sudo systemctl reload nginx

echo "NGINX configured for NFCS MVP"
echo "Access: http://localhost"
```

### SSL Certificate Setup (Let's Encrypt)
```bash
#!/bin/bash
# SSL certificate setup with Certbot

# Install Certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d nfcs.yourdomain.com

# Set up auto-renewal
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer

echo "SSL certificate installed and auto-renewal configured"
```

## 📈 Monitoring Integration

### Prometheus Metrics Export
```nginx
# NGINX Prometheus exporter
server {
    listen 9113;
    server_name localhost;
    
    location /metrics {
        access_log off;
        allow 127.0.0.1;
        allow 10.0.0.0/8;
        deny all;
        
        # nginx-prometheus-exporter
        stub_status on;
    }
}
```

### Log Analysis with ELK Stack
```bash
# Filebeat configuration for NGINX logs
# /etc/filebeat/conf.d/nginx.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/nginx/nfcs.access.log
  fields:
    service: nfcs
    component: nginx
  fields_under_root: true
```

## 🔗 Integration Points

### Docker Integration
```yaml
# docker-compose.yml snippet
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/sites-available:/etc/nginx/sites-available:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - nfcs-mvp
      - nfcs-api
```

### Kubernetes Integration
```yaml
# nginx-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config
data:
  nginx.conf: |
    # NGINX configuration for Kubernetes
    events { worker_connections 1024; }
    http {
      upstream nfcs-backend {
        server nfcs-api-service:8000;
      }
      
      server {
        listen 80;
        location / {
          proxy_pass http://nfcs-backend;
        }
      }
    }
```

## 🔗 Related Documentation
- [MVP Web Interface](../mvp_web_interface.py)
- [System Architecture](../docs/README.md)
- [Production Deployment](../README.md#deployment)

---
*Part of Vortex-Omega Neural Field Control System v2.4.3*