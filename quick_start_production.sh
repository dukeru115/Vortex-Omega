#!/bin/bash
# Vortex-Omega Production Quick Start Script
# Neural Field Control System v2.4.3

set -e

echo "🚀 Starting Vortex-Omega NFCS Production Deployment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${YELLOW}⚠️  Port $port is already in use${NC}"
        return 1
    else
        return 0
    fi
}

# Check required ports
echo -e "${BLUE}🔍 Checking port availability...${NC}"
PORTS=(80 443 3000 5432 6379 8080 9090 8765)  # Added 8765 for constitutional dashboard
for port in "${PORTS[@]}"; do
    if ! check_port $port; then
        echo -e "${RED}❌ Port $port is required but already in use${NC}"
        echo "Please stop the service using this port or modify docker-compose.yml"
        exit 1
    fi
done
echo -e "${GREEN}✅ All required ports are available${NC}"

# Create necessary directories
echo -e "${BLUE}📁 Creating directory structure...${NC}"
mkdir -p logs data monitoring/grafana/{dashboards,datasources} nginx/ssl static dashboard constitutional_data

# Generate self-signed SSL certificate if not exists
if [ ! -f "nginx/ssl/cert.pem" ]; then
    echo -e "${BLUE}🔒 Generating self-signed SSL certificate...${NC}"
    openssl req -x509 -newkey rsa:4096 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem \
        -days 365 -nodes -subj "/C=US/ST=State/L=City/O=VortexOmega/CN=localhost" 2>/dev/null || {
        echo -e "${YELLOW}⚠️  OpenSSL not available, using HTTP only${NC}"
        # Modify docker-compose to remove SSL ports if needed
    }
fi

# Set proper permissions
chmod +x scripts/*.sh 2>/dev/null || true
chmod 600 nginx/ssl/*.pem 2>/dev/null || true

# Check if .env file exists, create if not
if [ ! -f ".env" ]; then
    echo -e "${BLUE}⚙️  Creating .env file...${NC}"
    cat > .env << EOF
# Vortex-Omega Environment Configuration
NFCS_ENV=production
LOG_LEVEL=INFO
POSTGRES_PASSWORD=vortex123
REDIS_PASSWORD=
GRAFANA_PASSWORD=vortex123
KURAMOTO_COUPLING=0.5
ESC_MEMORY_LIMIT_MB=500
SYMBOLIC_CACHE_SIZE=1000
MAX_WORKERS=4
ENABLE_METRICS=true

# Constitutional Monitoring Configuration
ENABLE_CONSTITUTIONAL_MONITORING=true
CONSTITUTIONAL_DB_PATH=./constitutional_data/monitoring.db
CONSTITUTIONAL_DASHBOARD_PORT=8765
HA_WARNING_THRESHOLD=1.0
HA_CRITICAL_THRESHOLD=2.0
HA_EMERGENCY_THRESHOLD=4.0
INTEGRITY_WARNING_THRESHOLD=0.7
INTEGRITY_CRITICAL_THRESHOLD=0.5
ENABLE_EARLY_WARNING=true
WEBSOCKET_DASHBOARD=true
EOF
fi

# Function to run deployment
deploy_production() {
    echo -e "${BLUE}🐳 Building and starting Docker containers...${NC}"
    
    # Build images
    docker-compose build --parallel
    
    # Start core services first
    docker-compose up -d postgres redis
    
    # Wait for databases to be ready
    echo -e "${BLUE}⏳ Waiting for databases to be ready...${NC}"
    sleep 10
    
    # Start application
    docker-compose up -d vortex-omega
    
    # Wait for application to start
    echo -e "${BLUE}⏳ Waiting for Vortex-Omega to start...${NC}"
    sleep 15
    
    # Start monitoring
    docker-compose up -d prometheus grafana
    
    # Optionally start nginx (production profile)
    if [ "${WITH_NGINX:-false}" = "true" ]; then
        docker-compose --profile production up -d nginx
    fi
}

# Function to check deployment health
check_health() {
    echo -e "${BLUE}🏥 Checking deployment health...${NC}"
    
    # Check if containers are running
    local failed=0
    for service in postgres redis vortex-omega prometheus grafana; do
        if docker-compose ps $service | grep -q "Up"; then
            echo -e "${GREEN}✅ $service is running${NC}"
        else
            echo -e "${RED}❌ $service is not running${NC}"
            failed=1
        fi
    done
    
    # Check constitutional monitoring if enabled
    if [ "${ENABLE_CONSTITUTIONAL_MONITORING:-true}" = "true" ]; then
        echo -e "${BLUE}🏛️  Checking constitutional monitoring...${NC}"
        if curl -f http://localhost:8765 &>/dev/null; then
            echo -e "${GREEN}✅ Constitutional dashboard is accessible${NC}"
        else
            echo -e "${YELLOW}⚠️  Constitutional dashboard may still be starting${NC}"
        fi
    fi
    
    if [ $failed -eq 1 ]; then
        echo -e "${RED}❌ Some services failed to start${NC}"
        return 1
    fi
    
    # Check API health
    echo -e "${BLUE}🔍 Checking API health...${NC}"
    if curl -f http://localhost:8080/health &>/dev/null; then
        echo -e "${GREEN}✅ API is responding${NC}"
    else
        echo -e "${YELLOW}⚠️  API health check failed, but service may still be starting${NC}"
    fi
    
    return 0
}

# Function to show access information
show_access_info() {
    echo ""
    echo -e "${GREEN}🎉 Vortex-Omega NFCS is now running!${NC}"
    echo "=================================================="
    echo ""
    echo -e "${BLUE}📊 Service Endpoints:${NC}"
    echo "• Vortex-Omega API:         http://localhost:8080"
    echo "• API Documentation:        http://localhost:8080/docs"
    echo "• Health Check:             http://localhost:8080/health"
    echo "• Grafana Dashboard:        http://localhost:3000 (admin/vortex123)"
    echo "• Prometheus Metrics:       http://localhost:9090"
    echo "• Constitutional Monitor:   http://localhost:8765"
    echo "• Constitutional Dashboard: file://$(pwd)/dashboard/constitutional_monitor.html"
    echo ""
    echo -e "${BLUE}🔧 Management Commands:${NC}"
    echo "• View logs:           docker-compose logs -f vortex-omega"
    echo "• Stop services:       docker-compose down"
    echo "• Restart:             docker-compose restart vortex-omega"
    echo "• Scale API:           docker-compose up -d --scale vortex-omega=3"
    echo ""
    echo -e "${BLUE}🧪 NFCS Specific Features:${NC}"
    echo "• Hallucination Number monitoring available in Grafana"
    echo "• Real-time coherence metrics at /metrics endpoint"
    echo "• ESC semantic processing logs in PostgreSQL"
    echo "• Constitutional Module decisions logged"
    echo ""
    echo -e "${BLUE}🏛️  Constitutional Monitoring Features (NEW):${NC}"
    echo "• Real-time Ha monitoring with Algorithm 1 implementation"
    echo "• Early warning system with predictive alerts"
    echo "• Emergency protocol activation/deactivation"
    echo "• WebSocket dashboard for live monitoring"
    echo "• Constitutional compliance scoring"
    echo "• Automated threat level assessment"
    echo ""
}

# Main execution
echo -e "${BLUE}🏗️  Starting deployment process...${NC}"

# Deploy services
if deploy_production; then
    echo -e "${GREEN}✅ Deployment completed successfully${NC}"
else
    echo -e "${RED}❌ Deployment failed${NC}"
    exit 1
fi

# Health check
if check_health; then
    show_access_info
else
    echo -e "${YELLOW}⚠️  Deployment completed but some health checks failed${NC}"
    echo "Check logs with: docker-compose logs"
fi

echo ""
echo -e "${GREEN}🚀 Vortex-Omega NFCS is ready for Neural Field Control!${NC}"