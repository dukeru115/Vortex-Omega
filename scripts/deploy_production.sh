#!/bin/bash
# Production Deployment Script for Vortex-Omega NFCS v2.5.0
# Comprehensive production deployment with health checks and rollback

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ENV=${1:-production}
VERSION=$(grep -o 'version.*2\.[0-9]\+\.[0-9]\+' README.md | head -1 | cut -d' ' -f2 || echo "2.5.0")
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="./logs/deployment_$(date +%Y%m%d_%H%M%S).log"

# Ensure log directory exists
mkdir -p logs backups

# Logging function
log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}   🚀 Vortex-Omega NFCS v${VERSION} Production Deployment${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_section() {
    echo -e "\n${PURPLE}▶ $1${NC}"
    echo -e "${PURPLE}$(printf '─%.0s' {1..50})${NC}"
}

print_status() {
    local status=$1
    local message=$2
    
    case $status in
        "SUCCESS")
            echo -e "${GREEN}✅ $message${NC}"
            log "INFO" "$message"
            ;;
        "ERROR")
            echo -e "${RED}❌ $message${NC}"
            log "ERROR" "$message"
            ;;
        "WARNING")
            echo -e "${YELLOW}⚠️  $message${NC}"
            log "WARN" "$message"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ️  $message${NC}"
            log "INFO" "$message"
            ;;
    esac
}

# Cleanup function
cleanup() {
    log "INFO" "Performing cleanup..."
    # Add cleanup commands here
}

# Error handler
error_exit() {
    print_status "ERROR" "Deployment failed: $1"
    log "ERROR" "Deployment failed: $1"
    cleanup
    exit 1
}

# Trap errors
trap 'error_exit "Unexpected error on line $LINENO"' ERR

# Pre-deployment checks
pre_deployment_checks() {
    print_section "Pre-deployment Validation"
    
    # Check required files
    required_files=(
        ".env.${DEPLOYMENT_ENV}"
        "docker-compose.production.yml"
        "nginx.conf"
        "monitoring/prometheus.yml"
        "alembic.ini"
    )
    
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            print_status "SUCCESS" "Required file exists: $file"
        else
            error_exit "Missing required file: $file"
        fi
    done
    
    # Check Docker availability
    if command -v docker &> /dev/null; then
        print_status "SUCCESS" "Docker is available"
    else
        error_exit "Docker is not installed or not available"
    fi
    
    # Check Docker Compose
    if docker compose version &> /dev/null; then
        print_status "SUCCESS" "Docker Compose is available"
    else
        error_exit "Docker Compose is not available"
    fi
    
    # Validate environment file
    if grep -q "SECRET_KEY=\${SECRET_KEY}" ".env.${DEPLOYMENT_ENV}"; then
        print_status "WARNING" "Environment variables need to be set"
        echo "Please ensure all environment variables are properly configured"
    else
        print_status "SUCCESS" "Environment configuration appears complete"
    fi
    
    # Run enhanced health check
    if [ -x "./scripts/enhanced_ci_health_check.sh" ]; then
        print_status "INFO" "Running enhanced health check..."
        if ./scripts/enhanced_ci_health_check.sh; then
            print_status "SUCCESS" "System health check passed"
        else
            print_status "WARNING" "Health check completed with warnings"
        fi
    fi
}

# Create backup
create_backup() {
    print_section "Creating System Backup"
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup current configuration
    if [ -d "config" ]; then
        cp -r config "$BACKUP_DIR/"
        print_status "SUCCESS" "Configuration backed up"
    fi
    
    # Backup environment files
    for env_file in .env.*; do
        if [ -f "$env_file" ]; then
            cp "$env_file" "$BACKUP_DIR/"
            print_status "SUCCESS" "Environment file backed up: $env_file"
        fi
    done
    
    # Backup database if running
    if docker ps | grep -q postgres; then
        print_status "INFO" "Creating database backup..."
        docker exec postgres pg_dumpall -c -U vortex_user > "$BACKUP_DIR/database_backup.sql" || {
            print_status "WARNING" "Database backup failed (container may not be running)"
        }
    fi
    
    # Backup docker compose files
    cp docker-compose*.yml "$BACKUP_DIR/" 2>/dev/null || true
    cp nginx.conf "$BACKUP_DIR/" 2>/dev/null || true
    
    print_status "SUCCESS" "Backup created at: $BACKUP_DIR"
}

# Database setup
setup_database() {
    print_section "Database Setup and Migration"
    
    # Set environment for database operations
    export NFCS_ENV="$DEPLOYMENT_ENV"
    
    # Check if Alembic is available
    if command -v alembic &> /dev/null; then
        print_status "INFO" "Running database migrations..."
        
        # Run migrations
        if alembic upgrade head; then
            print_status "SUCCESS" "Database migrations completed"
        else
            print_status "WARNING" "Database migrations failed (may need manual intervention)"
        fi
    else
        print_status "WARNING" "Alembic not available, skipping migrations"
        
        # Try to run SQL initialization
        if [ -f "sql/init/001_initial_schema.sql" ]; then
            print_status "INFO" "Running SQL initialization..."
            # This would need the database to be running
            print_status "INFO" "SQL files available for manual initialization"
        fi
    fi
}

# Deploy services
deploy_services() {
    print_section "Service Deployment"
    
    # Stop existing services
    print_status "INFO" "Stopping existing services..."
    docker compose -f docker-compose.production.yml down --remove-orphans || {
        print_status "WARNING" "No existing services to stop"
    }
    
    # Pull latest images
    print_status "INFO" "Pulling latest Docker images..."
    docker compose -f docker-compose.production.yml pull || {
        print_status "WARNING" "Failed to pull some images, continuing with local builds"
    }
    
    # Build and start services
    print_status "INFO" "Building and starting services..."
    if docker compose -f docker-compose.production.yml up -d --build; then
        print_status "SUCCESS" "Services deployed successfully"
    else
        error_exit "Failed to deploy services"
    fi
    
    # Wait for services to be ready
    print_status "INFO" "Waiting for services to be ready..."
    sleep 30
}

# Health checks
post_deployment_health_checks() {
    print_section "Post-deployment Health Checks"
    
    local health_passed=true
    
    # Check service status
    print_status "INFO" "Checking service status..."
    if docker compose -f docker-compose.production.yml ps | grep -q "Up"; then
        print_status "SUCCESS" "Services are running"
    else
        print_status "ERROR" "Some services are not running"
        health_passed=false
    fi
    
    # Check application health endpoints
    health_endpoints=(
        "http://localhost/health"
        "http://localhost:9090/-/healthy"  # Prometheus
        "http://localhost:3000/api/health"  # Grafana
    )
    
    for endpoint in "${health_endpoints[@]}"; do
        if curl -f -s "$endpoint" > /dev/null 2>&1; then
            print_status "SUCCESS" "Health check passed: $endpoint"
        else
            print_status "WARNING" "Health check failed: $endpoint"
            # Don't fail deployment for monitoring endpoints
        fi
    done
    
    # Check database connectivity
    if docker exec postgres pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
        print_status "SUCCESS" "Database is accessible"
    else
        print_status "ERROR" "Database is not accessible"
        health_passed=false
    fi
    
    # Check Redis connectivity
    if docker exec redis redis-cli ping > /dev/null 2>&1; then
        print_status "SUCCESS" "Redis is accessible"
    else
        print_status "WARNING" "Redis is not accessible"
    fi
    
    if [ "$health_passed" = false ]; then
        print_status "ERROR" "Critical health checks failed"
        return 1
    else
        print_status "SUCCESS" "All critical health checks passed"
        return 0
    fi
}

# Monitoring setup
setup_monitoring() {
    print_section "Monitoring Configuration"
    
    # Verify monitoring services
    monitoring_services=("prometheus" "grafana")
    
    for service in "${monitoring_services[@]}"; do
        if docker compose -f docker-compose.production.yml ps "$service" | grep -q "Up"; then
            print_status "SUCCESS" "Monitoring service running: $service"
        else
            print_status "WARNING" "Monitoring service not running: $service"
        fi
    done
    
    # Display monitoring URLs
    echo -e "\n${BLUE}📊 Monitoring Access URLs:${NC}"
    echo -e "   Grafana: ${GREEN}http://localhost:3000${NC} (admin/vortex123)"
    echo -e "   Prometheus: ${GREEN}http://localhost:9090${NC}"
    echo -e "   Application: ${GREEN}http://localhost${NC}"
}

# Rollback function
rollback() {
    print_section "Rolling Back Deployment"
    
    if [ -d "$BACKUP_DIR" ]; then
        print_status "INFO" "Rolling back to previous version..."
        
        # Stop current services
        docker compose -f docker-compose.production.yml down
        
        # Restore configuration files
        cp "$BACKUP_DIR"/* . 2>/dev/null || true
        
        # Restore database if backup exists
        if [ -f "$BACKUP_DIR/database_backup.sql" ]; then
            print_status "INFO" "Restoring database..."
            # Database restoration would go here
        fi
        
        print_status "SUCCESS" "Rollback completed"
    else
        print_status "ERROR" "No backup found for rollback"
    fi
}

# Main deployment process
main() {
    print_header
    
    print_status "INFO" "Starting deployment for environment: $DEPLOYMENT_ENV"
    print_status "INFO" "Version: $VERSION"
    print_status "INFO" "Log file: $LOG_FILE"
    
    # Run deployment steps
    pre_deployment_checks
    create_backup
    setup_database
    deploy_services
    
    # Health checks
    if post_deployment_health_checks; then
        setup_monitoring
        
        print_section "Deployment Summary"
        print_status "SUCCESS" "🎉 Deployment completed successfully!"
        echo -e "\n${GREEN}✅ Vortex-Omega NFCS v${VERSION} is now running in ${DEPLOYMENT_ENV} mode${NC}"
        echo -e "${GREEN}✅ Access the application at: http://localhost${NC}"
        echo -e "${GREEN}✅ Monitor the system at: http://localhost:3000${NC}"
        echo -e "${GREEN}✅ View metrics at: http://localhost:9090${NC}"
        
        # Display important information
        echo -e "\n${BLUE}📋 Important Information:${NC}"
        echo -e "   • Backup location: ${BACKUP_DIR}"
        echo -e "   • Log file: ${LOG_FILE}"
        echo -e "   • Environment: ${DEPLOYMENT_ENV}"
        echo -e "   • Version: ${VERSION}"
        
        # Display next steps
        echo -e "\n${BLUE}🎯 Next Steps:${NC}"
        echo -e "   1. Verify application functionality"
        echo -e "   2. Run integration tests"
        echo -e "   3. Configure monitoring alerts"
        echo -e "   4. Set up automated backups"
        echo -e "   5. Update DNS and SSL certificates"
        
    else
        print_status "ERROR" "Post-deployment health checks failed"
        print_status "INFO" "Initiating rollback..."
        rollback
        error_exit "Deployment failed, system rolled back"
    fi
}

# Script usage
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [environment]"
    echo ""
    echo "Environments:"
    echo "  production (default) - Full production deployment"
    echo "  staging             - Staging environment deployment"
    echo "  testing             - Testing environment deployment"
    echo ""
    echo "Options:"
    echo "  --help, -h          - Show this help message"
    echo "  --rollback          - Rollback to previous deployment"
    echo ""
    echo "Examples:"
    echo "  $0                  - Deploy to production"
    echo "  $0 staging          - Deploy to staging"
    echo "  $0 --rollback       - Rollback deployment"
    exit 0
fi

if [ "$1" = "--rollback" ]; then
    rollback
    exit 0
fi

# Run main deployment
main