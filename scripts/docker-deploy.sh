#!/bin/bash

# Production deployment script for Vortex-Omega NFCS
# This script builds and deploys the production containers

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="vortex-omega"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env.production"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        log_warn "Production environment file not found, creating from example..."
        cp .env.example "$ENV_FILE"
        log_warn "Please edit $ENV_FILE with production values"
        exit 1
    fi
    
    log_info "All requirements met"
}

build_images() {
    log_info "Building Docker images..."
    
    # Build production image
    docker-compose -f "$COMPOSE_FILE" build --no-cache vortex-omega
    
    log_info "Images built successfully"
}

deploy_services() {
    log_info "Deploying services..."
    
    # Stop existing containers
    docker-compose -f "$COMPOSE_FILE" down
    
    # Start services in production mode
    docker-compose -f "$COMPOSE_FILE" --profile production up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10
    
    # Check health
    docker-compose -f "$COMPOSE_FILE" ps
    
    log_info "Services deployed successfully"
}

run_migrations() {
    log_info "Running database migrations..."
    
    # Execute migrations in the container
    docker-compose -f "$COMPOSE_FILE" exec -T vortex-omega python -m src.database.migrate
    
    log_info "Migrations completed"
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Ensure monitoring directories exist
    mkdir -p monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources
    
    # Start monitoring services
    docker-compose -f "$COMPOSE_FILE" up -d prometheus grafana
    
    log_info "Monitoring setup completed"
    log_info "Grafana available at http://localhost:3000 (admin/vortex123)"
    log_info "Prometheus available at http://localhost:9090"
}

show_status() {
    log_info "Deployment Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    log_info "Service URLs:"
    echo "  - Application: http://localhost:8080"
    echo "  - Redis: localhost:6379"
    echo "  - PostgreSQL: localhost:5432"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000"
    
    echo ""
    log_info "View logs with: docker-compose logs -f vortex-omega"
}

cleanup_old_images() {
    log_info "Cleaning up old images..."
    docker image prune -f
    log_info "Cleanup completed"
}

# Main execution
main() {
    log_info "Starting Vortex-Omega deployment..."
    
    check_requirements
    build_images
    deploy_services
    run_migrations
    setup_monitoring
    cleanup_old_images
    show_status
    
    log_info "Deployment completed successfully!"
}

# Parse arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    stop)
        log_info "Stopping services..."
        docker-compose -f "$COMPOSE_FILE" down
        log_info "Services stopped"
        ;;
    restart)
        log_info "Restarting services..."
        docker-compose -f "$COMPOSE_FILE" restart
        log_info "Services restarted"
        ;;
    logs)
        docker-compose -f "$COMPOSE_FILE" logs -f "${2:-vortex-omega}"
        ;;
    status)
        show_status
        ;;
    clean)
        log_warn "Removing all containers and volumes..."
        docker-compose -f "$COMPOSE_FILE" down -v
        cleanup_old_images
        log_info "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|clean}"
        exit 1
        ;;
esac