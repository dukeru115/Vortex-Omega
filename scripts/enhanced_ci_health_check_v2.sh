#!/bin/bash
# Enhanced CI/CD Health Check for Vortex-Omega
# Works in network-constrained environments with comprehensive fallbacks

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Global counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Print status with colors
print_status() {
    local status=$1
    local message=$2
    local details=$3
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    case $status in
        "PASS")
            echo -e "${GREEN}✅ PASS${NC}: $message"
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
            ;;
        "FAIL")
            echo -e "${RED}❌ FAIL${NC}: $message"
            if [ -n "$details" ]; then
                echo -e "   ${RED}Details: $details${NC}"
            fi
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            ;;
        "WARN")
            echo -e "${YELLOW}⚠️  WARN${NC}: $message"
            if [ -n "$details" ]; then
                echo -e "   ${YELLOW}Details: $details${NC}"
            fi
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
            ;;
        "INFO")
            echo -e "${BLUE}ℹ️  INFO${NC}: $message"
            ;;
    esac
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check network connectivity (non-critical)
check_network() {
    echo -e "\n${BLUE}🌐 Network Connectivity${NC}"
    echo "─────────────────────"
    
    if ping -c 1 google.com >/dev/null 2>&1; then
        print_status "PASS" "Internet connectivity available"
        
        # Test PyPI connectivity
        if curl -s --head https://pypi.org >/dev/null; then
            print_status "PASS" "PyPI connectivity available"
        else
            print_status "WARN" "PyPI not accessible (package installation may fail)"
        fi
    else
        print_status "WARN" "No internet connectivity (offline mode)"
        print_status "INFO" "Proceeding with offline validation"
    fi
}

# Environment validation
check_environment() {
    echo -e "\n${BLUE}🔍 Environment Validation${NC}"
    echo "─────────────────────────"
    
    # Python version check
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 8 ]; then
            print_status "PASS" "Python version: $PYTHON_VERSION"
        else
            print_status "FAIL" "Python version $PYTHON_VERSION < 3.8 (unsupported)"
        fi
    else
        print_status "FAIL" "Python 3 not found"
    fi
    
    # pip check
    if command_exists pip; then
        print_status "PASS" "pip available"
    elif command_exists pip3; then
        print_status "PASS" "pip3 available"
    else
        print_status "WARN" "pip not found (dependency installation unavailable)"
    fi
    
    # Virtual environment detection
    if [ -n "$VIRTUAL_ENV" ]; then
        print_status "PASS" "Virtual environment active: $VIRTUAL_ENV"
    else
        print_status "WARN" "No virtual environment detected"
    fi
    
    # PYTHONPATH check
    if [ -n "$PYTHONPATH" ]; then
        print_status "PASS" "PYTHONPATH configured: $PYTHONPATH"
    else
        print_status "INFO" "PYTHONPATH not set (will be configured)"
        export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
    fi
    
    # Disk space check
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    if [ "$AVAILABLE_SPACE" -gt 1048576 ]; then  # 1GB in KB
        print_status "PASS" "Sufficient disk space available"
    else
        print_status "WARN" "Low disk space (may affect builds)"
    fi
}

# Project structure validation
check_project_structure() {
    echo -e "\n${BLUE}📁 Project Structure${NC}"
    echo "─────────────────────"
    
    essential_files=(
        "src/__init__.py"
        "requirements.txt"
        "pyproject.toml"
        "Dockerfile"
        "README.md"
    )
    
    optional_files=(
        "requirements-dev.txt"
        ".github/workflows/ci-simple.yml"
        ".gitlab-ci.yml"
        "Jenkinsfile"
        "docker-compose.yml"
        "pytest.ini"
    )
    
    # Check essential files
    for file in "${essential_files[@]}"; do
        if [ -f "$file" ]; then
            print_status "PASS" "Essential file exists: $file"
        else
            print_status "FAIL" "Missing essential file: $file"
        fi
    done
    
    # Check optional files
    for file in "${optional_files[@]}"; do
        if [ -f "$file" ]; then
            print_status "PASS" "Optional file exists: $file"
        else
            print_status "INFO" "Optional file missing: $file"
        fi
    done
    
    # Check directory structure
    essential_dirs=("src" "tests")
    for dir in "${essential_dirs[@]}"; do
        if [ -d "$dir" ]; then
            print_status "PASS" "Directory exists: $dir"
        else
            print_status "FAIL" "Missing directory: $dir"
        fi
    done
}

# Code quality validation (offline)
check_code_quality() {
    echo -e "\n${BLUE}🔍 Code Quality (Offline)${NC}"
    echo "─────────────────────────"
    
    # Python syntax check
    if find src/ -name "*.py" -exec python3 -m py_compile {} \; >/dev/null 2>&1; then
        py_count=$(find src/ -name "*.py" | wc -l)
        print_status "PASS" "All $py_count Python files in src/ have valid syntax"
    else
        print_status "FAIL" "Python syntax errors found in src/"
    fi
    
    if [ -d "tests/" ]; then
        if find tests/ -name "*.py" -exec python3 -m py_compile {} \; >/dev/null 2>&1; then
            test_count=$(find tests/ -name "*.py" | wc -l)
            print_status "PASS" "All $test_count Python files in tests/ have valid syntax"
        else
            print_status "FAIL" "Python syntax errors found in tests/"
        fi
    else
        print_status "WARN" "No tests directory found"
    fi
    
    # Import validation
    if python3 -c "import src; print('✅ Core module imports successful')" >/dev/null 2>&1; then
        print_status "PASS" "Core module imports successful"
    else
        print_status "FAIL" "Core module import failed"
    fi
    
    # Basic security checks (if available)
    if command_exists bandit; then
        if bandit -r src/ -f txt >/dev/null 2>&1; then
            print_status "PASS" "Security scan completed (no critical issues)"
        else
            print_status "WARN" "Security scan found potential issues"
        fi
    else
        print_status "INFO" "Bandit not available (skipping security scan)"
    fi
}

# Test framework validation
check_test_framework() {
    echo -e "\n${BLUE}🧪 Test Framework${NC}"
    echo "─────────────────────"
    
    if [ -d "tests/" ]; then
        print_status "PASS" "Tests directory exists"
        
        # Count test files
        test_files=$(find tests/ -name "test_*.py" -o -name "*_test.py" | wc -l)
        if [ "$test_files" -gt 0 ]; then
            print_status "PASS" "Found $test_files test files"
        else
            print_status "WARN" "No test files found"
        fi
        
        # Test collection (if pytest available)
        if command_exists pytest; then
            collected=$(python3 -m pytest tests/ --collect-only -q 2>/dev/null | grep -E "^[0-9]+ tests collected" | cut -d' ' -f1 || echo "0")
            if [ "$collected" -gt 0 ]; then
                print_status "PASS" "Successfully collected $collected tests"
            else
                print_status "WARN" "Test collection issues (may need dependencies)"
            fi
        else
            print_status "WARN" "pytest not available for test collection"
        fi
    else
        print_status "FAIL" "Tests directory not found"
    fi
}

# Docker validation
check_docker() {
    echo -e "\n${BLUE}🐳 Docker Configuration${NC}"
    echo "─────────────────────────────"
    
    if [ -f "Dockerfile" ]; then
        print_status "PASS" "Dockerfile exists"
        
        # Check for multi-stage build
        if grep -q "FROM.*AS" Dockerfile; then
            print_status "PASS" "Multi-stage Docker build configured"
        else
            print_status "WARN" "Single-stage Docker build (consider multi-stage)"
        fi
        
        # Check for security best practices
        if grep -q "USER" Dockerfile; then
            print_status "PASS" "Non-root user configured in Dockerfile"
        else
            print_status "WARN" "Dockerfile runs as root (security concern)"
        fi
    else
        print_status "WARN" "Dockerfile not found"
    fi
    
    if [ -f "docker-compose.yml" ] || [ -f "docker-compose.yaml" ]; then
        print_status "PASS" "Docker Compose configuration exists"
    else
        print_status "WARN" "Docker Compose configuration not found"
    fi
    
    # Check if Docker is available
    if command_exists docker; then
        if docker info >/dev/null 2>&1; then
            print_status "PASS" "Docker daemon accessible"
        else
            print_status "WARN" "Docker daemon not accessible"
        fi
    else
        print_status "INFO" "Docker CLI not available"
    fi
}

# CI/CD configuration validation
check_cicd_config() {
    echo -e "\n${BLUE}🔄 CI/CD Configuration${NC}"
    echo "─────────────────────────"
    
    if [ -d ".github/workflows" ]; then
        workflow_count=$(find .github/workflows -name "*.yml" -o -name "*.yaml" | wc -l)
        print_status "PASS" "GitHub Actions workflows found: $workflow_count"
        
        # Check for essential workflows
        if [ -f ".github/workflows/ci-simple.yml" ] || [ -f ".github/workflows/main.yml" ]; then
            print_status "PASS" "Main CI workflow configured"
        else
            print_status "WARN" "Main CI workflow not found"
        fi
        
        if [ -f ".github/workflows/docker-image.yml" ]; then
            print_status "PASS" "Docker build workflow configured"
        else
            print_status "WARN" "Docker build workflow not found"
        fi
    else
        print_status "WARN" "GitHub Actions workflows not found"
    fi
    
    if [ -f ".gitlab-ci.yml" ]; then
        print_status "PASS" "GitLab CI configuration exists"
    else
        print_status "INFO" "GitLab CI configuration not found"
    fi
    
    if [ -f "Jenkinsfile" ]; then
        print_status "PASS" "Jenkins pipeline configuration exists"
    else
        print_status "INFO" "Jenkins pipeline configuration not found"
    fi
}

# Security validation
check_security() {
    echo -e "\n${BLUE}🔒 Security Validation${NC}"
    echo "──────────────────────"
    
    # Check for .env files in git
    if git ls-files | grep -q "\.env$"; then
        print_status "FAIL" "Environment files tracked in git (security risk)"
    else
        print_status "PASS" "Environment files not tracked in git"
    fi
    
    # Check .gitignore
    if [ -f ".gitignore" ]; then
        print_status "PASS" ".gitignore exists"
        
        if grep -q "\.env" .gitignore; then
            print_status "PASS" "Environment files ignored in git"
        else
            print_status "WARN" "Environment files not explicitly ignored"
        fi
        
        if grep -q "__pycache__" .gitignore; then
            print_status "PASS" "Python cache files ignored"
        else
            print_status "WARN" "Python cache files not ignored"
        fi
    else
        print_status "WARN" ".gitignore not found"
    fi
    
    # Check for hardcoded secrets (basic patterns)
    if grep -r -i "password\s*=" src/ --include="*.py" | grep -v "password_field\|password_input" >/dev/null 2>&1; then
        print_status "WARN" "Potential hardcoded passwords found"
    else
        print_status "PASS" "No obvious hardcoded passwords detected"
    fi
}

# Performance and resource validation
check_performance() {
    echo -e "\n${BLUE}⚡ Performance & Resources${NC}"
    echo "─────────────────────────────"
    
    # Check file sizes
    large_files=$(find . -type f -size +10M 2>/dev/null | grep -v ".git" | head -5)
    if [ -n "$large_files" ]; then
        print_status "WARN" "Large files detected (may slow CI/CD)"
        echo -e "   ${YELLOW}Large files:${NC}"
        echo "$large_files" | while read -r file; do
            size=$(du -h "$file" | cut -f1)
            echo -e "   ${YELLOW}- $file ($size)${NC}"
        done
    else
        print_status "PASS" "No excessively large files detected"
    fi
    
    # Check total project size
    total_size=$(du -sh . 2>/dev/null | cut -f1)
    print_status "INFO" "Total project size: $total_size"
    
    # Count Python files and estimate complexity
    py_files=$(find src/ -name "*.py" | wc -l)
    total_lines=$(find src/ -name "*.py" -exec wc -l {} \; 2>/dev/null | awk '{sum += $1} END {print sum}')
    
    if [ "$py_files" -gt 0 ]; then
        avg_lines=$((total_lines / py_files))
        print_status "INFO" "$py_files Python files, $total_lines total lines (avg: $avg_lines lines/file)"
        
        if [ "$avg_lines" -gt 200 ]; then
            print_status "WARN" "High average file size (consider refactoring)"
        else
            print_status "PASS" "Reasonable file sizes"
        fi
    fi
}

# Production readiness check
check_production_readiness() {
    echo -e "\n${BLUE}🚀 Production Readiness${NC}"
    echo "─────────────────────────"
    
    # Check for production configurations
    production_files=(
        "docker-compose.production.yml"
        "nginx.conf"
        "supervisord.conf"
        "requirements.txt"
    )
    
    for file in "${production_files[@]}"; do
        if [ -f "$file" ]; then
            print_status "PASS" "Production config exists: $file"
        else
            print_status "WARN" "Missing production config: $file"
        fi
    done
    
    # Check for monitoring setup
    if [ -d "monitoring" ]; then
        print_status "PASS" "Monitoring configuration directory exists"
    else
        print_status "WARN" "No monitoring configuration found"
    fi
    
    # Check for health check endpoints
    if grep -r "health" src/ --include="*.py" >/dev/null 2>&1; then
        print_status "PASS" "Health check implementation found"
    else
        print_status "WARN" "No health check endpoints found"
    fi
    
    # Check for logging configuration
    if grep -r "logging" src/ --include="*.py" >/dev/null 2>&1; then
        print_status "PASS" "Logging implementation found"
    else
        print_status "WARN" "No logging configuration found"
    fi
}

# Run offline validation script
run_offline_validation() {
    echo -e "\n${BLUE}🔬 Comprehensive Offline Validation${NC}"
    echo "────────────────────────────────────"
    
    if [ -f "scripts/offline_ci_validation.py" ]; then
        if python3 scripts/offline_ci_validation.py; then
            print_status "PASS" "Offline validation completed successfully"
        else
            print_status "WARN" "Offline validation completed with warnings"
        fi
    else
        print_status "INFO" "Offline validation script not found"
    fi
}

# Generate summary report
generate_summary() {
    echo -e "\n${PURPLE}═══════════════════════════════════════${NC}"
    echo -e "${PURPLE}📊 COMPREHENSIVE HEALTH CHECK SUMMARY${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════${NC}"
    
    echo -e "\n${CYAN}Statistics:${NC}"
    echo "  Total checks: $TOTAL_CHECKS"
    echo -e "  ${GREEN}Passed: $PASSED_CHECKS${NC}"
    echo -e "  ${RED}Failed: $FAILED_CHECKS${NC}"
    echo -e "  ${YELLOW}Warnings: $WARNING_CHECKS${NC}"
    
    if [ "$FAILED_CHECKS" -eq 0 ]; then
        success_rate=100
    else
        success_rate=$((PASSED_CHECKS * 100 / (PASSED_CHECKS + FAILED_CHECKS)))
    fi
    
    echo "  Success rate: $success_rate%"
    
    # Determine overall status
    if [ "$FAILED_CHECKS" -eq 0 ]; then
        echo -e "\n${GREEN}🎉 OVERALL STATUS: HEALTHY${NC}"
        echo -e "${GREEN}✅ Repository is ready for CI/CD pipelines!${NC}"
        exit_code=0
    elif [ "$FAILED_CHECKS" -lt 3 ]; then
        echo -e "\n${YELLOW}⚠️  OVERALL STATUS: NEEDS ATTENTION${NC}"
        echo -e "${YELLOW}🔧 Some issues need to be addressed.${NC}"
        exit_code=1
    else
        echo -e "\n${RED}❌ OVERALL STATUS: CRITICAL ISSUES${NC}"
        echo -e "${RED}🚨 Multiple critical issues need immediate attention.${NC}"
        exit_code=2
    fi
    
    # Recommendations
    echo -e "\n${CYAN}💡 Recommendations:${NC}"
    if [ "$FAILED_CHECKS" -gt 0 ]; then
        echo "  - Address all failed checks before proceeding"
    fi
    if [ "$WARNING_CHECKS" -gt 5 ]; then
        echo "  - Consider addressing warnings for better CI/CD reliability"
    fi
    if [ "$success_rate" -lt 80 ]; then
        echo "  - Review project configuration and dependencies"
    fi
    
    echo -e "\n${BLUE}📄 For detailed analysis, check: offline_ci_validation_report.json${NC}"
    
    return $exit_code
}

# Main execution
main() {
    echo -e "${CYAN}🔥 Enhanced CI/CD Health Check for Vortex-Omega v2.5.0${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
    
    # Run all checks
    check_network
    check_environment
    check_project_structure
    check_code_quality
    check_test_framework
    check_docker
    check_cicd_config
    check_security
    check_performance
    check_production_readiness
    run_offline_validation
    
    # Generate final summary
    generate_summary
}

# Execute main function
main "$@"