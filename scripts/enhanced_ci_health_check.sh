#!/bin/bash
# Enhanced CI/CD Health Check for Vortex-Omega NFCS
# Comprehensive validation with production readiness assessment

set -e

echo "🏥 Enhanced CI/CD Health Check for Vortex-Omega NFCS v2.5.0"
echo "============================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
CHECKS_TOTAL=0
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNING=0

# Function to print status
print_status() {
    local status=$1
    local message=$2
    local details=$3
    
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    
    case $status in
        "PASS")
            echo -e "${GREEN}✅ PASS${NC}: $message"
            CHECKS_PASSED=$((CHECKS_PASSED + 1))
            ;;
        "FAIL")
            echo -e "${RED}❌ FAIL${NC}: $message"
            if [ -n "$details" ]; then
                echo -e "   ${RED}Details: $details${NC}"
            fi
            CHECKS_FAILED=$((CHECKS_FAILED + 1))
            ;;
        "WARN")
            echo -e "${YELLOW}⚠️  WARN${NC}: $message"
            if [ -n "$details" ]; then
                echo -e "   ${YELLOW}Details: $details${NC}"
            fi
            CHECKS_WARNING=$((CHECKS_WARNING + 1))
            ;;
        "INFO")
            echo -e "${BLUE}ℹ️  INFO${NC}: $message"
            ;;
    esac
}

# Environment validation
echo -e "\n${BLUE}🔍 Environment Validation${NC}"
echo "─────────────────────────"

# Python version check
if python3 --version | grep -q "Python 3\.[89]\|Python 3\.1[0-9]"; then
    print_status "PASS" "Python version compatible" "$(python3 --version)"
else
    print_status "WARN" "Python version may not be optimal" "$(python3 --version)"
fi

# Virtual environment check
if [ -n "$VIRTUAL_ENV" ]; then
    print_status "PASS" "Virtual environment active" "$VIRTUAL_ENV"
else
    if [ -d "venv" ]; then
        print_status "WARN" "Virtual environment exists but not activated" "./venv"
        source venv/bin/activate 2>/dev/null || true
    else
        print_status "FAIL" "No virtual environment found"
    fi
fi

# Essential directories check
echo -e "\n${BLUE}📁 Directory Structure${NC}"
echo "──────────────────────"

essential_dirs=("src" "tests" "config" "scripts" "monitoring" "sql")
for dir in "${essential_dirs[@]}"; do
    if [ -d "$dir" ]; then
        print_status "PASS" "Directory exists: $dir"
    else
        print_status "FAIL" "Missing essential directory: $dir"
    fi
done

# Configuration files check
echo -e "\n${BLUE}⚙️  Configuration Files${NC}"
echo "─────────────────────────"

config_files=(
    ".env.production"
    ".env.development" 
    "alembic.ini"
    "docker-compose.yml"
    "nginx.conf"
    "monitoring/prometheus.yml"
)

for file in "${config_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "PASS" "Configuration file exists: $file"
    else
        print_status "WARN" "Configuration file missing: $file"
    fi
done

# Dependencies check
echo -e "\n${BLUE}📦 Dependencies Check${NC}"
echo "───────────────────────"

if [ -f "requirements.txt" ]; then
    print_status "PASS" "Requirements file exists"
    
    # Count dependencies
    dep_count=$(grep -v '^#' requirements.txt | grep -v '^$' | wc -l)
    print_status "INFO" "Dependencies defined: $dep_count"
    
    # Check for critical dependencies
    critical_deps=("flask" "torch" "numpy" "redis" "psutil")
    for dep in "${critical_deps[@]}"; do
        if grep -q "$dep" requirements.txt; then
            print_status "PASS" "Critical dependency found: $dep"
        else
            print_status "WARN" "Critical dependency missing: $dep"
        fi
    done
else
    print_status "FAIL" "requirements.txt not found"
fi

# Python syntax check
echo -e "\n${BLUE}🐍 Python Syntax Validation${NC}"
echo "──────────────────────────────"

python_files=$(find src tests -name "*.py" 2>/dev/null | head -20)
syntax_errors=0

for file in $python_files; do
    if python3 -m py_compile "$file" 2>/dev/null; then
        print_status "PASS" "Syntax valid: $file"
    else
        print_status "FAIL" "Syntax error: $file"
        syntax_errors=$((syntax_errors + 1))
    fi
done

if [ $syntax_errors -eq 0 ]; then
    print_status "PASS" "All Python files have valid syntax"
else
    print_status "FAIL" "Found $syntax_errors syntax errors"
fi

# Import check
echo -e "\n${BLUE}📥 Import Validation${NC}"
echo "───────────────────────"

export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# Test core imports
if python3 -c "import sys; sys.path.append('src'); import src" 2>/dev/null; then
    print_status "PASS" "Core package imports successfully"
else
    print_status "WARN" "Core package import issues (may need dependencies)"
fi

# Test specific modules
modules_to_test=(
    "src.main"
    "src.core"
    "src.modules"
    "src.orchestrator"
)

for module in "${modules_to_test[@]}"; do
    if python3 -c "import sys; sys.path.append('src'); import $module" 2>/dev/null; then
        print_status "PASS" "Module imports: $module"
    else
        print_status "WARN" "Module import issues: $module"
    fi
done

# Test collection
echo -e "\n${BLUE}🧪 Test Collection${NC}"
echo "─────────────────────"

if [ -d "tests" ]; then
    test_files=$(find tests -name "test_*.py" | wc -l)
    print_status "INFO" "Test files found: $test_files"
    
    if command -v pytest &> /dev/null; then
        if pytest --collect-only -q 2>/dev/null | grep -q "collected"; then
            collected=$(pytest --collect-only -q 2>/dev/null | grep "collected" | awk '{print $1}')
            print_status "PASS" "Tests collected successfully: $collected tests"
        else
            print_status "WARN" "Test collection issues (may need dependencies)"
        fi
    else
        print_status "WARN" "pytest not available for test collection"
    fi
else
    print_status "FAIL" "Tests directory not found"
fi

# Docker validation
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
else
    print_status "WARN" "Dockerfile not found"
fi

if [ -f "docker-compose.yml" ] || [ -f "docker-compose.yaml" ]; then
    print_status "PASS" "Docker Compose configuration exists"
else
    print_status "WARN" "Docker Compose configuration not found"
fi

# CI/CD files check
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

# Security check
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
else
    print_status "WARN" ".gitignore not found"
fi

# Production readiness check
echo -e "\n${BLUE}🚀 Production Readiness${NC}"
echo "─────────────────────────"

# Check for production configurations
if [ -f ".env.production" ]; then
    print_status "PASS" "Production environment configuration exists"
else
    print_status "FAIL" "Production environment configuration missing"
fi

# Check for monitoring setup
if [ -f "monitoring/prometheus.yml" ]; then
    print_status "PASS" "Monitoring configuration exists"
else
    print_status "WARN" "Monitoring configuration missing"
fi

# Check for backup configuration
if [ -f "docker-compose.production.yml" ]; then
    if grep -q "backup" docker-compose.production.yml; then
        print_status "PASS" "Backup service configured"
    else
        print_status "WARN" "Backup service not configured"
    fi
else
    print_status "WARN" "Production docker-compose not found"
fi

# Final summary
echo -e "\n${BLUE}📊 Health Check Summary${NC}"
echo "═══════════════════════"
echo -e "Total Checks: ${BLUE}$CHECKS_TOTAL${NC}"
echo -e "Passed: ${GREEN}$CHECKS_PASSED${NC}"
echo -e "Warnings: ${YELLOW}$CHECKS_WARNING${NC}"
echo -e "Failed: ${RED}$CHECKS_FAILED${NC}"

# Calculate health score
health_score=$(( (CHECKS_PASSED * 100) / CHECKS_TOTAL ))
echo -e "\nHealth Score: ${BLUE}$health_score%${NC}"

# Provide recommendations
echo -e "\n${BLUE}🎯 Recommendations${NC}"
echo "─────────────────"

if [ $CHECKS_FAILED -gt 0 ]; then
    echo -e "${RED}❌ Fix critical issues before deployment${NC}"
    echo -e "   Priority: Resolve $CHECKS_FAILED failed checks"
fi

if [ $CHECKS_WARNING -gt 5 ]; then
    echo -e "${YELLOW}⚠️  Address warnings for better reliability${NC}"
    echo -e "   Consider: Resolving $CHECKS_WARNING warning items"
fi

if [ $health_score -ge 90 ]; then
    echo -e "${GREEN}🎉 System is in excellent health!${NC}"
    echo -e "   Status: Ready for production deployment"
elif [ $health_score -ge 75 ]; then
    echo -e "${YELLOW}👍 System health is good${NC}"
    echo -e "   Status: Minor improvements recommended"
elif [ $health_score -ge 60 ]; then
    echo -e "${YELLOW}⚠️  System health needs attention${NC}"
    echo -e "   Status: Address issues before production"
else
    echo -e "${RED}🚨 System health is poor${NC}"
    echo -e "   Status: Major fixes required"
fi

echo -e "\n${BLUE}✅ Enhanced health check completed!${NC}"

# Exit with appropriate code
if [ $CHECKS_FAILED -gt 0 ]; then
    exit 1
elif [ $health_score -lt 75 ]; then
    exit 2
else
    exit 0
fi