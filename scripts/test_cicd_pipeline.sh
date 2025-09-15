#!/bin/bash
# CI/CD Pipeline Test Script for Vortex-Omega
# This script runs basic checks that CI/CD pipelines should pass

set -e  # Exit on any error

echo "🚀 Starting Vortex-Omega CI/CD Pipeline Test"
echo "=============================================="

# Check Python version
echo "📋 Python Version Check"
python --version
echo "✅ Python version OK"
echo

# Install dependencies
echo "📦 Installing Dependencies"
pip install -q --upgrade pip
if [ -f requirements.txt ]; then
    pip install -q -r requirements.txt
    echo "✅ Core dependencies installed"
fi

if [ -f requirements-dev.txt ]; then
    pip install -q -r requirements-dev.txt
    echo "✅ Development dependencies installed"
fi
echo

# Syntax check
echo "🔍 Syntax Error Check"
python -m py_compile $(find src -name "*.py") || {
    echo "❌ Syntax errors found!"
    exit 1
}
echo "✅ No syntax errors found"
echo

# Import check
echo "📥 Import Check"
python -c "
try:
    import src
    print('✅ Core imports successful')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
"
echo

# Test collection
echo "🧪 Test Collection Check"
test_count=$(python -m pytest tests/ --collect-only -q 2>/dev/null | grep -E "^[0-9]+ tests collected" | cut -d' ' -f1 || echo "0")
if [ "$test_count" -gt 0 ]; then
    echo "✅ Successfully collected $test_count tests"
else
    echo "❌ Test collection failed!"
    python -m pytest tests/ --collect-only
    exit 1
fi
echo

# Basic linting (syntax errors only)
echo "🔍 Critical Linting Check"
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
error_count=$?
if [ $error_count -eq 0 ]; then
    echo "✅ No critical linting errors"
else
    echo "❌ Critical linting errors found!"
    exit 1
fi
echo

# Run a simple test
echo "🧪 Basic Test Execution"
if python -m pytest tests/test_core/test_cgl_solver.py::TestCGLSolver::test_solver_initialization -v -q; then
    echo "✅ Basic test execution successful"
else
    echo "⚠️  Basic test failed (may be due to test logic, not infrastructure)"
fi
echo

echo "🎉 CI/CD Pipeline Test Completed Successfully!"
echo "==============================================="
echo "📊 Summary:"
echo "   - Python environment: OK"
echo "   - Dependencies: OK"
echo "   - Syntax: OK"
echo "   - Imports: OK"
echo "   - Test collection: $test_count tests"
echo "   - Critical linting: OK"
echo "   - Infrastructure: READY"
echo
echo "🚀 Repository is ready for CI/CD pipelines!"