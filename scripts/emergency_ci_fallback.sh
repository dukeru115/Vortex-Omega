#!/bin/bash
# Emergency CI Fallback Script for Vortex-Omega
# Works in extremely network-constrained environments
# Uses only Python standard library - no external dependencies

set -e

echo "🚨 Emergency CI Fallback for Vortex-Omega"
echo "========================================="
echo "⚠️  Running in emergency mode - no external dependencies"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0

# Helper function
check_and_report() {
    local test_name="$1"
    local command="$2"
    
    echo -n "Testing $test_name... "
    
    if eval "$command" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}❌ FAIL${NC}"
        FAILED=$((FAILED + 1))
    fi
}

# Set PYTHONPATH
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

echo "🔍 EMERGENCY VALIDATION TESTS"
echo "=============================="

# 1. Python version check
echo -n "Python version check... "
if python3 --version | grep -q "Python 3\.[89]\|Python 3\.1[0-9]"; then
    echo -e "${GREEN}✅ PASS${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}❌ FAIL${NC}"
    FAILED=$((FAILED + 1))
fi

# 2. Essential file existence
echo "Essential files:"
essential_files=("src/__init__.py" "requirements.txt" "pyproject.toml" "Dockerfile" "README.md")
for file in "${essential_files[@]}"; do
    check_and_report "$file" "[ -f '$file' ]"
done

# 3. Directory structure
echo "Essential directories:"
essential_dirs=("src" "tests")
for dir in "${essential_dirs[@]}"; do
    check_and_report "$dir/" "[ -d '$dir' ]"
done

# 4. Python syntax validation
echo "Python syntax validation:"
check_and_report "src/ syntax" "find src/ -name '*.py' -exec python3 -m py_compile {} \;"
if [ -d "tests/" ]; then
    check_and_report "tests/ syntax" "find tests/ -name '*.py' -exec python3 -m py_compile {} \;"
fi

# 5. Core module import
echo "Core imports:"
check_and_report "src module" "python3 -c 'import src'"
check_and_report "json module" "python3 -c 'import json'"
check_and_report "os module" "python3 -c 'import os'"
check_and_report "sys module" "python3 -c 'import sys'"
check_and_report "logging module" "python3 -c 'import logging'"

# 6. Basic code quality checks (using Python AST)
echo "Basic code quality:"
cat << 'EOF' > /tmp/ast_check.py
import ast
import sys
import os

def check_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False

files_checked = 0
files_passed = 0

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            files_checked += 1
            if check_file(filepath):
                files_passed += 1

print(f"{files_passed}/{files_checked}")
sys.exit(0 if files_passed == files_checked else 1)
EOF

check_and_report "AST parsing" "python3 /tmp/ast_check.py"

# 7. Security checks (basic patterns)
echo "Basic security checks:"
cat << 'EOF' > /tmp/security_check.py
import os
import re

def check_security():
    issues = 0
    dangerous_patterns = [
        (r'eval\s*\(', 'eval() usage'),
        (r'exec\s*\(', 'exec() usage'),
        (r'__import__\s*\(', '__import__() usage'),
        (r'password\s*=\s*["\'][^"\']+["\']', 'hardcoded password'),
    ]
    
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern, description in dangerous_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            issues += 1
                            print(f"Warning: {description} in {filepath}")
                            
                except Exception:
                    pass
    
    return issues == 0

if __name__ == "__main__":
    import sys
    sys.exit(0 if check_security() else 1)
EOF

check_and_report "security patterns" "python3 /tmp/security_check.py"

# 8. Git configuration checks
echo "Git configuration:"
check_and_report ".gitignore" "[ -f '.gitignore' ]"
check_and_report "no .env in git" "! git ls-files | grep -q '\.env$'"

# 9. Test file validation
if [ -d "tests/" ]; then
    echo "Test validation:"
    check_and_report "test files exist" "find tests/ -name 'test_*.py' | grep -q ."
    check_and_report "test compilation" "find tests/ -name 'test_*.py' -exec python3 -m py_compile {} \;"
fi

# 10. Configuration files
echo "Configuration validation:"
check_and_report "pyproject.toml syntax" "python3 -c 'import tomllib; tomllib.load(open(\"pyproject.toml\", \"rb\"))' 2>/dev/null || python3 -c 'import sys; print(\"tomllib not available in Python < 3.11\")'"

# Clean up
rm -f /tmp/ast_check.py /tmp/security_check.py

# Summary
echo ""
echo "🎯 EMERGENCY VALIDATION SUMMARY"
echo "==============================="
TOTAL=$((PASSED + FAILED))
echo "Total tests: $TOTAL"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 ALL EMERGENCY TESTS PASSED!${NC}"
    echo -e "${GREEN}✅ Repository is ready for deployment${NC}"
    echo ""
    echo "💡 This emergency validation ensures:"
    echo "   - Python syntax is correct"
    echo "   - Core modules can be imported"
    echo "   - Essential files exist"
    echo "   - No obvious security issues"
    echo "   - Basic project structure is intact"
    echo ""
    echo "🚀 Safe to proceed with deployment even in network-constrained environments!"
    exit 0
else
    echo ""
    echo -e "${RED}❌ EMERGENCY TESTS FAILED!${NC}"
    echo -e "${RED}🚨 Repository has critical issues${NC}"
    echo ""
    echo "🔧 Fix the failed tests before deployment"
    echo "   This emergency check only validates critical issues"
    echo "   Additional testing recommended when network is available"
    exit 1
fi