# CI/CD Troubleshooting Guide for Vortex-Omega

## Overview
This guide provides common CI/CD issues and their solutions for the Vortex-Omega repository.

## Common Issues and Solutions

### 1. Import Errors in Tests
**Symptoms:** Tests fail to collect with `ImportError` or `ModuleNotFoundError`

**Solutions:**
```bash
# Install missing dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Check for common missing packages
pip install websockets scikit-image hypothesis
```

**Class name mismatches fixed:**
- `SymbolicCore` → `SymbolicAI`
- `ESCCore` → `EchoSemanticConverter`
- `ConstitutionalModule` → `ConstitutionV0`
- `MetaCognitionModule` → `MetaReflectionModule`
- `MainLoop` → `NFCSMainOrchestrator`

### 2. Syntax Errors
**Symptoms:** `SyntaxError` in Python files

**Common fixes applied:**
- Fixed invalid import paths (`from ../../` → `from src.`)
- Fixed async function definitions (`def` → `async def` for functions using `async with`)
- Fixed boolean values (`true`/`false` → `True`/`False`)
- Fixed escaped quotes in docstrings

### 3. Missing Dependencies
**Required packages for development:**
```bash
# Core testing
pytest>=8.0.0
pytest-cov>=7.0.0
pytest-asyncio>=1.0.0
hypothesis>=6.0.0

# Code quality
black>=25.0.0
flake8>=7.0.0
mypy>=1.0.0
bandit>=1.7.0

# Additional dependencies
websockets>=12.0
scikit-image>=0.22.0
```

### 4. Docker Build Issues
**Common Docker targets:**
- `development` - For development environment
- `testing` - For running tests
- `production` - For production deployment
- `monitoring` - For metrics collection

### 5. Code Quality Standards

**Flake8 Configuration:**
```bash
flake8 src/ --max-line-length=100 --ignore=E203,W503,E501
```

**Black Formatting:**
```bash
black src/ tests/ --line-length=100
```

**Type Checking:**
```bash
mypy src/ --ignore-missing-imports
```

## Test Execution

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only
python -m pytest tests/test_core/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Test collection check
python -m pytest tests/ --collect-only
```

### Current Test Status
✅ **162 tests** successfully collected (as of latest fixes)
✅ **All import errors resolved**
✅ **All syntax errors fixed**

## Continuous Integration

### GitHub Actions Workflows
- `production-cicd.yml` - Main production pipeline
- `ci-simple.yml` - Basic CI checks
- `docker-image.yml` - Docker registry builds
- `build-test.yml` - Build and test workflow

### GitLab CI
- Full pipeline with validation, testing, building, security, and deployment stages
- Supports multiple environments (development, staging, production)

## Quick Health Check
```bash
# Check syntax
python -m py_compile src/**/*.py

# Check imports
python -c "import src; print('Imports OK')"

# Run basic test
python -m pytest tests/test_core/test_cgl_solver.py::TestCGLSolver::test_solver_initialization -v
```

## Emergency Recovery
If CI/CD completely fails:

1. **Reset to working state:**
   ```bash
   git checkout copilot/fix-29cebb28-92a7-4973-ab2b-18fe69da2556
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt -r requirements-dev.txt
   ```

3. **Verify test collection:**
   ```bash
   python -m pytest tests/ --collect-only
   ```

## Success Metrics
- ✅ 0 syntax errors
- ✅ 162 tests collected successfully  
- ✅ All critical imports resolved
- ✅ Development environment functional
- ✅ Code quality tools configured

---
*Last updated: September 2025*
*Repository: dukeru115/Vortex-Omega*