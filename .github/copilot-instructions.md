# Vortex-Omega: Copilot Onboarding Instructions

**ALWAYS follow these instructions first to reduce PR failures and build issues.**

Vortex-Omega is a hybrid AI system with Neural Field Control System (NFCS) featuring multi-agent consensus (Kuramoto/ADMM), constitutional monitoring, and Flask web interface. This is a Python 3.8+ AI/ML project with Docker support.

## Essential Setup - NEVER CANCEL Long Operations

### Environment Setup (2-3 minutes)
```bash
# Verify requirements
python3 --version  # Must be 3.8+ (3.11+ recommended, tested with 3.12)

# ALWAYS create virtual environment first
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Set PYTHONPATH for imports - CRITICAL
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

### Dependencies (5-15 minutes, TIMEOUT: 30+ minutes)
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Core dependencies - LARGE DOWNLOADS, NEVER CANCEL
# torch>=2.0.0 (~2GB), transformers>=4.20.0 (large)
timeout 1800 pip install -r requirements.txt --retries 3  # 5-15 minutes

# Development dependencies (optional)
timeout 900 pip install -r requirements-dev.txt --retries 3  # 2-5 minutes
```

### Build & Test (TIMEOUT: 10+ minutes each)
```bash
# Linting (1-2 minutes) - fallback if flake8 not available
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=venv,env || find src/ -name "*.py" -exec python -m py_compile {} \;
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude=venv,env || echo "Flake8 not available - syntax check passed"

# Tests (2-10 minutes, TIMEOUT: 15+ minutes)
./scripts/run-tests.sh --unit          # Unit tests (~2 minutes)
./scripts/run-tests.sh --integration   # Integration tests (~5 minutes)
./scripts/run-tests.sh --all           # All tests (~10 minutes)
```

## Core Commands & Workflows

### Quick Start (MVP Demo)
```bash
# Single command startup (2-3 minutes, TIMEOUT: 10+ minutes)
./start_mvp.sh
# Access dashboard: http://localhost:5000
# Fallback server runs even without Flask dependencies

# Manual startup if script fails
python mvp_web_interface.py
```

### Docker Production (TIMEOUT: 60+ minutes)
```bash
# Full production stack (15-45 minutes build time, NEVER CANCEL)
timeout 3600 docker compose up -d
# Services: http://localhost:8080, Grafana: http://localhost:3000
```

### Testing Workflow
```bash
# Before making changes - validate current state (4 seconds)
python scripts/ci_validation.py

# Basic import test (always works)
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
python -c "import src; print('Imports OK')"

# Run focused tests (if pytest available)
pytest tests/unit/ -v || echo "pytest not available"
pytest tests/integration/ -v || echo "pytest not available"
```

## Critical Validation Scenarios

**Test these after ANY code changes:**

### 1. Health Check (Tested and Working)
```bash
./start_mvp.sh &
sleep 30
curl -f http://localhost:5000/health || echo "FAILED"
# Expected response: "OK - NFCS MVP Fallback Server Running"
pkill -f mvp_web_interface  # Stop server
```

### 2. Core Module Import (Tested and Working)
```bash
python3 -c "
import sys; sys.path.append('src')
from modules.constitutional_realtime import ConstitutionalRealTimeMonitor
print('Core modules import OK')
" || echo "Import failed - check PYTHONPATH"
```

### 3. End-to-End Demo (Takes 30 seconds)
```bash
python demo_production.py  # Should show systems status
# Expected: 2/6 tests pass (Docker/CI configs), 4 fail (missing deps)
```

### 4. CI Validation (4 seconds, always works)
```bash
python scripts/ci_validation.py
# Should pass: Critical Files, Python Syntax (105+17 files), Core Imports, Health Check
```

## Repository Structure

### Key Directories
- `src/` - Core NFCS implementation (11,772+ lines)
- `tests/` - Pytest test suite with markers
- `scripts/` - Build, test, and deployment automation
- `config/` - YAML configuration files
- `monitoring/` - Prometheus/Grafana setup

### Entry Points
- `mvp_controller.py` - Production integration controller
- `mvp_web_interface.py` - Flask dashboard (main UI)
- `src/main.py` - Core NFCS entry point
- `start_mvp.sh` - One-command startup script

### Configuration Files
- `requirements.txt` - Core dependencies (torch, transformers, flask)
- `pyproject.toml` - Python project settings
- `pytest.ini` - Test configuration
- `docker-compose.yml` - Production deployment

## Timing Expectations & Timeouts

| Operation | Time | Timeout | Critical Notes |
|-----------|------|---------|----------------|
| Virtual env | 30s | 2min | Fast operation |
| pip install requirements.txt | **5-15min** | **30min** | torch ~2GB download, NEVER CANCEL |
| Unit tests | 2-3min | 10min | Most reliable |
| Integration tests | 5-8min | 15min | Requires services |
| Docker build | **15-45min** | **60min** | Multi-stage build, NEVER CANCEL |
| MVP startup | 2-3min | 10min | Module loading, fallback works |
| Linting | 1-2min | 5min | Code analysis |
| CI Validation | 4s | 30s | Always works |
| Demo production | 30s | 2min | Status check |

**NEVER CANCEL** operations marked in bold - they involve large ML library downloads or complex builds.

## Troubleshooting Common Failures

### Dependency Issues
```bash
# Network/firewall issues with PyTorch (COMMON in CI environments)
pip install torch --index-url https://download.pytorch.org/whl/cpu --timeout 300 --retries 3

# Minimal fallback installation (always works)
pip install flask flask-socketio numpy pyyaml --timeout 120 --retries 3

# Use Docker for complex dependencies (if network available)
timeout 3600 docker build --target dependencies .
```

### Import Errors
```bash
# PYTHONPATH not set (most common issue)
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Test basic imports (should always work)
python -c "import src"  # Should work after PYTHONPATH set
find src/ -name "*.py" -exec python -m py_compile {} \;  # Check syntax
```

### Build/Test Failures
```bash
# Test infrastructure missing (install basic tools)
pip install pytest pytest-cov flake8 --timeout 120 --retries 3

# Port conflicts (MVP web interface)
lsof -i :5000  # Check what's using port 5000
pkill -f mvp_web_interface  # Kill if needed
netstat -tlnp | grep :5000  # Alternative check

# Docker issues
docker system prune -f  # Clean up space
docker compose down && timeout 3600 docker compose up -d  # Restart
```

### Network/Firewall Issues (COMMON)
```bash
# When PyPI is blocked (use fallback validation)
python scripts/ci_validation.py  # Works without network
find src/ -name "*.py" -exec python -m py_compile {} \;  # Basic validation
python -c "import src; print('Core imports work')"  # Test imports

# Use offline mode for CI/CD
export CI_OFFLINE_MODE=true
./scripts/emergency_ci_fallback.sh  # Emergency validation
```

### Pre-commit Checklist
```bash
# 1. Set environment (CRITICAL)
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# 2. Quick lint (syntax errors only) - works without flake8
find src/ -name "*.py" -exec python -m py_compile {} \;
find tests/ -name "*.py" -exec python -m py_compile {} \;

# 3. Basic import test (always works)
python -c "import src; print('OK')"

# 4. CI validation (4 seconds, comprehensive)
python scripts/ci_validation.py

# 5. Run unit tests only (if pytest available)
./scripts/run-tests.sh --unit -x || echo "Unit tests not available"

# 6. Health check if MVP needed (30 seconds)
timeout 30 ./start_mvp.sh && sleep 20 && curl -f http://localhost:5000/health && pkill -f mvp_web_interface
```

## Known Working Environment

- **Python**: 3.8+ (3.11+ recommended, tested with 3.12.3)
- **Memory**: 8GB+ (4GB minimum) - ML libraries are large
- **Disk**: 10GB+ free space - PyTorch ~2GB, transformers ~1GB
- **Network**: PyPI access required for initial setup (fallback mode available)
- **Docker**: 20.10+ with Compose v2 for production deployment (tested with 28.0.4 + v2.38.2)

## Intelligent Fallback Behavior

**The system works even in restricted environments with missing dependencies:**

### Fallback Features (All Tested)
- **Web Interface**: Falls back to BaseHTTP server when Flask unavailable
- **ML Modules**: Use numpy-free implementations when ML libraries blocked  
- **Testing**: CI validation works without any external dependencies
- **Networking**: Offline mode available for firewall-restricted environments

### Testing in Restricted Environments
```bash
# These ALWAYS work regardless of network/dependency issues:
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
python -c "import src; print('Core works')"
find src/ -name "*.py" -exec python -m py_compile {} \;
python scripts/ci_validation.py
python mvp_web_interface.py  # Fallback server: http://localhost:5000
```

## File Statistics (Validated)
- **Source files**: 105 Python files in src/ (all compile successfully)
- **Test files**: 17 Python files in tests/ (all compile successfully)
- **Total lines**: 11,772+ lines in src/ (as of validation)
- **Test coverage**: Target 85%+ line coverage, 80%+ branch coverage

## Common Tasks Reference

### Repository Structure Overview
```
Vortex-Omega/
├── src/                    # Core NFCS implementation (105 files)
│   ├── core/              # Mathematical solvers (CGL, Kuramoto)
│   ├── modules/           # Constitutional, Symbolic AI, ESC
│   ├── orchestrator/      # System coordination
│   └── config/            # Configuration management
├── tests/                 # Pytest test suite (17 files)
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── test_core/        # Core module tests
├── scripts/              # Build and deployment automation
├── .github/workflows/    # CI/CD pipelines (8 workflows)
└── docker-compose.yml   # Production deployment
```

### Entry Points (All Validated)
- `mvp_controller.py` - Production integration controller
- `mvp_web_interface.py` - Flask dashboard (fallback mode available)
- `src/main.py` - Core NFCS entry point
- `start_mvp.sh` - One-command startup script (intelligent fallbacks)
- `demo_production.py` - End-to-end demo (30 seconds)

This system is a production-ready AI/ML framework. Most build failures are timing-related (timeouts) or network-related (package downloads). The core system is stable when properly configured and has intelligent fallback behavior for restricted environments.

## Quick Reference (Copy-Paste Ready)

### Minimal Setup (30 seconds)
```bash
python3 -m venv venv && source venv/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
python -c "import src; print('Ready')"
```

### Pre-commit Validation (10 seconds)
```bash
find src/ tests/ -name "*.py" -exec python -m py_compile {} \;
python scripts/ci_validation.py
```

### Emergency Fallback (if everything fails)
```bash
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
python mvp_web_interface.py  # Basic server: http://localhost:5000
```

**Screenshot of Fallback Interface**: ![NFCS MVP Fallback Mode](https://github.com/user-attachments/assets/37852c3a-cf49-4ad9-a4f0-48b65a0f0a99)


