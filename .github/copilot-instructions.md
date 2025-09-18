# Vortex-Omega: Copilot Onboarding Instructions

**ALWAYS follow these instructions first to reduce PR failures and build issues.**

Vortex-Omega is a hybrid AI system with Neural Field Control System (NFCS) featuring multi-agent consensus (Kuramoto/ADMM), constitutional monitoring, and Flask web interface. This is a Python 3.8+ AI/ML project with Docker support.

## Essential Setup - NEVER CANCEL Long Operations

### Environment Setup (2-3 minutes)
```bash
# Verify requirements
python3 --version  # Must be 3.8+ (3.11+ recommended)

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
pip install -r requirements.txt  # 5-15 minutes

# Development dependencies (optional)
pip install -r requirements-dev.txt  # 2-5 minutes
```

### Build & Test (TIMEOUT: 10+ minutes each)
```bash
# Linting (1-2 minutes)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=venv,env
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude=venv,env

# Tests (2-10 minutes, TIMEOUT: 15+ minutes)
./scripts/run-tests.sh unit          # Unit tests (~2 minutes)
./scripts/run-tests.sh integration   # Integration tests (~5 minutes)
./scripts/run-tests.sh --all         # All tests (~10 minutes)
```

## Core Commands & Workflows

### Quick Start (MVP Demo)
```bash
# Single command startup (2-3 minutes, TIMEOUT: 10+ minutes)
./start_mvp.sh
# Access dashboard: http://localhost:5000

# Manual startup if script fails
python mvp_web_interface.py
```

### Docker Production (TIMEOUT: 60+ minutes)
```bash
# Full production stack (15-45 minutes build time)
docker compose up -d
# Services: http://localhost:8080, Grafana: http://localhost:3000
```

### Testing Workflow
```bash
# Before making changes - validate current state
python scripts/ci_validation.py

# Basic import test (always works)
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
python -c "import src; print('Imports OK')"

# Run focused tests
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
```

## Critical Validation Scenarios

**Test these after ANY code changes:**

### 1. Health Check
```bash
./start_mvp.sh &
sleep 30
curl -f http://localhost:5000/health || echo "FAILED"
```

### 2. Core Module Import
```bash
python3 -c "
import sys; sys.path.append('src')
from modules.constitutional_realtime import ConstitutionalRealtimeMonitor
print('Core modules import OK')
"
```

### 3. End-to-End Demo
```bash
python demo_production.py  # Should show all systems active
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
| pip install requirements.txt | **5-15min** | **30min** | torch ~2GB download |
| Unit tests | 2-3min | 10min | Most reliable |
| Integration tests | 5-8min | 15min | Requires services |
| Docker build | **15-45min** | **60min** | Multi-stage build |
| MVP startup | 2-3min | 10min | Module loading |
| Linting | 1-2min | 5min | Code analysis |

**NEVER CANCEL** operations marked in bold - they involve large ML library downloads.

## Troubleshooting Common Failures

### Dependency Issues
```bash
# Network/firewall issues with PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Minimal fallback installation
pip install flask flask-socketio numpy

# Use Docker for complex dependencies
docker build --target dependencies .
```

### Import Errors
```bash
# PYTHONPATH not set (most common issue)
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Test basic imports
python -c "import src"  # Should work after PYTHONPATH set
```

### Build/Test Failures
```bash
# Test infrastructure missing
pip install pytest pytest-cov flake8

# Port conflicts (MVP web interface)
lsof -i :5000  # Check what's using port 5000
kill -9 $(lsof -ti :5000)  # Kill if needed

# Docker issues
docker system prune  # Clean up space
docker compose down && docker compose up -d  # Restart
```

### Pre-commit Checklist
```bash
# 1. Set environment
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# 2. Quick lint (syntax errors only)
find src/ -name "*.py" -exec python -m py_compile {} \;

# 3. Basic import test
python -c "import src; print('OK')"

# 4. Run unit tests only (fastest)
pytest tests/unit/ -x  # Stop on first failure

# 5. Health check if MVP needed
./start_mvp.sh && sleep 20 && curl -f http://localhost:5000/health
```

## Known Working Environment

- **Python**: 3.8+ (3.11+ recommended, tested with 3.12)
- **Memory**: 8GB+ (4GB minimum) - ML libraries are large
- **Disk**: 10GB+ free space - PyTorch ~2GB, transformers ~1GB
- **Network**: PyPI access required for initial setup (can cache afterwards)
- **Docker**: 20.10+ with Compose v2 for production deployment

This system is a production-ready AI/ML framework. Most build failures are timing-related (timeouts) or network-related (package downloads). The core system is stable when properly configured.


