# Vortex-Omega: Neural Field Control System (NFCS) v2.4.3

**Always follow these instructions first and fallback to additional search or bash commands only when you encounter unexpected information that does not match the info here.**

Vortex-Omega is a hybrid AI system with Neural Field Control System (NFCS) featuring constitutional monitoring, multi-agent consensus (Kuramoto/ADMM), causal world models, interpretable outputs, and a production-ready MVP with Flask web interface.

## Working Effectively

### Initial Setup (Required First Steps)

**CRITICAL: Set appropriate timeouts for all build commands. NEVER CANCEL long-running operations.**

```bash
# Environment check and setup
python3 --version  # Requires Python 3.8+ (3.11+ recommended)
which docker && docker --version  # Docker available
which git && git --version

# Create virtual environment - ALWAYS do this first
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip to avoid dependency issues
python -m pip install --upgrade pip
```

### Dependency Installation (5-15 minutes, NEVER CANCEL)

```bash
# Install core dependencies - takes 5-15 minutes depending on PyTorch/transformers
# NEVER CANCEL: Set timeout to 30+ minutes for safety
pip install -r requirements.txt

# For development work, also install dev dependencies
pip install -r requirements-dev.txt

# Known limitations: 
# - torch>=2.0.0 is large (~2GB download) and may fail on slow connections
# - transformers>=4.20.0 requires significant disk space
# If pip install fails due to network/firewall, use Docker approach instead
```

### Build and Test Commands (NEVER CANCEL)

```bash
# Run linting - takes 1-2 minutes
# NEVER CANCEL: Set timeout to 5+ minutes
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=venv,env
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude=venv,env

# Run tests - takes 2-10 minutes depending on scope
# NEVER CANCEL: Set timeout to 15+ minutes
./scripts/run-tests.sh

# Run specific test types
./scripts/run-tests.sh unit          # Unit tests only (~2 minutes)
./scripts/run-tests.sh integration   # Integration tests (~5 minutes)  
./scripts/run-tests.sh --all         # All tests (~10 minutes)
```

### Running the Application

#### MVP Quick Start (Recommended)
```bash
# Single command startup - takes 2-3 minutes
# NEVER CANCEL: Set timeout to 10+ minutes
./start_mvp.sh

# This will:
# 1. Install dependencies if missing
# 2. Start MVP web interface on http://localhost:5000
# 3. Initialize all NFCS components
# 4. Start constitutional monitoring
# 5. Activate ESC-Kuramoto integration
```

#### Manual MVP Startup
```bash
# If start_mvp.sh fails, run manually:
pip install -r requirements.txt

# Start MVP web interface directly
python mvp_web_interface.py
# Access at: http://localhost:5000

# Or start MVP controller only
python mvp_controller.py
```

#### Docker Deployment (Production)
```bash
# Build production image - takes 10-45 minutes, NEVER CANCEL
# NEVER CANCEL: Set timeout to 60+ minutes
docker build --target production -t vortex-omega .

# Or use docker-compose for full stack - takes 15-60 minutes
# NEVER CANCEL: Set timeout to 90+ minutes  
docker compose up -d

# Services will be available on:
# - Main app: http://localhost:8080
# - Grafana: http://localhost:3000 (admin/vortex123)
# - Prometheus: http://localhost:9090
```

### Development Mode
```bash
# For active development with hot reload
docker compose --profile development up -d

# Or run traditional development mode
python -m src.main --mode development
# Access at: http://localhost:8080
```

## Validation Scenarios

**ALWAYS test these scenarios after making changes to ensure system functionality:**

### 1. Basic MVP Functionality Test
```bash
# Start MVP system
./start_mvp.sh

# Test web dashboard accessibility
curl -f http://localhost:5000/health || echo "Health check failed"

# Test constitutional monitoring endpoint
curl -f http://localhost:5000/api/constitutional/status || echo "Constitutional endpoint failed"

# Test Kuramoto synchronization status
curl -f http://localhost:5000/api/kuramoto/sync || echo "Kuramoto endpoint failed"
```

### 2. Core System Integration Test
```bash
# Test main NFCS components can be imported
python3 -c "
import sys; sys.path.append('src')
from modules.constitutional_realtime import ConstitutionalRealtimeMonitor
from modules.esc_kuramoto_integration import ESCKuramotoIntegrationSystem
from modules.cognitive.constitution.constitution_core import ConstitutionModule
print('Core modules import successfully')
"
```

### 3. End-to-End Workflow Test
```bash
# Run the production demo to test full system
python demo_production.py

# Expected output: 
# - Constitutional monitoring active
# - Kuramoto oscillators synchronized  
# - Cognitive modules loaded
# - Real-time metrics displayed
```

## Key Project Components

### 1. MVP Production System (Recently Completed)
- **Location**: `mvp_controller.py`, `mvp_web_interface.py`
- **Purpose**: Production-ready demonstration of all NFCS capabilities
- **Usage**: `./start_mvp.sh` for one-command startup
- **Web Interface**: Flask + Socket.IO dashboard at http://localhost:5000

### 2. Core NFCS Components
- **Neural Field Control**: `src/core/` - Main NFCS algorithms and field processing
- **Constitutional Monitoring**: `src/modules/constitutional_realtime.py` - Algorithm 1 implementation with Ha monitoring
- **ESC-Kuramoto Integration**: `src/modules/esc_kuramoto_integration.py` - 64-oscillator semantic synchronization
- **Cognitive Modules**: `src/modules/cognitive/` - 5 complete systems (Constitution, Symbolic AI, Memory, Reflection, Freedom)

### 3. Testing and Validation
- **Unit Tests**: `tests/` - Comprehensive test suite with pytest
- **Integration Tests**: `tests/integration/` - Cross-module integration testing
- **Performance Tests**: Available via `./scripts/run-tests.sh --performance`
- **CI/CD**: `.github/workflows/` - GitHub Actions for automated testing

### 4. Configuration and Scripts
- **Scripts**: `scripts/` - Automation scripts for testing, deployment, Docker
- **Configuration**: `config/` - YAML configuration files
- **Docker**: `Dockerfile`, `docker-compose.yml` - Containerized deployment

## Common Tasks Reference

### Repository Root Structure
```
.
├── README.md                 # Main project documentation
├── requirements.txt          # Python dependencies (core ML/AI packages)
├── requirements-dev.txt      # Development dependencies
├── pyproject.toml           # Python project configuration
├── mvp_controller.py        # MVP integration controller
├── mvp_web_interface.py     # Flask web dashboard
├── start_mvp.sh             # One-command MVP startup
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # Full stack deployment
├── src/                     # Core source code (11,772+ lines)
├── tests/                   # Test suite with pytest
├── scripts/                 # Build and deployment scripts
├── config/                  # Configuration files
├── docs/                    # Extended documentation
├── notebooks/               # Jupyter notebooks and examples
└── monitoring/             # Prometheus/Grafana monitoring
```

### Common Commands Quick Reference
```bash
# Setup
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# MVP Demo
./start_mvp.sh

# Testing  
./scripts/run-tests.sh

# Docker Production
docker compose up -d

# Development Mode
python -m src.main --mode development

# Health Check
curl -f http://localhost:5000/health
```

## Timing Expectations and Timeouts

| Operation | Expected Time | Recommended Timeout | Notes |
|-----------|---------------|--------------------:|-------|
| Virtual env creation | 10-30 seconds | 2 minutes | Usually very fast |
| Pip install requirements.txt | 5-15 minutes | 30 minutes | PyTorch/transformers are large |
| Run unit tests | 1-3 minutes | 10 minutes | Depends on test scope |
| Run integration tests | 3-8 minutes | 15 minutes | Requires module loading |
| Run all tests | 5-15 minutes | 25 minutes | Full test suite |
| Docker build (production) | 15-45 minutes | 60 minutes | Multi-stage build with dependencies |
| Docker compose up | 10-30 minutes | 45 minutes | Downloads images + builds |
| MVP startup | 1-3 minutes | 10 minutes | Module initialization |
| Linting (flake8) | 30-90 seconds | 5 minutes | Code analysis |

**CRITICAL: NEVER CANCEL builds or tests. These are normal timings for ML/AI projects.**

## Troubleshooting Common Issues

### 1. Dependency Installation Failures
```bash
# If torch fails to install due to network/firewall issues:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# If requirements.txt fails completely, try minimal install:
pip install numpy scipy flask flask-socketio

# For firewall/network restrictions, use Docker instead:
docker build --target dependencies -t vortex-deps .
```

### 2. Import Errors
```bash
# If modules not found, ensure PYTHONPATH is set:
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# Or use absolute imports in tests:
python -m pytest tests/ --pythonpath=src
```

### 3. MVP Web Interface Issues
```bash
# If port 5000 is busy:
# Check what's running: lsof -i :5000
# Kill if necessary: kill -9 $(lsof -ti :5000)

# If PM2 installation fails in start_mvp.sh:
# Install manually: npm install -g pm2
# Or run directly: python mvp_web_interface.py
```

### 4. Docker Issues
```bash
# If Docker build fails with network issues:
docker build --network=host -t vortex-omega .

# If docker-compose fails:
# Use newer syntax: docker compose up -d
# Check Docker daemon: sudo systemctl status docker
```

### 5. Test Failures
```bash
# If tests fail due to missing dependencies:
pip install pytest pytest-cov pytest-asyncio

# Run tests with verbose output for debugging:
pytest -v --tb=long tests/

# Run specific test file:
pytest tests/test_constitutional_integration.py -v
```

## Pre-commit Validation

Always run these before committing changes:

```bash
# 1. Lint code
flake8 . --exclude=venv,env --max-line-length=127

# 2. Run tests  
./scripts/run-tests.sh unit

# 3. Test MVP functionality
./start_mvp.sh && sleep 30 && curl -f http://localhost:5000/health

# 4. Check imports work
python -c "import sys; sys.path.append('src'); from src.main import main"
```

## Known Working Configurations

- **Python**: 3.8+ (tested with 3.11, 3.12)
- **OS**: Linux (Ubuntu 20.04+), macOS 10.15+, Windows 10+ with WSL2
- **Docker**: 20.10+ with docker-compose v2.0+
- **Memory**: 8GB+ recommended (4GB minimum)
- **Disk**: 10GB+ free space for dependencies and Docker images
- **Network**: May require firewall exceptions for PyTorch/transformers downloads

## Architecture Notes

The system integrates:
- **Constitutional Monitoring** (Algorithm 1) for real-time oversight
- **ESC-Kuramoto Integration** with 64 semantic oscillators
- **Cognitive Modules** for advanced reasoning (321,922+ characters of code)
- **Empirical Validation** pipeline for testing and metrics
- **MVP Web Interface** for live monitoring and control

Key files to always check after making changes:
- `mvp_controller.py` - Central integration logic
- `src/modules/constitutional_realtime.py` - Constitutional monitoring
- `src/modules/esc_kuramoto_integration.py` - Synchronization system
- `tests/test_constitutional_integration.py` - Integration tests


