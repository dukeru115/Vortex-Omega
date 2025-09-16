#!/bin/bash
# NFCS MVP Startup Script
# Neural Field Control System v2.4.3 - Production Deployment

set -e

echo "🚀 NFCS MVP Startup Script v2.4.3"
echo "=================================="

# Check if we're in the correct directory
if [ ! -f "mvp_controller.py" ]; then
    echo "❌ Error: mvp_controller.py not found. Please run from Vortex-Omega directory."
    exit 1
fi

# Set up environment
export PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
echo "✅ PYTHONPATH configured: ${PYTHONPATH}"

# Create logs directory
mkdir -p logs

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "🐍 Python version: ${python_version}"

# Install dependencies with error handling
echo "📦 Installing Python dependencies..."
if ! timeout 900 pip install -r requirements.txt --retries 3; then
    echo "⚠️  Some dependencies failed to install, trying minimal setup..."
    pip install flask flask-socketio || echo "Minimal dependencies also failed - continuing anyway"
fi

# Try to install PM2, but don't fail if it doesn't work
if ! command -v pm2 &> /dev/null; then
    echo "📦 Attempting to install PM2..."
    if ! npm install -g pm2 2>/dev/null; then
        echo "⚠️  PM2 installation failed - will run directly"
        USE_PM2=false
    else
        USE_PM2=true
    fi
else
    USE_PM2=true
fi

echo "🔧 Starting NFCS MVP services..."

if [ "$USE_PM2" = true ]; then
    # PM2 mode
    echo "🌐 Starting MVP Web Interface with PM2..."
    
    # Stop any existing instances
    pm2 stop nfcs-mvp-web 2>/dev/null || true
    pm2 delete nfcs-mvp-web 2>/dev/null || true
    
    # Start MVP web interface with PM2
    pm2 start ecosystem.config.js
    
    # Show status
    echo "📊 PM2 Status:"
    pm2 status
    
    echo ""
    echo "✅ NFCS MVP Successfully Started with PM2!"
    echo "================================="
    echo "📱 Web Dashboard: http://localhost:5000"
    echo "🔧 PM2 Management: pm2 status"
    echo "📝 Logs: pm2 logs nfcs-mvp-web --nostream"
    echo "⏹️  Stop: pm2 stop nfcs-mvp-web"
else
    # Direct mode
    echo "🌐 Starting MVP Web Interface directly..."
    echo "📱 Web Dashboard will be available at: http://localhost:5000"
    echo "⏹️  Stop with Ctrl+C"
    echo ""
    
    # Run validation first
    python scripts/ci_validation.py || echo "Validation completed with warnings"
    
    # Start the web interface directly
    python mvp_web_interface.py
fi

echo ""
echo "🎯 Ready for MVP demonstration!"