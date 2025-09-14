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

# Create logs directory
mkdir -p logs

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install PM2 if not present
if ! command -v pm2 &> /dev/null; then
    echo "📦 Installing PM2..."
    npm install -g pm2
fi

echo "🔧 Starting NFCS MVP services..."

# Stop any existing instances
pm2 stop nfcs-mvp-web 2>/dev/null || true
pm2 delete nfcs-mvp-web 2>/dev/null || true

# Start MVP web interface with PM2
echo "🌐 Starting MVP Web Interface..."
pm2 start ecosystem.config.js

# Show status
echo "📊 PM2 Status:"
pm2 status

echo ""
echo "✅ NFCS MVP Successfully Started!"
echo "================================="
echo "📱 Web Dashboard: http://localhost:5000"
echo "🔧 PM2 Management: pm2 status"
echo "📝 Logs: pm2 logs nfcs-mvp-web --nostream"
echo "⏹️  Stop: pm2 stop nfcs-mvp-web"
echo ""
echo "🎯 Ready for MVP demonstration!"