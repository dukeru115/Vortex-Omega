#!/usr/bin/env python3
"""
NFCS MVP Web Interface
=====================

Real-time web dashboard for Neural Field Control System MVP demonstration.
Provides live monitoring, system status, and interactive controls.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any
import os
import sys

# Web framework imports with fallback
try:
    from flask import Flask, render_template_string, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    print("Installing required web dependencies...")
    try:
        import subprocess
        result = subprocess.run(
            ["pip", "install", "flask", "flask-socketio"], 
            capture_output=True, 
            timeout=60
        )
        if result.returncode == 0:
            from flask import Flask, render_template_string, jsonify, request
            from flask_socketio import SocketIO, emit
            FLASK_AVAILABLE = True
        else:
            raise ImportError("Flask installation failed")
    except Exception as e:
        print(f"Flask installation failed: {e}")
        print("⚠️ Using fallback HTTP server mode")
        FLASK_AVAILABLE = False
        
        # Fallback HTTP server implementation
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        
        class MVPFallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    html_content = '''
                    <html><head><title>NFCS MVP - Fallback Mode</title></head>
                    <body>
                    <h1>NFCS MVP - Fallback Mode</h1>
                    <p><strong>Status:</strong> Running in minimal mode (Flask not available)</p>
                    <p><strong>Note:</strong> Limited functionality due to missing dependencies</p>
                    <p>To get full functionality, install: pip install flask flask-socketio</p>
                    <p><a href="/health">Health Check</a> | <a href="/status">Status API</a></p>
                    </body></html>
                    '''
                    self.wfile.write(html_content.encode('utf-8'))
                elif self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'OK - NFCS MVP Fallback Server Running')
                elif self.path == '/status':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    status = {"status": "fallback", "message": "Running in minimal mode"}
                    self.wfile.write(json.dumps(status).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
                    
            def log_message(self, format, *args):
                # Suppress default logging
                pass

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from mvp_controller import NFCSMinimalViableProduct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup (conditional)
if FLASK_AVAILABLE:
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'nfcs-mvp-secret-key'
    socketio = SocketIO(app, cors_allowed_origins="*")
else:
    app = None
    socketio = None

# Global MVP instance
mvp_instance = None
monitoring_task = None

# HTML Template for MVP Dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NFCS MVP Dashboard v2.4.3</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: #fff;
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(90deg, #0f3460, #533483);
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            text-align: center;
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            text-align: center;
            opacity: 0.8;
            font-size: 1.1rem;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 1rem;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h3 {
            color: #64ffda;
            margin-bottom: 1rem;
            font-size: 1.2rem;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-operational { background-color: #4caf50; }
        .status-warning { background-color: #ff9800; }
        .status-error { background-color: #f44336; }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 0.5rem 0;
            padding: 0.5rem;
            background: rgba(255,255,255,0.05);
            border-radius: 5px;
        }
        
        .metric-value {
            font-weight: bold;
            color: #64ffda;
        }
        
        .controls {
            grid-column: 1 / -1;
            text-align: center;
        }
        
        .btn {
            background: linear-gradient(45deg, #533483, #0f3460);
            color: white;
            border: none;
            padding: 1rem 2rem;
            margin: 0.5rem;
            border-radius: 5px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(83, 52, 131, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .chart-container {
            height: 300px;
            margin-top: 1rem;
        }
        
        .log-container {
            grid-column: 1 / -1;
            max-height: 200px;
            overflow-y: auto;
            background: rgba(0,0,0,0.3);
            padding: 1rem;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }
        
        .log-entry {
            margin: 0.2rem 0;
            opacity: 0.8;
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Neural Field Control System</h1>
        <p>MVP Dashboard v2.4.3 - Real-time Constitutional AI Monitoring</p>
    </div>
    
    <div class="dashboard">
        <!-- System Status -->
        <div class="card">
            <h3>🚀 System Status</h3>
            <div class="metric">
                <span>Health:</span>
                <span class="metric-value" id="system-health">
                    <span class="status-indicator" id="health-indicator"></span>
                    <span id="health-text">Initializing</span>
                </span>
            </div>
            <div class="metric">
                <span>Constitutional Status:</span>
                <span class="metric-value" id="constitutional-status">Initializing</span>
            </div>
            <div class="metric">
                <span>Cognitive Modules:</span>
                <span class="metric-value" id="cognitive-modules">0/5</span>
            </div>
            <div class="metric">
                <span>Last Update:</span>
                <span class="metric-value" id="last-update">Never</span>
            </div>
        </div>
        
        <!-- Performance Metrics -->
        <div class="card">
            <h3>📊 Performance Metrics</h3>
            <div class="metric">
                <span>Kuramoto Sync Level:</span>
                <span class="metric-value" id="sync-level">0.000</span>
            </div>
            <div class="metric">
                <span>Validation Score:</span>
                <span class="metric-value" id="validation-score">0.000</span>
            </div>
            <div class="metric">
                <span>Active Predictions:</span>
                <span class="metric-value" id="active-predictions">0</span>
            </div>
            <div class="metric">
                <span>Safety Violations:</span>
                <span class="metric-value" id="safety-violations">0</span>
            </div>
        </div>
        
        <!-- System Capabilities -->
        <div class="card">
            <h3>⚡ Core Capabilities</h3>
            <div class="metric">
                <span>Constitutional Oversight:</span>
                <span class="metric-value">✅ Active</span>
            </div>
            <div class="metric">
                <span>ESC-Kuramoto Integration:</span>
                <span class="metric-value">✅ 64 Oscillators</span>
            </div>
            <div class="metric">
                <span>Cognitive Modules:</span>
                <span class="metric-value">✅ 5 Systems</span>
            </div>
            <div class="metric">
                <span>Empirical Validation:</span>
                <span class="metric-value">✅ Real-time</span>
            </div>
        </div>
        
        <!-- Real-time Chart -->
        <div class="card">
            <h3>📈 Real-time Synchronization</h3>
            <div class="chart-container">
                <canvas id="syncChart"></canvas>
            </div>
        </div>
        
        <!-- Validation Chart -->
        <div class="card">
            <h3>🎯 Validation Metrics</h3>
            <div class="chart-container">
                <canvas id="validationChart"></canvas>
            </div>
        </div>
        
        <!-- System Architecture -->
        <div class="card">
            <h3>🏗️ Architecture Overview</h3>
            <div style="text-align: center; margin-top: 1rem;">
                <div style="margin: 0.5rem 0;">🧠 Cognitive Layer (5 Modules)</div>
                <div style="margin: 0.5rem 0;">⚖️ Constitutional Framework</div>
                <div style="margin: 0.5rem 0;">🔄 ESC-Kuramoto Integration</div>
                <div style="margin: 0.5rem 0;">📊 Empirical Validation</div>
                <div style="margin: 0.5rem 0;">🚀 MVP Controller</div>
            </div>
        </div>
        
        <!-- Controls -->
        <div class="card controls">
            <h3>🎮 System Controls</h3>
            <button class="btn" id="start-btn" onclick="startSystem()">🚀 Start MVP</button>
            <button class="btn" id="stop-btn" onclick="stopSystem()" disabled>⏹️ Stop MVP</button>
            <button class="btn" onclick="refreshStatus()">🔄 Refresh Status</button>
            <button class="btn" onclick="demonstrateCapabilities()">🎯 Demo Capabilities</button>
        </div>
        
        <!-- System Log -->
        <div class="card">
            <h3>📝 System Log</h3>
            <div class="log-container" id="system-log">
                <div class="log-entry">System initialized - waiting for commands...</div>
            </div>
        </div>
    </div>

    <script>
        // Socket.IO connection
        const socket = io();
        
        // Chart instances
        let syncChart, validationChart;
        
        // Initialize charts
        function initCharts() {
            const syncCtx = document.getElementById('syncChart').getContext('2d');
            syncChart = new Chart(syncCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Synchronization Level',
                        data: [],
                        borderColor: '#64ffda',
                        backgroundColor: 'rgba(100, 255, 218, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: true, max: 1 }
                    }
                }
            });
            
            const validationCtx = document.getElementById('validationChart').getContext('2d');
            validationChart = new Chart(validationCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Validation Score',
                        data: [],
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: true, max: 1 }
                    }
                }
            });
        }
        
        // Update UI with status data
        function updateUI(data) {
            document.getElementById('system-health').innerHTML = 
                `<span class="status-indicator status-${data.system_health === 'operational' ? 'operational' : 'warning'}"></span>${data.system_health}`;
            document.getElementById('constitutional-status').textContent = data.constitutional_status;
            document.getElementById('cognitive-modules').textContent = `${data.cognitive_modules_active}/5`;
            document.getElementById('sync-level').textContent = data.kuramoto_sync_level.toFixed(3);
            document.getElementById('validation-score').textContent = data.validation_score.toFixed(3);
            document.getElementById('active-predictions').textContent = data.active_predictions;
            document.getElementById('safety-violations').textContent = data.safety_violations;
            
            const now = new Date().toLocaleTimeString();
            document.getElementById('last-update').textContent = now;
            
            // Update charts
            const timeLabel = now;
            
            syncChart.data.labels.push(timeLabel);
            syncChart.data.datasets[0].data.push(data.kuramoto_sync_level);
            if (syncChart.data.labels.length > 20) {
                syncChart.data.labels.shift();
                syncChart.data.datasets[0].data.shift();
            }
            syncChart.update('none');
            
            validationChart.data.labels.push(timeLabel);
            validationChart.data.datasets[0].data.push(data.validation_score);
            if (validationChart.data.labels.length > 20) {
                validationChart.data.labels.shift();
                validationChart.data.datasets[0].data.shift();
            }
            validationChart.update('none');
        }
        
        // Add log entry
        function addLogEntry(message) {
            const logContainer = document.getElementById('system-log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logContainer.appendChild(entry);
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        
        // Control functions
        function startSystem() {
            socket.emit('start_mvp');
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            addLogEntry('MVP startup initiated...');
        }
        
        function stopSystem() {
            socket.emit('stop_mvp');
            document.getElementById('start-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
            addLogEntry('MVP shutdown initiated...');
        }
        
        function refreshStatus() {
            socket.emit('get_status');
            addLogEntry('Status refresh requested...');
        }
        
        function demonstrateCapabilities() {
            socket.emit('demo_capabilities');
            addLogEntry('Capabilities demonstration started...');
        }
        
        // Socket event handlers
        socket.on('status_update', function(data) {
            updateUI(data);
        });
        
        socket.on('log_message', function(data) {
            addLogEntry(data.message);
        });
        
        socket.on('connect', function() {
            addLogEntry('Connected to MVP backend');
            refreshStatus();
        });
        
        // Initialize when page loads
        window.addEventListener('load', function() {
            initCharts();
            addLogEntry('Dashboard initialized - ready for MVP demonstration');
        });
    </script>
</body>
</html>
"""

# Flask/SocketIO Routes (only available when Flask is installed)
if FLASK_AVAILABLE:
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        logger.info("Client connected to MVP dashboard")
        emit('log_message', {'message': 'Connected to NFCS MVP Dashboard'})

    @socketio.on('start_mvp')
    def handle_start_mvp():
        """Start the MVP system."""
        global mvp_instance, monitoring_task
        
        async def start_mvp_async():
            global mvp_instance, monitoring_task
            
            try:
                mvp_instance = NFCSMinimalViableProduct()
                success = await mvp_instance.start_mvp()
                
                if success:
                    # Start monitoring task
                    monitoring_task = asyncio.create_task(mvp_monitoring_loop())
                    socketio.emit('log_message', {'message': 'MVP started successfully!'})
                else:
                    socketio.emit('log_message', {'message': 'MVP startup failed!'})
                    
            except Exception as e:
                logger.error(f"MVP start error: {e}")
                socketio.emit('log_message', {'message': f'MVP startup error: {e}'})
        
        # Run in asyncio loop
        asyncio.create_task(start_mvp_async())

    @socketio.on('stop_mvp')
    def handle_stop_mvp():
        """Stop the MVP system."""
        global mvp_instance, monitoring_task
        
        if mvp_instance:
            mvp_instance.stop_mvp()
            mvp_instance = None
        
        if monitoring_task:
            monitoring_task.cancel()
            monitoring_task = None
        
        emit('log_message', {'message': 'MVP stopped'})

    @socketio.on('get_status')
    def handle_get_status():
        """Get current MVP status."""
        if mvp_instance:
            status_dict = mvp_instance.status.__dict__
            emit('status_update', status_dict)
        else:
            emit('log_message', {'message': 'MVP not running'})

    @socketio.on('demo_capabilities')
    def handle_demo_capabilities():
        """Demonstrate MVP capabilities."""
        async def demo_async():
            if mvp_instance:
                capabilities = await mvp_instance.demonstrate_capabilities()
                for cap, desc in capabilities.items():
                    socketio.emit('log_message', {'message': f'✓ {cap}: {desc}'})
            else:
                socketio.emit('log_message', {'message': 'MVP not running - cannot demonstrate'})
        
        asyncio.create_task(demo_async())

    async def mvp_monitoring_loop():
        """Background monitoring loop for MVP."""
        global mvp_instance
        
        while mvp_instance and mvp_instance.running:
            try:
                await mvp_instance.run_monitoring_cycle()
                
                # Emit status update
                status_dict = mvp_instance.status.__dict__
                socketio.emit('status_update', status_dict)
                
                await asyncio.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                socketio.emit('log_message', {'message': f'Monitoring error: {e}'})
                break

    @app.route('/')
    def dashboard():
        """Main dashboard route."""
        return render_template_string(HTML_TEMPLATE)

    @app.route('/api/status')
    def api_status():
        """API endpoint for status."""
        if mvp_instance:
            return jsonify(mvp_instance.status.__dict__)
        else:
            return jsonify({"error": "MVP not running"})

    @app.route('/api/metrics')
    def api_metrics():
        """API endpoint for metrics summary."""
        if mvp_instance:
            return jsonify(mvp_instance.get_metrics_summary())
        else:
            return jsonify({"error": "MVP not running"})

def run_web_interface(host='0.0.0.0', port=5000, debug=False):
    """Run the web interface with fallback support."""
    logger.info(f"🌐 Starting NFCS MVP Web Interface on {host}:{port}")
    
    if FLASK_AVAILABLE:
        logger.info("✅ Using Flask web server")
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    else:
        logger.info("⚠️ Using fallback HTTP server (limited functionality)")
        httpd = HTTPServer((host, port), MVPFallbackHandler)
        logger.info(f"📱 Fallback server running at http://{host}:{port}")
        logger.info("🔧 Install Flask for full functionality: pip install flask flask-socketio")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("🛑 Server stopped by user")
            httpd.shutdown()

if __name__ == '__main__':
    try:
        run_web_interface(debug=True)
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        print("❌ Web interface failed to start")
        print("💡 This may be due to missing dependencies")
        print("🔧 Try: pip install flask flask-socketio")
        sys.exit(1)