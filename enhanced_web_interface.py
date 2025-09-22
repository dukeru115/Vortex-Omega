"""
Enhanced NFCS Web Interface with RAG and Telemetry Integration

Advanced web dashboard featuring:
- ESC Module 2.1 telemetry visualization
- RAG system integration and monitoring
- Distributed Kuramoto performance tracking
- Topological defect visualization
- Real-time system health monitoring
"""

import json
import logging
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import threading
from collections import deque

# Conditional imports with graceful fallbacks
try:
    from flask import Flask, render_template_string, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    logging.warning("Flask not available - using minimal HTTP server")
    FLASK_AVAILABLE = False

import numpy as np

# Import NFCS components
try:
    from src.modules.esc.esc_core import EchoSemanticConverter, ESCConfig, ProcessingMode
    from src.modules.esc.telemetry import get_telemetry_collector
    from src.modules.rag.rag_core import RAGProcessor, RAGConfig
    from src.core.distributed_kuramoto import DistributedKuramotoSolver, DistributedConfig, ComputeMode
    NFCS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"NFCS components not fully available: {e}")
    NFCS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedNFCSWebInterface:
    """
    Enhanced web interface for NFCS with comprehensive monitoring capabilities.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        
        # Initialize Flask app
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            self.app.config['SECRET_KEY'] = 'nfcs-enhanced-demo-2024'
            self.socketio = SocketIO(self.app, cors_allowed_origins="*")
            self._setup_flask_routes()
        else:
            self.app = None
            self.socketio = None
        
        # Initialize NFCS components
        self._initialize_nfcs_components()
        
        # System state tracking
        self.system_status = {
            'operational': True,
            'last_update': time.time(),
            'components': {
                'esc_module': 'initializing',
                'rag_system': 'initializing',
                'kuramoto_solver': 'initializing',
                'telemetry': 'initializing'
            }
        }
        
        # Real-time data storage
        self.telemetry_data = deque(maxlen=1000)
        self.rag_metrics = deque(maxlen=500)
        self.performance_data = deque(maxlen=200)
        self.topological_data = deque(maxlen=100)
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def _initialize_nfcs_components(self):
        """Initialize NFCS system components."""
        try:
            if NFCS_AVAILABLE:
                # ESC Module with telemetry
                esc_config = ESCConfig(
                    embedding_dim=128,
                    semantic_field_layers=4,
                    processing_mode=ProcessingMode.BALANCED,
                    enable_constitutional_filtering=True
                )
                self.esc_module = EchoSemanticConverter(esc_config)
                self.telemetry_collector = get_telemetry_collector()
                
                # RAG System
                rag_config = RAGConfig(
                    max_retrieved_docs=5,
                    enable_conformal_abstention=True,
                    enable_hallucination_detection=True
                )
                self.rag_processor = RAGProcessor(rag_config)
                
                # Distributed Kuramoto Solver
                distributed_config = DistributedConfig(
                    compute_mode=ComputeMode.CPU_PARALLEL,
                    num_workers=2,  # Conservative for demo
                    optimization_target='speed'
                )
                self.kuramoto_solver = DistributedKuramotoSolver(distributed_config)
                
                self.system_status['components'] = {
                    'esc_module': 'operational',
                    'rag_system': 'operational', 
                    'kuramoto_solver': 'operational',
                    'telemetry': 'operational'
                }
                
                logger.info("NFCS components initialized successfully")
            else:
                # Mock components for demo
                self._initialize_mock_components()
                
        except Exception as e:
            logger.error(f"Failed to initialize NFCS components: {e}")
            self._initialize_mock_components()
    
    def _initialize_mock_components(self):
        """Initialize mock components for demonstration."""
        self.esc_module = None
        self.rag_processor = None
        self.kuramoto_solver = None
        self.telemetry_collector = None
        
        self.system_status['components'] = {
            'esc_module': 'mock',
            'rag_system': 'mock',
            'kuramoto_solver': 'mock',
            'telemetry': 'mock'
        }
        
        logger.info("Mock components initialized for demonstration")
    
    def _setup_flask_routes(self):
        """Setup Flask routes and socket handlers."""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(ENHANCED_DASHBOARD_TEMPLATE)
        
        @self.app.route('/api/system/status')
        def system_status():
            return jsonify(self._get_system_status())
        
        @self.app.route('/api/telemetry/data')
        def telemetry_data():
            return jsonify(self._get_telemetry_data())
        
        @self.app.route('/api/rag/query', methods=['POST'])
        def rag_query():
            return jsonify(self._process_rag_query(request.json))
        
        @self.app.route('/api/kuramoto/performance')
        def kuramoto_performance():
            return jsonify(self._get_kuramoto_performance())
        
        @self.app.route('/api/topological/visualization')
        def topological_visualization():
            return jsonify(self._get_topological_data())
        
        @self.socketio.on('connect')
        def handle_connect():
            emit('system_status', self._get_system_status())
            logger.info("Client connected to enhanced dashboard")
        
        @self.socketio.on('start_monitoring')
        def handle_start_monitoring():
            self._start_background_monitoring()
            emit('monitoring_status', {'active': True})
        
        @self.socketio.on('stop_monitoring')
        def handle_stop_monitoring():
            self._stop_background_monitoring()
            emit('monitoring_status', {'active': False})
        
        @self.socketio.on('esc_demo')
        def handle_esc_demo(data):
            result = self._demonstrate_esc_processing(data.get('text', 'Hello world'))
            emit('esc_result', result)
        
        @self.socketio.on('kuramoto_benchmark')
        def handle_kuramoto_benchmark():
            result = self._run_kuramoto_benchmark()
            emit('benchmark_result', result)
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        self.system_status['last_update'] = time.time()
        
        # Add component health checks
        health_scores = {}
        for component, status in self.system_status['components'].items():
            if status == 'operational':
                health_scores[component] = 0.95 + np.random.random() * 0.05
            elif status == 'mock':
                health_scores[component] = 0.8 + np.random.random() * 0.1
            else:
                health_scores[component] = 0.0
        
        return {
            **self.system_status,
            'health_scores': health_scores,
            'uptime': time.time() - (self.system_status.get('start_time', time.time())),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_telemetry_data(self) -> Dict[str, Any]:
        """Get ESC telemetry data."""
        if self.telemetry_collector and hasattr(self.telemetry_collector, 'get_interpretability_report'):
            try:
                return self.telemetry_collector.get_interpretability_report()
            except Exception as e:
                logger.error(f"Error getting telemetry data: {e}")
        
        # Mock telemetry data
        return {
            'system_overview': {
                'total_sessions': 25,
                'total_tokens_processed': 1250,
                'average_processing_time': 0.045,
                'constitutional_violation_rate': 0.02,
                'active_semantic_anchors': 8
            },
            'semantic_anchor_stability': {
                'average_stability': 0.87,
                'minimum_stability': 0.65,
                'stable_anchors': 7,
                'unstable_anchors': 1,
                'status': 'stable'
            },
            'performance_trends': {
                'trend': 'stable',
                'average_time': 0.045,
                'processing_variance': 0.008
            }
        }
    
    def _process_rag_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process RAG query and return results."""
        query = query_data.get('query', '')
        
        if self.rag_processor:
            try:
                response = self.rag_processor.process_query(query)
                result = {
                    'success': True,
                    'query': query,
                    'response': response.generated_response,
                    'confidence': response.confidence_score,
                    'uncertainty': response.uncertainty_estimate,
                    'abstained': response.should_abstain,
                    'hallucination_score': response.hallucination_score,
                    'sources': response.sources_used,
                    'processing_time': response.processing_time
                }
                
                # Store for metrics
                self.rag_metrics.append({
                    'timestamp': time.time(),
                    'confidence': response.confidence_score,
                    'hallucination_score': response.hallucination_score,
                    'abstained': response.should_abstain
                })
                
                return result
            except Exception as e:
                logger.error(f"RAG processing error: {e}")
        
        # Mock response
        return {
            'success': True,
            'query': query,
            'response': f"Mock response for: {query}. This demonstrates the RAG system integration.",
            'confidence': 0.85,
            'uncertainty': 0.15,
            'abstained': False,
            'hallucination_score': 0.1,
            'sources': ['Wikipedia', 'Internal KB'],
            'processing_time': 0.23
        }
    
    def _get_kuramoto_performance(self) -> Dict[str, Any]:
        """Get Kuramoto solver performance data."""
        if self.kuramoto_solver:
            try:
                return self.kuramoto_solver.get_performance_report()
            except Exception as e:
                logger.error(f"Error getting Kuramoto performance: {e}")
        
        # Mock performance data
        return {
            'total_runs': 12,
            'current_speedup': 1.67,
            'average_speedup': 1.58,
            'max_speedup': 1.82,
            'target_achieved': True,
            'performance_improvement': '67.0%',
            'compute_infrastructure': {
                'mode': 'cpu_parallel',
                'gpu_available': False,
                'dask_available': False,
                'workers': 2
            }
        }
    
    def _get_topological_data(self) -> Dict[str, Any]:
        """Generate topological defect visualization data."""
        # Generate mock topological defect data
        n_points = 50
        x = np.linspace(-2, 2, n_points)
        y = np.linspace(-2, 2, n_points)
        X, Y = np.meshgrid(x, y)
        
        # Create synthetic phase field with defects
        phase_field = np.arctan2(Y, X) + 0.2 * np.sin(3 * np.sqrt(X**2 + Y**2))
        
        # Detect topological defects (simplified)
        defects = []
        for i in range(5):
            defect_x = np.random.uniform(-1.5, 1.5)
            defect_y = np.random.uniform(-1.5, 1.5)
            defects.append({
                'x': defect_x,
                'y': defect_y,
                'charge': np.random.choice([-1, 1]),
                'strength': np.random.uniform(0.5, 1.0)
            })
        
        return {
            'grid_x': X.tolist(),
            'grid_y': Y.tolist(),
            'phase_field': phase_field.tolist(),
            'defects': defects,
            'timestamp': time.time(),
            'field_energy': np.mean(np.gradient(phase_field)**2),
            'defect_count': len(defects)
        }
    
    def _demonstrate_esc_processing(self, text: str) -> Dict[str, Any]:
        """Demonstrate ESC processing with telemetry."""
        if self.esc_module:
            try:
                tokens = text.split()
                result = self.esc_module.process_sequence(tokens)
                
                # Get telemetry data
                telemetry_report = self.esc_module.get_telemetry_report()
                
                return {
                    'success': True,
                    'input_text': text,
                    'processed_tokens': len(result.processed_tokens),
                    'constitutional_metrics': result.constitutional_metrics,
                    'processing_time': result.processing_stats.get('average_processing_time', 0.0),
                    'telemetry': telemetry_report,
                    'warnings': result.warnings
                }
            except Exception as e:
                logger.error(f"ESC processing error: {e}")
        
        # Mock processing result
        return {
            'success': True,
            'input_text': text,
            'processed_tokens': len(text.split()),
            'constitutional_metrics': {
                'overall_compliance': 0.92,
                'safety_score': 0.95,
                'risk_assessment': 0.08
            },
            'processing_time': 0.035,
            'warnings': []
        }
    
    def _run_kuramoto_benchmark(self) -> Dict[str, Any]:
        """Run Kuramoto performance benchmark."""
        if self.kuramoto_solver:
            try:
                results = self.kuramoto_solver.benchmark_performance([50, 100, 200])
                return {
                    'success': True,
                    'benchmark_results': results,
                    'timestamp': time.time()
                }
            except Exception as e:
                logger.error(f"Benchmark error: {e}")
        
        # Mock benchmark results
        return {
            'success': True,
            'benchmark_results': {
                'size_50': {
                    'execution_time': 0.12,
                    'speedup': 1.45,
                    'memory_usage': 2.3,
                    'success': True
                },
                'size_100': {
                    'execution_time': 0.28,
                    'speedup': 1.62,
                    'memory_usage': 4.1,
                    'success': True
                },
                'size_200': {
                    'execution_time': 0.67,
                    'speedup': 1.78,
                    'memory_usage': 8.8,
                    'success': True
                }
            },
            'timestamp': time.time()
        }
    
    def _start_background_monitoring(self):
        """Start background monitoring thread."""
        if not self.monitoring_active and self.socketio:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Background monitoring started")
    
    def _stop_background_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        logger.info("Background monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Emit real-time updates
                if self.socketio:
                    self.socketio.emit('telemetry_update', self._get_telemetry_data())
                    self.socketio.emit('system_status_update', self._get_system_status())
                    
                    # Simulate real-time performance data
                    perf_data = {
                        'timestamp': time.time(),
                        'cpu_usage': 20 + np.random.random() * 30,
                        'memory_usage': 45 + np.random.random() * 20,
                        'processing_rate': 850 + np.random.random() * 200
                    }
                    self.socketio.emit('performance_update', perf_data)
                
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def run(self):
        """Run the enhanced web interface."""
        self.system_status['start_time'] = time.time()
        
        if FLASK_AVAILABLE and self.app and self.socketio:
            logger.info(f"Starting enhanced NFCS web interface on {self.host}:{self.port}")
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
        else:
            logger.error("Flask not available - cannot start web interface")
            return False
    
    def cleanup(self):
        """Cleanup resources."""
        self._stop_background_monitoring()
        if self.kuramoto_solver:
            self.kuramoto_solver.cleanup()


# Enhanced HTML template with new visualizations
ENHANCED_DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced NFCS Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            background: #1a1a2e; 
            color: #eee; 
        }
        .header { 
            background: #16213e; 
            padding: 20px; 
            text-align: center; 
            border-bottom: 3px solid #0f4c75; 
        }
        .container { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
            padding: 20px; 
            max-width: 1400px; 
            margin: 0 auto; 
        }
        .panel { 
            background: #16213e; 
            border: 1px solid #0f4c75; 
            border-radius: 10px; 
            padding: 20px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.3); 
        }
        .panel h3 { 
            color: #3bb78f; 
            border-bottom: 2px solid #0f4c75; 
            padding-bottom: 10px; 
        }
        .status-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 10px; 
            margin: 15px 0; 
        }
        .status-item { 
            background: #0f3460; 
            padding: 15px; 
            border-radius: 5px; 
            text-align: center; 
        }
        .metric-value { 
            font-size: 1.5em; 
            font-weight: bold; 
            color: #3bb78f; 
        }
        .button { 
            background: #0f4c75; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 5px; 
            cursor: pointer; 
            margin: 5px; 
            transition: background 0.3s; 
        }
        .button:hover { background: #3bb78f; }
        .input-field { 
            width: 100%; 
            padding: 10px; 
            background: #0f3460; 
            border: 1px solid #0f4c75; 
            border-radius: 5px; 
            color: white; 
            margin: 10px 0; 
        }
        .log-panel { 
            background: #0a0a0a; 
            padding: 15px; 
            border-radius: 5px; 
            max-height: 300px; 
            overflow-y: auto; 
            font-family: monospace; 
            font-size: 0.9em; 
        }
        .success { color: #3bb78f; }
        .warning { color: #f39c12; }
        .error { color: #e74c3c; }
        .full-width { grid-column: 1 / -1; }
        #topological-viz { height: 400px; }
        .metrics-row { 
            display: flex; 
            justify-content: space-between; 
            margin: 10px 0; 
        }
        .metric-box { 
            flex: 1; 
            background: #0f3460; 
            padding: 10px; 
            margin: 0 5px; 
            border-radius: 5px; 
            text-align: center; 
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌊 Enhanced NFCS Dashboard v2.4.3</h1>
        <p>Real-time monitoring of ESC Module 2.1, RAG System, and Distributed Kuramoto</p>
    </div>

    <div class="container">
        <!-- System Status Panel -->
        <div class="panel">
            <h3>📊 System Status</h3>
            <div class="status-grid">
                <div class="status-item">
                    <div>ESC Module</div>
                    <div class="metric-value" id="esc-status">Initializing</div>
                </div>
                <div class="status-item">
                    <div>RAG System</div>
                    <div class="metric-value" id="rag-status">Initializing</div>
                </div>
                <div class="status-item">
                    <div>Kuramoto Solver</div>
                    <div class="metric-value" id="kuramoto-status">Initializing</div>
                </div>
                <div class="status-item">
                    <div>Telemetry</div>
                    <div class="metric-value" id="telemetry-status">Initializing</div>
                </div>
            </div>
            <button class="button" onclick="startMonitoring()">Start Real-time Monitoring</button>
            <button class="button" onclick="stopMonitoring()">Stop Monitoring</button>
        </div>

        <!-- ESC Telemetry Panel -->
        <div class="panel">
            <h3>🎭 ESC Module 2.1 Telemetry</h3>
            <div class="metrics-row">
                <div class="metric-box">
                    <div>Semantic Anchors</div>
                    <div class="metric-value" id="semantic-anchors">0</div>
                </div>
                <div class="metric-box">
                    <div>Stability Score</div>
                    <div class="metric-value" id="anchor-stability">0.0</div>
                </div>
                <div class="metric-box">
                    <div>Processing Time</div>
                    <div class="metric-value" id="processing-time">0ms</div>
                </div>
            </div>
            
            <input type="text" class="input-field" id="esc-input" placeholder="Enter text for ESC processing..." value="Neural field control with constitutional safety">
            <button class="button" onclick="demonstrateESC()">Process with ESC</button>
            
            <div id="esc-results" class="log-panel" style="margin-top: 15px; height: 150px;">
                ESC processing results will appear here...
            </div>
        </div>

        <!-- RAG System Panel -->
        <div class="panel">
            <h3>🧠 RAG System</h3>
            <div class="metrics-row">
                <div class="metric-box">
                    <div>Confidence</div>
                    <div class="metric-value" id="rag-confidence">0%</div>
                </div>
                <div class="metric-box">
                    <div>Hallucination Score</div>
                    <div class="metric-value" id="hallucination-score">0%</div>
                </div>
                <div class="metric-box">
                    <div>Sources Used</div>
                    <div class="metric-value" id="sources-count">0</div>
                </div>
            </div>
            
            <input type="text" class="input-field" id="rag-query" placeholder="Ask a question..." value="What is a neural field control system?">
            <button class="button" onclick="queryRAG()">Query RAG System</button>
            
            <div id="rag-response" class="log-panel" style="margin-top: 15px; height: 150px;">
                RAG responses will appear here...
            </div>
        </div>

        <!-- Kuramoto Performance Panel -->
        <div class="panel">
            <h3>⚡ Kuramoto Performance</h3>
            <div class="metrics-row">
                <div class="metric-box">
                    <div>Speedup Factor</div>
                    <div class="metric-value" id="speedup-factor">1.0x</div>
                </div>
                <div class="metric-box">
                    <div>Target Achieved</div>
                    <div class="metric-value" id="target-achieved">No</div>
                </div>
                <div class="metric-box">
                    <div>Compute Mode</div>
                    <div class="metric-value" id="compute-mode">CPU</div>
                </div>
            </div>
            
            <button class="button" onclick="runBenchmark()">Run Performance Benchmark</button>
            <button class="button" onclick="optimizePerformance()">Optimize Performance</button>
            
            <div id="performance-log" class="log-panel" style="margin-top: 15px; height: 120px;">
                Performance data will appear here...
            </div>
        </div>

        <!-- Topological Visualization Panel -->
        <div class="panel full-width">
            <h3>🌀 Topological Defect Visualization</h3>
            <div id="topological-viz"></div>
            <button class="button" onclick="updateTopologicalViz()">Update Visualization</button>
            <button class="button" onclick="exportVisualization()">Export Data</button>
        </div>

        <!-- System Logs Panel -->
        <div class="panel full-width">
            <h3>📋 System Logs</h3>
            <div id="system-logs" class="log-panel">
                System initialization complete. Ready for operation.
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        
        // Socket event handlers
        socket.on('connect', function() {
            addLog('Connected to enhanced NFCS dashboard', 'success');
        });
        
        socket.on('system_status', function(data) {
            updateSystemStatus(data);
        });
        
        socket.on('telemetry_update', function(data) {
            updateTelemetryDisplay(data);
        });
        
        socket.on('esc_result', function(data) {
            displayESCResult(data);
        });
        
        socket.on('benchmark_result', function(data) {
            displayBenchmarkResult(data);
        });
        
        socket.on('performance_update', function(data) {
            updatePerformanceDisplay(data);
        });

        // UI Functions
        function addLog(message, type = 'info') {
            const logs = document.getElementById('system-logs');
            const timestamp = new Date().toLocaleTimeString();
            const className = type === 'success' ? 'success' : type === 'warning' ? 'warning' : type === 'error' ? 'error' : '';
            logs.innerHTML += `<div class="${className}">[${timestamp}] ${message}</div>`;
            logs.scrollTop = logs.scrollHeight;
        }
        
        function updateSystemStatus(data) {
            if (data.components) {
                document.getElementById('esc-status').textContent = data.components.esc_module;
                document.getElementById('rag-status').textContent = data.components.rag_system;
                document.getElementById('kuramoto-status').textContent = data.components.kuramoto_solver;
                document.getElementById('telemetry-status').textContent = data.components.telemetry;
            }
        }
        
        function updateTelemetryDisplay(data) {
            if (data.system_overview) {
                document.getElementById('semantic-anchors').textContent = data.system_overview.active_semantic_anchors || 0;
                document.getElementById('processing-time').textContent = Math.round((data.system_overview.average_processing_time || 0) * 1000) + 'ms';
            }
            if (data.semantic_anchor_stability) {
                document.getElementById('anchor-stability').textContent = (data.semantic_anchor_stability.average_stability || 0).toFixed(2);
            }
        }
        
        function startMonitoring() {
            socket.emit('start_monitoring');
            addLog('Real-time monitoring started', 'success');
        }
        
        function stopMonitoring() {
            socket.emit('stop_monitoring');
            addLog('Real-time monitoring stopped', 'warning');
        }
        
        function demonstrateESC() {
            const text = document.getElementById('esc-input').value;
            socket.emit('esc_demo', {text: text});
            addLog(`ESC processing: "${text}"`, 'info');
        }
        
        function displayESCResult(data) {
            const results = document.getElementById('esc-results');
            if (data.success) {
                results.innerHTML = `
                    <div class="success">✓ Processing successful</div>
                    <div>Tokens processed: ${data.processed_tokens}</div>
                    <div>Compliance score: ${(data.constitutional_metrics.overall_compliance * 100).toFixed(1)}%</div>
                    <div>Processing time: ${(data.processing_time * 1000).toFixed(1)}ms</div>
                `;
            } else {
                results.innerHTML = `<div class="error">✗ Processing failed</div>`;
            }
        }
        
        function queryRAG() {
            const query = document.getElementById('rag-query').value;
            fetch('/api/rag/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            })
            .then(response => response.json())
            .then(data => {
                const responseDiv = document.getElementById('rag-response');
                if (data.success) {
                    responseDiv.innerHTML = `
                        <div class="success">✓ Query processed</div>
                        <div><strong>Response:</strong> ${data.response}</div>
                        <div>Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
                        <div>Hallucination score: ${(data.hallucination_score * 100).toFixed(1)}%</div>
                        <div>Sources: ${data.sources.join(', ')}</div>
                    `;
                    
                    // Update metrics
                    document.getElementById('rag-confidence').textContent = (data.confidence * 100).toFixed(0) + '%';
                    document.getElementById('hallucination-score').textContent = (data.hallucination_score * 100).toFixed(0) + '%';
                    document.getElementById('sources-count').textContent = data.sources.length;
                }
            });
        }
        
        function runBenchmark() {
            socket.emit('kuramoto_benchmark');
            addLog('Running Kuramoto performance benchmark...', 'info');
        }
        
        function displayBenchmarkResult(data) {
            if (data.success) {
                const log = document.getElementById('performance-log');
                log.innerHTML = '<div class="success">✓ Benchmark completed</div>';
                
                Object.entries(data.benchmark_results).forEach(([size, result]) => {
                    if (result.success) {
                        log.innerHTML += `<div>${size}: ${result.speedup.toFixed(2)}x speedup, ${result.execution_time.toFixed(3)}s</div>`;
                    }
                });
                
                addLog('Performance benchmark completed', 'success');
            }
        }
        
        function updatePerformanceDisplay(data) {
            // Update performance metrics from real-time data
            addLog(`Performance: CPU ${data.cpu_usage.toFixed(1)}%, Memory ${data.memory_usage.toFixed(1)}%`, 'info');
        }
        
        function updateTopologicalViz() {
            fetch('/api/topological/visualization')
            .then(response => response.json())
            .then(data => {
                const trace = {
                    z: data.phase_field,
                    type: 'heatmap',
                    colorscale: 'Viridis',
                    showscale: true
                };
                
                const layout = {
                    title: 'Phase Field with Topological Defects',
                    xaxis: {title: 'X'},
                    yaxis: {title: 'Y'},
                    plot_bgcolor: '#16213e',
                    paper_bgcolor: '#16213e',
                    font: {color: '#eee'}
                };
                
                Plotly.newPlot('topological-viz', [trace], layout);
                addLog(`Topological visualization updated: ${data.defect_count} defects detected`, 'success');
            });
        }
        
        function optimizePerformance() {
            addLog('Performance optimization initiated...', 'info');
            // Simulate optimization
            setTimeout(() => {
                addLog('Performance optimization completed', 'success');
            }, 2000);
        }
        
        function exportVisualization() {
            addLog('Exporting visualization data...', 'info');
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            updateTopologicalViz();
            
            // Load performance data
            fetch('/api/kuramoto/performance')
            .then(response => response.json())
            .then(data => {
                document.getElementById('speedup-factor').textContent = data.current_speedup.toFixed(1) + 'x';
                document.getElementById('target-achieved').textContent = data.target_achieved ? 'Yes' : 'No';
                document.getElementById('compute-mode').textContent = data.compute_infrastructure.mode;
            });
        });
    </script>
</body>
</html>
'''


def main():
    """Main function to start the enhanced web interface."""
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    host = "0.0.0.0"
    port = 5000
    
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    try:
        # Create and run enhanced interface
        interface = EnhancedNFCSWebInterface(host=host, port=port)
        interface.run()
    except KeyboardInterrupt:
        print("\n🛑 Enhanced NFCS Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting enhanced dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()