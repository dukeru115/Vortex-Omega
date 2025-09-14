// PM2 Ecosystem Configuration for NFCS MVP
// Neural Field Control System v2.4.3 Production Deployment

module.exports = {
  apps: [{
    name: 'nfcs-mvp-web',
    script: 'mvp_web_interface.py',
    interpreter: 'python3',
    cwd: '/home/user/webapp/Vortex-Omega',
    
    // Instance configuration
    instances: 1,
    exec_mode: 'fork',
    
    // Environment
    env: {
      NODE_ENV: 'production',
      FLASK_ENV: 'production',
      PYTHONPATH: '/home/user/webapp/Vortex-Omega/src',
      PORT: 5000
    },
    
    // Restart policy
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    
    // Logging
    log_file: './logs/nfcs-mvp.log',
    out_file: './logs/nfcs-mvp-out.log',
    error_file: './logs/nfcs-mvp-error.log',
    time: true,
    
    // Advanced options
    kill_timeout: 5000,
    listen_timeout: 3000,
    
    // Health monitoring
    min_uptime: '10s',
    max_restarts: 10
  }]
};