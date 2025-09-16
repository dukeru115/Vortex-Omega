-- Initial Vortex-Omega Database Schema
-- Neural Field Control System v2.5.0

-- Create extension for UUID generation (PostgreSQL only)
-- This will be ignored by SQLite
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schema for NFCS components
CREATE SCHEMA IF NOT EXISTS nfcs;
CREATE SCHEMA IF NOT EXISTS constitutional;
CREATE SCHEMA IF NOT EXISTS cognitive;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- System configuration table
CREATE TABLE IF NOT EXISTS nfcs.system_config (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    value TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Constitutional monitoring events
CREATE TABLE IF NOT EXISTS constitutional.monitoring_events (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    description TEXT,
    metadata JSONB,
    ha_value DECIMAL(10, 6),
    threshold_value DECIMAL(10, 6),
    action_taken TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    INDEX idx_constitutional_events_type (event_type),
    INDEX idx_constitutional_events_severity (severity),
    INDEX idx_constitutional_events_created (created_at)
);

-- ESC-Kuramoto oscillator states
CREATE TABLE IF NOT EXISTS nfcs.kuramoto_states (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    oscillator_id INTEGER NOT NULL,
    phase DECIMAL(10, 6) NOT NULL,
    frequency DECIMAL(10, 6) NOT NULL,
    coupling_strength DECIMAL(10, 6),
    sync_parameter DECIMAL(10, 6),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(255),
    INDEX idx_kuramoto_oscillator (oscillator_id),
    INDEX idx_kuramoto_timestamp (timestamp),
    INDEX idx_kuramoto_session (session_id)
);

-- Cognitive module interactions
CREATE TABLE IF NOT EXISTS cognitive.module_interactions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    source_module VARCHAR(100) NOT NULL,
    target_module VARCHAR(100) NOT NULL,
    interaction_type VARCHAR(100) NOT NULL,
    input_data JSONB,
    output_data JSONB,
    processing_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_cognitive_modules (source_module, target_module),
    INDEX idx_cognitive_type (interaction_type),
    INDEX idx_cognitive_created (created_at)
);

-- Symbolic AI processing logs
CREATE TABLE IF NOT EXISTS cognitive.symbolic_processing (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    input_text TEXT NOT NULL,
    symbolic_representation JSONB,
    field_representation JSONB,
    verification_results JSONB,
    processing_stage VARCHAR(50) NOT NULL,
    success BOOLEAN DEFAULT TRUE,
    error_details TEXT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbolic_stage (processing_stage),
    INDEX idx_symbolic_success (success),
    INDEX idx_symbolic_created (created_at)
);

-- Memory system tables
CREATE TABLE IF NOT EXISTS cognitive.memory_entries (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    memory_type VARCHAR(50) NOT NULL CHECK (memory_type IN ('working', 'episodic', 'semantic', 'procedural')),
    content JSONB NOT NULL,
    strength DECIMAL(5, 4) DEFAULT 1.0,
    decay_rate DECIMAL(8, 6) DEFAULT 0.1,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    tags TEXT[],
    INDEX idx_memory_type (memory_type),
    INDEX idx_memory_strength (strength),
    INDEX idx_memory_created (created_at),
    INDEX idx_memory_expires (expires_at),
    INDEX idx_memory_tags USING GIN (tags)
);

-- Performance monitoring
CREATE TABLE IF NOT EXISTS monitoring.performance_metrics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 6) NOT NULL,
    metric_unit VARCHAR(20),
    component VARCHAR(100) NOT NULL,
    instance_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    INDEX idx_perf_name (metric_name),
    INDEX idx_perf_component (component),
    INDEX idx_perf_timestamp (timestamp)
);

-- System health checks
CREATE TABLE IF NOT EXISTS monitoring.health_checks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    check_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('HEALTHY', 'WARNING', 'CRITICAL', 'UNKNOWN')),
    response_time_ms INTEGER,
    error_message TEXT,
    details JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_health_name (check_name),
    INDEX idx_health_status (status),
    INDEX idx_health_timestamp (timestamp)
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS monitoring.api_requests (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    user_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    request_size INTEGER,
    response_size INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_api_endpoint (endpoint),
    INDEX idx_api_status (status_code),
    INDEX idx_api_timestamp (timestamp),
    INDEX idx_api_user (user_id)
);

-- Insert default system configuration
INSERT INTO nfcs.system_config (key, value, description) VALUES
    ('version', '2.5.0', 'NFCS System Version'),
    ('operational_mode', 'supervised', 'Current operational mode'),
    ('safety_level', '0.8', 'Default safety threshold'),
    ('kuramoto_oscillators', '64', 'Number of Kuramoto oscillators'),
    ('constitutional_monitoring', 'true', 'Enable constitutional monitoring'),
    ('max_concurrent_operations', '100', 'Maximum concurrent operations'),
    ('log_level', 'INFO', 'Default logging level'),
    ('api_rate_limit', '1000', 'API requests per hour limit'),
    ('session_timeout', '3600', 'Session timeout in seconds'),
    ('backup_enabled', 'true', 'Enable automated backups')
ON CONFLICT (key) DO UPDATE SET 
    value = EXCLUDED.value,
    updated_at = CURRENT_TIMESTAMP;