-- Vortex-Omega NFCS Database Initialization
-- Neural Field Control System database schema

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- NFCS Field State table
CREATE TABLE IF NOT EXISTS nfcs_field_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    coherence_measure DOUBLE PRECISION NOT NULL,
    hallucination_number DOUBLE PRECISION NOT NULL,
    defect_density DOUBLE PRECISION NOT NULL,
    field_energy DOUBLE PRECISION NOT NULL,
    phase_data JSONB,
    amplitude_data JSONB,
    metadata JSONB,
    
    -- Indexes for time-series queries
    INDEX idx_nfcs_field_timestamp ON nfcs_field_states(timestamp),
    INDEX idx_nfcs_field_ha ON nfcs_field_states(hallucination_number),
    INDEX idx_nfcs_field_coherence ON nfcs_field_states(coherence_measure)
);

-- ESC (Echo-Semantic Converter) processing log
CREATE TABLE IF NOT EXISTS esc_processing_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    input_tokens TEXT[],
    semantic_frequencies JSONB,
    echo_patterns JSONB,
    output_signal JSONB,
    processing_time_ms INTEGER,
    
    INDEX idx_esc_timestamp ON esc_processing_log(timestamp),
    INDEX idx_esc_processing_time ON esc_processing_log(processing_time_ms)
);

-- Kuramoto Module synchronization data
CREATE TABLE IF NOT EXISTS kuramoto_sync_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    module_name VARCHAR(100) NOT NULL,
    phase DOUBLE PRECISION NOT NULL,
    frequency DOUBLE PRECISION NOT NULL,
    coupling_strength DOUBLE PRECISION NOT NULL,
    sync_parameter DOUBLE PRECISION NOT NULL,
    
    INDEX idx_kuramoto_timestamp ON kuramoto_sync_states(timestamp),
    INDEX idx_kuramoto_module ON kuramoto_sync_states(module_name),
    INDEX idx_kuramoto_sync ON kuramoto_sync_states(sync_parameter)
);

-- Constitutional Module decisions log
CREATE TABLE IF NOT EXISTS constitutional_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    decision_type VARCHAR(50) NOT NULL, -- ACCEPT, REJECT, EMERGENCY_MODE
    integrity_score DOUBLE PRECISION NOT NULL,
    reason TEXT,
    context JSONB,
    
    INDEX idx_constitutional_timestamp ON constitutional_decisions(timestamp),
    INDEX idx_constitutional_type ON constitutional_decisions(decision_type)
);

-- System performance metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    labels JSONB,
    
    INDEX idx_metrics_timestamp ON system_metrics(timestamp),
    INDEX idx_metrics_name ON system_metrics(metric_name)
);

-- Create hypertables for time-series data (if TimescaleDB is available)
-- SELECT create_hypertable('nfcs_field_states', 'timestamp', if_not_exists => TRUE);
-- SELECT create_hypertable('esc_processing_log', 'timestamp', if_not_exists => TRUE);
-- SELECT create_hypertable('kuramoto_sync_states', 'timestamp', if_not_exists => TRUE);

-- Insert initial test data
INSERT INTO nfcs_field_states (coherence_measure, hallucination_number, defect_density, field_energy, metadata)
VALUES 
    (0.85, 0.3, 0.1, 1000.0, '{"init": true, "version": "2.4.3"}'),
    (0.92, 0.15, 0.05, 1200.0, '{"init": true, "version": "2.4.3"}');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO vortex;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO vortex;