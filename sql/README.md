# 🗄️ NFCS Database Schema & SQL Scripts

## Overview
Database schema definitions, migration scripts, and SQL utilities for the Neural Field Control System (NFCS) v2.4.3.

## 🏗️ Database Architecture

### Current Implementation
The MVP system currently operates with:
- **In-Memory Storage**: Real-time data processing
- **File-based Persistence**: Configuration and state storage
- **SQLite Support**: Lightweight database operations

### Production Database Design
```
🗄️ PostgreSQL Database
├── 📊 NFCS Core Tables
├── 🏛️ Constitutional Monitoring
├── 🔄 Kuramoto Synchronization
├── 🧠 Cognitive Modules Data  
├── 📈 Validation & Metrics
└── 👥 User Management
```

## 📁 Directory Structure

### Schema Files
- **schema.sql**: Complete database schema definition
- **initial_data.sql**: Default configuration and seed data
- **indexes.sql**: Performance optimization indexes
- **constraints.sql**: Data integrity constraints

### Migration Scripts
- **migrations/**: Version-controlled schema changes
- **rollbacks/**: Rollback scripts for each migration
- **seeds/**: Development and testing data

### Utilities
- **backup.sql**: Database backup procedures  
- **maintenance.sql**: Routine maintenance tasks
- **monitoring.sql**: Database health queries

## 🗃️ Core Tables

### 🧠 NFCS System Tables

#### `nfcs_fields` - Neural Field States
```sql
CREATE TABLE nfcs_fields (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    field_data JSONB NOT NULL,
    resolution INTEGER NOT NULL,
    coherence_level DECIMAL(5,4),
    energy_state DECIMAL(10,6),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    INDEX idx_nfcs_fields_timestamp (timestamp),
    INDEX idx_nfcs_fields_coherence (coherence_level)
);
```

#### `system_config` - Configuration Management
```sql
CREATE TABLE system_config (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    config_type VARCHAR(50) NOT NULL,
    environment VARCHAR(50) NOT NULL DEFAULT 'production',
    is_active BOOLEAN NOT NULL DEFAULT true,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    INDEX idx_config_key_env (config_key, environment)
);
```

### 🏛️ Constitutional Monitoring Tables

#### `constitutional_policies` - Policy Definitions
```sql
CREATE TABLE constitutional_policies (
    id SERIAL PRIMARY KEY,
    policy_name VARCHAR(255) NOT NULL,
    policy_type VARCHAR(50) NOT NULL, -- fundamental, operational, safety, ethical
    policy_content JSONB NOT NULL,
    priority_level INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_priority_level CHECK (priority_level BETWEEN 1 AND 10)
);
```

#### `constitutional_violations` - Violation Log
```sql
CREATE TABLE constitutional_violations (
    id BIGSERIAL PRIMARY KEY,
    violation_type VARCHAR(50) NOT NULL, -- minor, moderate, major, critical
    policy_id INTEGER REFERENCES constitutional_policies(id),
    violation_data JSONB NOT NULL,
    severity_score DECIMAL(3,2) NOT NULL,
    ha_value DECIMAL(5,4),
    resolution_status VARCHAR(50) DEFAULT 'pending',
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    INDEX idx_violations_type (violation_type),
    INDEX idx_violations_severity (severity_score),
    INDEX idx_violations_detected (detected_at)
);
```

### 🔄 Kuramoto Synchronization Tables

#### `kuramoto_oscillators` - Oscillator Network State
```sql
CREATE TABLE kuramoto_oscillators (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    oscillator_states JSONB NOT NULL, -- Array of 64 oscillator states
    sync_parameter DECIMAL(5,4),
    phase_coherence DECIMAL(5,4),
    coupling_strength DECIMAL(5,4),
    semantic_embedding VECTOR(768), -- Requires pgvector extension
    prediction_horizons JSONB, -- 30s, 3min, 10min predictions
    INDEX idx_kuramoto_timestamp (timestamp),
    INDEX idx_kuramoto_sync (sync_parameter)
);
```

#### `synchronization_events` - Sync State Changes
```sql
CREATE TABLE synchronization_events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL, -- sync_achieved, desync_detected, emergency_stop
    sync_level_before DECIMAL(5,4),
    sync_level_after DECIMAL(5,4),
    event_data JSONB,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    INDEX idx_sync_events_type (event_type),
    INDEX idx_sync_events_time (occurred_at)
);
```

### 🧠 Cognitive Modules Tables

#### `cognitive_sessions` - Module Sessions
```sql
CREATE TABLE cognitive_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    module_type VARCHAR(50) NOT NULL, -- constitution, symbolic_ai, memory, reflection, freedom
    session_data JSONB NOT NULL,
    performance_metrics JSONB,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    INDEX idx_cognitive_module_type (module_type),
    INDEX idx_cognitive_started (started_at)
);
```

#### `memory_experiences` - Long-term Memory Storage
```sql
CREATE TABLE memory_experiences (
    id BIGSERIAL PRIMARY KEY,
    experience_uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    experience_type VARCHAR(50) NOT NULL,
    context_data JSONB NOT NULL,
    emotional_weight DECIMAL(3,2),
    consolidation_level DECIMAL(3,2) DEFAULT 0.0,
    retrieval_count INTEGER DEFAULT 0,
    stored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMPTZ,
    INDEX idx_memory_type (experience_type),
    INDEX idx_memory_consolidation (consolidation_level),
    INDEX idx_memory_stored (stored_at)
);
```

### 📊 Validation & Metrics Tables

#### `validation_runs` - Empirical Validation Results
```sql
CREATE TABLE validation_runs (
    id BIGSERIAL PRIMARY KEY,
    run_uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    validation_type VARCHAR(50) NOT NULL, -- theoretical, performance, statistical
    test_parameters JSONB NOT NULL,
    results_data JSONB NOT NULL,
    overall_score DECIMAL(5,4),
    passed_tests INTEGER DEFAULT 0,
    total_tests INTEGER NOT NULL,
    duration_seconds INTEGER,
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    INDEX idx_validation_type (validation_type),
    INDEX idx_validation_score (overall_score),
    INDEX idx_validation_executed (executed_at)
);
```

#### `system_metrics` - Performance Metrics
```sql
CREATE TABLE system_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(20),
    component VARCHAR(50) NOT NULL,
    tags JSONB,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    INDEX idx_metrics_name_component (metric_name, component),
    INDEX idx_metrics_recorded (recorded_at)
);
```

### 👥 User Management Tables

#### `users` - System Users
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user', -- admin, operator, user, readonly
    is_active BOOLEAN NOT NULL DEFAULT true,
    last_login TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    INDEX idx_users_username (username),
    INDEX idx_users_email (email)
);
```

#### `user_sessions` - Authentication Sessions
```sql
CREATE TABLE user_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    jwt_token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    INDEX idx_sessions_uuid (session_uuid),
    INDEX idx_sessions_user (user_id),
    INDEX idx_sessions_expires (expires_at)
);
```

## 🚀 Setup Scripts

### Database Initialization
```sql
-- Create database and extensions
CREATE DATABASE nfcs_production;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector"; -- For semantic embeddings

-- Create schemas
CREATE SCHEMA nfcs_core;
CREATE SCHEMA constitutional;
CREATE SCHEMA cognitive;
CREATE SCHEMA validation;
CREATE SCHEMA user_mgmt;
```

### Development Data Seeds
```sql
-- Insert default configuration
INSERT INTO system_config (config_key, config_value, config_type) VALUES
('nfcs.field_resolution', '128', 'integer'),
('nfcs.ha_threshold', '0.85', 'float'),
('kuramoto.n_oscillators', '64', 'integer'),
('cognitive.modules_enabled', '["constitution", "symbolic_ai", "memory", "reflection", "freedom"]', 'array');

-- Create default admin user
INSERT INTO users (username, email, password_hash, role) VALUES
('admin', 'admin@nfcs.system', '$2b$12$...', 'admin');
```

## 📊 Performance Optimizations

### Indexing Strategy
```sql
-- Time-series optimization for high-frequency data
CREATE INDEX CONCURRENTLY idx_nfcs_fields_time_brin 
ON nfcs_fields USING BRIN (timestamp);

-- Composite indexes for common queries
CREATE INDEX CONCURRENTLY idx_violations_type_severity_time 
ON constitutional_violations (violation_type, severity_score, detected_at);

-- Partial indexes for active records
CREATE INDEX CONCURRENTLY idx_active_policies 
ON constitutional_policies (policy_type, priority_level) 
WHERE is_active = true;
```

### Partitioning Tables
```sql
-- Partition large tables by time
CREATE TABLE nfcs_fields_y2025m09 PARTITION OF nfcs_fields
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

-- Auto-partition creation function
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name text, start_date date)
RETURNS void AS $$
DECLARE
    partition_name text;
    end_date date;
BEGIN
    partition_name := table_name || '_y' || EXTRACT(year FROM start_date) || 'm' || LPAD(EXTRACT(month FROM start_date)::text, 2, '0');
    end_date := start_date + interval '1 month';
    
    EXECUTE format('CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;
```

## 🔧 Database Utilities

### Backup & Maintenance
```bash
#!/bin/bash
# Database backup script
pg_dump -h localhost -U nfcs_user -d nfcs_production --schema-only > schema_backup.sql
pg_dump -h localhost -U nfcs_user -d nfcs_production --data-only > data_backup.sql

# Compressed backup with timestamp
pg_dump -h localhost -U nfcs_user -d nfcs_production -Fc > "nfcs_backup_$(date +%Y%m%d_%H%M%S).dump"
```

### Health Monitoring Queries
```sql
-- Database size monitoring
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname IN ('nfcs_core', 'constitutional', 'cognitive')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Performance analysis
SELECT 
    schemaname,
    tablename,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_live_tup,
    n_dead_tup
FROM pg_stat_user_tables
WHERE schemaname IN ('nfcs_core', 'constitutional', 'cognitive');
```

## 🔗 Integration Points

### MVP Integration
The current MVP system can be enhanced with database support:
- **Configuration Storage**: System settings in database
- **Session Persistence**: User session management
- **Metrics History**: Long-term performance tracking

### Production Integration
Full database integration provides:
- **Scalable Storage**: Handle large-scale NFCS data
- **ACID Compliance**: Data consistency and reliability
- **Advanced Analytics**: Complex queries and reporting
- **Multi-user Support**: Concurrent access management

## 📝 Migration Management

### Version Control
```bash
# Apply migrations
psql -h localhost -U nfcs_user -d nfcs_production -f migrations/001_initial_schema.sql
psql -h localhost -U nfcs_user -d nfcs_production -f migrations/002_add_cognitive_tables.sql

# Rollback if needed
psql -h localhost -U nfcs_user -d nfcs_production -f rollbacks/002_remove_cognitive_tables.sql
```

### Migration Tracking
```sql
CREATE TABLE schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum VARCHAR(64)
);
```

## 🔗 Related Documentation
- [Configuration Management](../src/config/README.md)
- [API Layer](../src/api/README.md)  
- [System Architecture](../docs/README.md)

---
*Part of Vortex-Omega Neural Field Control System v2.4.3*