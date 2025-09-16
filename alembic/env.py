"""
Database migration environment for Vortex-Omega NFCS
Handles both SQLite (development) and PostgreSQL (production)
"""

import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import models
try:
    from src.modules.cognitive.constitution.constitution_core import Base as ConstitutionBase
    from src.modules.cognitive.memory.memory_core import Base as MemoryBase
    from src.core.base_models import Base as CoreBase
    
    # Combined metadata from all models
    target_metadata = [ConstitutionBase.metadata, MemoryBase.metadata, CoreBase.metadata]
except ImportError:
    # Fallback if models not available
    from sqlalchemy import MetaData
    target_metadata = MetaData()

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

def get_database_url():
    """Get database URL from environment or config"""
    # Try environment variable first
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        return db_url
    
    # Try .env files
    env_file = None
    if os.getenv('NFCS_ENV') == 'production':
        env_file = '.env.production'
    elif os.getenv('NFCS_ENV') == 'development':
        env_file = '.env.development'
    
    if env_file and os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.startswith('DATABASE_URL='):
                    return line.split('=', 1)[1].strip()
    
    # Fallback to config
    return config.get_main_option("sqlalchemy.url")

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    
    # Override config with environment database URL
    configuration = config.get_section(config.config_ini_section)
    configuration['sqlalchemy.url'] = get_database_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()