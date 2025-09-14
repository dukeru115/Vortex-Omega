#!/usr/bin/env python3
"""
NFCS FastAPI Server Launcher
============================

Production launcher for NFCS FastAPI server with proper configuration,
logging setup, and process management.

Author: Team Ω (Omega)
Date: September 13, 2025
Version: 2.4.3
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import uvicorn

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the application"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="NFCS FastAPI Server v2.4.3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py                          # Run with defaults
  python launcher.py --host 0.0.0.0 --port 8080  # Custom host/port
  python launcher.py --workers 4 --log-level DEBUG  # Production mode
  python launcher.py --reload                 # Development mode
        """
    )
    
    parser.add_argument(
        "--host", 
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--access-log",
        action="store_true",
        default=True,
        help="Enable access logging"
    )
    parser.add_argument(
        "--ssl-keyfile",
        help="SSL key file path (for HTTPS)"
    )
    parser.add_argument(
        "--ssl-certfile", 
        help="SSL certificate file path (for HTTPS)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting NFCS FastAPI Server v2.4.3")
    logger.info(f"   Host: {args.host}")
    logger.info(f"   Port: {args.port}")
    logger.info(f"   Workers: {args.workers}")
    logger.info(f"   Log Level: {args.log_level}")
    logger.info(f"   Reload: {args.reload}")
    
    # SSL configuration
    ssl_config = {}
    if args.ssl_keyfile and args.ssl_certfile:
        ssl_config = {
            "ssl_keyfile": args.ssl_keyfile,
            "ssl_certfile": args.ssl_certfile
        }
        logger.info("   SSL: Enabled")
    
    # Server configuration
    config = {
        "app": "api.server:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level.lower(),
        "access_log": args.access_log,
        "server_header": False,
        "date_header": False,
        **ssl_config
    }
    
    # Development vs Production configuration
    if args.reload:
        config.update({
            "reload": True,
            "reload_dirs": [str(src_dir)],
        })
        logger.info("   Mode: Development (auto-reload enabled)")
    else:
        config.update({
            "workers": args.workers if args.workers > 1 else None,
        })
        logger.info(f"   Mode: Production")
    
    try:
        # Start the server
        uvicorn.run(**config)
        
    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)
    finally:
        logger.info("✅ Server shutdown complete")


if __name__ == "__main__":
    main()