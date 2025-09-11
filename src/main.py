#!/usr/bin/env python3
"""
Neural Field Control System (NFCS) - Main Entry Point
====================================================

This is the main entry point for the Neural Field Control System.
It initializes and runs the complete NFCS with all orchestrator components,
cognitive modules, mathematical frameworks, and constitutional safety systems.

Usage:
    python src/main.py [options]
    
Options:
    --config PATH       Configuration file path
    --mode MODE         Operational mode (autonomous, supervised, manual)
    --log-level LEVEL   Logging level (DEBUG, INFO, WARNING, ERROR)
    --test              Run in test mode with reduced complexity
    --daemon            Run as background daemon
    --help              Show this help message

Example:
    python src/main.py --mode autonomous --log-level INFO
    python src/main.py --config nfcs_config.yaml --daemon
    python src/main.py --test
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.nfcs_orchestrator import (
    NFCSOrchestrator, 
    create_orchestrator, 
    create_default_config,
    OrchestrationConfig,
    OperationalMode
)


class NFCSMain:
    """Main NFCS application class"""
    
    def __init__(self, args):
        self.args = args
        self.orchestrator: NFCSOrchestrator = None
        self.logger = None
        self.running = False
        
    def setup_logging(self):
        """Configure logging based on arguments"""
        log_level = getattr(logging, self.args.log_level.upper())
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        
        # File handler if not in test mode
        if not self.args.test:
            log_file = Path("logs") / f"nfcs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_file.parent.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        self.logger = logging.getLogger("NFCSMain")
        self.logger.info(f"Logging configured - Level: {self.args.log_level}")
        
    def create_configuration(self) -> OrchestrationConfig:
        """Create orchestrator configuration"""
        config = create_default_config()
        
        # Apply command line overrides
        if self.args.mode:
            if self.args.mode == "autonomous":
                config.enable_autonomous_mode = True
            elif self.args.mode == "supervised":
                config.enable_autonomous_mode = False
        
        # Test mode adjustments
        if self.args.test:
            config.max_concurrent_processes = 3
            config.update_frequency_hz = 5.0
            config.performance_history_size = 100
            config.max_error_threshold = 10
            
        # Daemon mode adjustments  
        if self.args.daemon:
            config.log_level = "WARNING"  # Reduce daemon logging
            
        self.logger.info(f"Configuration created - Mode: {self.args.mode}, Test: {self.args.test}")
        return config
    
    async def initialize_nfcs(self) -> bool:
        """Initialize the complete NFCS system"""
        try:
            self.logger.info("🚀 Initializing Neural Field Control System...")
            
            # Create configuration
            config = self.create_configuration()
            
            # Load additional config file if specified
            if self.args.config and Path(self.args.config).exists():
                self.logger.info(f"Loading configuration from {self.args.config}")
                # Would load config file here in production
            
            # Create and initialize orchestrator
            self.orchestrator = await create_orchestrator(config)
            
            # Set operational mode if specified
            if self.args.mode:
                mode_mapping = {
                    "autonomous": OperationalMode.AUTONOMOUS,
                    "supervised": OperationalMode.SUPERVISED,
                    "manual": OperationalMode.MANUAL
                }
                
                if self.args.mode in mode_mapping:
                    await self.orchestrator.set_operational_mode(mode_mapping[self.args.mode])
            
            self.logger.info("✅ NFCS initialization completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ NFCS initialization failed: {e}")
            return False
    
    async def run_main_loop(self):
        """Main application loop"""
        self.running = True
        self.logger.info("🔄 Starting NFCS main loop...")
        
        try:
            # Print system status
            await self.print_system_status()
            
            # Main execution loop
            while self.running:
                # Get system status
                status = self.orchestrator.get_system_status()
                
                # Check for critical conditions
                if status['status'] in ['error', 'critical', 'emergency']:
                    self.logger.warning(f"⚠️  System in {status['status']} status")
                
                # In test mode, run for limited time
                if self.args.test:
                    await asyncio.sleep(5.0)
                    self.logger.info("🧪 Test mode - stopping after 5 seconds")
                    break
                
                # Normal operation - wait before next check
                await asyncio.sleep(10.0)
                
        except asyncio.CancelledError:
            self.logger.info("🛑 Main loop cancelled")
        except Exception as e:
            self.logger.error(f"💥 Error in main loop: {e}")
        finally:
            self.running = False
    
    async def print_system_status(self):
        """Print comprehensive system status"""
        try:
            status = self.orchestrator.get_system_status()
            
            self.logger.info("📊 NFCS System Status:")
            self.logger.info(f"   Status: {status['status']}")
            self.logger.info(f"   Mode: {status['mode']}")
            self.logger.info(f"   Uptime: {status['uptime_seconds']:.1f} seconds")
            self.logger.info(f"   Active Modules: {len(status['active_modules'])}")
            
            if status['active_modules']:
                self.logger.info(f"   Modules: {', '.join(status['active_modules'])}")
            
            # Performance metrics
            perf = status.get('performance_metrics', {})
            if perf:
                self.logger.info(f"   Operations: {perf.get('total_operations', 0)}")
                self.logger.info(f"   Success Rate: {perf.get('success_rate', 0):.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error printing system status: {e}")
    
    async def shutdown_nfcs(self):
        """Gracefully shutdown NFCS"""
        try:
            self.logger.info("🛑 Shutting down NFCS...")
            
            if self.orchestrator:
                # Print final status
                await self.print_system_status()
                
                # Shutdown orchestrator
                shutdown_success = await self.orchestrator.shutdown(timeout=30.0)
                
                if shutdown_success:
                    self.logger.info("✅ NFCS shutdown completed successfully")
                else:
                    self.logger.warning("⚠️  NFCS shutdown completed with warnings")
            
        except Exception as e:
            self.logger.error(f"❌ Error during NFCS shutdown: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"🔔 Received signal {signum}")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Run the complete NFCS application"""
        try:
            # Setup
            self.setup_logging()
            self.setup_signal_handlers()
            
            # Banner
            self.print_banner()
            
            # Initialize
            if not await self.initialize_nfcs():
                return 1
            
            # Run main loop
            await self.run_main_loop()
            
            # Shutdown
            await self.shutdown_nfcs()
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("🔔 Interrupted by user")
            await self.shutdown_nfcs()
            return 1
        except Exception as e:
            if self.logger:
                self.logger.error(f"💥 Unexpected error: {e}")
            else:
                print(f"💥 Unexpected error: {e}")
            return 1
    
    def print_banner(self):
        """Print startup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Neural Field Control System (NFCS)                      ║
║                            Production Release v1.0                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🧠 Cognitive Architecture with Constitutional Safety Framework              ║
║  🔬 Mathematical Neural Field Models with Kuramoto Synchronization          ║
║  🛡️  Constitutional Policies and Compliance Monitoring                      ║
║  🚨 Emergency Protocols and Safety Constraints                              ║
║  ⚡ Real-time Orchestration and Performance Monitoring                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """.strip()
        
        print(banner)
        
        if self.args.test:
            print("🧪 Running in TEST MODE")
        if self.args.daemon:
            print("🔄 Running in DAEMON MODE")
        
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Neural Field Control System (NFCS) - Advanced Cognitive Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode autonomous --log-level INFO
  %(prog)s --config nfcs_config.yaml --daemon
  %(prog)s --test
  %(prog)s --help
        """
    )
    
    parser.add_argument(
        "--config",
        help="Configuration file path (YAML or JSON)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["autonomous", "supervised", "manual"],
        default="supervised",
        help="Operational mode (default: supervised)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with reduced complexity and limited runtime"
    )
    
    parser.add_argument(
        "--daemon",
        action="store_true", 
        help="Run as background daemon with minimal output"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Create and run NFCS application
    nfcs_app = NFCSMain(args)
    return await nfcs_app.run()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 NFCS interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)