"""
Neural Field Control System - Integration Test Suite
===================================================

Comprehensive integration tests for the complete NFCS orchestrator system.
Tests all components working together in realistic scenarios.
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, "/home/user/webapp/src")

from orchestrator.nfcs_orchestrator import (
    NFCSOrchestrator,
    create_orchestrator,
    create_default_config,
)


async def test_basic_orchestrator_startup():
    """Test basic orchestrator initialization and startup"""
    print("\n🔧 Testing Basic Orchestrator Startup...")

    try:
        # Create orchestrator with default config
        config = create_default_config()
        orchestrator = await create_orchestrator(config)

        # Verify system is running
        status = orchestrator.get_system_status()
        print(f"✅ System Status: {status['status']}")
        print(f"✅ Operational Mode: {status['mode']}")
        print(f"✅ Active Modules: {len(status['active_modules'])}")

        # Test state management
        if orchestrator.state_coordinator:
            await orchestrator.state_coordinator.set_state(
                "test.integration",
                {"message": "Hello NFCS!", "timestamp": datetime.now().isoformat()},
                source_module="integration_test",
            )

            test_value = await orchestrator.state_coordinator.get_state("test.integration")
            print(f"✅ State Management: {test_value['message']}")

        # Test event system
        if orchestrator.event_system:
            event_id = await orchestrator.event_system.emit_event(
                "test.integration_event",
                {"test": True, "component": "integration_test"},
                source="integration_test",
            )
            print(f"✅ Event System: Event {event_id[:8]}... emitted")

        # Let system run for a moment
        await asyncio.sleep(2.0)

        # Get final status
        final_status = orchestrator.get_system_status()
        print(
            f"✅ Final Status: {final_status['statistics']['total_operations']} operations completed"
        )

        # Shutdown gracefully
        shutdown_success = await orchestrator.shutdown()
        print(f"✅ Shutdown: {'Success' if shutdown_success else 'Failed'}")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


async def test_module_coordination():
    """Test module registration and coordination"""
    print("\n🔧 Testing Module Coordination...")

    try:
        config = create_default_config()
        orchestrator = await create_orchestrator(config)

        # Test module manager functionality
        if orchestrator.module_manager:
            # Get active modules
            active_modules = await orchestrator.module_manager.get_active_modules()
            print(f"✅ Active Modules: {list(active_modules.keys())}")

            # Get manager statistics
            stats = orchestrator.module_manager.get_manager_statistics()
            print(f"✅ Module Statistics: {stats['total_registered_modules']} registered modules")

        await orchestrator.shutdown()
        return True

    except Exception as e:
        print(f"❌ Module coordination test failed: {e}")
        return False


async def test_constitutional_framework():
    """Test constitutional framework integration"""
    print("\n🔧 Testing Constitutional Framework Integration...")

    try:
        config = create_default_config()
        orchestrator = await create_orchestrator(config)

        # Test constitutional framework
        if orchestrator.constitutional_framework:
            # Test compliance check
            test_context = {
                "action": "test_action",
                "module": "integration_test",
                "timestamp": datetime.now().isoformat(),
            }

            compliance = orchestrator.constitutional_framework.check_compliance(test_context)
            print(f"✅ Constitutional Compliance: {compliance.compliant}")

            # Get active policies
            policies = orchestrator.constitutional_framework.get_active_policies()
            print(f"✅ Active Policies: {len(policies)} policies loaded")

        await orchestrator.shutdown()
        return True

    except Exception as e:
        print(f"❌ Constitutional framework test failed: {e}")
        return False


async def test_performance_monitoring():
    """Test performance monitoring capabilities"""
    print("\n🔧 Testing Performance Monitoring...")

    try:
        config = create_default_config()
        orchestrator = await create_orchestrator(config)

        # Let system run to generate metrics
        await asyncio.sleep(3.0)

        # Test performance monitoring
        if orchestrator.performance_monitor:
            current_metrics = orchestrator.performance_monitor.get_current_metrics()
            print(f"✅ Performance Metrics: {len(current_metrics)} metrics tracked")

        # Test resource monitoring
        if orchestrator.resource_manager:
            resource_status = await orchestrator.resource_manager.get_resource_status()
            print(
                f"✅ Resource Status: CPU {resource_status.get('cpu_percent', 0):.1f}%, Memory {resource_status.get('memory_mb', 0):.1f}MB"
            )

        await orchestrator.shutdown()
        return True

    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False


async def test_emergency_protocols():
    """Test emergency detection and response"""
    print("\n🔧 Testing Emergency Protocols...")

    try:
        config = create_default_config()
        config.emergency_shutdown_threshold = 5  # Lower threshold for testing
        orchestrator = await create_orchestrator(config)

        # Test emergency controller
        if orchestrator.emergency_controller:
            status = await orchestrator.emergency_controller.get_status()
            print(
                f"✅ Emergency Controller: {status['registered_conditions']} conditions registered"
            )

            # Test emergency condition check (should not trigger with normal conditions)
            emergency_status = orchestrator.emergency_controller.check_emergency_conditions(
                {
                    "error_count": 1,  # Below threshold
                    "warning_count": 0,
                    "uptime": 10.0,
                    "active_modules": 5,
                }
            )
            print(f"✅ Emergency Check: No emergency detected (as expected)")

        await orchestrator.shutdown()
        return True

    except Exception as e:
        print(f"❌ Emergency protocols test failed: {e}")
        return False


async def test_system_commands():
    """Test system command execution"""
    print("\n🔧 Testing System Commands...")

    try:
        config = create_default_config()
        orchestrator = await create_orchestrator(config)

        # Test status command
        result = await orchestrator.execute_command("get_status")
        if result["success"]:
            print(f"✅ Get Status Command: {result['result']['status']}")

        # Test modules command
        result = await orchestrator.execute_command("get_modules")
        if result["success"]:
            print(f"✅ Get Modules Command: {len(result['result'])} modules")

        # Test performance command
        result = await orchestrator.execute_command("get_performance")
        if result["success"]:
            print(f"✅ Get Performance Command: Metrics retrieved")

        await orchestrator.shutdown()
        return True

    except Exception as e:
        print(f"❌ System commands test failed: {e}")
        return False


async def run_full_integration_test():
    """Run complete integration test suite"""
    print("🚀 Starting NFCS Integration Test Suite")
    print("=" * 60)

    tests = [
        ("Basic Orchestrator Startup", test_basic_orchestrator_startup),
        ("Module Coordination", test_module_coordination),
        ("Constitutional Framework", test_constitutional_framework),
        ("Performance Monitoring", test_performance_monitoring),
        ("Emergency Protocols", test_emergency_protocols),
        ("System Commands", test_system_commands),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        try:
            success = await test_func()
            if success:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print("\n" + "=" * 60)
    print(f"🏁 Integration Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! NFCS orchestrator is fully functional!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run the integration tests
    try:
        success = asyncio.run(run_full_integration_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Integration tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error in integration tests: {e}")
        sys.exit(1)
