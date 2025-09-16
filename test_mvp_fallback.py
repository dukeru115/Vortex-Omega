#!/usr/bin/env python3
"""
Test MVP functionality with fallback mechanisms
Tests the production CI/L issues and verifies fixes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic Python imports"""
    print("🧪 Testing basic imports...")
    
    try:
        import json
        import os
        import sys
        import time
        import asyncio
        import logging
        print("✅ Basic Python modules available")
        return True
    except ImportError as e:
        print(f"❌ Basic import failed: {e}")
        return False

def test_constitutional_class_name():
    """Test if the constitutional class name is correct"""
    print("🧪 Testing constitutional class name...")
    
    try:
        # Try to load the module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "constitutional_realtime", 
            "src/modules/constitutional_realtime.py"
        )
        
        if spec is None:
            print("❌ Module spec not found")
            return False
            
        # Check the file for class definitions without importing
        with open("src/modules/constitutional_realtime.py", 'r') as f:
            content = f.read()
            
        if "class ConstitutionalRealTimeMonitor:" in content:
            print("✅ ConstitutionalRealTimeMonitor class found")
            return True
        elif "class ConstitutionalRealtimeMonitor:" in content:
            print("❌ Old class name ConstitutionalRealtimeMonitor found")
            return False
        else:
            print("❌ No constitutional monitor class found")
            return False
            
    except Exception as e:
        print(f"❌ Class name test failed: {e}")
        return False

def test_mvp_imports():
    """Test MVP controller imports with fallback"""
    print("🧪 Testing MVP controller imports...")
    
    try:
        # Check if mvp_controller.py has correct imports
        with open("mvp_controller.py", 'r') as f:
            content = f.read()
            
        if "from modules.constitutional_realtime import ConstitutionalRealTimeMonitor" in content:
            print("✅ MVP controller has correct import")
            return True
        elif "from modules.constitutional_realtime import ConstitutionalRealtimeMonitor" in content:
            print("❌ MVP controller has incorrect import")
            return False
        else:
            print("❌ No constitutional import found in MVP controller")
            return False
            
    except Exception as e:
        print(f"❌ MVP import test failed: {e}")
        return False

def test_websockets_fallback():
    """Test websockets fallback mechanism"""
    print("🧪 Testing websockets fallback...")
    
    try:
        # Check if constitutional module has websockets fallback
        with open("src/modules/constitutional_realtime.py", 'r') as f:
            content = f.read()
            
        if "WEBSOCKETS_AVAILABLE" in content and "except ImportError:" in content:
            print("✅ Websockets fallback mechanism present")
            return True
        else:
            print("❌ No websockets fallback mechanism")
            return False
            
    except Exception as e:
        print(f"❌ Websockets fallback test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Production CI/L Fallback Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_constitutional_class_name,
        test_mvp_imports,
        test_websockets_fallback
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All fallback tests PASSED!")
        print("✅ Production CI/L issues should be resolved")
        return True
    else:
        print("❌ Some tests failed - CI/L issues remain")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)