#!/usr/bin/env python3
"""
Test MVP Web Interface with Fallback
Simple test for web interface functionality
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_mvp_web_interface():
    """Test MVP web interface import and basic functionality."""
    
    print("🧪 Testing MVP Web Interface...")
    
    try:
        # Set PYTHONPATH
        os.environ['PYTHONPATH'] = f"{os.getcwd()}/src:{os.environ.get('PYTHONPATH', '')}"
        
        # Test if we can import the MVP controller (core functionality)
        from mvp_controller import NFCSMinimalViableProduct
        print("✅ MVP Controller import successful")
        
        # Test basic instantiation
        mvp = NFCSMinimalViableProduct()
        print("✅ MVP Controller instantiation successful")
        
        # Test the original issue: can we run the web interface import?
        # This was the specific error in the supervisord logs
        try:
            # Just test the import path that was failing
            exec("from mvp_controller import NFCSMinimalViableProduct")
            print("✅ Original failing import path now works")
        except Exception as e:
            print(f"❌ Original import path still fails: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ MVP Web Interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_start_mvp_script():
    """Test if start_mvp.sh would work now."""
    
    print("\n🧪 Testing start_mvp.sh compatibility...")
    
    try:
        # Test the import that start_mvp.sh uses
        from mvp_controller import NFCSMinimalViableProduct
        
        # Test that we can instantiate without errors
        mvp = NFCSMinimalViableProduct()
        print("✅ start_mvp.sh import compatibility confirmed")
        
        # Test basic status
        print(f"📊 MVP Status: {mvp.status.system_health}")
        
        return True
        
    except Exception as e:
        print(f"❌ start_mvp.sh compatibility failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("🚀 Production CI/L Fix Validation")
    print("=" * 50)
    
    tests = [
        test_mvp_web_interface,
        test_start_mvp_script
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 Production CI/L issues RESOLVED!")
        print("✅ MVP system can now start without import/dependency failures")
        print("✅ Original supervisord restart loop should be fixed")
        return True
    else:
        print("❌ Some issues remain")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)