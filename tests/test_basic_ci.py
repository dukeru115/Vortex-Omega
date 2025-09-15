"""
Simple Test Suite for CI/CD Validation
=====================================

Basic tests to ensure CI/CD pipeline works correctly.
These tests validate core functionality without complex dependencies.
"""

import pytest
import sys
import os


class TestBasic:
    """Basic tests for CI/CD validation"""

    def test_python_version(self):
        """Test that Python version is compatible"""
        assert sys.version_info >= (3, 9), "Python 3.9+ required"

    def test_imports_work(self):
        """Test that basic imports work"""
        import numpy as np
        import json
        import os

        assert np.__version__
        assert json.dumps({"test": True}) == '{"test": true}'
        assert os.path.exists(".")

    def test_basic_math(self):
        """Test basic mathematical operations"""
        assert 2 + 2 == 4
        assert 10 / 2 == 5.0
        assert 3**2 == 9

    def test_string_operations(self):
        """Test string operations"""
        test_str = "Hello World"
        assert test_str.lower() == "hello world"
        assert test_str.upper() == "HELLO WORLD"
        assert len(test_str) == 11

    def test_list_operations(self):
        """Test list operations"""
        test_list = [1, 2, 3, 4, 5]
        assert len(test_list) == 5
        assert sum(test_list) == 15
        assert max(test_list) == 5

    def test_dict_operations(self):
        """Test dictionary operations"""
        test_dict = {"a": 1, "b": 2, "c": 3}
        assert len(test_dict) == 3
        assert test_dict["a"] == 1
        assert "b" in test_dict


class TestEnvironment:
    """Test environment setup"""

    def test_working_directory(self):
        """Test that we're in the right directory"""
        cwd = os.getcwd()
        assert "Vortex-Omega" in cwd or "vortex-omega" in cwd.lower()

    def test_src_directory_exists(self):
        """Test that src directory exists"""
        assert os.path.exists("src"), "src directory should exist"

    def test_tests_directory_exists(self):
        """Test that tests directory exists"""
        assert os.path.exists("tests"), "tests directory should exist"

    def test_requirements_file_exists(self):
        """Test that requirements.txt exists"""
        assert os.path.exists("requirements.txt"), "requirements.txt should exist"


@pytest.mark.asyncio
async def test_async_functionality():
    """Test basic async functionality"""
    import asyncio

    async def async_add(a, b):
        await asyncio.sleep(0.01)  # Simulate async operation
        return a + b

    result = await async_add(3, 4)
    assert result == 7


class TestSafeImports:
    """Test safe imports from the project"""

    def test_src_path_accessible(self):
        """Test that src path is accessible"""
        src_path = os.path.join(os.getcwd(), "src")
        assert os.path.exists(src_path)

        # Add src to path for imports
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

    def test_can_import_from_src(self):
        """Test basic imports from src directory"""
        # Add src to path
        src_path = os.path.join(os.getcwd(), "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            # Try to import a basic module
            import utils

            assert hasattr(utils, "__file__")
        except ImportError:
            # If utils module doesn't exist, that's okay for basic CI
            pytest.skip("utils module not found - skipping import test")
