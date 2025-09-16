#!/usr/bin/env python3
"""
Offline CI Validation Script for Vortex-Omega
Validates code quality and functionality without external dependencies
"""

import os
import sys
import subprocess
import importlib.util
import ast
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class OfflineCIValidator:
    """Comprehensive offline validation for CI/CD pipelines."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.src_path = self.project_root / "src"
        self.tests_path = self.project_root / "tests"
        self.setup_python_path()
        
        self.results = {
            "syntax_check": {"passed": 0, "failed": 0, "errors": []},
            "import_check": {"passed": 0, "failed": 0, "errors": []},
            "structure_check": {"passed": 0, "failed": 0, "errors": []},
            "code_complexity": {"passed": 0, "failed": 0, "errors": []},
            "security_basic": {"passed": 0, "failed": 0, "errors": []},
        }
        
    def setup_python_path(self):
        """Setup PYTHONPATH for proper imports."""
        if str(self.src_path) not in sys.path:
            sys.path.insert(0, str(self.src_path))
        
        current_pythonpath = os.environ.get('PYTHONPATH', '')
        if str(self.src_path) not in current_pythonpath:
            os.environ['PYTHONPATH'] = f"{self.src_path}:{current_pythonpath}"
    
    def check_python_syntax(self) -> bool:
        """Check Python syntax for all .py files."""
        print("🔍 Checking Python syntax...")
        
        py_files = list(self.src_path.rglob("*.py")) + list(self.tests_path.rglob("*.py"))
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to check syntax
                ast.parse(content)
                
                # Also try py_compile for double-check
                result = subprocess.run(
                    [sys.executable, '-m', 'py_compile', str(py_file)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    self.results["syntax_check"]["passed"] += 1
                else:
                    self.results["syntax_check"]["failed"] += 1
                    self.results["syntax_check"]["errors"].append(
                        f"{py_file}: {result.stderr.strip()}"
                    )
                    
            except SyntaxError as e:
                self.results["syntax_check"]["failed"] += 1
                self.results["syntax_check"]["errors"].append(
                    f"{py_file}: SyntaxError line {e.lineno}: {e.msg}"
                )
            except Exception as e:
                self.results["syntax_check"]["failed"] += 1
                self.results["syntax_check"]["errors"].append(
                    f"{py_file}: {type(e).__name__}: {e}"
                )
        
        total_files = len(py_files)
        passed = self.results["syntax_check"]["passed"]
        failed = self.results["syntax_check"]["failed"]
        
        print(f"   ✅ {passed}/{total_files} files passed syntax check")
        if failed > 0:
            print(f"   ❌ {failed} files failed syntax check")
            for error in self.results["syntax_check"]["errors"][:3]:
                print(f"      - {error}")
            if len(self.results["syntax_check"]["errors"]) > 3:
                print(f"      ... and {len(self.results['syntax_check']['errors']) - 3} more")
        
        return failed == 0
    
    def check_imports(self) -> bool:
        """Test basic imports of core modules."""
        print("📦 Testing core module imports...")
        
        import_tests = [
            ("src", "Core src module"),
            ("json", "JSON module"),
            ("os", "OS module"),
            ("sys", "System module"),
            ("logging", "Logging module"),
            ("pathlib", "Pathlib module"),
            ("asyncio", "Asyncio module"),
            ("multiprocessing", "Multiprocessing module"),
        ]
        
        for module_name, description in import_tests:
            try:
                if module_name == "src":
                    # Special handling for src module
                    spec = importlib.util.find_spec("src")
                    if spec is None:
                        self.results["import_check"]["failed"] += 1
                        self.results["import_check"]["errors"].append(
                            f"{module_name}: Module not found"
                        )
                    else:
                        self.results["import_check"]["passed"] += 1
                        print(f"   ✅ {description}")
                else:
                    importlib.import_module(module_name)
                    self.results["import_check"]["passed"] += 1
                    print(f"   ✅ {description}")
                    
            except ImportError as e:
                self.results["import_check"]["failed"] += 1
                self.results["import_check"]["errors"].append(f"{module_name}: {e}")
                print(f"   ❌ {description}: {e}")
            except Exception as e:
                self.results["import_check"]["failed"] += 1
                self.results["import_check"]["errors"].append(f"{module_name}: {e}")
                print(f"   ⚠️  {description}: {e}")
        
        return self.results["import_check"]["failed"] == 0
    
    def check_project_structure(self) -> bool:
        """Check if essential project files exist."""
        print("📁 Checking project structure...")
        
        essential_files = [
            "src/__init__.py",
            "requirements.txt", 
            "pyproject.toml",
            "Dockerfile",
            ".github/workflows/ci-simple.yml",
            "README.md",
        ]
        
        optional_files = [
            "requirements-dev.txt",
            ".gitlab-ci.yml",
            "Jenkinsfile",
            "docker-compose.yml",
            "pytest.ini",
        ]
        
        # Check essential files
        for file_path in essential_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.results["structure_check"]["passed"] += 1
                print(f"   ✅ {file_path}")
            else:
                self.results["structure_check"]["failed"] += 1
                self.results["structure_check"]["errors"].append(f"Missing: {file_path}")
                print(f"   ❌ Missing: {file_path}")
        
        # Check optional files (warnings only)
        for file_path in optional_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"   ✅ {file_path} (optional)")
            else:
                print(f"   ⚠️  Missing: {file_path} (optional)")
        
        return self.results["structure_check"]["failed"] == 0
    
    def check_code_complexity(self) -> bool:
        """Basic code complexity analysis."""
        print("🧮 Analyzing code complexity...")
        
        py_files = list(self.src_path.rglob("*.py"))
        complex_files = []
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Count functions and classes
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                
                # Simple complexity heuristic
                complexity_score = len(functions) + len(classes) * 2
                lines_of_code = len([line for line in content.split('\n') if line.strip()])
                
                # Flag files with very high complexity
                if complexity_score > 50 or lines_of_code > 1000:
                    complex_files.append((py_file, complexity_score, lines_of_code))
                    
                self.results["code_complexity"]["passed"] += 1
                
            except Exception as e:
                self.results["code_complexity"]["failed"] += 1
                self.results["code_complexity"]["errors"].append(f"{py_file}: {e}")
        
        if complex_files:
            print(f"   ⚠️  {len(complex_files)} files with high complexity:")
            for file_path, score, loc in complex_files[:3]:
                print(f"      - {file_path.name}: {loc} lines, complexity {score}")
        else:
            print("   ✅ No files with excessive complexity detected")
        
        return self.results["code_complexity"]["failed"] == 0
    
    def check_basic_security(self) -> bool:
        """Basic security checks."""
        print("🔒 Running basic security checks...")
        
        py_files = list(self.src_path.rglob("*.py"))
        security_issues = []
        
        dangerous_patterns = [
            (r'eval\s*\(', 'Use of eval() function'),
            (r'exec\s*\(', 'Use of exec() function'),
            (r'__import__\s*\(', 'Use of __import__() function'),
            (r'subprocess\..*shell=True', 'Subprocess with shell=True'),
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret'),
        ]
        
        import re
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, description in dangerous_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        security_issues.append((py_file, description, len(matches)))
                        
                self.results["security_basic"]["passed"] += 1
                
            except Exception as e:
                self.results["security_basic"]["failed"] += 1
                self.results["security_basic"]["errors"].append(f"{py_file}: {e}")
        
        if security_issues:
            print(f"   ⚠️  {len(security_issues)} potential security issues found:")
            for file_path, issue, count in security_issues[:3]:
                print(f"      - {file_path.name}: {issue} ({count} occurrences)")
        else:
            print("   ✅ No obvious security issues detected")
        
        return True  # Don't fail CI for security warnings
    
    def generate_report(self) -> Dict:
        """Generate comprehensive validation report."""
        total_checks = sum(
            self.results[category]["passed"] + self.results[category]["failed"]
            for category in self.results
        )
        total_passed = sum(self.results[category]["passed"] for category in self.results)
        total_failed = sum(self.results[category]["failed"] for category in self.results)
        
        report = {
            "summary": {
                "total_checks": total_checks,
                "passed": total_passed,
                "failed": total_failed,
                "success_rate": total_passed / total_checks if total_checks > 0 else 0,
            },
            "details": self.results,
            "recommendations": []
        }
        
        # Add recommendations based on results
        if self.results["syntax_check"]["failed"] > 0:
            report["recommendations"].append(
                "Fix Python syntax errors before proceeding with CI/CD"
            )
        
        if self.results["import_check"]["failed"] > 0:
            report["recommendations"].append(
                "Resolve import issues and verify PYTHONPATH configuration"
            )
        
        if self.results["structure_check"]["failed"] > 0:
            report["recommendations"].append(
                "Add missing essential project files"
            )
        
        return report
    
    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("🚀 Starting Offline CI Validation for Vortex-Omega")
        print("=" * 60)
        
        checks = [
            ("Project Structure", self.check_project_structure),
            ("Python Syntax", self.check_python_syntax),
            ("Core Imports", self.check_imports),
            ("Code Complexity", self.check_code_complexity),
            ("Basic Security", self.check_basic_security),
        ]
        
        success = True
        
        for check_name, check_func in checks:
            print(f"\n📋 Running {check_name}...")
            try:
                result = check_func()
                if result:
                    print(f"✅ {check_name} PASSED")
                else:
                    print(f"❌ {check_name} FAILED")
                    success = False
            except Exception as e:
                print(f"💥 {check_name} ERROR: {e}")
                success = False
        
        # Generate and display report
        print("\n" + "=" * 60)
        print("📊 Validation Summary")
        print("=" * 60)
        
        report = self.generate_report()
        summary = report["summary"]
        
        print(f"Total checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        
        if report["recommendations"]:
            print("\n🔧 Recommendations:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")
        
        if success:
            print("\n🎉 All critical validation checks PASSED!")
            print("🚀 Repository is ready for offline CI/CD pipelines!")
        else:
            print("\n⚠️  Some validation checks FAILED!")
            print("🔧 Please address the issues before proceeding with CI/CD.")
        
        return success


def main():
    """Main entry point."""
    validator = OfflineCIValidator()
    success = validator.run_validation()
    
    # Save report for CI systems
    report = validator.generate_report()
    report_path = Path("offline_ci_validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())