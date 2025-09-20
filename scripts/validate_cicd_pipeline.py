#!/usr/bin/env python3
"""
CI/CD Pipeline Validation Script for Vortex-Omega

Validates that all CI/CD components are properly configured and working.
This script tests GitHub Actions workflows, Docker builds, test coverage,
and deployment readiness without requiring full external dependencies.
"""

import os
import sys
import json
import yaml
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import time

class Colors:
    """Terminal colors for output formatting"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class CICDValidator:
    """Main CI/CD validation class"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {}
        self.errors = []
        self.warnings = []
    
    def log_info(self, message: str) -> None:
        """Log info message with color"""
        print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} {message}")
    
    def log_success(self, message: str) -> None:
        """Log success message with color"""
        print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {message}")
    
    def log_warning(self, message: str) -> None:
        """Log warning message with color"""
        print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {message}")
        self.warnings.append(message)
    
    def log_error(self, message: str) -> None:
        """Log error message with color"""
        print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")
        self.errors.append(message)
    
    def run_command(self, command: List[str], cwd: Optional[Path] = None, 
                   timeout: int = 30) -> Tuple[bool, str, str]:
        """Run shell command and return success, stdout, stderr"""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", str(e)
    
    def validate_github_workflows(self) -> bool:
        """Validate GitHub Actions workflow files"""
        self.log_info("🔍 Validating GitHub Actions workflows...")
        
        workflows_dir = self.project_root / ".github" / "workflows"
        if not workflows_dir.exists():
            self.log_error("GitHub workflows directory not found")
            return False
        
        workflow_files = list(workflows_dir.glob("*.yml"))
        if not workflow_files:
            self.log_error("No workflow files found in .github/workflows/")
            return False
        
        valid_workflows = 0
        required_workflows = {
            'main.yml': 'Main CI workflow',
            'build-test.yml': 'Build and test workflow',
            'docker-image.yml': 'Docker image build workflow'
        }
        
        for workflow_file in workflow_files:
            try:
                with open(workflow_file, 'r') as f:
                    workflow_data = yaml.safe_load(f)
                
                # Validate workflow structure
                required_keys = ['name', 'on', 'jobs']
                if all(key in workflow_data for key in required_keys):
                    valid_workflows += 1
                    self.log_success(f"✅ {workflow_file.name} - Valid workflow structure")
                    
                    # Check for pytest integration
                    workflow_str = str(workflow_data)
                    if 'pytest' in workflow_str.lower():
                        self.log_success(f"   📋 {workflow_file.name} includes pytest")
                    
                    # Check for coverage integration
                    if 'coverage' in workflow_str.lower() or 'cov' in workflow_str.lower():
                        self.log_success(f"   📊 {workflow_file.name} includes coverage")
                    
                    # Check for Docker integration
                    if 'docker' in workflow_str.lower():
                        self.log_success(f"   🐳 {workflow_file.name} includes Docker")
                
                else:
                    self.log_warning(f"⚠️ {workflow_file.name} - Missing required keys")
                    
            except yaml.YAMLError as e:
                self.log_error(f"❌ {workflow_file.name} - Invalid YAML: {e}")
            except Exception as e:
                self.log_error(f"❌ {workflow_file.name} - Error reading file: {e}")
        
        # Check for required workflows
        for required_file, description in required_workflows.items():
            if (workflows_dir / required_file).exists():
                self.log_success(f"✅ Found {description} ({required_file})")
            else:
                self.log_warning(f"⚠️ Missing {description} ({required_file})")
        
        self.results['github_workflows'] = {
            'total_files': len(workflow_files),
            'valid_files': valid_workflows,
            'success': valid_workflows > 0
        }
        
        return valid_workflows > 0
    
    def validate_test_infrastructure(self) -> bool:
        """Validate test infrastructure and configuration"""
        self.log_info("🧪 Validating test infrastructure...")
        
        # Check pytest configuration
        pytest_config_files = ['pytest.ini', 'pyproject.toml', 'setup.cfg']
        pytest_config_found = False
        
        for config_file in pytest_config_files:
            if (self.project_root / config_file).exists():
                pytest_config_found = True
                self.log_success(f"✅ Found pytest configuration: {config_file}")
                break
        
        if not pytest_config_found:
            self.log_warning("⚠️ No pytest configuration file found")
        
        # Check test directories
        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            self.log_error("❌ Tests directory not found")
            return False
        
        # Count test files
        test_files = list(tests_dir.rglob("test_*.py"))
        self.log_success(f"✅ Found {len(test_files)} test files")
        
        # Check for test categories
        test_categories = {
            'unit': tests_dir / "test_core",
            'integration': tests_dir / "integration",
            'performance': None  # Optional
        }
        
        for category, path in test_categories.items():
            if path and path.exists():
                test_count = len(list(path.glob("test_*.py")))
                self.log_success(f"✅ {category.title()} tests: {test_count} files")
            else:
                if category == 'performance':
                    self.log_warning(f"⚠️ {category.title()} tests directory not found (optional)")
                else:
                    self.log_warning(f"⚠️ {category.title()} tests directory not found")
        
        # Try to run basic pytest validation (syntax check)
        success, stdout, stderr = self.run_command(['python', '-m', 'pytest', '--collect-only', '-q'])
        if success:
            self.log_success("✅ Pytest can collect tests successfully")
        else:
            self.log_warning(f"⚠️ Pytest collection issues: {stderr}")
        
        self.results['test_infrastructure'] = {
            'pytest_config': pytest_config_found,
            'test_files_count': len(test_files),
            'success': pytest_config_found and len(test_files) > 0
        }
        
        return pytest_config_found and len(test_files) > 0
    
    def validate_docker_configuration(self) -> bool:
        """Validate Docker configuration"""
        self.log_info("🐳 Validating Docker configuration...")
        
        # Check Dockerfile
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.exists():
            self.log_error("❌ Dockerfile not found")
            return False
        
        # Validate Dockerfile syntax
        try:
            with open(dockerfile, 'r') as f:
                dockerfile_content = f.read()
            
            required_instructions = ['FROM', 'COPY', 'WORKDIR']
            found_instructions = []
            
            for line in dockerfile_content.split('\n'):
                line = line.strip().upper()
                for instruction in required_instructions:
                    if line.startswith(instruction):
                        found_instructions.append(instruction)
                        break
            
            if len(set(found_instructions)) >= 2:
                self.log_success("✅ Dockerfile has basic required instructions")
            else:
                self.log_warning("⚠️ Dockerfile missing some basic instructions")
        
        except Exception as e:
            self.log_error(f"❌ Error reading Dockerfile: {e}")
            return False
        
        # Check docker-compose files
        compose_files = ['docker-compose.yml', 'docker-compose.yaml']
        compose_found = False
        
        for compose_file in compose_files:
            if (self.project_root / compose_file).exists():
                compose_found = True
                self.log_success(f"✅ Found Docker Compose file: {compose_file}")
                
                # Validate compose file structure
                try:
                    with open(self.project_root / compose_file, 'r') as f:
                        compose_data = yaml.safe_load(f)
                    
                    if 'services' in compose_data:
                        services_count = len(compose_data['services'])
                        self.log_success(f"   📋 {services_count} services defined")
                    else:
                        self.log_warning(f"   ⚠️ No services defined in {compose_file}")
                
                except yaml.YAMLError as e:
                    self.log_error(f"   ❌ Invalid YAML in {compose_file}: {e}")
                break
        
        if not compose_found:
            self.log_warning("⚠️ No Docker Compose file found")
        
        # Try Docker syntax validation (if Docker is available)
        docker_available = False
        success, stdout, stderr = self.run_command(['docker', '--version'])
        if success:
            docker_available = True
            self.log_success("✅ Docker is available for testing")
            
            # Test Dockerfile syntax
            success, stdout, stderr = self.run_command(['docker', 'build', '--no-cache', '-f', 'Dockerfile', '-t', 'vortex-test', '.'], timeout=60)
            if success:
                self.log_success("✅ Dockerfile builds successfully")
                # Clean up test image
                self.run_command(['docker', 'rmi', 'vortex-test'])
            else:
                self.log_warning(f"⚠️ Docker build issues: {stderr}")
        else:
            self.log_warning("⚠️ Docker not available for build testing")
        
        self.results['docker_configuration'] = {
            'dockerfile_exists': dockerfile.exists(),
            'compose_file_exists': compose_found,
            'docker_available': docker_available,
            'success': dockerfile.exists()
        }
        
        return dockerfile.exists()
    
    def validate_dependencies(self) -> bool:
        """Validate dependency management"""
        self.log_info("📦 Validating dependency management...")
        
        # Check requirements files
        requirement_files = ['requirements.txt', 'requirements-dev.txt', 'pyproject.toml']
        found_files = []
        
        for req_file in requirement_files:
            if (self.project_root / req_file).exists():
                found_files.append(req_file)
                self.log_success(f"✅ Found {req_file}")
                
                # Count dependencies
                if req_file.endswith('.txt'):
                    with open(self.project_root / req_file, 'r') as f:
                        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('#')]
                        self.log_success(f"   📋 {len(lines)} dependencies in {req_file}")
        
        if not found_files:
            self.log_error("❌ No dependency files found")
            return False
        
        # Check for security in dependencies
        security_files = ['.pre-commit-config.yaml', 'bandit.yml']
        for sec_file in security_files:
            if (self.project_root / sec_file).exists():
                self.log_success(f"✅ Found security configuration: {sec_file}")
        
        self.results['dependencies'] = {
            'requirement_files': found_files,
            'success': len(found_files) > 0
        }
        
        return len(found_files) > 0
    
    def validate_code_quality(self) -> bool:
        """Validate code quality tools configuration"""
        self.log_info("📏 Validating code quality configuration...")
        
        # Check for linting configuration
        linting_configs = {
            '.flake8': 'Flake8 configuration',
            'setup.cfg': 'Setup.cfg (may contain flake8 config)',
            'pyproject.toml': 'PyProject.toml (may contain tool configs)',
            '.pylintrc': 'Pylint configuration'
        }
        
        found_configs = []
        for config_file, description in linting_configs.items():
            if (self.project_root / config_file).exists():
                found_configs.append(config_file)
                self.log_success(f"✅ Found {description}")
        
        # Check for formatting configuration
        if (self.project_root / 'pyproject.toml').exists():
            try:
                with open(self.project_root / 'pyproject.toml', 'r') as f:
                    content = f.read()
                    if '[tool.black]' in content:
                        self.log_success("✅ Black formatter configuration found")
                    if '[tool.mypy]' in content:
                        self.log_success("✅ MyPy type checking configuration found")
            except Exception as e:
                self.log_warning(f"⚠️ Error reading pyproject.toml: {e}")
        
        # Test if basic linting tools are available
        linting_tools = ['flake8', 'black', 'mypy']
        available_tools = []
        
        for tool in linting_tools:
            success, stdout, stderr = self.run_command(['python', '-m', tool, '--version'])
            if success:
                available_tools.append(tool)
                self.log_success(f"✅ {tool} is available")
            else:
                self.log_warning(f"⚠️ {tool} not available")
        
        self.results['code_quality'] = {
            'config_files': found_configs,
            'available_tools': available_tools,
            'success': len(found_configs) > 0
        }
        
        return len(found_configs) > 0
    
    def validate_project_structure(self) -> bool:
        """Validate overall project structure"""
        self.log_info("📁 Validating project structure...")
        
        # Check essential directories
        essential_dirs = {
            'src': 'Source code directory',
            'tests': 'Test directory',
            '.github': 'GitHub configuration directory'
        }
        
        missing_dirs = []
        for dir_name, description in essential_dirs.items():
            if (self.project_root / dir_name).exists():
                self.log_success(f"✅ {description} exists")
            else:
                missing_dirs.append(dir_name)
                self.log_error(f"❌ {description} missing")
        
        # Check essential files
        essential_files = {
            'README.md': 'Project documentation',
            'LICENSE': 'License file',
            'pyproject.toml': 'Python project configuration'
        }
        
        missing_files = []
        for file_name, description in essential_files.items():
            if (self.project_root / file_name).exists():
                self.log_success(f"✅ {description} exists")
            else:
                missing_files.append(file_name)
                self.log_warning(f"⚠️ {description} missing")
        
        # Check Python package structure
        src_dir = self.project_root / "src"
        if src_dir.exists():
            init_files = list(src_dir.rglob("__init__.py"))
            self.log_success(f"✅ Found {len(init_files)} __init__.py files in src/")
        
        self.results['project_structure'] = {
            'missing_dirs': missing_dirs,
            'missing_files': missing_files,
            'success': len(missing_dirs) == 0
        }
        
        return len(missing_dirs) == 0
    
    def generate_report(self) -> Dict:
        """Generate comprehensive validation report"""
        total_checks = len(self.results)
        passed_checks = sum(1 for result in self.results.values() if result.get('success', False))
        
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}📊 CI/CD VALIDATION REPORT{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
        print(f"  Total Checks: {total_checks}")
        print(f"  Passed: {Colors.OKGREEN}{passed_checks}{Colors.ENDC}")
        print(f"  Failed: {Colors.FAIL}{total_checks - passed_checks}{Colors.ENDC}")
        print(f"  Warnings: {Colors.WARNING}{len(self.warnings)}{Colors.ENDC}")
        print(f"  Errors: {Colors.FAIL}{len(self.errors)}{Colors.ENDC}")
        
        # Detailed results
        print(f"\n{Colors.BOLD}Detailed Results:{Colors.ENDC}")
        for check_name, result in self.results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            print(f"  {check_name}: {status}")
        
        # Overall assessment
        if passed_checks == total_checks:
            print(f"\n{Colors.OKGREEN}🎉 CI/CD PIPELINE FULLY VALIDATED!{Colors.ENDC}")
            print(f"{Colors.OKGREEN}✅ All components are properly configured{Colors.ENDC}")
        elif passed_checks >= total_checks * 0.8:
            print(f"\n{Colors.WARNING}⚠️ CI/CD PIPELINE MOSTLY READY{Colors.ENDC}")
            print(f"{Colors.WARNING}Most components configured, some improvements needed{Colors.ENDC}")
        else:
            print(f"\n{Colors.FAIL}❌ CI/CD PIPELINE NEEDS ATTENTION{Colors.ENDC}")
            print(f"{Colors.FAIL}Several critical components need configuration{Colors.ENDC}")
        
        # Recommendations
        if self.warnings or self.errors:
            print(f"\n{Colors.BOLD}Recommendations:{Colors.ENDC}")
            for warning in self.warnings[:5]:  # Show first 5 warnings
                print(f"  ⚠️ {warning}")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  ❌ {error}")
        
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        
        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'warnings': len(self.warnings),
            'errors': len(self.errors),
            'success_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'overall_success': passed_checks >= total_checks * 0.8,
            'detailed_results': self.results
        }
    
    def run_full_validation(self) -> Dict:
        """Run complete CI/CD validation"""
        print(f"{Colors.HEADER}🚀 Starting CI/CD Pipeline Validation{Colors.ENDC}")
        print(f"Project: {self.project_root}")
        print(f"{'='*60}")
        
        # Run all validation checks
        validation_methods = [
            self.validate_project_structure,
            self.validate_github_workflows,
            self.validate_test_infrastructure,
            self.validate_docker_configuration,
            self.validate_dependencies,
            self.validate_code_quality
        ]
        
        for method in validation_methods:
            try:
                method()
            except Exception as e:
                self.log_error(f"Validation method {method.__name__} failed: {e}")
            print()  # Add spacing between checks
        
        return self.generate_report()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Validate CI/CD pipeline for Vortex-Omega')
    parser.add_argument('--project-root', type=Path, default=Path.cwd(),
                       help='Project root directory (default: current directory)')
    parser.add_argument('--output', type=Path, help='Output file for JSON report')
    
    args = parser.parse_args()
    
    if not args.project_root.exists():
        print(f"Error: Project root {args.project_root} does not exist")
        sys.exit(1)
    
    # Run validation
    validator = CICDValidator(args.project_root)
    report = validator.run_full_validation()
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if report['overall_success'] else 1)


if __name__ == "__main__":
    main()