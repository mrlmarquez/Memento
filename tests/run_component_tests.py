#!/usr/bin/env python3
"""
Component test runner for HierarchicalClient.

This script runs component tests that use actual backend services
(OpenAI and Anthropic) to verify end-to-end functionality.
"""

import subprocess
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from component_test_config import check_api_keys, print_setup_instructions, validate_setup


def run_component_tests():
    """Run component tests with proper configuration."""
    print("Running HierarchicalClient Component Tests")
    print("=" * 50)
    
    # Check setup
    if not validate_setup():
        print("❌ Setup incomplete. Please configure API keys first.")
        print()
        print_setup_instructions()
        return 1
    
    # Show available services
    status = check_api_keys()
    print("Available services:")
    for service, available in status.items():
        status_str = "✓ Available" if available else "✗ Missing"
        print(f"  {service}: {status_str}")
    print()
    
    # Change to project directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run component tests
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_hierarchical_client_component.py",
        "-v",
        "--tb=short",
        "--durations=10"  # Show slowest 10 tests
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Please install it with: pip install pytest")
        return 1


def run_specific_test(test_name: str):
    """Run a specific component test."""
    print(f"Running specific test: {test_name}")
    print("=" * 50)
    
    if not validate_setup():
        print("❌ Setup incomplete. Please configure API keys first.")
        return 1
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    cmd = [
        sys.executable, "-m", "pytest",
        f"tests/test_hierarchical_client_component.py::{test_name}",
        "-v",
        "--tb=short"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Please install it with: pip install pytest")
        return 1


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "setup":
            print_setup_instructions()
            return 0
        elif sys.argv[1] == "check":
            status = check_api_keys()
            print("API Key Status:")
            for service, available in status.items():
                status_str = "✓ Available" if available else "✗ Missing"
                print(f"  {service}: {status_str}")
            return 0
        elif sys.argv[1].startswith("test:"):
            test_name = sys.argv[1][5:]  # Remove "test:" prefix
            return run_specific_test(test_name)
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Available commands:")
            print("  setup  - Show setup instructions")
            print("  check  - Check API key status")
            print("  test:<name> - Run specific test")
            return 1
    else:
        return run_component_tests()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

