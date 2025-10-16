"""
Configuration helper for component tests.

This module provides utilities to check and configure API keys
for running component tests with actual backend services.
"""

import os
from typing import Dict, List, Optional


def check_api_keys() -> Dict[str, bool]:
    """
    Check which API keys are available for component tests.
    
    Returns:
        Dictionary mapping service names to availability status
    """
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "azure": bool(os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))
    }


def get_required_env_vars() -> List[str]:
    """
    Get list of required environment variables for component tests.
    
    Returns:
        List of environment variable names
    """
    return [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_URL",  # Optional, defaults to https://api.anthropic.com
        "AZURE_OPENAI_KEY",   # Optional
        "AZURE_OPENAI_ENDPOINT",  # Optional
        "OPENAI_API_VERSION",  # Optional for Azure
    ]


def print_setup_instructions():
    """Print setup instructions for component tests."""
    print("Component Test Setup Instructions")
    print("=" * 40)
    print()
    print("To run component tests with actual backend services, you need to set up API keys.")
    print()
    print("Required Environment Variables:")
    print("- OPENAI_API_KEY: Your OpenAI API key")
    print("- ANTHROPIC_API_KEY: Your Anthropic API key")
    print()
    print("Optional Environment Variables:")
    print("- ANTHROPIC_API_URL: Anthropic API URL (defaults to https://api.anthropic.com)")
    print("- AZURE_OPENAI_KEY: Your Azure OpenAI key")
    print("- AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint")
    print("- OPENAI_API_VERSION: Azure OpenAI API version (defaults to 2024-02-15-preview)")
    print()
    print("You can set these in your shell:")
    print("export OPENAI_API_KEY='your-openai-key'")
    print("export ANTHROPIC_API_KEY='your-anthropic-key'")
    print()
    print("Or create a .env file in the project root:")
    print("OPENAI_API_KEY=your-openai-key")
    print("ANTHROPIC_API_KEY=your-anthropic-key")
    print()
    print("Current API Key Status:")
    status = check_api_keys()
    for service, available in status.items():
        status_str = "✓ Available" if available else "✗ Missing"
        print(f"  {service}: {status_str}")


def validate_setup() -> bool:
    """
    Validate that the setup is correct for running component tests.
    
    Returns:
        True if setup is valid, False otherwise
    """
    status = check_api_keys()
    return status["openai"] or status["anthropic"]


if __name__ == "__main__":
    print_setup_instructions()
    if not validate_setup():
        print("\n❌ Setup incomplete. Please configure at least one API key.")
        exit(1)
    else:
        print("\n✅ Setup complete. You can run component tests.")

