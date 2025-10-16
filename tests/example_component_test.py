"""
Example component test demonstrating how to test HierarchicalClient with actual backends.

This is a simplified example that shows the basic pattern for component testing.
"""

import asyncio
import os
import pytest
import sys
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.agent import HierarchicalClient


class TestExampleComponent:
    """Example component test class."""
    
    @pytest.mark.asyncio
    async def test_simple_math_with_openai(self):
        """Example: Test simple math with OpenAI backend."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")
        
        # Set up environment
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
            'LLM_PROVIDER': 'openai'
        }):
            # Create client with actual backend
            client = HierarchicalClient("gpt-3.5-turbo", "gpt-3.5-turbo")
            
            # Test simple query
            result = await client.process_query(
                "What is 5 + 3? Please give me the final answer.",
                "test.txt"
            )
            
            # Verify result
            assert result is not None
            assert len(result) > 0
            # Should contain "8" or "eight"
            assert any(x in result.lower() for x in ["8", "eight"])
    
    @pytest.mark.asyncio
    async def test_simple_math_with_anthropic(self):
        """Example: Test simple math with Anthropic backend."""
        # Skip if no API key
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("Anthropic API key not available")
        
        # Set up environment
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': os.getenv("ANTHROPIC_API_KEY"),
            'ANTHROPIC_API_URL': os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com"),
            'LLM_PROVIDER': 'anthropic'
        }):
            # Create client with actual backend
            client = HierarchicalClient("claude-3-haiku-20240307", "claude-3-haiku-20240307")
            
            # Test simple query
            result = await client.process_query(
                "What is 6 + 4? Please give me the final answer.",
                "test.txt"
            )
            
            # Verify result
            assert result is not None
            assert len(result) > 0
            # Should contain "10" or "ten"
            assert any(x in result.lower() for x in ["10", "ten"])
    
    @pytest.mark.asyncio
    async def test_planning_task_with_mixed_backends(self):
        """Example: Test planning task with mixed backends."""
        # Skip if no API keys
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("Required API keys not available")
        
        # Set up environment
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
            'ANTHROPIC_API_KEY': os.getenv("ANTHROPIC_API_KEY"),
            'ANTHROPIC_API_URL': os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com")
        }):
            # Mock get_default_backend to return specific backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                from commons.llm import OpenAIBackend, AnthropicBackend
                
                mock_get_backend.side_effect = lambda model: (
                    OpenAIBackend(model) if model == "gpt-3.5-turbo" 
                    else AnthropicBackend(model)
                )
                
                # Create client with mixed backends
                client = HierarchicalClient("gpt-3.5-turbo", "claude-3-haiku-20240307")
                
                # Test planning task
                result = await client.process_query(
                    "I need to calculate the area of a rectangle with length 6 and width 4. "
                    "Please break this down into steps and give me the final answer.",
                    "test.txt"
                )
                
                # Verify result
                assert result is not None
                assert len(result) > 0
                # Should contain "24" (6 * 4)
                assert any(x in result.lower() for x in ["24", "twenty-four"])


if __name__ == "__main__":
    # Run the example tests
    pytest.main([__file__, "-v"])
