"""
Component tests for HierarchicalClient.process_query with actual backend instances.

These tests use real OpenAI and Anthropic backends to verify end-to-end functionality
and integration between different backend combinations.
"""

import asyncio
import os
import pytest
import sys
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons.llm import AnthropicBackend, OpenAIBackend, get_default_backend
from client.agent import HierarchicalClient


class MockToolSession:
    """Mock tool session for component testing."""
    
    def __init__(self, tools: List[Dict[str, Any]]):
        self.tools = tools
    
    async def list_tools(self):
        """Mock list_tools method."""
        return MagicMock(tools=self.tools)


class MockTool:
    """Mock tool object with attributes."""
    
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class TestHierarchicalClientComponent:
    """Component tests with actual backend instances."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Check if we have the required API keys
        self.has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
        self.has_anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
        
        # Skip tests if no API keys are available
        if not self.has_openai_key and not self.has_anthropic_key:
            pytest.skip("No API keys available for component tests")
    
    @pytest.mark.asyncio
    async def test_openai_meta_openai_exec_component(self):
        """Test HierarchicalClient with actual OpenAI backends."""
        if not self.has_openai_key:
            pytest.skip("OpenAI API key not available")
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
            'LLM_PROVIDER': 'openai'
        }):
            client = HierarchicalClient("gpt-3.5-turbo", "gpt-3.5-turbo")
            
            # Test with a simple query that should get a final answer
            result = await client.process_query(
                "What is 2 + 2? Please give me the final answer.",
                "test.txt"
            )
            
            # Should get a final answer
            assert result is not None
            assert len(result) > 0
            # Should contain some form of "4" or "four"
            assert any(x in result.lower() for x in ["4", "four"])
    
    @pytest.mark.asyncio
    async def test_anthropic_meta_anthropic_exec_component(self):
        """Test HierarchicalClient with actual Anthropic backends."""
        if not self.has_anthropic_key:
            pytest.skip("Anthropic API key not available")
        
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': os.getenv("ANTHROPIC_API_KEY"),
            'ANTHROPIC_API_URL': os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com"),
            'LLM_PROVIDER': 'anthropic'
        }):
            client = HierarchicalClient("claude-3-haiku-20240307", "claude-3-haiku-20240307")
            
            # Test with a simple query that should get a final answer
            result = await client.process_query(
                "What is 3 + 3? Please give me the final answer.",
                "test.txt"
            )
            
            # Should get a final answer
            assert result is not None
            assert len(result) > 0
            # Should contain some form of "6" or "six"
            assert any(x in result.lower() for x in ["6", "six"])
    
    @pytest.mark.asyncio
    async def test_openai_meta_anthropic_exec_component(self):
        """Test HierarchicalClient with OpenAI meta and Anthropic exec."""
        if not self.has_openai_key or not self.has_anthropic_key:
            pytest.skip("Required API keys not available")
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
            'ANTHROPIC_API_KEY': os.getenv("ANTHROPIC_API_KEY"),
            'ANTHROPIC_API_URL': os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com")
        }):
            # Mock get_default_backend to return specific backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_get_backend.side_effect = lambda model: (
                    OpenAIBackend(model) if model == "gpt-3.5-turbo" 
                    else AnthropicBackend(model)
                )
                client = HierarchicalClient("gpt-3.5-turbo", "claude-3-haiku-20240307")
                
                # Test with a simple query
                result = await client.process_query(
                    "What is 4 + 4? Please give me the final answer.",
                    "test.txt"
                )
                
                # Should get a final answer
                assert result is not None
                assert len(result) > 0
                # Should contain some form of "8" or "eight"
                assert any(x in result.lower() for x in ["8", "eight"])
    
    @pytest.mark.asyncio
    async def test_anthropic_meta_openai_exec_component(self):
        """Test HierarchicalClient with Anthropic meta and OpenAI exec."""
        if not self.has_openai_key or not self.has_anthropic_key:
            pytest.skip("Required API keys not available")
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
            'ANTHROPIC_API_KEY': os.getenv("ANTHROPIC_API_KEY"),
            'ANTHROPIC_API_URL': os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com")
        }):
            # Mock get_default_backend to return specific backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_get_backend.side_effect = lambda model: (
                    AnthropicBackend(model) if model == "claude-3-haiku-20240307" 
                    else OpenAIBackend(model)
                )
                client = HierarchicalClient("claude-3-haiku-20240307", "gpt-3.5-turbo")
                
                # Test with a simple query
                result = await client.process_query(
                    "What is 5 + 5? Please give me the final answer.",
                    "test.txt"
                )
                
                # Should get a final answer
                assert result is not None
                assert len(result) > 0
                # Should contain some form of "10" or "ten"
                assert any(x in result.lower() for x in ["10", "ten"])
    
    @pytest.mark.asyncio
    async def test_planning_and_execution_flow_openai(self):
        """Test the full planning and execution flow with OpenAI."""
        if not self.has_openai_key:
            pytest.skip("OpenAI API key not available")
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
            'LLM_PROVIDER': 'openai'
        }):
            client = HierarchicalClient("gpt-3.5-turbo", "gpt-3.5-turbo")
            
            # Test with a query that requires planning
            result = await client.process_query(
                "I need to calculate the area of a rectangle with length 5 and width 3. "
                "Please break this down into steps and give me the final answer.",
                "test.txt"
            )
            
            # Should get a final answer
            assert result is not None
            assert len(result) > 0
            # Should contain some form of "15" (5 * 3)
            assert any(x in result.lower() for x in ["15", "fifteen"])
    
    @pytest.mark.asyncio
    async def test_planning_and_execution_flow_anthropic(self):
        """Test the full planning and execution flow with Anthropic."""
        if not self.has_anthropic_key:
            pytest.skip("Anthropic API key not available")
        
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': os.getenv("ANTHROPIC_API_KEY"),
            'ANTHROPIC_API_URL': os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com"),
            'LLM_PROVIDER': 'anthropic'
        }):
            client = HierarchicalClient("claude-3-haiku-20240307", "claude-3-haiku-20240307")
            
            # Test with a query that requires planning
            result = await client.process_query(
                "I need to calculate the perimeter of a square with side length 4. "
                "Please break this down into steps and give me the final answer.",
                "test.txt"
            )
            
            # Should get a final answer
            assert result is not None
            assert len(result) > 0
            # Should contain some form of "16" (4 * 4)
            assert any(x in result.lower() for x in ["16", "sixteen"])
    
    @pytest.mark.asyncio
    async def test_tool_integration_openai(self):
        """Test tool integration with OpenAI backend."""
        if not self.has_openai_key:
            pytest.skip("OpenAI API key not available")
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
            'LLM_PROVIDER': 'openai'
        }):
            client = HierarchicalClient("gpt-3.5-turbo", "gpt-3.5-turbo")
            
            # Mock tool session
            mock_tool = MockTool("calculator", "A simple calculator tool", {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "description": "The operation to perform"},
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                }
            })
            mock_session = MockToolSession([mock_tool])
            client.sessions = {"calculator": mock_session}
            
            # Test with a query that might use tools
            result = await client.process_query(
                "I need to add 7 and 8. Please use the calculator tool if available.",
                "test.txt"
            )
            
            # Should get a response (may or may not use tools)
            assert result is not None
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_tool_integration_anthropic(self):
        """Test tool integration with Anthropic backend."""
        if not self.has_anthropic_key:
            pytest.skip("Anthropic API key not available")
        
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': os.getenv("ANTHROPIC_API_KEY"),
            'ANTHROPIC_API_URL': os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com"),
            'LLM_PROVIDER': 'anthropic'
        }):
            client = HierarchicalClient("claude-3-haiku-20240307", "claude-3-haiku-20240307")
            
            # Mock tool session
            mock_tool = MockTool("calculator", "A simple calculator tool", {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "description": "The operation to perform"},
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                }
            })
            mock_session = MockToolSession([mock_tool])
            client.sessions = {"calculator": mock_session}
            
            # Test with a query that might use tools
            result = await client.process_query(
                "I need to multiply 6 and 9. Please use the calculator tool if available.",
                "test.txt"
            )
            
            # Should get a response (may or may not use tools)
            assert result is not None
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_openai(self):
        """Test error handling with OpenAI backend."""
        if not self.has_openai_key:
            pytest.skip("OpenAI API key not available")
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
            'LLM_PROVIDER': 'openai'
        }):
            client = HierarchicalClient("gpt-3.5-turbo", "gpt-3.5-turbo")
            
            # Test with a query that might cause issues
            result = await client.process_query(
                "Please give me an invalid JSON response: {invalid json",
                "test.txt"
            )
            
            # Should handle the error gracefully
            assert result is not None
            # Should contain error information
            assert "error" in result.lower() or "invalid" in result.lower()
    
    @pytest.mark.asyncio
    async def test_error_handling_anthropic(self):
        """Test error handling with Anthropic backend."""
        if not self.has_anthropic_key:
            pytest.skip("Anthropic API key not available")
        
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': os.getenv("ANTHROPIC_API_KEY"),
            'ANTHROPIC_API_URL': os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com"),
            'LLM_PROVIDER': 'anthropic'
        }):
            client = HierarchicalClient("claude-3-haiku-20240307", "claude-3-haiku-20240307")
            
            # Test with a query that might cause issues
            result = await client.process_query(
                "Please give me an invalid JSON response: {invalid json",
                "test.txt"
            )
            
            # Should handle the error gracefully
            assert result is not None
            # Should contain error information
            assert "error" in result.lower() or "invalid" in result.lower()
    
    @pytest.mark.asyncio
    async def test_mixed_backend_tool_usage(self):
        """Test tool usage with mixed backends."""
        if not self.has_openai_key or not self.has_anthropic_key:
            pytest.skip("Required API keys not available")
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
            'ANTHROPIC_API_KEY': os.getenv("ANTHROPIC_API_KEY"),
            'ANTHROPIC_API_URL': os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com")
        }):
            # Mock get_default_backend to return specific backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_get_backend.side_effect = lambda model: (
                    OpenAIBackend(model) if model == "gpt-3.5-turbo" 
                    else AnthropicBackend(model)
                )
                client = HierarchicalClient("gpt-3.5-turbo", "claude-3-haiku-20240307")
                
                # Mock tool session
                mock_tool = MockTool("math_helper", "A math helper tool", {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                    }
                })
                mock_session = MockToolSession([mock_tool])
                client.sessions = {"math_helper": mock_session}
                
                # Test with a query that might use tools
                result = await client.process_query(
                    "I need to calculate 12 * 13. Please use the math_helper tool if available.",
                    "test.txt"
                )
                
                # Should get a response
                assert result is not None
                assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_backend_consistency_verification(self):
        """Test that backends work consistently across different scenarios."""
        if not self.has_openai_key or not self.has_anthropic_key:
            pytest.skip("Required API keys not available")
        
        # Test OpenAI backend
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
            'LLM_PROVIDER': 'openai'
        }):
            openai_client = HierarchicalClient("gpt-3.5-turbo", "gpt-3.5-turbo")
            openai_result = await openai_client.process_query(
                "What is the capital of France?",
                "test.txt"
            )
            assert openai_result is not None
            assert "paris" in openai_result.lower()
        
        # Test Anthropic backend
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': os.getenv("ANTHROPIC_API_KEY"),
            'ANTHROPIC_API_URL': os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com"),
            'LLM_PROVIDER': 'anthropic'
        }):
            anthropic_client = HierarchicalClient("claude-3-haiku-20240307", "claude-3-haiku-20240307")
            anthropic_result = await anthropic_client.process_query(
                "What is the capital of Germany?",
                "test.txt"
            )
            assert anthropic_result is not None
            assert "berlin" in anthropic_result.lower()
    
    @pytest.mark.asyncio
    async def test_performance_comparison(self):
        """Test performance characteristics of different backends."""
        if not self.has_openai_key or not self.has_anthropic_key:
            pytest.skip("Required API keys not available")
        
        import time
        
        # Test OpenAI performance
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
            'LLM_PROVIDER': 'openai'
        }):
            openai_client = HierarchicalClient("gpt-3.5-turbo", "gpt-3.5-turbo")
            start_time = time.time()
            openai_result = await openai_client.process_query(
                "What is 2 + 2?",
                "test.txt"
            )
            openai_time = time.time() - start_time
            assert openai_result is not None
        
        # Test Anthropic performance
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': os.getenv("ANTHROPIC_API_KEY"),
            'ANTHROPIC_API_URL': os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com"),
            'LLM_PROVIDER': 'anthropic'
        }):
            anthropic_client = HierarchicalClient("claude-3-haiku-20240307", "claude-3-haiku-20240307")
            start_time = time.time()
            anthropic_result = await anthropic_client.process_query(
                "What is 2 + 2?",
                "test.txt"
            )
            anthropic_time = time.time() - start_time
            assert anthropic_result is not None
        
        # Both should complete in reasonable time (less than 30 seconds)
        assert openai_time < 30
        assert anthropic_time < 30


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])

