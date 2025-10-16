"""
Test suite for HierarchicalClient with different backend combinations.

This test suite ensures that HierarchicalClient works correctly with
various combinations of AzureOpenAI and Anthropic backends for both
meta_model and exec_model.
"""

import asyncio
import json
import os
import pytest
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons.llm import AnthropicBackend, AzureOpenAIBackend, OpenAIBackend, get_default_backend
from client.agent import HierarchicalClient


class MockToolSession:
    """Mock tool session for testing."""
    
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


class MockAnthropicResponse:
    """Mock Anthropic API response for testing."""
    
    def __init__(self, content_text: str = "Test response from Claude"):
        self.content = [MockContentBlock("text", content_text)]


class MockContentBlock:
    """Mock content block for Anthropic responses."""
    
    def __init__(self, block_type: str, text: str = None, tool_use_data: dict = None):
        self.type = block_type
        self.text = text
        if tool_use_data:
            self.id = tool_use_data.get("id")
            self.name = tool_use_data.get("name")
            self.input = tool_use_data.get("input")


class TestHierarchicalClientBackends:
    """Test HierarchicalClient with different backend combinations."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Mock environment variables for different backends
        self.azure_env = {
            'AZURE_OPENAI_KEY': 'test-azure-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test-azure.openai.azure.com/',
            'OPENAI_API_VERSION': '2024-02-15-preview',
            'LLM_PROVIDER': 'azure'
        }
        self.anthropic_env = {
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'ANTHROPIC_API_URL': 'https://api.anthropic.com',
            'LLM_PROVIDER': 'anthropic'
        }
        self.openai_env = {
            'OPENAI_API_KEY': 'test-openai-key',
            'LLM_PROVIDER': 'openai'
        }
    
    def test_azure_meta_azure_exec(self):
        """Test HierarchicalClient with AzureOpenAI for both meta and exec models."""
        with patch.dict(os.environ, self.azure_env):
            client = HierarchicalClient("gpt-4", "gpt-4")
            
            # Verify both backends are AzureOpenAI
            assert isinstance(client.meta_llm, AzureOpenAIBackend)
            assert isinstance(client.exec_llm, AzureOpenAIBackend)
            assert client.meta_llm.model == "gpt-4"
            assert client.exec_llm.model == "gpt-4"
    
    def test_azure_meta_anthropic_exec(self):
        """Test HierarchicalClient with AzureOpenAI meta and Anthropic exec."""
        # Use specific provider for each model
        with patch.dict(os.environ, {
            'AZURE_OPENAI_KEY': 'test-azure-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test-azure.openai.azure.com/',
            'OPENAI_API_VERSION': '2024-02-15-preview',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'ANTHROPIC_API_URL': 'https://api.anthropic.com'
        }):
            # Mock get_default_backend to return specific backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_get_backend.side_effect = lambda model: (
                    AzureOpenAIBackend(model) if model == "gpt-4" 
                    else AnthropicBackend(model)
                )
                client = HierarchicalClient("gpt-4", "claude-3-sonnet-20240229")
                
                # Verify backend types
                assert isinstance(client.meta_llm, AzureOpenAIBackend)
                assert isinstance(client.exec_llm, AnthropicBackend)
                assert client.meta_llm.model == "gpt-4"
                assert client.exec_llm.model == "claude-3-sonnet-20240229"
    
    def test_anthropic_meta_azure_exec(self):
        """Test HierarchicalClient with Anthropic meta and AzureOpenAI exec."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_KEY': 'test-azure-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test-azure.openai.azure.com/',
            'OPENAI_API_VERSION': '2024-02-15-preview',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'ANTHROPIC_API_URL': 'https://api.anthropic.com'
        }):
            # Mock get_default_backend to return specific backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_get_backend.side_effect = lambda model: (
                    AnthropicBackend(model) if model == "claude-3-sonnet-20240229" 
                    else AzureOpenAIBackend(model)
                )
                client = HierarchicalClient("claude-3-sonnet-20240229", "gpt-4")
                
                # Verify backend types
                assert isinstance(client.meta_llm, AnthropicBackend)
                assert isinstance(client.exec_llm, AzureOpenAIBackend)
                assert client.meta_llm.model == "claude-3-sonnet-20240229"
                assert client.exec_llm.model == "gpt-4"
    
    def test_anthropic_meta_anthropic_exec(self):
        """Test HierarchicalClient with Anthropic for both meta and exec models."""
        with patch.dict(os.environ, self.anthropic_env):
            client = HierarchicalClient("claude-3-sonnet-20240229", "claude-3-haiku-20240307")
            
            # Verify both backends are Anthropic
            assert isinstance(client.meta_llm, AnthropicBackend)
            assert isinstance(client.exec_llm, AnthropicBackend)
            assert client.meta_llm.model == "claude-3-sonnet-20240229"
            assert client.exec_llm.model == "claude-3-haiku-20240307"
    
    def test_openai_meta_anthropic_exec(self):
        """Test HierarchicalClient with OpenAI meta and Anthropic exec."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'ANTHROPIC_API_URL': 'https://api.anthropic.com'
        }):
            # Mock get_default_backend to return specific backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_get_backend.side_effect = lambda model: (
                    OpenAIBackend(model) if model == "gpt-4" 
                    else AnthropicBackend(model)
                )
                client = HierarchicalClient("gpt-4", "claude-3-sonnet-20240229")
                
                # Verify backend types
                assert isinstance(client.meta_llm, OpenAIBackend)
                assert isinstance(client.exec_llm, AnthropicBackend)
                assert client.meta_llm.model == "gpt-4"
                assert client.exec_llm.model == "claude-3-sonnet-20240229"
    
    def test_anthropic_meta_openai_exec(self):
        """Test HierarchicalClient with Anthropic meta and OpenAI exec."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'ANTHROPIC_API_URL': 'https://api.anthropic.com'
        }):
            # Mock get_default_backend to return specific backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_get_backend.side_effect = lambda model: (
                    AnthropicBackend(model) if model == "claude-3-sonnet-20240229" 
                    else OpenAIBackend(model)
                )
                client = HierarchicalClient("claude-3-sonnet-20240229", "gpt-4")
                
                # Verify backend types
                assert isinstance(client.meta_llm, AnthropicBackend)
                assert isinstance(client.exec_llm, OpenAIBackend)
                assert client.meta_llm.model == "claude-3-sonnet-20240229"
                assert client.exec_llm.model == "gpt-4"
    
    def test_missing_meta_model_raises_error(self):
        """Test that missing meta_model raises SystemExit."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit, match="Missing meta_model"):
                HierarchicalClient(None, "gpt-4")
    
    def test_missing_exec_model_raises_error(self):
        """Test that missing exec_model raises SystemExit."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit, match="Missing exec_model"):
                HierarchicalClient("gpt-4", None)
    
    def test_environment_variable_fallback(self):
        """Test that environment variables are used as fallback."""
        with patch.dict(os.environ, {
            'META_PLANNER_MODEL': 'claude-3-sonnet-20240229',
            'EXEC_MODEL': 'gpt-4',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'ANTHROPIC_API_URL': 'https://api.anthropic.com',
            'OPENAI_API_KEY': 'test-openai-key'
        }):
            # Mock get_default_backend to return specific backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_get_backend.side_effect = lambda model: (
                    AnthropicBackend(model) if model == "claude-3-sonnet-20240229" 
                    else OpenAIBackend(model)
                )
                client = HierarchicalClient(None, None)
                
                # Should use environment variables
                assert isinstance(client.meta_llm, AnthropicBackend)
                assert isinstance(client.exec_llm, OpenAIBackend)
                assert client.meta_llm.model == "claude-3-sonnet-20240229"
                assert client.exec_llm.model == "gpt-4"


class TestHierarchicalClientProcessQuery:
    """Test HierarchicalClient.process_query with different backend combinations."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.azure_env = {
            'AZURE_OPENAI_KEY': 'test-azure-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test-azure.openai.azure.com/',
            'LLM_PROVIDER': 'azure'
        }
        self.anthropic_env = {
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'ANTHROPIC_API_URL': 'https://api.anthropic.com',
            'LLM_PROVIDER': 'anthropic'
        }
    
    @pytest.mark.asyncio
    async def test_process_query_azure_meta_azure_exec(self):
        """Test process_query with AzureOpenAI for both models."""
        with patch.dict(os.environ, self.azure_env):
            # Mock get_default_backend to return mock backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_backend = MagicMock()
                mock_backend.model = "gpt-4"
                mock_backend.chat = AsyncMock()
                mock_get_backend.return_value = mock_backend
                client = HierarchicalClient("gpt-4", "gpt-4")
            
            # Configure the mock backends
            meta_responses = [
                {"content": '{"plan": [{"id": 1, "description": "Test task"}]}'},
                {"content": "FINAL ANSWER: Task completed successfully"},
                {"content": "FINAL ANSWER: Task completed successfully"}  # Extra response for safety
            ]
            exec_response = {"content": "Task completed successfully"}
            
            client.meta_llm.chat.side_effect = meta_responses
            client.exec_llm.chat.return_value = exec_response
            
            result = await client.process_query("Test question", "test.txt")
            
            # Verify calls were made
            assert client.meta_llm.chat.called
            assert client.exec_llm.chat.called
            
            # Verify result is the final answer
            assert result == "Task completed successfully"
    
    @pytest.mark.asyncio
    async def test_process_query_azure_meta_anthropic_exec(self):
        """Test process_query with AzureOpenAI meta and Anthropic exec."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_KEY': 'test-azure-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test-azure.openai.azure.com/',
            'OPENAI_API_VERSION': '2024-02-15-preview',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'ANTHROPIC_API_URL': 'https://api.anthropic.com'
        }):
            # Mock get_default_backend to return specific backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_get_backend.side_effect = lambda model: (
                    AzureOpenAIBackend(model) if model == "gpt-4" 
                    else AnthropicBackend(model)
                )
                client = HierarchicalClient("gpt-4", "claude-3-sonnet-20240229")
                
                # Mock the chat responses - first cycle gives plan, second gives final answer
                meta_responses = [
                    {"content": '{"plan": [{"id": 1, "description": "Test task"}]}'},
                    {"content": "FINAL ANSWER: Task completed with Claude"}
                ]
                exec_response = {"content": "Task completed with Claude"}
                
                with patch.object(client.meta_llm, 'chat', new_callable=AsyncMock) as mock_meta_chat, \
                     patch.object(client.exec_llm, 'chat', new_callable=AsyncMock) as mock_exec_chat:
                    
                    mock_meta_chat.side_effect = meta_responses
                    mock_exec_chat.return_value = exec_response
                    
                    result = await client.process_query("Test question", "test.txt")
                    
                    # Verify calls were made
                    assert mock_meta_chat.called
                    assert mock_exec_chat.called
                    
                    # Verify result is the final answer
                    assert result == "Task completed with Claude"
    
    @pytest.mark.asyncio
    async def test_process_query_anthropic_meta_azure_exec(self):
        """Test process_query with Anthropic meta and AzureOpenAI exec."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_KEY': 'test-azure-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test-azure.openai.azure.com/',
            'OPENAI_API_VERSION': '2024-02-15-preview',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'ANTHROPIC_API_URL': 'https://api.anthropic.com'
        }):
            # Mock get_default_backend to return specific backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_get_backend.side_effect = lambda model: (
                    AnthropicBackend(model) if model == "claude-3-sonnet-20240229" 
                    else AzureOpenAIBackend(model)
                )
                client = HierarchicalClient("claude-3-sonnet-20240229", "gpt-4")
                
                # Mock the chat responses - first cycle gives plan, second gives final answer
                meta_responses = [
                    {"content": '{"plan": [{"id": 1, "description": "Test task"}]}'},
                    {"content": "FINAL ANSWER: Task completed with Azure"}
                ]
                exec_response = {"content": "Task completed with Azure"}
                
                with patch.object(client.meta_llm, 'chat', new_callable=AsyncMock) as mock_meta_chat, \
                     patch.object(client.exec_llm, 'chat', new_callable=AsyncMock) as mock_exec_chat:
                    
                    mock_meta_chat.side_effect = meta_responses
                    mock_exec_chat.return_value = exec_response
                    
                    result = await client.process_query("Test question", "test.txt")
                    
                    # Verify calls were made
                    assert mock_meta_chat.called
                    assert mock_exec_chat.called
                    
                    # Verify result is the final answer
                    assert result == "Task completed with Azure"
    
    @pytest.mark.asyncio
    async def test_process_query_anthropic_meta_anthropic_exec(self):
        """Test process_query with Anthropic for both models."""
        with patch.dict(os.environ, self.anthropic_env):
            client = HierarchicalClient("claude-3-sonnet-20240229", "claude-3-haiku-20240307")
            
            # Mock the chat responses - first cycle gives plan, second gives final answer
            meta_responses = [
                {"content": '{"plan": [{"id": 1, "description": "Test task"}]}'},
                {"content": "FINAL ANSWER: Task completed with both Claude models"}
            ]
            exec_response = {"content": "Task completed with both Claude models"}
            
            with patch.object(client.meta_llm, 'chat', new_callable=AsyncMock) as mock_meta_chat, \
                 patch.object(client.exec_llm, 'chat', new_callable=AsyncMock) as mock_exec_chat:
                
                mock_meta_chat.side_effect = meta_responses
                mock_exec_chat.return_value = exec_response
                
                result = await client.process_query("Test question", "test.txt")
                
                # Verify calls were made
                assert mock_meta_chat.called
                assert mock_exec_chat.called
                
                # Verify result is the final answer
                assert result == "Task completed with both Claude models"
    
    @pytest.mark.asyncio
    async def test_process_query_final_answer_azure_meta(self):
        """Test process_query with final answer from AzureOpenAI meta model."""
        with patch.dict(os.environ, self.azure_env):
            # Mock get_default_backend to return mock backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_backend = MagicMock()
                mock_backend.model = "gpt-4"
                mock_backend.chat = AsyncMock()
                mock_get_backend.return_value = mock_backend
                client = HierarchicalClient("gpt-4", "gpt-4")
            
            # Mock final answer response
            final_response = {"content": "FINAL ANSWER: This is the final answer"}
            
            with patch.object(client.meta_llm, 'chat', new_callable=AsyncMock) as mock_meta_chat:
                mock_meta_chat.return_value = final_response
                
                result = await client.process_query("Test question", "test.txt")
                
                # Should return the final answer without the prefix
                assert result == "This is the final answer"
    
    @pytest.mark.asyncio
    async def test_process_query_final_answer_anthropic_meta(self):
        """Test process_query with final answer from Anthropic meta model."""
        with patch.dict(os.environ, self.anthropic_env):
            client = HierarchicalClient("claude-3-sonnet-20240229", "claude-3-haiku-20240307")
            
            # Mock final answer response
            final_response = {"content": "FINAL ANSWER: This is the final answer from Claude"}
            
            with patch.object(client.meta_llm, 'chat', new_callable=AsyncMock) as mock_meta_chat:
                mock_meta_chat.return_value = final_response
                
                result = await client.process_query("Test question", "test.txt")
                
                # Should return the final answer without the prefix
                assert result == "This is the final answer from Claude"
    
    @pytest.mark.asyncio
    async def test_process_query_with_tools_azure_exec(self):
        """Test process_query with tools using AzureOpenAI exec model."""
        with patch.dict(os.environ, self.azure_env):
            # Mock get_default_backend to return mock backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_backend = MagicMock()
                mock_backend.model = "gpt-4"
                mock_backend.chat = AsyncMock()
                mock_get_backend.return_value = mock_backend
                client = HierarchicalClient("gpt-4", "gpt-4")
            
            # Mock tool session
            mock_tool = MockTool("test_tool", "A test tool", {"type": "object"})
            mock_session = MockToolSession([mock_tool])
            client.sessions = {"test_tool": mock_session}
            
            # Configure the mock backends
            meta_responses = [
                {"content": '{"plan": [{"id": 1, "description": "Use test_tool"}]}'},
                {"content": "FINAL ANSWER: Tool executed successfully"},
                {"content": "FINAL ANSWER: Tool executed successfully"}  # Extra response for safety
            ]
            exec_response = {"content": "Tool executed successfully"}
            
            client.meta_llm.chat.side_effect = meta_responses
            client.exec_llm.chat.return_value = exec_response
            
            result = await client.process_query("Test question", "test.txt")
            
            # Verify exec_llm.chat was called with sessions
            assert client.exec_llm.chat.called
            # Check that the second argument (sessions) was passed
            call_args = client.exec_llm.chat.call_args
            if call_args and len(call_args[0]) > 1:
                assert list(call_args[0][1]) == list(client.sessions.values())
            
            # Verify result is the final answer
            assert result == "Tool executed successfully"
    
    @pytest.mark.asyncio
    async def test_process_query_with_tools_anthropic_exec(self):
        """Test process_query with tools using Anthropic exec model."""
        with patch.dict(os.environ, self.anthropic_env):
            client = HierarchicalClient("claude-3-sonnet-20240229", "claude-3-haiku-20240307")
            
            # Mock tool session
            mock_tool = MockTool("test_tool", "A test tool", {"type": "object"})
            mock_session = MockToolSession([mock_tool])
            client.sessions = {"test_tool": mock_session}
            
            # Mock responses - first cycle gives plan, second gives final answer
            meta_responses = [
                {"content": '{"plan": [{"id": 1, "description": "Use test_tool"}]}'},
                {"content": "FINAL ANSWER: Tool executed with Claude"}
            ]
            exec_response = {"content": "Tool executed with Claude"}
            
            with patch.object(client.meta_llm, 'chat', new_callable=AsyncMock) as mock_meta_chat, \
                 patch.object(client.exec_llm, 'chat', new_callable=AsyncMock) as mock_exec_chat:
                
                mock_meta_chat.side_effect = meta_responses
                mock_exec_chat.return_value = exec_response
                
                result = await client.process_query("Test question", "test.txt")
                
                # Verify exec_llm.chat was called with sessions
                exec_call_args = mock_exec_chat.call_args
                assert list(exec_call_args[0][1]) == list(client.sessions.values())
                
                # Verify result is the final answer
                assert result == "Tool executed with Claude"
    
    @pytest.mark.asyncio
    async def test_process_query_planner_error_handling(self):
        """Test process_query error handling with different backends."""
        with patch.dict(os.environ, self.azure_env):
            # Mock get_default_backend to return mock backends
            with patch('client.agent.get_default_backend') as mock_get_backend:
                mock_backend = MagicMock()
                mock_backend.model = "gpt-4"
                mock_backend.chat = AsyncMock()
                mock_get_backend.return_value = mock_backend
                client = HierarchicalClient("gpt-4", "gpt-4")
            
            # Mock invalid JSON response
            invalid_response = {"content": "Invalid JSON response"}
            
            with patch.object(client.meta_llm, 'chat', new_callable=AsyncMock) as mock_meta_chat:
                mock_meta_chat.return_value = invalid_response
                
                result = await client.process_query("Test question", "test.txt")
                
                # Should return error message
                assert "[planner error]" in result
                assert "Invalid JSON response" in result
    
    @pytest.mark.asyncio
    async def test_process_query_max_cycles_reached(self):
        """Test process_query when max cycles are reached."""
        with patch.dict(os.environ, self.anthropic_env):
            client = HierarchicalClient("claude-3-sonnet-20240229", "claude-3-haiku-20240307")
            
            # Mock response that never gives final answer
            cycle_response = {"content": '{"plan": [{"id": 1, "description": "Test task"}]}'}
            
            with patch.object(client.meta_llm, 'chat', new_callable=AsyncMock) as mock_meta_chat, \
                 patch.object(client.exec_llm, 'chat', new_callable=AsyncMock) as mock_exec_chat:
                
                mock_meta_chat.return_value = cycle_response
                mock_exec_chat.return_value = {"content": "Task result"}
                
                result = await client.process_query("Test question", "test.txt")
                
                # Should have made MAX_CYCLES calls to meta_llm
                assert mock_meta_chat.call_count == client.MAX_CYCLES
                # Should have made calls to exec_llm for each task in each cycle
                assert mock_exec_chat.call_count == client.MAX_CYCLES


class TestHierarchicalClientBackendCompatibility:
    """Test compatibility between different backend combinations."""
    
    def test_backend_interface_consistency(self):
        """Test that all backend combinations have consistent interfaces."""
        backends = [
            (AzureOpenAIBackend, "gpt-4"),
            (AnthropicBackend, "claude-3-sonnet-20240229"),
        ]
        
        for backend_class, model in backends:
            with patch.dict(os.environ, {
                'AZURE_OPENAI_KEY': 'test-key',
                'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
                'OPENAI_API_VERSION': '2024-02-15-preview',
                'ANTHROPIC_API_KEY': 'test-key'
            }):
                if backend_class == AzureOpenAIBackend:
                    backend = AzureOpenAIBackend(model)
                else:
                    backend = AnthropicBackend(model)
                
                # All backends should have these methods
                assert hasattr(backend, 'chat')
                assert hasattr(backend, '_format_tools_schema')
                assert callable(backend.chat)
                assert callable(backend._format_tools_schema)
    
    def test_tool_schema_formatting_compatibility(self):
        """Test that tool schema formatting works across different backends."""
        mock_tool = MockTool("test_tool", "A test tool", {"type": "object"})
        mock_session = MockToolSession([mock_tool])
        
        with patch.dict(os.environ, {
            'AZURE_OPENAI_KEY': 'test-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'OPENAI_API_VERSION': '2024-02-15-preview',
            'ANTHROPIC_API_KEY': 'test-key'
        }):
            azure_backend = AzureOpenAIBackend("gpt-4")
            anthropic_backend = AnthropicBackend("claude-3-sonnet-20240229")
            
            # Both should be able to format tools
            assert callable(azure_backend._format_tools_schema)
            assert callable(anthropic_backend._format_tools_schema)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
