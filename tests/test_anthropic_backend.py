"""
Test suite for Anthropic backend implementation.

This test suite ensures that the Anthropic backend works correctly
and doesn't break existing functionality.
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Import the backend classes
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons.llm import AnthropicBackend, OpenAIBackend, get_default_backend


class MockAnthropicResponse:
    """Mock Anthropic API response for testing."""
    
    def __init__(self, content: List[Dict[str, Any]]):
        self.content = content


class MockContentBlock:
    """Mock content block for Anthropic responses."""
    
    def __init__(self, block_type: str, text: str = None, tool_use_data: Dict[str, Any] = None):
        self.type = block_type
        self.text = text
        if tool_use_data:
            self.id = tool_use_data.get("id")
            self.name = tool_use_data.get("name")
            self.input = tool_use_data.get("input")


class MockTool:
    """Mock tool object with attributes."""
    
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class MockToolSession:
    """Mock tool session for testing tool integration."""
    
    def __init__(self, tools: List[Dict[str, Any]]):
        self.tools = [MockTool(tool["name"], tool["description"], tool["inputSchema"]) for tool in tools]
    
    async def list_tools(self):
        """Mock list_tools method."""
        return MagicMock(tools=self.tools)


class TestAnthropicBackend:
    """Test cases for AnthropicBackend class."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Mock environment variables
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test-key',
            'ANTHROPIC_API_URL': 'https://api.anthropic.com'
        }):
            self.backend = AnthropicBackend("claude-3-sonnet-20240229")
    
    def test_initialization(self):
        """Test that AnthropicBackend initializes correctly."""
        assert self.backend.model == "claude-3-sonnet-20240229"
        assert self.backend.client is not None
    
    @pytest.mark.asyncio
    async def test_chat_basic_text_response(self):
        """Test basic chat functionality with text response."""
        # Mock the Anthropic client response
        mock_response = MockAnthropicResponse([
            MockContentBlock("text", "Hello! How can I help you today?")
        ])
        
        with patch.object(self.backend.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            result = await self.backend.chat(messages)
            
            assert result["content"] == "Hello! How can I help you today?"
            assert result["tool_calls"] is None
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self):
        """Test chat functionality with tool calls."""
        # Mock tool use content block
        tool_use_data = {
            "id": "tool_123",
            "name": "search_web",
            "input": '{"query": "test search"}'
        }
        
        mock_response = MockAnthropicResponse([
            MockContentBlock("tool_use", tool_use_data=tool_use_data)
        ])
        
        with patch.object(self.backend.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            messages = [{"role": "user", "content": "Search for something"}]
            result = await self.backend.chat(messages)
            
            assert result["content"] is None
            assert result["tool_calls"] is not None
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["id"] == "tool_123"
            assert result["tool_calls"][0]["type"] == "function"
            assert result["tool_calls"][0]["function"]["name"] == "search_web"
            assert result["tool_calls"][0]["function"]["arguments"] == '{"query": "test search"}'
    
    @pytest.mark.asyncio
    async def test_chat_mixed_content(self):
        """Test chat with both text and tool calls."""
        mock_response = MockAnthropicResponse([
            MockContentBlock("text", "I'll search for that information."),
            MockContentBlock("tool_use", tool_use_data={
                "id": "tool_456",
                "name": "get_weather",
                "input": '{"location": "New York"}'
            })
        ])
        
        with patch.object(self.backend.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            messages = [{"role": "user", "content": "What's the weather like?"}]
            result = await self.backend.chat(messages)
            
            assert result["content"] == "I'll search for that information."
            assert result["tool_calls"] is not None
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    
    @pytest.mark.asyncio
    async def test_format_tools_schema(self):
        """Test tool schema formatting."""
        # Create mock tool sessions
        mock_tools = [
            {
                "name": "search_web",
                "description": "Search the web for information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_weather",
                "description": "Get weather information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "Location to get weather for"}
                    },
                    "required": ["location"]
                }
            }
        ]
        
        session1 = MockToolSession([mock_tools[0]])
        session2 = MockToolSession([mock_tools[1]])
        sessions = [session1, session2]
        
        result = await self.backend._format_tools_schema(sessions)
        
        assert len(result) == 2
        assert result[0]["name"] == "search_web"
        assert result[0]["description"] == "Search the web for information"
        assert result[0]["input_schema"] == mock_tools[0]["inputSchema"]
        assert result[1]["name"] == "get_weather"
    
    @pytest.mark.asyncio
    async def test_format_tools_schema_caching(self):
        """Test that tool schema formatting caches results."""
        mock_tools = [{
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"type": "object"}
        }]
        
        session = MockToolSession(mock_tools)
        sessions = [session, session]  # Same session twice
        
        # Mock list_tools to track calls
        with patch.object(session, 'list_tools', new_callable=AsyncMock) as mock_list_tools:
            mock_list_tools.return_value = MagicMock(tools=session.tools)
            
            result = await self.backend._format_tools_schema(sessions)
            
            # Should only call list_tools once due to caching
            assert mock_list_tools.call_count == 1
            assert len(result) == 2  # But should return 2 tools (one from each session)
    
    @pytest.mark.asyncio
    async def test_chat_with_tools(self):
        """Test chat functionality with tools enabled."""
        mock_tools = [{
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"type": "object"}
        }]
        
        session = MockToolSession(mock_tools)
        
        # Mock the response
        mock_response = MockAnthropicResponse([
            MockContentBlock("text", "I'll use the test tool.")
        ])
        
        with patch.object(self.backend.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            messages = [{"role": "user", "content": "Test with tools"}]
            result = await self.backend.chat(messages, tools_session_values=[session])
            
            # Verify the API was called with tools
            call_args = mock_create.call_args
            assert "tools" in call_args.kwargs
            assert len(call_args.kwargs["tools"]) == 1
            assert call_args.kwargs["tools"][0]["name"] == "test_tool"
            assert call_args.kwargs["tool_choice"] == "auto"
    
    @pytest.mark.asyncio
    async def test_chat_without_tools(self):
        """Test chat functionality without tools."""
        mock_response = MockAnthropicResponse([
            MockContentBlock("text", "Simple response without tools.")
        ])
        
        with patch.object(self.backend.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            messages = [{"role": "user", "content": "Simple question"}]
            result = await self.backend.chat(messages)
            
            # Verify the API was called without tools
            call_args = mock_create.call_args
            assert "tools" not in call_args.kwargs
            assert "tool_choice" not in call_args.kwargs
    
    @pytest.mark.asyncio
    async def test_chat_parameters(self):
        """Test that chat parameters are passed correctly."""
        mock_response = MockAnthropicResponse([
            MockContentBlock("text", "Response with custom parameters.")
        ])
        
        with patch.object(self.backend.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            messages = [{"role": "user", "content": "Test parameters"}]
            await self.backend.chat(
                messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Verify parameters were passed
            call_args = mock_create.call_args
            assert call_args.kwargs["max_completion_tokens"] == 1000
            # Note: Anthropic doesn't use temperature in the same way as OpenAI
    
    @pytest.mark.asyncio
    async def test_prepare_response_empty_content(self):
        """Test prepare_response with empty content."""
        mock_response = MockAnthropicResponse([])
        result = self.backend.prepare_response(mock_response)
        
        assert result["content"] is None
        assert result["tool_calls"] is None
    
    @pytest.mark.asyncio
    async def test_prepare_response_text_only(self):
        """Test prepare_response with text only."""
        mock_response = MockAnthropicResponse([
            MockContentBlock("text", "First part"),
            MockContentBlock("text", "Second part")
        ])
        result = self.backend.prepare_response(mock_response)
        
        assert result["content"] == "First partSecond part"
        assert result["tool_calls"] is None
    
    @pytest.mark.asyncio
    async def test_prepare_response_tools_only(self):
        """Test prepare_response with tools only."""
        mock_response = MockAnthropicResponse([
            MockContentBlock("tool_use", tool_use_data={
                "id": "tool_1",
                "name": "tool_one",
                "input": '{"param": "value"}'
            }),
            MockContentBlock("tool_use", tool_use_data={
                "id": "tool_2",
                "name": "tool_two",
                "input": '{"param2": "value2"}'
            })
        ])
        result = self.backend.prepare_response(mock_response)
        
        assert result["content"] is None
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["function"]["name"] == "tool_one"
        assert result["tool_calls"][1]["function"]["name"] == "tool_two"
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test that retry mechanism works on failures."""
        with patch.object(self.backend.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            # First two calls fail, third succeeds
            mock_create.side_effect = [
                Exception("API Error 1"),
                Exception("API Error 2"),
                MockAnthropicResponse([MockContentBlock("text", "Success after retry")])
            ]
            
            messages = [{"role": "user", "content": "Test retry"}]
            result = await self.backend.chat(messages)
            
            assert result["content"] == "Success after retry"
            assert mock_create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        with patch.object(self.backend.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Persistent API Error")
            
            messages = [{"role": "user", "content": "Test max retries"}]
            
            with pytest.raises(Exception, match="Persistent API Error"):
                await self.backend.chat(messages)
            
            # Should have tried 3 times (initial + 2 retries)
            assert mock_create.call_count == 3


class TestBackendIntegration:
    """Test integration between different backends."""
    
    def test_backend_consistency(self):
        """Test that all backends have consistent interfaces."""
        # Test that AnthropicBackend has the same interface as OpenAIBackend
        anthropic_backend = AnthropicBackend("claude-3-sonnet-20240229")
        
        # Mock OpenAI backend to avoid API key requirement
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            openai_backend = OpenAIBackend("gpt-4")
        
        # Both should have these methods
        assert hasattr(anthropic_backend, 'chat')
        assert hasattr(anthropic_backend, '_format_tools_schema')
        assert hasattr(openai_backend, 'chat')
        assert hasattr(openai_backend, '_format_tools_schema')
        
    
    @pytest.mark.asyncio
    async def test_tool_schema_formatting_consistency(self):
        """Test that tool schema formatting is consistent between backends."""
        mock_tools = [{
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"type": "object"}
        }]
        
        session = MockToolSession(mock_tools)
        
        # Test Anthropic formatting
        anthropic_backend = AnthropicBackend("claude-3-sonnet-20240229")
        anthropic_result = await anthropic_backend._format_tools_schema([session])
        
        # Test OpenAI formatting with mocked environment
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            openai_backend = OpenAIBackend("gpt-4")
            openai_result = await openai_backend._format_tools_schema([session])
        
        # Both should return the same tool name and description
        assert anthropic_result[0]["name"] == openai_result[0]["function"]["name"]
        assert anthropic_result[0]["description"] == openai_result[0]["function"]["description"]
        
        # Schema structure should be similar (though Anthropic uses input_schema vs parameters)
        assert anthropic_result[0]["input_schema"] == openai_result[0]["function"]["parameters"]


class TestEnvironmentConfiguration:
    """Test environment configuration and backend selection."""
    
    def test_anthropic_environment_variables(self):
        """Test that Anthropic backend uses correct environment variables."""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'ANTHROPIC_API_URL': 'https://custom-anthropic-endpoint.com'
        }):
            backend = AnthropicBackend("claude-3-sonnet-20240229")
            
            # Verify the client was initialized with correct values
            assert backend.client.api_key == 'test-anthropic-key'
            assert backend.client.base_url == 'https://custom-anthropic-endpoint.com'
    
    def test_missing_anthropic_credentials(self):
        """Test behavior when Anthropic credentials are missing."""
        with patch.dict(os.environ, {}, clear=True):
            # Should still initialize but client will use None values
            backend = AnthropicBackend("claude-3-sonnet-20240229")
            assert backend.client.api_key is None


class TestDefaultBackendSelection:
    """Test the get_default_backend function with Anthropic support."""
    
    def test_get_anthropic_backend_with_provider(self):
        """Test getting Anthropic backend when LLM_PROVIDER is set to anthropic."""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'anthropic',
            'ANTHROPIC_API_KEY': 'test-key'
        }):
            backend = get_default_backend("claude-3-sonnet-20240229")
            assert isinstance(backend, AnthropicBackend)
            assert backend.model == "claude-3-sonnet-20240229"
    
    def test_get_anthropic_backend_fallback(self):
        """Test getting Anthropic backend as fallback when no provider specified."""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test-key'
        }, clear=True):
            backend = get_default_backend("claude-3-sonnet-20240229")
            assert isinstance(backend, AnthropicBackend)
            assert backend.model == "claude-3-sonnet-20240229"
    
    def test_get_anthropic_backend_priority(self):
        """Test that Anthropic backend has correct priority in fallback chain."""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test-key',
            'OPENAI_API_KEY': 'test-openai-key'
        }, clear=True):
            # Should prefer OpenAI over Anthropic in fallback
            backend = get_default_backend("gpt-4")
            assert isinstance(backend, OpenAIBackend)
    
    def test_no_valid_backend_raises_error(self):
        """Test that RuntimeError is raised when no valid backend is found."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="No valid LLM provider configuration found"):
                get_default_backend()
    
    def test_anthropic_backend_without_key_raises_error(self):
        """Test that RuntimeError is raised when Anthropic provider is set but no key."""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'anthropic'
        }, clear=True):
            with pytest.raises(RuntimeError, match="No valid LLM provider configuration found"):
                get_default_backend()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
