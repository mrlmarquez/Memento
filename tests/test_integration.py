"""
Integration test to verify Anthropic backend works with the actual client code.

This test ensures that the Anthropic backend integrates properly with the
existing client code and doesn't break any functionality.
"""

import asyncio
import os
import pytest
import sys
from unittest.mock import patch, AsyncMock, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons.llm import get_default_backend, AnthropicBackend


class MockAnthropicResponse:
    """Mock Anthropic API response for integration testing."""
    
    def __init__(self, content_text: str = "Hello! This is a test response from Claude."):
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


@pytest.mark.asyncio
async def test_anthropic_backend_integration():
    """Test that Anthropic backend integrates properly with client code."""
    print("Testing Anthropic backend integration...")
    
    # Test 1: Backend selection with environment variables
    print("1. Testing backend selection...")
    with patch.dict(os.environ, {
        'LLM_PROVIDER': 'anthropic',
        'ANTHROPIC_API_KEY': 'test-key'
    }):
        backend = get_default_backend("claude-3-sonnet-20240229")
        assert isinstance(backend, AnthropicBackend)
        print("   ✓ Anthropic backend selected correctly")
    
    # Test 2: Basic chat functionality
    print("2. Testing basic chat functionality...")
    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-key'
    }):
        backend = AnthropicBackend("claude-3-sonnet-20240229")
        
        # Mock the Anthropic client response
        mock_response = MockAnthropicResponse("Test response from Claude")
        
        with patch.object(backend.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello, Claude!"}]
            result = await backend.chat(messages)
            
            assert result["content"] == "Test response from Claude"
            assert result["tool_calls"] is None
            print("   ✓ Basic chat functionality works")
    
    # Test 3: Tool integration
    print("3. Testing tool integration...")
    
    # Mock tool session
    class MockTool:
        def __init__(self, name, description, input_schema):
            self.name = name
            self.description = description
            self.inputSchema = input_schema
    
    class MockToolSession:
        def __init__(self, tools):
            self.tools = tools
        
        async def list_tools(self):
            return MagicMock(tools=self.tools)
    
    mock_tool = MockTool(
        "test_tool",
        "A test tool for integration testing",
        {"type": "object", "properties": {"param": {"type": "string"}}}
    )
    
    session = MockToolSession([mock_tool])
    
    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-key'
    }):
        backend = AnthropicBackend("claude-3-sonnet-20240229")
        
        # Mock response with tool call
        mock_response = MockAnthropicResponse()
        mock_response.content = [
            MockContentBlock("text", "I'll use the test tool."),
            MockContentBlock("tool_use", tool_use_data={
                "id": "tool_123",
                "name": "test_tool",
                "input": '{"param": "test_value"}'
            })
        ]
        
        with patch.object(backend.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            messages = [{"role": "user", "content": "Use the test tool"}]
            result = await backend.chat(messages, tools_session_values=[session])
            
            assert result["content"] == "I'll use the test tool."
            assert result["tool_calls"] is not None
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["function"]["name"] == "test_tool"
            print("   ✓ Tool integration works")
    
    # Test 4: Error handling
    print("4. Testing error handling...")
    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-key'
    }):
        backend = AnthropicBackend("claude-3-sonnet-20240229")
        
        with patch.object(backend.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API Error")
            
            messages = [{"role": "user", "content": "This should fail"}]
            
            try:
                await backend.chat(messages)
                assert False, "Expected exception to be raised"
            except Exception as e:
                assert "API Error" in str(e)
                print("   ✓ Error handling works correctly")
    
    print("\n✅ All integration tests passed!")
    print("\nThe Anthropic backend is working correctly and integrates properly with the client code.")


@pytest.mark.asyncio
async def test_backend_compatibility():
    """Test that Anthropic backend is compatible with existing code patterns."""
    print("\nTesting backend compatibility...")
    
    # Test that the backend can be used in the same way as other backends
    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-key'
    }):
        backend = get_default_backend("claude-3-sonnet-20240229")
        
        # Test basic interface compatibility
        assert hasattr(backend, 'chat')
        assert hasattr(backend, '_format_tools_schema')
        assert callable(backend.chat)
        assert callable(backend._format_tools_schema)
        
        print("✓ Backend interface is compatible with existing code")
    
    print("✅ Backend compatibility test passed!")


if __name__ == "__main__":
    print("Running Anthropic Backend Integration Tests")
    print("=" * 50)
    
    # Run the integration tests
    asyncio.run(test_anthropic_backend_integration())
    asyncio.run(test_backend_compatibility())
    
    print("\n🎉 All integration tests completed successfully!")
    print("\nThe Anthropic backend is ready for use and won't break existing functionality.")
