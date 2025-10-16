# Anthropic Backend Tests

This directory contains comprehensive tests for the Anthropic/Claude backend implementation to ensure it works correctly and doesn't break existing functionality.

## Test Files

### `test_anthropic_backend.py`
Comprehensive unit tests for the Anthropic backend implementation:

- **Basic Functionality**: Tests basic chat functionality with text responses
- **Tool Integration**: Tests tool calling and schema formatting
- **Error Handling**: Tests retry mechanisms and error scenarios
- **Backend Integration**: Tests compatibility with existing backend interfaces
- **Environment Configuration**: Tests environment variable handling and backend selection

### `test_hierarchical_client_backends.py`
Unit tests for HierarchicalClient with different backend combinations:

- **Backend Combinations**: Tests all permutations of AzureOpenAI and Anthropic backends
- **Process Query Flow**: Tests the complete planning and execution process
- **Tool Integration**: Tests tool usage with different backend combinations
- **Error Handling**: Tests error scenarios and edge cases
- **Compatibility**: Verifies consistent behavior across backends

### `test_hierarchical_client_component.py`
Component tests using actual backend services:

- **Real Backend Integration**: Tests with actual OpenAI and Anthropic APIs
- **End-to-End Scenarios**: Tests complete workflows with real backends
- **Performance Testing**: Tests performance characteristics
- **Mixed Backend Support**: Tests combinations of different backends
- **Tool Integration**: Tests tool usage with real backends

### `test_integration.py`
Integration tests to verify the Anthropic backend works with the actual client code:

- **Backend Selection**: Tests that the correct backend is selected based on environment variables
- **Client Integration**: Tests that the backend integrates properly with existing client code
- **Tool Integration**: Tests tool calling in a realistic scenario
- **Error Handling**: Tests error handling in integration scenarios

### `example_component_test.py`
Example component tests demonstrating the testing patterns:

- **Simple Examples**: Basic math problems with different backends
- **Planning Tasks**: Multi-step problem solving
- **Mixed Backend Examples**: Using different backends for meta and exec models

## Running Tests

### Run All Tests
```bash
# Using uv (recommended) - from project root
uv run pytest tests/ -v

# Run specific test files
uv run pytest tests/test_anthropic_backend.py -v
uv run pytest tests/test_integration.py -v
```

### Run Integration Tests
```bash
# From project root
uv run python tests/test_integration.py
```

### Run Specific Test Classes
```bash
# Run only AnthropicBackend tests
uv run pytest tests/test_anthropic_backend.py::TestAnthropicBackend -v

# Run only integration tests
uv run pytest tests/test_anthropic_backend.py::TestBackendIntegration -v

# Run only HierarchicalClient backend tests
uv run pytest tests/test_hierarchical_client_backends.py -v
```

### Run Component Tests
```bash
# Check setup and run component tests
python tests/run_component_tests.py

# Check API key status
python tests/run_component_tests.py check

# Show setup instructions
python tests/run_component_tests.py setup

# Run specific component test
python tests/run_component_tests.py test:TestHierarchicalClientComponent::test_openai_meta_openai_exec_component

# Run component tests directly with pytest (requires API keys)
uv run pytest tests/test_hierarchical_client_component.py -v
```

## Test Coverage

The test suite covers:

1. **Initialization**: Backend creation and configuration
2. **Basic Chat**: Simple text-based conversations
3. **Tool Calls**: Function calling and tool integration
4. **Mixed Content**: Responses with both text and tool calls
5. **Schema Formatting**: Tool schema conversion between formats
6. **Caching**: Tool schema caching behavior
7. **Error Handling**: Retry mechanisms and failure scenarios
8. **Environment Variables**: Configuration and backend selection
9. **Backend Compatibility**: Interface consistency with other backends
10. **Integration**: Real-world usage scenarios

## Environment Variables

The tests use the following environment variables:

- `ANTHROPIC_API_KEY`: Anthropic API key for testing
- `ANTHROPIC_API_URL`: Optional custom Anthropic API endpoint
- `LLM_PROVIDER`: Set to "anthropic" to use Anthropic backend
- `OPENAI_API_KEY`: Used for compatibility testing

## Dependencies

The tests require the following additional dependencies (installed as dev dependencies):

- `pytest`: Test framework
- `pytest-asyncio`: Async test support

## Test Results

All tests should pass, indicating that:

✅ The Anthropic backend is properly implemented  
✅ Tool integration works correctly  
✅ Error handling is robust  
✅ Backend selection works as expected  
✅ Integration with existing code is seamless  
✅ No existing functionality is broken  

## Notes

- Tests use mocking to avoid making actual API calls
- The integration tests verify that the backend works with the actual client code patterns
- All tests are designed to be fast and reliable
- The test suite ensures backward compatibility with existing backends
