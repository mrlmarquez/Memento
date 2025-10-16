# Component Tests for HierarchicalClient

This directory contains component tests that verify the HierarchicalClient works correctly with actual backend services (OpenAI and Anthropic). These tests are more comprehensive than unit tests as they test real integration scenarios.

## Test Files

### `test_hierarchical_client_component.py`
Component tests that use actual backend instances:

- **Backend Combinations**: Tests all combinations of OpenAI and Anthropic backends
- **End-to-End Flow**: Tests the complete planning and execution process
- **Tool Integration**: Tests tool usage with different backends
- **Error Handling**: Tests error scenarios with real backends
- **Performance**: Tests performance characteristics
- **Consistency**: Verifies consistent behavior across backends

### `component_test_config.py`
Configuration helper for setting up API keys and validating the test environment.

### `run_component_tests.py`
Test runner script with setup validation and specific test execution.

## Setup

### 1. Install Dependencies
```bash
# Install test dependencies
uv add pytest pytest-asyncio --dev

# Install backend dependencies (if not already installed)
uv add anthropic openai
```

### 2. Configure API Keys

You need at least one of the following API keys:

#### OpenAI
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

#### Anthropic
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export ANTHROPIC_API_URL="https://api.anthropic.com"  # Optional
```

#### Azure OpenAI (Optional)
```bash
export AZURE_OPENAI_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export OPENAI_API_VERSION="2024-02-15-preview"
```

### 3. Verify Setup
```bash
# Check API key status
python tests/run_component_tests.py check

# Show setup instructions
python tests/run_component_tests.py setup
```

## Running Tests

### Run All Component Tests
```bash
# Using the test runner
python tests/run_component_tests.py

# Or directly with pytest
uv run pytest tests/test_hierarchical_client_component.py -v
```

### Run Specific Tests
```bash
# Run specific test
python tests/run_component_tests.py test:TestHierarchicalClientComponent::test_openai_meta_openai_exec_component

# Or with pytest
uv run pytest tests/test_hierarchical_client_component.py::TestHierarchicalClientComponent::test_openai_meta_openai_exec_component -v
```

### Run Tests by Backend
```bash
# Only OpenAI tests
uv run pytest tests/test_hierarchical_client_component.py -k "openai" -v

# Only Anthropic tests
uv run pytest tests/test_hierarchical_client_component.py -k "anthropic" -v

# Mixed backend tests
uv run pytest tests/test_hierarchical_client_component.py -k "mixed" -v
```

## Test Categories

### 1. Backend Combination Tests
- `test_openai_meta_openai_exec_component`
- `test_anthropic_meta_anthropic_exec_component`
- `test_openai_meta_anthropic_exec_component`
- `test_anthropic_meta_openai_exec_component`

### 2. Planning and Execution Flow Tests
- `test_planning_and_execution_flow_openai`
- `test_planning_and_execution_flow_anthropic`

### 3. Tool Integration Tests
- `test_tool_integration_openai`
- `test_tool_integration_anthropic`
- `test_mixed_backend_tool_usage`

### 4. Error Handling Tests
- `test_error_handling_openai`
- `test_error_handling_anthropic`

### 5. Performance and Consistency Tests
- `test_backend_consistency_verification`
- `test_performance_comparison`

## Test Scenarios

### Simple Math Problems
Tests use simple arithmetic problems that should reliably produce correct answers:
- "What is 2 + 2?" → Should return "4"
- "What is 3 + 3?" → Should return "6"
- "What is 4 + 4?" → Should return "8"

### Planning Tasks
Tests use problems that require the hierarchical planner to break down tasks:
- Rectangle area calculation
- Square perimeter calculation
- Multi-step mathematical operations

### Tool Integration
Tests verify that tools are properly passed to the executor:
- Mock calculator tools
- Math helper tools
- Tool schema formatting

## Expected Behavior

### Successful Tests
- All tests should complete without errors
- Math problems should return correct numerical answers
- Planning tasks should be broken down into steps
- Tool integration should work with both backends
- Error handling should be graceful

### Performance Expectations
- Individual queries should complete within 30 seconds
- Simple math problems should be fast (< 5 seconds)
- Complex planning tasks may take longer (10-30 seconds)

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   pytest.skip("OpenAI API key not available")
   ```
   - Solution: Set the required environment variables

2. **Rate Limiting**
   ```
   Rate limit exceeded
   ```
   - Solution: Wait a few minutes and retry, or use different API keys

3. **Network Issues**
   ```
   Connection timeout
   ```
   - Solution: Check internet connection and API endpoint URLs

4. **Invalid Response Format**
   ```
   AssertionError: assert "4" in result.lower()
   ```
   - Solution: The test may need adjustment for different model responses

### Debug Mode
Run tests with more verbose output:
```bash
uv run pytest tests/test_hierarchical_client_component.py -v -s --tb=long
```

### Skip Tests
Skip tests that require specific API keys:
```bash
# Skip Anthropic tests
uv run pytest tests/test_hierarchical_client_component.py -k "not anthropic" -v

# Skip OpenAI tests
uv run pytest tests/test_hierarchical_client_component.py -k "not openai" -v
```

## Cost Considerations

These tests make actual API calls and may incur costs:

- **OpenAI**: ~$0.001-0.01 per test run
- **Anthropic**: ~$0.001-0.01 per test run
- **Total**: ~$0.01-0.05 per full test suite

To minimize costs:
- Use cheaper models (gpt-3.5-turbo, claude-3-haiku)
- Run tests selectively
- Use test-specific API keys with usage limits

## Integration with CI/CD

For CI/CD pipelines, consider:

1. **Using test-specific API keys** with limited permissions
2. **Setting up rate limiting** to avoid hitting API limits
3. **Using mock backends** for most tests, with component tests only for critical paths
4. **Running component tests** only on specific branches or schedules

Example GitHub Actions workflow:
```yaml
- name: Run Component Tests
  if: github.ref == 'refs/heads/main'
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    python tests/run_component_tests.py
```

