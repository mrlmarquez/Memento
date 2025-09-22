import functools
import logging
import os
from typing import Any, Dict, List

import anthropic
import colorlog
from openai import AsyncAzureOpenAI, AsyncOpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
#   Logging setup
# ---------------------------------------------------------------------------
# Configure colored logging for better visibility of log levels
LOG_FORMAT = "%(log_color)s%(levelname)-8s%(reset)s %(message)s"
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ChatBackend:
    """Abstract base class for chat backends."""

    async def chat(self, *_, **__) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIBackend(ChatBackend):
    def __init__(self, model: str):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tool_session_values=None,
        tool_choice: str | None = "auto",
        max_tokens: int = 15000,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        if tool_session_values:
            tools = await self._format_tools_schema(tool_session_values)
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        resp = await self.client.chat.completions.create(**payload)  # type: ignore[arg-type]
        msg = resp.choices[0].message
        raw_calls = getattr(msg, "tool_calls", None)
        tool_calls = None
        if raw_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_calls
            ]
        return {"content": msg.content, "tool_calls": tool_calls}

    async def _format_tools_schema(self, session_values) -> List[Dict[str, Any]]:
        result, cached = [], {}
        for session in session_values:
            tools_resp = cached.get(id(session)) or await session.list_tools()
            cached[id(session)] = tools_resp
            for tool in tools_resp.tools:
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        return result


class AzureOpenAIBackend(OpenAIBackend):
    def __init__(self, model: str):
        self.model = model
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )


class AnthropicBackend(ChatBackend):
    """Simple Anthropic backend wrapper.

    This attempts to use the `anthropic` Python package. The package
    APIs vary; we call the blocking client in a thread to keep this
    method async. Tool-calls are not supported for Anthropic in this
    lightweight adapter.
    """

    def __init__(self, model: str):
        self.model = model
        self.client = anthropic.AsyncAnthropic(
            base_url=os.getenv("ANTHROPIC_API_URL"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools_session_values=None,
        tool_choice: str | None = "auto",
        max_tokens: int = 15000,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if tools_session_values:
            tools = await self._format_tools_schema(tools_session_values)
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        response = await self.client.messages.create(**payload)

        return self.prepare_response(response)

    def prepare_response(self, anthropic_response):
        text_blocks = []
        tool_calls = []

        # Iterate through all content blocks
        for block in anthropic_response.content:
            if block.type == "text":
                text_blocks.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": block.input},
                    }
                )

        # Combine all text blocks
        content = "".join(text_blocks) if text_blocks else None

        return {"content": content, "tool_calls": tool_calls if tool_calls else None}

    async def _format_tools_schema(self, session_values) -> List[Dict[str, Any]]:
        result, cached = [], {}
        for session in session_values:
            tools_resp = cached.get(id(session)) or await session.list_tools()
            cached[id(session)] = tools_resp
            for tool in tools_resp.tools:
                result.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                )
        return result


def get_default_backend(model: str = "gpt-4") -> ChatBackend:
    """Selects the default LLM backend based on environment variables."""
    provider = os.getenv("LLM_PROVIDER")
    if provider == "azure":
        if os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            return AzureOpenAIBackend(model)
    elif provider == "openai":
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIBackend(model)
    elif provider == "anthropic":
        if os.getenv("ANTHROPIC_API_KEY"):
            return AnthropicBackend(model)
    # Fallback logic
    if os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAIBackend(model)
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIBackend(model)
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicBackend(model)
    raise RuntimeError("No valid LLM provider configuration found.")


# Optionally, provide a default backend instance for convenience
default_backend = functools.partial(get_default_backend)
