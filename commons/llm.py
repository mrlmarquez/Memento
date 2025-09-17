import functools
import logging
import os
from typing import Any, Dict, List

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
        tools: List[Dict[str, Any]] | None = None,
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
        if tools:
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


class AzureOpenAIBackend(OpenAIBackend):
    def __init__(self, model: str):
        self.model = model
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )


def get_default_backend(model: str = "gpt-4") -> ChatBackend:
    """Selects the default LLM backend based on environment variables."""
    provider = os.getenv("LLM_PROVIDER")
    if provider == "azure":
        if os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            return AzureOpenAIBackend(model)
    elif provider == "openai":
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIBackend(model)
    # Fallback logic
    if os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAIBackend(model)
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIBackend(model)
    raise RuntimeError("No valid LLM provider configuration found.")


# Optionally, provide a default backend instance for convenience
default_backend = functools.partial(get_default_backend)
