from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from .config import ModelSpec


@dataclass
class GenerationResult:
    text: str
    provider: str
    model: str
    usage: dict[str, Any]
    response_id: str | None = None


class Provider:
    def __init__(self, spec: ModelSpec, timeout_seconds: float = 120):
        self.spec = spec
        self.timeout_seconds = timeout_seconds

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int,
    ) -> GenerationResult:
        raise NotImplementedError


def _api_key(spec: ModelSpec, allow_empty: bool = False) -> str:
    if not spec.api_key_env:
        if allow_empty:
            return "EMPTY"
        raise ValueError(f"{spec.id}: api_key_env is required")
    value = os.environ.get(spec.api_key_env)
    if value:
        return value
    if allow_empty:
        return "EMPTY"
    raise RuntimeError(f"Environment variable {spec.api_key_env} is not set")


class OpenAIChatProvider(Provider):
    def __init__(self, spec: ModelSpec, timeout_seconds: float, compatible: bool = False):
        super().__init__(spec, timeout_seconds)
        from openai import AsyncOpenAI

        self.compatible = compatible
        self.client = AsyncOpenAI(
            api_key=_api_key(spec, allow_empty=compatible),
            base_url=spec.base_url,
            timeout=timeout_seconds,
        )

    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def generate(self, messages, temperature, max_tokens) -> GenerationResult:
        kwargs: dict[str, Any] = {
            "model": self.spec.model,
            "messages": messages,
        }
        if self.compatible:
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["max_completion_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = await self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if response.usage else {}
        return GenerationResult(
            text=text,
            provider="openai_compatible" if self.compatible else "openai",
            model=response.model or self.spec.model,
            usage=usage,
            response_id=response.id,
        )


class AnthropicProvider(Provider):
    def __init__(self, spec: ModelSpec, timeout_seconds: float):
        super().__init__(spec, timeout_seconds)
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(
            api_key=_api_key(spec), timeout=timeout_seconds
        )

    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def generate(self, messages, temperature, max_tokens) -> GenerationResult:
        system_parts = [
            m["content"] for m in messages if m["role"] in {"system", "developer"}
        ]
        anthropic_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in {"user", "assistant"}
        ]
        kwargs: dict[str, Any] = {
            "model": self.spec.model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = await self.client.messages.create(**kwargs)
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = response.usage.model_dump() if response.usage else {}
        return GenerationResult(
            text=text,
            provider="anthropic",
            model=response.model,
            usage=usage,
            response_id=response.id,
        )


class GeminiProvider(Provider):
    def __init__(self, spec: ModelSpec, timeout_seconds: float):
        super().__init__(spec, timeout_seconds)
        from google import genai

        self.client = genai.Client(api_key=_api_key(spec))

    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def generate(self, messages, temperature, max_tokens) -> GenerationResult:
        from google.genai import types

        system_parts = [
            m["content"] for m in messages if m["role"] in {"system", "developer"}
        ]
        contents = []
        for message in messages:
            if message["role"] in {"system", "developer"}:
                continue
            role = "model" if message["role"] == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=message["content"])],
                )
            )
        config_kwargs: dict[str, Any] = {"max_output_tokens": max_tokens}
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        response = await self.client.aio.models.generate_content(
            model=self.spec.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        usage_metadata = getattr(response, "usage_metadata", None)
        usage = usage_metadata.model_dump() if usage_metadata else {}
        return GenerationResult(
            text=response.text or "",
            provider="gemini",
            model=self.spec.model,
            usage=usage,
            response_id=getattr(response, "response_id", None),
        )


def create_provider(spec: ModelSpec, timeout_seconds: float) -> Provider:
    provider = spec.provider.lower()
    if provider == "openai":
        return OpenAIChatProvider(spec, timeout_seconds, compatible=False)
    if provider == "openai_compatible":
        return OpenAIChatProvider(spec, timeout_seconds, compatible=True)
    if provider == "anthropic":
        return AnthropicProvider(spec, timeout_seconds)
    if provider == "gemini":
        return GeminiProvider(spec, timeout_seconds)
    raise ValueError(f"Unsupported provider: {spec.provider}")

