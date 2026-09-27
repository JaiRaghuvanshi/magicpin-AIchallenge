"""
Pluggable LLM provider layer.

Default: Anthropic (Claude) — chosen for controlled, register-sensitive
prose and reliable adherence to tone/vocab constraints (see README for the
full rationale). Swap providers with the VERA_LLM_PROVIDER env var; no other
code changes needed since composer.py only talks to the LLMProvider interface.

If no API key is configured (e.g. this dev/sandbox environment with no
network egress), `call()` raises LLMUnavailable and composer.py falls back
to the deterministic rule-based composer so the bot stays fully functional
and testable offline.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod


class LLMUnavailable(RuntimeError):
    """Raised when no provider/API key is configured, or the call fails."""


class LLMProvider(ABC):
    @abstractmethod
    def complete_json(self, system: str, user: str, max_tokens: int = 600) -> dict:
        """Call the model with temperature=0 and parse a JSON object from the response."""
        ...


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMUnavailable("ANTHROPIC_API_KEY not set")
        self.model = model
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise LLMUnavailable("anthropic package not installed") from e
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def complete_json(self, system: str, user: str, max_tokens: int = 600) -> dict:
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(block.text for block in resp.content if block.type == "text")
            return _parse_json_response(text)
        except Exception as e:  # noqa: BLE001
            raise LLMUnavailable(f"Anthropic call failed: {e}") from e


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMUnavailable("OPENAI_API_KEY not set")
        self.model = model
        try:
            import openai  # type: ignore
        except ImportError as e:
            raise LLMUnavailable("openai package not installed") from e
        self._client = openai.OpenAI(api_key=self.api_key)

    def complete_json(self, system: str, user: str, max_tokens: int = 600) -> dict:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return _parse_json_response(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            raise LLMUnavailable(f"OpenAI call failed: {e}") from e


def get_provider() -> LLMProvider:
    """
    Selects provider by VERA_LLM_PROVIDER env var (default: anthropic).
    Raises LLMUnavailable if construction fails (missing key/package) —
    composer.py catches this and falls back to the rule-based composer.
    """
    name = os.environ.get("VERA_LLM_PROVIDER", "anthropic").lower()
    if name == "anthropic":
        model = os.environ.get("VERA_LLM_MODEL", "claude-sonnet-4-6")
        return AnthropicProvider(model=model)
    if name == "openai":
        model = os.environ.get("VERA_LLM_MODEL", "gpt-4o-mini")
        return OpenAIProvider(model=model)
    raise LLMUnavailable(f"Unknown VERA_LLM_PROVIDER: {name}")
