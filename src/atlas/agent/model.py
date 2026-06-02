from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class ModelConfig:
    model: str
    api_key: str
    reasoning: bool = True
    max_new_tokens: int = 2048
    timeout: float = 60.0


class OpenRouterModel:
    def __init__(self, config: ModelConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        self.history: list[dict[str, str]] = []
        self._client = client or httpx.Client(timeout=config.timeout)
        self._owns_client = client is None

    def __call__(self, prompt: str) -> str:
        self._add_message("user", prompt)
        try:
            content = self._make_request(self.history)
        except Exception:
            self.history.pop()
            raise
        self._add_message("assistant", content)
        return content

    def reset(self) -> None:
        self.history.clear()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def _make_request(self, messages: list[dict[str, str]] | str) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        resp = self._client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.model,
                "messages": messages,
                "reasoning": {"enabled": self.config.reasoning},
                "max_tokens": self.config.max_new_tokens,
            },
        )
        resp.raise_for_status()

        payload = resp.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected OpenRouter response shape: {payload!r}") from exc

        if not isinstance(content, str):
            raise TypeError(f"Expected string content from OpenRouter, got {type(content)}")
        return content
