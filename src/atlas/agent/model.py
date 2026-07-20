from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Self

import httpx

from atlas.agent.contracts import ChatMessage, ModelResponse, ToolCall, ToolDefinition

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
        self.history: list[ChatMessage] = []
        self._client = client or httpx.Client(timeout=config.timeout)
        self._owns_client = client is None

    def __call__(self, prompt: str) -> str:
        self.history.append(ChatMessage(role="user", content=prompt))
        try:
            response = self.complete(self.history, [])
            content = response.content
            if content is None:
                raise ValueError("OpenRouter returned no text content")
        except Exception:
            self.history.pop()
            raise
        self.history.append(ChatMessage(role="assistant", content=content))
        return content

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        """Call OpenRouter using its OpenAI-compatible native tool-calling format."""
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [_serialize_message(message) for message in messages],
            "reasoning": {"enabled": self.config.reasoning},
            "max_tokens": self.config.max_new_tokens,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        response = self._post(payload)
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected OpenRouter response shape: {response!r}") from exc
        if not isinstance(message, dict):
            raise TypeError(f"Expected response message object, got {type(message)}")
        return _parse_response(message)

    def reset(self) -> None:
        self.history.clear()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()

        response = resp.json()
        if not isinstance(response, dict):
            raise TypeError(f"Expected OpenRouter response object, got {type(response)}")
        return response


def _serialize_message(message: ChatMessage) -> dict[str, Any]:
    data: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        data["content"] = message.content
    if message.tool_call_id is not None:
        data["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        data["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return data


def _parse_response(message: dict[str, Any]) -> ModelResponse:
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise TypeError(f"Expected string or null content, got {type(content)}")
    calls: list[ToolCall] = []
    raw_calls = message.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise TypeError(f"Expected list tool_calls, got {type(raw_calls)}")
    for raw_call in raw_calls:
        try:
            function = raw_call["function"]
            raw_arguments = function["arguments"]
            arguments = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            )
            calls.append(ToolCall(id=raw_call["id"], name=function["name"], arguments=arguments))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unexpected OpenRouter tool call: {raw_call!r}") from exc
    return ModelResponse(content=content, tool_calls=calls)
