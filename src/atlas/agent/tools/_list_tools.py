"""Agent tool to list other available tools."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolDefinition, ToolResult


class ListToolsInput(BaseModel):
    pass


class ListToolsTool(Tool):
    name: str = "list_tools"
    description: str = "List tools available in this agent session with their description and input in json format."
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    def __init__(self, definitions: Callable[[], list[ToolDefinition]]) -> None:
        self._definitions = definitions

    @property
    def input_model(self) -> type[BaseModel]:
        return ListToolsInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        ListToolsInput.model_validate(arguments)

        definitions = [definition.model_dump() for definition in self._definitions()]
        return ToolResult(text=json.dumps(definitions))
