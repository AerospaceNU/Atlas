from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar, Literal

import pytest
from pydantic import BaseModel, ValidationError

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolDefinition, ToolRegistry, ToolResult, default_registry
from atlas.agent.tools._list_tools import ListToolsTool

tool_def1 = ToolDefinition(
    name="read_file",
    description="Read a UTF-8 text file.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
)
tool_def2 = ToolDefinition(
    name="bash_tool",
    description="Run a shell command.",
    parameters={"type": "object", "properties": {}},
)


def test_tools_list_correctly_reads_as_json_format(tmp_path: Path) -> None:
    definitions = [tool_def1]
    tool = ListToolsTool(lambda: definitions)

    result = tool.run({}, LocalArtifactStore(tmp_path))
    result_list = json.loads(result.text)
    assert len(result_list) == 1
    assert result.artifacts == []

    result_tool1 = result_list[0]
    assert result_tool1.get("name") == "read_file"
    assert result_tool1.get("description") == "Read a UTF-8 text file."
    assert result_tool1.get("parameters") == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }


def test_tools_list_returns_info_for_each_current_definition(tmp_path: Path) -> None:
    definitions = [tool_def1, tool_def2]
    tool = ListToolsTool(lambda: definitions)

    result = tool.run({}, LocalArtifactStore(tmp_path))
    result_list = json.loads(result.text)
    assert len(result_list) == 2

    by_name = {definition["name"]: definition for definition in result_list}

    assert by_name["read_file"]["parameters"] == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }

    assert by_name["bash_tool"]["parameters"] == {"type": "object", "properties": {}}


def test_tools_list_looks_up_definitions_for_each_run(tmp_path: Path) -> None:
    calls = 0

    def definitions() -> list[ToolDefinition]:
        nonlocal calls
        calls += 1
        return [ToolDefinition(name=f"tool_{calls}", description="", parameters={})]

    tool = ListToolsTool(definitions)
    store = LocalArtifactStore(tmp_path)

    first = json.loads(tool.run({}, store).text)
    second = json.loads(tool.run({}, store).text)

    assert calls == 2
    assert first[0].get("name") == "tool_1"
    assert second[0].get("name") == "tool_2"


def test_tools_list_rejects_non_object_arguments(tmp_path: Path) -> None:
    tool = ListToolsTool(lambda: [])
    with pytest.raises(ValidationError):
        tool.run([], LocalArtifactStore(tmp_path))  # type: ignore


def test_tools_list_updates_with_registry(tmp_path: Path) -> None:

    class EmptyInput(BaseModel):
        pass

    class TestToolA(Tool):
        name = "Tool A"
        description = "temp"
        trust: ClassVar[Literal["default", "opt_in"]] = "default"

        @property
        def input_model(self) -> type[BaseModel]:
            return EmptyInput

        def run(self, arguments: dict[str, object], store: LocalArtifactStore) -> ToolResult:
            return ToolResult(text="hello")

    class TestToolB(Tool):
        name = "Tool B"
        description = "temp"
        trust: ClassVar[Literal["default", "opt_in"]] = "default"

        @property
        def input_model(self) -> type[BaseModel]:
            return EmptyInput

        def run(self, arguments: dict[str, object], store: LocalArtifactStore) -> ToolResult:
            return ToolResult(text="hi")

    test_tool1 = TestToolA()
    test_tool2 = TestToolB()

    registry = ToolRegistry()
    registry.register(test_tool1)
    assert len(registry.definitions) == 1

    list_tool = ListToolsTool(lambda: registry.definitions)
    store = LocalArtifactStore(tmp_path)
    assert len(json.loads(list_tool.run({}, store).text)) == 1

    registry.register(test_tool2)
    assert len(registry.definitions) == 2
    assert len(json.loads(list_tool.run({}, store).text)) == 2


def test_default_registry_registers_list_tools(tmp_path: Path) -> None:
    registry = default_registry()

    result = registry.execute("list_tools", {}, LocalArtifactStore(tmp_path))
    definitions = json.loads(result.text)
    names = {definition["name"] for definition in definitions}
    assert "list_tools" in names
