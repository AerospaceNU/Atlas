from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar, Literal

import pytest
from pydantic import BaseModel

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    Tool,
    ToolRegistry,
    ToolResult,
    build_registry,
    default_registry,
    discover_tool_classes,
)


class EmptyInput(BaseModel):
    pass


def _tool_module(class_name: str, tool_name: str, *, trust: str | None = None) -> str:
    trust_line = f"    trust: ClassVar[Literal['default', 'opt_in']] = '{trust}'\n" if trust else ""
    return (
        "from typing import ClassVar, Literal\n"
        "from pydantic import BaseModel\n"
        "from atlas.agent.contracts import Tool, ToolResult\n\n\n"
        "class EmptyInput(BaseModel):\n"
        "    pass\n\n\n"
        f"class {class_name}(Tool):\n"
        f"    name = '{tool_name}'\n"
        "    description = 'test tool'\n"
        f"{trust_line}"
        "    @property\n"
        "    def input_model(self) -> type[BaseModel]:\n"
        "        return EmptyInput\n\n"
        "    def run(self, arguments, store):\n"
        "        return ToolResult(text='ok')\n"
    )


def _write_package(root: Path, name: str, modules: dict[str, str]) -> None:
    package = root / name
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for module, source in modules.items():
        (package / f"{module}.py").write_text(source, encoding="utf-8")


@pytest.fixture
def sandbox_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    created: list[str] = []

    def make(name: str, modules: dict[str, str]) -> str:
        _write_package(tmp_path, name, modules)
        monkeypatch.syspath_prepend(str(tmp_path))
        created.append(name)
        return name

    yield make

    for name in created:
        for module_name in [
            key for key in sys.modules if key == name or key.startswith(f"{name}.")
        ]:
            del sys.modules[module_name]


def test_tools_package_does_not_export_the_framework() -> None:
    import atlas.agent.tools as tools_package

    assert not hasattr(tools_package, "Tool")
    assert not hasattr(tools_package, "ToolRegistry")
    assert not hasattr(tools_package, "ToolResult")
    assert not hasattr(tools_package, "default_registry")
    assert tools_package.ReadFileTool.name == "read_file"


def test_default_registry_has_discovered_tools_only() -> None:
    names = [tool.name for tool in default_registry().definitions]

    assert "read_file" in names
    assert "segment_landcover" in names
    assert "list_images" not in names
    assert "inspect_image" not in names
    assert "create_contact_sheet" not in names
    assert "stage_script_proposal" not in names
    assert "review_script_proposals" not in names


def test_discovery_sorts_by_module_then_qualname() -> None:
    classes = discover_tool_classes(("atlas.agent.tools",))

    keys = [(cls.__module__, cls.__qualname__) for cls in classes]
    assert keys == sorted(keys)


def test_duplicate_tool_name_raises_with_both_sources(sandbox_package) -> None:
    name = sandbox_package(
        "atlas_dup_pkg",
        {
            "first": _tool_module("Alpha", "duplicated"),
            "second": _tool_module("Beta", "duplicated"),
        },
    )

    with pytest.raises(ValueError) as excinfo:
        discover_tool_classes((name,))

    message = str(excinfo.value)
    assert "duplicated" in message
    assert "atlas_dup_pkg.first.Alpha" in message
    assert "atlas_dup_pkg.second.Beta" in message


def test_opt_in_tool_is_filtered_unless_requested(sandbox_package) -> None:
    name = sandbox_package(
        "atlas_opt_in_pkg",
        {
            "safe": _tool_module("Safe", "safe_tool"),
            "risky": _tool_module("Risky", "risky_tool", trust="opt_in"),
        },
    )

    default_names = [tool.name for tool in build_registry((name,)).definitions]
    assert default_names == ["safe_tool"]

    opt_in_names = [tool.name for tool in build_registry((name,), include_opt_in=True).definitions]
    assert opt_in_names == ["risky_tool", "safe_tool"]


def test_discovery_skips_private_modules_and_subpackages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_package(
        tmp_path,
        "atlas_skip_pkg",
        {"_private": _tool_module("Private", "private_tool")},
    )
    nested = tmp_path / "atlas_skip_pkg" / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "__init__.py").write_text(_tool_module("Nested", "nested_tool"), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert discover_tool_classes(("atlas_skip_pkg",)) == []

    for module_name in [
        key for key in sys.modules if key == "atlas_skip_pkg" or key.startswith("atlas_skip_pkg.")
    ]:
        del sys.modules[module_name]


def test_invalid_trust_value_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown tool trust level"):

        class BadTool(Tool):
            name = "bad"
            description = "bad"
            trust: ClassVar[Literal["default", "opt_in"]] = "anything"  # type: ignore[assignment]

            @property
            def input_model(self) -> type[BaseModel]:
                return EmptyInput

            def run(self, arguments: dict[str, object], store: LocalArtifactStore) -> ToolResult:
                return ToolResult(text="bad")


def test_concrete_tool_without_a_name_is_rejected(sandbox_package) -> None:
    source = _tool_module("Nameless", "ignored").replace("    name = 'ignored'\n", "")
    name = sandbox_package("atlas_nameless_pkg", {"blank": source})

    with pytest.raises(ValueError, match="without a string name"):
        discover_tool_classes((name,))


def test_reexported_tool_is_not_discovered(sandbox_package) -> None:
    name = sandbox_package(
        "atlas_reexport_pkg",
        {"reexport": "from atlas.agent.tools.read_file import ReadFileTool\n"},
    )

    assert discover_tool_classes((name,)) == []


def test_build_registry_registers_extra_after_discovery() -> None:
    class ExtraTool(Tool):
        name = "extra_tool"
        description = "extra"

        @property
        def input_model(self) -> type[BaseModel]:
            return EmptyInput

        def run(self, arguments: dict[str, object], store: LocalArtifactStore) -> ToolResult:
            return ToolResult(text="extra")

    registry = build_registry(extra=[ExtraTool()])
    assert "extra_tool" in [tool.name for tool in registry.definitions]

    duplicate = ToolRegistry()
    duplicate.register(ExtraTool())
    with pytest.raises(ValueError, match="already registered"):
        duplicate.register(ExtraTool())
