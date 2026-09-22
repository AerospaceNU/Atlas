from __future__ import annotations

import importlib
import inspect
import pkgutil
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore

_TRUST_VALUES = frozenset({"default", "opt_in"})


class Role(str, Enum):  # noqa: UP042 - str mixin keeps the lowercase wire values
    """Chat role. A str enum so the wire format stays lowercase."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolDefinition(BaseModel):
    """A function schema advertised to a tool-capable model."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """One structured request returned by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """A portable chat-completions message."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class ModelResponse(BaseModel):
    """The subset of a model response needed by the runtime."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolCapableModel(Protocol):
    """Minimal model contract needed by :class:`atlas.agent.runtime.Agent`."""

    def complete(
        self, messages: list[ChatMessage], tools: list[ToolDefinition]
    ) -> ModelResponse: ...


class ToolResult(BaseModel):
    """Structured tool output returned to the model and retained in the run log."""

    text: str
    artifacts: list[str] = Field(default_factory=list)


class Tool(ABC):
    """A local capability that can be safely advertised to an agent."""

    name: str
    description: str
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        trust = getattr(cls, "trust", "default")
        if trust not in _TRUST_VALUES:
            raise ValueError(f"Unknown tool trust level: {trust!r}")

    @property
    @abstractmethod
    def input_model(self) -> type[BaseModel]:
        """Pydantic input model used for both validation and JSON schema."""
        raise NotImplementedError

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_model.model_json_schema(),
        )

    @abstractmethod
    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        """Validate and execute a tool call within ``store``."""
        raise NotImplementedError


class ToolRegistry:
    """Explicit allow-list of tools available to an agent session."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    @property
    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

    def execute(
        self, name: str, arguments: dict[str, Any], store: LocalArtifactStore
    ) -> ToolResult:
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc
        return tool.run(arguments, store)


def discover_tool_classes(packages: Sequence[str]) -> list[type[Tool]]:
    """Import every module directly inside ``packages`` and collect concrete tools.

    A tool counts only when its class statement is in that module. Imported
    tool classes are ignored, so a re-export cannot grant a capability.
    Subpackages and private modules are skipped. Import errors propagate.

    A concrete tool with no string ``name`` raises ``ValueError``. Two classes
    that share a name raise ``ValueError`` naming both sources.
    """
    found: dict[type[Tool], str] = {}
    for package in packages:
        module = importlib.import_module(package)
        for info in pkgutil.iter_modules(module.__path__):
            if info.ispkg or info.name.startswith("_"):
                continue
            imported = importlib.import_module(f"{package}.{info.name}")
            for candidate in vars(imported).values():
                if not inspect.isclass(candidate) or not issubclass(candidate, Tool):
                    continue
                if candidate is Tool or inspect.isabstract(candidate):
                    continue
                if candidate.__module__ != imported.__name__:
                    continue
                source = f"{candidate.__module__}.{candidate.__qualname__}"
                if not isinstance(getattr(candidate, "name", None), str):
                    raise ValueError(f"{source} is a concrete tool without a string name")
                found.setdefault(candidate, source)

    classes = sorted(found, key=lambda cls: (cls.__module__, cls.__qualname__))
    seen: dict[str, str] = {}
    for cls in classes:
        source = found[cls]
        previous = seen.get(cls.name)
        if previous is not None:
            raise ValueError(f"Duplicate tool name {cls.name!r} defined by {previous} and {source}")
        seen[cls.name] = source
    return classes


def build_registry(
    packages: Sequence[str] = ("atlas.agent.tools",),
    *,
    include_opt_in: bool = False,
    extra: Sequence[Tool] = (),
) -> ToolRegistry:
    """Discover tools in ``packages``, optionally keep opt-in tools, then add ``extra``."""
    registry = ToolRegistry()
    for cls in discover_tool_classes(packages):
        if not include_opt_in and cls.trust == "opt_in":
            continue
        registry.register(cls())
    for tool in extra:
        registry.register(tool)
    return registry


def default_registry() -> ToolRegistry:
    """Build the normal tool allow-list for one agent session."""
    registry = build_registry(("atlas.agent.tools",))
    from atlas.models.registry import register_model_tools

    register_model_tools(registry)
    return registry
