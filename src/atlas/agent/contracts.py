from __future__ import annotations

import importlib
import inspect
import pkgutil
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore

_TRUST_VALUES = frozenset({"default", "opt_in"})
_SCHEMA_MAPS = frozenset({"properties", "$defs", "definitions", "patternProperties"})
_SCHEMA_VALUES = frozenset({"default", "const", "enum", "examples"})

_cancel_event_var: ContextVar[threading.Event | None] = ContextVar(
    "atlas_cancel_event", default=None
)
_output_sink_var: ContextVar[Callable[[str], None] | None] = ContextVar(
    "atlas_output_sink", default=None
)
_child_calls_var: ContextVar[list[ModelCall] | None] = ContextVar("atlas_child_calls", default=None)
_shared_budget_var: ContextVar[Any] = ContextVar("atlas_shared_budget", default=None)


def current_cancel_event() -> threading.Event | None:
    """Return the active turn's cancellation event, if executing inside an agent turn."""
    return _cancel_event_var.get()


def emit_tool_output(chunk: str) -> None:
    """Stream an incremental tool output chunk to the active RPC sink, if any."""
    if not chunk:
        return
    sink = _output_sink_var.get()
    if sink is not None:
        sink(chunk)


def current_child_calls() -> list[ModelCall] | None:
    """Return the parent turn's call collector when executing inside a subagent."""
    return _child_calls_var.get()


def current_shared_budget() -> Any:
    """Return the shared parent budget state when executing inside a subagent."""
    return _shared_budget_var.get()


@contextmanager
def tool_execution_scope(
    *,
    cancel_event: threading.Event | None = None,
    output_sink: Callable[[str], None] | None = None,
    child_calls: list[ModelCall] | None = None,
    shared_budget: Any = None,
) -> Iterator[None]:
    """Bind per-call cancellation, output streaming, and subagent accounting contextvars."""
    t_cancel = _cancel_event_var.set(cancel_event)
    t_sink = _output_sink_var.set(output_sink)
    t_calls = _child_calls_var.set(child_calls)
    t_budget = _shared_budget_var.set(shared_budget)
    try:
        yield
    finally:
        _cancel_event_var.reset(t_cancel)
        _output_sink_var.reset(t_sink)
        _child_calls_var.reset(t_calls)
        _shared_budget_var.reset(t_budget)


class Role(str, Enum):  # noqa: UP042 - str mixin keeps lowercase wire values
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
    arguments_error: str | None = None


class ModelImage(BaseModel):
    """One local image artifact encoded for a vision model."""

    path: str
    media_type: str
    data: str


class ChatMessage(BaseModel):
    """A portable chat-completions message."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    reasoning: str | None = None
    reasoning_details: list[dict[str, Any]] = Field(default_factory=list)
    images: list[ModelImage] = Field(default_factory=list)
    # Local history metadata, never sent to the model provider.
    tool_observation: bool = False


class TokenUsage(BaseModel):
    """Provider token counts and optional spend for one completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float | None = None


class CallTiming(BaseModel):
    """HTTP status and measured duration for one model call."""

    http_status: int
    elapsed_seconds: float
    time_to_first_token_seconds: float | None = None
    error_body: str | None = None


class ModelCall(BaseModel):
    """Usage and timing captured from one completion."""

    usage: TokenUsage = Field(default_factory=TokenUsage)
    timing: CallTiming | None = None
    finish_reason: str | None = None


class CatalogModel(BaseModel):
    """One row from the OpenRouter models catalog."""

    id: str
    name: str | None = None
    description: str | None = None
    context_length: int | None = None
    prompt_price: str | None = None
    completion_price: str | None = None
    cache_read_price: str | None = None
    supports_tools: bool = False


class ModelResponse(BaseModel):
    """The subset of a model response needed by the runtime."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    timing: CallTiming | None = None
    reasoning: str | None = None
    reasoning_details: list[dict[str, Any]] = Field(default_factory=list)
    # Why the provider stopped: "stop", "tool_calls", "length", or None when unreported.
    finish_reason: str | None = None
    # Tool-call id -> why its arguments could not be parsed. Such a call keeps
    # empty arguments and is answered with a validation error, never executed.
    argument_errors: dict[str, str] = Field(default_factory=dict)


class ToolCapableModel(Protocol):
    """Minimal model contract needed by :class:`atlas.agent.runtime.Agent`."""

    def complete(
        self, messages: list[ChatMessage], tools: list[ToolDefinition]
    ) -> ModelResponse: ...


class ToolResult(BaseModel):
    """Structured tool output returned to the model and retained in the run log."""

    text: str
    artifacts: list[str] = Field(default_factory=list)
    diff: str | None = None
    summary: str | None = None


def _clean_schema(node: Any, *, drop_description: bool = True) -> Any:
    """Strip Pydantic JSON-schema noise (titles, null anyOf unions, model docstrings)."""
    if isinstance(node, list):
        return [_clean_schema(item, drop_description=False) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title" and isinstance(value, str):
            continue
        if key == "description" and drop_description and isinstance(value, str):
            continue
        if key == "default" and value is None:
            continue
        if key in _SCHEMA_VALUES:
            out[key] = value
        elif key in _SCHEMA_MAPS and isinstance(value, dict):
            sub_drop = key in {"$defs", "definitions"}
            cleaned_map = {
                name: _clean_schema(schema, drop_description=sub_drop)
                for name, schema in value.items()
            }
            if cleaned_map or key == "properties":
                out[key] = cleaned_map
        elif key == "anyOf" and isinstance(value, list):
            non_null = [item for item in value if item != {"type": "null"}]
            if len(non_null) == 1 and len(non_null) < len(value):
                merged = _clean_schema(non_null[0], drop_description=False)
                if isinstance(merged, dict):
                    for mk, mv in merged.items():
                        out.setdefault(mk, mv)
                    continue
            out[key] = [_clean_schema(item, drop_description=False) for item in value]
        else:
            out[key] = _clean_schema(value, drop_description=False)
    return out


_without_titles = _clean_schema


class Tool(ABC):
    """A local capability that can be safely advertised to an agent."""

    name: str
    description: str
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    # Capability family used by tool profiles; "satellite" marks remote-sensing tools.
    domain: ClassVar[str] = ""

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
        """Return the schema advertised to the model, without Pydantic noise."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=_clean_schema(self.input_model.model_json_schema()),
        )

    @abstractmethod
    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        """Validate and execute a tool call within ``store``."""
        raise NotImplementedError


class UnknownToolError(KeyError):
    """A requested tool name that is not on the registry."""

    def __init__(self, name: str, available: Sequence[str] = ()) -> None:
        super().__init__(f"Unknown tool: {name}")
        self.name = name
        self.available = tuple(available)


class ToolRegistry:
    """Explicit allow-list of tools available to an agent session."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._definitions_cache: list[ToolDefinition] | None = None

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        self._definitions_cache = None

    def replace(self, tool: Tool) -> None:
        """Register ``tool``, replacing any tool already stored under its name."""
        self._tools[tool.name] = tool
        self._definitions_cache = None

    @property
    def definitions(self) -> list[ToolDefinition]:
        if self._definitions_cache is None:
            self._definitions_cache = [tool.definition() for tool in self._tools.values()]
        return list(self._definitions_cache)

    @property
    def names(self) -> list[str]:
        """Registered tool names, in registration order."""
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        """Return the registered tool, or None for an unknown name."""
        return self._tools.get(name)

    def execute(
        self, name: str, arguments: dict[str, Any], store: LocalArtifactStore
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(name, self.names)
        return tool.run(arguments, store)


def discover_tool_classes(packages: Sequence[str]) -> list[type[Tool]]:
    """Import every module directly inside ``packages`` and collect concrete tools."""
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

    # Private module so discovery does not call cls(). This tool needs the live registry.
    from atlas.agent.tools._list_tools import ListToolsTool

    registry.register(ListToolsTool(lambda: registry.definitions))

    from atlas.models.registry import register_model_tools

    register_model_tools(registry)
    return registry
