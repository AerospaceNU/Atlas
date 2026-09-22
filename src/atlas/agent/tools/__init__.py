from __future__ import annotations

import importlib
from typing import Any

# Concrete tools only. Framework contracts (Role, Tool, ToolRegistry,
# ToolResult, default_registry) live in atlas.agent.contracts. Discovery lives
# in atlas.agent.contracts.discover_tool_classes. Importing this package must
# not grant a capability, so every re-export below is lazy.
_LAZY_EXPORTS = {
    "ReadFileInput": "atlas.agent.tools.read_file",
    "ReadFileTool": "atlas.agent.tools.read_file",
}


def __getattr__(name: str) -> Any:
    """Re-export concrete tools lazily so importing this package stays inert."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)
