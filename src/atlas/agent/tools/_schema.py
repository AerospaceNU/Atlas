"""Shared helpers that shrink the JSON schema a tool advertises (private; discovery skips it).

Every tool definition is sent with every request, so a definition stays within
``MAX_DEFINITION_CHARS`` characters. Validation still runs on the full input model.
"""

from __future__ import annotations

from typing import Any

from atlas.agent.contracts import ToolDefinition

MAX_DEFINITION_CHARS = 1600


def slim_definition(base: ToolDefinition) -> ToolDefinition:
    """Return ``base`` without the model docstring, null branches and null defaults."""
    schema: dict[str, Any] = dict(base.parameters)
    schema.pop("description", None)
    props: dict[str, Any] = {}
    for name, prop in schema.get("properties", {}).items():
        slim = dict(prop)
        branches = [b for b in slim.pop("anyOf", []) if b.get("type") != "null"]
        if len(branches) == 1:
            slim.update(branches[0])
        elif branches:
            slim["anyOf"] = branches
        if slim.get("default") is None:
            slim.pop("default", None)
        props[name] = slim
    schema["properties"] = props
    return ToolDefinition(name=base.name, description=base.description, parameters=schema)
