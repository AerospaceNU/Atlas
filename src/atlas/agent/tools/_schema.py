"""Shared helpers that shrink the JSON schema a tool advertises (private; discovery skips it).

Every tool definition is sent with every request, so a definition stays within
``MAX_DEFINITION_CHARS`` characters. The measure is ``len(json.dumps(definition.model_dump()))``
(Python's default separators, see ``definition_chars``), which is stricter than the compact wire
form. ``tests/test_agent_runtime.py`` enforces it over ``default_registry()``. Validation still
runs on the full input model.
"""

from __future__ import annotations

import json
from typing import Any

from atlas.agent.contracts import ToolDefinition

MAX_DEFINITION_CHARS = 1600


def definition_chars(definition: ToolDefinition) -> int:
    """Return the size of a tool definition as json.dumps of its model dump."""
    return len(json.dumps(definition.model_dump()))


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
        items = slim.get("items")
        if isinstance(items, dict) and items.get("additionalProperties") is True:
            slim["items"] = {k: v for k, v in items.items() if k != "additionalProperties"}
        props[name] = slim
    schema["properties"] = props
    return ToolDefinition(name=base.name, description=base.description, parameters=schema)
