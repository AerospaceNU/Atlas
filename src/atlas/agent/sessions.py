"""Saved-session bookkeeping: listing, metadata, history repair, replay.

Stdlib, pydantic and contracts only; this module never imports the session
or the runtime. Nothing here executes caller-supplied code.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from atlas.agent.contracts import ChatMessage, Role
from atlas.agent.layout import project_artifact_dir, project_session_dir, project_sessions_dir

INTERRUPTED_NOTE = "interrupted: the process stopped before this call returned"
SKIP_PREFIXES = ("<untrusted_project_instructions", "<session_goal>")
_SESSION_ID = re.compile(r"^[0-9a-f]{12}$")
_PROMPT_CHARS = 120
_SUMMARY_CHARS = 80
_SUMMARY_KEYS = (
    "path",
    "file_path",
    "pattern",
    "command",
    "query",
    "url",
    "name",
    "stack",
    "scene",
)


class SessionSummary(BaseModel):
    """One saved session as listed from its ``meta.json``."""

    id: str
    saved: str = ""
    model: str = ""
    messages: int = 0
    prompt: str = ""
    interrupted: bool = False
    stopped_for_limit: bool = False


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _is_context(message: ChatMessage) -> bool:
    content = message.content or ""
    return content.startswith(SKIP_PREFIXES)


def first_prompt(messages: list[ChatMessage], *, limit: int = _PROMPT_CHARS) -> str:
    """Return the first real user request, whitespace-collapsed and cut."""
    for message in messages:
        if message.role is not Role.USER or message.tool_observation:
            continue
        if _is_context(message) or not message.content:
            continue
        return _cut(_collapse(message.content), limit)
    return ""


def write_meta(
    directory: Path,
    *,
    session_id: str,
    model: str,
    messages: int,
    saved: str,
    interrupted: bool,
    stopped_for_limit: bool,
    prompt: str = "",
) -> None:
    """Atomically write ``meta.json``, keeping the original ``created`` stamp.

    The caller passes an already-redacted ``prompt``.
    """
    target = directory / "meta.json"
    created = ""
    with contextlib.suppress(OSError, ValueError):
        old = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(old, dict):
            created = str(old.get("created") or "")
    payload = {
        "id": session_id,
        "model": model,
        "created": created or saved,
        "saved": saved,
        "messages": messages,
        "prompt": prompt,
        "interrupted": interrupted,
        "stopped_for_limit": stopped_for_limit,
    }
    temporary = directory / ".meta.json.tmp"
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, target)


def _summary(directory: Path) -> SessionSummary:
    meta: dict[str, Any] = {}
    with contextlib.suppress(OSError, ValueError):
        loaded = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            meta = loaded
    saved = meta.get("saved")
    if not isinstance(saved, str) or not saved:
        saved = ""
        with contextlib.suppress(OSError):
            stamp = (directory / "state.json").stat().st_mtime
            saved = datetime.fromtimestamp(stamp, UTC).isoformat()

    def text(key: str) -> str:
        value = meta.get(key)
        return value if isinstance(value, str) else ""

    count = meta.get("messages")
    return SessionSummary(
        id=directory.name,
        saved=saved,
        model=text("model"),
        messages=count if isinstance(count, int) and not isinstance(count, bool) else 0,
        prompt=text("prompt"),
        interrupted=meta.get("interrupted") is True,
        stopped_for_limit=meta.get("stopped_for_limit") is True,
    )


def list_sessions(workspace: Path, *, limit: int = 50) -> list[SessionSummary]:
    """List sessions holding a ``state.json``, newest first, from meta only."""
    found: list[SessionSummary] = []
    try:
        for entry in project_sessions_dir(workspace).iterdir():
            with contextlib.suppress(OSError):
                if entry.is_dir() and (entry / "state.json").is_file():
                    found.append(_summary(entry))
    except OSError:
        return []
    found.sort(key=lambda item: item.id)
    found.sort(key=lambda item: item.saved, reverse=True)
    return found[: max(0, limit)]


def newest_session(workspace: Path, *, exclude: str | None = None) -> str | None:
    """Return the newest saved session id other than ``exclude``."""
    for item in list_sessions(workspace, limit=1_000):
        if item.id != exclude:
            return item.id
    return None


def _has_regular_file(root: Path) -> bool:
    return any(names for _, _, names in os.walk(root))


def discard_unused_session(workspace: Path, session_id: str) -> bool:
    """Remove a session directory that never saved anything; never raises."""
    if not _SESSION_ID.match(session_id):
        return False
    try:
        session_dir = project_session_dir(workspace, session_id)
        artifact_dir = project_artifact_dir(workspace, session_id)
        if (session_dir / "state.json").exists():
            return False
        if artifact_dir.is_dir() and _has_regular_file(artifact_dir):
            return False
        removed = False
        for path in (session_dir, artifact_dir):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
                removed = True
        return removed
    except OSError:
        return False


def _is_image_observation(message: ChatMessage) -> bool:
    return message.role is Role.USER and message.tool_observation


def repair_history(messages: list[ChatMessage]) -> int:
    """Answer dangling tool calls and drop orphan results, in place.

    Returns:
        The number of messages inserted plus dropped; zero when valid.
    """
    changed = 0
    index = 0
    answerable: set[str] = set()
    while index < len(messages):
        message = messages[index]
        if message.role is Role.TOOL:
            if message.tool_call_id in answerable:
                index += 1
            else:
                del messages[index]
                changed += 1
            continue
        if message.role is Role.ASSISTANT and message.tool_calls:
            wanted = [call.id for call in message.tool_calls]
            answerable = set(wanted)
            end = index + 1
            answered: set[str] = set()
            while end < len(messages) and (
                messages[end].role is Role.TOOL or _is_image_observation(messages[end])
            ):
                if messages[end].role is Role.TOOL:
                    answered.add(messages[end].tool_call_id or "")
                end += 1
            insert_at = index + 1
            for position in range(index + 1, end):
                if messages[position].role is Role.TOOL:
                    insert_at = position + 1
            missing = [call_id for call_id in wanted if call_id not in answered]
            for offset, call_id in enumerate(missing):
                messages.insert(
                    insert_at + offset,
                    ChatMessage(role=Role.TOOL, tool_call_id=call_id, content=INTERRUPTED_NOTE),
                )
            changed += len(missing)
            index += 1
            continue
        if not _is_image_observation(message):
            answerable = set()
        index += 1
    return changed


def call_summary(name: str, arguments: dict[str, Any]) -> str:
    """Return a one-line ``name: value`` label for a tool call, at most 80 chars."""
    try:
        value: object = None
        for key in _SUMMARY_KEYS:
            if key in arguments and arguments[key] not in (None, ""):
                value = arguments[key]
                break
        if value is None:
            value = next((item for item in arguments.values() if isinstance(item, str)), None)
        if value is None:
            return _cut(_collapse(str(name)), _SUMMARY_CHARS)
        return _cut(_collapse(f"{name}: {value}"), _SUMMARY_CHARS)
    except Exception:
        return _cut(str(name), _SUMMARY_CHARS)


def replay_items(
    messages: list[ChatMessage], *, limit: int = 40, max_chars: int = 1500
) -> tuple[list[dict[str, str]], int]:
    """Flatten history into display items; returns the newest ``limit`` and the omitted count."""
    items: list[dict[str, str]] = []
    for message in messages:
        if message.role is Role.SYSTEM or _is_image_observation(message):
            continue
        if message.role is Role.USER:
            if _is_context(message) or not message.content:
                continue
            items.append({"role": "user", "text": _cut(message.content, max_chars)})
        elif message.role is Role.ASSISTANT:
            if message.content:
                items.append({"role": "assistant", "text": _cut(message.content, max_chars)})
            for call in message.tool_calls:
                items.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "summary": call_summary(call.name, call.arguments),
                    }
                )
    keep = max(0, limit)
    omitted = max(0, len(items) - keep)
    return (items[omitted:] if keep else []), omitted
