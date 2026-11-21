"""Cooperative cancellation and shared budget for one agent run.

Invariants: stdlib plus ``contracts.ModelCall`` only; this module never imports
the runtime, session, or model modules (they import it). With no scope
installed, ``cancel_requested()`` is false and behaviour is unchanged.

Python 3.12 threads start with an empty context: any thread that runs tool
code must call :func:`use_scope` itself to see the current scope.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from collections.abc import Callable, Iterator

from atlas.agent.contracts import ModelCall

__all__ = [
    "Cancelled",
    "ModelCancelled",
    "RunScope",
    "cancel_requested",
    "current_scope",
    "use_scope",
]


class Cancelled(Exception):
    """Cooperative stop requested for the current run."""


class ModelCancelled(Cancelled):
    """A model request was abandoned because of a cancel."""


class RunScope:
    """Cancel flag and token/spend budget shared by a run and its descendants.

    Charges propagate to every ancestor, so a parent sees the usage of all
    descendants. State is guarded by one lock because children charge from
    pool threads.
    """

    def __init__(
        self,
        should_cancel: Callable[[], bool] | None = None,
        *,
        max_tokens: int | None = None,
        max_spend_usd: float | None = None,
        parent: RunScope | None = None,
    ) -> None:
        self._should_cancel = should_cancel
        self.max_tokens = max_tokens
        self.max_spend_usd = max_spend_usd
        self.parent = parent
        self._lock = threading.Lock()
        self._calls: list[ModelCall] = []
        self._desc_calls_seen: list[ModelCall] = []
        self._desc_pending: list[ModelCall] = []
        self._desc_tools_seen = 0
        self._desc_tools_pending = 0

    def cancelled(self) -> bool:
        """Return true when this scope or any ancestor asks to stop."""
        if self._should_cancel is not None:
            try:
                if self._should_cancel():
                    return True
            except Exception:
                pass
        return self.parent is not None and self.parent.cancelled()

    def charge(self, call: ModelCall) -> None:
        """Record one model call here and in every ancestor."""
        with self._lock:
            self._calls.append(call)
        for ancestor in self._ancestors():
            with ancestor._lock:
                ancestor._desc_calls_seen.append(call)
                ancestor._desc_pending.append(call)

    def charge_tool_calls(self, n: int) -> None:
        """Record ``n`` executed tool calls in every ancestor; ``n <= 0`` is ignored."""
        if n <= 0:
            return
        for ancestor in self._ancestors():
            with ancestor._lock:
                ancestor._desc_tools_seen += n
                ancestor._desc_tools_pending += n

    def over_budget(self) -> bool:
        """Return true once own plus descendant usage, or an ancestor's, reaches a limit."""
        with self._lock:
            calls = [*self._calls, *self._desc_calls_seen]
        if self.max_tokens is not None:
            total = sum(c.usage.prompt_tokens + c.usage.completion_tokens for c in calls)
            if total >= self.max_tokens:
                return True
        if self.max_spend_usd is not None:
            spent = sum(c.usage.cost for c in calls if c.usage.cost is not None)
            if spent >= self.max_spend_usd:
                return True
        return self.parent is not None and self.parent.over_budget()

    def take_descendant_calls(self) -> list[ModelCall]:
        """Return descendant calls charged since the last take, oldest first."""
        with self._lock:
            taken, self._desc_pending = self._desc_pending, []
        return taken

    def take_descendant_tool_calls(self) -> int:
        """Return descendant tool calls charged since the last take."""
        with self._lock:
            taken, self._desc_tools_pending = self._desc_tools_pending, 0
        return taken

    def _ancestors(self) -> Iterator[RunScope]:
        node = self.parent
        while node is not None:
            yield node
            node = node.parent


_CURRENT: contextvars.ContextVar[RunScope | None] = contextvars.ContextVar(
    "atlas_run_scope", default=None
)


def current_scope() -> RunScope | None:
    """Return the scope installed in this context, if any."""
    return _CURRENT.get()


@contextlib.contextmanager
def use_scope(scope: RunScope | None) -> Iterator[RunScope | None]:
    """Install ``scope`` for the duration of the block."""
    token = _CURRENT.set(scope)
    try:
        yield scope
    finally:
        _CURRENT.reset(token)


def cancel_requested() -> bool:
    """Return true when the current scope, if any, has been cancelled."""
    scope = _CURRENT.get()
    return scope is not None and scope.cancelled()
