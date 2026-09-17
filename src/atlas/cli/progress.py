"""Search/pull progress: Rich TUI on a TTY, line logs otherwise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TextIO

from rich.live import Live
from rich.table import Table


@dataclass
class _SourceState:
    status: str = "searching"
    scenes: int = 0
    saved: int = 0
    error: str | None = None


class ProgressReporter(Protocol):
    """Per-source status updates for the data CLI."""

    def source_start(self, name: str) -> None:
        """Mark a source as searching."""

    def source_search_done(self, name: str, *, scenes: int, error: str | None) -> None:
        """Record search completion (or a captured source error)."""

    def source_pull(self, name: str, *, saved: int, total: int) -> None:
        """Update download counts while pulling previews."""

    def source_done(self, name: str) -> None:
        """Mark a source fully finished."""

    def close(self) -> None:
        """Release any TUI resources."""


class LogReporter:
    """One line per status change. Used when stdout is not a TTY."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def source_start(self, name: str) -> None:
        print(f"{name}: searching", file=self._stream, flush=True)

    def source_search_done(self, name: str, *, scenes: int, error: str | None) -> None:
        if error:
            print(f"{name}: error {error}", file=self._stream, flush=True)
        else:
            print(f"{name}: {scenes} scenes", file=self._stream, flush=True)

    def source_pull(self, name: str, *, saved: int, total: int) -> None:
        print(f"{name}: pulling {saved}/{total}", file=self._stream, flush=True)

    def source_done(self, name: str) -> None:
        print(f"{name}: done", file=self._stream, flush=True)

    def close(self) -> None:
        return None


class RichReporter:
    """Live table of per-source search and pull status."""

    def __init__(self) -> None:
        self._states: dict[str, _SourceState] = {}
        self._live = Live(self._render(), refresh_per_second=8)
        self._live.start()

    def source_start(self, name: str) -> None:
        self._states[name] = _SourceState()
        self._refresh()

    def source_search_done(self, name: str, *, scenes: int, error: str | None) -> None:
        state = self._states.setdefault(name, _SourceState())
        state.scenes = scenes
        state.error = error
        state.status = "error" if error else "pulling"
        self._refresh()

    def source_pull(self, name: str, *, saved: int, total: int) -> None:
        state = self._states.setdefault(name, _SourceState())
        state.saved = saved
        state.scenes = total
        state.status = "pulling"
        self._refresh()

    def source_done(self, name: str) -> None:
        state = self._states.setdefault(name, _SourceState())
        if state.error is None:
            state.status = "done"
        self._refresh()

    def close(self) -> None:
        self._live.stop()

    def _refresh(self) -> None:
        self._live.update(self._render())

    def _render(self) -> Table:
        table = Table(title="atlas data")
        table.add_column("Source")
        table.add_column("Status")
        table.add_column("Scenes", justify="right")
        table.add_column("Saved", justify="right")
        table.add_column("Error")
        for name, state in self._states.items():
            error = state.error or ""
            if len(error) > 60:
                error = error[:57] + "..."
            table.add_row(name, state.status, str(state.scenes), str(state.saved), error)
        return table


def pick_reporter(*, stream: TextIO) -> ProgressReporter:
    """Choose a Rich table on a TTY, otherwise line logs.

    Args:
        stream: Destination for log lines (ignored by the Rich reporter).
    """
    if not stream.isatty():
        return LogReporter(stream)
    return RichReporter()
