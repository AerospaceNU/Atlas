from __future__ import annotations

from typing import Protocol, TextIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressReporter(Protocol):
    """Per-source status updates for the data CLI."""

    def source_start(self, name: str) -> None:
        """Mark a source as searching."""

    def source_search_done(self, name: str, *, scenes: int, error: str | None) -> None:
        """Record search completion (or a captured source error)."""

    def source_pull(self, name: str, *, saved: int, total: int) -> None:
        """Update download counts while pulling rasters."""

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
    """Live progress bar with elapsed time and remaining-time estimate."""

    def __init__(self, stream: TextIO) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            console=Console(file=stream, highlight=False),
            expand=True,
            transient=False,
        )
        self._search_id: TaskID | None = None
        self._pull_id: TaskID | None = None
        self._n_sources = 0
        self._n_searched = 0
        self._saved: dict[str, int] = {}
        self._totals: dict[str, int] = {}
        self._stopped = False
        self._progress.start()

    def source_start(self, name: str) -> None:
        self._n_sources += 1
        if self._search_id is None:
            self._search_id = self._progress.add_task("searching", total=self._n_sources)
        else:
            self._progress.update(self._search_id, total=self._n_sources, description="searching")

    def source_search_done(self, name: str, *, scenes: int, error: str | None) -> None:
        self._n_searched += 1
        label = f"search {name} failed" if error else f"search {name}"
        if self._search_id is None:
            self._search_id = self._progress.add_task(label, total=max(1, self._n_sources))
        self._progress.update(
            self._search_id,
            completed=self._n_searched,
            total=max(self._n_sources, self._n_searched),
            description=label,
        )

    def source_pull(self, name: str, *, saved: int, total: int) -> None:
        self._saved[name] = saved
        self._totals[name] = total
        overall_saved = sum(self._saved.values())
        overall_total = sum(self._totals.values())
        label = f"downloading {name}"
        if self._pull_id is None:
            self._pull_id = self._progress.add_task(label, total=max(overall_total, 1))
        self._progress.update(
            self._pull_id,
            completed=overall_saved,
            total=max(overall_total, 1),
            description=label,
        )

    def source_done(self, name: str) -> None:
        if self._pull_id is not None:
            overall_saved = sum(self._saved.values())
            overall_total = sum(self._totals.values())
            self._progress.update(
                self._pull_id,
                completed=overall_saved,
                total=max(overall_total, overall_saved, 1),
                description="downloading",
            )

    def close(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._progress.stop()


def pick_reporter(*, stream: TextIO) -> ProgressReporter:
    """Choose a progress bar on a TTY, otherwise line logs.

    Args:
        stream: Destination for the bar or log lines.
    """
    if not stream.isatty():
        return LogReporter(stream)
    return RichReporter(stream)
