from __future__ import annotations

from datetime import UTC, date, datetime

from atlas.data.goes import _hours_in_range


def test_hours_in_range_drops_incomplete_and_future_hours() -> None:
    now = datetime(2026, 9, 18, 8, 15, tzinfo=UTC)
    hours = _hours_in_range(date(2026, 9, 17), date(2026, 9, 18), now=now)
    assert hours[0] == (2026, 260, 0)
    assert hours[-1] == (2026, 261, 7)
    assert (2026, 261, 8) not in hours
    assert (2026, 261, 21) not in hours


def test_hours_in_range_empty_when_window_is_still_in_the_future() -> None:
    now = datetime(2026, 9, 18, 0, 10, tzinfo=UTC)
    hours = _hours_in_range(date(2026, 9, 18), date(2026, 9, 18), now=now)
    assert hours == []
