from __future__ import annotations

from datetime import date

from atlas.data.firms import _nrt_window


def test_nrt_window_clamps_end_to_yesterday() -> None:
    start, end = _nrt_window(
        date(2026, 9, 11),
        date(2026, 9, 18),
        today_utc=date(2026, 9, 18),
    )
    assert start == date(2026, 9, 11)
    assert end == date(2026, 9, 17)


def test_nrt_window_today_only_is_empty_after_clamp() -> None:
    start, end = _nrt_window(
        date(2026, 9, 18),
        date(2026, 9, 18),
        today_utc=date(2026, 9, 18),
    )
    assert start > end
