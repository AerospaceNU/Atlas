from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

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


def test_goes_buckets_follow_the_operational_dates() -> None:
    from atlas.data.goes import _buckets_for

    assert _buckets_for("east", date(2024, 7, 1), date(2024, 7, 3)) == ["noaa-goes16"]
    assert _buckets_for("east", date(2025, 4, 7), date(2025, 4, 8)) == ["noaa-goes19"]
    assert _buckets_for("east", date(2025, 4, 6), date(2025, 4, 8)) == [
        "noaa-goes16",
        "noaa-goes19",
    ]
    assert _buckets_for("west", date(2024, 7, 1), date(2024, 7, 3)) == ["noaa-goes18"]
    assert _buckets_for("west", date(2020, 7, 1), date(2020, 7, 3)) == ["noaa-goes17"]
    assert _buckets_for("west", date(2015, 7, 1), date(2015, 7, 3)) == ["noaa-goes15"]
    assert _buckets_for("east", date(2015, 7, 1), date(2015, 7, 3)) == []
    assert _buckets_for("west", date(2009, 7, 1), date(2009, 7, 3)) == []


def test_goes_window_with_no_bucket_is_an_error() -> None:
    from atlas.data.base import BBox
    from atlas.data.goes import _list_targets

    with pytest.raises(ValueError, match="no public east bucket"):
        _list_targets(
            BBox(west=-100.0, south=30.0, east=-90.0, north=40.0),
            date(2010, 1, 1),
            date(2010, 1, 2),
        )


def test_goes_handover_window_lists_both_buckets() -> None:
    from atlas.data.base import BBox
    from atlas.data.goes import _list_targets

    targets = _list_targets(
        BBox(west=-100.0, south=30.0, east=-90.0, north=40.0),
        date(2025, 4, 6),
        date(2025, 4, 8),
    )
    assert {bucket for bucket, _side, _product in targets} == {"noaa-goes16", "noaa-goes19"}
