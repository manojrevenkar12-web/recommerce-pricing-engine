"""The company's day numbering: day 0 is 2024-08-01, the start of the data platform."""

from __future__ import annotations

from datetime import date, datetime, timedelta

EPOCH = date(2024, 8, 1)


def day(d: date | str) -> int:
    if isinstance(d, datetime):  # also pandas Timestamps
        d = d.date()
    d = d if isinstance(d, date) else date.fromisoformat(str(d)[:10])
    return (d - EPOCH).days


def to_date(n: int) -> date:
    return EPOCH + timedelta(days=int(n))
