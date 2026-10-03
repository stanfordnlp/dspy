"""Conservative parsing of advisory HTTP retry hints."""

import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def finite_seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        value = float(value)
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def retry_after_seconds(value):
    seconds = finite_seconds(value)
    if seconds is not None:
        return seconds
    if not isinstance(value, str):
        return None
    try:
        when = parsedate_to_datetime(value)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None
