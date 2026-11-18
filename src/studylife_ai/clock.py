"""The single source of "now" for prompts and listings.

StudyLife stores naive local timestamps (see docs/decisions.md), so every comparison against
them has to use a naive local "now" in the same zone. `Settings.timezone` names that zone; unset,
the process's own local time is used (the behavior before the setting existed).
"""

from datetime import datetime
from zoneinfo import ZoneInfo


def local_now(timezone: str | None) -> datetime:
    """Naive local wall-clock time in `timezone` (IANA name), or process-local time if unset."""
    if not timezone:
        return datetime.now()
    return datetime.now(ZoneInfo(timezone)).replace(tzinfo=None)
