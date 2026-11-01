"""Deterministic duration and report-time features; no use of the current clock.

Naive/date-only timestamps are interpreted as UTC. A supplied timestamp is a
report-time anchor, not proof of event onset. Only an explicitly ongoing phrase
("for ...", "... ধরে/dhore") supports an inferred interval ending at that
anchor. Multiple distinct durations are left unknown for human review.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import re

from ml.normalize import clean_text

_COUNTS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "এক": 1, "দুই": 2, "দু": 2, "তিন": 3, "চার": 4, "পাঁচ": 5,
    "ছয়": 6, "সাত": 7, "আট": 8, "নয়": 9, "দশ": 10, "এগারো": 11,
    "বারো": 12, "তেরো": 13, "চৌদ্দ": 14, "পনেরো": 15, "ষোলো": 16,
    "সতেরো": 17, "আঠারো": 18, "উনিশ": 19, "বিশ": 20,
    "ek": 1, "dui": 2, "du": 2, "tin": 3, "teen": 3, "char": 4,
    "pach": 5, "panch": 5, "choy": 6, "chhoy": 6, "sat": 7,
    "saat": 7, "at": 8, "aat": 8, "noy": 9, "dosh": 10, "dos": 10,
}
_COUNTS = {clean_text(key): value for key, value in _COUNTS.items()}
_UNITS = {
    "দিন": 24.0, "din": 24.0, "day": 24.0, "days": 24.0,
    "ঘণ্টা": 1.0, "ঘন্টা": 1.0, "ঘণ্টার": 1.0, "hour": 1.0, "hours": 1.0,
    "hr": 1.0, "hrs": 1.0, "ghonta": 1.0, "ghanta": 1.0, "gonta": 1.0,
    "মিনিট": 1/60, "minute": 1/60, "minutes": 1/60, "min": 1/60, "mins": 1/60,
    "সপ্তাহ": 168.0, "week": 168.0, "weeks": 168.0, "shoptaho": 168.0,
}
_NUM = r"\d+(?:\.\d+)?|" + "|".join(re.escape(key) for key in sorted(_COUNTS, key=len, reverse=True))
_UNIT = "|".join(re.escape(key) for key in sorted(_UNITS, key=len, reverse=True))
_DURATION = re.compile(
    rf"(?<![\w\u0980-\u09ff])(?P<for>for\s+)?(?P<number>{_NUM})\s*(?P<unit>{_UNIT})"
    r"(?:\s*(?P<ongoing>ধরেই|ধরে|ব্যাপী|dhore|dhorei))?(?![\w\u0980-\u09ff])"
)
_RANGE = re.compile(rf"\d+(?:\.\d+)?\s*[-–—−]\s*\d+(?:\.\d+)?\s*(?:{_UNIT})")


def _timestamp(value: str) -> datetime | None:
    if not isinstance(value, str):
        raise TypeError("timestamp must be a string")
    if not value.strip():
        return None
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-8601 timestamp {value!r}") from exc
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def extract_time_features(text: str, timestamp: str = "") -> dict:
    """Extract a supported duration and optional ongoing interval in hours/days.

    Missing values remain ``None``. Bare duration phrases are retained but do
    not imply an ongoing interval. Expressions such as "three days ago" and
    "after three days" are not treated as an outage duration. Numeric ranges
    and multiple distinct durations are conservatively unresolved.
    """
    normalized = clean_text(text)
    anchor = _timestamp(timestamp)
    result = {"duration_hours": None, "duration_days": None, "timestamp_utc": _iso(anchor),
              "interval_start_utc": None, "interval_end_utc": None, "ongoing": False,
              "confidence": 0.0, "matched_expressions": [], "errors": []}
    if _RANGE.search(normalized):
        result["errors"].append("A duration range is ambiguous; no point estimate was imputed.")
        return result
    durations = []
    for match in _DURATION.finditer(normalized):
        before, after = normalized[:match.start()].rstrip(), normalized[match.end():].lstrip()
        before_word = before.split()[-1] if before.split() else ""
        after_word = after.split()[0].strip(".,;:!?") if after.split() else ""
        if before.endswith(("-", "–", "—", "−")) or before_word in ("after", "পরে", "por") or after_word in ("ago", "আগে", "age", "পরে", "por"):
            continue
        # Do not read "twenty one days" as "one day" when compound counts are
        # outside this deliberately small explicit number-word dictionary.
        if before_word in _COUNTS:
            result["errors"].append("Unsupported compound number words in duration.")
            continue
        number_text = match.group("number")
        number = float(number_text) if re.fullmatch(r"\d+(?:\.\d+)?", number_text) else float(_COUNTS[number_text])
        hours = number * _UNITS[match.group("unit")]
        if not math.isfinite(hours):
            result["errors"].append("Duration is outside the supported finite numeric range.")
            continue
        durations.append((hours, bool(match.group("for") or match.group("ongoing"))))
        result["matched_expressions"].append(match.group())
    if not durations:
        return result
    values = {round(hours, 10) for hours, _ in durations}
    if len(values) != 1:
        result["errors"].append("Multiple distinct durations found; no duration was selected.")
        return result
    hours = durations[0][0]
    ongoing = any(value for _, value in durations)
    result.update(duration_hours=hours, duration_days=hours / 24, ongoing=ongoing,
                  confidence=1.0 if ongoing else 0.75)
    if ongoing and anchor is not None:
        try:
            start = anchor - timedelta(hours=hours)
        except OverflowError:
            result["errors"].append("Duration exceeds supported datetime range; no interval was inferred.")
        else:
            result["interval_start_utc"], result["interval_end_utc"] = _iso(start), _iso(anchor)
    return result


def time_pair_features(a: dict, b: dict) -> dict[str, float]:
    """Return numeric hours plus missing flags; zero never stands for unknown alone."""
    left, right = _timestamp(a.get("timestamp_utc") or ""), _timestamp(b.get("timestamp_utc") or "")
    time_missing = left is None or right is None
    duration_left, duration_right = a.get("duration_hours"), b.get("duration_hours")
    duration_missing = duration_left is None or duration_right is None
    starts = [_timestamp(row.get("interval_start_utc") or "") for row in (a, b)]
    ends = [_timestamp(row.get("interval_end_utc") or "") for row in (a, b)]
    interval_missing = any(value is None for value in (*starts, *ends))
    overlap = not interval_missing and max(starts) <= min(ends)
    interval_gap = max(0.0, (max(starts) - min(ends)).total_seconds() / 3600) if not interval_missing else 0.0
    return {
        "time_gap": abs((left - right).total_seconds()) / 3600 if not time_missing else 0.0,
        "time_missing": float(time_missing),
        "duration_gap": abs(float(duration_left) - float(duration_right)) if not duration_missing else 0.0,
        "duration_missing": float(duration_missing),
        "interval_overlap": float(overlap), "interval_missing": float(interval_missing),
        "interval_gap": interval_gap,
    }
