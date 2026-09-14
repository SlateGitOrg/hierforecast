"""Input data-quality gate.

Energy metering data is full of defects that do not look like defects: a
half-hour of missing settlement data reads as a demand dip, a meter reset reads
as a catastrophic load loss, and a local-time index silently contains 23 hours
one Sunday in spring and 25 hours one Sunday in autumn.  Each of these quietly
poisons a forecast rather than crashing it, so the gate FAILS THE RUN rather
than returning a cleaned series: a forecast built on history you have not
inspected is worse than no forecast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta


class DataQualityError(Exception):
    pass


@dataclass(frozen=True)
class Issue:
    kind: str
    index: int
    detail: str


# A drop of more than 80% of the local level in a single hour is not weather.
# Aggregate demand has strong inertia: the sharpest legitimate national ramp in
# the generator, and in published grid data, is well under 25% per hour.
RESET_DROP_FRACTION = 0.8
RECOVERY_TOLERANCE = 0.5


def check_index(timestamps: list[datetime], step_hours: int = 1) -> list[Issue]:
    """Gaps and duplicates in a supposedly regular index.

    A duplicated local hour in autumn and a missing one in spring are the DST
    signature; they are reported as their own kind because the remedy differs
    (aggregate the duplicate, interpolate the gap) from a genuine data outage.
    """
    issues: list[Issue] = []
    step = timedelta(hours=step_hours)
    seen: dict[datetime, int] = {}
    for i, ts in enumerate(timestamps):
        if ts in seen:
            issues.append(Issue("dst_duplicate_hour", i, "%s repeats index %d" % (ts.isoformat(), seen[ts])))
        else:
            seen[ts] = i
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        if gap == step or gap == timedelta(0):
            continue
        if gap == 2 * step:
            issues.append(Issue("dst_missing_hour", i, "single hour skipped at %s" % timestamps[i].isoformat()))
        elif gap > step:
            issues.append(
                Issue("missing_interval", i, "gap of %.0f hours before %s" % (gap.total_seconds() / 3600.0, timestamps[i].isoformat()))
            )
        else:
            issues.append(Issue("non_monotonic", i, "index goes backwards at %s" % timestamps[i].isoformat()))
    return issues


def check_values(values: list[float]) -> list[Issue]:
    issues: list[Issue] = []
    for i, v in enumerate(values):
        if v is None or not math.isfinite(v):
            issues.append(Issue("non_finite", i, "value is %r" % (v,)))
        elif v < 0.0:
            issues.append(Issue("negative_demand", i, "value is %.1f" % v))
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return issues
    for i in range(1, len(values)):
        prev, cur = values[i - 1], values[i]
        if prev is None or cur is None or not math.isfinite(prev) or not math.isfinite(cur):
            continue
        if prev <= 0.0:
            continue
        if cur < prev * (1.0 - RESET_DROP_FRACTION):
            # A reset is distinguished from a genuine (rare) collapse by the
            # series jumping straight back to its old level afterwards.
            recovered = (
                i + 1 < len(values)
                and values[i + 1] is not None
                and math.isfinite(values[i + 1])
                and abs(values[i + 1] - prev) < RECOVERY_TOLERANCE * prev
            )
            kind = "meter_reset" if recovered else "level_collapse"
            issues.append(Issue(kind, i, "%.1f -> %.1f" % (prev, cur)))
    return issues


def gate(timestamps: list[datetime], values: list[float], step_hours: int = 1) -> list[Issue]:
    if len(timestamps) != len(values):
        raise DataQualityError("timestamp/value length mismatch: %d vs %d" % (len(timestamps), len(values)))
    return check_index(timestamps, step_hours) + check_values(values)


def assert_clean(timestamps: list[datetime], values: list[float], step_hours: int = 1) -> None:
    issues = gate(timestamps, values, step_hours)
    if issues:
        head = "; ".join("%s@%d (%s)" % (i.kind, i.index, i.detail) for i in issues[:5])
        raise DataQualityError("%d data-quality issue(s): %s" % (len(issues), head))


def hourly_index(start: datetime, hours: int) -> list[datetime]:
    return [start + timedelta(hours=h) for h in range(hours)]
