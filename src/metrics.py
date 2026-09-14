"""Scoring rules, and the operational translation of the tail.

RMSE and MAPE score the *mean*.  A capacity decision is a decision about the
upper tail: the question is not "what will demand be" but "what number do we
have to be able to cover".  Pinball loss at quantile q is the proper scoring
rule for exactly that question, and it is asymmetric - at q=0.95 being short by
1 MW costs 19x more than being long by 1 MW, which is the correct economics for
a grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def pinball(actual: float, forecast: float, q: float) -> float:
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must be strictly inside (0,1)")
    d = actual - forecast
    return q * d if d >= 0 else (q - 1.0) * d


def pinball_loss(actuals: list[float], forecasts: list[float], q: float) -> float:
    if len(actuals) != len(forecasts):
        raise ValueError("length mismatch")
    return sum(pinball(a, f, q) for a, f in zip(actuals, forecasts)) / len(actuals)


def rmse(actuals: list[float], forecasts: list[float]) -> float:
    return math.sqrt(sum((a - f) ** 2 for a, f in zip(actuals, forecasts)) / len(actuals))


def mae(actuals: list[float], forecasts: list[float]) -> float:
    return sum(abs(a - f) for a, f in zip(actuals, forecasts)) / len(actuals)


def mape(actuals: list[float], forecasts: list[float]) -> float:
    """Percent, not fraction.  Undefined at zero demand, which never occurs in
    this generator; a real feed would need a MASE fallback."""
    tot = 0.0
    for a, f in zip(actuals, forecasts):
        if a == 0.0:
            raise ValueError("MAPE is undefined for zero actuals")
        tot += abs((a - f) / a)
    return 100.0 * tot / len(actuals)


@dataclass
class CapacityOutcome:
    """What a planner actually experiences if capacity is set to the forecast."""

    hours: int
    exceedance_hours: int
    exceedance_rate: float
    mean_shortfall_mw: float          # averaged over ALL hours
    mean_shortfall_when_short_mw: float
    worst_shortfall_mw: float
    unserved_mwh: float
    mean_headroom_mw: float           # cost of over-building, same units

    @property
    def exceedance_pct(self) -> float:
        return 100.0 * self.exceedance_rate


def capacity_outcome(actuals: list[float], capacity: list[float]) -> CapacityOutcome:
    """Score a capacity plan in MW, not in loss units.

    Hourly data, so 1 MW of shortfall sustained for 1 hour is 1 MWh unserved -
    the conversion is deliberately trivial and stated rather than hidden.
    """
    shorts = [max(0.0, a - c) for a, c in zip(actuals, capacity)]
    over = [max(0.0, c - a) for a, c in zip(actuals, capacity)]
    n = len(actuals)
    hit = [s for s in shorts if s > 0.0]
    return CapacityOutcome(
        hours=n,
        exceedance_hours=len(hit),
        exceedance_rate=len(hit) / n,
        mean_shortfall_mw=sum(shorts) / n,
        mean_shortfall_when_short_mw=(sum(hit) / len(hit)) if hit else 0.0,
        worst_shortfall_mw=max(shorts) if shorts else 0.0,
        unserved_mwh=sum(shorts),
        mean_headroom_mw=sum(over) / n,
    )
