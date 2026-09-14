"""Synthetic regional electricity demand with planted, known structure.

Everything the forecasters are supposed to find is planted here explicitly:

  * a shared national temperature driver (so regional errors are correlated -
    the reason MinT's covariance term beats bottom-up),
  * a genuine daily shape and weekday/weekend level shift,
  * a piecewise-linear temperature response (heating below HEAT_BASE_C,
    cooling above COOL_BASE_C) with per-node coefficients,
  * region-specific AR(1) noise plus independent sub-region noise,
  * and **cold snaps**: rare, multi-hour, nationally synchronised surges that
    make the demand error distribution genuinely right-skewed.

The cold snaps matter more than they look.  They are unforecastable from the
features available, so every model's residuals inherit their skew - and a model
that assumes Gaussian errors will place its 95th percentile too low.  That is
not a contrived flaw; it is the single most common way a demand forecast that
looks excellent on RMSE turns out to under-provision capacity.

The bottom level is generated first and every aggregate is defined as the exact
sum of its leaves, so the *data* is coherent by construction.  Any incoherence
observed later is therefore entirely the forecaster's doing, which is what the
reconciliation tests need.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .hierarchy import Hierarchy, default_hierarchy

HOURS_PER_DAY = 24
HEAT_BASE_C = 15.0  # below this, electric heating load switches on
COOL_BASE_C = 21.0  # above this, air conditioning load switches on

# Cold snaps: roughly one every four days, lasting most of a day, so they occupy
# a little under 10% of all hours.  That frequency is not arbitrary - it is the
# level at which the skew bites precisely in the 0.95-0.99 band a capacity
# planner uses.  Make them rarer and the surge hides above the 99th percentile;
# make them commoner and they stop being a tail and become the mean.
SNAP_PROB_PER_DAY = 1.0 / 4.0
SNAP_MIN_HOURS = 8
SNAP_MAX_HOURS = 20


@dataclass
class NodeParams:
    base_mw: float
    heat_mw_per_degc: float
    cool_mw_per_degc: float
    weekend_factor: float
    peak_evening: float
    peak_morning: float
    noise_sd: float
    temp_offset_c: float


@dataclass
class Dataset:
    hierarchy: Hierarchy
    hours: int
    hour_of_day: list[int]
    day_of_week: list[int]
    temperature: dict[str, list[float]]
    demand: dict[str, list[float]]  # every node, aggregates included

    def node_vector(self, t: int) -> list[float]:
        return [self.demand[name][t] for name in self.hierarchy.names]

    def slice(self, start: int, stop: int) -> "Dataset":
        return Dataset(
            hierarchy=self.hierarchy,
            hours=stop - start,
            hour_of_day=self.hour_of_day[start:stop],
            day_of_week=self.day_of_week[start:stop],
            temperature={k: v[start:stop] for k, v in self.temperature.items()},
            demand={k: v[start:stop] for k, v in self.demand.items()},
        )


def _daily_shape(hour: int, morning: float, evening: float) -> float:
    """Two-peak load shape, normalised to roughly 1.0 on average.

    Real demand is bimodal (breakfast and the evening ramp); a single sinusoid
    would be trivially fittable by the harmonic regression and would flatter it.
    """
    m = math.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
    e = math.exp(-0.5 * ((hour - 19.0) / 2.5) ** 2)
    night = -0.18 * math.exp(-0.5 * ((hour - 4.0) / 3.0) ** 2)
    return 1.0 + morning * m + evening * e + night


def _make_params(rng: random.Random, names: list[str]) -> dict[str, NodeParams]:
    params = {}
    for i, name in enumerate(names):
        params[name] = NodeParams(
            base_mw=700.0 + 500.0 * rng.random(),
            heat_mw_per_degc=18.0 + 14.0 * rng.random(),
            cool_mw_per_degc=6.0 + 10.0 * rng.random(),
            weekend_factor=0.86 + 0.06 * rng.random(),
            peak_evening=0.18 + 0.10 * rng.random(),
            peak_morning=0.10 + 0.08 * rng.random(),
            # Sub-region noise scales with size; 2.5% of base is in the range
            # a real substation meter shows once weather is accounted for.
            noise_sd=0.025 * (700.0 + 500.0 * rng.random()),
            temp_offset_c=-3.0 + 6.0 * (i / max(1, len(names) - 1)),
        )
    return params


def generate(
    hours: int = 24 * 140,
    seed: int = 20240,
    hierarchy: Hierarchy | None = None,
    region_noise_sd: float = 22.0,
    ar_rho: float = 0.55,
    snap_mw_per_node: float = 150.0,
    temp_measurement_sd: float = 3.0,
) -> Dataset:
    h = hierarchy or default_hierarchy()
    rng = random.Random(seed)
    params = _make_params(rng, h.bottom)

    # National temperature: annual swing plus a diurnal swing plus AR(1) weather
    # persistence.  Shared across every node, which is exactly what makes the
    # bottom-level forecast errors correlated.
    nat_temp = []
    w = 0.0
    for t in range(hours):
        day = t / 24.0
        seasonal = 10.0 + 9.0 * math.sin(2 * math.pi * (day - 100.0) / 365.0)
        diurnal = 4.0 * math.sin(2 * math.pi * ((t % 24) - 9.0) / 24.0)
        w = 0.86 * w + rng.gauss(0.0, 1.6)
        nat_temp.append(seasonal + diurnal + w)

    # Nationally synchronised cold-snap surge, in MW per bottom node.
    snap = [0.0] * hours
    d = 0
    while d < hours // 24:
        if rng.random() < SNAP_PROB_PER_DAY:
            length = rng.randint(SNAP_MIN_HOURS, SNAP_MAX_HOURS)
            intensity = snap_mw_per_node * (0.5 + rng.random())
            start = d * 24 + rng.randint(0, 6)
            for k in range(length):
                t = start + k
                if t < hours:
                    # Ramp up and down so the surge is not a rectangle.
                    ramp = math.sin(math.pi * (k + 0.5) / length)
                    snap[t] += intensity * ramp
            d += 2  # snaps do not immediately repeat
        d += 1

    region_state = {r: 0.0 for r in h.aggregates if r != "NAT"}
    temperature: dict[str, list[float]] = {name: [] for name in h.bottom}
    bottom: dict[str, list[float]] = {name: [] for name in h.bottom}
    hod, dow = [], []

    region_of = {}
    for name in h.bottom:
        node = next(nd for nd in h.nodes if nd.name == name)
        region_of[name] = node.parent

    for t in range(hours):
        hour = t % 24
        day = (t // 24) % 7
        hod.append(hour)
        dow.append(day)
        for r in region_state:
            region_state[r] = ar_rho * region_state[r] + rng.gauss(0.0, region_noise_sd)
        for name in h.bottom:
            p = params[name]
            temp = nat_temp[t] + p.temp_offset_c + rng.gauss(0.0, 0.4)
            # The model sees a NOISY reading from one local weather station;
            # demand responds to the true temperature.  This errors-in-variables
            # gap is why aggregate-level models are better estimated than
            # bottom-level ones - averaging 12 stations cuts the measurement
            # noise by sqrt(12) - and it is the mechanism that gives MinT
            # something to exploit that bottom-up cannot.
            temperature[name].append(temp + rng.gauss(0.0, temp_measurement_sd))
            hdd = max(0.0, HEAT_BASE_C - temp)
            cdd = max(0.0, temp - COOL_BASE_C)
            shape = _daily_shape(hour, p.peak_morning, p.peak_evening)
            week = p.weekend_factor if day >= 5 else 1.0
            mw = (
                p.base_mw * shape * week
                + p.heat_mw_per_degc * hdd
                + p.cool_mw_per_degc * cdd
                + region_state[region_of[name]]
                + snap[t] * (0.6 + 0.8 * rng.random())
                + rng.gauss(0.0, p.noise_sd)
            )
            bottom[name].append(mw)

    demand: dict[str, list[float]] = dict(bottom)
    for name in h.aggregates:
        leaves = h.leaves_under(name)
        demand[name] = [sum(bottom[l][t] for l in leaves) for t in range(hours)]
    for name in h.aggregates:
        temperature[name] = [
            sum(temperature[l][t] for l in h.leaves_under(name)) / len(h.leaves_under(name))
            for t in range(hours)
        ]

    return Dataset(h, hours, hod, dow, temperature, demand)
