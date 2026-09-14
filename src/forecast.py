"""Base forecasters: one independent model per node, each producing quantiles.

Independence is deliberate.  Fitting every node on its own is what a generic
implementation does, and it is what makes the forecasts incoherent - the whole
reason reconciliation exists.  This module produces that incoherent-but-honest
starting point; `reconcile.py` fixes it.

Two models are provided:

  * `SeasonalNaive` - last week's same hour.  The baseline that must always be
    computed and reported, so a clever model that fails to beat it cannot
    quietly ship.
  * `HarmonicRegression` - OLS on weather and calendar features via the normal
    equations, with a Gaussian predictive distribution whose width is the
    in-sample residual standard deviation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from .generator import COOL_BASE_C, HEAT_BASE_C, Dataset
from .linalg import Matrix, solve, transpose

_NORMAL = NormalDist()
HARMONICS = 3  # k=1..3 captures the two-peak daily shape; k=4+ added nothing


@dataclass(frozen=True)
class FeatureSpec:
    """Which regressors a base model is allowed to see.

    RICH is the correctly specified model.  COARSE omits the cooling response
    and the third harmonic, which leaves a small but systematic mean error -
    the realistic "slightly worse model" used in the loss-ranking experiment.
    """

    harmonics: int = HARMONICS
    use_cooling: bool = True
    use_weekend: bool = True


RICH = FeatureSpec()
COARSE = FeatureSpec(harmonics=1, use_cooling=False, use_weekend=True)


def z_score(q: float) -> float:
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must be strictly inside (0,1)")
    return _NORMAL.inv_cdf(q)


def features(data: Dataset, node: str, t: int, spec: FeatureSpec = RICH) -> list[float]:
    temp = data.temperature[node][t]
    hour = data.hour_of_day[t]
    row = [1.0, max(0.0, HEAT_BASE_C - temp)]
    if spec.use_cooling:
        row.append(max(0.0, temp - COOL_BASE_C))
    if spec.use_weekend:
        row.append(1.0 if data.day_of_week[t] >= 5 else 0.0)
    for k in range(1, spec.harmonics + 1):
        ang = 2 * math.pi * k * hour / 24.0
        row.append(math.sin(ang))
        row.append(math.cos(ang))
    return row


def design(data: Dataset, node: str, spec: FeatureSpec = RICH) -> tuple[Matrix, list[float]]:
    X = [features(data, node, t, spec) for t in range(data.hours)]
    y = list(data.demand[node])
    return X, y


def ols(X: Matrix, y: list[float], ridge: float = 1e-8) -> list[float]:
    """Least squares by the normal equations (X'X + ridge I) beta = X'y.

    The ridge term is 1e-8, eight orders of magnitude below the smallest
    diagonal entry of X'X here: it exists only to keep the solve from blowing up
    if a dummy column happens to be constant in a short backtest window, not to
    regularise anything.
    """
    Xt = transpose(X)
    p = len(Xt)
    XtX = [[sum(Xt[i][k] * Xt[j][k] for k in range(len(X))) for j in range(p)] for i in range(p)]
    for i in range(p):
        XtX[i][i] += ridge
    Xty = [[sum(Xt[i][k] * y[k] for k in range(len(X)))] for i in range(p)]
    return [row[0] for row in solve(XtX, Xty)]


@dataclass
class NodeModel:
    node: str
    coef: list[float]
    sigma: float
    spec: FeatureSpec = RICH
    residuals: tuple[float, ...] = ()

    def mean(self, data: Dataset, t: int) -> float:
        row = features(data, self.node, t, self.spec)
        return sum(c * x for c, x in zip(self.coef, row))

    def gaussian_quantile(self, data: Dataset, t: int, q: float) -> float:
        return self.mean(data, t) + z_score(q) * self.sigma

    def empirical_quantile(self, data: Dataset, t: int, q: float) -> float:
        """Mean plus the q-th quantile of the model's own residual sample.

        This is the only change needed to respect skew: no distributional
        assumption at all, just the shape the residuals actually had.
        """
        return self.mean(data, t) + empirical_quantile(list(self.residuals), q)


def empirical_quantile(sample: list[float], q: float) -> float:
    xs = sorted(sample)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


class HarmonicRegression:
    def __init__(self, spec: FeatureSpec = RICH, name: str = "harmonic-ols") -> None:
        self.spec = spec
        self.name = name
        self.models: dict[str, NodeModel] = {}

    def fit(self, train: Dataset) -> "HarmonicRegression":
        for node in train.hierarchy.names:
            X, y = design(train, node, self.spec)
            beta = ols(X, y)
            resid = [yi - sum(b * x for b, x in zip(beta, row)) for row, yi in zip(X, y)]
            n, p = len(resid), len(beta)
            dof = max(1, n - p)
            sigma = math.sqrt(sum(r * r for r in resid) / dof)
            self.models[node] = NodeModel(node, beta, sigma, self.spec, tuple(resid))
        return self

    def residuals(self, data: Dataset) -> dict[str, list[float]]:
        return {
            node: [data.demand[node][t] - self.models[node].mean(data, t) for t in range(data.hours)]
            for node in data.hierarchy.names
        }

    def mean_vector(self, data: Dataset, t: int) -> list[float]:
        return [self.models[node].mean(data, t) for node in data.hierarchy.names]

    def sigma_vector(self, data: Dataset) -> list[float]:
        return [self.models[node].sigma for node in data.hierarchy.names]


class SeasonalNaive:
    """yhat(t) = y(t - 168h).  Needs the realised history, so it is scored on a
    test set that is contiguous with its training set."""

    name = "seasonal-naive"
    LAG = 24 * 7

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {}
        self.sigma: dict[str, float] = {}

    def fit(self, train: Dataset) -> "SeasonalNaive":
        self.history = {node: list(train.demand[node]) for node in train.hierarchy.names}
        for node, series in self.history.items():
            errs = [series[t] - series[t - self.LAG] for t in range(self.LAG, len(series))]
            mu = sum(errs) / len(errs)
            self.sigma[node] = math.sqrt(sum((e - mu) ** 2 for e in errs) / (len(errs) - 1))
        return self

    def mean_vector(self, full: Dataset, t_abs: int) -> list[float]:
        return [full.demand[node][t_abs - self.LAG] for node in full.hierarchy.names]

    def sigma_vector(self, full: Dataset) -> list[float]:
        return [self.sigma[node] for node in full.hierarchy.names]
