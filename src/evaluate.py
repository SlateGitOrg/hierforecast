"""The two experiments this repo exists to run.

Experiment 1 - does reconciliation *improve accuracy*, or only fix coherence?
    Base forecasts are fit independently per node, so they are incoherent.
    Bottom-up, top-down, OLS and MinT all make them coherent.  Only the
    accuracy comparison tells you whether coherence cost you anything.

Experiment 2 - does the loss function change which model you would ship?
    Two candidate models are scored by RMSE/MAPE and by pinball loss at the
    upper quantiles, and the resulting capacity plans are compared in MW.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .forecast import COARSE, RICH, HarmonicRegression, SeasonalNaive, empirical_quantile, z_score
from .generator import Dataset, generate
from .hierarchy import Hierarchy
from .metrics import CapacityOutcome, capacity_outcome, mape, pinball_loss, rmse
from .reconcile import (
    Method,
    bottom_up,
    marginal_quantiles,
    mint_reconcile,
    ols_reconcile,
    reconciled_residuals,
    top_down,
    wls_reconcile,
)

UPPER_QUANTILES = (0.9, 0.95, 0.99)


def split(data: Dataset, train_fraction: float = 0.7) -> tuple[Dataset, Dataset]:
    cut = int(data.hours * train_fraction)
    return data.slice(0, cut), data.slice(cut, data.hours)


# --- experiment 1: reconciliation --------------------------------------------


@dataclass
class ReconResult:
    name: str
    rmse_by_level: dict[str, float]
    rmse_all: float
    pinball: dict[float, float]
    max_incoherence: float


def _level_of(h: Hierarchy, name: str) -> str:
    return next(n.level for n in h.nodes if n.name == name)


def evaluate_reconcilers(train: Dataset, test: Dataset, base: HarmonicRegression | None = None):
    """Score the base forecasts and each reconciliation of them on the test set."""
    h = train.hierarchy
    base = base or HarmonicRegression(RICH).fit(train)
    resid = base.residuals(train)

    props = [sum(train.demand[b]) / sum(train.demand["NAT"]) for b in h.bottom]
    variances = [max(1e-9, sum(r * r for r in resid[name]) / len(resid[name])) for name in h.names]
    mint, lam = mint_reconcile(h, resid)
    S = h.summing_matrix()
    structural = [sum(row) for row in S]  # number of leaves under each node
    methods: list[Method | None] = [
        None,  # the unreconciled base forecast
        bottom_up(h),
        top_down(h, props),
        ols_reconcile(h),
        wls_reconcile(h, structural, "wls-struct"),
        wls_reconcile(h, variances),
        mint,
    ]

    base_means = [base.mean_vector(test, t) for t in range(test.hours)]
    actuals = [[test.demand[name][t] for name in h.names] for t in range(test.hours)]

    results = []
    for method in methods:
        name = "base (independent)" if method is None else method.name
        fc = base_means if method is None else [method.apply(v) for v in base_means]
        if method is None:
            errs = [[resid[nm][t] for nm in h.names] for t in range(train.hours)]
        else:
            errs = reconciled_residuals(method, resid, h.names)
        per_level: dict[str, list[float]] = {}
        sq_all = 0.0
        for i, nm in enumerate(h.names):
            a = [row[i] for row in actuals]
            f = [row[i] for row in fc]
            e = rmse(a, f)
            per_level.setdefault(_level_of(h, nm), []).append(e)
            sq_all += e * e
        pb: dict[float, float] = {}
        for q in UPPER_QUANTILES:
            adj = marginal_quantiles(errs, q)
            tot = 0.0
            for i, nm in enumerate(h.names):
                a = [row[i] for row in actuals]
                f = [row[i] + adj[i] for row in fc]
                tot += pinball_loss(a, f, q)
            pb[q] = tot / len(h.names)
        incoh = max(h.max_incoherence(v) for v in fc)
        results.append(
            ReconResult(
                name=name,
                rmse_by_level={k: sum(v) / len(v) for k, v in per_level.items()},
                rmse_all=(sq_all / len(h.names)) ** 0.5,
                pinball=pb,
                max_incoherence=incoh,
            )
        )
    return results, lam


# --- experiment 2: which loss picks which model ------------------------------


@dataclass
class Candidate:
    """A base model plus a rule for turning it into quantiles."""

    label: str
    model: HarmonicRegression
    tail: str  # "gaussian" or "empirical"
    note: str = ""

    def mean(self, data: Dataset, node: str, t: int) -> float:
        return self.model.models[node].mean(data, t)

    def quantile(self, data: Dataset, node: str, t: int, q: float) -> float:
        m = self.model.models[node]
        if self.tail == "gaussian":
            return m.mean(data, t) + z_score(q) * m.sigma
        # Residual quantile is time-invariant: compute once per (node, q), not
        # once per hour (re-sorting thousands of residuals per call dominated
        # runtime before this cache existed).
        key = (node, q)
        cache = self.__dict__.setdefault("_qcache", {})
        if key not in cache:
            cache[key] = empirical_quantile(list(m.residuals), q)
        return m.mean(data, t) + cache[key]


@dataclass
class CandidateScore:
    label: str
    rmse: float
    mape: float
    mae: float
    pinball: dict[float, float] = field(default_factory=dict)
    capacity: dict[float, CapacityOutcome] = field(default_factory=dict)


def build_candidates(train: Dataset) -> list[Candidate]:
    """Three models, differing only in mean specification and tail rule.

    A is the model an RMSE-driven selection process ships: the best available
    mean, wrapped in the default Gaussian predictive interval.
    B is worse on the mean but reads its upper tail off the residuals instead of
    assuming symmetry.
    C exists so the comparison is not a straw man: it is the combination you
    should actually ship, and the demo reports it.
    """
    rich = HarmonicRegression(RICH, "rich").fit(train)
    coarse = HarmonicRegression(COARSE, "coarse").fit(train)
    return [
        Candidate("A rich-mean + gaussian tail", rich, "gaussian", "wins on RMSE/MAPE"),
        Candidate("B coarse-mean + empirical tail", coarse, "empirical", "wins on upper-quantile pinball"),
        Candidate("C rich-mean + empirical tail", rich, "empirical", "the combination to actually ship"),
    ]


def score_candidates(
    candidates: list[Candidate], test: Dataset, node: str = "NAT", quantiles=UPPER_QUANTILES
) -> list[CandidateScore]:
    actual = [test.demand[node][t] for t in range(test.hours)]
    out = []
    for c in candidates:
        means = [c.mean(test, node, t) for t in range(test.hours)]
        sc = CandidateScore(c.label, rmse(actual, means), mape(actual, means), 0.0)
        sc.mae = sum(abs(a - m) for a, m in zip(actual, means)) / len(actual)
        for q in quantiles:
            fq = [c.quantile(test, node, t, q) for t in range(test.hours)]
            sc.pinball[q] = pinball_loss(actual, fq, q)
            sc.capacity[q] = capacity_outcome(actual, fq)
        out.append(sc)
    return out


def default_experiment(hours: int = 24 * 200, seed: int = 20240, train_fraction: float = 0.7):
    data = generate(hours=hours, seed=seed)
    return (data,) + split(data, train_fraction)


def seasonal_naive_baseline(data: Dataset, train: Dataset, test: Dataset, node: str = "NAT") -> float:
    """RMSE of last-week's-same-hour on the test window.

    Always computed and always reported: a model that cannot beat it has not
    earned the right to be deployed, however good its reconciliation is.
    """
    sn = SeasonalNaive().fit(train)
    cut = train.hours
    a, f = [], []
    for t in range(test.hours):
        a.append(data.demand[node][cut + t])
        f.append(data.demand[node][cut + t - sn.LAG])
    return rmse(a, f)
