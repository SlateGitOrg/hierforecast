"""Forecast reconciliation: bottom-up, top-down, OLS, WLS and MinT.

Every method here is the same linear map in disguise:

    ytilde = S P yhat

where P (m x n) maps the n incoherent base forecasts to a bottom-level vector
and S maps that back up the tree.  Because the result is S times *something*,
coherence is exact by construction - not approximate, not enforced by a
post-hoc fudge.  What separates the methods is only the choice of P:

    bottom-up   P = [0 | I]                       ignores the aggregate forecasts
    top-down    P = p e_1'                        ignores the local ones
    OLS         P = (S'S)^-1 S'                   every node weighted equally
    WLS         P = (S'W^-1 S)^-1 S'W^-1, W diag  weighted by error variance
    MinT        as WLS but W = full error covariance (shrunk)

MinT is the only one that uses the *correlation* between node errors, and that
is why it can beat bottom-up on accuracy rather than merely matching it on
coherence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .hierarchy import Hierarchy
from .linalg import Matrix, matmul, matvec, solve, transpose, zeros


def projection(S: Matrix, W: Matrix) -> Matrix:
    """P = (S' W^-1 S)^-1 S' W^-1, computed without ever forming W^-1.

    Forming the inverse explicitly and multiplying is both slower and less
    accurate; two solves give the same answer with better conditioning.
    """
    Winv_S = solve(W, S)                     # n x m
    A = matmul(transpose(S), Winv_S)         # m x m
    return solve(A, transpose(Winv_S))       # m x n


def reconciler_matrix(S: Matrix, W: Matrix) -> Matrix:
    """S P: the full n x n map from base to reconciled forecasts."""
    return matmul(S, projection(S, W))


def diag(values: list[float]) -> Matrix:
    n = len(values)
    M = zeros(n, n)
    for i, v in enumerate(values):
        M[i][i] = v
    return M


def sample_covariance(residuals: dict[str, list[float]], names: list[str]) -> Matrix:
    n = len(names)
    cols = [residuals[name] for name in names]
    T = len(cols[0])
    means = [sum(c) / T for c in cols]
    C = zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            s = 0.0
            ci, cj, mi, mj = cols[i], cols[j], means[i], means[j]
            for t in range(T):
                s += (ci[t] - mi) * (cj[t] - mj)
            v = s / (T - 1)
            C[i][j] = v
            C[j][i] = v
    return C


def shrink_covariance(C: Matrix, residuals: dict[str, list[float]], names: list[str]) -> tuple[Matrix, float]:
    """Shrink the sample covariance toward its own diagonal (Schafer-Strimmer).

    The sample covariance of a 17-node hierarchy estimated from a few thousand
    residuals is fine, but MinT is famously fragile when it is not: the shrinkage
    intensity is *estimated from the data*, not picked, so the method degrades
    gracefully to WLS as the estimate gets noisy instead of silently inverting a
    near-singular matrix.
    """
    n = len(names)
    cols = [residuals[name] for name in names]
    T = len(cols[0])
    means = [sum(c) / T for c in cols]
    sd = [max(C[i][i], 1e-12) ** 0.5 for i in range(n)]
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = C[i][j] / (sd[i] * sd[j])
            # Var of the sample correlation estimate, per Schafer & Strimmer.
            w = [((cols[i][t] - means[i]) / sd[i]) * ((cols[j][t] - means[j]) / sd[j]) for t in range(T)]
            wbar = sum(w) / T
            var_r = (T / ((T - 1) ** 3)) * sum((x - wbar) ** 2 for x in w)
            num += var_r
            den += r * r
    lam = 1.0 if den <= 0.0 else max(0.0, min(1.0, num / den))
    out = zeros(n, n)
    for i in range(n):
        for j in range(n):
            out[i][j] = C[i][j] if i == j else (1.0 - lam) * C[i][j]
    return out, lam


@dataclass
class Method:
    name: str
    matrix: Matrix  # n x n, applied to a base forecast vector

    def apply(self, yhat: list[float]) -> list[float]:
        return matvec(self.matrix, yhat)


def bottom_up(h: Hierarchy) -> Method:
    S = h.summing_matrix()
    P = zeros(h.m, h.n)
    offset = len(h.aggregates)
    for j in range(h.m):
        P[j][offset + j] = 1.0
    return Method("bottom-up", matmul(S, P))


def top_down(h: Hierarchy, proportions: list[float]) -> Method:
    """Disaggregate the root forecast by fixed historical proportions."""
    if abs(sum(proportions) - 1.0) > 1e-9:
        raise ValueError("top-down proportions must sum to 1")
    S = h.summing_matrix()
    P = zeros(h.m, h.n)
    root = h.index["NAT"]
    for j, p in enumerate(proportions):
        P[j][root] = p
    return Method("top-down", matmul(S, P))


def ols_reconcile(h: Hierarchy) -> Method:
    n = h.n
    W = diag([1.0] * n)
    return Method("ols", reconciler_matrix(h.summing_matrix(), W))


def wls_reconcile(h: Hierarchy, variances: list[float], name: str = "wls-var") -> Method:
    """WLS with a diagonal W.  Passing the leaf counts gives "structural"
    scaling, which needs no residual history at all - the fallback when a node
    is too new to have one."""
    return Method(name, reconciler_matrix(h.summing_matrix(), diag(variances)))


def mint_reconcile(h: Hierarchy, residuals: dict[str, list[float]]) -> tuple[Method, float]:
    C = sample_covariance(residuals, h.names)
    W, lam = shrink_covariance(C, residuals, h.names)
    return Method("mint-shrink", reconciler_matrix(h.summing_matrix(), W)), lam


# --- probabilistic reconciliation -------------------------------------------

def reconciled_residuals(method: Method, residuals: dict[str, list[float]], names: list[str]) -> list[list[float]]:
    """Push each historical base-forecast error vector through the reconciler.

    Reconciliation is linear, so reconciling `base + e` is reconciling `base`
    plus reconciling `e`.  Doing the error part once, up front, turns a
    per-hour Monte Carlo into a table lookup - and it is exact, not an
    approximation of the simulation.
    """
    T = len(residuals[names[0]])
    cols = [residuals[name] for name in names]
    return [method.apply([cols[i][t] for i in range(len(names))]) for t in range(T)]


def reconciled_paths(
    method: Method,
    base_mean: list[float],
    residuals: dict[str, list[float]],
    names: list[str],
    draws: int,
    seed: int,
) -> list[list[float]]:
    """Coherent sample paths: reconciled point forecast plus a resampled
    reconciled error vector.

    Quantiles do not add up - the 95th percentile of a sum is not the sum of the
    95th percentiles, because the regions do not all peak at once.  So a
    "coherent quantile forecast" cannot mean applying S to a vector of
    quantiles.  It means every *realisation* is coherent, and the quantiles are
    then read off the marginals of that coherent joint distribution.  Resampling
    whole residual rows preserves the cross-node correlation that a per-node
    parametric draw would destroy.
    """
    rng = random.Random(seed)
    base = method.apply(base_mean)
    errs = reconciled_residuals(method, residuals, names)
    return [[base[i] + row[i] for i in range(len(base))] for row in (errs[rng.randrange(len(errs))] for _ in range(draws))]


def quantile_of(sample: list[float], q: float) -> float:
    xs = sorted(sample)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def marginal_quantiles(errs: list[list[float]], q: float) -> list[float]:
    """Per-node quantile of the reconciled error distribution."""
    n = len(errs[0])
    return [quantile_of([row[i] for row in errs], q) for i in range(n)]


def path_quantiles(paths: list[list[float]], q: float) -> list[float]:
    n = len(paths[0])
    return [quantile_of([p[i] for p in paths], q) for i in range(n)]
