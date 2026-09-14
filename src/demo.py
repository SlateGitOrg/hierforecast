"""60-second artefact: `python -m src.demo`.  ASCII output only, fixed seed."""

from __future__ import annotations

import time

from .evaluate import (
    build_candidates,
    default_experiment,
    evaluate_reconcilers,
    score_candidates,
    seasonal_naive_baseline,
)

SEED = 20240


def main() -> None:
    t0 = time.time()
    data, train, test = default_experiment(hours=24 * 200, seed=SEED)
    h = data.hierarchy
    print("hierforecast demo (seed=%d)" % SEED)
    print("hierarchy: %d nodes (%d aggregates, %d bottom), train=%dh test=%dh"
          % (h.n, len(h.aggregates), h.m, train.hours, test.hours))

    results, lam = evaluate_reconcilers(train, test)
    by = {r.name: r for r in results}
    print("\n[1] Reconciliation on held-out hours (MinT shrinkage lambda=%.4f)" % lam)
    print("%-20s %9s %9s %9s %9s | %8s %8s %8s | %s"
          % ("method", "rmse_all", "nat", "region", "sub", "pb90", "pb95", "pb99", "max|incoh| MW"))
    for r in results:
        L = r.rmse_by_level
        print("%-20s %9.2f %9.1f %9.1f %9.1f | %8.2f %8.2f %8.2f | %.2e"
              % (r.name, r.rmse_all, L["national"], L["region"], L["sub"],
                 r.pinball[0.9], r.pinball[0.95], r.pinball[0.99], r.max_incoherence))
    for other in ("bottom-up", "top-down", "base (independent)", "ols"):
        imp = 100.0 * (1.0 - by["mint-shrink"].pinball[0.95] / by[other].pinball[0.95])
        print("MinT pinball@0.95 vs %-20s: %+.1f%%" % (other, imp))

    snaive = seasonal_naive_baseline(data, train, test)
    model = by["base (independent)"].rmse_by_level["national"]
    print("\n[2] Seasonal-naive baseline (always reported): national RMSE %.1f MW vs model %.1f MW -> %s"
          % (snaive, model, "model beats baseline" if model < snaive else "MODEL FAILS BASELINE"))

    print("\n[3] Which model would you ship? (national node)")
    print("%-32s %8s %7s | %8s %8s %8s | %s" % ("candidate", "rmse", "mape%", "pb90", "pb95", "pb99", "hours short @p95 / @p99"))
    for s in score_candidates(build_candidates(train), test):
        print("%-32s %8.1f %7.2f | %8.2f %8.2f %8.2f | %.1f%% / %.1f%%"
              % (s.label, s.rmse, s.mape, s.pinball[0.9], s.pinball[0.95], s.pinball[0.99],
                 s.capacity[0.95].exceedance_pct, s.capacity[0.99].exceedance_pct))
    print("\n(nominal exceedance: 5.0%% at p95, 1.0%% at p99)  elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
