"""End-to-end tests against the planted synthetic hierarchy.

One 200-day experiment (seed 20240, the demo seed) is built once and shared.

Thresholds are derived from a 6-seed sweep at the same length (seeds 1-6),
not tuned on this seed:
  MinT vs bottom-up pinball gain:  p90 14.5..20.0%, p95 8.6..16.5%, p99 5.1..14.6%
  MinT vs independent base @p95:   3.6..7.3%
  Gaussian-tail exceedance @p99:   3.5..5.3% of hours (nominal 1%)
  seasonal-naive RMSE / model:     1.48..1.94
Each assertion uses roughly half the smallest observed margin, so a real
regression fails it but seed noise does not.  At 120 days of history the p95/p99
gains were NOT robust (-0.1..20% at p95), which is recorded in the README.
"""

import random
import unittest

from src.evaluate import (
    build_candidates,
    default_experiment,
    evaluate_reconcilers,
    score_candidates,
    seasonal_naive_baseline,
)
from src.forecast import RICH, HarmonicRegression
from src.generator import _make_params, generate

SEED = 20240


class Shared:
    built = False

    @classmethod
    def get(cls):
        if not cls.built:
            cls.data, cls.train, cls.test = default_experiment(hours=24 * 200, seed=SEED)
            cls.results, cls.lam = evaluate_reconcilers(cls.train, cls.test)
            cls.by = {r.name: r for r in cls.results}
            cls.cands = {s.label[0]: s for s in score_candidates(build_candidates(cls.train), cls.test)}
            cls.built = True
        return cls


def gain(better, worse):
    return 100.0 * (1.0 - better / worse)


class PlantedGroundTruthTests(unittest.TestCase):
    def test_generated_data_is_exactly_coherent(self):
        d = generate(hours=24 * 14, seed=5)
        h = d.hierarchy
        for t in range(d.hours):
            self.assertLess(h.max_incoherence(d.node_vector(t)), 1e-6)

    def test_cold_snaps_make_errors_right_skewed(self):
        s = Shared.get()
        r = s.by  # noqa: F841 - ensure experiment built
        model = HarmonicRegression(RICH).fit(s.train)
        res = sorted(model.models["NAT"].residuals)
        n = len(res)
        # Planted: synchronous surges -> upper tail longer than lower tail.
        med = res[n // 2]
        self.assertGreater(res[int(0.99 * n)] - med, 1.3 * (med - res[int(0.01 * n)]))

    def test_aggregate_model_recovers_planted_heating_slope_better_than_leaf_sum(self):
        # Planted mechanism behind MinT's gain: each leaf sees a noisy station
        # (sd 3C) so its slope is attenuated; the national average of 12
        # stations is far less noisy.  Planted national slope = sum of leaves'.
        s = Shared.get()
        h = s.train.hierarchy
        planted = _make_params(random.Random(SEED), h.bottom)
        truth = sum(planted[b].heat_mw_per_degc for b in h.bottom)
        model = HarmonicRegression(RICH).fit(s.train)
        nat_est = model.models["NAT"].coef[1]
        leaf_sum = sum(model.models[b].coef[1] for b in h.bottom)
        self.assertLess(leaf_sum, truth)  # attenuation really happens
        self.assertLess(abs(nat_est - truth), abs(leaf_sum - truth))


class ReconciliationExperimentTests(unittest.TestCase):
    def setUp(self):
        self.s = Shared.get()

    def test_base_forecasts_incoherent_reconciled_exact(self):
        by = self.s.by
        self.assertGreater(by["base (independent)"].max_incoherence, 10.0)  # MW
        for name in ("bottom-up", "top-down", "ols", "wls-struct", "wls-var", "mint-shrink"):
            # National demand ~2e4 MW; 1e-9 relative = float round-off.
            self.assertLess(by[name].max_incoherence, 1e-6, name)

    def test_mint_beats_bottom_up_on_pinball_every_upper_quantile(self):
        by = self.s.by
        floors = {0.9: 7.0, 0.95: 4.0, 0.99: 2.5}  # ~half the min observed gain
        for q, floor in floors.items():
            g = gain(by["mint-shrink"].pinball[q], by["bottom-up"].pinball[q])
            self.assertGreater(g, floor, "q=%s gain=%.1f%%" % (q, g))

    def test_mint_improves_on_the_unreconciled_base_not_just_coherence(self):
        by = self.s.by
        g = gain(by["mint-shrink"].pinball[0.95], by["base (independent)"].pinball[0.95])
        self.assertGreater(g, 1.5, "gain=%.1f%%" % g)  # observed 3.6..7.3%

    def test_bottom_up_is_the_generic_mistake_at_national_level(self):
        # Bottom-up throws away the better-estimated national model.
        L = lambda n: self.s.by[n].rmse_by_level["national"]
        self.assertGreater(L("bottom-up"), 1.2 * L("base (independent)"))
        self.assertLess(L("mint-shrink"), L("bottom-up"))

    def test_shrinkage_intensity_in_valid_range(self):
        self.assertTrue(0.0 <= self.s.lam <= 1.0)


class LossChoiceTests(unittest.TestCase):
    def setUp(self):
        self.c = Shared.get().cands

    def test_rmse_selects_a_model_that_under_provisions_the_tail(self):
        A, B, C = self.c["A"], self.c["B"], self.c["C"]
        self.assertLess(A.rmse, B.rmse)  # RMSE picks the Gaussian-tail model
        # ...which is short far more often than its nominal 1% at p99.
        self.assertGreater(A.capacity[0.99].exceedance_pct, 2.0)
        self.assertLess(C.pinball[0.99], A.pinball[0.99])
        self.assertLess(C.capacity[0.99].exceedance_pct, A.capacity[0.99].exceedance_pct)
        self.assertEqual(A.rmse, C.rmse)  # same mean: only the tail differs

    def test_seasonal_naive_always_computed_and_beaten(self):
        s = Shared.get()
        sn = seasonal_naive_baseline(s.data, s.train, s.test)
        self.assertGreater(sn, 0.0)
        model = s.by["base (independent)"].rmse_by_level["national"]
        self.assertGreater(sn / model, 1.2)  # observed 1.48..1.94


if __name__ == "__main__":
    unittest.main()
