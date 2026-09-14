"""Deterministic unit tests: linear algebra, hierarchy, reconciliation maths,
pinball loss and the data-quality gate.  Each asserts a property a broken
implementation would violate, not merely that the code runs."""

import random
import unittest
from datetime import datetime, timedelta

from src.hierarchy import Hierarchy, Node, default_hierarchy
from src.linalg import cholesky, identity, matmul, matvec, solve, transpose
from src.metrics import capacity_outcome, pinball, pinball_loss
from src.quality import DataQualityError, assert_clean, gate, hourly_index
from src.reconcile import (
    bottom_up,
    diag,
    mint_reconcile,
    ols_reconcile,
    projection,
    sample_covariance,
    shrink_covariance,
    top_down,
    wls_reconcile,
)

TOL = 1e-9


def rand_matrix(rng, r, c):
    return [[rng.uniform(-1, 1) for _ in range(c)] for _ in range(r)]


class LinalgTests(unittest.TestCase):
    def test_solve_recovers_known_solution(self):
        rng = random.Random(1)
        A = rand_matrix(rng, 8, 8)
        X = rand_matrix(rng, 8, 3)
        B = matmul(A, X)
        got = solve(A, B)
        for i in range(8):
            for j in range(3):
                self.assertAlmostEqual(got[i][j], X[i][j], places=9)

    def test_solve_needs_pivoting(self):
        # Zero leading pivot: naive elimination divides by zero.
        A = [[0.0, 1.0], [1.0, 0.0]]
        self.assertEqual(solve(A, [[2.0], [3.0]]), [[3.0], [2.0]])

    def test_singular_raises(self):
        with self.assertRaises(ValueError):
            solve([[1.0, 2.0], [2.0, 4.0]], [[1.0], [1.0]])

    def test_cholesky_reconstructs_and_rejects_indefinite(self):
        rng = random.Random(2)
        M = rand_matrix(rng, 5, 5)
        A = matmul(M, transpose(M))
        for i in range(5):
            A[i][i] += 0.5
        L = cholesky(A)
        LL = matmul(L, transpose(L))
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(LL[i][j], A[i][j], places=10)
        with self.assertRaises(ValueError):
            cholesky([[1.0, 2.0], [2.0, 1.0]])


class HierarchyTests(unittest.TestCase):
    def test_summing_matrix_structure(self):
        h = default_hierarchy()
        S = h.summing_matrix()
        self.assertEqual((h.n, h.m), (17, 12))
        self.assertEqual(sum(S[h.index["NAT"]]), 12.0)
        self.assertEqual(sum(S[h.index["R2"]]), 3.0)
        # Bottom block of S is the identity.
        for j in range(h.m):
            self.assertEqual(S[len(h.aggregates) + j], identity(h.m)[j])

    def test_unbalanced_hierarchy_from_data(self):
        h = Hierarchy([Node("T", None, "national"), Node("A", "T", "region"), Node("B", "T", "sub"),
                       Node("A1", "A", "sub"), Node("A2", "A", "sub")])
        S = h.summing_matrix()
        self.assertEqual(h.bottom, ["B", "A1", "A2"])
        self.assertEqual(S[h.index["T"]], [1.0, 1.0, 1.0])
        self.assertEqual(S[h.index["A"]], [0.0, 1.0, 1.0])

    def test_invalid_hierarchies_rejected(self):
        with self.assertRaises(ValueError):
            Hierarchy([Node("A", None, "x"), Node("B", None, "x")])
        with self.assertRaises(ValueError):
            Hierarchy([Node("A", None, "x"), Node("B", "Z", "x")])


def incoherent_vector(h, rng):
    return [rng.uniform(500, 5000) for _ in range(h.n)]


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.h = default_hierarchy()
        self.rng = random.Random(7)
        # Residuals with planted cross-node correlation (a common shock).
        T = 400
        res = {}
        common = [self.rng.gauss(0, 1) for _ in range(T)]
        leaf = {b: [5 * common[t] + self.rng.gauss(0, 3) for t in range(T)] for b in self.h.bottom}
        for name in self.h.names:
            leaves = self.h.leaves_under(name)
            res[name] = [sum(leaf[l][t] for l in leaves) + self.rng.gauss(0, 4) for t in range(T)]
        self.residuals = res

    def all_methods(self):
        h = self.h
        mint, _ = mint_reconcile(h, self.residuals)
        return [bottom_up(h), top_down(h, [1.0 / h.m] * h.m), ols_reconcile(h),
                wls_reconcile(h, [float(sum(r)) for r in h.summing_matrix()]), mint]

    def test_every_method_is_exactly_coherent(self):
        for _ in range(20):
            y = incoherent_vector(self.h, self.rng)
            self.assertGreater(self.h.max_incoherence(y), 1.0)  # input really is incoherent
            for m in self.all_methods():
                # Values ~1e4 MW; 1e-8 relative is float round-off, not approximation.
                self.assertLess(self.h.max_incoherence(m.apply(y)), 1e-8 * 5e4, m.name)

    def test_naive_rescaling_is_not_coherent_below_national(self):
        # The generic fix: scale the regions so they sum to national, leave subs alone.
        y = incoherent_vector(self.h, self.rng)
        h = self.h
        regs = h.children("NAT")
        k = y[h.index["NAT"]] / sum(y[h.index[r]] for r in regs)
        z = list(y)
        for r in regs:
            z[h.index[r]] *= k
        self.assertGreater(h.max_incoherence(z), 1.0)

    def test_projection_is_idempotent_and_preserves_coherent_input(self):
        # SP must be a projection: a forecast that is already coherent is unchanged.
        h = self.h
        S = h.summing_matrix()
        b = [self.rng.uniform(100, 900) for _ in range(h.m)]
        y = matvec(S, b)
        for m in self.all_methods()[2:]:  # OLS/WLS/MinT are projections (BU/TD are too, onto S, via P)
            out = m.apply(y)
            for a, c in zip(out, y):
                self.assertAlmostEqual(a, c, places=6)

    def test_ols_is_least_squares_closest_coherent_vector(self):
        # OLS reconciliation minimises ||y - S b||; perturbing b must not reduce it.
        h = self.h
        S = h.summing_matrix()
        y = incoherent_vector(h, self.rng)
        P = projection(S, diag([1.0] * h.n))
        b = matvec(P, y)
        best = sum((a - c) ** 2 for a, c in zip(y, matvec(S, b)))
        for _ in range(30):
            b2 = [x + self.rng.gauss(0, 5) for x in b]
            self.assertGreaterEqual(sum((a - c) ** 2 for a, c in zip(y, matvec(S, b2))), best - 1e-6)

    def test_bottom_up_ignores_aggregate_forecasts(self):
        y = incoherent_vector(self.h, self.rng)
        z = list(y)
        z[self.h.index["NAT"]] += 1e6
        self.assertEqual(bottom_up(self.h).apply(y), bottom_up(self.h).apply(z))

    def test_shrinkage_intensity_estimated_not_fixed(self):
        names = self.h.names
        C = sample_covariance(self.residuals, names)
        _, lam_corr = shrink_covariance(C, self.residuals, names)
        rng = random.Random(3)
        indep = {n: [rng.gauss(0, 1) for _ in range(400)] for n in names}
        _, lam_indep = shrink_covariance(sample_covariance(indep, names), indep, names)
        # Strongly correlated residuals -> little shrinkage; pure noise -> heavy.
        self.assertLess(lam_corr, 0.1)
        self.assertGreater(lam_indep, 0.5)


class PinballTests(unittest.TestCase):
    def test_asymmetry(self):
        self.assertAlmostEqual(pinball(110, 100, 0.95), 9.5)
        self.assertAlmostEqual(pinball(90, 100, 0.95), 0.5)

    def test_minimised_at_true_quantile(self):
        rng = random.Random(11)
        xs = [rng.expovariate(1.0) for _ in range(4000)]  # skewed on purpose
        true_q95 = sorted(xs)[int(0.95 * len(xs))]
        mean = sum(xs) / len(xs)
        at_q = pinball_loss(xs, [true_q95] * len(xs), 0.95)
        self.assertLess(at_q, pinball_loss(xs, [mean] * len(xs), 0.95))
        self.assertLess(at_q, pinball_loss(xs, [true_q95 * 1.2] * len(xs), 0.95))
        self.assertLess(at_q, pinball_loss(xs, [true_q95 * 0.8] * len(xs), 0.95))

    def test_capacity_outcome(self):
        c = capacity_outcome([10, 20, 30], [15, 15, 15])
        self.assertEqual(c.exceedance_hours, 2)
        self.assertAlmostEqual(c.unserved_mwh, 20.0)
        self.assertAlmostEqual(c.worst_shortfall_mw, 15.0)


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.ts = hourly_index(datetime(2024, 1, 1), 48)
        self.vals = [1000.0 + 50 * (i % 24) for i in range(48)]

    def test_clean_passes(self):
        assert_clean(self.ts, self.vals)

    def test_dst_spring_and_autumn(self):
        spring = self.ts[:10] + self.ts[11:]
        kinds = {i.kind for i in gate(spring, self.vals[:47])}
        self.assertIn("dst_missing_hour", kinds)
        autumn = self.ts[:11] + [self.ts[10]] + self.ts[11:47]
        kinds = {i.kind for i in gate(autumn, self.vals)}
        self.assertIn("dst_duplicate_hour", kinds)

    def test_missing_interval_and_meter_reset_fail_loudly(self):
        gap = self.ts[:10] + [t + timedelta(hours=5) for t in self.ts[10:]]
        with self.assertRaises(DataQualityError):
            assert_clean(gap, self.vals)
        v = list(self.vals)
        v[20] = 3.0
        kinds = [i.kind for i in gate(self.ts, v)]
        self.assertIn("meter_reset", kinds)
        with self.assertRaises(DataQualityError):
            assert_clean(self.ts, v)

    def test_legitimate_ramp_not_flagged(self):
        v = list(self.vals)
        v[30] = v[29] * 0.75  # a 25% drop: sharp but plausible
        self.assertEqual(gate(self.ts, v), [])


if __name__ == "__main__":
    unittest.main()
