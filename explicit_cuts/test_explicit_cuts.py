import unittest
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from cut_functions import a_star, b_star, s_log, s_c
from elastic_net.problems import make_e1
from baseline.protocol import method
from .protocol import build, METHOD_NAMES
from .runner import solve
from .verification import replay


RESULTS = Path(__file__).resolve().parents[1]/"results"


def requires(*paths):
    """Skip unless the upstream results exist; python run_all.py creates them."""
    missing = [p for p in paths if not (RESULTS/p).exists()]
    return unittest.skipIf(bool(missing), "needs " + ", ".join("results/"+p for p in missing) + " (run python run_all.py first)")


class CutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limiter = threadpool_limits(limits=1)

    def test_cut_ordering_over_attainable_data(self):
        rng = np.random.default_rng(1102)
        for L0, L1 in ((1., 0.), (.5, 2.), (1/128, 4.)):
            q = np.exp(rng.uniform(-8, 6, 400))
            delta = b_star(q, L0, L1)*(1+rng.exponential(1., q.size))  # attainable: delta >= B_*(q)
            clipped, log, cp = s_c(q, L0, L1), s_log(q, L0, L1), delta+a_star(q, L0, L1)
            self.assertTrue(np.all(clipped <= log*(1+1e-12)))
            self.assertTrue(np.all(np.maximum(delta, log) <= cp*(1+1e-12)))
            if L1 == 0.:
                np.testing.assert_allclose(clipped, log, rtol=1e-15)

    def test_clipped_equals_Log_without_curvature_growth(self):
        p = make_e1(m=8, n=32, beta=0., amplitude=.25, seed=0)
        a = solve(p, method("C-LBreI"), 200, 0)
        b = solve(p, method("Log-LBreI"), 200, 0)
        np.testing.assert_array_equal(a[2], b[2])

    def test_numba_trajectories_match_direct_replay(self):
        p = make_e1(m=8, n=32, beta=1., amplitude=.01, target_r=100., seed=0)
        for name in METHOD_NAMES:
            result, rows, *_ = solve(p, method(name), 30, 0)
            rep = replay(p, name, 30)
            m = min(len(rep), len(rows))
            np.testing.assert_allclose(rep[:m, 0], rows[:m, 5], rtol=1e-9, atol=1e-18)
            np.testing.assert_allclose(rep[:m, 1], rows[:m, 11], rtol=1e-8, atol=1e-14)

    @requires("baseline/config.json")
    def test_grid(self):
        jobs = build()
        self.assertEqual(len(jobs), 150)
        self.assertEqual(sum(len(j["methods"]) for j in jobs), 750)
        self.assertEqual(len(build(True)), 15)
        self.assertEqual({j["variant"] for j in jobs},
                         {"calibrated", "quadratic_control", "kappa10_zero", "kappa10_calibrated"})
        self.assertEqual(sum(j["source_instance"] is None for j in jobs), 50)


if __name__ == "__main__":
    unittest.main()
