import unittest
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from rates_and_inexact import inexact, protocol as base
from rates_and_inexact.common import load_problem, reference_interval
from .lean import lean_run
from .protocol import SOURCE, SSN_KKT, build, registered


RESULTS = Path(__file__).resolve().parents[1]/"results"


def requires(*paths):
    """Skip unless the upstream results exist; python run_all.py creates them."""
    missing = [p for p in paths if not (RESULTS/p).exists()]
    return unittest.skipIf(bool(missing), "needs " + ", ".join("results/"+p for p in missing) + " (run python run_all.py first)")


def archived_problem():
    directory = sorted(d for d in (SOURCE/"inexact/instances").iterdir() if (d/"instance.npz").exists())[0]
    return load_problem(directory), reference_interval(str(directory))["omega_upper"]


@requires("rates_and_inexact/inexact/instances")
class InexactCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limiter = threadpool_limits(limits=1)
        cls.p, cls.upper = archived_problem()

    def arguments(self, method, exact, budget):
        p = self.p
        eta_rel, exponent = (0., 0.) if exact else (1e-2, 2.)
        return (p.A, p.b, p.power, p.L0, p.L1, p.lam, p.p0, p.z, self.upper, method, budget, 1, exact,
                eta_rel, exponent, 100000, 1000., 1e-12)

    def test_lean_loop_reproduces_the_logged_loop(self):
        for method in (0, 1):
            for exact in (False, True):
                args = self.arguments(method, exact, 80)
                full = inexact.run(*args)
                evaluated, W, flag, first_N, first_W, p, x = lean_run(*args)
                stats = full[-1]
                self.assertEqual(flag, 0)
                self.assertEqual(evaluated, int(stats[1]))
                self.assertEqual(W, int(stats[3]))
                self.assertEqual((first_N, first_W), (int(stats[11]), int(stats[12])))
                self.assertTrue(np.array_equal(p, full[2]))
                self.assertTrue(np.array_equal(x, full[3]))

    def test_same_code_exact_accepts_only_kkt_points(self):
        for method in (0, 1):
            audit = inexact.run(*self.arguments(method, True, 60))[1]
            accepted = audit[:, 21] == 1.
            self.assertTrue(np.all(accepted))
            self.assertLessEqual(np.max(audit[accepted, 15]), 1e-12)

    def test_registration_is_temporary(self):
        names = [c["name"] for c in base.CONFIGURATIONS]
        with registered():
            self.assertIs(base.configuration("SSN-KKT"), SSN_KKT)
        self.assertEqual([c["name"] for c in base.CONFIGURATIONS], names)

    def test_jobs_cover_the_archived_instances(self):
        jobs = build()
        self.assertEqual(len(jobs), 40)
        self.assertEqual(len({j["id"] for j in jobs}), 40)
        self.assertTrue(all([m["configuration"] for m in j["methods"]] == ["SSN-KKT"]*2 for j in jobs))
        self.assertTrue(all((SOURCE/"inexact/instances"/j["archived_job"]).is_dir() for j in jobs))


if __name__ == "__main__":
    unittest.main()
