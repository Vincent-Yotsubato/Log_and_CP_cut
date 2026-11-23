import unittest
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from rates_and_inexact import inexact as frozen
from rates_and_inexact.common import load_problem, reference_interval
from . import inexact_residual as solver
from .protocol import SOURCES, build
from .verification import audit_residual


RESULTS = Path(__file__).resolve().parents[1]/"results"


def requires(*paths):
    """Skip unless the upstream results exist; python run_all.py creates them."""
    missing = [p for p in paths if not (RESULTS/p).exists()]
    return unittest.skipIf(bool(missing), "needs " + ", ".join("results/"+p for p in missing) + " (run python run_all.py first)")


def archived_problems(count=2):
    directories = sorted(d for d in (SOURCES["revision"]/"inexact/instances").iterdir() if (d/"instance.npz").exists())
    return [(load_problem(d), reference_interval(str(d))) for d in directories[:count]]


@requires("rates_and_inexact/inexact/instances")
class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limiter = threadpool_limits(limits=1)
        cls.problems = archived_problems()

    @staticmethod
    def arguments(p, upper, method, mode, budget, eta_rel=1e-2, exponent=2.):
        if mode == 0:
            eta_rel, exponent = 0., 0.
        return (p.A, p.b, p.power, p.L0, p.L1, p.lam, p.p0, p.z, upper, method, budget, 1, mode,
                eta_rel, exponent, 100000, 1000., 1e-12)

    def test_modes_zero_and_one_reproduce_the_frozen_solver(self):
        for p, interval in self.problems:
            for method in (0, 1):
                for mode in (0, 1):
                    args = self.arguments(p, interval["omega_upper"], method, mode, 150)
                    new = solver.run(*args)
                    old = frozen.run(*args[:12], mode == 0, *args[13:])
                    for a, b in zip(new, old):
                        self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))

    def test_lean_loop_reproduces_the_logged_loop(self):
        p, interval = self.problems[0]
        for method in (0, 1):
            for mode in range(4):
                args = self.arguments(p, interval["omega_upper"], method, mode, 120)
                full = solver.run(*args)
                evaluated, W, flag, first_N, first_W, dual, x = solver.lean_run(*args)
                stats = full[-1]
                self.assertEqual((flag, evaluated, W), (0, int(stats[1]), int(stats[3])))
                self.assertEqual((first_N, first_W), (int(stats[11]), int(stats[12])))
                self.assertTrue(np.array_equal(dual, full[2]) and np.array_equal(x, full[3]))

    def test_residual_acceptance_and_identity(self):
        p, interval = self.problems[0]
        for method in (0, 1):
            for mode in (2, 3):
                audit = solver.run(*self.arguments(p, interval["omega_upper"], method, mode, 200))[1]
                accepted = audit[:, 21] == 1.
                self.assertTrue(np.all(accepted))
                residual, eta = audit[:, 13], audit[:, 11]
                bound = np.abs(residual) if mode == 2 else residual
                self.assertTrue(np.all(bound <= eta))
                self.assertTrue(np.all(audit[:, 12] >= -1e-12*(1+np.abs(audit[:, 6]))))
                # Fenchel identity: the residual equals Omega(x+) - J+ (last audit column).
                self.assertLess(np.max(np.abs(residual-audit[:, 24])/(1+np.abs(audit[:, 7]))), 1e-12)

    def test_inline_audit_accepts_a_residual_run(self):
        p, interval = self.problems[0]
        for method, name in ((0, "H-CP"), (1, "H-Log")):
            ans = solver.run(*self.arguments(p, interval["omega_upper"], method, 2, 150))
            audit, history, stats = ans[1], ans[8], ans[-1]
            result = {k: float(v) for k, v in zip(solver.STAT_COLUMNS, stats)}
            for k in ("first_distance_N", "first_distance_W", "first_joint_N"):
                result[k] = int(result[k]) if result[k] >= 0 else None
            result.update(N=int(stats[1]), method=dict(name=name, configuration="SSN-R"))
            columns = dict(zip(solver.AUDIT_COLUMNS, audit.T))
            following = history[np.minimum(np.arange(len(audit))+1, len(history)-1)]
            errors = audit_residual(p, result, columns, np.arange(len(audit)), history[:len(audit)],
                                    following, interval, every=history)
            self.assertEqual(errors, [])

    def test_jobs(self):
        jobs = build()
        self.assertEqual(len(jobs), 40)
        self.assertEqual(len({j["id"] for j in jobs}), 40)
        self.assertTrue(all(len(j["methods"]) == 10 for j in jobs))


if __name__ == "__main__":
    unittest.main()
