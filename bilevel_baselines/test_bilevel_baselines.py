import unittest
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from baseline.numerics import run as archived_run, METHODS
from elastic_net.protocol import log_steps
from .numerics import project_ball_halfspace, run_mng, run_bigsam, run_agm
from .protocol import SOURCE, build, methods
from .runner import load, constants, solve
from .verification import project_ball_halfspace_bisection, replay, project_two_halfspaces


RESULTS = Path(__file__).resolve().parents[1]/"results"


def requires(*paths):
    """Skip unless the upstream results exist; python run_all.py creates them."""
    missing = [p for p in paths if not (RESULTS/p).exists()]
    return unittest.skipIf(bool(missing), "needs " + ", ".join("results/"+p for p in missing) + " (run python run_all.py first)")


def archived_instance(power=3, size=(10, 40)):
    import json
    for job in json.loads((SOURCE/"config.json").read_text(encoding="utf-8"))["jobs"]:
        m = job["model"]
        if (job["variant"] == "polyhedron" and m.get("correlation") == .8 and m["power"] == power
                and (m["m"], m["n"]) == size and m["seed"] == 0):
            return load(SOURCE/"instances"/job["id"])
    raise LookupError


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limiter = threadpool_limits(limits=1)

    def test_ball_halfspace_projection(self):
        rng = np.random.default_rng(1103)
        branches = set()
        for _ in range(2000):
            n = int(rng.integers(2, 6))
            a = rng.normal(size=n)
            rho = float(rng.uniform(.5, 2.))
            # Nonempty intersection: some point of the ball satisfies the cut.
            beta = float(a @ (rng.normal(size=n)*rho/np.sqrt(n)*.5)) + rng.uniform(-.5, 1.)*np.linalg.norm(a)*rho*.5
            if beta < -np.linalg.norm(a)*rho*.99:
                continue
            w = rng.normal(size=n)*rng.uniform(.1, 4.)
            u, branch, flag = project_ball_halfspace(w, rho, a, beta)
            self.assertEqual(flag, 0)
            branches.add(int(branch))
            v = project_ball_halfspace_bisection(w, rho, a, beta)
            np.testing.assert_allclose(u, v, atol=1e-9*(1+np.linalg.norm(w)))
            self.assertLessEqual(np.linalg.norm(u), rho*(1+1e-12))
            self.assertLessEqual(a @ u, beta+1e-10*(1+abs(beta)))
        self.assertEqual(branches, {0, 1, 2, 3})

    def test_two_halfspace_replay_projection(self):
        rng = np.random.default_rng(7)
        seen = set()
        for _ in range(500):
            x0 = rng.normal(size=4)*2
            a1, a2 = rng.normal(size=4), rng.normal(size=4)
            c1, c2 = rng.normal(), rng.normal()
            u = project_two_halfspaces(x0, a1, c1, a2, c2)
            self.assertLessEqual(a1 @ u, c1+1e-10*(1+abs(c1)))
            self.assertLessEqual(a2 @ u, c2+1e-10*(1+abs(c2)))
            # KKT: x0-u is a nonnegative combination of the active normals.
            active = [a for a, c in ((a1, c1), (a2, c2)) if abs(a @ u-c) < 1e-9*(1+abs(c))]
            seen.add(len(active))
            if not active:
                np.testing.assert_array_equal(u, x0)
                continue
            M = np.array(active).T
            coef, *_ = np.linalg.lstsq(M, x0-u, rcond=None)
            self.assertTrue(np.all(coef >= -1e-9))
            np.testing.assert_allclose(M @ coef, x0-u, atol=1e-9*(1+np.linalg.norm(x0)))
        self.assertEqual(seen, {0, 1, 2})

    @requires("baseline/config.json")
    def test_mng_is_archived_h_log_with_lipschitz_cut(self):
        p = archived_instance()
        c = constants(p["A"], p["b"], p["power"])
        steps = log_steps(299)
        rows, x, stats = run_mng(p["A"], p["b"], p["power"], c["L_ball"], p["z"], 300, steps)
        archived = archived_run(p["A"], p["b"], 1, 0., p["power"], c["L_ball"], 0., 0., p["p0"], p["z"],
                                METHODS["H-Log"], 1., 1., 299, 0, steps)
        old = archived[0]
        np.testing.assert_allclose(rows[:, 2], old[:len(rows), 5], rtol=1e-12, atol=1e-30)
        np.testing.assert_allclose(archived[2], x, rtol=1e-12, atol=1e-14)

    @requires("baseline/config.json")
    def test_bigsam_limits(self):
        p = archived_instance()
        steps = log_steps(49)
        # gamma = 0: no anchoring, plain gradient descent with step t.
        rows, x, _ = run_bigsam(p["A"], p["b"], p["power"], p["z"], 50, steps, 0, 1e-3, p["L0"], p["L1"], 0., 1e9)
        y = np.zeros_like(x)
        for _ in range(49):
            v = np.maximum(p["b"]-p["A"] @ y, 0.)
            y = y+1e-3*(p["A"].T @ v**(p["power"]-1))
        np.testing.assert_allclose(x, y, rtol=1e-11, atol=1e-14)
        # gamma = 20: alpha_k = 1 for k <= 20 keeps the anchor x0 = 0.
        rows, x, _ = run_bigsam(p["A"], p["b"], p["power"], p["z"], 21, log_steps(20), 0, 1e-3, p["L0"], p["L1"], 20., 1e9)
        np.testing.assert_array_equal(x, np.zeros_like(x))

    @requires("baseline/config.json")
    def test_numba_matches_numpy_replay(self):
        p = archived_instance(power=4, size=(20, 100))
        p["budget"] = 30
        c = constants(p["A"], p["b"], p["power"])
        for spec in methods():
            result, rows, x = solve(p, spec, 30)
            rep = replay(p, spec, 30, c)
            m = min(len(rep), len(rows))
            np.testing.assert_allclose(rep[:m, 1], rows[:m, 2], rtol=1e-8, atol=1e-25, err_msg=str(spec))
            np.testing.assert_allclose(rep[:m, 2], rows[:m, 3], rtol=1e-8, atol=1e-14, err_msg=str(spec))

    @requires("baseline/config.json")
    def test_agm_iterates_stay_in_Z(self):
        p = archived_instance(power=3, size=(20, 100))
        c = constants(p["A"], p["b"], p["power"])
        rows, x, stats = run_agm(p["A"], p["b"], p["power"], p["z"], 2000, log_steps(1999), 1., 1., c["rho"])
        self.assertLessEqual(stats[10], 1e-10)  # max violation of Z or the cut
        self.assertLessEqual(np.linalg.norm(x), c["rho"]*(1+1e-12))

    @requires("baseline/config.json")
    def test_grid(self):
        jobs = build()
        self.assertEqual(len(jobs), 40)
        self.assertEqual(sum(len(j["methods"]) for j in jobs), 40*14)
        self.assertEqual(len(build(True)), 4)
        self.assertEqual({j["model"]["power"] for j in jobs}, {3, 4})


if __name__ == "__main__":
    unittest.main()
