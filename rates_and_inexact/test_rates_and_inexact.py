import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, LinearConstraint
from threadpoolctl import threadpool_limits
from elastic_net.problems import make_poly, make_e1, value_gradient, soft
from cut_functions import rho, gamma
from .inexact import project_two, candidate, inner, run
from .runner import prepare
from .protocol import build


RESULTS = Path(__file__).resolve().parents[1]/"results"


def requires(*paths):
    """Skip unless the upstream results exist; python run_all.py creates them."""
    missing = [p for p in paths if not (RESULTS/p).exists()]
    return unittest.skipIf(bool(missing), "needs " + ", ".join("results/"+p for p in missing) + " (run python run_all.py first)")


class RatesAndInexactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limiter = threadpool_limits(limits=1)

    def test_general_offsets_against_independent_primal_qp(self):
        rng = np.random.default_rng(20261031)
        for _ in range(35):
            H = rng.normal(size=(2, 6))
            feasible = rng.normal(size=6)
            rhs = H@feasible+rng.uniform(.01, .5, 2)
            x = rng.normal(size=6)
            u, a, v, _, flag = project_two(x, H[0], rhs[0], H[1], rhs[1])
            fit = minimize(lambda z: .5*np.linalg.norm(z-x)**2, feasible,
                           jac=lambda z: z-x, constraints=LinearConstraint(H, -np.inf, rhs),
                           method="SLSQP", options={"ftol": 1e-14, "maxiter": 1000})
            self.assertTrue(fit.success, fit.message)
            self.assertEqual(flag, 0)
            self.assertLess(np.max(np.abs(u-fit.x)), 1e-10)
            self.assertGreaterEqual(min(a, v), 0.)
            self.assertLess(np.max(H@u-rhs), 1e-10)

    def test_parallel_opposite_and_zero_normals(self):
        x = np.array([3., 2.])
        h = np.array([1., 0.])
        for g, b, n, c, expected in [
                (h, 1., h, .5, [.5, 2.]),
                (h, 1., -h, 1., [1., 2.]),
                (h, 1., -h, -1., [1., 2.]),
                (np.zeros(2), 0., h, 1., [1., 2.])]:
            out = project_two(x, g, b, n, c)
            self.assertEqual(out[4], 0)
            np.testing.assert_allclose(out[0], expected, atol=1e-12)
        self.assertEqual(project_two(x, h, 0., -h, -1.)[4], 7)

    def test_complementarity_implies_J_equals_Omega(self):
        p0, x0 = np.zeros(2), np.zeros(2)
        g, normal = np.array([-1., 0.]), np.zeros(2)
        out = candidate(p0, x0, g, -1., normal, 0., 0., 0., .2, 1.2, 0., 1e-12, True, 1e-12)
        p, x, J = out[0], out[1], out[4]
        self.assertTrue(out[10])
        self.assertAlmostEqual(J, .5*x@x+.2*np.abs(x).sum(), places=13)
        self.assertLess(abs(out[6]), 1e-12)

    def test_small_eta_trajectory_matches_high_accuracy_PG(self):
        # A small orthogonal control keeps the raw dual PG problem sufficiently
        # conditioned to compare tolerances without first hitting its work cap.
        p = make_poly(m=2, n=4, power=3, correlation=0., lam=.2, seed=0, orthogonal=True)
        args = (p.A, p.b, p.power, p.L0, p.L1, p.lam, p.p0, p.z,
                .5*p.z@p.z+p.lam*np.abs(p.z).sum()+1e-10, 0, 5, 0)
        a = run(*args, False, .01, 2., 100000, 0., 1e-12, 1e-14)
        e = run(*args, True, 0., 0., 100000, 0., 1e-12)
        self.assertEqual(a[-1][5], 0)
        self.assertEqual(e[-1][5], 0)
        np.testing.assert_allclose(a[0][:, 9:11], e[0][:, 9:11], rtol=0., atol=1e-10)
        np.testing.assert_allclose(a[3], e[3], rtol=0., atol=1e-10)

    def test_prescribed_start_and_raw_PG_preserve_progress(self):
        rng = np.random.default_rng(72)
        for _ in range(12):
            p0 = rng.normal(size=5)
            lam = .4
            x0 = soft(p0, lam)
            normal = rng.normal(size=5)
            p = p0-normal
            x = soft(p, lam)
            c = normal@x
            J = -.5*x@x-c
            g = rng.normal(size=5)
            sigma = rng.uniform(.01, .1)
            b = g@x-sigma
            aa = sigma/(g@g)
            out = candidate(p0, x0, g, b, normal, c, J, .5*sigma**2/(g@g), lam,
                            aa, 1., 1e-8, False, 1e-12)
            self.assertGreaterEqual(out[5], -1e-12*(1+abs(J)))
            for solver in (0, 2):
                solved = inner(p0, x0, g, b, normal, c, J, sigma, lam, 1e-6, solver, False, 1000)
                self.assertEqual(solved[-1][9], 0.)

    def test_normalized_PG_accepts_where_raw_PG_stalls(self):
        # Tiny gradient against a large memory normal: the raw step moves lambda by
        # O(q^2/||n||^2) per iteration, the normalized step does not.
        rng = np.random.default_rng(5)
        lam, p0 = .3, rng.normal(size=8)
        normal = 4*rng.normal(size=8)
        p = p0-normal
        x = soft(p, lam)
        c = normal@x
        J = -.5*x@x-c
        g = 1e-3*rng.normal(size=8)
        sigma = 1e-4*np.linalg.norm(g)
        b = g@x-sigma
        raw = inner(p0, soft(p0, lam), g, b, normal, c, J, sigma, lam, 1e-9, 0, False, 20000)
        scaled = inner(p0, soft(p0, lam), g, b, normal, c, J, sigma, lam, 1e-9, 2, False, 20000)
        self.assertEqual(scaled[-1][3], 0.)
        self.assertEqual(scaled[-1][9], 0.)
        self.assertGreaterEqual(scaled[4]-J, .5*(sigma/np.linalg.norm(g))**2-1e-12*(1+abs(J)))
        self.assertLessEqual(scaled[-1][2], raw[-1][2])

    def test_Q_certificate_for_arbitrary_feasible_points(self):
        p = make_poly(m=3, n=8, power=3, correlation=.1, lam=.1, seed=5)
        out = run(p.A, p.b, p.power, p.L0, p.L1, p.lam, p.p0, p.z,
                  .5*p.z@p.z+p.lam*np.abs(p.z).sum()+1e-10, 0, 10, 1, False, .01, 2.)
        normal, c, witness = out[4], out[5], out[7]
        self.assertTrue(np.all(witness >= 0.))
        np.testing.assert_allclose(normal, -p.A.T@witness, atol=1e-10)
        rng = np.random.default_rng(3)
        right_inverse = p.A.T@np.linalg.inv(p.A@p.A.T)
        for _ in range(10):
            z = p.z+right_inverse@(np.maximum(p.b-p.A@p.z, 0.)+rng.uniform(.01, 1., 3))
            self.assertGreaterEqual(np.min(p.A@z-p.b), -1e-12)
            self.assertLessEqual(normal@z, c+1e-10)

    @requires("elastic_net/nonsmooth/config.json")
    def test_default_grid(self):
        self.assertEqual(sum(len(j["methods"]) for j in build("inexact")), 640)
        self.assertEqual(len(build("inexact", True)), 4)

    def test_cap_failure_does_not_accept_or_modify_state(self):
        p = make_poly(m=3, n=8, power=4, correlation=.8, lam=.5, seed=0)
        out = run(p.A, p.b, p.power, p.L0, p.L1, p.lam, p.p0, p.z,
                  .5*p.z@p.z+p.lam*np.abs(p.z).sum()+1e-10, 0, 10, 0, False,
                  1e-10, 2., 0, 0.)
        self.assertEqual(out[-1][5], 8.)
        self.assertEqual(out[1][-1, 21], 0.)
        np.testing.assert_array_equal(out[3], soft(p.p0, p.lam))

    def test_zero_gradient_refines_memory_instead_of_stopping(self):
        p0, x0 = np.zeros(2), np.zeros(2)
        g, normal = np.zeros(2), np.array([-1., 0.])
        c, J = -.5, -.5*.9**2+.5
        out = inner(p0, x0, g, 0., normal, c, J, 0., .1, 1e-12, 0, False, 20)
        self.assertEqual(out[-1][3], 0.)
        np.testing.assert_allclose(out[1], [.5, 0.], atol=1e-12)
        self.assertGreater(out[4], J)

    @requires("baseline/config.json", "elastic_net/nonsmooth/config.json", "elastic_net/hlog/config.json")
    def test_resume_guard_and_historical_archive_protection(self):
        from .common import SOURCES, read, write_json
        with self.assertRaises(ValueError):
            prepare(SOURCES["baseline"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"new"
            original = prepare(output, pilot=True)
            self.assertEqual(prepare(output, pilot=True), original)
            changed = read(output/"config.json")
            changed["settings"]["gradient_budget"] = 99
            write_json(output/"config.json", changed)
            with self.assertRaises(RuntimeError):
                prepare(output, pilot=True)

    @requires("elastic_net/nonsmooth/config.json")
    def test_saved_states_support_independent_acceptance_audit(self):
        from .runner import worker
        from .common import read, load_problem
        from .verification import verify_inexact
        with tempfile.TemporaryDirectory() as temporary:
            for j in (build("inexact", True)[0], build("inexact", True)[2]):
                job = dict(j, gradient_budget=70,
                           methods=[dict(name="H-CP", configuration="SSN-A")])
                answer = worker(job, temporary)
                self.assertNotIn("error", answer, answer.get("error"))
                directory = Path(temporary)/"inexact/instances"/job["id"]
                result_path = next(directory.glob("*/result.json"))
                r = read(result_path)
                self.assertEqual(r["flag"], 0)
                self.assertEqual(verify_inexact(result_path.parent, load_problem(directory), r), [])
                self.assertEqual(r["inline_audit_errors"], [])
                states_path = result_path.parent/"audit_states.npz"
                with np.load(states_path) as saved:
                    states = {k: saved[k].copy() for k in saved.files}
                # Reject a deliberately corrupted certificate, even when the
                # result file still claims that the step was accepted.
                states["u"][0] += 100.
                np.savez_compressed(states_path, **states)
                self.assertTrue(verify_inexact(result_path.parent, load_problem(directory), r))


if __name__ == "__main__":
    unittest.main()
