"""Independent mathematical checks required before launching the grids.

Run: python -m unittest baseline.test_baseline -v
"""
import unittest
import numpy as np
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from .numerics import cuts, evaluate, proj2, project_one, h_projected_gradient, soft, omega, METHODS
from .problems import make_poly, make_e1, value_gradient, reference
from .runner import solve
from .protocol import build


class MathematicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limiter = threadpool_limits(limits=1)

    def test_theorem22_random_points(self):
        rng = np.random.default_rng(20)
        models = [make_poly(seed=3, power=s, correlation=c) for s in (3, 4) for c in (0., .8)]
        models += [make_e1(seed=2, m=8, n=32, beta=beta, kappa=kappa)
                   for beta in (0., 1., 4.) for kappa in (1., 10.)]
        for model in models:
            for scale in (1e-4, .1, 1., 5.):
                for _ in range(10):
                    x = rng.normal(size=len(model.z)) * scale
                    f, g, tau = value_gradient(model.A, model.b, x, model.family, model.beta, model.power)
                    A, B, S, _ = cuts(np.linalg.norm(g), model.L0, model.L1)
                    self.assertLessEqual(A, (tau - f) * (1 + 1e-12))
                    self.assertLessEqual(B, f * (1 + 1e-12))
                    self.assertLessEqual(S, tau * (1 + 1e-12))

    def test_series_and_inverse(self):
        from cut_functions import b_star_inverse, b_star, rho, gamma
        for q in np.logspace(-14, 8, 150):
            A, B, S, C = cuts(q, 2., 3.)
            self.assertAlmostEqual((A + B) / S, 1., delta=1e-12)
            self.assertLessEqual(C, S * (1 + 1e-12))
        for L1 in (0., 1., 4.):
            for epsilon in np.logspace(-14, 6, 20):
                q = b_star_inverse(epsilon, 2., L1)
                self.assertAlmostEqual(float(b_star(q, 2., L1)) / epsilon, 1., delta=5e-12)
                self.assertGreater(rho(epsilon, 2., L1), 0)
                self.assertGreater(gamma(epsilon, 2., L1, 2), 0)

    def test_example32_difference_recurrence(self):
        problem = make_poly(example=True)
        np.testing.assert_allclose(problem.z, [1.5, 1.5], atol=1e-14)
        for name in ("Polyak-1", "Polyak-2", "CP", "C-LBreI", "Log-LBreI"):
            result, trace, *_ = solve(problem, {"name": name, "theta": 1., "c": 1.},
                                     {"max_updates": 100, "stop_mode": 0, "dense": True})
            x = np.zeros(2)
            for row in trace[:-1]:
                before = x[0] - x[1]
                f, g, _ = value_gradient(problem.A, problem.b, x, 1, power=4)
                t = row[10]
                expected = before + t * max(1 - x[0], 0)**3
                x -= t * g
                self.assertAlmostEqual(x[0] - x[1], expected, delta=2e-14)
            self.assertGreater(trace[-1, 28], 0)

    def test_euclidean_projection_identity(self):
        model = make_e1(m=8, n=32, lambda_factor=0., kappa=10.)
        for explicit, projected in (("C-LBreI", "Proj-C"), ("Log-LBreI", "Proj-Log")):
            outputs = [solve(model, {"name": name, "theta": 1., "c": 1.},
                             {"max_updates": 100, "stop_mode": 0, "dense": True}) for name in (explicit, projected)]
            np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
            np.testing.assert_array_equal(outputs[0][2], outputs[1][2])

    def test_h_halfspaces_and_omega(self):
        rng = np.random.default_rng(5)
        for problem in (make_poly(example=True), make_poly(power=3), make_poly(power=4, correlation=0.)):
            A, b = problem.A, problem.b
            right = A.T @ np.linalg.inv(A @ A.T)
            feasible = [problem.z]
            for _ in range(5):
                z = rng.normal(size=A.shape[1])
                z += right @ (b + rng.uniform(0, 3, size=len(b)) - A @ z)
                feasible.append(z)
            for name in ("H-Log", "H-CP"):
                x = problem.p0.copy()
                normal = np.zeros_like(x)
                for _ in range(300):
                    f, g, tau = value_gradient(A, b, x, 1, power=problem.power)
                    a, _, sl, _ = cuts(np.linalg.norm(g), problem.L0, problem.L1)
                    sigma = sl if name == "H-Log" else f + a
                    for z in feasible:
                        self.assertGreaterEqual(np.dot(g, x - z) + 1e-12, sigma)
                        self.assertLessEqual(np.dot(normal, z - x), 1e-11)
                    new, aa, vv, status, ips = proj2(problem.p0, x, g, sigma, normal)
                    self.assertEqual(status, 0)
                    self.assertGreaterEqual(omega(new, problem.p0, 0) + 1e-11, omega(x, problem.p0, 0))
                    self.assertLessEqual(omega(new, problem.p0, 0), omega(problem.z, problem.p0, 0) + 1e-11)
                    normal = aa * g + vv * normal
                    x = problem.p0 - normal

    def test_references_all_kernels(self):
        for lam in (0., .1):
            for correlation in (0., .8):
                model = make_poly(m=20, n=100, lam=lam, correlation=correlation)
                self.assertLess(model.certificate["gap"], 1e-10)
        for factor in (0., 1., 2.):
            model = make_e1(m=16, n=64, lambda_factor=factor, kappa=10)
            self.assertLess(model.certificate["gap"], 1e-10)

    def test_ill_conditioned_reference_absolute_gap(self):
        model = make_e1(seed=0, kappa=100, beta=4, amplitude=1)
        self.assertLess(model.certificate["gap"], 1e-10)
        self.assertLess(model.certificate["auxiliary_feasibility_residual"], 1e-40)

    def test_project_one_against_scipy(self):
        rng = np.random.default_rng(8)
        for _ in range(60):
            p, g = rng.normal(size=(2, 30))
            lam, sigma = .7, 10.**rng.uniform(-5, 1)
            x = soft(p, lam)
            t, calls, residual, flag = project_one(p, x, g, sigma, lam)
            self.assertEqual(flag, 0)
            self.assertGreaterEqual(t * (1 + 1e-12), sigma / (g @ g))
            f = lambda step: g @ (x - soft(p - step * g, lam)) - sigma
            hi = sigma / (g @ g)
            while f(hi) < 0:
                hi *= 2
            expected = brentq(f, 0., hi, xtol=1e-15, rtol=1e-14)
            self.assertAlmostEqual(t / expected, 1., delta=1e-8)

    def test_elastic_h_dual_is_compatible(self):
        problem = make_poly(example=True, lam=.1)
        x, normal = soft(problem.p0, .1), np.zeros(2)
        for _ in range(6):
            f, g, _ = value_gradient(problem.A, problem.b, x, 1, power=4)
            ca, *_ = cuts(np.linalg.norm(g), problem.L0, problem.L1)
            p, aa, vv, inner, inner_ip, flag = h_projected_gradient(problem.p0, x, g, f + ca, normal, .1)
            self.assertEqual(flag, 0)
            new = soft(p, .1)
            np.testing.assert_allclose(p, problem.p0 - aa * g - vv * normal, atol=1e-14)
            self.assertLessEqual(g @ (new - x) + f + ca, 2e-12)
            normal, x = problem.p0 - p, new

    def test_calibration_and_frozen_grid(self):
        for beta in (1., 4.):
            for target in (1., 10., 100., 1000.):
                model = make_e1(m=16, n=64, beta=beta, target_r=target, amplitude=.01)
                self.assertLess(model.metadata["calibration_relative_error"], 1e-10)
        jobs = build()
        grid = [j for j in jobs if j["variant"] == "polyhedron"]
        self.assertEqual(len(grid), 80)
        self.assertEqual(sum(len(j["methods"]) for j in jobs if j["panel"] == 2), 600)
        self.assertEqual({j["variant"] for j in jobs},
                         {"example32", "polyhedron", "orthogonal_control", "calibrated", "quadratic_control"})

    def test_overflow_is_explicit_failure(self):
        model = make_e1(m=8, n=32, beta=1000, amplitude=1, lambda_factor=0.)
        result, trace, p, x, witness = solve(model, {"name": "CP", "theta": 1., "c": 1.},
                                           {"max_updates": 10, "stop_mode": 0, "dense": True})
        self.assertEqual(result["status"], "overflow")
        self.assertFalse(result["success"])
        self.assertEqual(result["M"], 1)
        self.assertTrue(np.all(np.isfinite(x)))


if __name__ == "__main__":
    unittest.main()
