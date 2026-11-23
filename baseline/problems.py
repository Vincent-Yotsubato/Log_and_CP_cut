"""Seeded instances and independently certified selection references."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import mpmath as mp
from scipy.optimize import brentq, least_squares, minimize, nnls
from scipy.linalg import solve_triangular


def soft(p, lam):
    return np.sign(p) * np.maximum(np.abs(p) - lam, 0.0)


@dataclass
class Problem:
    A: np.ndarray
    b: np.ndarray
    family: int  # 0: cosh (including beta=0), 1: positive-part power
    beta: float
    power: int
    L0: float
    L1: float
    lam: float
    p0: np.ndarray
    z: np.ndarray
    dual: np.ndarray
    certificate: dict
    metadata: dict


def value_gradient(A, b, x, family, beta=1.0, power=4):
    r = A @ x - b
    with np.errstate(over="raise", invalid="raise"):
        if family == 1:
            v = np.maximum(-r, 0)
            f = float(np.sum(v**power) / power)
            g = -(A.T @ v**(power - 1))
            return f, g, power * f
        if beta == 0:
            f = float(r @ r / (2 * len(b)))
            return f, A.T @ r / len(b), 2 * f
        s = np.sinh(beta * r)
        f = float(2 * np.sum(np.sinh(beta * r / 2)**2) / (len(b) * beta**2))
        return f, A.T @ s / (len(b) * beta), float(s @ r / (len(b) * beta))


def reference(A, b, p0, lam, inequality, known_dual=None):
    """NNLS or elastic-net dual, then a feasible primal-dual gap.

The stable Fenchel decomposition avoids subtracting two large objectives.
No optimizer success flag is substituted for a certificate.
"""
    gram = A @ A.T
    iterations = 0
    if known_dual is not None:
        y = np.array(known_dual, copy=True)
    elif lam == 0:
        rhs = b - A @ p0
        if inequality:
            L = np.linalg.cholesky(gram)
            y, _ = nnls(L.T, solve_triangular(L, rhs, lower=True), maxiter=10000)
        else:
            y = np.linalg.solve(gram, rhs)
    else:
        def objective(y):
            x = soft(p0 + A.T @ y, lam)
            return 0.5 * (x @ x) - b @ y, A @ x - b

        fit = minimize(objective, np.zeros(len(b)), jac=True, method="L-BFGS-B",
                       bounds=[(0, None)] * len(b) if inequality else None,
                       options={"gtol": 1e-13, "ftol": 1e-15, "maxiter": 20000,
                                "maxls": 50, "maxcor": 30})
        y = fit.x
        iterations = int(fit.nit)

        def residual(y):
            slack = A @ soft(p0 + A.T @ y, lam) - b
            return np.minimum(y, slack) if inequality else slack

        def jacobian(y):
            p = p0 + A.T @ y
            active = A[:, np.abs(p) > lam]
            J = active @ active.T
            if inequality:
                inactive = y <= A @ soft(p, lam) - b
                J[inactive] = np.eye(len(b))[inactive]
            return J

        fit2 = least_squares(residual, y, jac=jacobian, ftol=1e-14,
                             xtol=1e-14, gtol=1e-14, max_nfev=2000)
        if np.linalg.norm(residual(fit2.x), np.inf) < np.linalg.norm(residual(y), np.inf):
            y = fit2.x
        if inequality:
            y = np.maximum(y, 0)
    right_inverse = np.linalg.solve(gram, A).T
    # Windows longdouble has only float64 precision. Use a separately stored
    # 60-decimal feasible point so certificate accuracy does not depend on it.
    # This matters for kappa=100, a=1; do not relax the absolute 1e-10 threshold.
    with mp.workdps(60):
        Am, bm, ym, p0m = [mp.matrix(v.tolist()) for v in (A, b, y, p0)]
        lm = mp.mpf(float(lam))
        pm = p0m + Am.T * ym
        mapped = mp.matrix([mp.sign(t) * max(abs(t) - lm, 0) for t in pm])
        um = mapped.copy()
        Rm = mp.matrix(right_inverse.tolist())
        for _ in range(4):
            defect = bm - Am * um
            if inequality:
                defect = mp.matrix([max(t, 0) for t in defect])
            um += Rm * defect
        slack = Am * um - bm
        defect = mp.matrix([max(-t, 0) if inequality else abs(t) for t in slack])
        terms = [mp.mpf('.5') * (um[j] - mapped[j])**2 +
                 (lm * (abs(um[j]) - mp.sign(pm[j]) * um[j]) if abs(pm[j]) > lm
                  else lm * abs(um[j]) - pm[j] * um[j]) for j in range(len(p0))]
        raw_mp = mp.fsum(terms) + (ym.T * slack)[0]
        # Certify a right inverse by ||I-AR||_2 <= sqrt(||.||_1 ||.||_inf),
        # including float64 multiplication roundoff in the computed bound.
        eps = np.finfo(float).eps
        gamma_n = A.shape[1] * eps / (1 - A.shape[1] * eps)
        E = np.abs(np.eye(len(b)) - A @ right_inverse) + gamma_n * (np.abs(A) @ np.abs(right_inverse)) * (1 + 4 * eps)
        eta = float(np.sqrt(np.max(E.sum(axis=0)) * np.max(E.sum(axis=1)))) * (1 + 1e-12)
        if eta >= 1:
            raise ArithmeticError("Cannot certify the reference right inverse")
        norm_R_upper = float(np.linalg.norm(right_inverse)) * (1 + 1e-12)
        repair_norm_mp = norm_R_upper / (1 - eta) * mp.norm(defect)
        growth = mp.norm(um - p0m) + lm * mp.sqrt(len(p0))
        repair_bound_mp = growth * repair_norm_mp + repair_norm_mp**2 / 2
        # A wide numerical guard for 60-digit scalar arithmetic, far below the
        # requested accuracy. The certificate is numerical, not interval proof.
        arithmetic_guard = mp.mpf('1e-45') * (1 + mp.norm(um)**2 + lm * mp.norm(um, 1))
        gap_mp = max(raw_mp, 0) + repair_bound_mp + arithmetic_guard
        u = np.asarray([float(t) for t in um])
        stored_error = mp.norm(mp.matrix(u.tolist()) - um)
        auxiliary_point = [mp.nstr(t, 60) for t in um]
        gap, raw_gap, repair_bound = float(gap_mp), float(raw_mp), float(repair_bound_mp)
        distance_bound = float(mp.sqrt(2 * gap_mp) + repair_norm_mp + stored_error)
        complementarity = max(abs(ym[i] * slack[i]) for i in range(len(b))) if inequality else 0
        high_precision_residual = float(mp.norm(defect))
    stored_slack = A @ u - b
    stored_defect = np.maximum(-stored_slack, 0) if inequality else np.abs(stored_slack)
    cert = {"gap": gap, "raw_gap": raw_gap,
            "feasibility": float(np.max(stored_defect)), "repair_bound": repair_bound,
            "distance_bound": distance_bound,
            "complementarity": float(complementarity),
            "nonnegative_dual": bool(np.all(y >= 0)) if inequality else True,
            "iterations": iterations, "type": "E.10" if inequality else "E.9",
            "arithmetic": "mpmath_60_decimal_digits_with_residual_correction",
            "auxiliary_feasible_point": auxiliary_point,
            "auxiliary_feasibility_residual": high_precision_residual,
            "right_inverse_error_bound": eta}
    if gap >= 1e-10 or cert["feasibility"] > 1e-11 or raw_gap < -1e-10:
        raise ArithmeticError(f"Reference certificate failed: {cert}")
    return u, y, cert


def make_poly(seed=0, m=10, n=40, power=4, correlation=0.8,
              orthogonal=False, example=False, lam=0.0, null_init=False):
    rng = np.random.default_rng(seed)
    if example:
        A = np.array([[1., 0.], [1., 1.]])
        b = np.array([1., 3.])
        m, n = A.shape
    else:
        common = rng.normal(size=n) / np.sqrt(n)
        rows = rng.normal(size=(m, n)) / np.sqrt(n)
        A = np.sqrt(correlation) * common[None, :] + np.sqrt(1 - correlation) * rows
        if orthogonal:
            A = np.linalg.qr(A.T)[0].T
        b = 1 + np.abs(rng.normal(size=m))
    singular = np.linalg.svd(A, compute_uv=False)
    p0 = np.zeros(n)
    if null_init:
        p0 = rng.normal(size=n)
        p0 -= A.T @ np.linalg.solve(A @ A.T, A @ p0)
        p0 /= np.linalg.norm(p0)
    z, dual, cert = reference(A, b, p0, lam, True)
    return Problem(np.ascontiguousarray(A), b, 1, 0., power,
                   (1 if power == 3 else 4) * singular[0]**2,
                   singular[0]**2 / singular[-1], lam, p0, z, dual, cert,
                   {"seed": seed, "m": m, "n": n, "s": power,
                    "rho_c": correlation, "orthogonal": orthogonal,
                    "example": example, "null_init": null_init})


def make_e1(seed=0, m=128, n=512, kappa=1., beta=1., amplitude=0.25,
            lambda_factor=1., target_r=None):
    """Seeded sparse cosh/power instance with a certified reference.

    The support is floor(n/20). Changing beta preserves B,b,lambda and z.
    Calibration uses a fixed a=0.01 in experiment 2, so r0=1 is attainable.
    """
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(n, m)))[0]
    U = np.linalg.qr(rng.normal(size=(m, m)))[0]
    singular = np.geomspace(1., 1. / kappa, m)
    A = np.ascontiguousarray((U * singular) @ Q.T)
    dual = rng.normal(size=m)
    p_ref = A.T @ dual
    support = max(1, n // 20)
    ordered = np.sort(np.abs(p_ref))
    lam = float((ordered[-support - 1] + ordered[-support]) / 2)
    z = soft(p_ref, lam)
    scale = amplitude / np.sqrt(np.mean((A @ z)**2))
    z *= scale
    dual *= scale
    lam *= scale
    b = A @ z
    p0 = np.zeros(n)
    L0, L1 = 1. / m, beta * kappa
    lam *= lambda_factor
    alpha, discrepancy = 0., 0.
    if target_r is not None:
        direction = A[0].copy()

        def equation(alpha):
            x = soft(alpha * direction, lam)
            _, g, _ = value_gradient(A, b, x, 0, beta)
            return L1 * np.linalg.norm(g) / L0 - target_r

        left, fl = 0., equation(0.)
        # The prescribed fixed grid; the first bracket fixes root ambiguity.
        for right in np.arange(.125, 128.0001, .125):
            fr = equation(right)
            if fl * fr <= 0:
                alpha = brentq(equation, left, right, xtol=1e-13, rtol=1e-14)
                break
            left, fl = right, fr
        else:
            raise ArithmeticError(f"No calibration bracket for r={target_r}, a={amplitude}")
        p0 = alpha * direction
        discrepancy = abs(equation(alpha)) / target_r
    known = dual.copy() if lambda_factor == 1 else None
    if known is not None:
        known[0] -= alpha
    z, dual, cert = reference(A, b, p0, lam, False, known)
    return Problem(A, b, 0, beta, 4, L0, L1, lam, p0, z, dual, cert,
                   {"seed": seed, "m": m, "n": n, "kappa": kappa,
                    "beta": beta, "a": amplitude, "lambda_factor": lambda_factor,
                    "target_r": target_r, "alpha": alpha,
                    "calibration_relative_error": discrepancy, "support": support})
