from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


def _validate(q, L0, L1):
    q = np.asarray(q, dtype=float)
    if L0 <= 0 or L1 < 0 or not np.isfinite(L0 + L1):
        raise ValueError("L0 必须为正数，L1 必须为非负有限数。")
    if np.any(q < 0) or not np.all(np.isfinite(q)):
        raise ValueError("梯度范数必须为非负有限数。")
    return q


def a_star(q, L0, L1):
    q = _validate(q, L0, L1)
    if L1 == 0:
        return q * q / (2 * L0)
    r = L1 * q / L0
    small = r < 1e-4
    result = np.empty_like(q)
    t = r[small]
    result[small] = q[small] ** 2 / L0 * (
        0.5 + t * (-1 / 6 + t * (1 / 12 + t * (-1 / 20 + t / 30)))
    )
    t = r[~small]
    result[~small] = L0 / L1**2 * ((1 + t) * np.log1p(t) - t)
    return result


def b_star(q, L0, L1):
    q = _validate(q, L0, L1)
    if L1 == 0:
        return q * q / (2 * L0)
    r = L1 * q / L0
    small = r < 1e-4
    result = np.empty_like(q)
    t = r[small]
    result[small] = q[small] ** 2 / L0 * (
        0.5 + t * (-1 / 3 + t * (1 / 4 + t * (-1 / 5 + t / 6)))
    )
    t = r[~small]
    result[~small] = L0 / L1**2 * (t - np.log1p(t))
    return result


def s_c(q, L0, L1):
    q = _validate(q, L0, L1)
    return q * (q / (L0 + L1 * q))


def s_log(q, L0, L1):
    q = _validate(q, L0, L1)
    if L1 == 0:
        return q * q / L0
    return q / L1 * np.log1p(L1 * q / L0)


def s_v(q, L0, L1):
    q = _validate(q, L0, L1)
    if L1 == 0:
        return q * q / L0
    r = L1 * q / L0
    small = r < 1e-4
    result = np.empty_like(q)
    t = r[small]
    result[small] = q[small] ** 2 / L0 * (
        1 + t * (-5 / 6 + t * (1 - 27 * t / 20 + 29 * t * t / 15))
    )
    t = r[~small]
    result[~small] = L0 / L1**2 * (
        (1 + 2 * t) * np.log1p(2 * t) - t * np.log1p(t) - 2 * t
    )
    return result


def phi(u):
    u = np.asarray(u, dtype=float)
    if np.any(u < 0) or not np.all(np.isfinite(u)):
        raise ValueError("指数余项函数的参数必须为非负有限数。")
    result = np.empty_like(u)
    small = u < 1e-4
    t = u[small]
    result[small] = t * t * (0.5 + t * (1 / 6 + t * (1 / 24 + t / 120)))
    result[~small] = np.expm1(u[~small]) - u[~small]
    return result


def phi_inverse(xi):
    xi = float(xi)
    if xi < 0 or not np.isfinite(xi):
        raise ValueError("逆函数参数必须为非负有限数。")
    if xi == 0:
        return 0.0
    upper = np.log1p(xi) + 1.0
    return brentq(
        lambda u: float(phi(u)) - xi,
        0.0,
        upper,
        xtol=np.finfo(float).tiny,
        rtol=8 * np.finfo(float).eps,
    )


def b_star_inverse(epsilon, L0, L1):
    _validate(epsilon, L0, L1)
    if L1 == 0:
        return float(np.sqrt(2 * L0 * epsilon))
    u = phi_inverse(L1 * L1 * epsilon / L0)
    return float(L0 / L1 * np.expm1(u))


def rho(epsilon, L0, L1):
    _validate(epsilon, L0, L1)
    if L1 == 0:
        return float(np.sqrt(2 * epsilon / L0))
    return phi_inverse(L1 * L1 * epsilon / L0) / L1


def gamma(epsilon, L0, L1, alpha=1.0):
    if epsilon <= 0 or not 0 < alpha <= 2:
        raise ValueError("epsilon 必须为正数，alpha 必须属于 (0,2]。")
    q = b_star_inverse(epsilon, L0, L1)
    ratio = epsilon / q
    return float(alpha * (2 - alpha) * ratio**2 + 2 * alpha * ratio * a_star(q, L0, L1) / q)


def complexity_threshold(method, epsilon, radius, L0, L1, theta=1.0):
    method = {"polyak1": "Polyak-1", "polyak2": "Polyak-2", "cp": "CP", "c": "C",
              "log": "Log", "h_cp": "H-CP", "h_log": "H-Log",
              "proj_c": "Proj-C", "proj_log": "Proj-Log"}.get(method, method)
    if epsilon <= 0 or radius < 0 or not 0 < theta < 2:
        raise ValueError("复杂度界参数不满足定义域。")
    if radius == 0:
        return 0.0
    if method in {"Polyak-1", "Polyak-2"}:
        return radius**2 / gamma(epsilon, L0, L1, float(method[-1]))
    if method in {"CP", "H-CP"}:
        return radius**2 / rho(epsilon, L0, L1) ** 2
    factor = 1.0 if method.startswith("H-") or method.startswith("Proj-") else theta * (2 - theta)
    if method in {"Log", "Log-LBreI", "H-Log", "Proj-Log"}:
        if L1 == 0:
            return L0**2 * radius**4 / (factor * epsilon**2)
        return L1**2 * radius**2 / (factor * np.log1p(L1 * epsilon / (L0 * radius)) ** 2)
    if method in {"C", "C-LBreI", "Proj-C"}:
        return radius**2 * (L0 * radius + L1 * epsilon) ** 2 / (factor * epsilon**2)
    raise ValueError(f"方法 {method} 没有此处实现的复杂度界。")


def gradient_bound(method, N, radius, L0, L1, theta=1.0):
    method = {"c": "C", "log": "Log", "h_log": "H-Log",
              "proj_c": "Proj-C", "proj_log": "Proj-Log"}.get(method, method)
    if N < 1 or radius < 0 or not 0 < theta < 2:
        raise ValueError("梯度界参数不满足定义域。")
    factor = 1.0 if method.startswith("H-") or method.startswith("Proj-") else theta * (2 - theta)
    w = radius / np.sqrt(factor * N)
    if method in {"C", "C-LBreI", "Proj-C"}:
        return np.inf if L1 * w >= 1 else L0 * w / (1 - L1 * w)
    if method in {"Log", "Log-LBreI", "H-Log", "Proj-Log"}:
        if L1 == 0:
            return L0 * w
        return L0 / L1 * np.expm1(L1 * w)
    raise ValueError(f"方法 {method} 没有此处实现的梯度界。")


@dataclass(frozen=True)
class ElasticNet:
    lam: float = 0.0
    mu: float = 1.0

    def __post_init__(self):
        if self.lam < 0 or self.mu <= 0 or not np.isfinite(self.lam + self.mu):
            raise ValueError("ElasticNet 要求 lam 非负、mu 为正且均为有限数。")

    def map(self, p):
        p = np.asarray(p, dtype=float)
        return np.sign(p) * np.maximum(np.abs(p) - self.lam, 0.0) / self.mu

    def primal(self, p):
        return self.map(p)

    def value(self, x):
        return float(self.lam * np.sum(np.abs(x)) + self.mu * np.dot(x, x) / 2)

    def conjugate(self, p):
        active = np.maximum(np.abs(p) - self.lam, 0.0)
        return float(np.dot(active, active) / (2 * self.mu))

    def omega(self, x, p0):
        return self.value(x) - float(np.dot(p0, x))

    def bregman(self, z, p, x=None):
        z, p = np.asarray(z), np.asarray(p)
        x = self.map(p) if x is None else np.asarray(x)
        distance = self.mu * np.dot(z - x, z - x) / 2
        if self.lam:
            active = np.abs(p) > self.lam
            distance += self.lam * np.sum(np.abs(z[active]) - np.sign(p[active]) * z[active])
            distance += np.sum(self.lam * np.abs(z[~active]) - p[~active] * z[~active])
        return float(distance)
