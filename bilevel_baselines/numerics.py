"""Float64 baselines for the Euclidean kernel with p0 = 0, so Omega(x) = ||x||^2/2.

Every method evaluates f and grad f of the polyhedral model once per iteration
(baseline.numerics.evaluate, the kernel of the archived H runs). Monitoring
quantities (distances, f at an averaged output point) are not counted.

MNG-local  H method (exact two-halfspace projection, baseline.numerics.proj2) with
           the Lipschitz cut q^2/Lc; at L1 = 0 this is exact H-Log with L0 = Lc.
BiG-SAM    Sabach-Shtern (2017), s = 1, so S(x) = x - grad Omega(x) = 0 and
           x_k = (1-alpha_k) T_k(x_{k-1}), alpha_k = min{gamma/k, 1};
           T_k(x) = x - t_k grad f(x) with t_k = t (fixed) or 1/(L0+L1||grad f(x)||).
AGM-BiO    Cao, Jiang, Yazdandoost Hamedani, Mokhtari (2024), Algorithm 1 with
           L_f = 1 (upper level Omega), g_k = f_star = 0 and Z = {||x|| <= radius}.
"""
import math
import numpy as np
from numba import njit

from baseline.numerics import evaluate, proj2

COLUMNS = ["k", "N", "f", "relative_distance", "distance", "feasibility", "Omega_gap", "q", "parameter"]
NCOLS = 9  # = len(COLUMNS); numba needs a scalar constant
STATS = ["K", "N", "first_distance_N", "first_joint_N", "relative_distance", "distance", "f",
         "feasibility", "Omega_gap", "flag", "diagnostic", "diagnostic_count", "max_norm"]
FLAGS = {0: "budget", 1: "exact_zero", 2: "projection_failure", 3: "nonfinite"}
DISTANCE_TARGET = 1e-2
JOINT_FEASIBILITY = 1e-8
JOINT_DISTANCE = 1e-6


@njit(cache=True)
def f_only(A, b, x, power):
    v = np.maximum(b - A @ x, 0.)
    return np.sum(v ** power) / power, np.linalg.norm(np.minimum(A @ x - b, 0.))


@njit(cache=True)
def project_ball_halfspace(w, rho, a, beta):
    """Euclidean projection onto {u: ||u|| <= rho, <a,u> <= beta}; a != 0.

    Returns (u, branch, flag): branch 0 interior, 1 halfspace only, 2 ball only,
    3 both active; flag 1 if the set is empty to working accuracy.
    """
    na2 = np.dot(a, a)
    nw = np.linalg.norm(w)
    if nw <= rho and np.dot(a, w) <= beta:
        return w.copy(), 0, 0
    ph = w - max(0., np.dot(a, w) - beta) / na2 * a
    if np.linalg.norm(ph) <= rho:
        return ph, 1, 0
    pb = w * (rho / nw)
    if np.dot(a, pb) <= beta:
        return pb, 2, 0
    c = beta / na2 * a
    r2 = rho * rho - np.dot(c, c)
    if r2 < 0.:
        return w.copy(), 3, 1
    d = (w - c) - np.dot(w - c, a) / na2 * a
    nd = np.linalg.norm(d)
    if nd == 0.:
        # w-c is parallel to a: every point of the circle is closest; take one.
        i = np.argmin(np.abs(a))
        d = -a[i] / na2 * a
        d[i] += 1.
        nd = np.linalg.norm(d)
    return c + math.sqrt(r2) / nd * d, 3, 0


@njit(cache=True)
def run_mng(A, b, power, Lc, z, budget, log_steps):
    n = A.shape[1]
    p0 = np.zeros(n)
    x = np.zeros(n)
    normal = np.zeros(n)
    nz, nb, oz = np.linalg.norm(z), np.linalg.norm(b), .5 * np.dot(z, z)
    rows = np.empty((len(log_steps) + 2, NCOLS))
    stored, li = 0, 0
    first, joint, flag = -1, -1, 0
    max_ratio, inadmissible, max_norm = 0., 0, 0.
    f = rel = dist = feas = og = q = 0.
    k = 0
    for k in range(budget):
        f, g, tau, residual, weights = evaluate(A, b, x, 1, 0., power)
        q2 = np.dot(g, g)
        q = math.sqrt(q2)
        dist = np.linalg.norm(x - z)
        rel = dist / nz
        feas = np.linalg.norm(np.minimum(residual, 0.)) / (1. + nb)
        og = .5 * np.dot(x, x) - oz
        max_norm = max(max_norm, np.linalg.norm(x))
        if first < 0 and rel <= DISTANCE_TARGET:
            first = k + 1
        if joint < 0 and feas <= JOINT_FEASIBILITY and dist / (1. + nz) <= JOINT_DISTANCE:
            joint = k + 1
        sigma = (q * q) / Lc  # as cuts(q, Lc, 0) in baseline.numerics: H-Log with L1 = 0
        ratio = sigma / tau if tau > 0. else 0.
        max_ratio = max(max_ratio, ratio)
        if ratio > 1. + 1e-10:
            inadmissible += 1
        terminal = k == budget - 1 or q == 0.
        logged = li < len(log_steps) and k == log_steps[li]
        if logged or terminal:
            if logged:
                li += 1
            rows[stored] = np.array([float(k), k + 1., f, rel, dist, feas, og, q, ratio])
            stored += 1
        if q == 0.:
            flag = 1
            break
        if terminal:
            break
        p, lam, nu, fl, ips = proj2(p0, x, g, sigma, normal)
        if fl != 0:
            flag = 2
            break
        normal = lam * g + nu * normal
        x = p0 - normal
        if not np.all(np.isfinite(x)):
            flag = 3
            break
    stats = np.array([float(k), k + 1., float(first), float(joint), rel, dist, f, feas, og, float(flag),
                      max_ratio, float(inadmissible), max_norm])
    return rows[:stored], x, stats


@njit(cache=True)
def run_bigsam(A, b, power, z, budget, log_steps, step_rule, t_fixed, L0, L1, gamma, radius):
    """step_rule 0: t = t_fixed; 1: t_k = 1/(L0 + L1 ||grad f(x_{k-1})||)."""
    n = A.shape[1]
    x = np.zeros(n)
    nz, nb, oz = np.linalg.norm(z), np.linalg.norm(b), .5 * np.dot(z, z)
    rows = np.empty((len(log_steps) + 2, NCOLS))
    stored, li = 0, 0
    first, joint, flag, exits = -1, -1, 0, 0
    max_norm = 0.
    f = rel = dist = feas = og = q = 0.
    k = 0
    for k in range(budget):
        # Evaluation k+1 at x = x_k; BiG-SAM's step index is k+1.
        f, g, tau, residual, weights = evaluate(A, b, x, 1, 0., power)
        q = np.linalg.norm(g)
        dist = np.linalg.norm(x - z)
        rel = dist / nz
        feas = np.linalg.norm(np.minimum(residual, 0.)) / (1. + nb)
        og = .5 * np.dot(x, x) - oz
        if first < 0 and rel <= DISTANCE_TARGET:
            first = k + 1
        if joint < 0 and feas <= JOINT_FEASIBILITY and dist / (1. + nz) <= JOINT_DISTANCE:
            joint = k + 1
        t = t_fixed if step_rule == 0 else 1. / (L0 + L1 * q)
        alpha = min(gamma / (k + 1.), 1.)
        terminal = k == budget - 1
        logged = li < len(log_steps) and k == log_steps[li]
        if logged or terminal:
            if logged:
                li += 1
            rows[stored] = np.array([float(k), k + 1., f, rel, dist, feas, og, q, alpha])
            stored += 1
        if terminal:
            break
        # y = T(x), z = S(x) = 0 for s = 1 and p0 = 0, x+ = alpha z + (1-alpha) y.
        x = (1. - alpha) * (x - t * g)
        nx = np.linalg.norm(x)
        max_norm = max(max_norm, nx)
        if nx > radius:
            exits += 1
        if not np.all(np.isfinite(x)):
            flag = 3
            break
    stats = np.array([float(k), k + 1., float(first), float(joint), rel, dist, f, feas, og, float(flag),
                      0., float(exits), max_norm])
    return rows[:stored], x, stats


@njit(cache=True)
def run_agm(A, b, power, z, budget, log_steps, gamma, Lf, radius):
    """Algorithm 1 of AGM-BiO; the cut uses g_k = f_star = 0. Output sequence x_k."""
    n = A.shape[1]
    x = np.zeros(n)
    zz = np.zeros(n)
    Ak = 0.
    nz, nb, oz = np.linalg.norm(z), np.linalg.norm(b), .5 * np.dot(z, z)
    rows = np.empty((len(log_steps) + 2, NCOLS))
    stored, li = 0, 0
    first, joint, flag = -1, -1, 0
    max_violation, both_active, max_norm = 0., 0, 0.
    f = rel = dist = feas = og = q = 0.
    k = 0
    for k in range(budget):
        ak = gamma * (k + 1.) / (4. * Lf)
        y = (Ak * x + ak * zz) / (Ak + ak)
        fy, gy, tauy, ry, wy = evaluate(A, b, y, 1, 0., power)  # evaluation k+1
        q = np.linalg.norm(gy)
        w = zz - ak * y  # grad Omega(y) = y for p0 = 0
        if q == 0.:
            nw = np.linalg.norm(w)
            znew = w * min(1., radius / nw) if nw > 0. else w
            branch = 2
        else:
            beta = np.dot(gy, y) - fy
            znew, branch, fl = project_ball_halfspace(w, radius, gy, beta)
            if fl != 0:
                flag = 2
                break
            max_violation = max(max_violation, (np.dot(gy, znew) - beta) / (1. + abs(beta)))
        if branch == 3:
            both_active += 1
        max_violation = max(max_violation, np.linalg.norm(znew) / radius - 1.)
        x = (Ak * x + ak * znew) / (Ak + ak)
        zz = znew
        Ak += ak
        # Monitoring at the output x_{k+1}, produced with k+1 evaluations.
        f, infeas = f_only(A, b, x, power)
        dist = np.linalg.norm(x - z)
        rel = dist / nz
        feas = infeas / (1. + nb)
        og = .5 * np.dot(x, x) - oz
        max_norm = max(max_norm, np.linalg.norm(x))
        if first < 0 and rel <= DISTANCE_TARGET:
            first = k + 1
        if joint < 0 and feas <= JOINT_FEASIBILITY and dist / (1. + nz) <= JOINT_DISTANCE:
            joint = k + 1
        terminal = k == budget - 1
        logged = li < len(log_steps) and k == log_steps[li]
        if logged or terminal:
            if logged:
                li += 1
            rows[stored] = np.array([float(k), k + 1., f, rel, dist, feas, og, q, ak])
            stored += 1
        if not np.all(np.isfinite(x)):
            flag = 3
            break
    stats = np.array([float(k), k + 1., float(first), float(joint), rel, dist, f, feas, og, float(flag),
                      max_violation, float(both_active), max_norm])
    return rows[:stored], x, stats
