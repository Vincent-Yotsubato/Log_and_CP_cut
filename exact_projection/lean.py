"""The outer loop of rates_and_inexact.inexact.run without per-step logging (timing only).

It calls the same evaluate, cuts and inner functions with the same arguments in the same
order, so it produces the same iterates and kernel-call counts; the tests and every timing
replay check this. Kept: the cut admissibility check, the Q-inclusion certificate
c_{k+1} + <w_{k+1}, b> >= 0 through the witness recursion, weak duality J_{k+1} <= Omega_upper,
and the distance test. Omitted: trace and audit rows, dual and correction histories, Omega(x_k),
and the diagnostic ||n_{k+1} + A^T w_{k+1}||, a matrix-vector product per outer step.
"""
import numpy as np
from numba import njit
from elastic_net.numerics import soft, omega, evaluate, cuts
from rates_and_inexact.inexact import inner


@njit(cache=True)
def lean_run(A, b, power, L0, L1, lam_en, p0, z, omega_upper, method, budget, solver,
             accept_exact, eta_rel, exponent, max_inner=100000, floor_factor=1000., tolerance=1e-12):
    """method: 0=H-CP, 1=H-Log. Return (evaluations, W, flag, first_N, first_W, p, x)."""
    p = p0.copy()
    x0 = soft(p0, lam_en)
    x = x0.copy()
    normal = np.zeros_like(p0)
    witness = np.zeros(b.size)
    c, J = 0., omega(x0, p0, lam_en)
    W, flag, evaluated, first_N, first_W = 1, 0, 0, -1, -1
    pi0 = 0.
    norm_z = np.linalg.norm(z)
    norm_b = np.linalg.norm(b)
    for k in range(budget):
        f, g, tau, residual, weights = evaluate(A, b, x, 1, 0., power)
        evaluated += 1
        q = np.linalg.norm(g)
        ca, _, sl, _ = cuts(q, L0, L1)
        sigma = f+ca if method == 0 else sl
        if q == 0.:
            sigma = 0.
        if k == 0:
            pi0 = .5*(sigma/q)**2 if q > 0. else 0.
        dist = np.linalg.norm(x-z)
        relative = dist/norm_z if norm_z > 0. else dist
        if first_N < 0 and relative <= .01:
            first_N, first_W = k+1, W
        if not np.isfinite(f+q+J):
            flag = 9
            break
        if tau > 0. and sigma/tau > 1+1e-10:
            flag = 6
            break
        if k == budget-1:
            break
        depth = np.dot(g, x)-sigma if q > 0. else 0.
        requested = eta_rel*pi0/(k+2.)**exponent
        eta = max(requested, floor_factor*np.finfo(np.float64).eps*(1+abs(J)))
        new_p, new_x, new_n, new_c, new_J, u, stats = inner(
            p0, x0, g, depth, normal, c, J, sigma, lam_en, eta,
            solver, accept_exact, max_inner, tolerance)
        W += int(stats[0])
        if stats[3] != 0.:
            flag = int(stats[3])
            break
        next_witness = stats[10]*weights+stats[11]*witness
        margin = new_c+np.dot(next_witness, b)
        if (margin < -tolerance*(1+abs(new_c)+norm_b*np.linalg.norm(next_witness))
                or new_J-omega_upper > 1e-10):
            flag = 7
            break
        p, x, normal, c, J, witness = new_p, new_x, new_n, new_c, new_J, next_witness
    return evaluated, W, flag, first_N, first_W, p, x
