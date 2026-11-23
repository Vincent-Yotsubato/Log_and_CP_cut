"""Algorithm 2 with general memory offsets and the two acceptance tests (i) and (ii).

solver 0: projected gradient in the original multipliers with step mu/(||g||^2+||n||^2).
          Kept only as a diagnostic: when q_k << ||n_k|| the lambda-direction moves by
          O(q_k^2/||n_k||^2) per step, so it reaches the inner cap on every instance.
solver 2: the same projected gradient on the same dual, in the normalized coordinates
          (lambda*||g||, nu*||n||), whose Lipschitz constant is (1+|cos(g,n)|)/mu.
solver 1: safeguarded semismooth Newton in the normalized coordinates.
All solvers start from (mu*sigma/q^2, 1) and offer every iterate to the acceptance tests. There is no
silent fallback from a failed approximate solve to the exact algorithm.
"""
import math
import numpy as np
from numba import njit
from elastic_net.numerics import soft, omega, evaluate, cuts

TRACE_COLUMNS = ["k", "n_grad", "n_mat", "n_omega", "n_certificate_ip", "delta", "q",
                 "best_delta", "best_q", "dist_to_zdag", "relative_distance", "affine_residual",
                 "Omega", "Omega_gap", "J", "selection_bound", "flag"]
AUDIT_COLUMNS = ["k", "lambda", "nu", "b", "sigma", "q", "J_previous", "J_next",
                 "c_previous", "c_next", "eta_requested", "eta_used", "progress_margin",
                 "gap", "u_feasibility", "kkt", "inner_iterations", "kernel_calls",
                 "certificate_ip", "start_accepted", "pg_i_violations", "accepted",
                 "Q_certificate_margin", "weak_duality_excess", "Omega_minus_J"]
STAT_COLUMNS = ["K", "N", "M", "W", "C", "flag", "delta0", "delta", "distance",
                "relative_distance", "feasibility", "first_distance_N", "first_distance_W",
                "first_joint_N", "eta_floor_uses", "pg_i_violations", "start_accepts",
                "max_progress_violation", "max_gap_violation", "max_u_infeasibility",
                "max_Q_violation", "max_weak_duality_excess", "max_normal_error",
                "max_cut_ratio", "first_distance_C"]


@njit(cache=True)
def project_two(point, g, b, normal, c):
    """Euclidean projection onto <g,u><=b, <n,u><=c, including degenerate normals.

    Return (u, multiplier_g, multiplier_n, certificate inner-products, flag).
    Opposite parallel normals describe a slab, not automatically an empty set.
    """
    q = np.linalg.norm(g)
    nn = np.linalg.norm(normal)
    cost = 2
    if q == 0. and b < -1e-12:
        return point.copy(), 0., 0., cost, 7
    if nn == 0. and c < -1e-12:
        return point.copy(), 0., 0., cost, 7
    if q == 0. and nn == 0.:
        return point.copy(), 0., 0., cost, 0
    if q == 0.:
        v = max((np.dot(normal, point)-c)/(nn*nn), 0.)
        return point-v*normal, 0., v, cost+1, 0
    if nn == 0.:
        a = max((np.dot(g, point)-b)/(q*q), 0.)
        return point-a*g, a, 0., cost+1, 0
    h, j = g/q, normal/nn
    sg, sn = np.dot(h, point)-b/q, np.dot(j, point)-c/nn
    cost += 2
    if sg <= 0. and sn <= 0.:
        return point.copy(), 0., 0., cost, 0
    a = max(sg, 0.)
    u = point-a*h
    cost += 1
    if np.dot(j, u)-c/nn <= 2e-15*(1+np.linalg.norm(u)):
        return u, a/q, 0., cost, 0
    v = max(sn, 0.)
    u = point-v*j
    cost += 1
    if np.dot(h, u)-b/q <= 2e-15*(1+np.linalg.norm(u)):
        return u, 0., v/nn, cost, 0
    cosine = np.dot(h, j)
    orth = j-cosine*h
    square = np.dot(orth, orth)
    cost += 2
    if square <= 1e-28:
        # A nonempty slab or identical-direction halfspaces was caught above.
        return point.copy(), 0., 0., cost, 7
    v = (sn-cosine*sg)/square
    a = sg-cosine*v
    if min(a, v) < -1e-12*(1+abs(a)+abs(v)):
        return point.copy(), 0., 0., cost, 7
    a, v = max(a, 0.), max(v, 0.)
    return point-a*h-v*j, a/q, v/nn, cost, 0


@njit(cache=True)
def stable_bregman(u, p, x, lam_en):
    total = .5*np.dot(u-x, u-x)
    for i in range(u.size):
        if p[i] > lam_en:
            total += lam_en*(abs(u[i])-u[i])
        elif p[i] < -lam_en:
            total += lam_en*(abs(u[i])+u[i])
        else:
            total += lam_en*abs(u[i])-p[i]*u[i]
    return total


@njit(cache=True)
def candidate(p0, x0, g, b, normal, c, J, progress, lam_en, a, v, eta, accept_exact, tolerance):
    """One actual kernel call. u is a certificate and never the next iterate."""
    new_normal = a*g+v*normal
    new_c = a*b+v*c
    p = p0-new_normal
    x = soft(p, lam_en)
    new_J = -.5*np.dot(x, x)-new_c
    margin = new_J-J-progress
    q, nn = np.linalg.norm(g), np.linalg.norm(normal)
    da, dv = b-np.dot(g, x), c-np.dot(normal, x)
    kkt = 0.
    if q > 0.:
        kkt = max(kkt, abs(min(a*q, da/q)), abs(a*da)/(1+abs(new_J)))
    if nn > 0.:
        kkt = max(kkt, abs(min(v*nn, dv/nn)), abs(v*dv)/(1+abs(new_J)))
    # Exact comparator does not pay the error-certificate cost at each PG step.
    if accept_exact and kkt > 1e-12:
        return p, x, new_normal, new_c, new_J, margin, np.inf, np.inf, kkt, 0, False, x
    u, _, _, cost, flag = project_two(x, g, b, normal, c)
    su, sn = b-np.dot(g, u), c-np.dot(normal, u)
    cost += 3  # feasibility products plus the stable Bregman norm product
    feasibility = max(0., -su, -sn)
    gap = stable_bregman(u, p, x, lam_en)+a*su+v*sn
    admissible = flag == 0 and feasibility <= 1e-12
    if accept_exact:
        accepted = admissible and kkt <= 1e-12
    else:
        accepted = (admissible and margin >= -tolerance*(1+abs(J))
                    and gap <= eta and gap >= -tolerance*(1+abs(new_J)))
    return p, x, new_normal, new_c, new_J, margin, gap, feasibility, kkt, cost, accepted, u


@njit(cache=True)
def inner(p0, x0, g, b, normal, c, J, sigma, lam_en, eta,
          solver=0, accept_exact=False, max_inner=100000, tolerance=1e-12):
    q2, nn2 = np.dot(g, g), np.dot(normal, normal)
    q, nn = math.sqrt(q2), math.sqrt(nn2)
    a, v = (sigma/q2 if q2 > 0. else 0.), 1.
    progress = .5*sigma*sigma/q2 if q2 > 0. else 0.
    calls, cost, violations = 0, 0, 0
    shortcut_steps = 0
    first = True
    # In SSN coordinates these are lambda*||g||, nu*||n||.
    h = g/q if q > 0. else np.zeros_like(g)
    j = normal/nn if nn > 0. else np.zeros_like(normal)
    ah, vh = a*q, v*nn
    for iteration in range(max_inner+1):
        out = candidate(p0, x0, g, b, normal, c, J, progress, lam_en,
                        a, v, eta, accept_exact, tolerance)
        p, x, new_n, new_c, new_J, margin, gap, feasibility, kkt, ips, accepted, u = out
        calls += 1
        cost += ips
        if solver != 1 and margin < -tolerance*(1+abs(J)):
            violations += 1
        if accepted:
            stats = np.array([calls, cost, iteration+shortcut_steps, 0., margin, gap, feasibility,
                              kkt, 1. if first else 0., violations, a, v])
            return p, x, new_n, new_c, new_J, u, stats
        if not np.isfinite(new_J) or violations > 0:
            break
        if not accept_exact and kkt <= 1e-12:
            # A converged SSN candidate cannot be silently substituted for the acceptance tests.
            if solver == 1:
                stats = np.array([calls, cost, iteration+shortcut_steps, 11., margin, gap, feasibility,
                                  kkt, 0., violations, a, v])
                return p, x, new_n, new_c, new_J, u, stats
        if iteration+shortcut_steps >= max_inner or q2+nn2 == 0.:
            break
        first = False
        if solver == 0:
            # The unscaled projected-gradient step with the constant L_Psi.
            a = max(0., a-(b-np.dot(g, x))/(q2+nn2))
            v = max(0., v-(c-np.dot(normal, x))/(q2+nn2))
            continue
        if solver == 2:
            # Same dual, coordinates (lambda*q, nu*||n||): gradient components are
            # divided by q and ||n||, and the Hessian is bounded by the Gram matrix
            # of the unit normals, whose largest eigenvalue is 1+|cos(g,n)|.
            lip = (1.+abs(np.dot(h, j)) if q > 0. and nn > 0. else 1.)
            if q > 0.:
                a = max(0., a*q-(b-np.dot(g, x))/(q*lip))/q
            if nn > 0.:
                v = max(0., v*nn-(c-np.dot(normal, x))/(nn*lip))/nn
            continue
        if iteration == 0 and q > 0.:
            # A safeguarded scalar semismooth-Newton single-active branch,
            # tested at every candidate against the stored Q offset and the acceptance tests.
            # Without
            # this branch a singular active-coordinate Hessian can force an
            # otherwise fast Newton solve into extremely slow gradient steps.
            depth0 = np.dot(h, x0)-b/q
            if depth0 > 0.:
                root, low, high = depth0, 0., np.inf
                for _ in range(min(60, max_inner)):
                    shortcut_steps += 1
                    shortcut = candidate(p0, x0, g, b, normal, c, J, progress, lam_en,
                                         root/q, 0., eta, accept_exact, tolerance)
                    calls += 1
                    cost += shortcut[9]
                    if shortcut[10]:
                        stats = np.array([calls, cost, shortcut_steps, 0., shortcut[5], shortcut[6], shortcut[7],
                                          shortcut[8], 0., violations, root/q, 0.])
                        return shortcut[0], shortcut[1], shortcut[2], shortcut[3], shortcut[4], shortcut[11], stats
                    error = np.dot(h, shortcut[1])-b/q
                    if abs(error) <= min(1e-13, 1e-11/max(q, 1.)):
                        break
                    if error > 0.:
                        low = root
                    else:
                        high = root
                    active0 = np.abs(shortcut[0]) > lam_en
                    derivative = np.dot(h[active0], h[active0])
                    trial_root = root+error/derivative if derivative > 1e-20 else np.inf
                    if not low < trial_root < high:
                        trial_root = (low+high)/2 if np.isfinite(high) else 2*max(root, depth0)
                    root = trial_root
        ga = (b-np.dot(g, x))/q if q > 0. else 0.
        gv = (c-np.dot(normal, x))/nn if nn > 0. else 0.
        active = np.abs(p) > lam_en
        hh = np.dot(h[active], h[active])
        jj = np.dot(j[active], j[active])
        hj = np.dot(h[active], j[active])
        if ah <= 1e-15 and ga >= 0.:
            da, dv = 0., -gv/max(jj, 1e-12)
        elif vh <= 1e-15 and gv >= 0.:
            da, dv = -ga/max(hh, 1e-12), 0.
        elif hh > 1e-14:
            orth = j[active]-(hj/hh)*h[active]
            den = np.dot(orth, orth)
            if den > 1e-22:
                dv = (-gv+(hj/hh)*ga)/den
                da = (-ga-hj*dv)/hh
            else:
                da, dv = max(0., ah-ga/2)-ah, max(0., vh-gv/2)-vh
        else:
            da, dv = max(0., ah-ga/2)-ah, max(0., vh-gv/2)-vh
        if ah <= 1e-15 and da < 0.:
            da = 0.
        if vh <= 1e-15 and dv < 0.:
            dv = 0.
        slope = ga*da+gv*dv
        if slope >= 0. or not np.isfinite(slope):
            da, dv = max(0., ah-ga/2)-ah, max(0., vh-gv/2)-vh
            slope = ga*da+gv*dv
        step = 1.
        if da < 0.:
            step = min(step, -ah/da)
        if dv < 0.:
            step = min(step, -vh/dv)
        descended = False
        for _ in range(50):
            na, nv = max(0., ah+step*da), max(0., vh+step*dv)
            aa, vv = (na/q if q > 0. else 0.), (nv/nn if nn > 0. else 0.)
            trial = candidate(p0, x0, g, b, normal, c, J, progress, lam_en,
                              aa, vv, eta, accept_exact, tolerance)
            calls += 1
            cost += trial[9]
            # Every trial is offered to the acceptance conditions before Armijo.
            if trial[10]:
                stats = np.array([calls, cost, iteration+shortcut_steps+1., 0., trial[5], trial[6], trial[7],
                                  trial[8], 0., violations, aa, vv])
                return trial[0], trial[1], trial[2], trial[3], trial[4], trial[11], stats
            if (trial[4] >= new_J-1e-4*step*slope or
                    (trial[4] >= new_J-4e-15*(1+abs(new_J)) and trial[8] < kkt)):
                ah, vh = na, nv
                descended = True
                break
            step *= .5
        if not descended:
            ah, vh = max(0., ah-ga/2), max(0., vh-gv/2)
        a, v = (ah/q if q > 0. else 0.), (vh/nn if nn > 0. else 0.)
    flag = 7. if violations > 0 else 8.
    stats = np.array([calls, cost, min(iteration+shortcut_steps, max_inner), flag, margin, gap, feasibility,
                      kkt, 0., violations, a, v])
    return p, x, new_n, new_c, new_J, u, stats


@njit(cache=True)
def run(A, b, power, L0, L1, lam_en, p0, z, omega_upper, method,
        budget, solver, accept_exact, eta_rel, exponent, max_inner=100000,
        floor_factor=1000., tolerance=1e-12, fixed_eta=-1.):
    """method: 0=H-CP, 1=H-Log; solver: 0=raw PG, 1=safeguarded SSN, 2=normalized PG."""
    p = p0.copy()
    x0 = soft(p0, lam_en)
    x = x0.copy()
    normal = np.zeros_like(p0)
    witness = np.zeros(b.size)
    c, J = 0., omega(x0, p0, lam_en)
    rows = np.empty((budget, 17))
    audit = np.empty((max(1, budget), 25))
    dual_history = np.empty((budget, p0.size))
    corrections = np.empty((budget, p0.size))
    used, n_audit, W, C = 0, 0, 1, 0
    flag, floor_uses, violations, starts = 0, 0, 0, 0
    first_N, first_W, joint_N, first_C = -1, -1, -1, -1
    max_i, max_ii, max_feas, max_Q, max_weak, max_normal, max_ratio = 0., 0., 0., 0., 0., 0., 0.
    best, best_q, f0, pi0 = np.inf, np.inf, 0., 0.
    norm_z, norm_b = np.linalg.norm(z), np.linalg.norm(b)
    omega_z = omega(z, p0, lam_en)
    for k in range(budget):
        dual_history[k] = p
        f, g, tau, residual, weights = evaluate(A, b, x, 1, 0., power)
        q = np.linalg.norm(g)
        if k == 0:
            f0 = f
        ca, _, sl, _ = cuts(q, L0, L1)
        sigma = f+ca if method == 0 else sl
        if q == 0.:
            sigma = 0.
        if k == 0:
            pi0 = .5*(sigma/q)**2 if q > 0. else 0.
        best, best_q = min(best, f), min(best_q, q)
        dist = np.linalg.norm(x-z)
        relative = dist/norm_z if norm_z > 0. else dist
        feasibility = np.linalg.norm(np.minimum(residual, 0.))/(1+norm_b)
        om = omega(x, p0, lam_en)
        if first_N < 0 and relative <= .01:
            first_N, first_W, first_C = k+1, W, C
        if joint_N < 0 and feasibility <= 1e-8 and dist/(1+norm_z) <= 1e-6:
            joint_N = k+1
        rows[used] = np.array([k, k+1., 2.*(k+1), W, C, f, q, best, best_q,
                              dist, relative, feasibility, om, omega_z-om, J, omega_upper-J, flag])
        used += 1
        if not np.isfinite(f+q+J):
            flag = 9
            break
        if tau > 0.:
            max_ratio = max(max_ratio, sigma/tau)
            if sigma/tau > 1+1e-10:
                flag = 6
                break
        if k == budget-1:
            break
        depth = np.dot(g, x)-sigma if q > 0. else 0.
        # Step k accepts with eta_{k+1}, where eta_j = eta_rel*pi0*(j+1)^(-p).
        requested = eta_rel*pi0/(k+2.)**exponent if fixed_eta < 0. else fixed_eta
        floor = floor_factor*np.finfo(np.float64).eps*(1+abs(J))
        eta = max(requested, floor)
        if not accept_exact and eta > requested:
            floor_uses += 1
        new_p, new_x, new_n, new_c, new_J, certificate_u, stats = inner(
            p0, x0, g, depth, normal, c, J, sigma, lam_en, eta,
            solver, accept_exact, max_inner, tolerance)
        W += int(stats[0])
        corrections[n_audit] = certificate_u
        C += int(stats[1])+2
        accepted = stats[3] == 0.
        aa, vv = stats[10], stats[11]
        next_witness = aa*weights+vv*witness
        Q_margin = new_c+np.dot(next_witness, b)
        excess = new_J-omega_upper
        deviation = omega(new_x, p0, lam_en)-new_J
        audit[n_audit] = np.array([k, aa, vv, depth, sigma, q, J, new_J, c, new_c,
                                  requested, eta, stats[4], stats[5], stats[6], stats[7],
                                  stats[2], stats[0], stats[1], stats[8], stats[9],
                                  1. if accepted else 0., Q_margin, excess, deviation])
        n_audit += 1
        violations += int(stats[9])
        if not accepted:
            flag = int(stats[3])
            break
        max_i = max(max_i, -stats[4])
        max_ii = max(max_ii, stats[5]-eta) if not accept_exact else max_ii
        max_feas = max(max_feas, stats[6])
        max_Q, max_weak = max(max_Q, -Q_margin), max(max_weak, excess)
        # This independently formed representation uses a matrix multiply only
        # as a diagnostic; it is outside the algorithm's M counter.
        normal_error = np.linalg.norm(new_n+A.T@next_witness)
        max_normal = max(max_normal, normal_error)
        if Q_margin < -tolerance*(1+abs(new_c)+np.linalg.norm(b)*np.linalg.norm(next_witness)) or excess > 1e-10:
            flag = 7
            audit[n_audit-1, 21] = 0.
            break
        starts += int(stats[8])
        p, x, normal, c, J, witness = new_p, new_x, new_n, new_c, new_J, next_witness
    rows[used-1, -1] = flag
    stats = np.array([used-1., used, 2.*used, W, C, flag, f0, f, dist, relative,
                      feasibility, first_N, first_W, joint_N, floor_uses, violations,
                      starts, max_i, max_ii, max_feas, max_Q, max_weak, max_normal, max_ratio, first_C])
    return rows[:used], audit[:n_audit], p, x, normal, c, J, witness, dual_history[:used], corrections[:n_audit], stats
