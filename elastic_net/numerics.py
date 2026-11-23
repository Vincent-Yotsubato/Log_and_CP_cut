"""Float64 algorithms, compiled as whole trajectories (no fast-math).

All counters refer to the numerical algorithm. Reference construction, logging
and diagnostic inner products are excluded. Matrix products inside the objective
are evaluated directly, including when the data happen to have orthogonal rows.
"""
import math
import numpy as np
from numba import njit

METHODS = {"Polyak-1": 0, "Polyak-2": 1, "CP": 2, "C-LBreI": 3,
           "Log-LBreI": 4, "H-Log": 5, "H-CP": 6, "Proj-C": 7, "Proj-Log": 8}
FLAGS = {0: "budget", 1: "objective_success", 2: "selection_success",
         3: "exact_zero", 4: "numerical_stagnation", 5: "overflow",
         6: "inadmissible_cut", 7: "h_certificate_failure",
         8: "inner_budget", 9: "nonfinite", 10: "projection_residual"}
COLUMNS = ["k", "n_grad", "n_mat", "n_omega", "n_ip", "f", "delta", "q",
           "sigma", "tau", "t", "dist_to_zdag", "Omega_gap", "flag",
           "relative_distance", "best_delta", "r", "A_over_delta",
           "rho_delta", "rho_CP", "rho_log", "rho_C", "A_over_Df", "B_over_delta",
           "affine_residual", "complementarity", "Omega", "h_Q_violation",
           "x1_minus_x2", "projection_residual", "inner_iterations",
           "zero_fraction", "explicit_depth_fraction", "projection_step_ratio", "h_kkt_residual"]


@njit(cache=True)
def cuts(q, L0, L1):
    q2 = q * q
    if L1 == 0:
        return q2 / (2 * L0), q2 / (2 * L0), q2 / L0, q2 / L0
    r = L1 * q / L0
    # Taylor series for small r; a slightly wider series region also avoids cancellation at 1e-4.
    if r < 1e-3:
        A = q2 / L0 * (0.5 + r * (-1/6 + r * (1/12 + r * (-1/20 + r * (1/30 - r/42)))))
        B = q2 / L0 * (0.5 + r * (-1/3 + r * (1/4 + r * (-1/5 + r * (1/6 - r/7)))))
    else:
        lr = math.log1p(r)
        A = L0 / (L1 * L1) * ((1 + r) * lr - r)
        B = L0 / (L1 * L1) * (r - lr)
    return A, B, q / L1 * math.log1p(r), q2 / (L0 + L1 * q)


@njit(cache=True)
def soft(p, lam):
    return np.sign(p) * np.maximum(np.abs(p) - lam, 0.)


@njit(cache=True)
def omega(x, p0, lam):
    return 0.5 * np.dot(x, x) + lam * np.sum(np.abs(x)) - np.dot(p0, x)


@njit(cache=True)
def evaluate(A, b, x, family, beta, power):
    residual = A @ x - b
    if family == 1:
        v = np.maximum(-residual, 0.)
        weights = v ** (power - 1)
        f = np.sum(v ** power) / power
        return f, -(A.T @ weights), power * f, residual, weights
    if beta == 0:
        f = np.dot(residual, residual) / (2 * b.size)
        return f, (A.T @ residual) / b.size, 2 * f, residual, residual / b.size
    # Overflow is a recorded failure; never silently clip the argument.
    if np.max(np.abs(beta * residual)) > 700:
        return np.inf, np.full(x.size, np.inf), np.inf, residual, residual
    s = np.sinh(beta * residual)
    f = 2 * np.sum(np.sinh(beta * residual / 2)**2) / (b.size * beta * beta)
    return f, (A.T @ s) / (b.size * beta), np.dot(s, residual) / (b.size * beta), residual, s / (b.size * beta)


@njit(cache=True)
def proj2(p0, x, g, sigma, normal):
    """Euclidean two-halfspace active set, normalized before solving.

    Orthogonalization avoids subtracting g.g*n.n-(g.n)^2 at small q.
    Returns the original (unnormalized) dual multipliers.
    """
    q = np.linalg.norm(g)
    nn = np.linalg.norm(normal)
    ips = 2
    h = g / q
    sg = np.dot(h, p0 - x) + sigma / q
    ips += 1
    if nn == 0:
        lam = max(sg, 0.) / q
        return p0 - lam * g, lam, 0., 0, ips
    j = normal / nn
    sn = np.dot(j, p0 - x)
    ips += 1
    if sg <= 0 and sn <= 0:
        return p0.copy(), 0., 0., 0, ips
    u = p0 - max(sg, 0.) * h
    ips += 2
    if np.dot(j, u - x) <= 2e-15 * (1 + np.linalg.norm(x)):
        return u, max(sg, 0.) / q, 0., 0, ips
    u = p0 - max(sn, 0.) * j
    ips += 1
    if np.dot(h, u - x) + sigma / q <= 0:
        return u, 0., max(sn, 0.) / nn, 0, ips
    cosine = np.dot(h, j)
    orth = j - cosine * h
    square = np.dot(orth, orth)
    ips += 2
    if square < 1e-28:
        # Parallel same-direction halfspaces: use the stronger one.
        if cosine > 0:
            if sg >= sn:
                return p0 - sg * h, sg / q, 0., 0, ips
            return p0 - sn * j, 0., sn / nn, 0, ips
        return x.copy(), 0., 0., 7, ips
    nu_scaled = (sn - cosine * sg) / square
    lam_scaled = sg - cosine * nu_scaled
    if min(nu_scaled, lam_scaled) < -1e-10 * (1 + abs(sg) + abs(sn)):
        return x.copy(), 0., 0., 7, ips
    lam, nu = max(lam_scaled, 0.) / q, max(nu_scaled, 0.) / nn
    return p0 - lam * g - nu * normal, lam, nu, 0, ips


@njit(cache=True)
def h_projected_gradient(p0, x, g, sigma, normal, lam_en, max_inner=200000):
    """Projected gradient on the two-multiplier dual with step exactly 1/L_Psi.

    Check the projected KKT residual, primal feasibility, and complementarity.
    The dual p is retained, including coordinates in soft-threshold dead zones.
    """
    # Solve the identical dual in variables (lambda*||g||, nu*||n||).
    # Unit normals prevent a vanishing gradient norm from freezing lambda.
    # The prescribed initial multipliers are unchanged in original units.
    q, nn = np.linalg.norm(g), np.linalg.norm(normal)
    ns = nn if nn > 0 else 1.
    # An exact single-active-constraint solution is a certified shortcut to the
    # same two-halfspace minimizer. This also handles coincident normals without
    # forcing projected gradient to resolve nonunique multipliers.
    x0 = soft(p0, lam_en)
    depth0 = np.dot(g, x0 - x) + sigma
    extra_w, extra_ip = 1, 3
    if depth0 > 0:
        t0, calls, root_residual, root_flag = project_one(p0, x0, g, depth0, lam_en)
        extra_w += calls + 1
        extra_ip += calls + 2
        candidate = soft(p0 - t0 * g, lam_en)
        if root_flag == 0 and np.dot(normal, candidate - x) <= 1e-13:
            return p0 - t0 * g, t0, 0., extra_w, extra_ip, 0
    h, j = g / q, normal / ns
    L = np.dot(h, h) + np.dot(j, j)
    bound = np.dot(h, x) - sigma / q
    c = np.dot(j, x)
    a, v = sigma / q, ns
    extra_ip += 4
    for iteration in range(max_inner):
        p = p0 - a * h - v * j
        u = soft(p, lam_en)
        da, dv = bound - np.dot(h, u), c - np.dot(j, u)
        residual = max(abs(min(a / q, da * q)), abs(min(v / ns, dv * ns)))
        comp = max(abs(a * da), abs(v * dv))
        if residual <= 1e-12 and comp <= 1e-12:
            return p, a / q, v / ns, iteration + 1 + extra_w, 2 * (iteration + 1) + extra_ip, 0
        a = max(0., a - da / L)
        v = max(0., v - dv / L)
    return p, a / q, v / ns, max_inner + extra_w, 2 * max_inner + extra_ip, 8


@njit(cache=True)
def boundary(t, p, x, g, sigma, lam):
    # On an unchanged active branch, x_i-T(p_i-t*g_i)=t*g_i exactly.
    # This avoids catastrophic cancellation near the stopping tolerance.
    total = 0.
    for i in range(len(p)):
        new_p = p[i] - t * g[i]
        if (p[i] > lam and new_p > lam) or (p[i] < -lam and new_p < -lam):
            difference = t * g[i]
        else:
            new_x = max(new_p - lam, 0.) if new_p >= 0 else min(new_p + lam, 0.)
            difference = x[i] - new_x
        total += g[i] * difference
    return total - sigma


@njit(cache=True)
def h_semismooth_newton(p0, x, g, sigma, normal, lam_en, max_inner=1000):
    """Solve the same convex two-multiplier dual with safeguarded Newton.

    Soft thresholding is piecewise affine. The active-coordinate Gram matrix
    is a generalized Hessian. Orthogonalization, bound-safe backtracking and
    projected-gradient fallback handle rank changes. No reference solution is
    used by the projection. Stop on normalized KKT and complementarity.
    """
    q,nn=np.linalg.norm(g),np.linalg.norm(normal)
    h=g/q;ns=nn if nn>0 else 1.;j=normal/ns
    b=np.dot(h,x)-sigma/q;c=np.dot(j,x)
    x0=soft(p0,lam_en);calls=1;ips=4
    # Exact one-active-constraint certificate.
    depth=np.dot(h,x0)-b;ips+=1
    if depth>0:
        t,w,res,flag=project_one(p0,x0,h,depth,lam_en)
        candidate=soft(p0-t*h,lam_en);calls+=w+1;ips+=w+1
        # project_one targets a relative cut residual. H's outer certificate
        # also checks the original (possibly large) normal, so refine this
        # shortcut to near-roundoff absolute accuracy on its active branch.
        for refine in range(6):
            error=boundary(t,p0,x0,h,depth,lam_en);calls+=1;ips+=1
            if abs(error)<=min(1e-13,1e-11/max(q,1.)):break
            active=np.abs(p0-t*h)>lam_en
            derivative=np.dot(h[active],h[active]);ips+=1
            if derivative<=1e-20:break
            trial=t-error/derivative
            if trial<0:break
            t=trial
        candidate=soft(p0-t*h,lam_en);calls+=1
        if flag==0 and abs(np.dot(h,candidate)-b)<=1e-12 and np.dot(j,candidate)-c<=1e-12:
            return p0-t*h,t/q,0.,calls,ips,0
    if nn==0:
        return p0.copy(),0.,0.,calls,ips,8
    a,v=0.,nn
    for iteration in range(max_inner):
        p=p0-a*h-v*j;u=soft(p,lam_en);calls+=1
        ga,gv=b-np.dot(h,u),c-np.dot(j,u);ips+=2
        obj=.5*np.dot(u,u)+a*b+v*c;ips+=1
        kkt=max(abs(min(a,ga)),abs(min(v,gv)))
        comp=max(abs(a*ga),abs(v*gv))/(1+abs(obj))
        if kkt<=1e-12 and comp<=1e-12:
            return p,a/q,v/ns,calls,ips,0
        active=np.abs(p)>lam_en
        hh=np.dot(h[active],h[active]);jj=np.dot(j[active],j[active]);hj=np.dot(h[active],j[active]);ips+=3
        if a<=1e-15 and ga>=0:
            da=0.;dv=-gv/max(jj,1e-12)
        elif v<=1e-15 and gv>=0:
            da=-ga/max(hh,1e-12);dv=0.
        elif hh>1e-14:
            ort=j[active]-(hj/hh)*h[active]
            den=np.dot(ort,ort);ips+=1
            if den>1e-22:
                dv=(-gv+(hj/hh)*ga)/den
                da=(-ga-hj*dv)/hh
            else:
                da=max(0.,a-ga/2)-a;dv=max(0.,v-gv/2)-v
        else:
            da=max(0.,a-ga/2)-a;dv=max(0.,v-gv/2)-v
        if a<=1e-15 and da<0:da=0.
        if v<=1e-15 and dv<0:dv=0.
        slope=ga*da+gv*dv
        if slope>=0 or not np.isfinite(slope):
            da=max(0.,a-ga/2)-a;dv=max(0.,v-gv/2)-v;slope=ga*da+gv*dv
        step=1.
        if da<0:step=min(step,-a/da)
        if dv<0:step=min(step,-v/dv)
        accepted=False
        for ls in range(50):
            na=max(0.,a+step*da);nv=max(0.,v+step*dv)
            nu=soft(p0-na*h-nv*j,lam_en);calls+=1
            no=.5*np.dot(nu,nu)+na*b+nv*c;ips+=1
            nga,ngv=b-np.dot(h,nu),c-np.dot(j,nu);ips+=2
            nkkt=max(abs(min(na,nga)),abs(min(nv,ngv)))
            # Objective roundoff may hide a useful final Newton correction.
            guard=4e-15*(1+abs(obj))
            if (no<=obj+1e-4*step*slope and nkkt<=max(1.,10*kkt)) or (no<=obj+guard and nkkt<kkt):
                a,v=na,nv;accepted=True;break
            step*=.5
        if not accepted:
            a=max(0.,a-ga/2);v=max(0.,v-gv/2)
    return p,a/q,v/ns,calls,ips,8


@njit(cache=True)
def project_one(p, x, g, sigma, lam, q2=0.):
    """Bracketed Brent-Dekker solver; phi evaluations count W and IP.

    Uses t/t_lower as the root variable so relative accuracy is stable for
    very small and very large multipliers. Tests compare against SciPy brentq.
    """
    lower = sigma / (q2 if q2 > 0 else np.dot(g, g))
    if lam == 0:
        return lower, 0, 0., 0  # exact Euclidean identity, no root evaluations
    fa = boundary(lower, p, x, g, sigma, lam)
    calls = 1
    if abs(fa) <= 1e-10 * sigma:
        return lower, calls, abs(fa) / sigma, 0
    if fa > 1e-10 * sigma:
        return lower, calls, fa / sigma, 10
    a, b = 1., 2.
    fb = boundary(b * lower, p, x, g, sigma, lam)
    calls += 1
    while fb < 0 and b < 2.**60:
        b *= 2
        fb = boundary(b * lower, p, x, g, sigma, lam)
        calls += 1
    if fb < 0 or not np.isfinite(fb):
        return lower, calls, np.inf, 10
    if abs(fa) < abs(fb):
        a, b, fa, fb = b, a, fb, fa
    c, fc, d = a, fa, a
    bisected = True
    tolerance = 1e-14 * max(abs(a), abs(b))
    for _ in range(150):
        if abs(fb) <= 1e-11 * sigma or abs(b - a) <= tolerance:
            return b * lower, calls, abs(fb) / sigma, 0 if abs(fb) <= 1e-8 * sigma else 10
        if fa != fc and fb != fc:
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            s = b - fb * (b - a) / (fb - fa)
        lo, hi = min((3 * a + b) / 4, b), max((3 * a + b) / 4, b)
        reject = (not lo < s < hi or
                  (bisected and abs(s - b) >= abs(b - c) / 2) or
                  (not bisected and abs(s - b) >= abs(c - d) / 2) or
                  (bisected and abs(b - c) < tolerance) or
                  (not bisected and abs(c - d) < tolerance))
        if reject:
            s, bisected = (a + b) / 2, True
        else:
            bisected = False
        fs = boundary(s * lower, p, x, g, sigma, lam)
        calls += 1
        d, c, fc = c, b, fb
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b, fa, fb = b, a, fb, fa
    return b * lower, calls, abs(fb) / sigma, 10


@njit(cache=True)
def run(A, b, family, beta, power, L0, L1, lam, p0, z, method,
        theta, constant_factor, max_updates, stop_mode, log_steps,
        objective_targets, distance_targets, diagnostics=False):
    """stop_mode: 0 full selection budget, 1 objective, 2 joint objective and distance tolerance."""
    p = p0.copy()
    x = soft(p, lam)
    normal = np.zeros(x.size)
    witness = np.zeros(b.size)  # explicit dual, or -normal = A.T @ witness
    rows = np.empty((len(log_steps) + 12, 35))
    hits = np.full(8, -1, dtype=np.int64)
    objective_hits = np.full(len(objective_targets), -1, dtype=np.int64)
    distance_hits = np.full(len(distance_targets), -1, dtype=np.int64)
    w, ips, stored, log_index = 1, 0, 0, 0
    omega_z = omega(z, p0, lam)
    omega_previous = omega(x, p0, lam)
    norm_z, norm_b = np.linalg.norm(z), np.linalg.norm(b)
    best, initial_f = np.inf, 0.
    max_ratio, max_q_violation, max_omega_drop, max_omega_excess = 0., 0., 0., 0.
    max_root_residual, total_inner, max_inner = 0., 0, 0
    flag = 0
    distance = np.linalg.norm(x - z)
    relative = distance / norm_z if norm_z > 0 else distance
    feasibility, complementarity = np.inf, 0.
    finite_p, finite_x, finite_witness = p, x, witness
    finite_f, finite_q, finite_tau, finite_k = np.nan, np.nan, np.nan, 0
    evaluations, matrix_calls = 0, 0
    for k in range(max_updates + 1):
        f, g, tau, residual, weights = evaluate(A, b, x, family, beta, power)
        evaluations += 1
        matrix_calls += 2
        ips += 1
        q2 = np.dot(g, g)
        q = math.sqrt(q2)
        if not np.isfinite(f + q + tau):
            flag = 5
            if family == 0 and beta > 0 and np.max(np.abs(beta * residual)) > 700:
                matrix_calls -= 1  # failed before the transpose application
            p, x, witness = finite_p, finite_x, finite_witness
            f, q, tau, k = finite_f, finite_q, finite_tau, finite_k
            break
        if k == 0:
            initial_f = f
        best = min(best, f)
        for j in range(8):
            if hits[j] < 0 and best <= initial_f * 10.**(-j - 1):
                hits[j] = k + 1  # N = number of gradient evaluations, k=0 counts
        ca, cb, sl, sc = cuts(q, L0, L1)
        if method <= 1:
            sigma, multiplier = f, float(method + 1)
        elif method == 2 or method == 6:
            sigma, multiplier = f + ca / constant_factor, 1.
        elif method == 3 or method == 7:
            sigma, multiplier = sc, theta
        else:
            sigma, multiplier = sl, theta
        distance = np.linalg.norm(x - z)
        relative = distance / norm_z if norm_z > 0 else distance
        for index in range(len(objective_targets)):
            if objective_hits[index] < 0 and best <= objective_targets[index]:
                objective_hits[index] = k + 1
        for index in range(len(distance_targets)):
            if distance_hits[index] < 0 and relative <= distance_targets[index]:
                distance_hits[index] = k + 1
        om = omega(x, p0, lam)
        feasibility = np.linalg.norm(residual if family == 0 else np.minimum(residual, 0.)) / (1 + norm_b)
        complementarity = np.max(np.abs(witness * residual)) if family == 1 else 0.
        q_violation = max(0., -np.dot(witness, b) - np.dot(normal, x)) if method in (5, 6) else 0.
        max_q_violation = max(max_q_violation, q_violation)
        if q_violation > 1e-10 * (1 + abs(om)):
            flag = 7
        if tau > 0:
            ratio = max(f / tau, (f + ca) / tau, sl / tau, sc / tau)
            max_ratio = max(max_ratio, ratio)
            if ratio > 1 + 1e-10:
                flag = 6
        if flag == 0 and stop_mode == 1 and f <= initial_f * 1e-9:
            flag = 1
        if flag == 0 and stop_mode == 3 and len(objective_targets) > 0 and best <= np.min(objective_targets):
            flag = 1
        if flag == 0 and stop_mode == 2 and k > 0 and feasibility <= 1e-8 and distance / (1 + norm_z) <= 1e-6:
            flag = 2
        if flag == 0 and (f == 0 or q == 0):
            flag = 3
        finite_p, finite_x, finite_witness = p, x, witness
        finite_f, finite_q, finite_tau, finite_k = f, q, tau, k
        if k == max_updates:
            # Success at the final evaluation is retained.
            pass
        terminal = flag != 0 or k == max_updates
        save = terminal or (log_index < len(log_steps) and k == log_steps[log_index])
        if save:
            if log_index < len(log_steps) and k == log_steps[log_index]:
                log_index += 1
            df = tau - f
            inv_tau, inv_f = (1 / tau if tau > 0 else 0.), (1 / f if f > 0 else 0.)
            row = np.array([float(k), float(k + 1), float(2 * (k + 1)), float(w), float(ips),
                f, f, q, sigma, tau, 0., distance, omega_z - om, float(flag), relative, best,
                L1 * q / L0, ca * inv_f, f * inv_tau, (f + ca) * inv_tau, sl * inv_tau,
                sc * inv_tau, ca / df if df > 0 else 0., cb * inv_f, feasibility,
                complementarity, om, q_violation, x[0] - x[1], 0., 0.,
                np.sum(x == 0.) / x.size, 0., 0., 0.])
            rows[stored] = row
            stored += 1
        if terminal:
            break
        step, root_residual, inner = multiplier * sigma / q2, 0., 0
        old_x = x
        if method in (5, 6):
            if lam == 0:
                proposed, aa, vv, failure, projection_ips = proj2(p0, x, g, sigma, normal)
                ips += projection_ips
                inner = 1
            else:
                proposed, aa, vv, inner, inner_ip, failure = h_semismooth_newton(p0, x, g, sigma, normal, lam)
                w += inner
                ips += inner_ip
            if failure:
                flag = failure
                total_inner += inner
                max_inner = max(max_inner, inner)
                break
            new_normal = aa * g + vv * normal
            proposed = p0 - new_normal  # preserve the compatible subgradient
            new_x = soft(proposed, lam)
            w += 1
            if family == 1:
                new_witness = aa * weights + vv * witness
            else:
                new_witness = witness
            next_omega = omega(new_x, p0, lam)
            drop = max(0., om - next_omega)
            excess = max(0., next_omega - omega_z)
            max_omega_drop, max_omega_excess = max(max_omega_drop, drop), max(max_omega_excess, excess)
            current_cut_violation = np.dot(g, new_x - x) + sigma
            old_Q_violation = np.dot(normal, new_x - x)
            nn = np.linalg.norm(normal)
            da = -current_cut_violation / q
            dv = -old_Q_violation / nn if nn > 0 else 0.
            h_kkt = max(abs(min(aa*q, da)), abs(min(vv*nn, dv)),
                        abs(aa*q*da)/(1+abs(next_omega)), abs(vv*nn*dv)/(1+abs(next_omega)))
            max_root_residual = max(max_root_residual, h_kkt)
            if save:
                rows[stored-1, 34] = h_kkt
            if (drop > 1e-10 * (1 + abs(om)) or excess > 1e-10 * (1 + abs(omega_z)) or
                current_cut_violation > 1e-10 * (1 + q * np.linalg.norm(x)) or
                old_Q_violation > 1e-10 * (1 + abs(om))):
                flag = 7
                break
            p, x, normal, witness = proposed, new_x, new_normal, new_witness
            step = aa
        else:
            if method in (7, 8):
                if diagnostics and save:
                    rows[stored-1, 32] = 1. + boundary(sigma/q2, p, x, g, sigma, lam)/sigma
                step, calls, root_residual, failure = project_one(p, x, g, sigma, lam, q2)
                w += calls
                ips += calls
                max_root_residual = max(max_root_residual, root_residual)
                if failure:
                    flag = failure
                    break
                if diagnostics and save:
                    rows[stored-1, 33] = step / (sigma/q2)
            p = p - step * g
            x = soft(p, lam)
            w += 1
            if family == 1:
                witness = witness + step * weights
        total_inner += inner
        max_inner = max(max_inner, inner)
        if save:
            rows[stored - 1, 10] = step
            rows[stored - 1, 29] = root_residual
            rows[stored - 1, 30] = inner
        if not np.all(np.isfinite(p)):
            flag = 9
            break
    # A failed update never counts as an accepted update. Keep the finite state.
    if stored > 0 and rows[stored-1,0] == k:
        rows[stored - 1, 13] = flag
    stats = np.array([float(k), float(evaluations), float(matrix_calls), float(w), float(ips),
                      float(flag), initial_f, f, distance, relative, feasibility,
                      max_ratio, max_q_violation, max_omega_drop, max_omega_excess,
                      max_root_residual, float(total_inner), float(max_inner), complementarity])
    # An inner failure can happen between logarithmic samples. Preserve the
    # last accepted/evaluated state instead of silently dropping that endpoint.
    if stored == 0 or rows[stored-1,0] != k:
        rows[stored] = np.array([float(k),float(evaluations),float(matrix_calls),float(w),float(ips),
            f,f,q,sigma,tau,0.,distance,omega_z-om,float(flag),relative,best,L1*q/L0,
            ca/f if f>0 else 0.,f/tau if tau>0 else 0.,(f+ca)/tau if tau>0 else 0.,
            sl/tau if tau>0 else 0.,sc/tau if tau>0 else 0.,ca/(tau-f) if tau>f else 0.,
            cb/f if f>0 else 0.,feasibility,complementarity,om,q_violation,x[0]-x[1],
            0.,float(inner),np.sum(x==0.)/x.size,0.,0.,0.])
        stored += 1
    return rows[:stored], p, x, witness, hits, stats, objective_hits, distance_hits
