"""Independent equations, reference-gap checks, first hits, and failed-run audit."""
from __future__ import annotations

import itertools
import math
from collections import Counter
from pathlib import Path

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits
from elastic_net.problems import value_gradient, soft
from .common import (read, write_json, trace, load_problem, digest, reference_interval,
                     instance, SOURCES, csv_write)
from .protocol import PANELS, build, configuration


def exact_distance_hits(p, name, budget):
    from elastic_net.numerics import run, METHODS
    ans = run(p.A, p.b, p.family, p.beta, p.power, p.L0, p.L1, p.lam, p.p0, p.z,
              METHODS[name], 1., 1., budget-1, 0, np.arange(budget, dtype=np.int64),
              np.empty(0), np.array([.01]), False)
    t = ans[0]
    distance = np.flatnonzero(t[:, 14] <= .01)
    joint = np.flatnonzero((t[:, 24] <= 1e-8) & (t[:, 11]/(1+np.linalg.norm(p.z)) <= 1e-6))
    return dict(first_distance_N=int(t[distance[0], 1]) if len(distance) else None,
                first_distance_W=int(t[distance[0], 3]) if len(distance) else None,
                first_joint_N=int(t[joint[0], 1]) if len(joint) else None)


def independent_projection(x, matrix, rhs):
    """Enumerate QP active sets with numpy.linalg.lstsq, independent of proj2."""
    best, best_value = None, math.inf
    norm = np.linalg.norm(matrix, axis=1)
    nonzero = norm > 0.
    if np.any((~nonzero) & (rhs < -1e-12)):
        raise ValueError("Empty zero-normal halfspace")
    H, h = matrix[nonzero]/norm[nonzero, None], rhs[nonzero]/norm[nonzero]
    for size in range(len(H)+1):
        for indices in itertools.combinations(range(len(H)), size):
            if not size:
                u = x.copy()
            else:
                B = H[list(indices)]
                multipliers = np.linalg.lstsq(B@B.T, B@x-h[list(indices)], rcond=1e-14)[0]
                if np.min(multipliers) < -1e-10:
                    continue
                u = x-B.T@multipliers
            if np.any(H@u-h > 1e-10*(1+np.linalg.norm(u))):
                continue
            value = np.linalg.norm(u-x)**2
            if value < best_value:
                best, best_value = u, value
    if best is None:
        raise ValueError("No feasible active-set correction")
    return best


def verify_reference(directory):
    p = load_problem(directory)
    cert = p.certificate
    with mp.workdps(65):
        A = mp.matrix(p.A.tolist())
        b = mp.matrix(p.b.tolist())
        p0 = mp.matrix(p.p0.tolist())
        y = mp.matrix(p.dual.tolist())
        u = mp.matrix([mp.mpf(v) for v in cert["auxiliary_feasible_point"]])
        lam = mp.mpf(float(p.lam))
        dual = p0+A.T*y
        mapped = [mp.sign(v)*max(abs(v)-lam, 0) for v in dual]
        residual = A*u-b
        inequality = p.family == 1
        defect = max(max(-v, 0) if inequality else abs(v) for v in residual)
        gap = ((u.T*u)[0]/2 + lam*sum(abs(v) for v in u) - (p0.T*u)[0]
               + sum(v*v for v in mapped)/2 - (y.T*b)[0])
        errors = []
        if defect > mp.mpf("1e-32"):
            errors.append("auxiliary feasible point is not feasible at 65 digits")
        if not -mp.mpf("1e-32") <= gap < mp.mpf("1e-10"):
            errors.append("reference primal-dual gap outside tolerance")
        if abs(float(gap)-cert["raw_gap"]) > 1e-30+1e-10*abs(float(gap)):
            errors.append("reference gap differs from its certificate")
        if inequality and np.min(p.dual) < 0.:
            errors.append("negative reference multiplier")
        return dict(errors=errors, gap=float(gap), feasibility=float(defect))


def audit_steps(n_audit, extra=()):
    """Outer steps whose states are saved: a dense prefix, log-spaced steps, targets, end."""
    from .protocol import AUDIT_DENSE, AUDIT_LOG
    if n_audit <= 0:
        return np.zeros(0, dtype=np.int64)
    steps = np.r_[np.arange(min(AUDIT_DENSE, n_audit)),
                  np.geomspace(1, max(1, n_audit-1), AUDIT_LOG).astype(np.int64),
                  [n_audit-1], np.asarray(extra, dtype=np.int64)]
    return np.unique(steps[(steps >= 0) & (steps < n_audit)]).astype(np.int64)


def verify_inexact(rd, p, result):
    """Re-check the saved sample of actual states (the full audit ran inline)."""
    audit = trace(rd/"audit.csv.gz")
    with np.load(rd/"state.npz") as saved:
        state = {k: saved[k] for k in saved.files}
    with np.load(rd/"audit_states.npz") as saved:
        states = {k: saved[k] for k in saved.files}
    errors = []
    if "steps" not in states:
        return ["audit_states.npz lacks the sampled-step index"]
    if len(states["steps"]) != len(audit["k"]) or not np.array_equal(states["steps"], audit["k"].astype(np.int64)):
        return ["saved audit rows and saved states describe different steps"]
    if np.max(np.abs(states["p_final"]-state["p"])) > 1e-12:
        errors.append("saved state history has a different terminal dual")
    if np.linalg.norm(state["normal"]+p.A.T@state["witness"]) > 1e-8*(1+np.linalg.norm(state["normal"])):
        errors.append("terminal normal representation")
    if abs(float(state["J"])+.5*state["x"]@state["x"]+float(state["c"])) > 1e-10*(1+abs(float(state["J"]))):
        errors.append("terminal J identity")
    return errors+audit_inexact(p, result, audit, states["steps"], states["p"], states["p_next"],
                                states["u"], reference_interval(str(rd.parent)))


def audit_inexact(p, result, audit, steps, current, following, feasible, interval, every=None):
    """Check each actual state independently, avoiding accumulated drift.

    Free-running replays of tiny gradients with very large multipliers amplify a
    single rounding difference, so every check is a one-step check of the actual
    computed states (p_k, p_{k+1}, u_k). With every=None the rows are a saved sample;
    with the full dual history (inline use) the witness recursion and the first
    hits, including their kernel-call counts, are recomputed as well.
    """
    errors = []
    full = every is not None
    cfg = configuration(result["method"]["configuration"])
    if full:
        if len(every) != result["N"] or len(steps) != len(audit["k"]) or len(feasible) != len(steps):
            return ["missing per-step dual states or feasible corrections"]
        if not np.array_equal(every[0], p.p0):
            errors.append("incorrect initial dual state")
    witness = np.zeros(p.b.size)
    first_N, first_W, joint_N, W = None, None, None, 1
    from cut_functions import a_star, s_log

    def hits(x):
        distance = np.linalg.norm(x-p.z)
        joint = (np.linalg.norm(np.minimum(p.A@x-p.b, 0.))/(1+np.linalg.norm(p.b)) <= 1e-8
                 and distance/(1+np.linalg.norm(p.z)) <= 1e-6)
        return distance/np.linalg.norm(p.z) <= .01, joint

    if full:
        for k, actual_p in enumerate(every):
            near, joint = hits(soft(actual_p, p.lam))
            if first_N is None and near:
                first_N, first_W = k+1, W
            if joint_N is None and joint:
                joint_N = k+1
            if k < len(audit["k"]):
                W += int(audit["kernel_calls"][k])
        if first_N != result["first_distance_N"] or first_W != result["first_distance_W"]:
            errors.append("first-distance hit or W differs from the independent audit")
        if joint_N != result["first_joint_N"]:
            errors.append("first-joint hit differs from the independent audit")
    else:
        # A sample cannot give the first hit; it can contradict it. The steps just
        # before each recorded hit are always saved.
        for name, index in (("first-distance", 0), ("first-joint", 1)):
            recorded = result["first_distance_N" if index == 0 else "first_joint_N"]
            for k, actual_p in zip(steps, current):
                hit = hits(soft(actual_p, p.lam))[index]
                if recorded is not None and k+1 == recorded and not hit:
                    errors.append(f"{name} target not met at the recorded step")
                if hit and (recorded is None or k+1 < recorded):
                    errors.append(f"{name} target met before the recorded step {recorded}")
                    break
    for i, k in enumerate(steps):
        actual_p = current[i]
        if audit["accepted"][i] != 1.:
            break
        x = soft(actual_p, p.lam)
        f, g, _ = value_gradient(p.A, p.b, x, 1, 0., p.power)
        q = np.linalg.norm(g)
        normal, c, J = p.p0-actual_p, audit["c_previous"][i], audit["J_previous"][i]
        a, v = audit["lambda"][i], audit["nu"][i]
        if min(a, v) < 0.:
            errors.append(f"negative multiplier at {k}")
        sigma = float(f+a_star(q, p.L0, p.L1)) if result["method"]["name"] == "H-CP" else float(s_log(q, p.L0, p.L1))
        if q == 0.:
            sigma = 0.
        if not np.isclose(audit["sigma"][i], sigma, rtol=1e-8, atol=1e-28):
            errors.append(f"cut mismatch at {k}")
        depth = audit["b"][i]
        expected_b = float(g@x-audit["sigma"][i]) if q > 0. else 0.
        new_n = a*g+v*normal
        expected_c = a*depth+v*c
        new_c = audit["c_next"][i]
        new_p = following[i]
        new_x = soft(new_p, p.lam)
        new_J = -.5*new_x@new_x-new_c
        u = feasible[i]
        active = np.abs(new_p) > p.lam
        Bregman = .5*np.linalg.norm(u-new_x)**2
        Bregman += np.sum(p.lam*(np.abs(u[active])-np.sign(new_p[active])*u[active]))
        Bregman += np.sum(p.lam*np.abs(u[~active])-new_p[~active]*u[~active])
        gap = Bregman+a*(depth-g@u)+v*(c-normal@u)
        progress = .5*(audit["sigma"][i]/q)**2 if q > 0. else 0.
        allowance = 2e-11*(1+abs(J))
        if abs(depth-expected_b) > 1e-10*(1+abs(depth)):
            errors.append(f"halfspace bias mismatch at {k}")
        if np.max(np.abs(new_p-(p.p0-new_n))) > allowance:
            errors.append(f"dual combination mismatch at {k}")
        if abs(new_c-expected_c) > allowance:
            errors.append(f"memory bias combination mismatch at {k}")
        if abs(J+.5*x@x+c) > allowance:
            errors.append(f"previous J identity at {k}")
        if new_J-J-progress < -allowance:
            errors.append(f"progress condition fails at {k}")
        if cfg["acceptance"] == "A" and gap > audit["eta_used"][i]+allowance:
            errors.append(f"acceptance gap fails at {k}")
        if np.max(np.vstack((g, normal))@u-np.array([depth, c])) > allowance:
            errors.append(f"infeasible correction at {k}")
        if full:
            # Q_{k+1} contains X: n_{k+1} = -A^T w_{k+1}, w >= 0, c_{k+1} + w^T b >= 0.
            witness = a*np.maximum(p.b-p.A@x, 0.)**(p.power-1)+v*witness
            if new_c+witness@p.b < -allowance:
                errors.append(f"Q inclusion certificate fails at {k}")
            if np.linalg.norm(new_n+p.A.T@witness) > 1e-8*(1+np.linalg.norm(new_n)):
                errors.append(f"normal representation fails at {k}")
        if new_J > interval["omega_upper"]+allowance:
            errors.append(f"weak duality fails at {k}")
        if abs(new_J-audit["J_next"][i]) > allowance:
            errors.append(f"stored J identity at {k}")
        if len(errors) >= 10:
            break
    return errors


def verify(output):
    output = Path(output)
    frozen = read(output/"config.json")
    errors, details, failures, counts = [], [], [], Counter()
    for name, sha in frozen["hashes"].items():
        if digest(output/"source"/name) != sha:
            errors.append("source snapshot mismatch: "+name)
    for name, sha in frozen["source_configs"].items():
        if digest(SOURCES[name]/"config.json") != sha:
            errors.append("input configuration changed: "+name)
    with threadpool_limits(limits=1):
        for panel in PANELS:
            for d in sorted((output/panel/"instances").glob("*")):
                if not (d/"instance.npz").exists():
                    continue
                job = read(d/"job.json")
                p = load_problem(d)
                reference = verify_reference(d)
                errors.extend(str(d)+": "+e for e in reference["errors"])
                if job["source"] is not None:
                    src = instance(job["source"], job["source_instance"])
                    for name in ("instance.npz", "reference.json"):
                        if digest(src/name) != digest(d/name):
                            errors.append(str(d)+": reused input differs: "+name)
                for path in sorted(d.glob("*/result.json")):
                    r = read(path)
                    rd = path.parent
                    state = np.load(rd/"state.npz")
                    local = []
                    f, _, _ = value_gradient(p.A, p.b, state["x"], p.family, p.beta, p.power)
                    if not np.isclose(f, r["delta"], rtol=2e-8, atol=1e-25):
                        local.append("endpoint objective")
                    if np.max(np.abs(soft(state["p"], p.lam)-state["x"])) > 1e-12:
                        local.append("incompatible primal/dual")
                    if r["N"] != r["K"]+1 or r["M"] != 2*r["N"]:
                        local.append("evaluation counters")
                    if panel == "inexact" and not r.get("exact_reference", False):
                        if "inline_audit_errors" not in r:
                            local.append("missing inline per-step audit")
                        local += ["inline: "+e for e in r.get("inline_audit_errors", [])]
                        local += verify_inexact(rd, p, r)
                    if panel == "inexact" and r.get("matched_original") is False:
                        local.append("SSN-E mismatch with original trajectory")
                    if r["flag"] >= 5:
                        failures.append(dict(panel=panel, instance_id=job["id"], method=r["method"], flag=r["flag"], status=r["status"]))
                    errors.extend(str(path)+": "+e for e in local)
                    details.append(dict(panel=panel, instance_id=job["id"], method=r["method"], errors=local))
                    counts[panel] += 1
    expected = {panel: sum(len(j["methods"]) for j in frozen["jobs"][panel]) for panel in PANELS}
    coverage = {panel: dict(actual=counts[panel], expected=expected[panel], complete=counts[panel] == expected[panel]) for panel in expected}
    suite_complete = all(c["complete"] for c in coverage.values())
    result = dict(integrity_passed=not errors, numerical_failures=failures, errors=errors,
                  verified_runs=sum(counts.values()), counts=dict(counts), coverage=coverage,
                  default_suite_complete=suite_complete, pilot=frozen["pilot"],
                  passed=not errors and not failures and suite_complete,
                  notes="A faithfully recorded cap failure can pass integrity checks while failing the algorithm experiment.")
    write_json(output/"verification_details.json", details)
    write_json(output/"verification.json", result)
    print(f"Verification: {sum(counts.values())} runs; {len(errors)} integrity errors; {len(failures)} numerical failures", flush=True)
    return result
