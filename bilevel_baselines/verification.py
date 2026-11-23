"""Independent checks: inputs, endpoints, counters, admissibility, containment, direct replays."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from rates_and_inexact.common import trace
from .protocol import SOURCE, H_SOURCE
from .runner import read, write_json, digest, load, constants, agm_gamma, source_hashes

REPLAY_STEPS = 30


def polyhedral(A, b, x, power):
    v = np.maximum(b-A@x, 0.)
    return float(np.sum(v**power)/power), -(A.T@(v**(power-1)))


def project_two_halfspaces(x0, a1, b1, a2, b2):
    """Projection of x0 onto {<a1,u> <= b1, <a2,u> <= b2} by active-set enumeration."""
    candidates = []
    if a1@x0 <= b1 and a2@x0 <= b2:
        return x0.copy()
    for a, c, other, d in ((a1, b1, a2, b2), (a2, b2, a1, b1)):
        if a@a > 0:
            u = x0-max(0., a@x0-c)/(a@a)*a
            if other@u <= d+1e-12*(1+abs(d)):
                candidates.append(u)
    G = np.array([[a1@a1, a1@a2], [a2@a1, a2@a2]])
    if abs(np.linalg.det(G)) > 1e-14*(G[0, 0]*G[1, 1]+1e-300):
        lam = np.linalg.solve(G, np.array([a1@x0-b1, a2@x0-b2]))
        if np.all(lam >= -1e-12):
            candidates.append(x0-lam[0]*a1-lam[1]*a2)
    if not candidates:
        raise RuntimeError("no feasible candidate")
    return min(candidates, key=lambda u: float(np.linalg.norm(u-x0)))


def project_ball_halfspace_bisection(w, rho, a, beta, iterations=200):
    """Projection onto {||u|| <= rho, <a,u> <= beta} as P_B(w - lam a) with the multiplier found by bisection."""
    def ball(v):
        n = np.linalg.norm(v)
        return v if n <= rho else v*(rho/n)
    u = ball(w)
    if a@u <= beta:
        return u
    lo, hi = 0., 1.
    while a@ball(w-hi*a) > beta:
        hi *= 2.
        if hi > 1e300:
            raise RuntimeError("empty intersection")
    for _ in range(iterations):
        mid = .5*(lo+hi)
        if a@ball(w-mid*a) > beta:
            lo = mid
        else:
            hi = mid
    return ball(w-hi*a)


def replay(p, spec, budget, c):
    """Plain numpy implementation of the first steps; returns (N, f, relative distance) per output point."""
    A, b, z, s = p["A"], p["b"], p["z"], p["power"]
    nz = np.linalg.norm(z)
    out = []
    x = np.zeros(A.shape[1])
    if spec["name"] == "MNG-local":
        normal = np.zeros_like(x)
        for k in range(budget):
            f, g = polyhedral(A, b, x, s)
            out.append((k+1, f, np.linalg.norm(x-z)/nz))
            q2 = g@g
            if q2 == 0.:
                break
            x = project_two_halfspaces(np.zeros_like(x), g, g@x-q2/c["L_ball"], -x, -(x@x))
    elif spec["name"] == "BiG-SAM":
        for k in range(budget):
            f, g = polyhedral(A, b, x, s)
            out.append((k+1, f, np.linalg.norm(x-z)/nz))
            t = 1./c["L_ball"] if spec["step"] == "ball" else 1./(p["L0"]+p["L1"]*np.linalg.norm(g))
            alpha = min(spec["gamma"]/(k+1.), 1.)
            x = (1.-alpha)*(x-t*g)
    else:
        gamma = agm_gamma(spec, c, s, budget_full(p))
        radius = spec["radius"]*c["rho"]
        zz, Ak = np.zeros_like(x), 0.
        for k in range(budget):
            ak = gamma*(k+1.)/4.
            y = (Ak*x+ak*zz)/(Ak+ak)
            fy, gy = polyhedral(A, b, y, s)
            w = zz-ak*y
            if gy@gy == 0.:
                zz = w if np.linalg.norm(w) <= radius else w*(radius/np.linalg.norm(w))
            else:
                zz = project_ball_halfspace_bisection(w, radius, gy, gy@y-fy)
            x = (Ak*x+ak*zz)/(Ak+ak)
            Ak += ak
            f, _ = polyhedral(A, b, x, s)
            out.append((k+1, f, np.linalg.norm(x-z)/nz))
    return np.array(out)


def budget_full(p):
    return p["budget"]


def check_run(rd, p, r, c):
    errors, warnings = [], []
    A, b, z, s = p["A"], p["b"], p["z"], p["power"]
    with np.load(rd/"state.npz") as state:
        x = state["x"]
    f, _ = polyhedral(A, b, x, s)
    distance = float(np.linalg.norm(x-z))
    feasibility = float(np.linalg.norm(np.maximum(b-A@x, 0.))/(1+np.linalg.norm(b)))
    if not np.isclose(f, r["f"], rtol=1e-9, atol=1e-28):
        errors.append("endpoint objective")
    if not np.isclose(distance, r["distance"], rtol=1e-9, atol=1e-15):
        errors.append("endpoint distance")
    if not np.isclose(feasibility, r["feasibility"], rtol=1e-8, atol=1e-20):
        errors.append("endpoint feasibility")
    if not np.isclose(.5*x@x-.5*z@z, r["Omega_gap"], rtol=1e-9, atol=1e-12):
        errors.append("endpoint Omega gap")
    if r["N"] != r["K"]+1:
        errors.append("evaluation counter")
    if r["flag"] not in (0, 1):
        errors.append("status "+r["status"])
    hit = r["first_distance_N"]
    if (hit is not None) != r["distance_success"] or (r["first_joint_N"] is not None) != r["joint_success"]:
        errors.append("success flags differ from the first hits")
    t = trace(rd/"trace.csv.gz")
    if hit is not None:
        before = t["N"] < hit
        if np.any(t["relative_distance"][before] <= 1e-2):
            errors.append("a stored point before the first hit is within 1%")
        at = t["N"] == hit
        if np.any(at) and not np.all(t["relative_distance"][at] <= 1e-2):
            errors.append("stored first-hit point is not within 1%")
    elif np.any(t["relative_distance"] <= 1e-2):
        errors.append("a stored point is within 1% but no hit was recorded")
    name = r["method"]["name"]
    if name == "MNG-local":
        if r["diagnostic"] > 1+1e-10 or r["diagnostic_count"] != 0:
            errors.append("inadmissible Lipschitz cut")
        if r["max_norm"] > 2*c["rho"]*(1+1e-12):
            warnings.append("MNG iterate outside the ball of L_ball")
    elif name == "BiG-SAM":
        if r["method"]["step"] == "ball" and r["diagnostic_count"] != 0:
            warnings.append(f"BiG-SAM iterate outside the ball of L_ball ({r['diagnostic_count']} steps)")
    elif name == "AGM-BiO":
        if r["diagnostic"] > 1e-10:
            errors.append("AGM-BiO projection violates Z or the cut")
        if r["max_norm"] > r["parameters"]["radius"]*(1+1e-12):
            errors.append("AGM-BiO output outside Z")
    rep = replay(p, r["method"], min(REPLAY_STEPS, r["N"]), c)
    dense = t["N"] <= len(rep)
    m = int(np.sum(dense))
    if m:
        idx = t["N"][dense].astype(int)-1
        if not np.allclose(rep[idx, 1], t["f"][dense], rtol=1e-8, atol=1e-25):
            errors.append("replayed objective differs")
        if not np.allclose(rep[idx, 2], t["relative_distance"][dense], rtol=1e-8, atol=1e-14):
            errors.append("replayed distance differs")
    return errors, warnings


def verify(output):
    output = Path(output)
    frozen = read(output/"config.json")
    errors, warnings, counts, statuses = [], [], Counter(), Counter()
    for name, sha in frozen["hashes"].items():
        if digest(output/"source"/name) != sha:
            errors.append("source snapshot mismatch: "+name)
    if digest(SOURCE/"config.json") != frozen["source_config"]:
        errors.append("archive configuration changed")
    if digest(H_SOURCE) != frozen["h_source"]:
        errors.append("H-method source table changed")
    for d in sorted((output/"instances").glob("*")):
        job = read(d/"job.json")
        src = SOURCE/"instances"/job["source_instance"]
        for name in ("instance.npz", "reference.json"):
            if digest(src/name) != digest(d/name):
                errors.append(f"{d.name}: input differs from the archive: {name}")
        p = load(d)
        p["budget"] = job["gradient_budget"]
        c = constants(p["A"], p["b"], p["power"])
        if c["rho"] < np.linalg.norm(p["z"])*(1-1e-12):
            errors.append(f"{d.name}: ||A^+ b|| below ||z_ref||")
        for path in sorted(d.glob("*/result.json")):
            r = read(path)
            local, warn = check_run(path.parent, p, r, c)
            errors.extend(f"{d.name}/{path.parent.name} {r['label']}: {e}" for e in local)
            warnings.extend(f"{d.name}/{path.parent.name} {r['label']}: {w}" for w in warn)
            counts[r["label"]] += 1
            statuses[r["status"]] += 1
    expected = sum(len(j["methods"]) for j in frozen["jobs"])
    result = dict(integrity_passed=not errors, errors=errors[:200], error_count=len(errors),
                  warnings=warnings[:200], warning_count=len(warnings),
                  runs=sum(counts.values()), expected_runs=expected, complete=sum(counts.values()) == expected,
                  statuses=dict(statuses), per_method=dict(counts), pilot=frozen["pilot"],
                  references="archived inputs byte-identical to the archive (certified there)")
    write_json(output/"verification.json", result)
    print(f"Verification: {result['runs']}/{expected} runs; {len(errors)} errors; {len(warnings)} warnings; "
          f"statuses {dict(statuses)}", flush=True)
    return result
