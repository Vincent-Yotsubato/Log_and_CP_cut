"""Independent one-step audit for the residual test, and integrity of the stored results."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from elastic_net.problems import value_gradient, soft
from rates_and_inexact.common import read, write_json, digest, trace, load_problem, instance, reference_interval
from rates_and_inexact.verification import verify_reference
from .protocol import SOURCES, TIMED, configuration


def audit_residual(p, result, audit, steps, current, following, interval, every=None):
    """rates_and_inexact.verification.audit_inexact with the residual test in place of (ii).

    Every check is a one-step check of the actual computed states (p_k, p_{k+1}). The residual
    is recomputed twice, through Omega(x+) - J+ and through the products
    lambda*(b - <g, x+>) + nu*(c - <n, x+>); with the full dual history (inline use) the witness
    recursion and the first hits with their kernel-call counts are recomputed as well.
    """
    errors = []
    full = every is not None
    cfg = configuration(result["method"]["configuration"])
    if full:
        if len(every) != result["N"] or len(steps) != len(audit["k"]):
            return ["missing per-step dual states"]
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
        through_omega = .5*new_x@new_x+p.lam*np.sum(np.abs(new_x))-p.p0@new_x-new_J
        through_products = a*(depth-g@new_x)+v*(c-normal@new_x)
        if abs(through_omega-through_products) > allowance:
            errors.append(f"residual identity fails at {k}")
        bound = through_products if cfg["acceptance"] == "R+" else abs(through_products)
        if bound > audit["eta_used"][i]+allowance:
            errors.append(f"residual acceptance fails at {k}")
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


def verify_saved(rd, p, result):
    """Re-check the saved sample of actual states (the full audit ran inline)."""
    audit = trace(rd/"audit.csv.gz")
    with np.load(rd/"state.npz") as saved:
        state = {k: saved[k] for k in saved.files}
    with np.load(rd/"audit_states.npz") as saved:
        states = {k: saved[k] for k in saved.files}
    if len(states["steps"]) != len(audit["k"]) or not np.array_equal(states["steps"], audit["k"].astype(np.int64)):
        return ["saved audit rows and saved states describe different steps"]
    errors = []
    if np.max(np.abs(states["p_final"]-state["p"])) > 1e-12:
        errors.append("saved state history has a different terminal dual")
    if np.linalg.norm(state["normal"]+p.A.T@state["witness"]) > 1e-8*(1+np.linalg.norm(state["normal"])):
        errors.append("terminal normal representation")
    if abs(float(state["J"])+.5*state["x"]@state["x"]+float(state["c"])) > 1e-10*(1+abs(float(state["J"]))):
        errors.append("terminal J identity")
    return errors+audit_residual(p, result, audit, states["steps"], states["p"], states["p_next"],
                                 reference_interval(str(rd.parent)))


def verify(output):
    output = Path(output)
    frozen = read(output/"config.json")
    errors, failures, expected_caps, details, found = [], [], [], [], set()
    for name, sha in frozen["hashes"].items():
        if digest(output/"source"/name) != sha:
            errors.append("source snapshot mismatch: "+name)
    for key, sha in frozen["source_configs"].items():
        if digest(SOURCES[key]/"config.json") != sha:
            errors.append("comparator configuration changed: "+key)
    for name, sha in frozen["comparator_results"].items():
        key, rest = name.split("/", 1)
        if digest(SOURCES[key]/rest) != sha:
            errors.append("comparator result changed: "+name)
    with threadpool_limits(limits=1):
        for d in sorted((output/"inexact/instances").glob("*")):
            if not (d/"instance.npz").exists():
                continue
            job = read(d/"job.json")
            p = load_problem(d)
            errors.extend(str(d)+": "+e for e in verify_reference(d)["errors"])
            src = instance(job["source"], job["source_instance"])
            for name in ("instance.npz", "reference.json"):
                if digest(src/name) != digest(d/name):
                    errors.append(str(d)+": reused input differs: "+name)
            for path in sorted(d.glob("*/result.json")):
                r, rd, local = read(path), path.parent, []
                state = np.load(rd/"state.npz")
                f, _, _ = value_gradient(p.A, p.b, state["x"], p.family, p.beta, p.power)
                if not np.isclose(f, r["delta"], rtol=2e-8, atol=1e-25):
                    local.append("endpoint objective")
                if np.max(np.abs(soft(state["p"], p.lam)-state["x"])) > 1e-12:
                    local.append("incompatible primal/dual")
                if r["N"] != r["K"]+1 or r["M"] != 2*r["N"]:
                    local.append("evaluation counters")
                if "inline_audit_errors" not in r:
                    local.append("missing inline per-step audit")
                local += ["inline: "+e for e in r.get("inline_audit_errors", [])]
                local += verify_saved(rd, p, r)
                if r["flag"] != 0:
                    cap = configuration(r["method"]["configuration"]).get("cap_expected", False) and r["flag"] == 8
                    (expected_caps if cap else failures).append(
                        dict(instance_id=job["id"], method=r["method"], flag=r["flag"], status=r["status"]))
                errors.extend(str(path)+": "+e for e in local)
                details.append(dict(instance_id=job["id"], method=r["method"], errors=local))
                found.add((job["id"], r["method"]["name"], r["method"]["configuration"]))
    expected = {(j["id"], m["name"], m["configuration"]) for j in frozen["jobs"] for m in j["methods"]}
    timing_path, timed = output/"timing/summary.csv", None
    if timing_path.exists():
        with timing_path.open(encoding="utf-8-sig", newline="") as stream:
            timed = Counter(r["timed"] for r in csv.DictReader(stream))
    result = dict(integrity_passed=not errors, numerical_failures=failures, expected_inner_caps=len(expected_caps),
                  expected_inner_cap_runs=expected_caps, errors=errors,
                  verified_runs=len(found), expected_runs=len(expected), complete=found == expected,
                  timing_replays=dict(timed) if timed is not None else None, timed_labels=[t[0] for t in TIMED],
                  passed=not errors and not failures and found == expected and timed is not None,
                  notes=("A faithfully recorded failure (flag != 0) passes integrity but fails the experiment; "
                         "inner-cap stops of the projected-gradient diagnostics are expected and listed separately."))
    write_json(output/"verification_details.json", details)
    write_json(output/"verification.json", result)
    print(f"Verification: {len(found)}/{len(expected)} runs; {len(errors)} integrity errors; "
          f"{len(failures)} numerical failures; timing {dict(timed) if timed else 'missing'}", flush=True)
    return result
