"""Integrity of the SSN-KKT runs, of the replayed archive, and of the timing replays."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from elastic_net.problems import value_gradient, soft
from rates_and_inexact.common import read, write_json, digest, trace, load_problem, instance
from rates_and_inexact.verification import verify_inexact, verify_reference
from .protocol import SOURCE, SETTINGS, TIMED, registered


def independent_kkt(p, audit, states):
    """Recompute the normalized KKT residual of every saved accepted step from the states."""
    worst = 0.
    for i in range(len(states["steps"])):
        if audit["accepted"][i] != 1.:
            break
        x = soft(states["p"][i], p.lam)
        _, g, _ = value_gradient(p.A, p.b, x, 1, 0., p.power)
        q, normal = np.linalg.norm(g), p.p0-states["p"][i]
        nn = np.linalg.norm(normal)
        new_x = soft(states["p_next"][i], p.lam)
        a, v = audit["lambda"][i], audit["nu"][i]
        new_J = -.5*new_x@new_x-audit["c_next"][i]
        da, dv = audit["b"][i]-g@new_x, audit["c_previous"][i]-normal@new_x
        kkt = 0.
        if q > 0.:
            kkt = max(kkt, abs(min(a*q, da/q)), abs(a*da)/(1+abs(new_J)))
        if nn > 0.:
            kkt = max(kkt, abs(min(v*nn, dv/nn)), abs(v*dv)/(1+abs(new_J)))
        worst = max(worst, kkt)
    return worst


def verify(output):
    output = Path(output)
    frozen = read(output/"config.json")
    errors, failures, details, found = [], [], [], set()
    for name, sha in frozen["hashes"].items():
        if digest(output/"source"/name) != sha:
            errors.append("source snapshot mismatch: "+name)
    if digest(SOURCE/"config.json") != frozen["source_config"]:
        errors.append("rates_and_inexact configuration changed")
    for name, sha in frozen["archived_results"].items():
        if digest(SOURCE/name) != sha:
            errors.append("archived result changed: "+name)
    worst_kkt = 0.
    with threadpool_limits(limits=1), registered():
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
                local += verify_inexact(rd, p, r)
                audit = trace(rd/"audit.csv.gz")
                accepted = audit["accepted"] == 1.
                if np.any(audit["kkt"][accepted] > SETTINGS["kkt_tolerance"]):
                    local.append("recorded KKT residual of an accepted step above the tolerance")
                with np.load(rd/"audit_states.npz") as saved:
                    states = {k: saved[k] for k in saved.files}
                kkt = independent_kkt(p, audit, states)
                worst_kkt = max(worst_kkt, kkt)
                if kkt > 1e-10:
                    local.append(f"independent KKT residual {kkt:.2e} at a saved step")
                if r["flag"] != 0:
                    failures.append(dict(instance_id=job["id"], method=r["method"], flag=r["flag"], status=r["status"]))
                errors.extend(str(path)+": "+e for e in local)
                details.append(dict(instance_id=job["id"], method=r["method"], errors=local))
                found.add((job["id"], r["method"]["name"]))
    expected = {(j["id"], m["name"]) for j in frozen["jobs"] for m in j["methods"]}
    timing_path, timed = output/"timing/summary.csv", None
    if timing_path.exists():
        with timing_path.open(encoding="utf-8-sig", newline="") as stream:
            timed = Counter(r["timed"] for r in csv.DictReader(stream))
    result = dict(integrity_passed=not errors, numerical_failures=failures, errors=errors,
                  verified_runs=len(found), expected_runs=len(expected), complete=found == expected,
                  max_independent_kkt_at_saved_steps=worst_kkt,
                  timing_replays=dict(timed) if timed is not None else None,
                  timed_labels=[t[0] for t in TIMED],
                  passed=not errors and not failures and found == expected and timed is not None)
    write_json(output/"verification_details.json", details)
    write_json(output/"verification.json", result)
    print(f"Verification: {len(found)}/{len(expected)} runs; {len(errors)} integrity errors; "
          f"{len(failures)} numerical failures; timing {dict(timed) if timed else 'missing'}", flush=True)
    return result
