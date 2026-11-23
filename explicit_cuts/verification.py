"""Independent checks: inputs, endpoints, counters, cut ordering, and a direct replay."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from cut_functions import a_star, s_log, s_c
from elastic_net.problems import value_gradient, soft
from rates_and_inexact.common import trace
from .protocol import SOURCE, LABELS
from .runner import read, write_json, digest, load, source_hashes

REPLAY_STEPS = 30


def cut(name, f, q, L0, L1):
    """The cut parameter and multiplier of each update, from cut_functions.py (not the numba core)."""
    q = np.atleast_1d(np.asarray(q, dtype=float))
    if name == "Polyak-1":
        return f, 1.
    if name == "Polyak-2":
        return f, 2.
    if name == "CP":
        return f+float(a_star(q, L0, L1)[0]), 1.
    if name == "Log-LBreI":
        return float(np.atleast_1d(s_log(q, L0, L1))[0]), 1.
    if name == "C-LBreI":
        return float(np.atleast_1d(s_c(q, L0, L1))[0]), 1.
    raise ValueError(name)


def replay(p, name, steps):
    """Plain numpy Algorithm 1 for the first steps; returns f and the distance to x_ref."""
    dual = p.p0.copy()
    out = []
    for _ in range(steps):
        x = soft(dual, p.lam)
        f, g, _ = value_gradient(p.A, p.b, x, 0, p.beta)
        out.append((f, np.linalg.norm(x-p.z)))
        q = np.linalg.norm(g)
        if q == 0. or f == 0.:
            break
        sigma, multiplier = cut(name, f, q, p.L0, p.L1)
        dual = dual-multiplier*sigma/(q*q)*g
    return np.array(out)


def check_run(rd, p, r):
    errors = []
    with np.load(rd/"state.npz") as s:
        state = {k: s[k] for k in s.files}
    f, _, _ = value_gradient(p.A, p.b, state["x"], 0, p.beta)
    if not np.isclose(f, r["delta"], rtol=2e-8, atol=1e-25):
        errors.append("endpoint objective")
    if np.max(np.abs(soft(state["p"], p.lam)-state["x"])) > 1e-12:
        errors.append("incompatible primal/dual")
    if r["N"] != r["K"]+1 or r["M"] != 2*r["N"] or r["W"] != r["N"]:
        errors.append("evaluation counters")
    if r["max_cut_ratio"] > 1+1e-10:
        errors.append("a cut exceeded the exact depth")
    distance = np.linalg.norm(state["x"]-p.z)
    feasibility = np.linalg.norm(p.A@state["x"]-p.b)/(1+np.linalg.norm(p.b))
    joint = bool(feasibility <= 1e-8 and distance/(1+np.linalg.norm(p.z)) <= 1e-6)
    if joint != r["selection_success"]:
        errors.append("selection flag differs from the endpoint")
    # Targets decrease, so first hits must not decrease, and a missed target stays missed.
    missed, last = False, 0
    for h in r["distance_hits"]:
        if h is None:
            missed = True
        elif missed or h < last:
            errors.append("distance hits are not ordered by target")
            break
        else:
            last = h
    t = trace(rd/"trace.csv.gz")
    # Cut ordering at every stored point, recomputed from q and delta with cut_functions.py.
    valid = t["tau"] > 0
    q, delta, tau = t["q"][valid], t["delta"][valid], t["tau"][valid]
    if q.size:
        clipped, log = s_c(q, p.L0, p.L1), s_log(q, p.L0, p.L1)
        cp = delta+a_star(q, p.L0, p.L1)
        slack = 1e-10*(1+tau)
        if np.any(clipped > log+slack):
            errors.append("clipped cut above the Log cut")
        if np.any(np.maximum(delta, log) > cp+slack):
            errors.append("CP cut below max(delta, Log)")
        if np.any(cp > tau*(1+1e-10)+1e-300):
            errors.append("CP cut above the exact depth")
        if p.L1 == 0. and np.any(np.abs(clipped-log) > 1e-14*(1+log)):
            errors.append("clipped and Log differ at L1=0")
        for column, values in (("rho_C", clipped), ("rho_log", log), ("rho_CP", cp)):
            if np.any(np.abs(t[column][valid]*tau-values) > 1e-9*(1+values)):
                errors.append("stored ratio differs from the recomputed cut: "+column)
    # Independent replay of the first steps against the stored dense prefix (k < 30).
    rep = replay(p, r["method"]["name"], min(REPLAY_STEPS, r["N"]))
    k = t["k"].astype(int)
    m = min(len(rep), int(np.sum(k < len(rep))))
    if m:
        if not np.allclose(rep[:m, 0], t["f"][:m], rtol=1e-9, atol=1e-18):
            errors.append("replayed objective differs")
        if not np.allclose(rep[:m, 1], t["dist_to_zdag"][:m], rtol=1e-8, atol=1e-14):
            errors.append("replayed distance differs")
    return errors


def verify(output):
    output = Path(output)
    frozen = read(output/"config.json")
    errors, counts, statuses = [], Counter(), Counter()
    for name, sha in frozen["hashes"].items():
        if digest(output/"source"/name) != sha:
            errors.append("source snapshot mismatch: "+name)
    if digest(SOURCE/"config.json") != frozen["source_config"]:
        errors.append("archive configuration changed")
    for d in sorted((output/"instances").glob("*")):
        job = read(d/"job.json")
        if job["source_instance"] is not None:
            src = SOURCE/"instances"/job["source_instance"]
            for name in ("instance.npz", "reference.json"):
                if digest(src/name) != digest(d/name):
                    errors.append(f"{d.name}: input differs from the archive: {name}")
        else:
            # New instance: recheck its certificate at 65 digits.
            from rates_and_inexact.verification import verify_reference
            errors.extend(f"{d.name}: {e}" for e in verify_reference(str(d))["errors"])
        p = load(d, job["model"])
        for path in sorted(d.glob("*/result.json")):
            r = read(path)
            local = check_run(path.parent, p, r)
            errors.extend(f"{d.name}/{path.parent.name} {LABELS[r['method']['name']]}: {e}" for e in local)
            counts[r["method"]["name"]] += 1
            statuses[r["status"]] += 1
    expected = sum(len(j["methods"]) for j in frozen["jobs"])
    result = dict(integrity_passed=not errors, errors=errors[:200], error_count=len(errors),
                  runs=sum(counts.values()), expected_runs=expected, complete=sum(counts.values()) == expected,
                  statuses=dict(statuses), per_method=dict(counts), pilot=frozen["pilot"],
                  references=("archived inputs byte-identical to the archive (certified there); "
                              "new kappa=10 certificates rechecked at 65 digits"))
    write_json(output/"verification.json", result)
    print(f"Verification: {result['runs']}/{expected} runs; {len(errors)} errors; statuses {dict(statuses)}", flush=True)
    return result
