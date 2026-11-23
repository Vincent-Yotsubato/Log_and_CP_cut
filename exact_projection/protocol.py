"""Same-code exact comparator and lean timing for the inexact H branch.

In results/rates_and_inexact, SSN-A (inexact acceptance, solver 1 of rates_and_inexact.inexact)
is compared with SSN-E, the exact code of elastic_net. SSN-E has a different start and a
Brent single-active search, so the W difference mixes the acceptance test with the
implementation. The SSN-A timing also includes per-step logging and the diagnostic
||n + A^T w||, a matrix-vector product per outer step. This study adds

    SSN-KKT  the SSN-A Newton iteration (same start, single-active branch and line search)
             accepting only candidates with normalized KKT residual <= 1e-12, i.e.
             rates_and_inexact.inexact.run with solver 1 and accept_exact=True;

and times SSN-KKT, SSN-A and SSN-E to the first 1% hit in one serial session. SSN-A and SSN-E
are not rerun: their saved first hits are replayed and checked.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from baseline.protocol import identity
from rates_and_inexact import protocol as base

EXPERIMENTS = Path(__file__).resolve().parents[1]
SOURCE = EXPERIMENTS/"results/rates_and_inexact"
SSN_KKT = dict(name="SSN-KKT", solver="ssn", acceptance="E", eta_rel=0., exponent=0., main=False)
METHODS = ("H-CP", "H-Log")
# (label, archived configuration whose first hit is replayed, harness)
TIMED = (("SSN-KKT", "SSN-KKT", "lean"), ("SSN-A", "SSN-A", "lean"),
         ("SSN-A (logged)", "SSN-A", "logged"), ("SSN-E", "SSN-E", "historical"))
SETTINGS = dict(
    schema=1, gradient_budget=20000, distance_target=.01, kkt_tolerance=1e-12,
    inner_cap=base.SETTINGS["inner_cap"], float_floor_factor=base.SETTINGS["float_floor_factor"],
    invariant_tolerance=base.SETTINGS["invariant_tolerance"], timing_repeats=3, blas_threads=1,
    # Per-process medians can differ by up to 1.7x between fresh processes, and differently
    # for the two branches; the timing therefore uses several processes.
    timing_processes=5, timing_seed=20261104, configuration=SSN_KKT, timed=[list(t) for t in TIMED],
    source="results/rates_and_inexact (inexact panel: inputs, SSN-A and SSN-E results; read-only)",
    harnesses=dict(
        lean=("exact_projection.lean.lean_run: the outer loop of rates_and_inexact.inexact.run "
              "with the same evaluate/cuts/inner calls; keeps the cut admissibility check, the "
              "Q-inclusion certificate through the witness recursion, weak duality and the distance "
              "test; omits trace/audit rows, histories, Omega(x_k) and ||n + A^T w||"),
        logged="rates_and_inexact.inexact.run, the archived timing scope of SSN-A",
        historical="elastic_net.numerics.run with two logged steps, the archived scope of SSN-E"),
)


@contextmanager
def registered():
    """Make SSN-KKT known to the rates_and_inexact helpers for the duration of a call.

    rates_and_inexact.runner.worker and verification.audit_inexact look configurations up in
    rates_and_inexact.protocol.CONFIGURATIONS; their source files stay unchanged.
    """
    base.CONFIGURATIONS.append(SSN_KKT)
    try:
        yield
    finally:
        base.CONFIGURATIONS.remove(SSN_KKT)


def archived_jobs():
    return json.loads((SOURCE/"config.json").read_text(encoding="utf-8"))["jobs"]["inexact"]


def build():
    """The 40 saved elastic-net jobs with SSN-KKT for H-CP and H-Log."""
    jobs = []
    for old in archived_jobs():
        job = {k: v for k, v in old.items() if k != "id"}
        job["methods"] = [dict(name=name, configuration=SSN_KKT["name"]) for name in METHODS]
        job["archived_job"] = old["id"]
        job["id"] = identity(job)
        jobs.append(job)
    return jobs


def run_id(name, configuration):
    return identity(dict(name=name, configuration=configuration))
