"""The residual acceptance test (ii) |c+ - <n+, x+>| <= eta_{k+1} on the 40 elastic-net instances.

Predeclared configurations (same Newton iteration as SSN-A and SSN-KKT; only the test differs):
    SSN-R   main         two-sided residual, eta_k = 1e-2 pi0 (k+1)^-2, the schedule of SSN-A
    SSN-R1  sensitivity  two-sided residual, eta_k = 1e-2 pi0 (k+1)^-1, the weakest order allowed
                         by the rate corollary
    SSN-R+  diagnostic   one-sided residual c+ - <n+, x+> <= eta, the part used in the proofs
    PG-R    diagnostic   two-sided residual with projected-gradient inner solves in the normalized
                         multipliers (lambda*||g||, nu*||n||), as PG-A of rates_and_inexact
    PG0-R   diagnostic   the same with the step 1/L_Psi in the original multipliers;
                         both PG configurations are expected to reach the inner cap (flag 8)
Comparators, not rerun: SSN-KKT (results/exact_projection) and SSN-A, SSN-E
(results/rates_and_inexact). Their saved first hits are replayed with the mode-based copy of
the solver (modes 0 and 1) and, for SSN-KKT, also with the frozen code.
"""
from __future__ import annotations

import json
from pathlib import Path

from baseline.protocol import identity

EXPERIMENTS = Path(__file__).resolve().parents[1]
SOURCES = dict(revision=EXPERIMENTS/"results/rates_and_inexact",
               cost=EXPERIMENTS/"results/exact_projection")
CONFIGURATIONS = [
    dict(name="SSN-R", solver="ssn", acceptance="R", mode=2, eta_rel=1e-2, exponent=2., main=True),
    dict(name="SSN-R1", solver="ssn", acceptance="R", mode=2, eta_rel=1e-2, exponent=1., main=False),
    dict(name="SSN-R+", solver="ssn", acceptance="R+", mode=3, eta_rel=1e-2, exponent=2., main=False),
    dict(name="PG-R", solver="pg", acceptance="R", mode=2, eta_rel=1e-2, exponent=2., main=False, cap_expected=True),
    dict(name="PG0-R", solver="pg_raw", acceptance="R", mode=2, eta_rel=1e-2, exponent=2., main=False,
         cap_expected=True),
]
SOLVER_CODES = {"pg_raw": 0, "ssn": 1, "pg": 2}
COMPARATORS = {  # name: (results root, mode, eta_rel, exponent)
    "SSN-KKT": ("cost", 0, 0., 0.),
    "SSN-A": ("revision", 1, 1e-2, 2.),
}
METHODS = ("H-CP", "H-Log")
# (label, configuration whose recorded first hit is replayed, harness)
TIMED = (("SSN-R", "SSN-R", "lean"), ("SSN-R1", "SSN-R1", "lean"), ("SSN-R+", "SSN-R+", "lean"),
         ("SSN-KKT", "SSN-KKT", "lean"), ("SSN-A", "SSN-A", "lean"),
         ("SSN-KKT (frozen code)", "SSN-KKT", "frozen"))
SETTINGS = dict(
    schema=1, gradient_budget=20000, distance_target=.01, kkt_tolerance=1e-12, inner_cap=100000,
    float_floor_factor=1000., invariant_tolerance=1e-12, timing_repeats=3, timing_processes=5,
    blas_threads=1, timing_seed=202611041, configurations=CONFIGURATIONS,
    comparators={k: list(v) for k, v in COMPARATORS.items()}, timed=[list(t) for t in TIMED],
    eta_index="outer step k accepts with eta_{k+1} = eta_rel*pi0*(k+2)^(-p), floor 1000*eps*(1+|J|)",
    residual="lambda*(b - <g, x+>) + nu*(c - <n, x+>) = c+ - <n+, x+> = Omega(x+) - J+",
    harnesses=dict(
        lean="inexact_projection.inexact_residual.lean_run: no per-step logging; keeps the cut, "
             "Q-inclusion and weak-duality checks",
        frozen="exact_projection.lean.lean_run, which calls the frozen rates_and_inexact inner()"),
)


def configuration(name):
    return next(c for c in CONFIGURATIONS if c["name"] == name)


def archived_jobs():
    return json.loads((SOURCES["revision"]/"config.json").read_text(encoding="utf-8"))["jobs"]["inexact"]


def build():
    """The 40 archived elastic-net jobs with the three residual configurations."""
    jobs = []
    for old in archived_jobs():
        job = {k: v for k, v in old.items() if k != "id"}
        job["methods"] = [dict(name=name, configuration=c["name"]) for name in METHODS for c in CONFIGURATIONS]
        job["archived_job"] = old["id"]
        job["id"] = identity(job)
        jobs.append(job)
    return jobs
