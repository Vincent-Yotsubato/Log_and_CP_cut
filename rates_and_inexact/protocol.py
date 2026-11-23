"""Inexact two-halfspace projections (panel "inexact") and the rate-window settings of rates.py."""
from __future__ import annotations

from baseline.protocol import identity
from .common import source_jobs

CONFIGURATIONS = [
    dict(name="PG-A", solver="pg", acceptance="A", eta_rel=1e-2, exponent=2., main=True),
    dict(name="PG-A2", solver="pg", acceptance="A", eta_rel=1e-2, exponent=1., main=False),
    dict(name="PG-A3", solver="pg", acceptance="A", eta_rel=1e-4, exponent=1., main=False),
    dict(name="PG-A4", solver="pg", acceptance="A", eta_rel=1e-4, exponent=2., main=False),
    dict(name="PG-E", solver="pg", acceptance="E", eta_rel=0., exponent=0., main=False),
    dict(name="SSN-E", solver="historical_ssn", acceptance="E", eta_rel=0., exponent=0., main=False),
    dict(name="SSN-A", solver="ssn", acceptance="A", eta_rel=1e-2, exponent=2., main=True),
    # Diagnostic only: the literal step 1/L_Psi in the original multipliers.
    dict(name="PG0-A", solver="pg_raw", acceptance="A", eta_rel=1e-2, exponent=2., main=False),
]
PANELS = ("inexact",)
SOLVER_CODES = {"pg_raw": 0, "ssn": 1, "pg": 2}
# Stored per run: every outer step up to AUDIT_DENSE, about AUDIT_LOG log-spaced steps,
# the steps around the first targets and the final/failed step. All steps are still
# checked in the algorithm and by the inline independent audit before saving.
AUDIT_DENSE, AUDIT_LOG = 200, 800
SETTINGS = dict(
    schema=2, mu=1., gradient_budget=20000, inner_cap=100000,
    kkt_tolerance=1e-12, invariant_tolerance=1e-12, float_floor_factor=1000.,
    distance_target=.01, feasibility_tolerance=1e-8, joint_distance_tolerance=1e-6,
    rate_window_fraction=.01, rate_gap_floor=1000., rate_initial_ceiling=.1,
    rate_minimum_points=6, blas_threads=1, configurations=CONFIGURATIONS,
    pg_coordinates=("PG-*: same dual in (lambda*||g||, nu*||n||), step mu/(1+|cos(g,n)|); "
                    "PG0-A: original multipliers, step mu/(||g||^2+||n||^2), diagnostic"),
    eta_index="outer step k accepts with eta_{k+1} = eta_rel*pi0*(k+2)^(-p)",
    audit_storage=dict(dense_steps=AUDIT_DENSE, log_steps=AUDIT_LOG,
                       note="full per-step independent audit runs inline before saving"),
    exact_baseline="SSN-E calls the exact elastic_net implementation",
    zero_gradient="inexact branch continues; a zero gradient alone is not selection",
)


def _job(panel, source, old, methods, budget, pilot):
    job = dict(panel=panel, source=source, source_instance=old["id"], model=old["model"],
               variant=old["variant"], methods=methods, gradient_budget=budget,
               pilot=pilot, source_job=old)
    job["id"] = identity(job)
    return job


def build(panel, pilot=False):
    jobs = []
    if panel == "inexact":
        for old in source_jobs("nonsmooth"):
            if pilot and old["model"]["seed"] != 0:
                continue
            methods = [dict(name=name, configuration=c["name"])
                       for name in ("H-CP", "H-Log") for c in CONFIGURATIONS]
            jobs.append(_job(panel, "nonsmooth", old, methods, 500 if pilot else 20000, pilot))
    else:
        raise ValueError(panel)
    return jobs


def configuration(name):
    return next(c for c in CONFIGURATIONS if c["name"] == name)


def manifest(pilot=False):
    return {p: build(p, pilot) for p in PANELS}
