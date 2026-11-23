"""Frozen grid: archived cosh instances, five cuts in the explicit update, one budget.

Every method is Algorithm 1 with t_k = mu*sigma_k/q_k^2 and one of the cuts
    clipped  q^2/(L0+L1 q)   (C-LBreI, theta=1; the clipped step mu/(L0+L1 q))
    Log      S_log(q)        (Log-LBreI, theta=1)
    Polyak-1 delta,  Polyak-2 (multiplier 2 on delta),  CP  delta+A_*(q).
For L1=0 the clipped and Log cuts both equal the Lipschitz cut q^2/L0, i.e. the
constant step 1/L0 of the linearized Bregman iteration.
"""
from __future__ import annotations

from pathlib import Path

from baseline.protocol import identity, method

EXPERIMENTS = Path(__file__).resolve().parents[1]
SOURCE = EXPERIMENTS/"results/baseline"
VARIANTS = ("calibrated", "quadratic_control")
METHOD_NAMES = ("C-LBreI", "Log-LBreI", "Polyak-1", "Polyak-2", "CP")
LABELS = {"C-LBreI": "clipped", "Log-LBreI": "Log", "Polyak-1": "Polyak-1", "Polyak-2": "Polyak-2", "CP": "CP"}
# The archived instances have kappa=1, i.e. B has orthonormal rows; then for beta=0 every
# cut equals the exact depth 2*delta and all updates except Polyak-1 coincide. A new
# kappa=10 family (singular values geometric from 1 to 1/10) separates the cuts.
NEW_FAMILY = dict(m=128, n=512, kappa=10., seeds=10,
                  zero_init=dict(betas=(0., 1., 4.), amplitude=.25),
                  calibrated=dict(betas=(1., 4.), amplitude=.01, target_r=1000.))
SETTINGS = dict(
    schema=1, mu=1., gradient_budget=50000, stop_mode=2, new_family=NEW_FAMILY,
    stop_rule="first evaluation with ||Bx-b||/(1+||b||)<=1e-8 and ||x-x_ref||/(1+||x_ref||)<=1e-6",
    distance_targets=[1e-2, 1e-4, 1e-6], blas_threads=1,
    source="results/baseline (calibrated and quadratic_control cosh instances)",
    cuts="clipped q^2/(L0+L1 q), Log S_log(q), Polyak delta (x1, x2), CP delta+A_*(q)",
)


def methods():
    return [method(name) for name in METHOD_NAMES]


def build(pilot=False):
    import json
    jobs = []
    budget = 500 if pilot else SETTINGS["gradient_budget"]

    def add(variant, model, source):
        job = dict(panel="cuts", variant=variant, source_instance=source, model=model,
                   methods=methods(), gradient_budget=budget, stop_mode=SETTINGS["stop_mode"], pilot=pilot)
        job["id"] = identity(job)
        jobs.append(job)

    for old in json.loads((SOURCE/"config.json").read_text(encoding="utf-8"))["jobs"]:
        if old["variant"] in VARIANTS and not (pilot and old["model"]["seed"] != 0):
            add(old["variant"], old["model"], old["id"])
    f = NEW_FAMILY
    for seed in range(1 if pilot else f["seeds"]):
        for beta in f["zero_init"]["betas"]:
            add("kappa10_zero", dict(kind="e1", seed=seed, m=f["m"], n=f["n"], kappa=f["kappa"], beta=beta,
                                     amplitude=f["zero_init"]["amplitude"]), None)
        for beta in f["calibrated"]["betas"]:
            add("kappa10_calibrated", dict(kind="e1", seed=seed, m=f["m"], n=f["n"], kappa=f["kappa"], beta=beta,
                                           amplitude=f["calibrated"]["amplitude"],
                                           target_r=f["calibrated"]["target_r"]), None)
    return jobs
