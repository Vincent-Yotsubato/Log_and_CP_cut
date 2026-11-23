"""Frozen grid: simple bilevel baselines on the 40 archived Euclidean polyhedral instances.

Instances: results/baseline, variant "polyhedron", rho_c = 0.8,
(m, n) in {(10, 40), (20, 100)}, s in {3, 4}, ten seeds each; p0 = 0, so z_dagger is the
minimal-norm point of X = {Ax >= b}. Budget: 1e5 gradient evaluations, as for the H runs.

Constants computed from (A, b) only, without z_dagger:
    rho  = ||A^+ b||  (A^+ b lies in X, so ||z_dagger|| <= rho),
    L(r) = (s-1) ||A||^2 (||b||_inf + max_i ||a_i|| r)^(s-2), a bound on ||Hess f_s|| on {||x|| <= r}.
The H iterates and, for s = 1, the BiG-SAM iterates satisfy ||x - z_dagger|| <= ||z_dagger||, hence
||x|| <= 2 rho; L_ball = L(2 rho). AGM-BiO uses Z = {||x|| <= c rho}, c in {1, 2}.

Methods:
    MNG-local                H method with the Lipschitz cut q^2/L_ball (Beck-Sabach MNG with a
                             local constant); admissibility is checked against s*f_s(x).
    BiG-SAM(step, gamma)     alpha_k = min{gamma/k, 1}, gamma in GAMMAS; step "ball": t = 1/L_ball,
                             step "clip": t_k = 1/(L0 + L1 q) (heuristic, outside the BiG-SAM theory).
    AGM-BiO(rule, c)         rule "thm41": gamma = 1 (Theorem 4.1);
                             rule "thm44": gamma = 1/((2 L(c rho)/L_f) T^((2s-2)/(2s-1)) + 2), T = budget
                             (Theorem 4.4 with Holder exponent r = s).
H-CP and H-Log are not rerun; their first hits come from results/selection_costs,
the source of Table 3 of the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

from baseline.protocol import identity

EXPERIMENTS = Path(__file__).resolve().parents[1]
SOURCE = EXPERIMENTS/"results/baseline"
H_SOURCE = EXPERIMENTS/"results/selection_costs/selection_costs_runs.csv"
GAMMAS = (0.2, 1., 2., 4., 20.)
SETTINGS = dict(
    schema=1, gradient_budget=100000, distance_target=1e-2,
    joint_rule="||(b-Ax)_+||/(1+||b||)<=1e-8 and ||x-x_ref||/(1+||x_ref||)<=1e-6",
    gammas=list(GAMMAS), agm_radius_factors=[1., 2.], upper_smoothness=1., blas_threads=1,
    source="results/baseline (polyhedron, rho_c=0.8, Euclidean kernel, p0=0)",
    h_source="results/selection_costs/selection_costs_runs.csv (Euclidean H-CP, H-Log)",
    counting=("one (f, grad f) evaluation per iteration for every method; N counts evaluations; "
              "AGM-BiO hits are recorded at the output x_{k+1} produced by k+1 evaluations"),
)


def methods():
    specs = [dict(name="MNG-local")]
    for step in ("ball", "clip"):
        for gamma in GAMMAS:
            specs.append(dict(name="BiG-SAM", step=step, gamma=gamma))
    specs += [dict(name="AGM-BiO", rule="thm41", radius=1.), dict(name="AGM-BiO", rule="thm41", radius=2.),
              dict(name="AGM-BiO", rule="thm44", radius=1.)]
    return specs


def label(spec):
    if spec["name"] == "BiG-SAM":
        return f"BiG-SAM({spec['step']}, gamma={spec['gamma']:g})"
    if spec["name"] == "AGM-BiO":
        return f"AGM-BiO({spec['rule']}, radius={spec['radius']:g}rho)"
    return spec["name"]


def build(pilot=False):
    jobs = []
    budget = 2000 if pilot else SETTINGS["gradient_budget"]
    for old in json.loads((SOURCE/"config.json").read_text(encoding="utf-8"))["jobs"]:
        model = old["model"]
        if old["variant"] != "polyhedron" or model.get("correlation") != .8:
            continue
        if pilot and model["seed"] != 0:
            continue
        job = dict(panel="bilevel_baselines", variant="polyhedron", source_instance=old["id"], model=model,
                   methods=methods(), gradient_budget=budget, pilot=pilot)
        job["id"] = identity(job)
        jobs.append(job)
    return jobs
