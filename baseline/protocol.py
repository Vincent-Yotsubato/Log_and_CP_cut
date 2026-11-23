"""Frozen job grid of panels 1 and 2; seeds and budgets are fixed in advance."""
import hashlib
import itertools
import json
import numpy as np

SELECTION = ["Polyak-1", "Polyak-2", "CP", "C-LBreI", "Log-LBreI", "H-Log", "H-CP"]


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def method(name, theta=1., c=1.):
    label = name
    if name == "CP" and c != 1:
        label += f"(c={c:g})"
    if name in ("C-LBreI", "Log-LBreI"):
        label += f"(theta={theta:g})"
    return {"name": name, "label": label, "theta": theta, "c": c}


def build(profile="full", panels=(1, 2)):
    smoke = profile == "smoke"
    seeds = range(1 if smoke else 10)
    jobs = []

    def add(panel, variant, model, methods, budget, stop, dense=False):
        if smoke:
            budget = min(budget, 250)
        job = {"panel": panel, "variant": variant, "model": model,
               "methods": methods, "max_updates": budget, "stop_mode": stop,
               "dense": dense}
        job["id"] = identity(job)
        jobs.append(job)

    methods = [method(s) for s in SELECTION]
    if 1 in panels:
        add(1, "example32", {"kind": "poly", "example": True}, methods, 99999, 0)
        for (m, n), power, correlation, seed in itertools.product(
                ((10, 40), (20, 100)), (3, 4), (0., .8), seeds):
            add(1, "polyhedron", {"kind": "poly", "seed": seed, "m": m, "n": n,
                 "power": power, "correlation": correlation}, methods, 99999, 0)
        # rho_c=0 Gaussian rows are only approximately orthogonal. This control
        # tests the exact complementarity mechanism without changing that grid.
        for power in (3, 4):
            add(1, "orthogonal_control", {"kind": "poly", "seed": 0, "m": 10,
                 "n": 40, "power": power, "correlation": 0., "orthogonal": True}, methods, 99999, 0)
    m, n = (16, 64) if smoke else (128, 512)
    if 2 in panels:
        comparison = [method("Polyak-1"), method("Polyak-2"), method("CP"),
                      method("CP", c=2), method("CP", c=10), method("Log-LBreI")]
        for beta, target, seed in itertools.product((1., 4.), (1., 10., 100., 1000.), seeds):
            add(2, "calibrated", {"kind": "e1", "seed": seed, "m": m, "n": n,
                 "beta": beta, "kappa": 1., "amplitude": .01, "target_r": target},
                comparison, 49999, 1)
        for amplitude, seed in itertools.product((.25, 1.), seeds):
            add(2, "quadratic_control", {"kind": "e1", "seed": seed, "m": m, "n": n,
                 "beta": 0., "kappa": 1., "amplitude": amplitude}, comparison, 49999, 1)
    return jobs


def log_steps(max_updates, dense=False):
    if dense:
        return np.arange(max_updates + 1, dtype=np.int64)
    steps = np.r_[np.arange(min(21, max_updates + 1)),
                  np.geomspace(21, max(21, max_updates), 350).astype(np.int64),
                  [999, 9999, 99999, max_updates]]
    return np.unique(steps[(steps >= 0) & (steps <= max_updates)]).astype(np.int64)
