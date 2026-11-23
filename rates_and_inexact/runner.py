"""Frozen configuration, immutable source inputs, resumable per-method results."""
from __future__ import annotations

import concurrent.futures
import platform
import shutil
import time
import traceback
from pathlib import Path

import numba
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from baseline.protocol import identity, method as old_method
from elastic_net import numerics as historical
from elastic_net.runner import solve as improved_solve
from .common import (EXPERIMENTS, SOURCES, read, write_json, digest, instance,
                     load_problem, save_trace, reference_interval)
from .protocol import SETTINGS, SOLVER_CODES, build, configuration, manifest
from . import inexact


def source_hashes():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [EXPERIMENTS/"cut_functions.py"]
    for module in ("baseline", "elastic_net"):
        paths += [EXPERIMENTS/module/name for name in
                  ("__init__.py", "numerics.py", "problems.py", "protocol.py", "runner.py")]
    return {str(p.relative_to(EXPERIMENTS)).replace("\\", "/"): digest(p) for p in sorted(paths)}


def prepare(output, pilot=False):
    output = Path(output).resolve()
    # Never write into an input result directory or beneath it.
    for historical_root in (EXPERIMENTS/"results/baseline",
                            EXPERIMENTS/"results/elastic_net",
                            EXPERIMENTS/"results/selection_costs"):
        if output == historical_root or historical_root in output.parents:
            raise ValueError("Choose a new output directory; input results are read-only")
    frozen = dict(settings=SETTINGS, pilot=pilot, jobs=manifest(pilot), hashes=source_hashes(),
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   scipy=scipy.__version__, numba=numba.__version__, blas_threads=1),
                  source_configs={k: digest(v/"config.json") for k, v in SOURCES.items()})
    path = output/"config.json"
    if output.exists() and not path.exists() and any(output.iterdir()):
        raise RuntimeError("The initial output directory must be empty; existing work is not overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != frozen:
            raise RuntimeError("Frozen source/config/environment differs. Use a NEW --out directory.")
    else:
        write_json(path, frozen)
        for name in frozen["hashes"]:
            target = output/"source"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(EXPERIMENTS/name, target)
    return frozen


def summarize(stats):
    keys = ["K", "N", "M", "W", "IP", "flag", "delta0", "delta", "distance", "relative_distance",
            "feasibility", "max_cut_ratio", "max_Q_violation", "max_Omega_drop", "max_Omega_excess",
            "max_projection_residual", "total_inner", "max_inner", "complementarity"]
    result = {k: float(v) for k, v in zip(keys, stats)}
    for key in ("K", "N", "M", "W", "IP", "flag", "total_inner", "max_inner"):
        result[key] = int(result[key])
    return result


def sparse(values):
    if len(values) < 600:
        return values
    steps = np.unique(np.r_[np.arange(30), np.geomspace(30, len(values)-1, 500).astype(int), len(values)-1])
    return values[steps]


def run_approximate(p, spec, budget, omega_upper):
    cfg = configuration(spec["configuration"])
    return inexact.run(p.A, p.b, p.power, p.L0, p.L1, p.lam, p.p0, p.z, omega_upper,
                       int(spec["name"] == "H-Log"), budget, SOLVER_CODES[cfg["solver"]],
                       cfg["acceptance"] == "E", cfg["eta_rel"], cfg["exponent"],
                       SETTINGS["inner_cap"], SETTINGS["float_floor_factor"], SETTINGS["invariant_tolerance"])


def initialize(job, directory):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"job.json", job)
    if job["source"] is not None:  # every job reads a saved instance
        src = instance(job["source"], job["source_instance"])
        for name in ("instance.npz", "reference.json"):
            if (directory/name).exists() and digest(directory/name) != digest(src/name):
                raise RuntimeError("Saved input differs from the frozen source: "+str(directory/name))
            if not (directory/name).exists():
                shutil.copy2(src/name, directory/name)
        write_json(directory/"provenance.json", dict(source=str(src.relative_to(EXPERIMENTS)),
                   hashes={name: digest(src/name) for name in ("job.json", "instance.npz", "reference.json")}))
    return load_problem(directory)


def worker(job, output):
    directory = Path(output)/job["panel"]/"instances"/job["id"]
    try:
        with threadpool_limits(limits=1):
            p = initialize(job, directory)
            statuses = []
            for spec in job["methods"]:
                run_id = identity(spec)
                rd = directory/run_id
                rd.mkdir(exist_ok=True)
                if (rd/"result.json").exists():
                    statuses.append(read(rd/"result.json")["status"])
                    continue
                # This wall clock is for execution progress only. Formal timing
                # warms the signatures and records three serial repeats.
                started = time.perf_counter()
                panel, budget = job["panel"], job["gradient_budget"]
                if panel == "inexact" and spec["configuration"] != "SSN-E":
                    interval = reference_interval(str(directory))
                    ans = run_approximate(p, spec, budget, interval["omega_upper"])
                    values, audit, dual, x, normal, c, J, witness, dual_history, corrections, stats = ans
                    result = {k: float(v) for k, v in zip(inexact.STAT_COLUMNS, stats)}
                    for k in inexact.STAT_COLUMNS[:6]+["first_distance_N", "first_distance_W", "first_joint_N",
                                                       "eta_floor_uses", "pg_i_violations", "start_accepts", "first_distance_C"]:
                        result[k] = int(result[k])
                    for k in ("first_distance_N", "first_distance_W", "first_joint_N", "first_distance_C"):
                        if result[k] < 0:
                            result[k] = None
                    levels, counts = np.unique(audit[:, 16].astype(np.int64), return_counts=True)
                    result.update(status={0: "budget", 6: "inadmissible_cut", 7: "invariant_failure",
                                          8: "inner_budget", 9: "nonfinite", 11: "acceptance_stalled"}.get(result["flag"], "failure"),
                                  best_q=float(values[-1, 8]), J=J, c=c,
                                  inner_iterations_median=float(np.median(audit[audit[:, 21] == 1, 16]))
                                  if np.any(audit[:, 21] == 1) else None,
                                  inner_iterations_max=int(np.max(audit[:, 16])) if len(audit) else 0,
                                  # All outer attempts, including a final failed one.
                                  inner_iterations_histogram=[[int(a), int(b)] for a, b in zip(levels, counts)],
                                  tolerance=configuration(spec["configuration"]), exact_reference=False)
                    # Independent one-step audit of every actual state, before only a
                    # sample of the states is written (full histories are ~11 MB/run).
                    from .verification import audit_inexact, audit_steps
                    audit_columns = dict(zip(inexact.AUDIT_COLUMNS, audit.T))
                    n_audit, N = len(audit), len(dual_history)
                    following = dual_history[np.minimum(np.arange(n_audit)+1, N-1)]
                    result["inline_audit_errors"] = audit_inexact(
                        p, {**result, "method": spec}, audit_columns, np.arange(n_audit), dual_history[:n_audit], following,
                        corrections, interval, every=dual_history)
                    extra = [h-o for h in (result["first_distance_N"], result["first_joint_N"])
                             if h is not None for o in (1, 2)]
                    keep = audit_steps(n_audit, extra)
                    columns = inexact.TRACE_COLUMNS
                    save_trace(rd/"audit.csv.gz", inexact.AUDIT_COLUMNS, audit[keep])
                    np.savez_compressed(rd/"audit_states.npz", steps=keep, p=dual_history[keep],
                                        p_next=following[keep], u=corrections[keep], p_final=dual_history[-1])
                    np.savez_compressed(rd/"state.npz", p=dual, x=x, normal=normal, c=c, J=J, witness=witness)
                else:
                    # Exact comparator (elastic_net code); preserve both path and W count.
                    m = old_method(spec["name"])
                    base = dict(panel="nonsmooth", max_updates=budget-1, stop_mode=0, diagnostic=False)
                    result, values, dual, x, witness = improved_solve(p, m, base)
                    columns = historical.COLUMNS
                    np.savez_compressed(rd/"state.npz", p=dual, x=x, witness=witness)
                    result["exact_reference"] = True
                    if panel == "inexact":
                        src = instance(job["source"], job["source_instance"])/identity(m)
                        original, state = read(src/"result.json"), np.load(src/"state.npz")
                        result["matched_original"] = (
                            bool(all(result[k] == original[k] for k in ("K", "N", "M", "W"))
                                 and np.max(np.abs(x-state["x"])) <= 1e-12
                                 and np.max(np.abs(dual-state["p"])) <= 1e-12)
                            if budget == job["source_job"]["max_updates"]+1 else None)
                        # Dense replay provides exact target costs, not log interpolation.
                        from .verification import exact_distance_hits
                        result.update(exact_distance_hits(p, spec["name"], budget))
                result.update(method=spec, instance_id=job["id"], panel=job["panel"],
                              source=job["source"], source_instance=job.get("source_instance"),
                              pilot=job["pilot"], metadata={**p.metadata, "lambda_en": p.lam},
                              selection_success=bool(result["feasibility"] <= 1e-8
                                  and result["distance"]/(1+np.linalg.norm(p.z)) <= 1e-6),
                              elapsed_including_io_seconds=time.perf_counter()-started,
                              timing_scope="Formal cost-to-target timing is a separate serial command; includes all certificate work")
                save_trace(rd/"trace.csv.gz", columns, sparse(values))
                write_json(rd/"result.json", result)
                statuses.append(result["status"])
            return dict(id=job["id"], runs=len(statuses), statuses={s: statuses.count(s) for s in set(statuses)})
    except Exception:
        error = traceback.format_exc()
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory/"error.json", dict(error=error))
        return dict(id=job["id"], runs=0, error=error)


def warmup():
    A, b, p0, z = np.eye(2), np.ones(2), np.zeros(2), np.ones(2)
    for solver in (0, 1):
        inexact.run(A, b, 3, 1., 1., .1, p0, z, 2., 0, 2, solver, False, .01, 2., 5)
    historical.run(A, b, 1, 0., 3, 1., 1., .1, p0, z, 4, 1., 1., 1, 3,
                   np.arange(2, dtype=np.int64), np.array([1e-4]), np.array([.01]), False)


def execute(panel, output, pilot=False, workers=4):
    jobs = build(panel, pilot)
    destination = Path(output)/panel
    destination.mkdir(parents=True, exist_ok=True)
    warmup()
    start, done, runs, errors = time.perf_counter(), 0, 0, []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, j, str(output)) for j in jobs]
        for f in concurrent.futures.as_completed(futures):
            answer = f.result()
            done += 1
            runs += answer["runs"]
            if "error" in answer:
                errors.append(answer)
            progress = dict(panel=panel, completed_jobs=done, expected_jobs=len(jobs),
                            completed_runs=runs, expected_runs=sum(len(j["methods"]) for j in jobs),
                            errors=errors, elapsed_seconds=time.perf_counter()-start, last=answer)
            write_json(destination/"progress.json", progress)
            print(f"{panel}: {done}/{len(jobs)} jobs, {runs} runs, {answer.get('statuses', 'ERROR')}", flush=True)
    write_json(destination/"completion.json", progress)
    if errors:
        raise RuntimeError(f"{panel}: {len(errors)} job exceptions; see error.json (not silently skipped)")
    return progress
