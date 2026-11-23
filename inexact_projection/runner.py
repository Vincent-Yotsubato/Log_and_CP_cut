"""Frozen configuration, residual-test runs with inline audit, and multi-process timing."""
from __future__ import annotations

import concurrent.futures
import csv
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numba
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from baseline.protocol import identity
from rates_and_inexact.common import (read, write_json, digest, csv_write, load_problem, reference_interval,
                                      save_trace)
from rates_and_inexact.runner import initialize, sparse
from rates_and_inexact.verification import audit_steps
from exact_projection.lean import lean_run as frozen_lean_run
from . import inexact_residual as solver
from .protocol import EXPERIMENTS, SOURCES, SETTINGS, COMPARATORS, TIMED, SOLVER_CODES, build, configuration
from .verification import audit_residual

STATUS = {0: "budget", 6: "inadmissible_cut", 7: "invariant_failure", 8: "inner_budget", 9: "nonfinite",
          11: "acceptance_stalled"}


def source_hashes():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += sorted((EXPERIMENTS/"exact_projection").glob("*.py"))
    paths += sorted((EXPERIMENTS/"rates_and_inexact").glob("*.py"))
    paths += [EXPERIMENTS/"elastic_net"/n for n in ("numerics.py", "problems.py", "runner.py")]
    paths += [EXPERIMENTS/"baseline"/n for n in ("protocol.py", "runner.py")] + [EXPERIMENTS/"cut_functions.py"]
    return {str(p.relative_to(EXPERIMENTS)).replace("\\", "/"): digest(p) for p in paths}


def comparator_results():
    """Digests of the replayed comparator results (SSN-KKT, SSN-A, SSN-E)."""
    out = {}
    for key, keep in (("cost", ("SSN-KKT",)), ("revision", ("SSN-A", "SSN-E"))):
        for path in sorted((SOURCES[key]/"inexact/instances").glob("*/*/result.json")):
            if read(path)["method"]["configuration"] in keep:
                out[key+"/"+str(path.relative_to(SOURCES[key])).replace("\\", "/")] = digest(path)
    return out


def prepare(output):
    output = Path(output).resolve()
    for root in SOURCES.values():
        if root == output or root in output.parents:
            raise ValueError("Choose a new output directory; the comparator archives are read-only")
    frozen = dict(settings=SETTINGS, jobs=build(), hashes=source_hashes(),
                  source_configs={k: digest(v/"config.json") for k, v in SOURCES.items()},
                  comparator_results=comparator_results(),
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   scipy=scipy.__version__, numba=numba.__version__, blas_threads=1))
    path = output/"config.json"
    if output.exists() and not path.exists() and any(output.iterdir()):
        raise RuntimeError("The initial output directory must be empty.")
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


def logged_run(p, name, code, mode, eta_rel, exponent, budget, omega_upper):
    return solver.run(p.A, p.b, p.power, p.L0, p.L1, p.lam, p.p0, p.z, omega_upper, int(name == "H-Log"),
                      budget, code, mode, eta_rel, exponent, SETTINGS["inner_cap"], SETTINGS["float_floor_factor"],
                      SETTINGS["invariant_tolerance"])


def worker(job, output):
    directory = Path(output)/"inexact"/"instances"/job["id"]
    try:
        with threadpool_limits(limits=1):
            p = initialize(job, directory)
            interval = reference_interval(str(directory))
            statuses = []
            for spec in job["methods"]:
                rd = directory/identity(spec)
                rd.mkdir(exist_ok=True)
                if (rd/"result.json").exists():
                    statuses.append(read(rd/"result.json")["status"])
                    continue
                started = time.perf_counter()
                cfg = configuration(spec["configuration"])
                ans = logged_run(p, spec["name"], SOLVER_CODES[cfg["solver"]], cfg["mode"], cfg["eta_rel"], cfg["exponent"],
                                 job["gradient_budget"], interval["omega_upper"])
                values, audit, dual, x, normal, c, J, witness, dual_history, corrections, stats = ans
                result = {k: float(v) for k, v in zip(solver.STAT_COLUMNS, stats)}
                for k in solver.STAT_COLUMNS[:6]+["first_distance_N", "first_distance_W", "first_joint_N",
                                                  "eta_floor_uses", "pg_i_violations", "start_accepts", "first_distance_C"]:
                    result[k] = int(result[k])
                for k in ("first_distance_N", "first_distance_W", "first_joint_N", "first_distance_C"):
                    if result[k] < 0:
                        result[k] = None
                accepted = audit[:, 21] == 1.
                before = np.arange(len(audit)) < ((result["first_distance_N"] or 0)-1)
                prefix = accepted & before
                levels, counts = np.unique(audit[:, 16].astype(np.int64), return_counts=True)
                result.update(
                    status=STATUS.get(result["flag"], "failure"), best_q=float(values[-1, 8]), J=J, c=c,
                    inner_iterations_max=int(np.max(audit[:, 16])) if len(audit) else 0,
                    inner_iterations_histogram=[[int(a), int(b)] for a, b in zip(levels, counts)],
                    negative_residual_steps=int(np.sum(accepted & (audit[:, 13] < 0.))),
                    prefix_accepted_steps=int(np.sum(prefix)),
                    prefix_at_kkt_tolerance=int(np.sum(prefix & (audit[:, 15] <= SETTINGS["kkt_tolerance"]))),
                    prefix_start_accepted=int(np.sum(prefix & (audit[:, 19] == 1.))),
                    prefix_negative_residual=int(np.sum(prefix & (audit[:, 13] < 0.))),
                    tolerance=cfg, exact_reference=False)
                audit_columns = dict(zip(solver.AUDIT_COLUMNS, audit.T))
                n_audit, N = len(audit), len(dual_history)
                following = dual_history[np.minimum(np.arange(n_audit)+1, N-1)]
                result["inline_audit_errors"] = audit_residual(
                    p, {**result, "method": spec}, audit_columns, np.arange(n_audit), dual_history[:n_audit],
                    following, interval, every=dual_history)
                extra = [h-o for h in (result["first_distance_N"], result["first_joint_N"]) if h is not None for o in (1, 2)]
                keep = audit_steps(n_audit, extra)
                save_trace(rd/"audit.csv.gz", solver.AUDIT_COLUMNS, audit[keep])
                np.savez_compressed(rd/"audit_states.npz", steps=keep, p=dual_history[keep],
                                    p_next=following[keep], p_final=dual_history[-1])
                np.savez_compressed(rd/"state.npz", p=dual, x=x, normal=normal, c=c, J=J, witness=witness)
                result.update(method=spec, instance_id=job["id"], panel=job["panel"], source=job["source"],
                              source_instance=job.get("source_instance"), pilot=job["pilot"],
                              metadata={**p.metadata, "lambda_en": p.lam},
                              selection_success=bool(result["feasibility"] <= 1e-8
                                                     and result["distance"]/(1+np.linalg.norm(p.z)) <= 1e-6),
                              elapsed_including_io_seconds=time.perf_counter()-started)
                save_trace(rd/"trace.csv.gz", solver.TRACE_COLUMNS, sparse(values))
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
    for code in (0, 1, 2):
        solver.run(A, b, 3, 1., 1., .1, p0, z, 2., 0, 2, code, 2, .01, 2., 5, 1000., 1e-12)
    for mode in range(4):
        solver.run(A, b, 3, 1., 1., .1, p0, z, 2., 0, 2, 1, mode, .01, 2., 5, 1000., 1e-12)
        solver.lean_run(A, b, 3, 1., 1., .1, p0, z, 2., 0, 2, 1, mode, .01, 2., 5, 1000., 1e-12)
    frozen_lean_run(A, b, 3, 1., 1., .1, p0, z, 2., 0, 2, 1, True, 0., 0., 5, 1000., 1e-12)


def execute(output, workers=4):
    jobs = build()
    destination = Path(output)/"inexact"
    destination.mkdir(parents=True, exist_ok=True)
    warmup()
    started, done, runs, errors = time.perf_counter(), 0, 0, []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, j, str(output)) for j in jobs]
        for f in concurrent.futures.as_completed(futures):
            answer = f.result()
            done += 1
            runs += answer["runs"]
            if "error" in answer:
                errors.append(answer)
            progress = dict(completed_jobs=done, expected_jobs=len(jobs), completed_runs=runs,
                            expected_runs=sum(len(j["methods"]) for j in jobs), errors=errors,
                            elapsed_seconds=time.perf_counter()-started, last=answer)
            write_json(destination/"progress.json", progress)
            print(f"residual: {done}/{len(jobs)} jobs, {runs} runs, {answer.get('statuses', 'ERROR')}", flush=True)
    write_json(destination/"completion.json", progress)
    if errors:
        raise RuntimeError(f"{len(errors)} job exceptions; see error.json")
    return progress


def results(output):
    """(source_instance, method, configuration) -> (instance directory, result)."""
    found = {}
    for root, keep in ((SOURCES["revision"], ("SSN-A", "SSN-E")), (SOURCES["cost"], ("SSN-KKT",)),
                       (Path(output), tuple(c["name"] for c in SETTINGS["configurations"]))):
        for path in sorted((root/"inexact/instances").glob("*/*/result.json")):
            r = read(path)
            if r["method"]["configuration"] in keep:
                found[(r["source_instance"], r["method"]["name"], r["method"]["configuration"])] = (path.parent.parent, r)
    return found


def replay(p, name, configuration_name, harness, N, omega_upper):
    """Exactly N gradient evaluations; return (evaluations, W, endpoint x)."""
    method = int(name == "H-Log")
    if configuration_name in COMPARATORS:
        _, mode, eta_rel, exponent = COMPARATORS[configuration_name]
    else:
        cfg = configuration(configuration_name)
        mode, eta_rel, exponent = cfg["mode"], cfg["eta_rel"], cfg["exponent"]
    tail = (SETTINGS["inner_cap"], SETTINGS["float_floor_factor"], SETTINGS["invariant_tolerance"])
    head = (p.A, p.b, p.power, p.L0, p.L1, p.lam, p.p0, p.z, omega_upper, method, N, 1)
    if harness == "frozen":
        evaluated, W, flag, _, _, _, x = frozen_lean_run(*head, mode == 0, eta_rel, exponent, *tail)
    else:
        evaluated, W, flag, _, _, _, x = solver.lean_run(*head, mode, eta_rel, exponent, *tail)
    if flag != 0:
        raise RuntimeError(f"replay flag {flag}")
    return evaluated, W, x


def timing_process(output, index):
    """One fresh process: every task in a shuffled order, each repeated, all replays checked."""
    output = Path(output)
    tasks = []
    for (source_instance, name, cfg), (directory, r) in sorted(results(output).items()):
        for label, source_cfg, harness in TIMED:
            if source_cfg == cfg and r.get("first_distance_N") is not None:
                tasks.append((label, harness, source_instance, name, cfg, directory, r))
    random.Random(SETTINGS["timing_seed"]+index).shuffle(tasks)
    rows = []
    with threadpool_limits(limits=SETTINGS["blas_threads"]):
        warmup()
        for label, harness, source_instance, name, cfg, directory, r in tasks:
            replay(load_problem(directory), name, cfg, harness, 2, reference_interval(str(directory))["omega_upper"])
        for order, (label, harness, source_instance, name, cfg, directory, r) in enumerate(tasks):
            p = load_problem(directory)
            omega_upper = reference_interval(str(directory))["omega_upper"]
            N = r["first_distance_N"]
            for repeat in range(SETTINGS["timing_repeats"]):
                started = time.perf_counter()
                evaluated, W, x = replay(p, name, cfg, harness, N, omega_upper)
                seconds = time.perf_counter()-started
                if (evaluated != N or W != r["first_distance_W"] or
                        np.linalg.norm(x-p.z)/np.linalg.norm(p.z) > .01+1e-12):
                    raise RuntimeError(f"Timing replay did not reproduce the first hit: {label} {name} {directory}")
                rows.append(dict(process=index, order=order, source_instance=source_instance, method=name,
                                 timed=label, harness=harness, repeat=repeat+1, N=N, W=W, seconds=seconds))
    csv_write(output/f"timing/process_{index}.csv", rows)


def timing(output):
    """Sequential fresh processes; per task, the median over processes of the per-process medians."""
    output = Path(output).resolve()
    processes = SETTINGS["timing_processes"]
    for index in range(processes):
        subprocess.run([sys.executable, "-c", "import sys; from inexact_projection.runner import "
                        f"timing_process; timing_process(sys.argv[1], {index})", str(output)],
                       cwd=EXPERIMENTS, check=True)
        print(f"Timing process {index+1}/{processes} done", flush=True)
    rows = []
    for index in range(processes):
        with (output/f"timing/process_{index}.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows += list(csv.DictReader(stream))
    groups = defaultdict(lambda: defaultdict(list))
    meta = {}
    for r in rows:
        key = (r["source_instance"], r["method"], r["timed"])
        groups[key][int(r["process"])].append(float(r["seconds"]))
        meta[key] = r
    summaries = []
    for key, by_process in sorted(groups.items()):
        if sorted(by_process) != list(range(processes)):
            raise RuntimeError(f"Missing timing process for {key}")
        medians = [float(np.median(by_process[i])) for i in range(processes)]
        first = meta[key]
        summaries.append(dict(source_instance=key[0], method=key[1], timed=key[2], harness=first["harness"],
                              N=int(first["N"]), W=int(first["W"]), median_seconds=float(np.median(medians)),
                              min_process_seconds=min(medians), max_process_seconds=max(medians)))
    csv_write(output/"timing/summary.csv", summaries)
    write_json(output/"timing/protocol.json", dict(
        processes=processes, repeats=SETTINGS["timing_repeats"], blas_threads=SETTINGS["blas_threads"],
        tasks=len(summaries), order=f"process i shuffles all tasks with seed {SETTINGS['timing_seed']}+i",
        aggregate="per task: median over processes of the median over repeats; tables: median over instances",
        harnesses=SETTINGS["harnesses"], excluded="JIT compilation, reference construction, file writing",
        check="every repeat reproduces the recorded N, W and the 1% distance at the endpoint"))
    print(f"Timing: {len(summaries)} tasks x {SETTINGS['timing_repeats']} repeats x {processes} processes", flush=True)
