"""Frozen configuration, SSN-KKT runs through the rates_and_inexact worker, and serial timing."""
from __future__ import annotations

import concurrent.futures
import csv
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numba
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from elastic_net.numerics import run as historical_run, METHODS as HISTORICAL
from rates_and_inexact import inexact
from rates_and_inexact import runner as base_runner
from rates_and_inexact.common import read, write_json, digest, csv_write, load_problem, reference_interval
from .lean import lean_run
from .protocol import EXPERIMENTS, SOURCE, SETTINGS, TIMED, build, registered


def source_hashes():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += sorted((EXPERIMENTS/"rates_and_inexact").glob("*.py"))
    paths += [EXPERIMENTS/"elastic_net"/n for n in ("numerics.py", "problems.py", "runner.py")]
    paths += [EXPERIMENTS/"baseline"/n for n in ("protocol.py", "runner.py")] + [EXPERIMENTS/"cut_functions.py"]
    return {str(p.relative_to(EXPERIMENTS)).replace("\\", "/"): digest(p) for p in paths}


def archived_results():
    """Digests of the archived inexact-panel results whose first hits are replayed."""
    paths = sorted((SOURCE/"inexact/instances").glob("*/*/result.json"))
    return {str(p.relative_to(SOURCE)).replace("\\", "/"): digest(p) for p in paths}


def prepare(output):
    output = Path(output).resolve()
    if SOURCE == output or SOURCE in output.parents:
        raise ValueError("Choose a new output directory; results/rates_and_inexact is read-only")
    frozen = dict(settings=SETTINGS, jobs=build(), hashes=source_hashes(),
                  source_config=digest(SOURCE/"config.json"), archived_results=archived_results(),
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


def worker(job, output):
    with registered():
        return base_runner.worker(job, output)


def execute(output, workers=4):
    jobs = build()
    destination = Path(output)/"inexact"
    destination.mkdir(parents=True, exist_ok=True)
    base_runner.warmup()
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
                            expected_runs=2*len(jobs), errors=errors,
                            elapsed_seconds=time.perf_counter()-started, last=answer)
            write_json(destination/"progress.json", progress)
            print(f"SSN-KKT: {done}/{len(jobs)} jobs, {runs} runs, {answer.get('statuses', 'ERROR')}", flush=True)
    write_json(destination/"completion.json", progress)
    if errors:
        raise RuntimeError(f"{len(errors)} job exceptions; see error.json")
    return progress


def results(output):
    """(source_instance, method, configuration) -> (instance directory, result)."""
    found = {}
    for root, keep in ((SOURCE, ("SSN-A", "SSN-E")), (Path(output), ("SSN-KKT",))):
        for path in sorted((root/"inexact/instances").glob("*/*/result.json")):
            r = read(path)
            if r["method"]["configuration"] in keep:
                found[(r["source_instance"], r["method"]["name"], r["method"]["configuration"])] = (path.parent.parent, r)
    return found


def replay(p, name, configuration, harness, N, omega_upper):
    """Run exactly N gradient evaluations; return (evaluations, W, endpoint x, extra)."""
    method = int(name == "H-Log")
    if harness == "historical":
        ans = historical_run(p.A, p.b, 1, 0., p.power, p.L0, p.L1, p.lam, p.p0, p.z, HISTORICAL[name],
                             1., 1., N-1, 0, np.unique(np.array([0, N-1], dtype=np.int64)),
                             np.empty(0), np.array([.01]), False)
        return int(ans[5][1]), int(ans[5][3]), ans[2], {}
    exact = configuration == "SSN-KKT"
    eta_rel, exponent = (0., 0.) if exact else (1e-2, 2.)
    args = (p.A, p.b, p.power, p.L0, p.L1, p.lam, p.p0, p.z, omega_upper, method, N, 1, exact,
            eta_rel, exponent, SETTINGS["inner_cap"], SETTINGS["float_floor_factor"],
            SETTINGS["invariant_tolerance"])
    if harness == "lean":
        evaluated, W, flag, _, _, _, x = lean_run(*args)
        if flag != 0:
            raise RuntimeError(f"lean replay flag {flag}")
        return evaluated, W, x, {}
    ans = inexact.run(*args)
    audit = ans[1]
    accepted = audit[:, 21] == 1.
    extra = dict(accepted_steps=int(np.sum(accepted)),
                 accepted_at_kkt_tolerance=int(np.sum(accepted & (audit[:, 15] <= SETTINGS["kkt_tolerance"]))),
                 start_accepted=int(np.sum(accepted & (audit[:, 19] == 1.))))
    return int(ans[-1][1]), int(ans[-1][3]), ans[3], extra


def warmup():
    A, b, p0, z = np.eye(2), np.ones(2), np.zeros(2), np.ones(2)
    for exact in (False, True):
        lean_run(A, b, 3, 1., 1., .1, p0, z, 2., 0, 2, 1, exact, .01, 2., 5, 1000., 1e-12)
        inexact.run(A, b, 3, 1., 1., .1, p0, z, 2., 0, 2, 1, exact, .01, 2., 5, 1000., 1e-12)
    base_runner.warmup()


def timing_process(output, index):
    """One fresh process: every task in a shuffled order, each repeated, all replays checked."""
    output = Path(output)
    found = results(output)
    tasks = []
    for (source_instance, name, configuration), (directory, r) in sorted(found.items()):
        for label, source_cfg, harness in TIMED:
            if source_cfg == configuration and r.get("first_distance_N") is not None:
                tasks.append((label, harness, source_instance, name, configuration, directory, r))
    random.Random(SETTINGS["timing_seed"]+index).shuffle(tasks)
    rows = []
    with threadpool_limits(limits=SETTINGS["blas_threads"]):
        warmup()
        # Compile every specialization with the argument types of the real inputs.
        for label, harness, source_instance, name, configuration, directory, r in tasks:
            p = load_problem(directory)
            replay(p, name, configuration, harness, 2, reference_interval(str(directory))["omega_upper"])
        for order, (label, harness, source_instance, name, configuration, directory, r) in enumerate(tasks):
            p = load_problem(directory)
            omega_upper = reference_interval(str(directory))["omega_upper"]
            N = r["first_distance_N"]
            for repeat in range(SETTINGS["timing_repeats"]):
                started = time.perf_counter()
                evaluated, W, x, extra = replay(p, name, configuration, harness, N, omega_upper)
                seconds = time.perf_counter()-started
                if (evaluated != N or W != r["first_distance_W"] or
                        np.linalg.norm(x-p.z)/np.linalg.norm(p.z) > .01+1e-12):
                    raise RuntimeError(f"Timing replay did not reproduce the first hit: {label} {name} {directory}")
                rows.append(dict(process=index, order=order, source_instance=source_instance, method=name,
                                 timed=label, harness=harness, repeat=repeat+1, N=N, W=W, seconds=seconds, **extra))
    csv_write(output/f"timing/process_{index}.csv", rows)


def timing(output):
    """Sequential fresh processes; per task, the median over processes of the per-process medians."""
    output = Path(output).resolve()
    processes = SETTINGS["timing_processes"]
    for index in range(processes):
        subprocess.run([sys.executable, "-c", "import sys; from exact_projection.runner import "
                        f"timing_process; timing_process(sys.argv[1], {index})", str(output)],
                       cwd=EXPERIMENTS, check=True)
        print(f"Timing process {index+1}/{processes} done", flush=True)
    rows = []
    for index in range(processes):
        with (output/f"timing/process_{index}.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows += list(csv.DictReader(stream))
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["source_instance"], r["method"], r["timed"])][int(r["process"])].append(float(r["seconds"]))
    summaries, per_process = [], defaultdict(list)
    for (source_instance, name, label), by_process in sorted(groups.items()):
        if sorted(by_process) != list(range(processes)):
            raise RuntimeError(f"Missing timing process for {label} {name} {source_instance}")
        medians = [float(np.median(by_process[i])) for i in range(processes)]
        first = next(r for r in rows if (r["source_instance"], r["method"], r["timed"]) == (source_instance, name, label))
        extra = {k: int(first[k]) for k in ("accepted_steps", "accepted_at_kkt_tolerance", "start_accepted") if first.get(k)}
        summaries.append(dict(source_instance=source_instance, method=name, timed=label, harness=first["harness"],
                              N=int(first["N"]), W=int(first["W"]), median_seconds=float(np.median(medians)),
                              min_process_seconds=min(medians), max_process_seconds=max(medians), **extra))
        for i, value in enumerate(medians):
            per_process[(name, label, i)].append(value)
    csv_write(output/"timing/summary.csv", summaries)
    csv_write(output/"timing/processes.csv", [dict(method=n, timed=l, process=i, instances=len(v),
                                                    median_ms=1000*float(np.median(v)))
                                               for (n, l, i), v in sorted(per_process.items())])
    write_json(output/"timing/protocol.json", dict(
        processes=processes, repeats=SETTINGS["timing_repeats"], blas_threads=SETTINGS["blas_threads"],
        tasks=len(summaries), order=f"process i shuffles all tasks with seed {SETTINGS['timing_seed']}+i",
        aggregate="per task: median over processes of the median over repeats; tables: median over instances",
        harnesses=SETTINGS["harnesses"], excluded="JIT compilation, reference construction, file writing",
        check="every repeat reproduces the recorded N, W and the 1% distance at the endpoint"))
    print(f"Timing: {len(summaries)} tasks x {SETTINGS['timing_repeats']} repeats x {processes} processes", flush=True)
