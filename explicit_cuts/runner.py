"""Frozen configuration, read-only archived inputs, resumable per-method results."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import platform
import shutil
import time
import traceback
from pathlib import Path

import numba
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from baseline.protocol import identity
from baseline.runner import write_json
from elastic_net.numerics import run, METHODS, FLAGS, COLUMNS
from elastic_net.protocol import log_steps
from elastic_net.problems import Problem
from rates_and_inexact.common import save_trace
from .protocol import EXPERIMENTS, SOURCE, SETTINGS, build

KEYS = ["K", "N", "M", "W", "IP", "flag", "delta0", "delta", "distance", "relative_distance", "feasibility",
        "max_cut_ratio", "max_Q_violation", "max_Omega_drop", "max_Omega_excess", "max_projection_residual",
        "total_inner", "max_inner", "complementarity"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [EXPERIMENTS/"elastic_net"/n for n in ("numerics.py", "problems.py", "protocol.py")]
    paths += [EXPERIMENTS/"baseline"/n for n in ("protocol.py", "runner.py")]
    return {str(p.relative_to(EXPERIMENTS)).replace("\\", "/"): digest(p) for p in paths}


def prepare(output, pilot=False):
    output = Path(output).resolve()
    if SOURCE == output or SOURCE in output.parents:
        raise ValueError("Choose a new output directory; the archive is read-only")
    frozen = dict(settings=SETTINGS, pilot=pilot, jobs=build(pilot), hashes=source_hashes(),
                  source_config=digest(SOURCE/"config.json"),
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


def load(directory, model):
    a = np.load(directory/"instance.npz")
    r = read(directory/"reference.json")
    return Problem(a["A"], a["b"], 0, model.get("beta", 0.), 4, r["L0"], r["L1"], r["lambda_en"],
                   a["p0"], a["z"], a["reference_dual"], r["certificate"], r["metadata"])


def solve(p, spec, budget, stop_mode):
    targets = np.array(SETTINGS["distance_targets"])
    started = time.perf_counter()
    rows, dual, x, witness, hits, stats, _, distance_hits = run(
        p.A, p.b, 0, p.beta, 4, p.L0, p.L1, p.lam, p.p0, p.z, METHODS[spec["name"]],
        float(spec["theta"]), float(spec["c"]), budget-1, stop_mode, log_steps(budget-1),
        np.empty(0), targets, False)
    elapsed = time.perf_counter()-started
    result = {k: float(v) for k, v in zip(KEYS, stats)}
    for k in ("K", "N", "M", "W", "IP", "flag", "total_inner", "max_inner"):
        result[k] = int(result[k])
    norm_z = float(np.linalg.norm(p.z))
    result.update(method=spec, status=FLAGS[result["flag"]], elapsed_seconds=elapsed,
                  objective_hits_relative=[int(v) if v >= 0 else None for v in hits],
                  distance_targets=targets.tolist(),
                  distance_hits=[int(v) if v >= 0 else None for v in distance_hits],
                  selection_success=bool(result["feasibility"] <= 1e-8 and result["distance"]/(1+norm_z) <= 1e-6),
                  L0=float(p.L0), L1=float(p.L1), lambda_en=float(p.lam),
                  initial_r=float(rows[0][COLUMNS.index("r")]))
    result["joint_N"] = result["N"] if result["selection_success"] else None
    return result, rows, dual, x, witness


def worker(job, output):
    directory = Path(output)/"instances"/job["id"]
    try:
        with threadpool_limits(limits=1):
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory/"job.json", job)
            if job["source_instance"] is not None:
                source = SOURCE/"instances"/job["source_instance"]
                for name in ("instance.npz", "reference.json"):
                    if (directory/name).exists() and digest(directory/name) != digest(source/name):
                        raise RuntimeError("Saved input differs from the archive: "+name)
                    if not (directory/name).exists():
                        shutil.copy2(source/name, directory/name)
            elif not (directory/"instance.npz").exists():
                # New kappa=10 instance: the archived generator with its 60-digit certificate.
                from elastic_net.problems import make_e1
                g = make_e1(**{k: v for k, v in job["model"].items() if k != "kind"})
                np.savez_compressed(directory/"instance.npz", A=g.A, b=g.b, p0=g.p0, z=g.z, reference_dual=g.dual)
                write_json(directory/"reference.json", dict(certificate=g.certificate, metadata=g.metadata,
                                                            L0=g.L0, L1=g.L1, lambda_en=g.lam))
            p = load(directory, job["model"])
            statuses = []
            for spec in job["methods"]:
                rd = directory/identity(spec)
                if (rd/"result.json").exists():
                    statuses.append(read(rd/"result.json")["status"])
                    continue
                rd.mkdir(exist_ok=True)
                result, rows, dual, x, witness = solve(p, spec, job["gradient_budget"], job["stop_mode"])
                result.update(instance_id=job["id"], source_instance=job["source_instance"],
                              variant=job["variant"], model=job["model"], pilot=job["pilot"])
                save_trace(rd/"trace.csv.gz", COLUMNS, rows)
                np.savez_compressed(rd/"state.npz", p=dual, x=x)
                write_json(rd/"result.json", result)
                statuses.append(result["status"])
            return dict(id=job["id"], runs=len(statuses), statuses={s: statuses.count(s) for s in set(statuses)})
    except Exception:
        error = traceback.format_exc()
        write_json(directory/"error.json", dict(error=error))
        return dict(id=job["id"], runs=0, error=error)


def warmup():
    A, b = np.eye(2), np.ones(2)
    run(A, b, 0, 1., 4, 1., 1., .1, np.zeros(2), b, 3, 1., 1., 2, 2, np.arange(3, dtype=np.int64),
        np.empty(0), np.array([.01]), False)


def execute(output, pilot=False, workers=4):
    jobs = build(pilot)
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
            progress = dict(completed_jobs=done, expected_jobs=len(jobs), completed_runs=runs,
                            expected_runs=sum(len(j["methods"]) for j in jobs), errors=errors,
                            elapsed_seconds=time.perf_counter()-start, last=answer)
            write_json(Path(output)/"progress.json", progress)
            print(f"cuts: {done}/{len(jobs)} jobs, {runs} runs, {answer.get('statuses', 'ERROR')}", flush=True)
    write_json(Path(output)/"completion.json", progress)
    if errors:
        raise RuntimeError(f"{len(errors)} job exceptions; see error.json")
    return progress
