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
from elastic_net.protocol import log_steps
from rates_and_inexact.common import save_trace
from .numerics import run_mng, run_bigsam, run_agm, COLUMNS, STATS, FLAGS
from .protocol import EXPERIMENTS, SOURCE, H_SOURCE, SETTINGS, build, label


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [EXPERIMENTS/"baseline"/n for n in ("numerics.py", "protocol.py", "runner.py")]
    paths += [EXPERIMENTS/"elastic_net"/"protocol.py", EXPERIMENTS/"rates_and_inexact"/"common.py"]
    return {str(p.relative_to(EXPERIMENTS)).replace("\\", "/"): digest(p) for p in paths}


def prepare(output, pilot=False):
    output = Path(output).resolve()
    if SOURCE == output or SOURCE in output.parents:
        raise ValueError("Choose a new output directory; the archive is read-only")
    frozen = dict(settings=SETTINGS, pilot=pilot, jobs=build(pilot), hashes=source_hashes(),
                  source_config=digest(SOURCE/"config.json"), h_source=digest(H_SOURCE),
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


def constants(A, b, power):
    """Data-only constants of protocol.py (no reference point is used)."""
    singular = np.linalg.svd(A, compute_uv=False)
    rho = float(np.linalg.norm(np.linalg.pinv(A) @ b))
    row_max = float(np.max(np.linalg.norm(A, axis=1)))

    def hessian_bound(radius):
        return float((power-1)*singular[0]**2*(np.max(np.abs(b))+row_max*radius)**(power-2))

    return dict(rho=rho, L_ball=hessian_bound(2*rho), hessian_bound=hessian_bound,
                norm_A=float(singular[0]), sigma_min=float(singular[-1]))


def load(directory):
    a = np.load(directory/"instance.npz")
    r = read(directory/"reference.json")
    return dict(A=np.ascontiguousarray(a["A"]), b=a["b"].copy(), p0=a["p0"].copy(), z=a["z"].copy(),
                L0=float(r["L0"]), L1=float(r["L1"]), lam=float(r["lambda_en"]), power=int(r["metadata"]["s"]),
                distance_bound=float(r["certificate"]["distance_bound"]))


def agm_gamma(spec, c, power, budget):
    if spec["rule"] == "thm41":
        return 1.
    L_g = c["hessian_bound"](spec["radius"]*c["rho"])
    exponent = (2*power-2)/(2*power-1)
    return 1./((2*L_g/SETTINGS["upper_smoothness"])*budget**exponent+2)


def solve(p, spec, budget):
    if np.any(p["p0"] != 0.) or p["lam"] != 0.:
        raise ValueError("The baselines are implemented for the Euclidean kernel with p0 = 0")
    c = constants(p["A"], p["b"], p["power"])
    steps = log_steps(budget-1)
    started = time.perf_counter()
    parameters = dict(rho=c["rho"], L_ball=c["L_ball"])
    if spec["name"] == "MNG-local":
        rows, x, stats = run_mng(p["A"], p["b"], p["power"], c["L_ball"], p["z"], budget, steps)
    elif spec["name"] == "BiG-SAM":
        rule = 0 if spec["step"] == "ball" else 1
        rows, x, stats = run_bigsam(p["A"], p["b"], p["power"], p["z"], budget, steps, rule, 1./c["L_ball"],
                                    p["L0"], p["L1"], float(spec["gamma"]), 2*c["rho"])
    elif spec["name"] == "AGM-BiO":
        gamma = agm_gamma(spec, c, p["power"], budget)
        parameters.update(gamma=gamma, radius=spec["radius"]*c["rho"],
                          L_Z=c["hessian_bound"](spec["radius"]*c["rho"]))
        rows, x, stats = run_agm(p["A"], p["b"], p["power"], p["z"], budget, steps, gamma,
                                 SETTINGS["upper_smoothness"], spec["radius"]*c["rho"])
    else:
        raise ValueError(spec["name"])
    elapsed = time.perf_counter()-started
    result = {k: float(v) for k, v in zip(STATS, stats)}
    for k in ("K", "N", "flag", "diagnostic_count"):
        result[k] = int(result[k])
    for k in ("first_distance_N", "first_joint_N"):
        result[k] = int(result[k]) if result[k] >= 0 else None
    result.update(method=spec, label=label(spec), status=FLAGS[result["flag"]], elapsed_seconds=elapsed,
                  parameters=parameters, L0=p["L0"], L1=p["L1"], power=p["power"],
                  distance_success=result["first_distance_N"] is not None,
                  joint_success=result["first_joint_N"] is not None)
    return result, rows, x


def worker(job, output):
    directory = Path(output)/"instances"/job["id"]
    try:
        with threadpool_limits(limits=1):
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory/"job.json", job)
            source = SOURCE/"instances"/job["source_instance"]
            for name in ("instance.npz", "reference.json"):
                if (directory/name).exists() and digest(directory/name) != digest(source/name):
                    raise RuntimeError("Saved input differs from the archive: "+name)
                if not (directory/name).exists():
                    shutil.copy2(source/name, directory/name)
            p = load(directory)
            statuses = []
            for spec in job["methods"]:
                rd = directory/identity(spec)
                if (rd/"result.json").exists():
                    statuses.append(read(rd/"result.json")["status"])
                    continue
                rd.mkdir(exist_ok=True)
                result, rows, x = solve(p, spec, job["gradient_budget"])
                result.update(instance_id=job["id"], source_instance=job["source_instance"],
                              model=job["model"], pilot=job["pilot"])
                save_trace(rd/"trace.csv.gz", COLUMNS, rows)
                np.savez_compressed(rd/"state.npz", x=x)
                write_json(rd/"result.json", result)
                statuses.append(result["status"])
            return dict(id=job["id"], runs=len(statuses), statuses={s: statuses.count(s) for s in set(statuses)})
    except Exception:
        error = traceback.format_exc()
        write_json(directory/"error.json", dict(error=error))
        return dict(id=job["id"], runs=0, error=error)


def warmup():
    A, b, z = np.array([[1., 0.], [1., 1.]]), np.array([1., 3.]), np.array([1.5, 1.5])
    steps = np.arange(3, dtype=np.int64)
    run_mng(A, b, 4, 12., z, 3, steps)
    run_bigsam(A, b, 4, z, 3, steps, 0, .1, 12., 6., 1., 10.)
    run_agm(A, b, 4, z, 3, steps, 1., 1., 10.)


def execute(output, pilot=False, workers=4):
    jobs = build(pilot)
    warmup()
    start, done, runs, errors = time.perf_counter(), 0, 0, []
    progress = {}
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
            print(f"baselines: {done}/{len(jobs)} jobs, {runs} runs, {answer.get('statuses', 'ERROR')}", flush=True)
    write_json(Path(output)/"completion.json", progress)
    if errors:
        raise RuntimeError(f"{len(errors)} job exceptions; see error.json")
    return progress
