from __future__ import annotations

import concurrent.futures
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
import math

import numpy as np
import scipy
import numba
from threadpoolctl import threadpool_limits

from .numerics import COLUMNS, FLAGS, METHODS, run
from .problems import make_e1, make_poly
from .protocol import identity, log_steps


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    def finite(data):
        if isinstance(data, float) and not math.isfinite(data):
            return None
        if isinstance(data, dict):
            return {k: finite(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [finite(v) for v in data]
        return data
    temporary.write_text(json.dumps(finite(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def hashes():
    root = Path(__file__).resolve().parent
    paths = [root / name for name in ("numerics.py", "problems.py", "protocol.py", "runner.py", "__main__.py")]
    return {p.relative_to(root.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def create_problem(job):
    arguments = dict(job["model"])
    kind = arguments.pop("kind")
    return make_e1(**arguments) if kind == "e1" else make_poly(**arguments)


def solve(problem, method, job):
    started = time.perf_counter()
    trace, p, x, witness, hits, stats = run(
        problem.A, problem.b, problem.family, problem.beta, problem.power,
        problem.L0, problem.L1, problem.lam, problem.p0, problem.z, METHODS[method["name"]],
        float(method["theta"]), float(method["c"]), job["max_updates"], job["stop_mode"],
        log_steps(job["max_updates"], job["dense"]))
    elapsed = time.perf_counter() - started
    objective_success = bool(stats[7] <= stats[6] * 1e-9)
    selection_success = bool(stats[10] <= 1e-8 and stats[8] / (1 + np.linalg.norm(problem.z)) <= 1e-6)
    result = {"method": method, "status": FLAGS[int(stats[5])], "flag": int(stats[5]),
              "K": int(stats[0]), "N": int(stats[1]), "M": int(stats[2]),
              "W": int(stats[3]), "IP": int(stats[4]), "elapsed_seconds": elapsed,
              "delta0": float(stats[6]), "delta": float(stats[7]),
              "distance": float(stats[8]), "relative_distance": float(stats[9]),
              "feasibility": float(stats[10]), "max_cut_ratio": float(stats[11]),
              "max_Q_violation": float(stats[12]), "max_Omega_drop": float(stats[13]),
              "max_Omega_excess": float(stats[14]), "max_projection_residual": float(stats[15]),
              "total_inner": int(stats[16]), "max_inner": int(stats[17]),
              "complementarity": float(stats[18]), "hits": [int(x) if x >= 0 else None for x in hits],
              "objective_success": objective_success, "selection_success": selection_success,
              "success": objective_success if job["stop_mode"] == 1 else selection_success,
              "timing_scope": "compiled trajectory including invariant diagnostics; excludes JIT, reference and disk I/O"}
    return result, trace, p, x, witness


def warmup():
    # Compile the complete signature once before measuring or launching workers.
    problem = make_poly(example=True)
    solve(problem, {"name": "CP", "theta": 1., "c": 1.},
          {"max_updates": 2, "stop_mode": 0, "dense": True})


def worker(job, output):
    with threadpool_limits(limits=1):
        directory = Path(output) / "instances" / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "job.json", job)
        warmup()
        try:
            problem = create_problem(job)
            np.savez_compressed(directory / "instance.npz", A=problem.A, b=problem.b,
                                p0=problem.p0, z=problem.z, reference_dual=problem.dual)
            write_json(directory / "reference.json", {"certificate": problem.certificate,
                       "metadata": problem.metadata, "L0": problem.L0, "L1": problem.L1,
                       "lambda_en": problem.lam})
        except Exception:
            error = {"stage": "reference_or_initialization", "traceback": traceback.format_exc()}
            write_json(directory / "error.json", error)
            return {"id": job["id"], "error": error, "runs": 0}
        summaries = []
        for method in job["methods"]:
            run_id = identity(method)
            run_dir = directory / run_id
            run_dir.mkdir(exist_ok=True)
            if (run_dir / "result.json").exists():
                summaries.append(json.loads((run_dir / "result.json").read_text(encoding="utf-8")))
                continue
            try:
                result, trace, p, x, witness = solve(problem, method, job)
                result.update({"instance_id": job["id"], "run_id": run_id, "panel": job["panel"],
                               "variant": job["variant"], "metadata": problem.metadata})
                np.savez_compressed(run_dir / "state.npz", p=p, x=x, witness=witness)
                with gzip.open(run_dir / "trace.csv.gz", "wt", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(COLUMNS)
                    writer.writerows(trace)
                write_json(run_dir / "result.json", result)
                summaries.append(result)
            except Exception:
                error = {"stage": "solve_or_write", "method": method, "traceback": traceback.format_exc()}
                write_json(run_dir / "error.json", error)
                summaries.append({"status": "exception", "error": error})
        return {"id": job["id"], "runs": len(summaries),
                "statuses": {s: sum(r["status"] == s for r in summaries) for s in sorted({r["status"] for r in summaries})}}


def execute(jobs, output, profile, workers=4):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = {"profile": profile, "jobs": jobs, "hashes": hashes(),
              "python": platform.python_version(), "numpy": np.__version__,
              "scipy": scipy.__version__, "numba": numba.__version__,
              "blas_threads": 1, "workers": workers,
              "notes": ["Experiment 2 uses a=0.01 for the entire calibrated grid.",
                        "rho_c=0 Gaussian rows are not exactly orthogonal; a separate control is included.",
                        "Full 1a/1b budget: 100000 gradient evaluations, including k=0.",
                        "Experiment 4 budget: 20000 accepted updates plus terminal evaluation.",
                        "No optional E2 latent-exponential or invalid-constant runs."]}
    config = output / "config.json"
    if config.exists():
        previous = json.loads(config.read_text(encoding="utf-8"))
        if previous != frozen:
            raise RuntimeError("Existing output uses different code/configuration; choose a new output directory.")
    else:
        write_json(config, frozen)
    warmup()
    start = time.perf_counter()
    completed, runs = 0, 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, job, str(output)): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            answer = future.result()
            completed += 1
            runs += answer["runs"]
            with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(answer, ensure_ascii=False) + "\n")
            status = {"completed_instances": completed, "total_instances": len(jobs),
                      "completed_runs": runs, "expected_runs": sum(len(j["methods"]) for j in jobs),
                      "elapsed_seconds": time.perf_counter() - start, "last": answer}
            write_json(output / "progress.json", status)
            print(f"{completed}/{len(jobs)} instances; {runs} runs; {status['elapsed_seconds']:.1f}s; {answer.get('statuses', answer.get('error'))}", flush=True)
    write_json(output / "completion.json", status)
    return output
