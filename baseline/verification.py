"""Independently verify saved data; do not rerun the numerical algorithms."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import mpmath as mp
from threadpoolctl import threadpool_limits

from .analysis import read
from .numerics import COLUMNS
from .problems import soft, value_gradient
from .protocol import identity
from .runner import write_json, hashes


def verify(output):
    output = Path(output).resolve()
    config = read(output / "config.json")
    failures, certificates, terminal_checks, counters = [], [], [], Counter()
    def check(condition, message):
        if not condition:
            failures.append(message)
    check(config["hashes"] == hashes(), "Computational source hashes changed")
    check(not list(output.glob("instances/**/error.json")), "Exception records exist")
    jobs = config["jobs"]
    expected = sum(len(job["methods"]) for job in jobs)
    check(len(list(output.glob("instances/*/*/result.json"))) == expected, "Missing method results")
    with threadpool_limits(limits=1), mp.workdps(65):
        for index, job in enumerate(jobs):
            directory = output / "instances" / job["id"]
            if not (directory / "reference.json").exists():
                failures.append(f"{job['id']}: missing reference")
                continue
            reference = read(directory / "reference.json")
            data = np.load(directory / "instance.npz")
            A, b, p0, z, y = [data[key] for key in ("A", "b", "p0", "z", "reference_dual")]
            cert, lam = reference["certificate"], reference["lambda_en"]
            inequality = job["model"]["kind"] == "poly"
            # Evaluate the gap bound directly at the saved high-precision feasible
            # auxiliary point, independently of the producer's decomposition.
            Am, bm, p0m, ym = [mp.matrix(v.tolist()) for v in (A, b, p0, y)]
            um = mp.matrix([mp.mpf(v) for v in cert["auxiliary_feasible_point"]])
            lm = mp.mpf(float(lam))
            pm = p0m + Am.T * ym
            mapped = mp.matrix([mp.sign(t) * max(abs(t) - lm, 0) for t in pm])
            slack = Am * um - bm
            gap = (um.T * um)[0] / 2 + lm * mp.norm(um, 1) - (p0m.T * um)[0] + (mapped.T * mapped)[0] / 2 - (ym.T * bm)[0]
            feasibility = max(max(-t, 0) if inequality else abs(t) for t in slack)
            check(float(feasibility) < 1e-35, f"{job['id']}: high-precision point not feasible")
            check(float(gap) >= -1e-35 and float(gap) < 1e-10, f"{job['id']}: direct reference gap")
            check(abs(float(gap) - cert["raw_gap"]) < 1e-30 + 1e-10 * abs(float(gap)), f"{job['id']}: reference gap mismatch")
            check(cert["gap"] < 1e-10, f"{job['id']}: reported certificate gap")
            if inequality:
                check(np.all(y >= 0), f"{job['id']}: reference dual is negative")
            certificates.append({"instance_id": job["id"], "independent_gap": float(gap),
                                 "auxiliary_feasibility": float(feasibility), "gap_upper": cert["gap"]})
            family, beta, power = (1, 0., job["model"].get("power", 4)) if inequality else (0, job["model"].get("beta", 1.), 4)
            initial_f, initial_g, _ = value_gradient(A, b, soft(p0, lam), family, beta, power)
            target = reference["metadata"].get("target_r")
            if target is not None:
                error = abs(reference["L1"] * np.linalg.norm(initial_g) / reference["L0"] - target) / target
                check(error < 1e-10, f"{job['id']}: r0 calibration")
            for method in job["methods"]:
                run_dir = directory / identity(method)
                if not (run_dir / "result.json").exists():
                    failures.append(f"{job['id']}/{method['label']}: no result")
                    continue
                r = read(run_dir / "result.json")
                state = np.load(run_dir / "state.npz")
                x, p, witness = [state[key] for key in ("x", "p", "witness")]
                f, g, tau = value_gradient(A, b, x, family, beta, power)
                prefix = f"{job['id']}/{method['label']}"
                check(r["flag"] < 5, prefix + ": numerical failure")
                check(np.max(np.abs(x - soft(p, lam))) <= 1e-12 * (1 + np.linalg.norm(x)), prefix + ": lost dual iterate")
                check(np.isclose(f, r["delta"], rtol=2e-8, atol=1e-26), prefix + ": final objective mismatch")
                distance = np.linalg.norm(x - z)
                check(np.isclose(distance, r["distance"], rtol=1e-10, atol=1e-13), prefix + ": final distance mismatch")
                check(r["N"] == r["K"] + 1 and r["M"] == 2 * r["N"], prefix + ": gradient/matrix counter mismatch")
                check(r["W"] >= r["K"] + 1, prefix + ": missing kernel calls")
                with gzip.open(run_dir / "trace.csv.gz", "rt", encoding="utf-8") as stream:
                    t = np.loadtxt(stream, delimiter=",", skiprows=1, ndmin=2)
                check(int(t[-1, 0]) == r["K"], prefix + ": missing terminal trace")
                check(np.all(np.diff(t[:, 0]) > 0), prefix + ": unordered trace")
                check(np.all(np.isfinite(t)), prefix + ": nonfinite trace")
                check(np.all(t[:, 8] <= (1 + 1e-10) * t[:, 9] + 1e-300), prefix + ": inadmissible recorded cut")
                check(r["max_cut_ratio"] <= 1 + 1e-10, prefix + ": inadmissible between saved steps")
                if method["name"].startswith("H-"):
                    check(r["max_Omega_drop"] < 1e-10 * (1 + np.linalg.norm(z)**2), prefix + ": Omega decreases")
                    check(r["max_Q_violation"] < 1e-10 * (1 + np.linalg.norm(z)**2), prefix + ": Q excludes feasible set")
                    normal = p0 - p
                    check(np.linalg.norm(normal + A.T @ witness) < 1e-9 * (1 + np.linalg.norm(normal)), prefix + ": normal representation")
                    check(np.min(witness) >= -1e-13, prefix + ": negative normal weights")
                    check(normal @ (z - x) < 1e-9 * (1 + np.linalg.norm(z)**2), prefix + ": Q excludes reference")
                for j, N in enumerate(r["hits"], 1):
                    if N is not None:
                        check(1 <= N <= r["N"], prefix + ": threshold hit outside actual budget")
                        # Log samples cannot prove an earlier exact hit, but no
                        # recorded qualifying sample may precede the saved hit.
                        reached = t[:, 15] <= initial_f * 10.**(-j)
                        if reached.any():
                            check(N <= int(t[reached, 1][0]), prefix + ": late first-hit record")
                counters[r["status"]] += 1
                terminal_checks.append({"instance_id": job["id"], "method": method["label"],
                                        "independent_delta": f, "independent_distance": distance})
            if (index + 1) % 50 == 0:
                print(f"Verified {index + 1}/{len(jobs)} instances", flush=True)
    result = {"passed": not failures, "expected_runs": expected, "verified_runs": len(terminal_checks),
              "verified_references": len(certificates), "failures": failures, "statuses": dict(counters),
              "max_reference_gap": max(r["gap_upper"] for r in certificates),
              "max_independent_gap": max(r["independent_gap"] for r in certificates),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(output / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    result = verify(args.output)
    raise SystemExit(0 if result["passed"] else 1)
