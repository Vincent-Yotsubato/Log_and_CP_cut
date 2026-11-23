"""Baseline campaign: non-affine selection (panel 1) and CP vs Polyak on cosh (panel 2).

    python -m baseline run --out results/baseline [--profile smoke] [--panels 1 2]
    python -m baseline analyze --out results/baseline
    python -m baseline verify --out results/baseline
"""
import os
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from collections import Counter
from pathlib import Path

from baseline.protocol import build


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("run", "analyze", "verify"))
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--panels", type=int, nargs="+", default=[1, 2], choices=(1, 2))
    parser.add_argument("--out", type=Path, required=True,
                        help="run: a new output directory; analyze/verify: an existing one such as results/baseline")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.stage == "run":
        jobs = build(args.profile, args.panels)
        if args.list:
            print(dict(Counter({p: sum(len(j["methods"]) for j in jobs if j["panel"] == p) for p in args.panels})))
            print(f"{len(jobs)} instances, {sum(len(j['methods']) for j in jobs)} runs")
            return
        from baseline.runner import execute
        execute(jobs, args.out, args.profile, args.workers)
    elif args.stage == "analyze":
        from baseline.analysis import generate
        generate(args.out)
    else:
        from baseline.verification import verify
        if not verify(args.out)["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
