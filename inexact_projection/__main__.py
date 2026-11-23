"""The residual acceptance test of the paper on the 40 elastic-net instances."""
import os
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "run", "timing", "verify", "analyze"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]/"results/inexact_projection")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    from inexact_projection.protocol import build
    if args.list:
        jobs = build()
        print(f"{len(jobs)} instances, {sum(len(j['methods']) for j in jobs)} runs")
        return
    from inexact_projection.runner import prepare, execute, timing
    if args.stage in ("all", "run", "timing"):
        prepare(args.out)  # freezes or checks the configuration; read-only stages skip it
    if args.stage in ("all", "run"):
        execute(args.out, args.workers)
    if args.stage in ("all", "timing"):
        timing(args.out)
    if args.stage in ("all", "verify"):
        from inexact_projection.verification import verify
        result = verify(args.out)
        if not result["integrity_passed"]:
            raise SystemExit(2)
    if args.stage in ("all", "analyze"):
        from inexact_projection.analysis import analyze
        analyze(args.out)


if __name__ == "__main__":
    main()
