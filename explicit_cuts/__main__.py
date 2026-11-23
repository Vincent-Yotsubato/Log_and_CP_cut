"""Cut comparison in the explicit update on the archived cosh inverse problems."""
import os
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "run", "verify", "analyze"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]/"results/explicit_cuts")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot", action="store_true", help="seed 0 and 500 gradient evaluations only")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    from explicit_cuts.protocol import build
    if args.list:
        jobs = build(args.pilot)
        print(f"{len(jobs)} instances, {sum(len(j['methods']) for j in jobs)} runs")
        return
    if args.stage in ("all", "run"):
        # Only computation is tied to the frozen source; verify/analyze read the results.
        from explicit_cuts.runner import prepare, execute
        prepare(args.out, args.pilot)
        execute(args.out, args.pilot, args.workers)
    if args.stage in ("all", "verify"):
        from explicit_cuts.verification import verify
        result = verify(args.out)
        if not result["integrity_passed"]:
            raise SystemExit(2)
    if args.stage in ("all", "analyze"):
        from explicit_cuts.analysis import analyze
        analyze(args.out)


if __name__ == "__main__":
    main()
