"""Inexact two-halfspace projections (panel "inexact") and the rate analysis of the exact H runs.

    python -m rates_and_inexact all --out results/rates_and_inexact  # inexact, rates, verify, analyze
    python -m rates_and_inexact inexact --out results/pilot --pilot
    python -m rates_and_inexact rates   --out results/rates_and_inexact
    python -m rates_and_inexact verify  --out results/rates_and_inexact
    python -m rates_and_inexact analyze --out results/rates_and_inexact

Only "inexact" and "all" compute new runs; they freeze the configuration in a new output directory.
"""
import os
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("all", "inexact", "rates", "verify", "analyze"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]/"results/rates_and_inexact")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot", action="store_true", help="Separate seed-0 / 500-gradient validation only")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    from rates_and_inexact.protocol import build
    if args.list:
        jobs = build("inexact", args.pilot)
        print(f"inexact: {len(jobs)} instances, {sum(len(j['methods']) for j in jobs)} runs")
        return
    if args.stage in ("all", "inexact"):
        from rates_and_inexact.runner import prepare, execute
        prepare(args.out, args.pilot)
        execute("inexact", args.out, args.pilot, args.workers)
    if args.stage in ("all", "rates"):
        from rates_and_inexact.rates import analyze
        analyze(args.out)
    if args.stage in ("all", "verify"):
        from rates_and_inexact.verification import verify
        result = verify(args.out)
        if not result["integrity_passed"]:
            raise SystemExit(2)
    if args.stage in ("all", "analyze"):
        from rates_and_inexact.analysis import analyze
        analyze(args.out)


if __name__ == "__main__":
    main()
