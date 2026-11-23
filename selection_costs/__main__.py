"""Replay the 320 retained selection runs and time their first 1% distance hits.

    python -m selection_costs --out results/selection_costs  # full replay, about 20 minutes
    python -m selection_costs --out results/pilot --pilot      # two runs per kernel
"""
import os
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory; bilevel_baselines reads results/selection_costs")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    from selection_costs.runner import main as replay
    replay(args.out, args.pilot)


if __name__ == "__main__":
    main()
