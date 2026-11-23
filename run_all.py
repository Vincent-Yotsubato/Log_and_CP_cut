"""Run every experiment of the paper in dependency order, then draw Figures 1-3.

    python run_all.py                    # everything, results in results/<package>
    python run_all.py --workers 8        # more worker processes for the parallel stages
    python run_all.py --from rates_and_inexact
    python run_all.py --list

A computation stage freezes its configuration in results/<package>/config.json on the first
run. Started again in the same directory with unchanged code, it keeps the finished runs.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (group, module arguments, accepts --workers); groups are listed in dependency order.
STEPS = [
    ("baseline", ["baseline", "run", "--out", "results/baseline"], True),
    ("baseline", ["baseline", "verify", "--out", "results/baseline"], False),
    ("baseline", ["baseline", "analyze", "--out", "results/baseline"], False),
    ("elastic_net", ["elastic_net", "run", "nonsmooth", "--out", "results/elastic_net/nonsmooth"], True),
    ("elastic_net", ["elastic_net", "run", "hlog", "--out", "results/elastic_net/hlog"], True),
    ("elastic_net", ["elastic_net", "verify", "--out", "results/elastic_net"], True),
    ("elastic_net", ["elastic_net", "analyze", "--out", "results/elastic_net"], False),
    ("selection_costs", ["selection_costs", "--out", "results/selection_costs"], False),
    ("explicit_cuts", ["explicit_cuts", "all", "--out", "results/explicit_cuts"], True),
    ("bilevel_baselines", ["bilevel_baselines", "all", "--out", "results/bilevel_baselines"], True),
    ("rates_and_inexact", ["rates_and_inexact", "all", "--out", "results/rates_and_inexact"], True),
    ("exact_projection", ["exact_projection", "all", "--out", "results/exact_projection"], True),
    ("inexact_projection", ["inexact_projection", "all", "--out", "results/inexact_projection"], True),
    ("paper_figures", ["paper_figures.fig1_explicit_cuts"], False),
    ("paper_figures", ["paper_figures.fig2_fig3_selection"], False),
]
GROUPS = list(dict.fromkeys(group for group, _, _ in STEPS))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4, help="worker processes of the parallel stages (default 4)")
    parser.add_argument("--from", dest="start", choices=GROUPS, default=GROUPS[0],
                        help="skip the groups before this one (their results must already exist)")
    parser.add_argument("--list", action="store_true", help="print the commands without running them")
    args = parser.parse_args()
    selected = [step for step in STEPS if GROUPS.index(step[0]) >= GROUPS.index(args.start)]
    started = time.perf_counter()
    for group, arguments, parallel in selected:
        command = [sys.executable, "-m", *arguments] + (["--workers", str(args.workers)] if parallel else [])
        print(f"[{group}] {' '.join(command[1:])}", flush=True)
        if args.list:
            continue
        step_start = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT)
        print(f"[{group}] finished in {(time.perf_counter() - step_start) / 60:.1f} min "
              f"(exit code {completed.returncode})", flush=True)
        if completed.returncode != 0:
            raise SystemExit(f"Stopped: {' '.join(arguments)} failed. Fix the cause and resume with --from {group}.")
    if not args.list:
        print(f"All steps finished in {(time.perf_counter() - started) / 60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
