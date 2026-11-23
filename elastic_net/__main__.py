"""Elastic-net campaign: run a panel, regenerate tables and figures, or verify saved results.

    python -m elastic_net run nonsmooth --out results/elastic_net/nonsmooth
    python -m elastic_net run hlog --out results/elastic_net/hlog
    python -m elastic_net analyze --out results/elastic_net
    python -m elastic_net verify --out results/elastic_net
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from pathlib import Path

from elastic_net.protocol import PANELS, build


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('run', 'analyze', 'verify'))
    p.add_argument('panel', nargs='?', choices=PANELS, help='panel to run (stage "run" only)')
    p.add_argument('--out', type=Path, required=True,
                   help='run: the new panel directory; analyze/verify: the campaign root holding the panels')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--pilot', action='store_true', help='seed 0 and at most 500 updates')
    p.add_argument('--list', action='store_true')
    a = p.parse_args()
    if a.stage == 'run':
        if a.panel is None:
            p.error('stage "run" needs a panel')
        jobs = build(a.panel, a.pilot)
        if a.list:
            print(f'{len(jobs)} instances, {sum(len(j["methods"]) for j in jobs)} runs')
            return
        from elastic_net.runner import execute
        execute(jobs, a.out, a.workers)
    elif a.stage == 'analyze':
        from elastic_net.analysis import main as analyze
        analyze(a.out)
    else:
        from elastic_net.verification import verify
        result = verify(a.out, a.workers)
        if not result['integrity_passed']:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
