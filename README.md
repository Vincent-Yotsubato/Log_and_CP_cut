# Log and CP cuts: numerical experiments

Code for the numerical experiments (Section 6) of the paper

> **Logarithmic and Curvature-Corrected Polyak Cuts for Bregman Solution Selection under (L0,L1)-Smoothness**

The repository holds code only. Every instance is generated from fixed seeds, every reference
solution is certified in 60-digit arithmetic, and every table and figure is recomputed from the
saved runs. The `results/` directory is created by the scripts and is not tracked.

## Requirements

- Python 3.12 (tested with Python 3.12.8 on Windows 11; numba 0.60 does not support Python 3.13).
- The pinned packages: `python -m pip install -r requirements.txt`.

Run every command from the repository root. Each entry point limits BLAS and numba to one thread
before NumPy is imported; the parallel stages run independent jobs in worker processes instead.

## Quick check

```
python -m unittest discover -s . -t . -p "test_*.py"
```

The 48 tests check the projections, cuts and replays against independent implementations
(SciPy, plain NumPy, high-precision arithmetic). On a fresh clone 19 of them are skipped because
they read experiment results; they run once `results/` exists.

## Reproducing all experiments

```
python run_all.py                  # all stages in dependency order, then Figures 1-3 (about 40 minutes)
python run_all.py --list           # print the commands only
python run_all.py --workers 8      # more worker processes for the parallel stages
python run_all.py --from explicit_cuts
```

`run_all.py` runs the commands below in this order; each one can also be run on its own. Times are
wall-clock times with 4 worker processes on a laptop with an Intel Core i5-1135G7 (4 cores,
8 threads); the whole pipeline took 38 minutes.

| Stage | Command | Grid | Time |
|---|---|---|---|
| Baseline | `python -m baseline run --out results/baseline`, then `verify` and `analyze` with the same `--out` | 183 instances, 1181 runs | 3.5 min |
| Elastic-net kernel | `python -m elastic_net run nonsmooth --out results/elastic_net/nonsmooth`; `run hlog --out results/elastic_net/hlog`; `verify`, `analyze --out results/elastic_net` | 160 runs; 20 H-Log runs with 10^6 gradients | 1.4 min |
| Selection costs | `python -m selection_costs --out results/selection_costs` | 320 replays with serial timing | 1.5 min |
| Explicit cuts | `python -m explicit_cuts all` | 150 instances, 750 runs | 1.8 min |
| Simple bilevel baselines | `python -m bilevel_baselines all` | 560 runs | 1.6 min |
| Inexact projections and rates | `python -m rates_and_inexact all` | 640 runs and the rate analysis | 18 min |
| Exact projection (SSN-KKT) | `python -m exact_projection all` | 80 runs and serial timing | 3.0 min |
| Residual acceptance test | `python -m inexact_projection all` | 400 runs and serial timing | 7.7 min |
| Figures 1-3 | `python -m paper_figures.fig1_explicit_cuts`; `python -m paper_figures.fig2_fig3_selection` | | 0.2 min |

Commands without `--out` write to `results/<package>`. The packages `explicit_cuts`,
`bilevel_baselines`, `rates_and_inexact`, `exact_projection` and `inexact_projection` take a stage
argument: `run` (or `inexact` for `rates_and_inexact`), `timing` where applicable, `verify`,
`analyze`, or `all`. `--list` prints the grid. `--pilot` (where available; for `baseline`:
`--profile smoke`) runs a small version to check the pipeline; give it its own `--out` directory.

Later stages read the results of earlier ones, so the order matters:

- `elastic_net` reuses the polyhedral instances of `baseline` and continues its H-Log runs;
- `selection_costs` replays runs of `baseline` and `elastic_net/nonsmooth`;
- `explicit_cuts` reuses the kappa = 1 cosh instances of `baseline`;
- `bilevel_baselines` uses the Euclidean polyhedral instances of `baseline` and the H-CP and
  H-Log first hits of `selection_costs`;
- `rates_and_inexact` reads `baseline` and `elastic_net`; `exact_projection` reads
  `rates_and_inexact`; `inexact_projection` reads both;
- `paper_figures` reads `baseline`, `elastic_net`, `explicit_cuts` and `exact_projection`.

## Where the numbers in the paper come from

| Paper | Results | Code |
|---|---|---|
| Section 6.2, Figure 1 | `results/explicit_cuts` (ratios in `tables/cuts_summary.csv`) | `explicit_cuts`, `paper_figures/fig1_explicit_cuts.py` |
| Section 6.3, endpoints of the explicit methods on the Euclidean polyhedra | `results/baseline/tables/all_runs.csv` (variant `polyhedron`, rho_c = 0.8) | `baseline` |
| Figure 2(a), Euclidean kernel | `results/baseline` | `paper_figures/fig2_fig3_selection.py` |
| Figure 2(b), elastic-net kernel | explicit CP and Log: `results/elastic_net/nonsmooth`; H-CP and H-Log: `results/exact_projection/inexact` | `paper_figures/fig2_fig3_selection.py` |
| Table 2, exact rows | successes, N and W/N: `results/exact_projection`; times: `results/inexact_projection` | `exact_projection`, `inexact_projection` |
| Table 2, inexact rows; Section 6.4 | `results/inexact_projection` (main configuration SSN-R; SSN-R+ without the lower bound; PG-R and PG0-R with projected-gradient inner solves) | `inexact_projection` |
| Section 6.5, Figure 3 | trajectories: `results/baseline`; H-Log with s = 4 continued to 10^6 gradients: `results/elastic_net/hlog`; slopes and chi_poly: `results/rates_and_inexact/rates` | `rates_and_inexact`, `paper_figures/fig2_fig3_selection.py` |
| Section 6.6, Table 3 | `results/bilevel_baselines`; first hits of H-CP and H-Log from `results/selection_costs` | `bilevel_baselines`, `selection_costs` |

N counts gradient evaluations, W counts evaluations of the kernel map inside the projections, M
counts matrix-vector products and C counts certificate inner products. Failed runs are kept:
success counts and median curves include them; N, W/N and times at a target are medians over the
successful runs; paired ratios use only the instances on which both methods succeed.

## Layout

```
cut_functions.py      the cut functions A*, B*, S_log, rho, Gamma_alpha and the gradient bounds
baseline/             models, explicit methods, exact H methods, certified references
elastic_net/          H methods with semismooth-Newton projections for the elastic-net kernel
selection_costs/      cost replay and serial timing of the 320 selection runs
explicit_cuts/        cut comparison in the explicit update on cosh inverse problems
bilevel_baselines/    MNG, BiG-SAM and AGM-BiO on the Euclidean polyhedral instances
rates_and_inexact/    inexact two-halfspace projections and the rate analysis of the exact H runs
exact_projection/     exact elastic-net projections (SSN-KKT) and their timing
inexact_projection/   the residual acceptance test of the paper and its timing
paper_figures/        Figures 1-3 of the paper
run_all.py            all of the above in dependency order
```

Each experiment package has the same parts: `__main__.py` (command line), `protocol.py` (the
frozen job grid), `runner.py` (computation), `verification.py` (independent checks of the saved
results, without rerunning the algorithms), `analysis.py` (tables, figures and a report) and
`test_*.py`. `baseline/numerics.py` and `elastic_net/numerics.py` are the two numerical cores;
the later packages import them.

## Output of a stage

- `config.json`: the job grid, settings, SHA-256 of the source files and the package versions,
  frozen on the first run. A directory can only be resumed with the same configuration and code;
  use a new `--out` directory after changing either.
- `source/`: a copy of the code that produced the results (most packages).
- `instances/<instance>/`: input matrices, the certified reference and, per method, `result.json`,
  the endpoint `state.npz` and the trajectory `trace.csv.gz`.
- `verification.json`, and from `analyze`: `tables/` (CSV, plus LaTeX rows for some tables),
  `figures/` (PDF and PNG) and `REPORT_zh.md`, a short report in Chinese.

`python -m rates_and_inexact verify` reports 480 numerical failures: all 80 runs of each of the six
projected-gradient configurations (PG-*) stop at the inner iteration cap and are recorded as
failures. This is the expected outcome and matches the results behind the paper. A stage exits
with a nonzero code only when an integrity check fails.

Counts such as N, W and the success flags are deterministic for the pinned package versions;
times depend on the machine. On our machine, `python run_all.py` in a fresh clone reproduced every
result record and endpoint behind the paper bit for bit, and Figures 1-3 pixel for pixel.
