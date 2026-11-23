"""Figure 1 of the paper (Section 6.2): explicit updates with different cuts on the cosh instances.

Reads results/explicit_cuts and writes cuts_explicit.pdf/.png into results/paper_figures.

    python -m paper_figures.fig1_explicit_cuts [--out DIR]
"""
from pathlib import Path
import argparse
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
from explicit_cuts.analysis import collect, curves, COLORS, STYLES  # noqa: E402
from explicit_cuts.protocol import LABELS, METHOD_NAMES  # noqa: E402
from rates_and_inexact.common import trace  # noqa: E402

# Log and CP are both solid in explicit_cuts.STYLES; give CP its own dash pattern for grayscale print.
STYLES = dict(STYLES, CP=(0, (3, 1, 1, 1, 1, 1)))

FIGURES = EXPERIMENTS/'results/paper_figures'
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--out', type=Path, default=FIGURES, help='output directory (default: results/paper_figures)')
OUT = parser.parse_args().out
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = EXPERIMENTS/'results/explicit_cuts'
plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
runs = collect(RESULTS)
budget = max(r['N'] for r in runs)
grid = np.unique(np.geomspace(1, budget, 200).astype(int))
fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.9))


def panel(ax, selected, title):
    for name in METHOD_NAMES:
        chosen = [r for r in selected if r['method']['name'] == name]
        v = curves(chosen, 'relative_distance', grid)
        ax.loglog(grid, np.median(v, axis=0), linestyle=STYLES[name], color=COLORS[name], label=LABELS[name], lw=1.1)
        ax.fill_between(grid, np.quantile(v, .25, axis=0), np.quantile(v, .75, axis=0), color=COLORS[name], alpha=.1)
    ax.set(title=title, xlabel='Gradient evaluations')
    ax.grid(alpha=.2)


panel(axs[0], [r for r in runs if r['variant'] == 'kappa10_calibrated' and r['model']['beta'] == 4.],
      r'(a) $\beta=4$, $r_0=1000$')
panel(axs[1], [r for r in runs if r['variant'] == 'kappa10_zero' and r['model']['beta'] == 0.],
      r'(b) $\beta=0$, $p_0=0$')
axs[0].set_ylabel('Relative distance to $z^\\dagger$')
pts = defaultdict(list)
for r in runs:
    # Panel (c) uses the same ten beta=4, r0=1000 runs as panel (a).
    if r['variant'] != 'kappa10_calibrated' or r['model']['beta'] != 4.:
        continue
    t = trace(r['path']/'trace.csv.gz')
    ok = (t['tau'] > 0) & (t['r'] > 0)
    for column in ('rho_C', 'rho_log', 'rho_delta', 'rho_CP'):
        pts[column].append(np.c_[t['r'][ok], t[column][ok]])
edges = np.geomspace(1e-3, max(np.max(np.vstack(v)[:, 0]) for v in pts.values()), 40)
for column, name in (('rho_C', 'C-LBreI'), ('rho_log', 'Log-LBreI'), ('rho_delta', 'Polyak-1'), ('rho_CP', 'CP')):
    data = np.vstack(pts[column])
    idx = np.digitize(data[:, 0], edges)
    centers = [np.sqrt(edges[i-1]*edges[i]) for i in range(1, len(edges)) if np.sum(idx == i) >= 20]
    meds = [np.median(data[idx == i, 1]) for i in range(1, len(edges)) if np.sum(idx == i) >= 20]
    # Logarithmic ratio axis: the clipped and Log cuts fall to about 1% and 5% of the exact depth.
    axs[2].loglog(centers, meds, linestyle=STYLES[name], color=COLORS[name], lw=1.1,
                  label={'Polyak-1': r'$\delta$'}.get(name, LABELS[name]))
axs[2].set(title=r'(c) $\beta=4$, $r_0=1000$', xlabel=r'$r=L_1q/L_0$',
           ylabel=r'$\sigma/\langle g,x-z^\dagger\rangle$', ylim=(5e-3, 1))
axs[2].grid(alpha=.2)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9, bbox_to_anchor=(.5, -.01))
fig.tight_layout(rect=(0, .1, 1, 1))
for ext in ('pdf', 'png'):
    fig.savefig(OUT/f'cuts_explicit.{ext}', bbox_inches='tight', dpi=180)
print(OUT/'cuts_explicit.pdf')
