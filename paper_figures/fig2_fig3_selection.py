"""Figures 2 and 3 of the paper (Sections 6.3 and 6.5), redrawn from frozen data.

Fig. 2: polyhedral selection, Euclidean (rho_c=0.8) and elastic-net panels.
Fig. 3: relative cut depth along the H trajectories and decay of E_k (Euclidean).
Writes PDF/PNG into results/paper_figures and the plotted summary numbers into
results/paper_figures/figures_summary.json.

The elastic-net H-CP and H-Log curves of Fig. 2 use the SSN-KKT exact projections of
results/exact_projection (the code behind the exact rows of Table 2); explicit CP and Log
come from results/elastic_net/nonsmooth.

    python -m paper_figures.fig2_fig3_selection [--out DIR]
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENTS = Path(__file__).resolve().parents[1]
BASE = EXPERIMENTS/'results/baseline'
DATA = EXPERIMENTS/'results/elastic_net'
KKT = EXPERIMENTS/'results/exact_projection/inexact'
SUMMARY = EXPERIMENTS/'results/paper_figures/figures_summary.json'
FIGURES = EXPERIMENTS/'results/paper_figures'
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--out', type=Path, default=FIGURES, help='output directory (default: results/paper_figures)')
OUT = parser.parse_args().out
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
COLORS = {'CP': 'tab:orange', 'Log-LBreI(theta=1)': 'tab:blue', 'H-CP': 'tab:red', 'H-Log': 'tab:green'}
NAMES = {'CP': 'CP', 'Log-LBreI(theta=1)': 'Log', 'H-CP': 'H-CP', 'H-Log': 'H-Log'}
# Explicit baselines dashed, so that coinciding curves remain visible.
LINE = {'CP': '--', 'Log-LBreI(theta=1)': '--', 'H-CP': '-', 'H-Log': '-'}
summary = {}


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT/f'{name}.{ext}', bbox_inches='tight', dpi=180)
    plt.close(fig)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def trace(path):
    return pd.read_csv(path/'trace.csv.gz')


runs = pd.read_csv(BASE/'tables/all_runs.csv')

# ------------------------------------------------------------ Fig. 2


def selection_runs():
    euclid = runs[(runs.panel == 1) & (runs.variant == 'polyhedron') & (runs.rho_c == .8)
                  & runs.label.isin(list(COLORS))]
    out = {'euclidean': [], 'elastic_net': []}
    for r in euclid.itertuples():
        d = BASE/'instances'/r.instance_id
        out['euclidean'].append(dict(label=r.label, status=r.status, N=r.N, path=d/r.run_id, instance=d))
    for path in sorted((DATA/'nonsmooth').glob('instances/*/*/result.json')):
        r = read(path)
        if r['method']['label'] in ('CP', 'Log-LBreI(theta=1)'):
            out['elastic_net'].append(dict(label=r['method']['label'], status=r['status'], N=r['N'],
                                           path=path.parent, instance=path.parent.parent))
    for path in sorted(KKT.glob('instances/*/*/result.json')):
        r = read(path)
        assert r['method']['configuration'] == 'SSN-KKT' and r['method']['name'] in ('H-CP', 'H-Log')
        out['elastic_net'].append(dict(label=r['method']['name'], status=r['status'], N=r['N'],
                                       path=path.parent, instance=path.parent.parent))
    return out


fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.5))
for ax, (group, items), budget, title in zip(axs, selection_runs().items(), (100000, 20000),
                                              ('(a) Euclidean kernel', '(b) Elastic-net kernel')):
    grid = np.unique(np.geomspace(1, budget, 160).astype(int))
    bounds = []
    for item in items:
        if item['label'] == 'H-CP':
            ref = read(item['instance']/'reference.json')
            z = np.load(item['instance']/'instance.npz')['z']
            bounds.append(ref['certificate']['distance_bound']/np.linalg.norm(z))
    summary[f'fig2_{group}'] = {}
    for label, color in COLORS.items():
        chosen = [i for i in items if i['label'] == label]
        assert len(chosen) == 40, (group, label, len(chosen))
        values = []
        for item in chosen:
            t = trace(item['path'])
            v = np.exp(np.interp(np.log(grid), np.log(t.n_grad), np.log(np.maximum(t.relative_distance, 1e-16))))
            if item['status'] != 'exact_zero':
                v[grid > item['N']] = np.nan
            values.append(v)
        values = np.asarray(values)
        ax.loglog(grid, np.nanmedian(values, axis=0), LINE[label], label=NAMES[label], color=color)
        ax.fill_between(grid, np.nanquantile(values, .25, axis=0), np.nanquantile(values, .75, axis=0),
                        alpha=.13, color=color)
        summary[f'fig2_{group}'][NAMES[label]] = dict(final_median=float(np.nanmedian(values[:, -1])),
                                                     final_max=float(np.nanmax(values[:, -1])))
    ax.axhline(max(bounds), color='grey', ls=':', lw=1, label='Reference uncertainty')
    ax.set(title=title, xlabel='Gradient evaluations', ylabel='Relative distance to reference')
    ax.grid(alpha=.2)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9)
fig.tight_layout(rect=(0, .1, 1, 1))
save(fig, 'polyhedral_selection')

# ------------------------------------------------------------ Fig. 3
euclid = runs[(runs.panel == 1) & (runs.variant == 'polyhedron') & (runs.rho_c == .8)
              & runs.label.isin(['H-CP', 'H-Log'])]
curves = {}
for r in euclid.itertuples():
    curves.setdefault((r.label, int(r.s)), []).append(BASE/'instances'/r.instance_id/r.run_id)
extended = []
for path in sorted((DATA/'hlog').glob('instances/*/*/result.json')):
    job = read(path.parent.parent/'job.json')
    assert job['model']['power'] == 4 and job['model']['correlation'] == .8
    extended.append(path.parent)
assert len(extended) == 20
curves[('H-Log', 4)] = extended  # the same 20 instances, continued to 10^6 evaluations
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.4))
styles = {3: '-', 4: '--'}
summary['fig3'] = {}
for (label, s), paths in sorted(curves.items()):
    budget = 1000000 if (label, s) == ('H-Log', 4) else 100000
    grid = np.unique(np.geomspace(1, budget, 200).astype(int))
    depth, gap = [], []
    column = 'rho_CP' if label == 'H-CP' else 'rho_log'
    for path in paths:
        t = trace(path)
        valid = (t.tau > 0) & (t.f > 0) & (t[column] > 0)
        last = t.n_grad.iloc[-1]
        d = np.interp(np.log(grid), np.log(t.n_grad[valid]), np.log(t[column][valid]),
                      right=np.nan) if valid.sum() > 1 else np.full(grid.size, np.nan)
        d = np.exp(d)
        d[grid > t.n_grad[valid].max()] = np.nan
        depth.append(d)
        e = np.maximum(t.Omega_gap/t.Omega_gap.iloc[0], 1e-12)
        g = np.exp(np.interp(np.log(grid), np.log(t.n_grad), np.log(e)))
        g[grid > last] = np.nan
        gap.append(g)
    depth, gap = np.asarray(depth), np.asarray(gap)
    color = COLORS[label]
    with np.errstate(all='ignore'):
        md = np.nanmedian(depth, axis=0)
        mg = np.nanmedian(gap, axis=0)
    axs[0].loglog(grid, md, styles[s], color=color, label=f'{label}, $s={s}$')
    axs[1].loglog(grid, mg, styles[s], color=color, label=f'{label}, $s={s}$')
    order = -1. if label == 'H-CP' else -1/(2*s-3)
    anchor = np.searchsorted(grid, 3 if label == 'H-CP' else 100)
    ref = mg[anchor]*(grid[anchor:]/grid[anchor])**order
    axs[1].loglog(grid[anchor:], ref, ':', color=color, lw=1)
    finite = np.isfinite(md)
    summary['fig3'][f'{label}_s{s}'] = dict(
        instances=len(paths), depth_ratio_min=float(np.nanmin(depth)),
        depth_ratio_median_at_end=float(md[finite][-1]) if finite.any() else None,
        depth_ratio_median_at_N=dict((str(n), float(md[np.searchsorted(grid, n)]))
                                     for n in (10, 100, 1000, 10000, 100000) if n <= budget),
        E_ratio_median_at_N=dict((str(n), float(mg[np.searchsorted(grid, n)]))
                                 for n in (10, 100, 1000, 10000, 100000, 1000000) if n <= budget))
for s in (3, 4):
    axs[0].axhline(1/s, color='grey', ls=styles[s], lw=.8)
axs[0].set(title=r'(a) Cut depth $\sigma_k/\inf_{z\in X}\langle g_k,x_k-z\rangle$', xlabel='Gradient evaluations')
axs[1].set(title=r'(b) $E_k/E_0$ (display floor $10^{-12}$)', xlabel='Gradient evaluations')
for ax in axs:
    ax.grid(alpha=.2)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9)
fig.tight_layout(rect=(0, .1, 1, 1))
save(fig, 'cut_depth_and_rates')
SUMMARY.parent.mkdir(parents=True, exist_ok=True)
SUMMARY.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(OUT)
print(json.dumps(summary['fig3'], indent=1)[:3000])
