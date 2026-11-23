"""Tables and figures of the elastic-net campaign, generated solely from saved results."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .protocol import BASELINE, PANELS
from .runner import read,write_json

def save(fig,root,name):
    (root/'figures').mkdir(exist_ok=True,parents=True)
    for ext in ('pdf','png'):fig.savefig(root/'figures'/f'{name}.{ext}',bbox_inches='tight',dpi=180)
    plt.close(fig)

def load(directory):
    rows=[]
    for f in sorted(Path(directory).glob('instances/*/*/result.json')):
        r=read(f);r['path']=str(f.parent);r['label']=r['method']['label']
        rows.append({**r,**r['metadata']})
    return rows

def trace(r):return pd.read_csv(Path(r['path'])/'trace.csv.gz')

def tab(root,name,df):
    p=root/'tables';p.mkdir(exist_ok=True,parents=True)
    df.to_csv(p/f'{name}.csv',index=False,encoding='utf-8-sig')

def nonsmooth(root):
    runs=load(root/'nonsmooth');d=pd.DataFrame(runs)
    d['above_1pct']=d.relative_distance>.01;d['numeric_failure']=d.flag>=5
    simple=['instance_id','seed','s','lambda_en','label','N','status','selection_success','relative_distance',
        'delta','feasibility','complementarity','W','total_inner','max_inner','max_projection_residual',
        'max_Q_violation','max_Omega_drop','max_Omega_excess','above_1pct','numeric_failure']
    tab(root,'nonsmooth_all_runs',d[simple])
    g=d.groupby(['s','lambda_en','label']).agg(n=('N','size'),strict_successes=('selection_success','sum'),
        above_1pct=('above_1pct','sum'),numeric_failures=('numeric_failure','sum'),median_distance=('relative_distance','median'),
        max_distance=('relative_distance','max'),median_gap=('delta','median'),median_W=('W','median'),
        max_projection_residual=('max_projection_residual','max')).reset_index()
    tab(root,'nonsmooth_summary',g)
    fig,axs=plt.subplots(2,2,figsize=(7.5,6.5),sharex=True)
    colors={'CP':'tab:orange','Log-LBreI(theta=1)':'tab:blue','H-CP':'tab:red','H-Log':'tab:green'}
    grid=np.unique(np.geomspace(1,20000,120).astype(int))
    for ax,(power,lam) in zip(axs.flat,[(3,.1),(3,.5),(4,.1),(4,.5)]):
        reference_bounds=[]
        for r in runs:
            if r['s']==power and r['lambda_en']==lam:
                rd=Path(r['path']).parent
                rr=read(rd/'reference.json');zz=np.load(rd/'instance.npz')['z']
                reference_bounds.append(rr['certificate']['distance_bound']/np.linalg.norm(zz))
        for name in colors:
            vals=[]
            for r in runs:
                if r['s']!=power or r['lambda_en']!=lam or r['label']!=name:continue
                t=trace(r)
                values=np.interp(np.log(grid),np.log(t.n_grad),np.log(np.maximum(t.relative_distance,1e-16)))
                # Interpolation is only for plot display, never for threshold hits.
                values=np.exp(values)
                if r['status']!='exact_zero':values[grid>r['N']]=np.nan
                vals.append(values)
            v=np.array(vals);ax.loglog(grid,np.nanmedian(v,axis=0),label='Log (theta=1)' if name.startswith('Log-') else name,color=colors[name])
            ax.fill_between(grid,np.nanquantile(v,.25,axis=0),np.nanquantile(v,.75,axis=0),alpha=.13,color=colors[name])
        uncertainty=max(reference_bounds)
        ax.axhline(uncertainty,color='grey',ls=':',lw=1,label='Reference error bound')
        ax.set(title=f's={power}, lambda={lam}',xlabel='Gradient evaluations',ylabel='Relative distance to reference');ax.grid(alpha=.2)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.tight_layout(rect=(0,.1,1,1));save(fig,root,'nonsmooth_selection')

def hlog(root):
    rows=[];prefix=[]
    for r in load(root/'hlog'):
        old=read(BASELINE/'instances'/r['source_instance']/r['run_id']/'result.json')
        t=trace(r);at=t[t.n_grad==100000].iloc[0]
        delta=float(abs(at.relative_distance-old['relative_distance']))
        prefix.append(delta)
        rows.append(dict(seed=r['seed'],m=r['m'],n=r['n'],status=r['status'],
            distance_100k=old['relative_distance'],distance_1m=r['relative_distance'],
            distance_ratio=r['relative_distance']/old['relative_distance'],gap_100k=old['delta'],gap_1m=r['delta'],
            feasibility=r['feasibility'],complementarity=r['complementarity'],
            N_1pct=r['distance_hits'][0],N_01pct=r['distance_hits'][1],N_001pct=r['distance_hits'][2]))
    d=pd.DataFrame(rows);tab(root,'hlog_extended_paired',d)
    g=d.groupby(['m','n']).agg(instances=('seed','size'),median_100k=('distance_100k','median'),
        median_1m=('distance_1m','median'),median_ratio=('distance_ratio','median'),max_1m=('distance_1m','max'),
        above_1pct_100k=('distance_100k',lambda x:int((x>.01).sum())),
        above_1pct_1m=('distance_1m',lambda x:int((x>.01).sum()))).reset_index()
    tab(root,'hlog_extended_summary',g)
    write_json(root/'hlog_prefix_check.json',dict(maximum_distance_difference_at_100k=max(prefix),passed=max(prefix)<1e-12))
    fig,axs=plt.subplots(1,2,figsize=(7.5,3.5))
    for ax,(m,n) in zip(axs,[(10,40),(20,100)]):
        for r in load(root/'hlog'):
            if r['m']!=m:continue
            t=trace(r);ax.loglog(t.n_grad,t.relative_distance,alpha=.65,lw=1)
        ax.axvline(1e5,color='k',ls='--',lw=1);ax.axhline(.01,color='grey',ls=':',lw=1)
        ax.set_xticks([1,1e2,1e4,1e6])
        ax.set(title=f'{m} constraints, {n} variables',xlabel='Gradient evaluations',ylabel='Relative selected-solution distance');ax.grid(alpha=.2)
    fig.tight_layout();save(fig,root,'hlog_extended')

def main(root):
    root=Path(root)
    for name,fn in [('nonsmooth',nonsmooth),('hlog',hlog)]:
        if (root/name/'completion.json').exists():
            fn(root);print(f'Analyzed {name}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');main(p.parse_args().output)
