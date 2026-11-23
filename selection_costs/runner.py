"""Reproduce the retained selection runs and time identical first-distance targets.

Original inputs, references and solver modules are reused without modification.
Dense traces locate the first hit; three serial replays measure that exact prefix.
Inputs: results/baseline (Euclidean, rho_c=0.8) and results/elastic_net/nonsmooth.
"""
from pathlib import Path
import json, time, hashlib, platform
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from baseline import numerics as old
from elastic_net import numerics as new

EXPERIMENTS = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENTS
SOURCES = [('Euclidean', EXPERIMENTS/'results/baseline'), ('Elastic net', EXPERIMENTS/'results/elastic_net/nonsmooth')]
METHODS = {'CP': 'CP', 'Log-LBreI(theta=1)': 'Log', 'H-CP': 'H-CP', 'H-Log': 'H-Log'}

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def collect():
    paths = []
    for panel, base in SOURCES:
        for p in sorted(base.glob('instances/*/*/result.json')):
            r = read(p)
            if r['method']['label'] not in METHODS:
                continue
            m = r['metadata']
            if panel == 'Euclidean' and not (r.get('variant') == 'polyhedron' and m.get('rho_c') == .8):
                continue
            paths.append((panel, p, r))
    assert len(paths) == 320, len(paths)
    return paths

def args_for(p, r, budget, steps):
    d=p.parent.parent
    a=np.load(d/'instance.npz'); ref=read(d/'reference.json')
    m=r['metadata']; method=r['method']; mod=old if ref['lambda_en']==0 else new
    args=(a['A'],a['b'],1,0.,int(m['s']),ref['L0'],ref['L1'],ref['lambda_en'],a['p0'],a['z'],
          mod.METHODS[method['name']],float(method['theta']),float(method['c']),budget-1,0,steps)
    if mod is new:
        args += (np.empty(0),np.array([.01]),False)
    return mod,args,a['z']

def main(OUT, pilot=False):
    OUT=Path(OUT); OUT.mkdir(parents=True, exist_ok=True)
    records=collect(); rows=[]; replay_rows=[]
    if pilot:  # two runs per kernel, for a quick check of the pipeline
        records=[r for g in ('Euclidean','Elastic net') for r in [v for v in records if v[0]==g][:2]]
    with threadpool_limits(limits=1):
        # Compile and load caches before timing any trajectory.
        for group in ('Euclidean','Elastic net'):
            _,p,r=next(v for v in records if v[0]==group)
            mod,args,_=args_for(p,r,3,np.arange(3,dtype=np.int64)); mod.run(*args)
        for i,(group,p,r) in enumerate(records):
            budget=100000 if group=='Euclidean' else 20000
            mod,args,z=args_for(p,r,budget,np.arange(budget,dtype=np.int64))
            ans=mod.run(*args); tr=ans[0]; stats=ans[5]
            # A reproduction audit, including the original stopping behavior.
            assert int(stats[1])==r['N'] and int(stats[3])==r['W'], (p,stats[:6],r['N'],r['W'])
            assert np.isclose(stats[9],r['relative_distance'],rtol=1e-8,atol=1e-12), p
            assert int(stats[5]) in (0,3), (p,stats[5])
            hit=np.flatnonzero(tr[:,14]<=.01)
            joint=np.flatnonzero((tr[:,24]<=1e-8)&(tr[:,11]/(1+np.linalg.norm(z))<=1e-6))
            first=int(tr[hit[0],1]) if len(hit) else None
            jfirst=int(tr[joint[0],1]) if len(joint) else None
            times=[]; W=None
            if first is not None:
                W=int(tr[hit[0],3])
                for repeat in range(3):
                    _,pa,_=args_for(p,r,first,np.array([0,first-1],dtype=np.int64) if first>1 else np.array([0],dtype=np.int64))
                    start=time.perf_counter(); out=mod.run(*pa); seconds=time.perf_counter()-start
                    assert int(out[5][1])==first and int(out[5][3])==W
                    assert out[5][9]<=.01+1e-14
                    times.append(seconds)
                    replay_rows.append(dict(group=group,instance_id=r['instance_id'],method=METHODS[r['method']['label']],repeat=repeat+1,N=first,W=W,seconds=seconds))
            row=dict(group=group,instance_id=r['instance_id'],method=METHODS[r['method']['label']],budget=budget,
                     m=r['metadata']['m'],n=r['metadata']['n'],s=r['metadata']['s'],seed=r['metadata']['seed'],
                     source=str(p.relative_to(ROOT)),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                     input_sha256=hashlib.sha256((p.parent.parent/'instance.npz').read_bytes()).hexdigest(),
                     distance_success=first is not None,joint_success=jfirst is not None,N=first,W=W,
                     W_per_gradient=W/first if first else None,time_seconds=float(np.median(times)) if times else None,
                     joint_first_N=jfirst,final_N=r['N'],final_W=r['W'],final_distance=float(stats[9]),
                     matched_original=True,full_trace_sha256=hashlib.sha256(tr.tobytes()).hexdigest())
            rows.append(row)
            if (i+1)%10==0:
                pd.DataFrame(rows).to_csv(OUT/'selection_costs_runs.csv',index=False)
                pd.DataFrame(replay_rows).to_csv(OUT/'selection_timing_repeats.csv',index=False)
                print(f'{i+1}/{len(records)} reproduced and costed',flush=True)
    d=pd.DataFrame(rows)
    d.to_csv(OUT/'selection_costs_runs.csv',index=False)
    pd.DataFrame(replay_rows).to_csv(OUT/'selection_timing_repeats.csv',index=False)
    summary=d.groupby(['group','method']).agg(instances=('instance_id','size'),distance_success=('distance_success','sum'),
        joint_success=('joint_success','sum'),median_N=('N','median'),median_W_per_gradient=('W_per_gradient','median'),median_seconds=('time_seconds','median')).reset_index()
    summary.to_csv(OUT/'selection_costs_summary.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps(dict(runs=len(d),timed_replays=len(replay_rows),repeats=3,blas_threads=1,
        counts_include_initial_evaluation=True,function_evaluations_equal_gradient_evaluations=True,
        timing='First iterate with relative distance <= 1%; median of three serial replays per instance, then median over successful instances; JIT/reference/I-O excluded; invariant diagnostics included.',
        python=platform.python_version(),numpy=np.__version__,all_runs_match_original=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pilot=pilot,
        solvers={str(Path(m.__file__).relative_to(ROOT)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (old,new)}),indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)

