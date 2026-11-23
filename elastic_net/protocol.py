"""Job grids of the elastic-net campaign: elastic-net polyhedral runs and the extended H-Log runs."""
import itertools
import json
from pathlib import Path
from baseline.protocol import method, identity

BASELINE = Path(__file__).resolve().parents[1] / 'results/baseline'
PANELS = ('nonsmooth', 'hlog')
DISTANCES = [1e-2, 1e-3, 1e-4]

def build(panel, pilot=False):
    old = json.loads((BASELINE / 'config.json').read_text(encoding='utf-8'))['jobs']
    jobs = []
    def add(variant, model, methods, updates, stop=0, source=None, diagnostic=False):
        job = dict(panel=panel, variant=variant, model=model, methods=methods,
                   max_updates=updates, stop_mode=stop, source_instance=source,
                   diagnostic=diagnostic, dense=False, pilot=pilot)
        job['id'] = identity(job)
        jobs.append(job)
    if panel == 'nonsmooth':
        for power, lam, seed in itertools.product((3,4), (.1,.5), range(10)):
            add('elastic_polyhedron', dict(kind='poly', m=10,n=40,power=power,
                correlation=.8,lam=lam,seed=seed),
                [method(s) for s in ('CP','Log-LBreI','H-CP','H-Log')],19999)
    elif panel == 'hlog':
        for j in old:
            if j['variant']=='polyhedron' and j['model']['power']==4 and j['model']['correlation']==.8:
                add('extended_hlog',j['model'],[method('H-Log')],999999,0,j['id'])
    else:
        raise ValueError(panel)
    if pilot:
        chosen=[]
        for j in jobs:
            if j['model'].get('seed',0)==0:
                j['max_updates']=min(500,j['max_updates'])
                j['id']=identity({k:v for k,v in j.items() if k!='id'})
                chosen.append(j)
        return chosen
    return jobs

def log_steps(budget):
    import numpy as np
    return np.unique(np.r_[np.arange(min(30,budget+1)),
        np.geomspace(30,max(30,budget),500).astype(int),
        [v for v in (999,9999,19999,99999,199999,499999,999999,budget) if v<=budget]]).astype(np.int64)
