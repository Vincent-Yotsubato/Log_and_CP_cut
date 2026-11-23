"""Isolated, resumable runs with frozen inputs and explicit failure records."""
import concurrent.futures
import csv
import gzip
import hashlib
import json
import platform
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
import scipy
import numba
from threadpoolctl import threadpool_limits
from baseline.runner import write_json
from baseline.protocol import identity
from .protocol import BASELINE, DISTANCES, log_steps
from .problems import Problem, make_e1, make_poly, value_gradient, soft
from .numerics import run, METHODS, FLAGS, COLUMNS

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def hashes():
    here=Path(__file__).parent
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [here/n for n in ('numerics.py','problems.py','protocol.py','runner.py')]}

def create_problem(job):
    if job.get('source_instance'):
        d=BASELINE/'instances'/job['source_instance']
        a=np.load(d/'instance.npz'); r=read(d/'reference.json'); m=job['model']
        return Problem(a['A'],a['b'],int(m['kind']=='poly'),m.get('beta',0.),m.get('power',4),
            r['L0'],r['L1'],r['lambda_en'],a['p0'],a['z'],a['reference_dual'],r['certificate'],r['metadata'])
    m=dict(job['model']); kind=m.pop('kind')
    return make_poly(**m) if kind=='poly' else make_e1(**m)

def solve(p, method, job):
    f0=value_gradient(p.A,p.b,soft(p.p0,p.lam),p.family,p.beta,p.power)[0]
    targets=np.empty(0)
    started=time.perf_counter()
    trace,dual,x,witness,hits,s,oh,dh=run(p.A,p.b,p.family,p.beta,p.power,p.L0,p.L1,p.lam,
        p.p0,p.z,METHODS[method['name']],float(method['theta']),float(method['c']),
        job['max_updates'],job['stop_mode'],log_steps(job['max_updates']),
        targets,np.array(DISTANCES),job.get('diagnostic',False))
    elapsed=time.perf_counter()-started
    keys=['K','N','M','W','IP','flag','delta0','delta','distance','relative_distance',
          'feasibility','max_cut_ratio','max_Q_violation','max_Omega_drop','max_Omega_excess',
          'max_projection_residual','total_inner','max_inner','complementarity']
    r={k:float(v) for k,v in zip(keys,s)}
    for k in ('K','N','M','W','IP','flag','total_inner','max_inner'):r[k]=int(r[k])
    r.update(method=method,status=FLAGS[r['flag']],elapsed_seconds=elapsed,
        hits=[int(v) if v>=0 else None for v in hits],
        objective_targets=targets.tolist(),objective_hits=[int(v) if v>=0 else None for v in oh],
        distance_targets=DISTANCES,distance_hits=[int(v) if v>=0 else None for v in dh],
        objective_success=bool(r['delta']<=f0*1e-9),
        selection_success=bool(r['feasibility']<=1e-8 and r['distance']/(1+np.linalg.norm(p.z))<=1e-6),
        timing_scope='JIT/reference/I-O excluded; standard invariant diagnostics included',
        inner_counter_definition='kernel calls within H projection, including single-halfspace shortcuts')
    r['success']=r['selection_success']
    return r,trace,dual,x,witness

def warmup():
    a=np.eye(2); b=np.ones(2)
    run(a,b,1,0.,4,4.,1.,0.,np.zeros(2),b,2,1.,1.,2,0,np.arange(3),
        np.empty(0),np.array(DISTANCES),False)

def worker(job,output):
    with threadpool_limits(limits=1):
        d=Path(output)/'instances'/job['id'];d.mkdir(parents=True,exist_ok=True)
        write_json(d/'job.json',job)
        warmup()
        try:
            if (d/'instance.npz').exists() and (d/'reference.json').exists():
                a=np.load(d/'instance.npz');r=read(d/'reference.json');m=job['model']
                p=Problem(a['A'],a['b'],int(m['kind']=='poly'),m.get('beta',0.),m.get('power',4),
                    r['L0'],r['L1'],r['lambda_en'],a['p0'],a['z'],a['reference_dual'],r['certificate'],r['metadata'])
            else:
                p=create_problem(job)
                np.savez_compressed(d/'instance.npz',A=p.A,b=p.b,p0=p.p0,z=p.z,reference_dual=p.dual)
                write_json(d/'reference.json',dict(certificate=p.certificate,metadata=p.metadata,
                    L0=p.L0,L1=p.L1,lambda_en=p.lam,source_instance=job.get('source_instance')))
            statuses=[]
            for method in job['methods']:
                rd=d/identity(method);rd.mkdir(exist_ok=True)
                if (rd/'result.json').exists():
                    statuses.append(read(rd/'result.json')['status']);continue
                r,t,dual,x,witness=solve(p,method,job)
                r.update(instance_id=job['id'],run_id=identity(method),panel=job['panel'],
                    variant=job['variant'],metadata={**p.metadata,'lambda_en':p.lam},source_instance=job.get('source_instance'))
                np.savez_compressed(rd/'state.npz',p=dual,x=x,witness=witness)
                with gzip.open(rd/'trace.csv.gz','wt',newline='',encoding='utf-8') as f:
                    writer=csv.writer(f);writer.writerow(COLUMNS);writer.writerows(t)
                write_json(rd/'result.json',r);statuses.append(r['status'])
            return dict(id=job['id'],runs=len(statuses),statuses={s:statuses.count(s) for s in set(statuses)})
        except Exception:
            error=traceback.format_exc();write_json(d/'error.json',dict(error=error))
            return dict(id=job['id'],runs=0,error=error)

def execute(jobs,output,workers=4):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    frozen=dict(jobs=jobs,hashes=hashes(),baseline=str(BASELINE),
        workers=workers,blas_threads=1,python=platform.python_version(),numpy=np.__version__,
        scipy=scipy.__version__,numba=numba.__version__,distance_targets=DISTANCES)
    config=output/'config.json'
    if config.exists() and read(config)!=frozen:raise RuntimeError('Output already has a different frozen protocol/source; use a new directory')
    if not config.exists():write_json(config,frozen)
    code=output/'source';code.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('*.py'):
        if not (code/p.name).exists():shutil.copy2(p,code/p.name)
    warmup();started=time.perf_counter();n=0;count=0;errors=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,j,str(output)) for j in jobs]
        for future in concurrent.futures.as_completed(futures):
            answer=future.result();n+=1;count+=answer['runs']
            if 'error' in answer:errors.append(answer)
            status=dict(completed_instances=n,total_instances=len(jobs),completed_runs=count,
                expected_runs=sum(len(j['methods']) for j in jobs),elapsed_seconds=time.perf_counter()-started,
                errors=errors,last=answer)
            write_json(output/'progress.json',status)
            with (output/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(answer)+'\n')
            print(f'{n}/{len(jobs)} instances; {count} runs; {status["elapsed_seconds"]:.1f}s; {answer.get("statuses",answer.get("error"))}',flush=True)
    write_json(output/'completion.json',status)
    return status
