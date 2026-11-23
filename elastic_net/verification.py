"""Independent audit of endpoints, reference certificates, counters and provenance."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
import mpmath as mp
from threadpoolctl import threadpool_limits
from baseline.problems import value_gradient,soft
from baseline.protocol import identity
from .runner import read,write_json
from .protocol import BASELINE, PANELS

def one(directory,job):
    failures=[];algorithm_failures=[];runs=0;statuses=Counter()
    def check(ok,message):
        if not ok:failures.append(message)
    d=Path(directory)/'instances'/job['id']
    with threadpool_limits(limits=1),mp.workdps(65):
        ref=read(d/'reference.json');a=np.load(d/'instance.npz')
        A,b,p0,z,y=[a[k] for k in ('A','b','p0','z','reference_dual')]
        inequality=job['model']['kind']=='poly';cert=ref['certificate'];lam=ref['lambda_en']
        certificate_mode='independently_recomputed'
        if job.get('source_instance'):
            old=BASELINE/'instances'/job['source_instance'];oldref=read(old/'reference.json');olddata=np.load(old/'instance.npz')
            for k in ('A','b','p0','z','reference_dual'):check(np.array_equal(a[k],olddata[k]),'baseline input mismatch '+k)
            check(cert==oldref['certificate'],'baseline certificate differs')
            check(read(BASELINE/'verification.json')['passed'],'baseline independent audit not passed')
            direct_gap=cert['raw_gap'];certificate_mode='identical_to_previously_independently_verified_baseline'
        else:
            Am,bm,pm0,ym=[mp.matrix(v.tolist()) for v in (A,b,p0,y)]
            u=mp.matrix([mp.mpf(v) for v in cert['auxiliary_feasible_point']]);lm=mp.mpf(float(lam))
            p=pm0+Am.T*ym;mapped=mp.matrix([mp.sign(v)*max(abs(v)-lm,0) for v in p])
            residual=Am*u-bm
            defect=max(max(-v,0) if inequality else abs(v) for v in residual)
            direct_gap=float((u.T*u)[0]/2+lm*mp.norm(u,1)-(pm0.T*u)[0]+(mapped.T*mapped)[0]/2-(ym.T*bm)[0])
            check(float(defect)<1e-35,'reference auxiliary feasibility')
            check(-1e-35<=direct_gap<1e-10,'direct primal-dual gap')
            check(abs(direct_gap-cert['raw_gap'])<1e-30+1e-10*abs(direct_gap),'direct gap mismatch')
        check(cert['gap']<1e-10,'reported reference gap')
        if inequality:check(np.min(y)>=0,'negative reference multiplier')
        for method in job['methods']:
            rd=d/identity(method);r=read(rd/'result.json');state=np.load(rd/'state.npz')
            x,p,witness=[state[k] for k in ('x','p','witness')]
            prefix=method['label']+': '
            f,g,tau=value_gradient(A,b,x,int(inequality),job['model'].get('beta',0.),job['model'].get('power',4))
            check(np.isclose(f,r['delta'],rtol=2e-8,atol=1e-26),prefix+'objective mismatch')
            check(np.allclose(soft(p,lam),x,rtol=1e-12,atol=1e-12),prefix+'incompatible primal/dual')
            distance=np.linalg.norm(x-z)
            check(np.isclose(distance,r['distance'],rtol=1e-10,atol=1e-13),prefix+'distance mismatch')
            check(r['N']==r['K']+1 and r['M']==2*r['N'],prefix+'evaluation/matrix counters')
            check(r['W']>=r['K']+1,prefix+'kernel counters')
            check(r['max_cut_ratio']<=1+1e-10,prefix+'invalid cut')
            with gzip.open(rd/'trace.csv.gz','rt') as stream:
                cols=stream.readline().strip().split(',');t=np.loadtxt(stream,delimiter=',',ndmin=2)
            check(int(t[-1,0])==r['K'],prefix+'terminal trace missing')
            check(np.all(np.isfinite(t)),prefix+'nonfinite trace')
            check(np.all(np.diff(t[:,0])>0),prefix+'trace order')
            check(np.all(t[:,8]<=t[:,9]*(1+1e-10)+1e-300),prefix+'logged cut invalid')
            check(r['selection_success']==bool(np.linalg.norm(np.minimum(A@x-b,0) if inequality else A@x-b)/(1+np.linalg.norm(b))<=1e-8 and distance/(1+np.linalg.norm(z))<=1e-6),prefix+'success criterion mismatch')
            if method['name'].startswith('H-'):
                normal=p0-p
                check(np.linalg.norm(normal+A.T@witness)<1e-9*(1+np.linalg.norm(normal)),prefix+'normal representation')
                check(np.min(witness)>=-1e-13,prefix+'negative normal coefficient')
                check(normal@(z-x)<1e-9*(1+np.linalg.norm(z)**2),prefix+'memory cut excludes reference')
                check(r['max_Omega_drop']<1e-10*(1+np.linalg.norm(z)**2),prefix+'Omega decreases')
                check(r['max_Omega_excess']<1e-10*(1+np.linalg.norm(z)**2),prefix+'Omega exceeds target')
                if lam>0:check(r['max_projection_residual']<1e-10,prefix+'projection KKT residual too large')
            for eps,hit in zip(r['objective_targets'],r['objective_hits']):
                qualifying=t[t[:,15]<=eps]
                if len(qualifying):check(hit is not None and hit<=qualifying[0,1],prefix+'late target hit')
                if hit is not None:check(1<=hit<=r['N'],prefix+'target outside run')
                if r['delta0']<=eps:check(hit==1,prefix+'initially met target not marked')
            if r['flag']>=5:algorithm_failures.append(dict(method=method['label'],status=r['status'],N=r['N']))
            runs+=1;statuses[r['status']]+=1
    return dict(instance=job['id'],panel=job['panel'],runs=runs,failures=failures,
        algorithm_failures=algorithm_failures,statuses=dict(statuses),gap=cert['gap'],
        direct_gap=direct_gap,reference_relative_distance_bound=cert['distance_bound']/(1+np.linalg.norm(z)),
        certificate_mode=certificate_mode)

def verify(root,workers=4,panels=PANELS):
    root=Path(root);jobs=[];errors=[];details=[]
    for panel in panels:
        d=root/panel;cfg=read(d/'config.json');completion=read(d/'completion.json')
        for file,expected in cfg['hashes'].items():
            if hashlib.sha256((d/'source'/file).read_bytes()).hexdigest()!=expected:errors.append(panel+': source snapshot mismatch '+file)
        if completion['completed_runs']!=completion['expected_runs']:errors.append(panel+': incomplete runs')
        if list(d.glob('instances/**/error.json')):errors.append(panel+': exception records')
        jobs += [(str(d),j) for j in cfg['jobs']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(one,d,j):(d,j) for d,j in jobs}
        for f in concurrent.futures.as_completed(futures):
            try:details.append(f.result())
            except Exception as exc:errors.append(str(futures[f][1]['id'])+': '+repr(exc))
            if len(details)%20==0:print(f'Verified {len(details)}/{len(jobs)} instances',flush=True)
    statuses=Counter();algorithm_failures=[]
    for r in details:
        errors += [r['instance']+': '+x for x in r['failures']]
        algorithm_failures += [dict(instance=r['instance'],**x) for x in r['algorithm_failures']]
        statuses.update(r['statuses'])
    result=dict(passed=not errors and not algorithm_failures,integrity_passed=not errors,
        expected_instances=len(jobs),verified_instances=len(details),verified_runs=sum(x['runs'] for x in details),
        errors=errors,algorithm_failures=algorithm_failures,statuses=dict(statuses),
        max_reference_gap=max(x['gap'] for x in details),
        max_relative_reference_distance_bound=max(x['reference_relative_distance_bound'] for x in details),
        newly_recomputed_certificates=sum(x['certificate_mode']=='independently_recomputed' for x in details),
        reused_certificates=sum(x['certificate_mode']!='independently_recomputed' for x in details))
    suffix='' if tuple(panels)==PANELS else '_partial'
    write_json(root/f'verification_details{suffix}.json',details);write_json(root/f'verification{suffix}.json',result)
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--panels',nargs='+',choices=PANELS,default=PANELS)
    a=p.parse_args();verify(a.output,a.workers,a.panels)
