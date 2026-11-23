import json
import unittest
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from threadpoolctl import threadpool_limits
from baseline.runner import solve as original_solve
from baseline.protocol import method
from .runner import solve,create_problem
from .problems import make_e1,make_poly
from .protocol import build
from .numerics import h_semismooth_newton, soft

RESULTS = Path(__file__).resolve().parents[1]/"results"


def requires(*paths):
    """Skip unless the upstream results exist; python run_all.py creates them."""
    missing = [p for p in paths if not (RESULTS/p).exists()]
    return unittest.skipIf(bool(missing), "needs " + ", ".join("results/"+p for p in missing) + " (run python run_all.py first)")


class ElasticNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.limiter=threadpool_limits(limits=1)

    def test_baseline_trajectory_unchanged(self):
        for p in (make_poly(example=True),make_e1(m=8,n=32,kappa=10)):
            for name in ('CP','H-CP','H-Log','C-LBreI','Proj-C','Proj-Log'):
                if p.family==0 and name.startswith('H-'):continue
                job=dict(panel='test',max_updates=100,stop_mode=0,dense=False,diagnostic=False)
                old=original_solve(p,method(name),job)
                new=solve(p,method(name),job)
                np.testing.assert_array_equal(old[2],new[2])
                np.testing.assert_array_equal(old[3],new[3])
                for key in ('K','N','M','W','status'):self.assertEqual(old[0][key],new[0][key])
                json.dumps(new[0],allow_nan=False)

    def test_instrumented_projection_same_algorithm(self):
        p=make_e1(m=8,n=32,kappa=10)
        job=dict(panel='test',max_updates=200,stop_mode=2,diagnostic=False)
        a=solve(p,method('Proj-Log'),job)
        b=solve(p,method('Proj-Log'),{**job,'diagnostic':True})
        np.testing.assert_array_equal(a[2],b[2]);np.testing.assert_array_equal(a[3],b[3])
        active=b[1][b[1][:,33]>0]
        self.assertTrue(np.all(active[:,33]>=1-1e-12))
        self.assertTrue(np.all(active[:,32]<=1+1e-8))

    @requires("baseline/config.json")
    def test_grid_and_sparsity(self):
        self.assertEqual([len(build(p)) for p in ('nonsmooth','hlog')],[40,20])
        p=make_e1(m=16,n=64,support_fraction=.2)
        self.assertEqual(np.count_nonzero(abs(p.z)>1e-10),12)
        self.assertLess(p.certificate['gap'],1e-10)

    def test_nonsmooth_projection_against_independent_primal_qp(self):
        rng=np.random.default_rng(26026)
        for index in range(30):
            n=8;lam=.4;p0=rng.normal(size=n);normal=rng.normal(size=n)
            x=soft(p0-normal,lam)
            h=normal/np.linalg.norm(normal)+10.**rng.uniform(-3,0)*rng.normal(size=n)
            h/=np.linalg.norm(h);g=h*10.**rng.uniform(-6,2)
            sigma=np.linalg.norm(g)*10.**rng.uniform(-4,0)
            p,aa,vv,calls,ips,flag=h_semismooth_newton(p0,x,g,sigma,normal,lam)
            self.assertEqual(flag,0,f'case {index}')
            candidate=soft(p,lam)
            j=normal/np.linalg.norm(normal)
            b=h@x-sigma/np.linalg.norm(g);c=j@x
            mat=np.vstack([np.c_[h[None,:],np.zeros((1,n))],
                np.c_[j[None,:],np.zeros((1,n))],np.c_[np.eye(n),-np.eye(n)],
                np.c_[-np.eye(n),-np.eye(n)]])
            upper=np.r_[b,c,np.zeros(2*n)]
            fun=lambda u:.5*u[:n]@u[:n]+lam*u[n:].sum()-p0@u[:n]
            jac=lambda u:np.r_[u[:n]-p0,np.full(n,lam)]
            fit=minimize(fun,np.zeros(2*n),jac=jac,method='SLSQP',
                constraints=LinearConstraint(mat,-np.inf,upper),options={'ftol':1e-12,'maxiter':1000})
            self.assertTrue(fit.success,fit.message)
            np.testing.assert_allclose(candidate,fit.x[:n],atol=2e-6,rtol=2e-6)
            self.assertLessEqual(h@candidate-b,2e-11)
            self.assertLessEqual(j@candidate-c,2e-11)

    def test_h_shortcut_absolute_cut_accuracy(self):
        p=make_poly(seed=5,m=10,n=40,power=4,lam=.5,correlation=.8)
        r,*_=solve(p,method('H-CP'),dict(panel='nonsmooth',max_updates=20000,stop_mode=0,diagnostic=False))
        self.assertLess(r['flag'],5)
        self.assertLess(r['relative_distance'],1e-4)

if __name__=='__main__':unittest.main()
