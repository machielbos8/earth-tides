# -*- coding: utf-8 -*-
#
# Depth sensitivity of tidal Love numbers and the choice of alpha in the upper
# mantle.
#
# This file is part of earth-tides 1.1.
#
# earth-tides is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.
#
# 6/10/2026 Machiel Bos
#===============================================================================
"""Depth sensitivity of tidal Love numbers and the choice of alpha in the upper mantle.

PREMs, sphere, static, M2 frequency for the anelastic correction (Sect. 3.7).
(1) anelastic changes of h2, l2, k2 for alpha = 0.15 / 0 in various depth ranges;
(2) share of dh, dk from mu - 1 % in each layer, for degrees 2, 8 and 30
    (on a sphere the order is irrelevant: m = n, one degree).
Results -> results/depth_sensitivity.txt
"""
import sys
sys.path.insert(0, 'scripts')
from provenance import Run
_run = Run('depth sensitivity of the Love numbers', params=dict(argv=sys.argv[1:]), inputs=['data/PREM.dat'])
import warnings, copy; warnings.filterwarnings('ignore')
lines = []
_print = print
def print(*a):
    s = ' '.join(str(x) for x in a); _print(s, flush=True); lines.append(s)
import numpy as np
from earthtides.model import EarthModel
from earthtides.solver import TidalSolver
md0=EarthModel('data/PREMs.dat'); w=1.4053e-4; a=md0.a/1e3
def modified(d):
    new=copy.copy(md0)
    new.mu=[m*(1+di) for m,di in zip(md0.mu,d)]
    new.lam=[l-2/3*m*di for l,m,di in zip(md0.lam,md0.mu,d)]
    new.is_complex=bool(np.any(np.imag(d)!=0)); return new
def love(md,n=2):
    r=TidalSolver(md,m=n,K=1,N=24 if n<10 else 40,omega_f=0.0,n_force=n).solve()
    return r['h'],r['l'],r['kk']
top=[i for i in range(md0.n_layers) if a-md0.r0[i]/1e3<=220.5]   # layers above 220 km
d15=md0.dmu_factor(w,alpha=0.15); d0=md0.dmu_factor(w,alpha=0.0)
E=love(md0)
cases={'alpha 0.15 everywhere':d15,
       'alpha 0.15, alpha 0 above 220 km':np.array([d0[i] if i in top else d15[i] for i in range(md0.n_layers)]),
       'alpha 0 everywhere':d0,
       'alpha 0.15 above 220 km only':np.array([d15[i] if i in top else 0 for i in range(md0.n_layers)]),
       'alpha 0.15 below 220 km only':np.array([0 if i in top else d15[i] for i in range(md0.n_layers)])}
print(f"elastic: h2 {E[0]:.5f} l2 {E[1]:.5f} k2 {E[2]:.5f}")
for k,d in cases.items():
    h,l,kk=love(modified(d))
    print(f"{k:34s} dh2 {(h-E[0]).real:+.5f}{(h-E[0]).imag:+.5f}i  dl2 {(l-E[1]).real:+.5f}  dk2 {(kk-E[2]).real:+.5f}{(kk-E[2]).imag:+.5f}i")
print('\nkernel: share of d(h), d(k) from mu -1% in each layer (sum = 100%)')
for n in (2,8,30):
    En=love(md0,n); rows=[]
    for i in range(md0.n_layers):
        if not np.isfinite(md0.Qmu[i]): rows.append((0,0)); continue
        d=np.zeros(md0.n_layers); d[i]=-0.01
        h,l,kk=love(modified(d),n); rows.append(((h-En[0]).real,(kk-En[2]).real))
    rows=np.array(rows); sh=rows[:,0]/rows[:,0].sum()*100; sk=rows[:,1]/rows[:,1].sum()*100
    print(f'n={n}: ' + '  '.join(f'{a-md0.r1[i]/1e3:.0f}-{a-md0.r0[i]/1e3:.0f}:{sh[i]:.0f}/{sk[i]:.0f}' for i in range(md0.n_layers-1,-1,-1) if md0.Qmu[i]==md0.Qmu[i]))
    print(f'      top 220 km: h {sh[top].sum():.1f}%  k {sk[top].sum():.1f}%;  top 400 km: h {sh[top+[7]].sum():.1f}%  k {sk[top+[7]].sum():.1f}%')

open('results/depth_sensitivity.txt', 'w').write('\n'.join(lines) + '\n')
_run.output('results/depth_sensitivity.txt')
_run.__exit__(None, None, None)
