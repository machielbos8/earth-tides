# -*- coding: utf-8 -*-
#
# b2_wahr_h0.
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
# 20/10/2026 Machiel Bos
#===============================================================================
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'src')
import numpy as np
from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit
md0 = EarthModel('data/PREM.dat'); ic = 1
def model(nsub):
    if nsub == 1: return md0
    r0, r1 = md0.r0[ic], md0.r1[ic]
    return md0.split_layer(ic, list(r0 + (r1 - r0) * np.arange(1, nsub) / nsub))
DEG = {1: ([2], [1, 3]), 2: ([2], [3])}
def run(md, m, s, N, ls, deg=None, K=None):
    g = ClairautGeometry(md)
    kw = dict(degrees=deg) if deg else dict(K=K)
    S = TidalSolver(md, g, m=m, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=s * OMEGA_EARTH, omega_rot=OMEGA_EARTH,
                    wahr_frame=(m != 2), dyn_fluid=True, consistent_reference=True, frame='mantle', linear_solver=ls, **kw)
    r = S.solve(); f = iers_fit(S, g)
    return np.array([f['h0'].real, f['l0'].real, r['kk'].real * (1 + g.flattening_surface / 3) ** -5])
for m, s, name in ((1, 0.927, 'O1'), (2, 1.92700, 'M2')):
    print(f'{name}: h0, l0, k (DDW norm.) | solver spread (1e-9)')
    for N, nsub in ((16, 1), (24, 1), (24, 4), (44, 1), (24, 8)):
        md = model(nsub)
        v = np.array([run(md, m, s, N, ls, deg=DEG[m]) for ls in ('sparse', 'block', 'dense')])
        print(f'   Wahr truncation, N {N} x {nsub}: {v[0,0]:.6f} {v[0,1]:.6f} {v[0,2]:.6f} | {np.abs(v - v[0]).max()*1e9:8.2f}', flush=True)
    v = run(md0, m, s, 24, 'sparse', K=2)
    print(f'   our K2/N24 (determinate):      {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}', flush=True)
