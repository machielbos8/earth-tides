# -*- coding: utf-8 -*-
#
# B2 stepping stone: radial h-refinement of the inviscid core (split into
# equal sub-layers) at K 2.
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
"""B2 stepping stone: radial h-refinement of the inviscid core (split into equal sub-layers) at K 2."""
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'src')
import numpy as np
from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit
md0 = EarthModel('data/PREM.dat'); ic = 1
def model(nsub):
    if nsub == 1:
        return md0
    r0, r1 = md0.r0[ic], md0.r1[ic]
    return md0.split_layer(ic, list(r0 + (r1 - r0) * np.arange(1, nsub) / nsub))
def h2(md, m, s, N, ls):
    g = ClairautGeometry(md)
    S = TidalSolver(md, g, m=m, K=2, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=s * OMEGA_EARTH, omega_rot=OMEGA_EARTH,
                    wahr_frame=(m != 2), dyn_fluid=True, consistent_reference=True, frame='mantle', linear_solver=ls)
    S.solve(); return iers_fit(S, g)['h2'].real * 1e4
for m, freqs in ((2, (1.9250, 1.9275, 1.9300, 1.93227)), (1, (0.9025, 0.905, 0.9075, 0.927))):
    print(f'm = {m}: h2 (1e-4) at', ' '.join(f'{f:.5f}' for f in freqs), '| spread block/SuperLU/dense (1e-6)')
    for N in (16, 24):
        for nsub in (1, 2, 4, 8):
            md = model(nsub)
            vals = np.array([[h2(md, m, f, N, ls) for ls in ('sparse', 'block', 'dense')] for f in freqs])
            spread = (vals.max(axis=1) - vals.min(axis=1)).max() * 100
            print(f'   N {N} x {nsub} core layers ({N * nsub:3d} per core): ' + ' '.join(f'{v:9.5f}' for v in vals[:, 0])
                  + f'  | {spread:9.3f}', flush=True)
