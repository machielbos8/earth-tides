# -*- coding: utf-8 -*-
#
# B2: Hough's Earth with an inviscid core: h and k against the radial resolution N, the truncation
# (Wahr's degrees or K 2) and the direct solver, for a sphere and the Maclaurin spheroid.
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
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'src'); sys.path.insert(0, 'examples/ex5')
import numpy as np
import hough1895 as ch
from earthtides.geometry import OMEGA_EARTH, ExactSpheroidGeometry, SphereGeometry
from earthtides.solver import TidalSolver
import tempfile, os
md = ch.make(path=os.path.join(tempfile.gettempdir(), 'HOUGH_b2.dat'))   # written by the example; not a result
print('Hough model: layers', md.n_layers, [(round(md.r0[i]/1e3), round(md.r1[i]/1e3)) for i in range(md.n_layers)])
fl = ch.maclaurin_flattening(OMEGA_EARTH)
DEG = {1: ([2], [1, 3]), 2: ([2], [3])}
def k(m, s, N, ls, trunc, flat):
    g = ExactSpheroidGeometry(fl) if flat else SphereGeometry()
    kw = dict(degrees=DEG[m]) if trunc == 'wahr' else dict(K=2)
    S = TidalSolver(md, g, m=m, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=s * OMEGA_EARTH, omega_rot=OMEGA_EARTH,
                    wahr_frame=(m != 2), dyn_fluid=True, consistent_reference=flat, frame='mantle', linear_solver=ls, **kw)
    r = S.solve(); return complex(r['kk']), complex(r['h'])
for flat in (False, True):
    for trunc in ('wahr', 'K2'):
        for m, s in ((2, 1.9275), (1, 0.927)):
            row = []
            for N in (16, 24, 32, 44, 60):
                v = np.array([k(m, s, N, ls, trunc, flat) for ls in ('sparse', 'block', 'dense')])
                row.append(f'N{N}: h {v[0,1].real:.7f} spread {np.abs(v - v[0]).max():.0e}')
            print(f'{"flat  " if flat else "sphere"} {trunc:4s} m={m} {s}: ' + ' | '.join(row), flush=True)
