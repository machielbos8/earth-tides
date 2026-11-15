# -*- coding: utf-8 -*-
#
# The DDW comparison (Table 2 of paper 1) with an inviscid core at low uniform K (Wahr's truncation)
# and increasing N. Plan 2026-10-19, item B.
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
# 19/10/2026 Machiel Bos
#===============================================================================
"""The DDW comparison (Table 2 of paper 1) with an inviscid core at low uniform K (Wahr's truncation)
and increasing N. Plan 2026-10-19, item B.

Wahr (1981, eq. 2.11): sigma_l + tau_{l+-1}, sigma_{l+-2} to first order. In our convention (K
degrees per field, parity of the forcing, starting at m) that is K = 2 for m = 1, 2 (degrees 2, 4)
and K = 3 for m = 0 (degrees 0, 2, 4). Same conventions as paper_pepi/make_tables_viscous.py
(DDW values, k in DDW's normalisation k (1 + f/3)^-5, K1 = mean of 1 -+ 1e-5 cpsd), elastic
hydrostatic PREM, true (inviscid) fluid core, rotating, consistent reference, mantle frame.
Usage: python scripts/ddw_lowK_table.py [WORKERS] -> results/ddw_lowK_table.txt"""
import os
import sys
import warnings
from multiprocessing import Pool
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
sys.path.insert(0, 'src')
sys.path.insert(0, 'paper_pepi')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit

KEYS = ('h0', 'h2', 'l0', 'l1', 'l2', 'k0')
RES = [(K, N) for K in (2, 3, 4, 6) for N in (16, 24, 30)]
SOLVERS = ('sparse', 'block')     # the value is SuperLU's; |block - SuperLU| flags an undetermined system


def ddw_lines():
    import make_tables_viscous as T                # the DDW values and lines of Table 2
    return T.DDW


def solve(job):
    m, sig, K, N, ls = job
    md = EarthModel('data/PREM.dat')
    g = ClairautGeometry(md)
    S = TidalSolver(md, g, m=m, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle', linear_solver=ls)
    r = S.solve()
    f = iers_fit(S, g)
    f['k0'] = r['kk'] * (1 + g.flattening_surface / 3) ** -5
    return job, ({k: float(np.real(f[k])) for k in KEYS}, S._solver_used)


if __name__ == '__main__':
    sys.path.insert(0, 'scripts')
    from provenance import Run
    _run = Run('Dehant et al. comparison at low uniform K (S11 truncation)', params=dict(argv=sys.argv[1:]), inputs=['data/PREM.dat'])
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    lines = ddw_lines()
    jobs = []
    for name, (m, s, _) in lines.items():
        for f in ((s - 1e-5, s + 1e-5) if s == 1.0 else (s,)):
            for K, N in RES:
                for ls in SOLVERS:
                    jobs.append((m, f, K, N, ls))
    with Pool(nproc, maxtasksperchild=8) as p:
        R = dict(p.map(solve, jobs, chunksize=1))
    out = ['DDW comparison, inviscid core, uniform K / N (h0 h2 l0 l1 l2 k; k in DDW normalisation); values: SuperLU',
           'Wahr (1981) truncation = K 2 (m = 1, 2), K 3 (m = 0); value - DDW in units of 1e-5', '']
    for name, (m, s, ref) in lines.items():
        fs = (s - 1e-5, s + 1e-5) if s == 1.0 else (s,)
        out.append(f'{name} ({s:.5f} cpsd, m = {m})')
        out.append('  %-10s ' % 'DDW' + ' '.join('%9s' % ('' if x is None else '%.5f' % x) for x in ref))
        for K, N in RES:
            v = {k: np.mean([R[(m, f, K, N, 'sparse')][0][k] for f in fs]) for k in KEYS}
            vb = {k: np.mean([R[(m, f, K, N, 'block')][0][k] for f in fs]) for k in KEYS}
            det = max(abs(v[k] - vb[k]) for k in KEYS)
            d = ' '.join('%9s' % ('' if x is None else '%+.1f' % ((v[k] - x) * 1e5)) for k, x in zip(KEYS, ref))
            out.append('  K%d/N%-6d ' % (K, N) + ' '.join('%9.5f' % v[k] for k in KEYS) + '   minus DDW: ' + d
                       + '   |block-SuperLU| %.0e' % det + ('   UNDETERMINED' if det > 1e-7 else ''))
        out.append('')
    open('results/ddw_lowK_table.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))
    _run.output('results/ddw_lowK_table.txt')
    _run.__exit__(None, None, None)
