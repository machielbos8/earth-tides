# -*- coding: utf-8 -*-
#
# How many Gauss-Legendre nodes in colatitude does the angular projection need? The default
# n_theta = 2 n_max + 24 (solver.py) was never tested; this varies it.
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
# 08/11/2026 Machiel Bos
#===============================================================================
"""Love numbers against the number of Gauss-Legendre nodes Q of the projection, eq. (projection) of P1.
A trial degree n times a test degree k is a polynomial of degree <= 2 n_max in cos(theta), integrated exactly
by Q >= n_max + 1 nodes; the flattening enters through 1/R, 1/R_s of the map r = s (1 - 2/3 eps P2), rational
in cos(theta), whose series adds degree 2 per power of eps ~ 1/300. Cases: inviscid PREM, K 5 N 30, at O1 and
M2 (the paper's standard), and viscous PREM delta 1 km, core K 24 N 30, band 12, at 0.9175 cpsd.
Q = n_max + 1, n_max + 3, n_max + 8, 2 n_max, 2 n_max + 24 (default), 2 n_max + 64 (reference).
Usage: python scripts/quadrature_nodes_test.py  ->  results/quadrature_nodes_test.txt"""
import os
import sys
import warnings
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
from multiprocessing import Pool

import numpy as np

CASES = (('O1', 1, 0.92700, None), ('M2', 2, 1.92700, None), ('att', 1, 0.9175, 24))
QS = (('n+1', 1, 1), ('n+3', 1, 3), ('n+8', 1, 8), ('2n', 2, 0), ('2n+24', 2, 24), ('2n+64', 2, 64))
KEYS = ('h0', 'h2', 'k0', 'l0', 'l1', 'l2')


def degrees(m, K):
    sph = [n for n in range(m, 400) if n % 2 == 0][:K]
    tor = [n for n in range(max(m, 1), 400) if n % 2 == 1][:K]
    return max(sph + tor + [2])


def one(job):
    name, m, sig, Kc, qn, a, b = job
    from earthtides.model import EarthModel
    from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
    from earthtides.solver import TidalSolver, resolution_per_layer
    from earthtides.iers import iers_fit
    if Kc is None:
        md = EarthModel('data/PREM.dat')
        kw = dict(K=5, N=30)
        nmax = degrees(m, 5)
    else:
        md = EarthModel('data/PREM.dat').viscous_core(sig * OMEGA_EARTH, delta=1.0e3)
        Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 24, 'rules': [{'fluid': True, 'K': Kc, 'N': 30}]})
        kw = dict(layer_K=Kl, layer_N=Nl, band=12)
        nmax = degrees(m, Kc)
    g = ClairautGeometry(md)
    S = TidalSolver(md, g, m=m, n_theta=a * nmax + b, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle', **kw)
    assert nmax == max(S._degF + S._degWF + [2]), (nmax, max(S._degF + S._degWF))
    r = S.solve()
    fit = iers_fit(S, g)
    fit['k0'] = r['kk'] * (1 + g.flattening_surface / 3) ** -5
    return name, qn, nmax, len(S.w), np.array([complex(fit[k]) for k in KEYS])


def mem(job):
    """Peak RSS (GB) and time of one viscous solve at 0.9175 cpsd, core K Kc, Q = a n_max + b."""
    import time
    import resource
    Kc, qn, a, b = job
    t = time.time()
    _, _, nmax, nq, love = one(('att', 1, 0.9175, Kc, qn, a, b))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1.0 if sys.platform == 'darwin' else 1024.0) / 1e9
    return Kc, qn, nmax, nq, rss, time.time() - t, love


def main_memory(Ks):
    """--memory K [K ...]: one solve at a time (separate processes), Q = n_max + 8 and 2 n_max + 24."""
    from provenance import Run
    jobs = [(K, qn, a, b) for K in Ks for qn, a, b in (('n+8', 1, 8), ('2n+24', 2, 24))]
    with Run('peak memory and time against the number of Gauss-Legendre nodes', params=dict(K_core=Ks),
             inputs=['data/PREM.dat']) as run_:
        res = []
        for j in jobs:
            with Pool(1, maxtasksperchild=1) as p:
                res.append(p.map(mem, [j])[0])
            print('K %d %-6s nodes %4d  %.2f GB  %.0f s' % (res[-1][0], res[-1][1], res[-1][3], res[-1][4],
                                                          res[-1][5]), flush=True)
        lines = ['# K Q n_max nodes peak_GB time_s  max|dLove| vs 2n+24']
        for K in Ks:
            ref = [x for x in res if x[0] == K and x[1] == '2n+24'][0][6]
            for Kc, qn, nmax, nq, rss, dt, love in [x for x in res if x[0] == K]:
                lines.append('%3d %-6s %3d %4d %6.2f %6.0f  %.1e' % (Kc, qn, nmax, nq, rss, dt,
                                                                   np.abs(love - ref).max()))
        run_.write_text('results/quadrature_nodes_memory.txt', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main_compare():
    """--compare: the solver's default against Q = 2 n_max + 24 (the default until solver-9) and 2 n_max + 64, at
    O1, psi1 (next to the FCN, ~1e3 amplification) and M2, inviscid K 5 and viscous core K 24."""
    from provenance import Run
    cases = (('O1', 1, 0.92700, None), ('psi1', 1, 1.00273, None), ('M2', 2, 1.92700, None),
             ('O1', 1, 0.92700, 24), ('psi1', 1, 1.00273, 24), ('att', 1, 0.9175, 24), ('M2', 2, 1.92700, 24))
    Qs = (('dflt', 0, 0), ('2n+24', 2, 24), ('2n+64', 2, 64))
    jobs = [(nm, m, s, K, qn, a, b) for nm, m, s, K in cases for qn, a, b in Qs]
    with Run('solver default n_theta against 2 n_max + 24 and + 64', params=dict(cases=cases),
             inputs=['data/PREM.dat']) as run_:
        with Pool(2, maxtasksperchild=1) as p:
            res = p.map(one, jobs, chunksize=1)
        lines = ['# case core nodes(default/old/ref)  max|dLove| over ' + ' '.join(KEYS)
                 + ': default-old  default-ref  old-ref   h2 (default)']
        for i in range(0, len(res), 3):
            (nm, _, _, nq0, l0), (_, _, _, nq1, l1), (_, _, _, nq2, l2) = res[i:i + 3]
            d = lambda x, y: np.abs(x - y).max()
            lines.append('%-5s %-8s %4d %4d %4d  %.1e  %.1e  %.1e  %.12f' % (
                nm, 'K5 inv' if jobs[i][3] is None else 'K24 visc', nq0, nq1, nq2, d(l0, l1), d(l0, l2), d(l1, l2),
                l0[1].real))
        run_.write_text('results/quadrature_nodes_compare.txt', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    if '--compare' in sys.argv:
        main_compare()
        sys.exit()
    if '--memory' in sys.argv:
        main_memory([int(k) for k in sys.argv[sys.argv.index('--memory') + 1:]])
        sys.exit()
    from provenance import Run
    jobs = [(nm, m, s, K, qn, a, b) for nm, m, s, K in CASES for qn, a, b in QS]
    with Run('number of Gauss-Legendre nodes of the angular projection', params=dict(cases=CASES, Q=QS),
             inputs=['data/PREM.dat']) as run_:
        with Pool(2, maxtasksperchild=1) as p:
            res = p.map(one, jobs, chunksize=1)
        lines = ['# case Q n_max nodes  max |dLove| vs 2n+64 over ' + ' '.join(KEYS) + '   h0 h2 (real)']
        for nm, *_ in CASES:
            rr = [x for x in res if x[0] == nm]
            ref = [x for x in rr if x[1] == '2n+64'][0][4]
            for _, qn, nmax, nq, love in rr:
                d = np.abs(love - ref)
                lines.append('%-4s %-6s %3d %4d  %.1e   %s' % (nm, qn, nmax, nq, d.max(),
                             ' '.join('%.12f' % v.real for v in love[:2])))
        run_.write_text('results/quadrature_nodes_test.txt', '\n'.join(lines) + '\n')
    print('\n'.join(lines))
