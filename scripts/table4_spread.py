# -*- coding: utf-8 -*-
#
# How much the tabulated inviscid Love numbers depend on the core resolution and on the linear solver.
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
# 16/11/2026 Machiel Bos
#===============================================================================
"""Review M4: the values of Table S13 (tab:nonhydro_love, make_table4.py --figure level) come from an inviscid
core that slips along its boundaries, K = 4, N = 24 everywhere, block solver. For the three elastic-support
states of that table (hydrostatic, Delta e_f of MHB, FCN at 431.4 d) and its eight waves, each value is solved
again with N = 16, 20, 30 in the liquid core and with the two other linear solvers (sparse = SuperLU,
layers) at N = 24. Printed: the largest deviation from the tabulated value (in 1e-4) per wave and parameter,
over the core resolutions and over the solvers, and the state in which it occurs.

    .venv/bin/python scripts/table4_spread.py [--workers 5]   ->  results/table4_spread_level.txt"""
import os
import sys
import warnings
from multiprocessing import Pool

for _v in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_v, '1')        # one BLAS thread per worker: threaded sums vary from run to run near the FCN
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
warnings.filterwarnings('ignore')
import numpy as np

from earthtides import love, solver

WAVES = (('Mf', 0, 0.07320), ('O1', 1, 0.92954), ('P1', 1, 0.99726), ('K1', 1, 1.00274),
         ('psi1', 1, 1.005475688), ('phi1', 1, 1.008213728), ('OO1', 1, 1.07594), ('M2', 2, 1.93227))
KEYS = ('h0', 'h2', 'l0', 'l1', 'k')
J2 = 1.14e-5
E_MHB = 9.73e-5
E_431 = 1.2382898371013964e-4          # results/paper1_table4_level.txt: e_f that puts the FCN at 431.4 sd
STATES = (('hydrostatic', None), ('MHB e_f', E_MHB), ('FCN 431', E_431))
CORE_N = (16, 20, 30)
SOLVERS = ('sparse', 'layers')
_TS = solver.TidalSolver


def opts(e_f, N=None):
    nh = None if e_f is None else dict(targets={'J2': J2, 'e_f': e_f})
    return love.Options(figure='level', nonhydrostatic=nh, fluidcore_N=N)


def job(j):
    state, e_f, wave, m, f, N, lin = j
    class TS(_TS):                       # love._job imports solver.TidalSolver at call time
        def __init__(self, *a, **kw):
            kw.setdefault('linear_solver', lin)
            super().__init__(*a, **kw)
    solver.TidalSolver = TS
    p = love._job((opts(e_f, N), 2, m, f))
    return j, {k: complex(p[k]).real for k in KEYS}


def main(run, workers):
    jobs = [(s, e, w, m, f, None, 'block') for s, e in STATES for w, m, f in WAVES]
    jobs += [(s, e, w, m, f, N, 'block') for s, e in STATES for w, m, f in WAVES for N in CORE_N]
    jobs += [(s, e, w, m, f, None, lin) for s, e in STATES for w, m, f in WAVES for lin in SOLVERS]
    with Pool(workers, maxtasksperchild=1) as pool:     # fresh process per job: the cached non-hydrostatic state is built with that job's solver
        res = dict(pool.map(job, jobs, chunksize=1))
    ref = {(j[0], j[2]): v for j, v in res.items() if j[5] is None and j[6] == 'block'}
    lines = ['Table S13 (level figure, inviscid slipping core K 4, N 24, block solver): largest deviation from the',
             'tabulated value, in 1e-4, over core N 16/20/30 (block) and over the solvers sparse/layers (N 24),',
             'maximum over the three states (hydrostatic, MHB e_f, FCN 431); [state] where the larger one occurs', '',
             f'{"wave":6s} ' + ' '.join(f'{k + " N":>9s} {k + " solv":>9s}' for k in KEYS) + '   worst']
    worst_all = {}
    for w, _, _ in WAVES:
        row, big = [], (0.0, '')
        for k in KEYS:
            dn = max(abs(v[k] - ref[(j[0], w)][k]) for j, v in res.items() if j[2] == w and j[5] is not None)
            ds = max(abs(v[k] - ref[(j[0], w)][k]) for j, v in res.items() if j[2] == w and j[6] != 'block')
            row += [1e4 * dn, 1e4 * ds]
            for j in sorted(res, key=lambda j: (j[0], j[2], str(j[5]), j[6])):   # deterministic tie-break
                d = round(abs(res[j][k] - ref[(j[0], w)][k]), 10)
                if j[2] == w and (j[5] is not None or j[6] != 'block') and d > big[0]:
                    big = (d, f'{k} {j[0]} {"N " + str(j[5]) if j[5] else j[6]}')
        worst_all[w] = big
        lines.append(f'{w:6s} ' + ' '.join(f'{x:9.2f}' for x in row) + f'   {1e4 * big[0]:.2f} ({big[1]})')
    lines += ['', 'values (real parts), per state and variant:']
    for j in sorted(res, key=lambda j: (j[0], j[2], str(j[5]), j[6])):
        lines.append(f'{j[0]:12s} {j[2]:5s} N {str(j[5] or 24):3s} {j[6]:7s} '
                     + ' '.join(f'{res[j][k]:+.6f}' for k in KEYS))
    run.write_text(str(ROOT / 'results' / 'table4_spread_level.txt'), '\n'.join(lines) + '\n')
    print('\n'.join(lines[:5 + len(WAVES)]))


if __name__ == '__main__':
    from provenance import Run
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 5
    with Run('Review M4: spread of Table S13 over core N and linear solvers', params=dict(workers=workers),
             inputs=['data/PREM.dat']) as run_:
        main(run_, workers)
