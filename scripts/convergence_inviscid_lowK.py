# -*- coding: utf-8 -*-
#
# Inviscid core at the low angular truncation of Wahr (1981) / DDW (1999): uniform K = 2..4,
# radial resolution N increased. Plan 2026-10-19, item B.
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
"""Inviscid core at the low angular truncation of Wahr (1981) / DDW (1999): uniform K = 2..4,
radial resolution N increased. Plan 2026-10-19, item B.

Wahr (1981, eq. 2.11) solves for sigma_l + tau_{l+-1} (our K = 2: U, V, P of degrees 2 and 4 for
m = 1, 2; W of 1 and 3; degree 4 enters his solution only to first order). Question: at such low K,
does increasing N alone give a smooth, converged inviscid response through the semidiurnal and
diurnal bands (where K = 8..24 gives spikes, Fig. 4)?
Same frequency grid and set-up as scripts/convergence_viscous_coreK.py (elastic hydrostatic PREM,
true fluid core, rotating, consistent reference, mantle frame), uniform K and N.
Usage: python scripts/convergence_inviscid_lowK.py K N [WORKERS]
-> results/convergence_inviscid_K<K>_N<N>.txt"""
import os
import sys
import warnings
from multiprocessing import Pool
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')    # Accelerate: one thread per worker
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit


def run(args):
    m, sig, K, N = args
    md = EarthModel('data/PREM.dat')
    g = ClairautGeometry(md)
    S = TidalSolver(md, g, m=m, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    S.solve()
    f = iers_fit(S, g)
    return (m, sig, K, N, f['h0'].real, f['h2'].real, f['l2'].real, f['l1'].real, S._solver_used)


if __name__ == '__main__':
    sys.path.insert(0, 'scripts')
    from provenance import Run
    _run = Run('inviscid core at low K: radial resolution (S11)', params=dict(argv=sys.argv[1:]), inputs=['data/PREM.dat'])
    K, N = int(sys.argv[1]), int(sys.argv[2])
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    jobs = [(2, s, K, N) for s in np.linspace(1.90, 1.96, 25)]
    jobs += [(1, s, K, N) for s in np.linspace(0.90, 0.95, 21)]
    with Pool(nproc, maxtasksperchild=8) as p:
        res = p.map(run, jobs, chunksize=1)
    out = 'results/convergence_inviscid_K%d_N%d.txt' % (K, N)
    with open(out, 'w') as fo:
        fo.write('m sigma K N h0 h2 l2 l1 solver\n')
        for r in res:
            fo.write('%d %.5f %d %d %.12f %.12f %.12f %.12f %s\n' % (r[:8] + (r[8].replace(' ', '_'),)))
    print(out, len(res), 'solves;', sum(r[8] != 'block' for r in res), 'not on the block path')
    _run.output(out)
    _run.__exit__(None, None, None)
