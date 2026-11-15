# -*- coding: utf-8 -*-
#
# Radial resolution of the Ekman layers of the viscous core
# (EarthModel.viscous_core): N, d_min, boundary layers on/off.
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
# 17/10/2026 Machiel Bos
#===============================================================================
"""Is the Ekman layer of the viscous core (EarthModel.viscous_core) resolved in radius? O1 and M2,
K = 6, delta = 1.5 and 0.75 km: vary N (24, 30, 40), the boundary-layer thickness d_min (10, 20,
40 km) and boundary layers on/off. -> results/check_ekman_resolution.txt"""
import sys
import warnings
from multiprocessing import Pool
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit


def run(a):
    m, sig, delta, N, dmin, bl = a
    w = sig * OMEGA_EARTH
    md = EarthModel('data/PREM.dat').viscous_core(w, delta=delta, boundary_layers=bl, d_min=dmin)
    g = ClairautGeometry(md)
    S = TidalSolver(md, g, m=m, K=6, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                    omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    S.solve(); f = iers_fit(S, g)
    line = 'm=%d %.5f delta %4.2f km N%d d_min %2.0f km layers %-3s h0 %.8f (%+.2e) h2 %.9f (%+.2e) l2 %.9f l1 %.9f' % (
        m, sig, delta / 1e3, N, dmin / 1e3, 'yes' if bl else 'no', f['h0'].real, f['h0'].imag,
        f['h2'].real, f['h2'].imag, f['l2'].real, f['l1'].real)
    print(line, flush=True)
    return line


if __name__ == '__main__':
    sys.path.insert(0, 'scripts')
    from provenance import Run
    _run = Run('Ekman sub-layer resolution of the viscous core (S9)', params=dict(argv=sys.argv[1:]), inputs=['data/PREM.dat'])
    jobs = []
    for m, sig in ((1, 0.92700), (2, 1.92700)):
        for delta in (1.5e3, 7.5e2):
            jobs += [(m, sig, delta, N, 2e4, True) for N in (24, 30, 40)]
            jobs += [(m, sig, delta, 30, dm, True) for dm in (1e4, 4e4)]
            jobs += [(m, sig, delta, 30, 2e4, False)]
    with Pool(3, maxtasksperchild=1) as p:
        res = p.map(run, jobs, chunksize=1)
    open('results/check_ekman_resolution.txt', 'w').write('\n'.join(res) + '\n')
    _run.output('results/check_ekman_resolution.txt')
    _run.__exit__(None, None, None)
