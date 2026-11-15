# -*- coding: utf-8 -*-
#
# What the slip term for a wall off the level surface changes in the Love numbers of the slipping core, per wave.
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
# 14/11/2026 Machiel Bos
#===============================================================================
"""Elastic hydrostatic PREM (first-order Clairaut figure, consistent reference potential), inviscid slipping core,
with TidalSolver(slip_nonlevel=False) and True: the IERS parameters h0, h2, l0, l1, l2 (iers_fit) at the main
diurnal waves and M2, and their differences. Which numbers of the paper that come from the slipping core move,
and by how much (lab notes 2026-11-14)?
Usage: python scripts/slip_nonlevel_waves.py [K [N]]  ->  results/slip_nonlevel_waves[_K<K>].txt"""
import sys
import warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np
from multiprocessing import Pool

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit
from provenance import Run

K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
MODEL = 'data/PREM.dat'
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2')
WAVES = (('Q1', 1, 0.89081), ('O1', 1, 0.92954), ('P1', 1, 0.99727), ('K1-', 1, 0.99990), ('psi1', 1, 1.00273),
         ('phi1', 1, 1.00546), ('OO1', 1, 1.07046), ('M2', 2, 1.93227))


def solve(job):
    name, m, sig, on = job
    md = EarthModel(MODEL)
    g = ClairautGeometry(md)
    w = sig * OMEGA_EARTH
    S = TidalSolver(md, g, m=m, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w, omega_rot=OMEGA_EARTH,
                    wahr_frame=(m == 1), dyn_fluid=True, consistent_reference=True, frame='mantle',
                    slip_nonlevel=on)
    S.solve()
    f = iers_fit(S, g)
    return np.array([complex(f[k]).real for k in KEYS])


def main():
    with Run('Love numbers of the slipping core with and without the slip term for a non-level wall',
             params=dict(K=K, N=N, waves=WAVES), inputs=[MODEL]) as run:
        jobs = [w + (on,) for w in WAVES for on in (False, True)]
        with Pool(4, maxtasksperchild=4) as pool:
            R = pool.map(solve, jobs, chunksize=1)
        lines = [f'elastic hydrostatic PREM, slipping inviscid core, K {K} N {N}; slip_nonlevel False / True / '
                 'difference (True - False)',
                 f'{"wave":5s} {"cpsd":>8s} {"":6s} ' + ' '.join(f'{k:>12s}' for k in KEYS)]
        for i, (name, m, sig) in enumerate(WAVES):
            a, b = R[2 * i], R[2 * i + 1]
            for lab, v in (('off', a), ('on', b), ('diff', b - a)):
                lines.append(f'{name:5s} {sig:8.5f} {lab:6s} ' + ' '.join(f'{x:+12.3e}' if lab == 'diff' else f'{x:+12.7f}'
                                                                           for x in v))
        out = 'results/slip_nonlevel_waves' + ('' if K == 3 else f'_K{K}')
        run.write_text(out + '.txt', '\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
