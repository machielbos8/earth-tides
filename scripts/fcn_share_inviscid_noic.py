# -*- coding: utf-8 -*-
#
# Inviscid resonance shares for PREM without an inner core: is the viscous limit the inviscid core there?
# (P1 v2 Sect. 5.3, review)
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
# 10/11/2026 Machiel Bos
#===============================================================================
"""With the inner core the inviscid slipping core and the viscous core (delta -> 0) differ in the resonant share
of h(2) (+0.0040 against +0.0015, fcn_share_inviscid.py, fcn_resonance_extrapolation_v3.py). Without an inner
core the inviscid problem is determinate (Table S13), and fcn_resonance_noic.py gives the viscous limit there
(+0.00152). This script gives the inviscid shares for PREM without inner core (data/PREM_noIC.dat) at several
core resolutions K (and N): if they converge to the viscous value, the difference with the inner core comes
from the ill-posed problem; if they stay near 0.0040, the difference is the boundary condition (slip vs no slip).
Same sweep and one-pole fit as fcn_resonance_noic.py.
Usage: python scripts/fcn_share_inviscid_noic.py [NPROC]  ->  results/fcn_share_inviscid_noic.txt"""
import sys
import warnings
from multiprocessing import Pool
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer
from earthtides.iers import iers_fit
import fcn_resonance_noic as fr

RES = ((4, 24), (6, 30), (10, 30), (10, 44))       # core K, N
NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def solve(job):
    s, Kc, Nc = job
    md = EarthModel(fr.MODEL)
    g = ClairautGeometry(md)
    Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 24, 'rules': [{'fluid': True, 'K': Kc, 'N': Nc}]})
    S = TidalSolver(md, g, m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=s * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    r = S.solve()
    f = iers_fit(S, g)
    f['k0'] = r['kk'] * (1 + g.flattening_surface / 3) ** -5
    return job, np.array([complex(f[k]) for k in fr.KEYS])


def main():
    from provenance import Run
    with Run('inviscid FCN resonance shares, PREM without inner core, several core resolutions',
             params=dict(res=RES, sweep=fr.SWEEP), inputs=[fr.MODEL]) as run:
        jobs = [(s, K, N) for K, N in RES for s in fr.SWEEP]
        with Pool(NPROC, maxtasksperchild=8) as p:
            R = dict(p.map(solve, jobs, chunksize=1))
        s = np.array(fr.SWEEP)
        lines = ['PREM without inner core, inviscid slipping core; one-pole fit (fcn_resonance_noic.fit) over '
                 f'{len(s)} frequencies; resonant parts relative to that of h0',
                 f'{"core K":>6s} {"N":>3s} {"FCN sd":>8s} ' + ' '.join(f'{k:>9s}' for k in fr.KEYS[1:]) + '  max |resid| h0']
        for K, N in RES:
            L = np.array([R[(x, K, N)] for x in fr.SWEEP])
            p, c, mx = fr.fit(s, L, 1.0)
            A = c[4]
            lines.append(f'{K:6d} {N:3d} {1 / (p.real - 1):8.1f} ' +
                         ' '.join(f'{(A[i] / A[0]).real:+9.5f}' for i in range(1, len(fr.KEYS))) + f'  {mx[0]:.1e}')
        lines += ['', 'viscous limit (fcn_resonance_noic_K12.txt): h2 +0.00152 l0 -0.03142 l1 +0.00158 l2 -0.00018 '
                  'k0 +0.50223; with the inner core: viscous +0.0015, inviscid +0.0040 (h2)']
        run.write_text('results/fcn_share_inviscid_noic.txt', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
