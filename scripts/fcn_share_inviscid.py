# -*- coding: utf-8 -*-
#
# The same diurnal sweep and resonance fit as fcn_resonance_extrapolation.py (v2), for the inviscid core
# (null pair removed): like-for-like resonance strengths (residue ratios) inviscid vs viscous limit.
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
"""Inviscid resonance strengths (-> results/fcn_share_inviscid.txt)."""
import sys
import warnings
from multiprocessing import Pool
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.iers import iers_fit
from convergence_deflated import Defl
import fcn_resonance_extrapolation as F


def solve(x):
    md = EarthModel('data/PREM.dat'); g = ClairautGeometry(md)
    S = Defl(md, g, m=1, K=6, N=30, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=x * OMEGA_EARTH,
             omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True, frame='mantle')
    r = S.solve(); f = iers_fit(S, g); f['k0'] = r['kk'] * (1 + g.flattening_surface / 3) ** -5
    return np.array([complex(f[k]) for k in F.KEYS])


if __name__ == '__main__':
    sys.path.insert(0, 'scripts')
    from provenance import Run
    _run = Run('inviscid FCN resonance strengths, core K 6 (Sect. 5.3)', params=dict(argv=sys.argv[1:]), inputs=['data/PREM.dat'])
    s = np.array(F.SWEEP)
    with Pool(4) as p:
        L = np.array(p.map(solve, F.SWEEP, chunksize=1))
    # inviscid poles are real: fit with a tiny damping start
    sf, c, mx = F.fit(s, L, 1e1)
    A = c[4]
    out = ('inviscid K 6: FCN %.7f%+.1ei (%.1f sd), FICN %.7f%+.1ei (%.0f sd), max |resid| %s\n' % (
        sf[0].real, sf[0].imag, 1 / (sf[0].real - 1), sf[1].real, sf[1].imag, 1 / (sf[1].real - 1),
        np.array2string(mx, precision=1)) +
        'residue ratios: ' + '  '.join('%s %+.6f' % (k, (A[i] / A[0]).real) for i, k in enumerate(F.KEYS) if i) + '\n')
    print(out)
    open('results/fcn_share_inviscid.txt', 'w').write(out)
    np.savez('results/fcn_share_inviscid.npz', sweep=s, L=L)
    _run.output('results/fcn_share_inviscid.txt')
    _run.output('results/fcn_share_inviscid.npz')
    _run.__exit__(None, None, None)
